"""Ataraxos Update-Equivalence Tabular MMD Search (Milestone 5 Tier-1 Search).

Implements exact Maximum Entropy Mirror Descent (MMD) policy improvement
over Common Random Numbers (CRN) belief worlds:

    pi_search(a) proportional to [ pi_bp(a) * U(a)^(eta * tau) * exp(eta * Q_hat(a)) ]^(1 / (1 + eta * tau))

Guarantees:
1. Exact closed-form solution to policy iteration anchored to blueprint + uniform magnet.
2. Common Random Numbers across all candidate root actions eliminate variance.
3. Compact active-indices stepping in C++ with centralized critic leaf evaluation.
4. Returns structured SearchResult with live Belief X-Ray diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import uno_ataraxos as ua
from uno_ataraxos import UnoEngine, VecUnoEnv
from uno_ataraxos.belief import UnoBeliefNet
from uno_ataraxos.networks import UnoActorNet, UnoQCriticNet
from uno_ataraxos.rl_math import (
    get_device,
    masked_log_softmax,
    uniform_legal_log_probs,
    wdl_expected_value,
)

# Card ID mappings for Belief X-Ray
DRAW_2_CARD_IDS = [42, 45, 48, 51]
WILD_DRAW_4_CARD_ID = 53
RED_CARD_IDS = list(range(0, 10)) + [40, 41, 42]
GREEN_CARD_IDS = list(range(10, 20)) + [43, 44, 45]
BLUE_CARD_IDS = list(range(20, 30)) + [46, 47, 48]
YELLOW_CARD_IDS = list(range(30, 40)) + [49, 50, 51]


@dataclass
class SearchResult:
    """Structured result of Ataraxos test-time search."""

    action: int
    search_probs: torch.Tensor          # [61] float32 on device
    bp_probs: torch.Tensor              # [61] float32 on device
    q_hat: torch.Tensor                 # [61] float32 on device (-1e9 on illegal)
    belief_xray: Dict[str, float]       # Live diagnostics P(+2), P(+4), colors, etc.
    latency_ms: float                   # Total wall-clock time in ms


def compute_belief_xray(
    sampled_opp_hands: torch.Tensor,     # [K, 54] int32
    opp_hand_size: int,
) -> Dict[str, float]:
    """Compute live Belief X-Ray probabilities from K sampled opponent hands."""
    K = sampled_opp_hands.shape[0]
    if K == 0 or opp_hand_size == 0:
        return {
            "p_plus_2": 0.0,
            "p_plus_4": 0.0,
            "p_red": 0.0,
            "p_green": 0.0,
            "p_blue": 0.0,
            "p_yellow": 0.0,
            "opp_hand_size": float(opp_hand_size),
        }

    p_p2 = float((sampled_opp_hands[:, DRAW_2_CARD_IDS].sum(dim=-1) > 0).float().mean().item())
    p_p4 = float((sampled_opp_hands[:, WILD_DRAW_4_CARD_ID] > 0).float().mean().item())
    p_r = float((sampled_opp_hands[:, RED_CARD_IDS].sum(dim=-1) > 0).float().mean().item())
    p_g = float((sampled_opp_hands[:, GREEN_CARD_IDS].sum(dim=-1) > 0).float().mean().item())
    p_b = float((sampled_opp_hands[:, BLUE_CARD_IDS].sum(dim=-1) > 0).float().mean().item())
    p_y = float((sampled_opp_hands[:, YELLOW_CARD_IDS].sum(dim=-1) > 0).float().mean().item())

    return {
        "p_plus_2": p_p2,
        "p_plus_4": p_p4,
        "p_red": p_r,
        "p_green": p_g,
        "p_blue": p_b,
        "p_yellow": p_y,
        "opp_hand_size": float(opp_hand_size),
    }


def ataraxos_mmd_search(
    root_engine: UnoEngine,
    actor_net: UnoActorNet,
    belief_net: UnoBeliefNet,
    q_critic_net: Optional[UnoQCriticNet] = None,
    rollout_pool: Optional[VecUnoEnv] = None,
    K: int = 128,
    depth: int = 12,
    eta: float = 3.0,
    tau: float = 0.05,
    uniform_mix: float = 0.03,
    max_total_worlds: int = 768,
    leaf_eval: str = "critic",
    deterministic: bool = False,
    seed: Optional[int] = None,
) -> SearchResult:
    """Run Tier-1 Ataraxos Update-Equivalence Tabular MMD Search.

    Args:
        root_engine: C++ UnoEngine instance at the current root state.
        actor_net: Blueprint actor policy network.
        belief_net: Generative belief network for opponent hand sampling.
        q_critic_net: Optional centralized Q-critic network for leaf evaluation.
        rollout_pool: Persistent C++ VecUnoEnv(auto_reset=False) instance.
        K: Target number of belief worlds (default 128).
        depth: Maximum rollout depth in plies (default 12).
        eta: MMD learning rate / inverse temperature (default 3.0).
        tau: Uniform legal policy magnet weight (default 0.05).
        uniform_mix: Safety mixture weight inside belief sampler (default 0.03).
        max_total_worlds: Maximum product of num_a * K_eff to cap latency (default 768).
        leaf_eval: Leaf evaluation mode, "critic" or "wdl" (default "critic").
        deterministic: If True, selects argmax action; else samples from search_probs.
        seed: Random seed for world generation and rollouts.

    Returns:
        SearchResult containing the chosen action, policy distributions, Q-values,
        Belief X-Ray diagnostics, and wall-clock latency.
    """
    t0 = time.perf_counter()
    device = next(actor_net.parameters()).device
    root_player = int(root_engine.current_player)

    # 1. Extract root observations
    legal_actions = np.array(root_engine.get_legal_action_ids(root_player), dtype=np.int32)
    num_a = len(legal_actions)

    legal_mask_np = root_engine.get_action_mask(root_player)
    legal_mask_t = torch.from_numpy(legal_mask_np).to(device)

    # Early exit if only 1 action is legal (0 ms)
    if num_a <= 1:
        chosen_act = int(legal_actions[0]) if num_a == 1 else 60
        single_p = torch.zeros(61, dtype=torch.float32, device=device)
        single_p[chosen_act] = 1.0
        q_zeros = torch.full((61,), -1e9, dtype=torch.float32, device=device)
        q_zeros[chosen_act] = 0.0
        xray_dummy = {"p_plus_2": 0.0, "p_plus_4": 0.0, "p_red": 0.0, "p_green": 0.0, "p_blue": 0.0, "p_yellow": 0.0, "opp_hand_size": 0.0}
        return SearchResult(
            action=chosen_act,
            search_probs=single_p,
            bp_probs=single_p,
            q_hat=q_zeros,
            belief_xray=xray_dummy,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )

    # Extract tensors for root player
    cf_np = root_engine.get_card_features_actor(root_player)
    gf_np = root_engine.get_global_features(root_player)
    ht_np = root_engine.get_history_tokens(root_player)

    cf_t = torch.from_numpy(cf_np).unsqueeze(0).to(device)
    gf_t = torch.from_numpy(gf_np).unsqueeze(0).to(device)
    ht_t = torch.from_numpy(ht_np).unsqueeze(0).to(device)

    # 2. Compute Blueprint and Uniform Magnet Log-Probabilities
    device_type = "mps" if device.type == "mps" else ("cuda" if device.type == "cuda" else "cpu")
    with torch.no_grad(), torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        bp_logits, _ = actor_net(cf_t, gf_t, ht_t)

    bp_log_p = masked_log_softmax(bp_logits.float().squeeze(0), legal_mask_t)
    bp_probs = torch.where(legal_mask_t, bp_log_p.exp(), torch.zeros_like(bp_log_p))
    mag_log_p = uniform_legal_log_probs(legal_mask_t)

    # 3. Compute Adaptive World Budget K_eff & Sample Opponent Hands
    K_eff = min(K, max(32, max_total_worlds // max(1, num_a)))
    opp_hand_size = int(round(gf_np[1] * 20.0))
    sampled_opp = belief_net.sample_opp_hands(
        cf_t, gf_t, ht_t, K=K_eff, temp=1.0, uniform_mix=uniform_mix
    )  # [K_eff, 54] int32 on device
    belief_xray = compute_belief_xray(sampled_opp, opp_hand_size)

    # 4. Fast C++ Batched World Loader with Common Random Numbers
    total_slots = num_a * K_eff
    if rollout_pool is None:
        rollout_pool = VecUnoEnv(num_envs=total_slots, seed=seed if seed is not None else 42, auto_reset=False)
    elif rollout_pool.num_envs != total_slots:
        rollout_pool.resize(total_slots)

    sampled_opp_np = sampled_opp.cpu().numpy()
    rng_seed = seed if seed is not None else random.randint(1, 10_000_000)

    obs, rew_root_np, dones_root_np = rollout_pool.load_worlds_and_step_root(
        root_engine, sampled_opp_np, legal_actions, rng_seed
    )

    root_values = np.zeros(total_slots, dtype=np.float32)
    root_values[dones_root_np] = rew_root_np[dones_root_np]
    active_indices = np.where(~dones_root_np)[0].astype(np.int32)

    # 5. Fast Rollout Loop (ply = 1 .. depth)
    for ply in range(1, depth):
        if len(active_indices) == 0:
            break

        act_cf = torch.from_numpy(obs["card_features_actor"][active_indices]).to(device)
        act_gf = torch.from_numpy(obs["global_features"][active_indices]).to(device)
        act_ht = torch.from_numpy(obs["history_tokens"][active_indices]).to(device)
        act_mask = torch.from_numpy(obs["legal_mask"][active_indices]).to(device)

        with torch.no_grad(), torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits_step, _ = actor_net(act_cf, act_gf, act_ht)

        p_step = F.softmax(logits_step.float().masked_fill(~act_mask, -1e9), dim=-1)

        # Sample actions with blueprint stochasticity
        sampled_actions = torch.multinomial(p_step, num_samples=1).squeeze(-1).cpu().numpy().astype(np.int32)

        # Step active environments in C++
        obs, rew_step, dones_step = rollout_pool.step_indices(active_indices, sampled_actions)

        # Handle newly terminated environments (pure CPU vectorization)
        newly_done_mask = dones_step[active_indices]
        if newly_done_mask.any():
            done_slots = active_indices[newly_done_mask]
            acting_players = obs["acting_player"][done_slots]
            signs = np.where(acting_players == root_player, 1.0, -1.0).astype(np.float32)
            root_values[done_slots] = signs * rew_step[done_slots]
            active_indices = active_indices[~newly_done_mask]

    # 6. Leaf Evaluation for Still-Active Leaves
    if len(active_indices) > 0:
        leaf_indices = active_indices
        leaf_cf = torch.from_numpy(obs["card_features_actor"][leaf_indices]).to(device)
        leaf_gf = torch.from_numpy(obs["global_features"][leaf_indices]).to(device)
        leaf_ht = torch.from_numpy(obs["history_tokens"][leaf_indices]).to(device)
        leaf_mask = torch.from_numpy(obs["legal_mask"][leaf_indices]).to(device)

        with torch.no_grad(), torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            if leaf_eval == "critic" and q_critic_net is not None:
                leaf_oh = torch.from_numpy(obs["opp_hand_counts"][leaf_indices]).to(device)
                q_leaf = q_critic_net(leaf_cf, leaf_oh, leaf_gf, leaf_ht)
                logits_leaf, _ = actor_net(leaf_cf, leaf_gf, leaf_ht)
                probs_leaf = masked_log_softmax(logits_leaf.float(), leaf_mask).exp()
                v_leaf = (probs_leaf * q_leaf).sum(dim=-1).cpu().numpy()
            else:
                _, wdl_leaf = actor_net(leaf_cf, leaf_gf, leaf_ht)
                v_leaf = wdl_expected_value(wdl_leaf.float()).cpu().numpy()

        curr_p = obs["current_player"][leaf_indices]
        signs = np.where(curr_p == root_player, 1.0, -1.0).astype(np.float32)
        root_values[leaf_indices] = signs * v_leaf.astype(np.float32)

    # 7. Exact Closed-Form Tabular MMD Policy Update
    root_values_t = torch.from_numpy(root_values).to(device)
    root_values_2d = root_values_t.view(num_a, K_eff)
    q_hat_legal = root_values_2d.mean(dim=-1)  # [num_a] on device

    q_hat = torch.full((61,), -1e9, dtype=torch.float32, device=device)
    for a_idx, a_id in enumerate(legal_actions):
        q_hat[a_id] = q_hat_legal[a_idx]

    denom = 1.0 + eta * tau
    search_logits = (bp_log_p + (eta * tau) * mag_log_p + eta * q_hat) / denom
    search_probs = masked_log_softmax(search_logits, legal_mask_t).exp()

    if deterministic:
        chosen_action = int(search_probs.argmax().item())
    else:
        chosen_action = int(torch.multinomial(search_probs, 1).item())

    latency_ms = (time.perf_counter() - t0) * 1000.0

    return SearchResult(
        action=chosen_action,
        search_probs=search_probs,
        bp_probs=bp_probs,
        q_hat=q_hat,
        belief_xray=belief_xray,
        latency_ms=latency_ms,
    )
