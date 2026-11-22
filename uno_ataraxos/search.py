"""Ataraxos Update-Equivalence Tabular MMD Search (Milestone 7 Upgraded).

Implements exact Maximum Entropy Mirror Descent (MMD) policy improvement
over Common Random Numbers (CRN) paired belief worlds:

    pi_search(a) proportional to [ pi_bp(a) * U(a)^(eta * tau) * exp(eta * Q_eff(a)) ]^(1 / (1 + eta * tau))

Research Upgrades (Ataraxos arXiv:2511.07312v2 Tables S21 & S31):
1. rollout_temp: Deterministic argmax rollouts (temp=0.0) or softmax sampling (temp>0.0).
2. bp_floor: Uniform legal mixture floor capping stubborn 99.9% blueprint lockouts without blurring 60/40 mixed turns.
3. eb_shrink: Paired Empirical-Bayes signal/noise shrinkage filter over CRN worlds.
4. Adaptive world budgeting up to 256 paired worlds on low-branching turns.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
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
    search_probs: torch.Tensor              # [61] float32 on device
    bp_probs: torch.Tensor                  # [61] float32 on device
    q_hat: torch.Tensor                     # [61] float32 on device (-1e9 on illegal)
    belief_xray: Dict[str, float]           # Live diagnostics P(+2), P(+4), colors, etc.
    latency_ms: float                       # Total wall-clock time in ms
    adv_se: Optional[torch.Tensor] = None   # [61] paired standard errors across CRN worlds
    ply1_opp_pred: Optional[torch.Tensor] = None  # [3, 61] marginalized ply-1 opponent action predictions


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
    depth: int = 14,
    eta: float = 3.0,
    tau: float = 0.05,
    rollout_temp: float = 0.0,
    bp_floor: float = 0.03,
    eb_shrink: float = 1.0,
    uniform_mix: float = 0.03,
    max_total_worlds: int = 768,
    leaf_eval: str = "critic",
    deterministic: bool = False,
    seed: Optional[int] = None,
    use_portfolio: bool = False,
    portfolio_weights: Tuple[float, float, float] = (0.50, 0.30, 0.20),
    risk_gamma: float = 4.0,
    belief_temp: float = 1.0,
    return_ply1_opp_pred: bool = False,
) -> SearchResult:
    """Run Tier-1 Ataraxos Update-Equivalence Tabular MMD Search.

    Args:
        root_engine: C++ UnoEngine instance at the current root state.
        actor_net: Blueprint actor policy network.
        belief_net: Generative belief network for opponent hand sampling.
        q_critic_net: Optional centralized Q-critic network for leaf evaluation.
        rollout_pool: Persistent C++ VecUnoEnv(auto_reset=False) instance.
        K: Target number of belief worlds (default 128).
        depth: Maximum rollout depth in plies (default 14).
        eta: MMD learning rate / inverse temperature (default 3.0).
        tau: Uniform legal policy magnet weight (default 0.05).
        rollout_temp: Rollout policy temperature (0.0 = deterministic argmax).
        bp_floor: Uniform mixture floor over legal actions to cap stubborn lockouts (default 0.03).
        eb_shrink: Empirical-Bayes signal/noise shrinkage coefficient (default 1.0).
        uniform_mix: Safety mixture weight inside belief sampler (default 0.03).
        max_total_worlds: Maximum product of num_a * K_eff to cap latency (default 768).
        leaf_eval: Leaf evaluation mode, "critic" or "wdl" (default "critic").
        deterministic: If True, selects argmax action; else samples from search_probs.
        seed: Random seed for world generation and rollouts.
        use_portfolio: If True, simulates 3-style opponent portfolio during rollouts.
        portfolio_weights: Prior/posterior weights over (Blueprint, Aggressive, Noisy).
        risk_gamma: Soft-Minimax entropic risk aversion parameter (0.0 = weighted average).
        belief_temp: Softmax temperature for belief opponent hand sampling.
        return_ply1_opp_pred: If True, computes and caches marginalized ply-1 opponent action predictions.

    Returns:
        SearchResult containing the chosen action, policy distributions, Q-values,
        adv_se, Belief X-Ray diagnostics, latency, and optional ply1_opp_pred.
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
        se_zeros = torch.zeros(61, dtype=torch.float32, device=device)
        xray_dummy = {"p_plus_2": 0.0, "p_plus_4": 0.0, "p_red": 0.0, "p_green": 0.0, "p_blue": 0.0, "p_yellow": 0.0, "opp_hand_size": 0.0}
        return SearchResult(
            action=chosen_act,
            search_probs=single_p,
            bp_probs=single_p,
            q_hat=q_zeros,
            belief_xray=xray_dummy,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            adv_se=se_zeros,
            ply1_opp_pred=None,
        )

    # Extract tensors for root player
    cf_np = root_engine.get_card_features_actor(root_player)
    gf_np = root_engine.get_global_features(root_player)
    ht_np = root_engine.get_history_tokens(root_player)

    cf_t = torch.from_numpy(cf_np).unsqueeze(0).to(device)
    gf_t = torch.from_numpy(gf_np).unsqueeze(0).to(device)
    ht_t = torch.from_numpy(ht_np).unsqueeze(0).to(device)

    # 2. Compute Blueprint and Uniform Magnet Log-Probabilities with bp_floor
    device_type = "mps" if device.type == "mps" else ("cuda" if device.type == "cuda" else "cpu")
    with torch.no_grad(), torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        bp_logits, _ = actor_net(cf_t, gf_t, ht_t)

    bp_probs_raw = F.softmax(bp_logits.float().squeeze(0).masked_fill(~legal_mask_t, -1e9), dim=-1)
    num_legal = legal_mask_t.sum().clamp_min(1)
    unif_probs = torch.where(legal_mask_t, 1.0 / num_legal.float(), torch.zeros_like(bp_probs_raw))

    if bp_floor > 0.0:
        bp_probs_floored = (1.0 - bp_floor) * bp_probs_raw + bp_floor * unif_probs
        neg_large = torch.tensor(-1e9, dtype=torch.float32, device=device)
        bp_log_p = torch.where(legal_mask_t, bp_probs_floored.clamp_min(1e-12).log(), neg_large)
    else:
        bp_probs_floored = bp_probs_raw
        bp_log_p = masked_log_softmax(bp_logits.float().squeeze(0), legal_mask_t)

    bp_probs = bp_probs_floored
    mag_log_p = uniform_legal_log_probs(legal_mask_t)

    # 3. Compute Adaptive World Budget K_eff & Sample Opponent Hands
    K_eff = min(256, max(32, max_total_worlds // max(1, num_a)))
    opp_hand_size = int(round(gf_np[1] * 20.0))
    sampled_opp = belief_net.sample_opp_hands(
        cf_t, gf_t, ht_t, K=K_eff, temp=belief_temp, uniform_mix=uniform_mix
    )  # [K_eff, 54] int32 on device
    belief_xray = compute_belief_xray(sampled_opp, opp_hand_size)

    # 3b. Deterministic World Style Assignment for Portfolio
    if use_portfolio:
        pw_sum = max(1e-6, sum(portfolio_weights))
        pw_norm = [w / pw_sum for w in portfolio_weights]
        k0 = max(4, int(round(pw_norm[0] * K_eff)))
        k1 = max(4, int(round(pw_norm[1] * K_eff)))
        k2 = max(4, K_eff - k0 - k1)
        diff = (k0 + k1 + k2) - K_eff
        if diff != 0:
            k0 = max(4, k0 - diff)
        world_styles_np = np.array([0] * k0 + [1] * k1 + [2] * (K_eff - k0 - k1), dtype=np.int32)
    else:
        world_styles_np = np.zeros(K_eff, dtype=np.int32)

    total_slots = num_a * K_eff
    slot_styles_np = np.tile(world_styles_np, num_a)

    # 4. Fast C++ Batched World Loader with Common Random Numbers
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

    ply1_opp_preds_per_action = torch.zeros((num_a, 3, 61), dtype=torch.float32, device=device)

    # 5. Fast Rollout Loop (ply = 1 .. depth) with Portfolio Support
    for ply in range(1, depth):
        if len(active_indices) == 0:
            break

        act_cf = torch.from_numpy(obs["card_features_actor"][active_indices]).to(device)
        act_gf = torch.from_numpy(obs["global_features"][active_indices]).to(device)
        act_ht = torch.from_numpy(obs["history_tokens"][active_indices]).to(device)
        act_mask = torch.from_numpy(obs["legal_mask"][active_indices]).to(device)

        with torch.no_grad(), torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits_step, _ = actor_net(act_cf, act_gf, act_ht)

        masked_logits = logits_step.masked_fill(~act_mask, -1e4)

        # Compute base greedy action for all envs on device
        base_actions_t = masked_logits.argmax(dim=-1).to(torch.int32)
        is_opp = (obs["current_player"][active_indices] != root_player)

        if not use_portfolio or not is_opp.any():
            if rollout_temp <= 0.0 or not is_opp.any():
                sampled_actions = base_actions_t.cpu().numpy()
            else:
                actions_t = base_actions_t.clone()
                idx_o = np.where(is_opp)[0]
                idx_o_t = torch.from_numpy(idx_o).to(device)
                p_step = F.softmax(masked_logits[idx_o].float() / rollout_temp, dim=-1)
                actions_t[idx_o_t] = torch.multinomial(p_step, num_samples=1).squeeze(-1).to(torch.int32)
                sampled_actions = actions_t.cpu().numpy()
        else:
            actions_t = base_actions_t.clone()
            idx_o = np.where(is_opp)[0]
            opp_env_slots = active_indices[idx_o]
            opp_styles = slot_styles_np[opp_env_slots]

            # Style 1: Aggressive No-VDraw (never draws voluntarily if any card is playable)
            m1 = (opp_styles == 1)
            if m1.any():
                idx1 = idx_o[m1]
                idx1_t = torch.from_numpy(idx1).to(device)
                agg_mask = act_mask[idx1].clone()
                can_play = agg_mask[:, :60].any(dim=-1)
                agg_mask[can_play, 60] = False
                agg_logits = logits_step[idx1].masked_fill(~agg_mask, -1e4)
                actions_t[idx1_t] = agg_logits.argmax(dim=-1).to(torch.int32)

            # Style 2: Unpredictable High-Temp (softmax with temp=2.0)
            m2 = (opp_styles == 2)
            if m2.any():
                idx2 = idx_o[m2]
                idx2_t = torch.from_numpy(idx2).to(device)
                p_noisy = F.softmax(masked_logits[idx2].float() / 2.0, dim=-1)
                actions_t[idx2_t] = torch.multinomial(p_noisy, num_samples=1).squeeze(-1).to(torch.int32)

            # Single host-device copy per ply
            sampled_actions = actions_t.cpu().numpy()

        # Cache marginalized Ply-1 Opponent Predictions for OnlineOpponentTracker (ZERO extra forward pass!)
        if ply == 1 and return_ply1_opp_pred and is_opp.any():
            with torch.no_grad():
                idx_o_all = np.where(is_opp)[0]
                idx_o_all_t = torch.from_numpy(idx_o_all).to(device)
                c_logits = masked_logits[idx_o_all_t].float()
                c_mask = act_mask[idx_o_all_t]

                p0_all = F.softmax(c_logits, dim=-1)
                agg_c_mask = c_mask.clone()
                agg_c_mask[agg_c_mask[:, :60].any(dim=-1), 60] = False
                p1_all = F.softmax(logits_step[idx_o_all_t].float().masked_fill(~agg_c_mask, -1e4), dim=-1)
                p2_all = F.softmax(c_logits / 2.0, dim=-1)

                opp_slots = active_indices[idx_o_all]
                a_indices = opp_slots // K_eff
                for a_idx in range(num_a):
                    m_a = (a_indices == a_idx)
                    if m_a.any():
                        m_a_t = torch.from_numpy(m_a).to(device)
                        ply1_opp_preds_per_action[a_idx, 0] = p0_all[m_a_t].mean(dim=0)
                        ply1_opp_preds_per_action[a_idx, 1] = p1_all[m_a_t].mean(dim=0)
                        ply1_opp_preds_per_action[a_idx, 2] = p2_all[m_a_t].mean(dim=0)

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

    # 7. Soft-Minimax Risk-Averse Aggregation Across Styles & Closed-Form MMD Update
    root_values_t = torch.from_numpy(root_values).to(device)
    root_values_2d = root_values_t.view(num_a, K_eff)  # [num_a, K_eff]

    # Blueprint-expected baseline across candidate root actions per world
    bp_probs_legal = bp_probs[legal_actions]
    bp_probs_legal = bp_probs_legal / bp_probs_legal.sum().clamp_min(1e-9)

    if use_portfolio:
        q_style = torch.zeros((num_a, 3), dtype=torch.float32, device=device)
        for m in range(3):
            m_mask = torch.from_numpy(world_styles_np == m).to(device)
            if m_mask.any():
                q_style[:, m] = root_values_2d[:, m_mask].mean(dim=-1)
            else:
                q_style[:, m] = root_values_2d.mean(dim=-1)

        w_t = torch.tensor(portfolio_weights, dtype=torch.float32, device=device)
        w_t = w_t / w_t.sum()

        if risk_gamma > 0.0:
            q_min = q_style.min(dim=-1, keepdim=True).values
            q_rob = q_min.squeeze(-1) - (1.0 / risk_gamma) * torch.log(
                (w_t * torch.exp(-risk_gamma * (q_style - q_min))).sum(dim=-1).clamp_min(1e-12)
            )
        else:
            q_rob = (q_style * w_t).sum(dim=-1)

        adv_mean = q_rob - (bp_probs_legal * q_rob).sum()
    else:
        v_bp_k = (bp_probs_legal.unsqueeze(-1) * root_values_2d).sum(dim=0)  # [K_eff]
        adv_mat = root_values_2d - v_bp_k.unsqueeze(0)  # [num_a, K_eff]
        adv_mean = adv_mat.mean(dim=-1)  # [num_a]

    # Paired standard error across all K_eff worlds for empirical-bayes shrinkage
    v_bp_k = (bp_probs_legal.unsqueeze(-1) * root_values_2d).sum(dim=0)
    adv_mat = root_values_2d - v_bp_k.unsqueeze(0)
    if K_eff > 1:
        adv_se = adv_mat.std(dim=-1, unbiased=True) / math.sqrt(K_eff)
    else:
        adv_se = torch.zeros_like(adv_mean)

    if eb_shrink > 0.0:
        shrink_weight = (adv_mean ** 2) / (adv_mean ** 2 + eb_shrink * (adv_se ** 2) + 1e-8)
        effective_q = adv_mean * shrink_weight
    else:
        effective_q = adv_mean

    effective_q_61 = torch.full((61,), 0.0, dtype=torch.float32, device=device)
    q_hat = torch.full((61,), -1e9, dtype=torch.float32, device=device)
    adv_se_61 = torch.zeros(61, dtype=torch.float32, device=device)

    for a_idx, a_id in enumerate(legal_actions):
        effective_q_61[a_id] = effective_q[a_idx]
        q_hat[a_id] = root_values_2d[a_idx].mean()  # Exact true win-equity in [-1, 1]
        adv_se_61[a_id] = adv_se[a_idx]

    denom = 1.0 + eta * tau
    search_logits = (bp_log_p + (eta * tau) * mag_log_p + eta * effective_q_61) / denom
    search_probs = masked_log_softmax(search_logits, legal_mask_t).exp()

    if deterministic:
        chosen_action = int(search_probs.argmax().item())
    else:
        chosen_action = int(torch.multinomial(search_probs, 1).item())

    chosen_ply1_opp_pred = None
    if return_ply1_opp_pred and chosen_action in legal_actions:
        c_idx = int(np.where(legal_actions == chosen_action)[0][0])
        chosen_ply1_opp_pred = ply1_opp_preds_per_action[c_idx]

    latency_ms = (time.perf_counter() - t0) * 1000.0

    return SearchResult(
        action=chosen_action,
        search_probs=search_probs,
        bp_probs=bp_probs,
        q_hat=q_hat,
        belief_xray=belief_xray,
        latency_ms=latency_ms,
        adv_se=adv_se_61,
        ply1_opp_pred=chosen_ply1_opp_pred,
    )
