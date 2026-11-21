"""Tier-2 Low-Card Endgame PDCFR+ Subgame Solver (Milestone 5).

Implements Depth-Limited 3-Ply Predictive Discounted CFR+ (PDCFR+) for
critical low-card endgames (my_hand <= 3 and opp_hand <= 3):

Algorithm:
1. Samples K=128 belief worlds and deduplicates into top M <= 16 distinct
   opponent hands with normalized posterior weights w_1..w_M.
2. Builds a compact 3-ply extensive-form subgame tree, resolving extra turns
   (Skip, Reverse) and stacking (+2, +4).
3. Evaluates all non-terminal depth-3 leaves in ONE single batched forward pass
   of actor_net + q_critic_net.
4. Solves the exact tabular subgame via 150 iterations of PDCFR+:
     alpha = 2.0, beta = 0.5, gamma = 2.0, R+ = max(0, R)
     predictive regret matching: sigma^t proportional to max(0, R^{t-1} + r^{t-1})
5. Anchors to blueprint policy (0.90 * cfr_avg_policy + 0.10 * bp_probs)
   and returns structured SearchResult with live Belief X-Ray.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import random
import time
from typing import Any, Dict, List, Optional, Tuple, Union

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
from uno_ataraxos.search import SearchResult, compute_belief_xray

DECK_MAX = [
    1, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    1, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    1, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    1, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    4, 4
]


@dataclass
class FlatPly2Node:
    player: int
    infoset_id: int
    actions: List[int]
    payoffs: np.ndarray          # [num_a2] float32 terminal/leaf payoffs for p0


@dataclass
class FlatPly1Node:
    player: int
    infoset_id: int
    actions: List[int]
    children: List[Union[float, FlatPly2Node]]


def solve_endgame_pdcfr(
    root_engine: UnoEngine,
    actor_net: UnoActorNet,
    belief_net: UnoBeliefNet,
    q_critic_net: Optional[UnoQCriticNet] = None,
    K: int = 128,
    M_max: int = 16,
    iterations: int = 150,
    alpha: float = 2.0,
    beta: float = 0.5,
    gamma: float = 2.0,
    blend_bp: float = 0.10,
    known_opp_hands: Optional[List[List[int]]] = None,
    deterministic: bool = False,
    seed: Optional[int] = None,
) -> SearchResult:
    """Solve the low-card endgame using ultra-fast 3-ply PDCFR+.

    Args:
        root_engine: C++ UnoEngine instance at the current root state.
        actor_net: Blueprint actor policy network.
        belief_net: Generative belief network for opponent hand sampling.
        q_critic_net: Optional centralized Q-critic network for leaf evaluation.
        K: Number of belief worlds to sample (default 128).
        M_max: Maximum number of distinct opponent hands (default 16).
        iterations: Number of PDCFR+ iterations (default 150).
        alpha: Positive regret discount parameter (default 2.0).
        beta: Negative regret discount parameter (default 0.5).
        gamma: Strategy accumulation discount parameter (default 2.0).
        blend_bp: Mixture weight for blueprint policy anchoring (default 0.10).
        deterministic: If True, selects argmax action.
        seed: Random seed.

    Returns:
        SearchResult containing the chosen action, policy distributions,
        diagnostics, and wall-clock latency.
    """
    t0 = time.perf_counter()
    device = next(actor_net.parameters()).device
    root_player = int(root_engine.current_player)
    opp_player = 1 - root_player

    legal_actions = root_engine.get_legal_action_ids(root_player)
    num_a = len(legal_actions)

    legal_mask_np = root_engine.get_action_mask(root_player)
    legal_mask_t = torch.from_numpy(legal_mask_np).to(device)

    # 0. Early exit if 1 or 0 actions legal
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

    # 1. Compute Blueprint Policy for Anchoring
    cf_np = root_engine.get_card_features_actor(root_player)
    gf_np = root_engine.get_global_features(root_player)
    ht_np = root_engine.get_history_tokens(root_player)

    cf_t = torch.from_numpy(cf_np).unsqueeze(0).to(device)
    gf_t = torch.from_numpy(gf_np).unsqueeze(0).to(device)
    ht_t = torch.from_numpy(ht_np).unsqueeze(0).to(device)

    device_type = "mps" if device.type == "mps" else ("cuda" if device.type == "cuda" else "cpu")
    with torch.no_grad(), torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        bp_logits, _ = actor_net(cf_t, gf_t, ht_t)

    bp_log_p = masked_log_softmax(bp_logits.float().squeeze(0), legal_mask_t)
    bp_probs = torch.where(legal_mask_t, bp_log_p.exp(), torch.zeros_like(bp_log_p))

    # 2. Sample K Opponent Hands & Deduplicate to Top M <= 16 (or use known_opp_hands)
    if known_opp_hands is not None:
        distinct_opp_hands = [list(h) for h in known_opp_hands]
        weights = [1.0 / len(distinct_opp_hands)] * len(distinct_opp_hands)
        opp_hand_size = len(distinct_opp_hands[0])
        belief_xray = {"p_plus_2": 0.0, "p_plus_4": 0.0, "p_red": 0.0, "p_green": 0.0, "p_blue": 0.0, "p_yellow": 0.0, "opp_hand_size": float(opp_hand_size)}
    else:
        opp_hand_size = len(root_engine.get_hand_ids(opp_player))
        sampled_opp = belief_net.sample_opp_hands(cf_t, gf_t, ht_t, K=K, temp=1.0, uniform_mix=0.03)
        belief_xray = compute_belief_xray(sampled_opp, opp_hand_size)

        # Deduplicate
        sampled_opp_np = sampled_opp.cpu().numpy()
        hand_counts_tuples = [tuple(sampled_opp_np[k]) for k in range(K)]
        counts = Counter(hand_counts_tuples)
        top_m = counts.most_common(M_max)

        total_weight = sum(cnt for _, cnt in top_m)
        weights = [cnt / total_weight for _, cnt in top_m]
        distinct_opp_hands = []
        for counts_tuple, _ in top_m:
            h = []
            for c, count in enumerate(counts_tuple):
                for _ in range(count):
                    h.append(c)
            distinct_opp_hands.append(h)

    # Prepare Draw Piles for each world
    my_hand_ids = root_engine.get_hand_ids(root_player)
    my_counts = [0] * 54
    for c in my_hand_ids:
        my_counts[c] += 1
    disc_counts = [0] * 54
    for c in root_engine.get_public_discard():
        disc_counts[c] += 1

    distinct_draw_piles = []
    for h in distinct_opp_hands:
        h_cnt = [0] * 54
        for c in h:
            h_cnt[c] += 1
        d_pile = []
        for c in range(54):
            rem = DECK_MAX[c] - my_counts[c] - disc_counts[c] - h_cnt[c]
            if rem > 0:
                d_pile.extend([c] * rem)
        rng_seed = (seed or 42) + len(distinct_draw_piles) * 10007
        rng = random.Random(rng_seed)
        rng.shuffle(d_pile)
        distinct_draw_piles.append(d_pile)

    M = len(distinct_opp_hands)

    # 3. Build 3-Ply Subgame Tree
    # Infoset registry: key -> id
    infoset_map: Dict[Tuple[Any, ...], int] = {}
    infoset_players: List[int] = []
    infoset_action_counts: List[int] = []

    def get_infoset_id(key: Tuple[Any, ...], p: int, n_act: int) -> int:
        if key not in infoset_map:
            new_id = len(infoset_map)
            infoset_map[key] = new_id
            infoset_players.append(p)
            infoset_action_counts.append(n_act)
            return new_id
        return infoset_map[key]

    root_infoset_id = get_infoset_id(("root",), root_player, num_a)

    # Leaves to evaluate
    pending_leaves: List[Tuple[FlatPly2Node, int, UnoEngine]] = []

    # branches[m][a_idx] -> Union[float, FlatPly1Node]
    branches: List[List[Union[float, FlatPly1Node]]] = [[0.0] * num_a for _ in range(M)]

    for m in range(M):
        h_m = distinct_opp_hands[m]
        d_m = distinct_draw_piles[m]

        for a_idx, a_0 in enumerate(legal_actions):
            eng_0 = root_engine.clone()
            eng_0.set_world(opp_player, h_m, d_m)
            eng_0.step(a_0)

            if eng_0.is_over:
                payoff = 1.0 if eng_0.winner == root_player else (-1.0 if eng_0.winner == opp_player else 0.0)
                branches[m][a_idx] = float(payoff)
                continue

            p1 = int(eng_0.current_player)
            legal_1 = eng_0.get_legal_action_ids(p1)
            info_key_1 = ("ply1", a_0, tuple(sorted(eng_0.get_hand_ids(p1)))) if p1 == opp_player else ("ply1_extra", a_0, tuple(sorted(eng_0.get_hand_ids(root_player))))
            id_1 = get_infoset_id(info_key_1, p1, len(legal_1))

            children_1: List[Union[float, FlatPly2Node]] = []
            for a_1 in legal_1:
                eng_1 = eng_0.clone()
                eng_1.step(a_1)

                if eng_1.is_over:
                    payoff_1 = 1.0 if eng_1.winner == root_player else (-1.0 if eng_1.winner == opp_player else 0.0)
                    children_1.append(float(payoff_1))
                    continue

                p2 = int(eng_1.current_player)
                legal_2 = eng_1.get_legal_action_ids(p2)
                info_key_2 = ("ply2", a_0, a_1, tuple(sorted(eng_1.get_hand_ids(p2))))
                id_2 = get_infoset_id(info_key_2, p2, len(legal_2))

                payoffs_2 = np.zeros(len(legal_2), dtype=np.float32)
                node_2 = FlatPly2Node(player=p2, infoset_id=id_2, actions=list(legal_2), payoffs=payoffs_2)
                children_1.append(node_2)

                for a2_idx, a_2 in enumerate(legal_2):
                    eng_2 = eng_1.clone()
                    eng_2.step(a_2)

                    if eng_2.is_over:
                        payoff_2 = 1.0 if eng_2.winner == root_player else (-1.0 if eng_2.winner == opp_player else 0.0)
                        payoffs_2[a2_idx] = float(payoff_2)
                    else:
                        # Depth-3 Leaf for neural network evaluation
                        pending_leaves.append((node_2, a2_idx, eng_2))

            node_1 = FlatPly1Node(player=p1, infoset_id=id_1, actions=list(legal_1), children=children_1)
            branches[m][a_idx] = node_1

    # 4. Batch Neural Leaf Evaluation
    if pending_leaves:
        n_leaves = len(pending_leaves)
        leaf_cf = np.empty((n_leaves, 54, 7), dtype=np.float32)
        leaf_gf = np.empty((n_leaves, 10), dtype=np.float32)
        leaf_ht = np.empty((n_leaves, 48, 5), dtype=np.int32)
        leaf_mask = np.empty((n_leaves, 61), dtype=bool)
        leaf_curr_p = np.empty(n_leaves, dtype=np.int32)

        for idx, (_, _, eng) in enumerate(pending_leaves):
            curr_p = int(eng.current_player)
            leaf_curr_p[idx] = curr_p
            leaf_cf[idx] = eng.get_card_features_actor(curr_p)
            leaf_gf[idx] = eng.get_global_features(curr_p)
            leaf_ht[idx] = eng.get_history_tokens(curr_p)
            leaf_mask[idx] = eng.get_action_mask(curr_p)

        l_cf_t = torch.from_numpy(leaf_cf).to(device)
        l_gf_t = torch.from_numpy(leaf_gf).to(device)
        l_ht_t = torch.from_numpy(leaf_ht).to(device)
        l_mask_t = torch.from_numpy(leaf_mask).to(device)

        with torch.no_grad(), torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            if q_critic_net is not None:
                leaf_oh = np.empty((n_leaves, 54), dtype=np.float32)
                for idx, (_, _, eng) in enumerate(pending_leaves):
                    leaf_oh[idx] = eng.get_opp_hand_counts(int(eng.current_player))
                l_oh_t = torch.from_numpy(leaf_oh).to(device)
                q_leaf = q_critic_net(l_cf_t, l_oh_t, l_gf_t, l_ht_t)
                logits_leaf, _ = actor_net(l_cf_t, l_gf_t, l_ht_t)
                probs_leaf = masked_log_softmax(logits_leaf.float(), l_mask_t).exp()
                v_curr = (probs_leaf * q_leaf).sum(dim=-1).cpu().numpy()
            else:
                _, wdl_leaf = actor_net(l_cf_t, l_gf_t, l_ht_t)
                v_curr = wdl_expected_value(wdl_leaf.float()).cpu().numpy()

        for idx, (node_2, a2_idx, _) in enumerate(pending_leaves):
            sign = 1.0 if leaf_curr_p[idx] == root_player else -1.0
            node_2.payoffs[a2_idx] = float(sign * v_curr[idx])

    # 5. Flat Vectorized PDCFR+ Solver
    num_infosets = len(infoset_map)
    cum_regrets = [np.zeros(cnt, dtype=np.float32) for cnt in infoset_action_counts]
    prev_regrets = [np.zeros(cnt, dtype=np.float32) for cnt in infoset_action_counts]
    cum_strategy = [np.zeros(cnt, dtype=np.float32) for cnt in infoset_action_counts]

    weights_np = np.array(weights, dtype=np.float32)
    root_action_values = np.zeros(num_a, dtype=np.float32)

    for t in range(1, iterations + 1):
        # 5.1 Predictive strategy
        sigma = []
        for i in range(num_infosets):
            pred = np.maximum(0.0, cum_regrets[i] + prev_regrets[i])
            s = pred.sum()
            if s > 1e-9:
                sigma.append(pred / s)
            else:
                sigma.append(np.full(infoset_action_counts[i], 1.0 / infoset_action_counts[i], dtype=np.float32))

        # Instantaneous regrets accumulator
        inst_r = [np.zeros(cnt, dtype=np.float32) for cnt in infoset_action_counts]

        # 5.2 Tree Evaluation & Counterfactual Values
        root_sigma = sigma[root_infoset_id]

        for a_idx in range(num_a):
            val_a0 = 0.0
            pi_p0 = float(root_sigma[a_idx])

            for m in range(M):
                w_m = float(weights_np[m])
                pi_p1 = w_m
                br = branches[m][a_idx]

                if isinstance(br, float):
                    val_a0 += w_m * br
                else:
                    # Ply 1 Node
                    node_1 = br
                    s1 = sigma[node_1.infoset_id]
                    v1_children = np.zeros(len(node_1.actions), dtype=np.float32)

                    # Player at ply 1
                    p1 = node_1.player
                    # Reach prob from opponent's view
                    pi_opp_1 = pi_p1 if p1 == root_player else pi_p0

                    for a1_idx, ch1 in enumerate(node_1.children):
                        # Update reach probabilities after ply 1 action
                        if p1 == root_player:
                            pi_p0_next = pi_p0 * float(s1[a1_idx])
                            pi_p1_next = pi_p1
                        else:
                            pi_p0_next = pi_p0
                            pi_p1_next = pi_p1 * float(s1[a1_idx])

                        if isinstance(ch1, float):
                            v1_children[a1_idx] = ch1
                        else:
                            # Ply 2 Node
                            node_2 = ch1
                            s2 = sigma[node_2.infoset_id]
                            v2_children = node_2.payoffs
                            v2_node = float(np.dot(s2, v2_children))
                            v1_children[a1_idx] = v2_node

                            # Regret for Ply 2
                            p2 = node_2.player
                            pi_opp_2 = pi_p1_next if p2 == root_player else pi_p0_next
                            if pi_opp_2 > 1e-9:
                                diff = v2_children - v2_node
                                if p2 == root_player:
                                    inst_r[node_2.infoset_id] += pi_opp_2 * diff
                                else:
                                    inst_r[node_2.infoset_id] += pi_opp_2 * (-diff)

                    v1_node = float(np.dot(s1, v1_children))
                    val_a0 += w_m * v1_node

                    # Regret for Ply 1
                    if pi_opp_1 > 1e-9:
                        diff1 = v1_children - v1_node
                        if p1 == root_player:
                            inst_r[node_1.infoset_id] += pi_opp_1 * diff1
                        else:
                            inst_r[node_1.infoset_id] += pi_opp_1 * (-diff1)

            root_action_values[a_idx] = val_a0

        root_v = float(np.dot(root_sigma, root_action_values))
        inst_r[root_infoset_id] += (root_action_values - root_v)

        # 5.3 Discounted Regret Updates (DCFR+)
        t_alpha = (t ** alpha) / (t ** alpha + 1.0)
        t_beta = (t ** beta) / (t ** beta + 1.0)

        for i in range(num_infosets):
            R = cum_regrets[i]
            r = inst_r[i]
            discount = np.where(R > 0.0, t_alpha, t_beta)
            new_R = np.maximum(0.0, discount * R + r)
            cum_regrets[i] = new_R
            prev_regrets[i] = r

        # 5.4 Strategy Accumulation (gamma = 2.0)
        t_gamma = (t / (t + 1.0)) ** gamma
        for i in range(num_infosets):
            cum_strategy[i] += t_gamma * sigma[i]

    # 6. Extract Root Strategy & Blend with Blueprint
    root_strat_accum = cum_strategy[root_infoset_id]
    s_root = root_strat_accum.sum()
    if s_root > 1e-9:
        cfr_root_policy = root_strat_accum / s_root
    else:
        cfr_root_policy = np.full(num_a, 1.0 / num_a, dtype=np.float32)

    cfr_probs_np = np.zeros(61, dtype=np.float32)
    for a_idx, a_id in enumerate(legal_actions):
        cfr_probs_np[a_id] = cfr_root_policy[a_idx]

    cfr_probs = torch.from_numpy(cfr_probs_np).to(device)

    # Blend: (1 - blend_bp) * cfr + blend_bp * bp
    blended_probs = (1.0 - blend_bp) * cfr_probs + blend_bp * bp_probs
    blended_probs = torch.where(legal_mask_t, blended_probs, torch.zeros_like(blended_probs))
    blended_probs = blended_probs / blended_probs.sum().clamp_min(1e-9)

    # Construct Q-hat estimate from root children counterfactual values
    q_hat = torch.full((61,), -1e9, dtype=torch.float32, device=device)
    for a_idx, a_id in enumerate(legal_actions):
        q_hat[a_id] = float(root_action_values[a_idx])

    if deterministic:
        chosen_action = int(blended_probs.argmax().item())
    else:
        chosen_action = int(torch.multinomial(blended_probs, 1).item())

    latency_ms = (time.perf_counter() - t0) * 1000.0

    return SearchResult(
        action=chosen_action,
        search_probs=blended_probs,
        bp_probs=bp_probs,
        q_hat=q_hat,
        belief_xray=belief_xray,
        latency_ms=latency_ms,
    )
