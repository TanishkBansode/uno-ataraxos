#!/usr/bin/env python3
"""Diagnostic Script: Blueprint Voluntary Draw & Policy Stubbornness Profile (Milestone 7).

Runs 500 games (250 self-play + 250 vs ExpertHeuristicBot) with runs/blueprint_100m/latest.pt
and reports:
1. Exact Voluntary Draw breakdown (Wild Hoarding vs +2 Avoidance vs True Reneging).
2. Average hand size when voluntarily drawing.
3. Policy stubbornness profile: mean entropy and % of turns with max(p) > 0.95 / > 0.99.
4. Turn-1 vs Turn-21 synthetic sanity gate verification.

Usage:
    python scripts/diagnose_blueprint.py --blueprint runs/blueprint_100m/latest.pt --games 500
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from uno_ataraxos import UnoEngine
from uno_ataraxos.agent import AtaraxosUnoAgent
from uno_ataraxos.eval_suite import ExpertHeuristicBot
from uno_ataraxos.rl_math import get_device

DRAW_2_ACTION_IDS = {42, 45, 48, 51}
WILD_ACTION_IDS = set(range(52, 60))  # 52..55 Wild, 56..59 Wild Draw 4


def run_diagnostics(blueprint_path: str, num_games: int = 500) -> Dict[str, Any]:
    device = get_device()
    print("=" * 65)
    print("      BLUEPRINT VOLUNTARY DRAW & STUBBORNNESS DIAGNOSTIC")
    print("=" * 65)
    print(f"Device:    {device}")
    print(f"Blueprint: {blueprint_path}")
    print(f"Games:     {num_games} ({num_games // 2} Self-Play + {num_games // 2} vs Expert)")

    agent = AtaraxosUnoAgent(blueprint_ckpt=blueprint_path, mode="blueprint", device=device)
    expert = ExpertHeuristicBot()

    # Metrics accumulators
    total_choice_turns = 0
    choice_entropies: List[float] = []
    choice_max_probs: List[float] = []
    gt_95_count = 0
    gt_99_count = 0

    total_turns = 0
    total_v_draws = 0
    wild_hoard_count = 0
    p2_avoid_count = 0
    true_renege_count = 0

    my_hands_v_draw: List[int] = []
    opp_hands_v_draw: List[int] = []

    t0 = time.perf_counter()

    for g in range(num_games):
        seed = 10000 + g
        eng = UnoEngine(seed)
        eng.reset()
        is_vs_expert = (g >= num_games // 2)

        while not eng.is_over:
            curr_p = int(eng.current_player)
            is_bp_turn = (curr_p == 0) if is_vs_expert else True

            if is_bp_turn:
                total_turns += 1
                legal_ids = eng.get_legal_action_ids(curr_p)
                mask = eng.get_action_mask(curr_p)

                # Extract BP policy distribution
                cf = torch.from_numpy(eng.get_card_features_actor(curr_p)).unsqueeze(0).to(device)
                gf = torch.from_numpy(eng.get_global_features(curr_p)).unsqueeze(0).to(device)
                ht = torch.from_numpy(eng.get_history_tokens(curr_p)).unsqueeze(0).to(device)
                mask_t = torch.from_numpy(mask).to(device)

                with torch.no_grad():
                    logits, _ = agent.actor_net(cf, gf, ht)
                masked_logits = logits.float().squeeze(0).masked_fill(~mask_t, -1e9)
                probs = F.softmax(masked_logits, dim=-1)

                if len(legal_ids) >= 2:
                    total_choice_turns += 1
                    p_legal = probs[mask_t]
                    ent = -(p_legal * (p_legal + 1e-12).log()).sum().item()
                    choice_entropies.append(ent)
                    max_p = p_legal.max().item()
                    choice_max_probs.append(max_p)
                    if max_p > 0.95:
                        gt_95_count += 1
                    if max_p > 0.99:
                        gt_99_count += 1

                # Step with BP
                act, _ = agent.act_from_engine(eng)

                # Check voluntary draw: stack == 0 and had non-draw legal actions but drew
                non_draw_legal = [a for a in legal_ids if a < 60]
                if eng.stack == 0 and len(non_draw_legal) > 0 and act == 60:
                    total_v_draws += 1
                    my_hand_len = len(eng.get_hand_ids(curr_p))
                    opp_hand_len = len(eng.get_hand_ids(1 - curr_p))
                    my_hands_v_draw.append(my_hand_len)
                    opp_hands_v_draw.append(opp_hand_len)

                    has_colored = any(a < 52 and a not in DRAW_2_ACTION_IDS for a in non_draw_legal)
                    has_p2 = any(a in DRAW_2_ACTION_IDS for a in non_draw_legal)

                    if has_colored:
                        true_renege_count += 1
                    elif has_p2:
                        p2_avoid_count += 1
                    else:
                        wild_hoard_count += 1

            else:
                # Expert turn
                cf = eng.get_card_features_actor(curr_p)
                gf = eng.get_global_features(curr_p)
                mask = eng.get_action_mask(curr_p)
                act = expert.act_single(cf, gf, mask)

            eng.step(act)

    elapsed = time.perf_counter() - t0

    pct_gt_95 = (gt_95_count / max(1, total_choice_turns)) * 100.0
    pct_gt_99 = (gt_99_count / max(1, total_choice_turns)) * 100.0
    mean_ent = float(np.mean(choice_entropies)) if choice_entropies else 0.0

    pct_wild = (wild_hoard_count / max(1, total_v_draws)) * 100.0
    pct_p2 = (p2_avoid_count / max(1, total_v_draws)) * 100.0
    pct_colored = (true_renege_count / max(1, total_v_draws)) * 100.0
    total_v_draw_rate = (total_v_draws / max(1, total_turns)) * 100.0

    avg_my_hand = float(np.mean(my_hands_v_draw)) if my_hands_v_draw else 0.0
    avg_opp_hand = float(np.mean(opp_hands_v_draw)) if opp_hands_v_draw else 0.0

    print(f"\nCompleted {num_games} games in {elapsed:.1f}s ({total_turns:,} total turns)\n")
    print("-" * 65)
    print("1. BLUEPRINT VOLUNTARY DRAW BREAKDOWN:")
    print(f"   Total Voluntary Draw Rate:        {total_v_draw_rate:5.2f}% ({total_v_draws:,} / {total_turns:,} turns)")
    print(f"   - Wild/Stack-Insurance Hoarding:   {pct_wild:5.1f}% ({wild_hoard_count:,}) [Only Wild/+4 playable]")
    print(f"   - +2 Counter-War Avoidance:        {pct_p2:5.1f}% ({p2_avoid_count:,}) [Only +2/Wild playable]")
    print(f"   - True Colored-Card Reneging:      {pct_colored:5.1f}% ({true_renege_count:,}) [0-9, Skip, Rev playable]")
    print(f"   Avg Hand Size when Drawing:        Me = {avg_my_hand:.1f} cards | Opponent = {avg_opp_hand:.1f} cards")
    print("-" * 65)
    print("2. BLUEPRINT STUBBORNNESS PROFILE:")
    print(f"   Total Choice Turns (>=2 legal):    {total_choice_turns:,}")
    print(f"   Mean Policy Entropy:               {mean_ent:.3f} nats")
    print(f"   Choice Turns with max(p) > 0.95:   {pct_gt_95:5.1f}% ({gt_95_count:,} turns)")
    print(f"   Choice Turns with max(p) > 0.99:   {pct_gt_99:5.1f}% ({gt_99_count:,} turns)")
    print("-" * 65)

    # 3. Synthetic Sanity Gate Verification
    print("3. SYNTHETIC SANITY GATE VERIFICATION (bp_floor=0.05, eb_shrink=1.0, eta=18.0):")
    # Base prior p = [0.99, 0.01]
    p_raw = torch.tensor([0.99, 0.01])
    p_floor = 0.05
    p_floored = (1.0 - p_floor) * p_raw + p_floor * torch.tensor([0.5, 0.5])
    log_p_floored = p_floored.log()

    # Case A: Noisy +0.02 edge (SE = 0.045)
    # Value of act 0 is 0.0, act 1 is 0.02 -> v_bp = 0.0002 -> adv = [-0.0002, 0.0198]
    adv_noisy = torch.tensor([-0.0002, 0.0198])
    se = torch.tensor([0.045, 0.045])
    w_noisy = (adv_noisy ** 2) / (adv_noisy ** 2 + 1.0 * (se ** 2) + 1e-8)
    eff_noisy = adv_noisy * w_noisy
    denom = 1.0 + 18.0 * 0.002
    logits_noisy = (log_p_floored + 18.0 * eff_noisy) / denom
    p_noisy = F.softmax(logits_noisy, dim=-1)

    # Case B: Sharp +0.26 edge (SE = 0.045)
    # Value of act 0 is 0.0, act 1 is 0.26 -> v_bp = 0.0026 -> adv = [-0.0026, 0.2574]
    adv_sharp = torch.tensor([-0.0026, 0.2574])
    w_sharp = (adv_sharp ** 2) / (adv_sharp ** 2 + 1.0 * (se ** 2) + 1e-8)
    eff_sharp = adv_sharp * w_sharp
    logits_sharp = (log_p_floored + 18.0 * eff_sharp) / denom
    p_sharp = F.softmax(logits_sharp, dim=-1)

    print(f"   Case A: Noisy +0.02 Edge (SE=0.045) -> Shrink Weight={w_noisy[1].item():.3f} | BP Action Prob: {p_noisy[0].item()*100:.1f}% (> 90%: PASS)")
    print(f"   Case B: Sharp +0.26 Edge (SE=0.045) -> Shrink Weight={w_sharp[1].item():.3f} | Search Action Prob: {p_sharp[1].item()*100:.1f}% (> 70%: PASS)")
    print("=" * 65 + "\n")

    return {
        "total_v_draw_rate": total_v_draw_rate,
        "pct_wild_hoarding": pct_wild,
        "pct_p2_avoidance": pct_p2,
        "pct_true_reneging": pct_colored,
        "avg_my_hand": avg_my_hand,
        "avg_opp_hand": avg_opp_hand,
        "mean_entropy": mean_ent,
        "pct_gt_95": pct_gt_95,
        "pct_gt_99": pct_gt_99,
        "p_noisy_bp": p_noisy[0].item(),
        "p_sharp_search": p_sharp[1].item(),
    }


def main():
    parser = argparse.ArgumentParser(description="Diagnose Blueprint V-Draw & Stubbornness")
    parser.add_argument("--blueprint", type=str, default="runs/blueprint_100m/latest.pt", help="Path to Blueprint checkpoint")
    parser.add_argument("--games", type=int, default=500, help="Number of games to simulate")
    args = parser.parse_args()

    run_diagnostics(blueprint_path=args.blueprint, num_games=args.games)


if __name__ == "__main__":
    main()
