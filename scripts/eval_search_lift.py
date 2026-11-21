#!/usr/bin/env python3
"""Duplicate-Deck Search-Lift Evaluation Benchmark (Milestone 5).

Runs mirrored-deck evaluation comparing:
1. Tier-1 MMD Search vs Pure Blueprint
2. Hybrid (Tier-1 + Tier-2 Endgame PDCFR+) vs Pure Blueprint
3. Hybrid vs ExpertHeuristicBot

Reports exact duplicate win rates, payoffs, and latency ms/move (mean & p95).

Usage:
    python scripts/eval_search_lift.py --num-decks 100 --blueprint runs/sanity_3m/latest.pt --belief runs/belief_smoke/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from uno_ataraxos import UnoEngine, VecUnoEnv
from uno_ataraxos.agent import AtaraxosUnoAgent
from uno_ataraxos.eval_suite import ExpertHeuristicBot
from uno_ataraxos.rl_math import get_device


def run_duplicate_match(
    name_a: str,
    act_fn_a: Callable[[UnoEngine, int], int],
    name_b: str,
    act_fn_b: Callable[[UnoEngine, int], int],
    num_decks: int = 100,
    base_seed: int = 20000,
    measure_latencies_a: bool = False,
    agent_a: Optional[AtaraxosUnoAgent] = None,
) -> Dict[str, Any]:
    """Run mirrored-deck tournament between Policy A and Policy B."""
    total_games = 2 * num_decks
    wins_a = 0
    draws = 0
    losses_a = 0
    turns_list: List[int] = []
    latencies_a: List[float] = []

    print(f"\nEvaluating: {name_a} vs {name_b} ({num_decks} decks = {total_games} mirrored games)...")
    t0_match = time.perf_counter()

    for d in range(num_decks):
        deck_seed = base_seed + d * 10007

        # --- Game 1: Policy A = Seat 0, Policy B = Seat 1 ---
        eng1 = UnoEngine(deck_seed)
        eng1.reset()
        if agent_a is not None:
            agent_a.reset()

        while not eng1.is_over:
            curr_p = int(eng1.current_player)
            if curr_p == 0:
                t0_step = time.perf_counter()
                act = act_fn_a(eng1, curr_p)
                if measure_latencies_a:
                    latencies_a.append((time.perf_counter() - t0_step) * 1000.0)
            else:
                act = act_fn_b(eng1, curr_p)
            eng1.step(act)

        if eng1.winner == 0:
            wins_a += 1
        elif eng1.winner == 1:
            losses_a += 1
        else:
            draws += 1
        turns_list.append(eng1.turn_count)

        # --- Game 2 (Mirrored): Policy B = Seat 0, Policy A = Seat 1 ---
        eng2 = UnoEngine(deck_seed)
        eng2.reset()
        if agent_a is not None:
            agent_a.reset()

        while not eng2.is_over:
            curr_p = int(eng2.current_player)
            if curr_p == 1:
                t0_step = time.perf_counter()
                act = act_fn_a(eng2, curr_p)
                if measure_latencies_a:
                    latencies_a.append((time.perf_counter() - t0_step) * 1000.0)
            else:
                act = act_fn_b(eng2, curr_p)
            eng2.step(act)

        if eng2.winner == 1:
            wins_a += 1
        elif eng2.winner == 0:
            losses_a += 1
        else:
            draws += 1
        turns_list.append(eng2.turn_count)

        if (d + 1) % max(1, num_decks // 5) == 0 or (d + 1) == num_decks:
            played = (d + 1) * 2
            curr_wr = (wins_a + 0.5 * draws) / played * 100.0
            print(f"  Deck {d+1:3d}/{num_decks} | Played {played:3d} | Wins: {wins_a} | Losses: {losses_a} | WinRate: {curr_wr:5.1f}%")

    total_time = time.perf_counter() - t0_match
    win_rate_a = (wins_a + 0.5 * draws) / total_games * 100.0
    mean_lat = float(np.mean(latencies_a)) if latencies_a else 0.0
    p95_lat = float(np.percentile(latencies_a, 95)) if latencies_a else 0.0

    print(f"Result: {name_a} win rate = {win_rate_a:.1f}% ({wins_a}W - {draws}D - {losses_a}L) in {total_time:.1f}s")
    if measure_latencies_a:
        print(f"  Latency for {name_a}: mean = {mean_lat:.2f} ms/move, p95 = {p95_lat:.2f} ms/move")

    return {
        "name_a": name_a,
        "name_b": name_b,
        "win_rate_a": win_rate_a,
        "wins_a": wins_a,
        "draws": draws,
        "losses_a": losses_a,
        "avg_turns": float(np.mean(turns_list)),
        "mean_latency_ms": mean_lat,
        "p95_latency_ms": p95_lat,
        "total_time_s": total_time,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate search lift in duplicate-deck matches")
    parser.add_argument("--num-decks", type=int, default=100, help="Number of distinct initial decks (x2 mirrored games)")
    parser.add_argument("--blueprint", type=str, default="runs/sanity_3m/latest.pt", help="Path to blueprint checkpoint")
    parser.add_argument("--belief", type=str, default="runs/belief_smoke/best.pt", help="Path to belief checkpoint")
    parser.add_argument("--critic", type=str, default=None, help="Path to critic checkpoint")
    parser.add_argument("--K", type=int, default=128, help="Number of belief worlds")
    parser.add_argument("--depth", type=int, default=12, help="Rollout depth")
    parser.add_argument("--seed", type=int, default=20000, help="Base random seed")
    parser.add_argument("--quick", action="store_true", help="Quick smoke check (25 decks)")
    args = parser.parse_args()

    num_decks = 25 if args.quick else args.num_decks
    device = get_device()

    print("=" * 65)
    print("      MILESTONE 5: SEARCH-LIFT DUPLICATE EVALUATION")
    print("=" * 65)
    print(f"Device: {device} | Decks: {num_decks} ({2*num_decks} games/matchup)")
    print(f"Blueprint: {args.blueprint}")
    print(f"Belief:    {args.belief}")

    # 1. Instantiate Agents
    print("\nInitializing agents...")
    agent_bp = AtaraxosUnoAgent(
        blueprint_ckpt=args.blueprint,
        mode="blueprint",
        device=device,
    )

    agent_mmd = AtaraxosUnoAgent(
        blueprint_ckpt=args.blueprint,
        belief_ckpt=args.belief,
        critic_ckpt=args.critic,
        mode="mmd_search",
        device=device,
        K=args.K,
        depth=args.depth,
    )

    agent_hybrid = AtaraxosUnoAgent(
        blueprint_ckpt=args.blueprint,
        belief_ckpt=args.belief,
        critic_ckpt=args.critic,
        mode="hybrid",
        device=device,
        K=args.K,
        depth=args.depth,
    )

    expert_bot = ExpertHeuristicBot()

    def act_bp(eng: UnoEngine, p: int) -> int:
        act, _ = agent_bp.act_from_engine(eng, deterministic=False)
        return act

    def act_mmd(eng: UnoEngine, p: int) -> int:
        act, _ = agent_mmd.act_from_engine(eng, deterministic=False)
        return act

    def act_hybrid(eng: UnoEngine, p: int) -> int:
        act, _ = agent_hybrid.act_from_engine(eng, deterministic=False)
        return act

    def act_expert(eng: UnoEngine, p: int) -> int:
        cf = eng.get_card_features_actor(p)
        gf = eng.get_global_features(p)
        mask = eng.get_action_mask(p)
        return expert_bot.act_single(cf, gf, mask)

    # Matchup 1: Tier-1 MMD Search vs Pure Blueprint
    res_mmd_vs_bp = run_duplicate_match(
        "Tier-1 MMD Search", act_mmd,
        "Pure Blueprint", act_bp,
        num_decks=num_decks,
        base_seed=args.seed,
        measure_latencies_a=True,
        agent_a=agent_mmd,
    )

    # Matchup 2: Hybrid vs Pure Blueprint
    res_hybrid_vs_bp = run_duplicate_match(
        "Hybrid (Tier-1 + Tier-2)", act_hybrid,
        "Pure Blueprint", act_bp,
        num_decks=num_decks,
        base_seed=args.seed + 1000,
        measure_latencies_a=True,
        agent_a=agent_hybrid,
    )

    # Matchup 3: Hybrid vs ExpertHeuristicBot
    res_hybrid_vs_expert = run_duplicate_match(
        "Hybrid (Tier-1 + Tier-2)", act_hybrid,
        "ExpertHeuristicBot", act_expert,
        num_decks=num_decks,
        base_seed=args.seed + 2000,
        measure_latencies_a=False,
        agent_a=agent_hybrid,
    )

    # Summary Report
    print("\n" + "=" * 65)
    print("                    SEARCH-LIFT SUMMARY REPORT")
    print("=" * 65)
    print(f"Tier-1 MMD Search vs Pure Blueprint:       {res_mmd_vs_bp['win_rate_a']:5.1f}% win rate (mean: {res_mmd_vs_bp['mean_latency_ms']:.1f} ms, p95: {res_mmd_vs_bp['p95_latency_ms']:.1f} ms)")
    print(f"Hybrid (Tier-1 + Tier-2) vs Pure Blueprint: {res_hybrid_vs_bp['win_rate_a']:5.1f}% win rate (mean: {res_hybrid_vs_bp['mean_latency_ms']:.1f} ms, p95: {res_hybrid_vs_bp['p95_latency_ms']:.1f} ms)")
    print(f"Hybrid vs ExpertHeuristicBot:              {res_hybrid_vs_expert['win_rate_a']:5.1f}% win rate")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
