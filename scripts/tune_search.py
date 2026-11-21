#!/usr/bin/env python3
"""Search Hyperparameter Sweep & Tuning Utility (Milestone 6).

Sweeps:
- eta: MMD step size in [1.5, 3.0, 5.0]
- tau: Entropy regularization temperature in [0.02, 0.05]
- leaf_eval: Leaf evaluation model in ["critic", "wdl"]

Runs duplicate-deck tournaments against pure Blueprint, measures win rate and
latency, and writes the winning configuration to runs/search_config.json.

Usage:
    python scripts/tune_search.py --num-decks 50 --blueprint runs/sanity_3m/latest.pt --belief runs/belief_smoke/best.pt
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from uno_ataraxos import UnoEngine
from uno_ataraxos.agent import AtaraxosUnoAgent
from uno_ataraxos.rl_math import get_device


def evaluate_search_config(
    blueprint_path: str,
    belief_path: str,
    eta: float,
    tau: float,
    leaf_eval: str,
    num_decks: int = 50,
    base_seed: int = 30000,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Evaluate one search hyperparameter configuration against pure Blueprint."""
    if device is None:
        device = get_device()

    agent_search = AtaraxosUnoAgent(
        blueprint_ckpt=blueprint_path,
        belief_ckpt=belief_path,
        mode="hybrid",
        K=128,
        eta=eta,
        tau=tau,
        leaf_eval=leaf_eval,
        max_total_worlds=768,
        device=device,
    )

    agent_bp = AtaraxosUnoAgent(
        blueprint_ckpt=blueprint_path,
        mode="blueprint",
        device=device,
    )

    wins = 0
    draws = 0
    losses = 0
    latencies: List[float] = []
    total_games = 2 * num_decks

    t0_start = time.perf_counter()

    for d in range(num_decks):
        deck_seed = base_seed + d * 10007

        # Game 1: Search = Seat 0, BP = Seat 1
        eng1 = UnoEngine(deck_seed)
        eng1.reset()
        agent_search.reset()
        agent_bp.reset()

        while not eng1.is_over:
            curr_p = int(eng1.current_player)
            if curr_p == 0:
                t0_step = time.perf_counter()
                act, _ = agent_search.act_from_engine(eng1)
                latencies.append((time.perf_counter() - t0_step) * 1000.0)
            else:
                act, _ = agent_bp.act_from_engine(eng1)
            eng1.step(act)

        if eng1.winner == 0:
            wins += 1
        elif eng1.winner == 1:
            losses += 1
        else:
            draws += 1

        # Game 2 (Mirrored): BP = Seat 0, Search = Seat 1
        eng2 = UnoEngine(deck_seed)
        eng2.reset()
        agent_search.reset()
        agent_bp.reset()

        while not eng2.is_over:
            curr_p = int(eng2.current_player)
            if curr_p == 1:
                t0_step = time.perf_counter()
                act, _ = agent_search.act_from_engine(eng2)
                latencies.append((time.perf_counter() - t0_step) * 1000.0)
            else:
                act, _ = agent_bp.act_from_engine(eng2)
            eng2.step(act)

        if eng2.winner == 1:
            wins += 1
        elif eng2.winner == 0:
            losses += 1
        else:
            draws += 1

    total_time = time.perf_counter() - t0_start
    win_rate = (wins + 0.5 * draws) / total_games * 100.0
    mean_lat = float(np.mean(latencies)) if latencies else 0.0
    p95_lat = float(np.percentile(latencies, 95)) if latencies else 0.0

    return {
        "eta": eta,
        "tau": tau,
        "leaf_eval": leaf_eval,
        "win_rate": win_rate,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "mean_latency_ms": mean_lat,
        "p95_latency_ms": p95_lat,
        "total_time_s": total_time,
    }


def main():
    parser = argparse.ArgumentParser(description="Sweep search hyperparameters against pure Blueprint")
    parser.add_argument("--blueprint", type=str, default="runs/sanity_3m/latest.pt", help="Path to Blueprint checkpoint")
    parser.add_argument("--belief", type=str, default="runs/belief_smoke/best.pt", help="Path to Belief checkpoint")
    parser.add_argument("--num-decks", type=int, default=50, help="Number of decks per config (x2 mirrored games)")
    parser.add_argument("--out-config", type=str, default="runs/search_config.json", help="Output winning config path")
    parser.add_argument("--quick", action="store_true", help="Quick run with 15 decks and reduced grid")
    args = parser.parse_args()

    device = get_device()
    print("=" * 70)
    print("      MILESTONE 6: SEARCH HYPERPARAMETER SWEEP")
    print("=" * 70)
    print(f"Device: {device} | Blueprint: {args.blueprint} | Belief: {args.belief}")

    if args.quick:
        etas = [1.5, 3.0]
        taus = [0.02]
        leaf_evals = ["critic"]
        num_decks = 15
    else:
        etas = [1.5, 3.0, 5.0]
        taus = [0.02, 0.05]
        leaf_evals = ["critic", "wdl"]
        num_decks = args.num_decks

    grid = list(itertools.product(etas, taus, leaf_evals))
    print(f"Configurations to evaluate: {len(grid)} (Decks per config: {num_decks} = {2 * num_decks} games)")

    results: List[Dict[str, Any]] = []

    for idx, (eta, tau, leaf_eval) in enumerate(grid, 1):
        print(f"\n[{idx}/{len(grid)}] Evaluating: eta={eta}, tau={tau}, leaf_eval='{leaf_eval}' ...")
        res = evaluate_search_config(
            blueprint_path=args.blueprint,
            belief_path=args.belief,
            eta=eta,
            tau=tau,
            leaf_eval=leaf_eval,
            num_decks=num_decks,
            device=device,
        )
        print(f"    >> WR: {res['win_rate']:.1f}% ({res['wins']}W - {res['draws']}D - {res['losses']}L) | "
              f"Mean Lat: {res['mean_latency_ms']:.1f} ms | p95: {res['p95_latency_ms']:.1f} ms | Time: {res['total_time_s']:.1f}s")
        results.append(res)

    # Sort results by win_rate descending, then lowest mean latency
    results.sort(key=lambda r: (r["win_rate"], -r["mean_latency_ms"]), reverse=True)

    print("\n" + "=" * 75)
    print("                      SWEEP RESULTS RANKING")
    print("=" * 75)
    print(f"{'Rank':<5} | {'eta':<5} | {'tau':<5} | {'leaf_eval':<9} | {'Win Rate':<10} | {'W-D-L':<10} | {'Mean Lat (ms)':<14} | {'p95 Lat (ms)':<12}")
    print("-" * 75)
    for rank, r in enumerate(results, 1):
        wdl = f"{r['wins']}-{r['draws']}-{r['losses']}"
        print(f"{rank:<5} | {r['eta']:<5.1f} | {r['tau']:<5.2f} | {r['leaf_eval']:<9} | {r['win_rate']:<9.1f}% | {wdl:<10} | {r['mean_latency_ms']:<14.1f} | {r['p95_latency_ms']:<12.1f}")
    print("=" * 75)

    winner = results[0]
    winning_config = {
        "eta": winner["eta"],
        "tau": winner["tau"],
        "leaf_eval": winner["leaf_eval"],
        "max_total_worlds": 768,
        "K": 128,
        "win_rate_vs_blueprint": winner["win_rate"],
        "mean_latency_ms": winner["mean_latency_ms"],
        "p95_latency_ms": winner["p95_latency_ms"],
        "evaluated_decks": num_decks,
    }

    out_p = Path(args.out_config)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump(winning_config, f, indent=2)

    print(f"\nWinning configuration written to: {out_p.resolve()}")
    print(json.dumps(winning_config, indent=2))


if __name__ == "__main__":
    main()
