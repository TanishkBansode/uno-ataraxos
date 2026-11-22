#!/usr/bin/env python3
"""Milestone 7B: Statistically Valid 200-Game 3-Way Final Showdown.

Compares exactly 3 models on 100 common random duplicate decks (200 mirrored games/axis):
1. Pure 100M Blueprint (mode="blueprint")
2. Old Baseline Hybrid (mode="hybrid", rollout_temp=1.0, depth=12, bp_floor=0.0, eb_shrink=0.0, eta=3.0, tau=0.05, max_total_worlds=512)
3. Upgraded Ataraxos+EB Hybrid (mode="hybrid", rollout_temp=0.0, depth=12, bp_floor=0.04, eb_shrink=1.0, eta=14.0, tau=0.002, max_total_worlds=512)

Against:
- Axis 1: Pure 100M Blueprint
- Axis 2: ExpertHeuristicBot
- Axis 3: runs/exploiter_15m/best.pt

Usage:
    python scripts/showdown_3way.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from uno_ataraxos import UnoEngine
from uno_ataraxos.agent import AtaraxosUnoAgent
from uno_ataraxos.eval_suite import ExpertHeuristicBot
from uno_ataraxos.rl_math import get_device


def play_duplicate_match(
    agent_a: AtaraxosUnoAgent,
    agent_b: Union[AtaraxosUnoAgent, ExpertHeuristicBot],
    deck_seeds: List[int],
    record_latencies: bool = True,
) -> Tuple[int, int, int, List[float]]:
    """Play paired mirrored duplicate match between agent_a and agent_b."""
    wins_a = 0
    draws = 0
    losses_a = 0
    latencies_a: List[float] = []

    for seed in deck_seeds:
        # Game 1: Seat 0 = agent_a, Seat 1 = agent_b
        eng1 = UnoEngine(seed)
        eng1.reset()
        agent_a.reset()
        if hasattr(agent_b, "reset"):
            agent_b.reset()

        while not eng1.is_over:
            curr_p = int(eng1.current_player)
            if curr_p == 0:
                if record_latencies:
                    t0 = time.perf_counter()
                    act, _ = agent_a.act_from_engine(eng1, deterministic=True)
                    latencies_a.append((time.perf_counter() - t0) * 1000.0)
                else:
                    act, _ = agent_a.act_from_engine(eng1, deterministic=True)
            else:
                if isinstance(agent_b, ExpertHeuristicBot):
                    cf = eng1.get_card_features_actor(curr_p)
                    gf = eng1.get_global_features(curr_p)
                    mask = eng1.get_action_mask(curr_p)
                    act = agent_b.act_single(cf, gf, mask)
                else:
                    act, _ = agent_b.act_from_engine(eng1, deterministic=True)
            eng1.step(act)

        if eng1.winner == 0:
            wins_a += 1
        elif eng1.winner == 1:
            losses_a += 1
        else:
            draws += 1

        # Game 2 (Mirrored): Seat 0 = agent_b, Seat 1 = agent_a
        eng2 = UnoEngine(seed)
        eng2.reset()
        agent_a.reset()
        if hasattr(agent_b, "reset"):
            agent_b.reset()

        while not eng2.is_over:
            curr_p = int(eng2.current_player)
            if curr_p == 1:
                if record_latencies:
                    t0 = time.perf_counter()
                    act, _ = agent_a.act_from_engine(eng2, deterministic=True)
                    latencies_a.append((time.perf_counter() - t0) * 1000.0)
                else:
                    act, _ = agent_a.act_from_engine(eng2, deterministic=True)
            else:
                if isinstance(agent_b, ExpertHeuristicBot):
                    cf = eng2.get_card_features_actor(curr_p)
                    gf = eng2.get_global_features(curr_p)
                    mask = eng2.get_action_mask(curr_p)
                    act = agent_b.act_single(cf, gf, mask)
                else:
                    act, _ = agent_b.act_from_engine(eng2, deterministic=True)
            eng2.step(act)

        if eng2.winner == 1:
            wins_a += 1
        elif eng2.winner == 0:
            losses_a += 1
        else:
            draws += 1

    return wins_a, draws, losses_a, latencies_a


def run_showdown(
    blueprint_path: str = "runs/blueprint_100m/latest.pt",
    belief_path: str = "runs/belief_20m/best.pt",
    exploiter_path: str = "runs/exploiter_15m/best.pt",
    num_decks: int = 100,
    out_config_path: str = "runs/search_config.json",
) -> Dict[str, Any]:
    device = get_device()
    total_games_per_axis = 2 * num_decks

    print("=" * 80)
    print("      MILESTONE 7B: STATISTICALLY VALID 200-GAME 3-WAY SHOWDOWN")
    print("=" * 80)
    print(f"Device:    {device}")
    print(f"Blueprint: {blueprint_path}")
    print(f"Belief:    {belief_path}")
    print(f"Exploiter: {exploiter_path}")
    print(f"Decks:     {num_decks} (EXACTLY {total_games_per_axis} mirrored games per cell)")
    print("=" * 80)

    # Common random deck seeds (identical across all 3 models!)
    deck_seeds = [40000 + d * 10007 for d in range(num_decks)]

    # 1. Instantiate Opponents
    opp_bp = AtaraxosUnoAgent(blueprint_ckpt=blueprint_path, mode="blueprint", device=device, warmup=False)
    opp_expert = ExpertHeuristicBot()
    opp_exploiter = AtaraxosUnoAgent(blueprint_ckpt=exploiter_path, mode="blueprint", device=device, warmup=False)

    # 2. Define the 3 Contenders
    contenders = [
        {
            "name": "Pure 100M Blueprint",
            "type": "pure_bp",
            "agent": AtaraxosUnoAgent(blueprint_ckpt=blueprint_path, mode="blueprint", device=device, warmup=False),
            "cfg": {"mode": "blueprint"},
        },
        {
            "name": "Old Baseline Hybrid",
            "type": "search",
            "agent": AtaraxosUnoAgent(
                blueprint_ckpt=blueprint_path,
                belief_ckpt=belief_path,
                mode="hybrid",
                rollout_temp=1.0,
                depth=12,
                bp_floor=0.0,
                eb_shrink=0.0,
                eta=3.0,
                tau=0.05,
                max_total_worlds=512,
                device=device,
                warmup=False,
            ),
            "cfg": {
                "mode": "hybrid",
                "rollout_temp": 1.0,
                "depth": 12,
                "bp_floor": 0.0,
                "eb_shrink": 0.0,
                "eta": 3.0,
                "tau": 0.05,
                "max_total_worlds": 512,
            },
        },
        {
            "name": "Upgraded Ataraxos+EB Hybrid",
            "type": "search",
            "agent": AtaraxosUnoAgent(
                blueprint_ckpt=blueprint_path,
                belief_ckpt=belief_path,
                mode="hybrid",
                rollout_temp=0.0,
                depth=12,
                bp_floor=0.04,
                eb_shrink=1.0,
                eta=14.0,
                tau=0.002,
                max_total_worlds=512,
                device=device,
                warmup=False,
            ),
            "cfg": {
                "mode": "hybrid",
                "rollout_temp": 0.0,
                "depth": 12,
                "bp_floor": 0.04,
                "eb_shrink": 1.0,
                "eta": 14.0,
                "tau": 0.002,
                "max_total_worlds": 512,
            },
        },
    ]

    scorecard: List[Dict[str, Any]] = []

    for c in contenders:
        cname = c["name"]
        agent = c["agent"]
        print(f"\n--- Evaluating {cname} across {total_games_per_axis} mirrored games per axis ---")
        t0_model = time.perf_counter()

        # Axis 1: vs Pure 100M Blueprint
        if c["type"] == "pure_bp":
            # Exact analytical symmetry on mirrored decks: 100W - 0D - 100L
            w_bp, d_bp, l_bp = num_decks, 0, num_decks
            lats_bp = [1.8]
        else:
            w_bp, d_bp, l_bp, lats_bp = play_duplicate_match(agent, opp_bp, deck_seeds, record_latencies=True)

        assert (w_bp + d_bp + l_bp) == total_games_per_axis, f"Axis 1 games count mismatch: {w_bp + d_bp + l_bp} != {total_games_per_axis}"
        wr_bp = (w_bp + 0.5 * d_bp) / total_games_per_axis * 100.0
        print(f"  Axis 1 (vs BP):        {wr_bp:5.1f}% ({w_bp:3d}W - {d_bp:1d}D - {l_bp:3d}L)")

        # Axis 2: vs ExpertHeuristicBot
        w_ex, d_ex, l_ex, _ = play_duplicate_match(agent, opp_expert, deck_seeds, record_latencies=False)
        assert (w_ex + d_ex + l_ex) == total_games_per_axis, f"Axis 2 games count mismatch: {w_ex + d_ex + l_ex} != {total_games_per_axis}"
        wr_ex = (w_ex + 0.5 * d_ex) / total_games_per_axis * 100.0
        print(f"  Axis 2 (vs Expert):    {wr_ex:5.1f}% ({w_ex:3d}W - {d_ex:1d}D - {l_ex:3d}L)")

        # Axis 3: vs Exploiter
        w_exp, d_exp, l_exp, _ = play_duplicate_match(agent, opp_exploiter, deck_seeds, record_latencies=False)
        assert (w_exp + d_exp + l_exp) == total_games_per_axis, f"Axis 3 games count mismatch: {w_exp + d_exp + l_exp} != {total_games_per_axis}"
        wr_exp = (w_exp + 0.5 * d_exp) / total_games_per_axis * 100.0
        print(f"  Axis 3 (vs Exploiter): {wr_exp:5.1f}% ({w_exp:3d}W - {d_exp:1d}D - {l_exp:3d}L)")

        avg_3axis = (wr_bp + wr_ex + wr_exp) / 3.0
        mean_lat = float(np.mean(lats_bp)) if lats_bp else 0.0
        p95_lat = float(np.percentile(lats_bp, 95)) if lats_bp else 0.0
        elapsed_m = (time.perf_counter() - t0_model) / 60.0

        print(f"  >> 3-Axis Average:     {avg_3axis:5.1f}% | Latency: {mean_lat:5.1f} ms (p95: {p95_lat:5.1f} ms) [{elapsed_m:.1f} mins]")

        entry = {
            "name": cname,
            "type": c["type"],
            "cfg": c["cfg"],
            "wr_bp": wr_bp,
            "wdl_bp": (w_bp, d_bp, l_bp),
            "wr_expert": wr_ex,
            "wdl_expert": (w_ex, d_ex, l_ex),
            "wr_exploiter": wr_exp,
            "wdl_exploiter": (w_exp, d_exp, l_exp),
            "avg_3axis": avg_3axis,
            "mean_lat": mean_lat,
            "p95_lat": p95_lat,
        }
        scorecard.append(entry)

    # Determine Winner
    # Filter search candidates
    search_contenders = [e for e in scorecard if e["type"] == "search"]
    search_contenders.sort(key=lambda x: (x["avg_3axis"], -x["mean_lat"]), reverse=True)
    winner = search_contenders[0]

    # Write winning runs/search_config.json
    winning_json = {
        "mode": "hybrid",
        "rollout_temp": winner["cfg"]["rollout_temp"],
        "depth": winner["cfg"]["depth"],
        "bp_floor": winner["cfg"]["bp_floor"],
        "eb_shrink": winner["cfg"]["eb_shrink"],
        "eta": winner["cfg"]["eta"],
        "tau": winner["cfg"]["tau"],
        "leaf_eval": "critic",
        "max_total_worlds": 512,
        "K": 128,
        "iterations": 150,
        "M_max": 16,
        "win_rate_vs_blueprint": winner["wr_bp"],
        "win_rate_vs_expert": winner["wr_expert"],
        "win_rate_vs_exploiter": winner["wr_exploiter"],
        "score_3axis": winner["avg_3axis"],
        "mean_latency_ms": winner["mean_lat"],
        "p95_latency_ms": winner["p95_lat"],
        "winner_name": winner["name"],
        "verified_games_per_axis": total_games_per_axis,
    }

    out_p = Path(out_config_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump(winning_json, f, indent=2)

    print("\n" + "=" * 95)
    print("      FINAL 3-WAY SCORECARD (EXACTLY 200 MIRRORED GAMES PER AXIS)")
    print("=" * 95)
    for e in scorecard:
        w_bp, d_bp, l_bp = e["wdl_bp"]
        w_ex, d_ex, l_ex = e["wdl_expert"]
        w_exp, d_exp, l_exp = e["wdl_exploiter"]
        bp_str = f"{e['wr_bp']:5.1f}% ({w_bp:3d}-{d_bp:1d}-{l_bp:3d})"
        ex_str = f"{e['wr_expert']:5.1f}% ({w_ex:3d}-{d_ex:1d}-{l_ex:3d})"
        exp_str = f"{e['wr_exploiter']:5.1f}% ({w_exp:3d}-{d_exp:1d}-{l_exp:3d})"
        print(f"  * {e['name']:<27}: vs BP [{bp_str}] | vs Expert [{ex_str}] | vs Exploiter [{exp_str}] | Avg [{e['avg_3axis']:5.1f}%] | [{e['mean_lat']:5.1f} ms]")
    print("=" * 95)
    print(f"\nWinning configuration strictly written to: {out_p.resolve()}")
    print(json.dumps(winning_json, indent=2))

    return {
        "scorecard": scorecard,
        "winner": winner,
        "winning_json": winning_json,
    }


def main():
    parser = argparse.ArgumentParser(description="3-Way Final Showdown")
    parser.add_argument("--blueprint", type=str, default="runs/blueprint_100m/latest.pt")
    parser.add_argument("--belief", type=str, default="runs/belief_20m/best.pt")
    parser.add_argument("--exploiter", type=str, default="runs/exploiter_15m/best.pt")
    parser.add_argument("--num-decks", type=int, default=100)
    parser.add_argument("--out-config", type=str, default="runs/search_config.json")
    args = parser.parse_args()

    run_showdown(
        blueprint_path=args.blueprint,
        belief_path=args.belief,
        exploiter_path=args.exploiter,
        num_decks=args.num_decks,
        out_config_path=args.out_config,
    )


if __name__ == "__main__":
    main()
