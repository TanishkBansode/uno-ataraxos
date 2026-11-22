#!/usr/bin/env python3
"""Scientific 2-Stage 3-Axis Pareto Search Tuning (Milestone 7).

Evaluates 10 principled research-backed search configurations over paired common
random decks across 3 axes:
1. vs Pure 100M Blueprint
2. vs ExpertHeuristicBot
3. vs Warm-Start Exploiter (fallback: runs/exploiter_20m/best.pt)

Stage 1: Coarse screening of 10 configs on 60 decks (120 mirrored games/axis = 240 games/config).
Stage 2: Finalist verification on 150 decks (300 mirrored games/axis) across all 3 axes.
Final Check: Winning Tier-1 MMD vs Winning Hybrid (Tier-1 + Tier-2 Endgame PDCFR+).
Saves the winning configuration to runs/search_config.json.

Usage:
    python scripts/tune_search.py \
        --blueprint runs/blueprint_100m/latest.pt \
        --belief runs/belief_20m/best.pt \
        --exploiter runs/exploiter_warm_3m/best.pt
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

STAGE_1_CONFIGS: List[Dict[str, Any]] = [
    {
        "id": 1,
        "name": "Old Baseline",
        "rollout_temp": 1.0,
        "depth": 12,
        "bp_floor": 0.00,
        "eb_shrink": 0.0,
        "eta": 3.0,
        "tau": 0.05,
    },
    {
        "id": 2,
        "name": "Ataraxos DouDizhu Pure",
        "rollout_temp": 0.0,
        "depth": 14,
        "bp_floor": 0.00,
        "eb_shrink": 0.0,
        "eta": 5.0,
        "tau": 0.002,
    },
    {
        "id": 3,
        "name": "Ataraxos Barrage Pure",
        "rollout_temp": 0.0,
        "depth": 14,
        "bp_floor": 0.00,
        "eb_shrink": 0.0,
        "eta": 10.0,
        "tau": 0.001,
    },
    {
        "id": 4,
        "name": "Floor + Stochastic",
        "rollout_temp": 1.0,
        "depth": 14,
        "bp_floor": 0.03,
        "eb_shrink": 0.0,
        "eta": 10.0,
        "tau": 0.002,
    },
    {
        "id": 5,
        "name": "Floor + Greedy Rollout (Mild)",
        "rollout_temp": 0.0,
        "depth": 14,
        "bp_floor": 0.03,
        "eb_shrink": 0.0,
        "eta": 10.0,
        "tau": 0.002,
    },
    {
        "id": 6,
        "name": "Floor + Greedy Rollout (Strong)",
        "rollout_temp": 0.0,
        "depth": 14,
        "bp_floor": 0.04,
        "eb_shrink": 0.0,
        "eta": 16.0,
        "tau": 0.002,
    },
    {
        "id": 7,
        "name": "Floor + EB-Shrink (Balanced)",
        "rollout_temp": 0.0,
        "depth": 14,
        "bp_floor": 0.03,
        "eb_shrink": 1.0,
        "eta": 12.0,
        "tau": 0.002,
    },
    {
        "id": 8,
        "name": "Floor + EB-Shrink (Aggressive)",
        "rollout_temp": 0.0,
        "depth": 14,
        "bp_floor": 0.05,
        "eb_shrink": 1.0,
        "eta": 18.0,
        "tau": 0.002,
    },
    {
        "id": 9,
        "name": "Soft Rollout + Floor + EB",
        "rollout_temp": 0.5,
        "depth": 14,
        "bp_floor": 0.03,
        "eb_shrink": 1.0,
        "eta": 14.0,
        "tau": 0.002,
    },
    {
        "id": 10,
        "name": "Deeper Greedy + Floor + EB",
        "rollout_temp": 0.0,
        "depth": 20,
        "bp_floor": 0.03,
        "eb_shrink": 1.0,
        "eta": 14.0,
        "tau": 0.002,
    },
]


def resolve_exploiter_path(path_str: str) -> str:
    """Resolve exploiter checkpoint with automatic fallback."""
    p = Path(path_str)
    if p.exists():
        return str(p)
    candidates = [
        "runs/exploiter_warm_3m/best.pt",
        "runs/exploiter_20m/best.pt",
        "runs/exploiter_15m/best.pt",
        "runs/exploiter_smoke/best.pt",
    ]
    for c in candidates:
        if Path(c).exists():
            print(f"Notice: exploiter checkpoint '{path_str}' not found. Falling back to: {c}")
            return c
    raise FileNotFoundError(f"No valid exploiter checkpoint found (checked {path_str} and {candidates})")


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


def run_stage_1(
    blueprint_path: str,
    belief_path: str,
    num_decks: int = 60,
    device: Optional[torch.device] = None,
) -> List[Dict[str, Any]]:
    """Stage 1: Coarse screening of 10 principled configurations."""
    if device is None:
        device = get_device()

    print("\n" + "=" * 80)
    print("STAGE 1 — COARSE SCREENING (10 Principled Configs on Common Random Decks)")
    print(f"Decks: {num_decks} | Games per axis: {2 * num_decks} | Total games per config: {4 * num_decks}")
    print("=" * 80)

    # Fixed common deck seeds
    common_seeds = [20000 + d * 10007 for d in range(num_decks)]

    # Pre-instantiate opponents to avoid reloading weights
    pure_bp = AtaraxosUnoAgent(blueprint_ckpt=blueprint_path, mode="blueprint", device=device, warmup=False)
    expert = ExpertHeuristicBot()

    stage1_results: List[Dict[str, Any]] = []

    for cfg in STAGE_1_CONFIGS:
        cid = cfg["id"]
        cname = cfg["name"]
        print(f"\n[Config {cid:2d}/10] {cname}: r_temp={cfg['rollout_temp']}, depth={cfg['depth']}, floor={cfg['bp_floor']}, eb={cfg['eb_shrink']}, eta={cfg['eta']}")

        search_agent = AtaraxosUnoAgent(
            blueprint_ckpt=blueprint_path,
            belief_ckpt=belief_path,
            mode="mmd_search",
            device=device,
            rollout_temp=cfg["rollout_temp"],
            depth=cfg["depth"],
            bp_floor=cfg["bp_floor"],
            eb_shrink=cfg["eb_shrink"],
            eta=cfg["eta"],
            tau=cfg["tau"],
            max_total_worlds=768,
            leaf_eval="critic",
            warmup=False,
        )

        t0_cfg = time.perf_counter()

        # Axis 1: vs Pure Blueprint
        w_bp, d_bp, l_bp, lats_bp = play_duplicate_match(search_agent, pure_bp, common_seeds, record_latencies=True)
        wr_bp = (w_bp + 0.5 * d_bp) / (2 * num_decks) * 100.0

        # Axis 2: vs ExpertHeuristicBot
        w_ex, d_ex, l_ex, lats_ex = play_duplicate_match(search_agent, expert, common_seeds, record_latencies=False)
        wr_ex = (w_ex + 0.5 * d_ex) / (2 * num_decks) * 100.0

        elapsed = time.perf_counter() - t0_cfg
        score_2axis = 0.5 * wr_bp + 0.5 * wr_ex
        mean_lat = float(np.mean(lats_bp)) if lats_bp else 0.0
        p95_lat = float(np.percentile(lats_bp, 95)) if lats_bp else 0.0

        res_entry = {
            **cfg,
            "wr_bp": wr_bp,
            "wr_expert": wr_ex,
            "score_2axis": score_2axis,
            "mean_lat": mean_lat,
            "p95_lat": p95_lat,
            "elapsed_s": elapsed,
        }
        stage1_results.append(res_entry)

        print(f"    >> vs BP: {wr_bp:5.1f}% | vs Expert: {wr_ex:5.1f}% | 2-Axis Score: {score_2axis:5.1f}% | Mean Lat: {mean_lat:5.1f} ms | p95: {p95_lat:5.1f} ms ({elapsed:.1f}s)")

    # Sort descending by 2-axis score
    stage1_results.sort(key=lambda r: (r["score_2axis"], -r["mean_lat"]), reverse=True)

    print("\n" + "=" * 90)
    print("STAGE 1 COARSE SCREENING TABLE (Top 5 + Old Baseline, 120 mirrored games/axis):")
    print(f"{'config_id':<9} | {'r_temp':<6} | {'depth':<5} | {'floor':<5} | {'eb':<4} | {'eta':<5} | {'vs BP':<7} | {'vs Expert':<9} | {'2-Axis':<7} | {'ms/move':<8}")
    print("-" * 90)
    for r in stage1_results:
        print(f"{r['id']:<9d} | {r['rollout_temp']:<6.1f} | {r['depth']:<5d} | {r['bp_floor']:<5.2f} | {r['eb_shrink']:<4.1f} | {r['eta']:<5.1f} | {r['wr_bp']:6.1f}% | {r['wr_expert']:8.1f}% | {r['score_2axis']:6.1f}% | {r['mean_lat']:7.1f}")
    print("=" * 90 + "\n")

    return stage1_results


def run_stage_2(
    blueprint_path: str,
    belief_path: str,
    exploiter_path: str,
    finalist_configs: List[Dict[str, Any]],
    num_decks: int = 150,
    device: Optional[torch.device] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Stage 2: Full 3-axis scorecard on 150 decks (300 mirrored games/axis)."""
    if device is None:
        device = get_device()

    print("\n" + "=" * 80)
    print("STAGE 2 — FULL 3-AXIS SCORECARD (Finalists vs Baseline vs Pure BP)")
    print(f"Decks: {num_decks} | Games per axis: {2 * num_decks} (300 mirrored games)")
    print("=" * 80)

    common_seeds = [50000 + d * 10007 for d in range(num_decks)]
    total_games_per_axis = 2 * num_decks

    pure_bp = AtaraxosUnoAgent(blueprint_ckpt=blueprint_path, mode="blueprint", device=device, warmup=False)
    expert = ExpertHeuristicBot()
    exploiter = AtaraxosUnoAgent(blueprint_ckpt=exploiter_path, mode="blueprint", device=device, warmup=False)

    scorecard: List[Dict[str, Any]] = []

    # 1. Pure 100M Blueprint Baseline Evaluation
    print("\nEvaluating Reference: Pure 100M Blueprint ...")
    t0_ref = time.perf_counter()
    w_ex, d_ex, _, lats_ref = play_duplicate_match(pure_bp, expert, common_seeds, record_latencies=True)
    wr_ref_ex = (w_ex + 0.5 * d_ex) / total_games_per_axis * 100.0

    w_exp, d_exp, _, _ = play_duplicate_match(pure_bp, exploiter, common_seeds, record_latencies=False)
    wr_ref_exp = (w_exp + 0.5 * d_exp) / total_games_per_axis * 100.0

    wr_ref_bp = 50.0  # By definition vs identical self
    score_3axis_ref = (wr_ref_bp + wr_ref_ex + wr_ref_exp) / 3.0
    mean_lat_ref = float(np.mean(lats_ref)) if lats_ref else 0.0
    p95_lat_ref = float(np.percentile(lats_ref, 95)) if lats_ref else 0.0

    ref_entry = {
        "name": "Pure 100M Blueprint",
        "wr_bp": wr_ref_bp,
        "wr_expert": wr_ref_ex,
        "wr_exploiter": wr_ref_exp,
        "score_3axis": score_3axis_ref,
        "ci_95": 1.96 * math.sqrt(score_3axis_ref * (100.0 - score_3axis_ref) / (3 * total_games_per_axis)),
        "mean_lat": mean_lat_ref,
        "p95_lat": p95_lat_ref,
    }
    scorecard.append(ref_entry)
    print(f"    >> vs BP: 50.0% | vs Expert: {wr_ref_ex:5.1f}% | vs Exploiter: {wr_ref_exp:5.1f}% | 3-Axis: {score_3axis_ref:5.1f}%")

    # 2. Evaluate Finalist Search Configs
    for cfg in finalist_configs:
        cname = cfg["name"]
        print(f"\nEvaluating Finalist: {cname} (ID {cfg['id']}) ...")
        t0_cfg = time.perf_counter()

        search_agent = AtaraxosUnoAgent(
            blueprint_ckpt=blueprint_path,
            belief_ckpt=belief_path,
            mode="mmd_search",
            device=device,
            rollout_temp=cfg["rollout_temp"],
            depth=cfg["depth"],
            bp_floor=cfg["bp_floor"],
            eb_shrink=cfg["eb_shrink"],
            eta=cfg["eta"],
            tau=cfg["tau"],
            max_total_worlds=768,
            leaf_eval="critic",
            warmup=False,
        )

        w_bp, d_bp, _, lats = play_duplicate_match(search_agent, pure_bp, common_seeds, record_latencies=True)
        wr_bp = (w_bp + 0.5 * d_bp) / total_games_per_axis * 100.0

        w_ex, d_ex, _, _ = play_duplicate_match(search_agent, expert, common_seeds, record_latencies=False)
        wr_ex = (w_ex + 0.5 * d_ex) / total_games_per_axis * 100.0

        w_exp, d_exp, _, _ = play_duplicate_match(search_agent, exploiter, common_seeds, record_latencies=False)
        wr_exp = (w_exp + 0.5 * d_exp) / total_games_per_axis * 100.0

        score_3axis = (wr_bp + wr_ex + wr_exp) / 3.0
        ci_95 = 1.96 * math.sqrt(score_3axis * (100.0 - score_3axis) / (3 * total_games_per_axis))
        mean_lat = float(np.mean(lats)) if lats else 0.0
        p95_lat = float(np.percentile(lats, 95)) if lats else 0.0

        entry = {
            **cfg,
            "wr_bp": wr_bp,
            "wr_expert": wr_ex,
            "wr_exploiter": wr_exp,
            "score_3axis": score_3axis,
            "ci_95": ci_95,
            "mean_lat": mean_lat,
            "p95_lat": p95_lat,
        }
        scorecard.append(entry)
        print(f"    >> vs BP: {wr_bp:5.1f}% | vs Expert: {wr_ex:5.1f}% | vs Exploiter: {wr_exp:5.1f}% | 3-Axis: {score_3axis:5.1f}% (+/- {ci_95:3.1f}%) | Lat: {mean_lat:5.1f} ms")

    # Identify #1 winning search config
    search_entries = [e for e in scorecard if "id" in e]
    search_entries.sort(key=lambda r: (r["score_3axis"], -r["mean_lat"]), reverse=True)
    winner = search_entries[0]

    print("\n" + "=" * 90)
    print("STAGE 2 FULL 3-AXIS SCORECARD (300 mirrored games/axis):")
    print(f"{'Model':<32} | {'vs BP':<7} | {'vs Expert':<9} | {'vs Exploiter':<12} | {'3-Axis Avg':<18} | {'mean/p95 ms':<12}")
    print("-" * 90)
    for e in scorecard:
        mname = e["name"]
        avg_str = f"{e['score_3axis']:5.1f}% (+/-{e['ci_95']:3.1f}%)"
        lat_str = f"{e['mean_lat']:5.1f} / {e['p95_lat']:5.1f}"
        print(f"{mname:<32} | {e['wr_bp']:6.1f}% | {e['wr_expert']:8.1f}% | {e['wr_exploiter']:11.1f}% | {avg_str:<18} | {lat_str:<12}")
    print("=" * 90 + "\n")

    return scorecard, winner


def run_tier1_vs_hybrid(
    blueprint_path: str,
    belief_path: str,
    exploiter_path: str,
    winner_cfg: Dict[str, Any],
    num_decks: int = 150,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Head-to-head comparison: Winning Tier-1 MMD vs Winning Hybrid (Tier-1 + Tier-2 PDCFR+)."""
    if device is None:
        device = get_device()

    print("\n" + "=" * 80)
    print("FINAL HEAD-TO-HEAD: Winning Tier-1 MMD vs Winning Hybrid (Tier-1 + Tier-2 PDCFR+)")
    print(f"Decks: {num_decks} | Games per axis: {2 * num_decks} (300 mirrored games)")
    print("=" * 80)

    common_seeds = [70000 + d * 10007 for d in range(num_decks)]
    total_games = 2 * num_decks

    pure_bp = AtaraxosUnoAgent(blueprint_ckpt=blueprint_path, mode="blueprint", device=device, warmup=False)
    expert = ExpertHeuristicBot()
    exploiter = AtaraxosUnoAgent(blueprint_ckpt=exploiter_path, mode="blueprint", device=device, warmup=False)

    # 1. Tier-1 MMD Agent
    agent_t1 = AtaraxosUnoAgent(
        blueprint_ckpt=blueprint_path,
        belief_ckpt=belief_path,
        mode="mmd_search",
        device=device,
        rollout_temp=winner_cfg["rollout_temp"],
        depth=winner_cfg["depth"],
        bp_floor=winner_cfg["bp_floor"],
        eb_shrink=winner_cfg["eb_shrink"],
        eta=winner_cfg["eta"],
        tau=winner_cfg["tau"],
        max_total_worlds=768,
        leaf_eval="critic",
        warmup=False,
    )

    # 2. Hybrid Agent (MMD Search + Endgame PDCFR+)
    agent_hybrid = AtaraxosUnoAgent(
        blueprint_ckpt=blueprint_path,
        belief_ckpt=belief_path,
        mode="hybrid",
        device=device,
        rollout_temp=winner_cfg["rollout_temp"],
        depth=winner_cfg["depth"],
        bp_floor=winner_cfg["bp_floor"],
        eb_shrink=winner_cfg["eb_shrink"],
        eta=winner_cfg["eta"],
        tau=winner_cfg["tau"],
        max_total_worlds=768,
        leaf_eval="critic",
        iterations=150,
        M_max=16,
        warmup=False,
    )

    print("Evaluating Tier-1 MMD ...")
    w1_bp, d1_bp, _, lats1 = play_duplicate_match(agent_t1, pure_bp, common_seeds, record_latencies=True)
    w1_ex, d1_ex, _, _ = play_duplicate_match(agent_t1, expert, common_seeds, record_latencies=False)
    w1_exp, d1_exp, _, _ = play_duplicate_match(agent_t1, exploiter, common_seeds, record_latencies=False)

    wr1_bp = (w1_bp + 0.5 * d1_bp) / total_games * 100.0
    wr1_ex = (w1_ex + 0.5 * d1_ex) / total_games * 100.0
    wr1_exp = (w1_exp + 0.5 * d1_exp) / total_games * 100.0
    avg1 = (wr1_bp + wr1_ex + wr1_exp) / 3.0
    mean_lat1 = float(np.mean(lats1)) if lats1 else 0.0

    print("Evaluating Hybrid (Tier-1 + Tier-2 PDCFR+) ...")
    w2_bp, d2_bp, _, lats2 = play_duplicate_match(agent_hybrid, pure_bp, common_seeds, record_latencies=True)
    w2_ex, d2_ex, _, _ = play_duplicate_match(agent_hybrid, expert, common_seeds, record_latencies=False)
    w2_exp, d2_exp, _, _ = play_duplicate_match(agent_hybrid, exploiter, common_seeds, record_latencies=False)

    wr2_bp = (w2_bp + 0.5 * d2_bp) / total_games * 100.0
    wr2_ex = (w2_ex + 0.5 * d2_ex) / total_games * 100.0
    wr2_exp = (w2_exp + 0.5 * d2_exp) / total_games * 100.0
    avg2 = (wr2_bp + wr2_ex + wr2_exp) / 3.0
    mean_lat2 = float(np.mean(lats2)) if lats2 else 0.0

    best_mode = "hybrid" if avg2 >= avg1 else "mmd_search"

    print("\n" + "=" * 90)
    print("TIER-1 VS HYBRID FINAL CHECK (300 mirrored games/axis):")
    print(f"{'Mode':<18} | {'vs BP':<7} | {'vs Expert':<9} | {'vs Exploiter':<12} | {'3-Axis Avg':<10} | {'mean ms':<8}")
    print("-" * 90)
    print(f"{'Tier-1 MMD':<18} | {wr1_bp:6.1f}% | {wr1_ex:8.1f}% | {wr1_exp:11.1f}% | {avg1:9.1f}% | {mean_lat1:7.1f}")
    print(f"{'Hybrid (+ PDCFR+)':<18} | {wr2_bp:6.1f}% | {wr2_ex:8.1f}% | {wr2_exp:11.1f}% | {avg2:9.1f}% | {mean_lat2:7.1f}")
    print("=" * 90)
    print(f"Recommended Default Mode: {best_mode.upper()} ({'+' if avg2 >= avg1 else ''}{avg2 - avg1:+.1f}% lift)\n")

    return {
        "best_mode": best_mode,
        "t1": {"wr_bp": wr1_bp, "wr_expert": wr1_ex, "wr_exploiter": wr1_exp, "avg": avg1, "lat": mean_lat1},
        "hybrid": {"wr_bp": wr2_bp, "wr_expert": wr2_ex, "wr_exploiter": wr2_exp, "avg": avg2, "lat": mean_lat2},
    }


def main():
    parser = argparse.ArgumentParser(description="Scientific 2-Stage 3-Axis Pareto Search Tuning")
    parser.add_argument("--blueprint", type=str, default="runs/blueprint_100m/latest.pt", help="Path to Blueprint checkpoint")
    parser.add_argument("--belief", type=str, default="runs/belief_20m/best.pt", help="Path to Belief checkpoint")
    parser.add_argument("--exploiter", type=str, default="runs/exploiter_warm_3m/best.pt", help="Path to Exploiter checkpoint")
    parser.add_argument("--out-config", type=str, default="runs/search_config.json", help="Path to output search_config.json")
    parser.add_argument("--stage1-decks", type=int, default=60, help="Decks for Stage 1 coarse screening")
    parser.add_argument("--stage2-decks", type=int, default=150, help="Decks for Stage 2 scorecard")
    parser.add_argument("--quick", action="store_true", help="Quick mode (15 decks stage 1, 30 decks stage 2) for verification")
    args = parser.parse_args()

    device = get_device()
    exploiter_path = resolve_exploiter_path(args.exploiter)

    print("=" * 80)
    print("      MILESTONE 7: SCIENTIFIC 2-STAGE 3-AXIS PARETO SEARCH TUNING")
    print("=" * 80)
    print(f"Device:    {device}")
    print(f"Blueprint: {args.blueprint}")
    print(f"Belief:    {args.belief}")
    print(f"Exploiter: {exploiter_path}")

    stage1_decks = 15 if args.quick else args.stage1_decks
    stage2_decks = 30 if args.quick else args.stage2_decks

    # --- STAGE 1: Coarse Screening ---
    stage1_results = run_stage_1(
        blueprint_path=args.blueprint,
        belief_path=args.belief,
        num_decks=stage1_decks,
        device=device,
    )

    # Pick Top 3 Configs from Stage 1 (plus Old Baseline #1 if not already in Top 3)
    top_3 = stage1_results[:3]
    finalists = list(top_3)
    baseline_entry = next((r for r in stage1_results if r["id"] == 1), None)
    if baseline_entry and baseline_entry["id"] not in [f["id"] for f in finalists]:
        finalists.append(baseline_entry)

    # --- STAGE 2: Full 3-Axis Scorecard ---
    scorecard, winner = run_stage_2(
        blueprint_path=args.blueprint,
        belief_path=args.belief,
        exploiter_path=exploiter_path,
        finalist_configs=finalists,
        num_decks=stage2_decks,
        device=device,
    )

    # --- Final Check: Tier-1 MMD vs Hybrid ---
    comp = run_tier1_vs_hybrid(
        blueprint_path=args.blueprint,
        belief_path=args.belief,
        exploiter_path=exploiter_path,
        winner_cfg=winner,
        num_decks=stage2_decks,
        device=device,
    )

    # Write Winning search_config.json
    final_config = {
        "mode": comp["best_mode"],
        "rollout_temp": winner["rollout_temp"],
        "depth": winner["depth"],
        "bp_floor": winner["bp_floor"],
        "eb_shrink": winner["eb_shrink"],
        "eta": winner["eta"],
        "tau": winner["tau"],
        "leaf_eval": "critic",
        "max_total_worlds": 768,
        "K": 128,
        "iterations": 150,
        "M_max": 16,
        "win_rate_vs_blueprint": winner["wr_bp"],
        "win_rate_vs_expert": winner["wr_expert"],
        "win_rate_vs_exploiter": winner["wr_exploiter"],
        "score_3axis": winner["score_3axis"],
        "mean_latency_ms": winner["mean_lat"],
        "p95_latency_ms": winner["p95_lat"],
        "winner_name": winner["name"],
    }

    out_p = Path(args.out_config)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump(final_config, f, indent=2)

    print(f"\nWinning configuration written to: {out_p.resolve()}")
    print(json.dumps(final_config, indent=2))


if __name__ == "__main__":
    main()
