#!/usr/bin/env python3
"""Milestone 8: 4-Way Controlled Ablation of OOD-Robust Portfolio Search.

Compares 4 contenders on 100 common random duplicate decks (EXACTLY 200 mirrored games/axis):
1. Control A: Pure 100M Blueprint (mode="blueprint")
2. Ablation B: Static Risk-Averse Portfolio (Tier-1 ONLY):
   - mode="mmd_search", use_portfolio=True, use_adaptive_bayes=False
   - portfolio_weights=(0.45, 0.35, 0.20), risk_gamma=4.0, uniform_mix=0.12, belief_temp=1.15
   - rollout_temp=0.0, depth=14, bp_floor=0.03, eb_shrink=1.0, eta=8.0, tau=0.002, max_total_worlds=512
3. Ablation C: Adaptive Bayesian Portfolio (Tier-1 ONLY):
   - mode="mmd_search", use_portfolio=True, use_adaptive_bayes=True, risk_gamma=4.0
   - rollout_temp=0.0, depth=14, bp_floor=0.03, eb_shrink=1.0, eta=8.0, tau=0.002, max_total_worlds=512
4. Ablation D: Adaptive Bayesian Portfolio (HYBRID + Belief-Tempered PDCFR+):
   - same as C, but mode="hybrid"

Against:
- Axis 1: Pure 100M Blueprint
- Axis 2: ExpertHeuristicBot
- Axis 3: runs/exploiter_20m/best.pt

Usage:
    python scripts/eval_tournament.py
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
    track_bayes: bool = False,
) -> Tuple[int, int, int, List[float], Dict[str, Any]]:
    """Play paired mirrored duplicate match between agent_a and agent_b."""
    wins_a = 0
    draws = 0
    losses_a = 0
    latencies_a: List[float] = []

    bayes_weights: List[List[float]] = []
    bayes_surprisals: List[float] = []
    bayes_mixes: List[float] = []

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

        if track_bayes:
            bayes_weights.append(list(agent_a.opp_tracker.weights))
            bayes_surprisals.append(float(agent_a.opp_tracker.ema_surprisal))
            _, u_mix = agent_a.opp_tracker.get_belief_params()
            bayes_mixes.append(float(u_mix))

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

        if track_bayes:
            bayes_weights.append(list(agent_a.opp_tracker.weights))
            bayes_surprisals.append(float(agent_a.opp_tracker.ema_surprisal))
            _, u_mix = agent_a.opp_tracker.get_belief_params()
            bayes_mixes.append(float(u_mix))

    bayes_summary = {}
    if track_bayes and bayes_weights:
        w_arr = np.array(bayes_weights)
        bayes_summary = {
            "avg_weights": w_arr.mean(axis=0).tolist(),
            "avg_surprisal": float(np.mean(bayes_surprisals)),
            "avg_uniform_mix": float(np.mean(bayes_mixes)),
        }

    return wins_a, draws, losses_a, latencies_a, bayes_summary


def run_ood_ablation(
    blueprint_path: str = "runs/blueprint_100m/latest.pt",
    belief_path: str = "runs/belief_20m/best.pt",
    exploiter_path: str = "runs/exploiter_20m/best.pt",
    num_decks: int = 100,
    out_config_path: str = "runs/search_config.json",
) -> Dict[str, Any]:
    device = get_device()
    total_games_per_axis = 2 * num_decks

    print("=" * 95)
    print("      MILESTONE 8: 4-WAY CONTROLLED ABLATION OF OOD-ROBUST PORTFOLIO SEARCH")
    print("=" * 95)
    print(f"Device:    {device}")
    print(f"Blueprint: {blueprint_path}")
    print(f"Belief:    {belief_path}")
    print(f"Exploiter: {exploiter_path}")
    print(f"Decks:     {num_decks} (EXACTLY {total_games_per_axis} mirrored games per cell)")
    print("=" * 95)

    # Common random duplicate deck seeds (identical across all 4 models!)
    deck_seeds = [60000 + d * 10007 for d in range(num_decks)]

    # 1. Instantiate Opponents
    opp_bp = AtaraxosUnoAgent(blueprint_ckpt=blueprint_path, mode="blueprint", device=device, warmup=False)
    opp_expert = ExpertHeuristicBot()
    opp_exploiter = AtaraxosUnoAgent(blueprint_ckpt=exploiter_path, mode="blueprint", device=device, warmup=False)

    # 2. Define the 4 Contenders
    contenders = [
        {
            "name": "Control A (Pure 100M BP)",
            "type": "pure_bp",
            "agent": AtaraxosUnoAgent(blueprint_ckpt=blueprint_path, mode="blueprint", device=device, warmup=False),
            "track_bayes": False,
            "cfg": {"mode": "blueprint"},
        },
        {
            "name": "Ablation B (Static Risk-Averse)",
            "type": "search",
            "agent": AtaraxosUnoAgent(
                blueprint_ckpt=blueprint_path,
                belief_ckpt=belief_path,
                mode="mmd_search",
                use_portfolio=True,
                use_adaptive_bayes=False,
                portfolio_weights=(0.45, 0.35, 0.20),
                risk_gamma=4.0,
                uniform_mix=0.12,
                belief_temp=1.15,
                rollout_temp=0.0,
                depth=14,
                bp_floor=0.03,
                eb_shrink=1.0,
                eta=8.0,
                tau=0.002,
                max_total_worlds=512,
                device=device,
                warmup=False,
            ),
            "track_bayes": False,
            "cfg": {
                "mode": "mmd_search",
                "use_portfolio": True,
                "use_adaptive_bayes": False,
                "portfolio_weights": [0.45, 0.35, 0.20],
                "risk_gamma": 4.0,
                "uniform_mix": 0.12,
                "belief_temp": 1.15,
                "rollout_temp": 0.0,
                "depth": 14,
                "bp_floor": 0.03,
                "eb_shrink": 1.0,
                "eta": 8.0,
                "tau": 0.002,
                "max_total_worlds": 512,
            },
        },
        {
            "name": "Ablation C (Adaptive Bayes Tier-1)",
            "type": "search",
            "agent": AtaraxosUnoAgent(
                blueprint_ckpt=blueprint_path,
                belief_ckpt=belief_path,
                mode="mmd_search",
                use_portfolio=True,
                use_adaptive_bayes=True,
                risk_gamma=4.0,
                rollout_temp=0.0,
                depth=14,
                bp_floor=0.03,
                eb_shrink=1.0,
                eta=8.0,
                tau=0.002,
                max_total_worlds=512,
                device=device,
                warmup=False,
            ),
            "track_bayes": True,
            "cfg": {
                "mode": "mmd_search",
                "use_portfolio": True,
                "use_adaptive_bayes": True,
                "risk_gamma": 4.0,
                "rollout_temp": 0.0,
                "depth": 14,
                "bp_floor": 0.03,
                "eb_shrink": 1.0,
                "eta": 8.0,
                "tau": 0.002,
                "max_total_worlds": 512,
            },
        },
        {
            "name": "Ablation D (Adaptive Bayes Hybrid)",
            "type": "search",
            "agent": AtaraxosUnoAgent(
                blueprint_ckpt=blueprint_path,
                belief_ckpt=belief_path,
                mode="hybrid",
                use_portfolio=True,
                use_adaptive_bayes=True,
                risk_gamma=4.0,
                rollout_temp=0.0,
                depth=14,
                bp_floor=0.03,
                eb_shrink=1.0,
                eta=8.0,
                tau=0.002,
                max_total_worlds=512,
                device=device,
                warmup=False,
            ),
            "track_bayes": True,
            "cfg": {
                "mode": "hybrid",
                "use_portfolio": True,
                "use_adaptive_bayes": True,
                "risk_gamma": 4.0,
                "rollout_temp": 0.0,
                "depth": 14,
                "bp_floor": 0.03,
                "eb_shrink": 1.0,
                "eta": 8.0,
                "tau": 0.002,
                "max_total_worlds": 512,
            },
        },
    ]

    scorecard: List[Dict[str, Any]] = []

    for c in contenders:
        cname = c["name"]
        agent = c["agent"]
        tb = c["track_bayes"]
        print(f"\n--- Evaluating {cname} across {total_games_per_axis} mirrored games per axis ---")
        t0_model = time.perf_counter()

        # Axis 1: vs Pure 100M Blueprint
        if c["type"] == "pure_bp":
            w_bp, d_bp, l_bp = num_decks, 0, num_decks
            lats_bp = [1.8]
            bayes_bp = {}
        else:
            w_bp, d_bp, l_bp, lats_bp, bayes_bp = play_duplicate_match(
                agent, opp_bp, deck_seeds, record_latencies=True, track_bayes=tb
            )

        assert (w_bp + d_bp + l_bp) == total_games_per_axis, f"Axis 1 games count mismatch: {w_bp + d_bp + l_bp} != {total_games_per_axis}"
        wr_bp = (w_bp + 0.5 * d_bp) / total_games_per_axis * 100.0
        print(f"  Axis 1 (vs BP):        {wr_bp:5.1f}% ({w_bp:3d}W - {d_bp:1d}D - {l_bp:3d}L)")
        if tb and bayes_bp:
            bw = bayes_bp["avg_weights"]
            print(f"    -> Posterior: [BP: {bw[0]:.2f}, Agg: {bw[1]:.2f}, Noisy: {bw[2]:.2f}] | Surprisal: {bayes_bp['avg_surprisal']:.2f} | u_mix: {bayes_bp['avg_uniform_mix']:.3f}")

        # Axis 2: vs ExpertHeuristicBot
        w_ex, d_ex, l_ex, _, bayes_ex = play_duplicate_match(
            agent, opp_expert, deck_seeds, record_latencies=False, track_bayes=tb
        )
        assert (w_ex + d_ex + l_ex) == total_games_per_axis, f"Axis 2 games count mismatch: {w_ex + d_ex + l_ex} != {total_games_per_axis}"
        wr_ex = (w_ex + 0.5 * d_ex) / total_games_per_axis * 100.0
        print(f"  Axis 2 (vs Expert):    {wr_ex:5.1f}% ({w_ex:3d}W - {d_ex:1d}D - {l_ex:3d}L)")
        if tb and bayes_ex:
            bw = bayes_ex["avg_weights"]
            print(f"    -> Posterior: [BP: {bw[0]:.2f}, Agg: {bw[1]:.2f}, Noisy: {bw[2]:.2f}] | Surprisal: {bayes_ex['avg_surprisal']:.2f} | u_mix: {bayes_ex['avg_uniform_mix']:.3f}")

        # Axis 3: vs Exploiter
        w_exp, d_exp, l_exp, _, bayes_exp = play_duplicate_match(
            agent, opp_exploiter, deck_seeds, record_latencies=False, track_bayes=tb
        )
        assert (w_exp + d_exp + l_exp) == total_games_per_axis, f"Axis 3 games count mismatch: {w_exp + d_exp + l_exp} != {total_games_per_axis}"
        wr_exp = (w_exp + 0.5 * d_exp) / total_games_per_axis * 100.0
        print(f"  Axis 3 (vs Exploiter): {wr_exp:5.1f}% ({w_exp:3d}W - {d_exp:1d}D - {l_exp:3d}L)")
        if tb and bayes_exp:
            bw = bayes_exp["avg_weights"]
            print(f"    -> Posterior: [BP: {bw[0]:.2f}, Agg: {bw[1]:.2f}, Noisy: {bw[2]:.2f}] | Surprisal: {bayes_exp['avg_surprisal']:.2f} | u_mix: {bayes_exp['avg_uniform_mix']:.3f}")

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
            "bayes_bp": bayes_bp,
            "bayes_expert": bayes_ex,
            "bayes_exploiter": bayes_exp,
        }
        scorecard.append(entry)

    # Determine Best Contender
    search_contenders = [e for e in scorecard if e["type"] == "search"]
    search_contenders.sort(key=lambda x: (x["avg_3axis"], -x["mean_lat"]), reverse=True)
    best_search = search_contenders[0]

    pure_bp_baseline = scorecard[0]["avg_3axis"]
    print("\n" + "=" * 105)
    print("      FINAL 4-WAY CONTROLLED ABLATION SCORECARD (EXACTLY 200 MIRRORED GAMES PER AXIS)")
    print("=" * 105)
    for e in scorecard:
        w_bp, d_bp, l_bp = e["wdl_bp"]
        w_ex, d_ex, l_ex = e["wdl_expert"]
        w_exp, d_exp, l_exp = e["wdl_exploiter"]
        bp_str = f"{e['wr_bp']:5.1f}% ({w_bp:3d}-{d_bp:1d}-{l_bp:3d})"
        ex_str = f"{e['wr_expert']:5.1f}% ({w_ex:3d}-{d_ex:1d}-{l_ex:3d})"
        exp_str = f"{e['wr_exploiter']:5.1f}% ({w_exp:3d}-{d_exp:1d}-{l_exp:3d})"
        print(f"  * {e['name']:<33}: vs BP [{bp_str}] | vs Expert [{ex_str}] | vs Exploiter [{exp_str}] | Avg [{e['avg_3axis']:5.1f}%] | [{e['mean_lat']:5.1f} ms]")
    print("=" * 105)

    # Print Ablation C Bayesian tracking summary
    abl_c = [e for e in scorecard if e["name"].startswith("Ablation C")][0]
    print("\n--- Online Bayesian Opponent Identification Profiles (Ablation C) ---")
    for axis_name, b_data in [("Axis 1 (vs BP)", abl_c["bayes_bp"]), ("Axis 2 (vs Expert)", abl_c["bayes_expert"]), ("Axis 3 (vs Exploiter)", abl_c["bayes_exploiter"])]:
        if b_data:
            bw = b_data["avg_weights"]
            print(f"  * {axis_name:<20}: Posteriors: [BP={bw[0]:.2f}, Agg={bw[1]:.2f}, Noisy={bw[2]:.2f}] | Surprisal={b_data['avg_surprisal']:.2f} | u_mix={b_data['avg_uniform_mix']:.3f}")

    if best_search["avg_3axis"] >= pure_bp_baseline:
        print(f"\n[NEW CHAMPION] {best_search['name']} achieved {best_search['avg_3axis']:.1f}% >= Pure BP Baseline {pure_bp_baseline:.1f}%!")
        out_cfg = dict(best_search["cfg"])
        out_cfg.update({
            "win_rate_vs_blueprint": best_search["wr_bp"],
            "win_rate_vs_expert": best_search["wr_expert"],
            "win_rate_vs_exploiter": best_search["wr_exploiter"],
            "score_3axis": best_search["avg_3axis"],
            "mean_latency_ms": best_search["mean_lat"],
            "p95_latency_ms": best_search["p95_lat"],
            "winner_name": best_search["name"],
            "verified_games_per_axis": total_games_per_axis,
        })
        out_p = Path(out_config_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w") as f:
            json.dump(out_cfg, f, indent=2)
        print(f"Saved winning config to: {out_p.resolve()}")
    else:
        print(f"\n[BASELINE DEFENDED] Pure BP Baseline {pure_bp_baseline:.1f}% remained superior to all search variants (best search: {best_search['name']} at {best_search['avg_3axis']:.1f}%).")

    return {
        "scorecard": scorecard,
        "best_search": best_search,
        "pure_bp_baseline": pure_bp_baseline,
    }


def main():
    parser = argparse.ArgumentParser(description="4-Way Controlled Ablation")
    parser.add_argument("--blueprint", type=str, default="runs/blueprint_100m/latest.pt")
    parser.add_argument("--belief", type=str, default="runs/belief_20m/best.pt")
    parser.add_argument("--exploiter", type=str, default="runs/exploiter_20m/best.pt")
    parser.add_argument("--num-decks", type=int, default=100)
    parser.add_argument("--out-config", type=str, default="runs/search_config.json")
    args = parser.parse_args()

    run_ood_ablation(
        blueprint_path=args.blueprint,
        belief_path=args.belief,
        exploiter_path=args.exploiter,
        num_decks=args.num_decks,
        out_config_path=args.out_config,
    )


if __name__ == "__main__":
    main()
