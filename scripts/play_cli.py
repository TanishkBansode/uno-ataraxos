#!/usr/bin/env python3
"""Interactive Terminal 1v1 UNO CLI with live Belief X-Ray and Move Equity Display.

Usage:
    python scripts/play_cli.py --mode hybrid --blueprint runs/sanity_3m/latest.pt --belief runs/belief_smoke/best.pt --xray
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

from uno_ataraxos import UnoEngine
from uno_ataraxos.agent import AtaraxosUnoAgent
from uno_ataraxos.rl_math import get_device

# ANSI Color Codes for terminal UI
COLORS = {
    "r": "\033[91m",       # Red
    "g": "\033[92m",       # Green
    "b": "\033[94m",       # Blue
    "y": "\033[93m",       # Yellow
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "cyan": "\033[96m",
    "magenta": "\033[95m",
}


def color_card(card_str: str) -> str:
    """Add ANSI color coding to card strings."""
    if not card_str:
        return card_str
    prefix = card_str[0].lower()
    color = COLORS.get(prefix, "")
    return f"{color}{card_str}{COLORS['reset']}"


def main():
    parser = argparse.ArgumentParser(description="Play 1v1 UNO vs Ataraxos AI with Belief X-Ray")
    parser.add_argument("--mode", type=str, default="hybrid", choices=["blueprint", "mmd_search", "hybrid"], help="Agent mode")
    parser.add_argument("--blueprint", type=str, default="runs/sanity_3m/latest.pt", help="Path to blueprint checkpoint")
    parser.add_argument("--belief", type=str, default="runs/belief_smoke/best.pt", help="Path to belief checkpoint")
    parser.add_argument("--critic", type=str, default=None, help="Path to critic checkpoint")
    parser.add_argument("--xray", action="store_true", default=True, help="Display live Belief X-Ray diagnostics")
    parser.add_argument("--human-player", type=int, default=0, choices=[0, 1], help="Seat of human player (0 or 1)")
    parser.add_argument("--seed", type=int, default=None, help="Initial random seed")
    parser.add_argument("--K", type=int, default=128, help="Number of belief worlds")
    parser.add_argument("--depth", type=int, default=12, help="Rollout depth")
    parser.add_argument("--portfolio", action="store_true", default=True, help="Enable 3-style portfolio rollout search")
    parser.add_argument("--no-portfolio", dest="portfolio", action="store_false", help="Disable portfolio rollout search")
    parser.add_argument("--adaptive", action="store_true", default=True, help="Enable online Bayesian opponent tracking")
    parser.add_argument("--no-adaptive", dest="adaptive", action="store_false", help="Disable online Bayesian tracking")
    args = parser.parse_args()

    device = get_device()
    print(f"\n{COLORS['bold']}{'='*60}")
    print("      ATARAXOS 1v1 UNO - TEST-TIME SEARCH & BELIEF X-RAY")
    print(f"{'='*60}{COLORS['reset']}")
    print(f"Device: {device} | Mode: {args.mode.upper()} | Portfolio: {args.portfolio} | Adaptive Bayes: {args.adaptive}")
    print(f"Loading AI agent...")

    agent = AtaraxosUnoAgent(
        blueprint_ckpt=args.blueprint,
        belief_ckpt=args.belief,
        critic_ckpt=args.critic,
        mode=args.mode,
        device=device,
        K=args.K,
        depth=args.depth,
        use_portfolio=args.portfolio,
        use_adaptive_bayes=args.adaptive,
    )
    print(f"AI loaded successfully! You are Player {args.human_player}.\n")

    # Initialize Game Engine
    seed = args.seed if args.seed is not None else int(torch.randint(1, 1000000, (1,)).item())
    engine = UnoEngine(seed)
    engine.reset()
    agent.reset()

    human_p = args.human_player
    ai_p = 1 - human_p

    col_names = ["Red", "Green", "Blue", "Yellow"]

    while not engine.is_over:
        curr_p = int(engine.current_player)
        target = engine.target_str
        act_col = engine.active_color
        stack = engine.stack
        turn = engine.turn_count

        print(f"\n{COLORS['dim']}--- Turn {turn} | Target: {color_card(target)} | Active Color: {col_names[act_col]} | Stack: {stack} ---{COLORS['reset']}")

        h_human = engine.hands[human_p]
        h_ai_len = len(engine.get_hand_ids(ai_p))
        print(f"AI Hand Size: {COLORS['bold']}{h_ai_len}{COLORS['reset']} cards | Deck: {len(engine.get_deck_ids())} cards")

        if curr_p == human_p:
            # Human Turn
            legal_ids = engine.get_legal_action_ids(human_p)
            legal_strs = engine.get_legal_actions(human_p)

            print(f"\n{COLORS['bold']}Your Hand ({len(h_human)} cards):{COLORS['reset']}")
            print("  " + "  ".join([f"[{color_card(c)}]" for c in h_human]))

            print(f"\nLegal Actions:")
            for idx, (a_id, a_str) in enumerate(zip(legal_ids, legal_strs)):
                print(f"  ({idx}) {color_card(a_str)} [ID {a_id}]")

            while True:
                try:
                    choice = input(f"\nSelect action (0-{len(legal_ids)-1}): ").strip()
                    choice_idx = int(choice)
                    if 0 <= choice_idx < len(legal_ids):
                        chosen_act = legal_ids[choice_idx]
                        break
                    print("Invalid index. Try again.")
                except EOFError:
                    print("\nGame exited by user.")
                    return
                except ValueError:
                    print("Please enter a valid integer index.")

            print(f">> You played: {color_card(legal_strs[choice_idx])}")
            engine.step(chosen_act)

        else:
            # AI Turn
            print(f"\n{COLORS['cyan']}AI is thinking...{COLORS['reset']}")
            chosen_act, s_res = agent.act_from_engine(engine, deterministic=True)
            legal_ids = engine.get_legal_action_ids(ai_p)
            legal_strs = engine.get_legal_actions(ai_p)
            act_name_map = dict(zip(legal_ids, legal_strs))
            chosen_str = act_name_map.get(chosen_act, f"Action {chosen_act}")

            if s_res is not None and args.xray:
                print(f"\n{COLORS['magenta']}=== [BELIEF X-RAY DIAGNOSTICS] ==={COLORS['reset']}")
                if hasattr(agent, "opp_tracker"):
                    w = agent.opp_tracker.weights
                    s = agent.opp_tracker.ema_surprisal
                    _, u_mix = agent.opp_tracker.get_belief_params()
                    print(f"  Opponent Profile: [BP: {w[0]*100:4.1f}% | Aggressive: {w[1]*100:4.1f}% | Unpredictable: {w[2]*100:4.1f}%] | Surprisal: {s:4.2f} | Belief Mix: {u_mix*100:4.1f}%")

                xr = s_res.belief_xray
                print(f"  Belief over YOUR hand ({int(xr['opp_hand_size'])} cards):")
                print(f"    P(You hold +2):     {xr['p_plus_2']*100:5.1f}%")
                print(f"    P(You hold +4):     {xr['p_plus_4']*100:5.1f}%")
                print(f"    P(You hold Red):    {xr['p_red']*100:5.1f}%  | P(Green): {xr['p_green']*100:5.1f}%")
                print(f"    P(You hold Blue):   {xr['p_blue']*100:5.1f}%  | P(Yellow): {xr['p_yellow']*100:5.1f}%")

                print(f"\n  Search Equities & Action Distribution:")
                for a_id in legal_ids:
                    p = s_res.search_probs[a_id].item()
                    q = s_res.q_hat[a_id].item()
                    name = act_name_map.get(a_id, str(a_id))
                    star = " *" if a_id == chosen_act else ""
                    print(f"    {color_card(name):15s} -> Prob: {p*100:5.1f}% | Q: {q:+6.3f}{star}")
                print(f"  Latency: {s_res.latency_ms:.1f} ms")
                print(f"{COLORS['magenta']}==================================={COLORS['reset']}")

            print(f">> AI played: {color_card(chosen_str)}")
            engine.step(chosen_act)

    print(f"\n{COLORS['bold']}{'='*60}")
    print("                     GAME OVER")
    print(f"{'='*60}{COLORS['reset']}")
    if engine.winner == human_p:
        print(f"{COLORS['bold']}{COLORS['g']}CONGRATULATIONS! You won the game!{COLORS['reset']}\n")
    elif engine.winner == ai_p:
        print(f"{COLORS['bold']}{COLORS['r']}AI won the game! Better luck next time.{COLORS['reset']}\n")
    else:
        print("Game ended in a DRAW (turn limit or empty passes).\n")


if __name__ == "__main__":
    main()
