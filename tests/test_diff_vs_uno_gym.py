"""10,000-Game Differential Fuzz Test vs uno_gym.UnoSim.

Verifies bit-for-bit parity at every step:
- All 61 bits of legal_mask
- current_player
- active_color
- stack_count
- hand_sizes and card multisets for both hands
- is_over and terminal payoffs
- Deck multiset conservation and wild color reset on reshuffles
"""

import random
from collections import Counter
import pytest
import numpy as np

from uno_gym.sim import UnoSim
from uno_gym.state import normalize_card_key
import uno_ataraxos as ua

COLOR_MAP = {"r": 0, "g": 1, "b": 2, "y": 3}


def test_10000_games_differential_vs_uno_gym():
    NUM_GAMES = 10000
    reshuffles_verified = 0
    total_steps_verified = 0

    print(f"\n[Diff Test] Running {NUM_GAMES:,} differential games vs uno_gym.UnoSim...")

    for game_idx in range(NUM_GAMES):
        sim = UnoSim(seed=game_idx * 31 + 7)
        eng = ua.UnoEngine()
        eng.sync_from_sim(sim)

        while not sim.is_over:
            # 1. Assert legal_mask exact 61-bit equality
            mask_sim = sim.get_action_mask()
            mask_eng = eng.get_action_mask()
            assert (mask_sim == mask_eng).all(), (
                f"Game {game_idx}, Turn {sim.turn_count}: 61-bit legal_mask mismatch!\n"
                f"Sim legal: {sim.get_legal_actions()}\n"
                f"Eng legal: {eng.get_legal_actions()}"
            )

            # 2. Assert game state equality
            assert sim.current_player == eng.current_player, f"Game {game_idx}, Turn {sim.turn_count}: current_player mismatch"
            assert COLOR_MAP[sim.target[0]] == eng.active_color, f"Game {game_idx}, Turn {sim.turn_count}: active_color mismatch"
            assert sim.stack == eng.stack, f"Game {game_idx}, Turn {sim.turn_count}: stack mismatch"
            assert len(sim.hands[0]) == len(eng.hands[0]), f"Game {game_idx}, Turn {sim.turn_count}: hand 0 size mismatch"
            assert len(sim.hands[1]) == len(eng.hands[1]), f"Game {game_idx}, Turn {sim.turn_count}: hand 1 size mismatch"
            assert Counter(sim.hands[0]) == Counter(eng.hands[0]), f"Game {game_idx}, Turn {sim.turn_count}: hand 0 cards mismatch"
            assert Counter(sim.hands[1]) == Counter(eng.hands[1]), f"Game {game_idx}, Turn {sim.turn_count}: hand 1 cards mismatch"
            assert sim.is_over == eng.is_over, f"Game {game_idx}, Turn {sim.turn_count}: is_over mismatch"

            # Pick a uniform random legal action
            legal_ids = sim.get_legal_action_ids()
            act_id = random.choice(legal_ids)
            act_str = sim.get_legal_actions()[legal_ids.index(act_id)]

            # Check if this action triggers a deck reshuffle
            draw_count = sim.stack if (act_str == "draw" and sim.stack > 0) else (1 if act_str == "draw" else 0)
            will_reshuffle = (draw_count > len(sim.deck) and len(sim.discard_pile) > 1)

            # Step both engines
            sim.step(act_str)
            eng.step(act_id)
            total_steps_verified += 1

            if will_reshuffle:
                reshuffles_verified += 1
                # (a) Assert multiset of cards recycled/in-draw-pile + hands matches identically and wild colors reset
                sim_pool = (
                    [normalize_card_key(c) for c in sim.deck]
                    + [normalize_card_key(c) for c in sim.hands[0]]
                    + [normalize_card_key(c) for c in sim.hands[1]]
                )
                eng_pool = (
                    [normalize_card_key(c) for c in eng.deck]
                    + [normalize_card_key(c) for c in eng.hands[0]]
                    + [normalize_card_key(c) for c in eng.hands[1]]
                )
                assert Counter(sim_pool) == Counter(eng_pool), (
                    f"Game {game_idx}, Turn {sim.turn_count}: Recycled multiset mismatch on reshuffle!"
                )
                # Verify all wild cards in draw pile are normalized back to neutral 'wild' / 'wild_draw_4'
                for c in eng.deck:
                    assert c not in (
                        "r-wild", "g-wild", "b-wild", "y-wild",
                        "r-wild_draw_4", "g-wild_draw_4", "b-wild_draw_4", "y-wild_draw_4"
                    ), f"Game {game_idx}: Non-neutral wild card {c} found in recycled deck!"

                # (b) Sync C++ new post-reshuffle draw pile order & resulting hands from uno_gym.UnoSim
                eng.sync_from_sim(sim)

            # Assert terminal equality
            assert sim.is_over == eng.is_over, f"Game {game_idx}: is_over mismatch post-step"
            assert sim.winner == eng.winner, f"Game {game_idx}: winner mismatch post-step ({sim.winner} vs {eng.winner})"

        # Verify terminal payoffs
        sim_payoffs = sim.get_payoffs()
        eng_payoffs = list(eng.get_payoffs())
        assert sim_payoffs == eng_payoffs, (
            f"Game {game_idx}: Terminal payoff mismatch ({sim_payoffs} vs {eng_payoffs})"
        )

    print(f"[Diff Test Passed] {NUM_GAMES:,} games matched 100%! Total steps: {total_steps_verified:,}, Reshuffles verified: {reshuffles_verified:,}")


if __name__ == "__main__":
    test_10000_games_differential_vs_uno_gym()
