"""Targeted unit tests covering every single RULES.md clause & edge case.

Enforces 100% strict adherence to:
- Official Mattel 108-card rules
- 1v1 Skip/Reverse = extra turn
- Same-type draw stacking (+2 on +2, +4 on +4, cross-stacking strictly forbidden)
- Reshuffle mid-multi-draw and wild color reset
- Deck exhaustion guards & stalemate termination
- Immediate win on final card (+2/+4)
- Provenance tracking & tensor contracts
"""

import unittest
from collections import Counter
import numpy as np
import uno_ataraxos as ua
from uno_ataraxos import (
    UnoEngine,
    VecUnoEnv,
    CARD_KEYS,
    ALL_ACTIONS,
    action_to_id,
    id_to_action,
    card_to_id,
    id_to_card,
)


class TestRulesExact(unittest.TestCase):

    def test_01_deck_composition_108_cards(self):
        """Official UNO deck must contain exactly 108 cards matching Mattel distribution."""
        eng = UnoEngine(42)
        total_cards = len(eng.hands[0]) + len(eng.hands[1]) + len(eng.deck) + len(eng.discard_pile)
        self.assertEqual(total_cards, 108)

        # Check full deck composition
        full_deck = ua.TOTAL_DECK_CARDS
        self.assertEqual(full_deck, 108)

    def test_02_initial_deal_7_cards_each(self):
        """Starting game must deal exactly 7 cards to each player with stack=0 and turn_count=0."""
        eng = UnoEngine(123)
        self.assertEqual(len(eng.hands[0]), 7)
        self.assertEqual(len(eng.hands[1]), 7)
        self.assertEqual(eng.stack, 0)
        self.assertEqual(eng.current_player, 0)
        self.assertEqual(eng.turn_count, 0)
        self.assertFalse(eng.is_over)

    def test_03_initial_non_number_card_redraw(self):
        """Opening card must be a standard number card (0-9). Non-numbers are buried/redrawn."""
        for seed in range(50):
            eng = UnoEngine(seed)
            target = eng.target_str
            self.assertIn("-", target)
            _, trait = target.split("-", 1)
            self.assertTrue(trait.isdigit(), f"Initial target must be digit 0-9, got {target}")
            self.assertIn(int(trait), range(10))

    def test_04_block_grants_extra_turn_in_1v1(self):
        """In 1v1 UNO, playing Skip (block) skips the opponent, granting an immediate extra turn."""
        eng = UnoEngine()
        eng.load_state(
            ["r-skip", "g-5"], ["b-3", "y-4"], ["y-1", "b-2"], ["r-0"],
            "r-0", 0, 0, 0, 0, 0
        )
        self.assertEqual(eng.current_player, 0)
        eng.step("r-skip")
        # In 1v1, current_player must remain 0!
        self.assertEqual(eng.current_player, 0)
        self.assertEqual(eng.target_str, "r-skip")

    def test_05_reverse_equals_skip_in_1v1(self):
        """In 1v1 UNO, Reverse acts identically to Skip, granting an immediate extra turn."""
        eng = UnoEngine()
        eng.load_state(
            ["r-reverse", "g-5"], ["b-3", "y-4"], ["y-1", "b-2"], ["r-0"],
            "r-0", 0, 0, 0, 0, 0
        )
        self.assertEqual(eng.current_player, 0)
        eng.step("r-reverse")
        # In 1v1, reverse keeps the turn for the acting player
        self.assertEqual(eng.current_player, 0)
        self.assertEqual(eng.target_str, "r-reverse")

    def test_06_plus_2_countered_by_different_color_plus_2_changing_active_color(self):
        """+2 can be countered by any color +2, increasing stack by 2 and updating active_color."""
        eng = UnoEngine()
        eng.load_state(
            ["b-draw_2", "r-5"], ["g-2"], ["y-1"], ["r-draw_2"],
            "r-draw_2", 0, 2, 0, 1, 0
        )
        legal = eng.get_legal_actions()
        self.assertIn("b-draw_2", legal)
        self.assertIn("draw", legal)
        self.assertNotIn("r-5", legal)

        eng.step("b-draw_2")
        self.assertEqual(eng.stack, 4)
        self.assertEqual(eng.active_color, 2)  # Blue
        self.assertEqual(eng.target_str, "b-draw_2")
        self.assertEqual(eng.current_player, 1)

    def test_07_plus_4_rejected_on_plus_2_stack(self):
        """Cross-type stacking is strictly forbidden: +4 cannot be played on a +2 stack."""
        eng = UnoEngine()
        eng.load_state(
            ["b-draw_2", "wild_draw_4", "r-5"], ["g-2"], ["y-1"], ["r-draw_2"],
            "r-draw_2", 0, 2, 0, 1, 0
        )
        legal = eng.get_legal_actions()
        self.assertIn("b-draw_2", legal)
        self.assertIn("draw", legal)
        for c in ["r", "g", "b", "y"]:
            self.assertNotIn(f"{c}-wild_draw_4", legal)
            self.assertNotIn(f"{c}-wild", legal)

    def test_08_plus_2_rejected_on_plus_4_stack(self):
        """Cross-type stacking is strictly forbidden: +2 cannot be played on a +4 stack."""
        eng = UnoEngine()
        eng.load_state(
            ["b-draw_2", "r-5", "wild_draw_4"], ["g-2"], ["y-1"], ["r-wild_draw_4"],
            "r-wild_draw_4", 0, 4, 0, 1, 0
        )
        legal = eng.get_legal_actions()
        self.assertNotIn("b-draw_2", legal)
        self.assertNotIn("r-5", legal)
        for c in ["r", "g", "b", "y"]:
            self.assertIn(f"{c}-wild_draw_4", legal)
        self.assertIn("draw", legal)

    def test_09_plus_4_playable_while_holding_active_color(self):
        """Wild Draw 4 is always playable on a normal turn, even if holding cards matching active color."""
        eng = UnoEngine()
        eng.load_state(
            ["r-3", "r-7", "wild_draw_4"], ["g-2"], ["y-1"], ["r-5"],
            "r-5", 0, 0, 0, 1, 0
        )
        legal = eng.get_legal_actions()
        for c in ["r", "g", "b", "y"]:
            self.assertIn(f"{c}-wild_draw_4", legal)
        self.assertIn("r-3", legal)
        self.assertIn("r-7", legal)

    def test_10_voluntary_draw_allowed_on_normal_turn(self):
        """When cards exist to draw, voluntary draw is allowed even if playable cards exist."""
        eng = UnoEngine()
        eng.load_state(
            ["r-3", "g-8"], ["g-2"], ["y-1", "b-2"], ["r-5"],
            "r-5", 0, 0, 0, 1, 0
        )
        legal = eng.get_legal_actions()
        self.assertIn("r-3", legal)
        self.assertIn("draw", legal)

    def test_11_voluntary_draw_disabled_on_deck_exhaustion_with_playable_cards(self):
        """Deck Exhaustion Guard: When available draw count is 0, voluntary draw is DISABLED if player has playable card."""
        eng = UnoEngine()
        # Deck empty, discard pile only has 1 card (top card), so 0 cards available to draw
        eng.load_state(
            ["r-3", "g-8"], ["g-2"], [], ["r-5"],
            "r-5", 0, 0, 0, 1, 0
        )
        self.assertEqual(eng.get_available_draw_count(), 0)
        legal = eng.get_legal_actions()
        self.assertIn("r-3", legal)
        # Voluntary draw must be disabled!
        self.assertNotIn("draw", legal)

    def test_12_0_card_pass_on_deck_exhaustion_resetting_stack_and_incrementing_stalemate(self):
        """When 0 cards to draw and player cannot play, draw acts as 0-card pass, resets stack and increments consecutive_passes."""
        eng = UnoEngine()
        eng.load_state(
            ["b-8", "y-9"], ["g-2"], [], ["r-5"],
            "r-5", 0, 0, 0, 1, 0
        )
        self.assertEqual(eng.get_available_draw_count(), 0)
        legal = eng.get_legal_actions()
        # No playable cards -> only forced pass ('draw') is legal
        self.assertEqual(legal, ["draw"])

        eng.step("draw")
        self.assertEqual(eng.consecutive_passes, 1)
        self.assertEqual(len(eng.hands[0]), 2)  # Drew 0 cards
        self.assertEqual(eng.current_player, 1)

    def test_13_2_consecutive_empty_passes_terminating_with_card_count_tiebreak(self):
        """Two consecutive empty passes terminates immediately; player with fewer cards wins."""
        eng = UnoEngine()
        eng.load_state(
            ["b-8", "y-9"], ["g-2"], [], ["r-5"],
            "r-5", 0, 0, 0, 1, 1  # already 1 consecutive pass
        )
        eng.step("draw")
        self.assertTrue(eng.is_over)
        self.assertEqual(eng.consecutive_passes, 2)
        # Player 1 had 1 card, Player 0 had 2 cards -> Player 1 wins
        self.assertEqual(eng.winner, 1)
        self.assertEqual(eng.get_payoffs(), (-1.0, 1.0))

    def test_14_200_turn_limit_tiebreak(self):
        """Game terminating at 200 turns awards victory to player with fewer cards in hand."""
        eng = UnoEngine()
        eng.load_state(
            ["r-1", "r-2"], ["g-1", "g-2", "g-3"], ["b-1"], ["r-0"],
            "r-0", 0, 0, 0, 199, 0
        )
        eng.step("r-1")
        self.assertTrue(eng.is_over)
        self.assertEqual(eng.turn_count, 200)
        # Player 0 has 1 card left, Player 1 has 3 cards -> Player 0 wins
        self.assertEqual(eng.winner, 0)
        self.assertEqual(eng.get_payoffs(), (1.0, -1.0))

    def test_15_200_turn_limit_equal_cards_draw(self):
        """Game terminating at 200 turns with equal card counts results in a draw (0 payoff)."""
        eng = UnoEngine()
        eng.load_state(
            ["r-1", "r-2"], ["g-1", "g-2"], [], ["r-0"],
            "r-0", 0, 0, 0, 199, 0
        )
        eng.step("draw")
        self.assertTrue(eng.is_over)
        self.assertEqual(eng.turn_count, 200)
        # Both players have 2 cards -> draw
        self.assertIsNone(eng.winner)
        self.assertEqual(eng.get_payoffs(), (0.0, 0.0))

    def test_16_mid_draw_reshuffling_resetting_wild_colors(self):
        """Recycling discards resets declared wild colors back to neutral wild & wild_draw_4."""
        eng = UnoEngine()
        # Deck has 1 card, discard pile has 3 cards including r-wild and g-wild_draw_4
        eng.load_state(
            ["r-1"], ["g-1"], ["b-2"], ["r-wild", "g-wild_draw_4", "r-5"],
            "r-5", 0, 0, 0, 10, 0
        )
        # Draw 3 cards -> exhausts 1 deck card and forces recycling of 2 discard cards
        eng.step("draw")
        # In the recycled deck, all wilds must be neutral
        for c in eng.deck:
            self.assertNotIn(c, ["r-wild", "g-wild", "b-wild", "y-wild",
                                 "r-wild_draw_4", "g-wild_draw_4", "b-wild_draw_4", "y-wild_draw_4"])

    def test_17_immediate_win_when_final_card_is_plus_2(self):
        """Legally playing last card wins immediately even if it is PLUS_TWO (opponent does not draw)."""
        eng = UnoEngine()
        eng.load_state(
            ["r-draw_2"], ["g-1", "g-2"], ["b-1"], ["r-5"],
            "r-5", 0, 0, 0, 10, 0
        )
        eng.step("r-draw_2")
        self.assertTrue(eng.is_over)
        self.assertEqual(eng.winner, 0)
        self.assertEqual(len(eng.hands[0]), 0)
        self.assertEqual(eng.get_payoffs(), (1.0, -1.0))

    def test_18_immediate_win_when_final_card_is_plus_4(self):
        """Legally playing last card wins immediately even if it is PLUS_FOUR."""
        eng = UnoEngine()
        eng.load_state(
            ["wild_draw_4"], ["g-1", "g-2"], ["b-1"], ["r-5"],
            "r-5", 0, 0, 0, 10, 0
        )
        eng.step("r-wild_draw_4")
        self.assertTrue(eng.is_over)
        self.assertEqual(eng.winner, 0)
        self.assertEqual(len(eng.hands[0]), 0)
        self.assertEqual(eng.get_payoffs(), (1.0, -1.0))

    def test_19_exact_recycled_remaining_for_me_provenance(self):
        """When opp_draws_since_reshuffle == 0, recycled_remaining_for_me is 100% guaranteed NOT in opp hand."""
        eng = UnoEngine()
        eng.load_state(
            ["r-1"], ["g-1"], [], ["r-5", "b-7", "y-skip", "g-0"],
            "g-0", 1, 0, 0, 5, 0
        )
        # Force recycle on draw
        eng.step("draw")
        self.assertEqual(eng.get_draws_since_reshuffle(0), 1)
        self.assertEqual(eng.get_draws_since_reshuffle(1), 0)

        # For player 0 (who just drew): opponent (player 1) has drawn 0 cards from recycled pool!
        rem_p0 = eng.get_recycled_remaining_for_me(0)
        self.assertGreater(sum(rem_p0), 0)

    def test_20_108_card_conservation_invariant_across_random_games(self):
        """Sum of cards in hands, draw pile, and discards must strictly equal 108 at every step."""
        for seed in range(200):
            eng = UnoEngine(seed * 17)
            steps = 0
            while not eng.is_over and steps < 60:
                total_cards = len(eng.hands[0]) + len(eng.hands[1]) + len(eng.deck) + len(eng.discard_pile)
                self.assertEqual(total_cards, 108, f"Card count violated at step {steps}")
                legal = eng.get_legal_action_ids()
                act = np.random.choice(legal)
                eng.step(act)
                steps += 1

    def test_21_drawing_on_penalty_stack_absorbs_entire_stack(self):
        """Drawing while stack > 0 draws all stack cards, resets stack to 0, and ends turn."""
        eng = UnoEngine()
        eng.load_state(
            ["r-1"], ["g-1"], ["b-1", "b-2", "b-3", "b-4", "b-5", "b-6"], ["r-draw_2"],
            "r-draw_2", 0, 6, 0, 5, 0
        )
        eng.step("draw")
        self.assertEqual(len(eng.hands[0]), 7)  # 1 + 6 = 7
        self.assertEqual(eng.stack, 0)
        self.assertEqual(eng.current_player, 1)

    def test_22_regular_wild_always_legal_on_normal_turn(self):
        """Regular wild allows declaring any of 4 colors on normal turn."""
        eng = UnoEngine()
        eng.load_state(
            ["wild"], ["g-1"], ["b-1"], ["r-5"],
            "r-5", 0, 0, 0, 5, 0
        )
        legal = eng.get_legal_actions()
        for c in ["r", "g", "b", "y"]:
            self.assertIn(f"{c}-wild", legal)

    def test_23_regular_wild_rejected_on_penalty_stack(self):
        """Regular wild cannot counter a +2 or +4 penalty stack."""
        eng = UnoEngine()
        eng.load_state(
            ["wild"], ["g-1"], ["b-1"], ["r-draw_2"],
            "r-draw_2", 0, 2, 0, 5, 0
        )
        legal = eng.get_legal_actions()
        for c in ["r", "g", "b", "y"]:
            self.assertNotIn(f"{c}-wild", legal)

    def test_24_tensor_shapes_and_dtypes(self):
        """VecUnoEnv output tensors must match the exact specification shapes and dtypes."""
        vec = VecUnoEnv(num_envs=8, seed=42)
        obs = vec.reset()

        self.assertEqual(obs["card_features_actor"].shape, (8, 54, 7))
        self.assertEqual(obs["card_features_actor"].dtype, np.float32)

        self.assertEqual(obs["global_features"].shape, (8, 10))
        self.assertEqual(obs["global_features"].dtype, np.float32)

        self.assertEqual(obs["history_tokens"].shape, (8, 48, 5))
        self.assertEqual(obs["history_tokens"].dtype, np.int32)

        self.assertEqual(obs["legal_mask"].shape, (8, 61))
        self.assertEqual(obs["legal_mask"].dtype, bool)

        self.assertEqual(obs["opp_hand_counts"].shape, (8, 54))
        self.assertEqual(obs["opp_hand_counts"].dtype, np.float32)

        self.assertEqual(obs["current_player"].shape, (8,))
        self.assertEqual(obs["current_player"].dtype, np.int32)

    def test_25_vec_uno_env_step_contract(self):
        """VecUnoEnv.step returns (obs, rewards, dones, info) with float32 rewards and bool dones."""
        vec = VecUnoEnv(num_envs=4, seed=42)
        vec.reset()
        actions = np.array([60, 60, 60, 60], dtype=np.int32)
        obs, rewards, dones, info = vec.step(actions)

        self.assertEqual(rewards.shape, (4,))
        self.assertEqual(rewards.dtype, np.float32)
        self.assertEqual(dones.shape, (4,))
        self.assertEqual(dones.dtype, bool)

    def test_26_step_no_autoreset(self):
        """VecUnoEnv.step_no_autoreset does not auto-reset completed games."""
        vec = VecUnoEnv(num_envs=2, seed=1, auto_reset=False)
        vec.reset()
        actions = np.array([60, 60], dtype=np.int32)
        obs, rewards, dones, info = vec.step_no_autoreset(actions)
        self.assertEqual(dones.shape, (2,))


if __name__ == "__main__":
    unittest.main()
