"""Tests for S4 (24-fold) color permutation invariance and tensor transformations."""

import unittest
import torch
import numpy as np

import uno_ataraxos as ua
from uno_ataraxos.symmetry import (
    COLOR_PERMUTATIONS,
    NUM_PERMUTATIONS,
    FWD_CARD_54,
    INV_CARD_54,
    FWD_ACTION_62,
    INV_ACTION_61,
    FWD_COLOR_5,
    INV_COLOR_5,
    INV_PERM_ID,
    get_inverse_perm_id,
    apply_s4_batch,
)


class TestSymmetry(unittest.TestCase):

    def setUp(self):
        self.device = ua.get_device()

    def test_01_permutation_table_shapes_and_invariants(self):
        """Verify dimensions and mathematical bijection of all S4 lookup tables."""
        self.assertEqual(len(COLOR_PERMUTATIONS), 24)
        self.assertEqual(FWD_CARD_54.shape, (24, 54))
        self.assertEqual(INV_CARD_54.shape, (24, 54))
        self.assertEqual(FWD_ACTION_62.shape, (24, 62))
        self.assertEqual(INV_ACTION_61.shape, (24, 61))
        self.assertEqual(FWD_COLOR_5.shape, (24, 5))
        self.assertEqual(INV_COLOR_5.shape, (24, 5))

        for p in range(24):
            # Card bijection
            card_fwd = FWD_CARD_54[p]
            card_inv = INV_CARD_54[p]
            self.assertTrue(torch.equal(card_inv[card_fwd], torch.arange(54)))
            self.assertTrue(torch.equal(card_fwd[card_inv], torch.arange(54)))

            # Action bijection (0..60)
            act_fwd = FWD_ACTION_62[p, :61]
            act_inv = INV_ACTION_61[p]
            self.assertTrue(torch.equal(act_inv[act_fwd], torch.arange(61)))
            self.assertTrue(torch.equal(act_fwd[act_inv], torch.arange(61)))

            # Padding tokens must map to themselves
            self.assertEqual(FWD_ACTION_62[p, 61].item(), 61)  # padded action stays 61
            self.assertEqual(FWD_ACTION_62[p, 60].item(), 60)  # draw stays 60
            self.assertEqual(FWD_CARD_54[p, 52].item(), 52)    # wild stays 52
            self.assertEqual(FWD_CARD_54[p, 53].item(), 53)    # wild_draw_4 stays 53
            self.assertEqual(FWD_COLOR_5[p, 4].item(), 4)      # padded color stays 4

    def test_02_inverse_permutation_roundtrip(self):
        """Applying permutation p followed by p^-1 must strictly yield the original permutation."""
        for p in range(24):
            inv_p = get_inverse_perm_id(p)
            sigma = COLOR_PERMUTATIONS[p]
            sigma_inv = COLOR_PERMUTATIONS[inv_p]
            # sigma_inv(sigma(c)) == c
            for c in range(4):
                self.assertEqual(sigma_inv[sigma[c]], c)

    def test_03_legal_mask_and_fwd_action_consistency(self):
        """Check invariant: legal_mask_new[fwd_action[p, a_old]] == legal_mask_old[a_old]."""
        # Pull real mid-game states from VecUnoEnv
        vec = ua.VecUnoEnv(num_envs=32, seed=123)
        # Step a few random moves to get rich mid-game states
        for _ in range(15):
            vec.step_random_legal()

        obs = vec.get_obs()
        legal_mask = torch.from_numpy(obs["legal_mask"]).to(self.device)  # [32, 61]
        card_feats = torch.from_numpy(obs["card_features_actor"]).to(self.device)
        hist_tokens = torch.from_numpy(obs["history_tokens"]).to(self.device)
        opp_hand = torch.from_numpy(obs["opp_hand_counts"]).to(self.device)

        # Test every single permutation p in S4
        for p in range(24):
            c_new, h_new, m_new, opp_new = apply_s4_batch(
                card_feats, hist_tokens, legal_mask, opp_hand_counts=opp_hand, perm_ids=p
            )
            # Verify for all 32 envs and all 61 actions
            fwd_actions = FWD_ACTION_62[p, :61].to(self.device)
            for old_a in range(61):
                new_a = fwd_actions[old_a].item()
                self.assertTrue(
                    torch.equal(m_new[:, new_a], legal_mask[:, old_a]),
                    f"Permutation {p}: legal_mask_new[{new_a}] != legal_mask_old[{old_a}]"
                )

    def test_04_card_count_conservation(self):
        """Sum of cards across colors must be strictly conserved under any permutation."""
        vec = ua.VecUnoEnv(num_envs=16, seed=456)
        for _ in range(10):
            vec.step_random_legal()

        obs = vec.get_obs()
        card_feats = torch.from_numpy(obs["card_features_actor"]).to(self.device)
        hist_tokens = torch.from_numpy(obs["history_tokens"]).to(self.device)
        legal_mask = torch.from_numpy(obs["legal_mask"]).to(self.device)
        opp_hand = torch.from_numpy(obs["opp_hand_counts"]).to(self.device)

        for p in range(24):
            c_new, h_new, m_new, opp_new = apply_s4_batch(
                card_feats, hist_tokens, legal_mask, opp_hand_counts=opp_hand, perm_ids=p
            )
            # Hand card sum (channel 0)
            self.assertTrue(torch.allclose(card_feats[:, :, 0].sum(dim=1), c_new[:, :, 0].sum(dim=1)))
            # Public discard sum (channel 1)
            self.assertTrue(torch.allclose(card_feats[:, :, 1].sum(dim=1), c_new[:, :, 1].sum(dim=1)))
            # Opponent hand sum
            self.assertTrue(torch.allclose(opp_hand.sum(dim=1), opp_new.sum(dim=1)))

    def test_05_roundtrip_identity(self):
        """Applying permutation p followed by p^-1 on tensors must recover exact original tensors."""
        vec = ua.VecUnoEnv(num_envs=8, seed=789)
        for _ in range(8):
            vec.step_random_legal()

        obs = vec.get_obs()
        card_feats = torch.from_numpy(obs["card_features_actor"]).to(self.device)
        hist_tokens = torch.from_numpy(obs["history_tokens"]).to(self.device)
        legal_mask = torch.from_numpy(obs["legal_mask"]).to(self.device)
        opp_hand = torch.from_numpy(obs["opp_hand_counts"]).to(self.device)
        actions = torch.randint(0, 61, (8,), device=self.device)

        for p in range(24):
            inv_p = get_inverse_perm_id(p)

            # Apply forward permutation p
            c_perm, h_perm, m_perm, opp_perm, a_perm = apply_s4_batch(
                card_feats, hist_tokens, legal_mask, opp_hand_counts=opp_hand, actions=actions, perm_ids=p
            )

            # Apply inverse permutation inv_p
            c_rec, h_rec, m_rec, opp_rec, a_rec = apply_s4_batch(
                c_perm, h_perm, m_perm, opp_hand_counts=opp_perm, actions=a_perm, perm_ids=inv_p
            )

            self.assertTrue(torch.equal(card_feats, c_rec), f"Card feats roundtrip failed on perm {p}")
            self.assertTrue(torch.equal(hist_tokens, h_rec), f"History tokens roundtrip failed on perm {p}")
            self.assertTrue(torch.equal(legal_mask, m_rec), f"Legal mask roundtrip failed on perm {p}")
            self.assertTrue(torch.equal(opp_hand, opp_rec), f"Opp hand roundtrip failed on perm {p}")
            self.assertTrue(torch.equal(actions, a_rec), f"Actions roundtrip failed on perm {p}")

    def test_06_padding_tokens_preserved(self):
        """Padding tokens in history (actor=2, action=61, color=4) must remain identical under all permutations."""
        # Create history with padded entries
        B = 4
        hist_tokens = torch.full((B, 48, 5), fill_value=2, dtype=torch.int32, device=self.device)
        hist_tokens[..., 1] = 61  # action pad
        hist_tokens[..., 2] = 4   # color pad
        hist_tokens[..., 3] = 0
        hist_tokens[..., 4] = 0

        card_feats = torch.zeros((B, 54, 7), dtype=torch.float32, device=self.device)
        legal_mask = torch.ones((B, 61), dtype=torch.bool, device=self.device)

        for p in range(24):
            _, h_new, _ = apply_s4_batch(card_feats, hist_tokens, legal_mask, perm_ids=p)
            self.assertTrue(torch.equal(h_new[..., 0], hist_tokens[..., 0]))
            self.assertTrue(torch.equal(h_new[..., 1], hist_tokens[..., 1]))
            self.assertTrue(torch.equal(h_new[..., 2], hist_tokens[..., 2]))


if __name__ == "__main__":
    unittest.main()
