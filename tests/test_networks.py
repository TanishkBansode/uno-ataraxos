"""Tests for UnoActorNet and UnoQCriticNet Transformer architectures on Apple Silicon MPS."""

import unittest
import torch
import numpy as np

import uno_ataraxos as ua
from uno_ataraxos.networks import UnoActorNet, UnoQCriticNet


class TestNetworks(unittest.TestCase):

    def setUp(self):
        self.device = ua.get_device()

    def test_01_parameter_counts(self):
        """Verify parameter counts of UnoActorNet (~2.2M) and UnoQCriticNet (~2.3M)."""
        actor = UnoActorNet().to(self.device)
        critic = UnoQCriticNet().to(self.device)

        actor_params = sum(p.numel() for p in actor.parameters())
        critic_params = sum(p.numel() for p in critic.parameters())

        self.assertGreater(actor_params, 1_900_000)
        self.assertLess(actor_params, 2_400_000)

        self.assertGreater(critic_params, 1_800_000)
        self.assertLess(critic_params, 2_400_000)

    def test_02_turn_0_all_padded_history_zero_nan_on_mps(self):
        """Turn 0: All 48 history tokens are padded. Forward & backward must produce ZERO NaNs on MPS."""
        B = 4
        actor = UnoActorNet().to(self.device)
        critic = UnoQCriticNet().to(self.device)

        # Fresh Turn 0 state from reset VecUnoEnv
        vec = ua.VecUnoEnv(num_envs=B, seed=42)
        obs = vec.reset()

        card_feats = torch.from_numpy(obs["card_features_actor"]).to(self.device)
        global_feats = torch.from_numpy(obs["global_features"]).to(self.device)
        hist_tokens = torch.from_numpy(obs["history_tokens"]).to(self.device)
        opp_hand = torch.from_numpy(obs["opp_hand_counts"]).to(self.device)

        # Assert that history is indeed all padded on Turn 0 (actor_rel == 2)
        self.assertTrue((hist_tokens[..., 0] == 2).all(), "Turn 0 history should be completely padded")

        # 1. Actor Forward
        policy_logits, wdl_logits = actor(card_feats, global_feats, hist_tokens)
        self.assertEqual(policy_logits.shape, (B, 61))
        self.assertEqual(wdl_logits.shape, (B, 3))
        self.assertEqual(policy_logits.dtype, torch.float32)
        self.assertEqual(wdl_logits.dtype, torch.float32)
        self.assertFalse(torch.isnan(policy_logits).any(), "NaN found in policy logits on Turn 0")
        self.assertFalse(torch.isinf(policy_logits).any(), "Inf found in policy logits on Turn 0")
        self.assertFalse(torch.isnan(wdl_logits).any(), "NaN found in wdl logits on Turn 0")

        # 2. Critic Forward
        q_values = critic(card_feats, opp_hand, global_feats, hist_tokens)
        self.assertEqual(q_values.shape, (B, 61))
        self.assertEqual(q_values.dtype, torch.float32)
        self.assertFalse(torch.isnan(q_values).any(), "NaN found in Q-values on Turn 0")
        self.assertFalse(torch.isinf(q_values).any(), "Inf found in Q-values on Turn 0")
        # Check tanh bounds [-1.0, 1.0]
        self.assertTrue((q_values >= -1.0).all() and (q_values <= 1.0).all())

        # 3. Backward Pass Gradient Flow on Turn 0
        loss_actor = policy_logits.sum() + wdl_logits.sum()
        loss_actor.backward()
        for name, p in actor.named_parameters():
            if p.requires_grad and p.grad is not None:
                self.assertFalse(torch.isnan(p.grad).any(), f"NaN in actor gradient {name} on Turn 0")
                self.assertFalse(torch.isinf(p.grad).any(), f"Inf in actor gradient {name} on Turn 0")

        loss_critic = q_values.sum()
        loss_critic.backward()
        for name, p in critic.named_parameters():
            if p.requires_grad and p.grad is not None:
                self.assertFalse(torch.isnan(p.grad).any(), f"NaN in critic gradient {name} on Turn 0")
                self.assertFalse(torch.isinf(p.grad).any(), f"Inf in critic gradient {name} on Turn 0")

    def test_03_mid_game_observations_with_history(self):
        """Mid-game observations with real history events pass through smoothly."""
        B = 8
        actor = UnoActorNet().to(self.device)
        critic = UnoQCriticNet().to(self.device)

        vec = ua.VecUnoEnv(num_envs=B, seed=99)
        for _ in range(25):
            vec.step_random_legal()

        obs = vec.get_obs()
        card_feats = torch.from_numpy(obs["card_features_actor"]).to(self.device)
        global_feats = torch.from_numpy(obs["global_features"]).to(self.device)
        hist_tokens = torch.from_numpy(obs["history_tokens"]).to(self.device)
        opp_hand = torch.from_numpy(obs["opp_hand_counts"]).to(self.device)

        # Has at least some real history events
        self.assertTrue((hist_tokens[..., 0] != 2).any(), "Mid-game should have non-padded history")

        policy_logits, wdl_logits = actor(card_feats, global_feats, hist_tokens)
        q_values = critic(card_feats, opp_hand, global_feats, hist_tokens)

        self.assertFalse(torch.isnan(policy_logits).any())
        self.assertFalse(torch.isnan(wdl_logits).any())
        self.assertFalse(torch.isnan(q_values).any())

    def test_04_no_opp_hand_leak_to_actor(self):
        """Confirm UnoQCriticNet output changes when opp_hand_counts changes, while UnoActorNet has no privileged input."""
        B = 2
        critic = UnoQCriticNet().to(self.device)
        critic.eval()

        vec = ua.VecUnoEnv(num_envs=B, seed=77)
        obs = vec.reset()

        card_feats = torch.from_numpy(obs["card_features_actor"]).to(self.device)
        global_feats = torch.from_numpy(obs["global_features"]).to(self.device)
        hist_tokens = torch.from_numpy(obs["history_tokens"]).to(self.device)
        opp_hand_1 = torch.from_numpy(obs["opp_hand_counts"]).to(self.device)

        # Create a completely different opp_hand
        opp_hand_2 = opp_hand_1.clone()
        opp_hand_2[:, 0:10] += 2.0  # add cards to opp hand

        with torch.no_grad():
            q1 = critic(card_feats, opp_hand_1, global_feats, hist_tokens)
            q2 = critic(card_feats, opp_hand_2, global_feats, hist_tokens)

        # Q-values must change because critic observes privileged opp_hand_counts!
        self.assertFalse(torch.allclose(q1, q2, atol=1e-4), "Critic output should change when opp_hand changes")

    def test_05_bfloat16_autocast_stability_on_mps(self):
        """bfloat16 autocast forward + VRPO/EMAgnet KL loss + backward + AdamW step on MPS.

        Verifies zero NaNs/Infs on both Turn-0 (all-padded history) and mid-game states.
        """
        device = self.device
        device_type = "mps" if device.type == "mps" else ("cuda" if device.type == "cuda" else "cpu")
        autocast_dtype = torch.bfloat16

        actor = UnoActorNet().to(device)
        critic = UnoQCriticNet().to(device)
        ema_actor = UnoActorNet().to(device)
        ema_actor.load_state_dict(actor.state_dict())
        ema_actor.eval()

        optimizer = torch.optim.AdamW(
            list(actor.parameters()) + list(critic.parameters()),
            lr=3e-4,
        )

        # Test on both Turn-0 (all-padded) and mid-game (populated history) batches
        for is_turn_0 in [True, False]:
            B = 8
            vec = ua.VecUnoEnv(num_envs=B, seed=123)
            obs = vec.reset()
            if not is_turn_0:
                for _ in range(20):
                    vec.step_random_legal()
                obs = vec.get_obs()

            card_feats = torch.from_numpy(obs["card_features_actor"]).to(device)
            global_feats = torch.from_numpy(obs["global_features"]).to(device)
            hist_tokens = torch.from_numpy(obs["history_tokens"]).to(device)
            opp_hand = torch.from_numpy(obs["opp_hand_counts"]).to(device)
            legal_mask = torch.from_numpy(obs["legal_mask"]).to(device)

            actions = torch.randint(0, 61, (B,), device=device)
            q_targets = torch.randn(B, dtype=torch.float32, device=device).clamp(-1.0, 1.0)
            wdl_targets = torch.randint(0, 3, (B,), device=device)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device_type, dtype=autocast_dtype):
                # 1. Critic forward & backward
                q_values = critic(card_feats, opp_hand, global_feats, hist_tokens)
                self.assertEqual(q_values.dtype, torch.float32)
                self.assertFalse(torch.isnan(q_values).any())
                self.assertFalse(torch.isinf(q_values).any())

                q_loss = torch.nn.functional.mse_loss(
                    q_values.gather(1, actions.unsqueeze(1)).squeeze(1),
                    q_targets,
                )
                self.assertFalse(torch.isnan(q_loss))
                self.assertFalse(torch.isinf(q_loss))

            q_loss.backward()

            # Verify critic grads
            for name, p in critic.named_parameters():
                if p.requires_grad and p.grad is not None:
                    self.assertFalse(torch.isnan(p.grad).any(), f"NaN in critic grad {name} (turn_0={is_turn_0})")
                    self.assertFalse(torch.isinf(p.grad).any(), f"Inf in critic grad {name} (turn_0={is_turn_0})")

            # 2. EMA Actor forward (no_grad)
            with torch.no_grad(), torch.autocast(device_type=device_type, dtype=autocast_dtype):
                ref_logits, _ = ema_actor(card_feats, global_feats, hist_tokens)
                ref_logp = ua.masked_log_softmax(ref_logits, legal_mask)

            # 3. Actor forward & backward
            with torch.autocast(device_type=device_type, dtype=autocast_dtype):
                policy_logits, wdl_logits = actor(card_feats, global_feats, hist_tokens)
                self.assertEqual(policy_logits.dtype, torch.float32)
                self.assertEqual(wdl_logits.dtype, torch.float32)
                self.assertFalse(torch.isnan(policy_logits).any())
                self.assertFalse(torch.isinf(policy_logits).any())
                self.assertFalse(torch.isnan(wdl_logits).any())

                kl_loss = ua.masked_reverse_kl(policy_logits, ref_logp, legal_mask).mean()
                ent_loss = -0.01 * ua.masked_entropy(policy_logits, legal_mask).mean()
                wdl_loss = torch.nn.functional.cross_entropy(wdl_logits, wdl_targets)
                actor_loss = kl_loss + ent_loss + wdl_loss
                self.assertFalse(torch.isnan(actor_loss))
                self.assertFalse(torch.isinf(actor_loss))

            actor_loss.backward()

            # Verify actor grads
            for name, p in actor.named_parameters():
                if p.requires_grad and p.grad is not None:
                    self.assertFalse(torch.isnan(p.grad).any(), f"NaN in actor grad {name} (turn_0={is_turn_0})")
                    self.assertFalse(torch.isinf(p.grad).any(), f"Inf in actor grad {name} (turn_0={is_turn_0})")

            # 4. Optimizer step & EMA
            optimizer.step()
            ua.update_ema(ema_actor, actor, decay=0.999)

            # Check parameter finite
            for p in list(actor.parameters()) + list(critic.parameters()):
                self.assertFalse(torch.isnan(p).any())
                self.assertFalse(torch.isinf(p).any())


if __name__ == "__main__":
    unittest.main()
