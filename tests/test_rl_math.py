"""Tests for reinforcement learning mathematics, VRPO targets, and masked distributions."""

import unittest
import torch
import torch.nn as nn
import uno_ataraxos as ua
from uno_ataraxos.rl_math import (
    get_device,
    masked_log_softmax,
    masked_reverse_kl,
    masked_entropy,
    uniform_legal_log_probs,
    compute_vrpo_targets,
    update_ema,
    wdl_expected_value,
    reward_to_wdl_target,
)


class TestRLMath(unittest.TestCase):

    def setUp(self):
        self.device = get_device()

    def test_01_non_alternating_vrpo_skip_reverse_combo(self):
        """Construct synthetic 6-step game with 3-card Skip/Reverse combo ending in +1 win.

        Step sequence:
        t=0: Player 0 (Skip)    -> same_player_next=1 (P0 acts next)
        t=1: Player 0 (Reverse) -> same_player_next=1 (P0 acts next)
        t=2: Player 0 (Normal)  -> same_player_next=0 (P1 acts next)
        t=3: Player 1 (Skip)    -> same_player_next=1 (P1 acts next)
        t=4: Player 1 (Normal)  -> same_player_next=0 (P0 acts next)
        t=5: Player 0 (Wins!)   -> dones=1, reward=+1.0 (P0 wins)

        Expected:
        Player 0 steps (t=0, 1, 2, 5) must receive exact +1.0
        Player 1 steps (t=3, 4) must receive exact -1.0
        """
        T, B = 6, 1
        rewards = torch.zeros((T, B), dtype=torch.float32, device=self.device)
        rewards[5, 0] = 1.0  # Player 0 wins on final step

        dones = torch.zeros((T, B), dtype=torch.float32, device=self.device)
        dones[5, 0] = 1.0

        same_player_next = torch.tensor(
            [[1.0], [1.0], [0.0], [1.0], [0.0], [0.0]],
            dtype=torch.float32,
            device=self.device,
        )

        q_taken = torch.zeros((T, B), dtype=torch.float32, device=self.device)
        v_expected = torch.zeros((T + 1, B), dtype=torch.float32, device=self.device)

        q_targets, advantages = compute_vrpo_targets(
            rewards, dones, same_player_next, q_taken, v_expected, lam=1.0
        )

        expected_q = torch.tensor([[1.0], [1.0], [1.0], [-1.0], [-1.0], [1.0]], device=self.device)
        self.assertTrue(torch.allclose(q_targets, expected_q, atol=1e-5), f"Got:\n{q_targets}\nExpected:\n{expected_q}")
        self.assertTrue(torch.allclose(advantages, expected_q, atol=1e-5))

    def test_02_zero_variance_exact_critic_vrpo(self):
        """When Q and V are exact (Bellman error delta == 0), VRPO targets must exactly equal Q*."""
        T, B = 10, 4
        # Generate random Bellman-consistent trajectory
        v_expected = torch.empty((T + 1, B), dtype=torch.float32, device=self.device).uniform_(-0.8, 0.8)
        rewards = torch.zeros((T, B), dtype=torch.float32, device=self.device)
        rewards[T - 1] = torch.tensor([1.0, -1.0, 0.0, 1.0], device=self.device)

        dones = torch.zeros((T, B), dtype=torch.float32, device=self.device)
        dones[T - 1] = 1.0

        same_player_next = torch.randint(0, 2, (T, B), device=self.device).float()

        # Construct exact Q* satisfying Q* = R + (1-done) * sign * V_{next}
        q_star = torch.empty((T, B), dtype=torch.float32, device=self.device)
        for t in range(T):
            not_done = 1.0 - dones[t]
            sign = 2.0 * same_player_next[t] - 1.0
            q_star[t] = rewards[t] + not_done * sign * v_expected[t + 1]

        q_targets, _ = compute_vrpo_targets(
            rewards, dones, same_player_next, q_star, v_expected, lam=0.95
        )

        self.assertTrue(torch.allclose(q_targets, q_star, atol=1e-5))

    def test_03_episode_boundary_isolation(self):
        """Verify that rewards after dones=1.0 never leak backward across episodes."""
        T, B = 4, 1
        rewards = torch.tensor([[0.0], [1.0], [0.0], [-1.0]], dtype=torch.float32, device=self.device)
        dones = torch.tensor([[0.0], [1.0], [0.0], [1.0]], dtype=torch.float32, device=self.device)
        same_player_next = torch.zeros((T, B), dtype=torch.float32, device=self.device)

        q_taken = torch.zeros((T, B), dtype=torch.float32, device=self.device)
        v_expected = torch.zeros((T + 1, B), dtype=torch.float32, device=self.device)

        q_targets_1, _ = compute_vrpo_targets(rewards, dones, same_player_next, q_taken, v_expected, lam=0.95)

        # Alter the reward of the SECOND episode
        rewards_alt = rewards.clone()
        rewards_alt[3, 0] = 100.0  # huge reward in episode 2
        q_targets_2, _ = compute_vrpo_targets(rewards_alt, dones, same_player_next, q_taken, v_expected, lam=0.95)

        # Steps 0 and 1 (episode 1) must be completely identical!
        self.assertTrue(torch.allclose(q_targets_1[:2], q_targets_2[:2], atol=1e-6))

    def test_04_safe_kl_entropy_extreme_logits_and_gradients(self):
        """Test masked log-softmax, reverse KL, and entropy under extreme logits and 1-action masks."""
        B = 8
        logits = torch.randn(B, 61, dtype=torch.float32, device=self.device, requires_grad=True)

        # Test with 1-legal-action masks (e.g. forced draw)
        mask_single = torch.zeros(B, 61, dtype=torch.bool, device=self.device)
        mask_single[:, 60] = True  # only action 60 is legal

        # Entropy of a deterministic 1-action policy must be exactly 0.0
        ent = masked_entropy(logits, mask_single)
        self.assertTrue(torch.allclose(ent, torch.zeros_like(ent), atol=1e-5))
        self.assertFalse(torch.isnan(ent).any())
        self.assertFalse(torch.isinf(ent).any())

        # Reverse KL with uniform distribution over 1 action must be exactly 0.0
        u_logp = uniform_legal_log_probs(mask_single)
        kl = masked_reverse_kl(logits, u_logp, mask_single)
        self.assertTrue(torch.allclose(kl, torch.zeros_like(kl), atol=1e-5))
        self.assertFalse(torch.isnan(kl).any())

        # Test extreme logits (+50.0 and -50.0)
        extreme_logits = torch.empty(B, 61, dtype=torch.float32, device=self.device, requires_grad=True)
        extreme_logits.data.uniform_(-50.0, 50.0)

        random_mask = torch.rand(B, 61, device=self.device) > 0.5
        random_mask[:, 60] = True  # guarantee at least 1 legal action

        log_p = masked_log_softmax(extreme_logits, random_mask)
        self.assertFalse(torch.isnan(log_p).any())

        ent_extreme = masked_entropy(extreme_logits, random_mask)
        self.assertFalse(torch.isnan(ent_extreme).any())

        u_extreme = uniform_legal_log_probs(random_mask)
        kl_extreme = masked_reverse_kl(extreme_logits, u_extreme, random_mask)
        self.assertFalse(torch.isnan(kl_extreme).any())

        # Gradient flow check: backward pass must produce finite gradients with zero NaNs
        loss = kl_extreme.sum() + ent_extreme.sum()
        loss.backward()
        self.assertIsNotNone(extreme_logits.grad)
        self.assertFalse(torch.isnan(extreme_logits.grad).any())
        self.assertFalse(torch.isinf(extreme_logits.grad).any())

    def test_05_ema_update(self):
        """Verify in-place EMA updates correctly interpolate parameters."""
        online = nn.Linear(10, 10, bias=True).to(self.device)
        ema = nn.Linear(10, 10, bias=True).to(self.device)

        with torch.no_grad():
            ema.weight.fill_(0.0)
            online.weight.fill_(1.0)
            ema.bias.fill_(0.0)
            online.bias.fill_(1.0)

        update_ema(ema, online, decay=0.9)
        # Expected: 0.9 * 0.0 + 0.1 * 1.0 = 0.1
        self.assertTrue(torch.allclose(ema.weight, torch.full_like(ema.weight, 0.1)))
        self.assertTrue(torch.allclose(ema.bias, torch.full_like(ema.bias, 0.1)))

    def test_06_wdl_expected_value_and_target_mapping(self):
        """Verify WDL value mapping and integer target mapping."""
        # Clean logits for Loss, Draw, Win
        logits = torch.tensor([
            [-10.0, -10.0, 10.0],  # 100% Win -> +1.0
            [10.0, -10.0, -10.0],  # 100% Loss -> -1.0
            [-10.0, 10.0, -10.0],  # 100% Draw -> 0.0
        ], device=self.device)

        v = wdl_expected_value(logits)
        self.assertAlmostEqual(v[0].item(), 1.0, places=4)
        self.assertAlmostEqual(v[1].item(), -1.0, places=4)
        self.assertAlmostEqual(v[2].item(), 0.0, places=4)

        # Target mapping
        rewards = torch.tensor([-1.0, 0.0, 1.0], device=self.device)
        targets = reward_to_wdl_target(rewards)
        self.assertTrue(torch.equal(targets, torch.tensor([0, 1, 2], device=self.device)))


if __name__ == "__main__":
    unittest.main()
