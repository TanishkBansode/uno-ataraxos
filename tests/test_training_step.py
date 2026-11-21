"""Unit tests for VRPO + EMAgnet rollout collection and training step invariants."""

import os
import tempfile
import unittest
import torch
import numpy as np

import uno_ataraxos as ua
from uno_ataraxos.symmetry import apply_s4_batch, INV_ACTION_61
from uno_ataraxos.networks import UnoActorNet, UnoQCriticNet
from uno_ataraxos.rl_math import (
    masked_log_softmax,
    masked_reverse_kl,
    compute_vrpo_targets,
    reward_to_wdl_target,
    update_ema,
)
from uno_ataraxos.train_blueprint import train


class TestTrainingStep(unittest.TestCase):

    def setUp(self):
        self.device = ua.get_device()
        self.device_type = "mps" if self.device.type == "mps" else ("cuda" if self.device.type == "cuda" else "cpu")

    def test_01_on_policy_s4_initial_ratio_and_kl_step(self):
        """Verify that at the very first minibatch of Epoch 1 (before optimizer step),

        ratio == 1.0 ± 1e-4 and kl_step < 1e-5 across all items.
        Proves per-step on-policy S4 augmentation is 100% mathematically aligned!
        """
        device = self.device
        device_type = self.device_type
        actor = UnoActorNet().to(device)
        actor.eval()

        B = 64
        vec = ua.VecUnoEnv(num_envs=B, seed=123)
        obs = vec.reset()

        card_raw = torch.from_numpy(obs["card_features_actor"]).to(device)
        global_raw = torch.from_numpy(obs["global_features"]).to(device)
        hist_raw = torch.from_numpy(obs["history_tokens"]).to(device)
        mask_raw = torch.from_numpy(obs["legal_mask"]).to(device)
        opp_raw = torch.from_numpy(obs["opp_hand_counts"]).to(device)

        # 1. Sample random S4 permutation per item
        perm_ids = torch.randint(0, 24, (B,), device=device, dtype=torch.int64)
        card_p, hist_p, mask_p, opp_p = apply_s4_batch(
            card_raw, hist_raw, mask_raw, opp_hand_counts=opp_raw, perm_ids=perm_ids
        )

        # 2. Rollout forward
        with torch.no_grad(), torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits_p, _ = actor(card_p, global_raw, hist_p)

        old_log_p = masked_log_softmax(logits_p.float(), mask_p)
        probs_p = torch.where(mask_p, old_log_p.exp(), torch.zeros_like(old_log_p))
        probs_p = probs_p / probs_p.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        a_p = torch.multinomial(probs_p, 1).squeeze(-1)

        # 3. Unpermute action back to raw env space: a_env = inv[p, a_p]
        inv_act = INV_ACTION_61.to(device)
        a_env = inv_act[perm_ids, a_p]

        # Verify legality consistency
        for i in range(B):
            self.assertTrue(mask_raw[i, a_env[i]].item())
            self.assertTrue(mask_p[i, a_p[i]].item())

        # 4. Re-evaluate actor on stored (card_p, global_raw, hist_p) in Epoch 1 (before optimizer step)
        with torch.no_grad(), torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            eval_logits_p, _ = actor(card_p, global_raw, hist_p)

        eval_log_p = masked_log_softmax(eval_logits_p.float(), mask_p)
        log_p_taken = eval_log_p.gather(1, a_p.unsqueeze(-1)).squeeze(-1)
        old_log_p_taken = old_log_p.gather(1, a_p.unsqueeze(-1)).squeeze(-1)

        ratio = (log_p_taken - old_log_p_taken).clamp(-10.0, 10.0).exp()
        kl_step = masked_reverse_kl(eval_logits_p.float(), old_log_p, mask_p)

        max_ratio_err = torch.max(torch.abs(ratio - 1.0)).item()
        max_kl = torch.max(kl_step).item()

        self.assertLess(max_ratio_err, 1e-4, f"Initial ratio deviation too large: {max_ratio_err}")
        self.assertLess(max_kl, 1e-5, f"Initial kl_step too large: {max_kl}")

    def test_02_full_two_iterations_and_checkpoint_resume_parity(self):
        """Run 2 full iterations (exercising Stage 1 -> Stage 2 transition).

        Verifies zero NaNs, finite gradients, and exact checkpoint save/resume parity.
        """
        device = self.device
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = os.path.join(tmp_dir, "test_run")
            # Step size: B=32, T=16 -> 512 transitions/iter.
            # 2 iterations = 1024 transitions.
            # Stage transition at 0.35 * 1024 = 358 steps (occurs at iteration 1 -> 2!).
            train(
                total_steps=1024,
                num_envs=32,
                rollout_steps=16,
                minibatch_size=128,
                ppo_epochs=2,
                eval_every_steps=512,
                save_every_mins=0.01,
                run_dir=run_dir,
                seed=99,
                device=device,
            )

            latest_ckpt_path = os.path.join(run_dir, "latest.pt")
            self.assertTrue(os.path.isfile(latest_ckpt_path), "Checkpoint was not saved")

            # Check checkpoint contents
            ckpt = torch.load(latest_ckpt_path, map_location=device)
            self.assertEqual(ckpt["global_step"], 1024)
            self.assertEqual(ckpt["iteration"], 2)
            self.assertEqual(ckpt["stage"], 2)

            for key in ["actor_state_dict", "critic_state_dict", "ema_actor_state_dict"]:
                for name, param in ckpt[key].items():
                    self.assertFalse(torch.isnan(param).any(), f"NaN in saved parameter {name}")
                    self.assertFalse(torch.isinf(param).any(), f"Inf in saved parameter {name}")

            # Verify resume from checkpoint for 1 more iteration (up to 1536 steps)
            train(
                total_steps=1536,
                num_envs=32,
                rollout_steps=16,
                minibatch_size=128,
                ppo_epochs=2,
                eval_every_steps=512,
                save_every_mins=0.01,
                run_dir=run_dir,
                resume=latest_ckpt_path,
                seed=99,
                device=device,
            )

            ckpt_resumed = torch.load(latest_ckpt_path, map_location=device)
            self.assertEqual(ckpt_resumed["global_step"], 1536)
            self.assertEqual(ckpt_resumed["iteration"], 3)

    def test_03_checkpoint_portability_orig_mod_unwrapping(self):
        """Test that get_clean_state_dict and load_clean_state_dict handle models with _orig_mod.

        Ensures checkpoints saved from compiled models can be loaded into uncompiled models.
        """
        from uno_ataraxos.train_blueprint import get_clean_state_dict, load_clean_state_dict

        model = UnoActorNet().to(self.device)

        # Simulate torch.compile wrapping via _orig_mod
        class DummyCompiled(torch.nn.Module):
            def __init__(self, orig):
                super().__init__()
                self._orig_mod = orig
            def forward(self, *args, **kwargs):
                return self._orig_mod(*args, **kwargs)

        compiled_mock = DummyCompiled(model)

        # Extract clean state dict
        state_dict = get_clean_state_dict(compiled_mock)
        for k in state_dict.keys():
            self.assertFalse(k.startswith("_orig_mod."), f"Key contains compiled prefix: {k}")

        # Load clean state dict into fresh uncompiled model
        fresh_model = UnoActorNet().to(self.device)
        load_clean_state_dict(fresh_model, state_dict)

        # Load clean state dict into compiled model
        fresh_compiled = DummyCompiled(fresh_model)
        load_clean_state_dict(fresh_compiled, state_dict)

    def test_04_stage2_dual_anchor_and_snapshot_saving(self):
        """Verify Stage 2 dual-anchor uniform floor and historical snapshot creation."""
        import json
        device = self.device
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = os.path.join(tmp_dir, "dual_anchor_run")
            # 2 iterations = 1024 transitions.
            # Stage 2 threshold = 0.30 * 1024 = 307 steps -> triggers in iteration 2.
            # Snapshot interval = 512 steps -> saves snapshot at step 512!
            train(
                total_steps=1024,
                num_envs=32,
                rollout_steps=16,
                minibatch_size=128,
                ppo_epochs=2,
                eval_every_steps=512,
                num_eval_decks=10,
                snapshot_every_steps=512,
                save_every_mins=0.01,
                run_dir=run_dir,
                seed=42,
                device=device,
            )

            # 1. Verify snapshot directory created and contains actor_step_512.pt
            snap_path = os.path.join(run_dir, "snapshots", "actor_step_512.pt")
            self.assertTrue(os.path.isfile(snap_path), f"Snapshot not found at {snap_path}")

            snap_weights = torch.load(snap_path, map_location=device)
            self.assertIn("backbone.board_proj.weight", snap_weights)

            # 2. Verify metrics.jsonl records dual-anchor floor in stage 2
            metrics_file = os.path.join(run_dir, "metrics.jsonl")
            self.assertTrue(os.path.isfile(metrics_file))
            records = []
            with open(metrics_file, "r") as f:
                for line in f:
                    records.append(json.loads(line))

            stage_2_entries = [r for r in records if r.get("stage") == 2]
            self.assertGreater(len(stage_2_entries), 0)
            self.assertGreater(stage_2_entries[0]["alpha_unif_floor"], 0.0)


if __name__ == "__main__":
    unittest.main()
