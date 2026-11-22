"""VRPO + Dynamically Damped EMAgnet Self-Play Trainer for 1v1 UNO.

Cross-Platform Architecture:
- Apple Silicon (macOS ARM64 MPS): bfloat16 autocast (no GradScaler, no torch.compile).
- NVIDIA CUDA (Compute >= 8.0): bfloat16 autocast + optional torch.compile.
- NVIDIA CUDA (Compute < 8.0): float16 autocast + torch.amp.GradScaler("cuda").
- CPU: float32 fallback.

RL Features:
- 100% On-Policy Per-Step S4 Color Permutation during rollout collection.
- VRPO Q-boosting Expected SARSA(λ) with exact extra-turn signs.
- 3-Class Win/Draw/Loss auxiliary value head targets.
- Two-Stage Regularization:
  * Stage 1 (< 30% total steps): Uniform Legal Policy Magnet (alpha_mag: 0.10 -> 0.015).
  * Stage 2 (>= 30% total steps): Dual-Anchor Regularization:
    - EMAgnet reference (alpha_mag: 0.04 -> 0.01) with shock-free transition.
    - Uniform magnet floor (alpha_unif_floor: 0.006 -> 0.002) preventing entropy collapse.
- Historical snapshot tracking & evaluation against past model generations.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

import uno_ataraxos as ua
from uno_ataraxos.eval_suite import (
    ExpertHeuristicBot,
    GreedyBot,
    RandomBot,
    evaluate_duplicate,
)
from uno_ataraxos.networks import UnoActorNet, UnoQCriticNet
from uno_ataraxos.rl_math import (
    compute_vrpo_targets,
    get_device,
    masked_entropy,
    masked_log_softmax,
    masked_reverse_kl,
    reward_to_wdl_target,
    uniform_legal_log_probs,
    update_ema,
)
from uno_ataraxos.symmetry import INV_ACTION_61, apply_s4_batch


def get_clean_state_dict(model: torch.nn.Module) -> dict:
    """Return state_dict unwrapped from torch.compile (_orig_mod) if present.

    Ensures 100% checkpoint portability between compiled CUDA and uncompiled devices.
    """
    raw_model = getattr(model, "_orig_mod", model)
    return raw_model.state_dict()


def load_clean_state_dict(model: torch.nn.Module, state_dict: dict) -> None:
    """Load state_dict cleanly into model, unwrapping _orig_mod if compiled."""
    raw_model = getattr(model, "_orig_mod", model)
    raw_model.load_state_dict(state_dict)


def train(
    total_steps: int = 100_000_000,
    num_envs: int = 512,
    rollout_steps: int = 64,
    minibatch_size: int = 1024,
    ppo_epochs: int = 2,
    eval_every_steps: int = 2_500_000,
    num_eval_decks: int = 250,
    snapshot_every_steps: int = 10_000_000,
    save_every_mins: float = 15.0,
    run_dir: str = "runs/blueprint_v1",
    resume: Optional[str] = None,
    compile_model: bool = False,
    cooldown_sec: float = 0.0,
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> None:
    """Main training loop for VRPO + EMAgnet self-play."""
    if device is None:
        device = get_device()

    # 1. Smart Mixed-Precision & Scaler Auto-Detection
    if device.type == "cuda":
        device_type = "cuda"
        capability = torch.cuda.get_device_capability()
        if capability[0] >= 8:
            # Modern CUDA (Compute >= 8.0)
            autocast_dtype = torch.bfloat16
            use_scaler = False
            scaler = None
            amp_desc = f"bfloat16 (compute capability {capability[0]}.{capability[1]})"
        else:
            # Legacy CUDA (Compute < 8.0) -> float16 + GradScaler
            autocast_dtype = torch.float16
            use_scaler = True
            scaler = torch.amp.GradScaler("cuda")
            amp_desc = f"float16 + GradScaler (compute capability {capability[0]}.{capability[1]})"
    elif device.type == "mps":
        device_type = "mps"
        autocast_dtype = torch.bfloat16
        use_scaler = False
        scaler = None
        amp_desc = "bfloat16 (Apple Silicon MPS)"
    else:
        device_type = "cpu"
        autocast_dtype = torch.float32
        use_scaler = False
        scaler = None
        amp_desc = "float32 (CPU)"

    os.makedirs(run_dir, exist_ok=True)
    snapshot_dir = os.path.join(run_dir, "snapshots")
    os.makedirs(snapshot_dir, exist_ok=True)
    metrics_path = os.path.join(run_dir, "metrics.jsonl")

    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 2. Instantiate Models
    actor = UnoActorNet().to(device)
    critic = UnoQCriticNet().to(device)
    ema_actor = UnoActorNet().to(device)
    ema_actor.load_state_dict(actor.state_dict())
    ema_actor.eval()

    # Optional torch.compile (auto/opt-in on CUDA)
    if compile_model and device.type == "cuda":
        print("[COMPILE] Compiling actor, critic, and ema_actor with torch.compile(mode='default')...")
        try:
            actor = torch.compile(actor, mode="default")
            critic = torch.compile(critic, mode="default")
            ema_actor = torch.compile(ema_actor, mode="default")
            print("[COMPILE] Compilation enabled successfully!")
        except Exception as e:
            print(f"[COMPILE WARNING] torch.compile failed ({e}), continuing uncompiled.")

    # Optimizers
    actor_opt = torch.optim.AdamW(actor.parameters(), lr=3e-4, weight_decay=1e-4)
    critic_opt = torch.optim.AdamW(critic.parameters(), lr=5e-4, weight_decay=1e-4)

    global_step = 0
    iteration = 0
    stage = 1
    best_win_rate = -1.0
    last_save_time = time.time()
    last_eval_step = -eval_every_steps
    last_snapshot_step = 0

    # 3. Handle Resume Checkpoint
    if resume and os.path.isfile(resume):
        print(f"[RESUME] Loading checkpoint from {resume}...")
        ckpt = torch.load(resume, map_location=device)
        load_clean_state_dict(actor, ckpt["actor_state_dict"])
        load_clean_state_dict(critic, ckpt["critic_state_dict"])
        load_clean_state_dict(ema_actor, ckpt["ema_actor_state_dict"])
        actor_opt.load_state_dict(ckpt["actor_opt_state_dict"])
        critic_opt.load_state_dict(ckpt["critic_opt_state_dict"])
        global_step = ckpt.get("global_step", 0)
        iteration = ckpt.get("iteration", 0)
        stage = ckpt.get("stage", 1)
        best_win_rate = ckpt.get("best_win_rate", -1.0)
        last_eval_step = global_step
        last_snapshot_step = (global_step // snapshot_every_steps) * snapshot_every_steps
        print(f"[RESUME] Resumed at global_step {global_step:,}, iteration {iteration}, stage {stage}")

    # 4. Instantiate Vectorized Environment
    vec = ua.VecUnoEnv(num_envs=num_envs, seed=seed, auto_reset=True)
    obs = vec.reset()

    # Preallocate Rollout Storage on device
    B = num_envs
    T = rollout_steps
    N = T * B

    buf_card = torch.empty((T, B, 54, 7), dtype=torch.float32, device=device)
    buf_global = torch.empty((T, B, 10), dtype=torch.float32, device=device)
    buf_hist = torch.empty((T, B, 48, 5), dtype=torch.int32, device=device)
    buf_mask = torch.empty((T, B, 61), dtype=torch.bool, device=device)
    buf_opp = torch.empty((T, B, 54), dtype=torch.float32, device=device)
    buf_action = torch.empty((T, B), dtype=torch.int64, device=device)
    buf_old_log_p = torch.empty((T, B, 61), dtype=torch.float32, device=device)
    buf_q_taken = torch.empty((T, B), dtype=torch.float32, device=device)
    buf_v_expected = torch.empty((T + 1, B), dtype=torch.float32, device=device)
    buf_rewards = torch.empty((T, B), dtype=torch.float32, device=device)
    buf_dones = torch.empty((T, B), dtype=torch.float32, device=device)
    buf_same_player = torch.empty((T, B), dtype=torch.float32, device=device)

    # Evaluation bots
    random_bot = RandomBot(seed=1000)
    greedy_bot = GreedyBot(seed=1001)
    expert_bot = ExpertHeuristicBot(seed=1002)

    inv_act_table = INV_ACTION_61.to(device)

    print("=" * 72)
    print("  UNO-ATARAXOS VRPO + DYNAMICALLY DAMPED EMAGNET TRAINER")
    print("=" * 72)
    print(f"Device:            {device} ({amp_desc})")
    print(f"Compiled Model:    {compile_model and device.type == 'cuda'}")
    print(f"Total Steps:       {total_steps:,}")
    print(f"Num Envs / Steps:  B = {num_envs}, T = {rollout_steps} ({N:,} transitions/iter)")
    print(f"PPO Minibatch:     {minibatch_size} ({N // minibatch_size} batches × {ppo_epochs} epochs)")
    print(f"Eval Interval:     Every {eval_every_steps:,} steps ({num_eval_decks * 2} duplicate games)")
    print(f"Snapshot Interval: Every {snapshot_every_steps:,} steps")
    print(f"Run Directory:     {run_dir}")
    print("-" * 72)

    # Initial Step 0 Evaluation
    if global_step == 0:
        print("[EVAL] Running initial baseline evaluations at step 0...")
        actor.eval()
        res_rand = evaluate_duplicate(actor, random_bot, num_decks=num_eval_decks, device=device)
        res_greedy = evaluate_duplicate(actor, greedy_bot, num_decks=num_eval_decks, device=device)
        res_expert = evaluate_duplicate(actor, expert_bot, num_decks=num_eval_decks, device=device)
        actor.train()

        print(f"  Step 0 | vs Random: {res_rand['win_rate_a']*100:.1f}% | vs Greedy: {res_greedy['win_rate_a']*100:.1f}% | vs Expert: {res_expert['win_rate_a']*100:.1f}%")
        metrics_entry = {
            "step": 0,
            "iter": 0,
            "eval/vs_random": res_rand["win_rate_a"],
            "eval/vs_greedy": res_greedy["win_rate_a"],
            "eval/vs_expert": res_expert["win_rate_a"],
        }
        with open(metrics_path, "a") as f:
            f.write(json.dumps(metrics_entry) + "\n")

    ep_lengths = []
    current_ep_lens = np.zeros(num_envs, dtype=np.int32)
    t_start = time.perf_counter()

    while global_step < total_steps:
        iter_t0 = time.perf_counter()
        iteration += 1

        # ---------------------------------------------------------------------
        # 5. Rollout Collection: 100% On-Policy Per-Step S4 Permutation
        # ---------------------------------------------------------------------
        actor.eval()
        critic.eval()

        for t in range(T):
            card_raw = torch.from_numpy(obs["card_features_actor"]).to(device)
            global_raw = torch.from_numpy(obs["global_features"]).to(device)
            hist_raw = torch.from_numpy(obs["history_tokens"]).to(device)
            mask_raw = torch.from_numpy(obs["legal_mask"]).to(device)
            opp_raw = torch.from_numpy(obs["opp_hand_counts"]).to(device)
            cur_player = torch.from_numpy(obs["current_player"]).to(device)

            perm_ids = torch.randint(0, 24, (B,), device=device, dtype=torch.int64)
            card_p, hist_p, mask_p, opp_p = apply_s4_batch(
                card_raw, hist_raw, mask_raw, opp_hand_counts=opp_raw, perm_ids=perm_ids
            )

            with torch.no_grad(), torch.autocast(device_type=device_type, dtype=autocast_dtype):
                logits_p, _ = actor(card_p, global_raw, hist_p)
                q_all_p = critic(card_p, opp_p, global_raw, hist_p)

            logits_f = logits_p.float()
            q_all_f = q_all_p.float()
            old_log_p = masked_log_softmax(logits_f, mask_p)

            probs_p = torch.where(mask_p, old_log_p.exp(), torch.zeros_like(old_log_p))
            probs_p = probs_p / probs_p.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            a_p = torch.multinomial(probs_p, 1).squeeze(-1)

            q_taken_t = q_all_f.gather(1, a_p.unsqueeze(-1)).squeeze(-1)
            v_expected_t = (probs_p * q_all_f).sum(dim=-1)

            buf_card[t] = card_p
            buf_global[t] = global_raw
            buf_hist[t] = hist_p
            buf_mask[t] = mask_p
            buf_opp[t] = opp_p
            buf_action[t] = a_p
            buf_old_log_p[t] = old_log_p
            buf_q_taken[t] = q_taken_t
            buf_v_expected[t] = v_expected_t

            a_env = inv_act_table[perm_ids, a_p]
            next_obs, rewards_np, dones_np, _ = vec.step(a_env.cpu().numpy().astype(np.int32))

            rewards_t = torch.from_numpy(rewards_np).to(device)
            dones_t = torch.from_numpy(dones_np.astype(np.float32)).to(device)
            next_player = torch.from_numpy(next_obs["current_player"]).to(device)
            same_player_next_t = (next_player == cur_player).float()

            buf_rewards[t] = rewards_t
            buf_dones[t] = dones_t
            buf_same_player[t] = same_player_next_t

            current_ep_lens += 1
            for env_i in np.where(dones_np)[0]:
                ep_lengths.append(int(current_ep_lens[env_i]))
                current_ep_lens[env_i] = 0

            obs = next_obs

        # Bootstrap at step T
        card_raw_T = torch.from_numpy(obs["card_features_actor"]).to(device)
        global_raw_T = torch.from_numpy(obs["global_features"]).to(device)
        hist_raw_T = torch.from_numpy(obs["history_tokens"]).to(device)
        mask_raw_T = torch.from_numpy(obs["legal_mask"]).to(device)
        opp_raw_T = torch.from_numpy(obs["opp_hand_counts"]).to(device)

        perm_ids_T = torch.randint(0, 24, (B,), device=device, dtype=torch.int64)
        card_p_T, hist_p_T, mask_p_T, opp_p_T = apply_s4_batch(
            card_raw_T, hist_raw_T, mask_raw_T, opp_hand_counts=opp_raw_T, perm_ids=perm_ids_T
        )

        with torch.no_grad(), torch.autocast(device_type=device_type, dtype=autocast_dtype):
            logits_T, wdl_logits_T = actor(card_p_T, global_raw_T, hist_p_T)
            q_all_T = critic(card_p_T, opp_p_T, global_raw_T, hist_p_T)

        log_p_T = masked_log_softmax(logits_T.float(), mask_p_T)
        probs_T = torch.where(mask_p_T, log_p_T.exp(), torch.zeros_like(log_p_T))
        probs_T = probs_T / probs_T.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        buf_v_expected[T] = (probs_T * q_all_T.float()).sum(dim=-1)
        wdl_bootstrap = wdl_logits_T.float()

        # ---------------------------------------------------------------------
        # 6. Exact Target Computations (float32)
        # ---------------------------------------------------------------------
        q_targets, advantages = compute_vrpo_targets(
            buf_rewards, buf_dones, buf_same_player, buf_q_taken, buf_v_expected, lam=0.95
        )

        wdl_targets = torch.empty((T, B, 3), dtype=torch.float32, device=device)
        wdl_next = F.softmax(wdl_bootstrap, dim=-1)
        for t in reversed(range(T)):
            is_done = (buf_dones[t] > 0.5)
            r_class = reward_to_wdl_target(buf_rewards[t])
            one_hot_r = F.one_hot(r_class, num_classes=3).float()
            not_done_wdl = torch.where(
                buf_same_player[t].unsqueeze(-1) > 0.5,
                wdl_next,
                wdl_next[:, [2, 1, 0]],
            )
            wdl_t = torch.where(is_done.unsqueeze(-1), one_hot_r, not_done_wdl)
            wdl_targets[t] = wdl_t
            wdl_next = wdl_t

        choice_mask = (buf_mask.sum(dim=-1) > 1)
        if choice_mask.any():
            adv_choice = advantages[choice_mask]
            adv_mean = adv_choice.mean()
            adv_std = adv_choice.std()
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)

        # ---------------------------------------------------------------------
        # 7. Magnet Reference & Stage 2 Dual-Anchor Regularization
        # ---------------------------------------------------------------------
        stage_2_start = 0.30 * total_steps
        is_stage_2 = (global_step >= stage_2_start)
        if is_stage_2 and stage == 1:
            print(f"\n[STAGE TRANSITION] Stage 1 (Uniform Magnet) -> Stage 2 (EMAgnet + Dual-Anchor) at step {global_step:,}!")
            load_clean_state_dict(ema_actor, get_clean_state_dict(actor))
            stage = 2

        flat_cards = buf_card.reshape(N, 54, 7)
        flat_globals = buf_global.reshape(N, 10)
        flat_hists = buf_hist.reshape(N, 48, 5)
        flat_masks = buf_mask.reshape(N, 61)

        if stage == 1:
            # Stage 1: Uniform Legal Policy Magnet (alpha_mag: 0.10 -> 0.015)
            mag_log_p = uniform_legal_log_probs(buf_mask)
            prog = min(1.0, max(0.0, global_step / max(1, stage_2_start)))
            alpha_mag = 0.10 - prog * (0.10 - 0.015)
            alpha_unif_floor = 0.0
        else:
            # Stage 2: EMAgnet Reference (alpha_mag: 0.04 -> 0.01)
            mag_log_p_flat = torch.empty((N, 61), dtype=torch.float32, device=device)
            chunk_size = 1024
            for c_start in range(0, N, chunk_size):
                c_end = min(c_start + chunk_size, N)
                with torch.no_grad(), torch.autocast(device_type=device_type, dtype=autocast_dtype):
                    ema_logits, _ = ema_actor(
                        flat_cards[c_start:c_end],
                        flat_globals[c_start:c_end],
                        flat_hists[c_start:c_end],
                    )
                mag_log_p_flat[c_start:c_end] = masked_log_softmax(
                    ema_logits.float(), flat_masks[c_start:c_end]
                )
            mag_log_p = mag_log_p_flat.reshape(T, B, 61)

            prog = min(1.0, max(0.0, (global_step - stage_2_start) / max(1, total_steps - stage_2_start)))
            alpha_mag = 0.04 - prog * (0.04 - 0.01)
            # Dual-Anchor Floor: small uniform anchor preventing entropy collapse over 100M+ steps
            alpha_unif_floor = 0.006 - prog * (0.006 - 0.002)

        # ---------------------------------------------------------------------
        # 8. PPO + VRPO + EMAgnet Minibatch Updates
        # ---------------------------------------------------------------------
        actor.train()
        critic.train()

        flat_opps = buf_opp.reshape(N, 54)
        flat_actions = buf_action.reshape(N)
        flat_old_log_p = buf_old_log_p.reshape(N, 61)
        flat_mag_log_p = mag_log_p.reshape(N, 61)
        flat_q_targets = q_targets.reshape(N)
        flat_adv = advantages.reshape(N)
        flat_wdl_targets = wdl_targets.reshape(N, 3)
        flat_choice_mask = choice_mask.reshape(N)

        lr_alpha = 0.5 * (1.0 + math.cos(math.pi * global_step / total_steps))
        cur_actor_lr = 5e-5 + (3e-4 - 5e-5) * lr_alpha
        cur_critic_lr = 1e-4 + (5e-4 - 1e-4) * lr_alpha
        for pg in actor_opt.param_groups:
            pg["lr"] = cur_actor_lr
        for pg in critic_opt.param_groups:
            pg["lr"] = cur_critic_lr

        loss_actor_accum = 0.0
        loss_critic_accum = 0.0
        loss_wdl_accum = 0.0
        loss_kl_mag_accum = 0.0
        loss_kl_step_accum = 0.0
        entropy_accum = 0.0
        clip_frac_accum = 0.0
        n_updates = 0

        for epoch in range(ppo_epochs):
            perm_indices = torch.randperm(N, device=device)
            for start_idx in range(0, N, minibatch_size):
                b_idx = perm_indices[start_idx : start_idx + minibatch_size]
                n_updates += 1

                # 8a. Centralized Q-Critic Update
                critic_opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device_type, dtype=autocast_dtype):
                    q_pred_all = critic(
                        flat_cards[b_idx],
                        flat_opps[b_idx],
                        flat_globals[b_idx],
                        flat_hists[b_idx],
                    )
                q_pred_taken = q_pred_all.float().gather(1, flat_actions[b_idx].unsqueeze(-1)).squeeze(-1)
                critic_loss = F.smooth_l1_loss(q_pred_taken, flat_q_targets[b_idx])

                if use_scaler and scaler is not None:
                    scaler.scale(critic_loss).backward()
                    scaler.unscale_(critic_opt)
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                    scaler.step(critic_opt)
                else:
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                    critic_opt.step()

                # 8b. Actor + WDL Auxiliary Update
                actor_opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device_type, dtype=autocast_dtype):
                    p_logits, wdl_logits = actor(
                        flat_cards[b_idx],
                        flat_globals[b_idx],
                        flat_hists[b_idx],
                    )
                p_logits = p_logits.float()
                wdl_logits = wdl_logits.float()

                mb_mask = flat_masks[b_idx]
                mb_actions = flat_actions[b_idx]
                mb_choice = flat_choice_mask[b_idx]
                mb_adv = flat_adv[b_idx]

                log_p = masked_log_softmax(p_logits, mb_mask)
                log_p_taken = log_p.gather(1, mb_actions.unsqueeze(-1)).squeeze(-1)
                old_log_p_taken = flat_old_log_p[b_idx].gather(1, mb_actions.unsqueeze(-1)).squeeze(-1)

                ratio = (log_p_taken - old_log_p_taken).clamp(-10.0, 10.0).exp()
                surr1 = ratio * mb_adv
                surr2 = ratio.clamp(0.8, 1.2) * mb_adv

                if mb_choice.any():
                    n_ch = mb_choice.float().sum()
                    ppo_loss = (-torch.min(surr1, surr2) * mb_choice.float()).sum() / n_ch
                    kl_mag = (masked_reverse_kl(p_logits, flat_mag_log_p[b_idx], mb_mask) * mb_choice.float()).sum() / n_ch
                    kl_step = (masked_reverse_kl(p_logits, flat_old_log_p[b_idx], mb_mask) * mb_choice.float()).sum() / n_ch

                    # Dual-Anchor Stage 2 uniform floor computation
                    if stage == 2:
                        flat_unif_log_p = uniform_legal_log_probs(mb_mask)
                        kl_unif = (masked_reverse_kl(p_logits, flat_unif_log_p, mb_mask) * mb_choice.float()).sum() / n_ch
                    else:
                        kl_unif = torch.tensor(0.0, device=device)
                else:
                    ppo_loss = torch.tensor(0.0, device=device)
                    kl_mag = torch.tensor(0.0, device=device)
                    kl_step = torch.tensor(0.0, device=device)
                    kl_unif = torch.tensor(0.0, device=device)

                wdl_loss = -(flat_wdl_targets[b_idx] * F.log_softmax(wdl_logits, dim=-1)).sum(dim=-1).mean()
                actor_loss = (
                    ppo_loss
                    + alpha_mag * kl_mag
                    + alpha_unif_floor * kl_unif
                    + 0.02 * kl_step
                    + 0.25 * wdl_loss
                )

                if use_scaler and scaler is not None:
                    scaler.scale(actor_loss).backward()
                    scaler.unscale_(actor_opt)
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                    scaler.step(actor_opt)
                    scaler.update()
                else:
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                    actor_opt.step()

                loss_actor_accum += actor_loss.item()
                loss_critic_accum += critic_loss.item()
                loss_wdl_accum += wdl_loss.item()
                loss_kl_mag_accum += kl_mag.item()
                loss_kl_step_accum += kl_step.item()
                entropy_accum += masked_entropy(p_logits, mb_mask).mean().item()
                clip_frac_accum += ((ratio < 0.8) | (ratio > 1.2)).float().mean().item()

        if stage == 2:
            update_ema(ema_actor, actor, decay=0.985)

        global_step += N
        iter_elapsed = time.perf_counter() - iter_t0
        total_elapsed = time.perf_counter() - t_start
        sps = N / max(1e-6, iter_elapsed)
        overall_sps = global_step / max(1e-6, total_elapsed)

        mean_actor_loss = loss_actor_accum / n_updates
        mean_critic_loss = loss_critic_accum / n_updates
        mean_wdl_loss = loss_wdl_accum / n_updates
        mean_kl_mag = loss_kl_mag_accum / n_updates
        mean_kl_step = loss_kl_step_accum / n_updates
        mean_entropy = entropy_accum / n_updates
        mean_clip_frac = clip_frac_accum / n_updates
        avg_ep_len = float(np.mean(ep_lengths[-500:])) if ep_lengths else 0.0

        metrics_dict: Dict[str, Any] = {
            "step": global_step,
            "iter": iteration,
            "sps": float(sps),
            "overall_sps": float(overall_sps),
            "actor_loss": float(mean_actor_loss),
            "critic_loss": float(mean_critic_loss),
            "wdl_loss": float(mean_wdl_loss),
            "kl_mag": float(mean_kl_mag),
            "kl_step": float(mean_kl_step),
            "entropy": float(mean_entropy),
            "clip_frac": float(mean_clip_frac),
            "avg_ep_len": float(avg_ep_len),
            "alpha_mag": float(alpha_mag),
            "alpha_unif_floor": float(alpha_unif_floor),
            "stage": int(stage),
            "lr_actor": float(cur_actor_lr),
            "lr_critic": float(cur_critic_lr),
        }

        # ---------------------------------------------------------------------
        # 9. Historical Snapshot Saving (Weights-Only)
        # ---------------------------------------------------------------------
        if (global_step - last_snapshot_step) >= snapshot_every_steps and global_step > 0:
            last_snapshot_step = global_step
            snap_path = os.path.join(snapshot_dir, f"actor_step_{global_step}.pt")
            torch.save(get_clean_state_dict(actor), snap_path)
            print(f"[SNAPSHOT] Saved weights snapshot to {snap_path}")

        # ---------------------------------------------------------------------
        # 10. Evaluation (Baselines + Earliest Historical Snapshot)
        # ---------------------------------------------------------------------
        if global_step - last_eval_step >= eval_every_steps or global_step >= total_steps:
            last_eval_step = global_step
            print(f"\n[EVAL] Running duplicate evaluation at step {global_step:,}...")
            actor.eval()
            res_rand = evaluate_duplicate(actor, random_bot, num_decks=num_eval_decks, device=device)
            res_greedy = evaluate_duplicate(actor, greedy_bot, num_decks=num_eval_decks, device=device)
            res_expert = evaluate_duplicate(actor, expert_bot, num_decks=num_eval_decks, device=device)

            metrics_dict["eval/vs_random"] = res_rand["win_rate_a"]
            metrics_dict["eval/vs_greedy"] = res_greedy["win_rate_a"]
            metrics_dict["eval/vs_expert"] = res_expert["win_rate_a"]
            metrics_dict["eval/v_draw_rate"] = res_expert["voluntary_draw_rate_a"]

            eval_str = (
                f"  >>> Step {global_step:,} EVAL | vs Random: {res_rand['win_rate_a']*100:.1f}% | "
                f"vs Greedy: {res_greedy['win_rate_a']*100:.1f}% | vs Expert: {res_expert['win_rate_a']*100:.1f}% | "
                f"V-Draw Rate: {res_expert['voluntary_draw_rate_a']*100:.1f}%"
            )

            # Check for earliest historical snapshot to benchmark against self-play lineage
            snapshot_files = sorted(glob.glob(os.path.join(snapshot_dir, "actor_step_*.pt")))
            if snapshot_files:
                earliest_snap_path = snapshot_files[0]
                snap_name = os.path.basename(earliest_snap_path).replace(".pt", "")
                try:
                    snap_bot = UnoActorNet().to(device)
                    snap_bot.load_state_dict(torch.load(earliest_snap_path, map_location=device))
                    snap_bot.eval()
                    res_snap = evaluate_duplicate(actor, snap_bot, num_decks=num_eval_decks, device=device)
                    metrics_dict[f"eval/vs_{snap_name}"] = res_snap["win_rate_a"]
                    eval_str += f" | vs {snap_name}: {res_snap['win_rate_a']*100:.1f}%"
                except Exception as e:
                    print(f"  [SNAPSHOT EVAL WARNING] Failed to evaluate against {earliest_snap_path}: {e}")

            print(eval_str)
            actor.train()

            # Save best checkpoint
            if res_greedy["win_rate_a"] > best_win_rate:
                best_win_rate = res_greedy["win_rate_a"]
                best_ckpt_path = os.path.join(run_dir, "best.pt")
                torch.save(
                    {
                        "global_step": global_step,
                        "iteration": iteration,
                        "actor_state_dict": get_clean_state_dict(actor),
                        "critic_state_dict": get_clean_state_dict(critic),
                        "ema_actor_state_dict": get_clean_state_dict(ema_actor),
                        "actor_opt_state_dict": actor_opt.state_dict(),
                        "critic_opt_state_dict": critic_opt.state_dict(),
                        "stage": stage,
                        "best_win_rate": best_win_rate,
                    },
                    best_ckpt_path,
                )

        # Console Progress Line
        floor_str = f" f={alpha_unif_floor:.3f}" if stage == 2 else ""
        print(
            f"Iter {iteration:4d} | Step {global_step:8,d}/{total_steps:,} ({global_step/total_steps*100:4.1f}%) | "
            f"SPS: {sps:6,.0f} | CriticL: {mean_critic_loss:6.4f} | ActorL: {mean_actor_loss:6.4f} | "
            f"Ent: {mean_entropy:5.2f} | KL_step: {mean_kl_step:6.4f} | "
            f"EpLen: {avg_ep_len:5.1f} | S{stage} α={alpha_mag:.3f}{floor_str}"
        )

        with open(metrics_path, "a") as f:
            f.write(json.dumps(metrics_dict) + "\n")

        # Periodic Checkpoint Saving
        if (time.time() - last_save_time) >= (save_every_mins * 60.0) or global_step >= total_steps:
            last_save_time = time.time()
            latest_ckpt_path = os.path.join(run_dir, "latest.pt")
            torch.save(
                {
                    "global_step": global_step,
                    "iteration": iteration,
                    "actor_state_dict": get_clean_state_dict(actor),
                    "critic_state_dict": get_clean_state_dict(critic),
                    "ema_actor_state_dict": get_clean_state_dict(ema_actor),
                    "actor_opt_state_dict": actor_opt.state_dict(),
                    "critic_opt_state_dict": critic_opt.state_dict(),
                    "stage": stage,
                    "best_win_rate": best_win_rate,
                },
                latest_ckpt_path,
            )

        if cooldown_sec > 0.0:
            time.sleep(cooldown_sec)

    print("\n" + "=" * 72)
    print(f"TRAINING COMPLETED: {global_step:,} steps in {total_elapsed/60.0:.2f} mins ({global_step/total_elapsed:,.0f} SPS)")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(description="uno-ataraxos VRPO + EMAgnet Self-Play Trainer")
    parser.add_argument("--total-steps", type=int, default=100_000_000, help="Total environment steps")
    parser.add_argument("--num-envs", type=int, default=512, help="Number of parallel environments (B)")
    parser.add_argument("--rollout-steps", type=int, default=64, help="Rollout steps per env (T)")
    parser.add_argument("--minibatch-size", type=int, default=1024, help="PPO minibatch size")
    parser.add_argument("--ppo-epochs", type=int, default=2, help="Number of PPO epochs per iteration")
    parser.add_argument("--eval-every-steps", type=int, default=2_500_000, help="Evaluation interval in steps")
    parser.add_argument("--num-eval-decks", type=int, default=250, help="Number of duplicate decks for eval")
    parser.add_argument("--snapshot-every-steps", type=int, default=10_000_000, help="Historical snapshot save interval")
    parser.add_argument("--save-every-mins", type=float, default=15.0, help="Checkpoint save interval in minutes")
    parser.add_argument("--run-dir", type=str, default="runs/blueprint_v1", help="Directory for logs & checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile on CUDA")
    parser.add_argument("--cooldown-sec", type=float, default=0.0, help="Optional cooldown delay between iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    train(
        total_steps=args.total_steps,
        num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,
        minibatch_size=args.minibatch_size,
        ppo_epochs=args.ppo_epochs,
        eval_every_steps=args.eval_every_steps,
        num_eval_decks=args.num_eval_decks,
        snapshot_every_steps=args.snapshot_every_steps,
        save_every_mins=args.save_every_mins,
        run_dir=args.run_dir,
        resume=args.resume,
        compile_model=args.compile,
        cooldown_sec=args.cooldown_sec,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
