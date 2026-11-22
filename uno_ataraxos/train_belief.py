"""Diverse-Mixture Streaming Belief Network Trainer for 1v1 UNO (Milestone 4).

Generates diverse online multi-policy trajectories across 256 parallel environments:
- 50% (128 envs): Blueprint vs Blueprint (loads --blueprint-ckpt)
- 20% (52 envs): Blueprint vs Snapshot / High-Temp Blueprint (temp = 1.5)
- 20% (52 envs): Blueprint vs ExpertHeuristicBot (OOD resilience)
- 10% (24 envs): Blueprint vs GreedyBot / RandomBot

Hardware & Precision:
- Apple Silicon (MPS): bfloat16 autocast (uncompiled)
- NVIDIA CUDA (Compute >= 8.0): bfloat16 autocast (optional torch.compile)
- NVIDIA CUDA (Compute < 8.0): float16 autocast + GradScaler
- CPU: float32
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import uno_ataraxos as ua
from uno_ataraxos import VecUnoEnv
from uno_ataraxos.belief import (
    UnoBeliefNet,
    compute_eligible,
    hypergeometric_hand_nll,
)
from uno_ataraxos.eval_suite import ExpertHeuristicBot, GreedyBot, RandomBot
from uno_ataraxos.networks import UnoActorNet
from uno_ataraxos.rl_math import get_device, masked_log_softmax


# Card IDs for evaluation
DRAW_2_CARD_IDS = [42, 45, 48, 51]
WILD_DRAW_4_CARD_ID = 53


def get_clean_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Extract clean state dict without '_orig_mod.' prefixes from torch.compile."""
    raw_model = getattr(model, "_orig_mod", model)
    return raw_model.state_dict()


def load_clean_state_dict(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    """Load state dict cleanly regardless of torch.compile wrapper."""
    raw_model = getattr(model, "_orig_mod", model)
    clean_sd = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            clean_sd[k[len("_orig_mod."):]] = v
        else:
            clean_sd[k] = v
    raw_model.load_state_dict(clean_sd)


class StreamingReplayBuffer:
    """Compact circular replay buffer for imperfect observation states & opp hand counts."""

    def __init__(self, capacity: int = 32_768):
        self.capacity = capacity
        self.card_feats = np.empty((capacity, 54, 7), dtype=np.float32)
        self.glob_feats = np.empty((capacity, 10), dtype=np.float32)
        self.hist_tokens = np.empty((capacity, 48, 5), dtype=np.int32)
        self.opp_hands = np.empty((capacity, 54), dtype=np.float32)
        self.size = 0
        self.ptr = 0

    def add_batch(
        self,
        card_feats: np.ndarray,
        glob_feats: np.ndarray,
        hist_tokens: np.ndarray,
        opp_hands: np.ndarray,
    ) -> None:
        """Add batch of transitions to circular buffer."""
        B = card_feats.shape[0]
        for i in range(B):
            self.card_feats[self.ptr] = card_feats[i]
            self.glob_feats[self.ptr] = glob_feats[i]
            self.hist_tokens[self.ptr] = hist_tokens[i]
            self.opp_hands[self.ptr] = opp_hands[i]
            self.ptr = (self.ptr + 1) % self.capacity
            if self.size < self.capacity:
                self.size += 1

    def sample(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample random minibatch of transitions onto device."""
        indices = np.random.randint(0, self.size, size=batch_size)
        cf = torch.from_numpy(self.card_feats[indices]).to(device)
        gf = torch.from_numpy(self.glob_feats[indices]).to(device)
        ht = torch.from_numpy(self.hist_tokens[indices]).to(device)
        oh = torch.from_numpy(self.opp_hands[indices]).to(device)
        return cf, gf, ht, oh


@torch.no_grad()
def evaluate_belief(
    belief_net: UnoBeliefNet,
    eval_cf: torch.Tensor,
    eval_gf: torch.Tensor,
    eval_ht: torch.Tensor,
    eval_oh: torch.Tensor,
    K_eval: int = 64,
) -> Dict[str, float]:
    """Evaluate belief network on validation batch: NLLs, gain, Brier score, and accuracy."""
    belief_net.eval()
    B = eval_cf.shape[0]

    # 1. Exact teacher-forced hand NLL without S4 permutation
    out = belief_net.compute_nll_loss(eval_cf, eval_gf, eval_ht, eval_oh, apply_s4=False)
    hand_nll = float(out["hand_nll"])
    hyper_nll = float(out["hypergeom_nll"])
    nll_gain = float(out["nll_gain"])

    # 2. Brier score & Top accuracy on subset of positions
    eval_subset = min(B, 32)
    brier_p2, acc_p2 = [], []
    brier_p4, acc_p4 = [], []
    brier_col, acc_col = [], []

    for i in range(eval_subset):
        k_i = int(eval_oh[i].sum().item())
        if k_i == 0:
            continue

        sampled = belief_net.sample_opp_hands(
            eval_cf[i:i+1], eval_gf[i:i+1], eval_ht[i:i+1], K=K_eval, temp=1.0, uniform_mix=0.03
        )

        # A) Draw 2
        p_p2 = (sampled[:, DRAW_2_CARD_IDS].sum(dim=-1) > 0).float().mean().item()
        y_p2 = 1.0 if eval_oh[i, DRAW_2_CARD_IDS].sum().item() > 0 else 0.0
        brier_p2.append((p_p2 - y_p2) ** 2)
        acc_p2.append(float((p_p2 >= 0.5) == (y_p2 > 0.5)))

        # B) Wild Draw 4
        p_p4 = (sampled[:, WILD_DRAW_4_CARD_ID] > 0).float().mean().item()
        y_p4 = 1.0 if eval_oh[i, WILD_DRAW_4_CARD_ID].item() > 0 else 0.0
        brier_p4.append((p_p4 - y_p4) ** 2)
        acc_p4.append(float((p_p4 >= 0.5) == (y_p4 > 0.5)))

        # C) Active Color
        active_col_mask = (eval_cf[i, :, 4] > 0.5)
        p_col = (sampled[:, active_col_mask].sum(dim=-1) > 0).float().mean().item()
        y_col = 1.0 if eval_oh[i, active_col_mask].sum().item() > 0 else 0.0
        brier_col.append((p_col - y_col) ** 2)
        acc_col.append(float((p_col >= 0.5) == (y_col > 0.5)))

    belief_net.train()
    return {
        "hand_nll": hand_nll,
        "hypergeom_nll": hyper_nll,
        "nll_gain": nll_gain,
        "brier_p2": float(np.mean(brier_p2)) if brier_p2 else 0.0,
        "acc_p2": float(np.mean(acc_p2)) if acc_p2 else 0.0,
        "brier_p4": float(np.mean(brier_p4)) if brier_p4 else 0.0,
        "acc_p4": float(np.mean(acc_p4)) if acc_p4 else 0.0,
        "brier_color": float(np.mean(brier_col)) if brier_col else 0.0,
        "acc_color": float(np.mean(acc_col)) if acc_col else 0.0,
    }


def collect_validation_positions(
    blueprint_actor: UnoActorNet,
    opponent_type: str,
    num_positions: int,
    device: torch.device,
    seed: int = 1000,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collect fixed validation positions for self-play or OOD opponent."""
    num_envs = 32
    env = VecUnoEnv(num_envs=num_envs, seed=seed)
    obs = env.reset()

    if opponent_type == "expert":
        opp_bot = ExpertHeuristicBot(seed=seed)
    else:
        opp_bot = None

    cf_list, gf_list, ht_list, oh_list = [], [], [], []

    while len(cf_list) < num_positions:
        p_acting = obs["current_player"]

        # Form actions
        actions = np.empty(num_envs, dtype=np.int32)
        bp_indices = []
        opp_indices = []

        for i in range(num_envs):
            p = p_acting[i]
            if opp_bot is None or p == 0:
                bp_indices.append(i)
            else:
                opp_indices.append(i)

        if bp_indices:
            bp_cf = torch.from_numpy(obs["card_features_actor"][bp_indices]).to(device)
            bp_gf = torch.from_numpy(obs["global_features"][bp_indices]).to(device)
            bp_ht = torch.from_numpy(obs["history_tokens"][bp_indices]).to(device)
            bp_mask = torch.from_numpy(obs["legal_mask"][bp_indices]).to(device)

            with torch.no_grad():
                logits, _ = blueprint_actor(bp_cf, bp_gf, bp_ht)
                log_p = masked_log_softmax(logits.float(), bp_mask)
                probs = torch.where(bp_mask, log_p.exp(), torch.zeros_like(log_p))
                probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)
                bp_acts = torch.multinomial(probs, 1).squeeze(-1).cpu().numpy().astype(np.int32)

            for idx, act in zip(bp_indices, bp_acts):
                actions[idx] = act

        if opp_indices:
            opp_cf = obs["card_features_actor"][opp_indices]
            opp_gf = obs["global_features"][opp_indices]
            opp_ht = obs["history_tokens"][opp_indices]
            opp_mask = obs["legal_mask"][opp_indices]
            opp_acts = opp_bot.act_batch(opp_cf, opp_gf, opp_ht, opp_mask)
            for idx, act in zip(opp_indices, opp_acts):
                actions[idx] = act

        # Collect mid-game positions with opp_hand_size >= 2
        for i in range(num_envs):
            k_i = obs["opp_hand_counts"][i].sum()
            turn_i = obs["global_features"][i, 6] * 200.0
            if k_i >= 2 and turn_i >= 3 and len(cf_list) < num_positions:
                cf_list.append(obs["card_features_actor"][i].copy())
                gf_list.append(obs["global_features"][i].copy())
                ht_list.append(obs["history_tokens"][i].copy())
                oh_list.append(obs["opp_hand_counts"][i].copy())

        obs, _, _, _ = env.step(actions)

    return (
        torch.from_numpy(np.array(cf_list)).to(device),
        torch.from_numpy(np.array(gf_list)).to(device),
        torch.from_numpy(np.array(ht_list)).to(device),
        torch.from_numpy(np.array(oh_list)).to(device),
    )


def train_belief(
    blueprint_ckpt: str = "runs/blueprint_100m/latest.pt",
    snapshots_dir: Optional[str] = None,
    total_positions: int = 500_000,
    batch_size: int = 256,
    lr: float = 4e-4,
    log_every_steps: int = 50,
    eval_every_steps: int = 250,
    run_dir: str = "runs/belief_20m",
    compile_model: bool = False,
    seed: int = 42,
) -> None:
    """Run online diverse-mixture streaming belief training."""
    os.makedirs(run_dir, exist_ok=True)
    metrics_path = os.path.join(run_dir, "metrics.jsonl")

    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = get_device()

    # Hardware & precision detection matching Milestone 3B
    is_cuda = (device.type == "cuda")
    is_mps = (device.type == "mps")
    scaler = None

    if is_cuda:
        cap = torch.cuda.get_device_capability(0)[0]
        if cap >= 8:
            autocast_dtype = torch.bfloat16
            precision_desc = f"CUDA (sm_{cap}) bfloat16 autocast"
        else:
            autocast_dtype = torch.float16
            scaler = torch.amp.GradScaler("cuda")
            precision_desc = f"CUDA (sm_{cap}) float16 + GradScaler"
    elif is_mps:
        autocast_dtype = torch.bfloat16
        precision_desc = "Apple Silicon MPS bfloat16 autocast"
    else:
        autocast_dtype = torch.float32
        precision_desc = "CPU float32"

    print("=" * 76)
    print("  UNO-ATARAXOS BELIEF NETWORK STREAMING TRAINER (MILESTONE 4)")
    print("=" * 76)
    print(f"Device:            {device} ({precision_desc})")
    print(f"Blueprint Checkpoint: {blueprint_ckpt}")
    print(f"Total Positions:   {total_positions:,}")
    print(f"Batch Size:        {batch_size}")
    print(f"Learning Rate:     {lr}")
    print(f"Run Directory:     {run_dir}")
    print("-" * 76)

    # 1. Load Blueprint Policy
    blueprint_actor = UnoActorNet().to(device)
    if os.path.isfile(blueprint_ckpt):
        ckpt_data = torch.load(blueprint_ckpt, map_location=device, weights_only=False)
        actor_sd = ckpt_data.get("actor_state_dict", ckpt_data.get("actor", ckpt_data))
        load_clean_state_dict(blueprint_actor, actor_sd)
        print(f"✓ Loaded blueprint weights from {blueprint_ckpt}")
    else:
        print(f"[WARNING] Blueprint checkpoint '{blueprint_ckpt}' not found! Using random blueprint weights.")
    blueprint_actor.eval()

    # Load snapshot policy if available
    snapshot_actor: Optional[UnoActorNet] = None
    if snapshots_dir and os.path.isdir(snapshots_dir):
        snap_files = glob.glob(os.path.join(snapshots_dir, "*.pt"))
        if snap_files:
            chosen_snap = random.choice(snap_files)
            snapshot_actor = UnoActorNet().to(device)
            s_data = torch.load(chosen_snap, map_location=device, weights_only=False)
            s_sd = s_data.get("actor_state_dict", s_data.get("actor", s_data))
            load_clean_state_dict(snapshot_actor, s_sd)
            snapshot_actor.eval()
            print(f"✓ Loaded snapshot opponent: {chosen_snap}")

    # 2. Instantiate UnoBeliefNet
    belief_net = UnoBeliefNet(history_dropout=0.15).to(device)
    param_count = sum(p.numel() for p in belief_net.parameters())
    print(f"✓ Initialized UnoBeliefNet ({param_count:,} parameters)")

    if compile_model and is_cuda:
        try:
            print("Compiling belief network with torch.compile...")
            belief_net = torch.compile(belief_net, mode="default")  # type: ignore
        except Exception as e:
            print(f"Warning: torch.compile failed ({e}), continuing uncompiled.")

    optimizer = torch.optim.AdamW(belief_net.parameters(), lr=lr, weight_decay=1e-4)

    # 3. Collect Fixed Validation Sets
    print("Collecting fixed evaluation sets (128 self-play & 128 expert positions)...")
    val_sp_cf, val_sp_gf, val_sp_ht, val_sp_oh = collect_validation_positions(
        blueprint_actor, "selfplay", num_positions=128, device=device, seed=1001
    )
    val_ex_cf, val_ex_gf, val_ex_ht, val_ex_oh = collect_validation_positions(
        blueprint_actor, "expert", num_positions=128, device=device, seed=1002
    )
    print("✓ Validation sets ready!")

    # 4. Multi-Policy Online Environment Mixture
    num_envs = 256
    vec_env = VecUnoEnv(num_envs=num_envs, seed=seed)
    obs = vec_env.reset()

    expert_bot = ExpertHeuristicBot(seed=seed + 1)
    greedy_bot = GreedyBot(seed=seed + 2)
    random_bot = RandomBot(seed=seed + 3)

    # Environment assignment rules:
    # 0..127 (128 envs): Blueprint vs Blueprint
    # 128..153 (26 envs): P0 = Blueprint, P1 = High-Temp (or Snapshot)
    # 154..179 (26 envs): P0 = High-Temp (or Snapshot), P1 = Blueprint
    # 180..205 (26 envs): P0 = Blueprint, P1 = ExpertHeuristic
    # 206..231 (26 envs): P0 = ExpertHeuristic, P1 = Blueprint
    # 232..243 (12 envs): P0 = Blueprint, P1 = GreedyBot
    # 244..255 (12 envs): P0 = Blueprint, P1 = RandomBot

    buffer = StreamingReplayBuffer(capacity=32_768)
    total_positions_seen = 0
    step = 0
    best_val_gain = -float("inf")

    t_start = time.perf_counter()

    print("\nStarting Online Diverse-Mixture Streaming Belief Training...")
    print(f"{'Step':>6} | {'Pos (k)':>8} | {'NLL Loss':>8} | {'Hand NLL':>8} | {'Hyper NLL':>9} | {'Gain':>6} | {'Val SP G':>8} | {'Val Ex G':>8} | {'P2 Brier':>8} | {'P4 Brier':>8} | {'Time (s)':>8}")
    print("-" * 115)

    # Initial eval
    init_sp = evaluate_belief(belief_net, val_sp_cf, val_sp_gf, val_sp_ht, val_sp_oh)
    init_ex = evaluate_belief(belief_net, val_ex_cf, val_ex_gf, val_ex_ht, val_ex_oh)
    print(f"{0:>6} | {0:>8.1f} | {'N/A':>8} | {'N/A':>8} | {'N/A':>9} | {'N/A':>6} | {init_sp['nll_gain']:>8.2f} | {init_ex['nll_gain']:>8.2f} | {init_sp['brier_p2']:>8.3f} | {init_sp['brier_p4']:>8.3f} | {0.0:>8.1f}")

    while total_positions_seen < total_positions:
        # 1. Step Environment Mixture
        p_acting = obs["current_player"]

        bp_norm_indices: List[int] = []
        bp_high_indices: List[int] = []
        expert_indices: List[int] = []
        greedy_indices: List[int] = []
        random_indices: List[int] = []

        for env_i in range(num_envs):
            p = p_acting[env_i]
            if env_i < 128:
                bp_norm_indices.append(env_i)
            elif env_i < 154:
                if p == 0:
                    bp_norm_indices.append(env_i)
                else:
                    bp_high_indices.append(env_i)
            elif env_i < 180:
                if p == 0:
                    bp_high_indices.append(env_i)
                else:
                    bp_norm_indices.append(env_i)
            elif env_i < 206:
                if p == 0:
                    bp_norm_indices.append(env_i)
                else:
                    expert_indices.append(env_i)
            elif env_i < 232:
                if p == 0:
                    expert_indices.append(env_i)
                else:
                    bp_norm_indices.append(env_i)
            elif env_i < 244:
                if p == 0:
                    bp_norm_indices.append(env_i)
                else:
                    greedy_indices.append(env_i)
            else:
                if p == 0:
                    bp_norm_indices.append(env_i)
                else:
                    random_indices.append(env_i)

        actions = np.empty(num_envs, dtype=np.int32)

        # Batch Blueprint / High-Temp neural evaluations
        all_nn_indices = bp_norm_indices + bp_high_indices
        if all_nn_indices:
            nn_cf = torch.from_numpy(obs["card_features_actor"][all_nn_indices]).to(device)
            nn_gf = torch.from_numpy(obs["global_features"][all_nn_indices]).to(device)
            nn_ht = torch.from_numpy(obs["history_tokens"][all_nn_indices]).to(device)
            nn_mask = torch.from_numpy(obs["legal_mask"][all_nn_indices]).to(device)

            with torch.no_grad():
                logits, _ = blueprint_actor(nn_cf, nn_gf, nn_ht)

            # Assign normal temp (1.0) and high temp (1.5)
            N_norm = len(bp_norm_indices)
            N_high = len(bp_high_indices)

            if N_norm > 0:
                log_p_norm = masked_log_softmax(logits[:N_norm].float(), nn_mask[:N_norm])
                p_norm = torch.where(nn_mask[:N_norm], log_p_norm.exp(), torch.zeros_like(log_p_norm))
                p_norm = p_norm / p_norm.sum(dim=-1, keepdim=True).clamp_min(1e-9)
                acts_norm = torch.multinomial(p_norm, 1).squeeze(-1).cpu().numpy().astype(np.int32)
                for idx, a in zip(bp_norm_indices, acts_norm):
                    actions[idx] = a

            if N_high > 0:
                log_p_high = masked_log_softmax(logits[N_norm:].float(), nn_mask[N_norm:])
                p_high = torch.where(nn_mask[N_norm:], (log_p_high / 1.5).exp(), torch.zeros_like(log_p_high))
                p_high = p_high / p_high.sum(dim=-1, keepdim=True).clamp_min(1e-9)
                acts_high = torch.multinomial(p_high, 1).squeeze(-1).cpu().numpy().astype(np.int32)
                for idx, a in zip(bp_high_indices, acts_high):
                    actions[idx] = a

        # Heuristic Bots
        if expert_indices:
            e_cf = obs["card_features_actor"][expert_indices]
            e_gf = obs["global_features"][expert_indices]
            e_ht = obs["history_tokens"][expert_indices]
            e_mask = obs["legal_mask"][expert_indices]
            e_acts = expert_bot.act_batch(e_cf, e_gf, e_ht, e_mask)
            for idx, a in zip(expert_indices, e_acts):
                actions[idx] = a

        if greedy_indices:
            g_cf = obs["card_features_actor"][greedy_indices]
            g_gf = obs["global_features"][greedy_indices]
            g_ht = obs["history_tokens"][greedy_indices]
            g_mask = obs["legal_mask"][greedy_indices]
            g_acts = greedy_bot.act_batch(g_cf, g_gf, g_ht, g_mask)
            for idx, a in zip(greedy_indices, g_acts):
                actions[idx] = a

        if random_indices:
            r_cf = obs["card_features_actor"][random_indices]
            r_gf = obs["global_features"][random_indices]
            r_ht = obs["history_tokens"][random_indices]
            r_mask = obs["legal_mask"][random_indices]
            r_acts = random_bot.act_batch(r_cf, r_gf, r_ht, r_mask)
            for idx, a in zip(random_indices, r_acts):
                actions[idx] = a

        # 2. Add current positions to buffer
        valid_pos = (obs["opp_hand_counts"].sum(axis=-1) > 0)
        buffer.add_batch(
            obs["card_features_actor"][valid_pos],
            obs["global_features"][valid_pos],
            obs["history_tokens"][valid_pos],
            obs["opp_hand_counts"][valid_pos],
        )
        total_positions_seen += int(valid_pos.sum())

        obs, _, _, _ = vec_env.step(actions)

        # 3. Train Step if buffer has sufficient data
        if buffer.size >= batch_size:
            step += 1
            b_cf, b_gf, b_ht, b_oh = buffer.sample(batch_size, device)

            optimizer.zero_grad()

            if scaler is not None:
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    loss_dict = belief_net.compute_nll_loss(b_cf, b_gf, b_ht, b_oh, apply_s4=True)
                    loss = loss_dict["loss"]
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(belief_net.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            elif is_cuda or is_mps:
                with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                    loss_dict = belief_net.compute_nll_loss(b_cf, b_gf, b_ht, b_oh, apply_s4=True)
                    loss = loss_dict["loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(belief_net.parameters(), max_norm=1.0)
                optimizer.step()
            else:
                loss_dict = belief_net.compute_nll_loss(b_cf, b_gf, b_ht, b_oh, apply_s4=True)
                loss = loss_dict["loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(belief_net.parameters(), max_norm=1.0)
                optimizer.step()

            # Cosine decay schedule
            progress = min(1.0, float(total_positions_seen) / float(total_positions))
            curr_lr = lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))
            for param_group in optimizer.param_groups:
                param_group["lr"] = curr_lr

            # 4. Periodic Evaluation & Logging
            if step % eval_every_steps == 0 or total_positions_seen >= total_positions:
                val_sp = evaluate_belief(belief_net, val_sp_cf, val_sp_gf, val_sp_ht, val_sp_oh)
                val_ex = evaluate_belief(belief_net, val_ex_cf, val_ex_gf, val_ex_ht, val_ex_oh)

                elapsed = time.perf_counter() - t_start
                pos_k = total_positions_seen / 1000.0

                print(
                    f"{step:>6} | {pos_k:>8.1f} | {loss_dict['card_nll']:>8.3f} | {loss_dict['hand_nll']:>8.2f} | "
                    f"{loss_dict['hypergeom_nll']:>9.2f} | {loss_dict['nll_gain']:>6.2f} | "
                    f"{val_sp['nll_gain']:>8.2f} | {val_ex['nll_gain']:>8.2f} | "
                    f"{val_sp['brier_p2']:>8.3f} | {val_sp['brier_p4']:>8.3f} | {elapsed:>8.1f}"
                )

                # Save metrics log
                record = {
                    "step": step,
                    "positions": total_positions_seen,
                    "train_card_nll": loss_dict["card_nll"],
                    "train_hand_nll": loss_dict["hand_nll"],
                    "train_hyper_nll": loss_dict["hypergeom_nll"],
                    "train_gain": loss_dict["nll_gain"],
                    "val_sp_hand_nll": val_sp["hand_nll"],
                    "val_sp_gain": val_sp["nll_gain"],
                    "val_ex_hand_nll": val_ex["hand_nll"],
                    "val_ex_gain": val_ex["nll_gain"],
                    "val_sp_brier_p2": val_sp["brier_p2"],
                    "val_sp_acc_p2": val_sp["acc_p2"],
                    "val_sp_brier_p4": val_sp["brier_p4"],
                    "val_sp_acc_p4": val_sp["acc_p4"],
                    "val_sp_brier_col": val_sp["brier_color"],
                    "val_sp_acc_col": val_sp["acc_color"],
                    "lr": curr_lr,
                    "elapsed_sec": elapsed,
                }
                with open(metrics_path, "a") as f:
                    f.write(json.dumps(record) + "\n")

                # Save latest checkpoint
                clean_sd = get_clean_state_dict(belief_net)
                torch.save(
                    {"belief_net": clean_sd, "step": step, "positions": total_positions_seen},
                    os.path.join(run_dir, "latest.pt"),
                )

                # Save best checkpoint
                avg_gain = 0.5 * (val_sp["nll_gain"] + val_ex["nll_gain"])
                if avg_gain > best_val_gain:
                    best_val_gain = avg_gain
                    torch.save(
                        {"belief_net": clean_sd, "step": step, "positions": total_positions_seen, "val_gain": avg_gain},
                        os.path.join(run_dir, "best.pt"),
                    )

            elif step % log_every_steps == 0:
                elapsed = time.perf_counter() - t_start
                pos_k = total_positions_seen / 1000.0
                print(
                    f"{step:>6} | {pos_k:>8.1f} | {loss_dict['card_nll']:>8.3f} | {loss_dict['hand_nll']:>8.2f} | "
                    f"{loss_dict['hypergeom_nll']:>9.2f} | {loss_dict['nll_gain']:>6.2f} | "
                    f"{'--':>8} | {'--':>8} | {'--':>8} | {'--':>8} | {elapsed:>8.1f}"
                )

    total_time = time.perf_counter() - t_start
    print("-" * 115)
    print(f"✓ Belief Network training complete! Processed {total_positions_seen:,} positions in {total_time:.1f}s ({total_positions_seen / total_time:.0f} pos/sec).")
    print(f"✓ Saved latest model to {os.path.join(run_dir, 'latest.pt')}")
    print(f"✓ Saved best model to {os.path.join(run_dir, 'best.pt')}")


def main():
    parser = argparse.ArgumentParser(description="Uno-Ataraxos Online Diverse-Mixture Belief Network Trainer")
    parser.add_argument("--blueprint-ckpt", type=str, default="runs/blueprint_100m/latest.pt", help="Path to blueprint actor checkpoint")
    parser.add_argument("--snapshots-dir", type=str, default=None, help="Path to historical opponent snapshots directory")
    parser.add_argument("--total-positions", type=int, default=500_000, help="Total positions to train on")
    parser.add_argument("--batch-size", type=int, default=256, help="Minibatch size")
    parser.add_argument("--lr", type=float, default=4e-4, help="Learning rate")
    parser.add_argument("--log-every-steps", type=int, default=50, help="Steps between training logs")
    parser.add_argument("--eval-every-steps", type=int, default=250, help="Steps between validation evaluations")
    parser.add_argument("--run-dir", type=str, default="runs/belief_20m", help="Directory to save checkpoints & metrics")
    parser.add_argument("--compile", action="store_true", help="Compile belief model with torch.compile on CUDA")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train_belief(
        blueprint_ckpt=args.blueprint_ckpt,
        snapshots_dir=args.snapshots_dir,
        total_positions=args.total_positions,
        batch_size=args.batch_size,
        lr=args.lr,
        log_every_steps=args.log_every_steps,
        eval_every_steps=args.eval_every_steps,
        run_dir=args.run_dir,
        compile_model=args.compile,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
