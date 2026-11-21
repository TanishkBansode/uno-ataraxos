"""Benchmark M5 MPS latency, parameter counts, and peak memory for uno-ataraxos (Milestone 2.5).

Benchmarks BOTH FP32 and BF16 autocast across:
1. Actor rollout inference (B = 512, no_grad)
2. Full Actor + Critic + EMA train step (B = 1024)
"""

import os
import platform
import resource
import time
import torch
import torch.nn.functional as F

import uno_ataraxos as ua
from uno_ataraxos.networks import UnoActorNet, UnoQCriticNet
from uno_ataraxos.rl_math import (
    get_device,
    masked_entropy,
    masked_log_softmax,
    masked_reverse_kl,
    update_ema,
)


def get_peak_ram_mb() -> float:
    """Return peak process RAM in MB."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return usage / (1024.0 * 1024.0)
    return usage / 1024.0


def sync_device(device: torch.device):
    """Synchronize device for accurate timing."""
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def benchmark_m5():
    device = get_device()
    device_type = "mps" if device.type == "mps" else ("cuda" if device.type == "cuda" else "cpu")
    print("=" * 72)
    print("  UNO-ATARAXOS MILESTONE 2.5: M5 MPS HYBRID NEURAL BENCHMARK")
    print("=" * 72)
    print(f"Platform:       {platform.platform()} ({platform.machine()})")
    print(f"Device:         {device} (PyTorch {torch.__version__})")
    print(f"CPU count:      {os.cpu_count()} cores")
    print("-" * 72)

    # 1. Instantiate models
    actor = UnoActorNet().to(device)
    critic = UnoQCriticNet().to(device)
    ema_actor = UnoActorNet().to(device)
    ema_actor.load_state_dict(actor.state_dict())
    ema_actor.eval()

    actor_params = sum(p.numel() for p in actor.parameters())
    critic_params = sum(p.numel() for p in critic.parameters())

    print("MODEL PARAMETER COUNTS:")
    print(f"  UnoActorNet params:   {actor_params:,} ({actor_params / 1e6:.2f}M)")
    print(f"  UnoQCriticNet params:  {critic_params:,} ({critic_params / 1e6:.2f}M)")
    print("-" * 72)

    # 2. Benchmark Actor Rollout Inference (B = 512, no_grad)
    B_inf = 512
    card_feats_inf = torch.randn(B_inf, 54, 7, dtype=torch.float32, device=device)
    global_feats_inf = torch.randn(B_inf, 10, dtype=torch.float32, device=device)
    hist_tokens_inf = torch.zeros((B_inf, 48, 5), dtype=torch.int32, device=device)

    # 2a. FP32 Inference
    print(f"[FP32] Benchmarking Actor Rollout Inference (B = {B_inf}, no_grad)...")
    for _ in range(15):
        with torch.no_grad():
            _ = actor(card_feats_inf, global_feats_inf, hist_tokens_inf)
    sync_device(device)

    n_inf_iters = 50
    t0 = time.perf_counter()
    for _ in range(n_inf_iters):
        with torch.no_grad():
            _ = actor(card_feats_inf, global_feats_inf, hist_tokens_inf)
        sync_device(device)
    fp32_inf_ms = ((time.perf_counter() - t0) / n_inf_iters) * 1000.0
    fp32_inf_sps = B_inf / (fp32_inf_ms / 1000.0)
    print(f"  [FP32] Actor Inference Latency:   {fp32_inf_ms:.2f} ms/batch ({fp32_inf_sps:,.0f} inf/sec)")

    # 2b. BF16 Autocast Inference
    print(f"[BF16 Autocast] Benchmarking Actor Rollout Inference (B = {B_inf}, no_grad)...")
    for _ in range(15):
        with torch.no_grad(), torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            _ = actor(card_feats_inf, global_feats_inf, hist_tokens_inf)
    sync_device(device)

    t0 = time.perf_counter()
    for _ in range(n_inf_iters):
        with torch.no_grad(), torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            _ = actor(card_feats_inf, global_feats_inf, hist_tokens_inf)
        sync_device(device)
    bf16_inf_ms = ((time.perf_counter() - t0) / n_inf_iters) * 1000.0
    bf16_inf_sps = B_inf / (bf16_inf_ms / 1000.0)
    print(f"  [BF16] Actor Inference Latency:   {bf16_inf_ms:.2f} ms/batch ({bf16_inf_sps:,.0f} inf/sec)")
    print("-" * 72)

    # 3. Benchmark Full Training Step (B = 1024)
    B_train = 1024
    card_feats_train = torch.randn(B_train, 54, 7, dtype=torch.float32, device=device)
    opp_hand_train = torch.randn(B_train, 54, dtype=torch.float32, device=device)
    global_feats_train = torch.randn(B_train, 10, dtype=torch.float32, device=device)
    hist_tokens_train = torch.zeros((B_train, 48, 5), dtype=torch.int32, device=device)
    legal_mask_train = torch.ones((B_train, 61), dtype=torch.bool, device=device)
    actions_train = torch.randint(0, 61, (B_train,), device=device)
    q_targets_train = torch.randn(B_train, dtype=torch.float32, device=device).clamp(-1.0, 1.0)
    wdl_targets_train = torch.randint(0, 3, (B_train,), device=device)

    optimizer = torch.optim.AdamW(
        list(actor.parameters()) + list(critic.parameters()),
        lr=3e-4,
        weight_decay=1e-4,
    )

    # 3a. FP32 Training Step
    print(f"[FP32] Benchmarking Full Actor + Critic + EMA Training Step (B = {B_train})...")
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        # Critic fwd & bwd
        q_vals = critic(card_feats_train, opp_hand_train, global_feats_train, hist_tokens_train)
        q_loss = F.mse_loss(q_vals.gather(1, actions_train.unsqueeze(1)).squeeze(1), q_targets_train)
        q_loss.backward()

        # EMA fwd (no_grad)
        with torch.no_grad():
            ref_logits, _ = ema_actor(card_feats_train, global_feats_train, hist_tokens_train)
            ref_logp = masked_log_softmax(ref_logits, legal_mask_train)

        # Actor fwd & bwd
        p_logits, wdl = actor(card_feats_train, global_feats_train, hist_tokens_train)
        kl_loss = masked_reverse_kl(p_logits, ref_logp, legal_mask_train).mean()
        ent_loss = -0.01 * masked_entropy(p_logits, legal_mask_train).mean()
        wdl_loss = F.cross_entropy(wdl, wdl_targets_train)
        actor_loss = kl_loss + ent_loss + wdl_loss
        actor_loss.backward()

        optimizer.step()
        update_ema(ema_actor, actor, decay=0.999)
    sync_device(device)

    n_train_iters = 25
    t0 = time.perf_counter()
    for _ in range(n_train_iters):
        optimizer.zero_grad(set_to_none=True)
        q_vals = critic(card_feats_train, opp_hand_train, global_feats_train, hist_tokens_train)
        q_loss = F.mse_loss(q_vals.gather(1, actions_train.unsqueeze(1)).squeeze(1), q_targets_train)
        q_loss.backward()

        with torch.no_grad():
            ref_logits, _ = ema_actor(card_feats_train, global_feats_train, hist_tokens_train)
            ref_logp = masked_log_softmax(ref_logits, legal_mask_train)

        p_logits, wdl = actor(card_feats_train, global_feats_train, hist_tokens_train)
        kl_loss = masked_reverse_kl(p_logits, ref_logp, legal_mask_train).mean()
        ent_loss = -0.01 * masked_entropy(p_logits, legal_mask_train).mean()
        wdl_loss = F.cross_entropy(wdl, wdl_targets_train)
        actor_loss = kl_loss + ent_loss + wdl_loss
        actor_loss.backward()

        optimizer.step()
        update_ema(ema_actor, actor, decay=0.999)
        sync_device(device)

    fp32_train_ms = ((time.perf_counter() - t0) / n_train_iters) * 1000.0
    fp32_train_sps = B_train / (fp32_train_ms / 1000.0)
    print(f"  [FP32] Training Step Latency:     {fp32_train_ms:.2f} ms/step ({fp32_train_sps:,.0f} samples/sec)")

    # 3b. BF16 Autocast Training Step
    print(f"[BF16 Autocast] Benchmarking Full Actor + Critic + EMA Training Step (B = {B_train})...")
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            q_vals = critic(card_feats_train, opp_hand_train, global_feats_train, hist_tokens_train)
            q_loss = F.mse_loss(q_vals.gather(1, actions_train.unsqueeze(1)).squeeze(1), q_targets_train)
        q_loss.backward()

        with torch.no_grad(), torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            ref_logits, _ = ema_actor(card_feats_train, global_feats_train, hist_tokens_train)
            ref_logp = masked_log_softmax(ref_logits, legal_mask_train)

        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            p_logits, wdl = actor(card_feats_train, global_feats_train, hist_tokens_train)
            kl_loss = masked_reverse_kl(p_logits, ref_logp, legal_mask_train).mean()
            ent_loss = -0.01 * masked_entropy(p_logits, legal_mask_train).mean()
            wdl_loss = F.cross_entropy(wdl, wdl_targets_train)
            actor_loss = kl_loss + ent_loss + wdl_loss
        actor_loss.backward()

        optimizer.step()
        update_ema(ema_actor, actor, decay=0.999)
    sync_device(device)

    t0 = time.perf_counter()
    for _ in range(n_train_iters):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            q_vals = critic(card_feats_train, opp_hand_train, global_feats_train, hist_tokens_train)
            q_loss = F.mse_loss(q_vals.gather(1, actions_train.unsqueeze(1)).squeeze(1), q_targets_train)
        q_loss.backward()

        with torch.no_grad(), torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            ref_logits, _ = ema_actor(card_feats_train, global_feats_train, hist_tokens_train)
            ref_logp = masked_log_softmax(ref_logits, legal_mask_train)

        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            p_logits, wdl = actor(card_feats_train, global_feats_train, hist_tokens_train)
            kl_loss = masked_reverse_kl(p_logits, ref_logp, legal_mask_train).mean()
            ent_loss = -0.01 * masked_entropy(p_logits, legal_mask_train).mean()
            wdl_loss = F.cross_entropy(wdl, wdl_targets_train)
            actor_loss = kl_loss + ent_loss + wdl_loss
        actor_loss.backward()

        optimizer.step()
        update_ema(ema_actor, actor, decay=0.999)
        sync_device(device)

    bf16_train_ms = ((time.perf_counter() - t0) / n_train_iters) * 1000.0
    bf16_train_sps = B_train / (bf16_train_ms / 1000.0)
    peak_ram = get_peak_ram_mb()

    print(f"  [BF16] Training Step Latency:     {bf16_train_ms:.2f} ms/step ({bf16_train_sps:,.0f} samples/sec)")
    print(f"  Peak Process RAM:                {peak_ram:.2f} MB")
    print("-" * 72)

    # 4. Comparative Speedup vs Milestone 2 Baseline
    m2_inf_baseline_ms = 194.80
    m2_train_baseline_ms = 3376.76
    best_inf_ms = min(fp32_inf_ms, bf16_inf_ms)
    best_train_ms = min(fp32_train_ms, bf16_train_ms)
    speedup_inf = m2_inf_baseline_ms / best_inf_ms
    speedup_train = m2_train_baseline_ms / best_train_ms

    print("SPEEDUP VS MILESTONE 2 BASELINE (194.8 ms inf / 3,376.8 ms train):")
    print(f"  Actor Rollout Inference Speedup:  {speedup_inf:.2f}x ({m2_inf_baseline_ms:.1f} ms -> {best_inf_ms:.2f} ms)")
    print(f"  Full Training Step Speedup:       {speedup_train:.2f}x ({m2_train_baseline_ms:.1f} ms -> {best_train_ms:.2f} ms)")
    print("=" * 72)


if __name__ == "__main__":
    benchmark_m5()
