"""Throughput benchmark for VecUnoEnv (B=512).

Measures:
- Steps per second (SPS) for 2,000,000 steps
- Games per second (GPS)
- Peak RAM usage on macOS ARM64
"""

import time
import os
import resource
import platform
import numpy as np
import uno_ataraxos as ua


def get_peak_ram_mb() -> float:
    """Return peak RAM usage in megabytes (macOS returns bytes)."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return usage / (1024.0 * 1024.0)
    return usage / 1024.0


def benchmark_vec_env(
    num_envs: int = 512,
    target_steps: int = 2_000_000,
    num_threads: int = 0,
):
    print("=" * 70)
    print("  UNO-ATARAXOS VECUNOENV THROUGHPUT BENCHMARK (MILESTONE 1)")
    print("=" * 70)
    print(f"Platform:       {platform.platform()} ({platform.machine()})")
    print(f"CPU count:      {os.cpu_count()} cores")
    print(f"Batch size (B): {num_envs} parallel environments")
    print(f"Target steps:   {target_steps:,} environment steps")
    print("-" * 70)

    # Initialize environment
    t_init_0 = time.perf_counter()
    env = ua.VecUnoEnv(num_envs=num_envs, seed=42, num_threads=num_threads, auto_reset=True)
    t_init = time.perf_counter() - t_init_0
    print(f"Env initialized in {t_init * 1000:.2f} ms")

    # Warmup
    print("Warming up for 20,480 steps (40 batches)...")
    for _ in range(40):
        env.step_random_legal()

    # Reset step/episode counters for accurate measurement
    start_steps = env.total_steps
    start_episodes = env.total_episodes

    num_batches = target_steps // num_envs
    actual_steps = num_batches * num_envs

    print(f"Running benchmark: {num_batches:,} batches x {num_envs} envs = {actual_steps:,} steps...")
    t0 = time.perf_counter()
    for _ in range(num_batches):
        env.step_random_legal()
    elapsed = time.perf_counter() - t0

    end_steps = env.total_steps
    end_episodes = env.total_episodes

    total_steps = end_steps - start_steps
    total_episodes = end_episodes - start_episodes

    steps_per_sec = total_steps / elapsed
    games_per_sec = total_episodes / elapsed
    peak_ram = get_peak_ram_mb()

    print("-" * 70)
    print("BENCHMARK RESULTS:")
    print(f"  Total Steps:      {total_steps:,} steps")
    print(f"  Wall Time:        {elapsed:.4f} seconds")
    print(f"  Throughput (SPS): {steps_per_sec:,.0f} steps/second ({steps_per_sec / 1e6:.2f}M SPS)")
    print(f"  Games Completed:  {total_episodes:,} games")
    print(f"  Throughput (GPS): {games_per_sec:,.0f} games/second")
    print(f"  Peak RAM:         {peak_ram:.2f} MB")
    print("=" * 70)

    target_sps = 5_000_000
    if steps_per_sec >= target_sps:
        print(f"TARGET ACHIEVED: {steps_per_sec:,.0f} SPS >= {target_sps:,.0f} SPS target! (PASS)")
    else:
        print(f"Target: {target_sps:,.0f} SPS, achieved {steps_per_sec:,.0f} SPS")

    return {
        "steps_per_sec": steps_per_sec,
        "games_per_sec": games_per_sec,
        "peak_ram_mb": peak_ram,
        "elapsed": elapsed,
        "total_steps": total_steps,
        "total_episodes": total_episodes,
    }


if __name__ == "__main__":
    benchmark_vec_env(num_envs=512, target_steps=2_000_000)
