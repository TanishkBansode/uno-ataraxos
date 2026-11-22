"""One-Click Launcher for uno-ataraxos 100M-Step Blueprint Self-Play Training Run.

Handles automatic environment detection, C++ extension verification, GPU precision
configuration, and resumable 100M-step self-play training across Apple Silicon (MPS),
NVIDIA CUDA, and CPU.
"""

import argparse
import os
import sys
import platform
import subprocess
import torch


def main():
    parser = argparse.ArgumentParser(description="uno-ataraxos 100M Blueprint Training Launcher")
    parser.add_argument("--run-dir", type=str, default="runs/blueprint_100m", help="Directory for checkpoints and metrics")
    parser.add_argument("--total-steps", type=int, default=100_000_000, help="Total environment steps to train")
    parser.add_argument("--num-envs", type=int, default=512, help="Number of vectorized parallel environments")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile (CUDA only)")
    parser.add_argument("--resume", type=str, default=None, help="Explicit checkpoint path to resume from")
    args = parser.parse_args()

    platform_desc = f"{platform.system()} {platform.machine()}"

    print("=" * 72)
    print("  UNO-ATARAXOS 100M-STEP BLUEPRINT TRAINING LAUNCHER")
    print("=" * 72)
    print(f"Platform:          {platform_desc}")
    print(f"Python Version:    {sys.version.split()[0]}")
    print(f"PyTorch Version:   {torch.__version__}")
    print(f"CUDA Available:    {torch.cuda.is_available()}")

    compile_flag = args.compile
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU:               {gpu_name} ({vram_gb:.1f} GB VRAM, sm_{capability[0]}{capability[1]})")
        if capability[0] >= 8:
            precision_note = "bfloat16 (Native Tensor Cores)"
            if not args.compile:
                compile_flag = True
        else:
            precision_note = "float16 + GradScaler"
            compile_flag = False
        print(f"Precision:         {precision_note}")
    elif torch.backends.mps.is_available():
        print("Device:            Apple Silicon MPS (bfloat16)")
        compile_flag = False
    else:
        print("Device:            CPU Fallback (float32)")
        compile_flag = False
    print("-" * 72)

    # Ensure repo root is on sys.path
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    # 1. Verify C++ extension is built
    try:
        import uno_ataraxos as ua
        from uno_ataraxos import VecUnoEnv
        vec = VecUnoEnv(num_envs=4, seed=42)
        obs = vec.reset()
        print("✓ C++ VecUnoEnv extension verified & functional!")
    except (ImportError, ModuleNotFoundError):
        print("Building C++ extension in-place...")
        subprocess.check_call([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=repo_root)
        import uno_ataraxos as ua
        from uno_ataraxos import VecUnoEnv
        vec = VecUnoEnv(num_envs=4, seed=42)
        obs = vec.reset()
        print("✓ C++ extension built successfully!")

    # 2. Check for existing checkpoint to resume
    if args.resume:
        resume_arg = args.resume
        print(f"✓ Explicit resume checkpoint specified: {resume_arg}")
    else:
        latest_ckpt = os.path.join(args.run_dir, "latest.pt")
        resume_arg = latest_ckpt if os.path.isfile(latest_ckpt) else None
        if resume_arg:
            print(f"✓ Found existing checkpoint: {resume_arg} -> Auto-resuming!")
        else:
            print(f"✓ Starting fresh run in {args.run_dir}")

    # 3. Launch Training
    from uno_ataraxos.train_blueprint import train

    train(
        total_steps=args.total_steps,
        num_envs=args.num_envs,
        rollout_steps=64,
        minibatch_size=1024,
        ppo_epochs=2,
        eval_every_steps=2_500_000,
        num_eval_decks=250,
        snapshot_every_steps=10_000_000,
        save_every_mins=15.0,
        run_dir=args.run_dir,
        resume=resume_arg,
        compile_model=compile_flag,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
