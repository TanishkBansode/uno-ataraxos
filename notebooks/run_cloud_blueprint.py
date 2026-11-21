"""One-Click Cloud Launcher for uno-ataraxos 100M-Step Blueprint Training Run.

Compatible with:
- Kaggle Notebooks (NVIDIA T4 / P100 / L4)
- Google Colab (Free T4 / Colab Pro A100 / L4)
- Dedicated Cloud GPU instances (RunPod, Lambda, Vast.ai, GCP, AWS)

Handles automatic environment detection, C++ extension verification, GPU precision
configuration, and resumable 100M-step self-play training.
"""

import os
import sys
import platform
import subprocess
import torch


def detect_environment():
    """Detect cloud notebook platform and setup default paths."""
    if os.path.exists("/kaggle/working"):
        env_name = "Kaggle"
        default_run_dir = "/kaggle/working/runs/blueprint_100m"
        data_dir = "/kaggle/working"
    elif "COLAB_GPU" in os.environ or os.path.exists("/content"):
        env_name = "Google Colab"
        # If Google Drive is mounted, prefer saving to Drive for persistence
        if os.path.exists("/content/drive/MyDrive"):
            default_run_dir = "/content/drive/MyDrive/uno_ataraxos/runs/blueprint_100m"
        else:
            default_run_dir = "/content/runs/blueprint_100m"
        data_dir = "/content"
    else:
        env_name = f"Local/Cloud ({platform.system()} {platform.machine()})"
        default_run_dir = "runs/blueprint_100m"
        data_dir = "."

    return env_name, default_run_dir, data_dir


def main():
    env_name, default_run_dir, data_dir = detect_environment()

    print("=" * 72)
    print("  UNO-ATARAXOS 100M-STEP CLOUD BLUEPRINT LAUNCHER")
    print("=" * 72)
    print(f"Platform:          {env_name}")
    print(f"Python Version:    {sys.version.split()[0]}")
    print(f"PyTorch Version:   {torch.__version__}")
    print(f"CUDA Available:    {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU:               {gpu_name} ({vram_gb:.1f} GB VRAM, sm_{capability[0]}{capability[1]})")
        if capability[0] >= 8:
            precision_note = "bfloat16 (Ampere+ Native Tensor Cores)"
            compile_flag = True
        else:
            precision_note = "float16 + GradScaler (Turing/Volta e.g. T4)"
            compile_flag = False  # torch.compile on T4 is optional / slower warmup
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
    except (ImportError, ModuleNotFoundError) as e:
        print("Building C++ extension in-place...")
        subprocess.check_call([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=repo_root)
        import uno_ataraxos as ua
        from uno_ataraxos import VecUnoEnv
        vec = VecUnoEnv(num_envs=4, seed=42)
        obs = vec.reset()
        print("✓ C++ extension built successfully!")

    # 2. Check for existing checkpoint to resume across session timeouts
    latest_ckpt = os.path.join(default_run_dir, "latest.pt")
    resume_arg = latest_ckpt if os.path.isfile(latest_ckpt) else None
    if resume_arg:
        print(f"✓ Found existing checkpoint: {resume_arg} -> Auto-resuming!")
    else:
        print(f"✓ Starting fresh run in {default_run_dir}")

    # 3. Launch Training
    from uno_ataraxos.train_blueprint import train

    train(
        total_steps=100_000_000,
        num_envs=512,
        rollout_steps=64,
        minibatch_size=1024,
        ppo_epochs=2,
        eval_every_steps=2_500_000,
        num_eval_decks=250,
        snapshot_every_steps=10_000_000,
        save_every_mins=15.0,
        run_dir=default_run_dir,
        resume=resume_arg,
        compile_model=compile_flag,
        seed=42,
    )


if __name__ == "__main__":
    main()
