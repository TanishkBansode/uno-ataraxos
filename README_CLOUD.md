# Cloud Training Guide (Kaggle / Colab / Linux CUDA)

`uno-ataraxos` is fully cross-platform and optimized for both **Apple Silicon (M-series MPS)** and **Linux NVIDIA CUDA (T4 / L4 / A100 / H100)**.

---

## 1. Quickstart: 3 Commands to Launch 100M-Step Self-Play

Copy and paste these commands into a Kaggle Notebook or Google Colab (with GPU accelerator enabled):

```bash
# 1. Install dependencies & build C++ extension in-place (takes ~15 seconds)
pip install -r requirements.txt
python setup.py build_ext --inplace

# 2. Run test suite to verify 100% environment health & parity
pytest tests/ -v

# 3. Launch the 100M-Step Blueprint Training Run (Stage A)
python -m uno_ataraxos.train_blueprint \
  --total-steps 100000000 \
  --num-envs 512 \
  --rollout-steps 64 \
  --minibatch-size 1024 \
  --ppo-epochs 2 \
  --eval-every-steps 2500000 \
  --snapshot-every-steps 10000000 \
  --save-every-mins 15 \
  --compile \
  --run-dir /kaggle/working/runs/blueprint_100m
```

*(If installing as an editable package with pip, use `pip install -e . --no-build-isolation` or `pip install .`)*.

Or simply execute the one-click script:
```bash
python notebooks/run_cloud_blueprint.py
```

---

## 2. End-to-End Cloud Pipeline (Stages A -> B -> C -> D)

Once Stage A (100M Blueprint) completes or reaches a strong checkpoint (`best.pt`), proceed with the downstream stages:

### Stage B: 20M-Position Hard-Masked Generative Belief Network
Trains the autoregressive generative belief network on streaming rollouts (50% self-play + 25% greedy + 25% random + expert heuristic OOD games) to learn accurate opponent hand posteriors:
```bash
python -m uno_ataraxos.train_belief \
  --total-positions 20000000 \
  --batch-size 256 \
  --lr 3e-4 \
  --eval-every-steps 5000 \
  --eval-positions 50000 \
  --blueprint-ckpt /kaggle/working/runs/blueprint_100m/best.pt \
  --save-dir /kaggle/working/runs/belief_20m
```
*Expected training time on Kaggle T4: ~2 hours.*

### Stage C: 15M-Step VRPO Neural Exploiter Red-Team
Measures the empirical Nash exploitability of the frozen Blueprint by training a dedicated adversarial agent (pure PPO + entropy floor) strictly on the frozen target across 500+ vectorized parallel environments:
```bash
python -m uno_ataraxos.train_exploiter \
  --target-ckpt /kaggle/working/runs/blueprint_100m/best.pt \
  --total-steps 15000000 \
  --num-envs 512 \
  --rollout-len 64 \
  --eval-every-steps 1000000 \
  --eval-decks 250 \
  --run-dir /kaggle/working/runs/exploiter_15m
```
*Expected training time on Kaggle T4: ~1.5 hours.*

### Stage D: Search Hyperparameter Sweep & Defense Verification
Runs mirrored duplicate tournaments against the frozen Blueprint to find the Pareto-optimal MMD parameters ($\eta, \tau, \text{leaf\_eval}$), saves `search_config.json`, and verifies search defends against the exploiter:
```bash
python scripts/tune_search.py \
  --blueprint /kaggle/working/runs/blueprint_100m/best.pt \
  --belief /kaggle/working/runs/belief_20m/best.pt \
  --num-decks 100 \
  --out-config /kaggle/working/runs/search_config.json
```

---

## 2. Automatic Precision & Optimization Matrix

The trainer automatically detects your hardware and configures the optimal mixed-precision pipeline:

| Hardware | Accelerator | Autocast Dtype | Scaler | `torch.compile` |
| :--- | :--- | :--- | :--- | :--- |
| **NVIDIA A100 / L4 / H100** | Compute capability $\ge 8.0$ | `torch.bfloat16` | None | Supported (`--compile`) |
| **NVIDIA T4 / V100** (Free Kaggle/Colab) | Compute capability $< 8.0$ | `torch.float16` | `torch.amp.GradScaler("cuda")` | Optional |
| **Apple Silicon M5 / M4 / M3** | Apple Silicon GPU (MPS) | `torch.bfloat16` | None | Disabled (MPS uncompiled) |
| **CPU Fallback** | Linux / macOS CPU | `torch.float32` | None | Disabled |

- **Numerical Invariant:** Regardless of precision (`bf16` or `fp16`), all network outputs are explicitly cast to `float32`, and all VRPO Expected SARSA(λ), reverse KL divergence, policy entropy, and WDL calculations execute in full `float32`.
- **Checkpoint Portability:** Checkpoints automatically unwrap `_orig_mod` if compiled, allowing seamless transfer between Kaggle CUDA and local MacBook Air M5.

---

## 3. Session Persistence & Resuming (Kaggle & Colab)

### A. Kaggle 12-Hour Session Persistence
1. Checkpoints (`latest.pt`, `best.pt`, `snapshots/`) are saved every 15 minutes to `/kaggle/working/runs/blueprint_100m`.
2. When your 12-hour session ends, save the output as a Kaggle Dataset (`uno-ataraxos-ckpt`).
3. In your next session, attach that dataset as an input and resume:
```bash
python -m uno_ataraxos.train_blueprint \
  --total-steps 100000000 \
  --run-dir /kaggle/working/runs/blueprint_100m \
  --resume /kaggle/input/uno-ataraxos-ckpt/latest.pt
```

### B. Google Colab Drive Persistence
Mount Google Drive at the start of your notebook:
```python
from google.colab import drive
drive.mount('/content/drive')
```
Run with `--run-dir /content/drive/MyDrive/uno_ataraxos/runs/blueprint_100m`. All checkpoints and `metrics.jsonl` are saved directly to your Google Drive. If disconnected, re-run with `--resume /content/drive/MyDrive/uno_ataraxos/runs/blueprint_100m/latest.pt`.

---

## 4. Expected Performance & Milestones

- **Rollout Throughput:** 512 parallel environments stepping at > 5,000,000 SPS in C++.
- **Training Throughput:**
  - NVIDIA A100 / L4: > 8,000 samples/sec (< 125 ms/step) $\to$ 100M steps in ~3.5 hours.
  - NVIDIA T4: > 4,500 samples/sec (< 220 ms/step) $\to$ 100M steps in ~6 hours.
  - Apple Silicon M5: > 5,300 samples/sec (< 190 ms/step) $\to$ 100M steps in ~5 hours.
- **Milestones:**
  - **Stage 1 (0M – 30M steps):** Uniform Legal Policy Magnet ($\alpha_{\text{mag}}: 0.10 \to 0.015$). Voluntary draw rate drops to < 1%.
  - **Stage 2 (30M – 100M steps):** EMAgnet ($\alpha_{\text{mag}}: 0.04 \to 0.01$) + Dual-Anchor Uniform Floor ($\alpha_{\text{unif\_floor}}: 0.006 \to 0.002$). Wins vs `ExpertHeuristicBot` exceed > 65%.
