# Training Guide

`uno-ataraxos` is cross-platform and optimized for both **Apple Silicon (MPS)** and **NVIDIA CUDA**.

---

## 1. Quickstart

To launch self-play training with automatic hardware detection and mixed precision:
```bash
python scripts/train.py
```

Or manually configure your run:
```bash
# 1. Install dependencies & build C++ extension in-place
pip install -r requirements.txt
python setup.py build_ext --inplace

# 2. Run test suite to verify environment health
pytest tests/ -q

# 3. Launch 100M-Step Blueprint Training (Stage A)
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
  --run-dir runs/blueprint_100m
```

---

## 2. End-to-End Pipeline (Stages A $\to$ B $\to$ C $\to$ D)

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
  --blueprint-ckpt runs/blueprint_100m/best.pt \
  --run-dir runs/belief_20m
```

### Stage C: 15M-Step VRPO Neural Exploiter Red-Team
Measures the empirical Nash exploitability of the frozen Blueprint by training a dedicated adversarial agent (pure PPO + entropy floor) strictly on the frozen target across 500+ vectorized parallel environments:
```bash
python -m uno_ataraxos.train_exploiter \
  --target-ckpt runs/blueprint_100m/best.pt \
  --total-steps 15000000 \
  --num-envs 512 \
  --rollout-len 64 \
  --eval-every-steps 1000000 \
  --eval-decks 250 \
  --run-dir runs/exploiter_20m
```

### Stage D: Search Hyperparameter Sweep & Defense Verification
Runs mirrored duplicate tournaments against the frozen Blueprint to find the Pareto-optimal MMD parameters ($\eta, \tau, \text{leaf\_eval}$), saves `search_config.json`, and verifies search defends against the exploiter:
```bash
python scripts/tune_search.py \
  --blueprint runs/blueprint_100m/best.pt \
  --belief runs/belief_20m/best.pt \
  --exploiter runs/exploiter_20m/best.pt \
  --num-decks 100 \
  --out-config runs/search_config.json
```

---

## 3. Hardware Precision & Optimization Matrix

The trainer automatically detects your hardware and configures the optimal mixed-precision pipeline:

| Hardware | Precision | Scaler | `torch.compile` |
| :--- | :--- | :--- | :--- |
| **NVIDIA CUDA (Compute $\ge 8.0$)** | `torch.bfloat16` | None | Supported (`--compile`) |
| **NVIDIA CUDA (Compute $< 8.0$)** | `torch.float16` | `torch.amp.GradScaler("cuda")` | Optional |
| **Apple Silicon (MPS)** | `torch.bfloat16` | None | Disabled |
| **CPU Fallback** | `torch.float32` | None | Disabled |

- **Numerical Invariant:** Regardless of precision (`bf16` or `fp16`), all network outputs are explicitly cast to `float32`, and all VRPO Expected SARSA(λ), reverse KL divergence, policy entropy, and WDL calculations execute in full `float32`.
- **Checkpoint Portability:** Checkpoints automatically unwrap `_orig_mod` if compiled, allowing seamless transfer between CUDA and Apple Silicon MPS.

---

## 4. Checkpoint Resuming

Checkpoints (`latest.pt`, `best.pt`, and `snapshots/`) are saved periodically (every 15 minutes by default). To resume an interrupted run at any time, pass the `--resume` flag:

```bash
python -m uno_ataraxos.train_blueprint \
  --total-steps 100000000 \
  --run-dir runs/blueprint_100m \
  --resume runs/blueprint_100m/latest.pt
```

---

## 5. Expected Throughput & Milestones

- **Rollout Throughput:** 512 parallel environments stepping at > 5,000,000 SPS in C++.
- **Training Throughput:**
  - Modern CUDA GPUs: > 8,000 samples/sec (< 125 ms/step) $\to$ 100M steps in ~3.5 hours.
  - Apple Silicon (MPS): > 5,300 samples/sec (< 190 ms/step) $\to$ 100M steps in ~5 hours.
- **Milestones:**
  - **Stage 1 (0M – 30M steps):** Uniform Legal Policy Magnet ($\alpha_{\text{mag}}: 0.10 \to 0.015$). Voluntary draw rate drops to < 1%.
  - **Stage 2 (30M – 100M steps):** EMAgnet ($\alpha_{\text{mag}}: 0.04 \to 0.01$) + Dual-Anchor Uniform Floor ($\alpha_{\text{unif\_floor}}: 0.006 \to 0.002$). Wins vs `ExpertHeuristicBot` exceed > 65%.
