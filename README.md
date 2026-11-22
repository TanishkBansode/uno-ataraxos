# Ataraxos-UNO

**A 1v1 UNO AI with a 66% win rate against expert heuristics, built with VRPO, EMAgnet, an Autoregressive Belief Network, and Risk-Averse Portfolio Search — running directly in your browser via ONNX Runtime Web (WebGPU/WASM).**

<p align="center">
  <a href="https://tanishkbansode.github.io/uno-ataraxos/">
    <img src="https://img.shields.io/badge/▶_Play_Live_in_Browser-GitHub_Pages-00C853?style=for-the-badge" alt="Play Live"/>
  </a>
  <img src="https://img.shields.io/badge/Pytest_Suite-73%2F73_Passing-blue?style=for-the-badge" alt="Tests"/>
  <img src="https://img.shields.io/badge/C%2B%2B17_Engine-7.8M_Steps%2Fsec-orange?style=for-the-badge" alt="Simulator Speed"/>
  <img src="https://img.shields.io/badge/Browser_Models-26.2_MB_Total-purple?style=for-the-badge" alt="Model Size"/>
</p>

<p align="center">
  <img src="docs/demo_gameplay.png" alt="Ataraxos-UNO Live Gameplay with Belief X-Ray" width="880"/>
</p>

> **[Play 1v1 against Ataraxos-UNO right now →](https://tanishkbansode.github.io/uno-ataraxos/)**  
> Toggle **👁 X-RAY** during a match to see the live probability distribution the AI infers over your hidden cards.

---

## 1v1 UNO Complexity

Classic 108-card 1v1 UNO with same-type draw stacking (`RULES.md`) is a **two-player, zero-sum, imperfect-information game**:
- **`Reverse = Skip` combo chains:** 16 extra-turn cards let a strong player shed 4–6 cards in one uninterrupted turn.
- **Same-type `+2` / `+4` stacking wars:** Only 8 `+2`s and 4 `+4`s exist in the finite deck — exact card counting and stack-insurance management decide the outcome.
- **Strategic voluntary drawing:** Players may draw even while holding playable cards to hoard Wild/`+4` closers.
- **Finite-deck reshuffle provenance:** Tracking which unseen cards were freshly recycled from the discard pile yields exact mathematical deductions about the opponent's hand.

Built with PyTorch and C++17, Ataraxos-UNO packages imperfect-information game algorithms (**Ataraxos**, **VRPO**, **EMAgnet**, **PDCFR+**) into a **2.24M-parameter** system that runs locally and inside modern web browsers.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│  1. C++17 Vectorized Engine (VecUnoEnv)                                  │
│     • 7.80M steps/sec on Apple Silicon • Exact 10,000-game parity vs uno-gym  │
│     • Tracks 48-move history + exact post-reshuffle card provenance      │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  2. Self-Play Blueprint (100M Steps / ~2.45M Games)                      │
│     • Hybrid SwiGLU ResMLP + Dilated-Conv History + 4-Query Cross-Attn   │
│     • VRPO Q-Boosting (Expected SARSA(λ) for non-alternating turns)      │
│     • Dynamically Damped EMAgnet + 24× S4 Color-Permutation Symmetry     │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  3. Hard-Masked Autoregressive Belief Network (20M Positions)            │
│     • Canonical descending multiset decoder with zero-dead-end masking   │
│     • 15% history dropout + diverse opponents for human robustness       │
│     • +1.47 nats/hand gain (4.35× sharper than exact Hypergeom prior)    │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  4. Risk-Averse Portfolio Search (Server & Browser-Native WebGPU ONNX)   │
│     • 3-style opponent rollouts: 45% Blueprint / 35% Aggressive / 20%    │
│       Unpredictable + Entropic Soft-Minimax (γ = 4.0)                    │
│     • Belief tempering + Empirical-Bayes paired-world shrinkage + MMD    │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Benchmarks & Key Research Findings

All head-to-head evaluations use **mirrored duplicate decks** (each shuffled 108-card deck is played twice with swapped seats) to eliminate first-order deal variance.

### 1. Blueprint Strength (100M Steps, Zero Search, `1.8 ms/move`)

| Opponent | Duplicate Win Rate | Notes |
| :--- | :---: | :--- |
| `RandomBot` | **98.4% – 98.8%** | Uniform legal moves |
| `10M-Step Self Snapshot` | **74.7% – 79.8%** | Strong monotone self-play climb |
| `GreedyBot` | **71.4% – 73.2%** | Never voluntarily draws, prioritizes actions |
| `ExpertHeuristicBot` | **63.6% – 65.5%** | Hand-crafted Skip/Reverse chainer & stack-counter |

**Emergent Wild-Hoarding Discovery (15,850-turn diagnostic):** Without any reward shaping, the 100M Blueprint learned a **12.45%** voluntary draw rate — of which **89.5%** occurs when *only* `Wild`/`Wild +4` is playable and **4.4%** when *only* `+2` is playable (**93.9%** deliberate Wild/stack-insurance hoarding).

---

### 2. Solving Rollout Overfitting — 2,400-Game Controlled Ablation

Standard single-opponent Ataraxos search (`Blueprint`-only rollouts) beat the Blueprint head-to-head (**52.0%**) but degraded against out-of-distribution opponents (`Expert` and `Exploiter`) because rollouts assumed every opponent hoards Wilds like the Blueprint.

Adding **3-Style Portfolio Rollouts + Entropic Soft-Minimax ($\gamma = 4.0$) + Belief Tempering** eliminated rollout bias across **2,400 mirrored duplicate games** (exactly `200` games per cell on identical deck seeds):

| Agent | vs 100M Blueprint (200g) | vs ExpertBot (200g) | vs 15M Exploiter (200g) | **3-Axis Avg** |
| :--- | :---: | :---: | :---: | :---: |
| Pure 100M Blueprint | 50.0% *(100–100)* | 61.0% *(122–78)* | 55.0% *(110–90)* | 55.3% |
| **Risk-Averse Portfolio (Champion)** ⭐ | **53.0%** *(106–94)* | **66.0%** *(132–68)* | **56.0%** *(112–88)* | **58.3% (+3.0%)** |
| Adaptive Bayesian (Tier-1) | **53.0%** *(106–94)* | 65.5% *(131–69)* | 55.0% *(110–90)* | 57.8% (+2.5%) |
| Adaptive Bayesian Hybrid (+ PDCFR+) | 50.0% *(100–100)* | 64.5% *(129–71)* | **57.0%** *(114–86)* | 57.2% (+1.9%) |

---

### 3. Parity, Speed & Model Footprint

| Component | Verified Result |
| :--- | :--- |
| **C++17 `VecUnoEnv` Throughput** | `7,797,175 steps/sec` (`40,408 games/sec`, `37.3 MB` RAM on Apple Silicon) |
| **C++ vs [`uno-gym`](https://github.com/tanishkbansode/uno-gym) Parity** | `10,000 / 10,000` games (`1,903,882` steps, `9,127` reshuffles) — **identical trajectory parity** |
| **Belief Network Calibration** | `+1.47` train / `+1.19` val nats/hand over exact Hypergeometric prior (`0` violations / `50,048` steps) |
| **ONNX vs PyTorch Parity** | `100 / 100` argmax match, max diff `1.97e-6` (Actor) & `1.04e-6` (Critic) |
| **Browser JS vs C++ Parity** | `4,000 / 4,000` steps exact (`0.0` diff) for both feature encoder and fast rollout engine |
| **Total Browser Model Size** | **26.23 MB** (`actor` 8.58 MB, `critic` 8.38 MB, `belief_enc` 2.62 MB, `belief_dec` 6.65 MB) |

---

## Quickstart

### Play Online (Zero Setup)
Open **[https://tanishkbansode.github.io/uno-ataraxos/](https://tanishkbansode.github.io/uno-ataraxos/)** — all 4 ONNX models download once (~26 MB), cache locally, and run Risk-Averse Portfolio Search + Belief X-Ray directly in your browser.

### Run Locally (Web UI + Full C++/PyTorch Daemon or Terminal CLI)
```bash
git clone https://github.com/TanishkBansode/uno-ataraxos.git
cd uno-ataraxos
pip install -e .

# Option 1: Launch local Web UI + full Risk-Averse Search server at http://localhost:8000
python scripts/serve.py

# Option 2: Play in terminal with live Belief X-Ray & move equities
python scripts/play.py \
  --blueprint runs/blueprint_100m/latest.pt \
  --belief runs/belief_20m/best.pt \
  --xray
```

### Run Verification Suite
```bash
pytest tests/ -q                           # 73 Python + C++ unit, parity, RL & API tests
node tests/test_js_features_parity.mjs     # 4,000-step JS vs C++ feature parity
node tests/test_fast_rollout_parity.mjs    # 4,000-step JS vs C++ rollout parity
```

### Train From Scratch
See **[`docs/TRAINING.md`](docs/TRAINING.md)** for step-by-step instructions to train the 100M Blueprint, 20M Belief Network, and Neural Exploiter Red-Team.

---

## Repository Layout

```text
uno-ataraxos/
├── RULES.md                     # Authoritative 108-card 1v1 UNO rules specification
├── cpp/                         # C++17 multi-threaded VecUnoEnv (7.8M SPS) + pybind11 bindings
├── uno_ataraxos/
│   ├── networks.py              # Hybrid ResMLP + Dilated-Conv + Cross-Attention Actor & Q-Critic
│   ├── rl_math.py               # Safe masked reverse-KL, VRPO Expected SARSA(λ), EMAgnet utils
│   ├── symmetry.py              # Exact 24-fold S4 color-permutation engine
│   ├── belief.py                # Zero-dead-end canonical descending Autoregressive Belief Net
│   ├── search.py                # Risk-Averse Portfolio Search + Empirical-Bayes MMD update
│   ├── endgame_cfr.py           # Depth-limited 3-ply PDCFR+ endgame subgame solver
│   ├── opponent_model.py        # Online Bayesian opponent style & surprisal tracker
│   ├── agent.py                 # Unified tournament-fair agent (VecUnoEnv + uno-gym adapter)
│   └── train_*.py               # Blueprint, Belief, and Neural Exploiter training loops
├── web/                         # Static WebGPU/WASM web app deployed to GitHub Pages
├── scripts/                     # serve.py, play.py, train.py, export_onnx.py, and tuning/evaluation
├── tests/                       # 73 pytest tests + JS/C++ bit-for-bit parity tests
└── docs/                        # Training guide & gameplay screenshot
```

---

## Research Foundations

- **Ataraxos** — Sokota et al. (*Nature*, 2026): dynamically damped self-play, dropout-regularized autoregressive belief networks, and update-equivalence MMD search.
- **VRPO** — Fan & Farina (*MIT*, 2026): Q-boosting with centralized full-information critics and Expected SARSA($\lambda$) for low-variance equilibrium RL.
- **EMAgnet** — Maidment et al. (2026): continuous parameter-space EMA magnet regularization for last-iterate Nash convergence.
- **PDCFR+** — Xu et al. (*AAAI*, 2026): fast predictive discounted CFR+ for localized imperfect-information endgame subgames.

## License

MIT License — see [`LICENSE`](LICENSE).
