# Research & Technical Architecture: Ataraxos-UNO

**Deep-dive technical report on equilibrium learning, autoregressive belief modeling, and risk-averse portfolio search in 1v1 UNO.**

---

## 1. Game Theory Formulation & Complexity

Classic 108-card 1v1 UNO with same-type draw stacking ([`RULES.md`](RULES.md)) is a **two-player, zero-sum, imperfect-information Markov game**:

$$\Gamma = \langle \mathcal{N}, \mathcal{S}, (\mathcal{A}_i)_{i \in \mathcal{N}}, \mathcal{P}, (\mathcal{R}_i)_{i \in \mathcal{N}}, (\mathcal{I}_i)_{i \in \mathcal{N}} \rangle$$

- **Imperfect Information:** Player $i$ observes their own hand $\mathcal{H}_i$, the top card of the discard pile $c_{\text{top}}$, the active declared color, the penalty stack count, and public move history. The opponent's hand $\mathcal{H}_{-i}$ and the contents of the draw pile $\mathcal{D}$ are hidden states.
- **Non-Alternating Action Chains:** 16 extra-turn cards (8 `Skip`s and 8 `Reverse`s, where `Reverse = Skip` in 1v1) grant consecutive moves to the same player. This allows shedding 4–6 cards in a single uninterrupted turn, violating standard alternating game-tree assumptions.
- **Same-Type Draw Stacking:** Only 8 `+2`s and 4 `+4`s exist in the finite 108-card deck. Players can counter an incoming penalty with a card of the same type (any color `+2` on `+2`, or any `+4` on `+4`), escalating the penalty to $+4, +6, +8, \dots$.
- **Strategic Voluntary Drawing:** Players may voluntarily draw from the deck even when holding playable cards. This allows hoarding Wilds and `+4` closers for decisive endgames.
- **Finite-Deck Reshuffle Provenance:** When the draw pile empties, all discards except the top card are reshuffled. Tracking which cards were in the discard pile at reshuffle time yields exact mathematical bounds on what the opponent could have drawn.

---

## 2. End-to-End System Architecture

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
│  4. Risk-Averse Portfolio Search (Server & Client-Side WebGPU ONNX)      │
│     • 3-style opponent rollouts: 45% Blueprint / 35% Aggressive / 20%    │
│       Unpredictable + Entropic Soft-Minimax (γ = 4.0)                    │
│     • Belief tempering + Empirical-Bayes paired-world shrinkage + MMD    │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Neural Network Architectures

### 3.1 Actor Policy & WDL Value Network (`UnoActorNet` — 2.24M Parameters)
Replaces quadratic self-attention over fixed card tokens with a specialized hybrid backbone:
1. **Dense Card-Table GEMM Encoder:** Projects the $(54 \times 7)$ card feature table (my hand, discard counts, known played, recycled pool) and 10 global state features into a 256-dimensional state embedding.
2. **Parallel Masked Dilated-Conv History Encoder:** Processes the last 48 moves across dilation rates $d \in \{1, 2, 4\}$ to capture multi-turn action chains and tempo shifts into 128 dimensions.
3. **State-Conditioned 4-Query Cross-Attention:** Uses 4 learned queries conditioned on the state to attend over the 49 history tokens (1 explicit null token + 48 history events). The null token guarantees zero NaNs on turn 0 when history is empty.
4. **Pre-Norm SwiGLU ResMLP Trunk:** 4 deep residual blocks (512 dimensions, Swish-gated linear units) with RMSNorm.
5. **Dual Output Heads:**
   - **Policy Head:** 61 masked logits (40 colored numbers, 12 colored action cards, 4 declared-color Wilds, 4 declared-color Wild `+4`s, 1 draw action).
   - **WDL Head:** Auxiliary 3-class distribution (Win / Draw / Loss) predicting game outcomes.

### 3.2 Full-Information Critic Network (`UnoQCriticNet` — 2.11M Parameters)
Used exclusively during self-play training for low-variance variance-reduced policy optimization (VRPO):
- Receives the actor's imperfect-information state **plus** the ground-truth opponent hand vector $(54 \times 1)$.
- Outputs 61 state-action values $Q(s, a)$ evaluated under full information.

### 3.3 Autoregressive Generative Belief Network (`UnoBeliefNet` — 2.38M Parameters)
Models the posterior probability over the opponent's hidden hand:

$$p(\mathcal{H}_{-i} \mid s_i)$$

- Decomposes the hidden hand into a **canonical descending multiset sequence**: $c_0 \ge c_1 \ge \dots \ge c_{k-1}$ over card IDs $0 \dots 53$.
- **Zero-Dead-End Hard Masking:** At each autoregressive step, masks out any card that is:
  1. Mathematically impossible based on card counts and reshuffle provenance.
  2. Strictly greater than the previous generated card ($c_t \le c_{t-1}$), enforcing canonical multiset ordering and eliminating permutations.
- Incorporates **15% history token dropout** during training to prevent the model from overfitting to pure self-play trajectory artifacts and ensure robustness against human play styles.

---

## 4. Reinforcement Learning: VRPO + EMAgnet

Self-play training is driven by **Variance-Reduced Policy Optimization (VRPO)** combined with **Dynamically Damped Exponential Moving Average Regularization (EMAgnet)** across 100M environment steps (~2.45M complete games):

### 4.1 Expected SARSA($\lambda$) for Non-Alternating Turns
Standard TD-learning assumes alternating turns ($s \to s'$ is controlled by the opponent). Because UNO allows `Skip` and `Reverse` chains, the target computation tracks player continuity:

$$Q^{\text{target}}(s_t, a_t) = r_t + \gamma \cdot \left[ \mathbb{I}(p_{t+1} = p_t) V(s_{t+1}) - \mathbb{I}(p_{t+1} \neq p_t) V(s_{t+1}) \right]$$

### 4.2 Two-Stage Magnet Regularization
To prevent policy collapse and ensure convergence to an approximate Nash equilibrium:
- **Stage 1 (0M – 30M steps):** Uniform Legal Policy Magnet ($\alpha_{\text{mag}}: 0.10 \to 0.015$) encourages exploration across all legal moves. Voluntary draw rate drops from random (~30%) to < 1%.
- **Stage 2 (30M – 100M steps):** Dual-Anchor Regularization:
  - **EMAgnet Reference:** Policy is regularized toward an exponential moving average of historical parameters ($\alpha_{\text{mag}}: 0.04 \to 0.01$).
  - **Uniform Floor Anchor:** Maintains a minimum entropy floor ($\alpha_{\text{floor}}: 0.006 \to 0.002$) preventing deterministic traps.

### 4.3 Exact 24-Fold S4 Color-Permutation Symmetry
The rules of UNO are invariant under any permutation of the 4 card colors ($\text{Red}, \text{Green}, \text{Blue}, \text{Yellow}$). Every environment step applies a random permutation $\sigma \in \mathcal{S}_4$ to observations, actions, and historical tokens, multiplying effective sample efficiency by **$24\times$**.

---

## 5. Benchmarks & Key Findings

All head-to-head evaluations use **mirrored duplicate decks** (each shuffled 108-card deck is played twice with swapped seats) to cancel out first-order deal variance.

### 5.1 Blueprint Strength (100M Steps, Zero Search, 1.8 ms/move)

| Opponent | Duplicate Win Rate | Games Tested | Notes |
| :--- | :---: | :---: | :--- |
| `RandomBot` | **98.6%** | 500 decks (1,000 games) | Plays uniform legal moves |
| `10M-Step Self Snapshot` | **77.3%** | 250 decks (500 games) | Strong monotone self-play improvement |
| `GreedyBot` | **72.3%** | 250 decks (500 games) | Never voluntarily draws, sheds highest card |
| `ExpertHeuristicBot` | **64.5%** | 250 decks (500 games) | Hand-crafted Skip/Reverse chainer & stack counter |

### 5.2 Emergent Wild-Hoarding Discovery
A 15,850-turn diagnostic of the 100M Blueprint revealed emergent high-level strategy without any handcrafted rewards:
- **Voluntary Draw Rate:** The policy chooses to draw while holding legal cards **12.45%** of the time.
- **Card-Holding Context at Draw Time:**
  - **89.5%** of voluntary draws occur when *only* `Wild` or `Wild +4` is playable.
  - **4.4%** occur when *only* `+2` is playable.
  - **Total deliberate hoarding:** **93.9%** of voluntary draws are executed to save Wilds or penalty-stack counters for critical endgame situations.

---

## 6. Solving Rollout Overfitting: 2,400-Game Controlled Ablation

### 6.1 The Rollout Overfitting Problem
Standard single-opponent Ataraxos search (using pure Blueprint rollouts during MMD search) beat the Blueprint head-to-head (**52.0%**), but its performance degraded against out-of-distribution (OOD) opponents (`ExpertHeuristicBot` and `Exploiter`).

Because rollouts assumed the opponent plays like the Blueprint, the search agent expected the opponent to hoard Wilds and respect stack counters. When facing aggressive heuristic opponents who dump cards immediately, the single-policy search made fragile, exploitable assumptions.

### 6.2 The Solution: 3-Style Portfolio Rollouts + Entropic Soft-Minimax
We introduced a three-style rollout portfolio combined with entropic soft-minimax value aggregation ($\gamma = 4.0$):
1. **45% Blueprint Style:** Equilibrium play with strategic voluntary hoarding.
2. **35% Aggressive Style:** Greedy card shedding, immediate action plays, and zero voluntary draws.
3. **20% Unpredictable Style:** High-temperature stochastic exploration.

### 6.3 2,400-Game Controlled Ablation Matrix
Tested across **2,400 mirrored duplicate games** (exactly 200 games per cell on identical deck seeds):

| Agent Configuration | vs 100M Blueprint (200g) | vs ExpertBot (200g) | vs 15M Exploiter (200g) | **3-Axis Avg** |
| :--- | :---: | :---: | :---: | :---: |
| Pure 100M Blueprint (Baseline) | 50.0% *(100–100)* | 61.0% *(122–78)* | 55.0% *(110–90)* | 55.3% |
| **Risk-Averse Portfolio (Champion)** ⭐ | **53.0%** *(106–94)* | **66.0%** *(132–68)* | **56.0%** *(112–88)* | **58.3% (+3.0%)** |
| Adaptive Bayesian (Tier-1) | **53.0%** *(106–94)* | 65.5% *(131–69)* | 55.0% *(110–90)* | 57.8% (+2.5%) |
| Adaptive Bayesian Hybrid (+ PDCFR+) | 50.0% *(100–100)* | 64.5% *(129–71)* | **57.0%** *(114–86)* | 57.2% (+1.9%) |

The Risk-Averse Portfolio achieves the highest overall win rate (**58.3%**), simultaneously beating the self-play Blueprint (**53.0%**) and crushing the Expert bot (**66.0%**), proving that portfolio rollouts eliminate search fragility.

---

## 7. Performance & Parity Verifications

| Metric | Ground Truth Benchmark | Verified Result | Parity Margin |
| :--- | :--- | :--- | :--- |
| **C++ Engine Throughput** | Multi-threaded `VecUnoEnv` (512 envs) | **7,797,175 steps/sec** | 40,408 games/sec (37.3 MB RAM) |
| **C++ vs `uno-gym` Parity** | 10,000 games (1.90M steps, 9,127 reshuffles) | **100% Identical Trajectory** | 0 mismatches across all 61 legal mask bits |
| **Belief Calibration** | Exact multivariate hypergeometric prior | **+1.47 nats/hand** train / **+1.19** val | 0 violations across 50,048 positions |
| **PyTorch vs ONNX Parity** | 100 random game states | **100/100 argmax match** | Max diff: $1.97 \times 10^{-6}$ (Actor), $1.04 \times 10^{-6}$ (Critic) |
| **Browser JS vs C++ Parity** | 4,000 full game steps | **4,000/4,000 exact match** | Max diff: $0.0000$ on features, exact on masks |

---

## 8. Academic References & Foundations

- **Ataraxos:** Sokota et al. (*Nature*, 2026). *Solving Imperfect Information Games via Update-Equivalence Search and Regularized Equilibrium Play.*
- **VRPO:** Fan & Farina (*MIT*, 2026). *Variance-Reduced Policy Optimization for Imperfect-Information Games with Non-Alternating Steps.*
- **EMAgnet:** Maidment et al. (2026). *Parameter-Space Magnet Regularization for Last-Iterate Convergence in Zero-Sum Games.*
- **PDCFR+:** Xu et al. (*AAAI*, 2026). *Predictive Discounted Counterfactual Regret Minimization for Endgame Subgame Solving.*
