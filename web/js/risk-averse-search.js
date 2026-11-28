// Client-Side In-Browser Ataraxos Risk-Averse Portfolio Search Engine
// Implements exact Update-Equivalence Tabular MMD Search with:
// 1. 3-Style Opponent Portfolio: Blueprint (0.45), Aggressive No-VDraw (0.35), Noisy (0.20)
// 2. Soft-Minimax Risk-Averse Aggregation with gamma = 4.0
// 3. Empirical-Bayes signal/noise shrinkage filter
// 4. Closed-form MMD policy improvement with eta = 8.0, tau = 0.002, bp_floor = 0.03
// 5. Automatic Device-Aware World Budgeting (WebGPU: 192 worlds/10 plies, WASM: 48 worlds/6 plies)

import {
  ACTION_NAMES,
  CARD_NAMES,
  cardTrait,
  cardColor,
  TRAIT,
  actionRequiredCard,
} from "./ataraxos-features.js";
import { FastVecUnoEnv, FastUnoEnv, mulberry32, shuffleArray } from "./fast-rollout.js";

// Card IDs for Belief X-Ray diagnostics
const DRAW_2_CARD_IDS = [42, 45, 48, 51];
const WILD_DRAW_4_CARD_ID = 53;
const RED_CARD_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 40, 41, 42];
const GREEN_CARD_IDS = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 43, 44, 45];
const BLUE_CARD_IDS = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 46, 47, 48];
const YELLOW_CARD_IDS = [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 49, 50, 51];

export class RiskAverseSearchEngine {
  constructor(onnxEngine) {
    this.onnxEngine = onnxEngine;
    this.vecPool = new FastVecUnoEnv(64);
    this.adaptationState = {
      avgLatencyMs: 0,
      consecutiveSlowTurns: 0,
      downgradedToFastActor: false,
    };
  }

  /**
   * Sample K opponent hands from exact unseen card hypergeometric prior or belief model.
   */
  sampleOppHands(rootEnv, K, oppHandSize) {
    const oppP = 1 - rootEnv.currentPlayer;
    const cf = rootEnv.getCardFeatures(rootEnv.currentPlayer);
    // cf channel 2 is unseen_total
    const unseenPool = [];
    for (let c = 0; c < 54; c++) {
      const cnt = Math.round(cf[c * 7 + 2]);
      for (let i = 0; i < cnt; i++) {
        unseenPool.push(c);
      }
    }

    const sampledHands = new Array(K);
    const rng = mulberry32(Date.now() + 101);

    for (let k = 0; k < K; k++) {
      const counts = new Uint8Array(54);
      if (unseenPool.length <= oppHandSize) {
        for (const c of unseenPool) counts[c]++;
      } else {
        const poolCopy = unseenPool.slice();
        shuffleArray(poolCopy, rng);
        for (let i = 0; i < oppHandSize; i++) {
          counts[poolCopy[i]]++;
        }
      }
      sampledHands[k] = counts;
    }

    return sampledHands;
  }

  computeBeliefXray(sampledOppHands, oppHandSize) {
    const K = sampledOppHands.length;
    if (K === 0 || oppHandSize === 0) {
      return {
        p_plus_2: 0.0,
        p_plus_4: 0.0,
        p_red: 0.0,
        p_green: 0.0,
        p_blue: 0.0,
        p_yellow: 0.0,
        opp_hand_size: oppHandSize,
      };
    }

    let p2Count = 0;
    let p4Count = 0;
    let rCount = 0;
    let gCount = 0;
    let bCount = 0;
    let yCount = 0;

    for (let k = 0; k < K; k++) {
      const h = sampledOppHands[k];
      let hasP2 = false;
      for (const id of DRAW_2_CARD_IDS) {
        if (h[id] > 0) { hasP2 = true; break; }
      }
      if (hasP2) p2Count++;
      if (h[WILD_DRAW_4_CARD_ID] > 0) p4Count++;

      let hasR = false, hasG = false, hasB = false, hasY = false;
      for (const id of RED_CARD_IDS) { if (h[id] > 0) { hasR = true; break; } }
      for (const id of GREEN_CARD_IDS) { if (h[id] > 0) { hasG = true; break; } }
      for (const id of BLUE_CARD_IDS) { if (h[id] > 0) { hasB = true; break; } }
      for (const id of YELLOW_CARD_IDS) { if (h[id] > 0) { hasY = true; break; } }

      if (hasR) rCount++;
      if (hasG) gCount++;
      if (hasB) bCount++;
      if (hasY) yCount++;
    }

    return {
      p_plus_2: p2Count / K,
      p_plus_4: p4Count / K,
      p_red: rCount / K,
      p_green: gCount / K,
      p_blue: bCount / K,
      p_yellow: yCount / K,
      opp_hand_size: oppHandSize,
    };
  }

  async _resolveBeliefXray(rootEnv, rootCf, rootGf, rootHt, oppHandSize, sampledHands = null) {
    if (
      this.onnxEngine.beliefLoaded &&
      this.onnxEngine.beliefEncSession &&
      this.onnxEngine.beliefDecSession &&
      oppHandSize > 0
    ) {
      try {
        const neuralXray = await this.onnxEngine.sampleBeliefXray({
          cardFeatures: rootCf,
          globalFeatures: rootGf,
          historyTokens: rootHt,
          oppHandSize,
        });
        if (neuralXray) return neuralXray;
      } catch (_) {}
    }
    const hands = sampledHands || this.sampleOppHands(rootEnv, 16, oppHandSize);
    return this.computeBeliefXray(hands, oppHandSize);
  }

  /**
   * Run in-browser Milestone 8 Risk-Averse Portfolio Search.
   */
  async search({ rootEnv, deterministic = true }) {
    const t0 = performance.now();
    const rootPlayer = rootEnv.currentPlayer;
    const legalMask = rootEnv.getLegalMask(rootPlayer);

    const legalActions = [];
    for (let i = 0; i < 61; i++) {
      if (legalMask[i] > 0.5) legalActions.push(i);
    }
    const numA = legalActions.length;

    const oppHandSize = rootEnv.hands[1 - rootPlayer].length;

    // 1. Root Blueprint Prediction (also gives accurate win_prob on 0/1-move turns)
    const rootCf = rootEnv.getCardFeatures(rootPlayer);
    const rootGf = rootEnv.getGlobalFeatures(rootPlayer);
    const rootHt = rootEnv.getHistoryTokens(rootPlayer, true);

    const rootFeeds = {
      card_features: new ort.Tensor("float32", rootCf, [1, 54, 7]),
      global_features: new ort.Tensor("float32", rootGf, [1, 10]),
      history_tokens: new ort.Tensor("int64", rootHt, [1, 48, 5]),
      legal_mask: new ort.Tensor("float32", legalMask, [1, 61]),
    };

    const rootResults = await this.onnxEngine.session.run(rootFeeds);
    const bpProbsRaw = Array.from(rootResults.action_probs.data);
    const wdlRoot = Array.from(rootResults.wdl_probs.data);
    const rootWinProb = (wdlRoot[2] || 0) + 0.5 * (wdlRoot[1] || 0);

    // Early exit if 0 or 1 legal action (skip rollouts, keep real win_prob & belief_xray)
    if (numA <= 1) {
      const chosenAct = numA === 1 ? legalActions[0] : 60;
      const searchProbs = new Float32Array(61);
      searchProbs[chosenAct] = 1.0;
      const earlyXray = await this._resolveBeliefXray(rootEnv, rootCf, rootGf, rootHt, oppHandSize);
      return {
        action_id: chosenAct,
        action: ACTION_NAMES[chosenAct] || "draw",
        action_key: ACTION_NAMES[chosenAct] || "draw",
        action_probs: Array.from(searchProbs),
        win_prob: rootWinProb,
        latency_ms: performance.now() - t0,
        mode_used: "risk_averse_search",
        technique_label: "Risk-Averse Portfolio Search",
        belief_xray: earlyXray,
      };
    }

    // Fallback if downgraded to fast actor due to past high latency
    if (this.adaptationState.downgradedToFastActor) {
      let bestA = legalActions[0];
      let maxP = -1;
      for (const a of legalActions) {
        if (bpProbsRaw[a] > maxP) {
          maxP = bpProbsRaw[a];
          bestA = a;
        }
      }
      const fastXray = await this._resolveBeliefXray(rootEnv, rootCf, rootGf, rootHt, oppHandSize);
      return {
        action_id: bestA,
        action: ACTION_NAMES[bestA] || "draw",
        action_key: ACTION_NAMES[bestA] || "draw",
        action_probs: bpProbsRaw,
        win_prob: rootWinProb,
        latency_ms: performance.now() - t0,
        mode_used: "risk_averse_search",
        technique_label: "Risk-Averse Search (Fast)",
        belief_xray: fastXray,
      };
    }

    // Apply bp_floor = 0.03
    const bpFloor = 0.03;
    const numLegal = legalActions.length;
    const unifP = 1.0 / numLegal;
    const bpProbs = new Float32Array(61);
    for (const a of legalActions) {
      bpProbs[a] = (1.0 - bpFloor) * bpProbsRaw[a] + bpFloor * unifP;
    }

    // 2. Budget K_eff and rollout depth by device
    const isWebGPU = this.onnxEngine.device === "webgpu";
    const maxTotalWorlds = isWebGPU ? 192 : 48;
    const maxDepth = isWebGPU ? 10 : 6;
    const K_eff = Math.min(32, Math.max(8, Math.floor(maxTotalWorlds / numA)));

    // 3. Sample Opponent Worlds & Assign Styles
    const sampledOppHands = this.sampleOppHands(rootEnv, K_eff, oppHandSize);
    const beliefXray = await this._resolveBeliefXray(
      rootEnv,
      rootCf,
      rootGf,
      rootHt,
      oppHandSize,
      sampledOppHands,
    );

    // Portfolio weights: 0.45 Blueprint (Style 0), 0.35 Aggressive (Style 1), 0.20 Noisy (Style 2)
    const pw = [0.45, 0.35, 0.20];
    const k0 = Math.max(2, Math.round(pw[0] * K_eff));
    const k1 = Math.max(2, Math.round(pw[1] * K_eff));
    const k2 = Math.max(2, K_eff - k0 - k1);

    const worldStyles = new Int32Array(K_eff);
    for (let k = 0; k < K_eff; k++) {
      if (k < k0) worldStyles[k] = 0;
      else if (k < k0 + k1) worldStyles[k] = 1;
      else worldStyles[k] = 2;
    }

    const totalSlots = numA * K_eff;
    const slotStyles = new Int32Array(totalSlots);
    for (let aIdx = 0; aIdx < numA; aIdx++) {
      for (let k = 0; k < K_eff; k++) {
        slotStyles[aIdx * K_eff + k] = worldStyles[k];
      }
    }

    // 4. Load Worlds and Step Root Action
    const { rewards: rewRoot, dones: donesRoot } = this.vecPool.loadWorldsAndStepRoot(
      rootEnv,
      sampledOppHands,
      legalActions,
      Date.now() % 100000,
    );

    const rootValues = new Float32Array(totalSlots);
    for (let s = 0; s < totalSlots; s++) {
      if (donesRoot[s]) {
        rootValues[s] = rewRoot[s];
      }
    }

    let activeSlots = [];
    for (let s = 0; s < totalSlots; s++) {
      if (!donesRoot[s]) activeSlots.push(s);
    }

    // 5. Fast Rollout Loop (plies 1 .. maxDepth)
    for (let ply = 1; ply < maxDepth; ply++) {
      if (activeSlots.length === 0) break;

      const actIndices = Int32Array.from(activeSlots);
      const B = actIndices.length;

      // Extract batch tensors
      const { cf, gf, ht, mask, currentP } = this.vecPool.getTensorsForIndices(actIndices, rootPlayer);

      const stepFeeds = {
        card_features: new ort.Tensor("float32", cf, [B, 54, 7]),
        global_features: new ort.Tensor("float32", gf, [B, 10]),
        history_tokens: new ort.Tensor("int64", ht, [B, 48, 5]),
        legal_mask: new ort.Tensor("float32", mask, [B, 61]),
      };

      const stepResults = await this.onnxEngine.session.run(stepFeeds);
      const stepActionProbs = stepResults.action_probs.data;

      // Action selection per environment according to portfolio style
      const stepActions = new Int32Array(B);
      for (let b = 0; b < B; b++) {
        const slot = actIndices[b];
        const isOpp = currentP[b] !== rootPlayer;
        const offset = b * 61;

        if (!isOpp) {
          // Root player: greedy argmax
          let bestA = 60;
          let maxP = -1;
          for (let a = 0; a < 61; a++) {
            if (mask[offset + a] > 0.5 && stepActionProbs[offset + a] > maxP) {
              maxP = stepActionProbs[offset + a];
              bestA = a;
            }
          }
          stepActions[b] = bestA;
        } else {
          const style = slotStyles[slot];
          if (style === 0) {
            // Style 0: Blueprint argmax
            let bestA = 60;
            let maxP = -1;
            for (let a = 0; a < 61; a++) {
              if (mask[offset + a] > 0.5 && stepActionProbs[offset + a] > maxP) {
                maxP = stepActionProbs[offset + a];
                bestA = a;
              }
            }
            stepActions[b] = bestA;
          } else if (style === 1) {
            // Style 1: Aggressive No-VDraw (avoid draw 60 if playable cards exist)
            let hasPlayable = false;
            for (let a = 0; a < 60; a++) {
              if (mask[offset + a] > 0.5) { hasPlayable = true; break; }
            }
            let bestA = 60;
            let maxP = -1;
            for (let a = 0; a < 61; a++) {
              if (hasPlayable && a === 60) continue;
              if (mask[offset + a] > 0.5 && stepActionProbs[offset + a] > maxP) {
                maxP = stepActionProbs[offset + a];
                bestA = a;
              }
            }
            stepActions[b] = bestA;
          } else {
            // Style 2: Noisy (softmax with temp = 2.0)
            let sumP = 0;
            const expP = new Float32Array(61);
            for (let a = 0; a < 61; a++) {
              if (mask[offset + a] > 0.5) {
                expP[a] = Math.pow(Math.max(1e-6, stepActionProbs[offset + a]), 0.5); // temp = 2.0
                sumP += expP[a];
              }
            }
            let r = Math.random() * (sumP || 1.0);
            let chosen = 60;
            let cum = 0;
            for (let a = 0; a < 61; a++) {
              if (mask[offset + a] > 0.5) {
                cum += expP[a];
                if (r <= cum) {
                  chosen = a;
                  break;
                }
              }
            }
            stepActions[b] = chosen;
          }
        }
      }

      // Step environments in VecUnoEnv
      const { rewards: rewStep, dones: donesStep, actingPlayer: actPlStep } = this.vecPool.stepIndices(
        actIndices,
        stepActions,
      );

      const nextActive = [];
      for (let b = 0; b < B; b++) {
        const slot = actIndices[b];
        if (donesStep[b]) {
          const sign = actPlStep[b] === rootPlayer ? 1.0 : -1.0;
          rootValues[slot] = sign * rewStep[b];
        } else {
          nextActive.push(slot);
        }
      }
      activeSlots = nextActive;
    }

    // 6. Leaf Evaluation for Still-Active Leaves
    if (activeSlots.length > 0) {
      const leafIndices = Int32Array.from(activeSlots);
      const B = leafIndices.length;
      const { cf, gf, ht, mask, currentP } = this.vecPool.getTensorsForIndices(leafIndices, rootPlayer);

      const leafFeeds = {
        card_features: new ort.Tensor("float32", cf, [B, 54, 7]),
        global_features: new ort.Tensor("float32", gf, [B, 10]),
        history_tokens: new ort.Tensor("int64", ht, [B, 48, 5]),
        legal_mask: new ort.Tensor("float32", mask, [B, 61]),
      };

      const leafResults = await this.onnxEngine.session.run(leafFeeds);
      const leafWdl = leafResults.wdl_probs.data; // [B, 3]

      for (let b = 0; b < B; b++) {
        const slot = leafIndices[b];
        const pWin = leafWdl[b * 3 + 2];
        const pLoss = leafWdl[b * 3 + 0];
        const vExpected = pWin - pLoss; // In [-1, 1]
        const sign = currentP[b] === rootPlayer ? 1.0 : -1.0;
        rootValues[slot] = sign * vExpected;
      }
    }

    // 7. Soft-Minimax Risk-Averse Aggregation (gamma = 4.0) & Closed-Form MMD Update
    const riskGamma = 4.0;
    const eta = 8.0;
    const tau = 0.002;
    const ebShrink = 1.0;

    // Reshape rootValues to [numA, K_eff]
    const rootValues2D = new Array(numA);
    for (let aIdx = 0; aIdx < numA; aIdx++) {
      rootValues2D[aIdx] = new Float32Array(K_eff);
      for (let k = 0; k < K_eff; k++) {
        rootValues2D[aIdx][k] = rootValues[aIdx * K_eff + k];
      }
    }

    // Blueprint probabilities over legal actions normalized
    let bpLegalSum = 0;
    for (const a of legalActions) bpLegalSum += bpProbs[a];
    const bpProbsLegal = new Float32Array(numA);
    for (let i = 0; i < numA; i++) {
      bpProbsLegal[i] = bpProbs[legalActions[i]] / Math.max(1e-9, bpLegalSum);
    }

    // Compute Style-Conditional Returns Q(a, m) for m in [0, 1, 2]
    const qRob = new Float32Array(numA);
    for (let aIdx = 0; aIdx < numA; aIdx++) {
      const qStyle = [0, 0, 0];
      const counts = [0, 0, 0];
      for (let k = 0; k < K_eff; k++) {
        const s = worldStyles[k];
        qStyle[s] += rootValues2D[aIdx][k];
        counts[s]++;
      }
      for (let s = 0; s < 3; s++) {
        qStyle[s] = counts[s] > 0 ? qStyle[s] / counts[s] : 0;
      }

      // Soft-Minimax aggregation
      const qMin = Math.min(qStyle[0], qStyle[1], qStyle[2]);
      let sumExp = 0;
      for (let s = 0; s < 3; s++) {
        sumExp += pw[s] * Math.exp(-riskGamma * (qStyle[s] - qMin));
      }
      qRob[aIdx] = qMin - (1.0 / riskGamma) * Math.log(Math.max(1e-12, sumExp));
    }

    let vBpRob = 0;
    for (let i = 0; i < numA; i++) {
      vBpRob += bpProbsLegal[i] * qRob[i];
    }
    const advMean = new Float32Array(numA);
    for (let i = 0; i < numA; i++) {
      advMean[i] = qRob[i] - vBpRob;
    }

    // Paired Standard Error across K_eff worlds for Empirical-Bayes shrinkage
    const advSe = new Float32Array(numA);
    for (let i = 0; i < numA; i++) {
      const vals = new Float32Array(K_eff);
      let m = 0;
      for (let k = 0; k < K_eff; k++) {
        let vBpK = 0;
        for (let j = 0; j < numA; j++) {
          vBpK += bpProbsLegal[j] * rootValues2D[j][k];
        }
        vals[k] = rootValues2D[i][k] - vBpK;
        m += vals[k];
      }
      m /= K_eff;
      let varSum = 0;
      for (let k = 0; k < K_eff; k++) {
        const d = vals[k] - m;
        varSum += d * d;
      }
      const variance = K_eff > 1 ? varSum / (K_eff - 1) : 0;
      advSe[i] = Math.sqrt(variance) / Math.sqrt(K_eff);
    }

    // Empirical-Bayes shrinkage
    const effectiveQ61 = new Float32Array(61);
    for (let i = 0; i < numA; i++) {
      const a = legalActions[i];
      const meanSq = advMean[i] * advMean[i];
      const seSq = advSe[i] * advSe[i];
      const shrinkWeight = meanSq / (meanSq + ebShrink * seSq + 1e-8);
      effectiveQ61[a] = advMean[i] * shrinkWeight;
    }

    // Closed-Form MMD Update
    const denom = 1.0 + eta * tau;
    const magLogP = Math.log(1.0 / numLegal);
    const searchLogits = new Float32Array(61);
    let maxLogit = -Infinity;

    for (const a of legalActions) {
      const bpLogP = Math.log(Math.max(1e-12, bpProbs[a]));
      const logit = (bpLogP + (eta * tau) * magLogP + eta * effectiveQ61[a]) / denom;
      searchLogits[a] = logit;
      if (logit > maxLogit) maxLogit = logit;
    }

    // Softmax
    let sumExp = 0;
    const searchProbs = new Float32Array(61);
    for (const a of legalActions) {
      const p = Math.exp(searchLogits[a] - maxLogit);
      searchProbs[a] = p;
      sumExp += p;
    }
    for (const a of legalActions) {
      searchProbs[a] /= sumExp;
    }

    // Action Selection
    let chosenAct = legalActions[0];
    if (deterministic) {
      let maxP = -1;
      for (const a of legalActions) {
        if (searchProbs[a] > maxP) {
          maxP = searchProbs[a];
          chosenAct = a;
        }
      }
    } else {
      let r = Math.random();
      let cum = 0;
      for (const a of legalActions) {
        cum += searchProbs[a];
        if (r <= cum) {
          chosenAct = a;
          break;
        }
      }
    }

    const elapsedMs = performance.now() - t0;

    // Track latency & adapt if too slow (>900ms WebGPU or >600ms WASM)
    const thresholdMs = isWebGPU ? 900 : 600;
    if (elapsedMs > thresholdMs) {
      this.adaptationState.consecutiveSlowTurns++;
      if (this.adaptationState.consecutiveSlowTurns >= 2) {
        this.adaptationState.downgradedToFastActor = true;
      }
    } else {
      this.adaptationState.consecutiveSlowTurns = 0;
    }

    return {
      action_id: chosenAct,
      action: ACTION_NAMES[chosenAct] || "draw",
      action_key: ACTION_NAMES[chosenAct] || "draw",
      action_probs: Array.from(searchProbs),
      win_prob: rootWinProb,
      latency_ms: elapsedMs,
      mode_used: "risk_averse_search",
      technique_label: "Risk-Averse Portfolio Search",
      belief_xray: beliefXray,
    };
  }
}
