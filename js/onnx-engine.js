// In-Browser ONNX Neural Inference Engine for Ataraxos-UNO
// Runs 2.24M-param UnoActorNet (~8.9 MB FP32) + optional UnoBeliefNet completely inside the browser.
// Supports native WebGPU acceleration (Metal on macOS, DirectX on Windows, Vulkan on Linux)
// with clean multithreaded SIMD WASM fallback. Zero tokenizer, zero external HF weights download!

import {
  ACTION_NAMES,
  CARD_NAMES,
  TRAIT,
  cardTrait,
  cardColor,
} from "./ataraxos-features.js";

const DEFAULT_ACTOR_URL = "model/ataraxos_actor.onnx";
const DEFAULT_BELIEF_ENC_URL = "model/belief_enc.onnx";
const DEFAULT_BELIEF_DEC_URL = "model/belief_dec.onnx";

/** ?model=<url> or <meta name="uno-model"> override the weights location. */
function resolveActorModelPath() {
  if (typeof window !== "undefined" && window.location) {
    try {
      const override = new URLSearchParams(window.location.search).get("model");
      if (override && override.trim()) return override.trim();
    } catch (_) {}
  }
  if (typeof document !== "undefined") {
    const meta = document.querySelector('meta[name="uno-model"]');
    const value = meta && (meta.getAttribute("content") || "").trim();
    if (value) return value;
  }
  return DEFAULT_ACTOR_URL;
}

export class InBrowserAtaraxosEngine {
  constructor() {
    this.session = null;
    this.beliefEncSession = null;
    this.beliefDecSession = null;
    this.isLoaded = false;
    this.isLoading = false;
    this.loadProgress = 0;
    this.device = "wasm";
    this.threads = 1;
    this.crossOriginIsolated = false;
    this.beliefLoaded = false;
  }

  static async _webgpuAvailable() {
    if (typeof navigator === "undefined" || !navigator.gpu) return false;
    try {
      const adapter = await navigator.gpu.requestAdapter({
        powerPreference: "high-performance",
      });
      return Boolean(adapter);
    } catch (_) {
      return false;
    }
  }

  _configureOrtEnv(ort) {
    const isolated =
      typeof crossOriginIsolated !== "undefined" && crossOriginIsolated === true;
    this.crossOriginIsolated = isolated;
    const hardwareThreads =
      typeof navigator !== "undefined" && navigator.hardwareConcurrency
        ? navigator.hardwareConcurrency
        : 4;
    this.threads = isolated ? Math.min(4, Math.max(1, hardwareThreads)) : 1;

    if (ort.env && ort.env.wasm) {
      ort.env.wasm.numThreads = this.threads;
      ort.env.wasm.simd = true;
      ort.env.wasm.proxy = false;
    }
    if (ort.env) {
      ort.env.logLevel = "error";
    }
  }

  async _fetchBuffer(url, progressCb, startPct = 10, endPct = 80, label = "Model") {
    const resp = await fetch(url);
    if (!resp.ok) {
      throw new Error(`Failed to load ${label} from ${url}: HTTP ${resp.status}`);
    }
    const contentLength = Number(resp.headers.get("Content-Length") || 0);

    if (!contentLength || !resp.body) {
      const buf = await resp.arrayBuffer();
      if (progressCb) progressCb(endPct, `Loaded ${label}`);
      return buf;
    }

    const reader = resp.body.getReader();
    const chunks = [];
    let received = 0;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.length;
      if (progressCb) {
        const pct = startPct + (received / contentLength) * (endPct - startPct);
        const mb = (received / (1024 * 1024)).toFixed(1);
        const totalMb = (contentLength / (1024 * 1024)).toFixed(1);
        progressCb(Math.min(endPct, Math.round(pct)), `Loading ${label} (${mb} / ${totalMb} MB)...`);
      }
    }

    const all = new Uint8Array(received);
    let offset = 0;
    for (const chunk of chunks) {
      all.set(chunk, offset);
      offset += chunk.length;
    }
    return all.buffer;
  }

  async init(progressCallback = null) {
    if (this.isLoaded) return;
    if (this.isLoading) {
      while (this.isLoading) {
        await new Promise((r) => setTimeout(r, 80));
      }
      return;
    }
    this.isLoading = true;

    try {
      if (typeof ort === "undefined") {
        throw new Error(
          "onnxruntime-web not found on page. Ensure ort.all.min.js script tag is loaded.",
        );
      }
      this._configureOrtEnv(ort);

      const hasWebGpu = await InBrowserAtaraxosEngine._webgpuAvailable();
      const modelPath = resolveActorModelPath();

      if (progressCallback) {
        progressCallback(10, `Fetching Ataraxos Actor (${hasWebGpu ? "WebGPU" : "WASM"})...`);
      }

      const modelBuffer = await this._fetchBuffer(
        modelPath,
        progressCallback,
        15,
        70,
        "Ataraxos-UNO Actor",
      );

      if (progressCallback) {
        progressCallback(75, "Creating Neural Inference Session...");
      }

      const executionProviders = hasWebGpu
        ? [
            {
              name: "webgpu",
              deviceType: "default",
              powerPreference: "high-performance",
            },
            "wasm",
          ]
        : ["wasm"];

      const sessionOptions = {
        executionProviders,
        graphOptimizationLevel: "all",
      };

      try {
        this.session = await ort.InferenceSession.create(modelBuffer, sessionOptions);
        this.device = hasWebGpu ? "webgpu" : "wasm";
      } catch (gpuErr) {
        if (hasWebGpu) {
          console.warn("[Ataraxos] WebGPU failed, falling back to multi-threaded WASM:", gpuErr);
          this.session = await ort.InferenceSession.create(modelBuffer, {
            executionProviders: ["wasm"],
            graphOptimizationLevel: "all",
          });
          this.device = "wasm";
        } else {
          throw gpuErr;
        }
      }

      // Prewarm Actor Session
      if (progressCallback) progressCallback(85, "Prewarming Neural Pipelines...");
      await this._prewarmActor();

      this.isLoaded = true;
      this.isLoading = false;
      if (progressCallback) {
        progressCallback(100, `⚡ Ataraxos-UNO Ready (${this.device.toUpperCase()})`);
      }
    } catch (err) {
      this.isLoading = false;
      this.isLoaded = false;
      console.error("[Ataraxos] In-Browser Neural Engine init failed:", err);
      throw err;
    }
  }

  async _prewarmActor() {
    if (!this.session) return;
    const dummyCf = new Float32Array(54 * 7);
    const dummyGf = new Float32Array(10);
    const dummyHt = new BigInt64Array(48 * 5);
    // Fill padding tokens: [2, 61, 4, 0, 0]
    for (let i = 0; i < 48; i++) {
      dummyHt[i * 5 + 0] = 2n;
      dummyHt[i * 5 + 1] = 61n;
      dummyHt[i * 5 + 2] = 4n;
      dummyHt[i * 5 + 3] = 0n;
      dummyHt[i * 5 + 4] = 0n;
    }
    const dummyMask = new Float32Array(61);
    dummyMask[60] = 1.0;

    const feeds = {
      card_features: new ort.Tensor("float32", dummyCf, [1, 54, 7]),
      global_features: new ort.Tensor("float32", dummyGf, [1, 10]),
      history_tokens: new ort.Tensor("int64", dummyHt, [1, 48, 5]),
      legal_mask: new ort.Tensor("float32", dummyMask, [1, 61]),
    };
    await this.session.run(feeds);
  }

  async initBeliefModels(progressCallback = null) {
    if (this.beliefLoaded) return;
    try {
      const hasWebGpu = this.device === "webgpu";
      const ep = hasWebGpu
        ? [
            {
              name: "webgpu",
              deviceType: "default",
              powerPreference: "high-performance",
            },
            "wasm",
          ]
        : ["wasm"];

      const encBuf = await this._fetchBuffer(DEFAULT_BELIEF_ENC_URL, progressCallback, 0, 50, "Belief Encoder");
      const decBuf = await this._fetchBuffer(DEFAULT_BELIEF_DEC_URL, progressCallback, 50, 95, "Belief Decoder");

      this.beliefEncSession = await ort.InferenceSession.create(encBuf, { executionProviders: ep });
      this.beliefDecSession = await ort.InferenceSession.create(decBuf, { executionProviders: ep });
      this.beliefLoaded = true;
      if (progressCallback) progressCallback(100, "Belief X-Ray Models Ready");
    } catch (e) {
      console.warn("[Ataraxos] Could not load optional in-browser belief models:", e);
    }
  }

  /**
   * Run Blueprint inference on provided Ataraxos features.
   */
  async predict({ cardFeatures, globalFeatures, historyTokens, legalMask }) {
    if (!this.isLoaded || !this.session) {
      throw new Error("Ataraxos In-Browser Neural Engine is not initialized.");
    }
    const t0 = performance.now();

    // Ensure BigInt64Array for history_tokens
    let htTensorData = historyTokens;
    if (!(historyTokens instanceof BigInt64Array)) {
      const bi = new BigInt64Array(historyTokens.length);
      for (let i = 0; i < historyTokens.length; i++) {
        bi[i] = BigInt(historyTokens[i]);
      }
      htTensorData = bi;
    }

    const feeds = {
      card_features: new ort.Tensor("float32", cardFeatures, [1, 54, 7]),
      global_features: new ort.Tensor("float32", globalFeatures, [1, 10]),
      history_tokens: new ort.Tensor("int64", htTensorData, [1, 48, 5]),
      legal_mask: new ort.Tensor("float32", legalMask, [1, 61]),
    };

    const results = await this.session.run(feeds);
    const actionProbs = Array.from(results.action_probs.data);
    const wdlProbs = Array.from(results.wdl_probs.data); // [P(Loss), P(Draw), P(Win)]
    const latencyMs = performance.now() - t0;

    // Argmax over legal actions
    let bestAction = 60;
    let maxP = -1.0;
    for (let i = 0; i < 61; i++) {
      if (legalMask[i] > 0.5 && actionProbs[i] > maxP) {
        maxP = actionProbs[i];
        bestAction = i;
      }
    }

    const winProb = (wdlProbs[2] || 0) + 0.5 * (wdlProbs[1] || 0);
    const actionKey = ACTION_NAMES[bestAction] || "draw";

    // Optional fast belief X-Ray if belief models are loaded
    let beliefXray = null;
    if (this.beliefLoaded && this.beliefEncSession && this.beliefDecSession) {
      try {
        beliefXray = await this.sampleBeliefXray({
          cardFeatures,
          globalFeatures,
          historyTokens: htTensorData,
          oppHandSize: Math.round(globalFeatures[1] * 20.0),
        });
      } catch (bErr) {
        console.warn("[Ataraxos] In-browser belief sampling error:", bErr);
      }
    }

    return {
      action_id: bestAction,
      action: actionKey,
      action_key: actionKey,
      action_probs: actionProbs,
      win_prob: winProb,
      latency_ms: latencyMs,
      device: this.device,
      mode_used: `in-browser (${this.device.toUpperCase()})`,
      belief_xray: beliefXray,
    };
  }

  async sampleBeliefXray({ cardFeatures, globalFeatures, historyTokens, oppHandSize = 7 }) {
    if (!this.beliefEncSession || !this.beliefDecSession || oppHandSize <= 0) {
      return null;
    }
    // Run Encoder
    const encFeeds = {
      card_features: new ort.Tensor("float32", cardFeatures, [1, 54, 7]),
      global_features: new ort.Tensor("float32", globalFeatures, [1, 10]),
      history_tokens: new ort.Tensor("int64", historyTokens, [1, 48, 5]),
    };
    const encOut = await this.beliefEncSession.run(encFeeds);
    const encContext = encOut.enc_context.data; // Float32Array [256]

    // Fast autoregressive sampling of K=16 belief hands
    const K = 16;
    let hasP2Count = 0;
    let hasP4Count = 0;
    let redCount = 0;
    let greenCount = 0;
    let blueCount = 0;
    let yellowCount = 0;

    // Unseen card limits from cardFeatures (channel 2 is unseen_total)
    const unseenBase = new Float32Array(54);
    for (let c = 0; c < 54; c++) {
      unseenBase[c] = cardFeatures[c * 7 + 2];
    }

    for (let k = 0; k < K; k++) {
      const picked = new Float32Array(54);
      const remElig = new Float32Array(unseenBase);
      let prevToken = 54n; // BOS
      let hasP2 = false;
      let hasP4 = false;
      let hasRed = false;
      let hasGreen = false;
      let hasBlue = false;
      let hasYellow = false;

      for (let j = 0; j < oppHandSize; j++) {
        const needed = oppHandSize - j;
        const stepMask = new Float32Array(54);
        let availSum = 0;
        for (let c = 0; c < 54; c++) {
          availSum += remElig[c];
          if (c <= Number(prevToken) && remElig[c] > 0 && availSum >= needed) {
            stepMask[c] = 1.0;
          }
        }

        const decFeeds = {
          enc_context: new ort.Tensor("float32", encContext, [1, 256]),
          picked_so_far: new ort.Tensor("float32", picked, [1, 54]),
          remaining_eligible: new ort.Tensor("float32", remElig, [1, 54]),
          prev_card_token: new ort.Tensor("int64", new BigInt64Array([prevToken]), [1]),
          cards_still_needed: new ort.Tensor("float32", new Float32Array([needed]), [1]),
          step_legal_mask: new ort.Tensor("float32", stepMask, [1, 54]),
        };

        const decOut = await this.beliefDecSession.run(decFeeds);
        const probs = decOut.step_probs.data;

        // Sample card proportional to probabilities
        let r = Math.random();
        let chosenCard = 0;
        let cum = 0;
        for (let c = 0; c < 54; c++) {
          cum += probs[c];
          if (r <= cum && stepMask[c] > 0.5) {
            chosenCard = c;
            break;
          }
        }

        picked[chosenCard]++;
        remElig[chosenCard] = Math.max(0, remElig[chosenCard] - 1);
        prevToken = BigInt(chosenCard);

        const tr = cardTrait(chosenCard);
        const col = cardColor(chosenCard);
        if (tr === TRAIT.DRAW_2) hasP2 = true;
        if (tr === TRAIT.WILD_DRAW_4) hasP4 = true;
        if (col === 0) hasRed = true;
        if (col === 1) hasGreen = true;
        if (col === 2) hasBlue = true;
        if (col === 3) hasYellow = true;
      }

      if (hasP2) hasP2Count++;
      if (hasP4) hasP4Count++;
      if (hasRed) redCount++;
      if (hasGreen) greenCount++;
      if (hasBlue) blueCount++;
      if (hasYellow) yellowCount++;
    }

    return {
      p_plus_2: hasP2Count / K,
      p_plus_4: hasP4Count / K,
      p_red: redCount / K,
      p_green: greenCount / K,
      p_blue: blueCount / K,
      p_yellow: yellowCount / K,
      opp_hand_size: oppHandSize,
    };
  }
}

// Backward-compatible alias
export { InBrowserAtaraxosEngine as InBrowserLayaEngine };
