// Strict Model Client for Ataraxos-UNO supporting both In-Browser WebGPU/WASM and Remote Daemon
import { CONFIG } from "./config.js";

export class FatalNeuralModelException extends Error {
  constructor(message, details = null) {
    super(message);
    this.name = "FatalNeuralModelException";
    this.details = details;
  }
}

export class AtaraxosModelClient {
  constructor(endpointUrl = CONFIG.DEFAULT_API_URL) {
    this.endpointUrl = endpointUrl;
    this.remoteUnavailable = false;
    this.isServerAvailable = false;
    this.serverHealthData = null;
    this.inBrowserEngine = null;
    this.searchEngine = null;
    this.preferInBrowser = false;
    this.timeoutMs = CONFIG.INFERENCE_TIMEOUT_MS;
    this.currentMode = "auto"; // "auto" | "blueprint" | "online_bayesian" | "hybrid" | "risk_averse" | "static" | "tier1"
    this.oppPosterior = null; // Stateless opponent tracking JSON pass-through

    this.telemetry = {
      totalCalls: 0,
      lastLatencyMs: 0,
      fallbackCount: 0,
    };
  }

  setEndpointUrl(url) {
    this.endpointUrl = url;
    this.remoteUnavailable = false;
  }

  setMode(mode) {
    this.currentMode = mode || "auto";
  }

  get mode() {
    return this.currentMode;
  }

  resetGame() {
    this.oppPosterior = null;
  }

  setPreferInBrowser(prefer) {
    this.preferInBrowser = Boolean(prefer);
  }

  get isInBrowserLoaded() {
    return Boolean(this.inBrowserEngine && this.inBrowserEngine.isLoaded);
  }

  get inBrowserDevice() {
    return this.inBrowserEngine ? this.inBrowserEngine.device : null;
  }

  async _ensureInBrowserEngine() {
    if (!this.inBrowserEngine) {
      const { InBrowserAtaraxosEngine } = await import("./onnx-engine.js");
      this.inBrowserEngine = new InBrowserAtaraxosEngine();
    }
    return this.inBrowserEngine;
  }

  async initInBrowser(progressCallback) {
    const engine = await this._ensureInBrowserEngine();
    return engine.init(progressCallback);
  }

  async initInBrowserBelief(progressCallback) {
    const engine = await this._ensureInBrowserEngine();
    return engine.initBeliefModels(progressCallback);
  }

  async checkHealth(probeUrl = this.endpointUrl) {
    let targetUrl = probeUrl || this.endpointUrl;
    if (typeof window !== "undefined" && window.location) {
      const isLocal =
        window.location.hostname === "localhost" ||
        window.location.hostname === "127.0.0.1";
      const hasExplicitApi = Boolean(
        new URLSearchParams(window.location.search).get("api"),
      );
      if (!isLocal && !hasExplicitApi && !targetUrl) {
        this.isServerAvailable = false;
        return false;
      }
    }
    if (!targetUrl || this.remoteUnavailable) {
      this.isServerAvailable = false;
      return false;
    }

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 600); // 600ms fast probe
    try {
      const response = await fetch(`${targetUrl}/api/health`, {
        signal: controller.signal,
        headers: { Accept: "application/json" },
      });
      if (response.ok) {
        this.serverHealthData = await response.json();
        const agentLoaded =
          this.serverHealthData &&
          this.serverHealthData.status === "ok" &&
          this.serverHealthData.agent_loaded !== false;
        this.isServerAvailable = Boolean(agentLoaded);
        this.remoteUnavailable = false;
        return this.isServerAvailable;
      } else {
        this.isServerAvailable = false;
        return false;
      }
    } catch (_) {
      this.isServerAvailable = false;
      return false;
    } finally {
      clearTimeout(timer);
    }
  }

  async predictAction(featureBundle) {
    this.telemetry.totalCalls++;
    const startTime = performance.now();

    let choice = null;
    let probs = {};
    let winProb = 0.5;
    let modeUsed = this.currentMode;
    let techniqueLabel = "Ataraxos AI";
    let engineBackend = "server";
    let beliefXray = null;

    // 1. Try Remote Daemon if active and reachable
    if (this.isServerAvailable && this.endpointUrl && !this.remoteUnavailable && !this.preferInBrowser) {
      const controller = new AbortController();
      const timeoutTimer = setTimeout(() => controller.abort(), this.timeoutMs);
      try {
        const payload = {
          state: featureBundle.state || featureBundle,
          mode: this.currentMode === "blueprint" ? "blueprint" : "auto",
          opp_posterior: this.oppPosterior,
        };

        const response = await fetch(`${this.endpointUrl}/api/predict_choice`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          signal: controller.signal,
          body: JSON.stringify(payload),
        });

        if (response.ok) {
          const data = await response.json();
          if (data && !data.__error__ && !data.error) {
            choice = data.action_key || data.action || data.choice;
            probs = data.action_probs || data.probabilities || {};
            winProb = Number(data.win_prob ?? 0.5);
            modeUsed = data.mode_used || this.currentMode;
            techniqueLabel = data.technique_label || "Ataraxos AI";
            engineBackend = "server";
            beliefXray = data.belief_xray || null;
            if (data.opp_posterior) {
              this.oppPosterior = data.opp_posterior;
            }
          }
        } else {
          this.isServerAvailable = false;
        }
      } catch (daemonErr) {
        this.isServerAvailable = false;
      } finally {
        clearTimeout(timeoutTimer);
      }
    }

    // 2. Client-Side In-Browser Inference (Risk-Averse Search or Blueprint)
    if (!choice) {
      try {
        const browserEngine = await this._ensureInBrowserEngine();
        if (!browserEngine.isLoaded) {
          if (!browserEngine.isLoading) {
            await browserEngine.init();
          } else {
            while (browserEngine.isLoading) {
              await new Promise((resolve) => setTimeout(resolve, 80));
            }
          }
        }

        if (this.currentMode === "blueprint") {
          const pred = await browserEngine.predict(featureBundle);
          choice = pred.action_key || pred.action;
          probs = pred.action_probs;
          winProb = pred.win_prob;
          modeUsed = "blueprint";
          techniqueLabel = "Ataraxos AI";
          engineBackend = browserEngine.device === "webgpu" ? "browser_webgpu" : "browser_wasm";
          beliefXray = pred.belief_xray;
        } else {
          // Native In-Browser Milestone 8 Risk-Averse Portfolio Search
          if (!this.searchEngine) {
            const { RiskAverseSearchEngine } = await import("./risk-averse-search.js");
            this.searchEngine = new RiskAverseSearchEngine(browserEngine);
          }

          if (featureBundle.fastEnv) {
            const pred = await this.searchEngine.search({
              rootEnv: featureBundle.fastEnv,
              deterministic: true,
            });
            choice = pred.action_key || pred.action;
            probs = pred.action_probs;
            winProb = pred.win_prob;
            modeUsed = "risk_averse_search";
            techniqueLabel = "Ataraxos AI";
            engineBackend = browserEngine.device === "webgpu" ? "browser_webgpu" : "browser_wasm";
            beliefXray = pred.belief_xray;
          } else {
            const pred = await browserEngine.predict(featureBundle);
            choice = pred.action_key || pred.action;
            probs = pred.action_probs;
            winProb = pred.win_prob;
            modeUsed = "blueprint";
            techniqueLabel = "Ataraxos AI";
            engineBackend = browserEngine.device === "webgpu" ? "browser_webgpu" : "browser_wasm";
            beliefXray = pred.belief_xray;
          }
        }
      } catch (browserErr) {
        throw new FatalNeuralModelException(
          `In-browser neural inference failed: ${browserErr.message}`,
          browserErr,
        );
      }
    }

    const elapsedMs = performance.now() - startTime;
    this.telemetry.lastLatencyMs = elapsedMs;

    return {
      action: choice,
      action_key: choice,
      probabilities: probs,
      win_prob: winProb,
      mode_used: modeUsed,
      technique_label: techniqueLabel,
      engine_backend: engineBackend,
      belief_xray: beliefXray,
      latencyMs: elapsedMs,
    };
  }
}

// Backward-compatible alias
export { AtaraxosModelClient as LayaModelClient };
