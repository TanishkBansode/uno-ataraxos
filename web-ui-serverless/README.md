# UNO Ataraxos AI — Serverless Web Edition

A static HTML/JS/CSS UNO game that plays against **Ataraxos AI**, an elite
neural decision and search agent trained via Ataraxos update-equivalence search.

- **Two inference paths cleanly supported:**
  1. **In-Browser Serverless ONNX (`WebGPU` / `WASM`):** Runs `ataraxos_actor.onnx` (~8.58 MB FP32), `ataraxos_critic.onnx` (~8.38 MB), and belief models directly inside the browser using ONNX Runtime Web. Zero tokenizer, zero external HF weight downloads, and zero CPU fallback for attention ops!
  2. **Daemon / Docker Space Path (`scripts/serve_model.py` `/api/predict_choice`):** Backed by `AtaraxosUnoAgent`, seamlessly supporting all modes (`blueprint`, `tier1`, `hybrid`, `static`, `risk_averse`, `online_bayesian`, and `auto` reading `runs/search_config.json`) with stateless JSON `opp_posterior` pass-through.
- **WebGPU first:** When the browser exposes a WebGPU adapter (Metal on Apple Silicon, DirectX on Windows, Vulkan on Linux), the local GPU engine is automatically prioritized for sub-millisecond turn latency. An explicit `?api=...` parameter in the URL forces the remote daemon endpoint.
- **Zero-Fallback Policy:** Every decision comes strictly from the neural model or search solver. If neural backends fail, the game halts with a clear diagnostic modal rather than playing random or heuristic cards.
- **Live Belief X-Ray HUD:** Live diagnostic panel visualizing the AI's Bayesian belief over opponent hand card colors (+2, +4, and color distributions).

---

## Model Assets

All models reside directly in `web-ui-serverless/model/`:

| Asset | Size | Purpose |
| ----- | ---- | ------- |
| `ataraxos_actor.onnx` | 8.58 MB | FP32 UnoActorNet policy & WDL value head (WebGPU/WASM) |
| `ataraxos_critic.onnx` | 8.38 MB | FP32 UnoQCriticNet leaf evaluation (WebGPU/WASM) |
| `belief_enc.onnx` | 2.62 MB | Optional in-browser UnoBeliefNet encoder |
| `belief_dec.onnx` | 6.65 MB | Optional in-browser UnoBeliefNet autoregressive decoder |

### Exporting Models

To re-export or verify parity against native PyTorch:

```bash
python scripts/export_ataraxos_onnx.py
```

This exports the models with WebGPU-safe explicit attention and runs mandatory `VecUnoEnv` parity tests asserting 100/100 argmax match and max probability difference `< 1e-4` on Actor, and max Q difference `< 1e-4` on Critic.

---

## Running Locally

From the repository root:

```bash
python scripts/serve_model.py --port 8000 --blueprint runs/blueprint_100m/latest.pt --belief runs/belief_20m/best.pt
```

Open <http://localhost:8000>. The server hosts both the static frontend and the `/api/predict_choice` endpoint.

To run purely serverless without any Python backend, serve `web-ui-serverless/` with any static web server (e.g. `npx serve web-ui-serverless` or GitHub Pages / Cloudflare Pages).
