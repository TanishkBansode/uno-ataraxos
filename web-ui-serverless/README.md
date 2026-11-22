# Ataraxos-UNO Web Interface

A browser-based UNO interface against **Ataraxos AI**, a neural decision
and search agent trained via update-equivalence search.

- **Supported inference backends:**
  1. **In-Browser ONNX (`WebGPU` / `WASM`):** Runs `ataraxos_actor.onnx` (8.58 MB), `ataraxos_critic.onnx` (8.38 MB), and belief models directly in the browser via ONNX Runtime Web. No tokenizers, external weight downloads, or CPU fallbacks for attention operations.
  2. **Daemon Path (`scripts/serve_model.py` `/api/predict_choice`):** Backed by `AtaraxosUnoAgent`, supporting all search modes (`blueprint`, `tier1`, `hybrid`, `static`, `risk_averse`, `online_bayesian`, and `auto` from `runs/search_config.json`) with stateless JSON opponent belief tracking.
- **WebGPU first:** Prioritizes local GPU execution (Metal on macOS, DirectX on Windows, Vulkan on Linux) for sub-millisecond move latency. An explicit `?api=...` query parameter directs moves to a remote daemon.
- **Strict neural execution:** Moves are generated exclusively by the neural network or search solver. If inference fails, the game halts with a diagnostic modal rather than falling back to heuristic play.
- **Belief X-Ray HUD:** Visualizes the AI's estimated probability distribution over your cards (+2, +4, and colors).

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
