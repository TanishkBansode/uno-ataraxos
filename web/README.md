# Ataraxos-UNO Web Interface

The `web/` directory contains the browser frontend and client-side inference engine for [**Ataraxos-UNO**](../README.md), a neural decision and search agent for competitive 1v1 UNO.

[**▶ Play Live on GitHub Pages**](https://tanishkbansode.github.io/uno-ataraxos/)

---

## 1. Inference Backends: Browser vs. Python Daemon

The interface supports two execution architectures depending on your setup:

| Feature | Client-Side In-Browser (Default) | Python Daemon Path |
| :--- | :--- | :--- |
| **Python / C++ Required** | **No** (Zero backend) | **Yes** (Python 3.9+, PyTorch, C++17 extension) |
| **Inference Engine** | ONNX Runtime Web (WebGPU / WASM) | Native PyTorch + C++ `VecUnoEnv` |
| **Model Execution** | **Client-side** (User's GPU via WebGPU, or WASM CPU fallback) | **Server-side** (Host CUDA, Apple Silicon MPS, or CPU) |
| **Search Capabilities** | In-browser 3-style Risk-Averse Portfolio Search (device-budgeted: 192 worlds on WebGPU, 48 on WASM) | Full `AtaraxosUnoAgent` (endgame PDCFR+ solver, online Bayesian style tracker, all ablation modes) |
| **Static Hosting** | **Supported** (GitHub Pages, Cloudflare Pages, `npx serve`) | Requires a running Python host process |
| **Network Requests** | **None during gameplay** (Models download once and cache locally in IndexedDB) | HTTP `POST /api/predict_choice` per bot turn |

---

## 2. Running the Web Interface

### Path A: Browser-Only Local Server (Zero Setup)
No PyTorch checkpoints or C++ compilers needed. Any static HTTP server will serve the client-side ONNX app:

```bash
# Using Node.js
npx serve web

# Or using Python's built-in static server
python3 -m http.server 8000 --directory web
```
Open <http://localhost:8000>. The bundled ONNX models (`web/model/*.onnx`) load directly into your browser's WebGPU/WASM engine.

---

### Path B: Local Python Daemon (Full PyTorch Agent)
If you want to run the full server-side PyTorch agent with deep search and custom checkpoint paths:

```bash
# 1. Build C++17 extension and install package
pip install -e .

# 2. Launch server (serves web/ and provides /api/predict_choice)
python scripts/serve.py \
  --port 8000 \
  --blueprint runs/blueprint_100m/latest.pt \
  --belief runs/belief_20m/best.pt
```

Open <http://localhost:8000>. The web interface automatically detects the local `/api/health` endpoint and routes bot decisions to the Python daemon.

---

## 3. Bundled Model Assets

All web models are stored in `web/model/` and tracked in the repository:

| Model Asset | File Size | Architecture & Role |
| :--- | :--- | :--- |
| `ataraxos_actor.onnx` | 8.58 MB | FP32 `UnoActorNet` policy (61 actions) & WDL value head |
| `ataraxos_critic.onnx` | 8.38 MB | FP32 `UnoQCriticNet` state-action leaf evaluator |
| `belief_enc.onnx` | 2.62 MB | Board & history context encoder for belief generation |
| `belief_dec.onnx` | 6.65 MB | Autoregressive descending multiset decoder |

### Re-Exporting Models & Parity Verification
To re-export ONNX assets from PyTorch checkpoints and run bit-for-bit parity tests:

```bash
python scripts/export_onnx.py \
  --blueprint runs/blueprint_100m/latest.pt \
  --belief runs/belief_20m/best.pt \
  --output-dir web/model
```
This script validates:
- 100/100 argmax action parity between PyTorch and ONNX Runtime.
- Max absolute difference $< 10^{-4}$ on output probability and Q-value tensors.

---

## 4. Browser Compatibility & Performance

- **WebGPU (Preferred):** Provides sub-millisecond tensor operations on Chrome, Edge, and Safari (macOS 14+ / iOS 17+). Runs on Metal, DirectX 12, or Vulkan.
- **WebAssembly (WASM Fallback):** Automatically activates if WebGPU is unavailable or disabled. Multi-threaded WASM utilizes SIMD instructions for CPU evaluation.
- **Cross-Origin Isolation:** The web server sends `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp` headers to enable high-precision timers and `SharedArrayBuffer` threading.
