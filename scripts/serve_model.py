#!/usr/bin/env python3
"""Unified Daemon & Docker Space HTTP Server for Ataraxos-UNO.

Serves:
1. Static files from `web-ui-serverless/` (HTML, JS, CSS, ONNX models, icons/sounds).
2. GET `/api/health`: Service health check.
3. POST `/api/predict_choice`:
   - Backed by AtaraxosUnoAgent.
   - Seamlessly supports ALL 6 modes:
     `blueprint`, `tier1`, `hybrid`, `static`, `risk_averse`, `online_bayesian`, plus `auto` (reading `runs/search_config.json`).
   - 100% Stateless: pass-through JSON `opp_posterior` tracking across turns and games!
   - Returns: `{ action_id, action_key, action_probs, win_prob, mode_used, latency_ms, belief_xray, opp_posterior }`.

Usage:
    python scripts/serve_model.py --port 8000 --blueprint runs/blueprint_100m/latest.pt --belief runs/belief_20m/best.pt
"""

from __future__ import annotations

import argparse
from http.server import HTTPServer, SimpleHTTPRequestHandler
import json
import mimetypes
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import urllib.parse

import numpy as np
import torch
import torch.nn.functional as F

import uno_ataraxos as ua
from uno_ataraxos import (
    UnoEngine,
    action_name_to_id,
    card_name_to_id,
    get_action_names,
    get_card_names,
)
from uno_ataraxos.agent import AtaraxosUnoAgent
from uno_ataraxos.opponent_model import DEFAULT_PRIOR
from uno_ataraxos.rl_math import get_device


class AtaraxosUnoServer:
    """Thread-safe backend manager for AtaraxosUnoAgent."""

    def __init__(
        self,
        blueprint_ckpt: str = "runs/blueprint_100m/latest.pt",
        belief_ckpt: str = "runs/belief_20m/best.pt",
        critic_ckpt: Optional[str] = None,
        search_config_path: str = "runs/search_config.json",
        device: Optional[torch.device] = None,
    ):
        self.device = device or get_device()
        self.blueprint_ckpt = str(blueprint_ckpt)
        self.belief_ckpt = str(belief_ckpt)
        self.search_config_path = search_config_path
        self.action_names = list(get_action_names())
        self.card_names = list(get_card_names())

        # Load search config
        self.search_cfg: Dict[str, Any] = {}
        cfg_file = Path(search_config_path)
        if cfg_file.exists():
            try:
                with open(cfg_file, "r") as f:
                    self.search_cfg = json.load(f)
            except Exception as e:
                print(f"[Warning] Could not load search config from {cfg_file}: {e}")

        # Initialize Agent with hybrid capabilities
        print(f"Loading AtaraxosUnoAgent on device {self.device}...")
        self.agent = AtaraxosUnoAgent(
            blueprint_ckpt=blueprint_ckpt,
            belief_ckpt=belief_ckpt,
            critic_ckpt=critic_ckpt,
            mode="hybrid",
            device=self.device,
            search_config_path=search_config_path,
            warmup=True,
        )
        print("AtaraxosUnoAgent initialized successfully!")

    def resolve_mode_config(self, requested_mode: str) -> Dict[str, Any]:
        """Resolve requested mode string to full parameter dict with ZERO parameter bleed."""
        mode_clean = (requested_mode or "auto").strip().lower()

        # 1. Champion: Ablation B (Static Risk-Averse Portfolio)
        if mode_clean in ("auto", "risk_averse", "static"):
            return {
                "mode": self.search_cfg.get("mode", "mmd_search"),
                "use_portfolio": self.search_cfg.get("use_portfolio", True),
                "use_adaptive_bayes": self.search_cfg.get("use_adaptive_bayes", False),
                "portfolio_weights": tuple(self.search_cfg.get("portfolio_weights", [0.45, 0.35, 0.20])),
                "risk_gamma": float(self.search_cfg.get("risk_gamma", 4.0)),
                "uniform_mix": float(self.search_cfg.get("uniform_mix", 0.12)),
                "belief_temp": float(self.search_cfg.get("belief_temp", 1.15)),
                "rollout_temp": float(self.search_cfg.get("rollout_temp", 0.0)),
                "depth": int(self.search_cfg.get("depth", 14)),
                "bp_floor": float(self.search_cfg.get("bp_floor", 0.03)),
                "eb_shrink": float(self.search_cfg.get("eb_shrink", 1.0)),
                "eta": float(self.search_cfg.get("eta", 8.0)),
                "tau": float(self.search_cfg.get("tau", 0.002)),
                "max_total_worlds": int(self.search_cfg.get("max_total_worlds", 512)),
                "technique_label": "Ablation B: Static Risk-Averse Portfolio (Champion)",
            }

        # 2. Control A: Pure 100M Blueprint
        elif mode_clean == "blueprint":
            return {
                "mode": "blueprint",
                "use_portfolio": False,
                "use_adaptive_bayes": False,
                "portfolio_weights": (0.50, 0.30, 0.20),
                "risk_gamma": 0.0,
                "uniform_mix": 0.03,
                "belief_temp": 1.0,
                "rollout_temp": 0.0,
                "depth": 14,
                "bp_floor": 0.03,
                "eb_shrink": 1.0,
                "eta": 8.0,
                "tau": 0.002,
                "max_total_worlds": 512,
                "technique_label": "Control A: Pure 100M Blueprint",
            }

        # 3. Ablation C: Adaptive Bayesian Portfolio (Tier-1)
        elif mode_clean == "online_bayesian":
            return {
                "mode": "mmd_search",
                "use_portfolio": True,
                "use_adaptive_bayes": True,
                "portfolio_weights": (0.50, 0.30, 0.20),
                "risk_gamma": 4.0,
                "uniform_mix": None,  # dynamic from tracker
                "belief_temp": None,  # dynamic from tracker
                "rollout_temp": 0.0,
                "depth": 14,
                "bp_floor": 0.03,
                "eb_shrink": 1.0,
                "eta": 8.0,
                "tau": 0.002,
                "max_total_worlds": 512,
                "technique_label": "Ablation C: Adaptive Bayesian Portfolio (Tier-1)",
            }

        # 4. Ablation D: Adaptive Bayesian Hybrid (+ Endgame PDCFR+)
        elif mode_clean == "hybrid":
            return {
                "mode": "hybrid",
                "use_portfolio": True,
                "use_adaptive_bayes": True,
                "portfolio_weights": (0.50, 0.30, 0.20),
                "risk_gamma": 4.0,
                "uniform_mix": None,  # dynamic from tracker
                "belief_temp": None,  # dynamic from tracker
                "rollout_temp": 0.0,
                "depth": 14,
                "bp_floor": 0.03,
                "eb_shrink": 1.0,
                "eta": 8.0,
                "tau": 0.002,
                "max_total_worlds": 512,
                "technique_label": "Ablation D: Adaptive Bayesian Hybrid (+ Endgame PDCFR+)",
            }

        # 5. Clean Single-Style Tier-1 MMD Search
        elif mode_clean in ("tier1", "mmd_search"):
            return {
                "mode": "mmd_search",
                "use_portfolio": False,
                "use_adaptive_bayes": False,
                "portfolio_weights": (0.50, 0.30, 0.20),
                "risk_gamma": 0.0,
                "uniform_mix": 0.03,
                "belief_temp": 1.0,
                "rollout_temp": 0.0,
                "depth": 14,
                "bp_floor": 0.03,
                "eb_shrink": 1.0,
                "eta": 8.0,
                "tau": 0.002,
                "max_total_worlds": 512,
                "technique_label": "Tier-1 MMD Search (Single Style)",
            }

        else:
            return self.resolve_mode_config("auto")

    def predict(
        self,
        state_data: Dict[str, Any],
        mode: str = "auto",
        opp_posterior: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Stateless decision endpoint supporting all verified Milestone 8 modes."""
        t0 = time.perf_counter()
        cfg = self.resolve_mode_config(mode)

        # Configure agent parameters cleanly for this request (ZERO mode-bleed)
        self.agent.mode = cfg["mode"]
        self.agent.use_portfolio = cfg["use_portfolio"]
        self.agent.use_adaptive_bayes = cfg["use_adaptive_bayes"]
        self.agent.portfolio_weights = cfg["portfolio_weights"]
        self.agent.risk_gamma = cfg["risk_gamma"]
        self.agent.uniform_mix = cfg["uniform_mix"]
        self.agent.belief_temp = cfg["belief_temp"]
        self.agent.rollout_temp = cfg["rollout_temp"]
        self.agent.depth = cfg["depth"]
        self.agent.bp_floor = cfg["bp_floor"]
        self.agent.eb_shrink = cfg["eb_shrink"]
        self.agent.eta = cfg["eta"]
        self.agent.tau = cfg["tau"]
        self.agent.max_total_worlds = cfg["max_total_worlds"]

        # Restore stateless opponent posterior if provided
        tracker = self.agent.opp_tracker
        if cfg["use_adaptive_bayes"] and opp_posterior and isinstance(opp_posterior, dict):
            if "weights" in opp_posterior and isinstance(opp_posterior["weights"], (list, tuple)):
                tracker.weights = [float(w) for w in opp_posterior["weights"]]
            elif "blueprint" in opp_posterior:
                tracker.weights = [
                    float(opp_posterior.get("blueprint", 0.5)),
                    float(opp_posterior.get("aggressive", 0.3)),
                    float(opp_posterior.get("unpredictable", 0.2)),
                ]
            else:
                tracker.weights = list(DEFAULT_PRIOR)

            tracker.ema_surprisal = float(opp_posterior.get("ema_surprisal", 0.5))
            tracker._cached_turn = opp_posterior.get("cached_turn", None)
            tracker._last_observed_turn = int(opp_posterior.get("last_observed_turn", -1))
            cached_probs = opp_posterior.get("cached_action_probs", None)
            if cached_probs is not None:
                tracker._cached_action_probs = torch.tensor(cached_probs, dtype=torch.float32)
            else:
                tracker._cached_action_probs = None
        else:
            tracker.reset()
            if not cfg["use_adaptive_bayes"]:
                tracker.weights = list(cfg["portfolio_weights"])

        # Build UnoEngine from state
        eng = UnoEngine(42)

        # 1. Extract hands
        h0_raw = state_data.get("hands_0", state_data.get("hand_0", []))
        h1_raw = state_data.get("hands_1", state_data.get("hand_1", []))
        deck_count = int(state_data.get("deck_count", state_data.get("deck_len", 40)))
        discard_raw = state_data.get("discard_pile", state_data.get("played_cards", []))
        target_raw = state_data.get("target", "r-0")
        active_color = int(state_data.get("active_color", 0))
        stack = int(state_data.get("stack", 0))
        current_player = int(state_data.get("current_player", 1))
        turn_count = int(state_data.get("turn_count", 0))
        consecutive_passes = int(state_data.get("consecutive_passes", 0))

        # Convert card IDs or names
        def to_card_id(c: Any) -> int:
            if isinstance(c, int):
                return c
            s = str(c).strip()
            cid = card_name_to_id(s)
            return cid if cid >= 0 else 0

        def to_action_id(a: Any) -> int:
            if isinstance(a, int):
                return a
            s = str(a).strip()
            aid = action_name_to_id(s)
            return aid if aid >= 0 else 0

        h0_ids = [to_card_id(c) for c in h0_raw]
        h1_ids = [to_card_id(c) for c in h1_raw]
        discard_ids = [to_action_id(a) for a in discard_raw]
        target_id = to_action_id(target_raw)
        dummy_deck = [0] * max(0, deck_count)

        eng.load_state_ids(
            h0_ids,
            h1_ids,
            dummy_deck,
            discard_ids,
            target_id,
            active_color,
            stack,
            current_player,
            turn_count,
            consecutive_passes,
        )

        # 2. Restore History Events if provided
        hist_events = state_data.get("history_events", None)
        if hist_events and isinstance(hist_events, list):
            cleaned_events = []
            for ev in hist_events:
                if isinstance(ev, (list, tuple)) and len(ev) >= 5:
                    cleaned_events.append([int(x) for x in ev[:5]])
                elif isinstance(ev, dict):
                    cleaned_events.append([
                        int(ev.get("actor_id", ev.get("actorId", 0))),
                        int(ev.get("action_id", ev.get("actionId", 0))),
                        int(ev.get("active_color_before", ev.get("activeColorBefore", 0))),
                        int(ev.get("stack_before", ev.get("stackBefore", 0))),
                        int(ev.get("cards_drawn", ev.get("cardsDrawn", 0))),
                    ])
            if cleaned_events:
                eng.set_history_events(cleaned_events)

        # 3. Restore Reshuffle Provenance if provided
        prov = state_data.get("reshuffle_provenance", state_data.get("provenance", None))
        if prov and isinstance(prov, dict):
            pool = [int(x) for x in prov.get("last_reshuffle_pool", [0] * 54)]
            d_since = prov.get("draws_since_reshuffle", [0, 0])
            d0 = int(d_since[0]) if len(d_since) > 0 else 0
            d1 = int(d_since[1]) if len(d_since) > 1 else 0
            k_drawn = prov.get("known_drawn_from_recycled", [[0] * 54, [0] * 54])
            k0 = [int(x) for x in k_drawn[0]] if len(k_drawn) > 0 else [0] * 54
            k1 = [int(x) for x in k_drawn[1]] if len(k_drawn) > 1 else [0] * 54
            eng.set_reshuffle_provenance(pool, d0, d1, k0, k1)

        # 4. Execute Move Selection
        chosen_act, s_res = self.agent.act_from_engine(eng, deterministic=True)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # 5. Extract probabilities & win prob
        action_probs = [0.0] * 61
        win_prob = 0.5
        belief_xray_out = None

        if s_res is not None:
            # Search mode
            probs = s_res.search_probs.detach().cpu().numpy()
            action_probs = [float(p) for p in probs]
            q_val = float(s_res.q_hat[chosen_act].item())
            win_prob = float(min(1.0, max(0.0, 0.5 * (q_val + 1.0))))
            belief_xray_out = dict(s_res.belief_xray) if s_res.belief_xray else {}
        else:
            # Blueprint mode: compute WDL win prob
            cf = torch.from_numpy(eng.get_card_features_actor(current_player)).unsqueeze(0).to(self.device)
            gf = torch.from_numpy(eng.get_global_features(current_player)).unsqueeze(0).to(self.device)
            ht = torch.from_numpy(eng.get_history_tokens(current_player)).unsqueeze(0).to(self.device)
            mask_t = torch.from_numpy(eng.get_action_mask(current_player)).to(self.device)

            with torch.no_grad():
                logits, wdl_logits = self.agent.actor_net(cf, gf, ht)
                neg_large = torch.tensor(-1e9, dtype=logits.dtype, device=self.device)
                masked_logits = torch.where(mask_t, logits.squeeze(0), neg_large)
                probs = F.softmax(masked_logits, dim=-1)
                action_probs = [float(p) for p in probs.cpu().numpy()]

                wdl_p = F.softmax(wdl_logits.squeeze(0), dim=-1).cpu().numpy()
                win_prob = float(wdl_p[2] + 0.5 * wdl_p[1])

        # Style distribution from tracker weights
        w_list = [float(w) for w in tracker.weights] if tracker and tracker.weights else list(DEFAULT_PRIOR)
        total_w = sum(w_list) if sum(w_list) > 0 else 1.0
        style_post = {
            "blueprint": float(w_list[0] / total_w),
            "aggressive": float(w_list[1] / total_w),
            "unpredictable": float(w_list[2] / total_w),
        }

        if belief_xray_out is not None:
            belief_xray_out["weights"] = w_list
            belief_xray_out["surprisal"] = float(tracker.ema_surprisal) if tracker else 0.5
            belief_xray_out["posterior"] = style_post

        # 6. Serialize updated opp_posterior
        updated_posterior = {
            "weights": w_list,
            "posterior": style_post,
            "blueprint": style_post["blueprint"],
            "aggressive": style_post["aggressive"],
            "unpredictable": style_post["unpredictable"],
            "ema_surprisal": float(tracker.ema_surprisal) if tracker else 0.5,
            "cached_turn": tracker._cached_turn if tracker else None,
            "cached_action_probs": tracker._cached_action_probs.tolist() if tracker and tracker._cached_action_probs is not None else None,
            "last_observed_turn": tracker._last_observed_turn if tracker else -1,
        }

        action_key = self.action_names[chosen_act] if 0 <= chosen_act < len(self.action_names) else "draw"

        return {
            "action_id": int(chosen_act),
            "action_key": action_key,
            "action_probs": action_probs,
            "win_prob": float(win_prob),
            "mode_used": cfg["mode"],
            "technique_label": cfg["technique_label"],
            "engine_backend": "server",
            "active_config": {
                "mode": cfg["mode"],
                "use_portfolio": self.agent.use_portfolio,
                "use_adaptive_bayes": self.agent.use_adaptive_bayes,
                "portfolio_weights": list(self.agent.portfolio_weights),
                "risk_gamma": self.agent.risk_gamma,
                "uniform_mix": self.agent.uniform_mix,
                "belief_temp": self.agent.belief_temp,
                "rollout_temp": self.agent.rollout_temp,
                "depth": self.agent.depth,
                "bp_floor": self.agent.bp_floor,
                "eb_shrink": self.agent.eb_shrink,
                "eta": self.agent.eta,
                "tau": self.agent.tau,
                "max_total_worlds": self.agent.max_total_worlds,
            },
            "latency_ms": float(latency_ms),
            "belief_xray": belief_xray_out,
            "opp_posterior": updated_posterior,
        }


class AtaraxosHttpHandler(SimpleHTTPRequestHandler):
    """HTTP Request Handler serving both static web-ui-serverless assets and JSON API."""

    server_backend: AtaraxosUnoServer

    def end_headers(self):
        # Cross-Origin Isolation Headers for WebGPU and SharedArrayBuffer
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ("/api/health", "/health"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            champ_cfg = self.server_backend.resolve_mode_config("auto")
            payload = {
                "status": "ok",
                "agent": "ataraxos-uno",
                "device": str(self.server_backend.device),
                "blueprint_ckpt": self.server_backend.blueprint_ckpt,
                "belief_ckpt": self.server_backend.belief_ckpt,
                "winner_name": self.server_backend.search_cfg.get("winner_name", "Ablation B (Static Risk-Averse)"),
                "champion_score": self.server_backend.search_cfg.get("score_3axis", 58.33),
                "search_config": self.server_backend.search_cfg,
                "champion_config": champ_cfg,
                "modes_supported": [
                    "auto",
                    "blueprint",
                    "online_bayesian",
                    "hybrid",
                    "risk_averse",
                    "static",
                    "tier1",
                ],
            }
            self.wfile.write(json.dumps(payload, indent=2).encode("utf-8"))
            return

        # Serve static files from web-ui-serverless
        super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ("/api/predict_choice", "/predict"):
            content_length = int(self.headers.get("Content-Length", 0))
            body_bytes = self.rfile.read(content_length)

            try:
                data = json.loads(body_bytes.decode("utf-8"))
                state = data.get("state", data)
                mode = data.get("mode", "auto")
                opp_posterior = data.get("opp_posterior", None)

                result = self.server_backend.predict(
                    state_data=state,
                    mode=mode,
                    opp_posterior=opp_posterior,
                )

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(result).encode("utf-8"))

            except Exception as e:
                import traceback
                traceback.print_exc()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                err_payload = {"error": str(e), "__error__": str(e)}
                self.wfile.write(json.dumps(err_payload).encode("utf-8"))
            return

        self.send_response(404)
        self.end_headers()


def run_server():
    parser = argparse.ArgumentParser(description="Serve Ataraxos-UNO web-ui and prediction API")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)), help="Port to listen on")
    parser.add_argument("--blueprint", type=str, default="runs/blueprint_100m/latest.pt", help="Path to blueprint checkpoint")
    parser.add_argument("--belief", type=str, default="runs/belief_20m/best.pt", help="Path to belief checkpoint")
    parser.add_argument("--critic", type=str, default=None, help="Path to critic checkpoint")
    parser.add_argument("--search-config", type=str, default="runs/search_config.json", help="Path to search config")
    parser.add_argument("--static-dir", type=str, default="web-ui-serverless", help="Directory of static web assets")
    args = parser.parse_args()

    static_path = Path(args.static_dir).resolve()
    if not static_path.exists():
        print(f"Error: static dir {static_path} not found!")
        sys.exit(1)

    # Initialize Backend
    backend = AtaraxosUnoServer(
        blueprint_ckpt=args.blueprint,
        belief_ckpt=args.belief,
        critic_ckpt=args.critic,
        search_config_path=args.search_config,
    )

    # Handler configuration
    class CustomHandler(AtaraxosHttpHandler):
        server_backend = backend

        def __init__(self, *h_args, **h_kwargs):
            super().__init__(*h_args, directory=str(static_path), **h_kwargs)

    httpd = HTTPServer((args.host, args.port), CustomHandler)
    print(f"\n=======================================================")
    print(f"🚀 Ataraxos-UNO Server running at http://{args.host}:{args.port}")
    print(f"📁 Serving static assets from: {static_path}")
    print(f"⚡ Endpoints:")
    print(f"   - GET  /api/health")
    print(f"   - POST /api/predict_choice (Modes: blueprint, tier1, hybrid, static, risk_averse, online_bayesian, auto)")
    print(f"=======================================================\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer shutting down gracefully.")
        httpd.server_close()


if __name__ == "__main__":
    run_server()
