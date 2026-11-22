#!/usr/bin/env python3
"""Automated integration test for scripts/serve_model.py.

Verifies:
1. Static asset serving (index.html, JS, CSS, ONNX model headers)
2. GET /api/health
3. POST /api/predict_choice across all 7 modes:
   - blueprint
   - tier1
   - hybrid
   - static
   - risk_averse
   - online_bayesian
   - auto
4. Stateless opp_posterior round-trip pass-through and belief_xray payload.
"""

import json
import socket
import subprocess
import sys
import time
import urllib.request
import urllib.error


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def make_sample_state():
    """Create a realistic state payload matching JS game-engine format."""
    # player 0 hand: red_1, red_2, green_5, blue_skip, wild
    # player 1 hand: yellow_3, yellow_7, plus_two_red, wild_draw_four
    return {
        "hands_0": [1, 2, 18, 38, 52],
        "hands_1": [30, 34, 40, 53],
        "deck_count": 80,
        "discard_pile": [0],
        "target": 0,  # red_0
        "active_color": 0,  # red
        "stack": 0,
        "current_player": 1,  # bot is player 1
        "turn_count": 5,
        "consecutive_passes": 0,
        "history_events": [
            [0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 2, 0, 0, 0],
            [1, 3, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        "reshuffle_provenance": {
            "last_reshuffle_pool": [0] * 54,
            "draws_since_reshuffle": [0, 0],
            "known_drawn_from_recycled": [[0] * 54, [0] * 54],
        },
    }


def main():
    port = find_free_port()
    print(f"Starting serve_model.py on port {port}...")

    cmd = [
        sys.executable,
        "scripts/serve_model.py",
        "--port",
        str(port),
        "--blueprint",
        "runs/blueprint_100m/latest.pt",
        "--belief",
        "runs/belief_20m/best.pt",
        "--search-config",
        "runs/search_config.json",
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    base_url = f"http://127.0.0.1:{port}"
    server_ready = False

    try:
        # Wait up to 25s for server to start
        for _ in range(50):
            time.sleep(0.5)
            try:
                with urllib.request.urlopen(f"{base_url}/api/health", timeout=1) as resp:
                    if resp.status == 200:
                        data = json.loads(resp.read().decode())
                        if data.get("status") == "ok":
                            server_ready = True
                            print(f"Server is ready! Winner: {data.get('winner_name')}")
                            # Assert Health endpoint reports exact Milestone 8 Champion
                            assert data.get("winner_name") == "Ablation B (Static Risk-Averse)", f"Unexpected winner: {data.get('winner_name')}"
                            champ_cfg = data.get("champion_config", {})
                            assert champ_cfg.get("use_portfolio") is True
                            assert champ_cfg.get("use_adaptive_bayes") is False
                            assert champ_cfg.get("risk_gamma") == 4.0
                            assert champ_cfg.get("uniform_mix") == 0.12
                            assert champ_cfg.get("belief_temp") == 1.15
                            assert champ_cfg.get("eta") == 8.0
                            assert champ_cfg.get("depth") == 14
                            assert champ_cfg.get("bp_floor") == 0.03
                            assert champ_cfg.get("eb_shrink") == 1.0
                            print("✓ GET /api/health validated: exact Milestone 8 Champion (Ablation B)")
                            break
            except Exception:
                pass
            if proc.poll() is not None:
                out, err = proc.communicate()
                print("Server failed to start:\nSTDOUT:\n", out, "\nSTDERR:\n", err)
                sys.exit(1)

        if not server_ready:
            print("Server timed out starting.")
            sys.exit(1)

        print("\n--- 1. Testing Static File Serving ---")
        with urllib.request.urlopen(f"{base_url}/", timeout=3) as resp:
            content = resp.read().decode()
            assert resp.status == 200
            assert "ataraxos_actor.onnx" in content or "ataraxos" in content.lower()
            print("✓ GET / (index.html) successfully served")

        req = urllib.request.Request(f"{base_url}/model/ataraxos_actor.onnx", method="HEAD")
        with urllib.request.urlopen(req, timeout=3) as resp:
            assert resp.status == 200
            cl = int(resp.headers.get("Content-Length", 0))
            assert cl > 8_000_000, f"Expected >8MB, got {cl}"
            print(f"✓ GET /model/ataraxos_actor.onnx served ({cl} bytes)")

        print("\n--- 2. Testing POST /api/predict_choice Across Modes ---")
        sample_state = make_sample_state()

        modes_to_test = [
            "blueprint",
            "tier1",
            "hybrid",
            "static",
            "risk_averse",
            "online_bayesian",
            "auto",
        ]

        test_posterior = {
            "blueprint": 0.6,
            "aggressive": 0.2,
            "unpredictable": 0.2,
        }

        for m in modes_to_test:
            t0 = time.perf_counter()
            payload = {
                "state": sample_state,
                "mode": m,
                "opp_posterior": test_posterior,
            }
            body = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                f"{base_url}/api/predict_choice",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                assert resp.status == 200
                res = json.loads(resp.read().decode())
                elapsed = (time.perf_counter() - t0) * 1000

                assert "action_id" in res, f"Missing action_id: {res}"
                assert "action_key" in res, f"Missing action_key: {res}"
                assert "action_probs" in res, f"Missing action_probs: {res}"
                assert "win_prob" in res, f"Missing win_prob: {res}"
                assert "mode_used" in res, f"Missing mode_used: {res}"
                assert "belief_xray" in res, f"Missing belief_xray: {res}"
                assert "active_config" in res, f"Missing active_config: {res}"

                # Strict validation for mode=auto (Champion Ablation B)
                if m in ("auto", "risk_averse", "static"):
                    ac = res["active_config"]
                    assert ac["mode"] == "mmd_search"
                    assert ac["use_portfolio"] is True
                    assert ac["use_adaptive_bayes"] is False
                    assert ac["risk_gamma"] == 4.0
                    assert ac["uniform_mix"] == 0.12
                    assert ac["belief_temp"] == 1.15
                    assert ac["eta"] == 8.0
                    assert ac["depth"] == 14
                    assert ac["bp_floor"] == 0.03
                    assert ac["eb_shrink"] == 1.0

                # Verify stateless opp_posterior pass-through for adaptive modes
                if m in ("online_bayesian", "hybrid"):
                    assert "opp_posterior" in res and res["opp_posterior"] is not None
                    opp_post = res["opp_posterior"]
                    assert "blueprint" in opp_post

                print(
                    f"✓ Mode '{m:<15}' -> action: {res['action_key']:<15} | "
                    f"win_prob: {res['win_prob']:.3f} | mode_used: {res['mode_used']:<12} | "
                    f"latency: {elapsed:.1f}ms"
                )

        print("\n--- 3. Testing Sequential No-Parameter-Bleed Across Mode Switches ---")
        # Call blueprint -> online_bayesian -> hybrid -> auto
        bleed_sequence = ["blueprint", "online_bayesian", "hybrid", "auto"]
        final_res = None
        for step_mode in bleed_sequence:
            payload = {
                "state": sample_state,
                "mode": step_mode,
                "opp_posterior": test_posterior,
            }
            body = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                f"{base_url}/api/predict_choice",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                assert resp.status == 200
                final_res = json.loads(resp.read().decode())

        # Assert final auto call has exact Ablation B parameters (no bleed from online_bayesian/hybrid!)
        assert final_res is not None
        final_ac = final_res["active_config"]
        assert final_ac["mode"] == "mmd_search", f"Mode bleed: {final_ac['mode']} != mmd_search"
        assert final_ac["use_portfolio"] is True
        assert final_ac["use_adaptive_bayes"] is False, f"Adaptive Bayes bled over: {final_ac['use_adaptive_bayes']}"
        assert final_ac["risk_gamma"] == 4.0
        assert final_ac["uniform_mix"] == 0.12, f"uniform_mix bled over: {final_ac['uniform_mix']}"
        assert final_ac["belief_temp"] == 1.15, f"belief_temp bled over: {final_ac['belief_temp']}"
        assert final_ac["eta"] == 8.0, f"eta bled over: {final_ac['eta']}"
        assert final_ac["depth"] == 14
        print("✓ Sequential No-Parameter-Bleed Test PASSED: final 'auto' call retains 100% exact Ablation B parameters!")

        print("\n--- 4. Testing ONNX vs PyTorch Blueprint 100M Parity ---")
        import torch
        import onnxruntime as ort
        import numpy as np
        from uno_ataraxos.networks import UnoActorNet
        from uno_ataraxos import UnoEngine

        actor = UnoActorNet()
        ckpt = torch.load("runs/blueprint_100m/latest.pt", map_location="cpu")
        raw_state = ckpt.get("actor_state_dict", ckpt.get("actor", ckpt))
        clean_state = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in raw_state.items()}
        actor.load_state_dict(clean_state)
        actor.eval()

        session = ort.InferenceSession("web-ui-serverless/model/ataraxos_actor.onnx")

        eng = UnoEngine(42)
        eng.reset()
        matches = 0
        max_diff = 0.0

        for _ in range(100):
            curr_p = int(eng.current_player)
            cf = eng.get_card_features_actor(curr_p)
            gf = eng.get_global_features(curr_p)
            ht = eng.get_history_tokens(curr_p)
            mask = eng.get_action_mask(curr_p)

            with torch.no_grad():
                pt_logits, _ = actor(
                    torch.from_numpy(cf).unsqueeze(0),
                    torch.from_numpy(gf).unsqueeze(0),
                    torch.from_numpy(ht).unsqueeze(0),
                )
                pt_mask_t = torch.from_numpy(mask).bool()
                pt_logits_masked = torch.where(pt_mask_t, pt_logits.squeeze(0), torch.tensor(-1e9))
                pt_probs = torch.softmax(pt_logits_masked, dim=-1).numpy()
                pt_act = int(np.argmax(pt_probs))

            feeds = {
                "card_features": np.expand_dims(cf, 0).astype(np.float32),
                "global_features": np.expand_dims(gf, 0).astype(np.float32),
                "history_tokens": np.expand_dims(ht, 0).astype(np.int64),
                "legal_mask": np.expand_dims(mask, 0).astype(np.float32),
            }
            onnx_out = session.run(None, feeds)
            onnx_probs = onnx_out[0][0]
            onnx_act = int(np.argmax(onnx_probs))

            if pt_act == onnx_act:
                matches += 1
            diff = float(np.max(np.abs(pt_probs - onnx_probs)))
            max_diff = max(max_diff, diff)

            legal_acts = eng.get_legal_action_ids(curr_p)
            eng.step(legal_acts[0])
            if eng.is_over:
                eng.reset()

        assert matches == 100, f"ONNX argmax matches {matches}/100"
        assert max_diff < 1e-4, f"ONNX max diff {max_diff:.6e} >= 1e-4"
        print(f"✓ ONNX vs PyTorch Blueprint 100M Parity PASSED: {matches}/100 matches | Max diff: {max_diff:.6e} < 1e-4")

        print("\n>>> ALL SERVER, API & ONNX INTEGRATION TESTS PASSED SUCCESSFULLY! <<<\n")

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_server_api():
    """Pytest entrypoint."""
    main()


if __name__ == "__main__":
    main()
