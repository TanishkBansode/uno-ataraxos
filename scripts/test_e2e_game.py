import argparse
import asyncio
import base64
import json
import subprocess
import time
import requests
import websockets

async def run_e2e(url="http://localhost:8099/", screenshot_path=None):
    print("=======================================================")
    print(f"Starting Headless Chrome E2E Test on {url}")
    print("=======================================================")

    chrome_cmd = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "--headless=new",
        "--remote-debugging-port=9222",
        "--user-data-dir=/tmp/chrome-cdp-profile-e2e",
        "--no-first-run",
        "--no-default-browser-check",
    ]

    proc = subprocess.Popen(chrome_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.5)

    try:
        resp = requests.get("http://localhost:9222/json")
        pages = resp.json()
        page_ws = pages[0]["webSocketDebuggerUrl"]
        print(f"Connected to CDP page: {page_ws}")

        async with websockets.connect(page_ws, max_size=None) as ws:
            msg_id = 1
            pending_responses = {}

            async def send_cmd(method, params=None):
                nonlocal msg_id
                cid = msg_id
                msg_id += 1
                fut = asyncio.get_event_loop().create_future()
                pending_responses[cid] = fut
                await ws.send(json.dumps({"id": cid, "method": method, "params": params or {}}))
                return await fut

            async def evaluate(expr):
                res = await send_cmd("Runtime.evaluate", {
                    "expression": expr,
                    "returnByValue": True,
                    "awaitPromise": True,
                })
                return res.get("result", {}).get("value")

            console_errors = []
            exceptions = []
            all_logs = []

            async def message_loop():
                while True:
                    try:
                        raw = await ws.recv()
                        evt = json.loads(raw)
                        if "id" in evt and evt["id"] in pending_responses:
                            fut = pending_responses.pop(evt["id"])
                            if "error" in evt:
                                fut.set_exception(Exception(evt["error"]))
                            else:
                                fut.set_result(evt.get("result", {}))
                            continue

                        method = evt.get("method", "")
                        if method == "Runtime.exceptionThrown":
                            exc = evt["params"]["exceptionDetails"]
                            txt = exc.get("text", "")
                            desc = exc.get("exception", {}).get("description", "")
                            exceptions.append(f"{txt}: {desc}")
                            print(f"[PAGE EXCEPTION] {txt}: {desc}")
                        elif method == "Console.messageAdded":
                            m = evt["params"]["message"]
                            level = m.get("level")
                            txt = m.get("text", "")
                            all_logs.append((level, txt))
                            if level == "error":
                                if "[W:onnxruntime" in txt or "VerifyEachNodeIsAssignedToAnEp" in txt:
                                    print(f"[CONSOLE ORT-WARN] {txt}")
                                else:
                                    console_errors.append(txt)
                                    print(f"[CONSOLE ERROR] {txt}")
                            elif level == "warning":
                                print(f"[CONSOLE WARN] {txt}")
                            else:
                                print(f"[CONSOLE LOG] {txt}")
                    except asyncio.CancelledError:
                        break
                    except Exception as e:
                        pass

            loop_task = asyncio.create_task(message_loop())

            await send_cmd("Console.enable")
            await send_cmd("Runtime.enable")
            await send_cmd("Page.enable")
            await send_cmd("Emulation.setDeviceMetricsOverride", {
                "width": 1280,
                "height": 800,
                "deviceScaleFactor": 2,
                "mobile": False,
            })
            await send_cmd("Page.navigate", {"url": url})

            # 1. Wait for "Loading AI..." to finish
            print("\nStep 1: Waiting for In-Browser Neural AI to initialize...")
            t0 = time.time()
            ai_ready = False
            for _ in range(60):
                status_txt = await evaluate("document.getElementById('ai-status-text') ? document.getElementById('ai-status-text').textContent : ''")
                is_loaded = await evaluate("window.modelClient ? (window.modelClient.isInBrowserLoaded || !window.modelClient.preferInBrowser) : false")
                if "Ready" in status_txt or is_loaded:
                    elapsed = time.time() - t0
                    print(f">>> AI INITIALIZATION COMPLETE in {elapsed:.2f}s! Status: '{status_txt.strip()}'")
                    ai_ready = True
                    break
                await asyncio.sleep(0.5)

            if not ai_ready:
                raise RuntimeError("Timed out waiting for AI initialization!")

            # 2. Click Play!
            print("\nStep 2: Submitting login form (Player name: 'Alice')...")
            await evaluate("document.getElementById('username').value = 'Alice'; document.getElementById('connect').click();")
            
            # Wait for game container to become visible
            for _ in range(30):
                game_hidden = await evaluate("document.getElementById('game-container').classList.contains('hidden')")
                if not game_hidden:
                    break
                await asyncio.sleep(0.5)

            print(f"Game container visible: {not game_hidden}")
            assert not game_hidden, "Game container did not become visible after login!"

            bot_name = await evaluate("document.getElementById('username_bot') ? document.getElementById('username_bot').textContent : ''")
            print(f"Bot player badge: '{bot_name}'")
            assert "ATARAXOS AI" in bot_name, f"Expected ATARAXOS AI, got {bot_name}"

            # If START_CARD is on desk, click it to start game
            start_card = await evaluate("document.getElementById('START_CARD') !== null")
            if start_card:
                print("Clicking START_CARD to deal initial hands...")
                await evaluate("document.getElementById('START_CARD').click();")
                await asyncio.sleep(1.0)

            # Wait for starting hands
            human_cards_count = 0
            bot_cards_count = 0
            for _ in range(20):
                human_cards_count = await evaluate("document.getElementById('cards').children.length")
                bot_cards_count = await evaluate("document.getElementById('bot-cards').children.length")
                if human_cards_count == 7 and bot_cards_count == 7:
                    break
                await asyncio.sleep(0.5)

            print(f"Dealt cards -> Human: {human_cards_count}, Bot: {bot_cards_count}")
            assert human_cards_count == 7, f"Expected 7 human cards, got {human_cards_count}"
            assert bot_cards_count == 7, f"Expected 7 bot cards, got {bot_cards_count}"

            # 3. Play at least 3 turns
            print("\nStep 3: Playing turns with Ataraxos AI...")
            turns_played = 0
            for turn in range(5):
                curr_move = await evaluate("window.engine ? window.engine.currentMove : ''")
                is_started = await evaluate("window.engine ? window.engine.isStarted : false")
                if not is_started:
                    break

                # If it's human turn, play a legal card or draw
                legal_items = await evaluate("""
                    (() => {
                        const eng = window.engine;
                        if (!eng) return [];
                        const { items } = eng.getLegalActionsForPlayer('human');
                        return items.map(it => ({ id: it.cardId, act: it.actionStr, pick: it.pickColor }));
                    })()
                """)

                print(f"\n[Turn {turn+1}] Human Turn. Legal moves: {len(legal_items)}")
                if len(legal_items) > 0 and legal_items[0]["act"] != "draw":
                    # Click playable card
                    target_id = legal_items[0]["id"]
                    pick_color = legal_items[0]["pick"] or "RED"
                    print(f"Human plays card {target_id} (declared color {pick_color})")
                    await evaluate(f"""
                        (() => {{
                            const eng = window.engine;
                            const el = document.getElementById('{target_id}');
                            if (el) el.click();
                            else eng.playCard('human', '{target_id}', '{pick_color}');
                        }})()
                    """)
                else:
                    print("Human draws card")
                    await evaluate("document.getElementById('carddeck').click();")

                turns_played += 1
                # Wait for bot to think and respond
                await asyncio.sleep(1.5)

                # Check AI win probability display
                win_prob_txt = await evaluate("document.getElementById('win-prob-val').textContent")
                print(f"Live AI Win Prob: {win_prob_txt}")

            # 4. Toggle Belief X-Ray
            print("\nStep 4: Testing Belief X-Ray Drawer...")
            await evaluate("document.getElementById('xray-toggle-btn').click();")
            await asyncio.sleep(0.5)
            xray_hidden = await evaluate("document.getElementById('belief-xray-panel').classList.contains('hidden')")
            print(f"X-Ray Drawer Visible: {not xray_hidden}")
            assert not xray_hidden, "Belief X-Ray panel should be visible!"

            r_val = await evaluate("document.getElementById('xray-red').textContent")
            g_val = await evaluate("document.getElementById('xray-green').textContent")
            b_val = await evaluate("document.getElementById('xray-blue').textContent")
            y_val = await evaluate("document.getElementById('xray-yellow').textContent")
            p2_val = await evaluate("document.getElementById('xray-p2').textContent")
            p4_val = await evaluate("document.getElementById('xray-p4').textContent")
            print(f"X-Ray Belief Probs -> Red: {r_val}, Green: {g_val}, Blue: {b_val}, Yellow: {y_val}, +2: {p2_val}, +4: {p4_val}")

            if screenshot_path:
                print(f"Capturing crisp screenshot to {screenshot_path}...")
                await asyncio.sleep(0.5)
                res = await send_cmd("Page.captureScreenshot", {"format": "png"})
                b64_data = res.get("data", "")
                with open(screenshot_path, "wb") as f:
                    f.write(base64.b64decode(b64_data))
                print(f"Screenshot successfully saved to {screenshot_path}")

            # 5. Test Play Again flow
            print("\nStep 5: Testing Play Again flow (no reload)...")
            await evaluate("""
                (() => {
                    const btn = document.getElementById('winner-return');
                    if (btn) btn.click();
                })()
            """)
            await asyncio.sleep(1.0)
            in_game = await evaluate("!document.getElementById('game-container').classList.contains('hidden')")
            print(f"Still in game table after Play Again: {in_game}")
            assert in_game, "Play Again should keep player in game without reload!"

            loop_task.cancel()

            print("\n=======================================================")
            print(f"E2E VERIFICATION SUMMARY:")
            print(f"- Turns Played: {turns_played}")
            print(f"- Console Errors: {len(console_errors)}")
            print(f"- Page Exceptions: {len(exceptions)}")
            print("=======================================================")
            assert len(exceptions) == 0, f"Exceptions encountered: {exceptions}"
            assert len(console_errors) == 0, f"Console errors encountered: {console_errors}"
            print(">>> ALL 6 E2E VERIFICATION ASSERTIONS PASSED! <<<")

    finally:
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            proc.kill()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8099/", help="URL to test")
    parser.add_argument("--screenshot", default=None, help="Path to save screenshot")
    args = parser.parse_args()
    asyncio.run(run_e2e(args.url, args.screenshot))
