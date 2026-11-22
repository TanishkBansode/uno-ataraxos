import asyncio
import json
import subprocess
import time
import requests
import websockets

async def inspect_url(target_url):
    print(f"\n=======================================================")
    print(f"Testing URL with Headless Chrome: {target_url}")
    print(f"=======================================================")

    chrome_cmd = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "--headless=new",
        "--remote-debugging-port=9222",
        "--user-data-dir=/tmp/chrome-cdp-profile",
        "--disable-gpu",
        "--no-first-run",
        "--no-default-browser-check",
    ]

    proc = subprocess.Popen(chrome_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.5)

    try:
        # Get page websocket endpoint
        resp = requests.get("http://localhost:9222/json")
        pages = resp.json()
        page_ws = pages[0]["webSocketDebuggerUrl"]
        print(f"Connected to CDP page: {page_ws}")

        async with websockets.connect(page_ws) as ws:
            msg_id = 1

            async def send(method, params=None):
                nonlocal msg_id
                cmd = {"id": msg_id, "method": method, "params": params or {}}
                msg_id += 1
                await ws.send(json.dumps(cmd))

            # Enable console, runtime, network, page
            await send("Console.enable")
            await send("Runtime.enable")
            await send("Network.enable")
            await send("Page.enable")

            # Navigate
            await send("Page.navigate", {"url": target_url})

            console_errors = []
            exceptions = []
            failed_requests = []
            logs = []

            # Listen for events for 8 seconds
            start_time = time.time()
            while time.time() - start_time < 8.0:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    evt = json.loads(msg)
                    method = evt.get("method", "")

                    if method == "Runtime.exceptionThrown":
                        exc = evt["params"]["exceptionDetails"]
                        txt = exc.get("text", "")
                        desc = exc.get("exception", {}).get("description", "")
                        exceptions.append(f"{txt}: {desc}")
                        print(f"[EXCEPTION] {txt}: {desc}")

                    elif method == "Console.messageAdded":
                        m = evt["params"]["message"]
                        level = m.get("level")
                        txt = m.get("text", "")
                        logs.append(f"[{level}] {txt}")
                        print(f"[CONSOLE {level}] {txt}")
                        if level == "error":
                            console_errors.append(txt)

                    elif method == "Network.loadingFailed":
                        p = evt["params"]
                        err = p.get("errorText", "")
                        req_id = p.get("requestId", "")
                        failed_requests.append(f"Request {req_id} failed: {err}")
                        print(f"[NET FAIL] {err}")

                    elif method == "Network.responseReceived":
                        resp = evt["params"]["response"]
                        status = resp.get("status")
                        url = resp.get("url")
                        if status >= 400:
                            failed_requests.append(f"{status} on {url}")
                            print(f"[HTTP {status}] {url}")

                except asyncio.TimeoutError:
                    pass

            print("\n--- SUMMARY ---")
            print(f"Exceptions: {len(exceptions)}")
            print(f"Console Errors: {len(console_errors)}")
            print(f"Failed Requests: {len(failed_requests)}")
            return exceptions, console_errors, failed_requests

    finally:
        proc.terminate()
        proc.wait()

if __name__ == "__main__":
    asyncio.run(inspect_url("https://tanishkbansode.github.io/uno-ataraxos/"))
