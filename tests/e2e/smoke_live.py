#!/usr/bin/env python3
"""OPT-IN live smoke test against the RUNNING vllm-omni stack.

Never run by default and never mutates deployments — read-only checks plus
two throwaway chat completions. Guarded by RUN_LIVE=1.

Usage:
    RUN_LIVE=1 [API_KEY=sk-...] [ROUTER_URL=http://127.0.0.1:11434] \
        [CENTRAL_URL=http://127.0.0.1:8080] python tests/e2e/smoke_live.py

Environment:
    RUN_LIVE     must be "1" or the script exits 0 doing nothing
    ROUTER_URL   router gateway base URL   (default http://127.0.0.1:11434)
    CENTRAL_URL  central dashboard base URL (default http://127.0.0.1:8080)
    API_KEY      optional; sent as "Authorization: Bearer <API_KEY>"
    SMOKE_MODEL  optional; model id to use. Defaults to the model_id of the
                 first healthy worker reported by the router.

Checks:
    1. router /workers returns at least one healthy worker
    2. non-streaming chat completion through the gateway returns 200 + choices
    3. streaming chat completion yields >=1 SSE chunk and terminates with [DONE]
    4. central /api/prometheus_stats returns rps_window consistent with
       requests_window over the selected window

Exit code 0 iff every check passed. Prints a PASS/FAIL table.
"""

import json
import os
import sys

import httpx

ROUTER_URL = os.environ.get("ROUTER_URL", "http://127.0.0.1:11434").rstrip("/")
CENTRAL_URL = os.environ.get("CENTRAL_URL", "http://127.0.0.1:8080").rstrip("/")
API_KEY = os.environ.get("API_KEY", "")

HEADERS = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}

results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f" - {detail}" if detail else ""))


def check_workers(client: httpx.Client) -> str | None:
    """Returns a model id to use for the inference checks, or None."""
    try:
        r = client.get(f"{ROUTER_URL}/workers", headers=HEADERS)
        if r.status_code != 200:
            record("router /workers reachable", False, f"HTTP {r.status_code}")
            return None
        body = r.json()
        workers = body.get("workers", [])
        healthy = [w for w in workers if w.get("is_healthy")]
        record(
            "router /workers non-empty",
            len(healthy) > 0,
            f"{len(healthy)} healthy / {len(workers)} total",
        )
        if not healthy:
            return None
        model = os.environ.get("SMOKE_MODEL") or healthy[0].get("model_id")
        if not model or model == "unknown":
            models = sorted({w.get("model_id") for w in healthy} - {None, "unknown"})
            model = models[0] if models else None
        record("resolved a routable model id", bool(model), str(model))
        return model
    except Exception as e:
        record("router /workers reachable", False, repr(e))
        return None


def check_non_stream(client: httpx.Client, model: str) -> None:
    try:
        r = client.post(
            f"{ROUTER_URL}/v1/chat/completions",
            headers=HEADERS,
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Say OK and nothing else."}],
                "max_tokens": 8,
                "stream": False,
            },
        )
        ok = r.status_code == 200 and bool(r.json().get("choices"))
        record("non-stream chat completion", ok, f"HTTP {r.status_code}")
    except Exception as e:
        record("non-stream chat completion", False, repr(e))


def check_stream(client: httpx.Client, model: str) -> None:
    try:
        chunks = 0
        done = False
        with client.stream(
            "POST",
            f"{ROUTER_URL}/v1/chat/completions",
            headers=HEADERS,
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Count to three."}],
                "max_tokens": 16,
                "stream": True,
            },
        ) as resp:
            if resp.status_code != 200:
                record("stream chat completion", False, f"HTTP {resp.status_code}")
                return
            for line in resp.iter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    done = True
                    break
                chunks += 1
        record(
            "stream chat completion (chunks + [DONE])",
            chunks >= 1 and done,
            f"{chunks} chunks, done={done}",
        )
    except Exception as e:
        record("stream chat completion (chunks + [DONE])", False, repr(e))


def check_central_stats(client: httpx.Client) -> None:
    try:
        window = 60
        r = client.get(
            f"{CENTRAL_URL}/api/prometheus_stats", params={"window": window}
        )
        if r.status_code != 200:
            record("central /api/prometheus_stats", False, f"HTTP {r.status_code}")
            return
        body = r.json()
        if body.get("incomplete"):
            record("central /api/prometheus_stats", False, "still warming up")
            return
        rps = body.get("rps_window")
        req_w = body.get("requests_window")
        win = body.get("window_seconds", window)
        record("central stats present", rps is not None and req_w is not None,
               f"rps_window={rps} requests_window={req_w} window={win}s")
        # Consistency: rps = requests_window / duration where the sample
        # duration is <= window (and realistically >= one 5s scrape period).
        if req_w and req_w > 0:
            lower = req_w / win * 0.9          # duration can't exceed the window
            upper = req_w / 1.0                 # duration is at least ~1s
            ok = lower <= rps <= upper
            record(
                "rps_window consistent with requests_window/window",
                ok,
                f"{lower:.2f} <= {rps} <= {upper:.2f}",
            )
        else:
            record(
                "rps_window consistent with requests_window/window",
                (rps or 0) <= 0.5,
                f"no window traffic, rps={rps}",
            )
    except Exception as e:
        record("central /api/prometheus_stats", False, repr(e))


def main() -> int:
    if os.environ.get("RUN_LIVE") != "1":
        print("smoke_live: RUN_LIVE != 1, skipping (this is an opt-in live test).")
        return 0

    print(f"Live smoke against router={ROUTER_URL} central={CENTRAL_URL}")
    with httpx.Client(timeout=30.0, verify=False) as client:
        model = check_workers(client)
        if model:
            check_non_stream(client, model)
            check_stream(client, model)
        check_central_stats(client)

    print("\n==== SUMMARY " + "=" * 47)
    width = max(len(name) for name, _, _ in results) + 2
    failed = 0
    for name, ok, detail in results:
        status = "PASS" if ok else "FAIL"
        failed += 0 if ok else 1
        print(f"  {name:<{width}} {status}   {detail}")
    print("=" * 60)
    print(f"  {len(results) - failed}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
