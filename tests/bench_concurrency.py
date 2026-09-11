#!/usr/bin/env python3
"""Concurrent chat-completion throughput benchmark.

Fires `--total` identical /v1/chat/completions requests at up to
`--concurrency` in flight against ONE endpoint and reports throughput
(req/s, completion tokens/s), latency percentiles, and error breakdown.

Hit vLLM workers DIRECTLY (https://host:port, self-signed → --insecure) for a
clean 1:1 comparison; going through the router would spread load across
replicas and measure the router, not the model.

    ulimit -n 16384   # 2048+ concurrent sockets need a raised fd limit
    python tests/bench_concurrency.py --url https://143.248.74.105:61002 \
        --model google/gemma-4-E4B-it --concurrency 2048 --total 4096 --insecure
"""
import argparse
import asyncio
import json
import statistics
import sys
import time

import httpx


def pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1)))))
    return xs[k]


async def run(a):
    payload = {
        "model": a.model,
        "messages": [{"role": "user", "content": a.prompt}],
        "max_tokens": a.max_tokens,
        "temperature": 0,
    }
    sem = asyncio.Semaphore(a.concurrency)
    lat, ttfb = [], []
    errs = {}
    stats = {"ok": 0, "completion_tokens": 0, "prompt_tokens": 0}
    limits = httpx.Limits(
        max_connections=a.concurrency + 64,
        max_keepalive_connections=a.concurrency,
    )
    async with httpx.AsyncClient(
        verify=not a.insecure, timeout=a.timeout, limits=limits
    ) as client:

        async def one():
            async with sem:
                t0 = time.perf_counter()
                try:
                    r = await client.post(f"{a.url}/v1/chat/completions", json=payload)
                    dt = time.perf_counter() - t0
                    if r.status_code == 200:
                        stats["ok"] += 1
                        lat.append(dt)
                        u = r.json().get("usage", {}) or {}
                        stats["completion_tokens"] += u.get("completion_tokens", 0) or 0
                        stats["prompt_tokens"] += u.get("prompt_tokens", 0) or 0
                    else:
                        k = f"HTTP {r.status_code}"
                        errs[k] = errs.get(k, 0) + 1
                except Exception as e:  # timeouts, connection errors, etc.
                    k = type(e).__name__
                    errs[k] = errs.get(k, 0) + 1

        # Warm-up: a single request so model/graph capture isn't in the timed window.
        if a.warmup:
            try:
                await client.post(f"{a.url}/v1/chat/completions", json=payload)
            except Exception:
                pass

        wall0 = time.perf_counter()
        await asyncio.gather(*(one() for _ in range(a.total)))
        wall = time.perf_counter() - wall0

    ok = stats["ok"]
    out = {
        "url": a.url,
        "model": a.model,
        "concurrency": a.concurrency,
        "total": a.total,
        "max_tokens": a.max_tokens,
        "wall_s": round(wall, 2),
        "ok": ok,
        "errors": errs,
        "req_per_s": round(ok / wall, 2) if wall else None,
        "completion_tok_per_s": round(stats["completion_tokens"] / wall, 1) if wall else None,
        "avg_completion_tokens": round(stats["completion_tokens"] / ok, 1) if ok else None,
        "latency_s": {
            "p50": round(pct(lat, 50), 2) if lat else None,
            "p95": round(pct(lat, 95), 2) if lat else None,
            "p99": round(pct(lat, 99), 2) if lat else None,
            "max": round(max(lat), 2) if lat else None,
            "mean": round(statistics.fmean(lat), 2) if lat else None,
        },
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True, help="base URL, e.g. https://host:port")
    ap.add_argument("--model", required=True)
    ap.add_argument("--concurrency", type=int, default=2048)
    ap.add_argument("--total", type=int, default=4096)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--prompt", default="Explain in three sentences why the sky is blue.")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--insecure", action="store_true", help="skip TLS verify (self-signed workers)")
    ap.add_argument("--no-warmup", dest="warmup", action="store_false")
    a = ap.parse_args()
    sys.exit(asyncio.run(run(a)))


if __name__ == "__main__":
    main()
