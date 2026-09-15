#!/usr/bin/env python3
"""Concurrent chat-completion throughput benchmark.

Fires `--total` identical /v1/chat/completions requests at up to
`--concurrency` in flight against ONE endpoint and reports throughput
(req/s, completion tokens/s), latency percentiles, and error breakdown.

Transport: raw asyncio HTTP/1.1 with one persistent keep-alive connection per
in-flight slot. No httpx/aiohttp on purpose — httpx's connection pool does
O(pending × connections) bookkeeping on every completion, and at ~1000+
keep-alive connections to a single origin the CLIENT becomes the bottleneck
(measured: 37 req/s via router with httpx vs 125 req/s with this client, same
engines). Direct-to-worker runs spread over several processes hid that.

    ulimit -n 16384   # 2048+ concurrent sockets need a raised fd limit
    python tests/bench_concurrency.py --url http://localhost:11434 \
        --model google/gemma-4-E4B-it --concurrency 2048 --total 4096
    python tests/bench_concurrency.py --url https://143.248.74.105:61002 \
        --model openai/gpt-oss-120b --concurrency 2048 --total 4096 --insecure
"""
import argparse
import asyncio
import json
import ssl
import statistics
import sys
import time
from urllib.parse import urlsplit


def pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1)))))
    return xs[k]


class Endpoint:
    def __init__(self, url, insecure):
        u = urlsplit(url)
        self.tls = u.scheme == "https"
        self.host = u.hostname
        self.port = u.port or (443 if self.tls else 80)
        self.path_prefix = u.path.rstrip("/")
        self.ssl = None
        if self.tls:
            self.ssl = ssl.create_default_context()
            if insecure:
                self.ssl.check_hostname = False
                self.ssl.verify_mode = ssl.CERT_NONE

    def request_bytes(self, path, payload, api_key=None):
        body = json.dumps(payload).encode()
        auth = f"Authorization: Bearer {api_key}\r\n" if api_key else ""
        head = (
            f"POST {self.path_prefix}{path} HTTP/1.1\r\n"
            f"Host: {self.host}:{self.port}\r\n"
            "Content-Type: application/json\r\n"
            "Accept: application/json\r\n"
            f"{auth}"
            f"Content-Length: {len(body)}\r\n\r\n"
        ).encode()
        return head + body

    async def connect(self):
        return await asyncio.open_connection(self.host, self.port, ssl=self.ssl)


async def read_response(reader):
    """Minimal HTTP/1.1 response reader: status + content-length or chunked body."""
    head = await reader.readuntil(b"\r\n\r\n")
    lines = head.decode("latin-1").split("\r\n")
    status = int(lines[0].split(" ", 2)[1])
    headers = {}
    for line in lines[1:]:
        if ":" in line:
            k, v = line.split(":", 1)
            headers[k.strip().lower()] = v.strip()
    if headers.get("transfer-encoding", "").lower() == "chunked":
        chunks = []
        while True:
            size = int((await reader.readline()).split(b";")[0].strip() or b"0", 16)
            if size == 0:
                await reader.readuntil(b"\r\n")  # trailer terminator
                break
            chunks.append(await reader.readexactly(size))
            await reader.readexactly(2)  # CRLF after chunk
        body = b"".join(chunks)
    else:
        body = await reader.readexactly(int(headers.get("content-length", "0")))
    keep_alive = headers.get("connection", "").lower() != "close"
    return status, body, keep_alive


_WORDS = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi "
          "rho sigma tau upsilon phi chi psi omega").split()


def filler(seed, n_words):
    """Deterministic pseudo-text of n_words words, unique per seed (≈1.3 tok/word)."""
    out = []
    x = seed * 2654435761 % (2**32) or 1
    for _ in range(n_words):
        x = (x * 1103515245 + 12345) % (2**31)
        out.append(_WORDS[x % len(_WORDS)])
    return " ".join(out)


_HARMONY_SYSTEM = (
    "You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\n"
    "Current date: 2026-09-11\n\nReasoning: high\n\n# Valid channels: analysis, commentary, final. "
    "Channel must be included for every message."
)
_REVIEWER_DEVELOPER = "# Instructions\n\n" + (
    "You are a meticulous biomedical literature reviewer. You will be given a HYPOTHESIS about a "
    "molecular or clinical relationship, and the TITLE and ABSTRACT of one research article. Decide "
    "whether the abstract SUPPORTS, CONTRADICTS, or is NEUTRAL toward the hypothesis. Base the verdict "
    "only on evidence stated in the abstract; do not rely on outside knowledge of the field, and do not "
    "assume a result that is merely implied. Treat indirect evidence (upstream or downstream effects, "
    "correlations without mechanism, results in unrelated tissues or species) as weaker than direct "
    "experimental manipulation. If the abstract discusses the entities but reports no relationship, or "
    "reports a relationship in the opposite direction, choose the corresponding verdict. Confidence must "
    "be a number between 0 and 1 reflecting how directly the abstract addresses the hypothesis and how "
    "conclusive the reported evidence is. The rationale must be two to four sentences quoting or closely "
    "paraphrasing the decisive statements of the abstract, naming the assay or study design when given, "
    "and explaining any caveat that lowered the confidence. Do not include any text outside the JSON.\n\n"
    "OUTPUT FORMAT (JSON): {\"verdict\": \"supports\" | \"contradicts\" | \"neutral\", "
    "\"confidence\": <float 0-1>, \"rationale\": <string>}"
)
_GENES = "PDCD1 DNM1L TP53 EGFR BRCA1 MTOR AKT1 STAT3 VEGFA TNF IL6 MYC KRAS PTEN CDK4".split()


def believe_payload(a, i):
    """One request shaped like the `believe` pipeline job (see docs/engine-bench.md):
    Harmony system + fixed developer instructions (shared by every request),
    user = HYPOTHESIS (shared within a job) + TITLE + ~1,400-char ABSTRACT (unique),
    no max_tokens, no temperature, non-streaming, reasoning high."""
    g1, g2 = _GENES[(i // 1000) % len(_GENES)], _GENES[(i // 1000 + 3) % len(_GENES)]
    abstract = (
        f"Background: The role of {g1} in regulating {g2} remains incompletely understood. "
        "Methods: " + filler(10_000 + i, 125).capitalize() + ". "
        "Results: " + filler(20_000 + i, 125).capitalize() + ". "
        "Conclusions: " + filler(30_000 + i, 55).capitalize() + "."
    )
    user = (f"HYPOTHESIS:\n{g1} activates {g2}\n\nTITLE:\n{filler(40_000 + i, 12).capitalize()}\n\n"
            f"ABSTRACT:\n{abstract}\n\nAnswer using JSON.")
    if getattr(a, "merge_system", False):
        # Dynamo 1.4.2's Harmony renderer keeps only ONE of system/developer when both are
        # present (measured: system+developer -> 569 prompt tokens == developer dropped).
        # Merging them into a single system message restores parity with vLLM (829 vs 835).
        messages = [
            {"role": "system", "content": _HARMONY_SYSTEM + "\n\n" + _REVIEWER_DEVELOPER},
            {"role": "user", "content": user},
        ]
    else:
        messages = [
            {"role": "system", "content": _HARMONY_SYSTEM},
            {"role": "developer", "content": _REVIEWER_DEVELOPER},
            {"role": "user", "content": user},
        ]
    p = {"model": a.model, "messages": messages}
    if a.max_tokens > 0:
        p["max_tokens"] = a.max_tokens
    # NOTE: the "Reasoning: high" line inside the Harmony system text is NOT honoured by
    # vLLM/Dynamo/SGLang chat renderers (they build their own system header from the API
    # field and default to medium). Pass --reasoning-effort high to get real high reasoning.
    if getattr(a, "reasoning_effort", None):
        p["reasoning_effort"] = a.reasoning_effort
    return p


def build_payloads(a):
    """Return a list of `total` request payloads.

    Default: identical short prompt (pure decode-throughput test).
    --workload believe: the real production job shape (shared ~450-token prefix,
    unique ~450-token abstract, unbounded reasoning-high output).
    With --sessions S: a prefix-sharing workload — every request carries the same
    long system prompt (--shared-words) plus a session-specific context
    (--session-words) and a short unique user message. Requests of one session
    are spread across the run, so a KV-aware router that pins a session to the
    worker holding its cache gets prefix hits that least-connections cannot.
    """
    if a.workload == "believe":
        return [believe_payload(a, i) for i in range(a.total)]
    if not a.sessions:
        p = {"model": a.model, "messages": [{"role": "user", "content": a.prompt}],
             "max_tokens": a.max_tokens, "temperature": 0}
        return [p] * a.total
    shared = "You are a meticulous assistant. Context: " + filler(0, a.shared_words)
    payloads = []
    for i in range(a.total):
        s = i % a.sessions
        payloads.append({
            "model": a.model,
            "messages": [
                {"role": "system", "content": shared},
                {"role": "user", "content": f"Session {s} notes: " + filler(s + 1, a.session_words)},
                {"role": "assistant", "content": "Understood. I have the notes."},
                {"role": "user", "content": f"Question {i}: " + a.prompt},
            ],
            "max_tokens": a.max_tokens,
            "temperature": 0,
        })
    return payloads


async def run(a):
    ep = Endpoint(a.url, a.insecure)
    payloads = build_payloads(a)
    reqs = [ep.request_bytes("/v1/chat/completions", p, a.api_key) for p in payloads]
    lat = []
    errs = {}
    stats = {"ok": 0, "completion_tokens": 0, "prompt_tokens": 0}
    remaining = [a.total]

    async def one_request(conn, req):
        reader, writer = conn
        writer.write(req)
        await writer.drain()
        return await asyncio.wait_for(read_response(reader), a.timeout)

    async def slot():
        conn = None
        while remaining[0] > 0:
            remaining[0] -= 1
            req = reqs[a.total - remaining[0] - 1]
            t0 = time.perf_counter()
            try:
                if conn is None:
                    conn = await asyncio.wait_for(ep.connect(), a.timeout)
                status, body, keep_alive = await one_request(conn, req)
                dt = time.perf_counter() - t0
                if status == 200:
                    stats["ok"] += 1
                    lat.append(dt)
                    u = json.loads(body).get("usage", {}) or {}
                    stats["completion_tokens"] += u.get("completion_tokens", 0) or 0
                    stats["prompt_tokens"] += u.get("prompt_tokens", 0) or 0
                else:
                    k = f"HTTP {status}"
                    errs[k] = errs.get(k, 0) + 1
                if not keep_alive:
                    conn[1].close()
                    conn = None
            except Exception as e:  # timeouts, connection errors, bad framing
                k = type(e).__name__
                errs[k] = errs.get(k, 0) + 1
                if conn is not None:
                    conn[1].close()
                    conn = None
        if conn is not None:
            conn[1].close()

    # Warm-up: a single request so model/graph capture isn't in the timed window.
    if a.warmup:
        try:
            conn = await ep.connect()
            await one_request(conn, reqs[0])
            conn[1].close()
        except Exception:
            pass

    wall0 = time.perf_counter()
    await asyncio.gather(*(slot() for _ in range(min(a.concurrency, a.total))))
    wall = time.perf_counter() - wall0

    ok = stats["ok"]
    out = {
        "url": a.url,
        "model": a.model,
        "concurrency": a.concurrency,
        "total": a.total,
        "max_tokens": a.max_tokens,
        "sessions": a.sessions or None,
        "avg_prompt_tokens": round(stats["prompt_tokens"] / ok, 1) if ok else None,
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
    ap.add_argument("--api-key", default=None, help="sent as `Authorization: Bearer <key>` (the gateway requires it)")
    ap.add_argument("--workload", choices=["simple", "believe"], default="simple",
                    help="believe = production job shape (Harmony prefix + abstract, reasoning high, no max_tokens unless >0)")
    ap.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default=None,
                    help="believe workload: set the API reasoning_effort field explicitly")
    ap.add_argument("--merge-system", action="store_true",
                    help="believe workload: merge system+developer into one system message (Dynamo workaround)")
    ap.add_argument("--sessions", type=int, default=0,
                    help="prefix-sharing workload: number of distinct sessions (0 = identical prompts)")
    ap.add_argument("--shared-words", type=int, default=1500, help="words in the system prompt shared by all")
    ap.add_argument("--session-words", type=int, default=1000, help="words of per-session context")
    ap.add_argument("--no-warmup", dest="warmup", action="store_false")
    a = ap.parse_args()
    sys.exit(asyncio.run(run(a)))


if __name__ == "__main__":
    main()
