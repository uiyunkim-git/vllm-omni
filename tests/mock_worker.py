"""Standalone mock vLLM worker for router / platform tests.

Implements just enough of the OpenAI-compatible surface for the Rust router
(central/router) to register it and route traffic to it:

    GET  /health              -> 200 "ok"
    GET  /v1/models           -> configurable served model id (the router reads
                                 data[0].id at add_worker time and routes by it)
    POST /v1/chat/completions -> stream:false (JSON) and stream:true (SSE + [DONE])
    POST /v1/embeddings       -> stream:false (JSON) and stream:true (SSE + [DONE])

Introspection / control endpoints (test-only, never part of the real API):

    GET  /_stats   -> {"requests", "peak_concurrency", "in_flight",
                       "aborted_streams", "completed_streams",
                       "died_streams", "failures_injected", "config"}
    POST /_reset   -> zero all counters (config untouched)
    POST /_config  -> JSON body patches the live config, e.g.
                      {"delay_ms": 500, "fail_mode": "429", "fail_n": -1}

Configuration (env var / CLI flag / per-request query param on direct calls):

    model id       MOCK_MODEL_ID       --model          (default "mock-model")
    delay ms       MOCK_DELAY_MS       --delay-ms       delay before response /
                                                        before the FIRST stream chunk
    failure mode   MOCK_FAIL_MODE      --fail-mode      "" | "500" | "429" | "hang"
                                                        | "die_mid_stream"
    failure budget MOCK_FAIL_N         --fail-n         -1 = fail forever,
                                                        N>0 = fail first N requests
    stream chunks  MOCK_STREAM_CHUNKS  --stream-chunks  content chunks before [DONE]
    chunk gap ms   MOCK_INTERCHUNK_MS  --interchunk-ms

Note: query-param overrides (?delay_ms=...&fail_mode=...) only work when
calling the worker DIRECTLY -- the router does not forward query params.
For router-mediated tests use /_config.

aborted_streams counts client disconnects mid-stream: the SSE generator's
``finally`` runs without the stream having reached [DONE] (uvicorn/starlette
cancel the generator task with asyncio.CancelledError / GeneratorExit when the
downstream connection drops). This is what the router cancellation-propagation
tests key on.
"""

import argparse
import asyncio
import json
import os
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from starlette.requests import ClientDisconnect


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def create_app(
    model_id: str | None = None,
    delay_ms: int | None = None,
    fail_mode: str | None = None,
    fail_n: int | None = None,
    stream_chunks: int | None = None,
    interchunk_ms: int | None = None,
) -> FastAPI:
    cfg = {
        "model_id": model_id if model_id is not None else _env("MOCK_MODEL_ID", "mock-model"),
        "delay_ms": int(delay_ms if delay_ms is not None else _env("MOCK_DELAY_MS", "0")),
        "fail_mode": fail_mode if fail_mode is not None else _env("MOCK_FAIL_MODE", ""),
        "fail_n": int(fail_n if fail_n is not None else _env("MOCK_FAIL_N", "-1")),
        "stream_chunks": int(
            stream_chunks if stream_chunks is not None else _env("MOCK_STREAM_CHUNKS", "3")
        ),
        "interchunk_ms": int(
            interchunk_ms if interchunk_ms is not None else _env("MOCK_INTERCHUNK_MS", "10")
        ),
    }

    stats = {
        "requests": 0,            # inference requests received (chat + embeddings)
        "in_flight": 0,           # currently being served (incl. active streams)
        "peak_concurrency": 0,    # max simultaneous in_flight ever observed
        "aborted_streams": 0,     # client disconnected before [DONE]
        "completed_streams": 0,   # streams that delivered [DONE]
        "died_streams": 0,        # streams the mock itself killed (die_mid_stream)
        "failures_injected": 0,   # 500/429/hang/die responses served
    }

    app = FastAPI()

    def _bump_inflight() -> None:
        stats["in_flight"] += 1
        if stats["in_flight"] > stats["peak_concurrency"]:
            stats["peak_concurrency"] = stats["in_flight"]

    def _merged(request: Request) -> dict:
        """Effective config for this request: query params > live config."""
        out = dict(cfg)
        q = request.query_params
        if "delay_ms" in q:
            out["delay_ms"] = int(q["delay_ms"])
        if "fail_mode" in q:
            out["fail_mode"] = q["fail_mode"]
        return out

    def _take_failure(effective: dict) -> str | None:
        mode = effective.get("fail_mode") or ""
        if not mode:
            return None
        if cfg["fail_n"] == 0:
            return None
        if cfg["fail_n"] > 0:
            cfg["fail_n"] -= 1
        stats["failures_injected"] += 1
        return mode

    # ------------------------------------------------------------------ admin

    @app.get("/health")
    async def health():
        return PlainTextResponse("ok")

    @app.get("/v1/models")
    async def models():
        return JSONResponse(
            {
                "object": "list",
                "data": [
                    {
                        "id": cfg["model_id"],
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "mock",
                    }
                ],
            }
        )

    @app.get("/get_model_info")
    async def model_info():
        return JSONResponse({"model": cfg["model_id"]})

    @app.get("/_stats")
    async def get_stats():
        return JSONResponse({**stats, "config": dict(cfg)})

    @app.post("/_reset")
    async def reset_stats():
        for k in stats:
            stats[k] = 0
        return JSONResponse({"status": "ok"})

    @app.post("/_config")
    async def patch_config(request: Request):
        body = await request.json()
        for k in ("model_id", "fail_mode"):
            if k in body:
                cfg[k] = str(body[k])
        for k in ("delay_ms", "fail_n", "stream_chunks", "interchunk_ms"):
            if k in body:
                cfg[k] = int(body[k])
        return JSONResponse({"status": "ok", "config": dict(cfg)})

    # -------------------------------------------------------------- inference

    def _chat_payload(body: dict) -> dict:
        now = time.time()
        return {
            "id": f"chatcmpl-mock-{int(now * 1000)}",
            "object": "chat.completion",
            "created": int(now),
            "model": cfg["model_id"],
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "mock response"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        }

    def _embedding_payload(body: dict) -> dict:
        inputs = body.get("input", "")
        n = len(inputs) if isinstance(inputs, list) else 1
        return {
            "object": "list",
            "model": cfg["model_id"],
            "data": [
                {"object": "embedding", "index": i, "embedding": [0.0] * 8}
                for i in range(n)
            ],
            "usage": {"prompt_tokens": 1, "total_tokens": 1},
        }

    def _stream_chunk(kind: str, i: int) -> str:
        if kind == "chat":
            payload = {
                "id": "chatcmpl-mock-stream",
                "object": "chat.completion.chunk",
                "model": cfg["model_id"],
                "choices": [
                    {"index": 0, "delta": {"content": f"tok{i} "}, "finish_reason": None}
                ],
            }
        else:
            payload = {
                "object": "embedding.chunk",
                "model": cfg["model_id"],
                "data": [{"object": "embedding", "index": i, "embedding": [0.0] * 8}],
            }
        return f"data: {json.dumps(payload)}\n\n"

    async def _serve(request: Request, kind: str):
        try:
            body = await request.json()
        except ClientDisconnect:
            # Client vanished while we were reading the request body.
            stats["aborted_streams"] += 1
            return PlainTextResponse("client disconnected", status_code=400)
        except Exception:
            body = {}

        effective = _merged(request)
        wants_stream = bool(body.get("stream", False))
        stats["requests"] += 1

        failure = _take_failure(effective)
        if failure in ("500", "429"):
            return JSONResponse(
                {"error": {"message": f"mock injected {failure}", "type": "mock_failure"}},
                status_code=int(failure),
            )
        if failure == "hang":
            await asyncio.sleep(3600)

        delay_s = effective["delay_ms"] / 1000.0

        if not wants_stream:
            _bump_inflight()
            try:
                if delay_s:
                    await asyncio.sleep(delay_s)
                payload = _chat_payload(body) if kind == "chat" else _embedding_payload(body)
                return JSONResponse(payload)
            finally:
                stats["in_flight"] -= 1

        die_mid_stream = failure == "die_mid_stream"

        async def gen():
            _bump_inflight()
            finished = False
            died = False
            try:
                if delay_s:
                    await asyncio.sleep(delay_s)
                for i in range(effective["stream_chunks"]):
                    yield _stream_chunk(kind, i)
                    if die_mid_stream and i == 0:
                        died = True
                        raise RuntimeError("mock worker dying mid-stream (configured)")
                    if effective["interchunk_ms"]:
                        await asyncio.sleep(effective["interchunk_ms"] / 1000.0)
                yield "data: [DONE]\n\n"
                finished = True
            except asyncio.CancelledError:
                # Client (or the router, on behalf of its client) disconnected.
                raise
            finally:
                stats["in_flight"] -= 1
                if finished:
                    stats["completed_streams"] += 1
                elif died:
                    stats["died_streams"] += 1
                else:
                    stats["aborted_streams"] += 1

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        return await _serve(request, "chat")

    @app.post("/v1/embeddings")
    async def embeddings(request: Request):
        return await _serve(request, "embeddings")

    return app


def main() -> None:
    import uvicorn

    p = argparse.ArgumentParser(description="Mock vLLM worker")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--model", default=None)
    p.add_argument("--delay-ms", type=int, default=None)
    p.add_argument("--fail-mode", default=None,
                   choices=["", "500", "429", "hang", "die_mid_stream"])
    p.add_argument("--fail-n", type=int, default=None)
    p.add_argument("--stream-chunks", type=int, default=None)
    p.add_argument("--interchunk-ms", type=int, default=None)
    args = p.parse_args()

    app = create_app(
        model_id=args.model,
        delay_ms=args.delay_ms,
        fail_mode=args.fail_mode,
        fail_n=args.fail_n,
        stream_chunks=args.stream_chunks,
        interchunk_ms=args.interchunk_ms,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
