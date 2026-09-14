"""Standalone mock of a vllm-omni WORKER AGENT (Dynamo-native platform).

The Rust router is gone, so a mock OpenAI endpoint is no longer what the
platform needs to be tested against. What central actually talks to is:

  1. the worker agent's internal API  (worker/main.py)      — deploy/stop/logs
  2. every dynamo instance's SYSTEM port (`base + 10000`)   — /health, /metrics

This module implements both. `POST /api/internal/deploy` accepts the current
request shape (including the dynamo fields central/manager.py:_dynamo_fields
adds), allocates a base port exactly like the real agent, and — for
engine="dynamo" — starts a tiny HTTP server on `base + 10000` that answers
`/health` with `{"status": "ready"}` and serves a plausible `/metrics` page
(`dynamo_component_*` + `vllm:*`), so central's health checks and metric
scraper can be driven end to end without docker or a GPU.

Worker agent API (subset central uses):

    GET  /api/internal/version
    POST /api/internal/deploy              -> {"id", "replica_id", "ports", "nodes"}
    POST /api/internal/stop/{deploy_id}
    POST /api/internal/stop_replica/{deploy_id}/{global_gpu_id}
    GET  /api/internal/images
    GET  /api/internal/models

Test-only introspection (never part of the real API):

    GET  /_state    -> {"deploy_requests": [...], "deployments": [...],
                        "instances": {port: {...}}}
    POST /_reset    -> forget every deployment (stops the instance servers)
    POST /_config   -> patch behaviour, e.g. {"ready": false, "fail_deploy": true,
                       "requests_total": 42}

Run it by hand:

    python mock_worker.py --port 8085 --worker-id mock-worker
    curl -s localhost:8085/_state | jq
"""

from __future__ import annotations

import argparse
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Must match worker/manager.py and central/dynamo.py.
SYSTEM_PORT_OFFSET = 10000
RPC_PORT_OFFSET = 12000
RESP_PORT_OFFSET = 42000
KV_PORT_OFFSET = 44000
PORT_START = 21001

DYNAMO_IMAGE = "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.2"
VLLM_IMAGE = "vllm/vllm-openai:latest"


class WorkerDeployRequest(BaseModel):
    """Mirror of worker/main.py:WorkerDeployRequest."""

    deploy_id: str
    replica_id: str
    name: str
    model: str
    served_model_name: Optional[str] = None
    is_embedding: bool = False
    engine: Optional[str] = "dynamo"
    gpus: List[int]
    tp: int = 1
    max_len: Optional[int] = None
    gpu_util: Optional[float] = 0.9
    extra_args: Optional[str] = None
    vllm_image: Optional[str] = None
    # engine == "dynamo" only
    advertise_host: Optional[str] = None
    etcd_endpoints: Optional[str] = None
    namespace: Optional[str] = "dynamo"
    reasoning_parser: Optional[str] = None
    tool_call_parser: Optional[str] = None
    block_size: Optional[int] = 64


# ---------------------------------------------------------------------------
# Fake dynamo instance: the system port (health + metrics)
# ---------------------------------------------------------------------------

METRICS_TEMPLATE = """\
# TYPE dynamo_component_requests_total counter
dynamo_component_requests_total{{dynamo_namespace="{ns}",dynamo_component="backend",dynamo_endpoint="generate"}} {requests}
dynamo_component_requests_total{{dynamo_namespace="{ns}",dynamo_component="backend",dynamo_endpoint="load_metrics"}} 4242
# TYPE dynamo_component_errors_total counter
dynamo_component_errors_total{{dynamo_namespace="{ns}",dynamo_component="backend",dynamo_endpoint="generate"}} {errors}
# TYPE dynamo_component_inflight_requests gauge
dynamo_component_inflight_requests{{dynamo_namespace="{ns}",dynamo_component="backend",dynamo_endpoint="generate"}} {inflight}
# TYPE dynamo_component_request_duration_seconds histogram
dynamo_component_request_duration_seconds_sum{{dynamo_namespace="{ns}",dynamo_component="backend",dynamo_endpoint="generate"}} {duration_sum}
dynamo_component_request_duration_seconds_count{{dynamo_namespace="{ns}",dynamo_component="backend",dynamo_endpoint="generate"}} {requests}
# TYPE dynamo_component_uptime_seconds gauge
dynamo_component_uptime_seconds{{dynamo_namespace="{ns}",dynamo_component="backend"}} {uptime}
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{{model_name="{model}"}} {running}
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{{model_name="{model}"}} {waiting}
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc{{model_name="{model}"}} {kv}
"""


class InstanceServer:
    """A dynamo worker's system port: /health, /live, /metrics."""

    def __init__(self, port: int, model: str, namespace: str, state: dict):
        self.port = port
        self.model = model
        self.namespace = namespace
        self.state = state  # shared, patchable through POST /_config
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _send(self, code: int, body: bytes, ctype: str) -> None:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):  # noqa: N802
                path = self.path.split("?")[0]
                if path in ("/health", "/live"):
                    ready = bool(outer.state.get("ready", True))
                    body = json.dumps(
                        {"status": "ready" if ready else "notready"}
                    ).encode()
                    self._send(200 if ready else 503, body, "application/json")
                elif path == "/metrics":
                    self._send(200, outer.render_metrics().encode(), "text/plain")
                else:
                    self._send(404, b"not found", "text/plain")

            def log_message(self, *args):  # silence
                pass

        self._httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def render_metrics(self) -> str:
        s = self.state
        return METRICS_TEMPLATE.format(
            ns=self.namespace,
            model=self.model,
            requests=s.get("requests_total", 0),
            errors=s.get("errors_total", 0),
            inflight=s.get("inflight", 0),
            duration_sum=s.get("duration_sum", 0.0),
            uptime=s.get("uptime_s", 60),
            running=s.get("running", 0),
            waiting=s.get("waiting", 0),
            kv=s.get("kv_cache_usage", 0.0),
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()


def _port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


# ---------------------------------------------------------------------------
# Worker agent app
# ---------------------------------------------------------------------------

def create_app(worker_id: str = "mock-worker", port_start: int = PORT_START) -> FastAPI:
    app = FastAPI()

    cfg = {
        "ready": True,          # instance /health answers ready
        "fail_deploy": False,   # deploy returns 500
        "requests_total": 0,
        "errors_total": 0,
        "inflight": 0,
        "duration_sum": 0.0,
        "running": 0,
        "waiting": 0,
        "kv_cache_usage": 0.0,
        "uptime_s": 60,
    }
    deployments: list = []          # like worker/data/local_deployments.json
    deploy_requests: list = []      # every request body we received
    instances: dict = {}            # system_port -> InstanceServer
    lock = threading.Lock()

    def _alloc_base_port() -> int:
        used = {p for d in deployments for p in d["ports"]}
        port = port_start
        while port in used or not _port_is_free(port + SYSTEM_PORT_OFFSET):
            port += 1
        return port

    def _stop_indices(indices: list) -> None:
        for i in reversed(sorted(indices)):
            dep = deployments.pop(i)
            for p in dep["ports"]:
                inst = instances.pop(p + SYSTEM_PORT_OFFSET, None)
                if inst is not None:
                    inst.stop()

    # -- worker agent API ---------------------------------------------------

    @app.get("/api/internal/version")
    async def version():
        return {"commit": "mock", "subtree": "", "updated_at": "", "updating": False}

    @app.post("/api/internal/deploy")
    async def deploy(req: WorkerDeployRequest):
        body = req.dict()
        with lock:
            deploy_requests.append(body)
            if cfg["fail_deploy"]:
                raise HTTPException(status_code=500, detail="mock deploy failure")
            engine = body.get("engine") or "dynamo"
            if engine not in ("dynamo", "vllm"):
                raise HTTPException(status_code=500, detail=f"Unsupported engine {engine!r}")

            base = _alloc_base_port()
            node_name = f"{engine}_{body['replica_id']}"
            dep = {
                "id": body["deploy_id"],
                "replica_id": body["replica_id"],
                "ports": [base],
                "nodes": [{"name": node_name, "port": base}],
                "engine": engine,
            }
            deployments.append(dep)

            if engine == "dynamo":
                inst = InstanceServer(
                    base + SYSTEM_PORT_OFFSET,
                    body.get("served_model_name") or body["model"],
                    body.get("namespace") or "dynamo",
                    cfg,
                )
                inst.start()
                instances[base + SYSTEM_PORT_OFFSET] = inst
        return {k: v for k, v in dep.items() if k != "engine"}

    @app.post("/api/internal/stop/{deploy_id}")
    async def stop(deploy_id: str):
        with lock:
            idx = [i for i, d in enumerate(deployments) if d["id"] == deploy_id]
            if not idx:
                raise HTTPException(status_code=404, detail="Deployment not found")
            _stop_indices(idx)
        return {"status": "success"}

    @app.post("/api/internal/stop_replica/{deploy_id}/{global_gpu_id}")
    async def stop_replica(deploy_id: str, global_gpu_id: str):
        wid, gid = global_gpu_id.rsplit("-", 1)
        replica_id = f"{deploy_id}_{wid}_{gid}"
        with lock:
            idx = [i for i, d in enumerate(deployments) if d["replica_id"] == replica_id]
            if not idx:
                raise HTTPException(status_code=404, detail="Replica not found")
            _stop_indices(idx)
        return {"status": "success"}

    @app.get("/api/internal/images")
    async def images():
        return [
            {"name": DYNAMO_IMAGE, "size": "20.6GB", "created": "", "engine": "dynamo"},
            {"name": VLLM_IMAGE, "size": "19GB", "created": "", "engine": "vllm"},
        ]

    @app.get("/api/internal/models")
    async def models():
        return []

    # -- test-only introspection -------------------------------------------

    @app.get("/_state")
    async def state():
        return JSONResponse({
            "worker_id": worker_id,
            "deploy_requests": deploy_requests,
            "deployments": deployments,
            "instances": {
                str(p): {"model": i.model, "namespace": i.namespace}
                for p, i in instances.items()
            },
            "config": dict(cfg),
        })

    @app.post("/_reset")
    async def reset():
        with lock:
            _stop_indices(list(range(len(deployments))))
            deploy_requests.clear()
        return {"status": "ok"}

    @app.post("/_config")
    async def config(patch: dict):
        cfg.update(patch)
        return {"status": "ok", "config": dict(cfg)}

    @app.on_event("shutdown")
    async def _shutdown():
        with lock:
            _stop_indices(list(range(len(deployments))))

    return app


def main() -> None:
    import uvicorn

    p = argparse.ArgumentParser(description="Mock vllm-omni worker agent")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--worker-id", default="mock-worker")
    p.add_argument("--port-start", type=int, default=PORT_START,
                   help="first base port handed out to deployments")
    args = p.parse_args()

    uvicorn.run(
        create_app(worker_id=args.worker_id, port_start=args.port_start),
        host=args.host, port=args.port, log_level="warning",
    )


if __name__ == "__main__":
    main()
