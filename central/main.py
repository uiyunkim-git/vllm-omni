from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse, PlainTextResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
from manager import CentralManager
import logging
import os

logger = logging.getLogger(__name__)
import httpx
import json
import random
import asyncio
import time
import re

logging.basicConfig(level=logging.INFO)

app = FastAPI()
manager = CentralManager()

templates = Jinja2Templates(directory="frontend")
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

class DeployRequest(BaseModel):
    name: str
    deployment_type: str # "replicas" or "tp"
    model: str
    served_model_name: Optional[str] = None
    is_embedding: bool = False
    engine: Optional[str] = "vllm"
    gpus: List[str] # Global GPU IDs e.g., ["alpha-worker-1-0", "alpha-worker-1-1"]
    tp: int = 1
    max_len: Optional[int] = None
    gpu_util: Optional[float] = 0.9
    extra_args: Optional[str] = None
    vllm_image: Optional[str] = None
    # engine == "dynamo": Dynamo-side parsers (inferred from the model name when omitted)
    reasoning_parser: Optional[str] = None
    tool_call_parser: Optional[str] = None
    block_size: Optional[int] = None

class ConfigSaveRequest(BaseModel):
    name: str
    config: dict

class EndpointAcceptRequest(BaseModel):
    custom_name: str

class RegisterNodeRequest(BaseModel):
    worker_id: str
    host: str
    port: int
    gpus: List[dict]
    version: Optional[dict] = None

@app.get("/", response_class=HTMLResponse)
async def read_dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request, "active_tab": "dashboard"})

@app.get("/configs", response_class=HTMLResponse)
async def read_configs(request: Request):
    return templates.TemplateResponse("configs.html", {"request": request, "active_tab": "configs"})

@app.get("/endpoints", response_class=HTMLResponse)
async def read_endpoints(request: Request):
    return templates.TemplateResponse("endpoints.html", {"request": request, "active_tab": "endpoints"})

@app.get("/gateway", response_class=HTMLResponse)
async def read_gateway(request: Request):
    return templates.TemplateResponse("gateway.html", {"request": request, "active_tab": "gateway"})

@app.get("/deploy", response_class=HTMLResponse)
async def deploy_page(request: Request):
    return templates.TemplateResponse("deploy.html", {"request": request, "active_tab": "dashboard"})

@app.get("/endpoints/{worker_id}/images", response_class=HTMLResponse)
async def endpoint_images_page(request: Request, worker_id: str):
    workers = manager.get_workers()
    if worker_id not in workers:
        raise HTTPException(status_code=404, detail="Worker not found")
    return templates.TemplateResponse("endpoint_images.html", {
        "request": request, "active_tab": "endpoints", "worker": workers[worker_id]
    })

@app.get("/endpoints/{worker_id}/models", response_class=HTMLResponse)
async def endpoint_models_page(request: Request, worker_id: str):
    workers = manager.get_workers()
    if worker_id not in workers:
        raise HTTPException(status_code=404, detail="Worker not found")
    return templates.TemplateResponse("endpoint_models.html", {
        "request": request, "active_tab": "endpoints", "worker": workers[worker_id]
    })

@app.get("/logs/{deploy_id}", response_class=HTMLResponse)
async def read_logs_page(request: Request, deploy_id: str):
    return templates.TemplateResponse("logs.html", {"request": request, "deploy_id": deploy_id})

@app.get("/api/endpoints")
async def get_endpoints():
    return manager.get_workers()

@app.post("/api/endpoints/{worker_id}/accept")
async def accept_endpoint(worker_id: str, req: EndpointAcceptRequest):
    success = manager.accept_worker(worker_id, custom_name=req.custom_name)
    if not success:
        raise HTTPException(status_code=404, detail="Worker not found")
    return {"status": "success"}

@app.delete("/api/endpoints/{worker_id}")
async def delete_endpoint(worker_id: str):
    success = manager.delete_worker(worker_id)
    if not success:
        raise HTTPException(status_code=404, detail="Worker not found")
    return {"status": "success"}

@app.get("/api/gpus")
async def get_gpus():
    return manager.get_all_gpus()

@app.get("/api/deployments")
async def get_deployments():
    return manager.load_deployments()

@app.get("/api/configs")
async def get_configs():
    return manager.load_configs()

@app.post("/api/configs")
async def save_config(req: ConfigSaveRequest):
    manager.save_config(req.dict())
    return {"status": "success"}

@app.delete("/api/configs/{name:path}")
async def delete_config(name: str):
    success = manager.delete_config(name)
    if not success:
        raise HTTPException(status_code=404, detail="Config not found")
    return {"status": "success"}

@app.post("/api/deploy")
async def deploy_model(req: DeployRequest):
    try:
        dep = await manager.deploy_model(req.dict())
        return dep
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/stop/{deployment_id}")
async def stop_deployment(deployment_id: str):
    success = await manager.stop_deployment(deployment_id)
    if not success:
        raise HTTPException(status_code=404, detail="Deployment not found")
    return {"status": "success"}

@app.post("/api/stop/{deployment_id}/gpu/{global_gpu_id}")
async def stop_replica(deployment_id: str, global_gpu_id: str):
    success = await manager.stop_replica(deployment_id, global_gpu_id)
    if not success:
        raise HTTPException(status_code=404, detail="Replica not found")
    return {"status": "success"}

def _parse_prometheus_full(text: str) -> dict:
    result: dict = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        m = re.match(r'^(\w+)(\{[^}]*\})?\s+(\S+)', line)
        if not m:
            continue
        name, labels_str, val_str = m.group(1), m.group(2) or "", m.group(3)
        try:
            val = float(val_str)
        except ValueError:
            continue
        labels = dict(re.findall(r'(\w+)="([^"]*)"', labels_str))
        result.setdefault(name, []).append({"labels": labels, "value": val})
    return result

from collections import deque

_BUFFER_SECONDS_MAX = 6 * 60 * 60  # keep 6h of samples per series. Capping at 6h
                                   # keeps per-series buffers around 4.3k entries
                                   # (vs ~17k at 24h), which made linear scans in
                                   # _window_delta and latency-bucket aggregation
                                   # noticeably slow once we hit ~40 workers.
# Single, unified window vocabulary shared by cards / per-worker bars / RPS chart.
_ALLOWED_WINDOWS = [30, 60, 300, 900, 3600, 6 * 3600]  # 30s, 1m, 5m, 15m, 1h, 6h
_ALLOWED_HISTORY = _ALLOWED_WINDOWS  # alias, used by rps_history endpoint
_metric_history: Dict[str, deque] = {}  # key -> deque[(ts, value)]


def _push_sample(key: str, ts: float, val: float) -> deque:
    dq = _metric_history.setdefault(key, deque())
    # Counters are monotonic — any drop means a real reset (router restart).
    # Partial /metrics scrapes are filtered upstream at the cycle level (see
    # `_required` short-circuit in _collect_metrics), so by the time we get
    # here, a drop is trustworthy.
    if dq and val < dq[-1][1]:
        dq.clear()
    dq.append((ts, val))
    cutoff = ts - _BUFFER_SECONDS_MAX
    while len(dq) > 1 and dq[0][0] < cutoff:
        dq.popleft()
    return dq


def _window_delta(key: str, window_s: int):
    """Return (delta, duration_s) for samples in the last `window_s` seconds, or None.

    The window is anchored to wall-clock `now`, NOT to the last sample's
    timestamp. This matters after a router restart: a worker that processed
    requests before the restart but none after stops being scraped (a counter
    that never increments post-reset is not even emitted by Prometheus), so its
    deque freezes with the old cumulative value. Anchoring the cutoff to the
    last sample would then report that entire historical climb as if it
    happened in the last `window_s` seconds — producing absurd RPS spikes on
    the windowed cards. Anchoring to `now` (and treating a stale series as no
    activity) correctly reports 0 once the series stops updating.
    """
    dq = _metric_history.get(key)
    if not dq or len(dq) < 2:
        return None
    now = time.time()
    cutoff = now - window_s
    newest_ts, newest_val = dq[-1]
    # Series hasn't updated within the window → nothing happened recently.
    if newest_ts < cutoff:
        return None
    oldest_ts, oldest_val = None, None
    for ts, val in dq:
        if ts >= cutoff:
            oldest_ts, oldest_val = ts, val
            break
    if oldest_ts is None:
        return None
    duration = max(1e-6, newest_ts - oldest_ts)
    delta = max(0.0, newest_val - oldest_val)
    return delta, duration


# ──────────────────────────────────────────────────────────────────────────
# Background metric collection.
#
# Previously /api/prometheus_stats both *scraped* the router and *served* the
# response, which meant nothing got into the ring buffers unless somebody had
# the metrics page open. That made the chart look like it was "ramping up from
# zero" every time a user opened it — because for that user, recent history
# really was empty. Now a single background task scrapes every 5s regardless
# of whether anyone is watching, and the endpoint just reads the snapshot it
# leaves behind.
# ──────────────────────────────────────────────────────────────────────────
_latest_scrape: dict = {}
_SCRAPE_PERIOD_S = 5.0

# Router endpoints — overridable so the service isn't hardwired to one host.
ROUTER_METRICS_URL = os.environ.get("ROUTER_METRICS_URL", "http://143.248.74.105:29000/metrics")
ROUTER_WORKERS_URL = os.environ.get("ROUTER_WORKERS_URL", "http://143.248.74.105:11434/workers")

# Last successful /workers snapshot. When the /workers fetch fails we reuse
# this instead of an empty dict — an empty dict made _is_live fail OPEN,
# counting every stale (worker, instance) series as live and inflating the
# aggregates for that cycle.
_last_live_instances: Dict[str, str] = {}

# ── Dynamo data plane metrics ────────────────────────────────────────────────
# engine=dynamo nodes are not behind the P2C router. Their per-worker counters
# come from each worker's system port (`dynamo_component_*` for the `generate`
# endpoint), and the global request stream from the Dynamo frontend
# (`dynamo_frontend_requests_total`). Both are folded into the SAME per-worker
# maps / ring buffers the router path fills, keyed by the node's system URL
# (`http://host:31xxx`), so the dashboard needs no separate code path.
DYNAMO_FRONTEND_URL = os.environ.get("DYNAMO_FRONTEND_URL", "http://143.248.74.105:11435")
_DYNAMO_SYSTEM_PORT_OFFSET = 10000


def _dynamo_node_url(host: str, base_port: int) -> str:
    return f"http://{host}:{base_port + _DYNAMO_SYSTEM_PORT_OFFSET}"


def _node_url(dep: dict, node: dict) -> str:
    """Endpoint URL central uses as the per-worker key for a deployment node:
    the vLLM API (https, +40000) for engine=vllm, the Dynamo system port for
    engine=dynamo."""
    if dep.get("engine", "vllm") == "dynamo":
        return _dynamo_node_url(node["host"], node["port"])
    return f'https://{node["host"]}:{node["port"] + 40000}'


def _dynamo_nodes() -> list:
    """[(url, dep)] for every node of every engine=dynamo deployment."""
    out = []
    for dep in manager.load_deployments():
        if dep.get("engine", "vllm") != "dynamo" or dep.get("status") not in ("running", "starting"):
            continue
        for node in dep.get("nodes", []):
            out.append((_dynamo_node_url(node["host"], node["port"]), dep, node))
    return out


async def _scrape_dynamo(client: httpx.AsyncClient) -> dict:
    """Scrape the Dynamo frontend + every dynamo worker's system port.
    Returns per-worker maps in the router path's shape (empty when nothing
    is deployed on the Dynamo data plane)."""
    res = {
        "total_requests": 0, "processed": {}, "running": {}, "lat_sum": {}, "lat_cnt": {},
        "bucket": {}, "succ": {}, "fail": {}, "live": {}, "active_workers": 0,
    }
    nodes = _dynamo_nodes()
    if not nodes:
        return res

    try:
        r = await client.get(f"{DYNAMO_FRONTEND_URL}/metrics", timeout=3.0)
        fraw = _parse_prometheus_full(r.text)
        res["total_requests"] = sum(int(float(e["value"])) for e in fraw.get("dynamo_frontend_requests_total", []))
    except Exception as e:
        logger.debug(f"dynamo frontend scrape failed: {e!r}")

    async def one(url, dep, node):
        try:
            r = await client.get(f"{url}/metrics", timeout=3.0)
        except Exception:
            return
        raw = _parse_prometheus_full(r.text)

        def gen(metric):
            return [e for e in raw.get(metric, []) if e["labels"].get("dynamo_endpoint") == "generate"]

        res["live"][url] = node.get("name", "dynamo")
        res["active_workers"] += 1
        res["processed"][url] = sum(int(float(e["value"])) for e in gen("dynamo_component_requests_total"))
        res["running"][url] = int(sum(float(e["value"]) for e in gen("dynamo_component_inflight_requests")))
        res["lat_sum"][url] = sum(float(e["value"]) for e in gen("dynamo_component_request_duration_seconds_sum"))
        res["lat_cnt"][url] = sum(int(float(e["value"])) for e in gen("dynamo_component_request_duration_seconds_count"))
        for e in gen("dynamo_component_request_duration_seconds_bucket"):
            le = e["labels"].get("le")
            if le is None or le == "+Inf":
                continue
            res["bucket"].setdefault(le, {})[url] = res["bucket"].get(le, {}).get(url, 0) + int(float(e["value"]))
        res["fail"][url] = sum(int(float(e["value"])) for e in gen("dynamo_component_errors_total"))
        res["succ"][url] = max(0, res["processed"][url] - res["fail"][url])

    await asyncio.gather(*(one(u, d, n) for u, d, n in nodes))
    return res


async def _collect_metrics():
    """One scrape cycle. Fetch the router's /metrics + /workers AND the Dynamo
    data plane (frontend + worker system ports), push samples to ring buffers,
    and update `_latest_scrape` for the API layer to read. Either source may be
    absent (router removed after the Dynamo cutover, or no dynamo nodes yet)."""
    raw: dict = {}
    router_ok = False
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(ROUTER_METRICS_URL, timeout=3.0)
            raw = _parse_prometheus_full(resp.text)
            # Partial-scrape guard: if any core series is missing we'd interpret
            # it as a counter reset and blow away the ring buffers — treat the
            # router as absent for this cycle instead.
            _required = (
                "vllm_router_requests_total",
                "vllm_router_processed_requests_total",
                "vllm_router_generate_duration_seconds_sum",
                "vllm_router_generate_duration_seconds_count",
            )
            router_ok = all(raw.get(k) for k in _required)
        except Exception as e:
            logger.debug(f"router metrics scrape failed ({ROUTER_METRICS_URL}): {e!r}")
        dyn = await _scrape_dynamo(client)

    if not router_ok and not dyn["live"]:
        return
    if not router_ok:
        raw = {}
    now = time.time()

    active_workers = int(next((e["value"] for e in raw.get("vllm_router_active_workers", []) if not e["labels"]), 0))
    active_workers += dyn["active_workers"]

    live_instances: Dict[str, str] = {}
    if router_ok:
        try:
            async with httpx.AsyncClient() as client:
                wresp = await client.get(ROUTER_WORKERS_URL, timeout=2.0)
                for w in wresp.json().get("workers", []):
                    if w.get("url") and w.get("instance_id"):
                        live_instances[w["url"]] = w["instance_id"]
            _last_live_instances.clear()
            _last_live_instances.update(live_instances)
        except Exception as e:
            # Reuse the previous snapshot rather than failing open (see comment on
            # _last_live_instances above).
            logger.warning(f"/workers fetch failed, using last-known live set: {e!r}")
            live_instances = dict(_last_live_instances)

    def _is_live(labels: dict) -> bool:
        if not live_instances:
            return True
        url = labels.get("worker", "")
        inst = labels.get("instance", "")
        live = live_instances.get(url)
        if live is None:
            return False
        if inst and inst != live:
            return False
        return True

    def _live_map(metric_name: str) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for e in raw.get(metric_name, []):
            if _is_live(e["labels"]):
                out[e["labels"].get("worker", "_")] = int(e["value"])
        return out

    processed_map = _live_map("vllm_router_processed_requests_total")
    decisions_map = _live_map("vllm_router_policy_decisions_total")
    running_map = _live_map("vllm_router_running_requests")

    total_requests = sum(int(e["value"]) for e in raw.get("vllm_router_requests_total", []))
    retries_per_worker: Dict[str, int] = {}
    for e in raw.get("vllm_router_retries_total", []):
        if not _is_live(e["labels"]):
            continue
        w = e["labels"].get("worker", "_")
        retries_per_worker[w] = retries_per_worker.get(w, 0) + int(e["value"])
    total_retries = sum(retries_per_worker.values())

    lat_sum_per_worker: Dict[str, float] = {}
    lat_cnt_per_worker: Dict[str, int] = {}
    bucket_per_worker: Dict[str, Dict[str, int]] = {}

    for e in raw.get("vllm_router_generate_duration_seconds_sum", []):
        if not _is_live(e["labels"]):
            continue
        w = e["labels"].get("worker", "_")
        lat_sum_per_worker[w] = lat_sum_per_worker.get(w, 0.0) + float(e["value"])
    for e in raw.get("vllm_router_generate_duration_seconds_count", []):
        if not _is_live(e["labels"]):
            continue
        w = e["labels"].get("worker", "_")
        lat_cnt_per_worker[w] = lat_cnt_per_worker.get(w, 0) + int(e["value"])
    for e in raw.get("vllm_router_generate_duration_seconds_bucket", []):
        le = e["labels"].get("le")
        if le is None or le == "+Inf" or not _is_live(e["labels"]):
            continue
        w = e["labels"].get("worker", "_")
        bucket_per_worker.setdefault(le, {})[w] = bucket_per_worker.get(le, {}).get(w, 0) + int(e["value"])

    cb_state_map = {
        e["labels"].get("worker", "_"): int(e["value"])
        for e in raw.get("vllm_router_cb_state", [])
        if _is_live(e["labels"])
    }

    cb_outcomes: dict = {}
    for e in raw.get("vllm_router_cb_outcomes_total", []):
        if not _is_live(e["labels"]):
            continue
        w = e["labels"].get("worker", "_")
        outcome = e["labels"].get("outcome", "unknown")
        cb_outcomes.setdefault(w, {})[outcome] = int(e["value"])

    cb_transitions_raw = [
        e for e in raw.get("vllm_router_cb_state_transitions_total", [])
        if _is_live(e["labels"])
    ]

    # ── Fold the Dynamo data plane into the same maps ─────────────────────
    total_requests += dyn["total_requests"]
    processed_map.update(dyn["processed"])
    running_map.update(dyn["running"])
    lat_sum_per_worker.update(dyn["lat_sum"])
    lat_cnt_per_worker.update(dyn["lat_cnt"])
    for le, by_worker in dyn["bucket"].items():
        bucket_per_worker.setdefault(le, {}).update(by_worker)
    for u in dyn["live"]:
        live_instances[u] = dyn["live"][u]
        cb_outcomes[u] = {"success": dyn["succ"].get(u, 0), "failure": dyn["fail"].get(u, 0)}

    # Push ring-buffer samples (same logic as before, just runs unconditionally).
    _push_sample("global:requests", now, total_requests)
    _push_sample("global:retries", now, total_retries)
    for w_url, s in lat_sum_per_worker.items():
        _push_sample(f"lat_sum:{w_url}", now, s)
    for w_url, c in lat_cnt_per_worker.items():
        _push_sample(f"lat_cnt:{w_url}", now, c)
    for le, by_worker in bucket_per_worker.items():
        for w_url, cum in by_worker.items():
            _push_sample(f"bucket:{le}:{w_url}", now, cum)
    # Push proc samples for every worker we have a current value for. Iterating
    # `processed_map` (post-filter) instead of `live_instances.keys()` matters:
    # when the /workers fetch fails, live_instances is empty and `_is_live`
    # fails open, so processed_map still gets populated — but a live_instances
    # iteration would silently push nothing and the chart would freeze at 0
    # despite traffic flowing.
    for w_url, val in processed_map.items():
        _push_sample(f"proc:{w_url}", now, val)
    for w_url, rcnt in retries_per_worker.items():
        _push_sample(f"retry:{w_url}", now, rcnt)
    for w_url, oc in cb_outcomes.items():
        _push_sample(f"cb_succ:{w_url}", now, oc.get("success", 0))
        _push_sample(f"cb_fail:{w_url}", now, oc.get("failure", 0))
    for e in cb_transitions_raw:
        lbls = e["labels"]
        key = f"cb_trans:{lbls.get('worker','_')}|{lbls.get('from','?')}|{lbls.get('to','?')}"
        _push_sample(key, now, int(e["value"]))

    # Publish snapshot for the API layer.
    _latest_scrape.update({
        "now": now,
        "live_instances": live_instances,
        "processed_map": processed_map,
        "decisions_map": decisions_map,
        "running_map": running_map,
        "cb_state_map": cb_state_map,
        "cb_outcomes": cb_outcomes,
        "cb_transitions_raw": cb_transitions_raw,
        "total_requests": total_requests,
        "total_retries": total_retries,
        "lat_sum_per_worker": lat_sum_per_worker,
        "lat_cnt_per_worker": lat_cnt_per_worker,
        "bucket_per_worker": bucket_per_worker,
        "active_workers": active_workers,
        "retries_per_worker": retries_per_worker,
    })

    # Prune ring-buffer keys whose series stopped updating beyond the maximum
    # retention window. Without this, every removed/redeployed worker leaves
    # its proc:/retry:/lat_*:/bucket:/cb_* keys behind forever — a slow memory
    # leak plus phantom data for the windowed aggregations.
    stale_cut = now - _BUFFER_SECONDS_MAX
    for key in [k for k, dq in _metric_history.items() if not dq or dq[-1][0] < stale_cut]:
        del _metric_history[key]


async def _scrape_loop():
    while True:
        try:
            await _collect_metrics()
        except Exception as e:
            import sys
            print(f"[collect] loop error: {e}", file=sys.stderr)
        await asyncio.sleep(_SCRAPE_PERIOD_S)


@app.get("/api/prometheus_stats")
async def get_prometheus_stats(window: int = 900, served_model_name: Optional[str] = None):
    # Clamp to allowed values; pick nearest if caller sends something odd
    if window not in _ALLOWED_WINDOWS:
        window = min(_ALLOWED_WINDOWS, key=lambda x: abs(x - window))

    snap = _latest_scrape
    if not snap:
        return {
            "error": "metrics still warming up",
            "incomplete": True,
            "timestamp": round(time.time(), 3),
        }

    now = snap["now"]
    live_instances = snap["live_instances"]
    processed_map = snap["processed_map"]
    decisions_map = snap["decisions_map"]
    running_map = snap["running_map"]
    cb_state_map = snap["cb_state_map"]
    cb_outcomes = snap["cb_outcomes"]
    cb_transitions_raw = snap["cb_transitions_raw"]
    lat_sum_per_worker = snap["lat_sum_per_worker"]
    lat_cnt_per_worker = snap["lat_cnt_per_worker"]
    bucket_per_worker = snap["bucket_per_worker"]
    active_workers = snap["active_workers"]
    total_requests = snap["total_requests"]
    total_retries = snap["total_retries"]
    bucket_les = sorted({le for le in bucket_per_worker.keys()}, key=lambda x: float(x))

    retries_per_worker = snap["retries_per_worker"]

    # P2C worker URL → deployment info + worker_id, derived from Central's deployment records.
    # Built fresh per request (cheap, manager.load_deployments() is in-memory).
    url_to_dep: Dict[str, dict] = {}        # url → {id, name, served_model_name}
    url_to_worker_id: Dict[str, str] = {}   # url → worker_id
    deployments: Dict[str, dict] = {}
    _worker_id_re = re.compile(r'^(?:vllm|dynamo)_[^_]+_(.+)_\d+$')
    for dep in manager.load_deployments():
        dep_key = dep["id"]
        served = dep.get("served_model_name") or dep.get("model", "")
        deployments[dep_key] = {
            "id": dep_key,
            "name": dep["name"],
            "model": dep.get("model", ""),
            "served_model_name": served,
            "engine": dep.get("engine", "vllm"),
            "worker_urls": [],
        }
        for node in dep.get("nodes", []):
            url = _node_url(dep, node)
            url_to_dep[url] = {"id": dep_key, "name": dep["name"], "served_model_name": served}
            m = _worker_id_re.match(node.get("name", "") or "")
            if m:
                url_to_worker_id[url] = m.group(1)
            deployments[dep_key]["worker_urls"].append(url)

    # Resolve target worker set for this request: either the deployment's workers
    # (filtered by served_model_name) or every live worker (the unfiltered global view).
    if served_model_name:
        target_urls = {
            u for u, info in url_to_dep.items()
            if info.get("served_model_name") == served_model_name
        }
    else:
        # Include processed-map keys: a worker that served requests but has no
        # latency series yet (or only failed traffic) must still count toward
        # the per-worker tables and windowed sums.
        target_urls = (
            set(processed_map.keys())
            | set(lat_cnt_per_worker.keys())
            | set(lat_sum_per_worker.keys())
        )

    # Cumulative sum/count over the chosen target workers.
    latency_sum = sum(lat_sum_per_worker.get(u, 0.0) for u in target_urls)
    latency_count = sum(lat_cnt_per_worker.get(u, 0) for u in target_urls)

    # Cumulative histogram (current scrape, aggregated over target workers).
    cum_buckets = []
    for le in bucket_les:
        cum = sum(bucket_per_worker.get(le, {}).get(u, 0) for u in target_urls)
        cum_buckets.append({"le": le, "cum": cum})

    latency_hist_cum = []
    prev = 0
    for b in cum_buckets:
        latency_hist_cum.append({"le": b["le"], "count": b["cum"] - prev})
        prev = b["cum"]

    # Windowed histogram: sum per-worker bucket deltas across target workers.
    latency_hist_window = []
    prev_w = 0.0
    for le in bucket_les:
        cum_w = 0.0
        for u in target_urls:
            d = _window_delta(f"bucket:{le}:{u}", window)
            if d:
                cum_w += float(d[0])
        latency_hist_window.append({"le": le, "count": int(max(0.0, cum_w - prev_w))})
        prev_w = cum_w

    # Fallback host→name map (only used when we don't have a deployment record for the URL)
    host_to_name: Dict[str, str] = {}
    for wid, w in manager.get_workers().items():
        host_to_name.setdefault(w["host"], w.get("name") or wid)

    def _friendly_name(url: str) -> str:
        m = re.match(r'https?://([^:/]+):(\d+)', url)
        if not m:
            return url
        host, port = m.group(1), m.group(2)
        wid = url_to_worker_id.get(url) or host_to_name.get(host, host)
        return f"{wid}:{port}"

    # (Ring-buffer pushes happen in _collect_metrics() — this endpoint is read-only.)

    # Union of: workers with counter series (traffic since router start),
    # workers currently registered in the router (live_instances), and workers
    # attached to known deployments. Counter series alone made every idle
    # worker vanish from the UI after a router restart (fresh Prometheus
    # registry → only workers that had received traffic appeared).
    all_workers = sorted(
        set(processed_map.keys())
        | set(decisions_map.keys())
        | set(running_map.keys())
        | set(live_instances.keys())
        | set(url_to_dep.keys())
    )
    per_worker = []
    for w in all_workers:
        proc = processed_map.get(w, 0)
        dec = decisions_map.get(w, proc)
        outcomes = cb_outcomes.get(w, {})
        dep_info = url_to_dep.get(w, {})
        proc_window = _window_delta(f"proc:{w}", window)
        succ_w = _window_delta(f"cb_succ:{w}", window)
        fail_w = _window_delta(f"cb_fail:{w}", window)
        per_worker.append({
            "url": w,
            "name": _friendly_name(w),
            "deployment_id": dep_info.get("id"),
            "deployment_name": dep_info.get("name"),
            "served_model_name": dep_info.get("served_model_name"),
            "processed": proc,                                  # cumulative since router start
            "processed_window": int(proc_window[0]) if proc_window else 0,
            "decisions": dec,
            "running": running_map.get(w, 0),                   # real in-flight gauge (NOW)
            "cb_state": cb_state_map.get(w, -1),
            "cb_success": outcomes.get("success", 0),
            "cb_failure": outcomes.get("failure", 0),
            "cb_success_window": int(succ_w[0]) if succ_w else 0,
            "cb_failure_window": int(fail_w[0]) if fail_w else 0,
        })

    # Windowed aggregates driven by ?window= and the served_model_name filter.
    requests_window_total = 0
    duration_w = 0.0
    retries_window_total = 0
    if not served_model_name:
        # Global view: use the SAME label-free aggregate series the RPS chart
        # uses (global:requests / global:retries). Summing per-(worker,instance)
        # series here made the card systematically LOWER than the chart: every
        # worker re-registration mints a new instance ULID and resets that
        # worker's per-URL ring buffer (counter-reset clear), dropping its
        # contribution for up to a full window while the label-free aggregate
        # keeps counting — the "71 rps card vs 165 rps chart" symptom.
        gd = _window_delta("global:requests", window)
        if gd:
            requests_window_total = int(gd[0])
            duration_w = gd[1]
        grd = _window_delta("global:retries", window)
        if grd:
            retries_window_total = int(grd[0])
    else:
        # Per-model view: both the card and the chart use the same per-worker
        # proc:{url} series, so they stay consistent within the tab.
        for u in target_urls:
            pd = _window_delta(f"proc:{u}", window)
            if pd:
                requests_window_total += int(pd[0])
                duration_w = max(duration_w, pd[1])
        for u in target_urls:
            rd = _window_delta(f"retry:{u}", window)
            if rd:
                retries_window_total += int(rd[0])
    if duration_w <= 0:
        duration_w = float(window)
    rps_window = requests_window_total / duration_w

    # Share of upstream attempts that were retries — bounded [0, 100). The old
    # retries/successes ratio had no ceiling (a burst of retries against few
    # successes rendered as "355.83%").
    _attempts_window = requests_window_total + retries_window_total
    retry_rate_window = (retries_window_total / _attempts_window * 100) if _attempts_window > 0 else 0

    # Per-worker latency deltas summed across target_urls — respects served_model_name filter.
    lat_sum_window = 0.0
    lat_cnt_window = 0
    for u in target_urls:
        ds = _window_delta(f"lat_sum:{u}", window)
        dc = _window_delta(f"lat_cnt:{u}", window)
        if ds:
            lat_sum_window += float(ds[0])
        if dc:
            lat_cnt_window += int(dc[0])
    avg_latency_window = (lat_sum_window / lat_cnt_window) if lat_cnt_window > 0 else 0

    cb_transitions = []
    for e in cb_transitions_raw:
        lbls = e["labels"]
        worker_url = lbls.get("worker", "_")
        frm = lbls.get("from", "?")
        to = lbls.get("to", "?")
        tw = _window_delta(f"cb_trans:{worker_url}|{frm}|{to}", window)
        win_cnt = int(tw[0]) if tw else 0
        if win_cnt == 0:
            continue   # only show transitions that actually happened in the window
        cb_transitions.append({
            "worker": worker_url,
            "name": _friendly_name(worker_url),
            "deployment_id": url_to_dep.get(worker_url, {}).get("id"),
            "from_state": frm,
            "to_state": to,
            "count": int(e["value"]),       # cumulative (kept for tooltip / hover)
            "count_window": win_cnt,        # transitions in selected rolling window
        })

    return {
        "timestamp": round(now, 3),
        "window_seconds": window,                              # window actually used (after clamping)
        "allowed_windows": _ALLOWED_WINDOWS,
        # Live (now)
        "active_workers": active_workers,
        "total_in_flight": sum(running_map.values()),
        # Windowed aggregates
        "rps_window": round(rps_window, 2) if rps_window is not None else None,
        "requests_window": requests_window_total,
        "retries_window": retries_window_total,
        "retry_rate_window_pct": round(retry_rate_window, 2),
        "avg_latency_window_s": round(avg_latency_window, 2),
        # All-time cumulative
        "total_requests": int(total_requests),
        "total_retries": int(total_retries),
        # Per-worker (each has cumulative + windowed + live)
        "per_worker": per_worker,
        # Deployment list
        "deployments": list(deployments.values()),
        # Cumulative latency histogram
        "latency_histogram": latency_hist_cum,           # legacy / cumulative view
        "latency_histogram_window": latency_hist_window, # window-bounded distribution
        "latency_count": latency_count,
        "latency_sum": round(latency_sum, 1),
        "cb_transitions": cb_transitions,
    }

@app.get("/api/rps_history")
async def get_rps_history(window: int = 3600, served_model_name: Optional[str] = None,
                          deployment: Optional[str] = None):
    """Return a downsampled RPS time-series for the requested window.

    Filtering precedence:
      - served_model_name (preferred): sums per-worker processed across all deployments
        whose served_model_name matches.
      - deployment (legacy fallback): sums by deployment.name.
      - neither: uses the global vllm_router_requests_total counter.
    """
    if window not in _ALLOWED_HISTORY:
        window = min(_ALLOWED_HISTORY, key=lambda x: abs(x - window))

    now = time.time()
    cutoff = now - window

    # Bucket size has to be at least the scrape interval (frontend polls every
    # 5s) — otherwise some buckets randomly land between two samples and inherit
    # the previous bucket's rate via carry-forward, producing visible step/plateau
    # artefacts on what should be a continuous line.
    _SCRAPE_INTERVAL = 5.0
    target_points = max(2, min(360, int(window / _SCRAPE_INTERVAL)))
    bucket_size = window / target_points

    # buffer_too_short=True means the ring buffer does not extend back far enough
    # to cover the full window (typical right after a central restart). When that
    # holds we draw a flat 0 line across the no-data region so the chart's x-axis
    # always equals [now-window, now], no matter how much history is available.
    # Build per-series cumulative-count timelines. Global view uses the single
    # vllm_router_requests_total series; deployment-filtered view uses each
    # matching worker's proc:{url} series (summed at bucket boundaries below).
    series_list: list[list[tuple]] = []  # list of (ts, cum) sequences
    if not served_model_name and not deployment:
        dq = _metric_history.get("global:requests")
        if dq:
            filtered = [(ts, val) for ts, val in dq if ts >= cutoff]
            if filtered:
                series_list.append(filtered)
    else:
        worker_urls = []
        for dep in manager.load_deployments():
            served = dep.get("served_model_name") or dep.get("model", "")
            matches = (served_model_name and served == served_model_name) or \
                      (deployment and dep.get("name") == deployment)
            if matches:
                for node in dep.get("nodes", []):
                    worker_urls.append(_node_url(dep, node))
        if not worker_urls:
            return {"window_seconds": window, "samples": [], "allowed": _ALLOWED_HISTORY}
        for u in worker_urls:
            dq = _metric_history.get(f"proc:{u}")
            if not dq:
                continue
            filtered = [(ts, val) for ts, val in dq if ts >= cutoff]
            if not filtered:
                continue
            series_list.append(filtered)

    if not series_list:
        return {"window_seconds": window, "samples": [], "allowed": _ALLOWED_HISTORY}

    # Chart x-axis is always rolling: [now - window, now]. When some workers
    # have buffers shorter than the window, the bucket walker handles them via
    # the first-observed-value carry-forward in last_vals, so partial buffers
    # contribute zero delta for the period before they were seen — no anchor
    # mode is needed and the chart slides as time advances.
    x_min = cutoff
    x_max = now

    # Per-worker rate: (Δcum / Δt) at each consecutive sample pair. Each rate
    # represents the average growth across that specific interval — robust to
    # irregular pushes because every rate is already normalized by its own dt.
    # Total RPS at a moment = sum of each worker's latest rate as of that moment.
    worker_rates: list[list[tuple]] = []
    for series in series_list:
        rates = []
        for j in range(1, len(series)):
            ts_a, val_a = series[j-1]
            ts_b, val_b = series[j]
            dt = ts_b - ts_a
            if dt > 0:
                rates.append((ts_b, max(0.0, (val_b - val_a) / dt)))
        worker_rates.append(rates)

    # Seed last_rates with each worker's FIRST computed rate. Without this seed
    # the leftmost buckets emit 0 (no rate observed yet for any worker), and
    # the chart starts from 0 every time the page is opened. Seeding makes the
    # entire visible range reflect "the rate when we first saw this worker",
    # which is the best estimate we have for the period before that.
    last_rates = [r[0][1] if r else 0.0 for r in worker_rates]
    # Timestamp of the sample backing each carried rate. Seeded with the first
    # rate's ts so pre-history buckets (b_end < first sample) are never treated
    # as stale.
    last_seen = [r[0][0] if r else 0.0 for r in worker_rates]
    ptrs = [0] * len(worker_rates)

    # A series that stops updating (worker died / deregistered) must decay to 0
    # instead of contributing its last observed rate forever — otherwise the
    # chart keeps "phantom" throughput from dead workers for the rest of the
    # window and reads higher than the cards.
    _STALE_AFTER_S = 3 * _SCRAPE_INTERVAL

    def advance_rates_to(ts_target: float) -> None:
        for w_idx, rates in enumerate(worker_rates):
            while ptrs[w_idx] < len(rates) and rates[ptrs[w_idx]][0] <= ts_target:
                last_rates[w_idx] = rates[ptrs[w_idx]][1]
                last_seen[w_idx] = rates[ptrs[w_idx]][0]
                ptrs[w_idx] += 1

    advance_rates_to(x_min)

    out = []
    for i in range(target_points):
        b_start = x_min + i * bucket_size
        b_end = b_start + bucket_size
        advance_rates_to(b_end)
        rps = sum(
            rate
            for rate, seen in zip(last_rates, last_seen)
            if b_end - seen <= _STALE_AFTER_S
        )
        out.append({"ts": round(b_start, 3), "rps": round(rps, 2)})

    return {"window_seconds": window, "samples": out, "allowed": _ALLOWED_HISTORY}


@app.get("/metrics", response_class=HTMLResponse)
async def read_metrics(request: Request):
    return templates.TemplateResponse("metrics.html", {"request": request, "active_tab": "metrics"})

@app.get("/api/deployments/{deploy_id}/logs")
async def get_deployment_logs(deploy_id: str, container_name: Optional[str] = None):
    # Retrieve the async generator and return it as a streaming response
    return StreamingResponse(manager.stream_logs(deploy_id, container_name), media_type="text/event-stream")

@app.get("/api/endpoints/{worker_id}/images")
async def get_worker_images(worker_id: str):
    workers = manager.get_workers()
    if worker_id not in workers:
        raise HTTPException(status_code=404, detail="Worker not found")
    worker = workers[worker_id]
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"http://{worker['host']}:{worker['port']}/api/internal/images",
                timeout=10.0
            )
            if resp.status_code == 404:
                raise HTTPException(status_code=501, detail="Worker does not support image management. Please update worker code.")
            resp.raise_for_status()
            return resp.json()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

@app.post("/api/endpoints/{worker_id}/images/pull")
async def pull_worker_image(worker_id: str, request: Request):
    workers = manager.get_workers()
    if worker_id not in workers:
        raise HTTPException(status_code=404, detail="Worker not found")
    worker = workers[worker_id]
    body = await request.json()
    image = body.get("image", "").strip()
    if not image:
        raise HTTPException(status_code=400, detail="image field required")

    async def stream_pull():
        try:
            timeout = httpx.Timeout(connect=10.0, read=None, write=10.0, pool=10.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "POST",
                    f"http://{worker['host']}:{worker['port']}/api/internal/images/pull",
                    json={"image": image}
                ) as resp:
                    async for chunk in resp.aiter_bytes():
                        yield chunk
        except Exception as e:
            yield f"\n[Central Error] {e}\n".encode()

    return StreamingResponse(stream_pull(), media_type="text/plain")

@app.get("/api/endpoints/{worker_id}/models")
async def get_worker_models(worker_id: str):
    workers = manager.get_workers()
    if worker_id not in workers:
        raise HTTPException(status_code=404, detail="Worker not found")
    worker = workers[worker_id]
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"http://{worker['host']}:{worker['port']}/api/internal/models",
                timeout=30.0
            )
            if resp.status_code == 404:
                raise HTTPException(status_code=501, detail="Worker does not support model management. Please update worker code.")
            resp.raise_for_status()
            return resp.json()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

@app.post("/api/endpoints/{worker_id}/models/download")
async def download_worker_model(worker_id: str, request: Request):
    workers = manager.get_workers()
    if worker_id not in workers:
        raise HTTPException(status_code=404, detail="Worker not found")
    worker = workers[worker_id]
    body = await request.json()
    model_id = body.get("model_id", "").strip()
    force = bool(body.get("force", False))
    if not model_id:
        raise HTTPException(status_code=400, detail="model_id field required")
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"http://{worker['host']}:{worker['port']}/api/internal/models/download",
                json={"model_id": model_id, "force": force},
                timeout=10.0
            )
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

@app.get("/api/endpoints/{worker_id}/models/jobs")
async def get_worker_model_jobs(worker_id: str):
    workers = manager.get_workers()
    if worker_id not in workers:
        raise HTTPException(status_code=404, detail="Worker not found")
    worker = workers[worker_id]
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"http://{worker['host']}:{worker['port']}/api/internal/models/jobs",
                timeout=5.0
            )
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

@app.get("/api/endpoints/{worker_id}/models/jobs/{job_id}/logs")
async def stream_worker_job_logs(worker_id: str, job_id: str, offset: int = 0):
    workers = manager.get_workers()
    if worker_id not in workers:
        raise HTTPException(status_code=404, detail="Worker not found")
    worker = workers[worker_id]

    async def stream():
        try:
            timeout = httpx.Timeout(connect=10.0, read=None, write=10.0, pool=10.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "GET",
                    f"http://{worker['host']}:{worker['port']}/api/internal/models/jobs/{job_id}/logs",
                    params={"offset": offset}
                ) as resp:
                    async for chunk in resp.aiter_bytes():
                        yield chunk
        except Exception as e:
            yield f"\n[Central Error] {e}\n".encode()

    return StreamingResponse(stream(), media_type="text/plain")

@app.post("/api/internal/register_node")
async def register_node(req: RegisterNodeRequest):
    manager.register_worker(req.worker_id, req.host, req.port, req.gpus, req.version)
    return {"status": "ok"}

@app.get("/api/version")
async def get_versions():
    """Target commit + each worker's reported commit and drift, for the UI/CI."""
    import manager as manager_module
    target = await manager.get_target_version()
    return {
        "target": target,
        "branch": manager_module.WORKER_BRANCH,
        "auto_update": manager_module.WORKER_AUTO_UPDATE,
        "workers": manager.worker_version_status(target),
    }

@app.post("/api/workers/{worker_id}/update")
async def update_worker(worker_id: str, branch: Optional[str] = None):
    try:
        return await manager.update_worker(worker_id, branch)
    except Exception as e:
        raise HTTPException(status_code=409, detail=str(e))

@app.post("/api/workers/update_all")
async def update_all_workers(branch: Optional[str] = None):
    """Trigger self-update on every active, drifted worker. A self-update only
    recreates the worker AGENT container — the vLLM model containers it manages
    keep serving — so we do NOT skip workers that are currently serving."""
    target = await manager.get_target_version()
    results = {}
    for w in manager.worker_version_status(target):
        wid = w["worker_id"]
        if w["status"] != "active" or not w.get("drift"):
            results[wid] = "skipped"
            continue
        try:
            await manager.update_worker(wid, branch)
            results[wid] = "updating"
        except Exception as e:
            results[wid] = f"error: {e}"
    return {"target": target, "results": results}

async def health_check_loop():
    while True:
        try:
            await manager.run_health_checks()
        except Exception as e:
            logging.error(f"Error in health check loop: {e}")
        await asyncio.sleep(10)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(health_check_loop())
    asyncio.create_task(manager.sync_p2c_workers())
    asyncio.create_task(_scrape_loop())
    asyncio.create_task(manager.auto_update_loop())

