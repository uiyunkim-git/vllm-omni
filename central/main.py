from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse, PlainTextResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
from manager import CentralManager
import dynamo
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
    deployment_type: str                 # "replicas" or "tp"
    model: str
    served_model_name: Optional[str] = None
    # "dynamo" (default): runs `python -m dynamo.vllm`, served through the Dynamo
    # frontend. "vllm" (legacy): plain `vllm serve` with its own HTTPS endpoint,
    # not reachable through the gateway — debugging only.
    engine: Optional[str] = dynamo.DEFAULT_ENGINE
    gpus: List[str]                      # global GPU ids, e.g. ["neuron-worker-0"]
    tp: int = 1
    is_embedding: bool = False           # --embedding-worker --runner pooling
    max_len: Optional[int] = None
    gpu_util: Optional[float] = 0.9
    extra_args: Optional[str] = None
    image: Optional[str] = None          # engine image override
    vllm_image: Optional[str] = None     # legacy alias for `image`
    # Dynamo-side parsers; inferred from the model name when omitted.
    reasoning_parser: Optional[str] = None
    tool_call_parser: Optional[str] = None
    block_size: Optional[int] = None     # must match the frontend's --kv-cache-block-size

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
    """Deployments with their stored engine settings; each node carries the URL
    central uses for health and metrics (system port for dynamo)."""
    deps = manager.load_deployments()
    for dep in deps:
        for node in dep.get("nodes", []):
            node["url"] = dynamo.node_url(dep, node)
    return deps

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

# ──────────────────────────────────────────────────────────────────────────
# Metrics: Dynamo data plane.
#
# Two sources, scraped every _SCRAPE_PERIOD_S by one background task (so the
# ring buffers fill whether or not anyone has the dashboard open):
#
#   1. the Dynamo frontend  — `dynamo_frontend_*`, per served model: requests,
#      active/queued, request-duration + TTFT histograms, migrations, rejections.
#   2. every deployment node's system port — `dynamo_component_*` (Dynamo-side
#      request handling) and `vllm:*` (engine queue depth, KV usage).
#
# There is no router any more: no circuit breaker, no retries, no policy
# decisions. Nothing here talks to vllm_router_p2c.
# ──────────────────────────────────────────────────────────────────────────
from collections import deque

_BUFFER_SECONDS_MAX = 6 * 60 * 60  # keep 6h of samples per series (~4.3k points)
_ALLOWED_WINDOWS = [30, 60, 300, 900, 3600, 6 * 3600]
_ALLOWED_HISTORY = _ALLOWED_WINDOWS
_metric_history: Dict[str, deque] = {}  # key -> deque[(ts, value)]
_latest_scrape: dict = {}
_SCRAPE_PERIOD_S = 5.0


def _push_sample(key: str, ts: float, val: float) -> deque:
    dq = _metric_history.setdefault(key, deque())
    # Counters are monotonic — a drop means a real reset (frontend or worker
    # restarted), so the old history is not comparable and must go.
    if dq and val < dq[-1][1]:
        dq.clear()
    dq.append((ts, val))
    cutoff = ts - _BUFFER_SECONDS_MAX
    while len(dq) > 1 and dq[0][0] < cutoff:
        dq.popleft()
    return dq


def _window_delta(key: str, window_s: int):
    """(delta, duration_s) over the last `window_s` seconds, or None.

    Anchored to wall-clock `now`, not to the newest sample: a series that
    stopped updating (worker gone) must read as "no activity", not replay its
    whole historical climb into the current window.
    """
    dq = _metric_history.get(key)
    if not dq or len(dq) < 2:
        return None
    now = time.time()
    cutoff = now - window_s
    newest_ts, newest_val = dq[-1]
    if newest_ts < cutoff:
        return None
    oldest_ts = oldest_val = None
    for ts, val in dq:
        if ts >= cutoff:
            oldest_ts, oldest_val = ts, val
            break
    if oldest_ts is None:
        return None
    return max(0.0, newest_val - oldest_val), max(1e-6, newest_ts - oldest_ts)


def _dep_nodes() -> list:
    """[(url, dep, node)] for every node of every live deployment."""
    out = []
    for dep in manager.load_deployments():
        if dep.get("status") not in ("running", "starting"):
            continue
        for node in dep.get("nodes", []):
            out.append((dynamo.node_url(dep, node), dep, node))
    return out


async def _collect_metrics():
    """One scrape cycle: frontend + every instance, into the ring buffers."""
    now = time.time()
    nodes = _dep_nodes()
    fe: dict = {}
    instances: Dict[str, dict] = {}

    async with httpx.AsyncClient() as client:
        try:
            fe = dynamo.frontend_summary(await dynamo.frontend_metrics(client))
        except Exception as e:
            logger.debug(f"frontend metrics scrape failed: {e!r}")

        async def one(url, dep, node):
            if not dynamo.is_dynamo(dep):
                return  # legacy vLLM endpoints expose no Dynamo metrics
            s = await dynamo.scrape_instance(client, url)
            if s is not None:
                instances[url] = s

        await asyncio.gather(*(one(u, d, n) for u, d, n in nodes))

    if not fe and not instances:
        return

    # Per-model frontend series.
    for model, m in (fe.get("per_model") or {}).items():
        _push_sample(f"fe:req:{model}", now, m["requests_total"])
        _push_sample(f"fe:lat_sum:{model}", now, m["latency_sum"])
        _push_sample(f"fe:lat_cnt:{model}", now, m["latency_count"])
        _push_sample(f"fe:ttft_sum:{model}", now, m["ttft_sum"])
        _push_sample(f"fe:ttft_cnt:{model}", now, m["ttft_count"])
        _push_sample(f"fe:tok:{model}", now, m["output_tokens_total"])
        _push_sample(f"fe:mig:{model}", now, m["migrations_total"])
        _push_sample(f"fe:rej:{model}", now, m["rejections_total"])
        for le, cum in m["latency_buckets"].items():
            _push_sample(f"fe:bucket:{le}:{model}", now, cum)
        for le, cum in m["ttft_buckets"].items():
            _push_sample(f"fe:ttftb:{le}:{model}", now, cum)
    if fe:
        _push_sample("fe:req:__all__", now, fe["requests_total"])

    # Per-instance series.
    for url, s in instances.items():
        _push_sample(f"inst:req:{url}", now, s["requests_total"])
        _push_sample(f"inst:err:{url}", now, s["errors_total"])
        _push_sample(f"inst:lat_sum:{url}", now, s["duration_sum"])
        _push_sample(f"inst:lat_cnt:{url}", now, s["duration_count"])

    _latest_scrape.update({"now": now, "frontend": fe, "instances": instances})

    # Drop series nobody updates any more (removed worker / renamed model).
    stale_cut = now - _BUFFER_SECONDS_MAX
    for key in [k for k, dq in _metric_history.items() if not dq or dq[-1][0] < stale_cut]:
        del _metric_history[key]


async def _scrape_loop():
    while True:
        try:
            await _collect_metrics()
        except Exception as e:
            logger.warning(f"[collect] loop error: {e!r}")
        await asyncio.sleep(_SCRAPE_PERIOD_S)


def _instance_rows() -> list:
    """Deployment nodes joined with their latest scrape — the row model shared
    by /api/instances and the per-instance table in /api/prometheus_stats."""
    snap_inst = (_latest_scrape.get("instances") or {})
    rows = []
    for url, dep, node in _dep_nodes():
        s = snap_inst.get(url, {})
        gpu = ""
        m = re.match(r"^(?:dynamo|vllm|ollama)_[^_]+_(.+)_(\d+)$", node.get("name", "") or "")
        worker_id, gpu = (m.group(1), m.group(2)) if m else ("", "")
        rows.append({
            "url": url,
            "system_url": url,           # contract alias
            "name": f"{worker_id or node['host']}:{dynamo.node_api_port(dep, node)}",
            "deployment_id": dep["id"],
            "deployment_name": dep.get("name"),
            "model": dep.get("model"),
            "served_model_name": dep.get("served_model_name") or dep.get("model"),
            "engine": dep.get("engine", dynamo.DEFAULT_ENGINE),
            "worker_id": worker_id,
            "host": node["host"],
            "gpu": gpu,
            "healthy": bool(node.get("is_healthy")),
            "running": s.get("running", 0),
            "waiting": s.get("waiting", 0),
            "inflight": s.get("inflight", 0),
            "kv_cache_usage_pct": s.get("kv_cache_usage_pct", 0.0),
            "processed": s.get("requests_total", 0),
            "requests_total": s.get("requests_total", 0),   # contract alias
            "errors": s.get("errors_total", 0),
            "errors_total": s.get("errors_total", 0),       # contract alias
        })
    return rows


@app.get("/api/frontend")
async def get_frontend_status():
    """Health + discovered models of the Dynamo frontend (the single ingress)."""
    healthy, models = False, []
    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(f"{dynamo.FRONTEND_URL}/health", timeout=3.0)
            healthy = r.status_code == 200
        except Exception:
            healthy = False
        try:
            models = await dynamo.frontend_models(client)
        except Exception:
            models = []
        ns_instances: Dict[str, list] = {}
        try:
            ns_instances = dynamo.generate_instances_by_namespace(await dynamo.etcd_instance_keys(client))
        except Exception as e:
            logger.debug(f"etcd instance listing failed: {e!r}")

    fe = _latest_scrape.get("frontend") or {}
    per_model = fe.get("per_model") or {}
    out_models = []
    for m in models:
        mid = m.get("id")
        ns = dynamo.namespace_for(mid)
        pm = per_model.get(mid, {})
        n_inst = len(ns_instances.get(ns, []))
        out_models.append({
            "id": mid,
            "namespace": ns,
            "instances": n_inst,
            # The frontend lists a model only once a worker set is registered, so
            # "discovered with at least one live instance" IS readiness here.
            # (dynamo_frontend_model_ready is not exported by 1.4.2.)
            "ready": n_inst > 0,
            "active_requests": pm.get("active_requests", 0),
            "queued_requests": pm.get("queued_requests", 0),
        })
    return {
        "url": dynamo.FRONTEND_URL,
        "healthy": healthy,
        "router_mode": os.environ.get("DYNAMO_ROUTER_MODE", "least-loaded"),
        "kv_block_size": dynamo.KV_BLOCK_SIZE,
        "models": out_models,
        "metrics": {
            "active_requests": fe.get("active_requests", 0),
            "queued_requests": fe.get("queued_requests", 0),
            "requests_total": fe.get("requests_total", 0),
            "output_tokens_total": fe.get("output_tokens_total", 0),
        },
    }


@app.get("/api/instances")
async def get_instances():
    return _instance_rows()


@app.get("/api/prometheus_stats")
async def get_prometheus_stats(window: int = 900, served_model_name: Optional[str] = None):
    if window not in _ALLOWED_WINDOWS:
        window = min(_ALLOWED_WINDOWS, key=lambda x: abs(x - window))

    fe = _latest_scrape.get("frontend") or {}
    per_model = fe.get("per_model") or {}
    rows = _instance_rows()
    if served_model_name:
        rows = [r for r in rows if r["served_model_name"] == served_model_name]
        models = [served_model_name]
    else:
        models = list(per_model.keys()) or sorted({r["served_model_name"] for r in rows})

    # Windowed aggregates come from the frontend counters (one authoritative
    # series per model) — the per-instance counters are for the table only.
    requests_window = 0
    duration_w = 0.0
    lat_sum_w = lat_cnt_w = 0.0
    ttft_sum_w = ttft_cnt_w = 0.0
    migrations_w = rejections_w = 0
    tokens_w = 0
    lat_buckets_w: Dict[str, float] = {}
    ttft_buckets_w: Dict[str, float] = {}
    for model in models:
        d = _window_delta(f"fe:req:{model}", window)
        if d:
            requests_window += int(d[0])
            duration_w = max(duration_w, d[1])
        for key, acc in (("lat_sum", "ls"), ("lat_cnt", "lc"), ("ttft_sum", "ts"), ("ttft_cnt", "tc")):
            dd = _window_delta(f"fe:{key}:{model}", window)
            if not dd:
                continue
            if acc == "ls":
                lat_sum_w += dd[0]
            elif acc == "lc":
                lat_cnt_w += dd[0]
            elif acc == "ts":
                ttft_sum_w += dd[0]
            else:
                ttft_cnt_w += dd[0]
        for name, target in (("mig", "m"), ("rej", "r"), ("tok", "t")):
            dd = _window_delta(f"fe:{name}:{model}", window)
            if not dd:
                continue
            if target == "m":
                migrations_w += int(dd[0])
            elif target == "r":
                rejections_w += int(dd[0])
            else:
                tokens_w += int(dd[0])
        for prefix, sink in (("fe:bucket", lat_buckets_w), ("fe:ttftb", ttft_buckets_w)):
            for key in [k for k in _metric_history if k.startswith(f"{prefix}:") and k.endswith(f":{model}")]:
                le = key[len(prefix) + 1: -(len(model) + 1)]
                dd = _window_delta(key, window)
                if dd:
                    sink[le] = sink.get(le, 0.0) + dd[0]

    if duration_w <= 0:
        duration_w = float(window)

    def _hist(cum: Dict[str, float]) -> list:
        out, prev = [], 0.0
        for le in sorted(cum, key=float):
            out.append({"le": le, "count": int(max(0.0, cum[le] - prev))})
            prev = cum[le]
        return out

    per_worker = []
    for r in rows:
        pw = _window_delta(f"inst:req:{r['url']}", window)
        ew = _window_delta(f"inst:err:{r['url']}", window)
        per_worker.append({
            **r,
            "processed_window": int(pw[0]) if pw else 0,
            "errors_window": int(ew[0]) if ew else 0,
        })

    active = sum(per_model.get(m, {}).get("active_requests", 0) for m in models) if per_model else 0
    queued = sum(per_model.get(m, {}).get("queued_requests", 0) for m in models) if per_model else 0

    return {
        "timestamp": round(_latest_scrape.get("now", time.time()), 3),
        "window_seconds": window,
        "allowed_windows": _ALLOWED_WINDOWS,
        # live
        "active_workers": len(rows),
        "active_requests": active,
        "queued_requests": queued,
        "total_in_flight": sum(r["running"] for r in rows),
        # windowed
        "rps_window": round(requests_window / duration_w, 2),
        "requests_window": requests_window,
        "output_tokens_window": tokens_w,
        "avg_latency_window_s": round(lat_sum_w / lat_cnt_w, 2) if lat_cnt_w else 0,
        "avg_ttft_window_s": round(ttft_sum_w / ttft_cnt_w, 3) if ttft_cnt_w else 0,
        "ttft_p50_s": round(dynamo.percentile_from_buckets(ttft_buckets_w, 0.50) or 0, 3),
        "ttft_p95_s": round(dynamo.percentile_from_buckets(ttft_buckets_w, 0.95) or 0, 3),
        "migrations_window": migrations_w,
        "rejections_window": rejections_w,
        # cumulative
        "total_requests": sum(per_model.get(m, {}).get("requests_total", 0) for m in models),
        # tables
        "per_worker": per_worker,
        "latency_histogram_window": _hist(lat_buckets_w),
        "ttft_histogram_window": _hist(ttft_buckets_w),
        "deployments": [
            {
                "id": d["id"], "name": d.get("name"), "model": d.get("model"),
                "served_model_name": d.get("served_model_name") or d.get("model"),
                "engine": d.get("engine", dynamo.DEFAULT_ENGINE),
            }
            for d in manager.load_deployments()
        ],
    }


@app.get("/api/rps_history")
async def get_rps_history(window: int = 3600, served_model_name: Optional[str] = None):
    """Downsampled requests/s time series over `window`, from the frontend's
    per-model counters (all models summed when no filter is given)."""
    if window not in _ALLOWED_HISTORY:
        window = min(_ALLOWED_HISTORY, key=lambda x: abs(x - window))
    now = time.time()
    cutoff = now - window
    target_points = max(2, min(360, int(window / _SCRAPE_PERIOD_S)))
    bucket_size = window / target_points

    if served_model_name:
        keys = [f"fe:req:{served_model_name}"]
    else:
        keys = [k for k in _metric_history if k.startswith("fe:req:") and k != "fe:req:__all__"] or ["fe:req:__all__"]

    series_list = []
    for k in keys:
        dq = _metric_history.get(k)
        if not dq:
            continue
        filtered = [(ts, v) for ts, v in dq if ts >= cutoff]
        if len(filtered) >= 2:
            series_list.append(filtered)
    if not series_list:
        return {"window_seconds": window, "samples": [], "allowed": _ALLOWED_HISTORY}

    # Per-series rate at each consecutive sample pair; total = sum of the most
    # recent rate of every series that is still fresh.
    rates_per_series = []
    for series in series_list:
        rates = []
        for j in range(1, len(series)):
            (ts_a, v_a), (ts_b, v_b) = series[j - 1], series[j]
            dt = ts_b - ts_a
            if dt > 0:
                rates.append((ts_b, max(0.0, (v_b - v_a) / dt)))
        rates_per_series.append(rates)

    last_rates = [r[0][1] if r else 0.0 for r in rates_per_series]
    last_seen = [r[0][0] if r else 0.0 for r in rates_per_series]
    ptrs = [0] * len(rates_per_series)
    _STALE_AFTER_S = 3 * _SCRAPE_PERIOD_S

    def advance_to(ts_target: float) -> None:
        for i, rates in enumerate(rates_per_series):
            while ptrs[i] < len(rates) and rates[ptrs[i]][0] <= ts_target:
                last_rates[i] = rates[ptrs[i]][1]
                last_seen[i] = rates[ptrs[i]][0]
                ptrs[i] += 1

    advance_to(cutoff)
    out = []
    for i in range(target_points):
        b_start = cutoff + i * bucket_size
        b_end = b_start + bucket_size
        advance_to(b_end)
        rps = sum(rate for rate, seen in zip(last_rates, last_seen) if b_end - seen <= _STALE_AFTER_S)
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
    asyncio.create_task(_scrape_loop())
    asyncio.create_task(manager.auto_update_loop())

