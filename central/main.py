from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse, PlainTextResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
from manager import CentralManager
import logging
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

_BUFFER_SECONDS_MAX = 24 * 60 * 60  # keep 24h of samples on the server
# Single, unified window vocabulary shared by cards / per-worker bars / RPS chart.
_ALLOWED_WINDOWS = [900, 3600, 6 * 3600, 12 * 3600, 24 * 3600]  # 15m, 1h, 6h, 12h, 24h
_ALLOWED_HISTORY = _ALLOWED_WINDOWS  # alias, used by rps_history endpoint
_metric_history: Dict[str, deque] = {}  # key -> deque[(ts, value)]


def _push_sample(key: str, ts: float, val: float) -> deque:
    dq = _metric_history.setdefault(key, deque())
    # Detect Prometheus counter reset (e.g. router restart): the counter went backwards.
    # All older samples are meaningless for windowed rate calc, so drop them.
    if dq and val < dq[-1][1]:
        dq.clear()
    dq.append((ts, val))
    cutoff = ts - _BUFFER_SECONDS_MAX
    while len(dq) > 1 and dq[0][0] < cutoff:
        dq.popleft()
    return dq


def _window_delta(key: str, window_s: int):
    """Return (delta, duration_s) for samples in the last `window_s` seconds, or None."""
    dq = _metric_history.get(key)
    if not dq or len(dq) < 2:
        return None
    newest_ts, newest_val = dq[-1]
    cutoff = newest_ts - window_s
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


@app.get("/api/prometheus_stats")
async def get_prometheus_stats(window: int = 900):
    # Clamp to allowed values; pick nearest if caller sends something odd
    if window not in _ALLOWED_WINDOWS:
        window = min(_ALLOWED_WINDOWS, key=lambda x: abs(x - window))
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get("http://143.248.74.105:29000/metrics", timeout=3.0)
            text = resp.text
    except Exception as e:
        return {"error": str(e)}

    raw = _parse_prometheus_full(text)
    now = time.time()

    active_workers = int(next((e["value"] for e in raw.get("vllm_router_active_workers", []) if not e["labels"]), 0))

    processed_map = {e["labels"].get("worker", "_"): int(e["value"]) for e in raw.get("vllm_router_processed_requests_total", [])}
    decisions_map = {e["labels"].get("worker", "_"): int(e["value"]) for e in raw.get("vllm_router_policy_decisions_total", [])}
    running_map = {e["labels"].get("worker", "_"): int(e["value"]) for e in raw.get("vllm_router_running_requests", [])}

    total_requests = sum(int(e["value"]) for e in raw.get("vllm_router_requests_total", []))
    total_retries = sum(int(e["value"]) for e in raw.get("vllm_router_retries_total", []))

    latency_count = int(next((e["value"] for e in raw.get("vllm_router_generate_duration_seconds_count", [])), 0))
    latency_sum = next((e["value"] for e in raw.get("vllm_router_generate_duration_seconds_sum", [])), 0.0)

    dur_buckets = raw.get("vllm_router_generate_duration_seconds_bucket", [])
    # Cumulative bucket values right now (current scrape).
    cum_buckets = sorted(
        [{"le": e["labels"]["le"], "cum": int(e["value"])} for e in dur_buckets if "le" in e["labels"] and e["labels"]["le"] != "+Inf"],
        key=lambda x: float(x["le"])
    )

    # Push per-bucket samples to the ring buffer so we can compute window-bounded
    # latency distributions instead of only ever-since-restart cumulative ones.
    for b in cum_buckets:
        _push_sample(f"bucket:{b['le']}", now, b["cum"])

    # Build the histogram BOTH ways:
    #  - cumulative (for cumulative section, if ever used)
    #  - windowed (cum_now - cum_window_start), shown in Rolling window
    latency_hist_cum = []
    prev = 0
    for b in cum_buckets:
        latency_hist_cum.append({"le": b["le"], "count": b["cum"] - prev})
        prev = b["cum"]

    latency_hist_window = []
    prev_w = 0.0
    for b in cum_buckets:
        delta = _window_delta(f"bucket:{b['le']}", window)
        cum_w = float(delta[0]) if delta else 0.0
        latency_hist_window.append({"le": b["le"], "count": int(max(0.0, cum_w - prev_w))})
        prev_w = cum_w

    cb_state_map = {e["labels"].get("worker", "_"): int(e["value"]) for e in raw.get("vllm_router_cb_state", [])}

    cb_outcomes: dict = {}
    for e in raw.get("vllm_router_cb_outcomes_total", []):
        w = e["labels"].get("worker", "_")
        outcome = e["labels"].get("outcome", "unknown")
        cb_outcomes.setdefault(w, {})[outcome] = int(e["value"])

    cb_transitions_raw = raw.get("vllm_router_cb_state_transitions_total", [])

    # P2C worker URL → deployment info + worker_id, derived from Central's deployment records.
    # Multiple workers can share a host (e.g. KBDS has 6 worker_ids on the same Tailscale IP),
    # so host alone is not a unique key — we use (host, ext_port) → deployment.node.
    url_to_dep: Dict[str, dict] = {}        # url → {id, name, served_model_name}
    url_to_worker_id: Dict[str, str] = {}   # url → worker_id (e.g. "kbds-worker-003")
    deployments: Dict[str, dict] = {}
    _worker_id_re = re.compile(r'^vllm_[^_]+_(.+)_\d+$')
    for dep in manager.load_deployments():
        dep_key = dep["id"]
        served = dep.get("served_model_name") or dep.get("model", "")
        deployments[dep_key] = {
            "id": dep_key,
            "name": dep["name"],
            "model": dep.get("model", ""),
            "served_model_name": served,
            "worker_urls": [],
        }
        for node in dep.get("nodes", []):
            ext_port = node["port"] + 40000
            url = f'https://{node["host"]}:{ext_port}'
            url_to_dep[url] = {"id": dep_key, "name": dep["name"], "served_model_name": served}
            m = _worker_id_re.match(node.get("name", "") or "")
            if m:
                url_to_worker_id[url] = m.group(1)
            deployments[dep_key]["worker_urls"].append(url)

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

    # Record 15-min ring-buffer samples
    _push_sample("global:requests", now, total_requests)
    _push_sample("global:retries", now, total_retries)
    _push_sample("global:latency_sum", now, latency_sum)
    _push_sample("global:latency_count", now, latency_count)
    for w_url, proc in processed_map.items():
        _push_sample(f"proc:{w_url}", now, proc)

    all_workers = sorted(set(list(processed_map.keys()) + list(decisions_map.keys()) + list(running_map.keys())))
    per_worker = []
    for w in all_workers:
        proc = processed_map.get(w, 0)
        dec = decisions_map.get(w, proc)
        outcomes = cb_outcomes.get(w, {})
        dep_info = url_to_dep.get(w, {})
        proc_window = _window_delta(f"proc:{w}", window)
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
        })

    # Windowed aggregates (driven by ?window=)
    requests_w = _window_delta("global:requests", window)
    retries_w  = _window_delta("global:retries", window)
    lat_sum_w  = _window_delta("global:latency_sum", window)
    lat_cnt_w  = _window_delta("global:latency_count", window)

    rps_window = (requests_w[0] / requests_w[1]) if requests_w else None
    requests_window_total = int(requests_w[0]) if requests_w else 0
    retries_window_total  = int(retries_w[0]) if retries_w else 0
    retry_rate_window     = (retries_window_total / requests_window_total * 100) if requests_window_total > 0 else 0
    avg_latency_window    = (lat_sum_w[0] / lat_cnt_w[0]) if lat_sum_w and lat_cnt_w and lat_cnt_w[0] > 0 else 0

    cb_transitions = [
        {
            "worker": e["labels"].get("worker", "_"),
            "name": _friendly_name(e["labels"].get("worker", "_")),
            "deployment_id": url_to_dep.get(e["labels"].get("worker", "_"), {}).get("id"),
            "from_state": e["labels"].get("from", "?"),
            "to_state": e["labels"].get("to", "?"),
            "count": int(e["value"]),
        }
        for e in cb_transitions_raw
    ]

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

    if not served_model_name and not deployment:
        dq = _metric_history.get("global:requests")
        samples = [(ts, val) for ts, val in (dq or []) if ts >= cutoff]
    else:
        worker_urls = []
        for dep in manager.load_deployments():
            served = dep.get("served_model_name") or dep.get("model", "")
            matches = (served_model_name and served == served_model_name) or \
                      (deployment and dep.get("name") == deployment)
            if matches:
                for node in dep.get("nodes", []):
                    worker_urls.append(f'https://{node["host"]}:{node["port"] + 40000}')
        if not worker_urls:
            return {"window_seconds": window, "samples": [], "allowed": _ALLOWED_HISTORY}
        # Merge ring buffers by timestamp using a per-second-level approximation
        # (we sum the closest sample per timestamp from each worker's series).
        per_worker = []
        for u in worker_urls:
            dq = _metric_history.get(f"proc:{u}")
            if dq:
                per_worker.append([(ts, val) for ts, val in dq if ts >= cutoff])
        if not per_worker:
            return {"window_seconds": window, "samples": [], "allowed": _ALLOWED_HISTORY}
        # Build a sorted union of timestamps, sum the latest-known per-worker value at each ts.
        ts_set = sorted({ts for series in per_worker for ts, _ in series})
        # Walking pointers per worker to find the latest value <= ts
        idxs = [0] * len(per_worker)
        last_vals = [0] * len(per_worker)
        samples = []
        for ts in ts_set:
            for i, series in enumerate(per_worker):
                while idxs[i] < len(series) and series[idxs[i]][0] <= ts:
                    last_vals[i] = series[idxs[i]][1]
                    idxs[i] += 1
            samples.append((ts, sum(last_vals)))

    if len(samples) < 2:
        return {"window_seconds": window, "samples": [], "allowed": _ALLOWED_HISTORY}

    # Downsample to ~360 buckets for any window
    target_points = 360
    bucket_size = window / target_points

    out = []
    prev_ts, prev_val = samples[0]
    next_bucket_end = prev_ts + bucket_size
    bucket_first_ts, bucket_first_val = prev_ts, prev_val
    for ts, val in samples[1:]:
        if ts >= next_bucket_end:
            # close out the previous bucket using first..last sample in the bucket
            dt = max(1e-6, prev_ts - bucket_first_ts)
            rate = max(0.0, (prev_val - bucket_first_val) / dt)
            out.append({"ts": round(prev_ts, 3), "rps": round(rate, 2)})
            bucket_first_ts, bucket_first_val = prev_ts, prev_val
            next_bucket_end = ts + bucket_size
        prev_ts, prev_val = ts, val
    # last bucket
    dt = max(1e-6, prev_ts - bucket_first_ts)
    rate = max(0.0, (prev_val - bucket_first_val) / dt)
    out.append({"ts": round(prev_ts, 3), "rps": round(rate, 2)})

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
    if not model_id:
        raise HTTPException(status_code=400, detail="model_id field required")
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"http://{worker['host']}:{worker['port']}/api/internal/models/download",
                json={"model_id": model_id},
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
    manager.register_worker(req.worker_id, req.host, req.port, req.gpus)
    return {"status": "ok"}

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

