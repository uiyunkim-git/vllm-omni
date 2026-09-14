"""Dynamo data-plane integration for central.

Single source of truth for everything central needs to know about the Dynamo
side of the platform:

  * how a served model maps to a Dynamo namespace,
  * which ports a worker instance exposes,
  * how to read the frontend's and the instances' Prometheus endpoints,
  * how to list live instances straight from etcd.

The deployment path (manager.py) and the metrics path (main.py) both import
from here so the two can never disagree about a URL or a namespace again.
"""
from __future__ import annotations

import base64
import hashlib
import os
import re
from typing import Dict, List, Optional

import httpx

# ── Topology ────────────────────────────────────────────────────────────────
FRONTEND_URL = os.environ.get("DYNAMO_FRONTEND_URL", "http://143.248.74.105:11434").rstrip("/")
ETCD_ENDPOINT = os.environ.get("DYNAMO_ETCD_ENDPOINTS", "http://143.248.74.105:2379").split(",")[0].rstrip("/")
NAMESPACE_PREFIX = os.environ.get("DYNAMO_NAMESPACE", "dynamo")
KV_BLOCK_SIZE = int(os.environ.get("DYNAMO_KV_BLOCK_SIZE", "64"))
ENGINE_IMAGE = os.environ.get("DYNAMO_IMAGE", "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.2")

# Port scheme. A deployment node stores one base `port`; every listener is
# derived from it. MUST match worker/manager.py.
#   dynamo: system(health/metrics) +10000, request-plane +12000,
#           response-stream +42000, kv-events +44000
#   vllm (legacy, direct endpoint): OpenAI API +40000 over HTTPS
SYSTEM_PORT_OFFSET = 10000
RPC_PORT_OFFSET = 12000
RESP_PORT_OFFSET = 42000
KV_PORT_OFFSET = 44000
VLLM_API_PORT_OFFSET = 40000

DEFAULT_ENGINE = "dynamo"


def is_dynamo(dep: dict) -> bool:
    return (dep or {}).get("engine", DEFAULT_ENGINE) == "dynamo"


def node_url(dep: dict, node: dict) -> str:
    """The URL central uses as this node's identity and metrics source."""
    if is_dynamo(dep):
        return f"http://{node['host']}:{node['port'] + SYSTEM_PORT_OFFSET}"
    return f"https://{node['host']}:{node['port'] + VLLM_API_PORT_OFFSET}"


def node_api_port(dep: dict, node: dict) -> int:
    off = SYSTEM_PORT_OFFSET if is_dynamo(dep) else VLLM_API_PORT_OFFSET
    return node["port"] + off


def namespace_for(served_model_name: str) -> str:
    """Dynamo allows ONE model per (namespace, component, endpoint): a second
    model registering on `<ns>/backend/generate` is rejected with "a different
    model is already registered there". So every served model gets its own
    namespace; the frontend discovers them all via --namespace-prefix.
    Replicas of one model share a namespace and form a single worker pool."""
    slug = re.sub(r"[^a-z0-9]+", "-", (served_model_name or "model").lower()).strip("-")
    ns = f"{NAMESPACE_PREFIX}-{slug}"
    if len(ns) <= 63:
        return ns
    # Truncating alone would map two long names sharing a prefix onto the same
    # namespace, and Dynamo rejects the second model with "a different model is
    # already registered there". Keep the truncation readable but unique.
    digest = hashlib.sha1(served_model_name.encode()).hexdigest()[:8]
    return f"{ns[:63 - 9].rstrip('-')}-{digest}"


def infer_parsers(model: str) -> tuple:
    """(reasoning_parser, tool_call_parser) for --dyn-* flags, by model family.
    These live on the frontend side in Dynamo, so they must be set at deploy."""
    m = (model or "").lower()
    if "gpt-oss" in m:
        return "gpt_oss", "harmony"
    if "gemma-4" in m or "gemma4" in m:
        return "gemma4", "gemma4"
    if "qwen3" in m:
        return "qwen3", "hermes"
    if "deepseek-v4" in m:
        return "deepseek_v4", "deepseek_v4"
    return None, None


# ── Prometheus ──────────────────────────────────────────────────────────────
_SAMPLE_RE = re.compile(r"^(\w[\w:]*)(\{[^}]*\})?\s+(\S+)")
_LABEL_RE = re.compile(r'(\w+)="([^"]*)"')


def parse_prometheus(text: str) -> Dict[str, list]:
    """{metric_name: [{"labels": {...}, "value": float}, ...]}"""
    out: Dict[str, list] = {}
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        m = _SAMPLE_RE.match(line)
        if not m:
            continue
        name, labels_str, val_str = m.group(1), m.group(2) or "", m.group(3)
        try:
            val = float(val_str)
        except ValueError:
            continue
        out.setdefault(name, []).append({"labels": dict(_LABEL_RE.findall(labels_str)), "value": val})
    return out


def _sum(raw: Dict[str, list], metric: str, **label_filter) -> float:
    total = 0.0
    for e in raw.get(metric, []):
        if all(e["labels"].get(k) == v for k, v in label_filter.items()):
            total += e["value"]
    return total


def _by_label(raw: Dict[str, list], metric: str, label: str = "model") -> Dict[str, float]:
    out: Dict[str, float] = {}
    for e in raw.get(metric, []):
        key = e["labels"].get(label)
        if key is None:
            continue
        out[key] = out.get(key, 0.0) + e["value"]
    return out


def _buckets_by_label(raw: Dict[str, list], metric: str, label: str = "model") -> Dict[str, Dict[str, float]]:
    """{model: {le: cumulative_count}} — '+Inf' dropped."""
    out: Dict[str, Dict[str, float]] = {}
    for e in raw.get(metric, []):
        le = e["labels"].get("le")
        key = e["labels"].get(label)
        if le is None or le == "+Inf" or key is None:
            continue
        out.setdefault(key, {})[le] = out.setdefault(key, {}).get(le, 0.0) + e["value"]
    return out


def percentile_from_buckets(buckets: Dict[str, float], q: float) -> Optional[float]:
    """Approximate quantile from cumulative histogram buckets {le: count}.
    `buckets` may hold window deltas; counts only have to be monotonic in le."""
    if not buckets:
        return None
    items = sorted(((float(le), c) for le, c in buckets.items()), key=lambda x: x[0])
    total = items[-1][1]
    if total <= 0:
        return None
    target = q * total
    prev_le, prev_c = 0.0, 0.0
    for le, c in items:
        if c >= target:
            # linear interpolation inside the bucket
            span = c - prev_c
            frac = (target - prev_c) / span if span > 0 else 0.0
            return prev_le + (le - prev_le) * frac
        prev_le, prev_c = le, c
    return items[-1][0]


# ── Frontend ────────────────────────────────────────────────────────────────
async def frontend_models(client: httpx.AsyncClient, timeout: float = 3.0) -> List[dict]:
    r = await client.get(f"{FRONTEND_URL}/v1/models", timeout=timeout)
    r.raise_for_status()
    return r.json().get("data", [])


async def frontend_metrics(client: httpx.AsyncClient, timeout: float = 3.0) -> Dict[str, list]:
    r = await client.get(f"{FRONTEND_URL}/metrics", timeout=timeout)
    r.raise_for_status()
    return parse_prometheus(r.text)


def frontend_summary(raw: Dict[str, list]) -> dict:
    """Per-model + global view of the frontend's own counters."""
    per_model = {}
    requests = _by_label(raw, "dynamo_frontend_requests_total")
    active = _by_label(raw, "dynamo_frontend_active_requests")
    queued = _by_label(raw, "dynamo_frontend_queued_requests")
    out_tokens = _by_label(raw, "dynamo_frontend_output_tokens_total")
    lat_sum = _by_label(raw, "dynamo_frontend_request_duration_seconds_sum")
    lat_cnt = _by_label(raw, "dynamo_frontend_request_duration_seconds_count")
    ttft_sum = _by_label(raw, "dynamo_frontend_time_to_first_token_seconds_sum")
    ttft_cnt = _by_label(raw, "dynamo_frontend_time_to_first_token_seconds_count")
    migrations = _by_label(raw, "dynamo_frontend_model_migration_total")
    rejections = _by_label(raw, "dynamo_frontend_model_rejection_total")
    ready = _by_label(raw, "dynamo_frontend_model_ready")
    lat_buckets = _buckets_by_label(raw, "dynamo_frontend_request_duration_seconds_bucket")
    ttft_buckets = _buckets_by_label(raw, "dynamo_frontend_time_to_first_token_seconds_bucket")

    for model in set().union(requests, active, queued, lat_cnt, ready):
        per_model[model] = {
            "requests_total": int(requests.get(model, 0)),
            "active_requests": int(active.get(model, 0)),
            "queued_requests": int(queued.get(model, 0)),
            "output_tokens_total": int(out_tokens.get(model, 0)),
            "latency_sum": lat_sum.get(model, 0.0),
            "latency_count": int(lat_cnt.get(model, 0)),
            "ttft_sum": ttft_sum.get(model, 0.0),
            "ttft_count": int(ttft_cnt.get(model, 0)),
            "migrations_total": int(migrations.get(model, 0)),
            "rejections_total": int(rejections.get(model, 0)),
            "ready": bool(ready.get(model, 0)),
            "latency_buckets": lat_buckets.get(model, {}),
            "ttft_buckets": ttft_buckets.get(model, {}),
        }
    return {
        "per_model": per_model,
        "requests_total": int(sum(requests.values())),
        "active_requests": int(sum(active.values())),
        "queued_requests": int(sum(queued.values())),
        "output_tokens_total": int(sum(out_tokens.values())),
    }


# ── Worker instances ────────────────────────────────────────────────────────
_GEN = {"dynamo_endpoint": "generate"}


def instance_summary(raw: Dict[str, list]) -> dict:
    """Per-instance view of a worker's system-port /metrics.

    `dynamo_component_*` counts Dynamo request handling; `vllm:*` is the engine
    itself (queue depth, KV usage) and is what the old per-worker table showed.
    """
    running = _sum(raw, "vllm:num_requests_running")
    waiting = _sum(raw, "vllm:num_requests_waiting")
    kv = _sum(raw, "vllm:gpu_cache_usage_perc")
    if not raw.get("vllm:gpu_cache_usage_perc"):
        kv = _sum(raw, "dynamo_component_gpu_cache_usage_percent") / 100.0
    return {
        "requests_total": int(_sum(raw, "dynamo_component_requests_total", **_GEN)),
        "errors_total": int(_sum(raw, "dynamo_component_errors_total", **_GEN)),
        "inflight": int(_sum(raw, "dynamo_component_inflight_requests", **_GEN)),
        "duration_sum": _sum(raw, "dynamo_component_request_duration_seconds_sum", **_GEN),
        "duration_count": int(_sum(raw, "dynamo_component_request_duration_seconds_count", **_GEN)),
        "running": int(running),
        "waiting": int(waiting),
        "kv_cache_usage_pct": round(kv * 100.0, 1),
        "uptime_s": int(_sum(raw, "dynamo_component_uptime_seconds")),
    }


async def scrape_instance(client: httpx.AsyncClient, url: str, timeout: float = 3.0) -> Optional[dict]:
    try:
        r = await client.get(f"{url}/metrics", timeout=timeout)
        r.raise_for_status()
    except Exception:
        return None
    return instance_summary(parse_prometheus(r.text))


async def instance_health(client: httpx.AsyncClient, url: str, timeout: float = 8.0) -> bool:
    """A dynamo worker is ready only once its engine is loaded and its
    `generate` endpoint is registered — /health reports that directly."""
    try:
        r = await client.get(f"{url}/health", timeout=timeout)
        return r.status_code == 200 and r.json().get("status") == "ready"
    except Exception:
        return False


# ── etcd (v3 gRPC-gateway) ──────────────────────────────────────────────────
def _b64(s: str) -> str:
    return base64.b64encode(s.encode()).decode()


def _range_end(prefix: str) -> str:
    b = bytearray(prefix.encode())
    for i in range(len(b) - 1, -1, -1):
        if b[i] < 0xFF:
            b[i] += 1
            return base64.b64encode(bytes(b[: i + 1])).decode()
    return _b64("\0")


async def etcd_instance_keys(client: httpx.AsyncClient, timeout: float = 3.0) -> List[str]:
    """Keys under `v1/instances/` — one per registered endpoint of every live
    worker. Key shape: v1/instances/<namespace>/<component>/<endpoint>/<id>."""
    prefix = "v1/instances/"
    r = await client.post(
        f"{ETCD_ENDPOINT}/v3/kv/range",
        json={"key": _b64(prefix), "range_end": _range_end(prefix), "keys_only": True},
        timeout=timeout,
    )
    r.raise_for_status()
    return [base64.b64decode(kv["key"]).decode() for kv in (r.json().get("kvs") or [])]


def generate_instances_by_namespace(keys: List[str]) -> Dict[str, List[str]]:
    """{namespace: [instance_id, ...]} for the `generate` endpoint only —
    i.e. the pool that actually serves inference for that model."""
    out: Dict[str, List[str]] = {}
    for k in keys:
        parts = k.split("/")
        # v1 / instances / ns / component / endpoint / id
        if len(parts) < 6 or parts[1] != "instances" or parts[4] != "generate":
            continue
        out.setdefault(parts[2], []).append(parts[5])
    return out
