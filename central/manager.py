import os
import re
import json
import time
import uuid
import subprocess
import httpx
import logging
import asyncio
from typing import List, Dict, Optional

import dynamo
import db

DATA_DIR = "/app/data"
os.makedirs(DATA_DIR, exist_ok=True)

logger = logging.getLogger(__name__)

# Data-plane topology, namespaces, port offsets and metric parsing all live in
# central/dynamo.py so the deploy path and the metrics path cannot disagree.
node_api_port = dynamo.node_api_port

# Deploy settings persisted with a deployment (and echoed back by the API), so a
# deployment can be inspected and recreated without retyping its configuration.
DEPLOY_CONFIG_KEYS = (
    "max_len", "gpu_util", "extra_args", "image", "is_embedding",
    "reasoning_parser", "tool_call_parser", "block_size", "tp",
)

# CI/CD self-update config.
HOST_REPO_DIR = os.environ.get("HOST_REPO_DIR", "")          # host path of the git checkout
HOST_GIT_DIR = os.environ.get("HOST_GIT_DIR", "")            # real git dir if submodule/worktree
WORKER_BRANCH = os.environ.get("WORKER_BRANCH", "main")
WORKER_AUTO_UPDATE = os.environ.get("WORKER_AUTO_UPDATE", "0") == "1"
GIT_IMAGE = os.environ.get("GIT_IMAGE", "alpine/git")

class CentralManager:
    def __init__(self):
        self._dep_lock = asyncio.Lock()
        # worker_id -> version dict reported in the heartbeat.
        self._worker_versions: dict = {}
        # Cached target commit (what workers should converge to) + timestamp.
        self._target_cache: dict = {"commit": None, "ts": 0.0}
        # worker_id -> monotonic ts of the last update we triggered (cooldown).
        self._update_cooldown: dict = {}
        db.init_db()

    def register_worker(self, worker_id: str, host: str, port: int, gpus: list, version: dict = None):
        if version:
            self._worker_versions[worker_id] = {**version, "seen": time.time()}
        conn = db.get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM workers WHERE worker_id = ?", (worker_id,))
        row = cursor.fetchone()
        gpus_json = json.dumps(gpus)
        
        if not row:
            cursor.execute('''
                INSERT INTO workers (worker_id, custom_name, host, port, status, gpus_json)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (worker_id, worker_id, host, port, 'pending', gpus_json))
            logger.info(f"Registered NEW pending worker {worker_id} at {host}:{port}")
        else:
            cursor.execute('''
                UPDATE workers SET host = ?, port = ?, gpus_json = ?, last_seen = CURRENT_TIMESTAMP
                WHERE worker_id = ?
            ''', (host, port, gpus_json, worker_id))
            logger.info(f"Updated worker {worker_id} at {host}:{port}")
            
        conn.commit()
        conn.close()
        # HAProxy should NOT reload on every single heartbeat, that drops active long-running connections!

    def accept_worker(self, worker_id: str, custom_name: str):
        conn = db.get_db()
        cursor = conn.cursor()
        cursor.execute("UPDATE workers SET status = 'active', custom_name = ? WHERE worker_id = ?", (custom_name, worker_id))
        rows_affected = cursor.rowcount
        conn.commit()
        conn.close()
        
        return rows_affected > 0

    def delete_worker(self, worker_id: str):
        conn = db.get_db()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM workers WHERE worker_id = ?", (worker_id,))
        rows_affected = cursor.rowcount
        conn.commit()
        conn.close()
        return rows_affected > 0

    def get_workers(self):
        conn = db.get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM workers")
        rows = cursor.fetchall()
        workers = {}
        for r in rows:
            workers[r["worker_id"]] = {
                "id": r["worker_id"],
                "name": r["custom_name"],
                "host": r["host"],
                "port": r["port"],
                "status": r["status"],
                "gpus": json.loads(r["gpus_json"]) if r["gpus_json"] else []
            }
        conn.close()
        return workers

    def get_all_gpus(self):
        conn = db.get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM workers WHERE status = 'active'")
        rows = cursor.fetchall()
        all_gpus = []
        for r in rows:
            worker_id = r["worker_id"]
            worker_name = r["custom_name"]
            gpus = json.loads(r["gpus_json"]) if r["gpus_json"] else []
            for g in gpus:
                all_gpus.append({
                    "id": f"{worker_id}-{g['id']}",
                    "worker_id": worker_id,
                    "worker_name": worker_name,
                    "local_id": g["id"],
                    "name": g["name"],
                    "utilization": g["utilization"],
                    "memory_used": g["memory_used"],
                    "memory_total": g["memory_total"]
                })
        conn.close()
        return all_gpus

    def load_configs(self):
        conn = db.get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM configs")
        rows = cursor.fetchall()
        configs = []
        for r in rows:
            conf = json.loads(r["config_json"])
            conf["name"] = r["name"]
            configs.append(conf)
        conn.close()
        return configs

    def save_config(self, config_data):
        conn = db.get_db()
        cursor = conn.cursor()
        name = config_data["name"]
        
        # The frontend sends { name: "str", config: { ... } }
        # We want to store ONLY the inner 'config' dict as the config_json.
        actual_config = config_data.get("config", config_data)
        
        # Ensure the inner config dictates its own embedded name as well
        actual_config["name"] = name
        
        config_json = json.dumps(actual_config)
        cursor.execute("INSERT OR REPLACE INTO configs (name, config_json) VALUES (?, ?)", (name, config_json))
        conn.commit()
        conn.close()

    def delete_config(self, name: str):
        conn = db.get_db()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM configs WHERE name = ?", (name,))
        rows_affected = cursor.rowcount
        conn.commit()
        conn.close()
        return rows_affected > 0

    def load_deployments(self):
        conn = db.get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM deployments")
        rows = cursor.fetchall()
        deps = []
        for r in rows:
            # Handle potential missing column for backwards compatibility
            served_name = r["model"]
            if "served_model_name" in r.keys() and r["served_model_name"]:
                served_name = r["served_model_name"]
                
            conf = {}
            if "config_json" in r.keys() and r["config_json"]:
                try:
                    conf = json.loads(r["config_json"])
                except Exception:
                    conf = {}
            deps.append({
                **conf,
                "id": r["id"],
                "name": r["name"],
                "model": r["model"],
                "served_model_name": served_name,
                "engine": r["engine"] if "engine" in r.keys() else dynamo.DEFAULT_ENGINE,
                "deployment_type": r["deployment_type"],
                "status": r["status"],
                "gpus": json.loads(r["gpus_json"]) if r["gpus_json"] else [],
                "nodes": json.loads(r["nodes_json"]) if r["nodes_json"] else [],
                "config": conf,
            })
        conn.close()
        return deps

    def save_deployments(self, deps):
        conn = db.get_db()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM deployments")
        for d in deps:
            conf = d.get("config") or {k: d[k] for k in DEPLOY_CONFIG_KEYS if k in d}
            cursor.execute('''
                INSERT INTO deployments (id, name, model, served_model_name, engine, deployment_type, status, gpus_json, nodes_json, config_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (d["id"], d["name"], d["model"], d.get("served_model_name", d["model"]),
                  d.get("engine", dynamo.DEFAULT_ENGINE), d["deployment_type"], d["status"],
                  json.dumps(d["gpus"]), json.dumps(d.get("nodes", [])), json.dumps(conf)))
        conn.commit()
        conn.close()

    async def deploy_model(self, req: dict):
        deploy_id = str(uuid.uuid4())[:8]
        touched_wids: set = set()
        try:
            return await self._deploy_model_inner(req, deploy_id, touched_wids)
        except Exception:
            # Roll back replicas already started on workers we touched. Without
            # this, a failure on replica N of M leaves replicas 1..N-1 running
            # but unrecorded — invisible to the UI, pinning their GPUs until
            # someone cleans up by hand.
            if touched_wids:
                all_workers = self.get_workers()
                async with httpx.AsyncClient() as client:
                    for wid in touched_wids:
                        w = all_workers.get(wid)
                        if not w:
                            continue
                        try:
                            await client.post(
                                f"http://{w['host']}:{w['port']}/api/internal/stop/{deploy_id}",
                                timeout=60.0,
                            )
                            logger.info(f"Rolled back partial deploy {deploy_id} on worker {wid}")
                        except Exception as e:
                            logger.error(f"Rollback of {deploy_id} on worker {wid} failed: {e}")
            raise

    # ── Dynamo data plane ────────────────────────────────────────────────────
    # engine == "dynamo": the worker runs `python -m dynamo.vllm` (no uvicorn,
    # no TLS), registers itself in etcd and is routed to by the Dynamo frontend.
    # Everything topological (namespace, ports, parsers) comes from dynamo.py.
    def _dynamo_fields(self, req: dict, worker: dict) -> dict:
        if req.get("engine", dynamo.DEFAULT_ENGINE) != "dynamo":
            return {}
        rp, tp = dynamo.infer_parsers(req.get("model", ""))
        served = req.get("served_model_name") or req.get("model", "")
        return {
            "advertise_host": worker["host"],
            "etcd_endpoints": dynamo.ETCD_ENDPOINT,
            "namespace": dynamo.namespace_for(served),
            "reasoning_parser": req.get("reasoning_parser") or rp,
            "tool_call_parser": req.get("tool_call_parser") or tp,
            "block_size": req.get("block_size") or dynamo.KV_BLOCK_SIZE,
            "is_embedding": bool(req.get("is_embedding")),
        }

    async def _deploy_model_inner(self, req: dict, deploy_id: str, touched_wids: set):
        # Group requested GPUs by worker
        worker_assignments = {}
        for global_gpu_id in req["gpus"]:
            wid, gid = global_gpu_id.rsplit("-", 1)
            if wid not in worker_assignments:
                worker_assignments[wid] = []
            worker_assignments[wid].append(int(gid))

        if not worker_assignments:
            raise Exception("No valid GPUs found in assignment")

        # Validation for TP mode
        if req["deployment_type"] == "tp":
            if len(worker_assignments) > 1:
                raise Exception("Tensor Parallelism must run on a single worker node.")
            total_gpus = sum(len(gpus) for gpus in worker_assignments.values())
            if total_gpus not in [1, 2, 4, 8]:
                raise Exception("Tensor Parallelism requires exactly 1, 2, 4, or 8 GPUs.")


        config = {
            "max_len": req.get("max_len"),
            "gpu_util": req.get("gpu_util"),
            "extra_args": req.get("extra_args"),
            "image": req.get("image") or req.get("vllm_image"),
            "is_embedding": bool(req.get("is_embedding")),
            "reasoning_parser": req.get("reasoning_parser"),
            "tool_call_parser": req.get("tool_call_parser"),
            "block_size": req.get("block_size") or dynamo.KV_BLOCK_SIZE,
            "tp": req.get("tp", 1),
        }
        if req.get("engine", dynamo.DEFAULT_ENGINE) == "dynamo" and not config["reasoning_parser"]:
            config["reasoning_parser"], config["tool_call_parser"] = dynamo.infer_parsers(req["model"])

        dep = {
            "id": deploy_id,
            "name": req["name"],
            "deployment_type": req["deployment_type"],
            "model": req["model"],
            "served_model_name": req.get("served_model_name") or req["model"],
            "engine": req.get("engine", dynamo.DEFAULT_ENGINE),
            "gpus": req["gpus"],
            "tp": req["tp"],
            "status": "starting",
            "nodes": [],
            "config": config,
            **config,
        }

        # Send deployment commands to workers.
        # touched_wids tracks workers that received at least one deploy command,
        # so a mid-sequence failure can roll back the replicas already started
        # instead of leaving orphaned containers pinning GPUs.
        touched_wids: set = set()
        all_workers = self.get_workers()
        async with httpx.AsyncClient() as client:
            if req["deployment_type"] == "replicas":
                # For independent replicas, we send a separate deployment command for EACH GPU
                for wid, gpus in worker_assignments.items():
                    if wid not in all_workers:
                        raise Exception(f"Worker {wid} is not registered or offline.")
                    
                    worker = all_workers[wid]
                    worker_url = f"http://{worker['host']}:{worker['port']}/api/internal/deploy"
                    
                    for gid in gpus:
                        worker_req = {
                            "deploy_id": deploy_id,
                            "replica_id": f"{deploy_id}_{wid}_{gid}", # Unique identifier for the worker to avoid collision
                            "name": req["name"],
                            "model": req["model"],
                            "served_model_name": req.get("served_model_name") or req["model"],
                            "is_embedding": bool(req.get("is_embedding")),
                            "engine": req.get("engine", dynamo.DEFAULT_ENGINE),
                            "gpus": [gid], # ONLY send one GPU
                            "tp": 1,
                            "max_len": req.get("max_len"),
                            "gpu_util": req.get("gpu_util"),
                            "extra_args": req.get("extra_args"),
                            "vllm_image": req.get("image") or req.get("vllm_image") or None,
                            **self._dynamo_fields(req, worker),
                        }

                        touched_wids.add(wid)
                        resp = await client.post(worker_url, json=worker_req, timeout=600.0)
                        if resp.status_code != 200:
                            raise Exception(f"Failed to deploy replica on worker {wid} GPU {gid}: {resp.text}")
                        
                        worker_resp = resp.json()
                        for node in worker_resp.get("nodes", []):
                            dep["nodes"].append({
                                "name": node["name"],
                                "host": worker["host"],
                                "port": node["port"],
                                "is_healthy": False
                            })
                            
            elif req["deployment_type"] == "tp":
                # For TP, we send ONE deployment command to the single worker with all selected GPUs
                wid = list(worker_assignments.keys())[0]
                gpus = worker_assignments[wid]
                
                if wid not in all_workers:
                    raise Exception(f"Worker {wid} is not registered or offline.")
                
                worker = all_workers[wid]
                worker_url = f"http://{worker['host']}:{worker['port']}/api/internal/deploy"
                
                worker_req = {
                    "deploy_id": deploy_id,
                    "replica_id": deploy_id, # Base ID
                    "name": req["name"],
                    "model": req["model"],
                    "served_model_name": req.get("served_model_name") or req["model"],
                    "is_embedding": bool(req.get("is_embedding")),
                    "engine": req.get("engine", dynamo.DEFAULT_ENGINE),
                    "gpus": gpus, # Send ALL selected GPUs
                    "tp": len(gpus), # Explicitly set TP to GPU count
                    "max_len": req.get("max_len"),
                    "gpu_util": req.get("gpu_util"),
                    "extra_args": req.get("extra_args"),
                    "vllm_image": req.get("image") or req.get("vllm_image") or None,
                    **self._dynamo_fields(req, worker),
                }

                touched_wids.add(wid)
                resp = await client.post(worker_url, json=worker_req, timeout=600.0)
                if resp.status_code != 200:
                    raise Exception(f"Failed to deploy TP model on worker {wid}: {resp.text}")
                
                worker_resp = resp.json()
                for node in worker_resp.get("nodes", []):
                    dep["nodes"].append({
                        "name": node["name"],
                        "host": worker["host"],
                        "port": node["port"],
                        "is_healthy": False
                    })
        
        # Persist under the lock, against a FRESH read. The old pattern loaded
        # the list BEFORE the (up to 600s) HTTP deploys and saved it after —
        # any stop/deploy that committed in between was silently clobbered by
        # save_deployments' delete-all-then-reinsert.
        async with self._dep_lock:
            deps_now = self.load_deployments()
            deps_now.append(dep)
            self.save_deployments(deps_now)
        return dep

    async def stop_deployment(self, deploy_id: str):
        async with self._dep_lock:
            deps = self.load_deployments()
            dep_index = -1
            for i, d in enumerate(deps):
                if d["id"] == deploy_id:
                    dep_index = i
                    break

            if dep_index == -1:
                return False
            dep = deps[dep_index]

            del deps[dep_index]
            self.save_deployments(deps)

        wids = set()
        for global_gpu_id in dep["gpus"]:
            wid, _ = global_gpu_id.rsplit("-", 1)
            wids.add(wid)

        all_workers = self.get_workers()
        async with httpx.AsyncClient() as client:
            for wid in wids:
                if wid in all_workers:
                    worker = all_workers[wid]
                    worker_url = f"http://{worker['host']}:{worker['port']}/api/internal/stop/{deploy_id}"
                    try:
                        await client.post(worker_url, timeout=60.0)
                    except Exception as e:
                        logger.error(f"Failed to stop deployment {deploy_id} on worker {wid}: {e}")

        return True

    async def stop_replica(self, deploy_id: str, global_gpu_id: str):
        # Hold lock only for the DB read-modify-write to prevent concurrent calls
        # from overwriting each other's changes.
        async with self._dep_lock:
            deps = self.load_deployments()
            dep_index = next((i for i, d in enumerate(deps) if d["id"] == deploy_id), -1)
            if dep_index == -1:
                return False

            dep = deps[dep_index]
            if global_gpu_id not in dep["gpus"]:
                return False

            wid, gpu_idx = global_gpu_id.rsplit("-", 1)
            dep["gpus"].remove(global_gpu_id)
            # Drop the node too. Leaving it behind kept a ghost instance in
            # /api/deployments (and so in the dashboard, health probes and the
            # metrics tables) for the rest of the deployment's life.
            dep["nodes"] = [
                n for n in dep.get("nodes", [])
                if not str(n.get("name", "")).endswith(f"_{wid}_{gpu_idx}")
            ]
            if not dep["gpus"]:
                del deps[dep_index]
            else:
                deps[dep_index] = dep

            self.save_deployments(deps)

        all_workers = self.get_workers()
        if wid in all_workers:
            worker = all_workers[wid]
            worker_url = f"http://{worker['host']}:{worker['port']}/api/internal/stop_replica/{deploy_id}/{global_gpu_id}"
            async with httpx.AsyncClient() as client:
                try:
                    await client.post(worker_url, timeout=60.0)
                except Exception as e:
                    logger.error(f"Failed to stop replica {global_gpu_id} of {deploy_id}: {e}")

        return True

    async def stream_logs(self, deploy_id: str, container_name: Optional[str] = None):
        import asyncio
        import httpx
        
        deps = self.load_deployments()
        dep = next((d for d in deps if d["id"] == deploy_id), None)
        if not dep:
            yield f"data: [Central] Deployment {deploy_id} not found\n\n"
            return
            
        wids = set()
        for global_gpu_id in dep["gpus"]:
            wid, _ = global_gpu_id.rsplit("-", 1)
            wids.add(wid)
            
        if not wids:
            yield "data: [Central] No workers attached to deployment\n\n"
            return
            
        all_workers = self.get_workers()
        queue = asyncio.Queue()
        tasks = []
        
        async def fetch_stream(wid: str):
            if wid not in all_workers:
                await queue.put(f"data: [Central] Worker {wid} is offline/unregistered.\n\n")
                return

            worker = all_workers[wid]
            worker_url = f"http://{worker['host']}:{worker['port']}/api/internal/logs/{deploy_id}"
            if container_name:
                worker_url += f"?container_name={container_name}"

            retry_delay = 2
            while True:
                try:
                    timeout = httpx.Timeout(connect=15.0, read=None, write=10.0, pool=10.0)
                    async with httpx.AsyncClient(timeout=timeout) as client:
                        async with client.stream("GET", worker_url) as response:
                            if response.status_code != 200:
                                err_msg = await response.aread()
                                await queue.put(f"data: [Central-Error] Worker {wid} returned {response.status_code}: {err_msg.decode('utf-8')}\n\n")
                                return
                            retry_delay = 2
                            async for line in response.aiter_lines():
                                if line:
                                    await queue.put(f"{line}\n\n")
                    # Stream ended cleanly (worker closed connection)
                    return
                except asyncio.CancelledError:
                    return
                except Exception as e:
                    await queue.put(f"data: [Central-Error] Connection to worker {wid} lost: {e} — reconnecting in {retry_delay}s...\n\n")
                    await asyncio.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, 30)

        for wid in wids:
            tasks.append(asyncio.create_task(fetch_stream(wid)))
            
        try:
            while True:
                line = await queue.get()
                yield line
        except asyncio.CancelledError:
            # Browser client disconnected
            pass
        finally:
            for task in tasks:
                task.cancel()

    async def _check_node_ready(self, client, host: int, api_port: int, dep: dict) -> bool:
        """Liveness + readiness check.

        /health alone is insufficient for vLLM: the HTTP server returns 200 while
        the engine is still loading model weights, which leads to a thundering-herd
        of failed requests (and CB trips) the moment the router starts routing.
        We additionally require /v1/models to list the served model, which only
        happens after the engine finishes initialization.
        """
        engine = dep.get("engine", dynamo.DEFAULT_ENGINE)

        if engine == "dynamo":
            return await dynamo.instance_health(client, f"http://{host}:{api_port}")

        try:
            resp = await client.get(f"https://{host}:{api_port}/health", timeout=8.0)
            if resp.status_code != 200:
                return False
        except Exception:
            return False

        served_name = dep.get("served_model_name") or dep.get("model", "")
        try:
            resp = await client.get(f"https://{host}:{api_port}/v1/models", timeout=8.0)
            if resp.status_code != 200:
                return False
            data = resp.json().get("data", [])
            if not data:
                return False
            if served_name and not any(m.get("id") == served_name for m in data):
                return False
        except Exception:
            return False

        return True

    async def _node_serves_other_model(self, client, host: int, api_port: int, expected: str) -> Optional[str]:
        """Return the served model id if the endpoint is alive but exposes a DIFFERENT
        model than this deployment expects. Returns None if it serves the expected model
        or if we can't tell."""
        try:
            resp = await client.get(f"https://{host}:{api_port}/v1/models", timeout=5.0)
            if resp.status_code != 200:
                return None
            ids = [m.get("id") for m in resp.json().get("data", [])]
            if not ids:
                return None
            if expected and expected not in ids:
                return ids[0]
        except Exception:
            return None
        return None

    async def run_health_checks(self):
        # ── Phase 1: probe every unique endpoint CONCURRENTLY (read-only). ──
        # The old sequential loop cost up to 13-21s per dead node; with ~40
        # nodes a single cycle could take 8-14 minutes, during which the UI
        # showed stale health (dead workers stayed green for many minutes).
        deps = self.load_deployments()

        # Deduplicate probe targets: same host:port:served_model only checked
        # once. Including served_model in the key matters when an endpoint has
        # been re-used by a new deployment serving a different model.
        probe_targets: dict = {}  # (host, api_port, served) -> dep (for engine/name)
        for dep in deps:
            if dep["status"] not in ["running", "starting"]:
                continue
            served_name = dep.get("served_model_name") or dep.get("model", "")
            for node in dep.get("nodes", []):
                key = (node["host"], node_api_port(dep, node), served_name)
                probe_targets.setdefault(key, dep)

        results: dict = {}  # key -> "other_model" | True | False
        sem = asyncio.Semaphore(16)

        async with httpx.AsyncClient(verify=False) as client:
            async def probe(key, dep):
                host, api_port, served_name = key
                async with sem:
                    other = None
                    if not dynamo.is_dynamo(dep):
                        other = await self._node_serves_other_model(client, host, api_port, served_name)
                    if other is not None:
                        logger.info(
                            f"Node {host}:{api_port} now serves {other!r}, not {served_name!r} "
                            f"— marking stale for cleanup"
                        )
                        results[key] = "other_model"
                        return
                    results[key] = await self._check_node_ready(client, host, api_port, dep)

            await asyncio.gather(*(probe(k, d) for k, d in probe_targets.items()))

        # ── Phase 2: apply results to a FRESH copy under the lock. ──────────
        # The old code mutated the list it loaded before the (multi-minute)
        # probe pass and wrote it back wholesale — clobbering any deploy/stop
        # that committed in between.
        async with self._dep_lock:
            deps = self.load_deployments()
            changed = False
            for dep in deps:
                if dep["status"] not in ["running", "starting"]:
                    continue
                served_name = dep.get("served_model_name") or dep.get("model", "")
                all_healthy = True
                stale_idx: list = []
                for node_idx, node in enumerate(dep.get("nodes", [])):
                    key = (node["host"], node_api_port(dep, node), served_name)
                    res = results.get(key)
                    if res == "other_model" and dynamo.is_dynamo(dep):
                        res = None  # never computed for dynamo nodes
                    if res is None:
                        # Node appeared after the probe pass — next cycle covers it.
                        if not node.get("is_healthy"):
                            all_healthy = False
                        continue
                    if res == "other_model":
                        stale_idx.append(node_idx)
                        continue

                    is_healthy = bool(res)
                    # Require 2 consecutive failures before marking unhealthy
                    # to avoid flapping from transient relay timeouts
                    if not is_healthy and node.get("is_healthy"):
                        fail_count = node.get("_fail_count", 0) + 1
                        node["_fail_count"] = fail_count
                        if fail_count >= 2:
                            node["is_healthy"] = False
                            node["_fail_count"] = 0
                            changed = True
                    elif is_healthy:
                        if not node.get("is_healthy"):
                            node["is_healthy"] = True
                            changed = True
                        node["_fail_count"] = 0

                    if not node.get("is_healthy"):
                        all_healthy = False

                # Prune stale nodes in reverse so deletion doesn't shift positions.
                for node_idx in reversed(stale_idx):
                    bad = dep["nodes"].pop(node_idx)
                    changed = True
                    logger.info(f"Pruned stale node {bad.get('name')} from deployment {dep['id']}")

                if all_healthy and dep["status"] == "starting" and dep.get("nodes"):
                    dep["status"] = "running"
                    changed = True

            if changed:
                self.save_deployments(deps)

    # ── CI/CD: worker version tracking + rolling self-update ────────────────
    def _target_version_blocking(self) -> Optional[str]:
        """Latest commit on WORKER_BRANCH as seen from central's checkout — the
        commit workers should converge to. Computed with a throwaway git
        container (central has the docker socket but not git/the repo)."""
        if not HOST_REPO_DIR:
            return None

        def _git(*args, timeout=20):
            base = ["docker", "run", "--rm", "-v", f"{HOST_REPO_DIR}:/repo"]
            env = []
            if HOST_GIT_DIR:
                # Submodule/worktree: mount the real git dir and point git at it.
                base += ["-v", f"{HOST_GIT_DIR}:/gitdir",
                         "-e", "GIT_DIR=/gitdir", "-e", "GIT_WORK_TREE=/repo"]
            base += [GIT_IMAGE, "-c", "safe.directory=*"]
            if not HOST_GIT_DIR:
                base += ["-C", "/repo"]
            return subprocess.run(base + list(args), capture_output=True, text=True, timeout=timeout)

        try:
            # fetch quietly so the target reflects the freshest origin state; if
            # the network/creds don't allow it, rev-parse still returns local HEAD.
            _git("fetch", "--quiet", "origin", WORKER_BRANCH, timeout=30)
            out = _git("rev-parse", f"origin/{WORKER_BRANCH}")
            sha = out.stdout.strip()
            if not sha:
                logger.warning(f"target-version rev-parse empty: {out.stderr.strip()}")
            return sha or None
        except Exception as e:
            logger.warning(f"target-version lookup failed: {e}")
            return None

    async def get_target_version(self, max_age: float = 60.0) -> Optional[str]:
        if self._target_cache["commit"] and (time.time() - self._target_cache["ts"]) < max_age:
            return self._target_cache["commit"]
        sha = await asyncio.to_thread(self._target_version_blocking)
        if sha:
            self._target_cache = {"commit": sha, "ts": time.time()}
        return self._target_cache["commit"]

    def worker_version_status(self, target: Optional[str]) -> list:
        """Per-worker version + drift, for the UI."""
        out = []
        for wid, w in self.get_workers().items():
            ver = self._worker_versions.get(wid, {})
            commit = ver.get("commit", "unknown")
            short = commit[:12] if commit and commit not in ("unknown", "unmanaged") else commit
            up_to_date = bool(target) and commit == target
            out.append({
                "worker_id": wid,
                "name": w.get("name"),
                "status": w.get("status"),
                "commit": short,
                "updating": ver.get("updating", False),
                "updated_at": ver.get("updated_at", ""),
                "up_to_date": up_to_date,
                "drift": bool(target) and commit not in (target, "unknown") ,
            })
        return out

    async def update_worker(self, worker_id: str, branch: Optional[str] = None) -> dict:
        """Ask one worker to self-update. Skips workers with an in-flight deploy
        (worker returns 409) and records a cooldown so the auto-loop doesn't
        hammer a worker that's mid-restart."""
        workers = self.get_workers()
        if worker_id not in workers:
            raise Exception(f"Worker {worker_id} not found")
        w = workers[worker_id]
        url = f"http://{w['host']}:{w['port']}/api/internal/self_update"
        self._update_cooldown[worker_id] = time.time()
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json={"branch": branch or WORKER_BRANCH}, timeout=30.0)
        if resp.status_code != 200:
            raise Exception(f"Worker {worker_id} update failed ({resp.status_code}): {resp.text}")
        return resp.json()

    async def auto_update_loop(self):
        """Opt-in (WORKER_AUTO_UPDATE=1). Rolls workers whose reported commit
        differs from the target, ONE at a time, with a per-worker cooldown. A
        self-update only recreates the agent (models keep serving), so serving
        workers are included. Never touches 'unmanaged' workers automatically —
        those were hand-deployed and adopting them is an explicit operator action."""
        if not WORKER_AUTO_UPDATE:
            logger.info("Worker auto-update disabled (set WORKER_AUTO_UPDATE=1 to enable).")
            return
        logger.info("Worker auto-update loop started.")
        while True:
            await asyncio.sleep(60)
            try:
                target = await self.get_target_version()
                if not target:
                    continue
                # A self-update only recreates the worker AGENT — the vLLM model
                # containers keep serving — so serving workers are updated too.
                for wid, w in self.get_workers().items():
                    if w.get("status") != "active":
                        continue
                    ver = self._worker_versions.get(wid, {})
                    commit = ver.get("commit")
                    if not commit or commit in ("unmanaged", "unknown") or commit == target:
                        continue
                    if ver.get("updating"):
                        continue
                    last = self._update_cooldown.get(wid, 0)
                    if time.time() - last < 300:
                        continue
                    logger.info(f"Auto-update: worker {wid} {commit[:12]} → {target[:12]}")
                    try:
                        await self.update_worker(wid)
                    except Exception as e:
                        logger.warning(f"Auto-update of {wid} failed: {e}")
                    break  # one at a time — re-evaluate next tick
            except Exception as e:
                logger.error(f"auto_update_loop error: {e}")
