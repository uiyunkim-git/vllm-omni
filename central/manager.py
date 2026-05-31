import os
import json
import uuid
import subprocess
import httpx
import logging
import asyncio
from typing import List, Dict, Optional
import db

DATA_DIR = "/app/data"
os.makedirs(DATA_DIR, exist_ok=True)

logger = logging.getLogger(__name__)

P2C_ROUTER_URL = "http://143.248.74.105:11434"

class CentralManager:
    def __init__(self):
        self._dep_lock = asyncio.Lock()
        db.init_db()

    def register_worker(self, worker_id: str, host: str, port: int, gpus: list):
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
                
            deps.append({
                "id": r["id"],
                "name": r["name"],
                "model": r["model"],
                "served_model_name": served_name,
                "engine": r["engine"] if "engine" in r.keys() else "vllm",
                "deployment_type": r["deployment_type"],
                "status": r["status"],
                "gpus": json.loads(r["gpus_json"]) if r["gpus_json"] else [],
                "nodes": json.loads(r["nodes_json"]) if r["nodes_json"] else []
            })
        conn.close()
        return deps

    def save_deployments(self, deps):
        conn = db.get_db()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM deployments")
        for d in deps:
            cursor.execute('''
                INSERT INTO deployments (id, name, model, served_model_name, engine, deployment_type, status, gpus_json, nodes_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (d["id"], d["name"], d["model"], d.get("served_model_name", d["model"]), d.get("engine", "vllm"), d["deployment_type"], d["status"], json.dumps(d["gpus"]), json.dumps(d.get("nodes", []))))
        conn.commit()
        conn.close()

    async def deploy_model(self, req: dict):
        deploy_id = str(uuid.uuid4())[:8]
        
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


        dep = {
            "id": deploy_id,
            "name": req["name"],
            "deployment_type": req["deployment_type"],
            "model": req["model"],
            "served_model_name": req.get("served_model_name") or req["model"],
            "is_embedding": False,
            "engine": req.get("engine", "vllm"),
            "gpus": req["gpus"],
            "tp": req["tp"],
            "status": "starting",
            "nodes": []
        }

        existing_deps = self.load_deployments()

        # Send deployment commands to workers
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
                            "is_embedding": False,
                            "engine": req.get("engine", "vllm"),
                            "gpus": [gid], # ONLY send one GPU
                            "tp": 1,
                            "max_len": req.get("max_len"),
                            "gpu_util": req.get("gpu_util"),
                            "extra_args": req.get("extra_args"),
                            "vllm_image": req.get("vllm_image") or None
                        }
                        
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
                    "is_embedding": False,
                    "engine": req.get("engine", "vllm"),
                    "gpus": gpus, # Send ALL selected GPUs
                    "tp": len(gpus), # Explicitly set TP to GPU count
                    "max_len": req.get("max_len"),
                    "gpu_util": req.get("gpu_util"),
                    "extra_args": req.get("extra_args"),
                    "vllm_image": req.get("vllm_image") or None
                }
                
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
        
        existing_deps.append(dep)
        self.save_deployments(existing_deps)
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

        for node in dep.get("nodes", []):
            await self._p2c_deregister(node["host"], node["port"])
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

            dep["gpus"].remove(global_gpu_id)
            if not dep["gpus"]:
                del deps[dep_index]
            else:
                deps[dep_index] = dep

            self.save_deployments(deps)

        # Find the node for this GPU before making the worker call
        wid, gpu_idx = global_gpu_id.rsplit("-", 1)
        removed_node = next(
            (n for n in dep.get("nodes", []) if n.get("name", "").endswith(f"_{gpu_idx}")),
            None,
        )

        all_workers = self.get_workers()
        if wid in all_workers:
            worker = all_workers[wid]
            worker_url = f"http://{worker['host']}:{worker['port']}/api/internal/stop_replica/{deploy_id}/{global_gpu_id}"
            async with httpx.AsyncClient() as client:
                try:
                    await client.post(worker_url, timeout=60.0)
                except Exception as e:
                    logger.error(f"Failed to stop replica {global_gpu_id} of {deploy_id}: {e}")

        if removed_node:
            await self._p2c_deregister(removed_node["host"], removed_node["port"])
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

    async def _p2c_register(self, host: str, port: int):
        url = f"https://{host}:{port + 40000}"
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(f"{P2C_ROUTER_URL}/add_worker", params={"url": url}, timeout=30.0)
            if resp.status_code == 200:
                logger.info(f"P2C: registered {url}")
            elif "already exists" in resp.text:
                logger.debug(f"P2C: {url} already registered")
            else:
                logger.warning(f"P2C: unexpected response for {url}: {resp.status_code} {resp.text}")
        except Exception as e:
            logger.warning(f"P2C: failed to register {url}: {e}")

    async def _p2c_deregister(self, host: str, port: int):
        url = f"https://{host}:{port + 40000}"
        try:
            async with httpx.AsyncClient() as client:
                await client.post(f"{P2C_ROUTER_URL}/remove_worker", params={"url": url}, timeout=10.0)
            logger.info(f"P2C: deregistered {url}")
        except Exception as e:
            logger.warning(f"P2C: failed to deregister {url}: {e}")

    async def sync_p2c_workers(self):
        """Register all currently healthy deployment nodes with the P2C router."""
        deps = self.load_deployments()
        for dep in deps:
            for node in dep.get("nodes", []):
                if node.get("is_healthy"):
                    await self._p2c_register(node["host"], node["port"])

    async def _check_node_ready(self, client, host: int, api_port: int, dep: dict) -> bool:
        """Liveness + readiness check.

        /health alone is insufficient for vLLM: the HTTP server returns 200 while
        the engine is still loading model weights, which leads to a thundering-herd
        of failed requests (and CB trips) the moment the router starts routing.
        We additionally require /v1/models to list the served model, which only
        happens after the engine finishes initialization.
        """
        engine = dep.get("engine", "vllm")
        health_path = "/health" if engine == "vllm" else "/"

        try:
            resp = await client.get(f"https://{host}:{api_port}{health_path}", timeout=8.0)
            if resp.status_code != 200:
                return False
        except Exception:
            return False

        if engine != "vllm":
            return True

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

    async def run_health_checks(self):
        deps = self.load_deployments()
        changed = False

        # Deduplicate health check targets: same host:port only checked once
        checked_endpoints: dict = {}  # (host, port) -> bool

        async with httpx.AsyncClient(verify=False) as client:
            for dep in deps:
                if dep["status"] not in ["running", "starting"]:
                    continue

                all_healthy = True
                for node in dep.get("nodes", []):
                    host = node["host"]
                    api_port = node["port"] + 40000
                    endpoint_key = (host, api_port)

                    if endpoint_key not in checked_endpoints:
                        checked_endpoints[endpoint_key] = await self._check_node_ready(
                            client, host, api_port, dep
                        )

                    is_healthy = checked_endpoints[endpoint_key]

                    # Require 2 consecutive failures before marking unhealthy
                    # to avoid flapping from transient relay timeouts
                    if not is_healthy and node.get("is_healthy"):
                        fail_count = node.get("_fail_count", 0) + 1
                        node["_fail_count"] = fail_count
                        if fail_count >= 2:
                            node["is_healthy"] = False
                            node["_fail_count"] = 0
                            changed = True
                            await self._p2c_deregister(host, node["port"])
                    elif is_healthy:
                        if not node.get("is_healthy"):
                            node["is_healthy"] = True
                            changed = True
                            await self._p2c_register(host, node["port"])
                        node["_fail_count"] = 0

                    if not node.get("is_healthy"):
                        all_healthy = False

                if all_healthy and dep["status"] == "starting":
                    dep["status"] = "running"
                    changed = True

        if changed:
            self.save_deployments(deps)
