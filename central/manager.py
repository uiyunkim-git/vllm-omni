import os
import json
import uuid
import subprocess
import httpx
import logging
import asyncio
from jinja2 import Environment, FileSystemLoader
from typing import List, Dict, Optional
import db

DATA_DIR = "/app/data"
os.makedirs(DATA_DIR, exist_ok=True)

logger = logging.getLogger(__name__)

class CentralManager:
    def __init__(self):
        self.env = Environment(loader=FileSystemLoader('/app/templates'))
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
        
        if rows_affected > 0:
            self.reload_go_proxy(self.load_deployments())
            return True
        return False

    def delete_worker(self, worker_id: str):
        conn = db.get_db()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM workers WHERE worker_id = ?", (worker_id,))
        rows_affected = cursor.rowcount
        conn.commit()
        conn.close()
        if rows_affected > 0:
            self.reload_go_proxy(self.load_deployments())
            return True
        return False

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
                            "engine": req.get("engine", "vllm"),
                            "gpus": [gid], # ONLY send one GPU
                            "tp": 1,
                            "max_len": req.get("max_len"),
                            "gpu_util": req.get("gpu_util"),
                            "extra_args": req.get("extra_args")
                        }
                        
                        resp = await client.post(worker_url, json=worker_req, timeout=60.0)
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
                    "engine": req.get("engine", "vllm"),
                    "gpus": gpus, # Send ALL selected GPUs
                    "tp": len(gpus), # Explicitly set TP to GPU count
                    "max_len": req.get("max_len"),
                    "gpu_util": req.get("gpu_util"),
                    "extra_args": req.get("extra_args")
                }
                
                resp = await client.post(worker_url, json=worker_req, timeout=60.0)
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
        
        dep["status"] = "running"
        existing_deps.append(dep)
        self.save_deployments(existing_deps)
        
        self.reload_go_proxy(existing_deps)
        return dep

    async def stop_deployment(self, deploy_id: str):
        deps = self.load_deployments()
        dep_index = -1
        for i, d in enumerate(deps):
            if d["id"] == deploy_id:
                dep_index = i
                break
                
        if dep_index == -1: return False
        dep = deps[dep_index]
        
        # We need to tell the workers to stop it.
        # Figure out which workers are involved
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

        del deps[dep_index]
        self.save_deployments(deps)
        self.reload_go_proxy(deps)
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
            
            try:
                # We use a long timeout since this is a persistent SSE connection
                timeout = httpx.Timeout(connect=10.0, read=None, write=10.0, pool=10.0)
                async with httpx.AsyncClient(timeout=timeout) as client:
                    async with client.stream("GET", worker_url) as response:
                        if response.status_code != 200:
                            err_msg = await response.aread()
                            await queue.put(f"data: [Central-Error] Worker {wid} returned {response.status_code}: {err_msg.decode('utf-8')}\n\n")
                            return
                            
                        async for line in response.aiter_lines():
                            if line:
                                # The worker already yields "data: ..." format
                                await queue.put(f"{line}\n\n")
            except asyncio.CancelledError:
                pass
            except Exception as e:
                await queue.put(f"data: [Central-Error] Connection to worker {wid} lost: {e}\n\n")

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

    def reload_go_proxy(self, deps):
        import os
        import json
        import subprocess

        models_map = {}
        for d in deps:
            if d.get("status") == "running" and d.get("deployment_type") != "embeddings": # Only active running
                model_name = d.get("served_model_name") or d.get("model", "default_model")
                if model_name not in models_map:
                    models_map[model_name] = []
                for node in d.get("nodes", []):
                    if node.get("is_healthy"):
                        models_map[model_name].append({
                            "host": node["host"],
                            "port": node["port"] + 40000
                        })
        
        config_data = {
            "auth_token": os.environ.get("VLLM_API_KEY", "bislaprom3#"),
            "models": models_map
        }
        
        # Ensure config directory exists
        os.makedirs("/app/go_proxy_config", exist_ok=True)
        
        with open("/app/go_proxy_config/config.json", "w") as f:
            json.dump(config_data, f, indent=4)
            
        # Hot reload without dropping connections
        subprocess.run(["curl", "-X", "POST", "http://vllm_omni_proxy:4000/reload"], check=False)

    async def run_health_checks(self):
        deps = self.load_deployments()
        changed = False

        async with httpx.AsyncClient(verify=False) as client:
            for dep in deps:
                if dep["status"] not in ["running", "starting"]:
                    continue

                all_healthy = True
                for node in dep.get("nodes", []):
                    host = node["host"]
                    # Calculate the API port. Usually port + 40000
                    api_port = node["port"] + 40000
                    health_path = "/health" if dep.get("engine", "vllm") == "vllm" else "/"
                    health_url = f"https://{host}:{api_port}{health_path}"

                    try:
                        resp = await client.get(health_url, timeout=2.0)
                        is_healthy = resp.status_code == 200
                    except Exception as e:
                        # logger.error(f"Health check failed for {health_url}: {e}")
                        is_healthy = False
                    
                    if node.get("is_healthy") != is_healthy:
                        node["is_healthy"] = is_healthy
                        changed = True

                    if not is_healthy:
                        all_healthy = False

                # If all nodes are healthy, mark dep as running securely
                if all_healthy and dep["status"] == "starting":
                    dep["status"] = "running"
                    changed = True

        if changed:
            self.save_deployments(deps)
            self.reload_go_proxy(deps)
