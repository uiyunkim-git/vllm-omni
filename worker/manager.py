import os
import json
import subprocess
import shlex
import shutil
import asyncio
import uuid
from datetime import datetime
from pydantic import BaseModel
from typing import List, Optional
from jinja2 import Environment, FileSystemLoader
import logging

logger = logging.getLogger(__name__)

DATA_DIR = "/app/data"

class DownloadJob:
    def __init__(self, job_id: str, model_id: str):
        self.job_id = job_id
        self.model_id = model_id
        self.status = "running"  # "running" | "done" | "failed"
        self.lines: list = []
        self.started_at = datetime.now().isoformat()
        self.finished_at = None
        self._cond = asyncio.Condition()

_download_jobs: dict = {}  # job_id -> DownloadJob
HOST_DATA_DIR = os.environ.get("HOST_DATA_DIR", "/home/uiyunkim/bisl-uiyunkim/applications/pons/vllm/vllm-omni/worker/data")
HOST_PORT_OFFSET = 40000
# Internal ports start at 21001 so host ports (internal + 40000) land at 61001+,
# above the OS ephemeral port range (32768-60999) to avoid bind conflicts.
_PORT_START = 21001
os.makedirs(DATA_DIR, exist_ok=True)

class WorkerManager:
    def __init__(self):
        self.env = Environment(loader=FileSystemLoader('/app/templates'))

    def get_gpu_status(self):
        try:
            # We use the docker socket to spin up a tiny container to query the host's GPUs since the worker container doesn't have nvidia-smi
            result = subprocess.run(
                ['docker', 'run', '--rm', '--gpus', 'all', '--entrypoint', 'nvidia-smi', 'vllm/vllm-openai:latest', '--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits'],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                logger.error(f"Docker nvidia-smi failed: {result.stderr}")
            else:
                logger.info(f"Docker nvidia-smi STDOUT: {result.stdout}")
            gpus = []
            for line in result.stdout.strip().split('\n'):
                if not line: continue
                parts = [x.strip() for x in line.split(',')]
                gpus.append({
                    "id": int(parts[0]),
                    "name": parts[1],
                    "memory_total": int(parts[2]),
                    "memory_used": int(parts[3]),
                    "memory_free": int(parts[4]),
                    "utilization": int(parts[5])
                })
            return gpus
        except Exception as e:
            logger.error(f"Exception in get_gpu_status: {e}")
            return []

    def load_local_deployments(self):
        deps_file = os.path.join(DATA_DIR, "local_deployments.json")
        try:
            with open(deps_file, "r") as f:
                return json.load(f)
        except:
            return []

    def save_local_deployments(self, deps):
        deps_file = os.path.join(DATA_DIR, "local_deployments.json")
        with open(deps_file, "w") as f:
            json.dump(deps, f, indent=2)

    def deploy_model(self, req: dict):
        deploy_id = req["deploy_id"]
        replica_id = req["replica_id"]

        # Check that the requested image exists locally before doing anything else
        requested_image = req.get("vllm_image") or "vllm/vllm-openai:latest"
        check = subprocess.run(
            ["docker", "image", "inspect", requested_image],
            capture_output=True
        )
        if check.returncode != 0:
            raise Exception(
                f"Image '{requested_image}' not found on this worker. "
                f"Go to Endpoints → Images and pull it first."
            )

        existing_deps = self.load_local_deployments()
        used_ports = set()
        for d in existing_deps:
            used_ports.update(d.get("ports", []))
            
        try:
            import re
            docker_ports_out = subprocess.check_output(["docker", "ps", "--format", "{{.Ports}}"], text=True)
            for match in re.finditer(r":(\d+)->", docker_ports_out):
                host_port = int(match.group(1))
                used_ports.add(host_port)
                if host_port >= HOST_PORT_OFFSET:
                    used_ports.add(host_port - HOST_PORT_OFFSET)
        except Exception as e:
            logging.error(f"Failed to check docker ports: {e}")
            
        ports = []
        current_port = _PORT_START
        
        nodes = []

        while current_port in used_ports:
            current_port += 1
        used_ports.add(current_port)
        ports.append(current_port)
        
        engine = req.get("engine", "vllm")
        if engine == "ollama":
            template = self.env.get_template("ollama_node.j2")
            node_name = f"ollama_{replica_id}"
        else:
            template = self.env.get_template("vllm_node.j2")
            node_name = f"vllm_{replica_id}"
        
        token = os.environ.get("HUGGING_FACE_HUB_TOKEN", "")
        if not token:
            # Fallback to reading the token file directly
            token_path = os.path.expanduser('~/.cache/huggingface/token')
            if os.path.exists(token_path):
                with open(token_path, 'r') as tf:
                    token = tf.read().strip()

        dep_dir = os.path.join(DATA_DIR, f"run_{replica_id}")
        
        # Aggressively delete the container if it exists before deleting the folder
        subprocess.run(["docker", "rm", "-f", node_name], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Aggressively delete the directory if it exists to clean up any fake 
        # empty directories created by Docker-in-Docker volume mount bugs
        if os.path.exists(dep_dir):
            shutil.rmtree(dep_dir, ignore_errors=True)
            
        os.makedirs(dep_dir, exist_ok=True)
                
        # Generate self-signed SSL certificate for DPI Evasion
        try:
            subprocess.run([
                "openssl", "req", "-x509", "-newkey", "rsa:4096",
                "-keyout", os.path.join(dep_dir, "vllm.key"),
                "-out", os.path.join(dep_dir, "vllm.crt"),
                "-days", "365", "-nodes", "-subj", "/CN=vllm_secure"
            ], check=True, capture_output=True)
            logger.info("Generated DPI-Evasion SSL Certificate successfully.")
        except Exception as e:
            logger.error(f"Failed to generate SSL certs: {e}")

        compose_path = os.path.join(dep_dir, "docker-compose.yml")

        def _write_compose(port):
            content = template.render(
                node_name=node_name,
                model_name=req["model"],
                served_model_name=req.get("served_model_name"),
                huggingface_token=token,
                tensor_parallel_size=req["tp"],
                gpu_ids=[str(g) for g in req["gpus"]],
                port=port,
                max_model_len=req.get("max_len"),
                gpu_memory_util=req.get("gpu_util"),
                replica_id=replica_id,
                host_cache_dir="/home/uiyunkim/.cache/huggingface",
                host_data_dir="/home/uiyunkim/.ollama",
                host_cert_path=os.path.join(HOST_DATA_DIR, f"run_{replica_id}", "vllm.crt"),
                host_key_path=os.path.join(HOST_DATA_DIR, f"run_{replica_id}", "vllm.key"),
                extra_args=shlex.split(req.get("extra_args") or ""),
                vllm_image=req.get("vllm_image") or "vllm/vllm-openai:latest"
            )
            with open(compose_path, "w") as f:
                f.write(content)

        _write_compose(current_port)

        result = None
        for _attempt in range(20):
            compose_cmd = ["docker", "compose", "-p", f"vllm_{replica_id}", "-f", compose_path, "up", "-d"]
            result = subprocess.run(compose_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                break
            error_text = result.stdout + result.stderr
            if "address already in use" in error_text:
                # Port was taken despite pre-check — advance to next free port and retry
                subprocess.run(["docker", "compose", "-p", f"vllm_{replica_id}", "-f", compose_path, "down", "-t", "0", "-v"],
                               check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                subprocess.run(["docker", "rm", "-f", node_name], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                current_port += 1
                while current_port in used_ports:
                    current_port += 1
                used_ports.add(current_port)
                logger.warning(f"Port conflict, retrying with port {current_port} (host {current_port + HOST_PORT_OFFSET})")
                _write_compose(current_port)
            else:
                break  # Non-port error — fail immediately

        if result.returncode != 0:
            cleanup_cmd = ["docker", "compose", "-p", f"vllm_{replica_id}", "-f", compose_path, "down", "-t", "0", "-v"]
            subprocess.run(cleanup_cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(["docker", "rm", "-f", node_name], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if engine == "ollama":
                subprocess.run(["docker", "rm", "-f", f"{node_name}_proxy"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            shutil.rmtree(dep_dir, ignore_errors=True)

            details = "\n".join(part for part in [result.stdout.strip(), result.stderr.strip()] if part)
            raise RuntimeError(f"docker compose failed for {node_name} on host port {current_port + HOST_PORT_OFFSET}: {details or 'no output'}")

        nodes.append({"name": node_name, "port": current_port})
            
        dep = {
            "id": deploy_id,
            "replica_id": replica_id,
            "ports": ports,
            "nodes": nodes
        }
        existing_deps.append(dep)
        self.save_local_deployments(existing_deps)
        
        return dep

    def stop_deployment(self, deploy_id: str):
        deps = self.load_local_deployments()
        # Find all replicas associated with this deploy_id
        matching_indices = []
        for i, d in enumerate(deps):
            if d["id"] == deploy_id:
                matching_indices.append(i)
                
        if not matching_indices: return False
        
        # Stop each replica
        for i in matching_indices:
            dep = deps[i]
            replica_id = dep.get("replica_id", deploy_id) # Fallback to deploy_id for old configs
            dep_dir = os.path.join(DATA_DIR, f"run_{replica_id}")
            compose_path = os.path.join(dep_dir, "docker-compose.yml")
            if os.path.exists(compose_path):
                subprocess.run(["docker", "compose", "-p", f"vllm_{replica_id}", "-f", compose_path, "down", "-t", "0", "-v"], check=False)
            
            # Clean up the directory regardless
            shutil.rmtree(dep_dir, ignore_errors=True)
                
        # Remove them from state (in reverse to avoid index shifting issues)
        for i in reversed(matching_indices):
            del deps[i]
            
        self.save_local_deployments(deps)
        return True

    def stop_replica(self, deploy_id: str, global_gpu_id: str):
        # replica_id format: "{deploy_id}_{wid}_{gid}"
        wid, gid = global_gpu_id.rsplit("-", 1)
        replica_id = f"{deploy_id}_{wid}_{gid}"

        deps = self.load_local_deployments()
        matching_indices = [i for i, d in enumerate(deps) if d.get("replica_id") == replica_id]
        if not matching_indices:
            return False

        for i in matching_indices:
            dep_dir = os.path.join(DATA_DIR, f"run_{replica_id}")
            compose_path = os.path.join(dep_dir, "docker-compose.yml")
            if os.path.exists(compose_path):
                subprocess.run(["docker", "compose", "-p", f"vllm_{replica_id}", "-f", compose_path, "down", "-t", "0", "-v"], check=False)
            shutil.rmtree(dep_dir, ignore_errors=True)

        for i in reversed(matching_indices):
            del deps[i]
        self.save_local_deployments(deps)
        return True

    def get_logs(self, deploy_id: str):
        import re
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        
        deps = self.load_local_deployments()
        target_nodes = []
        for d in deps:
            # Match exact deploy_id (TP mode) OR prefixes like deploy_id_0 (Replica mode)
            if d["id"] == deploy_id or d["id"].startswith(f"{deploy_id}_"):
                if d.get("nodes"):
                    target_nodes.extend([n["name"] for n in d["nodes"]])
                
        if not target_nodes:
            return "Deployment not found or has no nodes on this worker."
            
        all_lines = []
        for node in target_nodes:
            try:
                result = subprocess.run(
                    ['docker', 'logs', '--timestamps', '--tail', '1500', node],
                    capture_output=True, text=True
                )
                
                ts = ""
                for line in (result.stdout + result.stderr).splitlines():
                    if not line.strip():
                        continue
                        
                    parts = line.split(" ", 1)
                    if len(parts) == 2 and (parts[0].endswith("Z") or "T" in parts[0]):
                        ts = parts[0]
                        msg = parts[1]
                    else:
                        msg = line
                        
                    clean_msg = ansi_escape.sub('', msg)
                    all_lines.append((ts, f"[{node}] {clean_msg}"))
            except Exception as e:
                logger.error(f"Failed to fetch logs for {node}: {e}")
                
        # Sort chronologically by timestamp
        all_lines.sort(key=lambda x: x[0])
        
        logs_output = ""
        for ts, msg in all_lines:
            logs_output += f"{msg}\n"
            
        return logs_output

    async def stream_logs(self, target_nodes: list[str]):
        import asyncio
        import re
        
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        queue = asyncio.Queue()
        tasks = []
        processes = []
        
        async def read_stream(node: str):
            try:
                # -t is required to get timestamps for sorting/clean output
                proc = await asyncio.create_subprocess_exec(
                    'docker', 'logs', '-f', '-t', '--tail', '200', node,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT
                )
                processes.append(proc)
                
                # Read chunks to properly stream \r (carriage returns) for tqdm
                buffer = ""
                while True:
                    chunk = await proc.stdout.read(1024)
                    if not chunk:
                        break
                        
                    text = chunk.decode('utf-8', errors='replace')
                    buffer += text
                    
                    # Split by either \n or \r to dispatch lines immediately
                    while True:
                        if '\n' in buffer and ('\r' not in buffer or buffer.find('\n') < buffer.find('\r')):
                            line, buffer = buffer.split('\n', 1)
                            line = line.strip()
                        elif '\r' in buffer:
                            line, buffer = buffer.split('\r', 1)
                            line = line.strip()
                        else:
                            break
                            
                        if not line or "GET /health" in line:
                            continue
                            
                        # Extract timestamp if present to keep format consistent
                        parts = line.split(" ", 1)
                        if len(parts) == 2 and (parts[0].endswith("Z") or "T" in parts[0]):
                            msg = parts[1]
                        else:
                            msg = line
                            
                        clean_msg = ansi_escape.sub('', msg)
                        formatted_line = f"[{node}] {clean_msg}"
                        
                        await queue.put(formatted_line)
                        
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"Error reading stream for {node}: {e}")
            finally:
                if 'proc' in locals() and proc.returncode is None:
                    try:
                        proc.terminate()
                    except Exception:
                        pass

        # Start a reader task for each node
        for node in target_nodes:
            tasks.append(asyncio.create_task(read_stream(node)))
            
        try:
            while True:
                # We can use a timeout to send keep-alive pings if needed, 
                # but standard SSE often doesn't strictly require it if the proxy handles timeouts.
                # Let's just wait for the next log line.
                line = await queue.get()
                yield f"data: {line}\n\n"
        except asyncio.CancelledError:
            # Client disconnected
            pass
        except Exception as e:
            logger.error(f"Stream logs generator error: {e}")
            yield f"data: [System Error] Log stream disconnected: {e}\n\n"
        finally:
            # Cleanup
            for task in tasks:
                task.cancel()
            for proc in processes:
                if proc.returncode is None:
                    try:
                        proc.terminate()
                    except:
                        pass

    def list_hf_models(self):
        hf_hub_dir = os.path.join(
            os.environ.get("HOST_HF_CACHE_DIR", "/home/uiyunkim/.cache/huggingface"),
            "hub"
        )
        if not os.path.exists(hf_hub_dir):
            return []
        models = []
        try:
            for entry in os.scandir(hf_hub_dir):
                if not entry.is_dir() or not entry.name.startswith('models--'):
                    continue
                parts = entry.name[len('models--'):].split('--', 1)
                repo_id = '/'.join(parts) if len(parts) == 2 else parts[0]
                try:
                    result = subprocess.run(['du', '-sh', entry.path], capture_output=True, text=True, timeout=10)
                    size = result.stdout.split('\t')[0] if result.returncode == 0 else '?'
                except Exception:
                    size = '?'
                models.append({'repo_id': repo_id, 'size': size})
        except Exception as e:
            logger.error(f"Failed to list HF models: {e}")
        return sorted(models, key=lambda x: x['repo_id'])

    def start_download_job(self, model_id: str) -> str:
        job_id = uuid.uuid4().hex[:8]
        job = DownloadJob(job_id, model_id)
        _download_jobs[job_id] = job
        asyncio.create_task(self._run_download_job(job))
        return job_id

    async def _run_download_job(self, job: DownloadJob):
        import re
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

        async def emit(text: str):
            async with job._cond:
                job.lines.append(text)
                job._cond.notify_all()

        host_hf_cache = os.environ.get("HOST_HF_CACHE_DIR", "/home/uiyunkim/.cache/huggingface")
        token = os.environ.get("HUGGING_FACE_HUB_TOKEN", "")
        if not token:
            token_path = os.path.join(host_hf_cache, "token")
            if os.path.exists(token_path):
                with open(token_path, 'r') as f:
                    token = f.read().strip()

        download_py = """\
import os, sys
from huggingface_hub import HfApi, hf_hub_download

model_id = os.environ['MODEL_ID']
token = os.environ.get('HF_TOKEN') or None

print(f'Fetching file list for {model_id}...', flush=True)
api = HfApi(token=token)
try:
    files = list(api.list_repo_files(model_id))
except Exception as e:
    print(f'[✗] Failed to list files: {e}', flush=True)
    sys.exit(1)

print(f'{len(files)} files', flush=True)
for i, fname in enumerate(files, 1):
    print(f'[{i}/{len(files)}] {fname}', flush=True)
    try:
        hf_hub_download(model_id, filename=fname, token=token)
    except Exception as e:
        print(f'  [warn] {e}', flush=True)

print(f'\\n[✓] Done: {model_id}', flush=True)
"""

        try:
            proc = await asyncio.create_subprocess_exec(
                'docker', 'run', '--rm',
                '--entrypoint', 'python3',
                '-e', f'HF_TOKEN={token}',
                '-e', f'MODEL_ID={job.model_id}',
                '-v', f'{host_hf_cache}:/root/.cache/huggingface',
                'vllm/vllm-openai:latest',
                '-u', '-c', download_py,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT
            )
            async for line in proc.stdout:
                text = ansi_escape.sub('', line.decode('utf-8', errors='replace'))
                await emit(text)
            await proc.wait()
            if proc.returncode == 0:
                await emit(f"\n[✓] Successfully downloaded {job.model_id}\n")
                success = True
            else:
                await emit(f"\n[✗] Failed to download {job.model_id} (exit code {proc.returncode})\n")
                success = False
        except Exception as e:
            await emit(f"\n[✗] Exception: {e}\n")
            success = False

        async with job._cond:
            job.status = "done" if success else "failed"
            job.finished_at = datetime.now().isoformat()
            job._cond.notify_all()

    async def stream_job_logs(self, job_id: str, offset: int = 0):
        job = _download_jobs.get(job_id)
        if not job:
            yield f"[error] Job {job_id} not found\n"
            return

        sent = offset
        while True:
            async with job._cond:
                await job._cond.wait_for(
                    lambda: len(job.lines) > sent or job.status != "running"
                )
                batch = job.lines[sent:]
                sent += len(batch)
                is_done = job.status != "running" and sent >= len(job.lines)

            for line in batch:
                yield line

            if is_done:
                break

    def list_download_jobs(self) -> list:
        return [
            {
                "job_id": j.job_id,
                "model_id": j.model_id,
                "status": j.status,
                "started_at": j.started_at,
                "finished_at": j.finished_at,
                "line_count": len(j.lines),
            }
            for j in _download_jobs.values()
        ]

    def list_vllm_images(self):
        result = subprocess.run(
            ['docker', 'images', '--format', '{{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}'],
            capture_output=True, text=True
        )
        images = []
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue
            parts = line.split('\t')
            name = parts[0] if parts else ''
            size = parts[1] if len(parts) > 1 else ''
            created = parts[2] if len(parts) > 2 else ''
            if name.startswith('vllm/'):
                images.append({'name': name, 'size': size, 'created': created})
        return images

    async def pull_image_stream(self, image: str):
        proc = await asyncio.create_subprocess_exec(
            'docker', 'pull', image,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT
        )
        async for line in proc.stdout:
            yield line.decode('utf-8', errors='replace')
        await proc.wait()
        if proc.returncode == 0:
            yield f"\n[✓] Successfully pulled {image}\n"
        else:
            yield f"\n[✗] Failed to pull {image} (exit code {proc.returncode})\n"
