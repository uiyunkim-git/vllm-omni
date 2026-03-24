import os
import json
import subprocess
import shlex
import shutil
from pydantic import BaseModel
from typing import List, Optional
from jinja2 import Environment, FileSystemLoader
import logging

logger = logging.getLogger(__name__)

DATA_DIR = "/app/data"
HOST_DATA_DIR = os.environ.get("HOST_DATA_DIR", "/home/uiyunkim/bisl-uiyunkim/applications/pons/vllm/vllm-omni/worker/data")
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
        
        existing_deps = self.load_local_deployments()
        used_ports = set()
        for d in existing_deps:
            used_ports.update(d.get("ports", []))
            
        try:
            import subprocess
            import re
            docker_ports_out = subprocess.check_output(["docker", "ps", "--format", "{{.Ports}}"], text=True)
            for match in re.finditer(r":(\d+)->", docker_ports_out):
                used_ports.add(int(match.group(1)))
        except Exception as e:
            logging.error(f"Failed to check docker ports: {e}")
            
        ports = []
        current_port = 8001
        
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

        compose_content = template.render(
            node_name=node_name,
            model_name=req["model"],
            served_model_name=req.get("served_model_name"),
            huggingface_token=token,
            task="embed" if req.get("is_embedding", False) else "generate",
            tensor_parallel_size=req["tp"],
            gpu_ids=[str(g) for g in req["gpus"]],
            port=current_port,
            max_model_len=req.get("max_len"),
            gpu_memory_util=req.get("gpu_util"),
            replica_id=replica_id,
            host_cache_dir="/home/uiyunkim/.cache/huggingface",
            host_data_dir="/home/uiyunkim/.ollama",
            host_cert_path=os.path.join(HOST_DATA_DIR, f"run_{replica_id}", "vllm.crt"),
            host_key_path=os.path.join(HOST_DATA_DIR, f"run_{replica_id}", "vllm.key"),
            extra_args=shlex.split(req.get("extra_args") or "")
        )
        
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
        with open(compose_path, "w") as f:
            f.write(compose_content)
            
        subprocess.run(["docker", "compose", "-p", f"vllm_{replica_id}", "-f", compose_path, "up", "-d"], check=True)
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
