import os
import json
import subprocess
import shlex
import shutil
import asyncio
import threading
import uuid
from datetime import datetime
from pydantic import BaseModel
from typing import List, Optional
from jinja2 import Environment, FileSystemLoader
import logging

logger = logging.getLogger(__name__)

DATA_DIR = "/app/data"

class DownloadJob:
    def __init__(self, job_id: str, model_id: str, force: bool = False):
        self.job_id = job_id
        self.model_id = model_id
        self.force = force
        self.status = "running"  # "running" | "done" | "failed"
        self.lines: list = []
        self.started_at = datetime.now().isoformat()
        self.finished_at = None
        self._cond = asyncio.Condition()

_download_jobs: dict = {}  # job_id -> DownloadJob
HOST_DATA_DIR = os.environ.get("HOST_DATA_DIR", "/home/uiyunkim/bisl-uiyunkim/applications/pons/vllm/vllm-omni/worker/data")
# Absolute path of the git checkout ON THE HOST. The updater mounts this so it
# can `git pull` + `docker compose up -d --build worker`. Derived from
# HOST_DATA_DIR (…/worker/data → repo root) when not set explicitly.
HOST_REPO_DIR = os.environ.get(
    "HOST_REPO_DIR",
    os.path.dirname(os.path.dirname(HOST_DATA_DIR.rstrip("/"))),
)
WORKER_BRANCH = os.environ.get("WORKER_BRANCH", "main")
# When the checkout is a git submodule or linked worktree, its real git dir
# lives OUTSIDE the repo tree (e.g. superproject/.git/modules/…). Set this to
# that host path so the updater can mount it; leave empty for a plain clone.
HOST_GIT_DIR = os.environ.get("HOST_GIT_DIR", "")
# Written by the updater after a successful pull; read back to report version.
VERSION_FILE = os.path.join(DATA_DIR, ".worker_version")
HOST_PORT_OFFSET = 40000

# Dynamo data plane (engine == "dynamo"): worker image and the etcd the frontend uses
# for discovery. Central normally passes both explicitly in the deploy request.
DEFAULT_ENGINE = os.environ.get("DEFAULT_ENGINE", "dynamo")
DYNAMO_IMAGE = os.environ.get("DYNAMO_IMAGE", "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.2")
DYNAMO_ETCD_ENDPOINTS = os.environ.get("DYNAMO_ETCD_ENDPOINTS", "http://143.248.74.105:2379")
# Dynamo parses DYN_SYSTEM_PORT as i16 (max 32767), so the health/metrics port cannot use
# the +40000 slot the vLLM API uses. Keep every dynamo port derived from the allocated
# base `port` (21001..): system=+10000 (31xxx), response-stream=+42000, kv-events=+44000.
DYNAMO_SYSTEM_PORT_OFFSET = 10000   # 31xxx  health/metrics (DYN_SYSTEM_PORT, i16)
DYNAMO_RPC_PORT_OFFSET = 12000      # 33xxx  TCP request-plane listener (DYN_TCP_RPC_PORT; else OS-assigned!)
DYNAMO_RESP_PORT_OFFSET = 42000     # 63xxx  TCP response-stream server (DYN_TCP_RESPONSE_STREAM_PORT)
DYNAMO_KV_PORT_OFFSET = 44000       # 65xxx  ZMQ KV events
# Internal ports start at 21001 so host ports (internal + 40000) land at 61001+,
# above the OS ephemeral port range (32768-60999) to avoid bind conflicts.
_PORT_START = 21001
os.makedirs(DATA_DIR, exist_ok=True)

class WorkerManager:
    def __init__(self):
        self.env = Environment(loader=FileSystemLoader('/app/templates'))
        # Serializes deploy/stop state mutations. Deploys now run in worker
        # threads (asyncio.to_thread), so two concurrent deploys would race the
        # read-allocate-write of local_deployments.json and both pick the same
        # port — the second save silently dropping the first replica's record.
        self._state_lock = threading.Lock()
        # Host firewall rules for dynamo instances are inserted at deploy time
        # and do NOT survive a host reboot, so re-assert them for whatever this
        # worker is already running.
        try:
            self.reapply_host_ports()
        except Exception as e:
            logger.warning(f"startup firewall re-apply skipped: {e}")

    # ── Self-healing ───────────────────────────────────────────────────────
    # Docker's `unless-stopped` gives up after its own retry budget. After a host
    # reboot the NVIDIA driver is often not loaded yet when docker starts the
    # engine containers, so every one of them dies with
    #   nvidia-container-cli: initialization error: nvml error: driver not loaded
    # and STAYS dead — central still believes the deployment exists and nobody
    # brings it back (this took the whole gateway down on 2026-09-15). Reconcile
    # what is actually running against what this worker was told to run.
    def reconcile_local_deployments(self) -> list:
        """Re-`compose up` any locally recorded deployment whose container is not
        running. Returns the replica ids it restarted."""
        restarted = []
        for dep in self.load_local_deployments():
            replica_id = dep.get("replica_id") or dep.get("id")
            nodes = dep.get("nodes") or []
            if not replica_id or not nodes:
                continue
            name = nodes[0].get("name", "")
            try:
                out = subprocess.run(
                    ["docker", "inspect", "-f", "{{.State.Running}}", name],
                    capture_output=True, text=True, timeout=20,
                )
                running = out.returncode == 0 and out.stdout.strip() == "true"
            except Exception:
                continue  # docker unreachable — try again next tick
            if running:
                continue
            compose_path = os.path.join(DATA_DIR, f"run_{replica_id}", "docker-compose.yml")
            if not os.path.exists(compose_path):
                logger.warning(f"reconcile: {name} is down but {compose_path} is gone; leaving it to central")
                continue
            logger.warning(f"reconcile: {name} is not running — bringing it back up")
            r = subprocess.run(
                ["docker", "compose", "-p", f"vllm_{replica_id}", "-f", compose_path, "up", "-d"],
                capture_output=True, text=True, timeout=600,
            )
            if r.returncode == 0:
                restarted.append(replica_id)
            else:
                logger.error(f"reconcile: failed to restart {name}: {(r.stderr or r.stdout).strip()[:300]}")
        return restarted

    def reapply_host_ports(self) -> None:
        ports = []
        for dep in self.load_local_deployments():
            if not (dep.get("nodes") and str(dep["nodes"][0].get("name", "")).startswith("dynamo_")):
                continue
            for p in dep.get("ports", []):
                ports += [p + DYNAMO_SYSTEM_PORT_OFFSET, p + DYNAMO_RPC_PORT_OFFSET,
                          p + DYNAMO_RESP_PORT_OFFSET, p + DYNAMO_KV_PORT_OFFSET]
        if ports:
            self._ensure_host_ports_open(ports)

    def get_gpu_status(self):
        try:
            # We use the docker socket to spin up a tiny container to query the host's GPUs since the worker container doesn't have nvidia-smi
            # timeout guards against a wedged docker daemon — without it the
            # heartbeat loop would block forever and central would mark this
            # worker dead until a manual restart.
            result = subprocess.run(
                ['docker', 'run', '--rm', '--gpus', 'all', '--entrypoint', 'nvidia-smi', 'vllm/vllm-openai:latest', '--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=60
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

    # ── Self-update (CI/CD) ────────────────────────────────────────────────
    def get_version(self) -> dict:
        """Version this worker is running, for the heartbeat. `commit` is the
        git SHA the updater last pulled; `unmanaged` means this worker was
        deployed manually and has never self-updated (so central can't know
        exactly which commit it runs)."""
        try:
            with open(VERSION_FILE) as f:
                v = json.load(f)
            return {
                "commit": v.get("commit", "unmanaged"),
                "subtree": v.get("subtree", ""),
                "updated_at": v.get("updated_at", ""),
                "updating": os.path.exists(VERSION_FILE + ".lock"),
            }
        except Exception:
            return {"commit": "unmanaged", "subtree": "", "updated_at": "", "updating": os.path.exists(VERSION_FILE + ".lock")}

    def self_update(self, branch: Optional[str] = None) -> dict:
        """Launch a DETACHED updater container that pulls the latest repo and
        recreates this worker. Detached so it survives our own restart:
        `docker run -d` via the socket makes an independent sibling container on
        the host, so it can `compose up -d --build worker` even as we go down.

        Guards: refuses while a deploy/stop is in progress (state lock held) and
        while another updater is already running (fixed container name)."""
        branch = branch or WORKER_BRANCH

        # Don't update mid-deploy. Non-blocking probe of the state lock.
        if not self._state_lock.acquire(blocking=False):
            raise RuntimeError("A deploy/stop is in progress; update deferred.")
        try:
            # Discover our own image so the updater runs the exact same one that
            # is already present on this host (no dependency on a public image
            # having both git and docker+compose). Docker sets $HOSTNAME to the
            # container's short id, which `docker inspect` resolves.
            container_hint = os.environ.get("HOSTNAME", "")
            image = None
            try:
                image = subprocess.check_output(
                    ["docker", "inspect", "--format", "{{.Image}}", container_hint],
                    text=True, timeout=15,
                ).strip()
            except Exception:
                pass
            if not image:
                # Fallback: the image compose builds for this repo.
                image = os.environ.get("WORKER_IMAGE", "vllm-omni-worker")

            updater_name = "vllm_omni_worker_updater"
            # Clear any dead updater from a previous run.
            subprocess.run(["docker", "rm", "-f", updater_name],
                           check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Mount the repo (and, for a submodule/worktree, the real git dir) at
            # the SAME absolute path inside the updater as on the host. This is
            # essential for docker-out-of-docker compose: the compose CLI resolves
            # relative volumes (./worker/...) and the build context against the
            # project directory, then hands those ABSOLUTE paths to the host
            # daemon — so the path must be valid both inside the updater (for the
            # CLI's own reads) and on the host (for the bind mounts). Mounting at
            # /repo broke this: the host has no /repo, so `up --build` failed.
            rd = shlex.quote(HOST_REPO_DIR)
            gd = shlex.quote(HOST_GIT_DIR) if HOST_GIT_DIR else ""
            lock = shlex.quote(os.path.join(HOST_REPO_DIR, "worker/data/.worker_version.lock"))
            vfile = shlex.quote(os.path.join(HOST_REPO_DIR, "worker/data/.worker_version"))
            git_env = f"export GIT_DIR={gd} GIT_WORK_TREE={rd}" if HOST_GIT_DIR else f"cd {rd}"
            gd_safe = f"git config --global --add safe.directory {gd}" if HOST_GIT_DIR else ""
            branch_q = shlex.quote(branch)
            script = f"""set -e
# Always drop the lock on exit so a FAILED update never leaves the worker stuck
# reporting updating=true (the version file is only written on success).
trap 'rm -f {lock}' EXIT
{git_env}
git config --global --add safe.directory {rd}
{gd_safe}
touch {lock}
cd {rd}
export PWD={rd}
git fetch --prune origin {branch_q}
git checkout -f {branch_q}
git reset --hard origin/{branch_q}
NEWSHA=$(git rev-parse HEAD)
SUBTREE=$(git rev-parse HEAD:worker || echo "")
TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
docker compose --project-directory {rd} -f {rd}/docker-compose.worker.yml up -d --build worker
printf '{{"commit":"%s","subtree":"%s","updated_at":"%s"}}\\n' "$NEWSHA" "$SUBTREE" "$TS" > {vfile}
"""
            cmd = [
                "docker", "run", "-d", "--rm", "--name", updater_name,
                "-v", "/var/run/docker.sock:/var/run/docker.sock",
                "-v", f"{HOST_REPO_DIR}:{HOST_REPO_DIR}",
            ]
            if HOST_GIT_DIR:
                cmd += ["-v", f"{HOST_GIT_DIR}:{HOST_GIT_DIR}"]
            cmd += ["--entrypoint", "sh", image, "-c", script]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                # Make sure we don't leave a stale lock behind on launch failure.
                try:
                    os.remove(VERSION_FILE + ".lock")
                except Exception:
                    pass
                raise RuntimeError(f"Failed to launch updater: {result.stderr.strip() or result.stdout.strip()}")
            logger.info(f"Launched worker updater (branch={branch}, image={image})")
            return {"status": "updating", "branch": branch, "updater": result.stdout.strip()[:12]}
        finally:
            self._state_lock.release()

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

    def _ensure_host_ports_open(self, ports: list, comment: str = "vllm-omni-dynamo") -> None:
        """Make a host-network worker's ports reachable from other hosts.

        vLLM containers were published with `-p host:container`, and Docker
        itself inserts the firewall rules that let that traffic in (FORWARD
        chain). A Dynamo worker listens directly on the host network, so
        nothing does that for it — on hosts with a default-DROP INPUT policy
        (ufw) its ports time out even though the process is listening. Do what
        Docker does: insert an INPUT ACCEPT for each port, idempotently. The
        HOST's own iptables binary is run via nsenter so it lands in the same
        ruleset (nft or legacy) the host already uses.
        """
        rules = " ; ".join(
            f"iptables -C INPUT -p tcp --dport {p} -j ACCEPT -m comment --comment {comment} 2>/dev/null "
            f"|| iptables -I INPUT 1 -p tcp --dport {p} -j ACCEPT -m comment --comment {comment}"
            for p in ports
        )
        cmd = [
            "docker", "run", "--rm", "--privileged", "--pid=host", "--net=host",
            "alpine:3.20", "nsenter", "-t", "1", "-m", "-u", "-n", "-i", "--", "sh", "-c", rules,
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if r.returncode != 0:
                logger.warning(f"host firewall open for {ports} failed: {(r.stderr or r.stdout).strip()[:300]}")
            else:
                logger.info(f"host firewall: INPUT ACCEPT ensured for tcp {ports}")
        except Exception as e:
            logger.warning(f"host firewall open for {ports} skipped: {e}")

    def _default_advertise_host(self) -> str:
        """IP the Dynamo frontend should dial for this host. Central passes it
        explicitly (the address it already reaches this worker at); fall back to
        the same value the heartbeat advertises."""
        import socket
        return os.environ.get("WORKER_HOST") or socket.gethostbyname(socket.gethostname())

    def deploy_model(self, req: dict):
        # Serialize deploys/stops: they read-modify-write local_deployments.json
        # and allocate ports from a snapshot of it.
        with self._state_lock:
            return self._deploy_model_locked(req)

    def _deploy_model_locked(self, req: dict):
        deploy_id = req["deploy_id"]
        replica_id = req["replica_id"]

        engine = req.get("engine", DEFAULT_ENGINE)
        if engine not in ("dynamo", "vllm"):
            raise Exception(f"Unsupported engine {engine!r} (use 'dynamo', or 'vllm' for a legacy direct endpoint)")
        # Check that the requested image exists locally before doing anything else
        default_image = DYNAMO_IMAGE if engine == "dynamo" else "vllm/vllm-openai:latest"
        requested_image = req.get("vllm_image") or default_image
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
        
        template = self.env.get_template(f"{engine}_node.j2")
        node_name = f"{engine}_{replica_id}"
        
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
                
        # Self-signed cert for the legacy vLLM endpoint only — a dynamo worker
        # exposes no HTTPS server (the frontend owns ingress).
        if engine == "vllm":
            try:
                subprocess.run([
                    "openssl", "req", "-x509", "-newkey", "rsa:4096",
                    "-keyout", os.path.join(dep_dir, "vllm.key"),
                    "-out", os.path.join(dep_dir, "vllm.crt"),
                    "-days", "365", "-nodes", "-subj", "/CN=vllm_secure",
                ], check=True, capture_output=True)
                logger.info("Generated self-signed certificate for the legacy vLLM endpoint.")
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
                host_cert_path=os.path.join(HOST_DATA_DIR, f"run_{replica_id}", "vllm.crt"),
                host_key_path=os.path.join(HOST_DATA_DIR, f"run_{replica_id}", "vllm.key"),
                extra_args=shlex.split(req.get("extra_args") or ""),
                vllm_image=req.get("vllm_image") or "vllm/vllm-openai:latest",
                # ── dynamo engine only ──────────────────────────────────────
                # Host networking; three ports per instance, all derived from the
                # one allocated `port` so the existing allocator/conflict logic
                # keeps working (see DYNAMO_*_PORT_OFFSET).
                image=requested_image,
                system_port=port + DYNAMO_SYSTEM_PORT_OFFSET,
                rpc_port=port + DYNAMO_RPC_PORT_OFFSET,
                resp_port=port + DYNAMO_RESP_PORT_OFFSET,
                kv_port=port + DYNAMO_KV_PORT_OFFSET,
                advertise_host=req.get("advertise_host") or self._default_advertise_host(),
                etcd_endpoints=req.get("etcd_endpoints") or DYNAMO_ETCD_ENDPOINTS,
                namespace=req.get("namespace") or "dynamo",
                reasoning_parser=req.get("reasoning_parser"),
                tool_call_parser=req.get("tool_call_parser"),
                block_size=req.get("block_size") or 64,
                is_embedding=bool(req.get("is_embedding")),
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
            shutil.rmtree(dep_dir, ignore_errors=True)

            details = "\n".join(part for part in [result.stdout.strip(), result.stderr.strip()] if part)
            raise RuntimeError(f"docker compose failed for {node_name} on host port {current_port + HOST_PORT_OFFSET}: {details or 'no output'}")

        nodes.append({"name": node_name, "port": current_port})

        if engine == "dynamo":
            # Same effect as Docker's published-port rules for the vLLM path.
            self._ensure_host_ports_open([
                current_port + DYNAMO_SYSTEM_PORT_OFFSET,
                current_port + DYNAMO_RPC_PORT_OFFSET,
                current_port + DYNAMO_RESP_PORT_OFFSET,
                current_port + DYNAMO_KV_PORT_OFFSET,
            ])

        # current_port may have advanced past the originally allocated value via
        # the address-in-use retry loop above; persist the port actually bound,
        # otherwise future allocations reserve the wrong port.
        ports = [current_port]

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
        with self._state_lock:
            return self._stop_deployment_locked(deploy_id)

    def _stop_deployment_locked(self, deploy_id: str):
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
        with self._state_lock:
            return self._stop_replica_locked(deploy_id, global_gpu_id)

    def _stop_replica_locked(self, deploy_id: str, global_gpu_id: str):
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

    def start_download_job(self, model_id: str, force: bool = False) -> str:
        job_id = uuid.uuid4().hex[:8]
        job = DownloadJob(job_id, model_id, force=force)
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
from huggingface_hub.constants import HF_HUB_CACHE

model_id = os.environ['MODEL_ID']
token = os.environ.get('HF_TOKEN') or None
force = os.environ.get('FORCE_DOWNLOAD') == '1'

def local_commit():
    repo_dir = os.path.join(HF_HUB_CACHE, 'models--' + model_id.replace('/', '--'))
    ref = os.path.join(repo_dir, 'refs', 'main')
    if os.path.exists(ref):
        with open(ref) as f:
            return f.read().strip()
    return None

api = HfApi(token=token)
print(f'Checking {model_id}...', flush=True)
try:
    remote_sha = api.model_info(model_id).sha
except Exception as e:
    print(f'[✗] Failed to fetch model info: {e}', flush=True)
    sys.exit(1)

before = local_commit()
if force:
    print(f'Force redownload requested. Remote revision: {remote_sha[:12]}', flush=True)
elif before == remote_sha:
    print(f'[✓] Already up to date (revision {remote_sha[:12]}). Nothing to download.', flush=True)
    sys.exit(0)
elif before:
    print(f'Update available: {before[:12]} -> {remote_sha[:12]}', flush=True)
else:
    print(f'Not cached yet. Downloading revision {remote_sha[:12]}...', flush=True)

try:
    files = list(api.list_repo_files(model_id))
except Exception as e:
    print(f'[✗] Failed to list files: {e}', flush=True)
    sys.exit(1)

print(f'{len(files)} files', flush=True)
for i, fname in enumerate(files, 1):
    print(f'[{i}/{len(files)}] {fname}', flush=True)
    try:
        hf_hub_download(model_id, filename=fname, token=token, force_download=force)
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
                '-e', f'FORCE_DOWNLOAD={"1" if job.force else "0"}',
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
            # Engine images only: the Dynamo runtime (data plane) and vLLM
            # (legacy direct endpoints). Everything else on the host is noise.
            if name.startswith('vllm/') or 'ai-dynamo/' in name:
                images.append({
                    'name': name, 'size': size, 'created': created,
                    'engine': 'dynamo' if 'ai-dynamo/' in name else 'vllm',
                })
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
