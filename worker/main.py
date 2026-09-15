from fastapi import Depends, FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
from manager import WorkerManager
import subprocess
import httpx
import asyncio
import os
import logging
import socket

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
manager = WorkerManager()

# This is injected via docker-compose environment variable mapping
CENTRAL_URL = os.environ.get("CENTRAL_URL", "http://central:8080")
WORKER_ID = os.environ.get("WORKER_ID", socket.gethostname())
WORKER_HOST = os.environ.get("WORKER_HOST", socket.gethostbyname(socket.gethostname()))
WORKER_PORT = int(os.environ.get("WORKER_PORT", 8081))
RECONCILE_PERIOD_S = int(os.environ.get("RECONCILE_PERIOD_S", "60"))
# Shared secret for this agent's control API (deploy/stop/logs/images/models).
# The agent listens on the LAN and every route can start or kill GPU workloads,
# so it must not be open. Enforcement is skipped when unset, which keeps a host
# that has not been given the key yet working instead of bricking it.
WORKER_API_KEY = os.environ.get("WORKER_API_KEY", "").strip()


async def require_api_key(request: Request) -> None:
    if not WORKER_API_KEY:
        return
    auth = request.headers.get("authorization", "")
    presented = auth[7:].strip() if auth.lower().startswith("bearer ") else request.headers.get("x-api-key", "")
    if presented != WORKER_API_KEY:
        raise HTTPException(status_code=401, detail="invalid or missing worker api key")

class WorkerDeployRequest(BaseModel):
    deploy_id: str
    replica_id: str
    name: str
    model: str
    served_model_name: Optional[str] = None
    is_embedding: bool = False
    engine: Optional[str] = "dynamo"
    gpus: List[int]
    tp: int = 1
    max_len: Optional[int] = None
    gpu_util: Optional[float] = 0.9
    extra_args: Optional[str] = None
    vllm_image: Optional[str] = None
    # engine == "dynamo" only (see templates/dynamo_node.j2)
    advertise_host: Optional[str] = None      # IP the Dynamo frontend reaches this host at
    etcd_endpoints: Optional[str] = None      # e.g. http://143.248.74.105:2379
    namespace: Optional[str] = "dynamo"
    reasoning_parser: Optional[str] = None    # --dyn-reasoning-parser (gpt_oss, gemma4, qwen3, ...)
    tool_call_parser: Optional[str] = None    # --dyn-tool-call-parser (harmony, gemma4, hermes, ...)
    block_size: Optional[int] = 64            # must match the frontend's --kv-cache-block-size

class PullImageRequest(BaseModel):
    image: str

class DownloadModelRequest(BaseModel):
    model_id: str
    force: bool = False

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(register_loop())
    asyncio.create_task(reconcile_loop())

async def register_loop():
    while True:
        try:
            # get_gpu_status shells out to docker (blocking, seconds) — run it in
            # a thread so the heartbeat loop never stalls the event loop.
            gpus = await asyncio.to_thread(manager.get_gpu_status)
            payload = {
                "worker_id": WORKER_ID,
                "host": WORKER_HOST,
                "port": WORKER_PORT,
                "gpus": gpus,
                "version": manager.get_version(),
            }
            async with httpx.AsyncClient() as client:
                await client.post(f"{CENTRAL_URL}/api/internal/register_node", json=payload, timeout=5.0)
            logger.info(f"Registered with central server at {CENTRAL_URL}")
        except Exception as e:
            logger.error(f"Failed to register with central server: {e}")
            
        await asyncio.sleep(10)

async def reconcile_loop():
    """Bring back engine containers that died and stayed dead — most often a host
    reboot where the NVIDIA driver was not loaded yet when docker started them.
    Cheap (one `docker inspect` per local deployment) and a no-op when healthy."""
    while True:
        try:
            restarted = await asyncio.to_thread(manager.reconcile_local_deployments)
            if restarted:
                logger.warning(f"Reconcile restarted engine containers: {restarted}")
        except Exception as e:
            logger.error(f"reconcile_loop error: {e}")
        await asyncio.sleep(RECONCILE_PERIOD_S)

@app.get("/api/internal/version", dependencies=[Depends(require_api_key)])
async def get_version():
    return manager.get_version()

class SelfUpdateRequest(BaseModel):
    branch: Optional[str] = None

@app.post("/api/internal/self_update", dependencies=[Depends(require_api_key)])
async def self_update(req: SelfUpdateRequest = SelfUpdateRequest()):
    try:
        return await asyncio.to_thread(manager.self_update, req.branch)
    except Exception as e:
        raise HTTPException(status_code=409, detail=str(e))

@app.post("/api/internal/deploy", dependencies=[Depends(require_api_key)])
async def deploy_model(req: WorkerDeployRequest):
    try:
        # deploy_model shells out to docker compose (can take minutes on a cold
        # image). Run in a thread so the event loop — and the register_loop
        # heartbeat — keep running; otherwise central marks this worker offline
        # mid-deploy.
        dep = await asyncio.to_thread(manager.deploy_model, req.dict())
        return dep
    except Exception as e:
        import traceback
        err_str = f"Deploy failed: {e}\n{traceback.format_exc()}"
        logger.error(err_str)
        raise HTTPException(status_code=500, detail=err_str)

@app.post("/api/internal/stop/{deploy_id}", dependencies=[Depends(require_api_key)])
async def stop_deployment(deploy_id: str):
    success = await asyncio.to_thread(manager.stop_deployment, deploy_id)
    if not success:
        raise HTTPException(status_code=404, detail="Deployment not found")
    return {"status": "success"}

@app.post("/api/internal/stop_replica/{deploy_id}/{global_gpu_id}", dependencies=[Depends(require_api_key)])
async def stop_replica(deploy_id: str, global_gpu_id: str):
    success = await asyncio.to_thread(manager.stop_replica, deploy_id, global_gpu_id)
    if not success:
        raise HTTPException(status_code=404, detail="Replica not found")
    return {"status": "success"}

@app.get("/api/internal/logs/{deploy_id}", dependencies=[Depends(require_api_key)])
async def get_deployment_logs(deploy_id: str, container_name: Optional[str] = None):
    # Check if deployment exists on this worker
    deps = manager.load_local_deployments()
    target_nodes = []
    for d in deps:
        if d["id"] == deploy_id or d["id"].startswith(f"{deploy_id}_"):
            if d.get("nodes"):
                target_nodes.extend([n["name"] for n in d["nodes"]])
                
    if container_name:
        target_nodes = [n for n in target_nodes if n == container_name]
                
    if not target_nodes:
        raise HTTPException(status_code=404, detail="Deployment or container not found on this worker.")

    return StreamingResponse(manager.stream_logs(target_nodes), media_type="text/event-stream")

@app.get("/api/internal/images", dependencies=[Depends(require_api_key)])
async def list_images():
    return manager.list_vllm_images()

@app.post("/api/internal/images/pull", dependencies=[Depends(require_api_key)])
async def pull_image(req: PullImageRequest):
    return StreamingResponse(manager.pull_image_stream(req.image), media_type="text/plain")

@app.get("/api/internal/models", dependencies=[Depends(require_api_key)])
async def list_hf_models():
    return manager.list_hf_models()

@app.post("/api/internal/models/download", dependencies=[Depends(require_api_key)])
async def download_model(req: DownloadModelRequest):
    job_id = manager.start_download_job(req.model_id, force=req.force)
    return {"job_id": job_id}

@app.get("/api/internal/models/jobs", dependencies=[Depends(require_api_key)])
async def list_model_jobs():
    return manager.list_download_jobs()

@app.get("/api/internal/models/jobs/{job_id}/logs", dependencies=[Depends(require_api_key)])
async def stream_job_logs(job_id: str, offset: int = 0):
    return StreamingResponse(manager.stream_job_logs(job_id, offset), media_type="text/plain")
