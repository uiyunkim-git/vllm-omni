from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
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

class WorkerDeployRequest(BaseModel):
    deploy_id: str
    replica_id: str
    name: str
    model: str
    engine: Optional[str] = "vllm"
    gpus: List[int]
    tp: int = 1
    max_len: Optional[int] = None
    gpu_util: Optional[float] = 0.9
    extra_args: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(register_loop())

async def register_loop():
    while True:
        try:
            gpus = manager.get_gpu_status()
            payload = {
                "worker_id": WORKER_ID,
                "host": WORKER_HOST,
                "port": WORKER_PORT,
                "gpus": gpus
            }
            async with httpx.AsyncClient() as client:
                await client.post(f"{CENTRAL_URL}/api/internal/register_node", json=payload, timeout=5.0)
            logger.info(f"Registered with central server at {CENTRAL_URL}")
        except Exception as e:
            logger.error(f"Failed to register with central server: {e}")
            
        await asyncio.sleep(10)

@app.post("/api/internal/deploy")
async def deploy_model(req: WorkerDeployRequest):
    try:
        dep = manager.deploy_model(req.dict())
        return dep
    except Exception as e:
        import traceback
        err_str = f"Deploy failed: {e}\n{traceback.format_exc()}"
        logger.error(err_str)
        raise HTTPException(status_code=500, detail=err_str)

@app.post("/api/internal/stop/{deploy_id}")
async def stop_deployment(deploy_id: str):
    success = manager.stop_deployment(deploy_id)
    if not success:
        raise HTTPException(status_code=404, detail="Deployment not found")
    return {"status": "success"}

@app.get("/api/internal/logs/{deploy_id}")
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
