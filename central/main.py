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

@app.get("/api/proxy_stats")
async def get_proxy_stats():
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get("http://vllm_omni_proxy:4000/stats", timeout=2.0)
            return resp.json()
    except Exception:
        return {}

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

