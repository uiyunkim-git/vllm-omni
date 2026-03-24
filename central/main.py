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
    engine: Optional[str] = "vllm"
    gpus: List[str] # Global GPU IDs e.g., ["alpha-worker-1-0", "alpha-worker-1-1"]
    tp: int = 1
    max_len: Optional[int] = None
    gpu_util: Optional[float] = 0.9
    extra_args: Optional[str] = None

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

@app.delete("/api/configs/{name}")
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

@app.get("/api/deployments/{deploy_id}/logs")
async def get_deployment_logs(deploy_id: str, container_name: Optional[str] = None):
    # Retrieve the async generator and return it as a streaming response
    return StreamingResponse(manager.stream_logs(deploy_id, container_name), media_type="text/event-stream")

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

