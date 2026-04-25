"""FastAPI entry point for VLLMController."""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from manager import Manager

ROOT = Path(__file__).parent
manager = Manager(data_dir=ROOT / "data")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await manager.start()
    try:
        yield
    finally:
        await manager.stop()


app = FastAPI(title="VLLMController", lifespan=lifespan)


# ---------- aggregate state ----------
@app.get("/api/state")
async def get_state():
    return await manager.get_state()


@app.get("/api/stats")
async def get_stats():
    return await manager.get_stats()


# ---------- settings ----------
@app.get("/api/settings")
async def get_settings():
    return manager.settings


@app.post("/api/settings")
async def update_settings(req: Request):
    return await manager.update_settings(await req.json())


# ---------- containers ----------
@app.get("/api/containers")
async def list_containers():
    return await manager.list_containers()


@app.post("/api/containers")
async def create_container(req: Request):
    body = await req.json()
    if not body.get("model"):
        raise HTTPException(400, "model is required")
    try:
        return await manager.create_container(
            model=body["model"],
            port=body.get("port"),
            gpu_memory_utilization=body.get("gpu_memory_utilization"),
            extra_args=body.get("extra_args") or [],
            name=body.get("name"),
            image=body.get("image"),
        )
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/containers/{name}/start")
async def start_container(name: str):
    try:
        return await manager.start_container(name)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/containers/{name}/stop")
async def stop_container(name: str):
    try:
        return await manager.stop_container(name)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.delete("/api/containers/{name}")
async def remove_container(name: str):
    try:
        return await manager.remove_container(name)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/containers/{name}/logs")
async def get_logs(name: str, tail: int = 200):
    try:
        return {"logs": await manager.get_logs(name, tail)}
    except Exception as e:
        raise HTTPException(500, str(e))


# ---------- images ----------
@app.get("/api/images")
async def list_images():
    return await manager.list_images()


@app.post("/api/images/pull")
async def pull_image(req: Request):
    body = await req.json()
    image = body.get("image")
    if not image:
        raise HTTPException(400, "image is required")
    return StreamingResponse(
        manager.pull_image_stream(image),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.delete("/api/images/{image_id:path}")
async def remove_image(image_id: str):
    try:
        return await manager.remove_image(image_id)
    except Exception as e:
        raise HTTPException(500, str(e))


# ---------- inference / queue ----------
@app.post("/api/inference")
async def submit_inference(req: Request):
    body = await req.json()
    if not body.get("model") and not body.get("container"):
        raise HTTPException(400, "model or container is required")
    if not body.get("messages"):
        raise HTTPException(400, "messages is required")
    return await manager.submit_job(
        model=body.get("model") or "",
        messages=body["messages"],
        params=body.get("params") or {},
        container=body.get("container"),
    )


@app.get("/api/inference/{job_id}")
async def get_job(job_id: str):
    job = await manager.get_job(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return manager._public_job(job)


@app.post("/api/queue/{job_id}/run")
async def force_run(job_id: str):
    return await manager.force_run(job_id)


@app.delete("/api/queue/{job_id}")
async def cancel_job(job_id: str):
    return await manager.cancel_job(job_id)


@app.post("/api/queue/clear")
async def clear_finished():
    return await manager.clear_finished()


# ---------- static frontend ----------
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.get("/")
async def index():
    return FileResponse(ROOT / "static" / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=7878, log_level="info")
