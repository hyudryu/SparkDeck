"""FastAPI entry point for VLLMController."""
import asyncio
import json
import logging
import os
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import httpx

from disk_manager import DiskScanJobs, browse_directories, delete_entries
from manager import Manager, ClientAbort, FanSettingsConflict
from cluster import AGENT_PROTOCOL_VERSION, LOCAL_NODE_ID
from mcp_server import ControllerClient, build_server

ROOT = Path(__file__).parent
manager = Manager(data_dir=ROOT / "data")
disk_scan_jobs = DiskScanJobs()
mcp_control = build_server(
    ControllerClient("http://127.0.0.1:7878"),
    token=os.environ.get("VLLM_MCP_TOKEN"),
    public_url=os.environ.get(
        "VLLM_MCP_PUBLIC_URL", "http://127.0.0.1:7878/mcp"
    ),
)
mcp_http_app = mcp_control.streamable_http_app(
    streamable_http_path="/mcp",
    stateless_http=True,
    json_response=True,
    host="0.0.0.0",
)


def _watch_disconnect(req: Request, event: asyncio.Event) -> asyncio.Task:
    """Set `event` once the HTTP client goes away. uvicorn does not cancel
    running handlers on disconnect, so without this an aborted client leaves
    the upstream inference request running to its full token budget."""
    async def run():
        try:
            while not event.is_set():
                if await req.is_disconnected():
                    event.set()
                    return
                await asyncio.sleep(0.5)
        except Exception:
            pass
    return asyncio.create_task(run())


async def _guard_stream(stream, watcher: asyncio.Task):
    """Keep the disconnect watcher alive for the whole SSE stream; the
    endpoint returns before the generator is consumed, so the watcher must
    only be cancelled once iteration ends."""
    try:
        async for chunk in stream:
            yield chunk
    finally:
        watcher.cancel()

# ---------- in-memory server log buffer ----------
MAX_LOG_LINES = 5000
_log_buffer: deque[str] = deque(maxlen=MAX_LOG_LINES)


class _DequeHandler(logging.Handler):
    """Appends formatted log records to the in-memory deque."""
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            _log_buffer.append(msg)
        except Exception:
            pass


def _install_log_capture():
    """Route Python logging + uvicorn access logs into the in-memory buffer."""
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    handler = _DequeHandler()
    handler.setFormatter(fmt)
    # Capture all loggers
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
    # Also capture uvicorn access logs
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        lg = logging.getLogger(name)
        lg.addHandler(handler)
        lg.setLevel(logging.INFO)

_install_log_capture()


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with mcp_control.session_manager.run():
        await manager.start()
        try:
            yield
        finally:
            await manager.stop()


app = FastAPI(title="VLLMController", lifespan=lifespan)


def _require_agent(request: Request) -> None:
    authorization = request.headers.get("authorization", "")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not manager.agent_credentials.accepts_token(token):
        raise HTTPException(401, "invalid node-agent token")


async def _require_managed_agent_container(name: str, request: Request) -> None:
    _require_agent(request)
    if not await manager.is_managed_container(name):
        raise HTTPException(404, "managed container not found")


# ---------- aggregate state ----------
@app.get("/api/state")
async def get_state():
    return await manager.get_state()


@app.get("/api/stats")
async def get_stats():
    return await manager.get_stats()


@app.get("/api/inference-queue")
async def get_inference_queue():
    """Controller-side vLLM admission state, keyed by deployment/container."""
    return manager.inference_admission()


@app.get("/api/temperature-history")
async def get_temperature_history(node_id: str = LOCAL_NODE_ID):
    try:
        return await manager.temperature_history_for_node(node_id)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(502, str(exc)) from exc


@app.get("/api/active-request-rates")
async def get_active_request_rates():
    """Small endpoint polled by the token widget at a fixed cadence."""
    return manager.active_requests()


# ---------- cluster nodes / node agent ----------
@app.get("/api/agent/info")
async def agent_info():
    """Non-secret metadata used before pairing."""
    return {
        "node_id": manager.agent_credentials.node_id,
        "name": manager.settings.get("cluster_node_name"),
        "protocol_version": AGENT_PROTOCOL_VERSION,
        "pairing_required": True,
    }


@app.post("/api/agent/pair")
async def agent_pair(req: Request):
    body = await req.json()
    try:
        result = manager.agent_credentials.pair(body.get("pairing_code") or "")
    except ValueError as exc:
        raise HTTPException(403, str(exc)) from exc
    result["name"] = manager.settings.get("cluster_node_name")
    return result


@app.get("/api/agent/status")
async def agent_status(req: Request):
    _require_agent(req)
    return await manager.agent_status()


@app.get("/api/agent/temperature-history")
async def agent_temperature_history(req: Request):
    _require_agent(req)
    return manager.temperature_history()


@app.get("/api/agent/llama-rpc")
async def agent_llama_rpc_status(req: Request):
    _require_agent(req)
    return manager.llama_rpc_status()


@app.post("/api/agent/llama-rpc")
async def agent_start_llama_rpc(req: Request):
    _require_agent(req)
    body = await req.json()
    try:
        return await manager.start_llama_rpc_worker(
            body.get("host") or "", body.get("port")
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.delete("/api/agent/llama-rpc")
async def agent_stop_llama_rpc(req: Request):
    _require_agent(req)
    return await manager.stop_llama_rpc_worker()


@app.post("/api/agent/containers")
async def agent_create_container(req: Request):
    _require_agent(req)
    body = await req.json()
    if not body.get("model") or not body.get("cluster_member"):
        raise HTTPException(400, "model and cluster_member are required")
    try:
        return await manager.create_container(
            model=body["model"],
            port=body.get("port"),
            engine=body.get("engine", "vllm"),
            gpu_memory_utilization=body.get("gpu_memory_utilization"),
            gpu_memory_gb=body.get("gpu_memory_gb"),
            extra_args=body.get("extra_args") or [],
            name=body.get("name"),
            image=body.get("image"),
            sg_tp_size=body.get("sg_tp_size"),
            sg_context_length=body.get("sg_context_length"),
            sg_max_running_requests=body.get("sg_max_running_requests"),
            sg_mem_fraction=body.get("sg_mem_fraction"),
            sg_image=body.get("sg_image"),
            cluster_member=body.get("cluster_member"),
            hf_token=body.get("hf_token"),
        )
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


@app.post("/api/agent/containers/{name}/start")
async def agent_start_container(name: str, req: Request):
    await _require_managed_agent_container(name, req)
    return await manager.start_container(name)


@app.post("/api/agent/containers/{name}/stop")
async def agent_stop_container(name: str, req: Request):
    await _require_managed_agent_container(name, req)
    return await manager.stop_container(name)


@app.delete("/api/agent/containers/{name}")
async def agent_remove_container(name: str, req: Request):
    _require_agent(req)
    try:
        return await manager.remove_cluster_member(name)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/api/agent/containers/{name}/logs")
async def agent_container_logs(name: str, req: Request):
    _require_agent(req)
    try:
        return {"logs": await manager.get_cluster_member_logs(name, 300)}
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.post("/api/nodes")
async def pair_cluster_node(req: Request):
    try:
        return await manager.pair_node(await req.json())
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.patch("/api/nodes/{node_id}")
async def update_cluster_node(node_id: str, req: Request):
    try:
        return manager.node_registry.update(node_id, await req.json())
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.post("/api/nodes/{node_id}/refresh")
async def refresh_cluster_node(node_id: str):
    try:
        return await manager.refresh_node(node_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.delete("/api/nodes/{node_id}")
async def remove_cluster_node(node_id: str):
    if node_id == LOCAL_NODE_ID:
        raise HTTPException(400, "the coordinator node cannot be removed")
    try:
        removed = manager.remove_cluster_node(node_id)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    if not removed:
        raise HTTPException(404, "node not found")
    return {"ok": True}


@app.post("/api/deployments/{deployment_id}/{action}")
async def deployment_action(deployment_id: str, action: str):
    try:
        return await manager.deployment_action(deployment_id, action)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.put("/api/deployments/{deployment_id}/settings")
async def update_deployment_settings(deployment_id: str, req: Request):
    try:
        return manager.update_deployment_settings(deployment_id, await req.json())
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.put("/api/deployments/{deployment_id}/alias")
async def update_deployment_alias(deployment_id: str, req: Request):
    try:
        body = await req.json()
        return manager.update_deployment_alias(deployment_id, body.get("alias"))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.put("/api/deployments/{deployment_id}/pricing")
async def update_deployment_pricing(deployment_id: str, req: Request):
    try:
        return await manager.update_deployment_pricing(
            deployment_id, await req.json()
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/deployments/{deployment_id}/logs")
async def deployment_logs(deployment_id: str):
    try:
        return await manager.deployment_logs(deployment_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/api/disk")
async def get_disk():
    return await manager.get_disk()


@app.get("/api/disk-manager/browse")
async def browse_disk_manager(path: str = "/"):
    try:
        return await asyncio.to_thread(browse_directories, path)
    except (ValueError, FileNotFoundError, NotADirectoryError, PermissionError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/disk-manager/scan")
async def scan_disk_manager(payload: dict):
    try:
        return disk_scan_jobs.start(payload.get("path"))
    except (ValueError, FileNotFoundError, NotADirectoryError, PermissionError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/disk-manager/scan/{scan_id}")
async def poll_disk_manager_scan(scan_id: str, since: int = 0):
    try:
        return disk_scan_jobs.poll(scan_id, since)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/disk-manager/scan/{scan_id}/cancel")
async def cancel_disk_manager_scan(scan_id: str):
    try:
        return disk_scan_jobs.cancel(scan_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.post("/api/disk-manager/delete")
async def delete_disk_manager_entries(payload: dict):
    try:
        return await asyncio.to_thread(
            delete_entries, payload.get("root"), payload.get("paths"),
        )
    except (ValueError, FileNotFoundError, NotADirectoryError, PermissionError) as exc:
        raise HTTPException(400, str(exc)) from exc


# ---------- lifetime token stats ----------
@app.get("/api/token-stats")
async def get_token_stats():
    return manager.get_token_stats()


@app.post("/api/token-stats/reset")
async def reset_token_stats():
    return manager.reset_token_stats()


@app.post("/api/token-stats/session-reset")
async def reset_session_token_stats():
    """Clear only the session counters, not the persisted lifetime stats."""
    return manager.reset_session_token_stats()


@app.put("/api/token-stats/alias")
async def update_usage_alias(req: Request):
    try:
        body = await req.json()
        return manager.update_usage_alias(
            body.get("model"),
            body.get("alias"),
            merge_group=body.get("merge_group"),
            update_merge_group="merge_group" in body,
        )
    except ValueError as exc:
        status = 404 if str(exc) == "usage model not found" else 400
        raise HTTPException(status, str(exc)) from exc


@app.get("/api/token-cost/{model_path:path}")
async def get_token_cost(model_path: str, session: bool = False):
    stats = manager.session_token_stats.get(model_path) if session else None
    return manager.calculate_cost(model_path, stats)


# ---------- hourly / daily token stats (Usage → Analysis) ----------
@app.get("/api/token-stats/hourly")
async def get_hourly_token_stats(start: str | None = None,
                                 end: str | None = None):
    return manager.get_hourly_token_stats(start, end)


@app.get("/api/token-stats/daily")
async def get_daily_token_stats(start: str | None = None,
                                end: str | None = None,
                                weeks: int = 12):
    return manager.get_daily_token_stats(start, end, weeks)


# ---------- fan control ----------
@app.get("/api/fan/max-speed")
async def get_fan_max_speed():
    return manager.get_fan_max_speed()


@app.post("/api/fan/max-speed")
async def set_fan_max_speed(req: Request):
    body = await req.json()
    return manager.set_fan_max_speed(bool(body.get("enabled", False)))


@app.get("/api/fan/settings")
async def get_fan_settings():
    try:
        return manager.get_fan_settings()
    except FanSettingsConflict as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/fan/settings")
async def update_fan_settings(req: Request):
    try:
        body = await req.json()
    except Exception as exc:
        raise HTTPException(400, "request body is not valid JSON") from exc
    if not isinstance(body, dict):
        raise HTTPException(400, "request body must be an object")
    try:
        return manager.update_fan_settings(
            body.get("mode"),
            body.get("active_settings"),
            body.get("expected_mode"),
        )
    except FanSettingsConflict as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except OSError as exc:
        raise HTTPException(500, f"could not save FanController settings: {exc}") from exc


# ---------- settings ----------
@app.get("/api/settings")
async def get_settings():
    return manager.public_settings()


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
        # New clients send a deployment mode even for single-node launches.
        # Old clients omit it and retain the original local-container path.
        if body.get("deployment_mode") or body.get("node_ids"):
            return await manager.create_deployment(body)
        return await manager.create_container(
            model=body["model"],
            port=body.get("port"),
            engine=body.get("engine", "vllm"),
            gpu_memory_utilization=body.get("gpu_memory_utilization"),
            gpu_memory_gb=body.get("gpu_memory_gb"),
            extra_args=body.get("extra_args") or [],
            name=body.get("name"),
            image=body.get("image"),
            # SGLang-specific fields
            sg_tp_size=body.get("sg_tp_size"),
            sg_context_length=body.get("sg_context_length"),
            sg_max_running_requests=body.get("sg_max_running_requests"),
            sg_mem_fraction=body.get("sg_mem_fraction"),
            sg_image=body.get("sg_image"),
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


@app.put("/api/containers/{name}/settings")
async def update_container_settings(name: str, req: Request):
    body = await req.json()
    try:
        return await manager.update_container_settings(name, body.get("settings") or {})
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.put("/api/containers/{name}/alias")
async def update_container_alias(name: str, req: Request):
    body = await req.json()
    try:
        return await manager.update_container_alias(name, body.get("alias"))
    except ValueError as e:
        raise HTTPException(400, str(e))
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


@app.post("/api/containers/{name}/to-recipe")
async def container_to_recipe(name: str):
    try:
        return await manager.container_to_recipe(name)
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


# ---------- recipes ----------
@app.post("/api/recipes")
async def create_recipe(req: Request):
    body = await req.json()
    if not body.get("model"):
        raise HTTPException(400, "model is required")
    try:
        return await manager.add_recipe(
            model=body["model"],
            name=body.get("name"),
            image=body.get("image"),
            extra_args=body.get("extra_args"),
            gpu_memory_utilization=body.get("gpu_memory_utilization"),
            gpu_memory_gb=body.get("gpu_memory_gb"),
            engine=body.get("engine", "vllm"),
            # SGLang-specific fields
            sg_tp_size=body.get("sg_tp_size"),
            sg_context_length=body.get("sg_context_length"),
            sg_max_running_requests=body.get("sg_max_running_requests"),
            sg_mem_fraction=body.get("sg_mem_fraction"),
            sg_image=body.get("sg_image"),
            deployment_mode=body.get("deployment_mode", "single"),
            node_ids=body.get("node_ids") or [LOCAL_NODE_ID],
            launch_controls=body.get("launch_controls"),
            force_new=bool(body.get("force_new")),
        )
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/recipes/{rid}/launch")
async def launch_recipe(rid: str):
    recipe = await manager.get_recipe(rid)
    if not recipe:
        raise HTTPException(404, "recipe not found")
    manager.recipe_launches[rid] = {
        "phase": "Preparing launch", "started_at": time.time(),
    }
    try:
        if recipe.get("deployment_mode") or recipe.get("node_ids"):
            launch_body = dict(recipe)
            launch_body.pop("id", None)
            launch_body.pop("created_at", None)
            launch_body["recipe_id"] = rid
            result = await manager.create_deployment(launch_body)
            manager.recipe_launches[rid] = {
                "phase": "Starting cluster", "deployment_id": result.get("id"),
                "started_at": time.time(),
            }
            return result
        return await manager.create_container(
            model=recipe["model"],
            image=recipe.get("image"),
            extra_args=recipe.get("extra_args"),
            gpu_memory_utilization=recipe.get("gpu_memory_utilization"),
            gpu_memory_gb=recipe.get("gpu_memory_gb"),
            engine=recipe.get("engine", "vllm"),
            # SGLang-specific fields
            sg_tp_size=recipe.get("sg_tp_size"),
            sg_context_length=recipe.get("sg_context_length"),
            sg_max_running_requests=recipe.get("sg_max_running_requests"),
            sg_mem_fraction=recipe.get("sg_mem_fraction"),
            sg_image=recipe.get("sg_image"),
            recipe_id=rid,
        )
    except Exception as e:
        manager.recipe_launches[rid] = {
            "phase": "Failed", "error": str(e), "finished_at": time.time(),
        }
        raise HTTPException(500, str(e))


@app.put("/api/recipes/{rid}")
async def update_recipe(rid: str, req: Request):
    try:
        return await manager.update_recipe(rid, await req.json())
    except ValueError as exc:
        status = 404 if str(exc) == "recipe not found" else 400
        raise HTTPException(status, str(exc)) from exc


@app.delete("/api/recipes/{rid}")
async def delete_recipe(rid: str):
    ok = await manager.delete_recipe(rid)
    if not ok:
        raise HTTPException(404, "recipe not found")
    return {"ok": True}


# ---------- saved SparkRun references ----------
@app.post("/api/spark-launches")
async def create_spark_launch(req: Request):
    body = await req.json()
    try:
        return await manager.add_spark_launch(body.get("reference") or "")
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/spark-launches/{rid}/refresh")
async def refresh_spark_launch(rid: str):
    launch = await manager.refresh_spark_launch(rid)
    if not launch:
        raise HTTPException(404, "SparkRun not found")
    return launch


@app.delete("/api/spark-launches/{rid}")
async def delete_spark_launch(rid: str):
    ok = await manager.delete_spark_launch(rid)
    if not ok:
        raise HTTPException(404, "SparkRun not found")
    return {"ok": True}


@app.post("/api/spark-launches/{rid}/run")
async def run_spark_launch(rid: str, req: Request):
    body = await req.json() if (await req.body()) else {}
    launch = await manager.get_spark_launch(rid)
    if not launch:
        raise HTTPException(404, "SparkRun not found")
    return StreamingResponse(
        manager.run_spark_launch_stream(
            rid,
            solo=body.get("solo", True),
            overrides=body.get("overrides") or {},
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/spark-runs/{run_id}/cancel")
async def cancel_spark_run(run_id: str):
    return await manager.cancel_spark_run(run_id)


@app.get("/api/spark-runs/{run_id}/logs")
async def get_spark_run_logs(run_id: str):
    lines = manager.get_spark_run_logs(run_id)
    if lines is None:
        raise HTTPException(404, "run not found")
    return {"id": run_id, "lines": lines}


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


# ---------- ollama ----------
@app.post("/api/ollama/pull")
async def ollama_pull(req: Request):
    body = await req.json()
    name = body.get("name")
    if not name:
        raise HTTPException(400, "name is required")
    return StreamingResponse(
        manager.pull_ollama_model_stream(name),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.delete("/api/ollama/models/{name:path}")
async def ollama_delete_model(name: str):
    try:
        return await manager.delete_ollama_model(name)
    except RuntimeError as e:
        raise HTTPException(500, str(e))


# ---------- unsloth ----------
# Model identity comes in the body (ids contain "/", e.g.
# "unsloth/Qwen3.5-35B-A3B-GGUF") to avoid path-converter ambiguity.
@app.post("/api/unsloth/load")
async def unsloth_load(req: Request):
    body = await req.json()
    model_path = body.get("model_path")
    if not model_path:
        raise HTTPException(400, "model_path is required")
    # Run as a tracked task so /api/unsloth/load/cancel can abort a launch
    # that is still waiting for readiness (large models load for minutes).
    task = asyncio.create_task(
        manager.load_unsloth_model(
            model_path=model_path,
            overrides=body.get("settings"),
        )
    )
    manager._llama_load_task = task
    try:
        return await task
    except asyncio.CancelledError:
        return {"ok": False, "canceled": True, "model_path": model_path}
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        manager._llama_load_task = None


@app.post("/api/unsloth/load/cancel")
async def unsloth_cancel_load():
    try:
        return await manager.cancel_unsloth_load()
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/unsloth/unload")
async def unsloth_unload(req: Request):
    body = await req.json()
    model_path = body.get("model_path")
    if not model_path:
        raise HTTPException(400, "model_path is required")
    try:
        return await manager.unload_unsloth_model(model_path)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/unsloth/gguf-variants")
async def unsloth_gguf_variants(model_path: str):
    if not model_path:
        raise HTTPException(400, "model_path is required")
    try:
        return await manager.list_unsloth_gguf_variants(model_path)
    except httpx.HTTPStatusError as e:
        raise HTTPException(e.response.status_code, e.response.text[:300])
    except Exception as e:
        raise HTTPException(502, str(e))


@app.post("/api/unsloth/settings")
async def unsloth_save_settings(req: Request):
    body = await req.json()
    model_path = body.get("model_path")
    if not model_path:
        raise HTTPException(400, "model_path is required")
    try:
        return await manager.set_unsloth_settings(model_path, body.get("settings") or {})
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/unsloth/logs")
async def unsloth_logs(
    model_path: str | None = None, since: int = 0,
    limit_bytes: int = 262144,
):
    try:
        return manager.get_llama_server_logs(
            model_path=model_path, since=since, limit_bytes=limit_bytes
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


# ---------- OpenAI-compatible /v1 proxy ----------
@app.get("/v1/models")
async def v1_models():
    return await manager.proxy_models()


@app.post("/v1/chat/completions")
async def v1_chat_completions(req: Request):
    try:
        body = await req.json()
    except json.JSONDecodeError:
        raise HTTPException(400, "request body is not valid JSON")
    if not isinstance(body, dict) or not body.get("model"):
        raise HTTPException(400, "model is required")
    cancel = asyncio.Event()
    watcher = _watch_disconnect(req, cancel)
    stream = False
    try:
        result = await manager.proxy_chat_completions(body, cancel)
        stream = hasattr(result, "__aiter__")
    except ClientAbort:
        # Client left; upstream request was aborted too. Nothing to send.
        return Response(status_code=499)
    except LookupError as e:
        raise HTTPException(404, str(e))
    except TimeoutError as e:
        raise HTTPException(504, str(e))
    except httpx.HTTPStatusError as e:
        raise HTTPException(e.response.status_code, e.response.text[:500])
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        if not stream:
            watcher.cancel()
    # If result is an async generator (streaming), wrap in SSE
    if stream:
        return StreamingResponse(
            _guard_stream(result, watcher),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    return result


@app.post("/v1/completions")
async def v1_completions(req: Request):
    try:
        body = await req.json()
    except json.JSONDecodeError:
        raise HTTPException(400, "request body is not valid JSON")
    if not isinstance(body, dict) or not body.get("model"):
        raise HTTPException(400, "model is required")
    cancel = asyncio.Event()
    watcher = _watch_disconnect(req, cancel)
    stream = False
    try:
        result = await manager.proxy_completions(body, cancel)
        stream = hasattr(result, "__aiter__")
    except ClientAbort:
        return Response(status_code=499)
    except LookupError as e:
        raise HTTPException(404, str(e))
    except TimeoutError as e:
        raise HTTPException(504, str(e))
    except httpx.HTTPStatusError as e:
        raise HTTPException(e.response.status_code, e.response.text[:500])
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        if not stream:
            watcher.cancel()
    if stream:
        return StreamingResponse(
            _guard_stream(result, watcher),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    return result


# ---------- server logs ----------
@app.get("/api/server-logs")
async def get_server_logs(tail: int = 500):
    """Return the most recent server log lines from the in-memory buffer."""
    lines = list(_log_buffer)
    if tail and tail < len(lines):
        lines = lines[-tail:]
    return {"logs": lines, "total": len(_log_buffer)}


# ---------- static frontend ----------
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.get("/")
async def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/disk-manager")
async def disk_manager_page():
    return FileResponse(ROOT / "static" / "disk-manager.html")


# Keep MCP on the exact /mcp path without Starlette's trailing-slash redirect.
# This catch-all mount must remain after the controller's own routes.
app.mount("", mcp_http_app, name="mcp")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=7878,
        log_level="info",
        timeout_graceful_shutdown=10,
    )
