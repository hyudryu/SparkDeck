"""FastAPI entry point for SparkDeck."""
import asyncio
import json
import logging
import os
import re
import time
from urllib.parse import urlsplit
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import httpx

from disk_manager import DiskScanJobs, browse_directories, delete_entries
from manager import Manager, ClientAbort, FanSettingsConflict
from cluster import (
    AGENT_PROTOCOL_VERSION,
    COORDINATOR_ID_HEADER,
    LOCAL_NODE_ID,
)
from mcp_server import ControllerClient, build_server
from sparkdeck import SparkDeckService
from sparkdeck.onboarding import (
    FORWARD_HEADERS,
    OnboardingService,
    forward_management_request,
    is_forwardable_path,
)
from sparkdeck.updater import CONFIRMATION, UpdateService, current_revision
from sparkdeck.web import configure_static_asset_mime_types, register_spa_routes

ROOT = Path(__file__).parent
manager = Manager(data_dir=ROOT / "data")
sparkdeck = SparkDeckService(manager, data_dir=ROOT / "data")
onboarding = OnboardingService(manager, data_dir=ROOT / "data", port=7878)
updater = UpdateService(manager, root=ROOT, data_dir=ROOT / "data")
disk_scan_jobs = DiskScanJobs()
mcp_control = build_server(
    ControllerClient("http://127.0.0.1:7878"),
    token=(
        os.environ.get("SPARKDECK_MCP_TOKEN")
        or os.environ.get("VLLM_MCP_TOKEN")
    ),
    public_url=(
        os.environ.get("SPARKDECK_MCP_PUBLIC_URL")
        or os.environ.get("VLLM_MCP_PUBLIC_URL")
        or "http://127.0.0.1:7878/mcp"
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
            msg = _redact_log(self.format(record))
            _log_buffer.append(msg)
        except Exception:
            pass


_LOG_SECRET_PATTERNS = (
    re.compile(r"(?i)(authorization\s*[:=]\s*(?:bearer|basic)\s+)[^\s,;]+"),
    re.compile(
        r'''(?ix)
        ((?:["']?(?:api[_-]?key|access[_-]?token|hf[_-]?token|
        hugging[_-]?face[_-]?hub[_-]?token|password|secret)["']?)\s*[:=]\s*["']?)
        [^"'\s,;}]+
        ''',
    ),
    re.compile(r"(?i)([?&](?:token|key|api_key|access_token)=)[^&\s]+"),
)


def _redact_log(message: str) -> str:
    redacted = str(message)
    for pattern in _LOG_SECRET_PATTERNS:
        redacted = pattern.sub(r"\1[REDACTED]", redacted)
    configured_token = manager._resolved_hf_token()
    if configured_token:
        redacted = redacted.replace(configured_token, "[REDACTED]")
    return redacted


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
            await sparkdeck.close()
            await manager.stop()


app = FastAPI(
    title="SparkDeck",
    description="Run, compare, and benchmark open models across local inference runtimes.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    if is_forwardable_path(request.url.path):
        assignment = onboarding.assignment.load()
        forwarded = any(request.headers.get(name) for name in FORWARD_HEADERS)
        if assignment:
            response = await forward_management_request(request, manager, assignment)
        elif forwarded:
            valid, detail = onboarding.validate_forward_headers(request.headers)
            if not valid:
                response = Response(
                    json.dumps({"detail": detail}), status_code=401,
                    media_type="application/json",
                )
            else:
                response = await call_next(request)
        else:
            response = await call_next(request)
    else:
        response = await call_next(request)
    response.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; connect-src 'self'; img-src 'self' data:; "
        "style-src 'self' 'unsafe-inline'; font-src 'self'; object-src 'none'; "
        "base-uri 'self'; frame-ancestors 'none'",
    )
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    return response


def _require_agent(request: Request) -> None:
    authorization = request.headers.get("authorization", "")
    scheme, _, token = authorization.partition(" ")
    controller_id = request.headers.get(COORDINATOR_ID_HEADER, "")
    if scheme.lower() != "bearer" or not manager.agent_credentials.authorize_controller(
        token, controller_id
    ):
        raise HTTPException(401, "invalid node-agent token")


async def _require_managed_agent_container(name: str, request: Request) -> None:
    _require_agent(request)
    if not await manager.is_managed_container(name):
        raise HTTPException(404, "managed container not found")


_STORAGE_PRIVATE_KEYS = {
    "path", "cache_path", "snapshot_path", "blob_path", "agent_url",
    "agent_token", "token", "hf_token", "authorization",
}
_STORAGE_INSTRUCTIONS = [
    "Pair SparkDeck nodes over a cluster-private network such as Tailscale.",
    "Only complete Hugging Face cache weights are shown and transferable.",
    "Choose an online source and one or more online targets with enough free space.",
]


def _public_storage_payload(value):
    """Defense-in-depth redaction for every virtual NAS HTTP response."""
    if isinstance(value, dict):
        return {
            key: _public_storage_payload(item)
            for key, item in value.items()
            if str(key).casefold() not in _STORAGE_PRIVATE_KEYS
            and not str(key).casefold().endswith(("_path", "_token"))
        }
    if isinstance(value, list):
        return [_public_storage_payload(item) for item in value]
    return value


def _require_virtual_nas_enabled() -> None:
    if not manager.virtual_nas_enabled():
        raise HTTPException(
            409,
            "Virtual NAS is disabled. Enable it in Storage settings before using it.",
        )


def _storage_error(exc: Exception) -> HTTPException:
    if isinstance(exc, FileExistsError):
        return HTTPException(409, str(exc))
    if isinstance(exc, LookupError):
        return HTTPException(404, str(exc))
    if isinstance(exc, RuntimeError):
        return HTTPException(409, str(exc))
    return HTTPException(400, str(exc))


def _requested_node_ids(body: dict) -> list[str] | None:
    """Accept the versioned list field plus a scalar compatibility field."""
    raw = body.get("node_ids")
    scalar = body.get("node_id")
    if raw is None and scalar is None:
        return None
    if raw is None:
        raw = []
    if not isinstance(raw, list):
        raise ValueError("node_ids must be an array")
    if scalar is not None:
        raw = [*raw, scalar]
    result = []
    for value in raw:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("node_ids must contain non-empty node IDs")
        node_id = value.strip()
        if node_id not in result:
            result.append(node_id)
    if not result:
        raise ValueError("node_ids must contain at least one node ID")
    return result


# ---------- shared controller onboarding ----------
@app.get("/api/v1/onboarding")
async def onboarding_status(req: Request):
    return await onboarding.status(str(req.base_url))


@app.post("/api/v1/onboarding/join")
async def onboarding_join(req: Request):
    try:
        body = await req.json()
        if not isinstance(body, dict):
            raise ValueError("request body must be an object")
        return await onboarding.join(body, str(req.base_url))
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/v1/onboarding/register")
async def onboarding_register(req: Request):
    try:
        body = await req.json()
        if not isinstance(body, dict):
            raise ValueError("request body must be an object")
        client_id = req.client.host if req.client else "unknown"
        return await onboarding.register(body, str(req.base_url), client_id)
    except (ValueError, json.JSONDecodeError) as exc:
        status = 429 if "too many join attempts" in str(exc) else 400
        raise HTTPException(status, str(exc)) from exc


@app.post("/api/v1/onboarding/leave")
async def onboarding_leave(req: Request):
    return await onboarding.leave(str(req.base_url))


@app.post("/api/v1/onboarding/unregister")
async def onboarding_unregister(req: Request):
    try:
        return onboarding.unregister(req.headers)
    except PermissionError as exc:
        raise HTTPException(401, str(exc)) from exc
    except ValueError as exc:
        status = 409 if "still used by" in str(exc) else 404
        raise HTTPException(status, str(exc)) from exc


# ---------- aggregate state ----------
def _public_legacy_recipe(recipe: dict) -> dict:
    """Remove embedded credentials without changing the legacy recipe shape."""
    public = dict(recipe)
    if "extra_args" in public:
        public["extra_args"] = manager._without_hf_cli_credentials(
            public.get("extra_args") or []
        )
    return public


@app.get("/api/state")
async def get_state():
    state = await manager.get_state()
    # The MCP compatibility client still discovers recipes through this
    # aggregate endpoint. Keep these durable recipe records available while
    # the new SparkDeck UI uses the versioned application API.
    state["recipes"] = [_public_legacy_recipe(recipe) for recipe in manager.recipes]
    state["recipe_launches"] = dict(manager.recipe_launches)
    state["supported_runtimes"] = list(sparkdeck.registry.kinds)
    return state


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


@app.get("/api/temperature-runs")
async def get_temperature_runs():
    return manager.temperature_runs_state()


@app.get("/api/temperature-runs/{run_id}")
async def get_temperature_run(run_id: str):
    try:
        return manager.temperature_run(run_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.post("/api/temperature-runs")
async def arm_temperature_run(req: Request):
    body = await req.json()
    try:
        return await manager.arm_temperature_recording(
            body.get("node_id"),
            body.get("target_temp_c"),
            body.get("trigger_margin_pct", 5),
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(502, str(exc)) from exc


@app.post("/api/temperature-runs/cancel")
async def cancel_temperature_run():
    try:
        return await manager.cancel_temperature_recording()
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.put("/api/temperature-runs/{run_id}")
async def rename_temperature_run(run_id: str, req: Request):
    body = await req.json()
    try:
        return manager.rename_temperature_run(run_id, body.get("name"))
    except ValueError as exc:
        status = 404 if "not found" in str(exc) else 400
        raise HTTPException(status, str(exc)) from exc


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
        "app_revision": current_revision(ROOT),
    }


@app.post("/api/agent/pair")
async def agent_pair(req: Request):
    body = await req.json()
    try:
        pairing_code = body.get("pairing_code") or ""
        result = manager.agent_credentials.pair(
            pairing_code, body.get("controller_id") or ""
        )
    except ValueError as exc:
        raise HTTPException(403, str(exc)) from exc
    result["name"] = manager.settings.get("cluster_node_name")
    return result


@app.get("/api/agent/status")
async def agent_status(req: Request):
    _require_agent(req)
    return await manager.agent_status()


@app.get("/api/agent/system-update")
async def agent_system_update(req: Request):
    _require_agent(req)
    return updater.agent_status()


@app.post("/api/agent/system-update", status_code=202)
async def agent_start_system_update(req: Request):
    _require_agent(req)
    try:
        body = await req.json()
        if not isinstance(body, dict) or set(body) != {"tag", "revision"}:
            raise ValueError("request must contain only tag and revision")
        return await updater.start_local(str(body["tag"]), str(body["revision"]))
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.post("/api/agent/system-update/preflight")
async def agent_preflight_system_update(req: Request):
    _require_agent(req)
    try:
        body = await req.json()
        if not isinstance(body, dict) or set(body) != {"tag", "revision"}:
            raise ValueError("request must contain only tag and revision")
        return await updater.preflight_local(str(body["tag"]), str(body["revision"]))
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.patch("/api/agent/node")
async def agent_rename_node(req: Request):
    _require_agent(req)
    try:
        body = await req.json()
        if not isinstance(body, dict):
            raise ValueError("request body must be an object")
        if set(body) != {"name"}:
            raise ValueError("request body must contain only name")
        return await manager.rename_cluster_node(LOCAL_NODE_ID, body.get("name"))
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/agent/temperature-history")
async def agent_temperature_history(req: Request):
    _require_agent(req)
    return manager.temperature_history()


@app.get("/api/agent/stats")
async def agent_stats(req: Request):
    _require_agent(req)
    return await manager.get_stats()


@app.get("/api/agent/routeros")
async def agent_routeros(req: Request):
    _require_agent(req)
    return await manager.routeros.overview()


@app.put("/api/agent/routeros/connection")
async def agent_connect_routeros(req: Request):
    _require_agent(req)
    try:
        body = await req.json()
        return await manager.routeros.connect(body)
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(502, str(exc)) from exc


@app.delete("/api/agent/routeros/connection")
async def agent_disconnect_routeros(req: Request):
    _require_agent(req)
    try:
        return manager.routeros.disconnect()
    except RuntimeError as exc:
        raise HTTPException(500, str(exc)) from exc


@app.patch("/api/agent/routeros/fan-settings")
async def agent_update_routeros_fan(req: Request):
    _require_agent(req)
    try:
        body = await req.json()
        return await manager.routeros.update_fan_settings(body)
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(502, str(exc)) from exc


@app.post("/api/agent/images/pull")
async def agent_pull_image(req: Request):
    _require_agent(req)
    body = await req.json()
    image = str(body.get("image") or "").strip()
    if not image:
        raise HTTPException(400, "image is required")
    try:
        return await manager.pull_image_result(image)
    except Exception as exc:
        raise HTTPException(502, str(exc)[:500]) from exc


@app.get("/api/agent/images")
async def agent_images(req: Request):
    _require_agent(req)
    images, containers = await asyncio.gather(
        manager.list_images(), manager.list_containers(),
    )
    return {"images": images, "containers": containers}


@app.delete("/api/agent/images/{image_id:path}")
async def agent_remove_image(image_id: str, req: Request):
    _require_agent(req)
    try:
        return await manager.remove_image(image_id)
    except Exception as exc:
        raise HTTPException(500, str(exc)[:500]) from exc


@app.get("/api/agent/virtual-nas/inventory")
async def agent_virtual_nas_inventory(req: Request):
    _require_agent(req)
    models = await asyncio.to_thread(manager.virtual_nas.inventory)
    return _public_storage_payload({"models": models})


@app.get("/api/agent/virtual-nas/models/{model_id:path}/export")
async def agent_virtual_nas_export(model_id: str, req: Request):
    _require_agent(req)
    try:
        stream = manager.virtual_nas.export_model(model_id)
        return StreamingResponse(stream, media_type="application/x-tar")
    except (ValueError, LookupError, RuntimeError) as exc:
        raise _storage_error(exc) from exc


@app.put("/api/agent/virtual-nas/models/{model_id:path}/import")
async def agent_virtual_nas_import(model_id: str, req: Request):
    _require_agent(req)
    expected_header = req.headers.get("x-sparkdeck-expected-bytes")
    try:
        expected_bytes = int(expected_header) if expected_header is not None else None
        if expected_bytes is not None and expected_bytes < 0:
            raise ValueError("X-SparkDeck-Expected-Bytes must not be negative")
        result = await manager.virtual_nas.import_model(
            model_id, req.stream(), expected_bytes=expected_bytes,
        )
        return _public_storage_payload(result)
    except (TypeError, ValueError, LookupError, RuntimeError, FileExistsError) as exc:
        raise _storage_error(exc) from exc


@app.delete("/api/agent/virtual-nas/models/{model_id:path}")
async def agent_virtual_nas_delete(model_id: str, req: Request):
    _require_agent(req)
    try:
        return _public_storage_payload(
            await manager.delete_virtual_nas_model(LOCAL_NODE_ID, model_id)
        )
    except (ValueError, LookupError, RuntimeError) as exc:
        raise _storage_error(exc) from exc


@app.post("/api/agent/inference/health")
async def agent_inference_health(req: Request):
    _require_agent(req)
    body = await req.json()
    model = str(body.get("model") or "").strip()
    if not model:
        raise HTTPException(400, "model is required")
    container_name = body.pop("_sparkdeck_container_name", None)
    deployment_id = body.pop("_sparkdeck_deployment_id", None)
    try:
        container = await manager._resolve_vllm_target(
            model, container_name=container_name, deployment_id=deployment_id,
        )
        return {"ready": await manager._check_ready(container), "model": model}
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.post("/api/agent/inference/{endpoint:path}")
async def agent_inference(endpoint: str, req: Request):
    _require_agent(req)
    if endpoint not in {"chat/completions", "completions"}:
        raise HTTPException(404, "inference endpoint not found")
    body = await req.json()
    model = str(body.get("model") or "").strip()
    if not model:
        raise HTTPException(400, "model is required")
    container_name = body.pop("_sparkdeck_container_name", None)
    deployment_id = body.pop("_sparkdeck_deployment_id", None)
    cancel = asyncio.Event()
    watcher = _watch_disconnect(req, cancel)
    stream = False
    try:
        result = (
            await manager._vllm_chat(
                model, body, bool(body.get("stream")), cancel,
                container_name=container_name, deployment_id=deployment_id,
            )
            if endpoint == "chat/completions"
            else await manager._vllm_completions(
                model, body, bool(body.get("stream")), cancel,
                container_name=container_name, deployment_id=deployment_id,
            )
        )
        stream = hasattr(result, "__aiter__")
    except ClientAbort:
        return Response(status_code=499)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    finally:
        if not stream:
            watcher.cancel()
    if stream:
        return StreamingResponse(
            _guard_stream(result, watcher), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    return result


@app.get("/api/agent/token-usage")
async def agent_token_usage(req: Request):
    _require_agent(req)
    return manager.token_usage_sync_snapshot()


@app.post("/api/agent/token-usage")
async def agent_merge_token_usage(req: Request):
    _require_agent(req)
    try:
        changed = manager.merge_token_usage_sync(await req.json())
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"enabled": True, "changed": changed}


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
        detail = str(exc)
        credential = body.get("hf_token")
        if isinstance(credential, str) and credential:
            detail = detail.replace(credential, "[REDACTED]")
        detail = _redact_log(detail)
        raise HTTPException(500, detail) from exc


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
async def agent_container_logs(name: str, req: Request, tail: int = 300):
    _require_agent(req)
    try:
        return {
            "logs": await manager.get_cluster_member_logs(
                name, max(1, min(tail, 100_000))
            )
        }
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
        body = await req.json()
        if not isinstance(body, dict):
            raise ValueError("request body must be an object")
        if set(body) == {"name"}:
            return await manager.rename_cluster_node(node_id, body["name"])
        # Preserve the legacy general registry editor for enabled/network
        # configuration. Only its exact name-only form gains propagation.
        try:
            return manager.node_registry.update(node_id, body)
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


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


@app.put("/api/token-stats/rules")
async def update_usage_routing_rule(req: Request):
    try:
        body = await req.json()
        return manager.update_usage_routing_rule(
            body.get("source"), body.get("destination")
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.delete("/api/token-stats/rules/{source:path}")
async def delete_usage_routing_rule(source: str):
    try:
        return manager.delete_usage_routing_rule(source)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.delete("/api/token-stats/{model_path:path}")
async def erase_usage_model(model_path: str):
    try:
        return manager.erase_usage_model(model_path)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


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
    try:
        node_ids = _requested_node_ids(body)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if node_ids is not None:
        async def selected_pull_events():
            try:
                result = await manager.pull_image_on_nodes(image, node_ids)
                yield f"data: {json.dumps(result)}\n\n"
            except Exception as exc:
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"
            yield "data: {\"done\": true}\n\n"
        return StreamingResponse(
            selected_pull_events(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    return StreamingResponse(
        manager.pull_image_stream(image),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _v1_image_item(raw: dict, containers: list[dict]) -> dict:
    used_images = {str(item.get("image") or "") for item in containers}
    tags = raw.get("tags") or []
    first_tag = tags[0] if tags else None
    repository = first_tag
    tag = None
    if first_tag and ":" in first_tag.rsplit("/", 1)[-1]:
        repository, tag = first_tag.rsplit(":", 1)
    folded = " ".join(tags).casefold()
    runtimes = []
    if raw.get("is_vllm") or "vllm" in folded:
        runtimes.append("vllm")
    if "sglang" in folded:
        runtimes.append("sglang")
    if "llama.cpp" in folded or "ggml-org/llama" in folded:
        runtimes.append("llama.cpp")
    return {
        **raw,
        "repository": repository,
        "tag": tag,
        "created_at": raw.get("created"),
        "runtimes": runtimes,
        "in_use": raw.get("id") in used_images or any(
            image in used_images for image in tags
        ),
    }


async def _v1_image_inventory() -> dict:
    inventory = await manager.cluster_image_inventory()
    merged: dict[str, dict] = {}
    for result in inventory["results"]:
        node = result["node"]
        for raw in result["images"]:
            item = _v1_image_item(raw, result["containers"])
            key = str(item.get("id") or (item.get("tags") or [""])[0])
            if not key:
                continue
            current = merged.setdefault(key, {
                **item, "node_ids": [], "selected_nodes": [],
            })
            current["in_use"] = bool(current.get("in_use") or item.get("in_use"))
            current["node_ids"].append(node["id"])
            current["selected_nodes"].append(node)
    return {
        "items": list(merged.values()),
        "partial": inventory["partial"],
        "errors": inventory["errors"],
    }


async def _v1_image_items() -> list[dict]:
    return (await _v1_image_inventory())["items"]


@app.get("/api/v1/images")
async def v1_list_images():
    return await _v1_image_inventory()


@app.get("/api/v1/nodes")
async def v1_nodes():
    nodes = await manager.cluster_nodes()
    return {"items": [manager.public_target_node(node) for node in nodes]}


@app.patch("/api/v1/nodes/{node_id}")
async def v1_rename_node(node_id: str, req: Request):
    try:
        body = await req.json()
        if not isinstance(body, dict):
            raise ValueError("request body must be an object")
        if set(body) != {"name"}:
            raise ValueError("request body must contain only name")
        return await manager.rename_cluster_node(node_id, body.get("name"))
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/v1/images/pull", status_code=201)
async def v1_pull_image(req: Request):
    body = await req.json()
    image = str(body.get("image") or "").strip()
    if not image:
        raise HTTPException(400, "image is required")
    try:
        return await manager.pull_image_on_nodes(image, _requested_node_ids(body))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(502, str(exc)[:500]) from exc


@app.delete("/api/v1/images/{image_id:path}")
async def v1_remove_image(image_id: str):
    selected = next(
        (item for item in await _v1_image_items() if item["id"] == image_id), None
    )
    if not selected:
        raise HTTPException(404, "image not found in cluster inventory")
    if selected.get("in_use"):
        raise HTTPException(409, "image is used by a deployment")
    try:
        return await manager.remove_image_on_nodes(image_id, selected.get("node_ids") or [])
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        raise HTTPException(500, str(e))


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


# ---------- recipe compatibility API ----------
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
            sg_tp_size=body.get("sg_tp_size"),
            sg_context_length=body.get("sg_context_length"),
            sg_max_running_requests=body.get("sg_max_running_requests"),
            sg_mem_fraction=body.get("sg_mem_fraction"),
            sg_image=body.get("sg_image"),
            deployment_mode=body.get("deployment_mode", "single"),
            node_ids=body.get("node_ids"),
            launch_controls=body.get("launch_controls"),
            force_new=bool(body.get("force_new")),
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.put("/api/recipes/{rid}")
async def update_recipe(rid: str, req: Request):
    try:
        return await manager.update_recipe(rid, await req.json())
    except ValueError as exc:
        status = 404 if str(exc) == "recipe not found" else 400
        raise HTTPException(status, str(exc)) from exc


# ---------- versioned SparkDeck application API ----------
@app.get("/api/v1/catalog/models")
async def catalog_models(q: str = "", runtime: str | None = None, limit: int = 24):
    try:
        return await sparkdeck.catalog_search(q, limit, runtime)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except httpx.HTTPStatusError as e:
        raise HTTPException(e.response.status_code, "model catalog request failed")
    except httpx.HTTPError as e:
        raise HTTPException(502, f"model catalog unavailable: {e}")


@app.get("/api/v1/deployments")
async def v1_deployments():
    return {"items": await sparkdeck.deployments()}


def _public_recipe(recipe: dict) -> dict:
    """Return the safe saved-configuration contract used by the Models UI."""
    safe_recipe = {
        **recipe,
        "extra_args": manager._without_hf_cli_credentials(
            recipe.get("extra_args") or []
        ),
    }
    contract = manager.recipe_deployment_contract(safe_recipe)
    supported = recipe.get("supported", True) is not False and contract["supported"]
    return {
        "id": recipe.get("id"),
        "name": recipe.get("name") or recipe.get("model"),
        "model": recipe.get("model"),
        "engine": recipe.get("engine") or "vllm",
        "image": recipe.get("image"),
        "gpu_memory_utilization": recipe.get("gpu_memory_utilization"),
        "gpu_memory_gb": recipe.get("gpu_memory_gb"),
        "sg_tp_size": recipe.get("sg_tp_size"),
        "sg_context_length": recipe.get("sg_context_length"),
        "sg_max_running_requests": recipe.get("sg_max_running_requests"),
        "sg_mem_fraction": recipe.get("sg_mem_fraction"),
        "sg_image": recipe.get("sg_image"),
        **contract,
        "node_ids": list(recipe.get("node_ids") or [LOCAL_NODE_ID]),
        "extra_args_count": len(safe_recipe.get("extra_args") or []),
        "supported": supported,
        "error": recipe.get("error") or contract.get("error"),
        "launch": dict(manager.recipe_launches.get(recipe.get("id")) or {}),
    }


@app.get("/api/v1/recipes")
async def v1_recipes():
    """Expose durable legacy recipes in the SparkDeck Models UI."""
    return {"items": [_public_recipe(recipe) for recipe in manager.recipes]}


def _recipe_detail(recipe: dict) -> dict:
    """Public saved-configuration contract plus its editable launch inputs."""
    detail = _public_recipe(recipe)
    detail["extra_args"] = manager._without_hf_cli_credentials(
        recipe.get("extra_args") or []
    )
    detail["launch_controls"] = manager._deployment_launch_controls(
        manager._deployment_launch_settings(recipe)
    )
    return detail


@app.get("/api/v1/recipes/{recipe_id}")
async def v1_recipe_detail(recipe_id: str):
    """Return one saved configuration with its editable launch controls."""
    recipe = await manager.get_recipe(recipe_id)
    if not recipe:
        raise HTTPException(404, "saved configuration not found")
    return _recipe_detail(recipe)


@app.put("/api/v1/recipes/{recipe_id}")
async def v1_update_recipe(recipe_id: str, req: Request):
    """Update the editable fields of a saved configuration."""
    try:
        body = await req.json()
    except json.JSONDecodeError:
        raise HTTPException(400, "request body must be valid JSON")
    if not isinstance(body, dict):
        raise HTTPException(400, "request body must be an object")
    allowed = {
        "name", "extra_args", "launch_controls",
        "gpu_memory_utilization", "gpu_memory_gb",
    }
    unknown = sorted(set(body) - allowed)
    if unknown:
        raise HTTPException(400, f"unsupported field(s): {', '.join(unknown)}")
    try:
        updated = await manager.update_recipe(recipe_id, body)
    except ValueError as exc:
        status = 404 if str(exc) == "recipe not found" else 400
        raise HTTPException(status, str(exc)) from exc
    return _recipe_detail(updated)


@app.post("/api/v1/recipes/{recipe_id}/deploy", status_code=201)
async def v1_deploy_recipe(recipe_id: str, req: Request):
    recipe = await manager.get_recipe(recipe_id)
    if not recipe:
        raise HTTPException(404, "saved configuration not found")
    contract = manager.recipe_deployment_contract(recipe)
    if recipe.get("supported") is False or contract["supported"] is False:
        raise HTTPException(
            400,
            recipe.get("error") or contract.get("error") or "saved runtime is unsupported",
        )
    try:
        body = await req.json()
    except json.JSONDecodeError:
        body = {}
    if not isinstance(body, dict):
        raise HTTPException(400, "request body must be an object")
    try:
        selected_node_ids = _requested_node_ids(body)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    selected_node_ids = list(selected_node_ids or recipe.get("node_ids") or [LOCAL_NODE_ID])
    if len(selected_node_ids) != contract["required_node_count"]:
        raise HTTPException(
            400,
            f"this saved configuration requires exactly {contract['required_node_count']} node(s)",
        )
    if contract["deployment_mode"] == "sharded":
        if LOCAL_NODE_ID not in selected_node_ids:
            raise HTTPException(
                400, "sharded deployments must include the controller node"
            )
        selected_node_ids = [
            LOCAL_NODE_ID,
            *(node_id for node_id in selected_node_ids if node_id != LOCAL_NODE_ID),
        ]
    try:
        await manager.selected_cluster_nodes(selected_node_ids)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    local_model_path = manager._resolve_local_path(str(recipe.get("model") or ""))
    if local_model_path:
        nodes_with_weights = {LOCAL_NODE_ID}
    else:
        inventory = await manager.model_cache_inventory()
        requested_revision = contract.get("model_revision")
        cached_revision = requested_revision or "main"
        nodes_with_weights = {
            node.get("id")
            for node in inventory
            if any(
                model.get("model_id") == recipe.get("model")
                and cached_revision in (model.get("revisions") or [])
                for model in node.get("models") or []
            )
        }
    missing_weights = [
        node_id for node_id in selected_node_ids if node_id not in nodes_with_weights
    ]
    if missing_weights:
        raise HTTPException(
            409,
            "model weights are not available on selected node(s): "
            + ", ".join(missing_weights),
        )
    settings = {
        "extra_args": manager._without_hf_cli_credentials(
            recipe.get("extra_args") or []
        ),
        "image": recipe.get("image"),
        "gpu_memory_utilization": recipe.get("gpu_memory_utilization"),
        "gpu_memory_gb": recipe.get("gpu_memory_gb"),
    }
    if str(recipe.get("engine") or "vllm") == "sglang":
        settings.update({
            "tensor_parallel_size": contract["tensor_parallel_size"],
            "context_length": recipe.get("sg_context_length"),
            "max_running_requests": recipe.get("sg_max_running_requests"),
            "mem_fraction_static": recipe.get("sg_mem_fraction"),
            "image": recipe.get("sg_image") or recipe.get("image"),
        })
    try:
        return await sparkdeck.create_deployment({
            "model": recipe.get("model"),
            "alias": recipe.get("name") or recipe.get("model"),
            "runtime": recipe.get("engine") or "vllm",
            "kind": "managed",
            "settings": {key: value for key, value in settings.items() if value is not None},
            "node_ids": selected_node_ids,
            "deployment_mode": contract["deployment_mode"],
            "recipe_id": recipe_id,
        })
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(502, str(exc)) from exc


_APP_SETTING_DEFAULTS = {
    "theme": "system",
    "community_api_url": "",
}


@app.get("/api/v1/settings")
async def v1_settings():
    values = {
        key: sparkdeck.store.get_setting(key, default)
        for key, default in _APP_SETTING_DEFAULTS.items()
    }
    values["hf_token_configured"] = bool(manager._resolved_hf_token())
    return values


@app.put("/api/v1/settings")
async def v1_update_settings(req: Request):
    body = await req.json()
    if not isinstance(body, dict):
        raise HTTPException(400, "settings must be an object")
    theme = body.get("theme", _APP_SETTING_DEFAULTS["theme"])
    if theme not in ("system", "light", "dark"):
        raise HTTPException(400, "theme must be system, light, or dark")
    community_api_url = str(body.get("community_api_url") or "").strip().rstrip("/")
    if community_api_url:
        try:
            parsed = httpx.URL(community_api_url)
        except Exception as e:
            raise HTTPException(400, "community_api_url must be a valid URL") from e
        if parsed.scheme not in ("http", "https") or not parsed.host:
            raise HTTPException(400, "community_api_url must be an http or https URL")
    values = {
        "theme": theme,
        "community_api_url": community_api_url,
    }
    credential = body.get("hf_token")
    if credential is not None:
        if not isinstance(credential, str):
            raise HTTPException(400, "hf_token must be a string")
        credential = credential.strip()
        if credential:
            if len(credential) > 4096 or any(char.isspace() for char in credential):
                raise HTTPException(400, "hf_token is not valid")
    for key, value in values.items():
        sparkdeck.store.set_setting(key, value)
    if credential:
        await manager.update_settings({"hf_token": credential})
    values["hf_token_configured"] = bool(manager._resolved_hf_token())
    return values


@app.delete("/api/v1/settings/hf-token")
async def v1_clear_hf_token():
    await manager.clear_hf_token()
    values = {
        key: sparkdeck.store.get_setting(key, default)
        for key, default in _APP_SETTING_DEFAULTS.items()
    }
    values["hf_token_configured"] = bool(manager._resolved_hf_token())
    return values


def _require_same_origin_or_forwarded(req: Request, action: str = "system updates") -> None:
    if any(req.headers.get(name) for name in FORWARD_HEADERS):
        return
    origin = req.headers.get("origin")
    if not origin:
        return
    parsed = urlsplit(origin)
    if parsed.scheme not in {"http", "https"} or parsed.netloc.casefold() != req.headers.get("host", "").casefold():
        raise HTTPException(403, f"{action} require a same-origin request")


@app.get("/api/v1/routeros/presence")
async def routeros_presence():
    return await manager.routeros_cluster_presence()


@app.get("/api/v1/routeros")
async def routeros_overview():
    return await manager.routeros_cluster_overview()


@app.put("/api/v1/routeros/nodes/{node_id}/connection")
async def connect_routeros(node_id: str, req: Request):
    _require_same_origin_or_forwarded(req, "RouterOS changes")
    try:
        body = await req.json()
        if not isinstance(body, dict):
            raise ValueError("connection settings must be an object")
        return await manager.connect_routeros(node_id, body)
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(502, str(exc)) from exc


@app.delete("/api/v1/routeros/nodes/{node_id}/connection")
async def disconnect_routeros(node_id: str, req: Request):
    _require_same_origin_or_forwarded(req, "RouterOS changes")
    try:
        return await manager.disconnect_routeros(node_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(502, str(exc)) from exc


@app.patch("/api/v1/routeros/nodes/{node_id}/fan-settings")
async def update_routeros_fan(node_id: str, req: Request):
    _require_same_origin_or_forwarded(req, "RouterOS changes")
    try:
        body = await req.json()
        if not isinstance(body, dict):
            raise ValueError("fan settings must be an object")
        return await manager.update_routeros_fan_settings(node_id, body)
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(502, str(exc)) from exc


@app.get("/api/v1/system-update")
async def system_update_overview():
    return await updater.overview()


@app.post("/api/v1/system-update", status_code=202)
async def start_system_update(req: Request):
    _require_same_origin_or_forwarded(req)
    try:
        body = await req.json()
        if not isinstance(body, dict) or set(body) != {"confirm", "tag"}:
            raise ValueError("request must contain only confirm and tag")
        return await updater.start_cluster(
            str(body.get("confirm") or ""), str(body.get("tag") or ""),
        )
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.get("/api/v1/storage")
async def v1_storage():
    try:
        state = await manager.virtual_nas_inventory()
        return _public_storage_payload({
            "enabled": bool(state.get("enabled")),
            "nodes": state.get("nodes", []),
            "jobs": state.get("jobs", []),
            "instructions": list(_STORAGE_INSTRUCTIONS),
        })
    except (ValueError, LookupError, RuntimeError) as exc:
        raise _storage_error(exc) from exc


@app.get("/api/v1/model-cache")
async def v1_model_cache():
    """Read-only per-node model disk usage; independent of Virtual NAS."""
    try:
        return _public_storage_payload({
            "nodes": await manager.model_cache_inventory(),
        })
    except (ValueError, LookupError, RuntimeError) as exc:
        raise _storage_error(exc) from exc


@app.put("/api/v1/storage/settings")
async def v1_storage_settings(req: Request):
    try:
        body = await req.json()
    except json.JSONDecodeError as exc:
        raise HTTPException(400, "request body is not valid JSON") from exc
    if not isinstance(body, dict) or not isinstance(body.get("enabled"), bool):
        raise HTTPException(400, "enabled must be a boolean")
    await manager.update_settings({"virtual_nas_enabled": body["enabled"]})
    return await v1_storage()


@app.post("/api/v1/storage/transfers", status_code=202)
async def v1_storage_transfer(req: Request):
    _require_virtual_nas_enabled()
    try:
        body = await req.json()
        if not isinstance(body, dict):
            raise ValueError("request body must be an object")
        model_id = body.get("model_id")
        source_node_id = body.get("source_node_id")
        target_node_ids = body.get("target_node_ids")
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be a non-empty model ID")
        if not isinstance(source_node_id, str) or not source_node_id.strip():
            raise ValueError("source_node_id must be a non-empty node ID")
        if (
            not isinstance(target_node_ids, list)
            or not target_node_ids
            or any(not isinstance(item, str) or not item.strip() for item in target_node_ids)
        ):
            raise ValueError("target_node_ids must contain at least one node ID")
        targets = [item.strip() for item in target_node_ids]
        if len(set(targets)) != len(targets):
            raise ValueError("target_node_ids must not contain duplicates")
        result = await manager.queue_virtual_nas_transfer(
            model_id.strip(), source_node_id.strip(),
            targets,
        )
        return _public_storage_payload(result)
    except json.JSONDecodeError as exc:
        raise HTTPException(400, "request body is not valid JSON") from exc
    except (ValueError, LookupError, RuntimeError, FileExistsError) as exc:
        raise _storage_error(exc) from exc


@app.delete("/api/v1/storage/transfers/{job_id}")
async def v1_storage_cancel(job_id: str):
    _require_virtual_nas_enabled()
    if not job_id.strip():
        raise HTTPException(400, "job_id must not be empty")
    try:
        return _public_storage_payload(
            await manager.cancel_virtual_nas_transfer(job_id)
        )
    except (ValueError, LookupError, RuntimeError) as exc:
        raise _storage_error(exc) from exc


@app.delete("/api/v1/storage/nodes/{node_id}/models/{model_id:path}")
async def v1_storage_delete_model(node_id: str, model_id: str):
    _require_virtual_nas_enabled()
    if not node_id.strip() or not model_id.strip():
        raise HTTPException(400, "node_id and model_id must not be empty")
    try:
        return _public_storage_payload(
            await manager.delete_virtual_nas_model(node_id.strip(), model_id)
        )
    except (ValueError, LookupError, RuntimeError) as exc:
        raise _storage_error(exc) from exc


@app.post("/api/v1/deployments", status_code=201)
async def v1_create_deployment(req: Request):
    try:
        return await sparkdeck.create_deployment(await req.json())
    except (ValueError, json.JSONDecodeError) as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/v1/deployments/{deployment_id}/{action}")
async def v1_deployment_action(deployment_id: str, action: str):
    try:
        return await sparkdeck.deployment_action(deployment_id, action)
    except LookupError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.delete("/api/v1/deployments/{deployment_id}")
async def v1_delete_deployment(deployment_id: str):
    try:
        return await sparkdeck.delete_deployment(deployment_id)
    except LookupError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.patch("/api/v1/deployments/{deployment_id}")
async def v1_rename_deployment(deployment_id: str, req: Request):
    try:
        body = await req.json()
    except json.JSONDecodeError:
        raise HTTPException(400, "request body must be valid JSON")
    if not isinstance(body, dict):
        raise HTTPException(400, "request body must be an object")
    try:
        return await sparkdeck.rename_deployment(deployment_id, body.get("alias"))
    except LookupError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/v1/benchmarks")
async def v1_benchmarks(limit: int = 100, offset: int = 0):
    items, total = sparkdeck.store.benchmarks(limit, offset)
    return {"items": items, "total": total, "limit": min(500, max(1, limit)),
            "offset": max(0, offset)}


@app.delete("/api/v1/benchmarks/{sample_id}")
async def v1_delete_benchmark(sample_id: str):
    if not sparkdeck.store.delete_benchmark(sample_id):
        raise HTTPException(404, "benchmark sample not found")
    return {"ok": True, "id": sample_id}


@app.get("/api/v1/community/sync")
async def v1_community_sync():
    return sparkdeck.store.sync_status()


@app.put("/api/v1/community/consent")
async def v1_community_consent(req: Request):
    body = await req.json()
    enabled = body.get("enabled")
    if not isinstance(enabled, bool):
        raise HTTPException(400, "enabled must be a boolean")
    sparkdeck.store.set_community_consent(enabled)
    return sparkdeck.store.sync_status()


@app.post("/api/v1/community/retry")
async def v1_community_retry():
    return {"retried": sparkdeck.store.retry_outbox(),
            "sync": sparkdeck.store.sync_status()}


@app.post("/api/v1/community/pair")
async def v1_community_pairing():
    # The local contract is present so clients can expose the workflow without
    # pretending a hosted identity/upload service exists in this release.
    raise HTTPException(503, "community account pairing is not available in this release")


@app.get("/api/v1/community/aggregates")
async def v1_community_aggregates():
    try:
        return await sparkdeck.community_aggregates()
    except RuntimeError as exc:
        raise HTTPException(502, str(exc)) from exc


# ---------- OpenAI-compatible /v1 proxy ----------
@app.get("/v1/models")
async def v1_models():
    return await sparkdeck.models()


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
        result = await sparkdeck.proxy(body, "chat/completions", cancel)
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
        result = await sparkdeck.proxy(body, "completions", cancel)
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
FRONTEND_DIST = ROOT / "frontend" / "dist"
configure_static_asset_mime_types()
if FRONTEND_DIST.exists():
    app.mount("/static/app", StaticFiles(directory=FRONTEND_DIST), name="sparkdeck-app")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
register_spa_routes(app, FRONTEND_DIST)


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
