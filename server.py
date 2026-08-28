"""FastAPI entry point for SparkDeck."""
import asyncio
import json
import logging
import os
import re
import secrets
import sys
import time
from urllib.parse import urlsplit
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import httpx
import jwt

from disk_manager import DiskScanJobs, browse_directories, delete_entries
from manager import Manager, ClientAbort, FanSettingsConflict
from cluster import (
    AGENT_PROTOCOL_VERSION,
    COORDINATOR_ID_HEADER,
    LOCAL_NODE_ID,
)
from mcp_server import ControllerClient, build_server
from sparkdeck import SparkDeckService
from sparkdeck.benchy import BenchyError, BenchyService
from sparkdeck.service import (
    _COMMUNITY_MAX_RESPONSE_BYTES,
    _public_community_aggregates,
)
from sparkdeck.storage import COMMUNITY_API_URL, COMMUNITY_EVIDENCE_POLICY
from sparkdeck.onboarding import (
    FORWARD_HEADERS,
    FORWARD_SCHEME_HEADER,
    OnboardingService,
    forward_management_request,
    is_forwardable_path,
)
from sparkdeck.updater import CONFIRMATION, UpdateService
from sparkdeck.web import configure_static_asset_mime_types, register_spa_routes

ROOT = Path(__file__).parent
manager = Manager(data_dir=ROOT / "data")
sparkdeck = SparkDeckService(manager, data_dir=ROOT / "data")
benchy = BenchyService(manager, sparkdeck, data_dir=ROOT / "data")
onboarding = OnboardingService(
    manager, data_dir=ROOT / "data", port=7878,
    revoke_community_consent=sparkdeck.revoke_community_membership,
)
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
        uploader = asyncio.create_task(community_upload_loop())
        try:
            yield
        finally:
            uploader.cancel()
            await sparkdeck.close()
            await manager.stop()


app = FastAPI(
    title="SparkDeck",
    description="Run, compare, and benchmark open models across local inference runtimes.",
    version="1.0.0",
    lifespan=lifespan,
)

# Community account pairing is anchored on the Cognito user pool that hosts
# native email/password sign-in (see infra/cognito-community.yml). These
# identifiers are not secret; the env overrides keep forks pointed at their
# own pool.
COGNITO_USER_POOL_ID = os.environ.get(
    "SPARKDECK_COGNITO_USER_POOL_ID", "us-east-2_TjntedtdI")
COGNITO_CLIENT_ID = os.environ.get(
    "SPARKDECK_COGNITO_CLIENT_ID", "30ihrkeg4k1rn95d4mmkq00fvl")
COGNITO_ISSUER = os.environ.get(
    "SPARKDECK_COGNITO_ISSUER",
    f"https://cognito-idp.us-east-2.amazonaws.com/{COGNITO_USER_POOL_ID}")
# The browser calls the Cognito IDP API at the issuer's origin; the CSP must
# permit that connection when SparkDeck serves the app.
COGNITO_IDP_ORIGIN = urlsplit(COGNITO_ISSUER)._replace(path="", query="", fragment="").geturl()
_COMMUNITY_SESSION_COOKIE = "sparkdeck_community_session"
_COMMUNITY_SESSION_MAX_AGE = 3600
_COMMUNITY_SESSION_LIMIT = 256
_community_browser_sessions: dict[str, tuple[str, float]] = {}


@app.middleware("http")
async def security_headers(request: Request, call_next):
    forwarded = any(request.headers.get(name) for name in FORWARD_HEADERS)
    forgotten_unregister = (
        forwarded
        and request.method == "POST"
        and request.url.path == "/api/v1/onboarding/unregister"
        and onboarding.is_already_unregistered_worker(request.headers)
    )
    if forgotten_unregister:
        # The route turns this exact already-absent state into the 404 that a
        # force-forgotten worker recognizes as a successful local recovery.
        response = await call_next(request)
    elif is_forwardable_path(request.url.path):
        assignment = onboarding.assignment.load()
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
        f"default-src 'self'; connect-src 'self' {COGNITO_IDP_ORIGIN}; img-src 'self' data:; "
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
    "Partial Hugging Face caches are marked with a warning; only complete caches are transferable.",
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
@app.get("/healthz", include_in_schema=False, status_code=204)
async def healthz():
    """Return local process liveness without touching Docker or cluster nodes."""
    return Response(status_code=204)


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
        "app_revision": updater.runtime_revision,
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
    # agent_status runs the local launcher preflight. On Windows that launcher
    # probes this API's /healthz route, so it cannot run on the event loop that
    # must answer the probe.
    return await asyncio.to_thread(updater.agent_status)


@app.post("/api/agent/system-update", status_code=202)
async def agent_start_system_update(req: Request):
    _require_agent(req)
    try:
        body = await req.json()
        if not isinstance(body, dict) or set(body) != {"branch", "revision"}:
            raise ValueError("request must contain only branch and revision")
        return await updater.start_local(str(body["branch"]), str(body["revision"]))
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.post("/api/agent/system-update/preflight")
async def agent_preflight_system_update(req: Request):
    _require_agent(req)
    try:
        body = await req.json()
        if not isinstance(body, dict) or set(body) != {"branch", "revision"}:
            raise ValueError("request must contain only branch and revision")
        return await updater.preflight_local(str(body["branch"]), str(body["revision"]))
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


@app.post("/api/agent/onboarding/detach")
async def agent_detach_from_controller(req: Request):
    _require_agent(req)
    try:
        return await onboarding.detach()
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.get("/api/agent/temperature-history")
async def agent_temperature_history(req: Request):
    _require_agent(req)
    return manager.temperature_history()


@app.get("/api/agent/stats")
async def agent_stats(req: Request):
    _require_agent(req)
    return await manager.get_stats()


@app.get("/api/agent/fan-control")
async def agent_fan_control(req: Request):
    _require_agent(req)
    try:
        return manager.local_fan_control_overview()
    except FanSettingsConflict as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.patch("/api/agent/fan-control/max-speed")
async def agent_set_fan_max_speed(req: Request):
    _require_agent(req)
    try:
        body = await req.json()
        if not isinstance(body, dict) or set(body) != {"enabled"}:
            raise ValueError("request body must contain only enabled")
        if not isinstance(body["enabled"], bool):
            raise ValueError("enabled must be a boolean")
        return manager.set_fan_max_speed(body["enabled"])
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(400, str(exc)) from exc
    except FanSettingsConflict as exc:
        raise HTTPException(409, str(exc)) from exc
    except OSError as exc:
        raise HTTPException(500, f"could not update FanController: {exc}") from exc


@app.patch("/api/agent/fan-control/settings")
async def agent_update_fan_settings(req: Request):
    _require_agent(req)
    try:
        body = await req.json()
        if not isinstance(body, dict) or set(body) != {
            "mode", "active_settings", "expected_mode",
        }:
            raise ValueError(
                "request body must contain only mode, active_settings, and expected_mode"
            )
        if not isinstance(body["expected_mode"], str):
            raise ValueError("expected_mode must be a string")
        return manager.update_fan_settings(
            body["mode"], body["active_settings"], body["expected_mode"],
        )
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(400, str(exc)) from exc
    except FanSettingsConflict as exc:
        raise HTTPException(409, str(exc)) from exc
    except OSError as exc:
        raise HTTPException(500, f"could not save FanController settings: {exc}") from exc


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
    free_size = await asyncio.to_thread(manager.virtual_nas.free_bytes)
    return _public_storage_payload({"models": models, "free_size": free_size})


@app.post("/api/agent/virtual-nas/models/{model_id:path}/download")
async def agent_virtual_nas_download(model_id: str, req: Request):
    _require_agent(req)
    try:
        body = await req.json()
        if not isinstance(body, dict) or set(body) - {
            "revision", "requested_revision", "hf_token",
            "download_cache_baseline_bytes",
        }:
            raise ValueError(
                "request may contain only revision, requested_revision, hf_token, "
                "and download_cache_baseline_bytes"
            )
        revision = body.get("revision")
        requested_revision = body.get("requested_revision")
        token = body.get("hf_token")
        if revision is not None and not isinstance(revision, str):
            raise ValueError("revision must be a string")
        if token is not None and not isinstance(token, str):
            raise ValueError("hf_token must be a string")
        if requested_revision is not None and not isinstance(requested_revision, str):
            raise ValueError("requested_revision must be a string")
        baseline = body.get("download_cache_baseline_bytes")
        if baseline is not None and (
            isinstance(baseline, bool) or not isinstance(baseline, int) or baseline < 0
        ):
            raise ValueError("download_cache_baseline_bytes must be a non-negative integer")
        download_args = [
            model_id, revision or "main", token if token is not None else "",
            requested_revision or revision or "main",
        ]
        if baseline is not None:
            download_args.append(baseline)
        result = await manager.virtual_nas.download_model_checked(*download_args)
        return _public_storage_payload(result)
    except json.JSONDecodeError as exc:
        raise HTTPException(400, "request body is not valid JSON") from exc
    except (ValueError, LookupError, RuntimeError) as exc:
        raise _storage_error(exc) from exc


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
    model_bytes_header = req.headers.get("x-sparkdeck-model-bytes")
    try:
        expected_bytes = int(expected_header) if expected_header is not None else None
        model_bytes = int(model_bytes_header) if model_bytes_header is not None else None
        if expected_bytes is not None and expected_bytes < 0:
            raise ValueError("X-SparkDeck-Expected-Bytes must not be negative")
        if model_bytes is not None and model_bytes <= 0:
            raise ValueError("X-SparkDeck-Model-Bytes must be positive")
        result = await manager.virtual_nas.import_model(
            model_id, req.stream(), expected_bytes=expected_bytes,
            required_model_bytes=model_bytes,
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
        ready = await manager.inference_target_health(
            model, container_name=container_name, deployment_id=deployment_id,
        )
        return {"ready": ready, "model": model}
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
        raise HTTPException(404, {
            "type": "replica_unavailable", "message": str(exc),
        }) from exc
    except httpx.HTTPStatusError as exc:
        raise HTTPException(exc.response.status_code, {
            "type": "upstream_error", "message": exc.response.text[:500],
        }) from exc
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


@app.put("/api/agent/community-pairing")
async def agent_apply_community_pairing(req: Request):
    """Apply a controller-pushed community sign-in without overriding others."""
    _require_agent(req)
    body = await req.json()
    sub = body.get("sub") if isinstance(body, dict) else None
    email = body.get("email") if isinstance(body, dict) else None
    refresh_token = body.get("refresh_token") if isinstance(body, dict) else None
    if not isinstance(sub, str) or not sub:
        raise HTTPException(400, "sub is required")
    if refresh_token is not None and not isinstance(refresh_token, str):
        raise HTTPException(400, "refresh_token must be a string")
    with sparkdeck.store.locked():
        existing = sparkdeck.store.get_setting(
            "device_pairing", {"status": "not_paired"})
        if existing.get("status") == "paired":
            if existing.get("sub") == sub:
                if refresh_token and refresh_token != existing.get("refresh_token"):
                    # Merge a rotated token into the existing pairing; a fresh
                    # token also clears a stale token_invalid flag.
                    sparkdeck.store.set_setting("device_pairing", {
                        "status": "paired",
                        "sub": sub,
                        "email": email or existing.get("email"),
                        "refresh_token": refresh_token,
                    })
                sparkdeck.store.promote_outbox_for_pairing()
                return {"applied": True, "already": True}
            return {"applied": False, "existing": {"email": existing.get("email")}}
        pairing = {"status": "paired", "sub": sub, "email": email}
        if refresh_token:
            # Stored locally (never echoed back) so this node's uploader can
            # mint ID tokens for consented telemetry uploads.
            pairing["refresh_token"] = refresh_token
        sparkdeck.store.set_setting("device_pairing", pairing)
        sparkdeck.store.promote_outbox_for_pairing()
    return {"applied": True}


@app.delete("/api/agent/community-pairing")
async def agent_apply_community_unpairing(req: Request):
    """Unpair only when the local pairing belongs to the same account."""
    _require_agent(req)
    try:
        body = await req.json()
    except json.JSONDecodeError:
        body = {}
    sub = body.get("sub") if isinstance(body, dict) else None
    if not isinstance(sub, str) or not sub:
        # A missing sub would act as a wildcard; refuse it outright.
        raise HTTPException(400, "sub is required")
    result, existing = await sparkdeck.unpair_community_device(sub)
    if result == "already":
        return {"applied": True, "already": True}
    if result == "conflict":
        return {"applied": False, "existing": {"email": existing.get("email")}}
    _clear_community_browser_sessions(sub)
    return {"applied": True}


@app.put("/api/agent/community-consent")
async def agent_apply_community_consent(req: Request):
    """Apply controller-owned sharing consent on this worker."""
    _require_agent(req)
    try:
        body = await req.json()
    except json.JSONDecodeError as exc:
        raise HTTPException(400, "request body must be valid JSON") from exc
    enabled = body.get("enabled") if isinstance(body, dict) else None
    telemetry_cluster_id = (
        body.get("telemetry_cluster_id") if isinstance(body, dict) else None
    )
    if not isinstance(enabled, bool):
        raise HTTPException(400, "enabled must be a boolean")
    if telemetry_cluster_id is not None and not isinstance(telemetry_cluster_id, str):
        raise HTTPException(400, "telemetry_cluster_id must be a string")
    if telemetry_cluster_id is None:
        await sparkdeck.set_community_consent(enabled)
    else:
        await sparkdeck.set_community_consent(enabled, telemetry_cluster_id)
    return {"applied": True, "enabled": enabled}


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
    return await manager.start_container(name, explicit=True)


@app.post("/api/agent/containers/{name}/stop")
async def agent_stop_container(name: str, req: Request, explicit: bool = False):
    await _require_managed_agent_container(name, req)
    return await manager.stop_container(name, explicit=explicit)


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
        if set(body) == {"hidden_from_dashboard"}:
            return await manager.set_cluster_node_dashboard_hidden(
                node_id, body["hidden_from_dashboard"],
            )
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
    try:
        removed = await manager.detach_cluster_node(node_id)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    except RuntimeError as exc:
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
    try:
        body = await req.json()
        if not isinstance(body, dict) or set(body) != {"enabled"}:
            raise ValueError("request body must contain only enabled")
        if not isinstance(body["enabled"], bool):
            raise ValueError("enabled must be a boolean")
        return manager.set_fan_max_speed(body["enabled"])
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(400, str(exc)) from exc
    except FanSettingsConflict as exc:
        raise HTTPException(409, str(exc)) from exc
    except OSError as exc:
        raise HTTPException(500, f"could not update FanController: {exc}") from exc


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
        return await manager.start_container(name, explicit=True)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/containers/{name}/stop")
async def stop_container(name: str):
    try:
        return await manager.stop_container(name, explicit=True)
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
async def v1_update_node(node_id: str, req: Request):
    try:
        body = await req.json()
        if not isinstance(body, dict):
            raise ValueError("request body must be an object")
        if set(body) == {"name"}:
            return await manager.rename_cluster_node(node_id, body.get("name"))
        if set(body) == {"hidden_from_dashboard"}:
            return await manager.set_cluster_node_dashboard_hidden(
                node_id, body.get("hidden_from_dashboard"),
            )
        raise ValueError(
            "request body must contain only name or hidden_from_dashboard"
        )
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.delete("/api/v1/nodes/{node_id}")
async def v1_remove_node(node_id: str, force: bool = False):
    try:
        removed = await manager.detach_cluster_node(node_id, force=force)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    if not removed:
        raise HTTPException(404, "node not found")
    return {"ok": True, "node_id": node_id, "forced": force}


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


@app.delete("/api/recipes/{rid}")
async def delete_recipe(rid: str):
    """Preserve the legacy recipe deletion contract for existing clients."""
    if not await manager.delete_recipe(rid):
        raise HTTPException(404, "recipe not found")
    return {"ok": True}


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


@app.get("/api/v1/catalog/models/{model_id:path}")
async def catalog_model_details(model_id: str):
    try:
        return await sparkdeck.catalog_details(model_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except httpx.HTTPStatusError as exc:
        status = 404 if exc.response.status_code == 404 else 502
        raise HTTPException(status, "model metadata is unavailable") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(502, "model metadata is unavailable") from exc


@app.get("/api/v1/deployments")
async def v1_deployments():
    return {"items": await sparkdeck.deployments()}


@app.get("/api/v1/deployments/{deployment_id}")
async def v1_deployment_detail(deployment_id: str):
    try:
        return await sparkdeck.deployment_detail(deployment_id)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.put("/api/v1/deployments/{deployment_id}/settings")
async def v1_update_deployment_settings(deployment_id: str, req: Request):
    try:
        body = await req.json()
    except json.JSONDecodeError as exc:
        raise HTTPException(400, "request body must be valid JSON") from exc
    if not isinstance(body, dict):
        raise HTTPException(400, "request body must be an object")
    allowed = {
        "extra_args", "launch_controls",
        "gpu_memory_utilization", "gpu_memory_gb",
    }
    unknown = sorted(set(body) - allowed)
    if unknown:
        raise HTTPException(400, f"unsupported field(s): {', '.join(unknown)}")
    try:
        return await sparkdeck.update_deployment_settings(deployment_id, body)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        message = str(exc)
        status = 409 if (
            "stop the cluster" in message.lower()
            or "editable saved launch settings" in message.lower()
        ) else 400
        raise HTTPException(status, message) from exc


@app.get("/api/v1/deployments/{deployment_id}/logs")
async def v1_deployment_logs(deployment_id: str, tail: int = 300):
    try:
        return await sparkdeck.deployment_logs(deployment_id, tail)
    except LookupError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(500, str(exc))


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


@app.delete("/api/v1/recipes/{recipe_id}", status_code=204)
async def v1_delete_recipe(recipe_id: str):
    """Delete a saved launch recipe without changing existing deployments."""
    if not await manager.delete_recipe(recipe_id):
        raise HTTPException(404, "saved configuration not found")
    return Response(status_code=204)


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
    raw_body = await req.body()
    if raw_body.strip():
        try:
            body = json.loads(raw_body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HTTPException(400, "request body must be valid JSON") from exc
    else:
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
        try:
            readiness = await manager.recipe_model_revision_readiness(
                str(recipe.get("model") or ""),
                contract.get("model_revision"), selected_node_ids,
            )
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(409, str(exc)) from exc
        nodes_with_weights = set(selected_node_ids) - set(
            readiness["missing_node_ids"]
        )
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
        }, background=True)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(502, str(exc)) from exc


_APP_SETTING_DEFAULTS = {
    "theme": "system",
    "default_runtime": "vllm",
    "default_context_length": 8192,
}


@app.get("/api/v1/settings")
async def v1_settings():
    values = {
        key: sparkdeck.store.get_setting(key, default)
        for key, default in _APP_SETTING_DEFAULTS.items()
    }
    values["hf_token_configured"] = bool(manager._resolved_hf_token())
    # cluster_node_name is deliberately absent: this response is forwarded to
    # the controller on joined workers, so it would name the controller rather
    # than the node serving the browser. Clients read the unforwarded
    # /api/v1/onboarding status, which every node answers for itself.
    return values


@app.put("/api/v1/settings")
async def v1_update_settings(req: Request):
    body = await req.json()
    if not isinstance(body, dict):
        raise HTTPException(400, "settings must be an object")
    theme = body.get("theme", _APP_SETTING_DEFAULTS["theme"])
    if theme not in ("system", "light", "dark"):
        raise HTTPException(400, "theme must be system, light, or dark")
    default_runtime = str(body.get("default_runtime", _APP_SETTING_DEFAULTS["default_runtime"]))
    if default_runtime not in ("vllm", "llama.cpp", "sglang"):
        raise HTTPException(400, "default_runtime must be vllm, llama.cpp, or sglang")
    raw_context_length = body.get(
        "default_context_length", _APP_SETTING_DEFAULTS["default_context_length"]
    )
    try:
        default_context_length = int(raw_context_length)
    except (TypeError, ValueError) as e:
        raise HTTPException(400, "default_context_length must be an integer") from e
    if not 256 <= default_context_length <= 10_000_000:
        raise HTTPException(400, "default_context_length must be between 256 and 10000000")
    values = {
        "theme": theme,
        "default_runtime": default_runtime,
        "default_context_length": default_context_length,
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


@app.get("/api/v1/fan-control")
async def fan_control_overview():
    return await manager.fan_control_cluster_overview()


@app.patch("/api/v1/fan-control/nodes/{node_id}/max-speed")
async def update_fan_control_max_speed(node_id: str, req: Request):
    _require_same_origin_or_forwarded(req, "FanController changes")
    try:
        body = await req.json()
        if not isinstance(body, dict) or set(body) != {"enabled"}:
            raise ValueError("request body must contain only enabled")
        if not isinstance(body["enabled"], bool):
            raise ValueError("enabled must be a boolean")
        return await manager.set_node_fan_max_speed(node_id, body["enabled"])
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(400, str(exc)) from exc
    except FanSettingsConflict as exc:
        raise HTTPException(409, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(502, str(exc)) from exc


@app.patch("/api/v1/fan-control/nodes/{node_id}/settings")
async def update_fan_control_settings(node_id: str, req: Request):
    _require_same_origin_or_forwarded(req, "FanController changes")
    try:
        body = await req.json()
        if not isinstance(body, dict) or set(body) != {
            "mode", "active_settings", "expected_mode",
        }:
            raise ValueError(
                "request body must contain only mode, active_settings, and expected_mode"
            )
        if not isinstance(body["expected_mode"], str):
            raise ValueError("expected_mode must be a string")
        return await manager.update_node_fan_settings(
            node_id, body["mode"], body["active_settings"], body["expected_mode"],
        )
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(400, str(exc)) from exc
    except FanSettingsConflict as exc:
        raise HTTPException(409, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(502, str(exc)) from exc
    except OSError as exc:
        raise HTTPException(500, f"could not save FanController settings: {exc}") from exc


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
        if not isinstance(body, dict) or set(body) != {"confirm", "revision"}:
            raise ValueError("request must contain only confirm and revision")
        return await updater.start_cluster(
            str(body.get("confirm") or ""), str(body.get("revision") or ""),
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


@app.post("/api/v1/storage/transfers/preflight")
async def v1_storage_transfer_preflight(req: Request):
    try:
        body = await req.json()
        if not isinstance(body, dict):
            raise ValueError("request body must be an object")
        if set(body) - {"model_id", "revision"}:
            raise ValueError("request may contain only model_id and revision")
        model_id = body.get("model_id")
        revision = body.get("revision")
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be a non-empty model ID")
        if revision is not None and not isinstance(revision, str):
            raise ValueError("revision must be a string")
        return _public_storage_payload(
            await manager.virtual_nas_transfer_preflight(
                model_id.strip(), revision,
            )
        )
    except json.JSONDecodeError as exc:
        raise HTTPException(400, "request body is not valid JSON") from exc
    except (ValueError, LookupError, RuntimeError) as exc:
        raise _storage_error(exc) from exc


@app.post("/api/v1/storage/transfers", status_code=202)
async def v1_storage_transfer(req: Request):
    _require_virtual_nas_enabled()
    _require_same_origin_or_forwarded(req)
    try:
        body = await req.json()
        if not isinstance(body, dict):
            raise ValueError("request body must be an object")
        if set(body) - {"model_id", "source_node_id", "target_node_ids", "revision"}:
            raise ValueError("request contains unsupported fields")
        model_id = body.get("model_id")
        source_node_id = body.get("source_node_id")
        target_node_ids = body.get("target_node_ids")
        revision = body.get("revision")
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be a non-empty model ID")
        if not isinstance(source_node_id, str) or not source_node_id.strip():
            raise ValueError("source_node_id must be a non-empty node ID")
        if revision is not None and not isinstance(revision, str):
            raise ValueError("revision must be a string")
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
            targets, revision,
        )
        return _public_storage_payload(result)
    except json.JSONDecodeError as exc:
        raise HTTPException(400, "request body is not valid JSON") from exc
    except (ValueError, LookupError, RuntimeError, FileExistsError) as exc:
        raise _storage_error(exc) from exc


async def _recipe_preparation_request(recipe_id: str, req: Request) -> tuple[dict, dict, list[str]]:
    recipe = await manager.get_recipe(recipe_id)
    if not recipe:
        raise LookupError("saved configuration not found")
    contract = manager.recipe_deployment_contract(recipe)
    if recipe.get("supported") is False or contract.get("supported") is False:
        raise ValueError(recipe.get("error") or contract.get("error") or "saved runtime is unsupported")
    if manager._resolve_local_path(str(recipe.get("model") or "")):
        raise ValueError("local-path recipes do not use model preparation")
    body = await req.json()
    if not isinstance(body, dict) or set(body) != {"node_ids"}:
        raise ValueError("request must contain only node_ids")
    node_ids = body.get("node_ids")
    if not isinstance(node_ids, list) or not node_ids or any(
        not isinstance(item, str) or not item.strip() for item in node_ids
    ):
        raise ValueError("node_ids must contain at least one node ID")
    selected = [item.strip() for item in node_ids]
    if len(set(selected)) != len(selected):
        raise ValueError("node_ids must not contain duplicates")
    if len(selected) != contract["required_node_count"]:
        raise ValueError(
            f"this saved configuration requires exactly {contract['required_node_count']} node(s)"
        )
    if contract["deployment_mode"] == "sharded":
        if LOCAL_NODE_ID not in selected:
            raise ValueError("sharded deployments must include the controller node")
        selected = [LOCAL_NODE_ID, *(item for item in selected if item != LOCAL_NODE_ID)]
    await manager.selected_cluster_nodes(selected)
    return recipe, contract, selected


@app.post("/api/v1/recipes/{recipe_id}/prepare/preflight")
async def v1_recipe_preparation_preflight(recipe_id: str, req: Request):
    try:
        recipe, contract, node_ids = await _recipe_preparation_request(recipe_id, req)
        return _public_storage_payload(
            await manager.recipe_model_preparation_preflight(
                str(recipe.get("model") or ""), contract.get("model_revision"), node_ids,
            )
        )
    except json.JSONDecodeError as exc:
        raise HTTPException(400, "request body is not valid JSON") from exc
    except (ValueError, LookupError, RuntimeError) as exc:
        raise _storage_error(exc) from exc


@app.post("/api/v1/recipes/{recipe_id}/prepare", status_code=202)
async def v1_recipe_preparation(recipe_id: str, req: Request):
    _require_virtual_nas_enabled()
    _require_same_origin_or_forwarded(req)
    try:
        recipe, contract, node_ids = await _recipe_preparation_request(recipe_id, req)
        return _public_storage_payload(
            await manager.queue_recipe_model_preparation(
                str(recipe.get("model") or ""), contract.get("model_revision"), node_ids,
            )
        )
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


@app.post(
    "/api/v1/storage/nodes/{node_id}/models/{model_id:path}/download",
    status_code=202,
)
async def v1_storage_finish_model_download(node_id: str, model_id: str, req: Request):
    _require_virtual_nas_enabled()
    _require_same_origin_or_forwarded(req)
    try:
        body = await req.json()
        if not isinstance(body, dict) or set(body) - {"revision"}:
            raise ValueError("request may contain only revision")
        revision = body.get("revision")
        if revision is not None and not isinstance(revision, str):
            raise ValueError("revision must be a string")
        return _public_storage_payload(
            await manager.queue_virtual_nas_download(
                model_id.strip(), node_id.strip(), revision,
            )
        )
    except json.JSONDecodeError as exc:
        raise HTTPException(400, "request body is not valid JSON") from exc
    except (ValueError, LookupError, RuntimeError, FileExistsError) as exc:
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
async def v1_deployment_action(deployment_id: str, action: str, req: Request):
    try:
        raw = await req.body()
        body = json.loads(raw) if raw else {}
        if not isinstance(body, dict):
            raise ValueError("request body must be a JSON object")
        node_ids = body.get("node_ids")
        if node_ids is not None:
            if (
                not isinstance(node_ids, list)
                or not node_ids
                or any(not isinstance(item, str) or not item.strip() for item in node_ids)
            ):
                raise ValueError("node_ids must contain non-empty node IDs")
            node_ids = [item.strip() for item in node_ids]
        additional_node_ids = body.get("additional_node_ids")
        if additional_node_ids is not None:
            if (
                not isinstance(additional_node_ids, list)
                or not additional_node_ids
                or any(not isinstance(item, str) or not item.strip() for item in additional_node_ids)
            ):
                raise ValueError("additional_node_ids must contain non-empty node IDs")
            additional_node_ids = [item.strip() for item in additional_node_ids]
        return await sparkdeck.deployment_action(
            deployment_id, action, node_ids, additional_node_ids,
        )
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise HTTPException(400, "request body must be valid JSON")
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


@app.post("/api/v1/benchmark-runs", status_code=201)
async def v1_record_benchmark_run(req: Request):
    try:
        body = await req.json()
    except json.JSONDecodeError as exc:
        raise HTTPException(400, "request body must be valid JSON") from exc
    allowed = {
        "deployment_id", "concurrency", "request_count", "prompt_tokens",
        "generation_tokens", "prompt_seconds", "wall_seconds",
    }
    if not isinstance(body, dict):
        raise HTTPException(400, "request body must be an object")
    unknown = set(body) - allowed
    if unknown:
        raise HTTPException(400, f"unsupported benchmark fields: {', '.join(sorted(unknown))}")
    try:
        return await sparkdeck.record_benchmark_series_point(body)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/v1/benchmark-models")
async def v1_benchmark_models():
    return {"items": sparkdeck.store.benchmark_model_summaries()}


@app.get("/api/v1/benchmark-models/{model_id:path}")
async def v1_benchmark_model(model_id: str):
    detail = sparkdeck.store.benchmark_model_detail(model_id)
    if detail is None:
        raise HTTPException(404, "benchmark model not found")
    return detail


@app.delete("/api/v1/benchmarks/{sample_id}")
async def v1_delete_benchmark(sample_id: str):
    if not await sparkdeck.delete_benchmark(sample_id):
        raise HTTPException(404, "benchmark sample not found")
    return {"ok": True, "id": sample_id}


# ---------- llama-benchy benchmark runner ----------

@app.get("/api/v1/benchy/status")
async def v1_benchy_status():
    status = await benchy.detect()
    status["active_run_id"] = (benchy.active_run() or {}).get("id")
    return status


@app.post("/api/v1/benchy/install")
async def v1_benchy_install():
    try:
        return await benchy.install()
    except BenchyError as exc:
        raise HTTPException(502, str(exc)) from exc


@app.get("/api/v1/benchy/models")
async def v1_benchy_models():
    items = [
        {key: value for key, value in model.items() if not key.startswith("_")}
        for model in await benchy.served_models()
    ]
    return {"items": items}


@app.post("/api/v1/benchy/runs", status_code=202)
async def v1_benchy_start_run(req: Request):
    try:
        body = await req.json()
    except json.JSONDecodeError as exc:
        raise HTTPException(400, "request body must be valid JSON") from exc
    try:
        return await benchy.start_run(body if isinstance(body, dict) else {})
    except BenchyError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/v1/benchy/runs")
async def v1_benchy_runs():
    return {"items": benchy.list_runs()}


@app.get("/api/v1/benchy/runs/{run_id}")
async def v1_benchy_run(run_id: str):
    try:
        return benchy.get_run(run_id)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.post("/api/v1/benchy/runs/{run_id}/cancel")
async def v1_benchy_cancel_run(run_id: str):
    try:
        return await benchy.cancel_run(run_id)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except BenchyError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.delete("/api/v1/benchy/runs/{run_id}")
async def v1_benchy_delete_run(run_id: str):
    try:
        benchy.delete_run(run_id)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except BenchyError as exc:
        raise HTTPException(400, str(exc)) from exc
    except OSError as exc:
        raise HTTPException(500, f"could not delete benchmark run files: {exc}") from exc
    return {"ok": True, "id": run_id}


@app.get("/api/v1/benchy/runs/{run_id}/csv")
async def v1_benchy_run_csv(run_id: str):
    try:
        path = benchy.csv_path(run_id)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    return FileResponse(path, media_type="text/csv", filename=f"benchy-{run_id}.csv")


_cognito_jwks_client = None


def _cognito_jwks():
    global _cognito_jwks_client
    if _cognito_jwks_client is None:
        _cognito_jwks_client = jwt.PyJWKClient(
            f"{COGNITO_ISSUER}/.well-known/jwks.json")
    return _cognito_jwks_client


def _verify_cognito_id_token(id_token: str) -> dict:
    signing_key = _cognito_jwks().get_signing_key_from_jwt(id_token)
    claims = jwt.decode(
        id_token,
        signing_key.key,
        algorithms=["RS256"],
        audience=COGNITO_CLIENT_ID,
        issuer=COGNITO_ISSUER,
    )
    if claims.get("token_use") != "id" or not claims.get("sub"):
        raise jwt.InvalidTokenError("token is not a Cognito ID token")
    return claims


async def _verified_community_claims(req: Request) -> dict:
    authorization = req.headers.get("authorization", "")
    scheme, _, token = authorization.partition(" ")
    if scheme.casefold() != "bearer" or not token:
        raise HTTPException(401, "a Cognito ID token is required")
    try:
        return await asyncio.to_thread(_verify_cognito_id_token, token)
    except Exception as exc:
        raise HTTPException(401, "id_token could not be verified") from exc


def _prune_community_browser_sessions() -> None:
    now = time.time()
    expired = [
        token for token, (_, expires_at) in _community_browser_sessions.items()
        if expires_at <= now
    ]
    for token in expired:
        _community_browser_sessions.pop(token, None)


def _clear_community_browser_sessions(sub: str) -> None:
    for token, (session_sub, _) in list(_community_browser_sessions.items()):
        if session_sub == sub:
            _community_browser_sessions.pop(token, None)


def _community_session_response(
    req: Request, body: dict, sub: str | None = None,
) -> JSONResponse:
    response = JSONResponse(body, headers={"Cache-Control": "no-store"})
    if sub:
        _prune_community_browser_sessions()
        token = req.cookies.get(_COMMUNITY_SESSION_COOKIE)
        session = _community_browser_sessions.get(token or "")
        if not session or session[0] != sub:
            while len(_community_browser_sessions) >= _COMMUNITY_SESSION_LIMIT:
                oldest = min(
                    _community_browser_sessions,
                    key=lambda candidate: _community_browser_sessions[candidate][1],
                )
                _community_browser_sessions.pop(oldest, None)
            token = secrets.token_urlsafe(32)
        _community_browser_sessions[token] = (
            sub, time.time() + _COMMUNITY_SESSION_MAX_AGE,
        )
        forwarded_scheme = (
            req.headers.get(FORWARD_SCHEME_HEADER) or ""
        ).casefold()
        browser_scheme = (
            forwarded_scheme
            if forwarded_scheme in {"http", "https"}
            else req.url.scheme
        )
        response.set_cookie(
            _COMMUNITY_SESSION_COOKIE,
            token,
            max_age=_COMMUNITY_SESSION_MAX_AGE,
            httponly=True,
            secure=browser_scheme == "https",
            samesite="strict",
            path="/api/v1/community",
        )
    else:
        response.delete_cookie(
            _COMMUNITY_SESSION_COOKIE, path="/api/v1/community")
    return response


def _require_community_browser_session(req: Request, sub: str) -> None:
    _require_same_origin_or_forwarded(req, "Community account changes")
    _prune_community_browser_sessions()
    token = req.cookies.get(_COMMUNITY_SESSION_COOKIE)
    session = _community_browser_sessions.get(token or "")
    if not session or session[0] != sub:
        raise HTTPException(401, "a current node community session is required")


@app.get("/api/v1/community/auth-config")
async def v1_community_auth_config():
    return {
        "idp_endpoint": f"{COGNITO_IDP_ORIGIN.rstrip('/')}/",
        "client_id": COGNITO_CLIENT_ID,
    }


@app.get("/api/v1/community/session")
async def v1_community_session(req: Request):
    """Restore the sanitized account session held by this SparkDeck node.

    Cluster pairing keeps the reusable Cognito credential in the backend.
    Browsers receive display state only, never the subject, refresh token, or
    a short-lived hosted-service bearer.
    """
    pairing = sparkdeck.store.get_setting(
        "device_pairing", {"status": "not_paired"})
    if not isinstance(pairing, dict) or pairing.get("status") != "paired":
        return _community_session_response(req, {"status": "signed-out"})
    public = {
        "status": "reauth-required",
        "token_invalid": bool(pairing.get("token_invalid")),
    }
    email = pairing.get("email")
    if isinstance(email, str) and email:
        public["email"] = email
    refresh_token = pairing.get("refresh_token")
    if (
        public["token_invalid"]
        or not isinstance(refresh_token, str)
        or not refresh_token
    ):
        return _community_session_response(req, public)
    id_token = await _community_id_token(refresh_token)
    if not id_token:
        current = sparkdeck.store.get_setting(
            "device_pairing", {"status": "not_paired"})
        if isinstance(current, dict) and current.get("token_invalid"):
            public["token_invalid"] = True
            return _community_session_response(req, public)
        raise HTTPException(503, "community sign-in could not be restored")
    try:
        claims = await asyncio.to_thread(_verify_cognito_id_token, id_token)
    except Exception as exc:
        raise HTTPException(503, "community sign-in could not be verified") from exc
    if claims.get("sub") != pairing.get("sub"):
        _mark_community_token_invalid(refresh_token)
        public["token_invalid"] = True
        return _community_session_response(req, public)
    public["status"] = "signed-in"
    return _community_session_response(req, public, pairing["sub"])


@app.get("/api/v1/community/sync")
async def v1_community_sync():
    return _community_sync_status()


def _community_sync_status() -> dict:
    status = sparkdeck.store.sync_status()
    pairing = sparkdeck.store.get_setting(
        "device_pairing", {"status": "not_paired"})
    status["upload_configured"] = bool(
        _validated_community_api_url()
        and isinstance(pairing, dict)
        and pairing.get("refresh_token")
    )
    return status


@app.put("/api/v1/community/consent")
async def v1_community_consent(req: Request):
    body = await req.json()
    enabled = body.get("enabled")
    if not isinstance(enabled, bool):
        raise HTTPException(400, "enabled must be a boolean")
    snapshot = await sparkdeck.set_community_consent(enabled)
    cluster_id = (
        snapshot.get("telemetry_cluster_id")
        if isinstance(snapshot, dict) else None
    )
    cluster = (
        await manager.push_community_consent(enabled, cluster_id)
        if cluster_id
        else await manager.push_community_consent(enabled)
    )
    status = _community_sync_status()
    status["cluster"] = cluster
    return status


@app.post("/api/v1/community/retry")
async def v1_community_retry():
    return {"retried": sparkdeck.store.retry_outbox(),
            "sync": _community_sync_status()}


@app.post("/api/v1/community/pair")
async def v1_community_pair(req: Request):
    _require_same_origin_or_forwarded(req, "Community sign-in")
    body = await req.json()
    id_token = body.get("id_token") if isinstance(body, dict) else None
    if not isinstance(id_token, str) or not id_token:
        raise HTTPException(400, "id_token is required")
    refresh_token = body.get("refresh_token") if isinstance(body, dict) else None
    if refresh_token is not None and not isinstance(refresh_token, str):
        raise HTTPException(400, "refresh_token must be a string")
    try:
        claims = await asyncio.to_thread(_verify_cognito_id_token, id_token)
    except Exception as e:
        raise HTTPException(401, "id_token could not be verified") from e
    if refresh_token:
        refresh_id_token = await _community_id_token(refresh_token, force=True)
        if not refresh_id_token:
            raise HTTPException(401, "refresh_token could not be verified")
        try:
            refresh_claims = await asyncio.to_thread(
                _verify_cognito_id_token, refresh_id_token)
        except Exception as exc:
            raise HTTPException(401, "refresh_token could not be verified") from exc
        if refresh_claims.get("sub") != claims.get("sub"):
            if _community_token_cache.get("refresh_token") == refresh_token:
                _community_token_cache.update({
                    "refresh_token": None,
                    "id_token": None,
                    "expires_at": 0.0,
                })
            raise HTTPException(401, "refresh_token belongs to another account")
    with sparkdeck.store.locked():
        existing = sparkdeck.store.get_setting(
            "device_pairing", {"status": "not_paired"})
        if existing.get("status") == "paired" and existing.get("sub") != claims.get("sub"):
            # Never overwrite a different account's pairing on this node.
            return JSONResponse(
                {"error": "already_paired", "existing": {"email": existing.get("email")}},
                status_code=409,
            )
        retained = (
            existing.get("refresh_token")
            if existing.get("status") == "paired"
            and existing.get("sub") == claims.get("sub")
            else None
        )
        # Stored locally (never echoed back) so the uploader can mint ID
        # tokens for consented telemetry uploads. A re-pair that omits the
        # token keeps the stored one.
        effective_refresh_token = refresh_token or retained
        pairing = {
            "status": "paired",
            "sub": claims.get("sub"),
            "email": claims.get("email"),
        }
        if effective_refresh_token:
            pairing["refresh_token"] = effective_refresh_token
        sparkdeck.store.set_setting("device_pairing", pairing)
        sparkdeck.store.promote_outbox_for_pairing()
    # Best-effort cluster propagation; failures are reported, never raised.
    cluster = await manager.push_community_pairing(
        pairing["sub"], pairing["email"], effective_refresh_token)
    public_pairing = {k: v for k, v in pairing.items() if k != "refresh_token"}
    return _community_session_response(req, {
        "pairing": public_pairing,
        "cluster": cluster,
    }, pairing["sub"])


@app.delete("/api/v1/community/pair")
async def v1_community_unpair(req: Request):
    _require_same_origin_or_forwarded(req, "Community sign-out")
    claims = await _verified_community_claims(req)
    current = sparkdeck.store.get_setting(
        "device_pairing", {"status": "not_paired"})
    sub = current.get("sub") if isinstance(current, dict) else None
    if (
        isinstance(current, dict)
        and current.get("status") == "paired"
        and not isinstance(sub, str)
    ):
        raise HTTPException(409, "the paired account identity is invalid")
    if isinstance(sub, str) and sub != claims.get("sub"):
        raise HTTPException(403, "this account is not paired with the node")
    result, existing = await sparkdeck.unpair_community_device(claims["sub"])
    if result == "conflict":
        raise HTTPException(409, "the paired account changed; retry sign out")
    unpaired_sub = existing.get("sub") if result == "unpaired" else None
    pairing = {"status": "not_paired"}
    if unpaired_sub:
        cluster = await manager.push_community_unpair(unpaired_sub)
    else:
        # Nothing was paired locally, so there is nothing to propagate.
        cluster = {"applied": [], "conflicts": [], "errors": []}
    refresh_token = existing.get("refresh_token") if result == "unpaired" else None
    if isinstance(refresh_token, str) and refresh_token:
        await _revoke_community_refresh_token(refresh_token)
    if unpaired_sub:
        _clear_community_browser_sessions(unpaired_sub)
    return _community_session_response(req, {
        "pairing": pairing,
        "cluster": cluster,
    })


@app.get("/api/v1/community/aggregates")
async def v1_community_aggregates(req: Request):
    pairing = sparkdeck.store.get_setting(
        "device_pairing", {"status": "not_paired"})
    if not isinstance(pairing, dict) or pairing.get("status") != "paired":
        raise HTTPException(401, "community aggregates require sign-in")
    sub = pairing.get("sub")
    if not isinstance(sub, str) or not sub:
        raise HTTPException(401, "the paired account identity is invalid")
    _require_community_browser_session(req, sub)
    if pairing.get("token_invalid"):
        raise HTTPException(401, "community sign-in expired; sign in again")
    if not sparkdeck.store.get_setting("community_consent", False):
        raise HTTPException(403, "community sharing consent is required")
    api_url = _validated_community_api_url()
    if not api_url:
        try:
            return await sparkdeck.community_aggregates()
        except RuntimeError as exc:
            raise HTTPException(502, str(exc)) from exc
    refresh_token = pairing.get("refresh_token")
    if not isinstance(refresh_token, str) or not refresh_token:
        raise HTTPException(401, "community sign-in must be refreshed")
    id_token = await _community_id_token(refresh_token)
    if not id_token:
        current = sparkdeck.store.get_setting(
            "device_pairing", {"status": "not_paired"})
        if isinstance(current, dict) and current.get("token_invalid"):
            raise HTTPException(401, "community sign-in expired; sign in again")
        return _community_aggregates_unavailable()
    try:
        upstream = await _community_http.get(
            f"{api_url}/v2/aggregates",
            headers={
                "Authorization": f"Bearer {id_token}",
            },
        )
    except httpx.HTTPError:
        return _community_aggregates_unavailable()
    if upstream.status_code == 401:
        raise HTTPException(401, "community aggregates require sign-in")
    if upstream.status_code != 200:
        return _community_aggregates_unavailable()
    # Never pass an unvalidated or oversized upstream payload to the UI.
    if len(upstream.content) > _COMMUNITY_MAX_RESPONSE_BYTES:
        return _community_aggregates_unavailable()
    try:
        body = upstream.json()
        items = _public_community_aggregates(body)
        items = await sparkdeck.enrich_community_aggregates(items)
    except (TypeError, ValueError):
        return _community_aggregates_unavailable()
    evidence_policy = body.get("evidence_policy")
    return {
        "items": items,
        "availability": str(body.get("availability") or "ok"),
        "evidence_policy": (
            evidence_policy
            if isinstance(evidence_policy, dict)
            else COMMUNITY_EVIDENCE_POLICY
        ),
    }


def _community_aggregates_unavailable() -> dict:
    return {
        "items": [],
        "availability": "unavailable",
        "evidence_policy": COMMUNITY_EVIDENCE_POLICY,
    }


# ---------- Community telemetry uploader ----------
# Shared client for the community API and Cognito token minting. Tests swap in
# a MockTransport-backed client.
_community_http = httpx.AsyncClient(timeout=5)

COMMUNITY_UPLOAD_BATCH_SIZE = 50
COMMUNITY_UPLOAD_INTERVAL_SECONDS = 60
_COMMUNITY_UPLOAD_PACING_SECONDS = 0.15

logger = logging.getLogger(__name__)

_community_token_cache: dict = {
    "refresh_token": None, "id_token": None, "expires_at": 0.0,
}
# Rate limiting: skip uploads until this timestamp after a 429.
_community_upload_not_before = 0.0


def _validated_community_api_url() -> str | None:
    """Return the built-in community API base, or None when misconfigured.

    The URL is server-side configuration; an invalid override (non-HTTPS,
    missing host, embedded credentials) must not receive account tokens.
    """
    try:
        url = httpx.URL(COMMUNITY_API_URL.strip())
    except (TypeError, httpx.InvalidURL):
        return None
    if url.scheme != "https" or not url.host or url.userinfo:
        return None
    return str(url).rstrip("/")


def _mark_community_token_invalid(refresh_token: str) -> None:
    """Flag the stored pairing when Cognito rejects its refresh token for good."""
    pairing = sparkdeck.store.get_setting(
        "device_pairing", {"status": "not_paired"})
    if (
        isinstance(pairing, dict)
        and pairing.get("refresh_token") == refresh_token
        and not pairing.get("token_invalid")
    ):
        sparkdeck.store.set_setting(
            "device_pairing", {**pairing, "token_invalid": True})
    _community_token_cache.update({
        "refresh_token": None, "id_token": None, "expires_at": 0.0,
    })


async def _community_id_token(refresh_token: str, force: bool = False) -> str | None:
    """Mint (and cache) a Cognito ID token from the stored refresh token."""
    cache = _community_token_cache
    if (
        not force
        and cache["id_token"]
        and cache["refresh_token"] == refresh_token
        and cache["expires_at"] > time.time() + 60
    ):
        return cache["id_token"]
    try:
        response = await _community_http.post(
            f"{COGNITO_IDP_ORIGIN}/",
            headers={
                "Content-Type": "application/x-amz-json-1.1",
                "X-Amz-Target": "AWSCognitoIdentityProviderService.InitiateAuth",
            },
            json={
                "ClientId": COGNITO_CLIENT_ID,
                "AuthFlow": "REFRESH_TOKEN_AUTH",
                "AuthParameters": {"REFRESH_TOKEN": refresh_token},
            },
        )
    except httpx.HTTPError:
        return None
    if response.status_code != 200:
        error_code = ""
        try:
            body = response.json()
            error_code = str(body.get("__type") or body.get("error") or "")
        except (ValueError, AttributeError):
            pass
        if "NotAuthorizedException" in error_code or "invalid_grant" in error_code:
            # The refresh token is revoked or expired; retrying is pointless
            # until the account pairs again.
            _mark_community_token_invalid(refresh_token)
        return None
    result = response.json().get("AuthenticationResult") or {}
    id_token = result.get("IdToken")
    if not id_token:
        return None
    cache.update({
        "refresh_token": refresh_token,
        "id_token": id_token,
        "expires_at": time.time() + float(result.get("ExpiresIn", 3600)),
    })
    return id_token


async def _revoke_community_refresh_token(refresh_token: str) -> bool:
    """Best-effort single-session revocation without exposing the token."""
    if _community_token_cache.get("refresh_token") == refresh_token:
        _community_token_cache.update({
            "refresh_token": None, "id_token": None, "expires_at": 0.0,
        })
    try:
        response = await _community_http.post(
            f"{COGNITO_IDP_ORIGIN}/",
            headers={
                "Content-Type": "application/x-amz-json-1.1",
                "X-Amz-Target": "AWSCognitoIdentityProviderService.RevokeToken",
            },
            json={"ClientId": COGNITO_CLIENT_ID, "Token": refresh_token},
        )
    except httpx.HTTPError:
        return False
    return response.status_code == 200


async def _post_community_sample(
    api_url: str, id_token: str, payload: dict, sample_id: str,
) -> httpx.Response | None:
    try:
        return await _community_http.post(
            f"{api_url}/v2/samples",
            json=payload,
            headers={
                "Authorization": f"Bearer {id_token}",
                "Idempotency-Key": sample_id,
            },
        )
    except httpx.HTTPError:
        return None


async def community_upload_once() -> dict:
    """Upload one batch of pending consented samples; transient errors stay pending."""
    global _community_upload_not_before
    # Joined workers never contact the hosted service. Cluster inference is
    # observed and uploaded by the authoritative controller, which also makes
    # a controller-side opt-out fail closed even if a worker is unreachable.
    if manager.is_joined_worker():
        return {"uploaded": 0, "failed": 0, "reason": "controller_owned"}
    if time.time() < _community_upload_not_before:
        return {"uploaded": 0, "failed": 0, "reason": "rate_limited"}
    store = sparkdeck.store
    if not store.get_setting("community_consent", False):
        return {"uploaded": 0, "failed": 0, "reason": "consent_off"}
    pairing = store.get_setting("device_pairing", {"status": "not_paired"})
    if isinstance(pairing, dict) and pairing.get("token_invalid"):
        # The stored refresh token was rejected for good; wait for re-pairing.
        return {"uploaded": 0, "failed": 0, "reason": "token_invalid"}
    refresh_token = (
        pairing.get("refresh_token")
        if isinstance(pairing, dict) and pairing.get("status") == "paired"
        else None
    )
    api_url = _validated_community_api_url()
    if not refresh_token or not api_url:
        return {"uploaded": 0, "failed": 0, "reason": "not_configured"}
    batch = store.outbox_entries(COMMUNITY_UPLOAD_BATCH_SIZE)
    if not batch:
        return {"uploaded": 0, "failed": 0, "reason": "idle"}
    batch_sub = pairing.get("sub")
    id_token = await _community_id_token(refresh_token)
    if not id_token:
        return {"uploaded": 0, "failed": 0, "reason": "token_unavailable"}

    synced: list[str] = []
    failed: list[str] = []
    for entry in batch:
        sample_id = entry["sample_id"]
        # Hold the same mutation lock used by opt-out across the final consent
        # check and POST. Once disabling returns, no older upload can still be
        # in flight or begin from a stale batch snapshot.
        async with sparkdeck._community_upload_lock:
            if not store.get_setting("community_consent", False):
                break
            current_pairing = store.get_setting(
                "device_pairing", {"status": "not_paired"})
            if (
                not isinstance(current_pairing, dict)
                or current_pairing.get("status") != "paired"
                or current_pairing.get("sub") != batch_sub
                or current_pairing.get("refresh_token") != refresh_token
            ):
                break
            # The batch is only a scheduling snapshot; re-read the row at the
            # outbound boundary so a deleted or transitioned sample never sends.
            current = store.outbox_entry(sample_id)
            if current is None:
                continue
            response = await _post_community_sample(
                api_url, id_token, current["payload"], sample_id)
            if response is not None and response.status_code == 401:
                # The cached token expired early; mint once and retry this sample.
                id_token = await _community_id_token(refresh_token, force=True)
                if not id_token:
                    break
                response = await _post_community_sample(
                    api_url, id_token, current["payload"], sample_id)
        if response is None:
            continue  # transient transport error: leave pending
        if response.is_success:
            synced.append(sample_id)
        elif response.status_code == 401:
            break  # still unauthorized after one refresh: stop the batch
        elif response.status_code == 429:
            retry_after = response.headers.get("retry-after")
            delay = (
                float(retry_after)
                if retry_after and retry_after.isdigit()
                else float(COMMUNITY_UPLOAD_INTERVAL_SECONDS)
            )
            _community_upload_not_before = time.time() + delay
            break  # rate limited: everything stays pending until the next tick
        elif 400 <= response.status_code < 500:
            # Definitive rejection (e.g. schema validation): retryable via the
            # existing retry button, not hot-looped.
            failed.append(sample_id)
        # 5xx: leave pending for the next tick.
        await asyncio.sleep(_COMMUNITY_UPLOAD_PACING_SECONDS)
    store.mark_outbox_synced(synced)
    if failed:
        store.mark_outbox_failed(failed, "community API rejected the sample")
    return {"uploaded": len(synced), "failed": len(failed)}


async def community_upload_loop() -> None:
    while True:
        try:
            await community_upload_once()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("community sample upload failed")
        await asyncio.sleep(COMMUNITY_UPLOAD_INTERVAL_SECONDS)


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


_SHUTDOWN_REQUEST_PATTERN = re.compile(r"[1-9][0-9]*")


def _shutdown_request_process_ids() -> frozenset[int]:
    """Return process IDs that identify this server invocation.

    On Windows, a virtualenv ``python.exe`` redirector remains as the process
    recorded by ``Start-Process`` while the base interpreter runs this module
    as its child. Accept that redirector PID only for this specific venv case;
    other platforms and non-venv launches accept the server PID alone.
    """
    process_ids = {os.getpid()}
    if os.name == "nt" and sys.prefix != getattr(sys, "base_prefix", sys.prefix):
        parent_pid = os.getppid()
        if parent_pid > 0:
            process_ids.add(parent_pid)
    return frozenset(process_ids)


def _discard_shutdown_request(shutdown_file: Path) -> None:
    """Remove a request left behind by an earlier server invocation."""
    try:
        shutdown_file.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        logging.getLogger(__name__).warning(
            "Could not remove stale shutdown request %s: %s",
            shutdown_file,
            exc,
        )


def _consume_shutdown_request(
    shutdown_file: Path,
    process_ids: frozenset[int],
) -> bool:
    """Consume and validate a launcher request for this server invocation."""
    try:
        requested_pid = shutdown_file.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return False
    except OSError:
        # The launcher may still be replacing/writing the file. Retry on the
        # next watcher tick rather than losing a valid stop request.
        return False

    valid = bool(_SHUTDOWN_REQUEST_PATTERN.fullmatch(requested_pid))
    matches = valid and int(requested_pid) in process_ids
    _discard_shutdown_request(shutdown_file)
    return matches


async def _serve_application() -> None:
    """Serve one application instance and honor the Windows launcher stop file."""
    import uvicorn

    shutdown_file = ROOT / "data" / "run" / "shutdown.request"
    # A prior launcher/machine interruption can leave this persistent marker
    # behind. Clear it before serving; any subsequent request must name this
    # invocation so a foreground or Linux start cannot consume stale state.
    _discard_shutdown_request(shutdown_file)
    shutdown_process_ids = _shutdown_request_process_ids()
    instance = uvicorn.Server(uvicorn.Config(
        # Passing the application object avoids importing this module a second
        # time and constructing duplicate managers/background resources.
        app,
        host="0.0.0.0",
        port=7878,
        log_level="info",
        timeout_graceful_shutdown=10,
    ))

    async def watch_launcher_shutdown() -> None:
        while not instance.should_exit:
            if _consume_shutdown_request(shutdown_file, shutdown_process_ids):
                instance.should_exit = True
                return
            await asyncio.sleep(0.25)

    watcher = asyncio.create_task(watch_launcher_shutdown())
    try:
        await instance.serve()
    finally:
        watcher.cancel()
        try:
            await watcher
        except asyncio.CancelledError:
            pass
        shutdown_file.unlink(missing_ok=True)


if __name__ == "__main__":
    asyncio.run(_serve_application())
