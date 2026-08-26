"""Application service joining runtime adapters, persistence and proxy metrics."""

from __future__ import annotations

import json
import platform
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator
from urllib.parse import urlparse

import httpx

from .catalog import HuggingFaceCatalog
from .models import BenchmarkSample, Deployment, DeploymentKind, ModelIdentity, RuntimeKind
from .runtimes import RuntimeRegistry, launch_managed_container
from .storage import SparkDeckStore


_SAFE_CONFIGURATION_KEYS = {
    "context_size", "context_length", "max_model_len", "parallel",
    "gpu_layers", "split_mode", "tensor_split", "tensor_parallel_size",
    "pipeline_parallel_size", "data_parallel_size", "quantization", "dtype",
    "max_running_requests", "mem_fraction_static", "image", "runtime_version",
}


class SparkDeckService:
    def __init__(self, manager: Any, data_dir: Path):
        self.manager = manager
        self.store = SparkDeckStore(Path(data_dir) / "sparkdeck.sqlite3")
        self.registry = RuntimeRegistry()
        self.catalog = HuggingFaceCatalog(manager.http)

    async def close(self) -> None:
        self.store.close()

    async def catalog_search(self, query: str, limit: int) -> dict[str, Any]:
        items = await self.catalog.search(query, limit)
        local = {item["model"]["repository"]: item for item in await self.deployments()}
        for item in items:
            if item["id"] in local:
                item["local_deployment"] = local[item["id"]]
        return {"items": items, "next_cursor": None}

    async def deployments(self) -> list[dict[str, Any]]:
        registered = self.store.deployments()
        by_container = {item.get("container_name"): item for item in registered}
        try:
            containers = await self.manager.list_containers()
        except Exception:
            containers = []
        seen: set[str] = set()
        for container in containers:
            runtime = self._container_runtime(container)
            if runtime not in self.registry.kinds:
                continue
            stored = by_container.get(container.get("name"))
            if stored:
                stored["status"] = container.get("status", "unknown")
                stored["port"] = container.get("port")
                stored["managed"] = True
                seen.add(stored["id"])
                continue
            model = container.get("model") or container.get("served_model")
            if not model:
                continue
            generated_id = container.get("deployment_id") or f"container:{container.get('name')}"
            registered.append({
                "id": generated_id,
                "alias": container.get("alias") or container.get("served_model") or model,
                "runtime": runtime,
                "kind": "managed" if container.get("managed") else "external",
                "model": {"repository": model, "revision": None,
                          "artifact": None, "quantization": container.get("variant")},
                "status": container.get("status", "unknown"),
                "container_name": container.get("name"),
                "settings": self._safe_configuration(container.get("load_settings") or {}),
                "base_url_set": bool(container.get("port")),
                "port": container.get("port"),
                "managed": bool(container.get("managed")),
            })
        return registered

    async def create_deployment(self, body: dict[str, Any]) -> dict[str, Any]:
        model = str(body.get("model") or "").strip()
        alias = str(body.get("alias") or model).strip()
        if not model:
            raise ValueError("model is required")
        if not alias:
            raise ValueError("alias is required")
        runtime = RuntimeKind(str(body.get("runtime") or "vllm"))
        kind = DeploymentKind(str(body.get("kind") or ("external" if body.get("base_url") else "managed")))
        settings = dict(body.get("settings") or {})
        deployment_id = str(uuid.uuid4())
        identity = ModelIdentity(
            repository=model, revision=_optional_string(body.get("revision")),
            artifact=_optional_string(body.get("artifact") or settings.get("artifact")),
            quantization=_optional_string(body.get("quantization") or settings.get("quantization")),
        )
        deployment = Deployment(
            id=deployment_id, alias=alias, runtime=runtime, kind=kind,
            model=identity, settings=self._safe_configuration(settings),
            base_url_set=bool(body.get("base_url")),
        )
        # Force uniqueness before a potentially expensive container launch.
        if self.store.deployment(alias):
            raise ValueError(f"deployment alias '{alias}' is already in use")
        if kind is DeploymentKind.EXTERNAL:
            base_url = self._validate_base_url(body.get("base_url"))
            credential_ref = self._store_credential(deployment_id, body.get("api_key"))
            self.store.add_deployment(deployment, base_url, credential_ref)
            return (self.store.deployment(deployment_id) or deployment.to_dict())

        adapter = self.registry.get(runtime)
        launched = await launch_managed_container(
            self.manager, adapter, deployment_id, alias, model,
            {**settings, "artifact": identity.artifact},
        )
        deployment.container_name = launched.get("name")
        port = launched.get("port")
        if not deployment.container_name or not port:
            raise RuntimeError("runtime launched without a discoverable container endpoint")
        self.store.add_deployment(
            deployment, f"http://127.0.0.1:{int(port)}", None
        )
        result = self.store.deployment(deployment_id) or deployment.to_dict()
        result.update({"status": launched.get("status", "running"), "port": int(port)})
        return result

    async def deployment_action(self, deployment_id: str, action: str) -> dict[str, Any]:
        deployment = self.store.deployment(deployment_id, include_private=True)
        if not deployment:
            raise LookupError("deployment not found")
        if deployment["kind"] != DeploymentKind.MANAGED.value:
            raise ValueError("external endpoints cannot be started or stopped by SparkDeck")
        container = deployment.get("container_name")
        if not container:
            raise LookupError("managed container not found")
        if action == "start":
            result = await self.manager.start_container(container)
        elif action == "stop":
            result = await self.manager.stop_container(container)
        else:
            raise ValueError("action must be start or stop")
        return {"ok": True, "deployment_id": deployment_id, "action": action, "result": result}

    async def delete_deployment(self, deployment_id: str) -> dict[str, Any]:
        deployment = self.store.deployment(deployment_id, include_private=True)
        if not deployment:
            raise LookupError("deployment not found")
        if deployment["kind"] == DeploymentKind.MANAGED.value and deployment.get("container_name"):
            await self.manager.remove_container(deployment["container_name"])
        self._delete_credential(deployment_id, deployment.get("_credential_ref"))
        self.store.delete_deployment(deployment_id)
        return {"ok": True, "id": deployment_id}

    async def models(self) -> dict[str, Any]:
        data = []
        seen = set()
        for deployment in await self.deployments():
            if deployment.get("status") not in ("running", "registered"):
                continue
            alias = deployment["alias"]
            if alias in seen:
                continue
            seen.add(alias)
            data.append({
                "id": alias, "object": "model", "created": 0,
                "owned_by": "sparkdeck", "runtime": deployment["runtime"],
                "deployment_id": deployment["id"], "model": deployment["model"],
            })
        return {"object": "list", "data": data}

    async def proxy(self, body: dict[str, Any], endpoint: str,
                    cancel: Any = None) -> dict[str, Any] | AsyncIterator[str]:
        requested_model = str(body.get("model") or "")
        deployment = self.store.deployment(requested_model, include_private=True)
        if deployment:
            return await self._proxy_registered(deployment, body, endpoint, cancel)
        # Compatibility for existing vLLM/SGLang containers. The manager retains
        # admission control and cancellation for these established deployments.
        started = time.monotonic()
        stream = bool(body.get("stream"))
        result = (
            await self.manager._vllm_chat(requested_model, body, stream, cancel)
            if endpoint == "chat/completions"
            else await self.manager._vllm_completions(requested_model, body, stream, cancel)
        )
        runtime = await self._runtime_for_legacy_model(requested_model)
        if hasattr(result, "__aiter__"):
            return self._observe_stream(result, None, requested_model, runtime, {}, started)
        self._record_response(None, requested_model, runtime, {}, started, result)
        return result

    async def _proxy_registered(self, deployment: dict[str, Any], body: dict[str, Any],
                                endpoint: str, cancel: Any) -> dict[str, Any] | AsyncIterator[str]:
        base_url = deployment.get("_base_url")
        if not base_url:
            raise LookupError("deployment endpoint is unavailable")
        api_key = self._get_credential(deployment["id"], deployment.get("_credential_ref"))
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        upstream_body = dict(body)
        upstream_body["model"] = deployment["model"]["repository"]
        started = time.monotonic()
        if upstream_body.get("stream"):
            upstream_body["stream_options"] = {
                **(upstream_body.get("stream_options") or {}), "include_usage": True,
            }
            return self._http_stream(
                f"{base_url.rstrip('/')}/v1/{endpoint}", upstream_body, headers,
                deployment, started, cancel,
            )
        response = await self.manager.http.post(
            f"{base_url.rstrip('/')}/v1/{endpoint}", json=upstream_body,
            headers=headers, timeout=600,
        )
        response.raise_for_status()
        data = response.json()
        self._record_response(
            deployment["id"], deployment["model"]["repository"],
            deployment["runtime"], deployment.get("settings") or {}, started, data,
        )
        data["model"] = deployment["alias"]
        return data

    async def _http_stream(self, url: str, body: dict[str, Any], headers: dict[str, str],
                           deployment: dict[str, Any], started: float,
                           cancel: Any) -> AsyncIterator[str]:
        first_token_at: float | None = None
        usage: dict[str, Any] | None = None
        async with self.manager.http.stream(
            "POST", url, json=body, headers=headers, timeout=None
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if cancel is not None and cancel.is_set():
                    return
                if not line:
                    continue
                parsed = _parse_sse(line)
                if parsed:
                    if parsed.get("usage"):
                        usage = parsed["usage"]
                    if first_token_at is None and _chunk_has_output(parsed):
                        first_token_at = time.monotonic()
                    if parsed.get("model"):
                        parsed["model"] = deployment["alias"]
                        line = "data: " + json.dumps(parsed, separators=(",", ":"))
                yield f"{line}\n\n"
        if usage:
            self._record_usage(
                deployment["id"], deployment["model"]["repository"],
                deployment["runtime"], deployment.get("settings") or {}, started,
                usage, first_token_at,
            )

    async def _observe_stream(self, stream: AsyncIterator[str], deployment_id: str | None,
                              model: str, runtime: str, settings: dict[str, Any],
                              started: float) -> AsyncIterator[str]:
        first_token_at = None
        usage = None
        async for chunk in stream:
            text = chunk.decode("utf-8", errors="replace") if isinstance(chunk, bytes) else str(chunk)
            for line in text.splitlines():
                parsed = _parse_sse(line)
                if parsed and parsed.get("usage"):
                    usage = parsed["usage"]
                if parsed and first_token_at is None and _chunk_has_output(parsed):
                    first_token_at = time.monotonic()
            yield chunk
        if usage:
            self._record_usage(deployment_id, model, runtime, settings, started, usage, first_token_at)

    def _record_response(self, deployment_id: str | None, model: str, runtime: str,
                         settings: dict[str, Any], started: float, data: dict[str, Any]) -> None:
        usage = data.get("usage") if isinstance(data, dict) else None
        if isinstance(usage, dict):
            self._record_usage(deployment_id, model, runtime, settings, started, usage, None)

    def _record_usage(self, deployment_id: str | None, model: str, runtime: str,
                      settings: dict[str, Any], started: float, usage: dict[str, Any],
                      first_token_at: float | None) -> None:
        completed = time.monotonic()
        input_tokens = max(0, int(usage.get("prompt_tokens") or 0))
        output_tokens = max(0, int(usage.get("completion_tokens") or 0))
        latency = max(0.000001, completed - started)
        generation_seconds = max(0.000001, completed - (first_token_at or started))
        runtime_kind = RuntimeKind(runtime)
        safe_settings = self._safe_configuration(settings)
        quantization = _optional_string(safe_settings.get("quantization"))
        public_model = _public_model_id(model)
        eligible = bool(
            public_model != "local-model" and input_tokens > 0 and output_tokens >= 16 and latency > 0
            and runtime_kind.value in self.registry.kinds
        )
        sample = BenchmarkSample(
            id=str(uuid.uuid4()), created_at=datetime.now(timezone.utc).isoformat(),
            deployment_id=deployment_id,
            model=ModelIdentity(repository=public_model, quantization=quantization),
            runtime=runtime_kind,
            runtime_version=_optional_string(
                safe_settings.get("runtime_version") or safe_settings.get("image")
            ),
            hardware=self._hardware_snapshot(),
            configuration=safe_settings, input_tokens=input_tokens,
            output_tokens=output_tokens, latency_ms=round(latency * 1000, 3),
            ttft_ms=None if first_token_at is None else round((first_token_at - started) * 1000, 3),
            generation_tokens_per_second=(round(output_tokens / generation_seconds, 3)
                                          if output_tokens else None),
            prompt_tokens_per_second=(round(input_tokens / max(0.000001, (first_token_at or completed) - started), 3)
                                      if input_tokens else None),
            cold_start=None, eligible_for_community=eligible,
        )
        consent = bool(self.store.get_setting("community_consent", False))
        self.store.add_benchmark(sample, queue=eligible and consent)

    async def _runtime_for_legacy_model(self, model: str) -> str:
        try:
            for container in await self.manager.list_containers():
                ids = self.manager._container_model_ids(container)
                if model in ids:
                    return self._container_runtime(container)
        except Exception:
            pass
        return RuntimeKind.VLLM.value

    @staticmethod
    def _container_runtime(container: dict[str, Any]) -> str:
        runtime = str(container.get("runtime") or container.get("engine") or "vllm")
        return "llama.cpp" if runtime in ("llama", "llama-server", "llama_cpp") else runtime

    @staticmethod
    def _safe_configuration(settings: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in settings.items() if key in _SAFE_CONFIGURATION_KEYS}

    def _hardware_snapshot(self) -> dict[str, Any]:
        stats = getattr(self.manager, "_stats_cache", {}) or {}
        gpus = stats.get("gpus") or []
        public_gpus = [
            {"model": gpu.get("name"), "memory_mib": gpu.get("mem_total_mib")}
            for gpu in gpus if isinstance(gpu, dict) and gpu.get("name")
        ]
        names = " ".join(str(gpu.get("model") or "") for gpu in public_gpus).casefold()
        return {
            "architecture": platform.machine(),
            "device_class": "dgx-spark" if "gb10" in names or "dgx spark" in names else "local",
            "gpu_count": len(public_gpus),
            "gpus": public_gpus,
        }

    @staticmethod
    def _validate_base_url(value: Any) -> str:
        text = str(value or "").strip().rstrip("/")
        parsed = urlparse(text)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("base_url must be an http or https URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("base_url must not contain credentials, query parameters, or fragments")
        return text

    @staticmethod
    def _store_credential(deployment_id: str, api_key: Any) -> str | None:
        if not api_key:
            return None
        try:
            import keyring
            keyring.set_password("SparkDeck", deployment_id, str(api_key))
        except Exception as exc:
            raise RuntimeError("OS credential storage is unavailable; the API key was not stored") from exc
        return f"keyring:{deployment_id}"

    @staticmethod
    def _get_credential(deployment_id: str, credential_ref: str | None) -> str | None:
        if not credential_ref:
            return None
        try:
            import keyring
            return keyring.get_password("SparkDeck", deployment_id)
        except Exception:
            return None

    @staticmethod
    def _delete_credential(deployment_id: str, credential_ref: str | None) -> None:
        if not credential_ref:
            return
        try:
            import keyring
            keyring.delete_password("SparkDeck", deployment_id)
        except Exception:
            pass


def _optional_string(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _public_model_id(value: str) -> str:
    """Exclude endpoint URLs and local paths from persistent benchmark identity."""
    text = str(value or "").strip()
    if not text or urlparse(text).scheme or Path(text).is_absolute() or text.startswith(("./", "../", "~")):
        return "local-model"
    return text


def _parse_sse(line: str) -> dict[str, Any] | None:
    if not line.startswith("data:"):
        return None
    payload = line[5:].strip()
    if not payload or payload == "[DONE]":
        return None
    try:
        value = json.loads(payload)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _chunk_has_output(chunk: dict[str, Any]) -> bool:
    for choice in chunk.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta") or {}
        if isinstance(delta, dict) and (delta.get("content") or delta.get("reasoning_content")):
            return True
        if choice.get("text"):
            return True
    return False
