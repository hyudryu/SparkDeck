"""Runtime adapters for the three inference servers supported by SparkDeck."""

from __future__ import annotations

import asyncio
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

from .models import RuntimeKind


SPARKDECK_LABEL = "io.sparkdeck.managed"
SPARKDECK_MODEL_LABEL = "io.sparkdeck.model"
SPARKDECK_RUNTIME_LABEL = "io.sparkdeck.runtime"
SPARKDECK_DEPLOYMENT_LABEL = "io.sparkdeck.deployment"
_GGUF_SHARD_PATTERN = re.compile(
    r"^(?P<stem>.+)-(?P<index>\d{5})-of-(?P<count>\d{5})\.gguf$",
    re.IGNORECASE,
)


def normalize_openai_base_url(base_url: str) -> str:
    """Return an endpoint root so callers can append exactly one ``/v1``."""
    parsed = urlsplit(str(base_url or "").strip().rstrip("/"))
    path = parsed.path.rstrip("/")
    if path.rsplit("/", 1)[-1].casefold() == "v1":
        path = path.rsplit("/", 1)[0]
    return urlunsplit((parsed.scheme, parsed.netloc, path, parsed.query, parsed.fragment)).rstrip("/")


@dataclass(slots=True)
class LaunchSpec:
    image: str
    command: list[str]
    internal_port: int
    entrypoint: list[str] | None = None
    volumes: dict[str, dict[str, str]] | None = None
    environment: dict[str, str] | None = None


class RuntimeAdapter(ABC):
    kind: RuntimeKind
    default_image: str

    @abstractmethod
    def launch_spec(self, model: str, settings: dict[str, Any]) -> LaunchSpec:
        """Translate typed launch settings into an upstream server command."""

    async def health(self, http: httpx.AsyncClient, base_url: str,
                     api_key: str | None = None) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        root = normalize_openai_base_url(base_url)
        response = await http.get(f"{root}/v1/models", headers=headers, timeout=5)
        response.raise_for_status()
        data = response.json()
        return {"reachable": True, "models": data.get("data", [])}


class VllmAdapter(RuntimeAdapter):
    kind = RuntimeKind.VLLM
    default_image = "vllm/vllm-openai:latest"

    def launch_spec(self, model: str, settings: dict[str, Any]) -> LaunchSpec:
        command = ["vllm", "serve", model, "--host", "0.0.0.0", "--port", "8000"]
        if settings.get("gpu_memory_utilization") is not None:
            command += ["--gpu-memory-utilization", str(settings["gpu_memory_utilization"])]
        if settings.get("tensor_parallel_size"):
            command += ["--tensor-parallel-size", str(settings["tensor_parallel_size"])]
        if settings.get("pipeline_parallel_size"):
            command += ["--pipeline-parallel-size", str(settings["pipeline_parallel_size"])]
        max_model_len = settings.get("max_model_len") or settings.get("context_length")
        if max_model_len:
            command += ["--max-model-len", str(max_model_len)]
        if settings.get("quantization"):
            command += ["--quantization", str(settings["quantization"])]
        if settings.get("revision"):
            command += ["--revision", str(settings["revision"])]
        command.extend(str(item) for item in settings.get("extra_args", []))
        return LaunchSpec(settings.get("image") or self.default_image, command, 8000, entrypoint=[])


class LlamaCppAdapter(RuntimeAdapter):
    kind = RuntimeKind.LLAMA_CPP
    default_image = "ghcr.io/ggml-org/llama.cpp:server-cuda"

    def launch_spec(self, model: str, settings: dict[str, Any]) -> LaunchSpec:
        artifact = str(settings.get("artifact") or model).strip()
        artifact_path = Path(artifact).expanduser()
        if artifact_path.suffix.casefold() != ".gguf" or not artifact_path.is_file():
            raise ValueError(
                "llama.cpp managed deployments require an existing local GGUF artifact"
            )
        artifact_path = artifact_path.resolve()
        shard = _GGUF_SHARD_PATTERN.match(artifact_path.name)
        if shard:
            shard_count = int(shard.group("count"))
            missing = [
                artifact_path.with_name(
                    f"{shard.group('stem')}-{index:05d}-of-{shard_count:05d}.gguf"
                )
                for index in range(1, shard_count + 1)
                if not artifact_path.with_name(
                    f"{shard.group('stem')}-{index:05d}-of-{shard_count:05d}.gguf"
                ).is_file()
            ]
            if missing:
                raise ValueError(
                    "llama.cpp managed deployment requires every GGUF shard"
                )
            artifact_path = artifact_path.with_name(
                f"{shard.group('stem')}-00001-of-{shard_count:05d}.gguf"
            )
            volumes = {
                str(artifact_path.parent): {"bind": "/models", "mode": "ro"}
            }
            artifact = f"/models/{artifact_path.name}"
        else:
            volumes = {
                str(artifact_path): {"bind": "/models/model.gguf", "mode": "ro"}
            }
            artifact = "/models/model.gguf"
        command = ["--host", "0.0.0.0", "--port", "8080", "--model", artifact]
        context_size = settings.get("context_size") or settings.get("context_length")
        if context_size:
            command += ["--ctx-size", str(context_size)]
        parallel = settings.get("parallel") or settings.get("parallel_slots")
        if parallel:
            command += ["--parallel", str(parallel)]
        if settings.get("gpu_layers") is not None:
            command += ["--n-gpu-layers", str(settings["gpu_layers"])]
        if settings.get("split_mode"):
            command += ["--split-mode", str(settings["split_mode"])]
        if settings.get("tensor_split"):
            command += ["--tensor-split", str(settings["tensor_split"])]
        command.extend(str(item) for item in settings.get("extra_args", []))
        return LaunchSpec(settings.get("image") or self.default_image, command, 8080, volumes=volumes)


class SglangAdapter(RuntimeAdapter):
    kind = RuntimeKind.SGLANG
    default_image = "lmsysorg/sglang:latest"

    def launch_spec(self, model: str, settings: dict[str, Any]) -> LaunchSpec:
        command = ["-m", "sglang.launch_server", "--model-path", model,
                   "--host", "0.0.0.0", "--port", "8000"]
        if settings.get("tensor_parallel_size"):
            command += ["--tp-size", str(settings["tensor_parallel_size"])]
        if settings.get("data_parallel_size"):
            command += ["--dp-size", str(settings["data_parallel_size"])]
        if settings.get("context_length"):
            command += ["--context-length", str(settings["context_length"])]
        if settings.get("quantization"):
            command += ["--quantization", str(settings["quantization"])]
        if settings.get("revision"):
            command += ["--revision", str(settings["revision"])]
        if settings.get("mem_fraction_static"):
            command += ["--mem-fraction-static", str(settings["mem_fraction_static"])]
        command.extend(str(item) for item in settings.get("extra_args", []))
        return LaunchSpec(settings.get("image") or self.default_image, command, 8000,
                          entrypoint=["python3"])


class RuntimeRegistry:
    def __init__(self):
        adapters = (VllmAdapter(), LlamaCppAdapter(), SglangAdapter())
        self._adapters = {adapter.kind: adapter for adapter in adapters}

    @property
    def kinds(self) -> tuple[str, ...]:
        return tuple(kind.value for kind in self._adapters)

    def get(self, runtime: str | RuntimeKind) -> RuntimeAdapter:
        try:
            key = runtime if isinstance(runtime, RuntimeKind) else RuntimeKind(runtime)
        except ValueError as exc:
            raise ValueError(f"unsupported runtime '{runtime}'") from exc
        return self._adapters[key]


def safe_container_name(alias: str, deployment_id: str) -> str:
    safe = re.sub(r"[^a-z0-9_.-]+", "-", alias.lower()).strip("-.") or "model"
    return f"sparkdeck-{safe[:42]}-{deployment_id[:8]}"


async def launch_managed_container(manager: Any, adapter: RuntimeAdapter,
                                   deployment_id: str, alias: str, model: str,
                                   settings: dict[str, Any]) -> dict[str, Any]:
    """Launch a managed runtime while retaining Manager's mature vLLM/SGLang paths."""
    if adapter.kind in (RuntimeKind.VLLM, RuntimeKind.SGLANG):
        if adapter.kind is RuntimeKind.VLLM:
            extra: list[str] = []
            for key, flag in (
                ("tensor_parallel_size", "--tensor-parallel-size"),
                ("pipeline_parallel_size", "--pipeline-parallel-size"),
                ("context_length", "--max-model-len"),
                ("quantization", "--quantization"),
                ("revision", "--revision"),
            ):
                if settings.get(key) is not None:
                    extra += [flag, str(settings[key])]
            extra.extend(str(item) for item in settings.get("extra_args", []))
            return await manager.create_container(
                model=model, engine="vllm", image=settings.get("image"),
                gpu_memory_utilization=settings.get("gpu_memory_utilization"),
                extra_args=extra,
                name=safe_container_name(alias, deployment_id),
                sparkdeck_deployment_id=deployment_id,
            )
        extra = []
        if settings.get("data_parallel_size") is not None:
            extra += ["--dp-size", str(settings["data_parallel_size"])]
        if settings.get("quantization") is not None:
            extra += ["--quantization", str(settings["quantization"])]
        if settings.get("revision") is not None:
            extra += ["--revision", str(settings["revision"])]
        extra.extend(str(item) for item in settings.get("extra_args", []))
        return await manager.create_container(
            model=model, engine="sglang", sg_image=settings.get("image"),
            sg_tp_size=settings.get("tensor_parallel_size"),
            sg_context_length=settings.get("context_length"),
            sg_max_running_requests=settings.get("max_running_requests"),
            sg_mem_fraction=settings.get("mem_fraction_static"),
            extra_args=extra,
            name=safe_container_name(alias, deployment_id),
            sparkdeck_deployment_id=deployment_id,
        )

    spec = adapter.launch_spec(model, settings)
    await manager.evict_other_backends(protect=adapter.kind.value)
    port = int(settings.get("port") or await manager._allocate_port())
    name = safe_container_name(alias, deployment_id)

    def run() -> Any:
        try:
            manager.client.images.get(spec.image)
        except Exception as exc:
            # Only pull for the normal missing-image case; propagate daemon/auth errors.
            if exc.__class__.__name__ != "ImageNotFound":
                raise
            manager.client.images.pull(spec.image)
        options: dict[str, Any] = {
            "image": spec.image,
            "command": spec.command,
            "name": name,
            "detach": True,
            "ports": {f"{spec.internal_port}/tcp": port},
            "ipc_mode": "host",
            "shm_size": manager.settings.get("shm_size", "16g"),
            "labels": {
                SPARKDECK_LABEL: "1", SPARKDECK_MODEL_LABEL: model,
                SPARKDECK_RUNTIME_LABEL: adapter.kind.value,
                SPARKDECK_DEPLOYMENT_LABEL: deployment_id,
            },
            "restart_policy": {"Name": "unless-stopped"},
        }
        if spec.entrypoint is not None:
            options["entrypoint"] = spec.entrypoint
        if spec.volumes:
            options["volumes"] = spec.volumes
        if spec.environment:
            options["environment"] = spec.environment
        # Match the existing launcher's GPU exposure without coupling adapters to docker-py.
        try:
            import docker
            options["device_requests"] = [
                docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
            ]
        except Exception:
            pass
        container = manager._run_managed_container(options)
        container.reload()
        summary = manager._container_summary(container) or {}
        return {**summary, "port": summary.get("port") or port,
                "runtime": adapter.kind.value, "deployment_id": deployment_id}

    return await asyncio.to_thread(run)
