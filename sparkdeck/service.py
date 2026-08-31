"""Application service joining runtime adapters, persistence and proxy metrics."""

from __future__ import annotations

import asyncio
import copy
import contextvars
import hashlib
import inspect
import ipaddress
import json
import math
import platform
import re
import shlex
import socket
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, AsyncIterator
from urllib.parse import urlparse

import httpx

from .catalog import (
    HuggingFaceCatalog,
    canonical_quantization,
    quantization_from_text,
)
from .models import BenchmarkSample, Deployment, DeploymentKind, ModelIdentity, RuntimeKind
from .runtime_environment import normalize_runtime_environment
from .runtimes import (
    RuntimeRegistry,
    launch_managed_container,
    normalize_openai_base_url,
    safe_container_name,
)
from .storage import (
    COMMUNITY_API_URL,
    COMMUNITY_EVIDENCE_POLICY,
    SparkDeckStore,
    community_context_window,
)
from .virtual_nas import LOCAL_NODE_ID, download_required_free_bytes


_SAFE_CONFIGURATION_KEYS = {
    "context_size", "context_window", "context_length", "max_model_len", "parallel", "parallel_slots",
    "gpu_layers", "split_mode", "tensor_split", "gpu_split", "tensor_parallel_size",
    "pipeline_parallel_size", "data_parallel_size", "quantization", "dtype",
    "max_concurrency", "max_running_requests", "mem_fraction_static", "gpu_memory_utilization",
    "runtime_version",
}
_LOCAL_ROUTING_KEYS = {
    "deployment_mode", "node_ids", "manager_deployment_id", "model_source",
    "managed_by", "automation_run_id", "source_container_name",
    # Saved deployments relaunch from the persisted record, so their extra
    # argv must survive persistence alongside the routing keys above.
    "extra_args",
    # Manager consumes these launch inputs at relaunch; dropping them would
    # silently substitute default ports, images, and memory policies.
    "port", "image", "sg_image", "gpu_memory_gb",
    "sg_tp_size", "sg_context_length", "sg_max_running_requests",
    "sg_mem_fraction",
    # The immutable revision weight preparation resolved, so the launch uses
    # exactly the prepared snapshot instead of re-resolving a mutable name.
    "prepared_revision",
    # Preferred Hub seed node for distributing a saved GGUF artifact.
    "download_node_id",
    # Structured launch controls Manager merges into argv at launch.
    "launch_controls",
    # Non-secret vLLM environment variables are local launch inputs and must
    # never be included in community benchmark configuration.
    "environment",
}
_COMMUNITY_REDIRECT_STATUSES = {301, 302, 303, 307, 308}
_COMMUNITY_MAX_REDIRECTS = 5
_COMMUNITY_MAX_RESPONSE_BYTES = 8 * 1024 * 1024
_COMMUNITY_SAMPLE_INTERVAL_SECONDS = 4 * 60 * 60
_COMMUNITY_SAMPLE_MAX_INPUT_TOKENS = 10_000
_COMMUNITY_SAMPLE_MIN_DECODE_SECONDS = 3.0
_PUBLIC_GGUF_SHARD_PATTERN = re.compile(
    r"^(?P<stem>.+)-(?P<index>\d{5})(?P<separator>-of-)"
    r"(?P<count>\d{5})(?P<suffix>\.gguf)$",
    re.IGNORECASE,
)


def _discovered_launch_controls(
    manager: Any, engine: str, settings: dict[str, Any], extra_args: list[str],
) -> dict[str, Any]:
    """Launch controls for a discovered container's parsed command.

    ``Manager._container_load_settings`` lifts the managed flags out of
    ``extra_args`` into dedicated keys, so overlay them onto the controls
    parsed from the remaining flags.
    """
    controls = manager._deployment_launch_controls({
        "engine": engine,
        "extra_args": extra_args,
        "environment": settings.get("environment") or {},
        **(
            {
                "sg_context_length": settings.get("context_window"),
                "sg_max_running_requests": settings.get("max_concurrency"),
            }
            if engine == "sglang" else {}
        ),
    })
    if engine != "sglang":
        controls["context_window"] = settings.get("context_window")
        controls["max_concurrency"] = settings.get("max_concurrency")
    controls["kv_cache_dtype"] = settings.get("kv_cache_dtype")
    controls["thinking_mode"] = settings.get("thinking_mode")
    return controls


async def _public_connection_urls(
    url: httpx.URL, resolver: Any = socket.getaddrinfo,
) -> tuple[tuple[httpx.URL, ...], str, str]:
    """Resolve and pin every public destination for one outbound request.

    Each returned URL contains a validated IP address, so the HTTP transport
    cannot perform a second, attacker-controlled DNS lookup after validation.
    The caller must use the returned host header and SNI hostname so HTTPS
    authentication still applies to the configured service name.
    """
    if url.scheme not in {"http", "https"} or not url.host:
        raise ValueError("community aggregate URL must use HTTP or HTTPS")
    if url.userinfo:
        raise ValueError("community aggregate URL must not include credentials")

    hostname = url.raw_host.decode("ascii")
    port = url.port or (443 if url.scheme == "https" else 80)
    try:
        literal = ipaddress.ip_address(hostname)
    except ValueError:
        literal = None

    if literal is not None:
        addresses = [literal]
    else:
        try:
            answers = await asyncio.to_thread(
                resolver, hostname, port, type=socket.SOCK_STREAM,
            )
        except (OSError, UnicodeError) as exc:
            raise ValueError("community aggregate host could not be resolved") from exc
        addresses = []
        for answer in answers:
            try:
                address = ipaddress.ip_address(answer[4][0])
            except (IndexError, TypeError, ValueError):
                raise ValueError("community aggregate host returned an invalid address")
            if address not in addresses:
                addresses.append(address)

    # Reject the entire DNS answer if any address is unsafe. This prevents a
    # resolver or attacker from influencing which member of a mixed answer the
    # transport chooses, and covers loopback, private, link-local, multicast,
    # unspecified, reserved, and shared address space.
    if not addresses or any(not address.is_global for address in addresses):
        raise ValueError("community aggregate host must resolve only to public addresses")

    return (
        tuple(url.copy_with(host=str(address)) for address in addresses),
        url.netloc.decode("ascii"),
        hostname,
    )


async def _read_bounded_community_response(response: httpx.Response) -> bytes:
    """Read decoded response bytes without permitting an aggregate memory bomb."""
    content_length = response.headers.get("content-length")
    if content_length is not None:
        try:
            declared_length = int(content_length)
        except ValueError as exc:
            raise ValueError("community aggregate response has invalid length") from exc
        if declared_length < 0:
            raise ValueError("community aggregate response has invalid length")
        if declared_length > _COMMUNITY_MAX_RESPONSE_BYTES:
            raise ValueError("community aggregate response is too large")

    content = bytearray()
    async for chunk in response.aiter_bytes():
        if len(content) + len(chunk) > _COMMUNITY_MAX_RESPONSE_BYTES:
            raise ValueError("community aggregate response is too large")
        content.extend(chunk)
    return bytes(content)


async def _get_public_community_url(
    url: str | httpx.URL, *, transport: httpx.AsyncBaseTransport | None = None,
    resolver: Any = socket.getaddrinfo,
) -> httpx.Response:
    """GET a public URL with per-hop DNS pinning and redirect validation."""
    try:
        current = httpx.URL(url)
    except (TypeError, httpx.InvalidURL) as exc:
        raise ValueError("community aggregate URL is invalid") from exc

    for redirect_count in range(_COMMUNITY_MAX_REDIRECTS + 1):
        pinned_urls, host_header, sni_hostname = await _public_connection_urls(
            current, resolver,
        )
        response: httpx.Response | None = None
        last_connect_error: httpx.HTTPError | None = None
        for pinned in pinned_urls:
            # A fresh client for every candidate prevents IP-keyed pooled TLS
            # connections from crossing logical hosts or redirect hops.
            async with httpx.AsyncClient(
                transport=transport,
                trust_env=False,
                follow_redirects=False,
                timeout=15,
            ) as client:
                request = client.build_request(
                    "GET",
                    pinned,
                    headers={"Host": host_header, "Connection": "close"},
                    extensions={"sni_hostname": sni_hostname},
                )
                try:
                    streamed = await client.send(request, stream=True)
                except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                    last_connect_error = exc
                    continue
                try:
                    if (
                        streamed.status_code not in _COMMUNITY_REDIRECT_STATUSES
                        or streamed.headers.get("location") is None
                    ):
                        content = await _read_bounded_community_response(streamed)
                    else:
                        content = b""
                    response = httpx.Response(
                        streamed.status_code,
                        headers=streamed.headers,
                        content=content,
                        request=streamed.request,
                        extensions=streamed.extensions,
                    )
                finally:
                    await streamed.aclose()
                break
        if response is None:
            if last_connect_error is None:
                raise RuntimeError("community aggregate service has no connectable endpoints")
            raise last_connect_error

        if (
            response.status_code in _COMMUNITY_REDIRECT_STATUSES
            and response.headers.get("location") is not None
        ):
            if redirect_count >= _COMMUNITY_MAX_REDIRECTS:
                raise httpx.TooManyRedirects(
                    "community aggregate service redirected too many times",
                    request=response.request,
                )
            try:
                current = current.join(response.headers["location"])
            except (TypeError, httpx.InvalidURL) as exc:
                raise ValueError("community aggregate redirect is invalid") from exc
            continue
        return response

    raise AssertionError("unreachable")


def _community_aggregate_url(endpoint: str) -> httpx.URL:
    """Append the aggregate route to the URL path, never its query string."""
    try:
        url = httpx.URL(endpoint)
    except (TypeError, httpx.InvalidURL) as exc:
        raise ValueError("community aggregate URL is invalid") from exc
    path = url.path.rstrip("/")
    if not path.endswith("/aggregates"):
        path = f"{path}/aggregates"
    return url.copy_with(path=path or "/aggregates", fragment=None)


def _public_community_aggregates(payload: Any) -> list[dict[str, Any]]:
    """Validate and bound the public response from a configured aggregator."""
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise ValueError("community aggregate response must contain an items array")
    if len(payload["items"]) > 10_000:
        raise ValueError("community aggregate response is too large")

    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    for raw in payload["items"]:
        if not isinstance(raw, dict):
            raise ValueError("community aggregate item must be an object")
        model_id = str(raw.get("model_id") or "").strip()
        quantization = canonical_quantization(raw.get("quantization")) or "UNKNOWN"
        prompt_bucket = raw.get("prompt_tokens_bucket")
        speed = raw.get("inference_tokens_per_second")
        sample_count = raw.get("sample_count")
        unique_cluster_count = raw.get("unique_cluster_count", 1)
        if (
            not model_id
            or len(model_id) > 500
            or any(ord(char) < 32 for char in model_id)
            or isinstance(prompt_bucket, bool)
            or not isinstance(prompt_bucket, int)
            or isinstance(sample_count, bool)
            or not isinstance(sample_count, int)
            or isinstance(unique_cluster_count, bool)
            or not isinstance(unique_cluster_count, int)
            or isinstance(speed, bool)
            or not isinstance(speed, (int, float))
        ):
            raise ValueError("community aggregate item is invalid")
        speed = float(speed)
        if (
            prompt_bucket not in {400, *range(1_000, 10_000, 1_000)}
            or sample_count <= 0
            or sample_count > 1_000_000_000
            or unique_cluster_count <= 0
            or unique_cluster_count > sample_count
            or not math.isfinite(speed)
            or speed <= 0
        ):
            raise ValueError("community aggregate item is invalid")
        key = (model_id, quantization, prompt_bucket)
        if key in seen:
            raise ValueError("community aggregate response contains duplicate evidence")
        seen.add(key)
        result.append({
            "model_id": model_id,
            "quantization": quantization,
            "prompt_tokens_bucket": prompt_bucket,
            "inference_tokens_per_second": speed,
            "sample_count": sample_count,
            "unique_cluster_count": unique_cluster_count,
        })
    return result


class SparkDeckService:
    def __init__(self, manager: Any, data_dir: Path):
        self.manager = manager
        self.store = SparkDeckStore(Path(data_dir) / "sparkdeck.sqlite3")
        self.registry = RuntimeRegistry()
        self.catalog = HuggingFaceCatalog(
            manager.http,
            token_provider=lambda: getattr(manager, "_resolved_hf_token", lambda: "")(),
        )
        self._deployment_create_lock = asyncio.Lock()
        self._deployment_reconciliation_lock = asyncio.Lock()
        self._deployment_launches: dict[str, asyncio.Event] = {}
        self._deployment_launch_tasks: dict[str, asyncio.Task] = {}
        self._deployment_action_locks: dict[str, asyncio.Lock] = {}
        # Serializes community state mutations (consent, unpair, deletion,
        # coordinated-benchmark insertion) into one critical section.
        self._community_upload_lock = asyncio.Lock()
        self._community_observation: contextvars.ContextVar[dict[str, Any] | None] = (
            contextvars.ContextVar("sparkdeck_community_observation", default=None)
        )
        self._community_active_observations: dict[str, dict[str, Any]] = {}

    async def close(self) -> None:
        tasks = list(self._deployment_launch_tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self.store.close()

    async def set_community_consent(
        self, enabled: bool, telemetry_cluster_id: str | None = None,
    ) -> dict[str, Any]:
        """Serialize consent changes with benchmark queue mutations."""
        async with self._community_upload_lock:
            return await asyncio.to_thread(
                self.store.set_community_consent, enabled, telemetry_cluster_id
            )

    async def revoke_community_membership(self) -> dict[str, Any]:
        """Disable sharing and forget a former controller's cluster identity."""
        async with self._community_upload_lock:
            return await asyncio.to_thread(
                self.store.revoke_community_membership
            )

    async def delete_benchmark(self, sample_id: str) -> bool:
        """Serialize deletion with queue mutations; the uploader re-reads the
        outbox before sending, so deleted samples cannot start later."""
        async with self._community_upload_lock:
            return await asyncio.to_thread(self.store.delete_benchmark, sample_id)

    async def delete_benchmark_model(self, model_id: str) -> int:
        """Delete a consolidated model history row and all its samples."""
        async with self._community_upload_lock:
            return await asyncio.to_thread(
                self.store.delete_benchmark_model, model_id
            )

    async def benchmark_history_models(self) -> list[dict[str, Any]]:
        """Read bounded, consolidated local history off the event loop."""
        return await asyncio.to_thread(self.store.benchmark_history_models)

    async def unpair_community_device(
        self, expected_sub: str,
    ) -> tuple[str, dict[str, Any]]:
        """Serialize account-matched unpairing with queue mutations."""
        async with self._community_upload_lock:
            with self.store.locked():
                existing = self.store.get_setting(
                    "device_pairing", {"status": "not_paired"},
                )
                if existing.get("status") != "paired":
                    return "already", existing
                if existing.get("sub") != expected_sub:
                    return "conflict", existing
                self.store.set_setting(
                    "device_pairing", {"status": "not_paired"},
                )
                return "unpaired", existing

    async def community_aggregates(self) -> dict[str, Any]:
        """Return configured community evidence or privacy-safe local evidence."""
        endpoint = COMMUNITY_API_URL.strip().rstrip("/")
        if not endpoint:
            items = await asyncio.to_thread(self.store.community_aggregates)
            items = await self.enrich_community_aggregates(items)
            return {
                "items": items,
                "availability": "local" if items else "not_configured",
                "evidence_policy": COMMUNITY_EVIDENCE_POLICY,
            }

        try:
            aggregate_url = _community_aggregate_url(endpoint)
            response = await _get_public_community_url(
                aggregate_url,
                transport=getattr(self.manager, "community_http_transport", None),
                resolver=getattr(
                    self.manager, "community_resolver", socket.getaddrinfo,
                ),
            )
            response.raise_for_status()
            items = _public_community_aggregates(response.json())
            items = await self.enrich_community_aggregates(items)
        except (httpx.HTTPError, TypeError, ValueError) as exc:
            raise RuntimeError("community aggregate service is unavailable") from exc
        return {
            "items": items,
            "availability": "available",
            "evidence_policy": COMMUNITY_EVIDENCE_POLICY,
        }

    async def enrich_community_aggregates(
        self, items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Attach authoritative Hub size metadata without trusting uploads."""
        repositories = list(dict.fromkeys(
            str(item.get("model_id") or "") for item in items
            if str(item.get("model_id") or "").count("/") == 1
        ))[:100]
        semaphore = asyncio.Semaphore(8)

        async def load(repository: str) -> tuple[str, dict[str, Any] | None]:
            async with semaphore:
                try:
                    return repository, await self.catalog.details(repository)
                except (httpx.HTTPError, TypeError, ValueError):
                    return repository, None

        tasks = [asyncio.create_task(load(value)) for value in repositories]
        metadata: dict[str, dict[str, Any] | None] = {}
        if tasks:
            done, pending = await asyncio.wait(tasks, timeout=8)
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            metadata = dict(
                task.result() for task in done if not task.cancelled()
            )
        enriched: list[dict[str, Any]] = []
        for item in items:
            value = dict(item)
            detail = metadata.get(str(item.get("model_id") or ""))
            if detail:
                value["parameter_count"] = detail.get("parameter_count")
                value["weight_size_bytes"] = detail.get("weight_size_bytes")
                quantization = (
                    canonical_quantization(item.get("quantization")) or "UNKNOWN"
                )
                variant = next((
                    candidate for candidate in detail.get("quantizations") or []
                    if canonical_quantization(candidate.get("name")) == quantization
                ), None)
                if variant and variant.get("weight_size_bytes"):
                    value["weight_size_bytes"] = variant["weight_size_bytes"]
            enriched.append(value)
        return enriched

    async def record_benchmark_series_point(self, body: dict[str, Any]) -> dict[str, Any]:
        """Record aggregate benchmark counters; raw prompts and outputs are rejected upstream."""
        deployment_id = str(body.get("deployment_id") or "").strip()
        if not deployment_id:
            raise ValueError("deployment_id is required")
        deployment = await self._benchmark_deployment(deployment_id)
        settings = deployment.get("settings") or {}
        context_window = community_context_window(settings)
        if context_window is None:
            raise ValueError("deployment does not declare a context window")

        concurrency = _bounded_benchmark_integer(body.get("concurrency"), "concurrency")
        if concurrency not in {1, 2, 5, 10}:
            raise ValueError("concurrency must be one of 1, 2, 5, or 10")
        request_count = _bounded_benchmark_integer(body.get("request_count"), "request_count")
        if request_count < concurrency:
            raise ValueError("request_count must be at least the measured concurrency")
        if request_count % concurrency:
            raise ValueError("request_count must be divisible by the measured concurrency")
        prompt_tokens = _bounded_benchmark_integer(body.get("prompt_tokens"), "prompt_tokens", allow_zero=True)
        generation_tokens = _bounded_benchmark_integer(body.get("generation_tokens"), "generation_tokens", allow_zero=True)
        prompt_seconds = _positive_finite(body.get("prompt_seconds"), "prompt_seconds")
        wall_seconds = _positive_finite(body.get("wall_seconds"), "wall_seconds")
        if prompt_tokens == 0:
            raise ValueError("prompt throughput is unavailable without measured prompt tokens")

        tensor_parallel_size = 1
        raw_tp = settings.get("tensor_parallel_size")
        if raw_tp is not None:
            tensor_parallel_size = _bounded_benchmark_integer(raw_tp, "tensor_parallel_size")
        try:
            prompt_tokens_per_second = prompt_tokens / prompt_seconds
            generation_tokens_per_second = generation_tokens / wall_seconds
        except OverflowError as exc:
            raise ValueError("derived benchmark throughput is outside the supported range") from exc
        if not (
            math.isfinite(prompt_tokens_per_second)
            and math.isfinite(generation_tokens_per_second)
        ):
            raise ValueError("derived benchmark throughput is outside the supported range")
        raw_model_id = str(
            (deployment.get("model") or {}).get("repository")
            or deployment.get("alias") or deployment.get("id") or deployment_id
        ).strip()
        upload_model_id = _public_model_id(raw_model_id)
        if (
            upload_model_id != "local-model"
            and deployment.get("kind") == DeploymentKind.MANAGED.value
            and settings.get("model_source") != "public_repository"
        ):
            upload_model_id = "local-model"
        local_model_id = _local_benchmark_model_id(
            raw_model_id, deployment.get("id") or deployment_id,
            upload_model_id=upload_model_id,
        )
        deployment_model = deployment.get("model") or {}
        quantization = (
            canonical_quantization(deployment_model.get("quantization"))
            or canonical_quantization(settings.get("quantization"))
            or quantization_from_text(
                settings.get("gguf_variant"),
                deployment_model.get("artifact"),
                raw_model_id,
            )
            or "UNKNOWN"
        )
        point = {
            "id": str(uuid.uuid4()),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "deployment_id": deployment.get("id"),
            "model_id": local_model_id,
            "context_window_size": context_window,
            "concurrency": concurrency,
            "tensor_parallel_size": tensor_parallel_size,
            "prompt_tokens_per_second": prompt_tokens_per_second,
            "generation_tokens_per_second": generation_tokens_per_second,
            "request_count": request_count,
        }
        runtime = RuntimeKind(str(deployment.get("runtime") or RuntimeKind.VLLM.value))
        configuration = self._safe_configuration({
            **settings,
            "context_length": context_window,
            "tensor_parallel_size": tensor_parallel_size,
        })
        # This marker is trusted run metadata, not a deployment setting. Keep
        # it out of the general configuration allowlist so ordinary proxy
        # samples cannot impersonate coordinated runs.
        configuration["benchmark_concurrency"] = concurrency
        hardware, hardware_verified = await self._managed_hardware_snapshot(deployment)
        eligible_for_community = bool(
            upload_model_id != "local-model"
            and prompt_tokens > 0
            and generation_tokens >= 16
            and runtime.value in self.registry.kinds
            and hardware_verified
        )
        sample = BenchmarkSample(
            id=str(uuid.uuid4()), created_at=point["created_at"],
            deployment_id=deployment.get("id"),
            model=ModelIdentity(
                # Preserve the resolved identity for local history. Upload
                # eligibility still uses the separately sanitized model ID.
                repository=local_model_id, quantization=quantization,
            ), runtime=runtime,
            runtime_version=_optional_string(settings.get("runtime_version")),
            hardware=hardware, configuration=configuration,
            input_tokens=prompt_tokens, output_tokens=generation_tokens,
            latency_ms=wall_seconds * 1000, ttft_ms=None,
            generation_tokens_per_second=point["generation_tokens_per_second"],
            prompt_tokens_per_second=point["prompt_tokens_per_second"],
            cold_start=False, eligible_for_community=eligible_for_community,
        )
        # Keep the consent decision and outbox insertion in the same critical
        # section as opt-in/withdrawal and outbound uploads. Otherwise a
        # consent transition can scan or clear the outbox between these two
        # operations and leave sharing state inconsistent.
        async with self._community_upload_lock:
            consent = bool(await asyncio.to_thread(
                self.store.get_setting, "community_consent", False,
            ))
            await asyncio.to_thread(
                self.store.add_coordinated_benchmark, point, sample,
                sample.eligible_for_community and consent,
            )
        return point

    async def _benchmark_deployment(self, deployment_id: str) -> dict[str, Any]:
        """Resolve both SparkDeck records and Manager-only MCP deployments."""
        registered = self.store.deployments(include_private=True)
        deployment = next((item for item in registered if (
            item.get("id") == deployment_id
            or (item.get("settings") or {}).get("manager_deployment_id") == deployment_id
        )), None)
        manager_id = (
            (deployment.get("settings") or {}).get("manager_deployment_id")
            if deployment else deployment_id
        )
        if deployment is not None and not manager_id:
            return deployment

        lookup_manager_deployment = getattr(self.manager, "_deployment", None)
        manager_deployment = (
            lookup_manager_deployment(manager_id)
            if callable(lookup_manager_deployment) else None
        )
        if manager_deployment is None and deployment is not None:
            manager_deployment = next((item for item in getattr(self.manager, "deployments", []) if (
                isinstance(item, dict)
                and item.get("sparkdeck_record_id") == deployment.get("id")
            )), None)
        if manager_deployment is not None and not manager_deployment.get("launch_settings"):
            try:
                state = await self.manager.get_state()
                manager_deployment = next((item for item in state.get("deployments", []) if (
                    isinstance(item, dict)
                    and item.get("id") == manager_deployment.get("id")
                )), manager_deployment)
            except Exception:
                pass
        if manager_deployment is None:
            if deployment is not None:
                return deployment
            raise LookupError("deployment not found")

        launch_settings = manager_deployment.get("launch_settings") or {}
        launch_controls = self.manager._deployment_launch_controls(launch_settings)
        settings = dict((deployment or {}).get("settings") or {})
        settings["manager_deployment_id"] = manager_deployment.get("id")
        model_source = str(
            manager_deployment.get("model_source")
            or launch_settings.get("model_source")
            or settings.get("model_source")
            or "unknown"
        )
        settings["model_source"] = (
            model_source
            if model_source in {"local", "public_repository", "unknown"}
            else "unknown"
        )
        context_window = launch_controls.get("context_window") or launch_settings.get("sg_context_length")
        if context_window is not None:
            for key in ("max_model_len", "context_length", "context_size"):
                settings.pop(key, None)
            settings["context_length"] = context_window
        contract = self.manager.recipe_deployment_contract(launch_settings)
        if not contract.get("supported", True):
            raise ValueError(contract.get("error") or "deployment launch metadata is unsupported")
        tensor_parallel_size = contract.get("tensor_parallel_size")
        if tensor_parallel_size is not None:
            settings["tensor_parallel_size"] = tensor_parallel_size

        model_id = str(
            manager_deployment.get("model")
            or launch_settings.get("model")
            or ((deployment or {}).get("model") or {}).get("repository")
            or "local-model"
        ).strip()
        runtime = str(
            manager_deployment.get("engine")
            or launch_settings.get("engine")
            or (deployment or {}).get("runtime")
            or RuntimeKind.VLLM.value
        )
        return {
            **(deployment or {}),
            "id": (deployment or {}).get("id") or manager_deployment.get("id"),
            "kind": (deployment or {}).get("kind") or DeploymentKind.MANAGED.value,
            "runtime": runtime,
            "model": {**((deployment or {}).get("model") or {}), "repository": model_id},
            "settings": settings,
            "status": _deployment_status(manager_deployment.get("status")),
        }

    async def catalog_search(
        self, query: str, limit: int, runtime: str | None = None
    ) -> dict[str, Any]:
        bounded_limit = min(100, max(1, int(limit)))
        if runtime is not None:
            self.registry.get(runtime)
        deployments = await self.deployments()
        local: dict[str, list[dict[str, Any]]] = {}
        for deployment in deployments:
            repository = deployment["model"]["repository"]
            local.setdefault(repository, []).append(deployment)

        # Hugging Face enriches the catalog when online, while local models
        # remain searchable when the public catalog cannot be reached.
        try:
            remote_items = await self.catalog.search(query, bounded_limit)
        except (httpx.HTTPError, ValueError, TypeError):
            remote_items = []

        items_by_id = {
            item["id"]: copy.deepcopy(item)
            for item in remote_items if item.get("id")
        }
        folded_query = query.strip().casefold()
        matching_local_ids: list[str] = []
        for repository, matches in local.items():
            if folded_query and folded_query not in repository.casefold() and not any(
                folded_query in str(item.get("alias") or "").casefold()
                for item in matches
            ):
                continue
            matching_local_ids.append(repository)
            item = items_by_id.setdefault(
                repository,
                {
                    "id": repository,
                    "name": repository.split("/")[-1],
                    "author": repository.split("/", 1)[0] if "/" in repository else None,
                    "downloads": None,
                    "likes": None,
                    "tags": [],
                    "runtime_compatibility": [],
                    "community": None,
                },
            )
            item["local_deployment_ids"] = [match["id"] for match in matches]
            compatibility = {
                entry["runtime"]: entry
                for entry in item.get("runtime_compatibility") or []
            }
            for match in matches:
                compatibility[match["runtime"]] = {
                    "runtime": match["runtime"], "supported": True,
                }
            item["runtime_compatibility"] = list(compatibility.values())

        matching_local = set(matching_local_ids)
        items = [items_by_id[item_id] for item_id in matching_local_ids]
        items.extend(
            item for item_id, item in items_by_id.items()
            if item_id not in matching_local
        )
        if runtime:
            items = [
                item for item in items
                if any(
                    entry.get("runtime") == runtime and entry.get("supported")
                    for entry in item.get("runtime_compatibility") or []
                )
            ]
        items = items[:bounded_limit]
        return {"items": items, "total": len(items), "next_cursor": None}

    async def catalog_details(self, repository: str) -> dict[str, Any]:
        model = await self.catalog.details(repository)
        local = [
            item for item in await self.deployments()
            if (item.get("model") or {}).get("repository") == repository
        ]
        if local:
            model = copy.deepcopy(model)
            model["local_deployment_ids"] = [item["id"] for item in local]
        return {"model": model, "aggregates": []}

    async def deployments(self) -> list[dict[str, Any]]:
        raw_manager_deployments = getattr(self.manager, "deployments", [])
        registered = await self._adopt_unlinked_manager_deployments(
            raw_manager_deployments, skip_if_creating=True,
        )
        by_container = {item.get("container_name"): item for item in registered}
        cluster_state: dict[str, Any] = {}
        if any(item.get("settings", {}).get("manager_deployment_id") for item in registered):
            try:
                cluster_state = await self.manager.get_state()
            except Exception:
                cluster_state = {}
        manager_deployments = (
            cluster_state.get("deployments")
            or raw_manager_deployments
        )
        cluster_by_id = {
            item.get("id"): item
            for item in manager_deployments
            if isinstance(item, dict) and item.get("id")
        }
        cluster_by_record = {
            item.get("sparkdeck_record_id"): item
            for item in manager_deployments
            if isinstance(item, dict) and item.get("sparkdeck_record_id")
        }
        cluster_by_container = {
            member.get("container_name"): item
            for item in manager_deployments
            if isinstance(item, dict)
            for member in (item.get("members") or [])
            if isinstance(member, dict) and member.get("container_name")
        }
        cluster_nodes: dict[str, dict[str, Any]] = {}
        if cluster_state.get("nodes"):
            # get_state already collected the authoritative node inventory.
            # Reusing it avoids a second round of local Docker and remote node
            # requests on every Models-page poll.
            cluster_nodes = {
                node["id"]: node for node in cluster_state["nodes"]
                if isinstance(node, dict) and node.get("id")
            }
        elif any(
            item.get("settings", {}).get("manager_deployment_id") for item in registered
        ) or cluster_by_container:
            try:
                cluster_nodes = {
                    node["id"]: node for node in await self.manager.cluster_nodes()
                }
            except Exception:
                cluster_nodes = {}
        if cluster_state:
            # get_state also includes the container inventory. Do not call
            # list_containers a second time; on Windows a Docker Desktop
            # request can otherwise block the polling endpoint.
            containers = cluster_state.get("containers") or []
            docker_unavailable = not bool(cluster_state.get("docker_ready"))
        else:
            docker_unavailable = False
            try:
                containers = await self.manager.list_containers()
            except Exception:
                containers = []
                docker_unavailable = True
        seen: set[str] = set()
        local_cluster_members: dict[str, dict[str, Any]] = {}
        for stored in registered:
            manager_id = stored.get("settings", {}).get("manager_deployment_id")
            cluster = cluster_by_id.get(manager_id) or cluster_by_record.get(stored["id"])
            if not cluster:
                continue
            if not cluster.get("sparkdeck_record_id") and cluster.get("id") == manager_id:
                cluster["sparkdeck_record_id"] = stored["id"]
                durable_cluster = next(
                    (
                        item for item in raw_manager_deployments
                        if isinstance(item, dict) and item.get("id") == manager_id
                    ),
                    cluster,
                )
                durable_cluster["sparkdeck_record_id"] = stored["id"]
                launch_settings = durable_cluster.get("launch_settings")
                if isinstance(launch_settings, dict):
                    launch_settings["sparkdeck_record_id"] = stored["id"]
                save_deployments = getattr(self.manager, "_save_deployments", None)
                if save_deployments is not None:
                    save_deployments()
            if cluster.get("id") != manager_id:
                settings = self._local_configuration({
                    **(stored.get("settings") or {}),
                    "manager_deployment_id": cluster.get("id"),
                    "node_ids": list(cluster.get("node_ids") or []),
                })
                primary = (cluster.get("members") or [{}])[0]
                container_name = primary.get("container_name") or stored.get("container_name")
                port = cluster.get("api_port")
                self.store.update_managed_routing(
                    stored["id"], settings, container_name,
                    f"http://127.0.0.1:{int(port)}" if port else stored.get("_base_url"),
                )
                stored["settings"] = settings
                stored["container_name"] = container_name
                manager_id = cluster.get("id")
            manager_desired = cluster.get("desired_state")
            stored_desired = stored.get("desired_state")
            # A persisted stop is sticky until an explicit SparkDeck start.
            # This also migrates legacy SQLite rows when the Manager record is
            # already authoritative about a stopped deployment.
            if manager_desired == "stopped" and stored_desired != "stopped":
                self.store.update_desired_state(stored["id"], "stopped")
                stored["desired_state"] = "stopped"
            elif manager_desired == "running" and stored_desired != "stopped":
                if stored_desired != "running":
                    self.store.update_desired_state(stored["id"], "running")
                stored["desired_state"] = "running"
            stored["status"] = _deployment_status(cluster.get("status"))
            stored.update(_deployment_launch_progress(cluster))
            stored["last_used_at"] = cluster.get("last_used_at")
            launch_controls = cluster.get("launch_controls")
            if not isinstance(launch_controls, dict):
                controls_from_settings = getattr(
                    self.manager, "_deployment_launch_controls", None,
                )
                launch_controls = (
                    controls_from_settings(cluster.get("launch_settings") or {})
                    if callable(controls_from_settings) else {}
                )
            public_settings = dict(stored.get("settings") or {})
            if launch_controls.get("context_window") is not None:
                public_settings["context_length"] = launch_controls["context_window"]
            if launch_controls.get("max_concurrency") is not None:
                public_settings["max_concurrency"] = launch_controls["max_concurrency"]
            stored["settings"] = public_settings
            if cluster.get("error"):
                stored["last_error"] = str(cluster["error"])
            stored.update(self._layout_contract(cluster.get("launch_settings")))
            stored["port"] = cluster.get("api_port")
            stored["managed"] = True
            stored["managed_by"] = (
                cluster.get("managed_by")
                or stored.get("settings", {}).get("managed_by")
            )
            stored["automation_run_id"] = (
                cluster.get("automation_run_id")
                or stored.get("settings", {}).get("automation_run_id")
            )
            last_deployed_at = (
                cluster.get("last_deployed_at") or cluster.get("created_at")
            )
            if isinstance(last_deployed_at, (int, float)) and last_deployed_at:
                stored["last_deployed_at"] = datetime.fromtimestamp(
                    last_deployed_at, timezone.utc
                ).isoformat()
            elif isinstance(last_deployed_at, str) and last_deployed_at:
                stored["last_deployed_at"] = last_deployed_at
            stored["node_ids"] = list(cluster.get("node_ids") or [])
            stored["selected_nodes"] = [
                self.manager.public_target_node(cluster_nodes[node_id])
                for node_id in stored["node_ids"] if node_id in cluster_nodes
            ]
            for member in cluster.get("members") or []:
                if not isinstance(member, dict) or member.get("node_id") != "local":
                    continue
                container_name = member.get("container_name")
                if container_name:
                    local_cluster_members[container_name] = stored
            seen.add(stored["id"])
        # Reconciliation can replace a local primary container name. Refresh
        # ownership before scanning Docker so the replacement is not appended
        # again as a synthetic legacy deployment.
        by_container = {item.get("container_name"): item for item in registered}
        source_container_names = {
            item.get("settings", {}).get("source_container_name")
            for item in registered
            if item.get("settings", {}).get("source_container_name")
        }
        by_container.update(local_cluster_members)
        for container in containers:
            runtime = self._container_runtime(container)
            if runtime not in self.registry.kinds:
                continue
            stored = by_container.get(container.get("name"))
            if stored:
                # A local rank is not the whole clustered deployment. Preserve
                # Manager's aggregate state while another rank is still
                # pulling, starting, or failing.
                if not stored.get("settings", {}).get("manager_deployment_id"):
                    stored["status"] = _managed_container_status(container)
                stored["port"] = container.get("port")
                stored["managed"] = True
                seen.add(stored["id"])
                continue
            if container.get("name") in source_container_names:
                # Promotion deliberately leaves the original unmanaged
                # container available as a fallback. Hide that source card,
                # but never let it contribute state or routing to the managed
                # replacement: only Manager's deployment and ranks own those.
                continue
            model = container.get("model") or container.get("served_model")
            if not model:
                continue
            discovered = self._discovered_deployment(container, runtime, model)
            owner = cluster_by_container.get(container.get("name"))
            if owner:
                # One rank of a multi-node cluster: the card reports the whole
                # deployment's state and layout contract, not just this node's
                # rank container.
                discovered["status"] = _deployment_status(owner.get("status"))
                discovered.update(self._layout_contract(owner.get("launch_settings")))
                owner_node_ids = [
                    str(item) for item in owner.get("node_ids") or [] if str(item).strip()
                ]
                if owner_node_ids:
                    # Without the cluster node set the card would look like a
                    # standalone container: the running-node tooltip hides and
                    # an add-nodes picker cannot tell which nodes already run
                    # the model.
                    discovered["node_ids"] = owner_node_ids
                    discovered["selected_nodes"] = [
                        self.manager.public_target_node(cluster_nodes[node_id])
                        for node_id in owner_node_ids if node_id in cluster_nodes
                    ]
            registered.append(discovered)
        for deployment in registered:
            if (
                deployment.get("kind") == DeploymentKind.MANAGED.value
                and deployment.get("id") not in seen
                and not str(deployment.get("id") or "").startswith("container:")
            ):
                if deployment.get("id") in self._deployment_launches:
                    deployment.update({
                        "status": "starting",
                        "launch_phase": "queued",
                        "launch_message": "Preparing deployment launch",
                    })
                    continue
                settings = deployment.get("settings") or {}
                never_launched = (
                    not deployment.get("container_name")
                    and not settings.get("manager_deployment_id")
                    and deployment.get("desired_state") == "stopped"
                )
                if never_launched:
                    # A saved deployment is a launch bookmark; it owns no
                    # containers until its first explicit start.
                    deployment["status"] = "saved"
                    deployment["node_ids"] = list(settings.get("node_ids") or [])
                    deployment.update(self._saved_layout_contract(
                        settings, str(deployment.get("runtime") or ""),
                    ))
                    continue
                deployment["status"] = "missing"
                deployment["last_error"] = (
                    "Docker is unavailable" if docker_unavailable
                    else "Managed container is missing"
                )
        await asyncio.gather(
            *(self._probe_external_endpoint(item) for item in registered)
        )
        for deployment in registered:
            deployment.pop("_base_url", None)
            deployment.pop("_credential_ref", None)
        return registered

    async def register_manager_deployment(
        self, cluster: dict[str, Any],
    ) -> dict[str, Any]:
        """Persist one legacy Manager launch in the v1 deployment catalog."""
        # Match native v1 creation's mutation boundary. The fixed lock order is
        # create then reconciliation; read-side reconciliation never acquires
        # the create lock, so this cannot deadlock with catalog polling.
        async with self._deployment_create_lock:
            registered = await self._adopt_unlinked_manager_deployments([cluster])
        record_id = str(cluster.get("sparkdeck_record_id") or "")
        adopted = next(
            (item for item in registered if str(item.get("id") or "") == record_id),
            None,
        )
        if adopted is None:
            raise RuntimeError("manager deployment could not be registered")
        return adopted

    async def _adopt_unlinked_manager_deployments(
        self, manager_deployments: Any, *, skip_if_creating: bool = False,
    ) -> list[dict[str, Any]]:
        """Give legacy cluster launches durable cards in the v1 catalog.

        The legacy ``/api/containers`` path returns Manager deployment IDs and
        remains in use by MCP clients. Remote-only launches have no local
        container for normal Docker discovery, so without this reconciliation
        they can be healthy in Manager while remaining invisible in Models.
        """
        if not isinstance(manager_deployments, list):
            return self.store.deployments(include_private=True)
        # Native v1 creation intentionally holds the create lock across remote
        # launch work. Reads must keep returning the already durable catalog
        # instead of queueing behind a potentially minutes-long launch.
        if skip_if_creating and self._deployment_create_lock.locked():
            return self.store.deployments(include_private=True)
        async with self._deployment_reconciliation_lock:
            registered = self.store.deployments(include_private=True)
            by_manager_id = {
                str((item.get("settings") or {}).get("manager_deployment_id")): item
                for item in registered
                if (item.get("settings") or {}).get("manager_deployment_id")
            }
            by_record_id = {
                str(item.get("id")): item for item in registered if item.get("id")
            }
            changed = False
            for cluster in manager_deployments:
                if not isinstance(cluster, dict) or not cluster.get("id"):
                    continue
                manager_id = str(cluster["id"])
                launch_settings = (
                    dict(cluster.get("launch_settings"))
                    if isinstance(cluster.get("launch_settings"), dict) else {}
                )
                linked = by_manager_id.get(manager_id)
                record_id = str(cluster.get("sparkdeck_record_id") or "")
                if linked is None and record_id:
                    candidate = by_record_id.get(record_id)
                    candidate_manager_id = str(
                        ((candidate or {}).get("settings") or {}).get(
                            "manager_deployment_id"
                        ) or ""
                    )
                    if candidate_manager_id in {"", manager_id}:
                        linked = candidate
                    elif candidate is not None:
                        # Never let a caller-supplied reverse link steal a v1
                        # card from another Manager deployment. Allocate a new
                        # deterministic record for this cluster instead.
                        record_id = ""
                if linked is not None:
                    if cluster.get("sparkdeck_record_id") != linked["id"]:
                        cluster["sparkdeck_record_id"] = linked["id"]
                        durable_launch_settings = cluster.get("launch_settings")
                        if isinstance(durable_launch_settings, dict):
                            durable_launch_settings["sparkdeck_record_id"] = linked["id"]
                            launch_settings["sparkdeck_record_id"] = linked["id"]
                        changed = True
                    safe_args = list(launch_settings.get("extra_args") or [])
                    sanitize_args = getattr(
                        self.manager, "_without_sensitive_cli_credentials", None,
                    )
                    if callable(sanitize_args):
                        safe_args = sanitize_args(safe_args)
                    existing_settings = dict(linked.get("settings") or {})
                    existing_source = str(
                        existing_settings.get("model_source") or "unknown"
                    )
                    incoming_source = str(
                        cluster.get("model_source")
                        or launch_settings.get("model_source")
                        or "unknown"
                    )
                    if (
                        incoming_source not in {
                            "local", "public_repository", "unknown",
                        }
                        or (
                            incoming_source == "unknown"
                            and existing_source in {"local", "public_repository"}
                        )
                    ):
                        incoming_source = existing_source
                    node_ids = [
                        str(item) for item in (
                            cluster.get("node_ids")
                            or existing_settings.get("node_ids")
                            or []
                        ) if str(item).strip()
                    ]
                    settings = self._local_configuration({
                        **existing_settings,
                        **launch_settings,
                        "extra_args": safe_args,
                        "deployment_mode": (
                            cluster.get("mode")
                            or launch_settings.get("deployment_mode")
                            or existing_settings.get("deployment_mode")
                            or ("replicated" if len(node_ids) > 1 else "single")
                        ),
                        "node_ids": node_ids,
                        "manager_deployment_id": manager_id,
                        "model_source": incoming_source,
                        "managed_by": (
                            cluster.get("managed_by")
                            or existing_settings.get("managed_by")
                        ),
                        "automation_run_id": (
                            cluster.get("automation_run_id")
                            or existing_settings.get("automation_run_id")
                        ),
                    })
                    members = cluster.get("members") or []
                    primary = (
                        members[0]
                        if members and isinstance(members[0], dict) else {}
                    )
                    container_name = (
                        primary.get("container_name") or linked.get("container_name")
                    )
                    try:
                        port = int(cluster.get("api_port"))
                    except (TypeError, ValueError):
                        port = None
                    self.store.update_managed_routing(
                        linked["id"], settings, container_name,
                        f"http://127.0.0.1:{port}" if port else linked.get("_base_url"),
                    )
                    try:
                        runtime = RuntimeKind(str(
                            cluster.get("engine") or linked.get("runtime") or "vllm"
                        ))
                    except ValueError:
                        runtime = None
                    model = dict(linked.get("model") or {})
                    model["repository"] = str(
                        cluster.get("model") or model.get("repository") or ""
                    )
                    model["revision"] = (
                        self._persisted_revision(launch_settings)
                        or model.get("revision")
                    )
                    model["artifact"] = _optional_string(
                        (
                            launch_settings.get("llama_artifact")
                            if runtime is RuntimeKind.LLAMA_CPP
                            else launch_settings.get("artifact")
                        )
                        or model.get("artifact")
                    )
                    model["quantization"] = (
                        canonical_quantization(launch_settings.get("quantization"))
                        or model.get("quantization")
                    )
                    self.store.update_deployment_model(linked["id"], model)
                    refreshed = self.store.deployment(
                        linked["id"], include_private=True,
                    )
                    if refreshed is not None:
                        registered[registered.index(linked)] = refreshed
                        by_manager_id[manager_id] = refreshed
                        by_record_id[str(refreshed["id"])] = refreshed
                    continue
                model = str(cluster.get("model") or "").strip()
                try:
                    runtime = RuntimeKind(str(cluster.get("engine") or "vllm"))
                except ValueError:
                    continue
                if not model:
                    continue
                safe_args = list(launch_settings.get("extra_args") or [])
                sanitize_args = getattr(
                    self.manager, "_without_sensitive_cli_credentials", None,
                )
                if callable(sanitize_args):
                    safe_args = sanitize_args(safe_args)
                node_ids = [
                    str(item) for item in cluster.get("node_ids") or []
                    if str(item).strip()
                ]
                settings = self._local_configuration({
                    **launch_settings,
                    "extra_args": safe_args,
                    "deployment_mode": (
                        cluster.get("mode")
                        or launch_settings.get("deployment_mode")
                        or ("replicated" if len(node_ids) > 1 else "single")
                    ),
                    "node_ids": node_ids,
                    "manager_deployment_id": manager_id,
                    "model_source": (
                        cluster.get("model_source")
                        or launch_settings.get("model_source")
                        or "unknown"
                    ),
                    "managed_by": cluster.get("managed_by"),
                    "automation_run_id": cluster.get("automation_run_id"),
                })
                alias_base = str(
                    cluster.get("name")
                    or launch_settings.get("deployment_name")
                    or model
                ).strip() or model
                record_id = record_id or str(uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"sparkdeck:manager-deployment:{manager_id}",
                ))
                alias = alias_base
                suffix = 1
                while self.store.deployment(alias) is not None:
                    marker = manager_id[:8]
                    if suffix > 1:
                        marker = f"{marker}-{suffix}"
                    alias = f"{alias_base} ({marker})"
                    suffix += 1
                members = cluster.get("members") or []
                primary = members[0] if members and isinstance(members[0], dict) else {}
                deployment = Deployment(
                    id=record_id,
                    alias=alias,
                    runtime=runtime,
                    kind=DeploymentKind.MANAGED,
                    model=ModelIdentity(
                        repository=model,
                        revision=self._persisted_revision(launch_settings),
                        artifact=_optional_string(
                            launch_settings.get("llama_artifact")
                            if runtime is RuntimeKind.LLAMA_CPP
                            else launch_settings.get("artifact")
                        ),
                        quantization=canonical_quantization(
                            launch_settings.get("quantization")
                        ),
                    ),
                    container_name=primary.get("container_name"),
                    settings=settings,
                    desired_state=(
                        str(cluster.get("desired_state"))
                        if cluster.get("desired_state") in {"running", "stopped"}
                        else "running"
                    ),
                )
                try:
                    port = int(cluster.get("api_port"))
                except (TypeError, ValueError):
                    port = None
                self.store.add_deployment(
                    deployment,
                    f"http://127.0.0.1:{port}" if port else None,
                    None,
                )
                cluster["sparkdeck_record_id"] = record_id
                if isinstance(cluster.get("launch_settings"), dict):
                    cluster["launch_settings"]["sparkdeck_record_id"] = record_id
                adopted = self.store.deployment(record_id, include_private=True)
                if adopted is not None:
                    registered.append(adopted)
                    by_manager_id[manager_id] = adopted
                    by_record_id[record_id] = adopted
                changed = True
            if changed:
                save_deployments = getattr(self.manager, "_save_deployments", None)
                if callable(save_deployments):
                    try:
                        save_deployments()
                    except Exception:
                        # The SQLite forward link is already durable and is
                        # sufficient to prevent duplicate adoption next time.
                        pass
            return registered

    def remove_manager_deployment_registration(
        self, manager_id: str,
    ) -> str | None:
        """Delete the v1 card linked to a removed legacy Manager deployment."""
        manager_id = str(manager_id or "")
        linked = next(
            (
                item for item in self.store.deployments(include_private=True)
                if str((item.get("settings") or {}).get("manager_deployment_id") or "")
                == manager_id
            ),
            None,
        )
        if linked is None:
            return None
        record_id = str(linked["id"])
        self.store.delete_deployment(record_id)
        return record_id

    async def _probe_external_endpoint(self, deployment: dict[str, Any]) -> None:
        if deployment.get("kind") != DeploymentKind.EXTERNAL.value:
            return
        discovered = str(deployment.get("id") or "").startswith("container:")
        if discovered and deployment.get("status") not in {"running", "starting"}:
            return
        base_url = deployment.get("_base_url")
        if not base_url and discovered:
            # Discovered cards have no stored endpoint: only probe when the
            # container exposes a mappable port, and keep the Docker status
            # otherwise (host-network containers have none).
            port = deployment.get("port")
            if not port:
                return
            base_url = f"http://127.0.0.1:{int(port)}"
        if not base_url:
            return
        try:
            adapter = self.registry.get(deployment["runtime"])
            await asyncio.wait_for(
                adapter.health(
                    self.manager.http,
                    base_url,
                    self._get_credential(
                        deployment["id"], deployment.get("_credential_ref")
                    ),
                ),
                timeout=3,
            )
            deployment["status"] = "running"
        except Exception:
            if discovered:
                # Docker owns lifecycle truth for synthetic container cards.
                # The endpoint may still be loading or require credentials
                # that SparkDeck does not have; neither makes it startable.
                return
            deployment["status"] = "error"
            deployment["last_error"] = "Endpoint health check failed"

    async def deployment_detail(self, deployment_id: str) -> dict[str, Any]:
        """Return one deployment with only its safe, editable launch inputs."""
        public = next(
            (
                item for item in await self.deployments()
                if item.get("id") == deployment_id
            ),
            None,
        )
        if public is None:
            raise LookupError("deployment not found")

        stored = self.store.deployment(deployment_id, include_private=True)
        manager_deployment = None
        manager_id = None
        if stored is not None:
            manager_id = (stored.get("settings") or {}).get(
                "manager_deployment_id"
            )
            lookup = getattr(self.manager, "_deployment", None)
            if manager_id and callable(lookup):
                manager_deployment = lookup(manager_id)
            if manager_deployment is None:
                manager_deployment = next(
                    (
                        item
                        for item in getattr(self.manager, "deployments", [])
                        if isinstance(item, dict)
                        and (
                            (manager_id and item.get("id") == manager_id)
                            or item.get("sparkdeck_record_id") == deployment_id
                        )
                    ),
                    None,
                )

        launch_settings = (
            manager_deployment.get("launch_settings")
            if isinstance(manager_deployment, dict)
            and isinstance(manager_deployment.get("launch_settings"), dict)
            else None
        )
        # A saved deployment is a launch bookmark: it owns a record with the
        # model, settings, and node preferences but no launched runtime yet.
        # Local-GGUF bookmarks record their provenance up front, so ownership
        # (not model_source) decides bookmark state.
        saved_only = bool(
            stored is not None
            and stored.get("kind") == DeploymentKind.MANAGED.value
            and not manager_id
            and manager_deployment is None
            and not stored.get("container_name")
        )
        # Discovered containers have no saved launch settings, but Manager can
        # transactionally rebuild commands it knows how to parse.
        discovered_settings: dict[str, Any] | None = None
        discovered_image = None
        container: dict[str, Any] | None = None
        if launch_settings is None and str(deployment_id).startswith("container:"):
            try:
                container = await self._resolve_discovered_container(deployment_id)
            except LookupError:
                container = None
            if container:
                discovered_settings = container.get("load_settings") or {}
                discovered_image = container.get("image")

        if discovered_settings is not None:
            engine = str(public.get("runtime") or "vllm")
            extra_args = self.manager._without_sensitive_cli_credentials(
                [str(value) for value in discovered_settings.get("extra_args") or []]
            )
            launch_controls = _discovered_launch_controls(
                self.manager, engine, discovered_settings, extra_args,
            )
        else:
            saved_settings = (stored or {}).get("settings") or {}
            saved_extra_args = (
                [str(item) for item in saved_settings.get("extra_args") or []]
                if saved_only else []
            )
            extra_args = self.manager._without_sensitive_cli_credentials(
                (launch_settings or {}).get("extra_args") or saved_extra_args
            )
            launch_controls = (
                self.manager._deployment_launch_controls({
                    **launch_settings,
                    "extra_args": extra_args,
                })
                if launch_settings is not None
                else self._saved_bookmark_launch_controls(
                    str(public.get("runtime") or "vllm"), extra_args, saved_settings,
                ) if saved_only else {}
            )

        raw_status = str(
            (manager_deployment or {}).get("status") or public.get("status") or ""
        ).lower()
        repairable_error = bool(
            raw_status == "error"
            and (
                (manager_deployment or {}).get("launch_settings_error")
                or "persisted launch_settings.extra_args" in str(
                    (manager_deployment or {}).get("error") or ""
                )
            )
        )
        saved_only = bool(saved_only)
        _EDITABLE_RUNTIMES = {"vllm", "sglang", "llama.cpp"}
        discovered_editable = bool(
            discovered_settings is not None
            and discovered_settings.get("editable")
            and not self._owning_cluster_deployment(
                str((container or {}).get("name") or "")
            )
        )
        editable = bool(
            stored is not None
            and manager_id
            and manager_deployment is not None
            and launch_settings is not None
            and str(public.get("runtime") or "") in _EDITABLE_RUNTIMES
            and (raw_status == "stopped" or repairable_error)
        ) or saved_only or discovered_editable
        if editable:
            edit_reason = None
        elif str(deployment_id).startswith("container:"):
            edit_reason = (
                "This discovered deployment's launch command cannot be edited "
                "safely. Save it as a recipe to create editable launch settings."
            )
        elif stored is None:
            edit_reason = "Saved launch settings are unavailable for this deployment."
        elif public.get("kind") != DeploymentKind.MANAGED.value:
            edit_reason = "External deployments do not have SparkDeck-managed launch settings."
        elif not manager_id or manager_deployment is None or launch_settings is None:
            edit_reason = "Saved launch settings are unavailable for this deployment."
        elif str(public.get("runtime") or "") not in _EDITABLE_RUNTIMES:
            edit_reason = "This runtime does not support editing saved launch settings."
        else:
            edit_reason = "Stop the deployment before changing its launch settings."

        desired_state = (
            (manager_deployment or {}).get("desired_state")
            or (stored or {}).get("desired_state")
        )
        if desired_state not in {"running", "stopped"}:
            desired_state = (
                "running"
                if str(public.get("status") or "").lower() in {"running", "starting"}
                else "stopped"
            )

        # The ordinary deployment list carries controller-local routing keys
        # in ``settings`` so lifecycle actions can reconcile records. A detail
        # response is an editor contract, not a routing contract: expose only
        # safe runtime configuration and keep argv in the sanitized top-level
        # field below.
        public_settings = self._safe_configuration(public.get("settings") or {})
        public_settings.pop("extra_args", None)
        gpu_memory_utilization = (launch_settings or {}).get("gpu_memory_utilization")
        sg_tp_size = (launch_settings or {}).get("sg_tp_size")
        sg_mem_fraction = (launch_settings or {}).get("sg_mem_fraction")
        gpu_memory_gb = (launch_settings or {}).get("gpu_memory_gb")
        if saved_only:
            # Before the first launch the bookmark's own scalars are the only
            # source, so an unchanged editor save cannot erase them with null.
            sg_tp_size = saved_settings.get("tensor_parallel_size", sg_tp_size)
            sg_mem_fraction = saved_settings.get(
                "mem_fraction_static", sg_mem_fraction,
            )
            gpu_memory_gb = saved_settings.get("gpu_memory_gb", gpu_memory_gb)
        image = (launch_settings or {}).get("image")
        environment = (launch_settings or {}).get("environment")
        if saved_only:
            image = saved_settings.get("image", image)
            environment = saved_settings.get("environment", environment)
        if discovered_settings is not None:
            engine = str(public.get("runtime") or "vllm")
            if engine == "sglang":
                # The SGLang parser reports --mem-fraction-static through the
                # gpu_memory_utilization key; surface it under the sg fields.
                sg_mem_fraction = discovered_settings.get("gpu_memory_utilization")
                sg_tp_size = discovered_settings.get("tensor_parallel_size")
            else:
                gpu_memory_utilization = discovered_settings.get("gpu_memory_utilization")
            image = image or discovered_image
            environment = discovered_settings.get("environment") or environment
        return {
            **public,
            "settings": public_settings,
            "editable": editable,
            "edit_reason": edit_reason,
            "desired_state": desired_state,
            "extra_args": extra_args,
            "launch_controls": launch_controls,
            "gpu_memory_utilization": gpu_memory_utilization,
            "gpu_memory_gb": gpu_memory_gb,
            "sg_tp_size": sg_tp_size,
            "sg_mem_fraction": sg_mem_fraction,
            "image": image,
            "environment": environment or {},
        }

    async def update_deployment_settings(
        self, deployment_id: str, changes: dict[str, Any],
    ) -> dict[str, Any]:
        """Update a stopped manager-backed deployment by its public record ID."""
        if str(deployment_id).startswith("container:"):
            container = await self._resolve_discovered_container(deployment_id)
            return await self._update_discovered_deployment(container, changes)
        # Listing first performs the normal manager/store reconciliation, so a
        # settings save cannot target a manager deployment ID that was replaced
        # by an earlier relaunch.
        await self.deployments()
        stored = self.store.deployment(deployment_id, include_private=True)
        if stored is None:
            raise LookupError("deployment not found")
        manager_id = (stored.get("settings") or {}).get(
            "manager_deployment_id"
        )
        if not manager_id:
            saved_only = bool(
                stored.get("kind") == DeploymentKind.MANAGED.value
                and not stored.get("container_name")
            )
            if not saved_only:
                # Launched standalone deployments and external endpoints have
                # a live runtime or endpoint contract; their creator-form
                # identity fields must not change behind it.
                raise ValueError(
                    "deployment does not have editable saved launch settings"
                )
            return await self._update_saved_deployment(stored, changes)

        allowed = {
            "extra_args", "launch_controls",
            "environment",
            "gpu_memory_utilization", "gpu_memory_gb",
            "sg_tp_size", "sg_mem_fraction",
        }
        unknown = sorted(set(changes) - allowed)
        if unknown:
            raise ValueError(f"unsupported field(s): {', '.join(unknown)}")

        updated = self.manager.update_deployment_settings(manager_id, changes)
        launch_settings = updated.get("launch_settings") or {}
        controls = self.manager._deployment_launch_controls(launch_settings)
        contract = self.manager.recipe_deployment_contract(launch_settings)
        safe_args = self.manager._without_sensitive_cli_credentials(
            launch_settings.get("extra_args") or []
        )
        current_settings = dict(stored.get("settings") or {})
        refreshed_settings = self._local_configuration({
            **current_settings,
            **launch_settings,
            "extra_args": safe_args,
            "context_length": controls.get("context_window"),
            "tensor_parallel_size": contract.get("tensor_parallel_size"),
            "deployment_mode": updated.get("mode")
            or launch_settings.get("deployment_mode"),
            "node_ids": list(updated.get("node_ids") or []),
            "manager_deployment_id": manager_id,
        })
        self.store.update_managed_routing(
            deployment_id,
            refreshed_settings,
            stored.get("container_name"),
            stored.get("_base_url"),
        )
        return await self.deployment_detail(deployment_id)

    async def _update_discovered_deployment(
        self, container: dict[str, Any], changes: dict[str, Any],
    ) -> dict[str, Any]:
        """Apply the unified deployment editor contract to one Docker command."""
        settings = dict(container.get("load_settings") or {})
        name = str(container.get("name") or "")
        if not name or not settings.get("editable"):
            raise ValueError("discovered deployment launch settings are not editable")
        if self._owning_cluster_deployment(name) is not None:
            raise ValueError("edit the cluster deployment instead of one member")

        allowed = {
            "extra_args", "launch_controls", "environment",
            "gpu_memory_utilization", "gpu_memory_gb",
            "sg_tp_size", "sg_mem_fraction",
        }
        unknown = sorted(set(changes) - allowed)
        if unknown:
            raise ValueError(f"unsupported field(s): {', '.join(unknown)}")
        if changes.get("gpu_memory_gb") not in (None, ""):
            raise ValueError(
                "GPU memory reserve is not supported for discovered deployments"
            )

        engine = str(settings.get("engine") or self._container_runtime(container))
        args = changes.get("extra_args", settings.get("extra_args") or [])
        if not isinstance(args, list) or any(not isinstance(item, str) for item in args):
            raise ValueError("extra_args must be an array of strings")
        self._reject_sensitive_launch_args(args)
        current_controls = _discovered_launch_controls(
            self.manager, engine, settings,
            [str(item) for item in settings.get("extra_args") or []],
        )
        submitted_controls = changes.get("launch_controls")
        if submitted_controls is not None and not isinstance(submitted_controls, dict):
            raise ValueError("launch_controls must be an object")
        controls = {
            **current_controls,
            **(submitted_controls or {}),
        }
        original_environment = normalize_runtime_environment(
            settings.get("environment"), engine,
        )
        environment = normalize_runtime_environment(
            changes.get("environment", original_environment), engine,
        )
        merged_args = self.manager._apply_deployment_launch_controls(
            list(args), engine, controls, environment,
        )
        command_args = merged_args
        if engine == "vllm":
            # Docker exec-form argv never expands ${NAME}. Persist the readable
            # reference in managed settings, but rebuild discovered containers
            # with the same resolved JSON argv used by managed vLLM launches.
            command_args = self.manager._resolve_environment_backed_speculative_args(
                merged_args, environment,
            )

        if engine == "sglang":
            sg_tp_size = self.manager._validated_sg_scalar(
                "sg_tp_size", (
                    changes.get("sg_tp_size")
                    if "sg_tp_size" in changes
                    else settings.get("tensor_parallel_size")
                )
            )
            utilization = self.manager._validated_sg_scalar(
                "sg_mem_fraction", (
                    changes.get("sg_mem_fraction")
                    if "sg_mem_fraction" in changes
                    else settings.get("gpu_memory_utilization")
                )
            )
            command_args = self.manager._with_sglang_runtime_controls(
                merged_args,
                controls.get("context_window"),
                controls.get("max_concurrency"),
                sg_tp_size,
                utilization,
            )
        else:
            utilization = (
                changes.get("gpu_memory_utilization")
                if "gpu_memory_utilization" in changes
                else settings.get("gpu_memory_utilization")
            )

        replacement = {
            "command_flags": shlex.join(command_args),
            "context_window": controls.get("context_window"),
            "max_concurrency": controls.get("max_concurrency"),
            "kv_cache_dtype": controls.get("kv_cache_dtype"),
            "thinking_mode": controls.get("thinking_mode"),
            "gpu_memory_utilization": utilization,
        }
        if "environment" in changes or environment != original_environment:
            replacement["environment"] = environment
        await self.manager.update_container_settings(name, replacement)
        return await self.deployment_detail(f"container:{name}")

    async def _update_saved_deployment(
        self, stored: dict[str, Any], changes: dict[str, Any],
    ) -> dict[str, Any]:
        """Edit a saved deployment bookmark before its first launch."""
        # The keys the DeploymentPage editor submits on the manager-backed
        # contract are translated below so an unchanged detail form saves
        # cleanly against a bookmark that Manager has never seen.
        allowed = {
            "context_length", "tensor_parallel_size", "parallel_slots",
            "gpu_layers", "quantization", "artifact", "image", "extra_args",
            "gpu_memory_utilization", "node_ids", "deployment_mode",
            "launch_controls", "gpu_memory_gb",
            "sg_tp_size", "sg_mem_fraction", "alias",
            "environment",
        }
        unknown = sorted(set(changes) - allowed)
        if unknown:
            raise ValueError(f"unsupported field(s): {', '.join(unknown)}")
        alias = _optional_string(changes.get("alias")) or str(stored.get("alias"))
        if alias != stored.get("alias"):
            existing = self.store.deployment(alias)
            if existing and existing["id"] != stored["id"]:
                raise ValueError(f"deployment alias '{alias}' is already in use")
        settings = dict(stored.get("settings") or {})
        runtime_is_llama = str(stored.get("runtime")) == RuntimeKind.LLAMA_CPP.value
        if "environment" in changes:
            settings["environment"] = normalize_runtime_environment(
                changes.get("environment"), str(stored.get("runtime") or "vllm"),
            )
        if "image" in changes:
            image = _optional_string(changes.get("image"))
            if not runtime_is_llama and not image:
                raise ValueError("image must be a non-empty container image")
            settings["image"] = image
        integer_fields = (
            "context_length", "tensor_parallel_size", "parallel_slots",
        )
        for field in integer_fields:
            if field not in changes:
                continue
            value = changes.get(field)
            if value is None:
                settings[field] = None
                continue
            settings[field] = self._validated_positive_int_setting(
                field, value, minimum=256 if field == "context_length" else 1,
            )
        if "gpu_layers" in changes:
            value = changes.get("gpu_layers")
            if value is None:
                settings["gpu_layers"] = None
            else:
                number = self._validated_number("gpu_layers", value)
                if number != int(number):
                    raise ValueError("gpu_layers must be a whole number")
                if number < 0:
                    raise ValueError(
                        "gpu_layers must be zero or a positive integer"
                    )
                settings["gpu_layers"] = int(number)
        if "gpu_memory_gb" in changes:
            value = changes.get("gpu_memory_gb")
            if value is None:
                settings["gpu_memory_gb"] = None
            else:
                # Fractional reservations are valid: DeploymentPage edits this
                # field in tenths and Manager consumes it as a float.
                number = self._validated_number("gpu_memory_gb", value)
                if number <= 0:
                    raise ValueError("gpu_memory_gb must be positive")
                settings["gpu_memory_gb"] = number
        # Stored values created through the raw API can be numeric strings or
        # other shapes; normalize them through the same validator so the
        # minimum comparisons below never raise a TypeError-shaped 500.
        for field, minimum in (
            ("context_length", 256),
            ("tensor_parallel_size", 1),
            ("parallel_slots", 1),
        ):
            stored_value = settings.get(field)
            if stored_value is None:
                continue
            if self._validated_number(field, stored_value) < minimum:
                raise ValueError(f"{field} must be at least {minimum}")
        gpu_layers = settings.get("gpu_layers")
        if gpu_layers is not None:
            normalized_layers = self._validated_number("gpu_layers", gpu_layers)
            if normalized_layers != int(normalized_layers):
                raise ValueError("gpu_layers must be a whole number")
            if normalized_layers < 0:
                raise ValueError(
                    "gpu_layers must be zero or a positive integer"
                )
        gpu_memory_gb = settings.get("gpu_memory_gb")
        if gpu_memory_gb is not None and self._validated_number(
            "gpu_memory_gb", gpu_memory_gb,
        ) <= 0:
            raise ValueError("gpu_memory_gb must be positive")
        if "gpu_memory_utilization" in changes:
            value = changes.get("gpu_memory_utilization")
            if value is None:
                settings["gpu_memory_utilization"] = None
            elif (
                isinstance(value, (int, float)) and not isinstance(value, bool)
                and 0 < float(value) <= 1
            ):
                settings["gpu_memory_utilization"] = float(value)
            else:
                raise ValueError(
                    "gpu_memory_utilization must be between 0 and 1"
                )
        if "quantization" in changes:
            settings["quantization"] = canonical_quantization(
                changes.get("quantization")
            )
        if "artifact" in changes:
            artifact = _optional_string(changes.get("artifact"))
            if artifact and runtime_is_llama:
                try:
                    artifact_path = Path(artifact).expanduser()
                except (OSError, ValueError, RuntimeError) as exc:
                    raise ValueError(
                        "llama.cpp managed deployments require an existing local GGUF artifact"
                    ) from exc
                if artifact_path.is_absolute():
                    # Controller-local artifacts keep the same rule as
                    # creation: the file must already exist on the controller.
                    if not artifact_path.is_file():
                        raise ValueError(
                            "llama.cpp managed deployments require an existing local GGUF artifact"
                        )
                    artifact = str(artifact_path)
                else:
                    self._validate_public_gguf_artifact(
                        str((stored.get("model") or {}).get("repository") or ""),
                        artifact, settings.get("quantization"),
                    )
            settings["artifact"] = artifact
        if "launch_controls" in changes:
            controls = changes.get("launch_controls")
            if not isinstance(controls, dict):
                raise ValueError("launch_controls must be an object")
            context_window = controls.get("context_window")
            if context_window not in (None, ""):
                try:
                    settings["context_length"] = int(context_window)
                except (TypeError, ValueError) as exc:
                    raise ValueError("context_window must be an integer") from exc
            elif "context_window" in controls:
                # An explicit clear must also drop the saved scalar, or the
                # launch keeps the stale context length the editor removed.
                settings["context_length"] = None
            max_concurrency = controls.get("max_concurrency")
            if max_concurrency not in (None, ""):
                target = "parallel_slots" if runtime_is_llama else "max_running_requests"
                try:
                    settings[target] = int(max_concurrency)
                except (TypeError, ValueError) as exc:
                    raise ValueError("max_concurrency must be an integer") from exc
            elif "max_concurrency" in controls:
                settings["parallel_slots" if runtime_is_llama else "max_running_requests"] = None
            # Saved bookmarks never pass Manager's preflight until their first
            # Run, so the parallel topology controls must be validated and
            # normalized here instead of persisting an invalid layout.
            for key in ("tensor_parallel_size", "pipeline_parallel_size"):
                value = controls.get(key)
                if value in (None, ""):
                    continue
                if isinstance(value, bool) or (
                    isinstance(value, float) and not value.is_integer()
                ):
                    raise ValueError(f"{key} must be a positive integer")
                try:
                    parsed = int(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{key} must be a positive integer") from exc
                if parsed <= 0:
                    raise ValueError(f"{key} must be a positive integer")
                controls[key] = parsed
            # Persist the complete structured contract: Manager's preflight
            # merges every control (KV dtype, thinking, speculative tokens,
            # cudagraph size, batched tokens) into the launch argv for vLLM
            # and SGLang, including clearing a previously set value.
            settings["launch_controls"] = controls
            # Keep the saved topology scalar in sync so the launch picker's
            # node-count contract reflects the edited parallel layout.
            tp_control = controls.get("tensor_parallel_size")
            if isinstance(tp_control, int):
                settings["tensor_parallel_size"] = tp_control
            elif "tensor_parallel_size" in controls and str(
                stored.get("runtime")
            ) == RuntimeKind.VLLM.value:
                # An explicit clear drops the saved scalar too, or the layout
                # contract keeps demanding the stale topology. SGLang editors
                # always submit a null here; their scalar is sg-driven.
                settings["tensor_parallel_size"] = None
        if "sg_tp_size" in changes:
            settings["tensor_parallel_size"] = changes["sg_tp_size"]
        if "sg_mem_fraction" in changes:
            settings["mem_fraction_static"] = changes["sg_mem_fraction"]
        if "extra_args" in changes:
            extra_args = changes.get("extra_args")
            if not isinstance(extra_args, list) or any(
                not isinstance(item, str) for item in extra_args
            ):
                raise ValueError("extra_args must be an array of strings")
            self._reject_sensitive_launch_args(extra_args)
            settings["extra_args"] = extra_args
        if "node_ids" in changes:
            node_ids = changes.get("node_ids")
            if node_ids is not None:
                if (
                    not isinstance(node_ids, list) or not node_ids
                    or any(not isinstance(item, str) or not item.strip()
                           for item in node_ids)
                ):
                    raise ValueError("node_ids must contain non-empty node IDs")
                node_ids = list(dict.fromkeys(item.strip() for item in node_ids))
                await self.manager.selected_cluster_nodes(node_ids)
            settings["node_ids"] = node_ids
        if "deployment_mode" in changes:
            mode = changes.get("deployment_mode")
            if mode is not None and mode not in {"single", "replicated", "sharded"}:
                raise ValueError(
                    "deployment_mode must be single, sharded, or replicated"
                )
            settings["deployment_mode"] = mode
        # Validate the effective combination: locality of the final artifact
        # against the final nodes, and the artifact/quantization pair so a
        # partial update cannot relabel a saved GGUF with a mismatching
        # precision.
        effective_nodes = [
            str(item).strip() for item in settings.get("node_ids") or []
            if str(item).strip()
        ]
        if runtime_is_llama:
            effective_artifact = _optional_string(
                settings.get("artifact")
                or (stored.get("model") or {}).get("artifact")
            )
            if effective_artifact:
                if _artifact_is_controller_local(effective_artifact):
                    if any(node_id != LOCAL_NODE_ID for node_id in effective_nodes):
                        raise ValueError(
                            "local GGUF artifact files live outside the cluster cache "
                            "and cannot be distributed; select the controller only or "
                            "use a repo-relative Hub artifact"
                        )
                else:
                    self._validate_public_gguf_artifact(
                        str((stored.get("model") or {}).get("repository") or ""),
                        effective_artifact, settings.get("quantization"),
                    )
        # Validate the effective combination: the saved mode plus the new
        # nodes (or the new mode plus the saved nodes) must stay launchable.
        if str(stored.get("runtime")) == RuntimeKind.LLAMA_CPP.value and (
            settings.get("deployment_mode") == "sharded"
        ):
            raise ValueError(
                "llama.cpp deployments support single and replicated layouts, not sharded"
            )
        contract = self._saved_layout_contract(
            settings, str(stored.get("runtime") or ""),
        )
        if contract["deployment_mode"] == "single" and len(effective_nodes) > 1:
            raise ValueError("single deployment requires exactly one node")
        if contract["deployment_mode"] == "sharded" and len(effective_nodes) < 2:
            raise ValueError("sharded deployment requires at least two nodes")
        if alias != stored.get("alias"):
            self.store.update_saved_deployment_settings(
                stored["id"], self._local_configuration(settings), alias,
            )
        else:
            self.store.update_managed_routing(
                stored["id"],
                self._local_configuration(settings),
                stored.get("container_name"),
                stored.get("_base_url"),
            )
        model = dict(stored.get("model") or {})
        if "quantization" in changes:
            model["quantization"] = settings.get("quantization")
        if "artifact" in changes:
            model["artifact"] = settings.get("artifact")
        if model != (stored.get("model") or {}):
            self.store.update_deployment_model(stored["id"], model)
        return await self.deployment_detail(stored["id"])

    def _validate_public_gguf_artifact(
        self, repository: str, artifact: str, quantization: str | None,
    ) -> PurePosixPath:
        """Validate one repo-relative GGUF reference without touching the cache."""
        if _public_model_id(repository) != repository:
            raise ValueError(
                "repo-relative GGUF artifacts require a public Hugging Face repository"
            )
        relative = PurePosixPath(artifact)
        if (
            relative.is_absolute() or not relative.parts
            or any(part in {"", ".", ".."} for part in relative.parts)
            or artifact.startswith("~")
            or "\\" in artifact
            or re.match(r"^[A-Za-z]:", artifact)
            or relative.suffix.casefold() != ".gguf"
        ):
            raise ValueError("artifact must be a safe repo-relative .gguf filename")
        inferred = quantization_from_text(artifact)
        if quantization and inferred and quantization != inferred:
            raise ValueError("artifact quantization does not match the selected quantization")
        return relative

    @staticmethod
    def _expand_gguf_shard_files(relative: PurePosixPath) -> list[str]:
        """Expand one shard reference into its complete ordered shard set."""
        shard = _PUBLIC_GGUF_SHARD_PATTERN.match(relative.name)
        if not shard:
            return [relative.as_posix()]
        shard_count = int(shard.group("count"))
        return [
            str(relative.with_name(
                f"{shard.group('stem')}-{index:05d}"
                f"{shard.group('separator')}{shard_count:05d}"
                f"{shard.group('suffix')}"
            ))
            for index in range(1, shard_count + 1)
        ]

    async def _resolved_model_revision(
        self, repository: str, revision: str | None,
    ) -> str:
        """Resolve one public repository reference to an immutable commit SHA."""
        virtual_nas = getattr(self.manager, "virtual_nas", None)
        if virtual_nas is None:
            raise RuntimeError("model preparation is unavailable")
        resolution = await virtual_nas.resolve_download_revision(
            repository, revision or "main",
        )
        resolved_revision = str(resolution.get("resolved_revision") or "")
        if not re.fullmatch(r"[0-9a-f]{40}", resolved_revision):
            raise RuntimeError("model preparation did not resolve an immutable revision")
        return resolved_revision

    def _hub_relative_llama_artifact(
        self, repository: str, artifact: str, resolved_revision: str,
    ) -> str:
        """Return the cache-relative snapshot path for one GGUF artifact.

        Llama.cpp cluster members resolve this reference against each node's
        own Hugging Face cache, so one persisted value addresses every node.
        """
        relative = self._validate_public_gguf_artifact(repository, artifact, None)
        encoded = "models--" + repository.replace("/", "--")
        first_file = self._expand_gguf_shard_files(relative)[0]
        return f"{encoded}/snapshots/{resolved_revision}/{first_file}"

    def _llama_cpp_artifact_homes(
        self, requested_node_ids: list[str] | None, body: dict[str, Any],
    ) -> tuple[list[str] | None, str | None]:
        """Resolve GGUF home nodes and the optional Hub seed for a launch.

        Every selected node becomes a home that will hold the artifact (this
        PR generalizes llama.cpp beyond the controller, so a remote-only set
        is valid). The seed is validated here so the controller-only fast
        path cannot silently ignore a seed that names a node outside the
        selection.
        """
        requested_seed = _optional_string(body.get("download_node_id"))
        if requested_node_ids is None:
            if requested_seed is not None and requested_seed != LOCAL_NODE_ID:
                raise ValueError("download_node_id must be one of the selected nodes")
            return None, requested_seed
        if requested_seed is not None and requested_seed not in requested_node_ids:
            raise ValueError("download_node_id must be one of the selected nodes")
        return list(requested_node_ids), requested_seed

    async def _prepare_public_gguf_artifact(
        self, repository: str, artifact: str, revision: str,
        quantization: str | None,
        home_node_ids: list[str] | None = None,
        download_node_id: str | None = None,
    ) -> str:
        """Prepare one repo-relative GGUF through the existing Virtual NAS cache."""
        relative = self._validate_public_gguf_artifact(repository, artifact, quantization)

        resolved_revision = await self._resolved_model_revision(repository, revision)
        selected_files = self._expand_gguf_shard_files(relative)
        virtual_nas = self.manager.virtual_nas
        if home_node_ids and set(home_node_ids) - {LOCAL_NODE_ID}:
            await self._distribute_gguf_artifact(
                repository, resolved_revision, selected_files,
                home_node_ids, download_node_id, revision,
            )
        else:
            await virtual_nas.download_model_files_checked(
                repository, resolved_revision, selected_files,
                requested_revision=revision,
            )
        model_root = virtual_nas._model_path(repository).resolve()
        snapshot_root = model_root / "snapshots" / resolved_revision
        candidate = snapshot_root
        for part in relative.parts:
            candidate = candidate / part

        def validate_artifact_path(logical_path: Path, missing_message: str) -> None:
            try:
                resolved_path = logical_path.resolve(strict=True)
                allowed_root = (
                    (model_root / "blobs").resolve(strict=True)
                    if logical_path.is_symlink()
                    else snapshot_root.resolve(strict=True)
                )
                resolved_path.relative_to(allowed_root)
            except (OSError, ValueError) as exc:
                raise RuntimeError(missing_message) from exc
            if not logical_path.is_file():
                raise RuntimeError(missing_message)

        validate_artifact_path(
            candidate,
            "model preparation completed without the selected GGUF artifact",
        )
        if _PUBLIC_GGUF_SHARD_PATTERN.match(relative.name):
            for selected_file in selected_files:
                selected_relative = PurePosixPath(selected_file)
                logical_shard = snapshot_root.joinpath(*selected_relative.parts)
                validate_artifact_path(
                    logical_shard,
                    "model preparation completed without the complete selected GGUF shard set",
                )
        # Preserve the logical snapshot filename. Hugging Face cache entries
        # are normally symlinks into blobs/, whose content-addressed targets
        # have no .gguf suffix and do not retain multi-shard names.
        return str(candidate)

    @staticmethod
    def _validated_number(field: str, value: Any) -> float:
        """Coerce editor/API numeric input to a finite float or raise 400."""
        if isinstance(value, bool):
            raise ValueError(f"{field} must be a number")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be a number") from exc
        if not math.isfinite(number):
            raise ValueError(f"{field} must be a finite number")
        return number

    def _validated_positive_int_setting(
        self, field: str, value: Any, minimum: int = 1,
    ) -> int:
        number = self._validated_number(field, value)
        if number != int(number):
            raise ValueError(f"{field} must be a whole number")
        integer = int(number)
        if integer < minimum:
            raise ValueError(f"{field} must be at least {minimum}")
        return integer

    def _saved_bookmark_launch_controls(
        self, runtime: str, extra_args: list[str],
        saved_settings: dict[str, Any],
    ) -> dict[str, Any]:
        """Round-trip controls for a saved bookmark's editor.

        Order matters: controls parsed from the saved argv are authoritative,
        the structured controls saved through this editor come next, and the
        bookmark's scalar settings only seed keys nothing else provides — an
        unconditional seed would blank parsed values like --max-model-len and
        strip them from the next launch.
        """
        controls = self.manager._deployment_launch_controls({
            "engine": runtime,
            "extra_args": extra_args,
            "environment": saved_settings.get("environment") or {},
        })
        saved_controls = saved_settings.get("launch_controls") or {}
        controls.update(saved_controls)
        scalar_seeds = {
            "context_window": saved_settings.get("context_length"),
            "max_concurrency": (
                saved_settings.get("parallel_slots")
                if runtime == RuntimeKind.LLAMA_CPP.value
                else saved_settings.get("max_running_requests")
            ),
        }
        if runtime == RuntimeKind.VLLM.value:
            # The Models page saves sharded vLLM bookmarks with a bare
            # tensor_parallel_size scalar; seed the editor with it so an
            # unchanged Save cannot submit a null that strips the TP flag.
            scalar_seeds["tensor_parallel_size"] = saved_settings.get(
                "tensor_parallel_size"
            )
        for key, value in scalar_seeds.items():
            # A key present in the saved controls is an explicit editor value
            # — including an intentional clear — and must not be re-seeded.
            if key in saved_controls:
                continue
            if controls.get(key) is None and value is not None:
                controls[key] = value
        return controls

    @staticmethod
    def _saved_layout_contract(
        settings: dict[str, Any], runtime: str | None = None,
    ) -> dict[str, Any]:
        """Derive the launch layout contract persisted on a saved bookmark.

        Mirrors Manager's contract (replicated = the saved node count, sharded
        = the tensor-parallel node count) so launch pickers can enforce the
        required node count before the deployment exists in Manager.
        """
        node_ids = [
            str(item).strip() for item in settings.get("node_ids") or []
            if str(item).strip()
        ]
        mode = str(settings.get("deployment_mode") or "").strip() or (
            "replicated" if len(node_ids) > 1 else "single"
        )
        if mode == "sharded":
            controls = settings.get("launch_controls")
            if not isinstance(controls, dict):
                controls = {}

            def _positive_parallel(value: Any) -> int:
                return (
                    value
                    if isinstance(value, int) and not isinstance(value, bool)
                    and value > 0
                    else 1
                )

            tensor = _positive_parallel(controls.get("tensor_parallel_size"))
            if tensor == 1:
                tensor = _positive_parallel(settings.get("tensor_parallel_size"))
            world = tensor * _positive_parallel(
                controls.get("pipeline_parallel_size")
            )
            if world > 1:
                # Mirror Manager's launch gate: vLLM may place several ranks
                # per saved node when TP x PP divides the node count. SGLang
                # continues to require one selected host per rank.
                count = (
                    len(node_ids)
                    if runtime == RuntimeKind.VLLM.value
                    and len(node_ids) > 1 and world % len(node_ids) == 0
                    else world
                )
            else:
                count = max(2, len(node_ids))
        elif mode == "replicated":
            count = max(2, len(node_ids))
        else:
            mode = "single"
            count = 1
        return {"deployment_mode": mode, "required_node_count": count}

    def _reject_sensitive_launch_args(self, extra_args: Any) -> None:
        """Apply Manager's credential-argv policy before anything is saved."""
        reject = getattr(self.manager, "_reject_sensitive_cli_credentials", None)
        if callable(reject):
            reject(extra_args)

    async def _distribute_gguf_artifact(
        self, repository: str, resolved_revision: str, selected_files: list[str],
        home_node_ids: list[str], download_node_id: str | None,
        requested_revision: str,
    ) -> None:
        """Place the selected GGUF files on every home node with one Hub pull.

        Nodes that already hold the artifact act as streaming sources; when
        no node has it, exactly one home node seeds from Hugging Face and
        the rest receive file-scoped Virtual NAS streams, so the cluster
        never pays duplicate Hub bandwidth for the same artifact. The
        streams deliberately bypass the whole-repository transfer jobs:
        selective snapshots are not complete revisions, and targets may
        already cache the same repository under another quantization.
        """
        await self.manager.selected_cluster_nodes(home_node_ids)
        if download_node_id is not None and download_node_id not in home_node_ids:
            raise ValueError("download_node_id must be one of the selected nodes")
        if not self.manager.virtual_nas.enabled:
            raise RuntimeError(
                "Virtual NAS is required to distribute GGUF artifacts between nodes"
            )
        # Validate the whole set before any Hub download starts so a mixed
        # cluster fails fast instead of after creating partial side effects.
        unsupported = [
            node_id for node_id in home_node_ids
            if node_id != LOCAL_NODE_ID
            and not await self.manager.node_supports_selective_downloads(node_id)
        ]
        if unsupported:
            raise RuntimeError(
                "node(s) do not support selective model file transfers; "
                "update their SparkDeck agent: " + ", ".join(unsupported)
            )
        complete: dict[str, bool] = {}
        for node_id in home_node_ids:
            try:
                complete[node_id] = await self.manager.node_has_model_files(
                    node_id, repository, resolved_revision, selected_files,
                )
            except Exception:
                # A node that cannot answer the presence check (for example a
                # pre-selective-download agent) is treated as incomplete so
                # preparation falls back to copying or re-seeding instead of
                # silently assuming the artifact exists.
                complete[node_id] = False
        targets = [node_id for node_id in home_node_ids if not complete[node_id]]
        if not targets:
            return
        sources = [node_id for node_id in home_node_ids if complete[node_id]]
        if sources:
            source = download_node_id if download_node_id in sources else sources[0]
            for node_id in targets:
                await self.manager.node_transfer_model_files(
                    source, node_id, repository, resolved_revision,
                    selected_files, requested_revision,
                )
            return
        candidates = [download_node_id] if download_node_id else [
            node_id for node_id in home_node_ids
            if await self.manager.node_supports_selective_downloads(node_id)
        ]
        failures: list[str] = []
        for candidate in candidates:
            try:
                await self.manager.node_download_model_files(
                    candidate, repository, resolved_revision, selected_files,
                    requested_revision,
                )
            except Exception as exc:
                failures.append(f"{candidate}: {exc}")
                continue
            for node_id in targets:
                if node_id == candidate:
                    continue
                await self.manager.node_transfer_model_files(
                    candidate, node_id, repository, resolved_revision,
                    selected_files, requested_revision,
                )
            return
        raise RuntimeError(
            "no selected node could download the GGUF artifact: "
            + "; ".join(failures)
        )

    async def create_deployment(
        self, body: dict[str, Any], *, launch: bool = False,
        background: bool = False,
    ) -> dict[str, Any]:
        model = str(body.get("model") or "").strip()
        alias = str(body.get("alias") or model).strip()
        if not model:
            raise ValueError("model is required")
        if not alias:
            raise ValueError("alias is required")
        runtime = RuntimeKind(str(body.get("runtime") or "vllm"))
        kind = DeploymentKind(str(body.get("kind") or ("external" if body.get("base_url") else "managed")))
        settings = dict(body.get("settings") or {})
        # Runtime provenance is derived from the resolved launch input below;
        # callers cannot promote a local model to public benchmark evidence.
        settings.pop("model_source", None)
        # Saved argv persists in SQLite, so credential-bearing flags are
        # rejected at save time instead of failing (or leaking) at launch.
        self._reject_sensitive_launch_args(settings.get("extra_args"))
        environment = normalize_runtime_environment(
            settings.get("environment"), runtime.value,
        )
        if environment:
            settings["environment"] = environment
        else:
            settings.pop("environment", None)
        artifact = _optional_string(body.get("artifact") or settings.get("artifact"))
        quantization = canonical_quantization(
            body.get("quantization") or settings.get("quantization")
        )
        requested_node_ids = _requested_node_ids(body)
        deployment_mode = str(body.get("deployment_mode") or "").strip() or None
        if requested_node_ids is not None and kind is DeploymentKind.EXTERNAL:
            raise ValueError("node_ids are only supported for managed deployments")
        deployment_id = str(uuid.uuid4())
        # A launch can take minutes, but serializing creation is intentional:
        # alias uniqueness must be established before Docker is mutated.
        async with self._deployment_create_lock:
            if self.store.deployment(alias):
                raise ValueError(f"deployment alias '{alias}' is already in use")
            artifact_is_local = False
            model_is_local_path = False
            artifact_homes: list[str] | None = None
            artifact_seed: str | None = None
            if runtime is RuntimeKind.LLAMA_CPP and kind is DeploymentKind.MANAGED:
                artifact_homes, artifact_seed = self._llama_cpp_artifact_homes(
                    requested_node_ids, body,
                )
            resolve_local = getattr(self.manager, "_resolve_local_path", None)
            model_is_local_path = bool(
                resolve_local and resolve_local(model)
            )
            if (
                runtime is RuntimeKind.LLAMA_CPP
                and kind is DeploymentKind.MANAGED
                and artifact
            ):
                # Classify the caller's raw artifact. A repo-relative Hub path
                # such as ~/quantized/model.gguf must not become controller-
                # local merely because the same path exists under its home,
                # while a genuine tilde reference is expanded and must name a
                # real file to count as controller-local.
                try:
                    artifact_path = Path(artifact).expanduser()
                except (OSError, ValueError, RuntimeError) as exc:
                    raise ValueError(
                        "llama.cpp managed deployments require an existing local GGUF artifact"
                    ) from exc
                if artifact_path.is_absolute():
                    if not artifact_path.is_file():
                        raise ValueError(
                            "llama.cpp managed deployments require an existing local GGUF artifact"
                        )
                    if artifact_homes and any(
                        node_id != LOCAL_NODE_ID for node_id in artifact_homes
                    ):
                        raise ValueError(
                            "local GGUF artifact files live outside the cluster cache "
                            "and cannot be distributed; select the controller only or "
                            "use a repo-relative Hub artifact"
                        )
                    artifact_is_local = True
                    settings["model_source"] = "local"
                    # Persist the expanded absolute path so every later
                    # classification sees one unambiguous controller-local
                    # reference instead of a tilde shorthand.
                    artifact = str(artifact_path)
                else:
                    # A saved deployment only records the reference; the GGUF
                    # is resolved (and downloaded if needed) at launch time.
                    self._validate_public_gguf_artifact(model, artifact, quantization)
                    if launch:
                        artifact = await self._prepare_public_gguf_artifact(
                            model, artifact,
                            _optional_string(body.get("revision")) or "main",
                            quantization,
                            home_node_ids=artifact_homes,
                            download_node_id=artifact_seed,
                        )
                        settings["model_source"] = "public_repository"
                if artifact_seed is not None:
                    settings["download_node_id"] = artifact_seed
                settings["artifact"] = artifact
                if quantization:
                    settings["quantization"] = quantization
            identity = ModelIdentity(
                repository=model, revision=_optional_string(body.get("revision")),
                artifact=artifact,
                quantization=quantization,
            )
            deployment = Deployment(
                id=deployment_id, alias=alias, runtime=runtime, kind=kind,
                model=identity, settings=self._local_configuration(settings),
                base_url_set=bool(body.get("base_url")),
            )
            if kind is DeploymentKind.EXTERNAL:
                base_url = self._validate_base_url(body.get("base_url"))
                credential_ref = self._store_credential(deployment_id, body.get("api_key"))
                try:
                    self.store.add_deployment(deployment, base_url, credential_ref)
                except Exception:
                    self._delete_credential(deployment_id, credential_ref)
                    raise
                return (self.store.deployment(deployment_id) or deployment.to_dict())

            if requested_node_ids is not None:
                if any(item != "local" for item in requested_node_ids):
                    # Controller-local models can never run remotely, whatever
                    # the runtime: reject the mixed selection at save time
                    # instead of producing an unlaunchable bookmark.
                    if model_is_local_path:
                        raise ValueError(
                            "local model paths can only be saved for the controller node"
                        )
                    if runtime is RuntimeKind.LLAMA_CPP and (
                        artifact_is_local
                        or (not artifact and _public_model_id(model) == "local-model")
                    ):
                        raise ValueError(
                            "local GGUF artifacts can only be saved for the controller node"
                        )
                selected = await self.manager.selected_cluster_nodes(requested_node_ids)
                mode = deployment_mode or (
                    "replicated" if len(requested_node_ids) > 1 else "single"
                )
                if mode == "single" and len(requested_node_ids) != 1:
                    raise ValueError("single deployment requires exactly one node")
                if mode == "sharded" and runtime is RuntimeKind.LLAMA_CPP:
                    raise ValueError(
                        "llama.cpp deployments support single and replicated layouts, not sharded"
                    )
                if not launch:
                    # A saved deployment is a launch bookmark: persist the
                    # runtime, model, settings, and node preferences without
                    # mutating Docker or the cluster. Launch happens through
                    # the explicit start action.
                    deployment.desired_state = "stopped"
                    deployment.settings = self._local_configuration({
                        **settings,
                        "deployment_mode": mode,
                        "node_ids": requested_node_ids,
                    })
                    self.store.add_deployment(deployment, None, None)
                    result = self.store.deployment(deployment_id) or deployment.to_dict()
                    result.update({
                        "status": "saved",
                        "node_ids": requested_node_ids,
                        **self._saved_layout_contract({
                            "deployment_mode": mode,
                            "node_ids": requested_node_ids,
                        }, runtime.value),
                        "selected_nodes": [
                            self.manager.public_target_node(node) for node in selected
                        ],
                    })
                    return result
                launch_body = self._cluster_launch_body(
                    runtime, model, alias, deployment_id, identity, settings,
                    requested_node_ids, mode, llama_artifact=None,
                    recipe_id=body.get("recipe_id"),
                )
                if background:
                    return await self._begin_cluster_deployment(
                        deployment, settings, mode, requested_node_ids,
                        selected, launch_body,
                    )
                try:
                    cluster = await self.manager.create_deployment(launch_body)
                except Exception:
                    await self._recover_failed_cluster_launch(
                        deployment, settings, mode, requested_node_ids,
                    )
                    raise
                manager_id = cluster.get("id")
                members = cluster.get("members") or []
                primary = members[0] if members else {}
                deployment.container_name = primary.get("container_name")
                deployment.settings = self._local_configuration({
                    **settings,
                    "deployment_mode": mode,
                    "node_ids": requested_node_ids,
                    "manager_deployment_id": manager_id,
                    "model_source": cluster.get("model_source") or "unknown",
                })
                port = cluster.get("api_port")
                if not manager_id or not deployment.container_name or not port:
                    await self._recover_failed_cluster_launch(
                        deployment, settings, mode, requested_node_ids,
                    )
                    raise RuntimeError("cluster runtime launched without a discoverable endpoint")
                try:
                    self.store.add_deployment(
                        deployment, f"http://127.0.0.1:{int(port)}", None
                    )
                except Exception:
                    await self.manager.deployment_action(manager_id, "remove")
                    raise
                result = self.store.deployment(deployment_id) or deployment.to_dict()
                result.update({
                    "status": _deployment_status(cluster.get("status")),
                    "port": int(port),
                    "node_ids": requested_node_ids,
                    "selected_nodes": [
                        self.manager.public_target_node(node) for node in selected
                    ],
                })
                return result

            if not launch:
                # Controller-local bookmark without saved node preferences.
                # Everything about the record stays editable until the start
                # action launches it.
                deployment.desired_state = "stopped"
                self.store.add_deployment(deployment, None, None)
                result = self.store.deployment(deployment_id) or deployment.to_dict()
                result.update({"status": "saved"})
                return result

            adapter = self.registry.get(runtime)
            cleanup_name = safe_container_name(alias, deployment_id)
            # Persist ownership before Docker is mutated. If launch succeeds,
            # the record is filled with the discovered endpoint below. If
            # launch or cleanup fails, onboarding still has a durable signal
            # that this controller may own a managed workload.
            deployment.container_name = cleanup_name
            launch_complete = asyncio.Event()
            self._deployment_launches[deployment_id] = launch_complete
            try:
                self.store.add_deployment(deployment, None, None)
                try:
                    launched = await launch_managed_container(
                        self.manager, adapter, deployment_id, alias, model,
                        {
                            **settings,
                            "artifact": identity.artifact,
                            "revision": identity.revision,
                        },
                    )
                    deployment.container_name = launched.get("name")
                    deployment.settings = self._local_configuration({
                        **settings,
                        "model_source": (
                            settings.get("model_source")
                            or launched.get("model_source")
                            or "unknown"
                        ),
                    })
                    port = launched.get("port")
                    if not deployment.container_name or not port:
                        raise RuntimeError(
                            "runtime launched without a discoverable container endpoint"
                        )
                    cleanup_name = deployment.container_name
                    self.store.update_managed_routing(
                        deployment.id, deployment.settings, deployment.container_name,
                        f"http://127.0.0.1:{int(port)}",
                    )
                except Exception:
                    cleaned = False
                    try:
                        await self.manager.remove_container(cleanup_name)
                        cleaned = True
                    except Exception as cleanup_error:
                        cleaned = _is_missing_container_error(cleanup_error)
                    if cleaned:
                        self.store.delete_deployment(deployment_id)
                    raise
                result = self.store.deployment(deployment_id) or deployment.to_dict()
                result.update({
                    "status": launched.get("status", "running"), "port": int(port),
                })
                return result
            finally:
                launch_complete.set()
                self._deployment_launches.pop(deployment_id, None)

    def _cluster_launch_body(
        self, runtime: RuntimeKind, model: str, alias: str, deployment_id: str,
        identity: ModelIdentity, settings: dict[str, Any],
        node_ids: list[str], mode: str, llama_artifact: str | None,
        recipe_id: Any = None,
    ) -> dict[str, Any]:
        """Translate saved launch settings into a Manager cluster launch."""
        extra_args = list(settings.get("extra_args") or [])
        if runtime is RuntimeKind.VLLM:
            for key, flag in (
                ("tensor_parallel_size", "--tensor-parallel-size"),
                ("pipeline_parallel_size", "--pipeline-parallel-size"),
                ("quantization", "--quantization"),
                ("dtype", "--dtype"),
            ):
                if settings.get(key) is not None:
                    extra_args += [flag, str(settings[key])]
            context_length = settings.get("max_model_len") or settings.get("context_length")
            if context_length is not None:
                extra_args += ["--max-model-len", str(context_length)]
        if identity.revision and runtime is not RuntimeKind.LLAMA_CPP:
            # Llama.cpp pins its revision inside the cache-relative artifact
            # reference; an unknown --revision flag would break llama-server.
            extra_args += ["--revision", identity.revision]
        launch_body = {
            **settings,
            "model": model,
            "deployment_name": alias,
            "engine": runtime.value,
            "deployment_mode": mode,
            "node_ids": node_ids,
            "extra_args": extra_args,
            "managed_by": "sparkdeck",
            "sparkdeck_record_id": deployment_id,
            "recipe_id": recipe_id,
        }
        if runtime is RuntimeKind.SGLANG:
            launch_body.update({
                "sg_tp_size": settings.get("tensor_parallel_size"),
                "sg_context_length": settings.get("context_length"),
                "sg_max_running_requests": settings.get("max_running_requests"),
                "sg_mem_fraction": settings.get("mem_fraction_static"),
            })
            if (
                settings.get("data_parallel_size") is not None
                and not any(
                    str(arg) == "--dp-size"
                    or str(arg).startswith("--dp-size=")
                    for arg in launch_body["extra_args"]
                )
            ):
                launch_body["extra_args"] += [
                    "--dp-size", str(settings["data_parallel_size"]),
                ]
            if settings.get("quantization") is not None:
                launch_body["extra_args"] += [
                    "--quantization", str(settings["quantization"]),
                ]
        if runtime is RuntimeKind.LLAMA_CPP:
            launch_body.update({
                "llama_artifact": llama_artifact,
                "llama_context_length": settings.get("context_length"),
                "llama_parallel_slots": settings.get("parallel_slots"),
                "llama_gpu_layers": settings.get("gpu_layers"),
            })
        return launch_body

    def _saved_deployment_controller_only(
        self, runtime: RuntimeKind, model: str, artifact: str | None,
    ) -> bool:
        """True when the saved model can only ever run on the controller."""
        if runtime is not RuntimeKind.LLAMA_CPP:
            return False
        resolve_local = getattr(self.manager, "_resolve_local_path", None)
        if resolve_local and resolve_local(model):
            return True
        if artifact:
            return _artifact_is_controller_local(artifact)
        return _public_model_id(model) == "local-model"

    async def _launch_saved_deployment(
        self, deployment: dict[str, Any], node_ids: list[str] | None,
    ) -> dict[str, Any]:
        """Launch a saved deployment bookmark for the first time."""
        model = str((deployment.get("model") or {}).get("repository") or "")
        stored_model = deployment.get("model") or {}
        record = Deployment(
            id=deployment["id"],
            alias=deployment["alias"],
            runtime=RuntimeKind(str(deployment["runtime"])),
            kind=DeploymentKind(str(deployment["kind"])),
            model=ModelIdentity(
                repository=model,
                revision=_optional_string(stored_model.get("revision")),
                artifact=_optional_string(stored_model.get("artifact")),
                quantization=_optional_string(stored_model.get("quantization")),
            ),
            settings=dict(deployment.get("settings") or {}),
        )
        settings = record.settings
        artifact = str(record.model.artifact or settings.get("artifact") or "")
        if self._saved_deployment_controller_only(record.runtime, model, artifact):
            if any(item != "local" for item in node_ids or []):
                raise ValueError(
                    "controller-local model artifacts can only run on the controller node"
                )
            return await self._launch_controller_llama_deployment(
                record, settings, model, artifact,
            )
        if (
            record.runtime is RuntimeKind.LLAMA_CPP
            and not node_ids and not settings.get("node_ids")
        ):
            # No node preferences were saved: keep the llama.cpp bookmark
            # controller-local, matching deployments saved before node
            # preferences existed. Its GGUF is prepared on the controller.
            return await self._launch_controller_llama_deployment(
                record, settings, model, artifact,
            )
        return await self._launch_cluster_record(
            record, settings, model, artifact, node_ids,
        )

    async def _launch_controller_llama_deployment(
        self, record: Deployment, settings: dict[str, Any],
        model: str, artifact: str,
    ) -> dict[str, Any]:
        """Run a controller-local llama.cpp bookmark as a standalone container."""
        deployment_id = record.id
        adapter = self.registry.get(record.runtime)
        launch_settings = dict(settings)
        if artifact and not _artifact_is_controller_local(artifact):
            # Resolve (downloading if needed) the public GGUF on the controller.
            launch_settings["artifact"] = await self._prepare_public_gguf_artifact(
                model, artifact,
                record.model.revision or "main",
                record.model.quantization,
            )
        elif not artifact:
            raise ValueError("llama.cpp deployments require a GGUF artifact")
        cleanup_name = safe_container_name(record.alias, deployment_id)
        record.container_name = cleanup_name
        launch_complete = asyncio.Event()
        self._deployment_launches[deployment_id] = launch_complete
        try:
            launched = await launch_managed_container(
                self.manager, adapter, deployment_id, record.alias, model,
                {
                    **launch_settings,
                    "artifact": launch_settings.get("artifact") or artifact,
                    "revision": record.model.revision,
                },
            )
            record.container_name = launched.get("name")
            record.settings = self._local_configuration({
                **settings,
                "artifact": launch_settings.get("artifact") or artifact,
                "model_source": (
                    settings.get("model_source")
                    or launched.get("model_source")
                    or "unknown"
                ),
            })
            port = launched.get("port")
            if not record.container_name or not port:
                raise RuntimeError(
                    "runtime launched without a discoverable container endpoint"
                )
            self.store.update_managed_routing(
                record.id, record.settings, record.container_name,
                f"http://127.0.0.1:{int(port)}",
            )
        except Exception:
            try:
                await self.manager.remove_container(cleanup_name)
            except Exception as cleanup_error:
                if not _is_missing_container_error(cleanup_error):
                    raise
            raise
        finally:
            launch_complete.set()
            self._deployment_launches.pop(deployment_id, None)
        self.store.update_desired_state(deployment_id, "running")
        current = self.store.deployment(deployment_id) or record.to_dict()
        current.update({
            "status": launched.get("status", "running"), "port": int(port),
        })
        return current

    async def _launch_cluster_record(
        self, record: Deployment, settings: dict[str, Any],
        model: str, artifact: str, node_ids: list[str] | None,
    ) -> dict[str, Any]:
        """Launch a saved deployment as a Manager cluster on the given nodes."""
        saved_nodes = [
            str(item).strip() for item in settings.get("node_ids") or []
            if str(item).strip()
        ]
        selected_ids = list(node_ids) if node_ids else (saved_nodes or ["local"])
        selected_ids = list(dict.fromkeys(selected_ids))
        selected = await self.manager.selected_cluster_nodes(selected_ids)
        mode = str(settings.get("deployment_mode") or "").strip() or (
            "replicated" if len(selected_ids) > 1 else "single"
        )
        if mode == "single" and len(selected_ids) != 1:
            raise ValueError("single deployment requires exactly one node")
        if mode == "sharded" and record.runtime is RuntimeKind.LLAMA_CPP:
            raise ValueError(
                "llama.cpp deployments support single and replicated layouts, not sharded"
            )
        deployment_dict = record.to_dict()
        deployment_dict["settings"] = settings
        if record.runtime is not RuntimeKind.LLAMA_CPP:
            # Llama.cpp readiness is per-file inside the resolved snapshot and
            # is verified by each node when its container is created; the
            # whole-repository inventory check would reject selective GGUF
            # snapshots that are perfectly launchable.
            await self._validate_start_selection(deployment_dict, selected_ids, None)
        llama_artifact = None
        if record.runtime is RuntimeKind.LLAMA_CPP:
            if not artifact:
                raise ValueError("llama.cpp deployments require a GGUF artifact")
            prepared = _optional_string(settings.get("prepared_revision"))
            if prepared and re.fullmatch(r"[0-9a-f]{40}", prepared):
                # Reuse the exact snapshot weight preparation resolved so a
                # repository update cannot invalidate a just-prepared launch.
                resolved_revision = prepared
            else:
                resolved_revision = await self._resolved_model_revision(
                    model, record.model.revision or "main",
                )
            llama_artifact = self._hub_relative_llama_artifact(
                model, artifact, resolved_revision,
            )
        launch_body = self._cluster_launch_body(
            record.runtime, model, record.alias, record.id, record.model,
            settings, selected_ids, mode, llama_artifact=llama_artifact,
        )
        try:
            cluster = await self.manager.create_deployment(launch_body)
        except Exception:
            await self._recover_failed_cluster_launch(
                record, settings, mode, selected_ids,
            )
            raise
        self._link_cluster_record(record, settings, mode, selected_ids, cluster)
        self.store.update_desired_state(record.id, "running")
        current = self.store.deployment(record.id) or record.to_dict()
        current.update({
            "status": _deployment_status(cluster.get("status")),
            "port": cluster.get("api_port"),
            "node_ids": selected_ids,
            "selected_nodes": [
                self.manager.public_target_node(node) for node in selected
            ],
        })
        return current

    def _link_cluster_record(
        self, deployment: Deployment, settings: dict[str, Any], mode: str,
        node_ids: list[str], cluster: dict[str, Any],
    ) -> None:
        """Persist the public record's route to a durable Manager launch."""
        members = cluster.get("members") or []
        primary = members[0] if members else {}
        deployment.container_name = primary.get("container_name")
        deployment.settings = self._local_configuration({
            **settings,
            "deployment_mode": mode,
            "node_ids": node_ids,
            "manager_deployment_id": cluster.get("id"),
            "model_source": cluster.get("model_source") or "unknown",
        })
        port = cluster.get("api_port")
        self.store.update_managed_routing(
            deployment.id,
            deployment.settings,
            deployment.container_name,
            f"http://127.0.0.1:{int(port)}" if port else None,
        )

    async def _begin_cluster_deployment(
        self, deployment: Deployment, settings: dict[str, Any], mode: str,
        node_ids: list[str], selected: list[dict[str, Any]],
        launch_body: dict[str, Any],
    ) -> dict[str, Any]:
        """Persist a public card, then continue the slow launch in background."""
        launch_complete = asyncio.Event()
        self.store.add_deployment(deployment, None, None)
        self._deployment_launches[deployment.id] = launch_complete
        persisted = asyncio.get_running_loop().create_future()

        def consume_persisted_error(future: asyncio.Future) -> None:
            if not future.cancelled():
                future.exception()

        persisted.add_done_callback(consume_persisted_error)

        async def launch() -> None:
            try:
                cluster = await self.manager.create_deployment(
                    launch_body, launch_persisted=persisted,
                )
                self._link_cluster_record(
                    deployment, settings, mode, node_ids, cluster,
                )
            except BaseException:
                # Manager persists node-specific launch failures. Once linked,
                # retaining the SQLite row makes that diagnostic durable and
                # visible in Deployments. A preflight failure is handled by
                # the foreground waiter below before any response is returned.
                linked = any(
                    item.get("sparkdeck_record_id") == deployment.id
                    for item in getattr(self.manager, "deployments", [])
                    if isinstance(item, dict)
                )
                if not linked:
                    self.store.delete_deployment(deployment.id)
            finally:
                launch_complete.set()
                self._deployment_launches.pop(deployment.id, None)
                self._deployment_launch_tasks.pop(deployment.id, None)

        task = asyncio.create_task(
            launch(), name=f"sparkdeck-deploy-{deployment.id}",
        )
        self._deployment_launch_tasks[deployment.id] = task
        try:
            cluster = await asyncio.shield(persisted)
        except asyncio.CancelledError:
            # The accepted operation belongs to the service, not the client
            # connection. Keep the public row and retained launch task alive.
            raise
        except BaseException:
            await task
            raise

        self._link_cluster_record(
            deployment, settings, mode, node_ids, cluster,
        )
        result = self.store.deployment(deployment.id) or deployment.to_dict()
        result.update({
            "status": "starting",
            "port": cluster.get("api_port"),
            "node_ids": node_ids,
            "selected_nodes": [
                self.manager.public_target_node(node) for node in selected
            ],
            **_deployment_launch_progress(cluster),
        })
        return result

    async def _recover_failed_cluster_launch(
        self, deployment: Deployment, settings: dict[str, Any],
        mode: str, node_ids: list[str],
    ) -> None:
        """Remove, or durably adopt, a Manager record created before failure."""
        failed = next(
            (
                item for item in getattr(self.manager, "deployments", [])
                if isinstance(item, dict)
                and item.get("sparkdeck_record_id") == deployment.id
            ),
            None,
        )
        if not failed or not failed.get("id"):
            return
        try:
            removed = await self.manager.deployment_action(failed["id"], "remove")
            if removed.get("ok"):
                return
        except Exception:
            pass

        # If cleanup cannot reach every selected node, retain an actionable
        # SparkDeck record instead of leaving an invisible Manager orphan.
        members = failed.get("members") or []
        primary = members[0] if members else {}
        deployment.container_name = primary.get("container_name")
        deployment.settings = self._local_configuration({
            **settings,
            "deployment_mode": mode,
            "node_ids": node_ids,
            "manager_deployment_id": failed["id"],
        })
        try:
            port = failed.get("api_port")
            self.store.add_deployment(
                deployment,
                f"http://127.0.0.1:{int(port)}" if port else None,
                None,
            )
        except Exception:
            # A saved deployment already owns its SQLite row; adopt the failed
            # Manager record into it so the diagnostic stays reachable.
            try:
                self.store.update_managed_routing(
                    deployment.id, deployment.settings,
                    deployment.container_name,
                    f"http://127.0.0.1:{int(port)}" if port else None,
                )
            except Exception:
                # Preserve the original launch error. Manager still retains
                # its diagnostic record if even local SQLite adoption fails.
                pass

    def _persisted_revision(self, launch_settings: Any) -> str | None:
        """Revision pinned in the persisted launch args, if any."""
        contract_fn = getattr(self.manager, "recipe_deployment_contract", None)
        if not contract_fn:
            return None
        try:
            contract = contract_fn(launch_settings or {})
        except Exception:
            return None
        if not isinstance(contract, dict):
            return None
        value = contract.get("model_revision")
        return str(value).strip() if isinstance(value, str) and value.strip() else None

    def _owning_cluster_deployment(self, container_name: str | None) -> dict[str, Any] | None:
        """Return the managed cluster deployment a container is a rank of."""
        if not container_name:
            return None
        return next(
            (
                item for item in getattr(self.manager, "deployments", [])
                if isinstance(item, dict) and item.get("id")
                and any(
                    isinstance(member, dict)
                    and member.get("container_name") == container_name
                    for member in (item.get("members") or [])
                )
            ),
            None,
        )

    def _layout_contract(self, launch_settings: Any) -> dict[str, Any]:
        """Best-effort persisted layout for a cluster deployment.

        Returns the layout and revision fields the Models picker must enforce
        when the manager can derive them from persisted launch settings;
        callers fall back to their own defaults when absent.
        """
        contract_fn = getattr(self.manager, "recipe_deployment_contract", None)
        if not contract_fn:
            return {}
        try:
            contract = contract_fn(launch_settings or {})
        except Exception:
            return {}
        if not isinstance(contract, dict):
            return {}
        result: dict[str, Any] = {}
        if isinstance(contract.get("deployment_mode"), str):
            result["deployment_mode"] = contract["deployment_mode"]
        count = contract.get("required_node_count")
        if isinstance(count, int) and not isinstance(count, bool):
            result["required_node_count"] = count
        revision = contract.get("model_revision")
        if isinstance(revision, str) and revision.strip():
            result["model_revision"] = revision.strip()
        return result

    def _normalized_start_selection(
        self, launch_settings: Any, node_ids: list[str],
    ) -> list[str]:
        """Enforce the saved topology before Manager removes existing ranks."""
        selected = list(node_ids)
        if len(set(selected)) != len(selected):
            raise ValueError("node_ids must not contain duplicates")
        contract = self._layout_contract(launch_settings)
        required = contract.get("required_node_count")
        if required is not None and len(selected) != required:
            raise ValueError(
                f"this deployment requires exactly {required} node(s)"
            )
        return selected

    async def _validate_start_selection(
        self, deployment: dict[str, Any], node_ids: list[str],
        launch_settings: dict[str, Any] | None,
    ) -> None:
        """Mirror the recipe deploy gate for an explicit start selection."""
        repository = str((deployment.get("model") or {}).get("repository") or "")
        resolve_local = getattr(self.manager, "_resolve_local_path", None)
        is_local_path = bool(repository and resolve_local and resolve_local(repository))
        # llama.cpp GGUFs that live in a node's Hugging Face cache run on any
        # prepared node; only artifacts at absolute controller-local paths are
        # pinned to the controller.
        artifact = str(
            (deployment.get("model") or {}).get("artifact")
            or (deployment.get("settings") or {}).get("artifact")
            or ""
        )
        controller_artifact = (
            deployment.get("runtime") == RuntimeKind.LLAMA_CPP.value
            and (not artifact or _artifact_is_controller_local(artifact))
        )
        if (is_local_path or controller_artifact) and any(
            item != "local" for item in node_ids
        ):
            raise ValueError(
                "controller-local model paths can only run on the controller node"
                if is_local_path else
                "llama.cpp model artifacts can only run on the controller node"
            )
        if is_local_path or controller_artifact or not repository:
            return
        # A recipe-launched cluster pins its revision in the persisted launch
        # args, not in the deployment identity — prefer that when present.
        revision = (
            (deployment.get("model") or {}).get("revision")
            or self._persisted_revision(launch_settings)
            or "main"
        )
        inventory = await self.manager.model_cache_inventory()
        nodes_with_weights = {
            node.get("id")
            for node in inventory if isinstance(node, dict)
            for model in node.get("models") or []
            if isinstance(model, dict)
            and not model.get("partial")
            and model.get("model_id") == repository
            and revision in (model.get("revisions") or [])
        }
        missing = [node_id for node_id in node_ids if node_id not in nodes_with_weights]
        if missing:
            raise ValueError(
                "model weights are not available on selected node(s): "
                + ", ".join(missing)
            )

    def _adopt_manager_replacement(
        self, deployment: dict[str, Any], replacement: dict[str, Any],
        launch_settings: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Create durable SparkDeck routing for a relocated Manager-only card."""
        record_id = str(uuid.uuid4())
        alias = str(
            deployment.get("alias") or replacement.get("name")
            or (deployment.get("model") or {}).get("repository") or record_id
        ).strip()
        if self.store.deployment(alias):
            suffix = str(replacement["id"])[:8]
            candidate = f"{alias} ({suffix})"
            counter = 2
            while self.store.deployment(candidate):
                candidate = f"{alias} ({suffix}-{counter})"
                counter += 1
            alias = candidate

        model = deployment.get("model") or {}
        revision = (
            model.get("revision") or self._persisted_revision(launch_settings)
        )
        layout = self._layout_contract(launch_settings)
        settings = self._local_configuration({
            **(launch_settings or {}),
            **(deployment.get("settings") or {}),
            **layout,
            "manager_deployment_id": replacement["id"],
            "node_ids": list(replacement.get("node_ids") or []),
            "model_source": replacement.get("model_source") or "unknown",
            "source_container_name": (
                deployment.get("container_name")
                if str(deployment.get("id") or "").startswith("container:")
                else None
            ),
        })
        primary = (replacement.get("members") or [{}])[0]
        port = replacement.get("api_port")
        adopted = Deployment(
            id=record_id,
            alias=alias,
            runtime=RuntimeKind(str(deployment.get("runtime") or "vllm")),
            kind=DeploymentKind.MANAGED,
            model=ModelIdentity(
                # The alias remains the served/inference identity. The model
                # record must use the executable repository/path so cache and
                # locality validation remain correct on later restarts.
                repository=str(
                    (launch_settings or {}).get("model")
                    or replacement.get("model")
                    or model.get("repository")
                    or ""
                ),
                revision=_optional_string(revision),
                artifact=_optional_string(model.get("artifact")),
                quantization=_optional_string(model.get("quantization")),
            ),
            container_name=primary.get("container_name"),
            settings=settings,
        )
        self.store.add_deployment(
            adopted,
            f"http://127.0.0.1:{int(port)}" if port else None,
            None,
        )

        # The SQLite manager ID is sufficient for routing. Also persist the
        # reverse link when possible so later reconciliation can find the card
        # even if Manager replaces its own deployment ID again.
        replacement["sparkdeck_record_id"] = record_id
        persisted_launch = replacement.get("launch_settings")
        if isinstance(persisted_launch, dict):
            persisted_launch["sparkdeck_record_id"] = record_id
        save_deployments = getattr(self.manager, "_save_deployments", None)
        if callable(save_deployments):
            try:
                save_deployments()
            except Exception:
                # SQLite already has the durable forward link to this Manager
                # ID; do not report the successful relocation as failed only
                # because the optional reverse-link refresh could not flush.
                pass

        result = self.store.deployment(record_id) or adopted.to_dict()
        result.update({
            "status": _deployment_status(replacement.get("status")),
            "port": int(port) if port else None,
            "node_ids": list(replacement.get("node_ids") or []),
            **layout,
        })
        return result

    async def deployment_action(
        self, deployment_id: str, action: str,
        node_ids: list[str] | None = None,
        additional_node_ids: list[str] | None = None,
        promote: bool = False,
    ) -> dict[str, Any]:
        lock = self._deployment_action_locks.setdefault(
            deployment_id, asyncio.Lock(),
        )
        async with lock:
            launch_task = self._deployment_launch_tasks.get(deployment_id)
            if launch_task is not None and not launch_task.done():
                raise RuntimeError(
                    "deployment launch is still in progress; wait for container creation"
                )
            return await self._deployment_action_locked(
                deployment_id, action, node_ids, additional_node_ids, promote,
            )

    async def _promote_discovered_deployment(
        self, deployment: dict[str, Any], container: dict[str, Any],
        node_ids: list[str],
    ) -> dict[str, Any]:
        """Relaunch one discovered runtime as a managed cluster deployment."""
        settings = dict(container.get("load_settings") or {})
        runtime = str(deployment.get("runtime") or container.get("engine") or "vllm")
        try:
            tensor_parallel = max(1, int(settings.get("tensor_parallel_size") or 1))
        except (TypeError, ValueError):
            tensor_parallel = 1
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("node_ids must not contain duplicates")
        if len(node_ids) != tensor_parallel:
            raise ValueError(
                f"tensor parallel size {tensor_parallel} requires exactly "
                f"{tensor_parallel} node(s)"
            )
        self._reject_sensitive_launch_args(settings.get("extra_args"))
        if settings.get("editable") is False:
            raise ValueError(
                "discovered deployment cannot be promoted safely because its "
                "launch command contains credentials or unsupported arguments"
            )

        launch_model = str(settings.get("model") or container.get("model") or "")
        resolve_local = getattr(self.manager, "_resolve_local_path", None)
        if (
            launch_model and callable(resolve_local) and resolve_local(launch_model)
            and any(node_id != LOCAL_NODE_ID for node_id in node_ids)
        ):
            raise ValueError(
                "controller-local model paths can only run on the controller node"
            )

        recover = getattr(self.manager, "_recovered_deployment_launch_settings", None)
        if not callable(recover):
            raise RuntimeError("discovered deployment cannot be promoted on this controller")
        launch_settings = recover({
            "name": deployment.get("alias"),
            "model": launch_model,
            "engine": runtime,
            "mode": "sharded" if tensor_parallel > 1 else "single",
            "node_ids": list(node_ids),
            "api_port": container.get("port"),
        }, container)
        launch_settings.update({
            "deployment_name": deployment.get("alias"),
            "model": launch_model,
            "engine": runtime,
            "image": container.get("image"),
            "environment": settings.get("environment") or {},
            "deployment_mode": "sharded" if tensor_parallel > 1 else "single",
            "node_ids": list(node_ids),
        })
        replacement = await self.manager.create_deployment(launch_settings)
        adopted_launch_settings = {
            **launch_settings,
            **(replacement.get("launch_settings") or {}),
        }
        adopted_launch_settings["model"] = launch_model
        return self._adopt_manager_replacement(
            deployment, replacement, adopted_launch_settings,
        )

    async def _deployment_action_locked(
        self, deployment_id: str, action: str,
        node_ids: list[str] | None = None,
        additional_node_ids: list[str] | None = None,
        promote: bool = False,
    ) -> dict[str, Any]:
        deployment = self.store.deployment(deployment_id, include_private=True)
        discovered = None
        if not deployment and deployment_id.startswith("container:"):
            discovered = await self._resolve_discovered_container(deployment_id)
            deployment = self._discovered_deployment(
                discovered,
                self._container_runtime(discovered),
                discovered.get("model") or discovered.get("served_model"),
            )
        if not deployment:
            raise LookupError("deployment not found")
        if deployment["kind"] != DeploymentKind.MANAGED.value and discovered is None:
            raise ValueError("external endpoints cannot be started or stopped by SparkDeck")
        container = deployment.get("container_name")
        manager_id = deployment.get("settings", {}).get("manager_deployment_id")
        owner = self._owning_cluster_deployment(container)
        linked = next(
            (
                item for item in getattr(self.manager, "deployments", [])
                if isinstance(item, dict) and item.get("id") == manager_id
            ),
            None,
        ) if manager_id else None
        launch_settings = (owner or linked or {}).get("launch_settings")
        if (
            discovered is not None and not deployment.get("managed")
            and action == "start" and node_ids
        ):
            tensor_parallel = deployment.get("settings", {}).get(
                "tensor_parallel_size", 1,
            )
            try:
                required_nodes = max(1, int(tensor_parallel or 1))
            except (TypeError, ValueError):
                required_nodes = 1
            if len(node_ids) != required_nodes:
                raise ValueError(
                    f"tensor parallel size {required_nodes} requires exactly "
                    f"{required_nodes} node(s)"
                )
            if promote or required_nodes > 1 or node_ids != ["local"]:
                return await self._promote_discovered_deployment(
                    deployment, discovered, node_ids,
                )
        if (
            action == "start" and discovered is None
            and not manager_id and not owner and not container
        ):
            # A saved deployment (launch bookmark) has never been launched:
            # the start action performs the first launch using the recorded
            # runtime, model, settings, and node preferences.
            return await self._launch_saved_deployment(deployment, node_ids)
        relaunch_mode: str | None = None
        if additional_node_ids and action == "start":
            # "Launch on additional nodes" grows the running node set instead
            # of relocating it: the current cluster nodes stay first in the
            # merged selection so the primary replica is preserved.
            if not manager_id and not owner:
                raise ValueError(
                    "node selection is only available for cluster deployments; "
                    "this deployment starts on its existing node"
                )
            contract = self._layout_contract(launch_settings)
            if contract.get("deployment_mode") == "sharded":
                raise ValueError(
                    "sharded deployments cannot launch on additional nodes; "
                    "their tensor-parallel layout is fixed"
                )
            current_nodes = (owner or linked or {}).get("node_ids")
            if not isinstance(current_nodes, list) or not current_nodes:
                current_nodes = (deployment.get("settings") or {}).get("node_ids") or []
            merged = list(dict.fromkeys([
                *(str(item) for item in current_nodes if str(item).strip()),
                *(str(item).strip() for item in additional_node_ids),
            ]))
            if all(item in {str(item) for item in current_nodes} for item in merged):
                # Manager treats any explicit node selection as a relocation:
                # relaunching with an unchanged set would tear down every
                # running rank for a semantic no-op.
                raise ValueError(
                    "additional_node_ids must include at least one node that "
                    "is not already running this deployment"
                )
            # The picker constrains choices in the UI, but an API client can
            # bypass it and the cache can change after the inventory loads —
            # revalidate before relaunching.
            await self._validate_start_selection(deployment, merged, launch_settings)
            if len(merged) > 1 and contract.get("deployment_mode") != "replicated":
                relaunch_mode = "replicated"
            node_ids = merged
        elif node_ids and action == "start":
            node_ids = self._normalized_start_selection(launch_settings, node_ids)
            if not manager_id and not owner and any(item != "local" for item in node_ids):
                # A standalone container runs on the node that holds it; only
                # cluster deployments can be relocated by a start selection.
                raise ValueError(
                    "node selection is only available for cluster deployments; "
                    "this deployment starts on its existing node"
                )
            # The picker constrains choices in the UI, but an API client can
            # bypass it and the cache can change after the inventory loads —
            # revalidate before relaunching.
            await self._validate_start_selection(deployment, node_ids, launch_settings)
        if discovered is None and action == "stop":
            # Persist intent before the first container call. A failed or
            # partially completed stop must still prevent queued inference or
            # health traffic from waking the deployment again.
            self.store.update_desired_state(deployment_id, "stopped")
        if manager_id:
            if node_ids is None:
                result = await self.manager.deployment_action(manager_id, action)
            elif relaunch_mode:
                result = await self.manager.deployment_action(
                    manager_id, action, node_ids, relaunch_mode,
                )
            else:
                result = await self.manager.deployment_action(manager_id, action, node_ids)
            if not result.get("ok"):
                raise RuntimeError("; ".join(result.get("errors") or ["cluster action failed"]))
            replacement = result.get("deployment") if isinstance(result, dict) else None
            if isinstance(replacement, dict) and replacement.get("id"):
                primary = (replacement.get("members") or [{}])[0]
                settings = self._local_configuration({
                    **(deployment.get("settings") or {}),
                    "manager_deployment_id": replacement["id"],
                    "node_ids": list(replacement.get("node_ids") or []),
                })
                self.store.update_managed_routing(
                    deployment_id, settings,
                    primary.get("container_name") or deployment.get("container_name"),
                    f"http://127.0.0.1:{int(replacement['api_port'])}",
                )
            if action == "start":
                self.store.update_desired_state(deployment_id, "running")
            current = self.store.deployment(deployment_id) or deployment
            current["status"] = "running" if action == "start" else "stopped"
            current["node_ids"] = list(current.get("settings", {}).get("node_ids") or [])
            return current
        if owner:
            # A discovered card can be one rank of a manager-only cluster.
            # Acting on the single rank leaves the remaining ranks running,
            # and the cluster health monitor restarts the whole deployment —
            # the action must address the cluster instead.
            if action == "start" and node_ids:
                if relaunch_mode:
                    result = await self.manager.deployment_action(
                        owner["id"], action, node_ids, relaunch_mode,
                    )
                else:
                    result = await self.manager.deployment_action(owner["id"], action, node_ids)
            else:
                result = await self.manager.deployment_action(owner["id"], action)
            if not result.get("ok"):
                raise RuntimeError("; ".join(result.get("errors") or ["cluster action failed"]))
            replacement = result.get("deployment") if isinstance(result, dict) else None
            if isinstance(replacement, dict) and replacement.get("id"):
                return self._adopt_manager_replacement(
                    deployment, replacement, launch_settings,
                )
            return {**deployment, "status": "running" if action == "start" else "stopped"}
        if not container:
            raise LookupError("managed container not found")
        if discovered is None:
            try:
                current = next(
                    (item for item in await self.manager.list_containers()
                     if item.get("name") == container),
                    None,
                )
            except Exception as exc:
                raise LookupError("managed container is unavailable") from exc
            if current is None:
                raise LookupError("managed container not found")
        external_discovered = discovered is not None and not deployment.get("managed")
        if action == "start":
            if external_discovered:
                await self.manager.start_container(
                    container, explicit=True, managed=False,
                )
            else:
                await self.manager.start_container(container, explicit=True)
            if discovered is None:
                self.store.update_desired_state(deployment_id, "running")
        elif action == "stop":
            if external_discovered:
                await self.manager.stop_container(
                    container, explicit=True, managed=False,
                )
            else:
                await self.manager.stop_container(container, explicit=True)
        else:
            raise ValueError("action must be start or stop")
        current = self.store.deployment(deployment_id) or deployment
        current["status"] = "running" if action == "start" else "stopped"
        return current

    def _preparable_deployment_model(
        self, deployment_id: str,
    ) -> tuple[dict[str, Any], str, str]:
        """Resolve one managed deployment to its (record, model, revision)."""
        deployment = self.store.deployment(deployment_id, include_private=True)
        if deployment is None:
            raise LookupError("deployment not found")
        if deployment.get("kind") != DeploymentKind.MANAGED.value:
            raise ValueError("external endpoints do not use cached model weights")
        model = str((deployment.get("model") or {}).get("repository") or "")
        resolve_local = getattr(self.manager, "_resolve_local_path", None)
        if resolve_local and resolve_local(model):
            raise ValueError(
                "local model paths are not distributed through Virtual NAS"
            )
        if not model:
            raise ValueError("deployment does not reference a Hugging Face model")
        revision = (
            _optional_string((deployment.get("model") or {}).get("revision"))
            or "main"
        )
        return deployment, model, revision

    def _llama_selective_artifact(
        self, deployment: dict[str, Any], model: str,
    ) -> list[str] | None:
        """Return the selected GGUF files for a repo-relative llama bookmark.

        ``None`` means this deployment does not download a selected GGUF set
        (a controller-local artifact, or a non-llama runtime).
        """
        if deployment.get("runtime") != RuntimeKind.LLAMA_CPP.value:
            return None
        artifact = str(
            (deployment.get("model") or {}).get("artifact")
            or (deployment.get("settings") or {}).get("artifact")
            or ""
        )
        if not artifact or _artifact_is_controller_local(artifact):
            return None
        relative = self._validate_public_gguf_artifact(model, artifact, None)
        return self._expand_gguf_shard_files(relative)

    async def _selected_files_size(
        self, model: str, resolved_revision: str, files: list[str],
    ) -> int | None:
        """Estimate the Hub size of exactly the selected GGUF files.

        Returns ``None`` when the estimate is unavailable (offline, private
        repository, older agent) so capacity gating degrades to the exact
        per-node check that runs at prepare time.
        """
        virtual_nas = getattr(self.manager, "virtual_nas", None)
        estimate = getattr(virtual_nas, "estimate_selected_files_size", None)
        if not callable(estimate):
            return None
        try:
            size = await estimate(model, resolved_revision, files)
        except Exception:
            return None
        return size if isinstance(size, int) and not isinstance(size, bool) and size > 0 else None

    async def _llama_preparation_plan(
        self, model: str, revision: str, files: list[str],
        node_ids: list[str],
    ) -> dict[str, Any]:
        """Plan selective GGUF preparation for one llama.cpp bookmark.

        Readiness is per-file: a node is ready when its cache already holds
        every selected file of the resolved revision (including snapshots the
        selective downloader marked as deliberately partial). Capacity is
        estimated from the selected files' Hub sizes so undersized nodes are
        disabled with per-node reasons instead of failing after confirmation.
        """
        resolved = await self._resolved_model_revision(model, revision)
        required_bytes = await self._selected_files_size(model, resolved, files)
        inventory = {
            str(node.get("id")): node
            for node in await self.manager.model_cache_inventory()
        }
        nodes = {
            str(node.get("id")): node
            for node in await self.manager.cluster_nodes()
        }

        # Presence checks hit every selected agent; run them concurrently and
        # skip nodes cluster_nodes already reports offline. Results are
        # tri-state: True/False when the check answered, None when it could
        # not (offline, unreachable, or an older agent without the endpoint).
        presence = getattr(self.manager, "node_has_model_files", None)
        presence_available = callable(presence)

        async def check_presence(node_id: str) -> bool | None:
            node = nodes.get(node_id)
            if not presence_available or node is None or node.get("online") is False:
                return None
            try:
                return bool(await presence(node_id, model, resolved, files))
            except Exception:
                return None

        presence_results = await asyncio.gather(
            *(check_presence(node_id) for node_id in node_ids),
            return_exceptions=True,
        )

        targets: list[dict[str, Any]] = []
        for index, node_id in enumerate(node_ids):
            node = nodes.get(node_id)
            entry = inventory.get(node_id) or {}
            model_entry = next((
                item for item in entry.get("models") or []
                if isinstance(item, dict) and item.get("model_id") == model
            ), None)
            raw_presence = presence_results[index]
            has_weights: bool = (
                raw_presence
                if isinstance(raw_presence, bool)
                else False
            )
            if raw_presence is None:
                # A successful answer is authoritative even when it reports
                # files missing; the inventory fallback applies only when the
                # per-file check raised or was unavailable. A complete
                # snapshot of the resolved revision certainly contains the
                # selected files.
                has_weights = bool(
                    model_entry
                    and not model_entry.get("partial")
                    and resolved in (model_entry.get("revisions") or [])
                )
            if node is None:
                # An unknown, removed, or mistyped node ID must never look
                # downloadable; preparation would fail on it later anyway.
                targets.append({
                    "node_id": node_id,
                    "node_name": node_id,
                    "eligible": False,
                    "reason": "Unknown cluster node",
                    "free_bytes": None,
                    "required_free_bytes": None,
                    "active_job_id": None,
                    "active_job_status": None,
                    "active_job_kind": None,
                    "has_preparation_conflict": False,
                    "preparation_conflict_reason": None,
                    "has_required_weights": False,
                    "has_model_cache": False,
                    "download_eligible": False,
                    "download_reason": "Unknown cluster node",
                    "transfer_after_download_eligible": False,
                    "transfer_after_download_reason": None,
                    "transfer_after_download_required_free_bytes": None,
                })
                continue
            online = node.get("online") is not False
            capable = node_id == "local" or bool(
                entry.get("virtual_nas_download_capable", True)
            )
            free_bytes = entry.get("cache_free_size")
            required_free = (
                download_required_free_bytes(required_bytes, 0)
                if required_bytes is not None and not has_weights
                else None
            )
            capacity_reason = None
            if (
                not has_weights
                and required_free is not None
                and free_bytes is not None
                and model_entry is None
                and free_bytes < required_free
            ):
                # Hard-gate only cache-empty nodes, where nothing is reusable
                # and the estimate is exact. Partial caches resume from local
                # blobs the whole-set estimate cannot credit, so those defer
                # to the exact per-node check at prepare time.
                capacity_reason = (
                    "Not enough free cache space for the GGUF download"
                )
            download_eligible = online and capable and capacity_reason is None
            download_reason = (
                None if download_eligible
                else "Node is offline" if not online
                else "Node must be updated before downloading from Hugging Face"
                if not capable
                else capacity_reason
            )
            targets.append({
                "node_id": node_id,
                "node_name": str(node.get("name") or node_id),
                "eligible": False,
                "reason": None,
                "free_bytes": free_bytes,
                "required_free_bytes": required_free,
                "active_job_id": None,
                "active_job_status": None,
                "active_job_kind": None,
                "has_preparation_conflict": False,
                "preparation_conflict_reason": None,
                "has_required_weights": has_weights,
                "has_model_cache": model_entry is not None,
                "download_eligible": download_eligible,
                "download_reason": download_reason,
                "transfer_after_download_eligible": False,
                "transfer_after_download_reason": None,
                "transfer_after_download_required_free_bytes": None,
            })
        missing = [target for target in targets if not target["has_required_weights"]]
        blocked = [target for target in missing if not target["download_eligible"]]
        return {
            "enabled": True,
            "model_id": model,
            "revision": revision,
            "resolved_revision": resolved,
            "source": None,
            "sources": [],
            "download": None,
            "download_error": None,
            "targets": targets,
            "staging_reserve_bytes": 0,
            "node_ids": [str(node_id) for node_id in node_ids],
            "eligible": not blocked,
            "action": "ready" if not missing else "download",
            "download_node_id": missing[0]["node_id"] if missing else None,
            "download_node_ids": [
                target["node_id"] for target in missing if target["download_eligible"]
            ],
            "transfer_target_node_ids": [],
            "reason": (
                "; ".join(
                    f"{target['node_name']}: {target['download_reason']}"
                    for target in blocked
                )
            ) or None,
        }

    async def _prepare_llama_files(
        self, deployment: dict[str, Any], model: str, revision: str,
        files: list[str], node_ids: list[str],
        download_node_id: str | None = None,
    ) -> dict[str, Any]:
        """Place the selected GGUF files on every selected node.

        One node seeds from Hugging Face (the saved or request-provided seed
        preference when set) and the rest receive file-scoped Virtual NAS
        streams, so duplicate Hub bandwidth is never paid for the same
        artifact.
        """
        resolved = await self._resolved_model_revision(model, revision)
        seed = download_node_id or _optional_string(
            (deployment.get("settings") or {}).get("download_node_id")
        )
        await self._distribute_gguf_artifact(
            model, resolved, files, node_ids, seed, revision,
        )
        plan = await self._llama_preparation_plan(model, revision, files, node_ids)
        return {"workflow_id": None, "job_ids": [], "jobs": [], "plan": plan}

    async def _persist_prepared_revision(
        self, deployment_id: str, plan: dict[str, Any] | None,
    ) -> None:
        """Record the immutable revision preparation resolved for a launch."""
        resolved = (plan or {}).get("resolved_revision")
        if not isinstance(resolved, str) or not resolved:
            return
        stored = self.store.deployment(deployment_id, include_private=True)
        if stored is None:
            return
        settings = dict(stored.get("settings") or {})
        if settings.get("prepared_revision") == resolved:
            return
        settings["prepared_revision"] = resolved
        self.store.update_managed_routing(
            stored["id"],
            self._local_configuration(settings),
            stored.get("container_name"),
            stored.get("_base_url"),
        )

    async def deployment_preparation_preflight(
        self, deployment_id: str, node_ids: list[str],
    ) -> dict[str, Any]:
        """Plan per-node weight preparation for a saved deployment."""
        deployment, model, revision = self._preparable_deployment_model(deployment_id)
        files = self._llama_selective_artifact(deployment, model)
        if files is not None:
            return await self._llama_preparation_plan(model, revision, files, node_ids)
        return await self.manager.recipe_model_preparation_preflight(
            model, revision, node_ids,
        )

    async def deployment_prepare(
        self, deployment_id: str, node_ids: list[str],
        download_node_id: str | None = None,
    ) -> dict[str, Any]:
        """Queue Virtual NAS weight preparation for a saved deployment."""
        deployment, model, revision = self._preparable_deployment_model(deployment_id)
        files = self._llama_selective_artifact(deployment, model)
        if files is not None:
            result = await self._prepare_llama_files(
                deployment, model, revision, files, node_ids, download_node_id,
            )
        else:
            result = await self.manager.queue_recipe_model_preparation(
                model, revision, node_ids,
            )
        await self._persist_prepared_revision(deployment_id, result.get("plan"))
        return result

    async def deployment_logs(self, deployment_id: str, tail: Any = 300) -> dict[str, Any]:
        """Return recent container logs for a deployment, all ranks included."""
        deployment = self.store.deployment(deployment_id, include_private=True)
        if not deployment and deployment_id.startswith("container:"):
            container = await self._resolve_discovered_container(deployment_id)
            deployment = self._discovered_deployment(
                container, self._container_runtime(container),
                container.get("model") or container.get("served_model"),
            )
        if not deployment:
            raise LookupError("deployment not found")
        if (
            deployment.get("kind") != DeploymentKind.MANAGED.value
            and not str(deployment_id).startswith("container:")
        ):
            raise ValueError("external endpoints have no managed logs")
        try:
            bounded_tail = min(100_000, max(1, int(tail)))
        except (TypeError, ValueError) as exc:
            raise ValueError("tail must be an integer") from exc

        manager_id = deployment.get("settings", {}).get("manager_deployment_id")
        cluster = next(
            (
                item for item in getattr(self.manager, "deployments", [])
                if isinstance(item, dict) and item.get("id") == manager_id
            ),
            None,
        ) if manager_id else None
        if cluster is None:
            cluster = self._owning_cluster_deployment(deployment.get("container_name"))

        if cluster:
            members = [
                member for member in (cluster.get("members") or [])
                if isinstance(member, dict) and member.get("container_name")
            ]
            if members:
                results = await asyncio.gather(
                    *(self.manager._member_action(member, "logs", log_tail=bounded_tail)
                      for member in members),
                    return_exceptions=True,
                )

                def member_section(member: dict[str, Any], result: Any) -> str:
                    label = f"rank {member.get('rank')} · node {member.get('node_id')}"
                    logs = result.get("logs") if isinstance(result, dict) else None
                    if isinstance(logs, str) and logs:
                        body = logs
                    else:
                        # Ranks still queued/creating — and older agents that
                        # reply 404 until Docker creates the container — have
                        # no logs yet; the coordinator's launch status is the
                        # useful signal there.
                        phase = member.get("phase") or {}
                        message = phase.get("message") or (
                            "Waiting for the node agent to begin launch"
                            if member.get("status") in {"queued", "creating"}
                            else "No output has been reported yet"
                        )
                        body = f"=== Coordinator launch status ===\n{message}"
                        if isinstance(result, Exception):
                            body += f"\n\nAgent log request: {result}"
                    return f"===== {label} ({member.get('container_name')}) =====\n{body}"

                sections = [
                    member_section(member, result)
                    for member, result in zip(members, results)
                ]
                return {"logs": "\n\n".join(sections)}

        container = deployment.get("container_name")
        if not container:
            raise LookupError("managed container not found")
        if deployment_id.startswith("container:") and not deployment.get("managed"):
            return {"logs": await self.manager.get_logs(container, bounded_tail)}
        return {"logs": await self.manager.get_cluster_member_logs(container, bounded_tail)}

    async def delete_deployment(self, deployment_id: str) -> dict[str, Any]:
        # A provisional row is visible while Docker launch is in flight. Wait
        # only for that deployment to settle; unrelated removals must remain
        # responsive during a slow image pull or container startup.
        provisional = self.store.deployment(deployment_id, include_private=True)
        if provisional:
            launch_complete = self._deployment_launches.get(provisional["id"])
            if launch_complete is not None:
                await launch_complete.wait()
        deployment = self.store.deployment(deployment_id, include_private=True)
        discovered = False
        if not deployment and deployment_id.startswith("container:"):
            container = await self._resolve_discovered_container(deployment_id)
            deployment = self._discovered_deployment(
                container, self._container_runtime(container),
                container.get("model") or container.get("served_model"),
            )
            discovered = True
        if not deployment:
            raise LookupError("deployment not found")
        manager_id = deployment.get("settings", {}).get("manager_deployment_id")
        owner = self._owning_cluster_deployment(deployment.get("container_name"))
        if manager_id:
            result = await self.manager.deployment_action(manager_id, "remove")
            if not result.get("ok"):
                raise RuntimeError("; ".join(result.get("errors") or ["cluster removal failed"]))
        elif owner:
            # Removing one rank of a cluster would leave the health monitor to
            # resurrect the deployment; remove the whole cluster instead.
            result = await self.manager.deployment_action(owner["id"], "remove")
            if not result.get("ok"):
                raise RuntimeError("; ".join(result.get("errors") or ["cluster removal failed"]))
        elif (
            deployment["kind"] == DeploymentKind.MANAGED.value or discovered
        ) and deployment.get("container_name"):
            try:
                await self.manager.remove_container(deployment["container_name"])
            except Exception as exc:
                if not _is_missing_container_error(exc):
                    raise
        if not discovered:
            self._delete_credential(deployment_id, deployment.get("_credential_ref"))
            self.store.delete_deployment(deployment_id)
        return {"ok": True, "id": deployment_id}

    def _clone_launch_configuration(
        self, stored: dict[str, Any], runtime: RuntimeKind,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Normalize persisted and Manager-owned settings for a fresh launch."""
        settings = copy.deepcopy(stored.get("settings") or {})
        manager_id = settings.get("manager_deployment_id")
        manager_deployment = None
        lookup = getattr(self.manager, "_deployment", None)
        if manager_id and callable(lookup):
            manager_deployment = lookup(str(manager_id))
        if manager_deployment is None and manager_id:
            manager_deployment = next((
                item for item in getattr(self.manager, "deployments", [])
                if isinstance(item, dict) and str(item.get("id")) == str(manager_id)
            ), None)
        launch_settings = (
            copy.deepcopy(manager_deployment.get("launch_settings"))
            if isinstance(manager_deployment, dict)
            and isinstance(manager_deployment.get("launch_settings"), dict)
            else {}
        )
        launch_inputs = {
            **settings,
            **launch_settings,
            "engine": runtime.value,
        }
        sanitize_args = getattr(
            self.manager, "_without_sensitive_cli_credentials", None,
        )
        extra_args = list(launch_inputs.get("extra_args") or [])
        if callable(sanitize_args):
            extra_args = sanitize_args(extra_args)
        launch_inputs["extra_args"] = extra_args
        normalized = self._local_configuration(launch_inputs)
        controls = self.manager._deployment_launch_controls(launch_inputs)

        def preserve(key: str, value: Any) -> None:
            if value is not None:
                normalized[key] = value

        for pricing_key in (
            "input_cost_per_1m", "cache_cost_per_1m", "output_cost_per_1m",
        ):
            preserve(pricing_key, launch_inputs.get(pricing_key))
        preserve("context_length", controls.get("context_window"))
        if runtime is RuntimeKind.SGLANG:
            preserve("tensor_parallel_size", launch_inputs.get("sg_tp_size"))
            preserve("max_running_requests", controls.get("max_concurrency"))
            preserve("mem_fraction_static", launch_inputs.get("sg_mem_fraction"))
        elif runtime is RuntimeKind.LLAMA_CPP:
            preserve("parallel_slots", controls.get("max_concurrency"))
            preserve("gpu_layers", launch_inputs.get("llama_gpu_layers"))

        for runtime_identity_key in (
            "manager_deployment_id", "managed_by", "automation_run_id",
        ):
            normalized.pop(runtime_identity_key, None)
        return normalized, launch_settings

    def _clone_llama_artifact_identity(
        self, repository: str, artifact: Any,
    ) -> tuple[str | None, str | None]:
        """Restore a Manager cache reference and its pinned snapshot revision."""
        value = _optional_string(artifact)
        if value is None:
            return None, None
        normalized = value.replace("\\", "/")
        parts = PurePosixPath(normalized).parts
        encoded_repository = "models--" + repository.replace("/", "--")
        if (
            len(parts) >= 4
            and parts[0].casefold() == encoded_repository.casefold()
            and parts[1].casefold() == "snapshots"
            and re.fullmatch(r"[0-9a-f]{40}", parts[2], re.IGNORECASE)
        ):
            relative = "/".join(parts[3:])
            return (
                self._validate_public_gguf_artifact(
                    repository, relative, None,
                ).as_posix(),
                parts[2],
            )
        return value, None

    async def clone_deployment(self, deployment_id: str) -> dict[str, Any]:
        """Copy a persisted deployment's configuration without its live runtime."""
        if deployment_id.startswith("container:"):
            raise ValueError("discovered containers cannot be cloned")

        async with self._deployment_create_lock:
            stored = self.store.deployment(deployment_id, include_private=True)
            if not stored:
                raise LookupError("deployment not found")

            root_alias = re.sub(
                r"^\(Copy(?: \d+)?\)\s+", "", str(stored["alias"]), count=1,
            ).strip() or str(stored["alias"])
            aliases = {
                str(item.get("alias") or "").casefold()
                for item in self.store.deployments()
            }
            copy_number = 1
            while True:
                prefix = "(Copy)" if copy_number == 1 else f"(Copy {copy_number})"
                alias = f"{prefix} {root_alias}"
                if alias.casefold() not in aliases:
                    break
                copy_number += 1

            clone_id = str(uuid.uuid4())
            kind = DeploymentKind(str(stored["kind"]))
            runtime = RuntimeKind(str(stored["runtime"]))
            settings, launch_settings = self._clone_launch_configuration(
                stored, runtime,
            )
            if kind is DeploymentKind.MANAGED:
                settings.pop("port", None)
            model = stored.get("model") or {}
            repository = str(model.get("repository") or "")
            artifact = model.get("artifact")
            revision = _optional_string(model.get("revision"))
            if runtime is RuntimeKind.LLAMA_CPP:
                artifact, snapshot_revision = self._clone_llama_artifact_identity(
                    repository,
                    launch_settings.get("llama_artifact") or artifact,
                )
                revision = snapshot_revision or revision
            clone_base_url = (
                stored.get("_base_url")
                if kind is DeploymentKind.EXTERNAL else None
            )
            clone = Deployment(
                id=clone_id,
                alias=alias,
                runtime=runtime,
                kind=kind,
                model=ModelIdentity(
                    repository=repository,
                    revision=revision,
                    artifact=artifact,
                    quantization=_optional_string(model.get("quantization")),
                ),
                settings=settings,
                base_url_set=bool(clone_base_url),
                desired_state=(
                    "stopped" if kind is DeploymentKind.MANAGED
                    else str(stored.get("desired_state") or "running")
                ),
            )

            credential_ref = None
            if stored.get("_credential_ref"):
                api_key = self._get_credential(
                    stored["id"], stored.get("_credential_ref"),
                )
                if api_key is None:
                    raise RuntimeError(
                        "deployment credential could not be read from OS credential storage"
                    )
                credential_ref = self._store_credential(clone_id, api_key)
            try:
                self.store.add_deployment(
                    clone, clone_base_url, credential_ref,
                )
            except Exception:
                self._delete_credential(clone_id, credential_ref)
                raise

            result = self.store.deployment(clone_id) or clone.to_dict()
            if kind is DeploymentKind.MANAGED:
                result["status"] = "saved"
                node_ids = [str(item) for item in settings.get("node_ids") or []]
                if node_ids:
                    result["node_ids"] = node_ids
                result.update(self._saved_layout_contract(settings, clone.runtime.value))
            return result

    async def rename_deployment(self, deployment_id: str, alias: Any) -> dict[str, Any]:
        alias = str(alias or "").strip()
        if not alias:
            raise ValueError("alias is required")
        if deployment_id.startswith("container:"):
            raise ValueError("discovered containers cannot be renamed")
        stored = self.store.deployment(deployment_id, include_private=True)
        if not stored:
            raise LookupError("deployment not found")
        existing = self.store.deployment(alias)
        if existing and existing["id"] != stored["id"]:
            raise ValueError(f"deployment alias '{alias}' is already in use")
        manager_id = stored.get("settings", {}).get("manager_deployment_id")
        if manager_id:
            # Keep the Manager record in sync so /api/state and MCP cluster
            # listings (and future Manager-driven rebuilds) use the new name.
            self.manager.update_deployment_alias(manager_id, alias)
        self.store.update_alias(stored["id"], alias)
        stored.pop("_base_url", None)
        stored.pop("_credential_ref", None)
        return {**stored, "alias": alias}

    def _owning_cluster_deployment(self, container_name: str | None) -> dict[str, Any] | None:
        """Return the managed cluster deployment a container is a rank of."""
        if not container_name:
            return None
        return next(
            (
                item for item in getattr(self.manager, "deployments", [])
                if isinstance(item, dict) and item.get("id")
                and any(
                    isinstance(member, dict)
                    and member.get("container_name") == container_name
                    for member in (item.get("members") or [])
                )
            ),
            None,
        )

    async def _resolve_discovered_container(self, deployment_id: str) -> dict[str, Any]:
        name = deployment_id.removeprefix("container:")
        try:
            container = next(
                (item for item in await self.manager.list_containers()
                 if item.get("name") == name),
                None,
            )
        except Exception as exc:
            raise LookupError("managed container is unavailable") from exc
        if container is None or self._container_runtime(container) not in self.registry.kinds:
            raise LookupError("managed container not found")
        return container

    def _discovered_deployment(
        self, container: dict[str, Any], runtime: str, model: str
    ) -> dict[str, Any]:
        phase = container.get("phase") if isinstance(container.get("phase"), dict) else {}
        status = _managed_container_status(container)
        if not container.get("managed") and container.get("status") == "created":
            # Docker-created external containers are idle and startable. The
            # same state remains a launch phase for SparkDeck-managed runs.
            status = "stopped"
        settings = self._safe_configuration(container.get("load_settings") or {})
        if settings.get("context_length") is None and settings.get("context_window") is not None:
            settings["context_length"] = settings["context_window"]
        result = {
            # Synthetic IDs intentionally key by container name. A cluster
            # deployment ID may be shared by several ranks and cannot identify
            # an individual legacy container action safely.
            "id": f"container:{container.get('name')}",
            "alias": container.get("alias") or container.get("served_model") or model,
            "runtime": runtime,
            "kind": "managed" if container.get("managed") else "external",
            "model": {
                "repository": model, "revision": None, "artifact": None,
                "quantization": container.get("variant"),
            },
            "status": status,
            "container_name": container.get("name"),
            "settings": settings,
            "base_url_set": bool(container.get("port")),
            "port": container.get("port"),
            "managed": bool(container.get("managed")),
            "controllable": True,
            "logs_available": True,
            "removable": True,
        }
        last_deployed_at = container.get("started_at") or container.get("created")
        if (
            isinstance(last_deployed_at, (int, float)) and last_deployed_at > 0
        ) or (
            isinstance(last_deployed_at, str)
            and last_deployed_at
            and not last_deployed_at.startswith("0001-")
        ):
            result["last_deployed_at"] = last_deployed_at
        try:
            tensor_parallel = max(
                1, int(result["settings"].get("tensor_parallel_size") or 1),
            )
        except (TypeError, ValueError):
            tensor_parallel = 1
        result.update({
            "deployment_mode": "sharded" if tensor_parallel > 1 else "single",
            "required_node_count": tensor_parallel,
        })
        if status == "starting" and phase:
            result.update({
                "launch_phase": str(phase.get("phase") or "starting"),
                "launch_message": str(phase.get("message") or "Container is starting"),
            })
        return result

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
                "container_name": deployment.get("container_name"),
                "port": deployment.get("port"),
            })
        loaded_llama = await self._native_llama_model()
        if loaded_llama and loaded_llama not in seen:
            data.append({
                "id": loaded_llama, "object": "model", "created": 0,
                "owned_by": "llama.cpp", "runtime": RuntimeKind.LLAMA_CPP.value,
                "model": {
                    "repository": loaded_llama, "revision": None,
                    "artifact": None, "quantization": None,
                },
            })
        return {"object": "list", "data": data}

    async def proxy(self, body: dict[str, Any], endpoint: str,
                    cancel: Any = None, *, caller_ip: str | None = None,
                    ) -> dict[str, Any] | AsyncIterator[str]:
        requested_model = str(body.get("model") or "")
        deployment = self.store.deployment(requested_model, include_private=True)
        observation = self._community_observation_start(
            self._community_observation_scopes(deployment, requested_model)
        )
        context_token = self._community_observation.set(observation)
        streaming = False
        try:
            if deployment:
                result = await self._proxy_registered(
                    deployment, body, endpoint, cancel, caller_ip=caller_ip,
                )
            else:
                # Compatibility for existing vLLM/SGLang containers. Resolve the
                # container's source repository rather than publishing an
                # arbitrary served alias as the model identity.
                started = time.monotonic()
                caller_kwargs = {"caller_ip": caller_ip} if caller_ip else {}
                result = (
                    await self.manager.proxy_chat_completions(
                        body, cancel, **caller_kwargs,
                    )
                    if endpoint == "chat/completions"
                    else await self.manager.proxy_completions(
                        body, cancel, **caller_kwargs,
                    )
                )
                model, runtime, settings = await self._legacy_model_identity(
                    requested_model
                )
                if hasattr(result, "__aiter__"):
                    result = self._observe_stream(
                        result, None, model, runtime, settings, started
                    )
                else:
                    self._record_response(
                        None, model, runtime, settings, started, result
                    )
            if hasattr(result, "__aiter__"):
                streaming = True
                return self._community_observed_stream(result, observation)
            return result
        finally:
            self._community_observation.reset(context_token)
            if not streaming:
                self._community_observation_end(observation)

    def _community_observation_scopes(
        self, deployment: dict[str, Any] | None, requested_model: str,
    ) -> frozenset[str]:
        """Return private in-memory identities for potentially shared hardware."""
        if deployment:
            settings = deployment.get("settings") or {}
            manager_id = settings.get("manager_deployment_id")
            linked = next((
                item for item in getattr(self.manager, "deployments", [])
                if isinstance(item, dict) and item.get("id") == manager_id
            ), None)
            node_ids = {
                str(node_id)
                for source in (
                    deployment.get("node_ids") or [],
                    settings.get("node_ids") or [],
                    (linked or {}).get("node_ids") or [],
                )
                for node_id in source
                if str(node_id)
            }
            node_ids.update(
                str(member.get("node_id"))
                for member in (linked or {}).get("members") or []
                if isinstance(member, dict) and member.get("node_id")
            )
            if node_ids:
                return frozenset(f"node:{node_id}" for node_id in node_ids)
            if deployment.get("kind") == DeploymentKind.MANAGED.value:
                return frozenset({"node:local"})
            endpoint = normalize_openai_base_url(deployment.get("_base_url") or "")
            if endpoint:
                return frozenset({f"endpoint:{endpoint.casefold()}"})
            return frozenset({f"deployment:{deployment.get('id')}"})
        return frozenset({f"legacy-model:{requested_model.casefold()}"})

    def _community_observation_start(
        self, scopes: frozenset[str] | None = None,
    ) -> dict[str, Any] | None:
        snapshot = self.store.community_consent_snapshot()
        scopes = scopes or frozenset({"unspecified"})
        observation = {
            "id": str(uuid.uuid4()),
            # Opted-out requests carry only lifecycle state. They never reach
            # sample collection, but remain visible long enough to invalidate
            # an opted-in request whose decode overlaps them.
            "enabled": bool(snapshot.get("enabled")),
            "generation": int(snapshot.get("generation") or 0),
            "scopes": scopes,
            "contaminated": False,
        }
        for active in self._community_active_observations.values():
            if scopes.intersection(active.get("scopes") or ()):
                observation["contaminated"] = True
                active["contaminated"] = True
        self._community_active_observations[observation["id"]] = observation
        return observation

    def _community_observation_end(self, observation: dict[str, Any] | None) -> None:
        if observation is not None and observation.get("id"):
            self._community_active_observations.pop(observation["id"], None)

    async def _community_observed_stream(
        self, stream: AsyncIterator[str], observation: dict[str, Any] | None,
    ) -> AsyncIterator[str]:
        token = self._community_observation.set(observation)
        try:
            async for chunk in stream:
                yield chunk
        finally:
            self._community_observation.reset(token)
            self._community_observation_end(observation)

    async def _proxy_registered(self, deployment: dict[str, Any], body: dict[str, Any],
                                endpoint: str, cancel: Any, *,
                                caller_ip: str | None = None,
                                ) -> dict[str, Any] | AsyncIterator[str]:
        manager_desired = None
        manager_id = (deployment.get("settings") or {}).get(
            "manager_deployment_id"
        )
        if manager_id:
            linked = next(
                (
                    item for item in getattr(self.manager, "deployments", [])
                    if isinstance(item, dict) and item.get("id") == manager_id
                ),
                None,
            )
            manager_desired = (linked or {}).get("desired_state")
        if (
            deployment.get("kind") == DeploymentKind.MANAGED.value
            and (
                deployment.get("desired_state") == "stopped"
                or manager_desired == "stopped"
            )
        ):
            raise RuntimeError(
                "deployment is stopped; start it before sending inference requests"
            )
        if (
            deployment.get("kind") == DeploymentKind.MANAGED.value
            and deployment.get("runtime") in (RuntimeKind.VLLM.value, RuntimeKind.SGLANG.value)
        ):
            return await self._proxy_managed(
                deployment, body, endpoint, cancel, caller_ip=caller_ip,
            )
        base_url = normalize_openai_base_url(deployment.get("_base_url") or "")
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
            return await self._http_stream(
                f"{base_url}/v1/{endpoint}", upstream_body, headers,
                deployment, started, cancel,
            )
        upstream = self.manager.http.post(
            f"{base_url}/v1/{endpoint}", json=upstream_body,
            headers=headers, timeout=600,
        )
        response = (
            await self.manager._await_or_cancel(upstream, cancel)
            if cancel is not None
            else await upstream
        )
        response.raise_for_status()
        data = response.json()
        managed = deployment.get("kind") == DeploymentKind.MANAGED.value
        self._record_response(
            deployment["id"], deployment["model"]["repository"],
            deployment["runtime"], self._model_observation_settings(deployment),
            started, data,
            revision=deployment["model"].get("revision"),
            hardware=(
                self._hardware_snapshot()
                if managed else self._unknown_hardware_snapshot()
            ),
            hardware_verified=managed,
        )
        data["model"] = deployment["alias"]
        return data

    async def _proxy_managed(self, deployment: dict[str, Any], body: dict[str, Any],
                             endpoint: str, cancel: Any, *,
                             caller_ip: str | None = None,
                             ) -> dict[str, Any] | AsyncIterator[str]:
        """Keep managed vLLM/SGLang requests on Manager's admission path."""
        model = deployment["model"]["repository"]
        upstream_body = {**body, "model": model}
        stream = bool(upstream_body.get("stream"))
        started = time.monotonic()
        settings = self._model_observation_settings(deployment)
        manager_id = settings.get("manager_deployment_id")
        route_observation: dict[str, Any] = {}
        caller_kwargs = {"caller_ip": caller_ip} if caller_ip else {}
        result = (
            await self.manager.proxy_cluster_inference(
                manager_id, model, upstream_body, endpoint, cancel,
                route_observation=route_observation,
                **caller_kwargs,
            )
            if manager_id
            else (
                await self.manager._vllm_chat(
                    model, upstream_body, stream, cancel,
                    container_name=deployment.get("container_name"),
                    deployment_id=deployment["id"],
                    **caller_kwargs,
                )
                if endpoint == "chat/completions"
                else await self.manager._vllm_completions(
                    model, upstream_body, stream, cancel,
                    container_name=deployment.get("container_name"),
                    deployment_id=deployment["id"],
                    **caller_kwargs,
                )
            )
        )
        revision = deployment["model"].get("revision")
        if hasattr(result, "__aiter__"):
            async def serving_hardware():
                return await self._managed_hardware_snapshot(
                    deployment, route_observation.get("member"),
                    require_serving_member=bool(manager_id),
                )

            return self._observe_stream(
                result, deployment["id"], model, deployment["runtime"], settings,
                started, revision=revision, response_model=deployment["alias"],
                hardware_resolver=serving_hardware,
            )
        hardware, hardware_verified = await self._managed_hardware_snapshot(
            deployment, route_observation.get("member"),
            require_serving_member=bool(manager_id),
        )
        self._record_response(
            deployment["id"], model, deployment["runtime"], settings, started,
            result, revision=revision, hardware=hardware,
            hardware_verified=hardware_verified,
        )
        if isinstance(result, dict):
            result["model"] = deployment["alias"]
        return result

    async def _http_stream(self, url: str, body: dict[str, Any], headers: dict[str, str],
                           deployment: dict[str, Any], started: float,
                           cancel: Any) -> AsyncIterator[str]:
        upstream_body = dict(body)
        retried_without_stream_options = False
        while True:
            response_context = self.manager.http.stream(
                "POST", url, json=upstream_body, headers=headers, timeout=None
            )
            enter = response_context.__aenter__()
            response = (
                await self.manager._await_or_cancel(enter, cancel)
                if cancel is not None
                else await enter
            )
            if (
                response.status_code == 400
                and upstream_body.get("stream_options")
                and not retried_without_stream_options
            ):
                # Older llama-server/SGLang releases may not support this
                # OpenAI extension. Streaming still works; only terminal
                # usage-based benchmark capture is unavailable for that run.
                try:
                    await response.aread()
                finally:
                    await response_context.__aexit__(None, None, None)
                upstream_body = {
                    key: value for key, value in upstream_body.items()
                    if key != "stream_options"
                }
                retried_without_stream_options = True
                continue
            try:
                # Validate while the route can still return the real upstream
                # status. Deferring this until iteration would cause FastAPI's
                # StreamingResponse to commit a misleading HTTP 200 first.
                response.raise_for_status()
            except BaseException as exc:
                await response_context.__aexit__(type(exc), exc, exc.__traceback__)
                raise
            return self._consume_http_stream(
                response_context, response, deployment, started, cancel,
            )

    async def _consume_http_stream(
        self, response_context: Any, response: Any, deployment: dict[str, Any],
        started: float, cancel: Any,
    ) -> AsyncIterator[str]:
        first_token_at: float | None = None
        usage: dict[str, Any] | None = None
        stream_error: BaseException | None = None
        try:
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
        except BaseException as exc:
            stream_error = exc
            raise
        finally:
            if stream_error is None:
                await response_context.__aexit__(None, None, None)
            else:
                await response_context.__aexit__(
                    type(stream_error), stream_error, stream_error.__traceback__,
                )
        if usage:
            self._record_usage(
                deployment["id"], deployment["model"]["repository"],
                deployment["runtime"], self._model_observation_settings(deployment),
                started,
                usage, first_token_at,
                revision=deployment["model"].get("revision"),
                hardware=self._unknown_hardware_snapshot(), hardware_verified=False,
                stream_timing_trusted=False,
            )

    async def _observe_stream(self, stream: AsyncIterator[str], deployment_id: str | None,
                              model: str, runtime: str, settings: dict[str, Any],
                              started: float, revision: str | None = None,
                              response_model: str | None = None,
                              hardware: dict[str, Any] | None = None,
                              hardware_verified: bool = True,
                              hardware_resolver: Any = None) -> AsyncIterator[str]:
        first_token_at = None
        usage = None
        async for chunk in stream:
            text = chunk.decode("utf-8", errors="replace") if isinstance(chunk, bytes) else str(chunk)
            if response_model:
                text = _rewrite_sse_model(text, response_model)
            for line in text.splitlines():
                parsed = _parse_sse(line)
                if parsed and parsed.get("usage"):
                    usage = parsed["usage"]
                if parsed and first_token_at is None and _chunk_has_output(parsed):
                    first_token_at = time.monotonic()
            yield text.encode("utf-8") if isinstance(chunk, bytes) else text
        if usage:
            if hardware_resolver is not None:
                hardware, hardware_verified = await hardware_resolver()
            self._record_usage(
                deployment_id, model, runtime, settings, started, usage,
                first_token_at, revision=revision, hardware=hardware,
                hardware_verified=hardware_verified,
                stream_timing_trusted=False,
            )

    def _record_response(self, deployment_id: str | None, model: str, runtime: str,
                         settings: dict[str, Any], started: float, data: dict[str, Any],
                         revision: str | None = None,
                         hardware: dict[str, Any] | None = None,
                         hardware_verified: bool = True) -> None:
        usage = data.get("usage") if isinstance(data, dict) else None
        if isinstance(usage, dict):
            timings = data.get("timings") or {}
            self._record_usage(
                deployment_id, model, runtime, settings, started, usage, None,
                native_generation_tps=_positive_float(timings.get("predicted_per_second")),
                native_prompt_tps=_positive_float(timings.get("prompt_per_second")),
                revision=revision,
                hardware=hardware, hardware_verified=hardware_verified,
            )

    def _record_usage(self, deployment_id: str | None, model: str, runtime: str,
                      settings: dict[str, Any], started: float, usage: dict[str, Any],
                      first_token_at: float | None,
                      native_generation_tps: float | None = None,
                      native_prompt_tps: float | None = None,
                      revision: str | None = None,
                      hardware: dict[str, Any] | None = None,
                      hardware_verified: bool = True,
                      stream_timing_trusted: bool = True) -> None:
        observation = self._community_observation.get()
        passive_observation = observation is not None
        # Community sharing is the authority for passive inference telemetry.
        # When it is off, do not even create a local sample row.
        if passive_observation and (
            not observation.get("enabled") or observation.get("contaminated")
            or not stream_timing_trusted
        ):
            return
        completed = time.monotonic()
        input_tokens = max(0, int(usage.get("prompt_tokens") or 0))
        output_tokens = max(0, int(usage.get("completion_tokens") or 0))
        latency = max(0.000001, completed - started)
        generation_seconds = max(0.000001, completed - (first_token_at or started))
        runtime_kind = RuntimeKind(runtime)
        safe_settings = self._safe_configuration(settings)
        quantization = (
            canonical_quantization(settings.get("quantization"))
            or quantization_from_text(
                settings.get("gguf_variant"), settings.get("artifact"), model,
            )
            or "UNKNOWN"
        )
        observed_generation_tps = (
            round(output_tokens / generation_seconds, 3)
            if output_tokens and first_token_at is not None else None
        )
        observed_prompt_tps = (
            round(input_tokens / max(0.000001, first_token_at - started), 3)
            if first_token_at is not None and input_tokens else None
        )
        generation_tps = native_generation_tps or observed_generation_tps
        prompt_tps = native_prompt_tps or observed_prompt_tps
        public_model = _public_model_id(model)
        measured_decode_seconds = (
            generation_seconds
            if first_token_at is not None
            else (
                output_tokens / native_generation_tps
                if native_generation_tps and output_tokens else 0.0
            )
        )
        passive_eligible = bool(
            public_model != "local-model"
            and 0 < input_tokens < _COMMUNITY_SAMPLE_MAX_INPUT_TOKENS
            and output_tokens >= 32
            and measured_decode_seconds >= _COMMUNITY_SAMPLE_MIN_DECODE_SECONDS
            and generation_tps is not None
            and runtime_kind.value in self.registry.kinds
            and hardware_verified
            and self._community_sample_due(public_model, quantization)
        )
        legacy_eligible = bool(
            public_model != "local-model" and input_tokens > 0
            and output_tokens >= 16 and latency > 0
            and generation_tps is not None
            and community_context_window(safe_settings) is not None
            and runtime_kind.value in self.registry.kinds
            and hardware_verified
        )
        eligible = passive_eligible if passive_observation else legacy_eligible
        if passive_observation and not eligible:
            return
        # For community evidence this compatibility field represents observed
        # prompt occupancy, not the deployment's configured maximum context.
        if passive_observation:
            safe_settings["context_length"] = input_tokens
        sample = BenchmarkSample(
            id=str(uuid.uuid4()), created_at=datetime.now(timezone.utc).isoformat(),
            deployment_id=deployment_id,
            model=ModelIdentity(
                # Preserve public identities and give private models a stable,
                # non-sensitive local label instead of the old "local-model".
                repository=_local_benchmark_model_id(
                    model, deployment_id, upload_model_id=public_model,
                ),
                revision=_optional_string(revision),
                quantization=quantization,
            ),
            runtime=runtime_kind,
            runtime_version=_optional_string(safe_settings.get("runtime_version")),
            hardware=hardware if hardware is not None else self._hardware_snapshot(),
            configuration=safe_settings, input_tokens=input_tokens,
            output_tokens=output_tokens, latency_ms=round(latency * 1000, 3),
            ttft_ms=None if first_token_at is None else round((first_token_at - started) * 1000, 3),
            generation_tokens_per_second=generation_tps,
            prompt_tokens_per_second=prompt_tps,
            cold_start=None, eligible_for_community=eligible,
        )
        if not passive_observation:
            consent = bool(self.store.get_setting("community_consent", False))
            self.store.add_benchmark(sample, queue=eligible and consent)
            return
        inserted = self.store.add_benchmark_if_consented(
            sample, int(observation.get("generation") or 0)
        )
        if inserted:
            self.store.set_setting(
                self._community_sample_setting(public_model, quantization),
                datetime.now(timezone.utc).isoformat(),
            )

    @staticmethod
    def _community_sample_setting(model: str, quantization: str) -> str:
        quantization = canonical_quantization(quantization) or "UNKNOWN"
        digest = hashlib.sha256(
            f"{model.casefold()}\0{quantization.casefold()}".encode("utf-8")
        ).hexdigest()
        return f"community_sampled_at:{digest}"

    def _community_sample_due(self, model: str, quantization: str) -> bool:
        value = self.store.get_setting(
            self._community_sample_setting(model, quantization), None
        )
        if not isinstance(value, str):
            return True
        try:
            sampled_at = datetime.fromisoformat(value)
            if sampled_at.tzinfo is None:
                sampled_at = sampled_at.replace(tzinfo=timezone.utc)
        except ValueError:
            return True
        return (
            datetime.now(timezone.utc) - sampled_at
        ).total_seconds() >= _COMMUNITY_SAMPLE_INTERVAL_SECONDS

    async def _runtime_for_legacy_model(self, model: str) -> str:
        _, runtime, _ = await self._legacy_model_identity(model)
        return runtime

    async def _legacy_model_identity(
        self, requested_model: str,
    ) -> tuple[str, str, dict[str, Any]]:
        if await self._native_llama_model() == requested_model:
            variant = getattr(self.manager, "_unsloth_variant", lambda value: None)(
                requested_model
            )
            return requested_model, RuntimeKind.LLAMA_CPP.value, {
                "gguf_variant": variant,
            }
        try:
            for container in await self.manager.list_containers():
                ids = self.manager._container_model_ids(container)
                if requested_model in ids:
                    repository = str(container.get("model") or "").strip()
                    if _public_model_id(repository) == "local-model":
                        repository = _local_benchmark_model_id(
                            repository or requested_model, None,
                            upload_model_id="local-model",
                        )
                    settings = dict(
                        container.get("load_settings")
                        or container.get("settings")
                        or {}
                    )
                    for key in ("quantization", "artifact", "gguf_variant"):
                        if container.get(key) is not None:
                            settings[key] = container[key]
                    return repository, self._container_runtime(container), settings
        except Exception:
            pass
        # An unlinked served alias is not a trustworthy public repository.
        return requested_model or "Unknown model", RuntimeKind.VLLM.value, {}

    @staticmethod
    def _model_observation_settings(deployment: dict[str, Any]) -> dict[str, Any]:
        settings = dict(deployment.get("settings") or {})
        model = deployment.get("model") or {}
        for key in ("quantization", "artifact"):
            if model.get(key) is not None:
                settings[key] = model[key]
        return settings

    async def _native_llama_model(self) -> str | None:
        loaded_model = getattr(self.manager, "_unsloth_loaded_model", None)
        if loaded_model is None:
            return None
        result = loaded_model()
        if not inspect.isawaitable(result):
            return None
        return await result

    @staticmethod
    def _container_runtime(container: dict[str, Any]) -> str:
        runtime = str(container.get("runtime") or container.get("engine") or "vllm")
        return "llama.cpp" if runtime in ("llama", "llama-server", "llama_cpp") else runtime

    @staticmethod
    def _safe_configuration(settings: dict[str, Any]) -> dict[str, Any]:
        configuration = {
            key: value
            for key, value in settings.items()
            if key in _SAFE_CONFIGURATION_KEYS and key != "gpu_memory_utilization"
        }
        utilization = settings.get("gpu_memory_utilization")
        if (
            isinstance(utilization, (int, float))
            and not isinstance(utilization, bool)
            and math.isfinite(float(utilization))
            and 0 < float(utilization) <= 1
        ):
            configuration["gpu_memory_utilization"] = float(utilization)
        return configuration

    @staticmethod
    def _local_configuration(settings: dict[str, Any]) -> dict[str, Any]:
        configuration = SparkDeckService._safe_configuration(settings)
        configuration.update({
            key: value for key, value in settings.items()
            if key in _LOCAL_ROUTING_KEYS
        })
        return configuration

    async def _managed_hardware_snapshot(
        self, deployment: dict[str, Any], serving_member: dict[str, Any] | None = None,
        *, require_serving_member: bool = False,
    ) -> tuple[dict[str, Any], bool]:
        """Resolve benchmark hardware from the cluster member that serves requests."""
        if deployment.get("kind") != DeploymentKind.MANAGED.value:
            return self._unknown_hardware_snapshot(), False
        manager_id = (deployment.get("settings") or {}).get("manager_deployment_id")
        if not manager_id:
            return self._hardware_snapshot(), True
        try:
            if serving_member is None:
                if require_serving_member:
                    return self._unknown_hardware_snapshot(), False
                _, serving_member = self.manager._cluster_primary_member(manager_id)
            node_id = serving_member.get("node_id")
            if node_id == "local":
                return self._hardware_snapshot(), True
            if not node_id:
                return self._unknown_hardware_snapshot(), False
            stats = await self.manager.node_registry.request(
                node_id, "GET", "/api/agent/stats", timeout=5,
            )
            if not isinstance(stats, dict):
                return self._unknown_hardware_snapshot(), False
            return self._hardware_snapshot(stats), True
        except Exception:
            # Hardware collection must never make an otherwise valid inference
            # fail. Unknown remote hardware remains local-only evidence.
            return self._unknown_hardware_snapshot(), False

    def _hardware_snapshot(self, stats: dict[str, Any] | None = None) -> dict[str, Any]:
        if stats is None:
            stats = getattr(self.manager, "_stats_cache", {}) or {}
        gpus = stats.get("gpus") or []
        public_gpus = [
            {"model": gpu.get("name"), "memory_mib": gpu.get("mem_total_mib")}
            for gpu in gpus if isinstance(gpu, dict) and gpu.get("name")
        ]
        names = " ".join(str(gpu.get("model") or "") for gpu in public_gpus).casefold()
        return {
            "architecture": platform.machine(),
            "hardware_class": "dgx-spark" if "gb10" in names or "dgx spark" in names else "local",
            "gpu_count": len(public_gpus),
            "gpus": public_gpus,
        }

    @staticmethod
    def _unknown_hardware_snapshot() -> dict[str, Any]:
        return {"hardware_class": "unknown", "gpu_count": None, "gpus": []}

    @staticmethod
    def _validate_base_url(value: Any) -> str:
        text = str(value or "").strip().rstrip("/")
        parsed = urlparse(text)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("base_url must be an http or https URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("base_url must not contain credentials, query parameters, or fragments")
        return normalize_openai_base_url(text)

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


def _bounded_benchmark_integer(value: Any, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise ValueError(f"{name} must be an integer")
        parsed = int(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text or not text.lstrip("+-").isdigit():
            raise ValueError(f"{name} must be an integer")
        parsed = int(text)
    else:
        raise ValueError(f"{name} must be an integer")
    minimum = 0 if allow_zero else 1
    if parsed < minimum or parsed > 100_000_000:
        raise ValueError(f"{name} is outside the supported range")
    return parsed


def _positive_finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive number") from exc
    if not math.isfinite(parsed) or parsed <= 0 or parsed > 86_400:
        raise ValueError(f"{name} is outside the supported range")
    return parsed


def _requested_node_ids(body: dict[str, Any]) -> list[str] | None:
    raw = body.get("node_ids")
    scalar = body.get("node_id")
    if raw is None and scalar is None:
        return None
    if raw is None:
        raw = []
    if not isinstance(raw, list):
        raise ValueError("node_ids must be an array")
    values = [*raw, scalar] if scalar is not None else list(raw)
    result = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("node_ids must contain non-empty node IDs")
        node_id = value.strip()
        if node_id not in result:
            result.append(node_id)
    if not result:
        raise ValueError("node_ids must contain at least one node ID")
    return result


def _deployment_status(value: Any) -> str:
    status = str(value or "unknown").casefold()
    if status in ("running", "ready"):
        return "running"
    if status in ("created", "restarting", "launching", "starting", "recovering"):
        return "starting"
    if status in ("exited", "dead", "removed", "stopped"):
        return "stopped"
    if status in ("error", "unhealthy", "degraded"):
        return "error"
    return "unknown"


def _deployment_launch_progress(deployment: dict[str, Any]) -> dict[str, str]:
    """Flatten honest per-member launch state for deployment list cards."""
    if deployment.get("status") == "error" or deployment.get("error"):
        return {
            "launch_phase": "error",
            "launch_message": str(
                deployment.get("error") or "Deployment launch failed"
            ),
        }
    members = [
        member for member in (deployment.get("members") or [])
        if isinstance(member, dict)
    ]
    # Report the least-advanced active rank. Rank order is only a tie-breaker:
    # a queued worker must win over a rank-0 image pull, and an image pull must
    # win over another rank that has already started loading model weights.
    phase_order = {
        "queued": 0,
        "recovering": 0,
        "preparing": 1,
        "checking_image": 2,
        "pulling_image": 3,
        "creating": 4,
        "creating_container": 4,
        "created": 5,
        "starting": 5,
        "running": 5,
        "downloading": 6,
        "loading": 7,
        "initializing": 8,
        "ready": 9,
    }
    terminal_order = {
        "error": 0, "dead": 1, "unreachable": 2,
        "missing": 3, "unknown": 4,
    }

    def member_order(member: dict[str, Any]) -> tuple[int, int, int]:
        phase = member.get("phase")
        raw_phase = phase.get("phase") if isinstance(phase, dict) else phase
        phase_name = str(
            raw_phase or member.get("status") or "starting"
        ).casefold()
        rank = int(member.get("rank") or 0)
        if phase_name in terminal_order:
            # Another rank may still be actively pulling/creating/loading.
            # Keep reporting that work until no active phase remains; only
            # then surface terminal inventory state ahead of already-ready ranks.
            return 1, terminal_order[phase_name], rank
        if phase_name == "ready":
            return 2, phase_order[phase_name], rank
        return 0, phase_order.get(phase_name, phase_order["starting"]), rank

    members.sort(key=member_order)
    for member in members:
        raw_phase = member.get("phase")
        phase = raw_phase if isinstance(raw_phase, dict) else {}
        phase_name = str(
            phase.get("phase") or raw_phase or member.get("status") or ""
        ).strip()
        message = str(phase.get("message") or "").strip()
        if not message and member.get("error"):
            message = str(member["error"])
        if phase_name or message:
            public_phase = (
                "error" if phase_name.casefold() in terminal_order else phase_name
            )
            return {
                "launch_phase": public_phase or "starting",
                "launch_message": (
                    message or phase_name.replace("_", " ").title()
                ),
            }
    status = str(deployment.get("status") or "starting").strip()
    return {
        "launch_phase": status,
        "launch_message": "Preparing deployment launch",
    }


def _managed_container_status(container: dict[str, Any]) -> str:
    status = _deployment_status(container.get("status"))
    if status != "running":
        return status
    phase = container.get("phase") or {}
    return "running" if phase.get("phase") == "ready" else "starting"


def _positive_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _is_missing_container_error(exc: Exception) -> bool:
    if exc.__class__.__name__ == "NotFound":
        return True
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    return status == 404


def _artifact_is_controller_local(artifact: str | None) -> bool:
    """True when the artifact names a file on this host (platform-aware)."""
    if not artifact:
        return False
    try:
        return Path(artifact).expanduser().is_absolute()
    except (OSError, ValueError, RuntimeError):
        # An unresolvable "~user" home makes the reference unusable; callers
        # treat this as "not controller-local" and repo-relative validation
        # rejects it as a malformed artifact.
        return False


def _public_model_id(value: str) -> str:
    """Exclude endpoint URLs and local paths from the upload-facing identity."""
    text = str(value or "").strip()
    if (
        not text
        or urlparse(text).scheme
        or Path(text).is_absolute()
        or text.startswith(("./", "../", "~"))
        or Path(text).expanduser().exists()
        or "\\" in text
        or text.casefold().endswith((".gguf", ".bin", ".safetensors", ".pt", ".pth"))
        or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", text
        )
    ):
        return "local-model"
    return text


def _local_benchmark_model_id(
    value: str, deployment_id: str | None, *, upload_model_id: str | None = None,
) -> str:
    """Return a local grouping key without exposing private model identifiers."""
    public_model_id = upload_model_id or _public_model_id(value)
    if public_model_id != "local-model":
        return public_model_id
    private_model_id = str(value or "").strip()
    if re.fullmatch(r"Private model [0-9a-f]{8}", private_model_id):
        return private_model_id
    identity = json.dumps(
        [str(deployment_id or "").strip(), private_model_id],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(identity).hexdigest()[:16]
    # This deliberately does not resemble an owner/repository ID, so the
    # upload-facing sanitizer will reject it if it ever reaches that boundary.
    return f"Private model {digest[:8]}"


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


def _rewrite_sse_model(text: str, alias: str) -> str:
    """Rewrite model identities without disturbing SSE framing or event types."""
    output = []
    for framed_line in text.splitlines(keepends=True):
        line = framed_line.rstrip("\r\n")
        ending = framed_line[len(line):]
        parsed = _parse_sse(line)
        if parsed is not None and "model" in parsed:
            parsed["model"] = alias
            line = "data: " + json.dumps(parsed, separators=(",", ":"))
        output.append(line + ending)
    return "".join(output)


def _chunk_has_output(chunk: dict[str, Any]) -> bool:
    for choice in chunk.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta") or {}
        if isinstance(delta, dict) and (
            delta.get("content")
            or delta.get("reasoning_content")
            or delta.get("reasoning")
        ):
            return True
        if choice.get("text"):
            return True
    return False
