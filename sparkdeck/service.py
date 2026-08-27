"""Application service joining runtime adapters, persistence and proxy metrics."""

from __future__ import annotations

import asyncio
import copy
import inspect
import ipaddress
import json
import logging
import math
import platform
import re
import socket
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator
from urllib.parse import urlparse

import httpx

from .catalog import HuggingFaceCatalog
from .models import BenchmarkSample, Deployment, DeploymentKind, ModelIdentity, RuntimeKind
from .runtimes import (
    RuntimeRegistry,
    launch_managed_container,
    normalize_openai_base_url,
    safe_container_name,
)
from .storage import (
    COMMUNITY_EVIDENCE_POLICY,
    SparkDeckStore,
    community_context_window,
)


_SAFE_CONFIGURATION_KEYS = {
    "context_size", "context_length", "max_model_len", "parallel", "parallel_slots",
    "gpu_layers", "split_mode", "tensor_split", "gpu_split", "tensor_parallel_size",
    "pipeline_parallel_size", "data_parallel_size", "quantization", "dtype",
    "max_running_requests", "mem_fraction_static", "gpu_memory_utilization",
    "runtime_version",
}
_LOCAL_ROUTING_KEYS = {"deployment_mode", "node_ids", "manager_deployment_id"}
_COMMUNITY_REDIRECT_STATUSES = {301, 302, 303, 307, 308}
_COMMUNITY_MAX_REDIRECTS = 5
_COMMUNITY_MAX_RESPONSE_BYTES = 8 * 1024 * 1024
_COMMUNITY_UPLOAD_INTERVAL_SECONDS = 5.0
logger = logging.getLogger(__name__)


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
            assert last_connect_error is not None
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


def _community_upload_url(endpoint: str) -> httpx.URL:
    """Append the benchmark-ingestion route to a configured API base."""
    try:
        url = httpx.URL(endpoint)
    except (TypeError, httpx.InvalidURL) as exc:
        raise ValueError("community upload URL is invalid") from exc
    if url.scheme != "https":
        raise ValueError("community upload URL must use HTTPS")
    path = url.path.rstrip("/")
    if not path.endswith("/benchmarks"):
        path = f"{path}/benchmarks"
    return url.copy_with(path=path or "/benchmarks", fragment=None)


async def _post_public_community_url(
    url: str | httpx.URL, payload: dict[str, Any], *, token: str,
    idempotency_key: str, transport: httpx.AsyncBaseTransport | None = None,
    resolver: Any = socket.getaddrinfo,
) -> httpx.Response:
    """POST one sample to a DNS-pinned public URL without forwarding redirects."""
    try:
        target = httpx.URL(url)
    except (TypeError, httpx.InvalidURL) as exc:
        raise ValueError("community upload URL is invalid") from exc
    if target.scheme != "https":
        raise ValueError("community upload URL must use HTTPS")
    pinned_urls, host_header, sni_hostname = await _public_connection_urls(
        target, resolver,
    )
    last_connect_error: httpx.HTTPError | None = None
    for pinned in pinned_urls:
        async with httpx.AsyncClient(
            transport=transport,
            trust_env=False,
            follow_redirects=False,
            timeout=15,
        ) as client:
            request = client.build_request(
                "POST",
                pinned,
                headers={
                    "Host": host_header,
                    "Connection": "close",
                    "Authorization": f"Bearer {token}",
                    "Idempotency-Key": idempotency_key,
                    "Content-Type": "application/json",
                },
                json=payload,
                extensions={"sni_hostname": sni_hostname},
            )
            try:
                streamed = await client.send(request, stream=True)
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                last_connect_error = exc
                continue
            try:
                response = httpx.Response(
                    streamed.status_code,
                    headers=streamed.headers,
                    request=streamed.request,
                    extensions=streamed.extensions,
                )
            finally:
                # Upload responses have no response-body contract. Never read
                # or buffer an untrusted body merely to inspect the status.
                await streamed.aclose()
            if response.status_code in _COMMUNITY_REDIRECT_STATUSES:
                raise ValueError("community upload service must not redirect")
            return response
    assert last_connect_error is not None
    raise last_connect_error


def _public_community_aggregates(payload: Any) -> list[dict[str, Any]]:
    """Validate and bound the public response from a configured aggregator."""
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise ValueError("community aggregate response must contain an items array")
    if len(payload["items"]) > 10_000:
        raise ValueError("community aggregate response is too large")

    result: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for raw in payload["items"]:
        if not isinstance(raw, dict):
            raise ValueError("community aggregate item must be an object")
        model_id = str(raw.get("model_id") or "").strip()
        context_window = raw.get("context_window_size")
        speed = raw.get("inference_tokens_per_second")
        sample_count = raw.get("sample_count")
        if (
            not model_id
            or len(model_id) > 500
            or any(ord(char) < 32 for char in model_id)
            or isinstance(context_window, bool)
            or not isinstance(context_window, int)
            or isinstance(sample_count, bool)
            or not isinstance(sample_count, int)
            or isinstance(speed, bool)
            or not isinstance(speed, (int, float))
        ):
            raise ValueError("community aggregate item is invalid")
        speed = float(speed)
        if (
            context_window <= 0
            or context_window > 100_000_000
            or sample_count <= 0
            or sample_count > 1_000_000_000
            or not math.isfinite(speed)
            or speed <= 0
        ):
            raise ValueError("community aggregate item is invalid")
        key = (model_id, context_window)
        if key in seen:
            raise ValueError("community aggregate response contains duplicate evidence")
        seen.add(key)
        result.append({
            "model_id": model_id,
            "context_window_size": context_window,
            "inference_tokens_per_second": speed,
            "sample_count": sample_count,
        })
    return result


class SparkDeckService:
    def __init__(
        self, manager: Any, data_dir: Path, *,
        community_upload_url: str = "", community_upload_token: str = "",
    ):
        self.manager = manager
        self.store = SparkDeckStore(Path(data_dir) / "sparkdeck.sqlite3")
        self.registry = RuntimeRegistry()
        self.catalog = HuggingFaceCatalog(
            manager.http,
            token_provider=lambda: getattr(manager, "_resolved_hf_token", lambda: "")(),
        )
        self._deployment_create_lock = asyncio.Lock()
        self._community_upload_task: asyncio.Task[None] | None = None
        self._community_upload_lock = asyncio.Lock()
        self._community_upload_drain_lock = asyncio.Lock()
        # Upload authentication is deliberately process configuration, not a
        # browser token or user-editable setting. Deployments can issue a
        # revocable, node-scoped telemetry credential without exposing Cognito
        # credentials to peers or arbitrary configured aggregate endpoints.
        self._community_upload_url = community_upload_url.strip().rstrip("/")
        self._community_upload_token = community_upload_token.strip()

    async def start(self) -> None:
        if self._community_upload_task is None:
            self._community_upload_task = asyncio.create_task(
                self._community_upload_loop(), name="sparkdeck-community-upload",
            )

    @property
    def community_upload_configured(self) -> bool:
        return bool(self._community_upload_url and self._community_upload_token)

    async def close(self) -> None:
        if self._community_upload_task is not None:
            self._community_upload_task.cancel()
            try:
                await self._community_upload_task
            except asyncio.CancelledError:
                pass
            self._community_upload_task = None
        self.store.close()

    async def set_community_consent(self, enabled: bool) -> None:
        """Serialize consent withdrawal with all outbound uploads."""
        async with self._community_upload_lock:
            await asyncio.to_thread(self.store.set_community_consent, enabled)

    async def unpair_community_device(
        self, expected_sub: str,
    ) -> tuple[str, dict[str, Any]]:
        """Serialize account-matched unpairing with outbound upload batches."""
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

    async def upload_community_once(self) -> int:
        async with self._community_upload_drain_lock:
            return await self._upload_community_once_locked()

    async def _upload_community_once_locked(self) -> int:
        """Drain one bounded outbox batch when consent and credentials exist."""
        endpoint = self._community_upload_url
        token = self._community_upload_token
        if not endpoint or not token:
            return 0

        entries = await asyncio.to_thread(self.store.outbox_entries)
        synced = 0
        upload_url = _community_upload_url(endpoint)
        for entry in entries:
            async with self._community_upload_lock:
                consent = await asyncio.to_thread(
                    self.store.get_setting, "community_consent", False,
                )
                pairing = await asyncio.to_thread(
                    self.store.get_setting,
                    "device_pairing", {"status": "not_paired"},
                )
                if not consent or pairing.get("status") != "paired":
                    break
                sample_id = entry["sample_id"]
                try:
                    response = await _post_public_community_url(
                        upload_url,
                        entry["payload"],
                        token=token,
                        idempotency_key=sample_id,
                        transport=getattr(
                            self.manager, "community_http_transport", None,
                        ),
                        resolver=getattr(
                            self.manager, "community_resolver", socket.getaddrinfo,
                        ),
                    )
                    if response.is_success or response.status_code == 409:
                        await asyncio.to_thread(
                            self.store.mark_outbox_synced, [sample_id],
                        )
                        synced += 1
                    else:
                        await asyncio.to_thread(
                            self.store.mark_outbox_failed,
                            [sample_id],
                            f"community service returned HTTP {response.status_code}",
                        )
                except asyncio.CancelledError:
                    raise
                except (httpx.HTTPError, TypeError, ValueError) as exc:
                    await asyncio.to_thread(
                        self.store.mark_outbox_failed, [sample_id], str(exc),
                    )
        return synced

    async def _community_upload_loop(self) -> None:
        while True:
            try:
                await self.upload_community_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                # Per-item failures are persisted; this protects the lifecycle
                # task from an unexpected storage or configuration failure.
                logger.exception("community upload worker failed")
            await asyncio.sleep(_COMMUNITY_UPLOAD_INTERVAL_SECONDS)

    async def community_aggregates(self) -> dict[str, Any]:
        """Return configured community evidence or privacy-safe local evidence."""
        endpoint = str(
            await asyncio.to_thread(
                self.store.get_setting, "community_api_url", ""
            ) or ""
        ).strip().rstrip("/")
        if not endpoint:
            items = await asyncio.to_thread(self.store.community_aggregates)
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
        except (httpx.HTTPError, TypeError, ValueError) as exc:
            raise RuntimeError("community aggregate service is unavailable") from exc
        return {
            "items": items,
            "availability": "available",
            "evidence_policy": COMMUNITY_EVIDENCE_POLICY,
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

    async def deployments(self) -> list[dict[str, Any]]:
        registered = self.store.deployments(include_private=True)
        by_container = {item.get("container_name"): item for item in registered}
        cluster_state: dict[str, Any] = {}
        if any(item.get("settings", {}).get("manager_deployment_id") for item in registered):
            try:
                cluster_state = await self.manager.get_state()
            except Exception:
                cluster_state = {}
        raw_manager_deployments = getattr(self.manager, "deployments", [])
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
        cluster_nodes: dict[str, dict[str, Any]] = {}
        if any(item.get("settings", {}).get("manager_deployment_id") for item in registered):
            try:
                cluster_nodes = {
                    node["id"]: node for node in await self.manager.cluster_nodes()
                }
            except Exception:
                cluster_nodes = {}
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
            stored["status"] = _deployment_status(cluster.get("status"))
            stored["port"] = cluster.get("api_port")
            stored["managed"] = True
            created_at = cluster.get("created_at")
            if isinstance(created_at, (int, float)) and created_at:
                stored["created_at"] = datetime.fromtimestamp(
                    created_at, timezone.utc
                ).isoformat()
            elif isinstance(created_at, str) and created_at:
                stored["created_at"] = created_at
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
        by_container.update(local_cluster_members)
        for container in containers:
            runtime = self._container_runtime(container)
            if runtime not in self.registry.kinds:
                continue
            stored = by_container.get(container.get("name"))
            if stored:
                stored["status"] = _managed_container_status(container)
                stored["port"] = container.get("port")
                stored["managed"] = True
                seen.add(stored["id"])
                continue
            model = container.get("model") or container.get("served_model")
            if not model:
                continue
            registered.append(self._discovered_deployment(container, runtime, model))
        for deployment in registered:
            if (
                deployment.get("kind") == DeploymentKind.MANAGED.value
                and deployment.get("id") not in seen
                and not str(deployment.get("id") or "").startswith("container:")
            ):
                deployment["status"] = "missing"
                deployment["last_error"] = (
                    "Docker is unavailable" if docker_unavailable
                    else "Managed container is missing"
                )
        async def probe_external(deployment: dict[str, Any]) -> None:
            if deployment.get("kind") != DeploymentKind.EXTERNAL.value:
                return
            try:
                adapter = self.registry.get(deployment["runtime"])
                await asyncio.wait_for(
                    adapter.health(
                        self.manager.http,
                        deployment.get("_base_url") or "",
                        self._get_credential(
                            deployment["id"], deployment.get("_credential_ref")
                        ),
                    ),
                    timeout=3,
                )
                deployment["status"] = "running"
            except Exception:
                deployment["status"] = "error"
                deployment["last_error"] = "Endpoint health check failed"

        await asyncio.gather(*(probe_external(item) for item in registered))
        for deployment in registered:
            deployment.pop("_base_url", None)
            deployment.pop("_credential_ref", None)
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
        requested_node_ids = _requested_node_ids(body)
        deployment_mode = str(body.get("deployment_mode") or "").strip() or None
        if requested_node_ids is not None and kind is DeploymentKind.EXTERNAL:
            raise ValueError("node_ids are only supported for managed deployments")
        deployment_id = str(uuid.uuid4())
        identity = ModelIdentity(
            repository=model, revision=_optional_string(body.get("revision")),
            artifact=_optional_string(body.get("artifact") or settings.get("artifact")),
            quantization=_optional_string(body.get("quantization") or settings.get("quantization")),
        )
        deployment = Deployment(
            id=deployment_id, alias=alias, runtime=runtime, kind=kind,
            model=identity, settings=self._local_configuration(settings),
            base_url_set=bool(body.get("base_url")),
        )
        # A launch can take minutes, but serializing creation is intentional:
        # alias uniqueness must be established before Docker is mutated.
        async with self._deployment_create_lock:
            if self.store.deployment(alias):
                raise ValueError(f"deployment alias '{alias}' is already in use")
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
                if runtime is RuntimeKind.LLAMA_CPP:
                    raise ValueError(
                        "explicit node selection currently supports managed vLLM and SGLang deployments"
                    )
                selected = await self.manager.selected_cluster_nodes(requested_node_ids)
                mode = deployment_mode or (
                    "replicated" if len(requested_node_ids) > 1 else "single"
                )
                if mode == "single" and len(requested_node_ids) != 1:
                    raise ValueError("single deployment requires exactly one node")
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
                if identity.revision:
                    extra_args += ["--revision", identity.revision]
                launch_body = {
                    **settings,
                    "model": model,
                    "deployment_name": alias,
                    "engine": runtime.value,
                    "deployment_mode": mode,
                    "node_ids": requested_node_ids,
                    "extra_args": extra_args,
                    "managed_by": "sparkdeck",
                    "sparkdeck_record_id": deployment_id,
                    "recipe_id": body.get("recipe_id"),
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

            adapter = self.registry.get(runtime)
            cleanup_name = safe_container_name(alias, deployment_id)
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
                port = launched.get("port")
                if not deployment.container_name or not port:
                    raise RuntimeError("runtime launched without a discoverable container endpoint")
                cleanup_name = deployment.container_name
                self.store.add_deployment(
                    deployment, f"http://127.0.0.1:{int(port)}", None
                )
            except Exception:
                try:
                    await self.manager.remove_container(cleanup_name)
                except Exception:
                    pass
                raise
            result = self.store.deployment(deployment_id) or deployment.to_dict()
            result.update({"status": launched.get("status", "running"), "port": int(port)})
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
            # Preserve the original launch error. Manager still retains its
            # diagnostic record if even local SQLite adoption fails.
            pass

    async def deployment_action(self, deployment_id: str, action: str) -> dict[str, Any]:
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
        if deployment["kind"] != DeploymentKind.MANAGED.value:
            raise ValueError("external endpoints cannot be started or stopped by SparkDeck")
        manager_id = deployment.get("settings", {}).get("manager_deployment_id")
        if manager_id:
            result = await self.manager.deployment_action(manager_id, action)
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
            current = self.store.deployment(deployment_id) or deployment
            current["status"] = "running" if action == "start" else "stopped"
            current["node_ids"] = list(current.get("settings", {}).get("node_ids") or [])
            return current
        container = deployment.get("container_name")
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
        if action == "start":
            await self.manager.start_container(container)
        elif action == "stop":
            await self.manager.stop_container(container)
        else:
            raise ValueError("action must be start or stop")
        current = self.store.deployment(deployment_id) or deployment
        current["status"] = "running" if action == "start" else "stopped"
        return current

    async def delete_deployment(self, deployment_id: str) -> dict[str, Any]:
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
        if discovered and deployment["kind"] != DeploymentKind.MANAGED.value:
            raise ValueError(
                "unmanaged discovered containers cannot be removed by SparkDeck"
            )
        manager_id = deployment.get("settings", {}).get("manager_deployment_id")
        if manager_id:
            result = await self.manager.deployment_action(manager_id, "remove")
            if not result.get("ok"):
                raise RuntimeError("; ".join(result.get("errors") or ["cluster removal failed"]))
        elif deployment["kind"] == DeploymentKind.MANAGED.value and deployment.get("container_name"):
            try:
                await self.manager.remove_container(deployment["container_name"])
            except Exception as exc:
                if not _is_missing_container_error(exc):
                    raise
        if not discovered:
            self._delete_credential(deployment_id, deployment.get("_credential_ref"))
            self.store.delete_deployment(deployment_id)
        return {"ok": True, "id": deployment_id}

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
        return {
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
            "status": _managed_container_status(container),
            "container_name": container.get("name"),
            "settings": self._safe_configuration(container.get("load_settings") or {}),
            "base_url_set": bool(container.get("port")),
            "port": container.get("port"),
            "managed": bool(container.get("managed")),
        }

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
            await self.manager.proxy_chat_completions(body, cancel)
            if endpoint == "chat/completions"
            else await self.manager.proxy_completions(body, cancel)
        )
        runtime = await self._runtime_for_legacy_model(requested_model)
        if hasattr(result, "__aiter__"):
            return self._observe_stream(result, None, requested_model, runtime, {}, started)
        self._record_response(None, requested_model, runtime, {}, started, result)
        return result

    async def _proxy_registered(self, deployment: dict[str, Any], body: dict[str, Any],
                                endpoint: str, cancel: Any) -> dict[str, Any] | AsyncIterator[str]:
        if (
            deployment.get("kind") == DeploymentKind.MANAGED.value
            and deployment.get("runtime") in (RuntimeKind.VLLM.value, RuntimeKind.SGLANG.value)
        ):
            return await self._proxy_managed(deployment, body, endpoint, cancel)
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
            deployment["runtime"], deployment.get("settings") or {}, started, data,
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
                             endpoint: str, cancel: Any) -> dict[str, Any] | AsyncIterator[str]:
        """Keep managed vLLM/SGLang requests on Manager's admission path."""
        model = deployment["model"]["repository"]
        upstream_body = {**body, "model": model}
        stream = bool(upstream_body.get("stream"))
        started = time.monotonic()
        settings = deployment.get("settings") or {}
        manager_id = settings.get("manager_deployment_id")
        hardware, hardware_verified = await self._managed_hardware_snapshot(deployment)
        result = (
            await self.manager.proxy_cluster_inference(
                manager_id, model, upstream_body, endpoint, cancel,
            )
            if manager_id
            else (
                await self.manager._vllm_chat(
                    model, upstream_body, stream, cancel,
                    container_name=deployment.get("container_name"),
                    deployment_id=deployment["id"],
                )
                if endpoint == "chat/completions"
                else await self.manager._vllm_completions(
                    model, upstream_body, stream, cancel,
                    container_name=deployment.get("container_name"),
                    deployment_id=deployment["id"],
                )
            )
        )
        revision = deployment["model"].get("revision")
        if hasattr(result, "__aiter__"):
            return self._observe_stream(
                result, deployment["id"], model, deployment["runtime"], settings,
                started, revision=revision, response_model=deployment["alias"],
                hardware=hardware, hardware_verified=hardware_verified,
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
                deployment["runtime"], deployment.get("settings") or {}, started,
                usage, first_token_at,
                revision=deployment["model"].get("revision"),
                hardware=self._unknown_hardware_snapshot(), hardware_verified=False,
            )

    async def _observe_stream(self, stream: AsyncIterator[str], deployment_id: str | None,
                              model: str, runtime: str, settings: dict[str, Any],
                              started: float, revision: str | None = None,
                              response_model: str | None = None,
                              hardware: dict[str, Any] | None = None,
                              hardware_verified: bool = True) -> AsyncIterator[str]:
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
            self._record_usage(
                deployment_id, model, runtime, settings, started, usage,
                first_token_at, revision=revision, hardware=hardware,
                hardware_verified=hardware_verified,
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
                      hardware_verified: bool = True) -> None:
        completed = time.monotonic()
        input_tokens = max(0, int(usage.get("prompt_tokens") or 0))
        output_tokens = max(0, int(usage.get("completion_tokens") or 0))
        latency = max(0.000001, completed - started)
        generation_seconds = max(0.000001, completed - (first_token_at or started))
        runtime_kind = RuntimeKind(runtime)
        safe_settings = self._safe_configuration(settings)
        quantization = _optional_string(safe_settings.get("quantization"))
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
        eligible = bool(
            public_model != "local-model" and input_tokens > 0 and output_tokens >= 16 and latency > 0
            and generation_tps is not None
            and community_context_window(safe_settings) is not None
            and runtime_kind.value in self.registry.kinds
            and hardware_verified
        )
        sample = BenchmarkSample(
            id=str(uuid.uuid4()), created_at=datetime.now(timezone.utc).isoformat(),
            deployment_id=deployment_id,
            model=ModelIdentity(
                repository=public_model,
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
        consent = bool(self.store.get_setting("community_consent", False))
        self.store.add_benchmark(sample, queue=eligible and consent)

    async def _runtime_for_legacy_model(self, model: str) -> str:
        if await self._native_llama_model() == model:
            return RuntimeKind.LLAMA_CPP.value
        try:
            for container in await self.manager.list_containers():
                ids = self.manager._container_model_ids(container)
                if model in ids:
                    return self._container_runtime(container)
        except Exception:
            pass
        return RuntimeKind.VLLM.value

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
        self, deployment: dict[str, Any]
    ) -> tuple[dict[str, Any], bool]:
        """Resolve benchmark hardware from the cluster member that serves requests."""
        manager_id = (deployment.get("settings") or {}).get("manager_deployment_id")
        if not manager_id:
            return self._hardware_snapshot(), True
        try:
            _, primary = self.manager._cluster_primary_member(manager_id)
            node_id = primary.get("node_id")
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
    if status in ("created", "restarting", "launching", "starting"):
        return "starting"
    if status in ("exited", "dead", "removed", "stopped"):
        return "stopped"
    if status in ("error", "unhealthy"):
        return "error"
    return "unknown"


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


def _public_model_id(value: str) -> str:
    """Exclude endpoint URLs and local paths from persistent benchmark identity."""
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
        if isinstance(delta, dict) and (delta.get("content") or delta.get("reasoning_content")):
            return True
        if choice.get("text"):
            return True
    return False
