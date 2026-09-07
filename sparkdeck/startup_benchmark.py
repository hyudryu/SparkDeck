"""Consent-gated, one-shot decode measurements for newly started engines."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from .stream_cleanup import close_async_stream

log = logging.getLogger(__name__)

# Inventory polling is expensive (full deployment reconciliation, Docker
# inventory, cluster state, and health probes). Poll fast only while a new
# boot is being discovered or benchmarked; back way off in the stable state
# where every discovered boot is already seen.
_POLL_INITIAL_INTERVAL = 2.0
_POLL_MAX_INTERVAL = 30.0

# A probe that completes without recording an eligible sample (e.g. an older
# llama.cpp/SGLang endpoint that cannot stream terminal usage, a short output,
# or a timeout) must not be re-attempted every poll. Back off exponentially
# per fingerprint so a non-transient incompatibility cannot become a permanent
# synthetic GPU workload, while still allowing transient failures to retry.
_RETRY_BACKOFF_BASE = 60.0
_RETRY_BACKOFF_MAX = 3600.0
_RETRY_BACKOFF_MAX_EXPONENT = 8


@dataclass
class StartupTarget:
    deployment: dict[str, Any]
    fingerprint: str
    cluster: dict[str, Any] | None = None
    member: dict[str, Any] | None = None


def startup_fingerprint(containers: list[tuple[str, dict[str, Any]]]) -> str | None:
    """Require real start timestamps; a Docker name alone cannot identify a boot."""
    if not containers or any(not item.get("started_at") for _, item in containers):
        return None
    identity = sorted(
        (node, str(item.get("name")), str(item.get("id") or ""),
         str(item["started_at"]), str(item.get("restart_count") or 0))
        for node, item in containers
    )
    return hashlib.sha256(json.dumps(identity).encode()).hexdigest()


class StartupBenchmarkMonitor:
    def __init__(self, service: Any):
        self.service = service
        self.manager = service.manager
        self._tasks: dict[str, asyncio.Task] = {}
        self._serial = asyncio.Lock()
        # Per-fingerprint retry/backoff state so a probe that records no sample
        # is not re-attempted on every poll.
        self._retry_after: dict[str, float] = {}
        self._retry_attempts: dict[str, int] = {}

    def cancel_active(self) -> None:
        """Cancel every in-flight startup probe immediately (consent withdrew)."""
        for task in list(self._tasks.values()):
            task.cancel()

    async def run(self) -> None:
        try:
            interval = _POLL_INITIAL_INTERVAL
            while True:
                try:
                    progress = await self.tick()
                except Exception:
                    log.exception("Startup benchmark inventory failed")
                    progress = False
                if progress:
                    interval = _POLL_INITIAL_INTERVAL
                else:
                    # No unseen boot exists (or consent is off), so back off
                    # rather than repeatedly running the expensive inventory.
                    interval = min(interval * 2, _POLL_MAX_INTERVAL)
                await asyncio.sleep(interval)
        finally:
            tasks = list(self._tasks.values())
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def tick(self) -> bool:
        """Discover and start benchmark tasks; return True if any new boot began."""
        snapshot = self.service.store.community_consent_snapshot()
        if not snapshot.get("enabled"):
            for task in self._tasks.values():
                task.cancel()
            return False
        targets = await self.targets()
        progress = False
        for target in targets:
            key = target.fingerprint
            if key in self._tasks or self.service.store.get_setting(self._seen_key(key), False):
                continue
            retry_after = max(
                self._retry_after.get(key, 0.0), self._retry_deadline(key),
            )
            if time.time() < retry_after:
                continue
            task = asyncio.create_task(self._attempt(target, snapshot))
            self._tasks[key] = task
            task.add_done_callback(lambda done, key=key: self._tasks.pop(key, None))
            progress = True
        return progress

    @staticmethod
    def _seen_key(fingerprint: str) -> str:
        return "community_startup_benchmark:" + fingerprint

    @staticmethod
    def _retry_key(fingerprint: str) -> str:
        return "community_startup_benchmark_retry:" + fingerprint

    def _retry_deadline(self, fingerprint: str) -> float:
        """Return the persisted wall-clock deadline for the next probe."""
        state = self.service.store.get_setting(self._retry_key(fingerprint), {})
        if not isinstance(state, dict):
            return 0.0
        try:
            return float(state.get("retry_after") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _back_off(self, fingerprint: str) -> None:
        """Persist exponential retry state so restarts cannot create a probe loop."""
        state = self.service.store.get_setting(self._retry_key(fingerprint), {})
        try:
            previous = int(state.get("attempts") or 0) if isinstance(state, dict) else 0
        except (TypeError, ValueError):
            previous = 0
        # Attempts only select a bounded exponential delay. Cap the persisted
        # counter too, so corrupt or long-lived state cannot make exponentiation
        # overflow before the maximum delay is applied.
        attempts = min(max(previous, 0) + 1, _RETRY_BACKOFF_MAX_EXPONENT + 1)
        delay = min(
            _RETRY_BACKOFF_BASE
            * (2 ** min(attempts - 1, _RETRY_BACKOFF_MAX_EXPONENT)),
            _RETRY_BACKOFF_MAX,
        )
        retry_after = time.time() + delay
        self._retry_attempts[fingerprint] = attempts
        self._retry_after[fingerprint] = retry_after
        self.service.store.set_setting(
            self._retry_key(fingerprint),
            {"attempts": attempts, "retry_after": retry_after},
        )

    async def targets(self) -> list[StartupTarget]:
        deployments = await self.service.deployments()
        nodes = await self.manager.cluster_nodes()
        inventory = {
            (str(node.get("id")), str(container.get("name"))): container
            for node in nodes if node.get("online", True)
            for container in node.get("containers") or []
            if container.get("status") == "running"
        }
        # Local inventory includes legacy containers as well as cluster ranks.
        for container in await self.manager.list_containers():
            if container.get("status") == "running":
                inventory[("local", str(container.get("name")))] = container
        clusters = {str(item.get("id")): item for item in self.manager.deployments}
        targets = []
        for public in deployments:
            if public.get("kind") != "managed" or public.get("desired_state") == "stopped":
                continue
            stored = self.service.store.deployment(public["id"], include_private=True) or {}
            deployment = {**stored, **public}
            cluster = clusters.get(str((deployment.get("settings") or {}).get("manager_deployment_id")))
            if cluster:
                if cluster.get("desired_state") == "stopped":
                    continue
                members = self.manager._cluster_members_sorted(cluster)
                mode = cluster.get("mode")
                serving = (self.manager._grouped_coordinators(cluster) if mode == "grouped_sharded"
                           else members if mode == "replicated" else members[:1])
                for member in serving:
                    if member.get("desired_state") == "stopped":
                        continue
                    ranks = ([m for m in members if m.get("instance_id") == member.get("instance_id")]
                             if mode == "grouped_sharded" else [member] if mode == "replicated" else members)
                    containers = [(str(m.get("node_id")), inventory.get(
                        (str(m.get("node_id")), str(m.get("container_name"))), {})) for m in ranks]
                    fingerprint = startup_fingerprint(containers)
                    if fingerprint:
                        targets.append(StartupTarget(deployment, fingerprint, cluster, member))
            else:
                container = inventory.get(("local", str(deployment.get("container_name"))), {})
                # Remote agents also discover controller-owned rank containers.
                # Only the controller with the linked deployment may benchmark
                # those engines; agents must not issue a second competing probe.
                owner = container.get("deployment_id")
                if owner and owner != deployment["id"]:
                    continue
                fingerprint = startup_fingerprint([("local", container)])
                if fingerprint:
                    if not deployment.get("_base_url") and container.get("port"):
                        deployment["_base_url"] = f"http://127.0.0.1:{int(container['port'])}"
                    targets.append(StartupTarget(deployment, fingerprint))
        return targets

    async def _healthy(self, target: StartupTarget) -> bool:
        deployment = target.deployment
        if target.cluster:
            if target.member.get("node_id") == "local":
                return await self.manager.inference_target_health(
                    deployment["model"]["repository"],
                    container_name=target.member.get("container_name"),
                    deployment_id=target.cluster["id"],
                    strict_health=True,
                )
            result = await self.manager.node_registry.request(
                target.member["node_id"], "POST", "/api/agent/inference/health",
                json_body={
                    "model": deployment["model"]["repository"],
                    "_sparkdeck_container_name": target.member.get("container_name"),
                    "_sparkdeck_deployment_id": target.cluster["id"],
                    "strict_health": True,
                }, timeout=10,
            )
            return (result or {}).get("health_status") == 200
        if deployment.get("runtime") in {"vllm", "sglang"}:
            return await self.manager.inference_target_health(
                deployment["model"]["repository"], container_name=deployment.get("container_name"),
                deployment_id=deployment["id"],
                strict_health=True,
            )
        base_url = str(deployment.get("_base_url") or "").rstrip("/")
        if not base_url:
            return False
        key = self.service._get_credential(deployment["id"], deployment.get("_credential_ref"))
        response = await self.manager.http.get(
            f"{base_url}/health", headers={"Authorization": f"Bearer {key}"} if key else {}, timeout=5,
        )
        return response.status_code == 200

    async def _attempt(self, target: StartupTarget, snapshot: dict[str, Any]) -> None:
        try:
            if not await self._healthy(target):
                return
            async with self._serial:
                current = self.service.store.community_consent_snapshot()
                if not current.get("enabled") or current.get("generation") != snapshot.get("generation"):
                    return
                # Do not benchmark a replacement engine using a queued boot's identity.
                fresh = next((item for item in await self.targets()
                              if item.fingerprint == target.fingerprint), None)
                if fresh is None or not await self._healthy(fresh):
                    return
                scopes = self.service._community_observation_scopes(fresh.deployment, fresh.deployment["id"])
                if any(scopes.intersection(item.get("scopes") or ())
                       for item in self.service._community_active_observations.values()):
                    return
                if getattr(self.manager, "_active_reqs", {}):
                    return
                if getattr(self.service, "_startup_benchmark_busy", lambda: False)():
                    return
                try:
                    recorded = await asyncio.wait_for(
                        self._benchmark(fresh, snapshot), timeout=120,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    log.debug("Startup benchmark measurement failed", exc_info=True)
                    recorded = False
                if recorded:
                    self._retry_after.pop(target.fingerprint, None)
                    self._retry_attempts.pop(target.fingerprint, None)
                    self.service.store.set_setting(
                        self._retry_key(target.fingerprint), None,
                    )
                else:
                    # No eligible sample was recorded. Back off this fingerprint
                    # so the synthetic probe is not retried on every poll.
                    self._back_off(target.fingerprint)
        except Exception:
            log.debug("Startup benchmark skipped or failed", exc_info=True)

    async def _benchmark(self, target: StartupTarget, snapshot: dict[str, Any]) -> bool:
        deployment = target.deployment
        model = deployment["model"]["repository"]
        observation = self.service._community_observation_start(
            self.service._community_observation_scopes(deployment, deployment["id"]),
        )
        observation.update(
            startup_benchmark=True, generation=snapshot.get("generation"),
            seen_key=self._seen_key(target.fingerprint),
            manager_request_sequence=getattr(self.manager, "_req_seq", 0),
            manager_requests_expected=int(bool(target.cluster) or deployment["runtime"] in {"vllm", "sglang"}),
        )
        token = self.service._community_observation.set(observation)
        stream = None
        try:
            body = {
                "model": model, "prompt": "Write a detailed description of a peaceful garden in spring.",
                "max_tokens": 200, "temperature": 0, "stream": True,
                "stream_options": {"include_usage": True}, "ignore_eos": True,
            }
            started = time.monotonic()
            if target.cluster:
                upstream = await self.manager._proxy_cluster_member(
                    target.cluster, target.member, model, body, "completions", None,
                    startup_benchmark=True,
                )
                async def hardware():
                    return await self.service._managed_hardware_snapshot(
                        deployment, target.member, require_serving_member=True,
                    )
                stream = self.service._observe_stream(
                    upstream, deployment["id"], model, deployment["runtime"],
                    self.service._model_observation_settings(deployment), started,
                    revision=deployment["model"].get("revision"), hardware_resolver=hardware,
                )
            else:
                stream = await self.service._proxy_registered(
                    deployment, body, "completions", None, startup_benchmark=True,
                )
                if deployment["runtime"] == "llama.cpp" and hasattr(stream, "__aiter__"):
                    # The normal HTTP relay deliberately treats arbitrary
                    # external endpoints as untrusted. This target has verified
                    # local container ownership and is consumed independently.
                    stream = self.service._observe_stream(
                        stream, deployment["id"], model, deployment["runtime"],
                        self.service._model_observation_settings(deployment), started,
                        revision=deployment["model"].get("revision"),
                        hardware=self.service._hardware_snapshot(),
                        hardware_verified=deployment.get("kind") == "managed",
                    )
            if hasattr(stream, "__aiter__"):
                async for _ in stream:
                    pass
        finally:
            try:
                if hasattr(stream, "__aiter__"):
                    await close_async_stream(stream)
            finally:
                self.service._community_observation.reset(token)
                self.service._community_observation_end(observation)
        return bool(observation.get("startup_recorded"))
