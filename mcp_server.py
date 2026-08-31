"""HTTP MCP control plane for automated SparkDeck experiments."""

from __future__ import annotations

import asyncio
import copy
import hmac
import json
import math
import statistics
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit, urlunsplit

import httpx
from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings
from mcp.server.mcpserver import Context, MCPServer
from starlette.requests import Request
from starlette.responses import JSONResponse


ROOT = Path(__file__).resolve().parent
OWNER = "sparkdeck-mcp"
# Keep recognizing deployments created before the SparkDeck rename so users can
# safely stop or remove them through the same ownership guard.
LEGACY_OWNERS = frozenset({"vllm-controller-mcp"})
DEFAULT_CONTROLLER_URL = "http://127.0.0.1:7878"
MAX_INFERENCE_ERROR_BYTES = 500
DEFAULT_PROMPTS = [
    "Explain why speculative decoding can improve language-model serving throughput.",
    "Write a concise checklist for diagnosing a distributed GPU inference launch.",
    "Compare tensor parallelism and pipeline parallelism in two paragraphs.",
]


def _stream_event(line: str) -> dict[str, Any] | None:
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


def _stream_output_text(event: dict[str, Any]) -> str:
    values: list[str] = []
    for choice in event.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta") or {}
        if isinstance(delta, dict):
            for key in ("content", "reasoning_content"):
                value = delta.get(key)
                if isinstance(value, str) and value:
                    values.append(value)
        value = choice.get("text")
        if isinstance(value, str) and value:
            values.append(value)
    return "".join(values)


class ControllerError(RuntimeError):
    """A controller or inference endpoint returned an actionable error."""


class _InferenceHTTPError(RuntimeError):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


async def _bounded_stream_error(response: httpx.Response) -> str:
    """Read enough of a streaming error to diagnose it without unbounded buffering."""
    content = bytearray()
    truncated = False
    async for chunk in response.aiter_bytes():
        remaining = MAX_INFERENCE_ERROR_BYTES - len(content)
        if remaining <= 0:
            truncated = True
            break
        content.extend(chunk[:remaining])
        if len(chunk) > remaining or len(content) == MAX_INFERENCE_ERROR_BYTES:
            truncated = True
            break
    detail = bytes(content).decode("utf-8", errors="replace").strip()
    if truncated:
        detail += "..."
    return detail or response.reason_phrase


class StaticTokenVerifier:
    def __init__(self, expected_token: str):
        self.expected_token = expected_token

    async def verify_token(self, token: str) -> AccessToken | None:
        if not hmac.compare_digest(token, self.expected_token):
            return None
        return AccessToken(
            token=token,
            client_id="sparkdeck-agent",
            scopes=["cluster:control"],
            subject="automation-agent",
        )


class ControllerClient:
    def __init__(
        self,
        controller_url: str = DEFAULT_CONTROLLER_URL,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        inference_transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = 60.0,
    ):
        self.controller_url = controller_url.rstrip("/")
        self._transport = transport
        self._inference_transport = inference_transport
        self.timeout = timeout

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> Any:
        async with httpx.AsyncClient(
            transport=self._transport,
            timeout=timeout or self.timeout,
        ) as client:
            try:
                response = await client.request(
                    method, f"{self.controller_url}{path}", json=json_body
                )
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                try:
                    detail = exc.response.json().get("detail")
                except Exception:
                    detail = exc.response.text
                raise ControllerError(
                    f"controller {method} {path} failed ({exc.response.status_code}): {detail}"
                ) from exc
            except httpx.HTTPError as exc:
                raise ControllerError(f"controller is unavailable: {exc}") from exc
        return response.json()

    async def state(self) -> dict[str, Any]:
        return await self._request("GET", "/api/state")

    async def recipe(self, recipe_id: str) -> dict[str, Any]:
        state = await self.state()
        recipe = next(
            (item for item in state.get("recipes", []) if item.get("id") == recipe_id),
            None,
        )
        if not recipe:
            raise ControllerError(f"recipe not found: {recipe_id}")
        return recipe

    async def deployment(self, deployment_id: str) -> dict[str, Any]:
        state = await self.state()
        deployment = next(
            (
                item
                for item in state.get("deployments", [])
                if (
                    item.get("id") == deployment_id
                    or item.get("sparkdeck_record_id") == deployment_id
                )
            ),
            None,
        )
        if not deployment:
            raise ControllerError(f"deployment not found: {deployment_id}")
        return deployment

    async def _deployment_record_id(
        self, deployment_id: str, *, require_owned: bool,
    ) -> str:
        """Resolve either a Manager ID or stable SparkDeck record ID."""
        state = await self.state()
        deployment = next(
            (
                item for item in state.get("deployments", [])
                if (
                    item.get("id") == deployment_id
                    or item.get("sparkdeck_record_id") == deployment_id
                )
            ),
            None,
        )
        if deployment is not None:
            if (
                require_owned
                and deployment.get("managed_by") not in {OWNER, *LEGACY_OWNERS}
            ):
                raise ControllerError(
                    f"refusing to modify deployment {deployment_id}: "
                    "it was not created by this MCP server"
                )
            return str(deployment.get("sparkdeck_record_id") or deployment_id)
        if require_owned:
            # Distinguish an existing non-MCP record from an invalid ID while
            # preserving the default fail-closed ownership policy.
            detail = await self._request(
                "GET", f"/api/v1/deployments/{quote(deployment_id, safe='')}",
            )
            if detail.get("managed_by") not in {OWNER, *LEGACY_OWNERS}:
                raise ControllerError(
                    f"refusing to modify deployment {deployment_id}: "
                    "it was not created by this MCP server"
                )
        return deployment_id

    async def deployment_configuration(
        self, deployment_id: str,
    ) -> dict[str, Any]:
        record_id = await self._deployment_record_id(
            deployment_id, require_owned=False,
        )
        return await self._request(
            "GET", f"/api/v1/deployments/{quote(record_id, safe='')}",
        )

    async def update_deployment_configuration(
        self,
        deployment_id: str,
        changes: dict[str, Any],
        *,
        require_owned: bool = True,
    ) -> dict[str, Any]:
        if not isinstance(changes, dict):
            raise ControllerError("changes must be an object")
        record_id = await self._deployment_record_id(
            deployment_id, require_owned=require_owned,
        )
        return await self._request(
            "PUT", f"/api/v1/deployments/{quote(record_id, safe='')}/settings",
            json_body=changes,
        )

    async def create_recipe(self, recipe: dict[str, Any]) -> dict[str, Any]:
        return await self._request("POST", "/api/recipes", json_body=recipe)

    async def update_recipe(
        self, recipe_id: str, changes: dict[str, Any]
    ) -> dict[str, Any]:
        return await self._request(
            "PUT", f"/api/recipes/{recipe_id}", json_body=changes
        )

    async def clone_recipe(
        self,
        recipe_id: str,
        name: str,
        overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        recipe = await self.recipe(recipe_id)
        payload = self.variant_payload(recipe, overrides or {})
        for key in ("id", "created_at", "updated_at"):
            payload.pop(key, None)
        payload["name"] = name
        payload["force_new"] = True
        return await self.create_recipe(payload)

    @staticmethod
    def variant_payload(
        base: dict[str, Any], overrides: dict[str, Any]
    ) -> dict[str, Any]:
        if not isinstance(overrides, dict):
            raise ControllerError("overrides must be an object")
        payload = copy.deepcopy(base)
        for key, value in overrides.items():
            if key == "launch_controls" and isinstance(value, dict):
                current = payload.get("launch_controls")
                payload[key] = {**(current if isinstance(current, dict) else {}), **value}
            else:
                payload[key] = copy.deepcopy(value)
        return payload

    async def deploy(
        self,
        payload: dict[str, Any],
        *,
        run_id: str | None = None,
        deployment_name: str | None = None,
    ) -> dict[str, Any]:
        body = copy.deepcopy(payload)
        for key in ("id", "created_at", "updated_at"):
            body.pop(key, None)
        if deployment_name:
            body["deployment_name"] = deployment_name
        body["managed_by"] = OWNER
        body["automation_run_id"] = run_id or uuid.uuid4().hex
        if "deployment_mode" not in body:
            body["deployment_mode"] = "single"
        return await self._request(
            "POST", "/api/containers", json_body=body, timeout=1800
        )

    async def deploy_recipe(
        self,
        recipe_id: str,
        *,
        overrides: dict[str, Any] | None = None,
        run_id: str | None = None,
        deployment_name: str | None = None,
    ) -> dict[str, Any]:
        recipe = await self.recipe(recipe_id)
        payload = self.variant_payload(recipe, overrides or {})
        payload["recipe_id"] = recipe_id
        return await self.deploy(
            payload, run_id=run_id, deployment_name=deployment_name
        )

    async def action(
        self,
        deployment_id: str,
        action: str,
        *,
        require_owned: bool = True,
    ) -> dict[str, Any]:
        if action == "remove":
            # Preserve the legacy Manager removal contract used by A/B cleanup
            # and delete_cluster_deployment. v1 record deletion has a distinct
            # DELETE route and must not be conflated with start/stop actions.
            deployment = await self.deployment(deployment_id)
            if (
                require_owned
                and deployment.get("managed_by") not in {OWNER, *LEGACY_OWNERS}
            ):
                raise ControllerError(
                    f"refusing to remove deployment {deployment_id}: "
                    "it was not created by this MCP server"
                )
            return await self._request(
                "POST",
                f"/api/deployments/{quote(str(deployment['id']), safe='')}/remove",
                timeout=300,
            )
        if action not in {"start", "stop"}:
            raise ControllerError(f"unsupported deployment action: {action}")
        record_id = await self._deployment_record_id(
            deployment_id, require_owned=require_owned,
        )
        return await self._request(
            "POST", f"/api/v1/deployments/{quote(record_id, safe='')}/{action}",
            timeout=300,
        )

    async def storage(self) -> dict[str, Any]:
        """Return the controller's public cluster-wide Storage snapshot."""
        result = await self._request("GET", "/api/v1/storage")
        if not isinstance(result, dict):
            raise ControllerError("controller returned an invalid Storage response")
        return result

    async def storage_weights(self, node_id: str | None = None) -> dict[str, Any]:
        state = await self.storage()
        nodes = state.get("nodes") or []
        if node_id is not None:
            selected = str(node_id).strip()
            if not selected:
                raise ControllerError("node_id must not be empty")
            nodes = [node for node in nodes if str(node.get("id")) == selected]
            if not nodes:
                raise ControllerError(f"storage node not found: {selected}")
        return {"enabled": bool(state.get("enabled")), "nodes": nodes}

    async def storage_transfers(
        self, status: str | None = None,
    ) -> dict[str, Any]:
        jobs = (await self.storage()).get("jobs") or []
        if status is not None:
            selected = str(status).strip().casefold()
            if not selected:
                raise ControllerError("status must not be empty")
            jobs = [
                job for job in jobs
                if str(job.get("status") or "").casefold() == selected
            ]
        return {"jobs": jobs}

    async def storage_transfer(self, job_id: str) -> dict[str, Any]:
        selected = str(job_id).strip()
        if not selected:
            raise ControllerError("job_id must not be empty")
        jobs = (await self.storage()).get("jobs") or []
        job = next((job for job in jobs if str(job.get("id")) == selected), None)
        if not job:
            raise ControllerError(f"storage transfer not found: {selected}")
        return job

    async def pull_storage_weights(
        self,
        model_id: str,
        node_ids: list[str],
        *,
        revision: str | None = None,
        download_node_id: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"model_id": model_id, "node_ids": node_ids}
        if revision is not None:
            body["revision"] = revision
        if download_node_id is not None:
            body["download_node_id"] = download_node_id
        return await self._request(
            "POST", "/api/v1/storage/preparations", json_body=body,
            timeout=1800,
        )

    async def transfer_storage_weights(
        self,
        model_id: str,
        source_node_id: str,
        target_node_ids: list[str],
        *,
        revision: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model_id": model_id,
            "source_node_id": source_node_id,
            "target_node_ids": target_node_ids,
        }
        if revision is not None:
            body["revision"] = revision
        return await self._request(
            "POST", "/api/v1/storage/transfers", json_body=body,
            timeout=300,
        )

    async def delete_storage_weights(
        self, node_id: str, model_id: str,
    ) -> dict[str, Any]:
        node = quote(str(node_id).strip(), safe="")
        model = quote(str(model_id).strip(), safe="")
        if not node or not model:
            raise ControllerError("node_id and model_id must not be empty")
        return await self._request(
            "DELETE", f"/api/v1/storage/nodes/{node}/models/{model}",
            timeout=600,
        )

    async def wait_ready(
        self,
        deployment_id: str,
        *,
        timeout_seconds: float = 1800,
        poll_seconds: float = 3,
        progress: Any = None,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        last_status = None
        while time.monotonic() < deadline:
            deployment = await self.deployment(deployment_id)
            status = deployment.get("status")
            if status != last_status and progress:
                await progress(status or "unknown")
            last_status = status
            if status == "ready":
                return deployment
            if status in {"error", "degraded", "missing"}:
                raise ControllerError(
                    f"deployment {deployment_id} entered {status}: {deployment.get('error') or 'inspect deployment logs'}"
                )
            await asyncio.sleep(max(0.1, poll_seconds))
        raise ControllerError(
            f"deployment {deployment_id} did not become ready within {timeout_seconds:g}s (last status: {last_status})"
        )

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = max(0, math.ceil(percentile * len(ordered)) - 1)
        return ordered[index]

    @staticmethod
    def _interval_union_seconds(intervals: list[tuple[float, float]]) -> float:
        """Return elapsed time covered by one or more possibly overlapping intervals."""
        if not intervals:
            return 0.0
        ordered = sorted(intervals)
        current_start, current_end = ordered[0]
        total = 0.0
        for started_at, ended_at in ordered[1:]:
            if started_at <= current_end:
                current_end = max(current_end, ended_at)
                continue
            total += current_end - current_start
            current_start, current_end = started_at, ended_at
        return total + current_end - current_start

    def _inference_url(self, deployment: dict[str, Any]) -> str:
        port = deployment.get("api_port")
        if not port:
            raise ControllerError("deployment does not expose an API port")
        parsed = urlsplit(self.controller_url)
        hostname = parsed.hostname or "127.0.0.1"
        host = f"[{hostname}]" if ":" in hostname and not hostname.startswith("[") else hostname
        return urlunsplit((parsed.scheme or "http", f"{host}:{int(port)}", "", "", ""))

    async def benchmark(
        self,
        deployment_id: str,
        *,
        prompts: list[str] | None = None,
        repetitions: int = 2,
        concurrency: int = 1,
        max_tokens: int = 256,
        temperature: float = 0.0,
        warmup_requests: int = 1,
        request_timeout_seconds: float = 600,
    ) -> dict[str, Any]:
        if repetitions < 1 or concurrency < 1 or max_tokens < 1 or warmup_requests < 0:
            raise ControllerError("repetitions, concurrency, and max_tokens must be positive")
        if concurrency not in {1, 2, 5, 10}:
            raise ControllerError("concurrency must be one of 1, 2, 5, or 10")
        deployment = await self.deployment(deployment_id)
        if deployment.get("status") != "ready":
            raise ControllerError(
                f"deployment {deployment_id} is {deployment.get('status')}, not ready"
            )
        base_url = self._inference_url(deployment)
        prompt_values = [str(value) for value in (prompts or DEFAULT_PROMPTS) if str(value).strip()]
        if not prompt_values:
            raise ControllerError("at least one non-empty prompt is required")

        timeout = httpx.Timeout(request_timeout_seconds)
        async with httpx.AsyncClient(
            timeout=timeout, transport=self._inference_transport
        ) as client:
            try:
                model_response = await client.get(f"{base_url}/v1/models")
                model_response.raise_for_status()
                model = model_response.json()["data"][0]["id"]
            except (httpx.HTTPError, KeyError, IndexError, TypeError) as exc:
                raise ControllerError(f"could not discover served model at {base_url}: {exc}") from exc

            async def request_once(
                prompt: str, *, include_stream_usage: bool,
                request_max_tokens: int | None = None,
            ) -> dict[str, Any]:
                request_body = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": request_max_tokens or max_tokens,
                    "temperature": temperature,
                    "stream": True,
                }
                if include_stream_usage:
                    request_body["stream_options"] = {"include_usage": True}
                started = time.perf_counter()
                first_token_at: float | None = None
                usage: dict[str, Any] = {}
                text_parts: list[str] = []

                async def consume(body: dict[str, Any]) -> None:
                    nonlocal first_token_at, usage
                    async with client.stream(
                        "POST", f"{base_url}/v1/chat/completions", json=body,
                    ) as response:
                        if not 200 <= response.status_code < 300:
                            detail = await _bounded_stream_error(response)
                            raise _InferenceHTTPError(response.status_code, detail)
                        async for line in response.aiter_lines():
                            event = _stream_event(line)
                            if event is None:
                                continue
                            event_usage = event.get("usage")
                            if isinstance(event_usage, dict):
                                usage = event_usage
                            output = _stream_output_text(event)
                            if output:
                                if first_token_at is None:
                                    first_token_at = time.perf_counter()
                                if sum(len(value) for value in text_parts) < 240:
                                    text_parts.append(output)

                await consume(request_body)
                completed = time.perf_counter()
                first_token_seconds = (
                    first_token_at - started if first_token_at is not None else None
                )
                return {
                    "latency_seconds": completed - started,
                    "time_to_first_token_seconds": first_token_seconds,
                    "_prompt_started_at": started,
                    "_first_token_at": first_token_at,
                    "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                    "completion_tokens": int(usage.get("completion_tokens") or 0),
                    "sample": "".join(text_parts)[:240],
                }

            # Resolve stream_options support once before starting the measured
            # batch. Servers that reject it are then called directly with the
            # cached fallback shape, so failed capability attempts cannot
            # inflate batch wall time or generation/request throughput.
            stream_usage_supported = True
            probe_max_tokens = max_tokens if warmup_requests else 1
            probe_prompt = (
                prompt_values[0] if warmup_requests
                else f"__sparkdeck_capability_probe_{uuid.uuid4().hex}__"
            )
            while not warmup_requests and probe_prompt in prompt_values:
                probe_prompt = f"__sparkdeck_capability_probe_{uuid.uuid4().hex}__"
            try:
                await request_once(
                    probe_prompt, include_stream_usage=True,
                    request_max_tokens=probe_max_tokens,
                )
            except _InferenceHTTPError as exc:
                if exc.status_code not in {400, 422}:
                    raise ControllerError(
                        f"benchmark capability probe failed "
                        f"({exc.status_code}): {exc.detail}"
                    ) from exc
                stream_usage_supported = False
                try:
                    await request_once(
                        probe_prompt, include_stream_usage=False,
                        request_max_tokens=probe_max_tokens,
                    )
                except _InferenceHTTPError as fallback_exc:
                    raise ControllerError(
                        f"benchmark capability fallback failed "
                        f"({fallback_exc.status_code}): {fallback_exc.detail}"
                    ) from fallback_exc
                except httpx.HTTPError as fallback_exc:
                    raise ControllerError(
                        f"benchmark capability fallback failed: {fallback_exc}"
                    ) from fallback_exc
            except httpx.HTTPError as exc:
                raise ControllerError(f"benchmark capability probe failed: {exc}") from exc

            for index in range(1, warmup_requests):
                try:
                    await request_once(
                        prompt_values[index % len(prompt_values)],
                        include_stream_usage=stream_usage_supported,
                    )
                except _InferenceHTTPError as exc:
                    raise ControllerError(
                        f"benchmark warmup failed ({exc.status_code}): {exc.detail}"
                    ) from exc
                except httpx.HTTPError as exc:
                    raise ControllerError(f"benchmark warmup failed: {exc}") from exc

            jobs = [
                prompt
                for _ in range(repetitions)
                for prompt in prompt_values
            ]
            # Every wave must contain the requested concurrency. Pad a partial
            # final wave by cycling supplied prompts; otherwise a result can be
            # labelled C5/C10 even though part of the run measured fewer active
            # requests.
            while len(jobs) % concurrency:
                jobs.append(prompt_values[len(jobs) % len(prompt_values)])
            semaphore = asyncio.Semaphore(concurrency)

            async def limited(prompt: str) -> dict[str, Any]:
                async with semaphore:
                    try:
                        return await request_once(
                            prompt, include_stream_usage=stream_usage_supported,
                        )
                    except _InferenceHTTPError as exc:
                        raise ControllerError(
                            f"inference request failed ({exc.status_code}): {exc.detail}"
                        ) from exc
                    except httpx.HTTPError as exc:
                        raise ControllerError(f"inference request failed: {exc}") from exc

            wall_started = time.perf_counter()
            results = await asyncio.gather(*(limited(prompt) for prompt in jobs))
            wall_seconds = time.perf_counter() - wall_started

        completion_tokens = sum(result["completion_tokens"] for result in results)
        prompt_tokens = sum(result["prompt_tokens"] for result in results)
        latencies = [result["latency_seconds"] for result in results]
        first_token_times = [
            result["time_to_first_token_seconds"] for result in results
            if result["time_to_first_token_seconds"] is not None
        ]
        prompt_intervals = [
            (result["_prompt_started_at"], result["_first_token_at"])
            for result in results
            if result["_first_token_at"] is not None
        ]
        prompt_metric_available = bool(
            prompt_tokens > 0
            and len(first_token_times) == len(results)
            and len(prompt_intervals) == len(results)
            and all(value > 0 for value in first_token_times)
            and all(result["prompt_tokens"] > 0 for result in results)
        )
        prompt_seconds = (
            self._interval_union_seconds(prompt_intervals)
            if prompt_metric_available else None
        )
        prompt_tokens_per_second = (
            prompt_tokens / prompt_seconds
            if prompt_seconds is not None and prompt_seconds > 0 else None
        )
        for sample in results:
            sample.pop("_prompt_started_at", None)
            sample.pop("_first_token_at", None)
        result = {
            "deployment_id": deployment_id,
            "model": model,
            "configuration": {
                "prompts": len(prompt_values),
                "repetitions": repetitions,
                "requests": len(results),
                "concurrency": concurrency,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "warmup_requests": warmup_requests,
            },
            "metrics": {
                "wall_seconds": wall_seconds,
                "completion_tokens": completion_tokens,
                "prompt_tokens": prompt_tokens,
                "prompt_seconds": prompt_seconds,
                "prompt_tokens_per_second": prompt_tokens_per_second,
                "output_tokens_per_second": completion_tokens / wall_seconds if wall_seconds else 0.0,
                "requests_per_second": len(results) / wall_seconds if wall_seconds else 0.0,
                "mean_latency_seconds": statistics.fmean(latencies),
                "p50_latency_seconds": self._percentile(latencies, 0.50),
                "p95_latency_seconds": self._percentile(latencies, 0.95),
                "mean_time_to_first_token_seconds": (
                    statistics.fmean(first_token_times) if prompt_metric_available else None
                ),
                "p50_time_to_first_token_seconds": (
                    self._percentile(first_token_times, 0.50) if prompt_metric_available else None
                ),
                "p95_time_to_first_token_seconds": (
                    self._percentile(first_token_times, 0.95) if prompt_metric_available else None
                ),
            },
            "samples": results,
        }
        if not prompt_metric_available:
            result["recording"] = {
                "status": "not_recorded",
                "reason": "prompt throughput unavailable because a request lacked first-token timing or prompt-token usage",
            }
            return result
        try:
            recorded = await self._request(
                "POST", "/api/v1/benchmark-runs",
                json_body={
                    "deployment_id": deployment_id,
                    "concurrency": concurrency,
                    "request_count": len(results),
                    "prompt_tokens": prompt_tokens,
                    "generation_tokens": completion_tokens,
                    "prompt_seconds": prompt_seconds,
                    "wall_seconds": wall_seconds,
                },
            )
            result["recording"] = {"status": "recorded", "id": recorded.get("id")}
        except ControllerError as exc:
            # Preserve the benchmark result for legacy deployments/controllers.
            result["recording"] = {"status": "not_recorded", "reason": str(exc)}
        return result


def _save_ab_result(result: dict[str, Any]) -> None:
    path = ROOT / "data" / "mcp_ab_results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        values = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        if not isinstance(values, list):
            values = []
    except (OSError, json.JSONDecodeError):
        values = []
    values.append(result)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(values[-100:], indent=2), encoding="utf-8")
    temporary.replace(path)


def _load_ab_results(limit: int = 20) -> list[dict[str, Any]]:
    path = ROOT / "data" / "mcp_ab_results.json"
    if not path.exists():
        return []
    try:
        values = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return values[-max(1, min(limit, 100)):] if isinstance(values, list) else []


def build_server(
    client: ControllerClient,
    *,
    token: str | None = None,
    public_url: str = "http://127.0.0.1:7878/mcp",
) -> MCPServer:
    auth = None
    verifier = None
    if token:
        base_url = public_url.removesuffix("/mcp")
        auth = AuthSettings(
            issuer_url=base_url,
            resource_server_url=public_url,
            required_scopes=["cluster:control"],
        )
        verifier = StaticTokenVerifier(token)

    server = MCPServer(
        name="sparkdeck",
        title="SparkDeck Cluster Automation",
        description=(
            "Manage clustered model storage and create, configure, start, stop, tune, "
            "benchmark, compare, and remove clustered model deployments."
        ),
        instructions=(
            "Prefer recipe IDs and deployment IDs. Run A/B variants sequentially unless the "
            "selected nodes have enough independent GPUs. Destructive tools protect deployments "
            "not created through this MCP server unless allow_unowned is explicitly set. "
            "For vLLM image-specific runtime variables, use environment as an object of "
            "non-secret string NAME/value pairs; never place credentials in environment."
            " Stop a running deployment before changing its saved configuration, then "
            "start it again to apply the updated environment, launch_controls, or extra_args."
            " Storage tools use the controller's Virtual NAS validation and never expose "
            "cache paths, node credentials, or Hugging Face tokens."
        ),
        version="1.2.0",
        auth=auth,
        token_verifier=verifier,
    )

    @server.custom_route("/health", methods=["GET"])
    async def health(_: Request) -> JSONResponse:
        try:
            state = await client.state()
            return JSONResponse({
                "ok": True,
                "controller_url": client.controller_url,
                "nodes": len(state.get("nodes", [])),
            })
        except Exception as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=503)

    @server.tool()
    async def get_cluster_state() -> dict[str, Any]:
        """Return nodes, recipes, deployments, and controller summary state."""
        return await client.state()

    @server.tool()
    async def list_storage_weights(
        node_id: str | None = None,
    ) -> dict[str, Any]:
        """List model weights by storage node, optionally limited to one stable node ID."""
        return await client.storage_weights(node_id)

    @server.tool()
    async def pull_storage_weights(
        model_id: str,
        node_ids: list[str],
        revision: str | None = None,
        download_node_id: str | None = None,
    ) -> dict[str, Any]:
        """Ensure Hugging Face weights are present on selected nodes.

        SparkDeck downloads once and uses Virtual NAS fan-out when possible. Pass
        ``download_node_id`` to choose the seed node; it must also be in
        ``node_ids``. The result includes workflow and queued job IDs for status
        checks with the transfer tools.
        """
        return await client.pull_storage_weights(
            model_id,
            node_ids,
            revision=revision,
            download_node_id=download_node_id,
        )

    @server.tool()
    async def transfer_storage_weights(
        model_id: str,
        source_node_id: str,
        target_node_ids: list[str],
        revision: str | None = None,
    ) -> dict[str, Any]:
        """Queue a validated Virtual NAS copy from one node to other nodes."""
        return await client.transfer_storage_weights(
            model_id,
            source_node_id,
            target_node_ids,
            revision=revision,
        )

    @server.tool()
    async def list_storage_transfers(
        status: str | None = None,
    ) -> dict[str, Any]:
        """List weight downloads and transfers, optionally filtered by exact status."""
        return await client.storage_transfers(status)

    @server.tool()
    async def get_storage_transfer(job_id: str) -> dict[str, Any]:
        """Get current progress, rate, phase, and error details for one Storage job ID."""
        return await client.storage_transfer(job_id)

    @server.tool()
    async def delete_storage_weights(
        node_id: str,
        model_id: str,
        confirm: bool = False,
    ) -> dict[str, Any]:
        """Delete an exact cached model from one node after explicit confirmation.

        Set ``confirm=true``. SparkDeck refuses deletion while the model is serving,
        participating in a transfer, externally managed, or otherwise unsafe to remove.
        """
        if not confirm:
            raise ControllerError(
                "refusing to delete weights without confirm=true"
            )
        return await client.delete_storage_weights(node_id, model_id)

    @server.tool()
    async def list_cluster_recipes() -> list[dict[str, Any]]:
        """List reusable cluster recipes, including their stable recipe IDs."""
        return (await client.state()).get("recipes", [])

    @server.tool()
    async def list_cluster_deployments() -> list[dict[str, Any]]:
        """List deployments with IDs, status, launch controls, ownership, and API ports."""
        return (await client.state()).get("deployments", [])

    @server.tool()
    async def get_cluster_deployment(deployment_id: str) -> dict[str, Any]:
        """Get one Manager deployment by its Manager or stable SparkDeck deployment ID."""
        return await client.deployment(deployment_id)

    @server.tool()
    async def get_cluster_deployment_configuration(
        deployment_id: str,
    ) -> dict[str, Any]:
        """View one deployment's safe editable configuration by Manager or stable ID.

        Returns whether the deployment is editable, why it may not be editable,
        its desired state, sanitized ``extra_args`` runtime flags, non-secret
        ``environment``, structured ``launch_controls``, and runtime-specific
        memory settings. Credentials and controller-local routing are omitted.
        """
        return await client.deployment_configuration(deployment_id)

    @server.tool()
    async def update_cluster_deployment_configuration(
        deployment_id: str,
        changes: dict[str, Any],
        allow_unowned: bool = False,
    ) -> dict[str, Any]:
        """Modify a stopped deployment's saved runtime configuration.

        ``changes`` may contain sanitized runtime flags as an ``extra_args``
        token array, non-secret string ``environment`` values, structured
        ``launch_controls`` (for example ``context_window``,
        ``max_concurrency``, tensor/pipeline parallel size, KV cache dtype,
        thinking mode, or speculative decoding controls), and supported vLLM
        or SGLang memory fields. Stop a running deployment first, update it,
        then start it to apply the new settings. Unknown fields, secrets, and
        unsafe flags are rejected by SparkDeck. Deployments not created by this
        MCP server require ``allow_unowned=true``.
        """
        return await client.update_deployment_configuration(
            deployment_id, changes, require_owned=not allow_unowned,
        )

    @server.tool()
    async def create_cluster_recipe(recipe: dict[str, Any]) -> dict[str, Any]:
        """Create a reusable cluster recipe from structured launch settings.

        Pass engine flags in ``extra_args`` (array of tokens). SGLang also
        accepts ``launch_settings`` with ``context_length``,
        ``max_running_requests``, ``mem_fraction_static``, and
        ``tensor_parallel_size``; vLLM accepts ``max_model_len``,
        ``max_running_requests``, ``gpu_memory_utilization``, and
        ``environment``. Put environment at the recipe top level or inside
        launch_settings. It must be an object of non-secret string NAME/value
        pairs, for example ``{"NCCL_DEBUG": "WARN"}``, and is applied to every
        vLLM rank. Values outside these fields are ignored.
        """
        return await client.create_recipe(recipe)

    @server.tool()
    async def update_cluster_recipe(
        recipe_id: str, changes: dict[str, Any]
    ) -> dict[str, Any]:
        """Modify a recipe. Changes may include launch_controls, extra_args, and vLLM environment."""
        return await client.update_recipe(recipe_id, changes)

    @server.tool()
    async def clone_cluster_recipe(
        recipe_id: str,
        name: str,
        overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Clone a recipe and apply overrides, including a vLLM environment object."""
        return await client.clone_recipe(recipe_id, name, overrides)

    @server.tool()
    async def deploy_cluster_recipe(
        recipe_id: str,
        overrides: dict[str, Any] | None = None,
        deployment_name: str | None = None,
        automation_run_id: str | None = None,
    ) -> dict[str, Any]:
        """Launch a recipe with overrides such as vLLM environment and add MCP ownership metadata."""
        return await client.deploy_recipe(
            recipe_id,
            overrides=overrides,
            run_id=automation_run_id,
            deployment_name=deployment_name,
        )

    @server.tool()
    async def wait_for_cluster_ready(
        deployment_id: str,
        timeout_seconds: float = 1800,
        poll_seconds: float = 3,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Wait until a deployment is ready, fails, or reaches the timeout."""
        async def progress(status: str) -> None:
            if ctx:
                try:
                    await ctx.report_progress(0, 1, f"Deployment {deployment_id}: {status}")
                except (RuntimeError, ValueError):
                    pass

        return await client.wait_ready(
            deployment_id,
            timeout_seconds=timeout_seconds,
            poll_seconds=poll_seconds,
            progress=progress,
        )

    @server.tool()
    async def benchmark_cluster_deployment(
        deployment_id: str,
        prompts: list[str] | None = None,
        repetitions: int = 2,
        concurrency: int = 1,
        max_tokens: int = 256,
        temperature: float = 0.0,
        warmup_requests: int = 1,
        request_timeout_seconds: float = 600,
    ) -> dict[str, Any]:
        """Benchmark a ready deployment and return throughput and latency metrics."""
        return await client.benchmark(
            deployment_id,
            prompts=prompts,
            repetitions=repetitions,
            concurrency=concurrency,
            max_tokens=max_tokens,
            temperature=temperature,
            warmup_requests=warmup_requests,
            request_timeout_seconds=request_timeout_seconds,
        )

    @server.tool()
    async def stop_cluster_deployment(
        deployment_id: str, allow_unowned: bool = False
    ) -> dict[str, Any]:
        """Stop a deployment by ID. Non-MCP deployments require allow_unowned=true."""
        return await client.action(
            deployment_id, "stop", require_owned=not allow_unowned
        )

    @server.tool()
    async def start_cluster_deployment(
        deployment_id: str, allow_unowned: bool = False
    ) -> dict[str, Any]:
        """Start a deployment by Manager or stable ID.

        This applies its currently saved configuration. Non-MCP deployments
        require ``allow_unowned=true``.
        """
        return await client.action(
            deployment_id, "start", require_owned=not allow_unowned
        )

    @server.tool()
    async def delete_cluster_deployment(
        deployment_id: str, allow_unowned: bool = False
    ) -> dict[str, Any]:
        """Remove a deployment by ID. Non-MCP deployments require allow_unowned=true."""
        return await client.action(
            deployment_id, "remove", require_owned=not allow_unowned
        )

    @server.tool()
    async def run_cluster_ab_test(
        recipe_id: str,
        variant_a_overrides: dict[str, Any],
        variant_b_overrides: dict[str, Any],
        prompts: list[str] | None = None,
        repetitions: int = 2,
        concurrency: int = 1,
        max_tokens: int = 256,
        temperature: float = 0.0,
        warmup_requests: int = 1,
        startup_timeout_seconds: float = 1800,
        cleanup: bool = True,
        require_idle_cluster: bool = True,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Deploy and benchmark A/B variants; overrides may include vLLM environment objects."""
        run_id = uuid.uuid4().hex
        state = await client.state()
        if require_idle_cluster:
            occupied = [
                item
                for item in state.get("deployments", [])
                if item.get("status") in {"launching", "starting", "ready"}
            ]
            if occupied:
                ids = ", ".join(str(item.get("id")) for item in occupied)
                raise ControllerError(
                    f"cluster is not idle; active deployment IDs: {ids}. Stop them or set require_idle_cluster=false."
                )
        base_recipe = await client.recipe(recipe_id)
        outcomes: dict[str, Any] = {}

        async def report(step: int, message: str) -> None:
            if ctx:
                try:
                    await ctx.report_progress(step, 8, message)
                except (RuntimeError, ValueError):
                    pass

        async def run_variant(label: str, overrides: dict[str, Any], offset: int) -> None:
            deployment: dict[str, Any] | None = None
            await report(offset, f"Launching variant {label}")
            try:
                payload = client.variant_payload(base_recipe, overrides)
                deployment = await client.deploy(
                    payload,
                    run_id=run_id,
                    deployment_name=f"AB {run_id[:8]} variant {label}",
                )
                deployment_id = deployment["id"]

                async def status_progress(status: str) -> None:
                    await report(offset + 1, f"Variant {label}: {status}")

                await client.wait_ready(
                    deployment_id,
                    timeout_seconds=startup_timeout_seconds,
                    progress=status_progress,
                )
                await report(offset + 2, f"Benchmarking variant {label}")
                benchmark = await client.benchmark(
                    deployment_id,
                    prompts=prompts,
                    repetitions=repetitions,
                    concurrency=concurrency,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    warmup_requests=warmup_requests,
                )
                outcomes[label] = {
                    "overrides": overrides,
                    "deployment_id": deployment_id,
                    "benchmark": benchmark,
                }
            finally:
                if deployment and deployment.get("id"):
                    lifecycle_action = "remove" if cleanup else "stop"
                    await report(
                        offset + 3,
                        f"{lifecycle_action.title()}ing variant {label}",
                    )
                    cleanup_result = await client.action(
                        deployment["id"], lifecycle_action, require_owned=True
                    )
                    if label in outcomes:
                        outcomes[label]["lifecycle"] = cleanup_result
                    if not cleanup_result.get("ok", False):
                        raise ControllerError(
                            f"could not {lifecycle_action} variant {label}: "
                            f"{cleanup_result.get('errors') or cleanup_result}"
                        )

        try:
            await run_variant("A", variant_a_overrides, 0)
            await run_variant("B", variant_b_overrides, 4)
        except Exception as exc:
            failed = {
                "run_id": run_id,
                "recipe_id": recipe_id,
                "status": "failed",
                "error": str(exc),
                "variants": outcomes,
                "created_at": time.time(),
            }
            _save_ab_result(failed)
            raise

        a_rate = outcomes["A"]["benchmark"]["metrics"]["output_tokens_per_second"]
        b_rate = outcomes["B"]["benchmark"]["metrics"]["output_tokens_per_second"]
        delta = b_rate - a_rate
        result = {
            "run_id": run_id,
            "recipe_id": recipe_id,
            "status": "completed",
            "winner": "B" if delta > 0 else ("A" if delta < 0 else "tie"),
            "comparison": {
                "a_output_tokens_per_second": a_rate,
                "b_output_tokens_per_second": b_rate,
                "absolute_delta": delta,
                "percent_change_b_vs_a": (delta / a_rate * 100) if a_rate else None,
            },
            "variants": outcomes,
            "created_at": time.time(),
        }
        _save_ab_result(result)
        return result

    @server.tool()
    async def list_cluster_ab_results(limit: int = 20) -> list[dict[str, Any]]:
        """Return recently persisted A/B outcomes, newest last."""
        return _load_ab_results(limit)

    return server
