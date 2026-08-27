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
from urllib.parse import urlsplit, urlunsplit

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
DEFAULT_PROMPTS = [
    "Explain why speculative decoding can improve language-model serving throughput.",
    "Write a concise checklist for diagnosing a distributed GPU inference launch.",
    "Compare tensor parallelism and pipeline parallelism in two paragraphs.",
]


class ControllerError(RuntimeError):
    """A controller or inference endpoint returned an actionable error."""


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
                if item.get("id") == deployment_id
            ),
            None,
        )
        if not deployment:
            raise ControllerError(f"deployment not found: {deployment_id}")
        return deployment

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
        if require_owned:
            deployment = await self.deployment(deployment_id)
            if deployment.get("managed_by") not in {OWNER, *LEGACY_OWNERS}:
                raise ControllerError(
                    f"refusing to {action} deployment {deployment_id}: it was not created by this MCP server"
                )
        return await self._request(
            "POST", f"/api/deployments/{deployment_id}/{action}", timeout=300
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

            async def request_once(prompt: str) -> dict[str, Any]:
                started = time.perf_counter()
                response = await client.post(
                    f"{base_url}/v1/chat/completions",
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "stream": False,
                    },
                )
                response.raise_for_status()
                elapsed = time.perf_counter() - started
                body = response.json()
                usage = body.get("usage") or {}
                text = ((body.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
                return {
                    "latency_seconds": elapsed,
                    "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                    "completion_tokens": int(usage.get("completion_tokens") or 0),
                    "sample": text[:240],
                }

            for index in range(warmup_requests):
                try:
                    await request_once(prompt_values[index % len(prompt_values)])
                except httpx.HTTPError as exc:
                    raise ControllerError(f"benchmark warmup failed: {exc}") from exc

            jobs = [
                prompt
                for _ in range(repetitions)
                for prompt in prompt_values
            ]
            # A C10 data point must actually have ten requests available to run
            # together. Fill short prompt sets by cycling the supplied prompts;
            # otherwise the result would only be labelled with the requested
            # concurrency instead of measuring it.
            while len(jobs) < concurrency:
                jobs.append(prompt_values[len(jobs) % len(prompt_values)])
            semaphore = asyncio.Semaphore(concurrency)

            async def limited(prompt: str) -> dict[str, Any]:
                async with semaphore:
                    try:
                        return await request_once(prompt)
                    except httpx.HTTPStatusError as exc:
                        raise ControllerError(
                            f"inference request failed ({exc.response.status_code}): {exc.response.text[:500]}"
                        ) from exc
                    except httpx.HTTPError as exc:
                        raise ControllerError(f"inference request failed: {exc}") from exc

            wall_started = time.perf_counter()
            results = await asyncio.gather(*(limited(prompt) for prompt in jobs))
            wall_seconds = time.perf_counter() - wall_started

        completion_tokens = sum(result["completion_tokens"] for result in results)
        prompt_tokens = sum(result["prompt_tokens"] for result in results)
        latencies = [result["latency_seconds"] for result in results]
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
                "output_tokens_per_second": completion_tokens / wall_seconds if wall_seconds else 0.0,
                "requests_per_second": len(results) / wall_seconds if wall_seconds else 0.0,
                "mean_latency_seconds": statistics.fmean(latencies),
                "p50_latency_seconds": self._percentile(latencies, 0.50),
                "p95_latency_seconds": self._percentile(latencies, 0.95),
            },
            "samples": results,
        }
        try:
            recorded = await self._request(
                "POST", "/api/v1/benchmark-runs",
                json_body={
                    "deployment_id": deployment_id,
                    "concurrency": concurrency,
                    "request_count": len(results),
                    "prompt_tokens": prompt_tokens,
                    "generation_tokens": completion_tokens,
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
        description="Create, tune, benchmark, compare, stop, and remove clustered model deployments.",
        instructions=(
            "Prefer recipe IDs and deployment IDs. Run A/B variants sequentially unless the "
            "selected nodes have enough independent GPUs. Destructive tools protect deployments "
            "not created through this MCP server unless allow_unowned is explicitly set."
        ),
        version="1.0.0",
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
    async def list_cluster_recipes() -> list[dict[str, Any]]:
        """List reusable cluster recipes, including their stable recipe IDs."""
        return (await client.state()).get("recipes", [])

    @server.tool()
    async def list_cluster_deployments() -> list[dict[str, Any]]:
        """List deployments with IDs, status, launch controls, ownership, and API ports."""
        return (await client.state()).get("deployments", [])

    @server.tool()
    async def get_cluster_deployment(deployment_id: str) -> dict[str, Any]:
        """Get one deployment by its stable deployment ID."""
        return await client.deployment(deployment_id)

    @server.tool()
    async def create_cluster_recipe(recipe: dict[str, Any]) -> dict[str, Any]:
        """Create a reusable cluster recipe from structured launch settings."""
        return await client.create_recipe(recipe)

    @server.tool()
    async def update_cluster_recipe(
        recipe_id: str, changes: dict[str, Any]
    ) -> dict[str, Any]:
        """Modify an existing recipe. Changes may include launch_controls and extra_args."""
        return await client.update_recipe(recipe_id, changes)

    @server.tool()
    async def clone_cluster_recipe(
        recipe_id: str,
        name: str,
        overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Clone a recipe under a new name and apply launch-setting overrides."""
        return await client.clone_recipe(recipe_id, name, overrides)

    @server.tool()
    async def deploy_cluster_recipe(
        recipe_id: str,
        overrides: dict[str, Any] | None = None,
        deployment_name: str | None = None,
        automation_run_id: str | None = None,
    ) -> dict[str, Any]:
        """Launch a recipe with optional non-destructive overrides and MCP ownership metadata."""
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
        """Sequentially deploy and benchmark A/B recipe variants, then compare throughput."""
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
