"""Manager and service-side tests for the Laya decision runtime.

Laya is a non-autoregressive System 1 decision model rather than a text
generator, so SparkDeck treats it like the other single-engine runtimes: one
complete server per selected node, no tensor or pipeline parallelism. These
tests pin the container contract the node agent must honour and the request
path the controller uses to reach a running decision server.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import httpx

from manager import (
    DEFAULT_LAYA_IMAGE, DEPLOYMENT_LABEL, ENGINE_LABEL, Manager, MODE_LABEL,
    NNODES_LABEL, NODE_LABEL, RANK_LABEL, _LAYA_SERVE_PORT,
)
from sparkdeck.models import Deployment, DeploymentKind, ModelIdentity, RuntimeKind
from sparkdeck.service import SparkDeckService


def _manager() -> Manager:
    """A Manager with only the Docker-facing surface stubbed."""
    manager = Manager.__new__(Manager)
    container = Mock()
    container.reload = Mock()
    manager.client = Mock()
    manager.client.images.get = Mock()
    manager._allocate_port = AsyncMock(return_value=8123)
    manager._build_volumes = Mock(return_value={"/host/cache": {"bind": "/root/.cache/huggingface", "mode": "rw"}})
    manager._run_managed_container = Mock(return_value=container)
    manager._container_summary = Mock(return_value={
        "name": "laya-convaiinnovations-laya-8123", "port": 8123, "status": "running",
    })
    manager.settings = {"hf_cache": "/host/cache", "shm_size": "16g"}
    manager.cluster_member_launches = {}
    return manager


async def _launch(manager, **overrides):
    """Call the Laya container launcher with the full node-agent signature."""
    kwargs = {
        "model": "org/laya", "port": None, "image": None,
        "environment": None, "extra_args": None, "name": None,
        "cluster_member": None, "hf_token": None,
        "sparkdeck_deployment_id": None, "shm_size": None,
    }
    kwargs.update(overrides)
    return await manager._create_laya_container(**kwargs)


class LayaContainerTests(unittest.IsolatedAsyncioTestCase):
    async def test_laya_container_runs_the_decision_server(self):
        manager = _manager()

        result = await _launch(
            manager, model="convaiinnovations/laya",
            sparkdeck_deployment_id="dep-1",
        )

        options = manager._run_managed_container.call_args.args[0]
        self.assertEqual(options["image"], DEFAULT_LAYA_IMAGE)
        self.assertEqual(
            options["command"],
            [
                "--model", "convaiinnovations/laya",
                "--host", "0.0.0.0",
                "--port", str(_LAYA_SERVE_PORT),
            ],
        )
        self.assertEqual(
            options["ports"], {f"{_LAYA_SERVE_PORT}/tcp": 8123},
        )
        self.assertEqual(options["labels"]["io.sparkdeck.runtime"], "laya")
        self.assertEqual(options["labels"]["io.sparkdeck.deployment"], "dep-1")
        # The model reference is forwarded so a controller-local checkpoint
        # directory is mounted, not only the shared Hugging Face cache.
        manager._build_volumes.assert_called_once_with(
            "convaiinnovations/laya", "/host/cache", DEFAULT_LAYA_IMAGE,
        )
        self.assertEqual(result["model_source"], "public_repository")

    async def test_hugging_face_credential_reaches_the_container(self):
        manager = _manager()
        manager._container_hf_environment = Mock(
            return_value={"HF_TOKEN": "secret", "HUGGING_FACE_HUB_TOKEN": "secret"}
        )

        await _launch(manager, hf_token="secret")

        options = manager._run_managed_container.call_args.args[0]
        self.assertEqual(
            options["environment"],
            {"HF_TOKEN": "secret", "HUGGING_FACE_HUB_TOKEN": "secret"},
        )
        manager._container_hf_environment.assert_called_once_with("secret")

    async def test_deployment_environment_is_merged_with_the_credential(self):
        manager = _manager()
        manager._container_hf_environment = Mock(return_value={"HF_TOKEN": "secret"})

        await _launch(
            manager, environment={"HF_HUB_OFFLINE": "1"}, hf_token="secret",
        )

        options = manager._run_managed_container.call_args.args[0]
        self.assertEqual(
            options["environment"], {"HF_HUB_OFFLINE": "1", "HF_TOKEN": "secret"},
        )

    async def test_no_environment_block_without_variables(self):
        manager = _manager()
        manager._container_hf_environment = Mock(return_value={})

        await _launch(manager)

        options = manager._run_managed_container.call_args.args[0]
        self.assertNotIn("environment", options)

    async def test_gpu_request_follows_driver_availability(self):
        manager = _manager()
        with patch("manager._node_has_nvidia_driver", return_value=True):
            await _launch(manager)
        self.assertIn("device_requests", manager._run_managed_container.call_args.args[0])

        manager = _manager()
        with patch("manager._node_has_nvidia_driver", return_value=False):
            await _launch(manager)
        # A CPU-only node must not receive a request Docker would reject.
        self.assertNotIn("device_requests", manager._run_managed_container.call_args.args[0])

    async def test_explicit_cpu_device_skips_the_gpu_request(self):
        manager = _manager()
        with patch("manager._node_has_nvidia_driver", return_value=True):
            await _launch(manager, extra_args=["--device", "cpu"])
        self.assertNotIn("device_requests", manager._run_managed_container.call_args.args[0])

    async def test_explicit_cuda_device_requests_a_gpu_without_a_probe(self):
        manager = _manager()
        with patch("manager._node_has_nvidia_driver", return_value=False) as probe:
            await _launch(manager, extra_args=["--device=cuda:1"])
        self.assertIn("device_requests", manager._run_managed_container.call_args.args[0])
        probe.assert_not_called()

    async def test_extra_args_extend_the_served_configuration(self):
        manager = _manager()

        await _launch(
            manager, port=8200, image="registry.test/laya:1",
            extra_args=["--served-model-name", "laya-decide", "--device", "cpu"],
            name="laya-test", shm_size="4g",
        )

        options = manager._run_managed_container.call_args.args[0]
        self.assertEqual(options["image"], "registry.test/laya:1")
        self.assertEqual(
            options["command"],
            [
                "--model", "org/laya", "--host", "0.0.0.0", "--port",
                str(_LAYA_SERVE_PORT),
                "--served-model-name", "laya-decide", "--device", "cpu",
            ],
        )
        self.assertEqual(options["name"], "laya-test")
        self.assertEqual(options["shm_size"], "4g")

    async def test_cluster_member_labels_are_recorded(self):
        manager = _manager()
        member = {
            "deployment_id": "dep-9", "node_id": "spark-2", "rank": 1,
            "mode": "replicated", "nnodes": 2,
        }

        await _launch(
            manager, name="laya-replica", cluster_member=member,
            sparkdeck_deployment_id="dep-9",
        )

        labels = manager._run_managed_container.call_args.args[0]["labels"]
        self.assertEqual(labels[ENGINE_LABEL], "laya")
        self.assertEqual(labels[DEPLOYMENT_LABEL], "dep-9")
        self.assertEqual(labels[NODE_LABEL], "spark-2")
        self.assertEqual(labels[RANK_LABEL], "1")
        self.assertEqual(labels[MODE_LABEL], "replicated")
        self.assertEqual(labels[NNODES_LABEL], "2")

    async def test_revision_pin_is_accepted(self):
        # SparkDeck appends --revision for a cached bookmark launch, so the
        # decision server must translate it instead of failing the launch.
        manager = _manager()

        await _launch(
            manager, extra_args=["--revision", "a" * 40],
        )

        options = manager._run_managed_container.call_args.args[0]
        self.assertEqual(
            options["command"][-2:], ["--revision", "a" * 40],
        )

    async def test_sharded_laya_is_rejected(self):
        manager = _manager()

        with self.assertRaisesRegex(ValueError, "cannot run sharded"):
            await _launch(
                manager, name="laya-shard",
                cluster_member={
                    "mode": "sharded", "deployment_id": "d",
                    "node_id": "n", "rank": 0,
                },
            )
        manager._run_managed_container.assert_not_called()

    async def test_missing_image_is_pulled_once(self):
        import docker

        manager = _manager()
        manager.client.images.get = Mock(
            side_effect=[docker.errors.ImageNotFound("missing"), Mock()]
        )

        await _launch(manager, image="registry.test/laya:1", name="laya-pull")

        manager.client.images.pull.assert_called_once_with("registry.test/laya:1")

    async def test_launch_failure_is_reported_without_leaking_the_token(self):
        manager = _manager()
        manager.client.images.get = Mock(side_effect=RuntimeError("boom hf_secret_value"))
        manager._redact_hf_secret = Mock(return_value="boom [REDACTED]")

        with self.assertRaisesRegex(RuntimeError, r"boom \[REDACTED\]"):
            await _launch(manager, name="laya-fail")


class LayaPreflightTests(unittest.IsolatedAsyncioTestCase):
    async def test_preflight_rejects_sharded_laya_layouts(self):
        manager = Manager.__new__(Manager)
        manager._reject_hf_cli_credentials = Mock()
        manager.cluster_nodes = Mock()
        manager._validate_runtime_file_mount_nodes = Mock()

        with self.assertRaisesRegex(ValueError, "single and replicated layouts"):
            await manager._preflight_deployment_launch({
                "model": "convaiinnovations/laya",
                "engine": "laya",
                "deployment_mode": "sharded",
                "node_ids": ["local", "spark-2"],
            })

    async def test_preflight_accepts_a_single_node_laya_deployment(self):
        manager = Manager.__new__(Manager)
        manager._reject_hf_cli_credentials = Mock()
        manager._validate_runtime_file_mount_nodes = Mock()

        async def cluster_nodes():
            return [{
                "id": "spark-2", "name": "Spark 2", "online": True,
                "docker_ready": True,
            }]

        manager.cluster_nodes = cluster_nodes

        plan = await manager._preflight_deployment_launch({
            "model": "convaiinnovations/laya",
            "engine": "laya",
            "deployment_mode": "single",
            "node_ids": ["spark-2"],
        })

        self.assertEqual(plan["engine"], "laya")
        self.assertEqual(plan["mode"], "single")
        self.assertEqual(plan["node_ids"], ["spark-2"])


class FakeProxyManager:
    """The slice of Manager the controller's managed-proxy path needs."""

    def __init__(self, transport):
        self.http = httpx.AsyncClient(transport=transport)
        self.deployments = []
        self.list_containers = AsyncMock(return_value=[])
        self._prompt_waiting_requests = {}
        self.proxy_cluster_inference = AsyncMock()

    def source_ip_routing_rule(self, caller_ip, requested_model):
        return None

    @staticmethod
    async def _await_or_cancel(coro, cancel):
        return await coro


class LayaProxyTests(unittest.IsolatedAsyncioTestCase):
    """The controller's /v1 path must reach a managed decision server cleanly."""

    async def test_managed_laya_uses_manager_member_routing(self):
        """A managed Laya record must be routed by Manager, not by its stored URL.

        A managed record's stored endpoint is the controller's own port
        mapping, so proxying it directly would send a deployment placed on a
        remote node to the controller and fail. It must go through the same
        member-aware path vLLM and SGLANG use, which is what balances replicas
        and fails over.
        """
        with tempfile.TemporaryDirectory() as directory:
            manager = FakeProxyManager(httpx.MockTransport(
                lambda request: httpx.Response(200, json={})
            ))
            manager.proxy_cluster_inference.return_value = {
                "id": "chatcmpl-1", "object": "chat.completion", "created": 0,
                "model": "convaiinnovations/laya",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "{\"answers\":{}}"},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 12, "completion_tokens": 0, "total_tokens": 12},
            }
            service = SparkDeckService(manager, Path(directory))
            service.store.add_deployment(Deployment(
                id="record-laya", alias="laya-decide",
                runtime=RuntimeKind.LAYA, kind=DeploymentKind.MANAGED,
                model=ModelIdentity("convaiinnovations/laya"),
                container_name="laya-laya-8123", status="running",
                settings={"manager_deployment_id": "cluster-laya"},
            ))
            service.store.update_managed_routing(
                "record-laya", {"manager_deployment_id": "cluster-laya"},
                "laya-laya-8123", "http://127.0.0.1:8123",
            )

            try:
                response = await service.proxy(
                    {
                        "model": "laya-decide",
                        "state": {"body": "please refund"},
                        "questions": {"urgent": {"type": "noul"}},
                    },
                    "chat/completions",
                )

                manager.proxy_cluster_inference.assert_awaited_once()
                args = manager.proxy_cluster_inference.await_args.args
                self.assertEqual(args[0], "cluster-laya")
                self.assertEqual(args[1], "convaiinnovations/laya")
                # The decision payload survives the hop untouched.
                self.assertEqual(args[2]["state"], {"body": "please refund"})
                self.assertEqual(args[2]["questions"], {"urgent": {"type": "noul"}})
                self.assertEqual(args[2]["model"], "convaiinnovations/laya")
                # Callers keep the alias they selected; the decision server
                # keeps the repository it was loaded with.
                self.assertEqual(response["model"], "laya-decide")
                # A decision model reports prompt tokens and no output.
                self.assertEqual(response["usage"]["completion_tokens"], 0)
            finally:
                await manager.http.aclose()
                await service.close()

    async def test_unmanaged_laya_still_reaches_its_own_endpoint(self):
        """A Laya server registered by URL is proxied directly, as before."""
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            body = json.loads(request.content)
            assert body["state"] == {"body": "please refund"}
            assert body["questions"] == {"urgent": {"type": "noul"}}
            assert body["model"] == "convaiinnovations/laya"
            return httpx.Response(200, json={
                "id": "chatcmpl-1", "object": "chat.completion", "created": 0,
                "model": "convaiinnovations/laya",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "{\"answers\":{}}"},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 12, "completion_tokens": 0, "total_tokens": 12},
            })

        # The service store keeps a SQLite handle open, so the temporary
        # directory must outlive the closed service (notably on Windows).
        with tempfile.TemporaryDirectory() as directory:
            manager = FakeProxyManager(httpx.MockTransport(handler))
            service = SparkDeckService(manager, Path(directory))
            service.store.add_deployment(Deployment(
                id="record-laya", alias="laya-decide",
                runtime=RuntimeKind.LAYA, kind=DeploymentKind.EXTERNAL,
                model=ModelIdentity("convaiinnovations/laya"),
                status="registered", settings={},
            ))
            service.store.update_managed_routing(
                "record-laya", {}, None, "http://127.0.0.1:8123",
            )

            try:
                response = await service.proxy(
                    {
                        "model": "laya-decide",
                        "state": {"body": "please refund"},
                        "questions": {"urgent": {"type": "noul"}},
                    },
                    "chat/completions",
                )

                # Reconciliation may also probe the endpoint, so assert on the
                # decision request itself rather than a total call count.
                decision_requests = [
                    request for request in captured
                    if request.url.path == "/v1/chat/completions"
                ]
                self.assertEqual(len(decision_requests), 1)
                self.assertEqual(
                    str(decision_requests[0].url),
                    "http://127.0.0.1:8123/v1/chat/completions",
                )
                self.assertEqual(response["model"], "laya-decide")
                self.assertEqual(response["usage"]["completion_tokens"], 0)
            finally:
                await manager.http.aclose()
                await service.close()


if __name__ == "__main__":
    unittest.main()
