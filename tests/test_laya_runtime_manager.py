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
from unittest.mock import AsyncMock, Mock

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


class LayaContainerTests(unittest.IsolatedAsyncioTestCase):
    async def test_laya_container_runs_the_decision_server(self):
        manager = _manager()

        result = await manager._create_laya_container(
            model="convaiinnovations/laya", port=None, image=None,
            extra_args=None, name=None, cluster_member=None,
            sparkdeck_deployment_id="dep-1", shm_size=None,
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
        # Weights come from the node's shared Hugging Face cache.
        manager._build_volumes.assert_called_once_with(
            "", "/host/cache", DEFAULT_LAYA_IMAGE,
        )
        self.assertEqual(result["model_source"], "public_repository")

    async def test_extra_args_extend_the_served_configuration(self):
        manager = _manager()

        await manager._create_laya_container(
            model="org/laya", port=8200, image="registry.test/laya:1",
            extra_args=["--served-model-name", "laya-decide", "--device", "cpu"],
            name="laya-test", cluster_member=None,
            sparkdeck_deployment_id=None, shm_size="4g",
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

        await manager._create_laya_container(
            model="org/laya", port=None, image=None, extra_args=None,
            name="laya-replica", cluster_member=member,
            sparkdeck_deployment_id="dep-9", shm_size=None,
        )

        labels = manager._run_managed_container.call_args.args[0]["labels"]
        self.assertEqual(labels[ENGINE_LABEL], "laya")
        self.assertEqual(labels[DEPLOYMENT_LABEL], "dep-9")
        self.assertEqual(labels[NODE_LABEL], "spark-2")
        self.assertEqual(labels[RANK_LABEL], "1")
        self.assertEqual(labels[MODE_LABEL], "replicated")
        self.assertEqual(labels[NNODES_LABEL], "2")

    async def test_sharded_laya_is_rejected(self):
        manager = _manager()

        with self.assertRaisesRegex(ValueError, "cannot run sharded"):
            await manager._create_laya_container(
                model="org/laya", port=None, image=None, extra_args=None,
                name="laya-shard",
                cluster_member={"mode": "sharded", "deployment_id": "d", "node_id": "n", "rank": 0},
                sparkdeck_deployment_id=None, shm_size=None,
            )
        manager._run_managed_container.assert_not_called()

    async def test_missing_image_is_pulled_once(self):
        import docker

        manager = _manager()
        manager.client.images.get = Mock(
            side_effect=[docker.errors.ImageNotFound("missing"), Mock()]
        )

        await manager._create_laya_container(
            model="org/laya", port=None, image="registry.test/laya:1",
            extra_args=None, name="laya-pull", cluster_member=None,
            sparkdeck_deployment_id=None, shm_size=None,
        )

        manager.client.images.pull.assert_called_once_with("registry.test/laya:1")

    async def test_launch_failure_is_reported_without_leaking_the_token(self):
        manager = _manager()
        manager.client.images.get = Mock(side_effect=RuntimeError("boom hf_secret_value"))
        manager._redact_hf_secret = Mock(return_value="boom [REDACTED]")

        with self.assertRaisesRegex(RuntimeError, r"boom \[REDACTED\]"):
            await manager._create_laya_container(
                model="org/laya", port=None, image=None, extra_args=None,
                name="laya-fail", cluster_member=None,
                sparkdeck_deployment_id=None, shm_size=None,
            )


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
    """The slice of Manager the controller's registered-proxy path needs."""

    def __init__(self, transport):
        self.http = httpx.AsyncClient(transport=transport)
        self.deployments = []
        self.list_containers = AsyncMock(return_value=[])
        self._prompt_waiting_requests = {}

    def source_ip_routing_rule(self, caller_ip, requested_model):
        return None

    @staticmethod
    async def _await_or_cancel(coro, cancel):
        return await coro


class LayaProxyTests(unittest.IsolatedAsyncioTestCase):
    """The controller's /v1 path must reach a managed decision server cleanly."""

    async def test_decision_request_reaches_the_managed_laya_server(self):
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            body = json.loads(request.content)
            # The controller forwards the decision payload untouched and swaps
            # only the model field for the repository the server was loaded with.
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

                self.assertEqual(len(captured), 1)
                self.assertEqual(
                    str(captured[0].url),
                    "http://127.0.0.1:8123/v1/chat/completions",
                )
                # Callers keep the alias they selected; the decision server
                # keeps the repository it was loaded with.
                self.assertEqual(response["model"], "laya-decide")
                # A decision model reports prompt tokens and no output.
                self.assertEqual(response["usage"]["completion_tokens"], 0)
            finally:
                await manager.http.aclose()
                await service.close()


if __name__ == "__main__":
    unittest.main()
