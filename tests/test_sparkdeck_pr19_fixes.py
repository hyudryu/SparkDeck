import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import httpx

from sparkdeck.models import Deployment, DeploymentKind, ModelIdentity, RuntimeKind
from sparkdeck.runtimes import LlamaCppAdapter, SglangAdapter, launch_managed_container
from sparkdeck.service import SparkDeckService, _deployment_launch_progress


# server.py constructs its process-wide Manager at import time. The HTTP route
# contract tests do not need a live Docker daemon.
with patch("docker.from_env", return_value=Mock()):
    import server


class FakeManager:
    def __init__(self):
        self.http = httpx.AsyncClient()
        self.list_containers = AsyncMock(return_value=[])
        self.start_container = AsyncMock(return_value={"status": "running"})
        self.stop_container = AsyncMock(return_value={"status": "exited"})
        self.remove_container = AsyncMock(return_value={"ok": True})


class DeploymentLifecycleFixTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.manager = FakeManager()
        self.service = SparkDeckService(self.manager, Path(self.temp.name))

    async def asyncTearDown(self):
        await self.manager.http.aclose()
        await self.service.close()
        self.temp.cleanup()

    async def test_registered_managed_deployment_is_missing_without_docker(self):
        self.service.store.add_deployment(Deployment(
            id="dep-1", alias="model", runtime=RuntimeKind.VLLM,
            kind=DeploymentKind.MANAGED, model=ModelIdentity("org/model"),
            container_name="sparkdeck-model",
        ))
        self.manager.list_containers.side_effect = RuntimeError("daemon offline")

        result = await self.service.deployments()

        self.assertEqual(result[0]["status"], "missing")
        self.assertEqual(result[0]["last_error"], "Docker is unavailable")

    def test_cluster_progress_uses_least_advanced_active_member(self):
        deployment = {
            "status": "starting",
            "members": [
                {
                    "rank": 0,
                    "phase": {
                        "phase": "pulling_image", "message": "Pulling image",
                    },
                },
                {
                    "rank": 1,
                    "phase": {"phase": "queued", "message": "Waiting"},
                },
            ],
        }
        self.assertEqual(
            _deployment_launch_progress(deployment)["launch_phase"], "queued",
        )

        deployment["members"] = [
            {
                "rank": 0,
                "phase": {
                    "phase": "unreachable", "message": "Node unreachable",
                },
            },
            {
                "rank": 1,
                "phase": {
                    "phase": "pulling_image", "message": "Pulling image",
                },
            },
        ]
        self.assertEqual(
            _deployment_launch_progress(deployment)["launch_phase"],
            "pulling_image",
        )

        deployment["members"][1]["phase"] = {
            "phase": "ready", "message": "Ready",
        }
        self.assertEqual(
            _deployment_launch_progress(deployment)["launch_phase"],
            "error",
        )
        self.assertEqual(
            _deployment_launch_progress(deployment)["launch_message"],
            "Node unreachable",
        )

    async def test_discovered_legacy_managed_container_is_actionable(self):
        self.manager.list_containers.return_value = [{
            "name": "legacy-model", "model": "org/model", "engine": "vllm",
            "managed": True, "status": "running", "port": 8000,
        }]

        discovered = (await self.service.deployments())[0]
        stopped = await self.service.deployment_action(discovered["id"], "stop")
        removed = await self.service.delete_deployment(discovered["id"])

        self.assertEqual(discovered["id"], "container:legacy-model")
        self.assertEqual(stopped["status"], "stopped")
        self.assertEqual(removed, {"ok": True, "id": "container:legacy-model"})
        self.manager.stop_container.assert_awaited_once_with(
            "legacy-model", explicit=True,
        )
        self.manager.remove_container.assert_awaited_once_with("legacy-model")

    async def test_cluster_member_container_actions_target_whole_cluster(self):
        self.manager.deployments = [{
            "id": "cluster-1", "status": "ready",
            "members": [
                {"node_id": "local", "container_name": "cluster-1-r0-model", "rank": 0},
                {"node_id": "node-2", "container_name": "cluster-1-r1-model", "rank": 1},
            ],
        }]
        self.manager.deployment_action = AsyncMock(return_value={"ok": True, "errors": []})
        self.manager.list_containers.return_value = [{
            "name": "cluster-1-r0-model", "model": "org/model", "engine": "vllm",
            "managed": True, "status": "running", "port": 8000,
        }]

        cards = await self.service.deployments()
        stopped = await self.service.deployment_action(cards[0]["id"], "stop")

        self.assertEqual(cards[0]["id"], "container:cluster-1-r0-model")
        # The card reports the whole cluster's state, not just the local rank.
        self.assertEqual(cards[0]["status"], "running")
        self.manager.deployment_action.assert_awaited_once_with("cluster-1", "stop")
        self.manager.stop_container.assert_not_awaited()
        self.assertEqual(stopped["status"], "stopped")

    async def test_stopped_cluster_member_card_reports_cluster_status(self):
        self.manager.deployments = [{
            "id": "cluster-1", "status": "stopped",
            "members": [{"node_id": "local", "container_name": "cluster-1-r0-model", "rank": 0}],
        }]
        self.manager.list_containers.return_value = [{
            "name": "cluster-1-r0-model", "model": "org/model", "engine": "vllm",
            "managed": True, "status": "exited", "port": 8000,
        }]

        cards = await self.service.deployments()

        self.assertEqual(cards[0]["status"], "stopped")

    async def test_cluster_member_remove_targets_whole_cluster(self):
        self.manager.deployments = [{
            "id": "cluster-1", "status": "running",
            "members": [
                {"node_id": "local", "container_name": "cluster-1-r0-model", "rank": 0},
                {"node_id": "node-2", "container_name": "cluster-1-r1-model", "rank": 1},
            ],
        }]
        self.manager.deployment_action = AsyncMock(return_value={"ok": True, "errors": []})
        self.manager.list_containers.return_value = [{
            "name": "cluster-1-r0-model", "model": "org/model", "engine": "vllm",
            "managed": True, "status": "running", "port": 8000,
        }]

        removed = await self.service.delete_deployment("container:cluster-1-r0-model")

        self.assertEqual(removed, {"ok": True, "id": "container:cluster-1-r0-model"})
        self.manager.deployment_action.assert_awaited_once_with("cluster-1", "remove")
        self.manager.remove_container.assert_not_awaited()

    async def test_deployment_logs_include_every_cluster_rank(self):
        self.manager.deployments = [{
            "id": "cluster-1", "status": "running",
            "members": [
                {
                    "node_id": "node-2", "node_name": "Worker", "rank": 1,
                    "container_name": "cluster-1-r1-model", "status": "running",
                    "agent_token": "must-not-leak", "phase": {"private": True},
                },
                {
                    "node_id": "local", "node_name": "Controller", "rank": 0,
                    "container_name": "cluster-1-r0-model", "status": "running",
                    "container_id": "must-not-leak",
                },
            ],
        }]
        tails = []

        async def member_action(member, action, log_tail=300):
            tails.append((member["container_name"], action, log_tail))
            return {"logs": f"logs from {member['container_name']}"}

        self.manager._member_action = member_action
        self.manager.list_containers.return_value = [{
            "name": "cluster-1-r0-model", "model": "org/model", "engine": "vllm",
            "managed": True, "status": "running", "port": 8000,
        }]

        result = await self.service.deployment_logs("container:cluster-1-r0-model", 150)

        self.assertIn("rank 0 · node local", result["logs"])
        self.assertIn("logs from cluster-1-r0-model", result["logs"])
        self.assertIn("rank 1 · node node-2", result["logs"])
        self.assertIn("logs from cluster-1-r1-model", result["logs"])
        self.assertEqual(result["members"], [
            {
                "node_id": "local", "node_name": "Controller", "rank": 0,
                "container_name": "cluster-1-r0-model", "status": "running",
                "logs": "logs from cluster-1-r0-model",
            },
            {
                "node_id": "node-2", "node_name": "Worker", "rank": 1,
                "container_name": "cluster-1-r1-model", "status": "running",
                "logs": "logs from cluster-1-r1-model",
            },
        ])
        self.assertNotIn("agent_token", result["members"][1])
        self.assertNotIn("phase", result["members"][1])
        self.assertNotIn("container_id", result["members"][0])
        self.assertEqual(sorted(tails), [
            ("cluster-1-r0-model", "logs", 150),
            ("cluster-1-r1-model", "logs", 150),
        ])

    async def test_deployment_logs_fall_back_to_coordinator_status(self):
        self.manager.deployments = [{
            "id": "cluster-1", "status": "starting",
            "members": [
                {"node_id": "local", "container_name": "cluster-1-r0-model", "rank": 0},
                {"node_id": "node-2", "container_name": "cluster-1-r1-model", "rank": 1,
                 "status": "queued", "phase": {"message": "pulling runtime image"}},
            ],
        }]

        async def member_action(member, action, log_tail=300):
            if member["node_id"] == "local":
                return {"logs": "docker logs here"}
            raise RuntimeError("agent returned 404")

        self.manager._member_action = member_action
        self.manager.list_containers.return_value = [{
            "name": "cluster-1-r0-model", "model": "org/model", "engine": "vllm",
            "managed": True, "status": "running", "port": 8000,
        }]

        result = await self.service.deployment_logs("container:cluster-1-r0-model", 100)

        self.assertIn("docker logs here", result["logs"])
        # A rank without logs yet reports the coordinator's launch status
        # instead of a bare transport error.
        self.assertIn("=== Coordinator launch status ===", result["logs"])
        self.assertIn("pulling runtime image", result["logs"])
        self.assertIn("Agent log request: agent returned 404", result["logs"])
        self.assertNotIn("logs unavailable", result["logs"])
        self.assertEqual(result["members"][0]["logs"], "docker logs here")
        self.assertNotIn("error", result["members"][0])
        self.assertEqual(
            result["members"][1]["logs"],
            "=== Coordinator launch status ===\npulling runtime image"
            "\n\nAgent log request: agent returned 404",
        )
        self.assertEqual(result["members"][1]["error"], "agent returned 404")
        self.assertEqual(
            set(result["members"][1]),
            {"node_id", "rank", "container_name", "status", "logs", "error"},
        )

    async def test_deployment_logs_read_single_container(self):
        self.manager.list_containers.return_value = [{
            "name": "legacy-model", "model": "org/model", "engine": "vllm",
            "managed": True, "status": "running", "port": 8000,
        }]
        self.manager.get_cluster_member_logs = AsyncMock(return_value="single log line")

        result = await self.service.deployment_logs("container:legacy-model")

        self.assertEqual(result, {"logs": "single log line"})
        self.manager.get_cluster_member_logs.assert_awaited_once_with("legacy-model", 300)

    async def test_start_forwards_node_selection_to_manager(self):
        self.service.store.add_deployment(Deployment(
            id="dep-1", alias="model", runtime=RuntimeKind.VLLM,
            kind=DeploymentKind.MANAGED, model=ModelIdentity("org/model"),
            settings={"manager_deployment_id": "cluster-1"},
        ))
        self.manager.deployment_action = AsyncMock(return_value={"ok": True, "errors": []})
        self.manager._resolve_local_path = Mock(return_value=None)
        self.manager.model_cache_inventory = AsyncMock(return_value=[
            {"id": "a", "models": [{"model_id": "org/model", "revisions": ["main"], "partial": False}]},
            {"id": "b", "models": []},
        ])

        await self.service.deployment_action("dep-1", "start", node_ids=["a"])

        self.manager.deployment_action.assert_awaited_once_with("cluster-1", "start", ["a"])

    async def test_start_rejects_nodes_without_cached_weights(self):
        self.service.store.add_deployment(Deployment(
            id="dep-1", alias="model", runtime=RuntimeKind.VLLM,
            kind=DeploymentKind.MANAGED, model=ModelIdentity("org/model"),
            settings={"manager_deployment_id": "cluster-1"},
        ))
        self.manager.deployment_action = AsyncMock(return_value={"ok": True, "errors": []})
        self.manager._resolve_local_path = Mock(return_value=None)
        self.manager.model_cache_inventory = AsyncMock(return_value=[
            {"id": "a", "models": [{"model_id": "org/model", "revisions": ["main"], "partial": False}]},
            {"id": "b", "models": []},
        ])

        with self.assertRaisesRegex(ValueError, "not available on selected node"):
            await self.service.deployment_action("dep-1", "start", node_ids=["a", "b"])

        self.manager.deployment_action.assert_not_awaited()

    async def test_start_enforces_saved_count_and_preserves_sharded_coordinator(self):
        self.service.store.add_deployment(Deployment(
            id="dep-sharded", alias="sharded", runtime=RuntimeKind.VLLM,
            kind=DeploymentKind.MANAGED, model=ModelIdentity("org/model"),
            settings={"manager_deployment_id": "cluster-1"},
        ))
        self.manager.deployments = [{
            "id": "cluster-1", "status": "stopped", "engine": "vllm",
            "launch_settings": {
                "deployment_mode": "sharded",
                "node_ids": ["local", "worker-1"],
            },
        }]
        self.manager.recipe_deployment_contract = lambda _recipe: {
            "deployment_mode": "sharded", "required_node_count": 2,
        }
        self.manager.deployment_action = AsyncMock(return_value={"ok": True, "errors": []})
        self.manager._resolve_local_path = Mock(return_value=None)
        self.manager.model_cache_inventory = AsyncMock(return_value=[
            {"id": node_id, "models": [{
                "model_id": "org/model", "revisions": ["main"], "partial": False,
            }]}
            for node_id in ("local", "worker-1", "worker-2")
        ])

        with self.assertRaisesRegex(ValueError, "requires exactly 2"):
            await self.service.deployment_action(
                "dep-sharded", "start",
                node_ids=["local", "worker-1", "worker-2"],
            )
        self.manager.deployment_action.assert_not_awaited()

        await self.service.deployment_action(
            "dep-sharded", "start", node_ids=["worker-1", "local"],
        )

        self.manager.deployment_action.assert_awaited_once_with(
            "cluster-1", "start", ["worker-1", "local"],
        )

    async def test_start_rejects_node_selection_for_standalone_container(self):
        self.manager.list_containers.return_value = [{
            "name": "legacy-model", "model": "org/model", "engine": "vllm",
            "managed": True, "status": "exited", "port": 8000,
        }]

        with self.assertRaisesRegex(ValueError, "node selection is only available for cluster deployments"):
            await self.service.deployment_action("container:legacy-model", "start", node_ids=["remote-1"])
        self.manager.start_container.assert_not_awaited()

    async def test_cluster_member_start_with_selection_dispatches_to_owner(self):
        self.manager.deployments = [{
            "id": "cluster-1", "status": "stopped", "engine": "vllm",
            "members": [
                {"node_id": "local", "container_name": "cluster-1-r0-model", "rank": 0},
                {"node_id": "node-2", "container_name": "cluster-1-r1-model", "rank": 1},
            ],
            "launch_settings": {"engine": "vllm", "deployment_mode": "sharded"},
        }]
        self.manager.deployment_action = AsyncMock(return_value={"ok": True, "errors": []})
        self.manager._resolve_local_path = Mock(return_value=None)
        self.manager.model_cache_inventory = AsyncMock(return_value=[
            {"id": "local", "models": [{"model_id": "org/model", "revisions": ["main"], "partial": False}]},
            {"id": "node-2", "models": [{"model_id": "org/model", "revisions": ["main"], "partial": False}]},
        ])
        self.manager.list_containers.return_value = [{
            "name": "cluster-1-r0-model", "model": "org/model", "engine": "vllm",
            "managed": True, "status": "exited", "port": 8000,
        }]

        started = await self.service.deployment_action(
            "container:cluster-1-r0-model", "start", node_ids=["local", "node-2"],
        )

        # The discovered rank card addresses its whole cluster, never the
        # single local container — otherwise the health monitor would
        # resurrect the deployment.
        self.assertEqual(started["status"], "running")
        self.manager.deployment_action.assert_awaited_once_with("cluster-1", "start", ["local", "node-2"])
        self.manager.start_container.assert_not_awaited()

    async def test_remote_relocation_adopts_manager_only_card_durably(self):
        original = {
            "id": "cluster-1", "status": "stopped", "engine": "vllm",
            "members": [{
                "node_id": "local", "container_name": "cluster-1-r0-model", "rank": 0,
            }],
            "launch_settings": {
                "engine": "vllm", "model": "org/model",
                "deployment_mode": "single", "node_ids": ["local"],
            },
        }
        replacement = {
            "id": "cluster-2", "status": "ready", "engine": "vllm",
            "model": "org/model", "api_port": 8020, "node_ids": ["worker-1"],
            "members": [{
                "node_id": "worker-1", "container_name": "cluster-2-r0-model", "rank": 0,
            }],
            "launch_settings": {
                "engine": "vllm", "model": "org/model",
                "deployment_mode": "single", "node_ids": ["worker-1"],
            },
        }
        self.manager.deployments = [original]
        self.manager.recipe_deployment_contract = lambda _recipe: {
            "deployment_mode": "single", "required_node_count": 1,
        }
        self.manager._resolve_local_path = Mock(return_value=None)
        self.manager.model_cache_inventory = AsyncMock(return_value=[{
            "id": "worker-1", "models": [{
                "model_id": "org/model", "revisions": ["main"], "partial": False,
            }],
        }])
        self.manager.list_containers.return_value = [{
            "name": "cluster-1-r0-model", "model": "org/model", "engine": "vllm",
            "managed": True, "status": "exited", "port": 8000,
        }]

        async def relocate(*_args):
            self.manager.deployments = [replacement]
            self.manager.list_containers.return_value = []
            return {"ok": True, "errors": [], "deployment": replacement}

        self.manager.deployment_action = AsyncMock(side_effect=relocate)
        card = (await self.service.deployments())[0]

        started = await self.service.deployment_action(
            card["id"], "start", node_ids=["worker-1"],
        )
        listed = await self.service.deployments()

        self.assertNotEqual(started["id"], card["id"])
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0]["id"], started["id"])
        self.assertEqual(listed[0]["node_ids"], ["worker-1"])
        stored = self.service.store.deployment(started["id"], include_private=True)
        self.assertEqual(stored["settings"]["manager_deployment_id"], "cluster-2")
        self.assertEqual(stored["container_name"], "cluster-2-r0-model")
        self.assertEqual(stored["_base_url"], "http://127.0.0.1:8020")
        self.assertEqual(replacement["sparkdeck_record_id"], started["id"])

    async def test_start_rejects_remote_node_for_controller_local_models(self):
        self.service.store.add_deployment(Deployment(
            id="dep-local", alias="local weights", runtime=RuntimeKind.VLLM,
            kind=DeploymentKind.MANAGED, model=ModelIdentity("/models/weights"),
            settings={"manager_deployment_id": "cluster-1"},
        ))
        self.manager.deployment_action = AsyncMock(return_value={"ok": True, "errors": []})
        self.manager._resolve_local_path = Mock(return_value="/models/weights")

        with self.assertRaisesRegex(ValueError, "controller-local model paths"):
            await self.service.deployment_action("dep-local", "start", node_ids=["remote-1"])
        self.manager.deployment_action.assert_not_awaited()

    async def test_llama_cpp_local_artifact_start_bypasses_hf_cache(self):
        self.service.store.add_deployment(Deployment(
            id="dep-llama", alias="gguf", runtime=RuntimeKind.LLAMA_CPP,
            kind=DeploymentKind.MANAGED,
            model=ModelIdentity("/models/model.gguf"),
            container_name="sparkdeck-gguf",
        ))
        self.manager._resolve_local_path = Mock(return_value=None)
        self.manager.model_cache_inventory = AsyncMock(return_value=[])
        self.manager.list_containers.return_value = [{
            "name": "sparkdeck-gguf", "model": "/models/model.gguf",
            "engine": "llama.cpp", "managed": True,
            "status": "exited", "port": 8000,
        }]

        started = await self.service.deployment_action(
            "dep-llama", "start", node_ids=["local"],
        )

        self.assertEqual(started["status"], "running")
        self.manager.start_container.assert_awaited_once_with(
            "sparkdeck-gguf", explicit=True,
        )
        self.manager.model_cache_inventory.assert_not_awaited()

    async def test_start_weight_check_uses_persisted_revision(self):
        self.service.store.add_deployment(Deployment(
            id="dep-pin", alias="pinned", runtime=RuntimeKind.VLLM,
            kind=DeploymentKind.MANAGED, model=ModelIdentity("org/model"),
            settings={"manager_deployment_id": "cluster-1"},
        ))
        self.manager.deployments = [{
            "id": "cluster-1", "status": "stopped", "engine": "vllm",
            "launch_settings": {"extra_args": ["--revision=release-b"]},
        }]
        self.manager.recipe_deployment_contract = lambda recipe: {
            "model_revision": "release-b", "deployment_mode": "single",
            "required_node_count": 1,
        }
        self.manager.deployment_action = AsyncMock(return_value={"ok": True, "errors": []})
        self.manager._resolve_local_path = Mock(return_value=None)
        # "main" is absent everywhere; only the persisted pin exists.
        self.manager.model_cache_inventory = AsyncMock(return_value=[
            {"id": "a", "models": [{"model_id": "org/model", "revisions": ["release-b"], "partial": False}]},
        ])

        await self.service.deployment_action("dep-pin", "start", node_ids=["a"])

        self.manager.deployment_action.assert_awaited_once_with("cluster-1", "start", ["a"])

    async def test_cluster_member_card_exposes_persisted_layout(self):
        self.manager.deployments = [{
            "id": "cluster-1", "status": "stopped", "engine": "vllm",
            "members": [{"node_id": "local", "container_name": "cluster-1-r0-model", "rank": 0}],
            "launch_settings": {
                "engine": "vllm", "node_ids": ["local", "node-2"],
                "deployment_mode": "replicated", "extra_args": [],
            },
        }]
        self.manager.recipe_deployment_contract = lambda recipe: {
            "required_node_count": 2, "deployment_mode": "replicated",
            "model_revision": "release-b",
        }
        self.manager.list_containers.return_value = [{
            "name": "cluster-1-r0-model", "model": "org/model", "engine": "vllm",
            "managed": True, "status": "exited", "port": 8000,
        }]

        cards = await self.service.deployments()

        self.assertEqual(cards[0]["required_node_count"], 2)
        self.assertEqual(cards[0]["deployment_mode"], "replicated")
        self.assertEqual(cards[0]["model_revision"], "release-b")

    async def test_concurrent_duplicate_alias_launches_only_one_container(self):
        async def launch(*_args, **_kwargs):
            await asyncio.sleep(0.01)
            return {"name": "sparkdeck-shared-dep", "port": 8000, "status": "running"}

        launch_mock = AsyncMock(side_effect=launch)
        body = {"model": "org/model", "alias": "shared", "runtime": "vllm"}
        with patch("sparkdeck.service.launch_managed_container", launch_mock):
            results = await asyncio.gather(
                self.service.create_deployment(body, launch=True),
                self.service.create_deployment(body, launch=True),
                return_exceptions=True,
            )

        self.assertEqual(launch_mock.await_count, 1)
        self.assertEqual(len([item for item in results if isinstance(item, dict)]), 1)
        errors = [item for item in results if isinstance(item, Exception)]
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], ValueError)
        self.assertEqual(len(self.service.store.deployments()), 1)

    async def test_background_cluster_launch_returns_after_durable_start_and_reports_progress(self):
        release_launch = asyncio.Event()
        cluster = {
            "id": "cluster-queued", "status": "launching", "api_port": 8123,
            "node_ids": ["local"], "model_source": "unknown",
            "members": [{
                "node_id": "local", "node_name": "Controller", "rank": 0,
                "container_name": "cluster-queued-r0-model", "status": "creating",
                "phase": {
                    "phase": "pulling_image",
                    "message": "Downloading Docker image example/vllm:latest",
                },
            }],
        }

        async def launch(_body, *, launch_persisted=None):
            self.manager.deployments = [cluster]
            launch_persisted.set_result(cluster)
            await release_launch.wait()
            cluster.update({
                "status": "error", "error": "Controller: image pull failed",
            })
            raise RuntimeError(cluster["error"])

        self.manager.deployments = []
        self.manager.selected_cluster_nodes = AsyncMock(return_value=[{
            "id": "local", "name": "Controller",
        }])
        self.manager.create_deployment = AsyncMock(side_effect=launch)
        self.manager.public_target_node = Mock(side_effect=lambda node: node)
        self.manager.get_state = AsyncMock(side_effect=lambda: {
            "deployments": self.manager.deployments,
        })
        self.manager.cluster_nodes = AsyncMock(return_value=[{
            "id": "local", "name": "Controller",
        }])

        created = await self.service.create_deployment({
            "model": "org/model", "alias": "model", "runtime": "vllm",
            "node_ids": ["local"], "deployment_mode": "single",
        }, launch=True, background=True)

        self.assertEqual(created["status"], "starting")
        self.assertEqual(created["launch_phase"], "pulling_image")
        self.assertIn("Downloading Docker image", created["launch_message"])
        self.assertFalse(self.service._deployment_launch_tasks[created["id"]].done())
        with self.assertRaisesRegex(RuntimeError, "launch is still in progress"):
            await self.service.deployment_action(created["id"], "stop")
        stored = self.service.store.deployment(created["id"], include_private=True)
        self.assertEqual(
            stored["settings"]["manager_deployment_id"], "cluster-queued",
        )
        self.assertEqual(stored["_base_url"], "http://127.0.0.1:8123")

        loading = (await self.service.deployments())[0]
        self.assertEqual(loading["status"], "starting")
        self.assertEqual(loading["launch_phase"], "pulling_image")

        launch_complete = self.service._deployment_launches[created["id"]]
        release_launch.set()
        await asyncio.wait_for(launch_complete.wait(), 1)
        failed = (await self.service.deployments())[0]
        self.assertEqual(failed["status"], "error")
        self.assertEqual(failed["launch_phase"], "error")
        self.assertEqual(failed["last_error"], "Controller: image pull failed")

    async def test_background_cluster_preflight_failure_removes_provisional_card(self):
        async def reject(_body, *, launch_persisted=None):
            error = ValueError("selected port is unavailable")
            launch_persisted.set_exception(error)
            raise error

        self.manager.selected_cluster_nodes = AsyncMock(return_value=[{
            "id": "local", "name": "Controller",
        }])
        self.manager.create_deployment = AsyncMock(side_effect=reject)

        with self.assertRaisesRegex(ValueError, "selected port is unavailable"):
            await self.service.create_deployment({
                "model": "org/model", "alias": "model", "runtime": "vllm",
                "node_ids": ["local"], "deployment_mode": "single",
            }, launch=True, background=True)

        self.assertIsNone(self.service.store.deployment("model"))

    async def test_client_cancellation_keeps_accepted_launch_owned_by_service(self):
        entered = asyncio.Event()
        accept = asyncio.Event()
        cluster = {
            "id": "cluster-accepted", "status": "starting", "api_port": 8123,
            "node_ids": ["local"], "members": [{
                "node_id": "local", "node_name": "Controller", "rank": 0,
                "container_name": "cluster-accepted-r0-model", "status": "starting",
                "phase": {"phase": "starting", "message": "Starting model"},
            }],
        }

        async def launch(_body, *, launch_persisted=None):
            entered.set()
            await accept.wait()
            self.manager.deployments = [cluster]
            launch_persisted.set_result(cluster)
            return cluster

        self.manager.deployments = []
        self.manager.selected_cluster_nodes = AsyncMock(return_value=[{
            "id": "local", "name": "Controller",
        }])
        self.manager.create_deployment = AsyncMock(side_effect=launch)
        self.manager.public_target_node = Mock(side_effect=lambda node: node)
        request = asyncio.create_task(self.service.create_deployment({
            "model": "org/model", "alias": "model", "runtime": "vllm",
            "node_ids": ["local"], "deployment_mode": "single",
        }, launch=True, background=True))
        await asyncio.wait_for(entered.wait(), 1)

        request.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await request

        provisional = self.service.store.deployment("model")
        self.assertIsNotNone(provisional)
        launch_complete = next(iter(self.service._deployment_launches.values()))
        accept.set()
        await asyncio.wait_for(launch_complete.wait(), 1)
        stored = self.service.store.deployment("model")
        self.assertEqual(
            stored["settings"]["manager_deployment_id"], "cluster-accepted",
        )

    async def test_service_close_removes_launch_cancelled_before_manager_persistence(self):
        manager = FakeManager()
        manager.deployments = []
        manager.selected_cluster_nodes = AsyncMock(return_value=[{
            "id": "local", "name": "Controller",
        }])
        entered = asyncio.Event()
        never = asyncio.Event()

        async def launch(_body, *, launch_persisted=None):
            entered.set()
            try:
                await never.wait()
            except asyncio.CancelledError:
                launch_persisted.set_exception(RuntimeError(
                    "deployment launch stopped before it was accepted"
                ))
                raise

        manager.create_deployment = AsyncMock(side_effect=launch)
        with tempfile.TemporaryDirectory() as directory:
            service = SparkDeckService(manager, Path(directory))
            request = asyncio.create_task(service.create_deployment({
                "model": "org/model", "alias": "model", "runtime": "vllm",
                "node_ids": ["local"], "deployment_mode": "single",
            }, launch=True, background=True))
            await asyncio.wait_for(entered.wait(), 1)
            with patch.object(
                service.store, "delete_deployment",
                wraps=service.store.delete_deployment,
            ) as delete:
                await service.close()
                with self.assertRaisesRegex(
                    RuntimeError, "stopped before it was accepted",
                ):
                    await request
                delete.assert_called_once()
        await manager.http.aclose()

    async def test_failed_persistence_removes_launched_container(self):
        launched = {
            "name": "sparkdeck-model-deadbeef", "port": 8000, "status": "running",
        }
        with (
            patch("sparkdeck.service.launch_managed_container", AsyncMock(return_value=launched)),
            patch.object(
                self.service.store, "update_managed_routing",
                side_effect=OSError("disk full"),
            ),
        ):
            with self.assertRaisesRegex(OSError, "disk full"):
                await self.service.create_deployment({
                    "model": "org/model", "alias": "model", "runtime": "vllm",
                }, launch=True)

        self.manager.remove_container.assert_awaited_once_with(launched["name"])
        self.assertIsNone(self.service.store.deployment("model"))

    async def test_failed_prelaunch_persistence_never_starts_docker(self):
        launch = AsyncMock()
        with (
            patch("sparkdeck.service.launch_managed_container", launch),
            patch.object(
                self.service.store, "add_deployment", side_effect=OSError("disk full"),
            ),
        ):
            with self.assertRaisesRegex(OSError, "disk full"):
                await self.service.create_deployment({
                    "model": "org/model", "alias": "model", "runtime": "vllm",
                }, launch=True)

        launch.assert_not_awaited()
        self.manager.remove_container.assert_not_awaited()

    async def test_delete_waits_for_inflight_creation_then_removes_container(self):
        launch_started = asyncio.Event()
        release_launch = asyncio.Event()

        async def launch(*_args, **_kwargs):
            launch_started.set()
            await release_launch.wait()
            return {
                "name": "sparkdeck-model-deadbeef",
                "port": 8000,
                "status": "running",
            }

        with patch(
            "sparkdeck.service.launch_managed_container",
            AsyncMock(side_effect=launch),
        ):
            create_task = asyncio.create_task(self.service.create_deployment({
                "model": "org/model", "alias": "model", "runtime": "vllm",
            }, launch=True))
            await asyncio.wait_for(launch_started.wait(), 1)
            provisional = self.service.store.deployment("model")
            self.assertIsNotNone(provisional)

            delete_task = asyncio.create_task(
                self.service.delete_deployment(provisional["id"])
            )
            await asyncio.sleep(0)
            self.assertFalse(delete_task.done())

            release_launch.set()
            created = await create_task
            removed = await delete_task

        self.assertEqual(removed, {"ok": True, "id": created["id"]})
        self.manager.remove_container.assert_awaited_once_with(
            "sparkdeck-model-deadbeef"
        )
        self.assertIsNone(self.service.store.deployment(created["id"]))

    async def test_unrelated_delete_does_not_wait_for_inflight_creation(self):
        self.service.store.add_deployment(Deployment(
            id="existing-deployment",
            alias="existing",
            runtime=RuntimeKind.VLLM,
            kind=DeploymentKind.MANAGED,
            model=ModelIdentity("org/existing"),
            container_name="sparkdeck-existing",
        ))
        launch_started = asyncio.Event()
        release_launch = asyncio.Event()

        async def launch(*_args, **_kwargs):
            launch_started.set()
            await release_launch.wait()
            return {
                "name": "sparkdeck-new-deadbeef",
                "port": 8000,
                "status": "running",
            }

        with patch(
            "sparkdeck.service.launch_managed_container",
            AsyncMock(side_effect=launch),
        ):
            create_task = asyncio.create_task(self.service.create_deployment({
                "model": "org/new", "alias": "new", "runtime": "vllm",
            }, launch=True))
            await asyncio.wait_for(launch_started.wait(), 1)

            removed = await asyncio.wait_for(
                self.service.delete_deployment("existing-deployment"), 1
            )

            self.assertEqual(
                removed, {"ok": True, "id": "existing-deployment"}
            )
            self.assertFalse(create_task.done())
            release_launch.set()
            await create_task

        self.manager.remove_container.assert_awaited_once_with("sparkdeck-existing")

    async def test_delete_removes_stale_row_when_manager_deployment_is_absent(self):
        self.service.store.add_deployment(Deployment(
            id="stale-record",
            alias="stale",
            runtime=RuntimeKind.VLLM,
            kind=DeploymentKind.MANAGED,
            model=ModelIdentity("org/model"),
            container_name="stale-r0",
            settings={"manager_deployment_id": "missing-manager-deployment"},
        ))
        self.manager.deployment_action = AsyncMock(
            side_effect=ValueError("deployment not found"),
        )

        result = await self.service.delete_deployment("stale-record")

        self.assertEqual(result, {"ok": True, "id": "stale-record"})
        self.assertIsNone(self.service.store.deployment("stale-record"))
        self.manager.deployment_action.assert_awaited_once_with(
            "missing-manager-deployment", "remove",
        )

    async def test_delete_preserves_row_for_real_manager_removal_failure(self):
        self.service.store.add_deployment(Deployment(
            id="live-record",
            alias="live",
            runtime=RuntimeKind.VLLM,
            kind=DeploymentKind.MANAGED,
            model=ModelIdentity("org/model"),
            container_name="live-r0",
            settings={"manager_deployment_id": "live-manager-deployment"},
        ))
        self.manager.deployment_action = AsyncMock(
            side_effect=ValueError("selected node is unavailable"),
        )

        with self.assertRaisesRegex(ValueError, "selected node is unavailable"):
            await self.service.delete_deployment("live-record")

        self.assertIsNotNone(self.service.store.deployment("live-record"))

    async def test_delete_stale_manager_id_removes_current_container_owner(self):
        self.service.store.add_deployment(Deployment(
            id="adopted-record",
            alias="adopted",
            runtime=RuntimeKind.VLLM,
            kind=DeploymentKind.MANAGED,
            model=ModelIdentity("org/model"),
            container_name="replacement-r0",
            settings={"manager_deployment_id": "stale-manager-deployment"},
        ))
        self.manager.deployments = [{
            "id": "replacement-manager-deployment",
            "members": [{"container_name": "replacement-r0"}],
        }]
        self.manager.deployment_action = AsyncMock(side_effect=[
            ValueError("deployment not found"),
            {"ok": True, "errors": []},
        ])

        result = await self.service.delete_deployment("adopted-record")

        self.assertEqual(result, {"ok": True, "id": "adopted-record"})
        self.assertIsNone(self.service.store.deployment("adopted-record"))
        self.assertEqual(
            [call.args for call in self.manager.deployment_action.await_args_list],
            [
                ("stale-manager-deployment", "remove"),
                ("replacement-manager-deployment", "remove"),
            ],
        )
        self.manager.remove_container.assert_not_awaited()

    async def test_delete_stale_manager_id_uses_reverse_linked_owner(self):
        self.service.store.add_deployment(Deployment(
            id="linked-record",
            alias="linked",
            runtime=RuntimeKind.VLLM,
            kind=DeploymentKind.MANAGED,
            model=ModelIdentity("org/model"),
            container_name="old-primary-r0",
            settings={"manager_deployment_id": "stale-manager-deployment"},
        ))
        self.manager.deployments = [{
            "id": "replacement-manager-deployment",
            "sparkdeck_record_id": "linked-record",
            "members": [{"container_name": "new-primary-r0"}],
        }]
        self.manager.deployment_action = AsyncMock(side_effect=[
            ValueError("deployment not found"),
            {"ok": True, "errors": []},
        ])

        result = await self.service.delete_deployment("linked-record")

        self.assertEqual(result, {"ok": True, "id": "linked-record"})
        self.assertEqual(
            [call.args for call in self.manager.deployment_action.await_args_list],
            [
                ("stale-manager-deployment", "remove"),
                ("replacement-manager-deployment", "remove"),
            ],
        )
        self.manager.remove_container.assert_not_awaited()

    async def test_delete_refreshes_owner_after_manager_side_redeployment(self):
        self.service.store.add_deployment(Deployment(
            id="auto-redeployed-record",
            alias="auto-redeployed",
            runtime=RuntimeKind.VLLM,
            kind=DeploymentKind.MANAGED,
            model=ModelIdentity("org/model"),
            container_name="old-primary-r0",
            settings={"manager_deployment_id": "old-manager"},
        ))
        self.manager.deployments = [{
            "id": "old-manager",
            "sparkdeck_record_id": "auto-redeployed-record",
            "members": [{"container_name": "old-primary-r0"}],
        }]

        async def remove_manager(deployment_id, action):
            if deployment_id == "old-manager":
                self.service.store.update_managed_routing(
                    "auto-redeployed-record",
                    {"manager_deployment_id": "replacement-manager"},
                    "replacement-primary-r0",
                    None,
                )
                self.manager.deployments = [{
                    "id": "replacement-manager",
                    "sparkdeck_record_id": "auto-redeployed-record",
                    "members": [{"container_name": "replacement-primary-r0"}],
                }]
                raise ValueError("deployment not found")
            return {"ok": True, "errors": []}

        self.manager.deployment_action = AsyncMock(side_effect=remove_manager)

        result = await self.service.delete_deployment("auto-redeployed-record")

        self.assertEqual(result, {"ok": True, "id": "auto-redeployed-record"})
        self.assertEqual(
            [call.args for call in self.manager.deployment_action.await_args_list],
            [("old-manager", "remove"), ("replacement-manager", "remove")],
        )
        self.assertIsNone(
            self.service.store.deployment("auto-redeployed-record"),
        )
        self.manager.remove_container.assert_not_awaited()

    async def test_delete_accepts_current_owner_removed_concurrently(self):
        self.service.store.add_deployment(Deployment(
            id="racing-record",
            alias="racing",
            runtime=RuntimeKind.VLLM,
            kind=DeploymentKind.MANAGED,
            model=ModelIdentity("org/model"),
            container_name="replacement-r0",
            settings={"manager_deployment_id": "stale-manager-deployment"},
        ))
        self.manager.deployments = [{
            "id": "replacement-manager-deployment",
            "members": [{"container_name": "replacement-r0"}],
        }]
        self.manager.deployment_action = AsyncMock(side_effect=[
            ValueError("deployment not found"),
            ValueError("deployment not found"),
        ])

        result = await self.service.delete_deployment("racing-record")

        self.assertEqual(result, {"ok": True, "id": "racing-record"})
        self.assertIsNone(self.service.store.deployment("racing-record"))
        self.manager.remove_container.assert_not_awaited()

    async def test_delete_refreshes_routing_after_same_record_action(self):
        self.service.store.add_deployment(Deployment(
            id="relaunching-record",
            alias="relaunching",
            runtime=RuntimeKind.VLLM,
            kind=DeploymentKind.MANAGED,
            model=ModelIdentity("org/model"),
            container_name="old-r0",
            settings={"manager_deployment_id": "old-manager"},
        ))
        lock = self.service._deployment_action_locks.setdefault(
            "relaunching-record", asyncio.Lock(),
        )
        await lock.acquire()
        delete_task = asyncio.create_task(
            self.service.delete_deployment("relaunching-record"),
        )
        await asyncio.sleep(0)
        self.manager.deployment_action = AsyncMock(
            return_value={"ok": True, "errors": []},
        )
        self.service.store.update_managed_routing(
            "relaunching-record",
            {"manager_deployment_id": "replacement-manager"},
            "replacement-r0",
            None,
        )
        self.manager.deployments = [{
            "id": "replacement-manager",
            "sparkdeck_record_id": "relaunching-record",
            "members": [{"container_name": "replacement-r0"}],
        }]
        lock.release()

        result = await delete_task

        self.assertEqual(result, {"ok": True, "id": "relaunching-record"})
        self.manager.deployment_action.assert_awaited_once_with(
            "replacement-manager", "remove",
        )
        self.assertIsNone(self.service.store.deployment("relaunching-record"))

    async def test_delete_removes_every_persisted_orphaned_cluster_rank(self):
        self.service.store.add_deployment(Deployment(
            id="orphaned-record",
            alias="orphaned",
            runtime=RuntimeKind.VLLM,
            kind=DeploymentKind.MANAGED,
            model=ModelIdentity("org/model"),
            container_name="cluster-missing-manager-r0-org-model",
            settings={
                "manager_deployment_id": "missing-manager",
                "node_ids": ["local", "worker-1"],
            },
        ))
        self.manager.deployment_action = AsyncMock(
            side_effect=ValueError("deployment not found"),
        )
        self.manager.remove_orphaned_deployment_members = AsyncMock(
            return_value={"ok": True, "errors": []},
        )

        result = await self.service.delete_deployment("orphaned-record")

        self.assertEqual(result, {"ok": True, "id": "orphaned-record"})
        self.assertIsNone(self.service.store.deployment("orphaned-record"))
        self.assertEqual(
            self.manager.remove_orphaned_deployment_members.await_args.args[0],
            [{
                "node_id": "local", "rank": 0,
                "container_name": "cluster-missing-manager-r0-org-model",
            }, {
                "node_id": "worker-1", "rank": 1,
                "container_name": "cluster-missing-manager-r1-org-model",
            }],
        )
        self.manager.remove_container.assert_not_awaited()

    async def test_delete_preserves_row_when_orphaned_rank_cannot_be_removed(self):
        self.service.store.add_deployment(Deployment(
            id="unreachable-record",
            alias="unreachable",
            runtime=RuntimeKind.VLLM,
            kind=DeploymentKind.MANAGED,
            model=ModelIdentity("org/model"),
            container_name="cluster-missing-manager-r0-org-model",
            settings={
                "manager_deployment_id": "missing-manager",
                "node_ids": ["local", "worker-1"],
            },
        ))
        self.manager.deployment_action = AsyncMock(
            side_effect=ValueError("deployment not found"),
        )
        self.manager.remove_orphaned_deployment_members = AsyncMock(
            return_value={"ok": False, "errors": ["Worker agent is offline"]},
        )

        with self.assertRaisesRegex(RuntimeError, "Worker agent is offline"):
            await self.service.delete_deployment("unreachable-record")

        self.assertIsNotNone(self.service.store.deployment("unreachable-record"))

    async def test_delete_keeps_remote_orphan_with_unverifiable_member_names(self):
        self.service.store.add_deployment(Deployment(
            id="unsafe-record",
            alias="unsafe",
            runtime=RuntimeKind.VLLM,
            kind=DeploymentKind.MANAGED,
            model=ModelIdentity("org/model"),
            container_name="legacy-primary",
            settings={
                "manager_deployment_id": "missing-manager",
                "node_ids": ["local", "worker-1"],
            },
        ))
        self.manager.deployment_action = AsyncMock(
            side_effect=ValueError("deployment not found"),
        )
        self.manager.remove_orphaned_deployment_members = AsyncMock()

        with self.assertRaisesRegex(RuntimeError, "remote ranks may remain"):
            await self.service.delete_deployment("unsafe-record")

        self.assertIsNotNone(self.service.store.deployment("unsafe-record"))
        self.manager.remove_orphaned_deployment_members.assert_not_awaited()


class RuntimeForwardingFixTests(unittest.IsolatedAsyncioTestCase):
    async def test_sglang_forwards_max_running_requests(self):
        manager = Mock()
        manager.create_container = AsyncMock(return_value={"name": "model", "port": 8000})

        await launch_managed_container(
            manager, SglangAdapter(), "dep-1", "model", "org/model",
            {"max_running_requests": 37},
        )

        self.assertEqual(
            manager.create_container.await_args.kwargs["sg_max_running_requests"], 37
        )

    async def test_llama_cpp_expands_home_path_before_mounting(self):
        with patch.object(Path, "is_file", return_value=True):
            spec = LlamaCppAdapter().launch_spec("unused", {"artifact": "~/model.gguf"})

        expected = str((Path.home() / "model.gguf").resolve())
        self.assertEqual(spec.volumes, {
            expected: {"bind": "/models/model.gguf", "mode": "ro"},
        })
        self.assertEqual(spec.command[-1], "/models/model.gguf")


class RecipeCompatibilityFixTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app), base_url="http://test",
        )

    async def asyncTearDown(self):
        await self.client.aclose()

    async def test_create_recipe_forwards_mcp_recipe_fields(self):
        created = {"id": "recipe-1", "model": "org/model"}
        add_recipe = AsyncMock(return_value=created)
        payload = {
            "model": "org/model", "name": "variant", "engine": "sglang",
            "sg_max_running_requests": 19, "launch_controls": {"max_concurrency": 19},
            "deployment_mode": "single", "node_ids": ["local"], "force_new": True,
        }
        with patch.object(server.manager, "add_recipe", add_recipe):
            response = await self.client.post("/api/recipes", json=payload)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), created)
        self.assertEqual(add_recipe.await_args.kwargs["sg_max_running_requests"], 19)
        self.assertEqual(add_recipe.await_args.kwargs["launch_controls"], {"max_concurrency": 19})
        self.assertTrue(add_recipe.await_args.kwargs["force_new"])

    async def test_update_recipe_preserves_not_found_contract(self):
        update_recipe = AsyncMock(side_effect=ValueError("recipe not found"))
        with patch.object(server.manager, "update_recipe", update_recipe):
            response = await self.client.put("/api/recipes/missing", json={"name": "new"})

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "recipe not found")

    async def test_state_keeps_recipes_for_mcp_discovery(self):
        with (
            patch.object(server.manager, "get_state", AsyncMock(return_value={"deployments": []})),
            patch.object(server.manager, "recipes", [{"id": "recipe-1"}]),
            patch.object(server.manager, "recipe_launches", {"recipe-1": {"phase": "ready"}}),
        ):
            response = await self.client.get("/api/state")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["recipes"], [{"id": "recipe-1"}])
        self.assertEqual(response.json()["recipe_launches"]["recipe-1"]["phase"], "ready")

    def test_public_legacy_recipe_preserves_shape_while_removing_credentials(self):
        self.assertEqual(server._public_legacy_recipe({"id": "recipe-1"}), {"id": "recipe-1"})
        self.assertEqual(
            server._public_legacy_recipe({
                "id": "recipe-2",
                "extra_args": ["--dtype", "auto", "--hf-token", "hf_secret"],
            }),
            {"id": "recipe-2", "extra_args": ["--dtype", "auto"]},
        )


if __name__ == "__main__":
    unittest.main()
