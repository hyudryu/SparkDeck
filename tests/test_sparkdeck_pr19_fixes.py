import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import httpx

from sparkdeck.models import Deployment, DeploymentKind, ModelIdentity, RuntimeKind
from sparkdeck.runtimes import LlamaCppAdapter, SglangAdapter, launch_managed_container
from sparkdeck.service import SparkDeckService


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
        self.manager.stop_container.assert_awaited_once_with("legacy-model")
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
                {"node_id": "local", "container_name": "cluster-1-r0-model", "rank": 0},
                {"node_id": "node-2", "container_name": "cluster-1-r1-model", "rank": 1},
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
        self.assertEqual(sorted(tails), [
            ("cluster-1-r0-model", "logs", 150),
            ("cluster-1-r1-model", "logs", 150),
        ])

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

    async def test_start_rejects_node_selection_for_standalone_container(self):
        self.manager.list_containers.return_value = [{
            "name": "legacy-model", "model": "org/model", "engine": "vllm",
            "managed": True, "status": "exited", "port": 8000,
        }]

        with self.assertRaisesRegex(ValueError, "node selection is only available for cluster deployments"):
            await self.service.deployment_action("container:legacy-model", "start", node_ids=["remote-1"])
        self.manager.start_container.assert_not_awaited()

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
        }
        self.manager.list_containers.return_value = [{
            "name": "cluster-1-r0-model", "model": "org/model", "engine": "vllm",
            "managed": True, "status": "exited", "port": 8000,
        }]

        cards = await self.service.deployments()

        self.assertEqual(cards[0]["required_node_count"], 2)
        self.assertEqual(cards[0]["deployment_mode"], "replicated")

    async def test_concurrent_duplicate_alias_launches_only_one_container(self):
        async def launch(*_args, **_kwargs):
            await asyncio.sleep(0.01)
            return {"name": "sparkdeck-shared-dep", "port": 8000, "status": "running"}

        launch_mock = AsyncMock(side_effect=launch)
        body = {"model": "org/model", "alias": "shared", "runtime": "vllm"}
        with patch("sparkdeck.service.launch_managed_container", launch_mock):
            results = await asyncio.gather(
                self.service.create_deployment(body),
                self.service.create_deployment(body),
                return_exceptions=True,
            )

        self.assertEqual(launch_mock.await_count, 1)
        self.assertEqual(len([item for item in results if isinstance(item, dict)]), 1)
        errors = [item for item in results if isinstance(item, Exception)]
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], ValueError)
        self.assertEqual(len(self.service.store.deployments()), 1)

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
                })

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
                })

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
            }))
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
            }))
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
