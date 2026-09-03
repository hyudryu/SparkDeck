import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import httpx

from manager import Manager
from sparkdeck.models import Deployment, DeploymentKind, ModelIdentity, RuntimeKind
from sparkdeck.service import SparkDeckService


def cluster_node(node_id: str, *, local: bool = False) -> dict:
    return {
        "id": node_id, "name": "Controller" if local else node_id,
        "local": local, "enabled": True, "online": True,
        "docker_ready": True, "agent_url": f"http://{node_id}.private",
        "agent_token": "secret",
    }


class ClusterImageInventoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_inventory_collects_local_and_remote_with_safe_partial_errors(self):
        manager = Manager.__new__(Manager)
        manager.cluster_nodes = AsyncMock(return_value=[
            cluster_node("local", local=True),
            cluster_node("worker-1"),
            cluster_node("worker-2"),
        ])
        manager.list_images = AsyncMock(return_value=[{"id": "sha256:a", "tags": ["org/a:latest"]}])
        manager.list_containers = AsyncMock(return_value=[])
        manager.node_registry = Mock()

        async def remote(node_id, *_args, **_kwargs):
            if node_id == "worker-2":
                raise RuntimeError("offline")
            return {
                "images": [{"id": "sha256:a", "tags": ["org/a:latest"]}],
                "containers": [],
            }

        manager.node_registry.request = AsyncMock(side_effect=remote)

        inventory = await manager.cluster_image_inventory()

        self.assertTrue(inventory["partial"])
        self.assertEqual([row["node"]["id"] for row in inventory["results"]], ["local", "worker-1"])
        self.assertEqual(inventory["errors"][0]["node"]["id"], "worker-2")
        self.assertNotIn("agent_url", str(inventory))
        self.assertNotIn("secret", str(inventory))


class FakeManager:
    def __init__(self):
        self.http = httpx.AsyncClient()
        self.deployments = []
        self.cluster_nodes = AsyncMock(return_value=[cluster_node("worker-1")])
        self.list_containers = AsyncMock(return_value=[])
        self.deployment_action = AsyncMock(return_value={"ok": True, "errors": []})

    @staticmethod
    def public_target_node(node):
        return Manager.public_target_node(node)


class DeploymentRenameSynchronizationTests(unittest.IsolatedAsyncioTestCase):
    async def test_cluster_rename_persists_service_and_manager_aliases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = Manager.__new__(Manager)
            manager.http = httpx.AsyncClient()
            manager.deployments_path = root / "manager-deployments.json"
            manager.deployments = [{
                "id": "manager-1",
                "name": "Old name",
                "model": "org/model",
                "launch_settings": {
                    "deployment_name": "Old name",
                    "model": "org/model",
                },
            }]
            service = SparkDeckService(manager, root)
            service.store.add_deployment(Deployment(
                id="record-1",
                alias="Old name",
                runtime=RuntimeKind.VLLM,
                kind=DeploymentKind.MANAGED,
                model=ModelIdentity("org/model"),
                settings={"manager_deployment_id": "manager-1"},
            ))
            try:
                renamed = await service.rename_deployment(
                    "record-1", "Production model"
                )

                self.assertEqual(renamed["alias"], "Production model")
                self.assertEqual(manager.deployments[0]["name"], "Production model")
                self.assertEqual(
                    manager.deployments[0]["launch_settings"]["deployment_name"],
                    "Production model",
                )
                self.assertEqual(
                    json.loads(manager.deployments_path.read_text())[0]["name"],
                    "Production model",
                )
                self.assertEqual(
                    service.store.deployment("record-1")["alias"],
                    "Production model",
                )
            finally:
                await service.close()
                await manager.http.aclose()

    async def test_discovered_and_stored_renames_cannot_claim_same_alias(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = Manager.__new__(Manager)
            manager.http = httpx.AsyncClient()
            manager.deployments = []
            manager.container_aliases = {}
            service = SparkDeckService(manager, root)
            service.store.add_deployment(Deployment(
                id="record-1",
                alias="Stored name",
                runtime=RuntimeKind.VLLM,
                kind=DeploymentKind.MANAGED,
                model=ModelIdentity("org/stored"),
            ))
            inventory_calls = 0
            persistence_entered = asyncio.Event()
            allow_persistence = asyncio.Event()

            async def deployments():
                nonlocal inventory_calls
                inventory_calls += 1
                return [
                    *service.store.deployments(),
                    {
                        "id": "container:discovered",
                        "alias": manager.container_aliases.get(
                            "discovered", "Discovered name",
                        ),
                        "runtime": "vllm",
                        "kind": "external",
                        "model": {"repository": "org/discovered"},
                    },
                ]

            async def update_container_alias(name, alias):
                persistence_entered.set()
                await allow_persistence.wait()
                manager.container_aliases[name] = alias
                return {"ok": True, "name": name, "alias": alias}

            service.deployments = deployments
            service._resolve_discovered_container = AsyncMock(
                return_value={"name": "discovered"},
            )
            service.deployment_detail = AsyncMock(return_value={
                "id": "container:discovered", "alias": "Shared name",
            })
            manager.update_container_alias = update_container_alias
            try:
                discovered = asyncio.create_task(service.rename_deployment(
                    "container:discovered", "Shared name",
                ))
                await asyncio.wait_for(persistence_entered.wait(), timeout=1)
                stored = asyncio.create_task(service.rename_deployment(
                    "record-1", "shared NAME",
                ))
                await asyncio.sleep(0)

                # The stored rename cannot run its awaited inventory check
                # until the discovered alias has been durably persisted.
                self.assertEqual(inventory_calls, 1)
                allow_persistence.set()
                await discovered
                with self.assertRaisesRegex(ValueError, "already in use"):
                    await stored
                self.assertEqual(inventory_calls, 2)
                self.assertEqual(
                    service.store.deployment("record-1")["alias"], "Stored name",
                )
            finally:
                allow_persistence.set()
                await service.close()
                await manager.http.aclose()


class ReplacementReconciliationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.manager = FakeManager()
        self.service = SparkDeckService(self.manager, Path(self.temp.name))
        self.service.store.add_deployment(Deployment(
            id="record-1", alias="friendly", runtime=RuntimeKind.VLLM,
            kind=DeploymentKind.MANAGED, model=ModelIdentity("org/model"),
            container_name="old-rank",
            settings={
                "context_length": 8192,
                "node_ids": ["worker-1"],
                "manager_deployment_id": "old-manager",
            },
        ), "http://127.0.0.1:8000")

    async def asyncTearDown(self):
        await self.manager.http.aclose()
        await self.service.close()
        self.temp.cleanup()

    async def test_computed_ready_replacement_updates_sqlite_and_model_visibility(self):
        replacement = {
            "id": "new-manager", "sparkdeck_record_id": "record-1",
            "status": "ready", "api_port": 8010, "node_ids": ["worker-1"],
            "members": [{"rank": 0, "node_id": "worker-1", "container_name": "new-rank"}],
        }
        self.manager.get_state = AsyncMock(return_value={"deployments": [replacement]})
        self.manager.list_containers.return_value = [{
            "name": "new-rank", "model": "org/model", "runtime": "vllm",
            "managed": True, "status": "running", "port": 8010,
            "phase": {"phase": "ready"},
        }]

        listed = await self.service.deployments()
        models = await self.service.models()

        self.assertEqual(listed[0]["status"], "running")
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0]["id"], "record-1")
        self.assertEqual(models["data"][0]["id"], "friendly")
        stored = self.service.store.deployment("record-1", include_private=True)
        self.assertEqual(stored["settings"]["manager_deployment_id"], "new-manager")
        self.assertEqual(stored["container_name"], "new-rank")
        self.assertEqual(stored["_base_url"], "http://127.0.0.1:8010")

        await self.service.deployment_action("record-1", "stop")
        self.manager.deployment_action.assert_awaited_once_with("new-manager", "stop")
        self.assertEqual(
            self.service.store.deployment("record-1")["desired_state"],
            "stopped",
        )

    async def test_remote_only_legacy_manager_deployment_is_adopted_once(self):
        self.service.store.delete_deployment("record-1")
        self.manager._save_deployments = Mock()
        self.manager.deployments = [{
            "id": "manager-hermes",
            "name": "glm53-exl3-2x",
            "model": "Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw",
            "engine": "vllm",
            "mode": "sharded",
            "node_ids": ["worker-1", "worker-2"],
            "status": "ready",
            "desired_state": "running",
            "managed_by": "sparkdeck-mcp",
            "automation_run_id": "hermes-run-1",
            "api_port": 8000,
            "members": [
                {
                    "node_id": "worker-1", "rank": 0,
                    "container_name": "cluster-manager-hermes-r0-glm",
                },
                {
                    "node_id": "worker-2", "rank": 1,
                    "container_name": "cluster-manager-hermes-r1-glm",
                },
            ],
            "launch_settings": {
                "deployment_name": "glm53-exl3-2x",
                "deployment_mode": "sharded",
                "extra_args": ["--tensor-parallel-size", "2"],
            },
        }]
        self.manager.cluster_nodes.return_value = [
            cluster_node("worker-1"), cluster_node("worker-2"),
        ]

        first = await self.service.deployments()
        second = await self.service.deployments()

        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 1)
        adopted = first[0]
        self.assertEqual(adopted["alias"], "glm53-exl3-2x")
        self.assertEqual(adopted["status"], "running")
        self.assertEqual(adopted["node_ids"], ["worker-1", "worker-2"])
        self.assertTrue(adopted["managed"])
        self.assertEqual(adopted["managed_by"], "sparkdeck-mcp")
        self.assertEqual(adopted["automation_run_id"], "hermes-run-1")
        record_id = self.manager.deployments[0]["sparkdeck_record_id"]
        self.assertEqual(second[0]["id"], record_id)
        self.assertEqual(
            self.service.store.deployment(record_id)["settings"][
                "manager_deployment_id"
            ],
            "manager-hermes",
        )
        stored_settings = self.service.store.deployment(record_id)["settings"]
        self.assertEqual(stored_settings["managed_by"], "sparkdeck-mcp")
        self.assertEqual(stored_settings["automation_run_id"], "hermes-run-1")
        self.assertEqual(
            self.manager.deployments[0]["launch_settings"]["sparkdeck_record_id"],
            record_id,
        )
        self.manager._save_deployments.assert_called_once_with()

        await self.service.deployment_action(record_id, "stop")
        self.manager.deployment_action.assert_awaited_once_with(
            "manager-hermes", "stop",
        )

    async def test_completed_legacy_launch_refreshes_early_adoption(self):
        self.service.store.delete_deployment("record-1")
        self.manager._save_deployments = Mock()
        cluster = {
            "id": "manager-hermes",
            "name": "model",
            "model": "org/model",
            "engine": "vllm",
            "mode": "single",
            "node_ids": ["worker-1"],
            "status": "starting",
            "api_port": None,
            "members": [],
            "launch_settings": {"extra_args": []},
        }
        self.manager.deployments = [cluster]

        early = await self.service.deployments()
        record_id = early[0]["id"]
        self.assertIsNone(
            self.service.store.deployment(record_id, include_private=True)[
                "_base_url"
            ]
        )
        cluster.update({
            "status": "ready",
            "api_port": 8123,
            "model_source": "public_repository",
            "members": [{
                "node_id": "worker-1", "rank": 0,
                "container_name": "manager-hermes-r0",
            }],
        })

        registered = await self.service.register_manager_deployment(cluster)

        stored = self.service.store.deployment(record_id, include_private=True)
        self.assertEqual(registered["_base_url"], "http://127.0.0.1:8123")
        self.assertEqual(stored["_base_url"], "http://127.0.0.1:8123")
        self.assertEqual(stored["container_name"], "manager-hermes-r0")
        self.assertEqual(stored["settings"]["model_source"], "public_repository")

    async def test_llama_adoption_preserves_manager_gguf_artifact(self):
        self.service.store.delete_deployment("record-1")
        self.manager._save_deployments = Mock()
        self.manager.deployments = [{
            "id": "manager-llama",
            "name": "GGUF deployment",
            "model": "org/gguf-model",
            "engine": "llama.cpp",
            "mode": "single",
            "node_ids": ["worker-1"],
            "status": "ready",
            "api_port": 8124,
            "members": [{
                "node_id": "worker-1", "rank": 0,
                "container_name": "manager-llama-r0",
            }],
            "launch_settings": {
                "llama_artifact": "weights/model-Q4_K_M.gguf",
                "extra_args": [],
            },
        }]

        listed = await self.service.deployments()

        self.assertEqual(
            listed[0]["model"]["artifact"], "weights/model-Q4_K_M.gguf",
        )
        record_id = listed[0]["id"]
        model = dict(listed[0]["model"])
        model["artifact"] = None
        self.service.store.update_deployment_model(record_id, model)

        registered = await self.service.register_manager_deployment(
            self.manager.deployments[0],
        )

        self.assertEqual(
            registered["model"]["artifact"], "weights/model-Q4_K_M.gguf",
        )

    async def test_deployment_reads_do_not_wait_for_long_create_lock(self):
        self.manager.deployments = [{
            "id": "manager-in-flight", "name": "in-flight",
            "model": "org/in-flight", "engine": "vllm",
            "node_ids": ["worker-1"], "members": [],
            "launch_settings": {},
        }]
        await self.service._deployment_create_lock.acquire()
        try:
            listed = await asyncio.wait_for(self.service.deployments(), timeout=0.2)
        finally:
            self.service._deployment_create_lock.release()

        self.assertEqual(listed[0]["id"], "record-1")
        self.assertFalse(any(item["alias"] == "in-flight" for item in listed))
        refreshed = await self.service.deployments()
        self.assertTrue(any(item["alias"] == "in-flight" for item in refreshed))

    async def test_reverse_link_collision_allocates_a_distinct_record(self):
        self.manager._save_deployments = Mock()
        self.manager.deployments = [{
            "id": "new-manager",
            "sparkdeck_record_id": "record-1",
            "name": "Another deployment",
            "model": "org/other-model",
            "engine": "vllm",
            "mode": "single",
            "node_ids": ["worker-1"],
            "status": "ready",
            "desired_state": "running",
            "api_port": 8010,
            "members": [{
                "node_id": "worker-1", "rank": 0,
                "container_name": "new-manager-r0",
            }],
            "launch_settings": {"extra_args": []},
        }]

        listed = await self.service.deployments()

        original = self.service.store.deployment("record-1")
        self.assertEqual(
            original["settings"]["manager_deployment_id"], "old-manager",
        )
        adopted = next(item for item in listed if item["alias"] == "Another deployment")
        self.assertNotEqual(adopted["id"], "record-1")
        self.assertEqual(
            adopted["settings"]["manager_deployment_id"], "new-manager",
        )
        self.assertEqual(
            self.manager.deployments[0]["sparkdeck_record_id"], adopted["id"],
        )

        await self.service.deployment_action("record-1", "stop")
        self.manager.deployment_action.assert_awaited_once_with(
            "old-manager", "stop",
        )
        self.manager.deployment_action.reset_mock()
        await self.service.deployment_action(adopted["id"], "stop")
        self.manager.deployment_action.assert_awaited_once_with(
            "new-manager", "stop",
        )

    async def test_remove_manager_registration_is_scoped_and_idempotent(self):
        self.service.store.add_deployment(Deployment(
            id="record-2", alias="unrelated", runtime=RuntimeKind.VLLM,
            kind=DeploymentKind.MANAGED, model=ModelIdentity("org/other"),
            settings={"manager_deployment_id": "another-manager"},
        ))

        removed = self.service.remove_manager_deployment_registration(
            "old-manager",
        )

        self.assertEqual(removed, "record-1")
        self.assertIsNone(self.service.store.deployment("record-1"))
        self.assertIsNotNone(self.service.store.deployment("record-2"))
        self.assertIsNone(
            self.service.remove_manager_deployment_registration("old-manager")
        )

    async def test_settings_dirty_start_result_immediately_reconciles_replacement(self):
        self.manager.deployment_action.return_value = {
            "ok": True,
            "errors": [],
            "replaced_deployment_id": "old-manager",
            "deployment": {
                "id": "replacement-manager", "api_port": 8020,
                "node_ids": ["worker-1"],
                "members": [{"container_name": "replacement-rank"}],
            },
        }

        started = await self.service.deployment_action("record-1", "start")

        self.assertEqual(started["status"], "running")
        stored = self.service.store.deployment("record-1", include_private=True)
        self.assertEqual(
            stored["settings"]["manager_deployment_id"], "replacement-manager"
        )
        self.assertEqual(stored["container_name"], "replacement-rank")
        self.assertEqual(stored["_base_url"], "http://127.0.0.1:8020")
        self.assertEqual(stored["desired_state"], "running")

    async def test_concurrent_start_then_stop_finishes_durably_stopped(self):
        start_entered = asyncio.Event()
        release_start = asyncio.Event()

        async def action(_deployment_id, action):
            if action == "start":
                start_entered.set()
                await release_start.wait()
            return {"ok": True, "errors": []}

        self.manager.deployment_action.side_effect = action
        start = asyncio.create_task(
            self.service.deployment_action("record-1", "start")
        )
        await start_entered.wait()
        stop = asyncio.create_task(
            self.service.deployment_action("record-1", "stop")
        )
        await asyncio.sleep(0)
        release_start.set()
        await asyncio.gather(start, stop)

        self.assertEqual(
            self.service.store.deployment("record-1")["desired_state"],
            "stopped",
        )
        self.assertEqual(
            [call.args[1] for call in self.manager.deployment_action.await_args_list],
            ["start", "stop"],
        )

    async def test_remote_first_replica_does_not_duplicate_later_local_member(self):
        self.service.store.update_container("record-1", "remote-rank-0")
        deployment = {
            "id": "old-manager", "sparkdeck_record_id": "record-1",
            "status": "ready", "api_port": 8000,
            "node_ids": ["worker-1", "local"],
            "members": [
                {
                    "rank": 0, "node_id": "worker-1",
                    "container_name": "remote-rank-0",
                },
                {
                    "rank": 1, "node_id": "local",
                    "container_name": "local-rank-1",
                },
            ],
        }
        self.manager.get_state = AsyncMock(return_value={"deployments": [deployment]})
        self.manager.list_containers.return_value = [{
            "name": "local-rank-1", "model": "org/model", "runtime": "vllm",
            "managed": True, "status": "running", "port": 8000,
            "phase": {"phase": "ready"},
        }]

        listed = await self.service.deployments()

        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0]["id"], "record-1")
        self.assertEqual(listed[0]["status"], "running")
        self.assertFalse(any(item["id"] == "container:local-rank-1" for item in listed))


class DeploymentSettingsContractTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.manager = Manager.__new__(Manager)
        self.manager.http = httpx.AsyncClient()
        self.manager.deployments_path = root / "manager-deployments.json"
        self.manager.deployments = [{
            "id": "manager-1",
            "sparkdeck_record_id": "record-1",
            "name": "Editable model",
            "model": "org/model",
            "engine": "vllm",
            "mode": "single",
            "node_ids": ["local"],
            "status": "stopped",
            "desired_state": "stopped",
            "api_port": 8000,
            "members": [],
            "launch_settings": {
                "deployment_name": "Editable model",
                "model": "org/model",
                "engine": "vllm",
                "deployment_mode": "single",
                "node_ids": ["local"],
                "extra_args": [
                    "--hf-token", "do-not-expose",
                    "--api-key", "also-do-not-expose",
                    "--max-model-len", "32768",
                    "--enable-prefix-caching",
                ],
                "gpu_memory_utilization": 0.9,
                "gpu_memory_gb": 12,
                "image": "example/vllm:test",
            },
        }]
        self.manager.get_state = AsyncMock(
            side_effect=lambda: {"deployments": self.manager.deployments},
        )
        self.manager.cluster_nodes = AsyncMock(return_value=[
            cluster_node("local", local=True),
        ])
        self.manager.list_containers = AsyncMock(return_value=[])
        self.service = SparkDeckService(self.manager, root)
        self.service.store.add_deployment(Deployment(
            id="record-1", alias="Editable model", runtime=RuntimeKind.VLLM,
            kind=DeploymentKind.MANAGED, model=ModelIdentity("org/model"),
            settings={"manager_deployment_id": "manager-1"},
            desired_state="stopped",
        ), "http://127.0.0.1:8000")

    async def asyncTearDown(self):
        await self.service.close()
        await self.manager.http.aclose()
        self.temp.cleanup()

    async def test_detail_resolves_public_record_and_redacts_launch_credentials(self):
        detail = await self.service.deployment_detail("record-1")

        self.assertEqual(detail["id"], "record-1")
        self.assertTrue(detail["editable"])
        self.assertEqual(detail["desired_state"], "stopped")
        self.assertEqual(detail["launch_controls"]["context_window"], 32768)
        self.assertEqual(detail["extra_args"], [
            "--max-model-len", "32768", "--enable-prefix-caching",
        ])
        self.assertEqual(detail["gpu_memory_utilization"], 0.9)
        self.assertEqual(detail["gpu_memory_gb"], 12)
        self.assertEqual(detail["image"], "example/vllm:test")
        self.assertNotIn("do-not-expose", str(detail))
        self.assertNotIn("also-do-not-expose", str(detail))
        self.assertNotIn("launch_settings", detail)
        self.assertNotIn("manager_deployment_id", str(detail))

    async def test_update_maps_record_to_manager_and_refreshes_public_settings(self):
        detail = await self.service.update_deployment_settings("record-1", {
            "extra_args": ["--enable-prefix-caching"],
            "launch_controls": {"context_window": 65536},
            "gpu_memory_utilization": 0.8,
            "gpu_memory_gb": None,
        })

        manager_deployment = self.manager.deployments[0]
        self.assertTrue(manager_deployment["settings_dirty"])
        self.assertEqual(
            self.manager._cli_option(
                manager_deployment["launch_settings"]["extra_args"],
                {"--max-model-len"}, int,
            ),
            65536,
        )
        stored = self.service.store.deployment("record-1")
        self.assertEqual(
            stored["settings"]["manager_deployment_id"], "manager-1",
        )
        self.assertEqual(stored["settings"]["context_length"], 65536)
        self.assertEqual(detail["launch_controls"]["context_window"], 65536)
        self.assertEqual(detail["gpu_memory_utilization"], 0.8)

    async def test_update_rejects_credential_bearing_runtime_flags(self):
        with self.assertRaisesRegex(ValueError, "credentials in Settings"):
            await self.service.update_deployment_settings("record-1", {
                "extra_args": ["--api-key", "never-persist-this"],
            })

        self.assertNotIn(
            "never-persist-this", str(self.manager.deployments[0]),
        )

    async def test_manager_stopped_state_repairs_store_and_blocks_proxy(self):
        self.service.store.update_desired_state("record-1", "running")

        listed = await self.service.deployments()

        self.assertEqual(listed[0]["desired_state"], "stopped")
        stored = self.service.store.deployment("record-1", include_private=True)
        self.assertEqual(stored["desired_state"], "stopped")
        with self.assertRaisesRegex(RuntimeError, "deployment is stopped"):
            await self.service._proxy_registered(
                stored, {"model": "Editable model"}, "chat/completions", None,
            )


class WorkerSchedulerTests(unittest.IsolatedAsyncioTestCase):
    async def test_joined_worker_keeps_local_inference_nudger(self):
        manager = Manager.__new__(Manager)
        manager.data_dir = Path("unused")
        manager.is_joined_worker = Mock(return_value=True)
        manager._start_controller_tasks = Mock()
        blocker = asyncio.Event()

        async def loop():
            await blocker.wait()

        manager._temperature_history_monitor_loop = loop
        manager._inference_nudger_loop = loop
        manager._start_mem_bw_monitor = Mock()

        await manager.start()

        manager._start_controller_tasks.assert_not_called()
        self.assertFalse(manager.inference_nudger_task.done())
        for task in (manager.inference_nudger_task, manager.temperature_history_task):
            task.cancel()
        await asyncio.gather(
            manager.inference_nudger_task, manager.temperature_history_task,
            return_exceptions=True,
        )


if __name__ == "__main__":
    unittest.main()
