import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx

from sparkdeck.onboarding import is_forwardable_path
from sparkdeck.virtual_nas import VirtualNAS


with patch("docker.from_env", return_value=Mock()):
    import server


class VirtualNASApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app),
            base_url="http://test",
        )

    async def asyncTearDown(self):
        await self.client.aclose()

    async def test_disabled_public_operations_are_rejected_but_can_be_enabled(self):
        enabled = False

        def is_enabled():
            return enabled

        async def update_settings(values):
            nonlocal enabled
            enabled = values["virtual_nas_enabled"]
            return {}

        async def inventory():
            return {"enabled": enabled, "nodes": [], "jobs": []}

        with (
            patch.object(server.manager, "virtual_nas_enabled", side_effect=is_enabled),
            patch.object(server.manager, "update_settings", side_effect=update_settings),
            patch.object(
                server.manager, "virtual_nas_inventory",
                AsyncMock(side_effect=inventory),
            ),
        ):
            disabled = await self.client.get("/api/v1/storage")
            responses = [
                await self.client.post("/api/v1/storage/transfers", json={}),
                await self.client.delete("/api/v1/storage/transfers/job-1"),
                await self.client.delete(
                    "/api/v1/storage/nodes/local/models/org/model"
                ),
            ]
            invalid = await self.client.put(
                "/api/v1/storage/settings", json={"enabled": "yes"},
            )
            changed = await self.client.put(
                "/api/v1/storage/settings", json={"enabled": True},
            )

        self.assertEqual(disabled.status_code, 200)
        self.assertEqual(disabled.json()["enabled"], False)
        self.assertEqual(disabled.json()["nodes"], [])
        self.assertEqual(disabled.json()["jobs"], [])
        self.assertTrue(disabled.json()["instructions"])
        self.assertTrue(all(item.status_code == 409 for item in responses))
        self.assertTrue(all("Enable it in Storage" in item.text for item in responses))
        self.assertEqual(invalid.status_code, 400)
        self.assertTrue(changed.json()["enabled"])

    async def test_legacy_cluster_create_registers_v1_deployment_before_return(self):
        cluster = {
            "id": "manager-hermes", "status": "starting",
            "sparkdeck_record_id": "record-hermes",
        }
        create = AsyncMock(return_value=cluster)
        register = AsyncMock(return_value={"id": "record-hermes"})
        with (
            patch.object(server.manager, "create_deployment", create),
            patch.object(server.sparkdeck, "register_manager_deployment", register),
        ):
            response = await self.client.post("/api/containers", json={
                "model": "org/model",
                "deployment_mode": "single",
                "node_ids": ["local"],
                "managed_by": "sparkdeck-mcp",
            })

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["id"], "manager-hermes")
        create.assert_awaited_once()
        register.assert_awaited_once_with(cluster)

    async def test_legacy_cluster_create_rolls_back_when_registration_fails(self):
        cluster = {"id": "manager-hermes", "status": "starting"}
        create = AsyncMock(return_value=cluster)
        remove = AsyncMock(return_value={"ok": True, "errors": []})
        with (
            patch.object(server.manager, "create_deployment", create),
            patch.object(server.manager, "deployment_action", remove),
            patch.object(
                server.sparkdeck, "register_manager_deployment",
                AsyncMock(side_effect=RuntimeError("store unavailable")),
            ),
        ):
            response = await self.client.post("/api/containers", json={
                "model": "org/model",
                "deployment_mode": "single",
                "node_ids": ["local"],
            })

        self.assertEqual(response.status_code, 500)
        self.assertIn("store unavailable", response.json()["detail"])
        remove.assert_awaited_once_with("manager-hermes", "remove")

    async def test_legacy_cluster_remove_deletes_linked_v1_registration(self):
        remove = AsyncMock(return_value={"ok": True, "errors": []})
        unregister = Mock(return_value="record-hermes")
        with (
            patch.object(server.manager, "deployment_action", remove),
            patch.object(
                server.sparkdeck, "remove_manager_deployment_registration",
                unregister,
            ),
        ):
            response = await self.client.post(
                "/api/deployments/manager-hermes/remove",
            )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["ok"])
        remove.assert_awaited_once_with("manager-hermes", "remove")
        unregister.assert_called_once_with("manager-hermes")

    async def test_failed_legacy_cluster_remove_keeps_v1_registration(self):
        remove = AsyncMock(return_value={"ok": False, "errors": ["busy"]})
        unregister = Mock()
        with (
            patch.object(server.manager, "deployment_action", remove),
            patch.object(
                server.sparkdeck, "remove_manager_deployment_registration",
                unregister,
            ),
        ):
            response = await self.client.post(
                "/api/deployments/manager-hermes/remove",
            )

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["ok"])
        remove.assert_awaited_once_with("manager-hermes", "remove")
        unregister.assert_not_called()

    async def test_shared_inventory_and_transfer_payloads_are_redacted(self):
        inventory = {
            "enabled": True,
            "nodes": [{
                "id": "local", "name": "Spark", "online": True,
                "total_size": 42, "agent_url": "http://worker.private:7878",
                "models": [{
                    "model_id": "org/model", "size_bytes": 42,
                    "cache_path": "/home/private/cache", "agent_token": "secret",
                }],
            }],
            "jobs": [{
                "id": "job-1", "status": "running", "snapshot_path": "/private",
            }],
        }
        with (
            patch.object(server.manager, "virtual_nas_enabled", return_value=True),
            patch.object(
                server.manager, "virtual_nas_inventory", AsyncMock(return_value=inventory),
            ),
        ):
            response = await self.client.get("/api/v1/storage")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["nodes"][0]["models"][0]["model_id"], "org/model")
        self.assertEqual(payload["jobs"][0]["id"], "job-1")
        serialized = str(payload)
        self.assertNotIn("cache_path", serialized)
        self.assertNotIn("agent_url", serialized)
        self.assertNotIn("agent_token", serialized)
        self.assertNotIn("worker.private", serialized)

    async def test_transfer_contract_validates_shape_and_delegates_node_checks(self):
        queue = AsyncMock(return_value={
            "job_ids": ["job-1"],
            "jobs": [{"id": "job-1", "status": "queued", "token": "hidden"}],
        })
        with (
            patch.object(server.manager, "virtual_nas_enabled", return_value=True),
            patch.object(server.manager, "queue_virtual_nas_transfer", queue),
        ):
            invalid = await self.client.post(
                "/api/v1/storage/transfers",
                json={"model_id": "org/model", "source_node_id": "local"},
            )
            duplicate = await self.client.post(
                "/api/v1/storage/transfers",
                json={
                    "model_id": "org/model", "source_node_id": "local",
                    "target_node_ids": ["worker-1", "worker-1"],
                },
            )
            response = await self.client.post(
                "/api/v1/storage/transfers",
                json={
                    "model_id": " org/model ", "source_node_id": " local ",
                    "target_node_ids": ["worker-1"],
                },
            )

        self.assertEqual(invalid.status_code, 400)
        self.assertEqual(duplicate.status_code, 400)
        self.assertEqual(response.status_code, 202)
        self.assertNotIn("token", str(response.json()))
        queue.assert_awaited_once_with("org/model", "local", ["worker-1"], None)

    async def test_recipe_transfer_preflight_is_available_before_queueing(self):
        preflight = AsyncMock(return_value={
            "enabled": True,
            "model_id": "org/model",
            "revision": "main",
            "source": {"node_id": "local", "node_name": "Controller", "size_bytes": 20},
            "targets": [{
                "node_id": "worker-1", "node_name": "Worker", "eligible": True,
                "free_bytes": 100, "required_free_bytes": 80,
            }],
        })
        with patch.object(
            server.manager, "virtual_nas_transfer_preflight", preflight,
        ):
            response = await self.client.post(
                "/api/v1/storage/transfers/preflight",
                json={"model_id": " org/model ", "revision": "main"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["targets"][0]["eligible"])
        preflight.assert_awaited_once_with("org/model", "main")

    async def test_recipe_preparation_derives_model_contract_and_rejects_duplicate_nodes(self):
        recipe = {
            "id": "recipe-1", "model": "org/model", "engine": "vllm",
            "node_ids": ["local", "worker-1"],
        }
        contract = {
            "supported": True, "required_node_count": 2,
            "deployment_mode": "replicated", "model_revision": "release-1",
        }
        preflight = AsyncMock(return_value={
            "enabled": True, "eligible": True, "action": "download",
            "model_id": "org/model", "revision": "release-1",
            "node_ids": ["local", "worker-1"], "targets": [],
            "transfer_target_node_ids": ["worker-1"],
        })
        with (
            patch.object(server.manager, "get_recipe", AsyncMock(return_value=recipe)),
            patch.object(server.manager, "recipe_deployment_contract", return_value=contract),
            patch.object(server.manager, "_resolve_local_path", return_value=None),
            patch.object(server.manager, "selected_cluster_nodes", AsyncMock(return_value=[])),
            patch.object(server.manager, "recipe_model_preparation_preflight", preflight),
        ):
            response = await self.client.post(
                "/api/v1/recipes/recipe-1/prepare/preflight",
                json={"node_ids": ["local", "worker-1"]},
            )
            duplicate = await self.client.post(
                "/api/v1/recipes/recipe-1/prepare/preflight",
                json={"node_ids": ["local", "local"]},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(duplicate.status_code, 400)
        preflight.assert_awaited_once_with(
            "org/model", "release-1", ["local", "worker-1"],
            download_node_id=None,
        )

    async def test_delete_maps_absent_and_in_use_without_exposing_core_details(self):
        delete = AsyncMock(side_effect=[
            LookupError("cached model not found"),
            RuntimeError("model is in use by deployment chat"),
            ValueError("invalid model ID"),
        ])
        with (
            patch.object(server.manager, "virtual_nas_enabled", return_value=True),
            patch.object(server.manager, "delete_virtual_nas_model", delete),
        ):
            absent = await self.client.delete(
                "/api/v1/storage/nodes/local/models/org/missing"
            )
            in_use = await self.client.delete(
                "/api/v1/storage/nodes/local/models/org/serving"
            )
            arbitrary = await self.client.delete(
                "/api/v1/storage/nodes/local/models/etc/passwd"
            )

        self.assertEqual(absent.status_code, 404)
        self.assertEqual(in_use.status_code, 409)
        self.assertEqual(arbitrary.status_code, 400)

    async def test_transfer_existing_target_is_a_conflict(self):
        with (
            patch.object(server.manager, "virtual_nas_enabled", return_value=True),
            patch.object(
                server.manager, "queue_virtual_nas_transfer",
                AsyncMock(side_effect=FileExistsError(
                    "cached model already exists on target node"
                )),
            ),
        ):
            response = await self.client.post(
                "/api/v1/storage/transfers",
                json={
                    "model_id": "org/model", "source_node_id": "local",
                    "target_node_ids": ["worker-1"],
                },
            )

        self.assertEqual(response.status_code, 409)

    async def test_finish_partial_download_contract_queues_selected_node(self):
        queue = AsyncMock(return_value={
            "job_ids": ["download-1"],
            "jobs": [{"id": "download-1", "status": "queued", "token": "hidden"}],
        })
        with (
            patch.object(server.manager, "virtual_nas_enabled", return_value=True),
            patch.object(server.manager, "queue_virtual_nas_download", queue),
        ):
            response = await self.client.post(
                "/api/v1/storage/nodes/worker-1/models/org/model/download",
                json={"revision": "main"},
            )

        self.assertEqual(response.status_code, 202)
        queue.assert_awaited_once_with("org/model", "worker-1", "main")
        self.assertNotIn("token", response.text)

    async def test_agent_storage_routes_require_a_paired_node_token(self):
        with patch.object(
            server.manager.agent_credentials, "authorize_controller", return_value=False,
        ):
            responses = [
                await self.client.get("/api/agent/virtual-nas/inventory"),
                await self.client.post(
                    "/api/agent/virtual-nas/models/org/model/download",
                    json={"revision": "main", "hf_token": ""},
                ),
                await self.client.get(
                    "/api/agent/virtual-nas/models/org/model/export"
                ),
                await self.client.put(
                    "/api/agent/virtual-nas/models/org/model/import", content=b"tar",
                ),
                await self.client.delete(
                    "/api/agent/virtual-nas/models/org/model"
                ),
            ]

        self.assertTrue(all(item.status_code == 401 for item in responses))

    async def test_agent_download_uses_local_capacity_checked_operation(self):
        resolved = "a" * 40
        checked = AsyncMock(return_value={
            "ok": True, "model_id": "org/model", "revision": resolved,
            "size_bytes": 20,
        })
        with (
            patch.object(server, "_require_agent"),
            patch.object(server.manager.virtual_nas, "download_model_checked", checked),
        ):
            response = await self.client.post(
                "/api/agent/virtual-nas/models/org/model/download",
                json={
                    "revision": resolved,
                    "requested_revision": "main",
                    "hf_token": "ephemeral",
                    "download_cache_baseline_bytes": 7,
                },
            )

        self.assertEqual(response.status_code, 200)
        checked.assert_awaited_once_with(
            "org/model", resolved, "ephemeral", "main", 7,
        )
        self.assertNotIn("ephemeral", response.text)

    async def test_agent_download_supports_selective_files(self):
        resolved = "b" * 40
        checked_files = AsyncMock(return_value={
            "ok": True, "model_id": "org/model", "revision": resolved,
            "size_bytes": 12, "files": ["q4/model.gguf"],
        })
        with (
            patch.object(server, "_require_agent"),
            patch.object(server.manager.virtual_nas, "download_model_files_checked", checked_files),
        ):
            response = await self.client.post(
                "/api/agent/virtual-nas/models/org/model/download",
                json={
                    "revision": resolved,
                    "requested_revision": "main",
                    "hf_token": "ephemeral",
                    "files": ["q4/model.gguf"],
                },
            )

        self.assertEqual(response.status_code, 200)
        checked_files.assert_awaited_once_with(
            "org/model", resolved, ["q4/model.gguf"],
            explicit_token="ephemeral", requested_revision="main",
        )
        self.assertNotIn("ephemeral", response.text)

    async def test_agent_files_check_reports_selected_file_presence(self):
        has_files = Mock(return_value={
            "model_id": "org/model", "revision": "main",
            "present_files": ["q4/model.gguf"], "missing_files": [],
            "complete": True,
        })
        with (
            patch.object(server, "_require_agent"),
            patch.object(server.manager.virtual_nas, "has_model_files", has_files),
        ):
            response = await self.client.post(
                "/api/agent/virtual-nas/models/org/model/files/check",
                json={"revision": "main", "files": ["q4/model.gguf"]},
            )
            invalid = await self.client.post(
                "/api/agent/virtual-nas/models/org/model/files/check",
                json={"revision": "main", "files": []},
            )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["complete"])
        has_files.assert_called_once_with("org/model", "main", ["q4/model.gguf"])
        self.assertEqual(invalid.status_code, 400)

    async def test_generic_model_preparation_queues_with_a_seed(self):
        queue = AsyncMock(return_value={"workflow_id": "wf-1", "job_ids": [], "jobs": []})
        with (
            patch.object(server.manager, "virtual_nas_enabled", return_value=True),
            patch.object(server.manager, "selected_cluster_nodes", AsyncMock(return_value=[])),
            patch.object(server.manager, "queue_recipe_model_preparation", queue),
        ):
            response = await self.client.post(
                "/api/v1/storage/preparations",
                json={
                    "model_id": "org/model", "revision": "main",
                    "node_ids": ["local", "worker-1"],
                    "download_node_id": "worker-1",
                },
            )
            missing_seed = await self.client.post(
                "/api/v1/storage/preparations",
                json={
                    "model_id": "org/model", "revision": "main",
                    "node_ids": ["local", "worker-1"],
                    "download_node_id": "worker-9",
                },
            )

        self.assertEqual(response.status_code, 202)
        queue.assert_awaited_once_with("org/model", "main", ["local", "worker-1"], "worker-1")
        self.assertEqual(missing_seed.status_code, 400)

    async def test_recipe_preparation_preflight_forwards_an_explicit_seed(self):
        recipe = {
            "id": "recipe-1", "model": "org/model", "engine": "vllm",
            "node_ids": ["local", "worker-1"],
        }
        contract = {
            "supported": True, "required_node_count": 2,
            "deployment_mode": "replicated", "model_revision": "release-1",
        }
        preflight = AsyncMock(return_value={
            "enabled": True, "eligible": True, "action": "download",
            "model_id": "org/model", "revision": "release-1",
            "node_ids": ["local", "worker-1"], "targets": [],
            "transfer_target_node_ids": ["worker-1"],
        })
        with (
            patch.object(server.manager, "get_recipe", AsyncMock(return_value=recipe)),
            patch.object(server.manager, "recipe_deployment_contract", return_value=contract),
            patch.object(server.manager, "_resolve_local_path", return_value=None),
            patch.object(server.manager, "selected_cluster_nodes", AsyncMock(return_value=[])),
            patch.object(server.manager, "recipe_model_preparation_preflight", preflight),
        ):
            response = await self.client.post(
                "/api/v1/recipes/recipe-1/prepare/preflight",
                json={"node_ids": ["local", "worker-1"], "download_node_id": "worker-1"},
            )

        self.assertEqual(response.status_code, 200)
        preflight.assert_awaited_once_with(
            "org/model", "release-1", ["local", "worker-1"],
            download_node_id="worker-1",
        )

    async def test_agent_inventory_export_import_and_delete_contracts(self):
        async def export(_model_id):
            yield b"tar-"
            yield b"bytes"

        received = {}

        async def import_model(
            model_id, chunks, expected_bytes=None, required_model_bytes=None,
        ):
            received["model_id"] = model_id
            received["expected_bytes"] = expected_bytes
            received["body"] = b"".join([chunk async for chunk in chunks])
            return {"ok": True, "model_id": model_id, "cache_path": "/private"}

        virtual_nas = SimpleNamespace(
            inventory=Mock(return_value=[{
                "model_id": "org/model", "size_bytes": 9, "cache_path": "/private",
            }]),
            free_bytes=Mock(return_value=42),
            export_model=export,
            import_model=import_model,
        )
        delete = AsyncMock(return_value={
            "ok": True, "model_id": "org/model", "node_id": "local",
        })
        with (
            patch.object(
                server.manager.agent_credentials, "authorize_controller", return_value=True,
            ),
            patch.object(server.manager, "virtual_nas", virtual_nas, create=True),
            patch.object(server.manager, "delete_virtual_nas_model", delete),
        ):
            headers = {"authorization": "Bearer paired-secret"}
            inventory = await self.client.get(
                "/api/agent/virtual-nas/inventory", headers=headers,
            )
            exported = await self.client.get(
                "/api/agent/virtual-nas/models/org/model/export", headers=headers,
            )
            imported = await self.client.put(
                "/api/agent/virtual-nas/models/org/model/import",
                headers={**headers, "x-sparkdeck-expected-bytes": "3"},
                content=b"tar",
            )
            deleted = await self.client.delete(
                "/api/agent/virtual-nas/models/org/model", headers=headers,
            )

        self.assertEqual(inventory.status_code, 200)
        self.assertNotIn("cache_path", str(inventory.json()))
        self.assertEqual(inventory.json()["free_size"], 42)
        self.assertEqual(exported.content, b"tar-bytes")
        self.assertEqual(
            exported.headers["content-type"],
            "application/vnd.sparkdeck.file-stream",
        )
        self.assertEqual(imported.status_code, 200)
        self.assertNotIn("cache_path", str(imported.json()))
        self.assertEqual(received, {
            "model_id": "org/model", "expected_bytes": 3, "body": b"tar",
        })
        self.assertEqual(deleted.status_code, 200)
        delete.assert_awaited_once_with("local", "org/model")

    async def test_agent_inventory_runs_recursive_scan_off_event_loop(self):
        inventory_scan = Mock(return_value=[{
            "model_id": "org/model", "size_bytes": 9,
        }])
        free_scan = Mock(return_value=123)
        virtual_nas = SimpleNamespace(inventory=inventory_scan, free_bytes=free_scan)
        delegated = AsyncMock(side_effect=[inventory_scan.return_value, free_scan.return_value])
        with (
            patch.object(
                server.manager.agent_credentials, "authorize_controller", return_value=True,
            ),
            patch.object(server.manager, "virtual_nas", virtual_nas, create=True),
            patch.object(server.asyncio, "to_thread", delegated),
        ):
            response = await self.client.get(
                "/api/agent/virtual-nas/inventory",
                headers={"authorization": "Bearer paired-secret"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["models"][0]["model_id"], "org/model")
        self.assertEqual(response.json()["free_size"], 123)
        self.assertEqual(delegated.await_count, 2)
        delegated.assert_any_call(inventory_scan)
        delegated.assert_any_call(free_scan)
        inventory_scan.assert_not_called()
        free_scan.assert_not_called()

    async def test_agent_export_reports_missing_model_before_stream_headers(self):
        virtual_nas = SimpleNamespace(
            export_model=Mock(side_effect=LookupError("cached model not found")),
        )
        with (
            patch.object(
                server.manager.agent_credentials, "authorize_controller", return_value=True,
            ),
            patch.object(server.manager, "virtual_nas", virtual_nas, create=True),
        ):
            response = await self.client.get(
                "/api/agent/virtual-nas/models/org/missing/export",
                headers={"authorization": "Bearer paired-secret"},
            )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "cached model not found")

    def test_public_storage_is_shared_through_normal_worker_forwarding(self):
        self.assertTrue(is_forwardable_path("/api/v1/storage"))
        self.assertTrue(is_forwardable_path("/api/v1/storage/transfers"))
        self.assertFalse(is_forwardable_path("/api/agent/virtual-nas/inventory"))


class VirtualNASInventoryTests(unittest.TestCase):
    def _nas(self, root: Path) -> tuple[VirtualNAS, Path]:
        hub = root / "hub"
        hub.mkdir()
        return VirtualNAS(root / "data", lambda: hub, Mock(), lambda: True), hub

    @staticmethod
    def _snapshot(hub: Path, revision: str = "revision-1") -> tuple[Path, Path]:
        repository = hub / "models--org--model"
        snapshot = repository / "snapshots" / revision
        blobs = repository / "blobs"
        snapshot.mkdir(parents=True)
        blobs.mkdir()
        return snapshot, blobs

    def test_inventory_marks_partial_transformer_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            nas, hub = self._nas(Path(directory))
            snapshot, blobs = self._snapshot(hub)
            (snapshot / "config.json").write_text("{}", encoding="utf-8")
            (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")
            (snapshot / "model-00001-of-00002.safetensors").write_bytes(b"one")
            (blobs / "unrelated-complete-blob").write_bytes(b"not a shard")

            models = nas.inventory()
            self.assertEqual(len(models), 1)
            self.assertTrue(models[0]["partial"])
            self.assertEqual(models[0]["revisions"], [])

    def test_free_bytes_measures_the_hub_filesystem(self):
        with tempfile.TemporaryDirectory() as directory:
            nas, hub = self._nas(Path(directory))

            free = nas.free_bytes()

            # Concurrent writes make exact byte equality racy across calls,
            # so compare against a fresh reading within a small tolerance.
            fresh = shutil.disk_usage(hub).free
            self.assertIsInstance(free, int)
            self.assertGreaterEqual(free, 0)
            self.assertLessEqual(abs(free - fresh), max(1024 * 1024, fresh // 1000))

    def test_inventory_requires_config_tokenizer_and_all_indexed_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            nas, hub = self._nas(Path(directory))
            snapshot, _ = self._snapshot(hub)
            (snapshot / "config.json").write_text("{}", encoding="utf-8")
            (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")
            (snapshot / "model-00001-of-00002.safetensors").write_bytes(b"one")
            (snapshot / "model-00002-of-00002.safetensors").write_bytes(b"two")
            (snapshot / "model.safetensors.index.json").write_text(
                '{"weight_map":{"a":"model-00001-of-00002.safetensors",'
                '"b":"model-00002-of-00002.safetensors"}}',
                encoding="utf-8",
            )

            models = nas.inventory()

            self.assertEqual(len(models), 1)
            self.assertEqual(models[0]["model_id"], "org/model")
            self.assertEqual(models[0]["revisions"], ["revision-1"])

    def test_inventory_marks_missing_runtime_requirements_partial(self):
        # A snapshot with config but no tokenizer is unusable by Transformers
        # runtimes and stays partial; a weights-only snapshot ships neither by
        # design and counts as complete.
        cases = {
            "configuration": ({"tokenizer.json", "model.safetensors"}, False),
            "tokenizer": ({"config.json", "model.safetensors"}, True),
        }
        for missing, (filenames, partial) in cases.items():
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as directory:
                nas, hub = self._nas(Path(directory))
                snapshot, _ = self._snapshot(hub)
                for filename in filenames:
                    (snapshot / filename).write_bytes(b"content")

                self.assertIs(nas.inventory()[0]["partial"], partial)

    def test_inventory_marks_weight_index_with_missing_file_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            nas, hub = self._nas(Path(directory))
            snapshot, _ = self._snapshot(hub)
            (snapshot / "config.json").write_text("{}", encoding="utf-8")
            (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")
            (snapshot / "model-a.safetensors").write_bytes(b"one")
            (snapshot / "model.safetensors.index.json").write_text(
                '{"weight_map":{"a":"model-a.safetensors",'
                '"b":"model-b.safetensors"}}',
                encoding="utf-8",
            )

            self.assertTrue(nas.inventory()[0]["partial"])

    def test_inventory_advertises_only_complete_revision_and_matching_ref(self):
        with tempfile.TemporaryDirectory() as directory:
            nas, hub = self._nas(Path(directory))
            incomplete, _ = self._snapshot(hub, "incomplete")
            (incomplete / "config.json").write_text("{}", encoding="utf-8")
            complete = incomplete.parent / "complete"
            complete.mkdir()
            (complete / "model.gguf").write_bytes(b"complete gguf")
            refs = incomplete.parent.parent / "refs"
            refs.mkdir()
            (refs / "main").write_text("complete", encoding="utf-8")
            (refs / "broken").write_text("incomplete", encoding="utf-8")

            models = nas.inventory()

            self.assertEqual(models[0]["revisions"], ["complete", "main"])

    @unittest.skipIf(os.name == "nt", "creating cache symlinks requires Windows privileges")
    def test_inventory_marks_snapshot_with_dangling_blob_link_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            nas, hub = self._nas(Path(directory))
            snapshot, blobs = self._snapshot(hub)
            config_blob = blobs / "config"
            tokenizer_blob = blobs / "tokenizer"
            config_blob.write_text("{}", encoding="utf-8")
            tokenizer_blob.write_text("{}", encoding="utf-8")
            (snapshot / "config.json").symlink_to(config_blob)
            (snapshot / "tokenizer.json").symlink_to(tokenizer_blob)
            (snapshot / "model.safetensors").symlink_to(blobs / "missing-weight")

            self.assertTrue(nas.inventory()[0]["partial"])


if __name__ == "__main__":
    unittest.main()
