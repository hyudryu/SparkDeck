import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx

from sparkdeck.onboarding import is_forwardable_path


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
        queue.assert_awaited_once_with("org/model", "local", ["worker-1"])

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

    async def test_agent_storage_routes_require_a_paired_node_token(self):
        with patch.object(
            server.manager.agent_credentials, "authorize_controller", return_value=False,
        ):
            responses = [
                await self.client.get("/api/agent/virtual-nas/inventory"),
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

    async def test_agent_inventory_export_import_and_delete_contracts(self):
        async def export(_model_id):
            yield b"tar-"
            yield b"bytes"

        received = {}

        async def import_model(model_id, chunks, expected_bytes=None):
            received["model_id"] = model_id
            received["expected_bytes"] = expected_bytes
            received["body"] = b"".join([chunk async for chunk in chunks])
            return {"ok": True, "model_id": model_id, "cache_path": "/private"}

        virtual_nas = SimpleNamespace(
            inventory=Mock(return_value=[{
                "model_id": "org/model", "size_bytes": 9, "cache_path": "/private",
            }]),
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
        self.assertEqual(exported.content, b"tar-bytes")
        self.assertEqual(exported.headers["content-type"], "application/x-tar")
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
        virtual_nas = SimpleNamespace(inventory=inventory_scan)
        delegated = AsyncMock(return_value=inventory_scan.return_value)
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
        delegated.assert_awaited_once_with(inventory_scan)
        inventory_scan.assert_not_called()

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


if __name__ == "__main__":
    unittest.main()
