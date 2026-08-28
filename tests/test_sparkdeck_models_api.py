import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import httpx


with patch("docker.from_env", return_value=Mock()):
    import server

from sparkdeck.models import Deployment, DeploymentKind, ModelIdentity, RuntimeKind
from sparkdeck.storage import SparkDeckStore


RECIPE = {
    "id": "r1",
    "name": "DS config",
    "model": "deepseek-ai/DeepSeek-V4",
    "engine": "vllm",
    "extra_args": ["--max-model-len", "32768", "--enable-prefix-caching"],
    "deployment_mode": "single",
    "node_ids": ["local"],
}


class ModelsApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app), base_url="http://test",
        )
        # Keep requests local even if the data dir holds a worker assignment.
        self.assignment = patch.object(
            server.onboarding.assignment, "load", return_value=None,
        )
        self.assignment.start()

    async def asyncTearDown(self):
        await self.client.aclose()
        self.assignment.stop()

    async def test_recipe_detail_returns_editable_args_and_launch_controls(self):
        with patch.object(server.manager, "get_recipe", AsyncMock(return_value=dict(RECIPE))):
            response = await self.client.get("/api/v1/recipes/r1")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["extra_args"], RECIPE["extra_args"])
        self.assertEqual(body["extra_args_count"], 3)
        self.assertEqual(body["launch_controls"]["context_window"], 32768)

    async def test_recipe_detail_maps_missing_recipe(self):
        with patch.object(server.manager, "get_recipe", AsyncMock(return_value=None)):
            response = await self.client.get("/api/v1/recipes/missing")

        self.assertEqual(response.status_code, 404)

    async def test_update_recipe_delegates_and_returns_public_detail(self):
        update = AsyncMock(return_value=dict(RECIPE))
        with patch.object(server.manager, "update_recipe", update):
            response = await self.client.put(
                "/api/v1/recipes/r1",
                json={
                    "launch_controls": {"context_window": 65536},
                    "extra_args": ["--enable-prefix-caching"],
                },
            )

        self.assertEqual(response.status_code, 200)
        update.assert_awaited_once_with("r1", {
            "launch_controls": {"context_window": 65536},
            "extra_args": ["--enable-prefix-caching"],
        })
        self.assertEqual(response.json()["extra_args"], RECIPE["extra_args"])

    async def test_update_recipe_rejects_unknown_fields(self):
        response = await self.client.put("/api/v1/recipes/r1", json={"bogus": 1})

        self.assertEqual(response.status_code, 400)

    async def test_update_recipe_maps_validation_and_missing_recipe(self):
        update = AsyncMock(side_effect=ValueError("recipe not found"))
        with patch.object(server.manager, "update_recipe", update):
            missing = await self.client.put("/api/v1/recipes/r1", json={"name": "x"})
        update.side_effect = ValueError("context_window must be a positive integer")
        with patch.object(server.manager, "update_recipe", update):
            invalid = await self.client.put(
                "/api/v1/recipes/r1", json={"launch_controls": {"context_window": -1}},
            )

        self.assertEqual(missing.status_code, 404)
        self.assertEqual(invalid.status_code, 400)

    async def test_delete_recipe_delegates_and_returns_no_content(self):
        delete = AsyncMock(return_value=True)
        with patch.object(server.manager, "delete_recipe", delete):
            response = await self.client.delete("/api/v1/recipes/r1")

        self.assertEqual(response.status_code, 204)
        self.assertEqual(response.content, b"")
        delete.assert_awaited_once_with("r1")

    async def test_delete_recipe_maps_missing_recipe(self):
        with patch.object(server.manager, "delete_recipe", AsyncMock(return_value=False)):
            response = await self.client.delete("/api/v1/recipes/missing")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "saved configuration not found")

    async def test_legacy_delete_recipe_contract_remains_available(self):
        delete = AsyncMock(return_value=True)
        with patch.object(server.manager, "delete_recipe", delete):
            response = await self.client.delete("/api/recipes/r1")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"ok": True})
        delete.assert_awaited_once_with("r1")

    async def test_rename_deployment_delegates_to_service(self):
        rename = AsyncMock(return_value={"id": "dep-1", "alias": "Renamed"})
        with patch.object(server.sparkdeck, "rename_deployment", rename):
            response = await self.client.patch(
                "/api/v1/deployments/dep-1", json={"alias": "Renamed"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["alias"], "Renamed")
        rename.assert_awaited_once_with("dep-1", "Renamed")

    async def test_rename_deployment_maps_validation_and_missing(self):
        rename = AsyncMock(side_effect=ValueError("alias is required"))
        with patch.object(server.sparkdeck, "rename_deployment", rename):
            invalid = await self.client.patch("/api/v1/deployments/dep-1", json={"alias": " "})
        rename.side_effect = LookupError("deployment not found")
        with patch.object(server.sparkdeck, "rename_deployment", rename):
            missing = await self.client.patch(
                "/api/v1/deployments/missing", json={"alias": "Renamed"},
            )

        self.assertEqual(invalid.status_code, 400)
        self.assertEqual(missing.status_code, 404)

    async def test_deployment_detail_returns_curated_editable_settings(self):
        detail = {
            "id": "dep-1", "alias": "Model", "runtime": "vllm",
            "kind": "managed", "model": {"repository": "org/model"},
            "status": "stopped", "settings": {}, "editable": True,
            "edit_reason": None, "desired_state": "stopped",
            "extra_args": ["--enable-prefix-caching"],
            "launch_controls": {"context_window": 32768},
            "gpu_memory_utilization": 0.9, "gpu_memory_gb": None,
            "image": "example/vllm:test",
        }
        get_detail = AsyncMock(return_value=detail)
        with patch.object(server.sparkdeck, "deployment_detail", get_detail):
            response = await self.client.get("/api/v1/deployments/dep-1")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), detail)
        get_detail.assert_awaited_once_with("dep-1")

    async def test_deployment_detail_maps_missing_deployment(self):
        get_detail = AsyncMock(side_effect=LookupError("deployment not found"))
        with patch.object(server.sparkdeck, "deployment_detail", get_detail):
            response = await self.client.get("/api/v1/deployments/missing")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "deployment not found")

    async def test_update_deployment_settings_uses_exact_public_contract(self):
        changes = {
            "extra_args": ["--enable-prefix-caching"],
            "launch_controls": {"context_window": 65536},
            "gpu_memory_utilization": 0.9,
            "gpu_memory_gb": None,
        }
        updated = AsyncMock(return_value={
            "id": "dep-1", "editable": True, "extra_args": changes["extra_args"],
        })
        with patch.object(server.sparkdeck, "update_deployment_settings", updated):
            response = await self.client.put(
                "/api/v1/deployments/dep-1/settings", json=changes,
            )

        self.assertEqual(response.status_code, 200)
        updated.assert_awaited_once_with("dep-1", changes)

        response = await self.client.put(
            "/api/v1/deployments/dep-1/settings",
            json={"desired_state": "running"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("unsupported field", response.json()["detail"])

    async def test_update_deployment_settings_maps_missing_and_running_state(self):
        update = AsyncMock(side_effect=LookupError("deployment not found"))
        with patch.object(server.sparkdeck, "update_deployment_settings", update):
            missing = await self.client.put(
                "/api/v1/deployments/missing/settings", json={"extra_args": []},
            )

        update.side_effect = ValueError(
            "stop the cluster before changing its launch settings"
        )
        with patch.object(server.sparkdeck, "update_deployment_settings", update):
            running = await self.client.put(
                "/api/v1/deployments/dep-1/settings", json={"extra_args": []},
            )

        self.assertEqual(missing.status_code, 404)
        self.assertEqual(running.status_code, 409)


class DeploymentRenameStoreTests(unittest.TestCase):
    def test_update_alias_persists_new_name(self):
        with tempfile.TemporaryDirectory() as temp:
            store = SparkDeckStore(Path(temp) / "sparkdeck.sqlite3")
            try:
                store.add_deployment(Deployment(
                    id="dep-1", alias="old-name", runtime=RuntimeKind.VLLM,
                    kind=DeploymentKind.MANAGED, model=ModelIdentity("org/model"),
                ))
                store.update_alias("dep-1", "new-name")
                self.assertEqual(store.deployment("dep-1")["alias"], "new-name")
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
