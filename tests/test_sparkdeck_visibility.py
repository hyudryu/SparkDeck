import unittest
from unittest.mock import AsyncMock, Mock, patch

import httpx

from manager import Manager


with patch("docker.from_env", return_value=Mock()):
    import server


class ModelCacheInventoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_inventory_reports_each_node_without_enabling_transfers(self):
        manager = Manager.__new__(Manager)
        manager.cluster_nodes = AsyncMock(return_value=[
            {"id": "local", "name": "Spark One", "online": True, "disk": {"total": 100}},
            {"id": "node-2", "name": "Spark Two", "online": True, "disk": {"total": 200}},
        ])
        manager.virtual_nas = Mock()
        manager.virtual_nas.inventory.return_value = [{"model_id": "org/model", "size_bytes": 10}]
        manager.node_registry = Mock()
        manager.node_registry.request = AsyncMock(return_value={"models": [{"model_id": "org/model", "size_bytes": 11}]})

        nodes = await manager.model_cache_inventory()

        self.assertEqual([node["models"][0]["size_bytes"] for node in nodes], [10, 11])
        manager.node_registry.request.assert_awaited_once_with(
            "node-2", "GET", "/api/agent/virtual-nas/inventory", timeout=30,
        )


class DailyUsageSeriesTests(unittest.TestCase):
    def test_daily_usage_preserves_per_model_series(self):
        manager = Manager.__new__(Manager)
        manager.hourly_token_stats = {
            "2026-08-26T10": {
                "org/a": {"input": 10, "output": 4, "cached": 2, "requests": 1},
                "org/b": {"input": 20, "output": 8, "cached": 3, "requests": 2},
            },
        }

        result = manager.get_daily_token_stats("2026-08-26", "2026-08-26")

        self.assertEqual(result[0]["input"], 30)
        self.assertEqual(result[0]["models"]["org/a"]["output"], 4)
        self.assertEqual(result[0]["models"]["org/b"]["requests"], 2)


class SavedConfigurationApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app), base_url="http://test",
        )

    async def asyncTearDown(self):
        await self.client.aclose()

    async def test_legacy_recipe_defaults_are_visible_and_launch_through_sparkdeck(self):
        recipe = {
            "id": "recipe-1", "model": "org/model",
            "extra_args": ["--dtype", "auto", "--hf-token", "secret"],
        }
        created = {
            "id": "dep-1", "alias": "org/model", "runtime": "vllm",
            "kind": "managed", "model": {"repository": "org/model"},
            "status": "starting", "settings": {},
        }
        with (
            patch.object(server.manager, "recipes", [recipe]),
            patch.object(server.manager, "recipe_launches", {}),
            patch.object(server.manager, "get_recipe", AsyncMock(return_value=recipe)),
            patch.object(server.sparkdeck, "create_deployment", AsyncMock(return_value=created)) as create,
        ):
            listed = await self.client.get("/api/v1/recipes")
            launched = await self.client.post("/api/v1/recipes/recipe-1/deploy")

        self.assertEqual(listed.json()["items"][0]["node_ids"], ["local"])
        self.assertEqual(listed.json()["items"][0]["engine"], "vllm")
        self.assertEqual(listed.json()["items"][0]["extra_args_count"], 4)
        self.assertNotIn("secret", listed.text)
        self.assertEqual(launched.status_code, 201)
        self.assertEqual(
            create.await_args.args[0]["settings"]["extra_args"],
            ["--dtype", "auto", "--hf-token", "secret"],
        )
        self.assertEqual(create.await_args.args[0]["recipe_id"], "recipe-1")

    async def test_model_cache_endpoint_redacts_private_paths(self):
        with patch.object(server.manager, "model_cache_inventory", AsyncMock(return_value=[{
            "id": "local", "name": "Spark", "models": [{
                "model_id": "org/model", "size_bytes": 12, "cache_path": "/private/cache",
            }],
        }])):
            response = await self.client.get("/api/v1/model-cache")

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("cache_path", response.text)
        self.assertEqual(response.json()["nodes"][0]["models"][0]["size_bytes"], 12)


if __name__ == "__main__":
    unittest.main()
