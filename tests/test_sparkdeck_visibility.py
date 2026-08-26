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


class SavedConfigurationContractTests(unittest.TestCase):
    def setUp(self):
        self.manager = Manager.__new__(Manager)

    def test_vllm_tp_and_pp_product_sets_sharded_node_count(self):
        contract = self.manager.recipe_deployment_contract({
            "engine": "vllm",
            "extra_args": ["--tensor-parallel-size=2", "-pp", "2"],
        })

        self.assertEqual(contract["deployment_mode"], "sharded")
        self.assertEqual(contract["required_node_count"], 4)
        self.assertEqual(contract["tensor_parallel_size"], 2)
        self.assertEqual(contract["pipeline_parallel_size"], 2)

    def test_sglang_tp_sets_sharded_node_count(self):
        contract = self.manager.recipe_deployment_contract({
            "engine": "sglang", "sg_tp_size": 2,
        })

        self.assertEqual(contract["deployment_mode"], "sharded")
        self.assertEqual(contract["required_node_count"], 2)
        self.assertEqual(contract["tensor_parallel_size"], 2)
        self.assertEqual(contract["pipeline_parallel_size"], 1)


class SavedConfigurationApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.assignment_patch = patch.object(
            server.onboarding.assignment, "load", return_value=None,
        )
        self.assignment_patch.start()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app), base_url="http://test",
        )

    async def asyncTearDown(self):
        await self.client.aclose()
        self.assignment_patch.stop()

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
            patch.object(server.manager, "selected_cluster_nodes", AsyncMock(return_value=[{"id": "local"}])),
            patch.object(server.manager, "model_cache_inventory", AsyncMock(return_value=[{
                "id": "local", "models": [{"model_id": "org/model", "size_bytes": 12}],
            }])),
            patch.object(server.sparkdeck, "create_deployment", AsyncMock(return_value=created)) as create,
        ):
            listed = await self.client.get("/api/v1/recipes")
            launched = await self.client.post("/api/v1/recipes/recipe-1/deploy")

        self.assertEqual(listed.json()["items"][0]["node_ids"], ["local"])
        self.assertEqual(listed.json()["items"][0]["engine"], "vllm")
        self.assertEqual(listed.json()["items"][0]["required_node_count"], 1)
        self.assertEqual(listed.json()["items"][0]["extra_args_count"], 4)
        self.assertNotIn("secret", listed.text)
        self.assertEqual(launched.status_code, 201)
        self.assertEqual(
            create.await_args.args[0]["settings"]["extra_args"],
            ["--dtype", "auto", "--hf-token", "secret"],
        )
        self.assertEqual(create.await_args.args[0]["recipe_id"], "recipe-1")

    async def test_tp2_recipe_requires_two_nodes_with_cached_weights(self):
        recipe = {
            "id": "recipe-tp2", "name": "DeepSeek TP2", "model": "deepseek/model",
            "engine": "vllm",
            "node_ids": ["local", "node-2"],
            "extra_args": ["--tensor-parallel-size", "2"],
        }
        created = {
            "id": "dep-tp2", "alias": "DeepSeek TP2", "runtime": "vllm",
            "kind": "managed", "model": {"repository": "deepseek/model"},
            "status": "starting", "settings": {},
        }
        inventory = [
            {"id": "local", "models": [{"model_id": "deepseek/model", "size_bytes": 12}]},
            {"id": "node-2", "models": [{"model_id": "deepseek/model", "size_bytes": 12}]},
            {"id": "node-3", "models": []},
        ]
        with (
            patch.object(server.manager, "recipes", [recipe]),
            patch.object(server.manager, "recipe_launches", {}),
            patch.object(server.manager, "get_recipe", AsyncMock(return_value=recipe)),
            patch.object(server.manager, "selected_cluster_nodes", AsyncMock(return_value=[
                {"id": "local"}, {"id": "node-2"},
            ])),
            patch.object(server.manager, "model_cache_inventory", AsyncMock(return_value=inventory)),
            patch.object(server.sparkdeck, "create_deployment", AsyncMock(return_value=created)) as create,
        ):
            listed = await self.client.get("/api/v1/recipes")
            too_few = await self.client.post(
                "/api/v1/recipes/recipe-tp2/deploy", json={"node_ids": ["local"]},
            )
            missing_weights = await self.client.post(
                "/api/v1/recipes/recipe-tp2/deploy",
                json={"node_ids": ["local", "node-3"]},
            )
            remote_only = await self.client.post(
                "/api/v1/recipes/recipe-tp2/deploy",
                json={"node_ids": ["node-2", "node-3"]},
            )
            duplicate = await self.client.post(
                "/api/v1/recipes/recipe-tp2/deploy",
                json={"node_ids": ["local", "local"]},
            )
            launched = await self.client.post(
                "/api/v1/recipes/recipe-tp2/deploy",
                json={"node_ids": ["node-2", "local"]},
            )

        item = listed.json()["items"][0]
        self.assertEqual(item["tensor_parallel_size"], 2)
        self.assertEqual(item["required_node_count"], 2)
        self.assertEqual(item["deployment_mode"], "sharded")
        self.assertEqual(too_few.status_code, 400)
        self.assertIn("exactly 2", too_few.text)
        self.assertEqual(missing_weights.status_code, 409)
        self.assertIn("node-3", missing_weights.text)
        self.assertEqual(remote_only.status_code, 400)
        self.assertIn("controller node", remote_only.text)
        self.assertEqual(duplicate.status_code, 400)
        self.assertIn("exactly 2", duplicate.text)
        self.assertEqual(launched.status_code, 201)
        self.assertEqual(create.await_args.args[0]["node_ids"], ["local", "node-2"])
        self.assertEqual(create.await_args.args[0]["deployment_mode"], "sharded")

    async def test_node_readiness_is_validated_before_weight_inventory(self):
        recipe = {
            "id": "recipe-1", "model": "org/model", "extra_args": [],
        }
        inventory = AsyncMock(return_value=[])
        with (
            patch.object(server.manager, "get_recipe", AsyncMock(return_value=recipe)),
            patch.object(
                server.manager, "selected_cluster_nodes",
                AsyncMock(side_effect=ValueError("unknown cluster node(s): missing")),
            ),
            patch.object(server.manager, "model_cache_inventory", inventory),
        ):
            response = await self.client.post(
                "/api/v1/recipes/recipe-1/deploy", json={"node_ids": ["missing"]},
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("unknown cluster node", response.text)
        inventory.assert_not_awaited()

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
