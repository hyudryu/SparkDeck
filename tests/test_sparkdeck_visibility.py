import unittest
from unittest.mock import AsyncMock, Mock, patch

import httpx

from manager import Manager


PINNED_A = "a" * 40
PINNED_B = "b" * 40


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

    async def test_inventory_prefers_cache_mount_free_space_and_null_when_missing(self):
        manager = Manager.__new__(Manager)
        manager.cluster_nodes = AsyncMock(return_value=[
            {"id": "local", "name": "Spark One", "online": True, "disk": {"total": 100, "free": 7}},
            {"id": "node-2", "name": "Spark Two", "online": True, "disk": {"total": 200, "free": 9}},
            {"id": "node-3", "name": "Spark Three", "online": True},
            {"id": "node-4", "name": "Spark Four", "online": False, "disk": {"total": 300, "free": 0}},
        ])
        manager.virtual_nas = Mock()
        manager.virtual_nas.inventory.return_value = [{"model_id": "org/model", "size_bytes": 10}]
        # Zero free bytes is a genuinely full cache mount, not missing data.
        manager.virtual_nas.free_bytes.return_value = 0
        manager.node_registry = Mock()
        manager.node_registry.request = AsyncMock(side_effect=[
            {"models": [{"model_id": "org/model", "size_bytes": 11}], "free_size": 5678},
            {"models": []},
        ])

        nodes = await manager.model_cache_inventory()

        self.assertEqual(nodes[0]["free_size"], 0)
        self.assertEqual(nodes[0]["total_size"], 100)
        self.assertEqual(nodes[1]["free_size"], 5678)
        self.assertEqual(nodes[1]["total_size"], 200)
        self.assertIsNone(nodes[2]["total_size"])
        self.assertIsNone(nodes[2]["free_size"])
        self.assertEqual(nodes[3]["free_size"], 0)
        self.assertEqual(nodes[3]["total_size"], 300)


class DeploymentStartNodeSelectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_start_with_node_selection_relaunches_on_chosen_nodes(self):
        manager = Manager.__new__(Manager)
        manager.deployments = [{
            "id": "cluster-1", "status": "stopped", "engine": "vllm",
            "members": [{"node_id": "local", "container_name": "old-r0", "rank": 0}],
            "launch_settings": {
                "model": "org/model", "node_ids": ["local"],
                "deployment_mode": "sharded", "engine": "vllm",
                "extra_args": ["--tensor-parallel-size", "2"],
            },
        }]
        manager._member_action = AsyncMock(return_value={"ok": True})
        manager._preflight_deployment_launch = AsyncMock(return_value={})
        manager.create_deployment = AsyncMock(return_value={
            "id": "cluster-2", "members": [], "node_ids": ["a", "b"], "api_port": 8000,
        })
        manager._save_deployments = Mock()

        result = await manager.deployment_action("cluster-1", "start", node_ids=["a", "b"])

        self.assertTrue(result["ok"])
        self.assertEqual(result.get("deployment", {}).get("id"), "cluster-2")
        # The old rank container is removed and the saved launch settings are
        # relaunched with the explicitly chosen nodes.
        manager._member_action.assert_awaited_once()
        body = manager.create_deployment.await_args.args[0]
        self.assertEqual(body["node_ids"], ["a", "b"])
        self.assertEqual(body["model"], "org/model")
        self.assertNotIn("cluster-1", [item.get("id") for item in manager.deployments])

    async def test_start_without_node_selection_keeps_existing_ranks(self):
        manager = Manager.__new__(Manager)
        manager.deployments = [{
            "id": "cluster-1", "status": "stopped", "engine": "vllm",
            "members": [{"node_id": "local", "container_name": "old-r0", "rank": 0}],
            "launch_settings": {"model": "org/model", "node_ids": ["local"]},
        }]
        manager._member_action = AsyncMock(return_value={"ok": True})
        manager.create_deployment = AsyncMock()
        manager._save_deployments = Mock()

        result = await manager.deployment_action("cluster-1", "start")

        self.assertTrue(result["ok"])
        manager._member_action.assert_awaited_once_with(
            {"node_id": "local", "container_name": "old-r0", "rank": 0}, "start",
        )
        manager.create_deployment.assert_not_awaited()


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

    def test_explicit_single_mode_keeps_multi_gpu_parallelism_on_one_node(self):
        contract = self.manager.recipe_deployment_contract({
            "engine": "vllm", "deployment_mode": "single",
            "extra_args": ["--tensor-parallel-size", "4"],
        })

        self.assertEqual(contract["deployment_mode"], "single")
        self.assertEqual(contract["required_node_count"], 1)
        self.assertEqual(contract["tensor_parallel_size"], 4)

    def test_malformed_sglang_parallelism_does_not_hide_saved_recipes(self):
        contract = self.manager.recipe_deployment_contract({
            "engine": "sglang", "sg_tp_size": "not-a-number",
            "extra_args": ["--tp-size", "2"],
        })

        self.assertEqual(contract["deployment_mode"], "sharded")
        self.assertEqual(contract["required_node_count"], 2)
        self.assertEqual(contract["tensor_parallel_size"], 2)

    def test_unknown_persisted_mode_is_not_rewritten_to_single(self):
        contract = self.manager.recipe_deployment_contract({
            "engine": "vllm", "deployment_mode": "shardded", "extra_args": [],
        })

        self.assertEqual(contract["deployment_mode"], "shardded")
        self.assertFalse(contract["supported"])
        self.assertIn("unsupported persisted deployment mode", contract["error"])

    def test_unhashable_saved_node_id_marks_only_that_recipe_unsupported(self):
        contract = self.manager.recipe_deployment_contract({
            "engine": "vllm", "deployment_mode": "replicated",
            "node_ids": ["local", {"id": "node-2"}, ["node-3"]],
            "extra_args": [],
        })

        self.assertEqual(contract["deployment_mode"], "replicated")
        self.assertEqual(contract["required_node_count"], 2)
        self.assertFalse(contract["supported"])
        self.assertIn("node_ids", contract["error"])

    def test_malformed_saved_extra_args_marks_only_that_recipe_unsupported(self):
        contract = self.manager.recipe_deployment_contract({
            "engine": "vllm", "extra_args": True,
        })

        self.assertFalse(contract["supported"])
        self.assertIn("extra_args", contract["error"])


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

    async def test_deployment_start_route_forwards_node_selection(self):
        with patch.object(server.sparkdeck, "deployment_action", AsyncMock(return_value={"ok": True})) as action:
            without_body = await self.client.post("/api/v1/deployments/dep-1/start")
            with_nodes = await self.client.post(
                "/api/v1/deployments/dep-1/start",
                json={"node_ids": ["local", "node-2"]},
            )
            with_additional = await self.client.post(
                "/api/v1/deployments/dep-1/start",
                json={"additional_node_ids": ["node-3"]},
            )
            with_invalid = await self.client.post(
                "/api/v1/deployments/dep-1/start",
                json={"node_ids": []},
            )
            with_invalid_additional = await self.client.post(
                "/api/v1/deployments/dep-1/start",
                json={"additional_node_ids": [" "]},
            )
            invalid_utf8 = await self.client.post(
                "/api/v1/deployments/dep-1/start",
                content=b"\xff\xfe{}",
                headers={"content-type": "application/json"},
            )

        self.assertEqual(without_body.status_code, 200)
        self.assertEqual(with_nodes.status_code, 200)
        self.assertEqual(with_additional.status_code, 200)
        self.assertEqual(with_invalid.status_code, 400)
        self.assertEqual(with_invalid_additional.status_code, 400)
        self.assertEqual(invalid_utf8.status_code, 400)
        action.assert_any_await("dep-1", "start", None, None)
        action.assert_any_await("dep-1", "start", ["local", "node-2"], None)
        action.assert_any_await("dep-1", "start", None, ["node-3"])

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
            patch.object(server.manager, "get_state", AsyncMock(return_value={})),
            patch.object(server.manager, "selected_cluster_nodes", AsyncMock(return_value=[{"id": "local"}])),
            patch.object(server.manager, "model_cache_inventory", AsyncMock(return_value=[{
                "id": "local", "models": [{
                    "model_id": "org/model", "size_bytes": 12,
                    "revisions": ["main", PINNED_A],
                    "revision_refs": {"main": PINNED_A},
                }],
            }])),
            patch.object(
                server.manager.virtual_nas, "resolve_download_revision",
                AsyncMock(return_value={
                    "requested_revision": "main",
                    "resolved_revision": PINNED_A, "size_bytes": 12,
                }),
            ),
            patch.object(server.sparkdeck, "create_deployment", AsyncMock(return_value=created)) as create,
        ):
            listed = await self.client.get("/api/v1/recipes")
            state = await self.client.get("/api/state")
            launched = await self.client.post("/api/v1/recipes/recipe-1/deploy")

        self.assertEqual(listed.json()["items"][0]["node_ids"], ["local"])
        self.assertEqual(listed.json()["items"][0]["engine"], "vllm")
        self.assertEqual(listed.json()["items"][0]["required_node_count"], 1)
        self.assertEqual(listed.json()["items"][0]["extra_args_count"], 2)
        self.assertNotIn("secret", listed.text)
        self.assertNotIn("secret", state.text)
        self.assertEqual(state.json()["recipes"][0]["extra_args"], ["--dtype", "auto"])
        self.assertEqual(launched.status_code, 201)
        self.assertEqual(
            create.await_args.args[0]["settings"]["extra_args"],
            ["--dtype", "auto"],
        )
        self.assertEqual(create.await_args.args[0]["recipe_id"], "recipe-1")

    async def test_recipe_deploy_rejects_malformed_nonempty_json(self):
        recipe = {
            "id": "recipe-1",
            "model": "org/model",
            "engine": "vllm",
            "extra_args": [],
        }
        selected = AsyncMock()
        inventory = AsyncMock()
        create = AsyncMock()
        with (
            patch.object(
                server.manager, "get_recipe", AsyncMock(return_value=recipe)
            ),
            patch.object(server.manager, "selected_cluster_nodes", selected),
            patch.object(server.manager, "model_cache_inventory", inventory),
            patch.object(server.sparkdeck, "create_deployment", create),
        ):
            response = await self.client.post(
                "/api/v1/recipes/recipe-1/deploy",
                content=b'{"node_ids":["local"]',
                headers={"content-type": "application/json"},
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("valid JSON", response.text)
        selected.assert_not_awaited()
        inventory.assert_not_awaited()
        create.assert_not_awaited()

    async def test_malformed_parallelism_does_not_hide_other_saved_configurations(self):
        recipes = [
            {"id": "bad-tp", "model": "org/bad", "engine": "sglang", "sg_tp_size": "bad"},
            {"id": "valid", "model": "org/valid", "engine": "vllm", "extra_args": []},
        ]
        with (
            patch.object(server.manager, "recipes", recipes),
            patch.object(server.manager, "recipe_launches", {}),
        ):
            response = await self.client.get("/api/v1/recipes")

        self.assertEqual(response.status_code, 200)
        self.assertEqual([item["id"] for item in response.json()["items"]], ["bad-tp", "valid"])
        self.assertEqual(response.json()["items"][0]["tensor_parallel_size"], 1)

    async def test_unhashable_node_id_does_not_hide_other_saved_configurations(self):
        recipes = [
            {
                "id": "bad-nodes", "model": "org/bad", "engine": "vllm",
                "deployment_mode": "replicated",
                "node_ids": ["local", {"id": "node-2"}, ["node-3"]],
            },
            {"id": "valid", "model": "org/valid", "engine": "vllm", "extra_args": []},
        ]
        with (
            patch.object(server.manager, "recipes", recipes),
            patch.object(server.manager, "recipe_launches", {}),
        ):
            response = await self.client.get("/api/v1/recipes")

        self.assertEqual(response.status_code, 200)
        self.assertEqual([item["id"] for item in response.json()["items"]], ["bad-nodes", "valid"])
        self.assertFalse(response.json()["items"][0]["supported"])
        self.assertIn("node_ids", response.json()["items"][0]["error"])
        self.assertTrue(response.json()["items"][1]["supported"])

    async def test_unknown_deployment_mode_is_listed_as_unsupported_and_rejected(self):
        recipe = {
            "id": "bad-mode", "model": "org/model", "engine": "vllm",
            "deployment_mode": "shardded", "extra_args": [],
        }
        valid = {"id": "valid", "model": "org/valid", "engine": "vllm", "extra_args": []}
        create = AsyncMock()
        selected = AsyncMock()
        inventory = AsyncMock()
        with (
            patch.object(server.manager, "recipes", [recipe, valid]),
            patch.object(server.manager, "recipe_launches", {}),
            patch.object(server.manager, "get_recipe", AsyncMock(return_value=recipe)),
            patch.object(server.manager, "selected_cluster_nodes", selected),
            patch.object(server.manager, "model_cache_inventory", inventory),
            patch.object(server.sparkdeck, "create_deployment", create),
        ):
            listed = await self.client.get("/api/v1/recipes")
            launched = await self.client.post("/api/v1/recipes/bad-mode/deploy")

        item = listed.json()["items"][0]
        self.assertEqual([entry["id"] for entry in listed.json()["items"]], ["bad-mode", "valid"])
        self.assertEqual(item["deployment_mode"], "shardded")
        self.assertFalse(item["supported"])
        self.assertIn("unsupported persisted deployment mode", item["error"])
        self.assertTrue(listed.json()["items"][1]["supported"])
        self.assertEqual(launched.status_code, 400)
        selected.assert_not_awaited()
        inventory.assert_not_awaited()
        create.assert_not_awaited()

    async def test_recipe_creation_rejects_unknown_deployment_mode(self):
        response = await self.client.post("/api/recipes", json={
            "model": "org/model", "deployment_mode": "shardded",
        })

        self.assertEqual(response.status_code, 400)
        self.assertIn("deployment_mode must be", response.text)

    async def test_recipe_creation_rejects_malformed_node_ids_before_persisting(self):
        invalid_values = [
            [{"id": "node-2"}], [1], [""], ["   "], "local", {"id": "local"}, [],
        ]
        for node_ids in invalid_values:
            with (
                self.subTest(node_ids=node_ids),
                patch.object(server.manager, "recipes", []),
                patch.object(server.manager, "_save_recipes") as save,
            ):
                response = await self.client.post("/api/recipes", json={
                    "model": "org/model", "node_ids": node_ids,
                })

                self.assertEqual(response.status_code, 400)
                self.assertIn("node_ids", response.text)
                self.assertEqual(server.manager.recipes, [])
                save.assert_not_called()

    async def test_recipe_update_rejects_malformed_node_ids_before_persisting(self):
        original = {
            "id": "recipe-1", "model": "org/model", "engine": "vllm",
            "deployment_mode": "single", "node_ids": ["local"], "extra_args": [],
        }
        for node_ids in ([{"id": "node-2"}], [2], [" "]):
            recipe = dict(original)
            with (
                self.subTest(node_ids=node_ids),
                patch.object(server.manager, "recipes", [recipe]),
                patch.object(server.manager, "_save_recipes") as save,
            ):
                response = await self.client.put("/api/recipes/recipe-1", json={
                    "node_ids": node_ids,
                })

                self.assertEqual(response.status_code, 400)
                self.assertIn("node_ids", response.text)
                self.assertEqual(recipe, original)
                save.assert_not_called()

    async def test_valid_args_repair_only_the_persisted_args_error(self):
        marker = "unsupported persisted extra_args: expected an array of strings"
        cases = [
            ({"supported": False, "error": marker}, True, None),
            (
                {"supported": False, "error": f"manual unsupported reason; {marker}"},
                False,
                "manual unsupported reason",
            ),
        ]
        for flags, becomes_supported, expected_error in cases:
            recipe = {
                "id": "recipe-1", "model": "org/model", "engine": "vllm",
                "deployment_mode": "single", "node_ids": ["local"],
                "extra_args": [], **flags,
            }
            with (
                self.subTest(flags=flags),
                patch.object(server.manager, "recipes", [recipe]),
                patch.object(server.manager, "recipe_launches", {}),
                patch.object(server.manager, "_save_recipes") as save,
            ):
                updated = await self.client.put(
                    "/api/recipes/recipe-1",
                    json={"extra_args": ["--dtype", "auto"]},
                )
                listed = await self.client.get("/api/v1/recipes")

            self.assertEqual(updated.status_code, 200)
            item = listed.json()["items"][0]
            self.assertEqual(item["supported"], becomes_supported)
            self.assertEqual(item["error"], expected_error)
            self.assertNotIn(marker, str(updated.json().get("error") or ""))
            save.assert_called_once()

    async def test_revision_pinned_recipe_requires_the_exact_cached_revision(self):
        recipe = {
            "id": "revision-b", "model": "org/model", "engine": "vllm",
            "extra_args": ["--revision=release-b"],
        }
        created = {
            "id": "dep-revision", "alias": "org/model", "runtime": "vllm",
            "kind": "managed", "model": {"repository": "org/model"},
            "status": "starting", "settings": {},
        }
        inventory = AsyncMock(side_effect=[[{
            "id": "local", "models": [{
                "model_id": "org/model", "size_bytes": 12,
                "revisions": ["release-a", "revision-a"],
            }],
        }], [{
            "id": "local", "models": [{
                "model_id": "org/model", "size_bytes": 12,
                "revisions": ["release-b", PINNED_B],
                "revision_refs": {"release-b": PINNED_B},
            }],
        }]])
        with (
            patch.object(server.manager, "get_recipe", AsyncMock(return_value=recipe)),
            patch.object(server.manager, "selected_cluster_nodes", AsyncMock(return_value=[{"id": "local"}])),
            patch.object(server.manager, "model_cache_inventory", inventory),
            patch.object(
                server.manager.virtual_nas, "resolve_download_revision",
                AsyncMock(return_value={
                    "requested_revision": "release-b",
                    "resolved_revision": PINNED_B, "size_bytes": 12,
                }),
            ),
            patch.object(
                server.sparkdeck, "create_deployment", AsyncMock(return_value=created),
            ) as create,
        ):
            mismatch = await self.client.post("/api/v1/recipes/revision-b/deploy")
            matched = await self.client.post("/api/v1/recipes/revision-b/deploy")

        self.assertEqual(mismatch.status_code, 409)
        self.assertIn("local", mismatch.text)
        self.assertEqual(matched.status_code, 201)
        create.assert_awaited_once()
        self.assertEqual(
            create.await_args.args[0]["settings"]["extra_args"],
            ["--revision=release-b"],
        )

    async def test_recipe_without_revision_requires_cached_main_ref(self):
        recipe = {
            "id": "default-revision", "model": "org/model", "engine": "vllm",
            "extra_args": [],
        }
        created = {
            "id": "dep-main", "alias": "org/model", "runtime": "vllm",
            "kind": "managed", "model": {"repository": "org/model"},
            "status": "starting", "settings": {},
        }
        inventory = AsyncMock(side_effect=[[{
            "id": "local", "models": [{
                "model_id": "org/model", "size_bytes": 12,
                "revisions": ["unrelated-snapshot"],
            }],
        }], [{
            "id": "local", "models": [{
                "model_id": "org/model", "size_bytes": 12,
                "revisions": ["main", PINNED_A],
                "revision_refs": {"main": PINNED_A},
            }],
        }]])
        with (
            patch.object(server.manager, "get_recipe", AsyncMock(return_value=recipe)),
            patch.object(server.manager, "selected_cluster_nodes", AsyncMock(return_value=[{"id": "local"}])),
            patch.object(server.manager, "model_cache_inventory", inventory),
            patch.object(
                server.manager.virtual_nas, "resolve_download_revision",
                AsyncMock(return_value={
                    "requested_revision": "main",
                    "resolved_revision": PINNED_A, "size_bytes": 12,
                }),
            ),
            patch.object(
                server.sparkdeck, "create_deployment", AsyncMock(return_value=created),
            ) as create,
        ):
            missing_main = await self.client.post(
                "/api/v1/recipes/default-revision/deploy"
            )
            cached_main = await self.client.post(
                "/api/v1/recipes/default-revision/deploy"
            )

        self.assertEqual(missing_main.status_code, 409)
        self.assertIn("local", missing_main.text)
        self.assertEqual(cached_main.status_code, 201)
        create.assert_awaited_once()

    async def test_recipe_rejects_partial_cache_even_if_it_advertises_main(self):
        recipe = {
            "id": "partial-cache", "model": "org/model", "engine": "vllm",
            "extra_args": [],
        }
        inventory = [{
            "id": "local", "models": [{
                "model_id": "org/model", "size_bytes": 12,
                "partial": True, "revisions": ["main"],
            }],
        }]
        with (
            patch.object(server.manager, "get_recipe", AsyncMock(return_value=recipe)),
            patch.object(server.manager, "selected_cluster_nodes", AsyncMock(return_value=[{"id": "local"}])),
            patch.object(server.manager, "model_cache_inventory", AsyncMock(return_value=inventory)),
            patch.object(
                server.manager.virtual_nas, "resolve_download_revision",
                AsyncMock(return_value={
                    "requested_revision": "main",
                    "resolved_revision": PINNED_A, "size_bytes": 12,
                }),
            ),
            patch.object(server.sparkdeck, "create_deployment", AsyncMock()) as create,
        ):
            response = await self.client.post("/api/v1/recipes/partial-cache/deploy")

        self.assertEqual(response.status_code, 409)
        self.assertIn("model weights are not available", response.text)
        create.assert_not_awaited()

    async def test_valid_local_path_recipe_skips_hub_cache_preflight(self):
        recipe = {
            "id": "local-path", "model": "/models/local", "engine": "vllm",
            "extra_args": [],
        }
        created = {
            "id": "dep-local", "alias": "/models/local", "runtime": "vllm",
            "kind": "managed", "model": {"repository": "/models/local"},
            "status": "starting", "settings": {},
        }
        inventory = AsyncMock()
        with (
            patch.object(server.manager, "get_recipe", AsyncMock(return_value=recipe)),
            patch.object(server.manager, "selected_cluster_nodes", AsyncMock(return_value=[{"id": "local"}])),
            patch.object(server.manager, "_resolve_local_path", return_value="/models/local"),
            patch.object(server.manager, "model_cache_inventory", inventory),
            patch.object(
                server.sparkdeck, "create_deployment", AsyncMock(return_value=created),
            ) as create,
        ):
            response = await self.client.post("/api/v1/recipes/local-path/deploy")

        self.assertEqual(response.status_code, 201)
        inventory.assert_not_awaited()
        create.assert_awaited_once()

    async def test_recipe_deploy_rejects_nodes_with_mismatched_main_commits(self):
        recipe = {
            "id": "mixed-main", "model": "org/model", "engine": "vllm",
            "extra_args": ["--tensor-parallel-size", "2"],
        }
        inventory = [{
            "id": "local", "models": [{
                "model_id": "org/model", "partial": False,
                "size_bytes": 12, "revisions": ["main", PINNED_A],
                "revision_refs": {"main": PINNED_A},
            }],
        }, {
            "id": "node-2", "models": [{
                "model_id": "org/model", "partial": False,
                "size_bytes": 12, "revisions": ["main", PINNED_B],
                "revision_refs": {"main": PINNED_B},
            }],
        }]
        with (
            patch.object(server.manager, "get_recipe", AsyncMock(return_value=recipe)),
            patch.object(
                server.manager, "selected_cluster_nodes",
                AsyncMock(return_value=[{"id": "local"}, {"id": "node-2"}]),
            ),
            patch.object(
                server.manager, "model_cache_inventory",
                AsyncMock(return_value=inventory),
            ),
            patch.object(
                server.manager.virtual_nas, "resolve_download_revision",
                AsyncMock(return_value={
                    "requested_revision": "main",
                    "resolved_revision": PINNED_A, "size_bytes": 12,
                }),
            ),
            patch.object(server.sparkdeck, "create_deployment", AsyncMock()) as create,
        ):
            response = await self.client.post(
                "/api/v1/recipes/mixed-main/deploy",
                json={"node_ids": ["local", "node-2"]},
            )

        self.assertEqual(response.status_code, 409)
        self.assertIn("node-2", response.text)
        create.assert_not_awaited()

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
            {"id": "local", "models": [{
                "model_id": "deepseek/model", "size_bytes": 12,
                "revisions": ["main", PINNED_A],
                "revision_refs": {"main": PINNED_A},
            }]},
            {"id": "node-2", "models": [{
                "model_id": "deepseek/model", "size_bytes": 12,
                "revisions": ["main", PINNED_A],
                "revision_refs": {"main": PINNED_A},
            }]},
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
            patch.object(
                server.manager.virtual_nas, "resolve_download_revision",
                AsyncMock(return_value={
                    "requested_revision": "main",
                    "resolved_revision": PINNED_A, "size_bytes": 12,
                }),
            ),
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
