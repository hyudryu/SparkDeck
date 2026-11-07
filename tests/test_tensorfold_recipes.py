"""TensorFold engine support for saved launch recipes.

TensorFold already launches through the deployments API (see
:mod:`tests.test_tensorfold_runtime_manager`); these tests pin the recipe
contract that brings it into the saved-configuration and MCP flows:

* ``add_recipe``/``update_recipe`` accept the tensorfold engine with its
  plain ``serve MODEL`` argv (context via the ``context_window`` control) and
  refuse fields TensorFold has no flags for;
* the deployment contract always plans one single-GPU server per selected
  node and flags sharded layouts as unlaunchable;
* persisted tensorfold recipes load as supported while the remaining
  deployments-API-only runtimes stay marked unsupported;
* the HTTP launch-settings merge accepts tensorfold, maps its context knob,
  and refuses silently-dropped memory/concurrency fields;
* the MCP client forwards a tensorfold recipe verbatim on deploy.
"""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

import httpx
from fastapi import HTTPException

import server
from manager import Manager
from mcp_server import ControllerClient


class TensorfoldRecipeTests(unittest.IsolatedAsyncioTestCase):
    def _manager(self, directory: str, recipes=None) -> Manager:
        instance = Manager.__new__(Manager)
        instance.lock = asyncio.Lock()
        instance.recipes = list(recipes or [])
        instance.recipes_path = Path(directory) / "recipes.json"
        return instance

    async def test_add_tensorfold_recipe_applies_context_control(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            instance = self._manager(directory)

            recipe = await instance.add_recipe(
                "model/a", engine="tensorfold", force_new=True,
                launch_controls={"context_window": 262144},
            )

            self.assertEqual(recipe["engine"], "tensorfold")
            self.assertEqual(recipe["deployment_mode"], "single")
            self.assertEqual(recipe["extra_args"], ["--context", "262144"])
            contract = instance.recipe_deployment_contract(recipe)
            self.assertTrue(contract["supported"])
            self.assertEqual(contract["deployment_mode"], "single")
            self.assertEqual(contract["required_node_count"], 1)
            self.assertEqual(contract["tensor_parallel_size"], 1)
            self.assertEqual(contract["pipeline_parallel_size"], 1)

    async def test_add_tensorfold_recipe_rejects_engine_specific_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            instance = self._manager(directory)

            with self.assertRaisesRegex(ValueError, "sg_\\* fields"):
                await instance.add_recipe(
                    "model/a", engine="tensorfold", force_new=True,
                    sg_context_length=8192,
                )
            with self.assertRaisesRegex(ValueError, "gpu_memory_utilization"):
                await instance.add_recipe(
                    "model/a", engine="tensorfold", force_new=True,
                    gpu_memory_utilization=0.9,
                )
            with self.assertRaisesRegex(
                ValueError, "engine must be vllm, sglang, or tensorfold",
            ):
                await instance.add_recipe("model/a", engine="laya", force_new=True)

    async def test_add_tensorfold_recipe_rejects_sharded_layouts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            instance = self._manager(directory)

            for mode in ("sharded", "grouped_sharded"):
                with self.subTest(mode=mode):
                    with self.assertRaisesRegex(
                        ValueError, "single or replicated deployment modes",
                    ):
                        await instance.add_recipe(
                            "model/a", engine="tensorfold", force_new=True,
                            deployment_mode=mode,
                            node_ids=["node-a", "node-b"],
                        )

    async def test_update_tensorfold_recipe_rejects_unsupported_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            instance = self._manager(directory, [{
                "id": "tf-1", "name": "TF", "model": "model/a",
                "engine": "tensorfold", "deployment_mode": "single",
                "node_ids": ["local"], "extra_args": [],
            }])

            with self.assertRaisesRegex(ValueError, "gpu_memory_utilization"):
                await instance.update_recipe(
                    "tf-1", {"gpu_memory_utilization": 0.9},
                )
            with self.assertRaisesRegex(ValueError, "single or replicated"):
                await instance.update_recipe(
                    "tf-1", {"deployment_mode": "sharded"},
                )
            updated = await instance.update_recipe(
                "tf-1", {"extra_args": ["--top-k", "20"]},
            )
            self.assertEqual(updated["extra_args"], ["--top-k", "20"])
            self.assertEqual(updated["engine"], "tensorfold")

    def test_tensorfold_contract_ignores_parallel_flags(self) -> None:
        instance = Manager.__new__(Manager)
        contract = instance.recipe_deployment_contract({
            "engine": "tensorfold", "deployment_mode": "single",
            "extra_args": ["--context", "8192", "-tp", "4"],
        })
        self.assertEqual(contract["tensor_parallel_size"], 1)
        self.assertEqual(contract["pipeline_parallel_size"], 1)
        self.assertEqual(contract["required_node_count"], 1)

    def test_tensorfold_contract_flags_sharded_layouts_unsupported(self) -> None:
        instance = Manager.__new__(Manager)
        for mode in ("sharded", "grouped_sharded"):
            with self.subTest(mode=mode):
                contract = instance.recipe_deployment_contract({
                    "engine": "tensorfold", "deployment_mode": mode,
                    "node_ids": ["node-a", "node-b"], "extra_args": [],
                })
                self.assertFalse(contract["supported"])
                self.assertIn(
                    "single or replicated deployment modes", str(contract["error"]),
                )

    def test_load_recipes_keeps_tensorfold_supported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            instance = self._manager(directory)
            instance.recipes_path.write_text(json.dumps([
                {"id": "tf-1", "model": "model/a", "engine": "tensorfold"},
                {"id": "laya-1", "model": "model/b", "engine": "laya"},
            ]))

            loaded = instance._load_recipes()
            by_id = {recipe["id"]: recipe for recipe in loaded}

            self.assertNotIn("supported", by_id["tf-1"])
            self.assertNotIn("error", by_id["tf-1"])
            self.assertFalse(by_id["laya-1"]["supported"])
            self.assertEqual(
                by_id["laya-1"]["error"], "unsupported persisted runtime: laya",
            )


class TensorfoldRecipeMergeTests(unittest.TestCase):
    def test_merge_accepts_tensorfold_context(self) -> None:
        payload = server._merge_recipe_launch_settings({
            "model": "model/a",
            "engine": "tensorfold",
            "launch_settings": {
                "context_length": 131072,
                "environment": {"TF_MEMORY_RESERVE_GIB": "12"},
            },
        })

        self.assertEqual(
            payload["launch_controls"], {"context_window": 131072},
        )
        self.assertEqual(
            payload["environment"], {"TF_MEMORY_RESERVE_GIB": "12"},
        )

    def test_merge_rejects_tensorfold_memory_and_concurrency(self) -> None:
        for field, value in (
            ("gpu_memory_utilization", 0.9),
            ("max_running_requests", 8),
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(
                    HTTPException, f"{field} not supported for tensorfold",
                ):
                    server._merge_recipe_launch_settings({
                        "model": "model/a",
                        "engine": "tensorfold",
                        "launch_settings": {field: value},
                    })

    def test_merge_rejects_unknown_engine(self) -> None:
        with self.assertRaisesRegex(
            HTTPException, "engine must be vllm, sglang, or tensorfold",
        ):
            server._merge_recipe_launch_settings({
                "model": "model/a",
                "engine": "ninfer",
                "launch_settings": {"context_length": 8192},
            })


class TensorfoldRecipeClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_deploy_recipe_forwards_tensorfold_engine(self) -> None:
        requests = []

        async def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            if request.method == "GET":
                return httpx.Response(200, json={"recipes": [{
                    "id": "tf-1", "name": "TF", "model": "model/a",
                    "engine": "tensorfold", "extra_args": ["--context", "8192"],
                    "deployment_mode": "single", "node_ids": ["node-a"],
                }]})
            body = json.loads(request.content)
            return httpx.Response(200, json={"id": "deployed-1", **body})

        client = ControllerClient(transport=httpx.MockTransport(handler))
        result = await client.deploy_recipe("tf-1", deployment_name="TF Ablit")

        self.assertEqual(result["engine"], "tensorfold")
        deployed = json.loads(requests[-1].content)
        self.assertEqual(deployed["engine"], "tensorfold")
        self.assertEqual(deployed["deployment_name"], "TF Ablit")
        self.assertEqual(deployed["deployment_mode"], "single")
        self.assertEqual(
            [request.method for request in requests], ["GET", "POST"],
        )


if __name__ == "__main__":
    unittest.main()
