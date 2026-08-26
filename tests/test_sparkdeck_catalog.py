import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

import httpx

from sparkdeck.catalog import HuggingFaceCatalog
from sparkdeck.models import Deployment, DeploymentKind, ModelIdentity, RuntimeKind
from sparkdeck.service import SparkDeckService


class HuggingFaceCatalogTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_forwards_query_limit_sort_and_private_token(self):
        captured = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(200, json=[{
                "id": "org/Model",
                "author": "org",
                "downloads": "42",
                "likes": -5,
                "tags": ["Transformers", "safetensors", 123],
                "pipeline_tag": "text-generation",
                "lastModified": "2026-01-01T00:00:00Z",
                "private": False,
                "gated": "auto",
                "safetensors": {
                    "total": 1_500,
                    "parameters": {"BF16": 1_000, "F32": 500},
                },
            }, {
                "id": "org/private-model", "private": True,
            }])

        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        catalog = HuggingFaceCatalog(http, token_provider=lambda: "hf_private_secret")

        items = await catalog.search("  qwen coder  ", 999)

        request = captured[0]
        self.assertEqual(str(request.url.copy_with(query=None)), "https://huggingface.co/api/models")
        self.assertEqual(request.url.params["search"], "qwen coder")
        self.assertEqual(request.url.params["limit"], "100")
        self.assertEqual(request.url.params["sort"], "downloads")
        self.assertEqual(request.url.params["direction"], "-1")
        self.assertEqual(
            set(request.url.params.get_list("expand[]")),
            {
                "author", "downloads", "likes", "tags", "safetensors", "gguf",
                "pipeline_tag", "gated", "private", "lastModified",
            },
        )
        self.assertEqual(request.headers["authorization"], "Bearer hf_private_secret")
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["id"], "org/Model")
        self.assertEqual(items[0]["downloads"], 42)
        self.assertEqual(items[0]["likes"], 0)
        self.assertEqual(items[0]["parameter_count"], 1_500)
        self.assertEqual(items[0]["weight_size_bytes"], 4_000)
        self.assertEqual(items[0]["weight_size_source"], "safetensors")
        self.assertEqual(items[0]["tags"], ["Transformers", "safetensors"])
        self.assertTrue(next(
            item["supported"] for item in items[0]["runtime_compatibility"]
            if item["runtime"] == "vllm"
        ))
        self.assertNotIn("hf_private_secret", str(items))
        await http.aclose()

    async def test_gguf_metadata_exposes_parameter_and_weight_size(self):
        item = HuggingFaceCatalog._public_item({
            "id": "org/model-GGUF", "tags": ["gguf"],
            "gguf": {"total": 30_532_122_624, "totalFileSize": 17_310_784_672},
        })

        self.assertEqual(item["parameter_count"], 30_532_122_624)
        self.assertEqual(item["weight_size_bytes"], 17_310_784_672)
        self.assertEqual(item["weight_size_source"], "gguf")

    async def test_public_search_omits_authorization(self):
        captured = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(200, json=[])

        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        await HuggingFaceCatalog(http).search("model", 5)

        self.assertNotIn("authorization", captured[0].headers)
        await http.aclose()


class CatalogFallbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_matching_local_models_are_reserved_inside_catalog_limit(self):
        class Manager:
            def __init__(self):
                self.http = httpx.AsyncClient()
                self.list_containers = AsyncMock(return_value=[])

        with tempfile.TemporaryDirectory() as directory:
            manager = Manager()
            service = SparkDeckService(manager, Path(directory))
            service.store.add_deployment(Deployment(
                id="local-1", alias="Local model", runtime=RuntimeKind.VLLM,
                kind=DeploymentKind.MANAGED, model=ModelIdentity("local/Model"),
                container_name="sparkdeck-local",
            ))
            service.catalog.search = AsyncMock(return_value=[
                {"id": "remote/One", "runtime_compatibility": []},
                {"id": "remote/Two", "runtime_compatibility": []},
            ])

            result = await service.catalog_search("", 2)

            self.assertEqual(
                [item["id"] for item in result["items"]],
                ["local/Model", "remote/One"],
            )
            self.assertEqual(result["items"][0]["local_deployment_ids"], ["local-1"])
            await service.close()
            await manager.http.aclose()

    async def test_local_enrichment_does_not_mutate_cached_remote_results(self):
        class Manager:
            def __init__(self):
                self.http = httpx.AsyncClient()
                self.list_containers = AsyncMock(return_value=[])

        cached_item = {
            "id": "org/Model",
            "runtime_compatibility": [{"runtime": "vllm", "supported": False}],
        }
        with tempfile.TemporaryDirectory() as directory:
            manager = Manager()
            service = SparkDeckService(manager, Path(directory))
            service.store.add_deployment(Deployment(
                id="local-1", alias="Local model", runtime=RuntimeKind.VLLM,
                kind=DeploymentKind.MANAGED, model=ModelIdentity("org/Model"),
                container_name="sparkdeck-local",
            ))
            service.catalog.search = AsyncMock(return_value=[cached_item])

            enriched = await service.catalog_search("model", 10)
            service.store.delete_deployment("local-1")
            refreshed = await service.catalog_search("model", 10)

            self.assertEqual(enriched["items"][0]["local_deployment_ids"], ["local-1"])
            self.assertTrue(enriched["items"][0]["runtime_compatibility"][0]["supported"])
            self.assertNotIn("local_deployment_ids", refreshed["items"][0])
            self.assertFalse(refreshed["items"][0]["runtime_compatibility"][0]["supported"])
            self.assertEqual(cached_item, {
                "id": "org/Model",
                "runtime_compatibility": [{"runtime": "vllm", "supported": False}],
            })
            await service.close()
            await manager.http.aclose()

    async def test_hugging_face_outage_keeps_matching_local_models_searchable(self):
        class Manager:
            def __init__(self):
                self.http = httpx.AsyncClient()
                self.list_containers = AsyncMock(return_value=[])

        with tempfile.TemporaryDirectory() as directory:
            manager = Manager()
            service = SparkDeckService(manager, Path(directory))
            service.store.add_deployment(Deployment(
                id="local-1", alias="Local Qwen", runtime=RuntimeKind.VLLM,
                kind=DeploymentKind.MANAGED, model=ModelIdentity("org/Qwen-Local"),
                container_name="sparkdeck-local",
            ))
            service.catalog.search = AsyncMock(
                side_effect=httpx.ConnectError("Hugging Face unavailable")
            )

            result = await service.catalog_search("qwen", 24, "vllm")

            self.assertEqual([item["id"] for item in result["items"]], ["org/Qwen-Local"])
            self.assertEqual(result["items"][0]["local_deployment_ids"], ["local-1"])
            await service.close()
            await manager.http.aclose()


if __name__ == "__main__":
    unittest.main()
