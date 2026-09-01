import asyncio
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
                "pipeline_tag", "gated", "private", "lastModified", "siblings",
                "sha",
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

    async def test_search_classifies_and_exposes_sibling_only_gguf_artifacts(self):
        captured = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(200, json=[{
                "id": "org/model", "tags": [],
                "siblings": [
                    {"rfilename": "model-Q4_K_M.gguf", "size": 8_000_000_000},
                    {"rfilename": "README.md", "size": 100},
                ],
            }])

        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        items = await HuggingFaceCatalog(http).search("model", 5)

        self.assertIn("siblings", captured[0].url.params.get_list("expand[]"))
        self.assertEqual(items[0]["formats"], ["gguf"])
        self.assertTrue(next(
            value["supported"] for value in items[0]["runtime_compatibility"]
            if value["runtime"] == "llama.cpp"
        ))
        self.assertEqual(items[0]["quantizations"], [{
            "name": "Q4_K_M",
            "files": [{
                "filename": "model-Q4_K_M.gguf", "size_bytes": 8_000_000_000,
            }],
            "weight_size_bytes": 8_000_000_000,
            "artifacts": [{
                "filename": "model-Q4_K_M.gguf",
                "files": [{
                    "filename": "model-Q4_K_M.gguf", "size_bytes": 8_000_000_000,
                }],
                "weight_size_bytes": 8_000_000_000,
                "sharded": False,
            }],
        }])
        await http.aclose()

    async def test_gguf_metadata_exposes_parameter_and_weight_size(self):
        item = HuggingFaceCatalog._public_item({
            "id": "org/model-GGUF", "tags": ["gguf"],
            "gguf": {"total": 30_532_122_624, "totalFileSize": 17_310_784_672},
        })

        self.assertEqual(item["parameter_count"], 30_532_122_624)
        self.assertEqual(item["weight_size_bytes"], 17_310_784_672)
        self.assertEqual(item["weight_size_source"], "gguf")
        self.assertTrue(next(
            value["supported"] for value in item["runtime_compatibility"]
            if value["runtime"] == "llama.cpp"
        ))

    async def test_gguf_sibling_enables_llama_without_gguf_tag(self):
        item = HuggingFaceCatalog._public_item({
            "id": "org/model", "tags": [],
            "siblings": [{"rfilename": "model-Q4_K_M.gguf"}],
        })

        self.assertTrue(next(
            value["supported"] for value in item["runtime_compatibility"]
            if value["runtime"] == "llama.cpp"
        ))

    async def test_public_search_omits_authorization(self):
        captured = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(200, json=[])

        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        await HuggingFaceCatalog(http).search("model", 5)

        self.assertNotIn("authorization", captured[0].headers)
        await http.aclose()

    async def test_details_lists_all_primary_gguf_quantizations_and_shards(self):
        captured = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(200, json={
                "id": "unsloth/Qwen3.8-27B-GGUF",
                "sha": "d" * 40,
                "tags": ["gguf"],
                "gguf": {"total": 27_000_000_000},
                "siblings": [
                    {"rfilename": "Q4_K_M/model-Q4_K_M-00001-of-00002.gguf", "size": 10},
                    {"rfilename": "Q4_K_M/model-Q4_K_M-00002-of-00002.gguf", "lfs": {"size": 12}},
                    {"rfilename": "Q8_0/model-Q8_0.gguf", "size": 30},
                    {"rfilename": "mmproj-model-f16.gguf", "size": 5},
                    {"rfilename": "README.md", "size": 9},
                ],
            })

        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        item = await HuggingFaceCatalog(http).details(
            "unsloth/Qwen3.8-27B-GGUF"
        )

        self.assertEqual(captured[0].url.path, "/api/models/unsloth/Qwen3.8-27B-GGUF")
        self.assertIn("siblings", captured[0].url.params.get_list("expand[]"))
        # The resolved commit must be requested so clients can compare
        # cached snapshot revisions against the current file listing.
        self.assertIn("sha", captured[0].url.params.get_list("expand[]"))
        self.assertEqual(item["revision"], "d" * 40)
        self.assertEqual(item["parameter_count"], 27_000_000_000)
        self.assertEqual(
            [(value["name"], value["weight_size_bytes"]) for value in item["quantizations"]],
            [("Q4_K_M", 22), ("Q8_0", 30)],
        )
        self.assertEqual(len(item["quantizations"][0]["files"]), 2)
        await http.aclose()

    async def test_details_keeps_gguf_without_quantization_marker(self):
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "id": "org/model-GGUF", "tags": ["gguf"],
                "siblings": [{"rfilename": "model.gguf", "size": 10}],
            })

        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        item = await HuggingFaceCatalog(http).details("org/model-GGUF")

        self.assertEqual(item["quantizations"], [{
            "name": "unknown",
            "files": [{"filename": "model.gguf", "size_bytes": 10}],
            "weight_size_bytes": 10,
            "artifacts": [{
                "filename": "model.gguf",
                "files": [{"filename": "model.gguf", "size_bytes": 10}],
                "weight_size_bytes": 10,
                "sharded": False,
            }],
        }])
        await http.aclose()

    async def test_details_rejects_non_repository_ids_without_network(self):
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(500)

        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        with self.assertRaisesRegex(ValueError, "owner/name"):
            await HuggingFaceCatalog(http).details("served-alias")
        self.assertEqual(calls, 0)
        await http.aclose()

    async def test_details_deduplicates_same_key_without_serializing_other_models(self):
        calls: list[str] = []
        active = 0
        maximum_active = 0
        different_models_started = asyncio.Event()

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal active, maximum_active
            calls.append(request.url.path)
            active += 1
            maximum_active = max(maximum_active, active)
            if len(set(calls)) >= 2:
                different_models_started.set()
            try:
                await asyncio.wait_for(different_models_started.wait(), timeout=1)
                return httpx.Response(200, json={
                    "id": request.url.path.removeprefix("/api/models/"),
                    "tags": ["safetensors"], "siblings": [],
                })
            finally:
                active -= 1

        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        catalog = HuggingFaceCatalog(http)

        first, duplicate, other = await asyncio.gather(
            catalog.details("org/first"),
            catalog.details("org/first"),
            catalog.details("org/second"),
        )

        self.assertEqual(first, duplicate)
        self.assertEqual(other["id"], "org/second")
        self.assertEqual(calls.count("/api/models/org/first"), 1)
        self.assertEqual(calls.count("/api/models/org/second"), 1)
        self.assertEqual(maximum_active, 2)
        self.assertEqual(catalog._detail_locks, {})

        async def fail(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(404)

        await http.aclose()
        failed_http = httpx.AsyncClient(transport=httpx.MockTransport(fail))
        failed_catalog = HuggingFaceCatalog(failed_http)
        with self.assertRaises(httpx.HTTPStatusError):
            await failed_catalog.details("org/missing")
        self.assertEqual(failed_catalog._detail_locks, {})
        await failed_http.aclose()

    async def test_details_uses_tree_sizes_and_excludes_auxiliary_ggufs(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/tree/main"):
                return httpx.Response(200, json=[
                    {"path": "model-Q4_0.gguf", "size": 16},
                    {"path": "MTP/mtp-model-Q4_0.gguf", "size": 2},
                ])
            return httpx.Response(200, json={
                "id": "org/model-GGUF", "tags": ["gguf"],
                "siblings": [
                    {"rfilename": "model-Q4_0.gguf"},
                    {"rfilename": "MTP/mtp-model-Q4_0.gguf"},
                ],
            })

        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        item = await HuggingFaceCatalog(http).details("org/model-GGUF")

        self.assertEqual(item["quantizations"], [{
            "name": "Q4_0",
            "files": [{"filename": "model-Q4_0.gguf", "size_bytes": 16}],
            "weight_size_bytes": 16,
            "artifacts": [{
                "filename": "model-Q4_0.gguf",
                "files": [{"filename": "model-Q4_0.gguf", "size_bytes": 16}],
                "weight_size_bytes": 16,
                "sharded": False,
            }],
        }])
        await http.aclose()

    async def test_details_keeps_same_quantization_alternatives_separate(self):
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "id": "org/model-GGUF", "tags": ["gguf"],
                "siblings": [
                    {"rfilename": "original/model-Q4_K_M.gguf", "size": 10},
                    {"rfilename": "imatrix/model-Q4_K_M.gguf", "size": 12},
                ],
            })

        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        item = await HuggingFaceCatalog(http).details("org/model-GGUF")

        variant = item["quantizations"][0]
        self.assertEqual(variant["weight_size_bytes"], 10)
        self.assertEqual(len(variant["artifacts"]), 2)
        self.assertEqual(
            {artifact["weight_size_bytes"] for artifact in variant["artifacts"]},
            {10, 12},
        )
        self.assertTrue(all(
            len(artifact["files"]) == 1 for artifact in variant["artifacts"]
        ))
        await http.aclose()

    async def test_details_omits_incomplete_gguf_shard_groups(self):
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "id": "org/model-GGUF", "tags": ["gguf"],
                "siblings": [{
                    "rfilename": "model-Q4_K_M-00001-of-00002.gguf", "size": 10,
                }],
            })

        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        item = await HuggingFaceCatalog(http).details("org/model-GGUF")

        self.assertEqual(item["quantizations"], [])
        await http.aclose()

    async def test_details_paginates_entire_tree_before_summing_gguf_shards(self):
        tree_requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/tree/main"):
                tree_requests.append(request)
                if request.url.params.get("cursor") == "second-page":
                    return httpx.Response(200, json=[
                        {"path": "Q4_K_M/model-Q4_K_M-00002-of-00002.gguf", "size": 12},
                    ])
                return httpx.Response(200, headers={
                    "Link": (
                        '<https://huggingface.co/api/models/org/model-GGUF/tree/main'
                        '?recursive=true&limit=1000&cursor=second-page>; rel="next"'
                    ),
                }, json=[
                    {"path": "Q4_K_M/model-Q4_K_M-00001-of-00002.gguf", "size": 10},
                ])
            return httpx.Response(200, json={
                "id": "org/model-GGUF", "tags": ["gguf"],
                "siblings": [
                    {"rfilename": "Q4_K_M/model-Q4_K_M-00001-of-00002.gguf"},
                    {"rfilename": "Q4_K_M/model-Q4_K_M-00002-of-00002.gguf"},
                ],
            })

        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        item = await HuggingFaceCatalog(http).details("org/model-GGUF")

        self.assertEqual(len(tree_requests), 2)
        self.assertEqual(tree_requests[1].url.params["cursor"], "second-page")
        self.assertEqual(item["quantizations"][0]["weight_size_bytes"], 22)
        self.assertEqual(
            [file["size_bytes"] for file in item["quantizations"][0]["files"]],
            [10, 12],
        )
        await http.aclose()

    async def test_details_follows_one_same_origin_repository_rename(self):
        paths = []

        def handler(request: httpx.Request) -> httpx.Response:
            paths.append(request.url.path)
            if request.url.path.endswith("/org/old-name"):
                return httpx.Response(
                    307, headers={"location": "/api/models/org/new-name"}
                )
            return httpx.Response(200, json={
                "id": "org/new-name", "tags": ["safetensors"],
                "safetensors": {"total": 10}, "siblings": [],
            })

        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        item = await HuggingFaceCatalog(http).details("org/old-name")

        self.assertEqual(paths, [
            "/api/models/org/old-name", "/api/models/org/new-name",
        ])
        self.assertEqual(item["id"], "org/new-name")
        await http.aclose()

    async def test_details_prefers_tree_weight_size_over_inflated_safetensors(self):
        # Hub safetensors metadata double-counts tensors shared across shards,
        # so the element-count estimate (304 bytes) is far above the real
        # weight files (100 bytes).
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/tree/main"):
                return httpx.Response(200, json=[
                    {"type": "file", "path": "model.safetensors", "size": 100},
                    {"type": "file", "path": "config.json", "size": 10},
                ])
            return httpx.Response(200, json={
                "id": "org/model", "tags": ["safetensors"],
                "safetensors": {
                    "total": 300,
                    "parameters": {"I8": 296, "BF16": 4},
                },
                "siblings": [{"rfilename": "model.safetensors"}],
            })

        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        item = await HuggingFaceCatalog(http).details("org/model")

        self.assertEqual(item["parameter_count"], 300)
        self.assertEqual(item["weight_size_bytes"], 100)
        self.assertEqual(item["weight_size_source"], "tree")
        await http.aclose()

    async def test_details_keeps_safetensors_estimate_when_tree_fails(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/tree/main"):
                return httpx.Response(500)
            return httpx.Response(200, json={
                "id": "org/model", "tags": ["safetensors"],
                "safetensors": {
                    "total": 300,
                    "parameters": {"I8": 296, "BF16": 4},
                },
                "siblings": [{"rfilename": "model.safetensors"}],
            })

        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        item = await HuggingFaceCatalog(http).details("org/model")

        self.assertEqual(item["weight_size_bytes"], 304)
        self.assertEqual(item["weight_size_source"], "safetensors")
        await http.aclose()

    async def test_details_keeps_estimate_when_tree_lacks_weight_sizes(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/tree/main"):
                return httpx.Response(200, json=[
                    {"type": "file", "path": "model.safetensors"},
                    {"type": "file", "path": "config.json", "size": 10},
                ])
            return httpx.Response(200, json={
                "id": "org/model", "tags": ["safetensors"],
                "safetensors": {
                    "total": 300,
                    "parameters": {"I8": 296, "BF16": 4},
                },
                "siblings": [{"rfilename": "model.safetensors"}],
            })

        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        item = await HuggingFaceCatalog(http).details("org/model")

        self.assertEqual(item["weight_size_bytes"], 304)
        self.assertEqual(item["weight_size_source"], "safetensors")
        await http.aclose()

    async def test_details_sums_only_safetensors_among_alternative_weight_copies(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if "/tree/" in request.url.path:
                return httpx.Response(200, json=[
                    {"type": "file", "path": "model.safetensors", "size": 100},
                    {"type": "file", "path": "pytorch_model.bin", "size": 200},
                    {"type": "file", "path": "model-Q4_0.gguf", "size": 50},
                ])
            return httpx.Response(200, json={
                "id": "org/model", "tags": ["safetensors"],
                "safetensors": {
                    "total": 300,
                    "parameters": {"I8": 296, "BF16": 4},
                },
                "siblings": [{"rfilename": "model.safetensors"}],
            })

        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        item = await HuggingFaceCatalog(http).details("org/model")

        self.assertEqual(item["weight_size_bytes"], 100)
        self.assertEqual(item["weight_size_source"], "tree")
        await http.aclose()

    async def test_details_fetches_tree_at_resolved_revision(self):
        tree_paths = []
        sha = "a" * 40

        def handler(request: httpx.Request) -> httpx.Response:
            if "/tree/" in request.url.path:
                tree_paths.append(request.url.path)
                return httpx.Response(200, json=[
                    {"type": "file", "path": "model.safetensors", "size": 100},
                ])
            return httpx.Response(200, json={
                "id": "org/model", "tags": ["safetensors"], "sha": sha,
                "safetensors": {
                    "total": 300,
                    "parameters": {"I8": 296, "BF16": 4},
                },
                "siblings": [{"rfilename": "model.safetensors"}],
            })

        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        item = await HuggingFaceCatalog(http).details("org/model")

        self.assertEqual(tree_paths, [f"/api/models/org/model/tree/{sha}"])
        self.assertEqual(item["revision"], sha)
        self.assertEqual(item["weight_size_bytes"], 100)
        self.assertEqual(item["weight_size_source"], "tree")
        await http.aclose()

    async def test_search_enriches_every_rendered_result_at_its_revision(self):
        tree_requests = []
        sha = "b" * 40

        def handler(request: httpx.Request) -> httpx.Response:
            if "/tree/" in request.url.path:
                tree_requests.append(request.url.path)
                return httpx.Response(200, json=[
                    {"type": "file", "path": "model.safetensors", "size": 100},
                ])
            return httpx.Response(200, json=[{
                "id": f"org/model-{index}", "tags": ["safetensors"],
                "sha": sha,
                "safetensors": {"total": 300, "parameters": {"I8": 296, "BF16": 4}},
                "siblings": [{"rfilename": "model.safetensors"}],
            } for index in range(40)])

        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        items = await HuggingFaceCatalog(http).search("model", 40)

        self.assertEqual(len(items), 40)
        self.assertEqual(len(tree_requests), 40)
        self.assertTrue(all(
            path.endswith(f"/tree/{sha}") for path in tree_requests
        ))
        self.assertTrue(all(
            item["weight_size_source"] == "tree" for item in items
        ))
        await http.aclose()

    async def test_details_sums_only_primary_safetensors_checkpoint(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if "/tree/" in request.url.path:
                return httpx.Response(200, json=[
                    {"type": "file", "path": "model.safetensors", "size": 100},
                    {"type": "file", "path": "awq/model.safetensors", "size": 40},
                ])
            return httpx.Response(200, json={
                "id": "org/model", "tags": ["safetensors"],
                "safetensors": {
                    "total": 300,
                    "parameters": {"I8": 296, "BF16": 4},
                },
                "siblings": [{"rfilename": "model.safetensors"}],
            })

        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        item = await HuggingFaceCatalog(http).details("org/model")

        # The root checkpoint wins over the quantized copy in a subdirectory.
        self.assertEqual(item["weight_size_bytes"], 100)
        self.assertEqual(item["weight_size_source"], "tree")
        await http.aclose()

    async def test_details_uses_largest_checkpoint_group_without_root_weights(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if "/tree/" in request.url.path:
                return httpx.Response(200, json=[
                    {"type": "file", "path": "bf16/model.safetensors", "size": 200},
                    {"type": "file", "path": "awq/model.safetensors", "size": 50},
                ])
            return httpx.Response(200, json={
                "id": "org/model", "tags": ["safetensors"],
                "safetensors": {
                    "total": 300,
                    "parameters": {"I8": 296, "BF16": 4},
                },
                "siblings": [{"rfilename": "bf16/model.safetensors"}],
            })

        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        item = await HuggingFaceCatalog(http).details("org/model")

        self.assertEqual(item["weight_size_bytes"], 200)
        self.assertEqual(item["weight_size_source"], "tree")
        await http.aclose()

    async def test_details_sums_all_component_directories_of_one_checkpoint(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if "/tree/" in request.url.path:
                return httpx.Response(200, json=[
                    {"type": "file", "path": "transformer/model.safetensors", "size": 200},
                    {"type": "file", "path": "text_encoder/model.safetensors", "size": 30},
                    {"type": "file", "path": "vae/model.safetensors", "size": 10},
                    {"type": "file", "path": "awq/model.safetensors", "size": 60},
                ])
            return httpx.Response(200, json={
                "id": "org/model", "tags": ["safetensors"],
                "safetensors": {
                    "total": 300,
                    "parameters": {"I8": 296, "BF16": 4},
                },
                "siblings": [{"rfilename": "transformer/model.safetensors"}],
            })

        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        item = await HuggingFaceCatalog(http).details("org/model")

        # Component directories belong to the primary checkpoint; the AWQ
        # copy is an alternative that is never co-loaded.
        self.assertEqual(item["weight_size_bytes"], 240)
        self.assertEqual(item["weight_size_source"], "tree")
        await http.aclose()

    async def test_details_separates_variants_sharing_one_directory(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if "/tree/" in request.url.path:
                return httpx.Response(200, json=[
                    {"type": "file", "path": "unet/diffusion_model.safetensors", "size": 100},
                    {"type": "file", "path": "unet/diffusion_model.fp16.safetensors", "size": 150},
                ])
            return httpx.Response(200, json={
                "id": "org/model", "tags": ["safetensors"],
                "safetensors": {
                    "total": 300,
                    "parameters": {"I8": 296, "BF16": 4},
                },
                "siblings": [{"rfilename": "unet/diffusion_model.safetensors"}],
            })

        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        item = await HuggingFaceCatalog(http).details("org/model")

        # The unmarked file is the primary checkpoint; the fp16 sibling in
        # the same folder is an alternative that is never co-loaded.
        self.assertEqual(item["weight_size_bytes"], 100)
        self.assertEqual(item["weight_size_source"], "tree")
        await http.aclose()

    async def test_details_aggregates_one_variant_across_component_directories(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if "/tree/" in request.url.path:
                return httpx.Response(200, json=[
                    {"type": "file", "path": "unet/model.fp16.safetensors", "size": 100},
                    {"type": "file", "path": "text_encoder/model.fp16.safetensors", "size": 30},
                    {"type": "file", "path": "vae/model.safetensors", "size": 5},
                ])
            return httpx.Response(200, json={
                "id": "org/model", "tags": ["safetensors"],
                "safetensors": {
                    "total": 300,
                    "parameters": {"I8": 296, "BF16": 4},
                },
                "siblings": [{"rfilename": "unet/model.fp16.safetensors"}],
            })

        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        item = await HuggingFaceCatalog(http).details("org/model")

        # Component directories existing in only one dtype are required parts
        # of the pipeline and sum with the unmarked component.
        self.assertEqual(item["weight_size_bytes"], 135)
        self.assertEqual(item["weight_size_source"], "tree")
        await http.aclose()

    async def test_details_excludes_training_snapshot_directories(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if "/tree/" in request.url.path:
                return httpx.Response(200, json=[
                    {"type": "file", "path": "model.safetensors", "size": 100},
                    {"type": "file", "path": "checkpoint-100/model.safetensors", "size": 90},
                    {"type": "file", "path": "checkpoint-200/model.safetensors", "size": 95},
                ])
            return httpx.Response(200, json={
                "id": "org/model", "tags": ["safetensors"],
                "safetensors": {
                    "total": 300,
                    "parameters": {"I8": 296, "BF16": 4},
                },
                "siblings": [{"rfilename": "model.safetensors"}],
            })

        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        item = await HuggingFaceCatalog(http).details("org/model")

        self.assertEqual(item["weight_size_bytes"], 100)
        self.assertEqual(item["weight_size_source"], "tree")
        await http.aclose()

    async def test_search_enriches_safetensors_weight_size_from_tree(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/tree/main"):
                return httpx.Response(200, json=[
                    {"type": "file", "path": "model.safetensors", "size": 100},
                ])
            if request.url.path == "/api/models":
                return httpx.Response(200, json=[{
                    "id": "org/model", "tags": ["safetensors"],
                    "safetensors": {
                        "total": 300,
                        "parameters": {"I8": 296, "BF16": 4},
                    },
                    "siblings": [{"rfilename": "model.safetensors"}],
                }])
            return httpx.Response(200, json={
                "id": "org/model", "tags": ["safetensors"],
                "safetensors": {
                    "total": 300,
                    "parameters": {"I8": 296, "BF16": 4},
                },
                "siblings": [{"rfilename": "model.safetensors"}],
            })

        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        items = await HuggingFaceCatalog(http).search("model", 5)

        self.assertEqual(items[0]["weight_size_bytes"], 100)
        self.assertEqual(items[0]["weight_size_source"], "tree")
        await http.aclose()

    async def test_search_keeps_estimate_when_enrichment_fails(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/models":
                return httpx.Response(200, json=[{
                    "id": "org/model", "tags": ["safetensors"],
                    "safetensors": {
                        "total": 300,
                        "parameters": {"I8": 296, "BF16": 4},
                    },
                    "siblings": [{"rfilename": "model.safetensors"}],
                }])
            return httpx.Response(500)

        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        items = await HuggingFaceCatalog(http).search("model", 5)

        self.assertEqual(items[0]["weight_size_bytes"], 304)
        self.assertEqual(items[0]["weight_size_source"], "safetensors")
        await http.aclose()

    async def test_search_does_not_cache_partially_enriched_results(self):
        list_requests = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal list_requests
            if "/tree/" in request.url.path:
                await asyncio.sleep(5)
                return httpx.Response(200, json=[
                    {"type": "file", "path": "model.safetensors", "size": 100},
                ])
            list_requests += 1
            return httpx.Response(200, json=[{
                "id": "org/model", "tags": ["safetensors"],
                "safetensors": {"total": 300, "parameters": {"I8": 296, "BF16": 4}},
                "siblings": [{"rfilename": "model.safetensors"}],
            }])

        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        catalog = HuggingFaceCatalog(http)
        catalog._enrich_timeout = 0.05

        items = await catalog.search("model", 5)

        self.assertEqual(items[0]["weight_size_bytes"], 304)
        self.assertEqual(items[0]["weight_size_source"], "safetensors")
        self.assertEqual(catalog._cache, {})
        await catalog.search("model", 5)
        self.assertEqual(list_requests, 2)
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
