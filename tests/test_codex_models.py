"""The Codex and OpenAI catalog views expose the same routable models."""

from unittest.mock import AsyncMock, MagicMock

import unittest

from sparkdeck.service import SparkDeckService


def _service(deployments, native=None):
    service = SparkDeckService.__new__(SparkDeckService)
    service.manager = MagicMock(deployments=[])
    service.manager.embedding_models = AsyncMock(return_value={"models": []})
    service.store = MagicMock()
    service.store.deployment.return_value = None
    service.deployments = AsyncMock(return_value=deployments)
    service._native_llama_model = AsyncMock(return_value=native)
    service._model_routing_policies = {}
    return service


def _deployment(identifier, alias, served_models, status="running"):
    return {
        "id": identifier, "alias": alias, "served_models": served_models,
        "status": status, "runtime": "vllm",
        "model": {"repository": "org/model"},
    }


class CodexModelsTests(unittest.IsolatedAsyncioTestCase):
    async def test_codex_catalog_has_model_info_fields_and_preserves_openai_view(self):
        service = _service([_deployment("dep", "friendly", ["served-model"])])
        result = await service.models()

        assert result["object"] == "list"
        assert result["data"] == [{
            "id": "served-model", "object": "model", "created": 0,
            "owned_by": "sparkdeck", "runtime": "vllm", "deployment_id": "dep",
            "model": {"repository": "org/model"}, "container_name": None,
            "port": None,
        }]
        info, = result["models"]
        # Required non-optional fields from Codex's ModelInfo wire contract. A
        # models=data alias passes the envelope check but fails this contract.
        required = {
            "slug": str, "display_name": str, "supported_reasoning_levels": list,
            "shell_type": str, "visibility": str, "supported_in_api": bool,
            "priority": int, "support_verbosity": bool, "truncation_policy": dict,
            "experimental_supported_tools": list,
        }
        for field, field_type in required.items():
            assert type(info[field]) is field_type
        assert info["slug"] == "served-model"
        assert info["visibility"] == "list"
        assert info["shell_type"] == "unified_exec"
        assert info["base_instructions"]
        assert info["model_messages"]["instructions_template"] == info["base_instructions"]
        assert info["input_modalities"] == ["text"]


    async def test_codex_catalog_matches_collision_resolution_and_filters_stopped(self):
        result = await _service([
            _deployment("one", "first", ["shared"]),
            _deployment("two", "second", ["shared"]),
            _deployment("three", "stopped", ["offline"], "stopped"),
        ]).models()
        # The shared name advertises on its deterministic winner (id
        # tiebreaker here); the other profile stays visible by its alias.
        assert [model["slug"] for model in result["models"]] == ["first", "shared"]
        assert [model["slug"] for model in result["models"]] == [
            model["id"] for model in result["data"]
        ]


    async def test_native_model_is_available_in_both_catalogs_without_duplicates(self):
        native = await _service([], native="local.gguf").models()
        assert [model["slug"] for model in native["models"]] == ["local.gguf"]
        duplicate = await _service([
            _deployment("one", "first", ["local.gguf"]),
        ], native="local.gguf").models()
        assert len(duplicate["models"]) == len(duplicate["data"]) == 1


    async def test_empty_catalog_has_both_envelopes(self):
        assert await _service([]).models() == {"object": "list", "data": [], "models": []}

    async def test_embedding_models_keep_openai_discovery_without_coding_advertisement(self):
        service = _service([])
        service.manager.embedding_models.return_value = {
            "models": [{"model_id": "org/embed", "revision": "rev", "node_ids": ["node"]}],
        }
        result = await service.models()
        assert result["data"][0]["id"] == "org/embed"
        assert result["data"][0]["type"] == "embedding"
        assert result["models"] == []
