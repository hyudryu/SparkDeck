"""Embedding-model discovery, routing, HTTP contract, and runtime lifecycle."""

import asyncio
import base64
import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import httpx


with patch("docker.from_env", return_value=Mock()):
    import server

from manager import Manager
from sparkdeck import embeddings as embeddings_module
from sparkdeck.embeddings import (
    EMBEDDINGS_CAPABILITY,
    EmbeddingError,
    EmbeddingRuntime,
    embedding_descriptor,
    embeddings_response,
    normalize_embedding_inputs,
    parse_embedding_request,
)
from sparkdeck.virtual_nas import VirtualNAS


REVISION = "a" * 40
NEWER_REVISION = "b" * 40
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

_ST_MODULES = (
    ("sentence_transformers.models.Transformer", ""),
    ("sentence_transformers.models.Pooling", "1_Pooling"),
    ("sentence_transformers.models.Normalize", "2_Normalize"),
)


def write_manifest(snapshot: Path, modules=_ST_MODULES) -> None:
    snapshot.mkdir(parents=True, exist_ok=True)
    snapshot.joinpath("modules.json").write_text(
        json.dumps([
            {"idx": index, "name": str(index), "path": path, "type": module_type}
            for index, (module_type, path) in enumerate(modules)
        ]),
        encoding="utf-8",
    )


def create_cached_embedding(
    hub: Path, *, revision: str = REVISION, dimension: int = 384,
    model_id: str = MODEL_ID,
) -> Path:
    """Build a Hugging Face cache entry for a SentenceTransformers repository."""
    owner, repository = model_id.split("/", 1)
    root = hub / f"models--{owner}--{repository}"
    (root / "blobs").mkdir(parents=True, exist_ok=True)
    snapshot = root / "snapshots" / revision
    snapshot.mkdir(parents=True, exist_ok=True)
    snapshot.joinpath("1_Pooling").mkdir(exist_ok=True)
    snapshot.joinpath("1_Pooling", "config.json").write_text("{}", encoding="utf-8")
    snapshot.joinpath("model.safetensors").write_bytes(b"encoder-weights")
    snapshot.joinpath("tokenizer.json").write_text("{}", encoding="utf-8")
    snapshot.joinpath("config.json").write_text(
        json.dumps({"hidden_size": dimension}), encoding="utf-8",
    )
    write_manifest(snapshot)
    return root


def create_cached_llm(hub: Path, model_id: str = "org/model") -> Path:
    """Build a cache entry for an ordinary text-generation repository."""
    owner, repository = model_id.split("/", 1)
    root = hub / f"models--{owner}--{repository}"
    (root / "blobs").mkdir(parents=True, exist_ok=True)
    snapshot = root / "snapshots" / REVISION
    snapshot.mkdir(parents=True, exist_ok=True)
    snapshot.joinpath("model.safetensors").write_bytes(b"decoder-weights")
    snapshot.joinpath("tokenizer.json").write_text("{}", encoding="utf-8")
    snapshot.joinpath("config.json").write_text("{}", encoding="utf-8")
    return root


class EmbeddingDescriptorTests(unittest.TestCase):
    """A cached repository is only servable when it is really an embedding model."""

    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.hub = Path(self._temporary.name) / "hub"
        self.hub.mkdir(parents=True)

    def tearDown(self):
        self._temporary.cleanup()

    def test_manifest_identifies_a_sentence_transformers_repository(self):
        repository = create_cached_embedding(self.hub)

        descriptor = embedding_descriptor(repository, {REVISION})

        self.assertIsNotNone(descriptor)
        self.assertEqual(descriptor.revision, REVISION)
        self.assertEqual(descriptor.module_count, 3)
        self.assertEqual(descriptor.dimension, 384)
        self.assertEqual(descriptor.snapshot, repository / "snapshots" / REVISION)

    def test_public_descriptor_never_leaks_a_cache_path(self):
        descriptor = embedding_descriptor(create_cached_embedding(self.hub), {REVISION})

        public = descriptor.public()

        self.assertEqual(
            public, {"revision": REVISION, "module_count": 3, "dimension": 384},
        )
        self.assertNotIn("snapshot", public)

    def test_repository_without_a_manifest_is_not_an_embedding_model(self):
        self.assertIsNone(
            embedding_descriptor(create_cached_llm(self.hub), {REVISION}),
        )

    def test_unknown_revision_is_not_servable(self):
        repository = create_cached_embedding(self.hub)

        self.assertIsNone(embedding_descriptor(repository, {"c" * 40}))
        self.assertIsNone(embedding_descriptor(repository, set()))

    def test_newest_valid_revision_wins(self):
        repository = create_cached_embedding(self.hub, revision=REVISION)
        newer = create_cached_embedding(self.hub, revision=NEWER_REVISION)
        # Both snapshots live in one repository; the newer one is preferred.
        self.assertEqual(repository, newer)
        os.utime(
            repository / "snapshots" / NEWER_REVISION,
            (2_000_000_000, 2_000_000_000),
        )

        descriptor = embedding_descriptor(repository, {REVISION, NEWER_REVISION})

        self.assertEqual(descriptor.revision, NEWER_REVISION)

    def test_revision_directory_name_cannot_escape_the_cache(self):
        repository = create_cached_embedding(self.hub)

        descriptor = embedding_descriptor(repository, {"../../outside"})

        self.assertIsNone(descriptor)

    def test_module_outside_the_sentence_transformers_namespace_is_refused(self):
        # `SentenceTransformer.load` imports each module's `type` by dotted
        # path, so a manifest naming arbitrary code must never be advertised.
        repository = create_cached_embedding(self.hub)
        write_manifest(
            repository / "snapshots" / REVISION,
            (
                ("sentence_transformers.models.Transformer", ""),
                ("os.system", ""),
            ),
        )

        self.assertIsNone(embedding_descriptor(repository, {REVISION}))

    def test_manifest_without_an_encoder_is_refused(self):
        repository = create_cached_embedding(self.hub)
        write_manifest(
            repository / "snapshots" / REVISION,
            (("sentence_transformers.models.Pooling", "1_Pooling"),),
        )

        self.assertIsNone(embedding_descriptor(repository, {REVISION}))

    def test_malformed_and_oversized_manifests_are_refused(self):
        repository = create_cached_embedding(self.hub)
        snapshot = repository / "snapshots" / REVISION
        for content in ("{not json", "[]", '{"type": "x"}', "null"):
            with self.subTest(content=content):
                snapshot.joinpath("modules.json").write_text(content, encoding="utf-8")
                self.assertIsNone(embedding_descriptor(repository, {REVISION}))
        snapshot.joinpath("modules.json").write_text(
            "[" + " " * (embeddings_module._MANIFEST_MAX_BYTES + 1) + "]",
            encoding="utf-8",
        )
        self.assertIsNone(embedding_descriptor(repository, {REVISION}))

    def test_missing_dimension_is_simply_absent(self):
        repository = create_cached_embedding(self.hub, dimension=384)
        snapshot = repository / "snapshots" / REVISION
        snapshot.joinpath("config.json").write_text("{}", encoding="utf-8")

        descriptor = embedding_descriptor(repository, {REVISION})

        self.assertIsNotNone(descriptor)
        self.assertIsNone(descriptor.dimension)
        self.assertNotIn("dimension", descriptor.public())

    @unittest.skipIf(os.name == "nt", "creating cache symlinks requires privileges")
    def test_manifest_symlinked_outside_the_cache_is_refused(self):
        repository = create_cached_embedding(self.hub)
        snapshot = repository / "snapshots" / REVISION
        outside = Path(self._temporary.name) / "outside" / "modules.json"
        outside.parent.mkdir()
        outside.write_text(
            (snapshot / "modules.json").read_text(encoding="utf-8"), encoding="utf-8",
        )
        (snapshot / "modules.json").unlink()
        (snapshot / "modules.json").symlink_to(outside)

        self.assertIsNone(embedding_descriptor(repository, {REVISION}))

    @unittest.skipIf(os.name == "nt", "creating cache symlinks requires privileges")
    def test_manifest_symlinked_into_blobs_is_accepted(self):
        # The Hub stores snapshot entries as symlinks into the same
        # repository's blobs directory, so this is the normal layout.
        repository = create_cached_embedding(self.hub)
        snapshot = repository / "snapshots" / REVISION
        blob = repository / "blobs" / ("c" * 64)
        blob.write_text(
            (snapshot / "modules.json").read_text(encoding="utf-8"), encoding="utf-8",
        )
        (snapshot / "modules.json").unlink()
        (snapshot / "modules.json").symlink_to(blob)

        self.assertIsNotNone(embedding_descriptor(repository, {REVISION}))


class EmbeddingRequestTests(unittest.TestCase):
    def test_input_accepts_a_string_or_a_list_of_strings(self):
        self.assertEqual(normalize_embedding_inputs("hello world"), ["hello world"])
        self.assertEqual(normalize_embedding_inputs(["a", "b"]), ["a", "b"])

    def test_input_rejects_shapes_that_would_encode_something_else(self):
        for value in (None, "", [], [1, 2], [["a"]], 5, {"input": "a"}):
            with self.subTest(value=value):
                with self.assertRaises(EmbeddingError):
                    normalize_embedding_inputs(value)

    def test_input_bounds_batch_size_and_length(self):
        with self.assertRaises(EmbeddingError):
            normalize_embedding_inputs(
                ["x"] * (embeddings_module.MAX_EMBEDDING_INPUTS + 1)
            )
        with self.assertRaises(EmbeddingError):
            normalize_embedding_inputs(
                ["x" * (embeddings_module.MAX_EMBEDDING_INPUT_CHARS + 1)]
            )

    def test_request_round_trips_openai_fields(self):
        request = parse_embedding_request({
            "model": MODEL_ID, "input": ["a"], "encoding_format": "base64",
        })

        self.assertEqual(request.model, MODEL_ID)
        self.assertEqual(request.inputs, ["a"])
        self.assertEqual(request.encoding_format, "base64")
        self.assertTrue(request.normalize)

    def test_request_refuses_what_cannot_be_honoured(self):
        for body in (
            {"input": "a"},
            {"model": "", "input": "a"},
            {"model": MODEL_ID},
            {"model": MODEL_ID, "input": "a", "dimensions": 128},
            {"model": MODEL_ID, "input": "a", "encoding_format": "int8"},
            {"model": MODEL_ID, "input": "a", "normalize": "yes"},
        ):
            with self.subTest(body=body):
                with self.assertRaises(EmbeddingError):
                    parse_embedding_request(body)

    def test_response_matches_the_openai_embeddings_shape(self):
        response = embeddings_response(MODEL_ID, [[1.0, 2.0]], 7)

        self.assertEqual(response["object"], "list")
        self.assertEqual(response["model"], MODEL_ID)
        self.assertEqual(response["usage"], {"prompt_tokens": 7, "total_tokens": 7})
        self.assertEqual(
            response["data"],
            [{"object": "embedding", "index": 0, "embedding": [1.0, 2.0]}],
        )

    def test_base64_response_is_a_little_endian_float32_vector(self):
        response = embeddings_response(MODEL_ID, [[1.0, -2.5]], 3, "base64")

        encoded = response["data"][0]["embedding"]
        self.assertEqual(
            struct.unpack("<2f", base64.b64decode(encoded)), (1.0, -2.5),
        )


class EmbeddingInventoryTests(unittest.IsolatedAsyncioTestCase):
    """The node that holds the files is what decides a model is servable."""

    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.hub = Path(self._temporary.name) / "hub"
        self.hub.mkdir(parents=True)
        self.virtual_nas = VirtualNAS(
            Path(self._temporary.name) / "data",
            lambda: self.hub,
            Mock(),
            lambda: True,
        )

    def tearDown(self):
        self._temporary.cleanup()

    async def test_inventory_marks_only_embedding_repositories(self):
        create_cached_embedding(self.hub)
        create_cached_llm(self.hub)

        models = {model["model_id"]: model for model in self.virtual_nas.inventory()}

        self.assertEqual(
            models[MODEL_ID]["embedding"],
            {"revision": REVISION, "module_count": 3, "dimension": 384},
        )
        self.assertNotIn("embedding", models["org/model"])

    async def test_embedding_snapshot_resolves_only_a_cached_revision(self):
        repository = create_cached_embedding(self.hub)

        snapshot = self.virtual_nas.embedding_snapshot(MODEL_ID)
        self.assertEqual(snapshot.snapshot, repository / "snapshots" / REVISION)

        self.assertIsNone(self.virtual_nas.embedding_snapshot(MODEL_ID, "c" * 40))
        self.assertIsNone(self.virtual_nas.embedding_snapshot("org/model"))

    async def test_embedding_snapshot_refuses_an_unsafe_model_id(self):
        with self.assertRaises(ValueError):
            self.virtual_nas.embedding_snapshot("org/../../escape")


def embedding_node(node_id, *, capabilities=(EMBEDDINGS_CAPABILITY,), online=True):
    return {
        "id": node_id,
        "name": node_id,
        "online": online,
        "capabilities": list(capabilities),
        "models": [{
            "model_id": MODEL_ID,
            "embedding": {"revision": REVISION, "module_count": 3, "dimension": 384},
        }],
    }


def bare_manager(**attributes):
    manager = Manager.__new__(Manager)
    manager.settings = {"embeddings_enabled": True}
    manager.embedding_runtime = Mock()
    manager.node_registry = Mock()
    manager.node_registry.request = AsyncMock()
    for key, value in attributes.items():
        setattr(manager, key, value)
    return manager


class EmbeddingDiscoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_models_are_merged_across_nodes_that_can_serve_them(self):
        manager = bare_manager(
            model_cache_inventory=AsyncMock(return_value=[
                embedding_node("local"),
                embedding_node("node-2"),
            ]),
        )

        discovered = await manager.embedding_models()

        self.assertTrue(discovered["enabled"])
        self.assertEqual(
            discovered["models"],
            [{
                "model_id": MODEL_ID, "module_count": 3, "dimension": 384,
                "nodes": [
                    {"id": "local", "revision": REVISION},
                    {"id": "node-2", "revision": REVISION},
                ],
            }],
        )

    async def test_each_node_keeps_its_own_resolved_revision(self):
        # Two nodes can hold the same repository at different revisions, so
        # merging them must not flatten one node's revision onto the other.
        other = embedding_node("node-2")
        other["models"][0]["embedding"] = {
            "revision": NEWER_REVISION, "module_count": 3, "dimension": 384,
        }
        manager = bare_manager(
            model_cache_inventory=AsyncMock(return_value=[
                embedding_node("local"), other,
            ]),
        )

        discovered = await manager.embedding_models()

        self.assertEqual(
            discovered["models"][0]["nodes"],
            [
                {"id": "local", "revision": REVISION},
                {"id": "node-2", "revision": NEWER_REVISION},
            ],
        )

        request = AsyncMock(return_value={"embeddings": [[1.0]]})
        manager.node_registry.request = request
        manager.embed_local = AsyncMock(side_effect=RuntimeError("torch unavailable"))

        await manager.embed(MODEL_ID, ["hello"])

        self.assertEqual(
            request.await_args.kwargs["json_body"]["revision"], NEWER_REVISION,
        )

    async def test_nodes_without_the_capability_are_not_advertised(self):
        manager = bare_manager(
            model_cache_inventory=AsyncMock(return_value=[
                embedding_node("node-2", capabilities=[]),
                embedding_node("node-3", online=False),
            ]),
        )

        discovered = await manager.embedding_models()

        self.assertEqual(discovered["models"], [])

    async def test_partial_snapshots_are_never_advertised(self):
        node = embedding_node("local")
        node["models"][0]["partial"] = True
        manager = bare_manager(
            model_cache_inventory=AsyncMock(return_value=[node]),
        )

        self.assertEqual((await manager.embedding_models())["models"], [])

    async def test_disabled_serving_lists_nothing_and_skips_the_fan_out(self):
        inventory = AsyncMock(return_value=[embedding_node("local")])
        manager = bare_manager(model_cache_inventory=inventory)
        manager.settings = {"embeddings_enabled": False}

        discovered = await manager.embedding_models()

        self.assertEqual(discovered, {"enabled": False, "models": []})
        inventory.assert_not_awaited()

    async def test_discovery_is_cached_between_polls(self):
        inventory = AsyncMock(return_value=[embedding_node("local")])
        manager = bare_manager(model_cache_inventory=inventory)

        await manager.embedding_models()
        await manager.embedding_models()

        inventory.assert_awaited_once()

    async def test_status_reports_runtime_state_without_installing(self):
        runtime = Mock()
        runtime.state.return_value = {"installed": False, "version": None}
        manager = bare_manager(
            model_cache_inventory=AsyncMock(return_value=[embedding_node("local")]),
            embedding_runtime=runtime,
        )

        status = await manager.embedding_status()

        self.assertEqual(status["runtime"], {"installed": False, "version": None})
        runtime.install.assert_not_called()


class EmbeddingRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_local_node_is_preferred_over_a_remote_copy(self):
        manager = bare_manager(
            model_cache_inventory=AsyncMock(return_value=[
                embedding_node("node-2"), embedding_node("local"),
            ]),
        )
        manager.embed_local = AsyncMock(return_value={"embeddings": [[1.0]]})

        result = await manager.embed(MODEL_ID, ["hello"])

        self.assertEqual(result, {"embeddings": [[1.0]]})
        manager.embed_local.assert_awaited_once_with(
            MODEL_ID, REVISION, ["hello"], normalize=True,
        )
        manager.node_registry.request.assert_not_awaited()

    async def test_remote_copy_is_requested_from_an_agent(self):
        request = AsyncMock(return_value={"embeddings": [[2.0]], "prompt_tokens": 1})
        manager = bare_manager(
            model_cache_inventory=AsyncMock(return_value=[embedding_node("node-2")]),
        )
        manager.node_registry.request = request

        result = await manager.embed(MODEL_ID, ["hello"], normalize=False)

        self.assertEqual(result, {"embeddings": [[2.0]], "prompt_tokens": 1})
        self.assertEqual(request.await_args.args[:3], (
            "node-2", "POST", "/api/agent/embeddings",
        ))
        self.assertEqual(request.await_args.kwargs["json_body"], {
            "model_id": MODEL_ID, "revision": REVISION,
            "inputs": ["hello"], "normalize": False,
        })
        # The request may be the one that installs the runtime on that node.
        self.assertGreaterEqual(
            request.await_args.kwargs["timeout"],
            embeddings_module.DEFAULT_INSTALL_TIMEOUT,
        )

    async def test_a_failing_node_does_not_hide_a_working_one(self):
        request = AsyncMock(side_effect=[
            RuntimeError("install failed"), {"embeddings": [[3.0]]},
        ])
        manager = bare_manager(
            model_cache_inventory=AsyncMock(return_value=[
                embedding_node("node-2"), embedding_node("node-3"),
            ]),
        )
        manager.node_registry.request = request

        result = await manager.embed(MODEL_ID, ["hello"])

        self.assertEqual(result, {"embeddings": [[3.0]]})
        self.assertEqual(request.await_args_list[-1].args[0], "node-3")

    async def test_no_servable_model_reports_a_lookup_failure(self):
        manager = bare_manager(
            model_cache_inventory=AsyncMock(return_value=[embedding_node("local")]),
        )

        with self.assertRaises(LookupError):
            await manager.embed("org/other", ["hello"])

    async def test_every_node_failing_reports_a_runtime_failure(self):
        manager = bare_manager(
            model_cache_inventory=AsyncMock(return_value=[embedding_node("node-2")]),
        )
        manager.node_registry.request = AsyncMock(side_effect=RuntimeError("boom"))

        with self.assertRaises(RuntimeError):
            await manager.embed(MODEL_ID, ["hello"])

    async def test_disabled_serving_refuses_requests(self):
        manager = bare_manager()
        manager.settings = {"embeddings_enabled": False}

        with self.assertRaises(RuntimeError):
            await manager.embed(MODEL_ID, ["hello"])

    async def test_local_serving_resolves_the_snapshot_before_encoding(self):
        runtime = Mock()
        runtime.encode = AsyncMock(return_value={"embeddings": [[1.0]]})
        snapshot = Mock()
        snapshot.snapshot = Path("C:/cache/snapshot")
        virtual_nas = Mock()
        virtual_nas.embedding_snapshot = Mock(return_value=snapshot)
        manager = bare_manager(virtual_nas=virtual_nas, embedding_runtime=runtime)

        result = await manager.embed_local(MODEL_ID, REVISION, ["hello"])

        self.assertEqual(result, {"embeddings": [[1.0]]})
        self.assertEqual(runtime.encode.await_args.kwargs["snapshot"], snapshot.snapshot)

    async def test_local_serving_refuses_a_model_that_is_not_cached(self):
        virtual_nas = Mock()
        virtual_nas.embedding_snapshot = Mock(return_value=None)
        manager = bare_manager(virtual_nas=virtual_nas, embedding_runtime=Mock())

        with self.assertRaises(LookupError):
            await manager.embed_local(MODEL_ID, None, ["hello"])

    async def test_agents_advertise_the_embedding_capability(self):
        manager = Manager.__new__(Manager)
        manager.agent_credentials = Mock(node_id="local")
        manager.settings = {"cluster_node_name": "spark-one"}

        self.assertIn(
            EMBEDDINGS_CAPABILITY, manager.agent_health()["capabilities"],
        )


class EmbeddingRuntimeTests(unittest.IsolatedAsyncioTestCase):
    def runtime(self, **kwargs) -> EmbeddingRuntime:
        return EmbeddingRuntime(Path(self._temporary.name), **kwargs)

    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._temporary.cleanup()

    async def test_install_creates_the_venv_installs_and_verifies(self):
        runtime = self.runtime()
        calls: list[list[str]] = []

        async def fake_run(argv, *, timeout, label):
            calls.append(argv)
            if "venv" in argv:
                return ""
            if "pip" in argv:
                return "Successfully installed sentence-transformers-3.0.1\n"
            # The import writes warnings of its own; the version must still be
            # read from the marked line rather than the last line of output.
            return (
                "Some weights were not initialized\n"
                "sentence-transformers 3.0.1\n"
                "trailing chatter\n"
            )

        with patch.object(runtime, "_run", side_effect=fake_run):
            result = await runtime.install()

        self.assertEqual(result["installed"], True)
        self.assertEqual(result["version"], "3.0.1")
        self.assertEqual(calls[0][1:4], ["-m", "venv", str(runtime.venv_dir)])
        self.assertIn("sentence-transformers", calls[1])
        self.assertIn("import sentence_transformers", calls[2][2])
        self.assertEqual(
            json.loads(runtime.marker_path.read_text(encoding="utf-8"))["requirement"],
            "sentence-transformers",
        )

    async def test_install_is_skipped_once_the_runtime_is_recorded(self):
        runtime = self.runtime()
        runtime._venv_python().parent.mkdir(parents=True)
        runtime._venv_python().write_bytes(b"")
        runtime.marker_path.write_text(json.dumps({
            "requirement": "sentence-transformers", "version": "3.0.1",
        }), encoding="utf-8")

        with patch.object(runtime, "_run", side_effect=AssertionError("must not run")):
            result = await runtime.install()

        self.assertEqual(result["version"], "3.0.1")

    async def test_a_deleted_environment_is_not_reported_as_installed(self):
        runtime = self.runtime()
        runtime.marker_path.parent.mkdir(parents=True)
        runtime.marker_path.write_text(json.dumps({
            "requirement": "sentence-transformers", "version": "3.0.1",
        }), encoding="utf-8")

        self.assertFalse(runtime.state()["installed"])

    async def test_a_failed_install_is_recorded_and_raised(self):
        runtime = self.runtime()

        with patch.object(
            runtime, "_run", side_effect=EmbeddingError("pip exploded"),
        ):
            with self.assertRaises(EmbeddingError):
                await runtime.install()

        self.assertEqual(runtime.state()["error"], "pip exploded")
        self.assertFalse(runtime.state()["installed"])

    async def test_install_failure_does_not_record_a_runtime(self):
        runtime = self.runtime()

        with patch.object(runtime, "_run", new=AsyncMock(return_value="")) as run:
            run.side_effect = EmbeddingError("no network")
            with self.assertRaises(EmbeddingError):
                await runtime.install()

        self.assertFalse(runtime.marker_path.exists())

    async def test_status_never_installs_anything(self):
        runtime = self.runtime()

        with patch.object(runtime, "_run", side_effect=AssertionError("must not run")):
            state = runtime.state()

        self.assertEqual(state["installed"], False)
        self.assertEqual(state["workers"], 0)
        self.assertIsNone(state["error"])


_WORKER_STUB = '''
import json, sys
print(json.dumps({"ready": True, "dimension": 3}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    if request["inputs"] == ["boom"]:
        print(json.dumps({"id": request["id"], "error": "RuntimeError: boom"}), flush=True)
        continue
    if request["inputs"] == ["empty"]:
        print(json.dumps({"id": request["id"], "embeddings": [], "prompt_tokens": 0}), flush=True)
        continue
    print(json.dumps({
        "id": request["id"],
        "embeddings": [[0.5, 0.25, 0.125] for _ in request["inputs"]],
        "prompt_tokens": 2 * len(request["inputs"]),
    }), flush=True)
'''

_FAILING_WORKER_STUB = '''
import json, sys
print(json.dumps({"ready": False, "error": "OSError: weights are missing"}), flush=True)
'''


class EmbeddingWorkerProtocolTests(unittest.IsolatedAsyncioTestCase):
    """The parent half of the worker protocol, without installing torch."""

    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.snapshot = self.root / "snapshot"
        self.snapshot.mkdir()
        self.stub = self.root / "worker.py"
        self.stub.write_text(_WORKER_STUB, encoding="utf-8")

    def tearDown(self):
        self._temporary.cleanup()

    def runtime(self, **kwargs) -> EmbeddingRuntime:
        runtime = EmbeddingRuntime(
            self.root / "data", idle_seconds=kwargs.pop("idle_seconds", 900.0),
            **kwargs,
        )
        patcher = patch.object(runtime, "_installed", return_value={
            "installed": True, "version": "stub",
        })
        patcher.start()
        self.addCleanup(patcher.stop)
        interpreter = patch.object(
            runtime, "_venv_python", return_value=Path(sys.executable),
        )
        interpreter.start()
        self.addCleanup(interpreter.stop)
        return runtime

    def worker_script(self, path: Path):
        patcher = patch.object(embeddings_module, "_WORKER_SCRIPT", path)
        patcher.start()
        self.addCleanup(patcher.stop)

    async def test_encode_returns_vectors_and_token_usage(self):
        runtime = self.runtime()
        self.worker_script(self.stub)

        result = await runtime.encode(
            model_id=MODEL_ID, snapshot=self.snapshot, inputs=["a", "b"],
        )
        await runtime.stop()

        self.assertEqual(result["embeddings"], [[0.5, 0.25, 0.125]] * 2)
        self.assertEqual(result["prompt_tokens"], 4)
        self.assertEqual(result["dimension"], 3)

    async def test_one_warm_worker_serves_repeated_requests(self):
        runtime = self.runtime()
        self.worker_script(self.stub)

        await runtime.encode(model_id=MODEL_ID, snapshot=self.snapshot, inputs=["a"])
        first = runtime._workers[str(self.snapshot)]
        await runtime.encode(model_id=MODEL_ID, snapshot=self.snapshot, inputs=["b"])
        await runtime.stop()

        self.assertIs(runtime._workers.get(str(self.snapshot)), None)
        self.assertIsNotNone(first.dimension)

    async def test_a_worker_error_is_reported_as_an_embedding_error(self):
        runtime = self.runtime()
        self.worker_script(self.stub)

        with self.assertRaises(EmbeddingError):
            await runtime.encode(
                model_id=MODEL_ID, snapshot=self.snapshot, inputs=["boom"],
            )
        await runtime.stop()

    async def test_a_wrong_vector_count_is_refused(self):
        runtime = self.runtime()
        self.worker_script(self.stub)

        with self.assertRaises(EmbeddingError):
            await runtime.encode(
                model_id=MODEL_ID, snapshot=self.snapshot, inputs=["empty"],
            )
        await runtime.stop()

    async def test_a_worker_that_cannot_load_reports_its_reason(self):
        runtime = self.runtime()
        failing = self.root / "failing.py"
        failing.write_text(_FAILING_WORKER_STUB, encoding="utf-8")
        self.worker_script(failing)

        with self.assertRaises(EmbeddingError) as raised:
            await runtime.encode(
                model_id=MODEL_ID, snapshot=self.snapshot, inputs=["a"],
            )

        self.assertIn("weights are missing", str(raised.exception))
        await runtime.stop()

    async def test_workers_beyond_the_limit_are_evicted(self):
        runtime = self.runtime(max_workers=1)
        self.worker_script(self.stub)
        other = self.root / "other-snapshot"
        other.mkdir()

        await runtime.encode(model_id=MODEL_ID, snapshot=self.snapshot, inputs=["a"])
        await runtime.encode(model_id=MODEL_ID, snapshot=other, inputs=["b"])

        self.assertEqual(list(runtime._workers), [str(other)])
        await runtime.stop()

    async def test_idle_workers_are_released(self):
        runtime = self.runtime(idle_seconds=0.05)
        self.worker_script(self.stub)

        await runtime.encode(model_id=MODEL_ID, snapshot=self.snapshot, inputs=["a"])
        self.assertEqual(len(runtime._workers), 1)

        for _ in range(50):
            if not runtime._workers:
                break
            await asyncio.sleep(0.05)

        self.assertEqual(runtime._workers, {})
        await runtime.stop()


class EmbeddingHttpTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app), base_url="http://test",
        )
        self.assignment = patch.object(
            server.onboarding.assignment, "load", return_value=None,
        )
        self.assignment.start()

    async def asyncTearDown(self):
        await self.client.aclose()
        self.assignment.stop()

    async def test_models_lists_a_cached_embedding_model(self):
        with patch.object(server.manager, "embedding_models", AsyncMock(return_value={
            "enabled": True,
            "models": [{
                "model_id": MODEL_ID, "module_count": 3, "dimension": 384,
                "nodes": [{"id": "local", "revision": REVISION}],
            }],
        })):
            response = await self.client.get("/v1/models")

        self.assertEqual(response.status_code, 200)
        entry = next(
            model for model in response.json()["data"] if model["id"] == MODEL_ID
        )
        self.assertEqual(entry["type"], "embedding")
        self.assertEqual(entry["owned_by"], "sentence-transformers")
        self.assertEqual(entry["dimension"], 384)
        self.assertEqual(entry["nodes"], ["local"])
        self.assertNotIn("snapshot", json.dumps(entry))

    async def test_models_still_answers_when_discovery_fails(self):
        with patch.object(
            server.manager, "embedding_models",
            AsyncMock(side_effect=RuntimeError("no nodes")),
        ):
            response = await self.client.get("/v1/models")

        self.assertEqual(response.status_code, 200)
        self.assertNotIn(
            MODEL_ID, [model["id"] for model in response.json()["data"]],
        )

    async def test_embeddings_route_returns_an_openai_response(self):
        embed = AsyncMock(return_value={
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.5, 0.25]}],
            "model": MODEL_ID,
            "usage": {"prompt_tokens": 2, "total_tokens": 2},
        })
        with patch.object(server.sparkdeck, "embeddings", embed):
            response = await self.client.post("/v1/embeddings", json={
                "model": MODEL_ID, "input": "hello world",
            })

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"][0]["embedding"], [0.5, 0.25])
        self.assertEqual(embed.await_args.args, (MODEL_ID, ["hello world"]))
        self.assertEqual(embed.await_args.kwargs["normalize"], True)

    async def test_embeddings_route_rejects_an_unusable_body(self):
        with patch.object(server.sparkdeck, "embeddings", AsyncMock()) as embed:
            response = await self.client.post("/v1/embeddings", json={
                "model": MODEL_ID, "input": [1, 2, 3],
            })

        self.assertEqual(response.status_code, 400)
        embed.assert_not_awaited()

    async def test_embeddings_route_maps_serving_failures(self):
        for error, expected in (
            (LookupError("no cached embedding model"), 404),
            (RuntimeError("embedding serving is disabled"), 409),
            (EmbeddingError("unusable result"), 400),
            (ValueError("model_id must be a safe Hugging Face owner/repository ID"), 400),
        ):
            with self.subTest(error=error):
                with patch.object(
                    server.sparkdeck, "embeddings", AsyncMock(side_effect=error),
                ):
                    response = await self.client.post("/v1/embeddings", json={
                        "model": MODEL_ID, "input": "hello",
                    })
                self.assertEqual(response.status_code, expected)

    async def test_embeddings_route_reports_a_timeout_as_unavailable(self):
        with patch.object(
            server.sparkdeck, "embeddings", AsyncMock(side_effect=TimeoutError("slow")),
        ):
            response = await self.client.post("/v1/embeddings", json={
                "model": MODEL_ID, "input": "hello",
            })

        self.assertEqual(response.status_code, 504)

    async def test_agent_route_requires_agent_authorization(self):
        response = await self.client.post("/api/agent/embeddings", json={
            "model_id": MODEL_ID, "inputs": ["hello"],
        })

        self.assertEqual(response.status_code, 401)

    async def test_agent_route_serves_a_cached_model(self):
        with patch.object(server, "_require_agent"), patch.object(
            server.manager, "embed_local",
            AsyncMock(return_value={"embeddings": [[1.0]], "prompt_tokens": 1}),
        ) as embed:
            response = await self.client.post("/api/agent/embeddings", json={
                "model_id": MODEL_ID, "revision": REVISION, "inputs": ["hello"],
            })

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["embeddings"], [[1.0]])
        embed.assert_awaited_once_with(MODEL_ID, REVISION, ["hello"], normalize=True)

    async def test_agent_route_refuses_unknown_fields(self):
        with patch.object(server, "_require_agent"), patch.object(
            server.manager, "embed_local", AsyncMock(),
        ) as embed:
            response = await self.client.post("/api/agent/embeddings", json={
                "model_id": MODEL_ID, "inputs": ["hello"], "trust_remote_code": True,
            })

        self.assertEqual(response.status_code, 400)
        embed.assert_not_awaited()

    async def test_agent_route_rejects_an_unsafe_model_id(self):
        with patch.object(server, "_require_agent"):
            response = await self.client.post("/api/agent/embeddings", json={
                "model_id": "../../escape", "inputs": ["hello"],
            })

        self.assertEqual(response.status_code, 400)

    async def test_agent_route_maps_a_missing_model_to_not_found(self):
        with patch.object(server, "_require_agent"), patch.object(
            server.manager, "embed_local",
            AsyncMock(side_effect=LookupError("cached model not found")),
        ):
            response = await self.client.post("/api/agent/embeddings", json={
                "model_id": MODEL_ID, "inputs": ["hello"],
            })

        self.assertEqual(response.status_code, 404)

    async def test_status_endpoint_reports_discovery_and_runtime(self):
        with patch.object(server.manager, "embedding_status", AsyncMock(return_value={
            "enabled": True,
            "models": [{
                "model_id": MODEL_ID,
                "nodes": [{"id": "local", "revision": REVISION}],
            }],
            "runtime": {"installed": False, "version": None},
        })):
            response = await self.client.get("/api/v1/embeddings/status")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["enabled"])
        self.assertEqual(response.json()["runtime"]["installed"], False)

    async def test_install_endpoint_installs_and_reports_state(self):
        with patch.object(
            server.manager, "install_embedding_runtime",
            AsyncMock(return_value={"installed": True, "version": "3.0.1"}),
        ):
            response = await self.client.post("/api/v1/embeddings/install")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["version"], "3.0.1")

    async def test_install_endpoint_maps_a_failed_install(self):
        with patch.object(
            server.manager, "install_embedding_runtime",
            AsyncMock(side_effect=EmbeddingError("pip could not reach the index")),
        ):
            response = await self.client.post("/api/v1/embeddings/install")

        self.assertEqual(response.status_code, 502)
