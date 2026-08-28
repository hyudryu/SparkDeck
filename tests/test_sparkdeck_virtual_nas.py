import asyncio
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from manager import DEFAULT_SETTINGS, Manager
from sparkdeck.virtual_nas import (
    DOWNLOAD_STAGING_RESERVE_BYTES,
    VIRTUAL_NAS_DOWNLOAD_CAPABILITY,
    VIRTUAL_NAS_DOWNLOAD_BASELINE_CAPABILITY,
    VirtualNAS,
    validate_model_id,
    validate_revision,
)


RESOLVED_REVISION = "a" * 40


def create_cached_model(hub: Path, model_id: str = "org/model") -> Path:
    owner, repository = model_id.split("/", 1)
    root = hub / f"models--{owner}--{repository}"
    (root / "blobs").mkdir(parents=True)
    (root / "snapshots" / "revision-1").mkdir(parents=True)
    (root / "blobs" / "weights").write_bytes(b"model-weights")
    snapshot = root / "snapshots" / "revision-1"
    (snapshot / "config.json").write_text("{}")
    (snapshot / "tokenizer.json").write_text("{}")
    (snapshot / "model.safetensors").write_bytes(b"model-weights")
    return root


async def bytes_stream(value: bytes):
    for offset in range(0, len(value), 7):
        yield value[offset:offset + 7]


class FakeResponse:
    status_code = 200

    def __init__(self, payload: bytes = b'{"ok":true}'):
        self.payload = payload
        self.closed = False

    async def aread(self):
        return self.payload

    async def aclose(self):
        self.closed = True


class FakeRegistry:
    def __init__(
        self, *, fail_target: str | None = None, slow: bool = False,
        block_upload: bool = False,
    ):
        self.nodes = {
            "worker-a": {"id": "worker-a", "name": "Worker A", "enabled": True},
            "worker-b": {"id": "worker-b", "name": "Worker B", "enabled": True},
        }
        self.fail_target = fail_target
        self.slow = slow
        self.block_upload = block_upload
        self.upload_started = asyncio.Event()
        self.release_upload = asyncio.Event()
        self.active = 0
        self.max_active = 0
        self.received: dict[str, bytes] = {}
        self.remote_models: dict[str, list[dict]] = {}

    def get(self, node_id):
        return self.nodes.get(node_id)

    async def probe(self, node, force=False):
        return {
            **node, "online": True,
            "capabilities": [
                VIRTUAL_NAS_DOWNLOAD_CAPABILITY,
                VIRTUAL_NAS_DOWNLOAD_BASELINE_CAPABILITY,
            ],
            "disk": {"free": 10 * 1024 * 1024 * 1024},
        }

    async def request(self, node_id, method, path, **_kwargs):
        return {
            "models": self.remote_models.get(node_id, []),
            "free_size": 10 * 1024 * 1024 * 1024,
        }

    async def open_stream(self, node_id, method, path, **kwargs):
        if node_id == self.fail_target:
            raise RuntimeError("simulated target outage")
        if method != "PUT":
            raise AssertionError("test registry only expects target uploads")
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        body = bytearray()
        try:
            async for chunk in kwargs["content"]:
                body.extend(chunk)
                self.upload_started.set()
                if self.block_upload:
                    await self.release_upload.wait()
                if self.slow:
                    await asyncio.sleep(0.02)
        finally:
            self.active -= 1
        self.received[node_id] = bytes(body)
        self.remote_models[node_id] = [{
            "model_id": "org/model", "size_bytes": len(body),
            "partial": False, "revisions": ["revision-1", "main"],
        }]
        return FakeResponse()


class InventoryAndArchiveTests(unittest.IsolatedAsyncioTestCase):
    def test_virtual_nas_is_disabled_by_default(self):
        self.assertIs(DEFAULT_SETTINGS["virtual_nas_enabled"], False)

    def test_download_uses_the_configured_hub_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            hub = Path(directory) / "configured-cache" / "hub"
            nas = VirtualNAS(
                Path(directory), lambda: hub, FakeRegistry(), lambda: True,
            )

            def download_into_cache(**kwargs):
                cache_dir = Path(kwargs["cache_dir"])
                create_cached_model(cache_dir)
                return str(
                    cache_dir / "models--org--model" / "snapshots" / "revision-1"
                )

            snapshot_download = Mock(side_effect=download_into_cache)
            huggingface_hub = Mock(snapshot_download=snapshot_download)
            with patch.dict(
                "sys.modules", {"huggingface_hub": huggingface_hub},
            ):
                result = nas.download_model("org/model", "revision-1")

            self.assertTrue(result["ok"])
            self.assertEqual(result["revision"], "revision-1")
            self.assertEqual(
                snapshot_download.call_args.kwargs["cache_dir"], str(hub.resolve()),
            )
            self.assertTrue((hub / "models--org--model").is_dir())

    async def test_mutable_revision_resolution_is_fresh_and_returns_commit_sha(self):
        with tempfile.TemporaryDirectory() as directory:
            first_sha = "a" * 40
            second_sha = "b" * 40
            api = Mock()
            api.model_info.side_effect = [
                Mock(sha=first_sha, siblings=[Mock(size=10), Mock(size=20)]),
                Mock(sha=second_sha, siblings=[Mock(size=10), Mock(size=20)]),
            ]
            huggingface_hub = Mock(HfApi=Mock(return_value=api))
            nas = VirtualNAS(
                Path(directory), lambda: Path(directory) / "hub",
                FakeRegistry(), lambda: True,
            )

            with patch.dict("sys.modules", {"huggingface_hub": huggingface_hub}):
                first = await nas.resolve_download_revision("org/model", "main")
                second = await nas.resolve_download_revision("org/model", "main")

            self.assertEqual(first["resolved_revision"], first_sha)
            self.assertEqual(second["resolved_revision"], second_sha)
            self.assertEqual(first["size_bytes"], 30)
            self.assertEqual(api.model_info.call_count, 2)

    async def test_revision_resolution_rejects_non_commit_hub_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            api = Mock()
            api.model_info.return_value = Mock(
                sha="main", siblings=[Mock(size=20)],
            )
            huggingface_hub = Mock(HfApi=Mock(return_value=api))
            nas = VirtualNAS(
                Path(directory), lambda: Path(directory) / "hub",
                FakeRegistry(), lambda: True,
            )

            with (
                patch.dict("sys.modules", {"huggingface_hub": huggingface_hub}),
                self.assertRaisesRegex(RuntimeError, "immutable revision"),
            ):
                await nas.resolve_download_revision("org/model", "main")

    def test_pinned_download_writes_requested_ref_alias(self):
        with tempfile.TemporaryDirectory() as directory:
            hub = Path(directory) / "hub"
            resolved = "a" * 40

            def download_into_cache(**kwargs):
                repository = create_cached_model(hub)
                original = repository / "snapshots" / "revision-1"
                original.rename(repository / "snapshots" / resolved)
                return str(repository / "snapshots" / resolved)

            snapshot_download = Mock(side_effect=download_into_cache)
            huggingface_hub = Mock(snapshot_download=snapshot_download)
            nas = VirtualNAS(
                Path(directory), lambda: hub, FakeRegistry(), lambda: True,
            )

            with patch.dict("sys.modules", {"huggingface_hub": huggingface_hub}):
                result = nas.download_model(
                    "org/model", resolved, requested_revision="main",
                )

            model = nas.inventory()[0]
            self.assertTrue(result["ok"])
            self.assertEqual(model["revision_refs"], {"main": resolved})
            self.assertEqual(
                (hub / "models--org--model" / "refs" / "main").read_text().strip(),
                resolved,
            )

    async def test_selective_download_uses_selected_capacity_and_stays_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            hub = Path(directory) / "hub"
            resolved = "b" * 40
            selected_name = "MODEL-00001-OF-00001.GGUF"
            selected_size = 8
            repository_size = selected_size + 500_000_000
            repository = hub / "models--org--model"
            snapshot = repository / "snapshots" / resolved

            def download_file_into_cache(**kwargs):
                (repository / "blobs").mkdir(parents=True, exist_ok=True)
                snapshot.mkdir(parents=True, exist_ok=True)
                self.assertEqual(kwargs["filename"], selected_name)
                (snapshot / selected_name).write_bytes(b"12345678")
                return str(snapshot / selected_name)

            def download_full_snapshot(**kwargs):
                (snapshot / "README.md").write_text("full", encoding="utf-8")
                return str(snapshot)

            api = Mock()
            api.model_info.return_value = Mock(
                sha=resolved,
                siblings=[
                    Mock(rfilename=selected_name, size=selected_size),
                    Mock(rfilename="README.md", size=repository_size - selected_size),
                ],
            )
            hf_hub_download = Mock(side_effect=download_file_into_cache)
            snapshot_download = Mock(side_effect=download_full_snapshot)
            huggingface_hub = Mock(
                HfApi=Mock(return_value=api),
                hf_hub_download=hf_hub_download,
                snapshot_download=snapshot_download,
            )
            nas = VirtualNAS(
                Path(directory), lambda: hub, FakeRegistry(), lambda: True,
            )
            selected_required = (
                selected_size * 2 + DOWNLOAD_STAGING_RESERVE_BYTES
            )
            nas.free_bytes = Mock(return_value=selected_required)

            with patch.dict("sys.modules", {"huggingface_hub": huggingface_hub}):
                result = await nas.download_model_files_checked(
                    "org/model", resolved, [selected_name],
                    requested_revision="release-gguf",
                )

                self.assertTrue(result["ok"])
                self.assertEqual(result["size_bytes"], selected_size)
                self.assertTrue(
                    (snapshot / ".sparkdeck-selective.incomplete").is_file()
                )
                partial_model = nas.inventory()[0]
                self.assertTrue(partial_model["partial"])
                self.assertEqual(partial_model["revision"], "release-gguf")
                self.assertEqual(partial_model["revision_refs"], {})
                self.assertEqual(
                    partial_model["partial_revision_refs"],
                    {"release-gguf": resolved},
                )
                self.assertEqual(hf_hub_download.call_count, 1)
                self.assertEqual(
                    hf_hub_download.call_args.kwargs["filename"], selected_name,
                )

                full = nas.download_model("org/model", resolved)

            self.assertTrue(full["ok"])
            self.assertEqual(snapshot_download.call_count, 1)
            self.assertEqual(hf_hub_download.call_count, 1)
            self.assertFalse(
                (snapshot / ".sparkdeck-selective.incomplete").exists()
            )
            completed_model = nas.inventory()[0]
            self.assertFalse(completed_model["partial"])
            self.assertEqual(
                completed_model["revision_refs"], {"release-gguf": resolved},
            )

    async def test_selective_download_credits_only_selected_resumable_blob(self):
        with tempfile.TemporaryDirectory() as directory:
            hub = Path(directory) / "hub"
            repository = hub / "models--org--model"
            blobs = repository / "blobs"
            blobs.mkdir(parents=True)
            resolved = "b" * 40
            selected_name = "model.gguf"
            selected_size = 8
            selected_sha = "1" * 64
            incomplete = blobs / f"{selected_sha}.incomplete"
            incomplete.write_bytes(b"123")
            snapshot = repository / "snapshots" / resolved

            def download_file_into_cache(**kwargs):
                incomplete.unlink()
                snapshot.mkdir(parents=True, exist_ok=True)
                (snapshot / selected_name).write_bytes(b"12345678")
                return str(snapshot / selected_name)

            api = Mock()
            api.model_info.return_value = Mock(
                sha=resolved,
                siblings=[Mock(
                    rfilename=selected_name,
                    size=selected_size,
                    lfs=Mock(sha256=selected_sha),
                    blob_id="2" * 40,
                )],
            )
            hf_hub_download = Mock(side_effect=download_file_into_cache)
            huggingface_hub = Mock(
                HfApi=Mock(return_value=api),
                hf_hub_download=hf_hub_download,
            )
            nas = VirtualNAS(
                Path(directory), lambda: hub, FakeRegistry(), lambda: True,
            )
            nas.free_bytes = Mock(return_value=(
                selected_size * 2 + DOWNLOAD_STAGING_RESERVE_BYTES - 3
            ))

            with patch.dict("sys.modules", {"huggingface_hub": huggingface_hub}):
                result = await nas.download_model_files_checked(
                    "org/model", resolved, [selected_name],
                )

            self.assertTrue(result["ok"])
            hf_hub_download.assert_called_once()

    async def test_selective_download_does_not_credit_unrelated_resumable_blob(self):
        with tempfile.TemporaryDirectory() as directory:
            hub = Path(directory) / "hub"
            repository = hub / "models--org--model"
            blobs = repository / "blobs"
            blobs.mkdir(parents=True)
            resolved = "b" * 40
            selected_name = "model.gguf"
            selected_size = 8
            selected_blob_id = "1" * 40
            (blobs / f"{'2' * 40}.incomplete").write_bytes(b"123")

            api = Mock()
            api.model_info.return_value = Mock(
                sha=resolved,
                siblings=[Mock(
                    rfilename=selected_name,
                    size=selected_size,
                    lfs=None,
                    blob_id=selected_blob_id,
                )],
            )
            hf_hub_download = Mock()
            huggingface_hub = Mock(
                HfApi=Mock(return_value=api),
                hf_hub_download=hf_hub_download,
            )
            nas = VirtualNAS(
                Path(directory), lambda: hub, FakeRegistry(), lambda: True,
            )
            nas.free_bytes = Mock(return_value=(
                selected_size * 2 + DOWNLOAD_STAGING_RESERVE_BYTES - 3
            ))

            with (
                patch.dict("sys.modules", {"huggingface_hub": huggingface_hub}),
                self.assertRaisesRegex(RuntimeError, "insufficient free cache space"),
            ):
                await nas.download_model_files_checked(
                    "org/model", resolved, [selected_name],
                )

            hf_hub_download.assert_not_called()

    async def test_inventory_lists_complete_and_partial_models_without_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            hub = Path(directory) / "hub"
            complete = create_cached_model(hub)
            (complete / "blobs" / "next.incomplete").write_bytes(b"partial")
            next_revision = "b" * 40
            older_revision = "c" * 40
            empty_revision = "d" * 40
            lock_only_revision = "e" * 40
            next_snapshot = complete / "snapshots" / next_revision
            next_snapshot.mkdir()
            completed_blob = complete / "blobs" / "next-complete"
            completed_blob.write_bytes(b"completed")
            try:
                (next_snapshot / "config.json").symlink_to(completed_blob)
            except OSError:
                (next_snapshot / "config.json").write_bytes(b"completed")
            older_snapshot = complete / "snapshots" / older_revision
            older_snapshot.mkdir()
            older_blob = complete / "blobs" / "older-complete"
            older_blob.write_bytes(b"older")
            try:
                (older_snapshot / "config.json").symlink_to(older_blob)
            except OSError:
                (older_snapshot / "config.json").write_bytes(b"older")
            (complete / "refs").mkdir()
            (complete / "refs" / "main").write_text("revision-1")
            (complete / "refs" / "stale").write_text("missing-revision")
            (complete / "snapshots" / empty_revision).mkdir()
            (complete / "refs" / "empty").write_text(empty_revision)
            lock_only = complete / "snapshots" / lock_only_revision
            lock_only.mkdir()
            (lock_only / "download.lock").write_text("locked", encoding="utf-8")
            (complete / "refs" / "locked").write_text(lock_only_revision)
            non_commit = complete / "snapshots" / "release-partial"
            non_commit.mkdir()
            (non_commit / "config.json").write_text("{}", encoding="utf-8")
            (complete / "refs" / "non-commit").write_text("release-partial")
            (hub / "models--partial--repo" / "snapshots").mkdir(parents=True)
            nas = VirtualNAS(Path(directory), lambda: hub, FakeRegistry(), lambda: False)

            models = nas.inventory()

            self.assertEqual(
                [item["model_id"] for item in models],
                ["org/model", "partial/repo"],
            )
            self.assertEqual(models[0]["revisions"], ["main", "revision-1"])
            self.assertEqual(models[0]["revision_refs"], {"main": "revision-1"})
            self.assertFalse(models[0]["partial"])
            self.assertTrue(models[0]["has_partial_download"])
            self.assertEqual(
                models[0]["partial_size_bytes"],
                len(b"completed") + len(b"older"),
            )
            self.assertEqual(models[0]["partial_revision_size_bytes"], {
                next_revision: len(b"completed"),
                older_revision: len(b"older"),
            })
            self.assertEqual(
                models[0]["partial_revisions"],
                [next_revision, older_revision],
            )
            self.assertNotIn("empty", models[0]["partial_revision_refs"])
            self.assertNotIn("locked", models[0]["partial_revision_refs"])
            self.assertNotIn("non-commit", models[0]["partial_revision_refs"])
            self.assertTrue(models[1]["partial"])
            self.assertEqual(models[1]["revisions"], [])
            self.assertGreater(models[0]["size_bytes"], 0)
            self.assertNotIn(str(complete), json.dumps(models))
            self.assertNotIn("path", models[0])

    async def test_partial_revision_survives_when_its_blob_is_shared_with_complete_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            hub = Path(directory) / "hub"
            repository = hub / "models--org--model"
            blobs = repository / "blobs"
            blobs.mkdir(parents=True)
            shared_blob = blobs / "shared-content-hash"
            shared_blob.write_bytes(b"shared-weights")
            complete_revision = "c" * 40
            partial_revision = "d" * 40
            complete = repository / "snapshots" / complete_revision
            partial = repository / "snapshots" / partial_revision
            complete.mkdir(parents=True)
            partial.mkdir(parents=True)
            complete.joinpath("model.gguf").symlink_to(
                Path("../../blobs/shared-content-hash")
            )
            partial.joinpath("model-00001-of-00002.gguf").symlink_to(
                Path("../../blobs/shared-content-hash")
            )
            partial.joinpath(".sparkdeck-selective.incomplete").write_text(
                "selective cache marker", encoding="utf-8",
            )
            refs = repository / "refs"
            refs.mkdir()
            refs.joinpath("release-shared").write_text(
                partial_revision, encoding="utf-8",
            )
            nas = VirtualNAS(
                Path(directory), lambda: hub, FakeRegistry(), lambda: True,
            )

            model = nas.inventory()[0]

            self.assertFalse(model["partial"])
            self.assertTrue(model["has_partial_download"])
            self.assertEqual(model["partial_revisions"], [partial_revision])
            self.assertEqual(
                model["partial_revision_size_bytes"], {partial_revision: 0},
            )
            self.assertEqual(
                model["partial_revision_refs"],
                {"release-shared": partial_revision},
            )
            self.assertEqual(model["revision"], "release-shared")

    async def test_inventory_accepts_complete_windows_snapshot_without_blob_links(self):
        with tempfile.TemporaryDirectory() as directory:
            hub = Path(directory) / "hub"
            snapshot = (
                hub / "models--org--windows-model" / "snapshots" / "revision-1"
            )
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text("{}", encoding="utf-8")
            (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")
            (snapshot / "model.safetensors").write_bytes(b"complete weights")
            nas = VirtualNAS(
                Path(directory), lambda: hub, FakeRegistry(), lambda: False,
            )

            model = nas.inventory()[0]

            self.assertFalse(model["partial"])
            self.assertFalse(model["has_partial_download"])
            self.assertEqual(model["revisions"], ["revision-1"])

    async def test_complete_model_ignores_unassigned_incomplete_blob(self):
        with tempfile.TemporaryDirectory() as directory:
            hub = Path(directory) / "hub"
            repository = create_cached_model(hub)
            (repository / "blobs" / "orphan.incomplete").write_bytes(b"stale")
            nas = VirtualNAS(
                Path(directory), lambda: hub, FakeRegistry(), lambda: False,
            )

            model = nas.inventory()[0]

            self.assertFalse(model["partial"])
            self.assertFalse(model["has_partial_download"])
            self.assertEqual(model["partial_size_bytes"], 0)

    def test_inventory_requires_matching_index_for_complete_transformer_shards(self):
        cases = (
            ("model", ".safetensors", None, False),
            ("pytorch_model", ".bin", "other.bin.index.json", False),
            ("model", ".safetensors", None, True),
        )
        for prefix, suffix, index_name, with_gguf in cases:
            with (
                self.subTest(suffix=suffix, with_gguf=with_gguf),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                hub = root / "hub"
                snapshot = (
                    hub / "models--org--model" / "snapshots" / "revision-1"
                )
                snapshot.mkdir(parents=True)
                (snapshot.parent.parent / "blobs").mkdir()
                (snapshot / "config.json").write_text("{}", encoding="utf-8")
                (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")
                first = f"{prefix}-00001-of-00002{suffix}"
                second = f"{prefix}-00002-of-00002{suffix}"
                (snapshot / first).write_bytes(b"one")
                (snapshot / second).write_bytes(b"two")
                if with_gguf:
                    (snapshot / "model.gguf").write_bytes(b"standalone")
                if index_name:
                    (snapshot / index_name).write_text(json.dumps({
                        "weight_map": {"a": first, "b": second},
                    }), encoding="utf-8")
                nas = VirtualNAS(root, lambda: hub, FakeRegistry(), lambda: False)

                self.assertTrue(nas.inventory()[0]["partial"])

    async def test_streamed_export_import_uses_exact_repository(self):
        with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as target_dir:
            source_hub = Path(source_dir) / "hub"
            target_hub = Path(target_dir) / "hub"
            create_cached_model(source_hub)
            source = VirtualNAS(Path(source_dir), lambda: source_hub, FakeRegistry(), lambda: True)
            target = VirtualNAS(Path(target_dir), lambda: target_hub, FakeRegistry(), lambda: True)

            result = await target.import_model("org/model", source.export_model("org/model"))

            self.assertTrue(result["ok"])
            copied = target_hub / "models--org--model"
            self.assertEqual((copied / "blobs" / "weights").read_bytes(), b"model-weights")
            with self.assertRaises(FileExistsError):
                await target.import_model("org/model", bytes_stream(b"unused"))

    async def test_delete_is_blocked_while_local_export_is_reserved(self):
        with tempfile.TemporaryDirectory() as directory:
            hub = Path(directory) / "hub"
            create_cached_model(hub)
            nas = VirtualNAS(Path(directory), lambda: hub, FakeRegistry(), lambda: True)

            stream = nas.export_model("org/model")
            with self.assertRaisesRegex(RuntimeError, "transfer"):
                nas.delete_model("org/model")
            await stream.aclose()

            self.assertTrue(nas.delete_model("org/model")["ok"])

    async def test_import_rejects_traversal_and_escaping_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            hub = Path(directory) / "hub"
            nas = VirtualNAS(Path(directory), lambda: hub, FakeRegistry(), lambda: True)
            archive_bytes = io.BytesIO()
            with tarfile.open(fileobj=archive_bytes, mode="w") as archive:
                root = tarfile.TarInfo("models--org--model")
                root.type = tarfile.DIRTYPE
                archive.addfile(root)
                link = tarfile.TarInfo("models--org--model/snapshots/rev/weights")
                link.type = tarfile.SYMTYPE
                link.linkname = "../../../../outside"
                archive.addfile(link)

            with self.assertRaisesRegex(ValueError, "link escapes"):
                await nas.import_model("org/model", bytes_stream(archive_bytes.getvalue()))

            self.assertFalse((Path(directory) / "outside").exists())
            self.assertFalse((hub / "models--org--model").exists())

    def test_model_id_validation_rejects_paths_and_encoded_cache_names(self):
        for value in ("../model", "org/../model", "/absolute", "org/repo/extra", "org--x/repo"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_model_id(value)

    def test_revision_validation_accepts_hub_refs_and_rejects_unsafe_git_refs(self):
        self.assertEqual(validate_revision("v1.0+cuda"), "v1.0+cuda")
        self.assertEqual(validate_revision("release@candidate"), "release@candidate")
        for value in (
            "@", "feature@{one}", "../main", "a//b", "a b", "a\\b",
            "a~b", "a^b", "a:b", "a?b", "a*b", "a[b", "trailing.",
            "refs/name.lock", "/main", "main/", "line\nbreak",
        ):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_revision(value)


class QueueTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.hub = Path(self.temp.name) / "hub"
        create_cached_model(self.hub)

    async def asyncTearDown(self):
        self.temp.cleanup()

    async def wait_final(self, nas: VirtualNAS, count: int, timeout: float = 5):
        async def wait():
            while True:
                final = [job for job in nas.jobs if job["status"] in {"completed", "failed", "canceled"}]
                if len(final) == count:
                    return final
                await asyncio.sleep(0.01)
        return await asyncio.wait_for(wait(), timeout)

    async def test_multi_target_jobs_are_globally_serialized_and_persisted(self):
        registry = FakeRegistry(slow=True)
        nas = VirtualNAS(Path(self.temp.name), lambda: self.hub, registry, lambda: True)

        result = await nas.queue_transfer(
            "org/model", "local", ["worker-a", "worker-b"],
        )
        final = await self.wait_final(nas, 2)

        self.assertEqual(len(result["job_ids"]), 2)
        self.assertTrue(all(job["status"] == "completed" for job in final))
        self.assertEqual(registry.max_active, 1)
        self.assertEqual(set(registry.received), {"worker-a", "worker-b"})
        saved = json.loads((Path(self.temp.name) / "virtual_nas_transfers.json").read_text())
        self.assertTrue(all(job["status"] == "completed" for job in saved))
        await nas.stop()

    async def test_concurrent_queues_cannot_persist_duplicate_targets(self):
        registry = FakeRegistry()
        nas = VirtualNAS(
            Path(self.temp.name), lambda: self.hub, registry, lambda: True,
        )
        nas.start = Mock()
        first_target_check = asyncio.Event()
        release_target_check = asyncio.Event()
        original_storage = nas._node_storage

        async def blocked_storage(node_id):
            if node_id == "worker-a":
                first_target_check.set()
                await release_target_check.wait()
            return await original_storage(node_id)

        nas._node_storage = blocked_storage
        first = asyncio.create_task(
            nas.queue_transfer("org/model", "local", ["worker-a"]),
        )
        await asyncio.wait_for(first_target_check.wait(), 2)
        second = asyncio.create_task(
            nas.queue_transfer("org/model", "local", ["worker-a"]),
        )
        await asyncio.sleep(0.05)
        release_target_check.set()

        results = await asyncio.gather(first, second, return_exceptions=True)

        self.assertEqual(sum(isinstance(item, dict) for item in results), 1)
        self.assertEqual(sum(isinstance(item, ValueError) for item in results), 1)
        self.assertEqual(len(nas.jobs), 1)
        self.assertEqual(nas.jobs[0]["target_node_id"], "worker-a")

    async def test_restart_recovers_running_job_to_queue(self):
        path = Path(self.temp.name) / "virtual_nas_transfers.json"
        path.write_text(json.dumps([{
            "id": "job-1", "model_id": "org/model",
            "source_node_id": "local", "target_node_id": "worker-a",
            "status": "running", "bytes_total": 20, "bytes_transferred": 5,
            "created_at": 1, "started_at": 2,
        }]))

        nas = VirtualNAS(Path(self.temp.name), lambda: self.hub, FakeRegistry(), lambda: True)

        self.assertEqual(nas.jobs[0]["status"], "queued")
        self.assertEqual(nas.jobs[0]["bytes_transferred"], 0)
        self.assertEqual(json.loads(path.read_text())[0]["status"], "queued")

    async def test_running_transfer_can_be_canceled(self):
        registry = FakeRegistry(block_upload=True)
        nas = VirtualNAS(Path(self.temp.name), lambda: self.hub, registry, lambda: True)
        result = await nas.queue_transfer("org/model", "local", ["worker-a"])
        job_id = result["job_ids"][0]
        await asyncio.wait_for(registry.upload_started.wait(), 2)

        canceled = await nas.cancel_transfer(job_id)
        registry.release_upload.set()
        await self.wait_final(nas, 1)

        self.assertEqual(canceled["status"], "canceled")
        self.assertEqual(nas.jobs[0]["status"], "canceled")
        await nas.stop()

    async def test_target_failure_is_recorded(self):
        registry = FakeRegistry(fail_target="worker-a")
        nas = VirtualNAS(Path(self.temp.name), lambda: self.hub, registry, lambda: True)
        await nas.queue_transfer("org/model", "local", ["worker-a"])

        final = await self.wait_final(nas, 1)

        self.assertEqual(final[0]["status"], "failed")
        self.assertIn("simulated target outage", final[0]["error"])
        await nas.stop()

    async def test_stopping_dispatcher_requeues_running_transfer(self):
        registry = FakeRegistry(block_upload=True)
        nas = VirtualNAS(Path(self.temp.name), lambda: self.hub, registry, lambda: True)
        await nas.queue_transfer("org/model", "local", ["worker-a"])
        await asyncio.wait_for(registry.upload_started.wait(), 2)

        await nas.stop()

        self.assertEqual(nas.jobs[0]["status"], "queued")
        self.assertEqual(nas._active, {})

    async def test_existing_target_is_rejected_without_persisting_jobs(self):
        registry = FakeRegistry()

        async def inventory(node_id, method, path, **kwargs):
            return {"models": [{"model_id": "org/model", "size_bytes": 10}]}

        registry.request = inventory
        nas = VirtualNAS(Path(self.temp.name), lambda: self.hub, registry, lambda: True)

        with self.assertRaisesRegex(FileExistsError, "already exists"):
            await nas.queue_transfer("org/model", "local", ["worker-a"])

        self.assertEqual(nas.jobs, [])
        await nas.stop()

    async def test_insufficient_target_capacity_is_rejected_without_jobs(self):
        registry = FakeRegistry()

        async def low_capacity(node_id, method, path, **kwargs):
            return {"models": [], "free_size": 1}

        registry.request = low_capacity
        nas = VirtualNAS(Path(self.temp.name), lambda: self.hub, registry, lambda: True)

        with self.assertRaisesRegex(RuntimeError, "insufficient free disk space"):
            await nas.queue_transfer("org/model", "local", ["worker-a"])

        self.assertEqual(nas.jobs, [])

    async def test_unknown_target_capacity_fails_closed(self):
        registry = FakeRegistry()

        async def unknown_capacity(node_id, method, path, **kwargs):
            return {"models": []}

        registry.request = unknown_capacity
        nas = VirtualNAS(Path(self.temp.name), lambda: self.hub, registry, lambda: True)

        with self.assertRaisesRegex(RuntimeError, "did not report free disk capacity"):
            await nas.queue_transfer("org/model", "local", ["worker-a"])

        self.assertEqual(nas.jobs, [])

    async def test_capacity_uses_cache_mount_instead_of_generic_node_disk(self):
        registry = FakeRegistry()

        async def small_root_disk(node, force=False):
            return {**node, "online": True, "disk": {"free": 1}}

        registry.probe = small_root_disk
        nas = VirtualNAS(Path(self.temp.name), lambda: self.hub, registry, lambda: True)

        result = await nas.queue_transfer("org/model", "local", ["worker-a"])

        self.assertEqual(len(result["jobs"]), 1)
        await nas.stop()

    async def test_agent_checked_download_fails_closed_on_local_cache_capacity(self):
        nas = VirtualNAS(
            Path(self.temp.name), lambda: self.hub, FakeRegistry(), lambda: True,
        )
        nas.estimate_download_size = AsyncMock(return_value=100)
        nas.free_bytes = Mock(return_value=1)
        nas.download_model = Mock()

        with self.assertRaisesRegex(RuntimeError, "insufficient free cache space"):
            await nas.download_model_checked(
                "other/model", RESOLVED_REVISION, "ephemeral", "main",
            )

        nas.estimate_download_size.assert_awaited_once_with(
            "other/model", RESOLVED_REVISION, "ephemeral", force_refresh=True,
        )
        nas.download_model.assert_not_called()

    async def test_remote_download_requires_advertised_agent_capability(self):
        registry = FakeRegistry()

        async def legacy_probe(node, force=False):
            return {**node, "online": True, "capabilities": []}

        registry.probe = legacy_probe
        nas = VirtualNAS(
            Path(self.temp.name), lambda: self.hub, registry, lambda: True,
        )
        nas.start = Mock()

        with self.assertRaisesRegex(RuntimeError, "must be updated"):
            await nas.queue_download_and_transfer(
                "org/model", RESOLVED_REVISION, "worker-a", [], 100,
                requested_revision="main",
            )

        self.assertEqual(nas.jobs, [])

    async def test_remote_download_reuses_fresh_probe_instead_of_forcing_a_second_probe(self):
        registry = FakeRegistry()
        force_values = []

        async def transient_probe(node, force=False):
            force_values.append(force)
            if force:
                return {**node, "online": False}
            return {
                **node, "online": True,
                "capabilities": [
                    VIRTUAL_NAS_DOWNLOAD_CAPABILITY,
                    VIRTUAL_NAS_DOWNLOAD_BASELINE_CAPABILITY,
                ],
            }

        registry.probe = transient_probe
        nas = VirtualNAS(
            Path(self.temp.name), lambda: self.hub, registry, lambda: True,
        )
        nas.start = Mock()
        nas._node_storage = AsyncMock(return_value={
            "models": [{
                "model_id": "org/model", "size_bytes": 97,
                "partial": True, "revisions": [],
                "partial_revision_size_bytes": {RESOLVED_REVISION: 97},
            }],
            "free_size": 10 * 1024 * 1024 * 1024,
        })

        result = await nas.queue_download_and_transfer(
            "org/model", RESOLVED_REVISION, "worker-a", [], 100,
            requested_revision="main", require_partial_cache=True,
        )

        self.assertEqual(force_values, [False])
        self.assertEqual(result["jobs"][0]["bytes_transferred"], 97)

    async def test_finish_download_never_starts_without_the_partial_cache(self):
        nas = VirtualNAS(
            Path(self.temp.name), lambda: self.hub, FakeRegistry(), lambda: True,
        )
        nas.start = Mock()

        with self.assertRaisesRegex(LookupError, "partial model cache no longer exists"):
            await nas.queue_download_and_transfer(
                "other/model", RESOLVED_REVISION, "local", [], 100,
                requested_revision="release-1", require_partial_cache=True,
            )

        self.assertEqual(nas.jobs, [])

    async def test_resume_queue_capacity_uses_attempt_baseline_credit(self):
        expected = 100
        cached = 25
        required = expected * 2 + DOWNLOAD_STAGING_RESERVE_BYTES - cached
        for free_bytes, succeeds in ((required, True), (required - 1, False)):
            with self.subTest(free_bytes=free_bytes), tempfile.TemporaryDirectory() as directory:
                nas = VirtualNAS(
                    Path(directory), lambda: Path(directory) / "hub",
                    FakeRegistry(), lambda: True,
                )
                nas.start = Mock()
                nas._node_storage = AsyncMock(return_value={
                    "models": [{
                        "model_id": "org/model", "size_bytes": 125,
                        "partial": False, "has_partial_download": True,
                        "partial_size_bytes": cached,
                    }],
                    "free_size": free_bytes,
                })
                if succeeds:
                    result = await nas.queue_download_and_transfer(
                        "org/model", RESOLVED_REVISION, "local", [], expected,
                        requested_revision="release-1", require_partial_cache=True,
                        download_cache_baseline_bytes=100,
                    )
                    self.assertEqual(len(result["jobs"]), 1)
                    self.assertEqual(
                        result["jobs"][0]["bytes_transferred"], cached,
                    )
                else:
                    with self.assertRaisesRegex(RuntimeError, "insufficient free cache space"):
                        await nas.queue_download_and_transfer(
                            "org/model", RESOLVED_REVISION, "local", [], expected,
                            requested_revision="release-1", require_partial_cache=True,
                            download_cache_baseline_bytes=100,
                        )
                    self.assertEqual(nas.jobs, [])

    async def test_agent_resume_capacity_uses_attempt_baseline_credit(self):
        expected = 100
        cached = 25
        required = expected * 2 + DOWNLOAD_STAGING_RESERVE_BYTES - cached
        for free_bytes, succeeds in ((required, True), (required - 1, False)):
            with self.subTest(free_bytes=free_bytes), tempfile.TemporaryDirectory() as directory:
                nas = VirtualNAS(
                    Path(directory), lambda: Path(directory) / "hub",
                    FakeRegistry(), lambda: True,
                )
                nas.inventory = Mock(return_value=[{
                    "model_id": "org/model", "size_bytes": 125,
                    "partial": False, "has_partial_download": True,
                    "partial_size_bytes": cached, "revisions": [],
                }])
                nas.free_bytes = Mock(return_value=free_bytes)
                nas.estimate_download_size = AsyncMock(return_value=expected)
                nas.download_model = Mock(return_value={
                    "ok": True, "model_id": "org/model", "size_bytes": expected,
                })
                if succeeds:
                    await nas.download_model_checked(
                        "org/model", RESOLVED_REVISION,
                        requested_revision="release-1",
                        download_cache_baseline_bytes=100,
                    )
                    nas.download_model.assert_called_once()
                else:
                    with self.assertRaisesRegex(RuntimeError, "insufficient free cache space"):
                        await nas.download_model_checked(
                            "org/model", RESOLVED_REVISION,
                            requested_revision="release-1",
                            download_cache_baseline_bytes=100,
                        )
                    nas.download_model.assert_not_called()


class DeleteGuardTests(unittest.IsolatedAsyncioTestCase):
    async def test_stale_requested_ref_cannot_seed_pinned_workflow(self):
        manager = Manager.__new__(Manager)
        manager.settings = {"virtual_nas_enabled": True}
        manager.model_cache_inventory = AsyncMock(return_value=[{
            "id": "stale", "name": "Stale", "online": True,
            "cache_free_size": 10**9,
            "virtual_nas_download_capable": True,
            "models": [{
                "model_id": "org/model", "size_bytes": 20,
                "partial": False,
                "revisions": ["main", "a" * 40, "b" * 40],
                "revision_refs": {"main": "b" * 40},
            }],
        }])
        manager.virtual_nas_transfers = Mock(return_value={"items": []})
        manager.virtual_nas = Mock()
        manager.virtual_nas.resolve_download_revision = AsyncMock(return_value={
            "requested_revision": "main",
            "resolved_revision": "a" * 40,
            "size_bytes": 20,
        })

        result = await manager.virtual_nas_transfer_preflight("org/model", "main")

        self.assertIsNone(result["source"])
        self.assertFalse(result["targets"][0]["has_required_weights"])

    async def test_legacy_agent_is_transfer_capable_but_not_download_capable(self):
        manager = Manager.__new__(Manager)
        manager.settings = {"virtual_nas_enabled": True}
        manager.model_cache_inventory = AsyncMock(return_value=[{
            "id": "legacy", "name": "Legacy", "online": True,
            "cache_free_size": 10**9,
            "virtual_nas_download_capable": False, "models": [],
        }])
        manager.virtual_nas_transfers = Mock(return_value={"items": []})
        manager.virtual_nas = Mock()
        manager.virtual_nas.resolve_download_revision = AsyncMock(return_value={
            "requested_revision": "main",
            "resolved_revision": RESOLVED_REVISION,
            "size_bytes": 20,
        })

        result = await manager.virtual_nas_transfer_preflight("org/model", "main")

        self.assertFalse(result["targets"][0]["download_eligible"])
        self.assertIn("updated", result["targets"][0]["download_reason"])

    async def test_manager_queues_resumable_download_for_partial_cache(self):
        manager = Manager.__new__(Manager)
        manager.settings = {}
        manager.node_registry = FakeRegistry()
        manager.virtual_nas = Mock()
        manager.virtual_nas.list_transfers.return_value = {"items": [
            {
                "id": "failed-download", "kind": "download",
                "model_id": "org/model", "target_node_id": "worker-a",
                "requested_revision": "release-1", "revision": RESOLVED_REVISION,
                "status": "failed", "started_at": 4,
                "download_attempted_at": 4.5,
                "legacy_download_attempt_tracking": False, "created_at": 5,
                "bytes_total": 20, "download_cache_baseline_bytes": 3,
            },
            {
                "id": "failed-before-attempt", "kind": "download",
                "model_id": "org/model", "target_node_id": "worker-a",
                "requested_revision": "release-2", "revision": "b" * 40,
                "status": "failed", "started_at": 6,
                "download_attempted_at": None,
                "legacy_download_attempt_tracking": False, "created_at": 7,
            },
            {
                "id": "canceled-before-start", "kind": "download",
                "model_id": "org/model", "target_node_id": "worker-a",
                "requested_revision": "release-3", "revision": "c" * 40,
                "status": "canceled", "started_at": None,
                "download_attempted_at": None,
                "legacy_download_attempt_tracking": False, "created_at": 8,
            },
        ]}
        manager.virtual_nas.resolve_download_revision = AsyncMock(return_value={
            "requested_revision": "release-1",
            "resolved_revision": RESOLVED_REVISION,
            "size_bytes": 20,
        })
        manager.virtual_nas_transfer_preflight = AsyncMock(return_value={
            "resolved_revision": RESOLVED_REVISION,
            "download": {"size_bytes": 20},
            "targets": [{
                "node_id": "worker-a",
                "has_partial_model_cache": True,
                "download_eligible": True,
            }],
        })
        manager.virtual_nas.queue_download_and_transfer = AsyncMock(return_value={
            "job_ids": ["download-1"],
            "jobs": [{
                "id": "download-1", "kind": "download", "model_id": "org/model",
                "source_node_id": "huggingface", "target_node_id": "worker-a",
                "status": "queued", "bytes_total": 20, "bytes_transferred": 0,
                "created_at": 1,
            }],
        })

        result = await manager.queue_virtual_nas_download(
            "org/model", "worker-a",
        )

        self.assertEqual(result["job_ids"], ["download-1"])
        manager.virtual_nas.queue_download_and_transfer.assert_awaited_once_with(
            "org/model", RESOLVED_REVISION, "worker-a", [], 20,
            requested_revision="release-1", require_partial_cache=True,
            download_cache_baseline_bytes=3,
        )
        manager.virtual_nas.resolve_download_revision.assert_not_awaited()
        manager.virtual_nas_transfer_preflight.assert_awaited_once_with(
            "org/model", "release-1", {
                "requested_revision": "release-1",
                "resolved_revision": RESOLVED_REVISION,
                "size_bytes": 20,
                "resume_node_id": "worker-a",
                "download_cache_baseline_bytes": 3,
            },
        )

        manager.virtual_nas_transfer_preflight.return_value = {
            "resolved_revision": RESOLVED_REVISION,
            "download": {"size_bytes": 20},
            "targets": [{
                "node_id": "worker-a",
                "has_partial_model_cache": False,
                "download_eligible": False,
            }],
        }
        with self.assertRaisesRegex(LookupError, "partial model cache not found"):
            await manager.queue_virtual_nas_download(
                "org/model", "worker-a",
            )
        self.assertEqual(
            manager.virtual_nas.queue_download_and_transfer.await_count, 1,
        )

    async def test_manager_finishes_selective_cache_at_its_pinned_non_main_revision(self):
        manager = Manager.__new__(Manager)
        manager.settings = {"virtual_nas_enabled": True}
        manager.virtual_nas = Mock()
        manager.virtual_nas.list_transfers.return_value = {"items": []}
        manager.virtual_nas.resolve_download_revision = AsyncMock()
        manager.virtual_nas.estimate_download_size = AsyncMock(return_value=100)
        manager.virtual_nas.queue_download_and_transfer = AsyncMock(return_value={
            "job_ids": ["download-1"],
            "jobs": [{
                "id": "download-1", "kind": "download", "model_id": "org/model",
                "source_node_id": "huggingface", "target_node_id": "local",
                "status": "queued", "bytes_total": 100, "bytes_transferred": 0,
                "created_at": 1,
            }],
        })
        manager.model_cache_inventory = AsyncMock(return_value=[{
            "id": "local", "name": "Controller", "online": True,
            "cache_free_size": 10**9,
            "models": [{
                "model_id": "org/model", "size_bytes": 8,
                "partial": True, "has_partial_download": True,
                "revision": "release-gguf",
                "partial_revision_refs": {"release-gguf": RESOLVED_REVISION},
                "partial_revisions": [RESOLVED_REVISION],
                "partial_revision_size_bytes": {RESOLVED_REVISION: 0},
            }],
        }])
        manager.virtual_nas_transfer_preflight = AsyncMock(return_value={
            "resolved_revision": RESOLVED_REVISION,
            "download": {"size_bytes": 100},
            "targets": [{
                "node_id": "local", "has_partial_model_cache": True,
                "download_eligible": True,
            }],
        })

        result = await manager.queue_virtual_nas_download(
            "org/model", "local",
        )

        self.assertEqual(result["job_ids"], ["download-1"])
        manager.virtual_nas.resolve_download_revision.assert_not_awaited()
        manager.virtual_nas.estimate_download_size.assert_awaited_once_with(
            "org/model", RESOLVED_REVISION,
        )
        pinned_resolution = {
            "requested_revision": "release-gguf",
            "resolved_revision": RESOLVED_REVISION,
            "size_bytes": 100,
            "resume_node_id": "local",
            "download_cache_baseline_bytes": None,
        }
        manager.virtual_nas_transfer_preflight.assert_awaited_once_with(
            "org/model", "release-gguf", pinned_resolution,
        )
        manager.virtual_nas.queue_download_and_transfer.assert_awaited_once_with(
            "org/model", RESOLVED_REVISION, "local", [], 100,
            requested_revision="release-gguf", require_partial_cache=True,
            download_cache_baseline_bytes=None,
        )

    async def test_resume_preflight_credits_only_bytes_after_attempt_baseline(self):
        manager = Manager.__new__(Manager)
        manager.settings = {"virtual_nas_enabled": True}
        cached = 25
        expected = 100
        required = expected * 2 + DOWNLOAD_STAGING_RESERVE_BYTES - cached
        manager.model_cache_inventory = AsyncMock(return_value=[{
            "id": "worker-a", "name": "Worker", "online": True,
            "cache_free_size": required,
            "virtual_nas_download_capable": True,
            "models": [{
                "model_id": "org/model", "size_bytes": 125,
                "partial": False, "has_partial_download": True,
                "partial_size_bytes": cached,
                "partial_revision_size_bytes": {
                    RESOLVED_REVISION: cached,
                    "b" * 40: 90,
                },
                "revisions": ["old-revision"],
            }],
        }])
        manager.virtual_nas_transfers = Mock(return_value={"items": []})
        manager.virtual_nas = Mock()
        resolution = {
            "requested_revision": "release-1",
            "resolved_revision": RESOLVED_REVISION,
            "size_bytes": expected,
            "resume_node_id": "worker-a",
            "download_cache_baseline_bytes": None,
        }

        result = await manager.virtual_nas_transfer_preflight(
            "org/model", "release-1", resolution,
        )
        target = result["targets"][0]

        self.assertTrue(target["has_partial_model_cache"])
        self.assertTrue(target["download_eligible"])
        self.assertEqual(target["download_required_free_bytes"], required)

    async def test_finish_rejects_legacy_attempt_without_immutable_revision(self):
        manager = Manager.__new__(Manager)
        manager.settings = {}
        manager.virtual_nas = Mock()
        manager.virtual_nas.list_transfers.return_value = {"items": [{
            "id": "legacy-download", "kind": "download",
            "model_id": "org/model", "target_node_id": "worker-a",
            "requested_revision": "release-1", "revision": "release-1",
            "status": "failed", "started_at": 4,
            "download_attempted_at": 4.5,
            "legacy_download_attempt_tracking": False, "created_at": 5,
        }, {
            "id": "older-pinned-download", "kind": "download",
            "model_id": "org/model", "target_node_id": "worker-a",
            "requested_revision": "older", "revision": RESOLVED_REVISION,
            "status": "failed", "started_at": 2,
            "download_attempted_at": 2.5,
            "legacy_download_attempt_tracking": False, "created_at": 3,
        }]}
        manager.virtual_nas.resolve_download_revision = AsyncMock()
        manager.virtual_nas_transfer_preflight = AsyncMock()

        with self.assertRaisesRegex(LookupError, "cannot be recovered safely"):
            await manager.queue_virtual_nas_download(
                "org/model", "worker-a",
            )

        manager.virtual_nas.resolve_download_revision.assert_not_awaited()
        manager.virtual_nas_transfer_preflight.assert_not_awaited()

    async def test_recipe_transfer_preflight_requires_exact_revision_and_capacity(self):
        manager = Manager.__new__(Manager)
        manager.settings = {"virtual_nas_enabled": True}
        required = 20 * 2 + 64 * 1024 * 1024
        manager.model_cache_inventory = AsyncMock(return_value=[
            {
                "id": "source", "name": "Source", "online": True,
                "free_size": required, "cache_free_size": required,
                "models": [{
                    "model_id": "org/model", "size_bytes": 20,
                    "partial": False,
                    "revisions": ["main", RESOLVED_REVISION],
                    "revision_refs": {"main": RESOLVED_REVISION},
                }],
            },
            {
                "id": "enough", "name": "Enough", "online": True,
                "free_size": required, "cache_free_size": required, "models": [],
            },
            {
                "id": "short", "name": "Short", "online": True,
                "free_size": required - 1, "cache_free_size": required - 1, "models": [],
            },
            {
                "id": "wrong-revision", "name": "Wrong revision", "online": True,
                "free_size": required, "cache_free_size": required,
                "models": [{
                    "model_id": "org/model", "size_bytes": 20,
                    "partial": False, "revisions": ["snapshot-a"],
                }],
            },
            {
                "id": "generic-only", "name": "Legacy worker", "online": True,
                "free_size": required, "models": [],
            },
        ])
        manager.virtual_nas_transfers = Mock(return_value={"items": []})
        manager.virtual_nas = Mock()
        manager.virtual_nas.estimate_download_size = AsyncMock(return_value=20)
        manager.virtual_nas.resolve_download_revision = AsyncMock(return_value={
            "requested_revision": "main",
            "resolved_revision": RESOLVED_REVISION,
            "size_bytes": 20,
        })

        result = await manager.virtual_nas_transfer_preflight("org/model", "main")
        targets = {item["node_id"]: item for item in result["targets"]}

        self.assertEqual(result["source"]["node_id"], "source")
        self.assertEqual(targets["enough"]["required_free_bytes"], required)
        self.assertTrue(targets["enough"]["eligible"])
        self.assertFalse(targets["short"]["eligible"])
        self.assertEqual(targets["short"]["reason"], "Not enough free cache space")
        self.assertFalse(targets["wrong-revision"]["eligible"])
        self.assertIn("already exists", targets["wrong-revision"]["reason"])
        self.assertFalse(targets["generic-only"]["eligible"])
        self.assertEqual(
            targets["generic-only"]["reason"],
            "Free cache capacity is unavailable",
        )

    async def test_recipe_transfer_preflight_blocks_but_does_not_adopt_other_revision(self):
        manager = Manager.__new__(Manager)
        manager.settings = {"virtual_nas_enabled": True}
        required = 20 * 2 + 64 * 1024 * 1024
        manager.model_cache_inventory = AsyncMock(return_value=[
            {
                "id": "source", "name": "Source", "online": True,
                "cache_free_size": required,
                "models": [{
                    "model_id": "org/model", "size_bytes": 20,
                    "partial": False,
                    "revisions": ["rev-b", RESOLVED_REVISION],
                    "revision_refs": {"rev-b": RESOLVED_REVISION},
                }],
            },
            {
                "id": "target", "name": "Target", "online": True,
                "cache_free_size": required, "models": [],
            },
        ])
        manager.virtual_nas_transfers = Mock(return_value={"items": [{
            "id": "rev-a-job", "model_id": "org/model", "revision": "rev-a",
            "target_node_id": "target", "status": "running",
        }]})
        manager.virtual_nas = Mock()
        manager.virtual_nas.estimate_download_size = AsyncMock(return_value=20)
        manager.virtual_nas.resolve_download_revision = AsyncMock(return_value={
            "requested_revision": "rev-b",
            "resolved_revision": RESOLVED_REVISION,
            "size_bytes": 20,
        })

        result = await manager.virtual_nas_transfer_preflight("org/model", "rev-b")
        target = next(item for item in result["targets"] if item["node_id"] == "target")

        self.assertFalse(target["eligible"])
        self.assertFalse(target["download_eligible"])
        self.assertIn("Another revision", target["reason"])
        self.assertIsNone(target["active_job_id"])

    async def test_manager_refuses_active_container_then_deletes_exact_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            hub = Path(directory) / "hub"
            repository = create_cached_model(hub)
            manager = Manager.__new__(Manager)
            manager.settings = {"hf_cache": directory}
            manager.deployments = []
            manager.node_registry = FakeRegistry()
            manager.virtual_nas = VirtualNAS(
                Path(directory), lambda: hub, manager.node_registry, lambda: True,
            )
            manager.list_containers = AsyncMock(return_value=[{
                "model": "org/model", "served_models": [], "status": "running",
            }])
            manager._container_model_ids = Mock(return_value=["org/model"])
            manager._unsloth_loaded_model = AsyncMock(return_value=None)

            with self.assertRaisesRegex(RuntimeError, "local deployment"):
                await manager.delete_virtual_nas_model("local", "org/model")

            self.assertTrue(repository.exists())
            manager.list_containers.return_value = []
            result = await manager.delete_virtual_nas_model("local", "org/model")
            self.assertTrue(result["ok"])
            self.assertFalse(repository.exists())

    async def test_public_inventory_uses_node_disk_capacity(self):
        manager = Manager.__new__(Manager)
        manager.settings = {
            "virtual_nas_enabled": True, "cluster_node_name": "Coordinator",
        }
        manager.virtual_nas = Mock()
        manager.virtual_nas.inventory.return_value = [{
            "model_id": "org/model", "size_bytes": 40,
        }]
        manager.virtual_nas.list_transfers.return_value = {"items": []}
        manager.cluster_nodes = AsyncMock(return_value=[{
            "id": "local", "name": "Coordinator", "online": True,
            "disk": {"total": 1000, "free": 600},
        }])

        result = await manager.virtual_nas_inventory()

        self.assertEqual(result["nodes"][0]["total_size"], 1000)
        self.assertEqual(result["nodes"][0]["free_size"], 600)

    async def test_public_inventory_reports_live_overall_download_progress(self):
        manager = Manager.__new__(Manager)
        manager.settings = {
            "virtual_nas_enabled": True, "cluster_node_name": "Coordinator",
        }
        manager.virtual_nas = Mock()
        manager.virtual_nas.list_transfers.return_value = {"items": [{
            "id": "download-1", "kind": "download", "model_id": "org/model",
            "source_node_id": "huggingface", "target_node_id": "worker-a",
            "revision": RESOLVED_REVISION, "status": "running",
            "bytes_total": 1000, "bytes_transferred": 970,
            "download_cache_baseline_bytes": 1000, "created_at": 1,
        }]}
        manager.node_registry = Mock()
        manager.node_registry.get.return_value = {
            "id": "worker-a", "name": "Worker A",
        }
        manager.model_cache_inventory = AsyncMock(return_value=[{
            "id": "worker-a", "name": "Worker A", "online": True,
            "models": [{
                "model_id": "org/model", "size_bytes": 1975,
                "partial": False, "has_partial_download": True,
                "partial_revision_size_bytes": {RESOLVED_REVISION: 975},
            }],
        }])

        result = await manager.virtual_nas_inventory()

        self.assertEqual(result["jobs"][0]["bytes_transferred"], 975)
        self.assertEqual(result["jobs"][0]["progress"], 0.975)


if __name__ == "__main__":
    unittest.main()
