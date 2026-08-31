import asyncio
import json
import os
import shutil
import stat
import struct
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from manager import DEFAULT_SETTINGS, Manager
from sparkdeck import virtual_nas
from sparkdeck.virtual_nas import (
    DOWNLOAD_STAGING_RESERVE_BYTES,
    VIRTUAL_NAS_DOWNLOAD_CAPABILITY,
    VIRTUAL_NAS_DOWNLOAD_BASELINE_CAPABILITY,
    VIRTUAL_NAS_DIRECT_TRANSFER_CAPABILITY,
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
        self.stream_calls: list[tuple[str, str, str, dict]] = []

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
        self.stream_calls.append((node_id, method, path, kwargs))
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


class RemoteSourceRegistry(FakeRegistry):
    """A fabric-configured remote source for archive-leg routing tests."""

    def __init__(self):
        super().__init__()
        self.last_source_response = None

    async def request(self, node_id, method, path, **_kwargs):
        if "files/size" in path:
            return {"size_bytes": 7}
        return await super().request(node_id, method, path)

    async def open_stream(self, node_id, method, path, **kwargs):
        self.stream_calls.append((node_id, method, path, kwargs))
        if method == "GET":
            class SourceResponse(FakeResponse):
                async def aiter_bytes(self, chunk_size=None):
                    self.chunk_size = chunk_size
                    yield b"archive"
            self.last_source_response = SourceResponse()
            return self.last_source_response
        return await super().open_stream(node_id, method, path, **kwargs)


class DirectTransferRegistry(FakeRegistry):
    def __init__(self):
        super().__init__()
        self.direct_requests: list[tuple[str, str, str, dict]] = []
        self.direct_resolutions: list[str] = []

    def direct_transfer_source(self, node_id):
        return {"worker-a": "http://169.254.10.4:7878", "worker-b": "http://169.254.10.3:7878"}.get(node_id)

    async def resolve_direct_transfer_source(self, node_id):
        self.direct_resolutions.append(node_id)
        return self.direct_transfer_source(node_id)

    async def probe(self, node, force=False):
        return {
            **node, "online": True,
            "capabilities": [VIRTUAL_NAS_DIRECT_TRANSFER_CAPABILITY],
            "disk": {"free": 10 * 1024 * 1024 * 1024},
        }

    async def request(self, node_id, method, path, **kwargs):
        body = kwargs.get("json_body") or {}
        self.direct_requests.append((node_id, method, path, body))
        if path.endswith("/export-capability"):
            return {"capability": "single-use-capability"}
        if path.endswith("/import-from-peer"):
            self.remote_models[node_id] = [{
                "model_id": "org/model", "size_bytes": body["model_bytes"],
                "partial": False, "revisions": ["revision-1"],
            }]
            # Tar headers and record padding make wire bytes larger than the
            # model's logical size; that must not turn a valid import into a
            # false failure.
            return {"bytes_received": body["model_bytes"] + 10_240}
        return {
            "models": self.remote_models.get(node_id, []),
            "free_size": 10 * 1024 * 1024 * 1024,
        }


class InventoryAndArchiveTests(unittest.IsolatedAsyncioTestCase):
    async def test_direct_export_capability_is_single_use_and_revision_scoped(self):
        with tempfile.TemporaryDirectory() as directory:
            hub = Path(directory) / "hub"
            create_cached_model(hub)
            nas = VirtualNAS(Path(directory), lambda: hub, FakeRegistry(), lambda: True)
            capability = nas.issue_direct_export_capability("org/model", "revision-1")

            stream = nas.export_model_with_capability("org/model", capability)
            await stream.aclose()
            with self.assertRaisesRegex(PermissionError, "invalid or expired"):
                nas.export_model_with_capability("org/model", capability)

    async def test_import_reports_each_durable_finalization_phase(self):
        with tempfile.TemporaryDirectory() as directory:
            hub = Path(directory) / "hub"
            source_hub = Path(directory) / "source-hub"
            create_cached_model(source_hub)
            source = VirtualNAS(Path(directory) / "source", lambda: source_hub, FakeRegistry(), lambda: True)
            target = VirtualNAS(Path(directory) / "target", lambda: hub, FakeRegistry(), lambda: True)
            phases: list[str] = []

            result = await target.import_model(
                "org/model", source.export_model("org/model"), phase=phases.append,
            )

            self.assertTrue(result["ok"])
            self.assertEqual(phases, ["receiving", "syncing", "validating", "registering"])

    async def test_cancel_during_finalization_waits_for_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_hub = root / "source-hub"
            target_hub = root / "target-hub"
            create_cached_model(source_hub)
            source = VirtualNAS(root / "source", lambda: source_hub, FakeRegistry(), lambda: True)
            target = VirtualNAS(root / "target", lambda: target_hub, FakeRegistry(), lambda: True)
            entered = threading.Event()
            release = threading.Event()
            original = target._finalize_received_model

            def blocked_finalize(*args, **kwargs):
                entered.set()
                release.wait(2)
                return original(*args, **kwargs)

            with patch.object(target, "_finalize_received_model", side_effect=blocked_finalize):
                task = asyncio.create_task(target.import_model(
                    "org/model", source.export_model("org/model"),
                ))
                self.assertTrue(await asyncio.to_thread(entered.wait, 2))
                task.cancel()
                await asyncio.sleep(0.05)
                self.assertFalse(task.done())
                release.set()
                result = await task

            self.assertTrue(result["ok"])
            self.assertTrue((target_hub / "models--org--model").is_dir())
            self.assertEqual(list(target_hub.glob(".sparkdeck-vnas-stage-*")), [])

    def test_virtual_nas_is_disabled_by_default(self):
        self.assertIs(DEFAULT_SETTINGS["virtual_nas_enabled"], False)

    def test_has_model_files_reports_presence_per_selected_file(self):
        with tempfile.TemporaryDirectory() as directory:
            hub = Path(directory) / "hub"
            nas = VirtualNAS(
                Path(directory), lambda: hub, FakeRegistry(), lambda: True,
            )
            create_cached_model(hub)
            revision_sha = "a" * 40
            snapshot = hub / "models--org--model" / "snapshots" / revision_sha
            snapshot.mkdir(parents=True)
            (snapshot / "q4" / "model.gguf").parent.mkdir()
            (snapshot / "q4" / "model.gguf").write_bytes(b"gguf")

            complete = nas.has_model_files(
                "org/model", revision_sha, ["q4/model.gguf"],
            )
            partial = nas.has_model_files(
                "org/model", revision_sha, ["q4/model.gguf", "missing.gguf"],
            )
            absent = nas.has_model_files(
                "org/model", "b" * 40, ["q4/model.gguf"],
            )

            self.assertTrue(complete["complete"])
            self.assertEqual(complete["missing_files"], [])
            self.assertEqual(partial["present_files"], ["q4/model.gguf"])
            self.assertEqual(partial["missing_files"], ["missing.gguf"])
            self.assertFalse(partial["complete"])
            self.assertFalse(absent["complete"])

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

    def test_download_forces_hf_xet_high_performance_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            hub = Path(directory) / "configured-cache" / "hub"
            nas = VirtualNAS(
                Path(directory), lambda: hub, FakeRegistry(), lambda: True,
            )

            def download_into_cache(**kwargs):
                create_cached_model(Path(kwargs["cache_dir"]))
                return str(hub / "models--org--model" / "snapshots" / "revision-1")

            huggingface_hub = Mock(snapshot_download=Mock(side_effect=download_into_cache))
            with patch.dict(
                os.environ, {"HF_XET_HIGH_PERFORMANCE": "0"}, clear=False,
            ), patch.dict("sys.modules", {"huggingface_hub": huggingface_hub}):
                nas.download_model("org/model", "revision-1")
                self.assertEqual(os.environ.get("HF_XET_HIGH_PERFORMANCE"), "1")

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
                self.assertEqual(
                    partial_model["selective_files_by_revision"],
                    {resolved: [selected_name]},
                )
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

    async def test_selective_download_merges_prior_selected_files_into_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            hub = Path(directory) / "hub"
            repository = hub / "models--org--model"
            resolved = "c" * 40
            first = "model-Q4_K_M.gguf"
            second = "model-Q8_K_XL.gguf"
            snapshot = repository / "snapshots" / resolved

            def download_file_into_cache(**kwargs):
                filename = kwargs["filename"]
                snapshot.mkdir(parents=True, exist_ok=True)
                (snapshot / filename).write_bytes(b"weights")
                return str(snapshot / filename)

            api = Mock()
            api.model_info.return_value = Mock(
                sha=resolved,
                siblings=[
                    Mock(rfilename=first, size=7),
                    Mock(rfilename=second, size=7),
                    Mock(rfilename="README.md", size=1),
                ],
            )
            huggingface_hub = Mock(
                HfApi=Mock(return_value=api),
                hf_hub_download=Mock(side_effect=download_file_into_cache),
            )
            nas = VirtualNAS(
                Path(directory), lambda: hub, FakeRegistry(), lambda: True,
            )
            nas.free_bytes = Mock(return_value=10 * 1024 * 1024 * 1024)

            with patch.dict("sys.modules", {"huggingface_hub": huggingface_hub}):
                await nas.download_model_files_checked("org/model", resolved, [first])
                await nas.download_model_files_checked("org/model", resolved, [second])

            marker = json.loads((snapshot / ".sparkdeck-selective.incomplete").read_text())
            self.assertEqual(marker, {"files": [first, second]})
            self.assertEqual(nas.inventory()[0]["selective_files_by_revision"], {
                resolved: [first, second],
            })

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

    async def test_inventory_reports_snapshot_files_by_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            hub = Path(directory) / "hub"
            repository = hub / "models--org--model"
            blobs = repository / "blobs"
            blobs.mkdir(parents=True)
            weights = blobs / "weights"
            weights.write_bytes(b"gguf-weights")
            complete_revision = "c" * 40
            complete = repository / "snapshots" / complete_revision
            complete.mkdir(parents=True)
            nested = complete / "sub" / "dir"
            nested.mkdir(parents=True)
            try:
                (complete / "model-Q4_K_M.gguf").symlink_to(weights)
                (nested / "tokenizer.json").symlink_to(weights)
            except OSError:
                (complete / "model-Q4_K_M.gguf").write_bytes(b"gguf-weights")
                (nested / "tokenizer.json").write_bytes(b"gguf-weights")
            (complete / "README-AWQ.md").write_text("not weights", encoding="utf-8")
            (complete / "mmproj-F16.gguf").write_bytes(b"projector")
            # A selective snapshot carries its marker and a download lock;
            # its real files still count as cached, the residue does not.
            selective_revision = "d" * 40
            selective = repository / "snapshots" / selective_revision
            selective.mkdir()
            (selective / "model-Q8_0.gguf").write_bytes(b"gguf-weights")
            (selective / ".sparkdeck-selective.incomplete").write_text(
                "selective", encoding="utf-8",
            )
            (selective / "download.lock").write_text("locked", encoding="utf-8")
            # A symlink escaping the cache (and a dangling one) is reported
            # by the same containment rules the launch path applies.
            outside = Path(directory) / "outside.gguf"
            outside.write_bytes(b"outside")
            escape_linked = False
            dangling_linked = False
            try:
                (selective / "escape.gguf").symlink_to(outside)
                escape_linked = True
                (selective / "dangling.gguf").symlink_to(
                    repository / "blobs" / "missing-blob",
                )
                dangling_linked = True
            except OSError:
                pass
            nas = VirtualNAS(
                Path(directory), lambda: hub, FakeRegistry(), lambda: False,
            )

            models = nas.inventory()

            self.assertEqual(len(models), 1)
            self.assertEqual(set(models[0]["snapshot_files"]), {
                complete_revision, selective_revision,
            })
            self.assertEqual(set(models[0]["snapshot_files"][complete_revision]), {
                "README-AWQ.md", "mmproj-F16.gguf", "model-Q4_K_M.gguf",
                "sub/dir/tokenizer.json",
            })
            self.assertEqual(
                models[0]["snapshot_files"][selective_revision],
                ["model-Q8_0.gguf"],
            )
            self.assertEqual(models[0]["quantizations"], ["Q4_K_M", "Q8_0"])
            self.assertEqual(models[0]["selective_files_by_revision"], {
                selective_revision: ["model-Q8_0.gguf"],
            })
            if escape_linked:
                self.assertNotIn("escape.gguf", models[0]["snapshot_files"][selective_revision])
            if dangling_linked:
                self.assertNotIn("dangling.gguf", models[0]["snapshot_files"][selective_revision])
            self.assertNotIn(str(repository), json.dumps(models))

    async def test_snapshot_file_inventory_enforces_cap_during_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            hub = Path(directory) / "hub"
            repository = hub / "models--org--model"
            snapshot = repository / "snapshots" / ("c" * 40)
            snapshot.mkdir(parents=True)
            for index in range(2100):
                (snapshot / f"file-{index:05}.txt").write_text("x", encoding="utf-8")
            nas = VirtualNAS(
                Path(directory), lambda: hub, FakeRegistry(), lambda: False,
            )

            files_by_revision = virtual_nas._snapshot_files_by_revision(repository)

            reported = sum(len(names) for names in files_by_revision.values())
            self.assertLessEqual(
                reported,
                virtual_nas._INVENTORY_FILE_LIMIT,
            )

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

    async def test_inventory_accepts_complete_diffusers_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            hub = Path(directory) / "hub"
            snapshot = (
                hub / "models--Tongyi-MAI--Z-Image-Turbo"
                / "snapshots" / RESOLVED_REVISION
            )
            (snapshot / "transformer").mkdir(parents=True)
            (snapshot / "tokenizer").mkdir()
            (snapshot / "scheduler").mkdir()
            (snapshot / "model_index.json").write_text(json.dumps({
                "_class_name": "ZImagePipeline",
                "transformer": ["diffusers", "ZImageTransformer2DModel"],
                "tokenizer": ["transformers", "Qwen2Tokenizer"],
                "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
            }), encoding="utf-8")
            (snapshot / "transformer" / "config.json").write_text("{}")
            (snapshot / "transformer" / "diffusion_pytorch_model.safetensors").write_bytes(b"weights")
            (snapshot / "tokenizer" / "tokenizer.json").write_text("{}")
            (snapshot / "scheduler" / "scheduler_config.json").write_text("{}")
            nas = VirtualNAS(
                Path(directory), lambda: hub, FakeRegistry(), lambda: False,
            )

            model = nas.inventory()[0]

            self.assertEqual(model["model_id"], "Tongyi-MAI/Z-Image-Turbo")
            self.assertFalse(model["partial"])
            self.assertFalse(model["has_partial_download"])
            self.assertEqual(model["revisions"], [RESOLVED_REVISION])

            (snapshot / "text_encoder").mkdir()
            (snapshot / "text_encoder" / "config.json").write_text("{}")
            (snapshot / "model_index.json").write_text(json.dumps({
                "_class_name": "ZImagePipeline",
                "transformer": ["diffusers", "ZImageTransformer2DModel"],
                "text_encoder": ["transformers", "Qwen2Model"],
                "tokenizer": ["transformers", "Qwen2Tokenizer"],
                "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
            }), encoding="utf-8")
            self.assertTrue(nas.inventory()[0]["partial"])

            (snapshot / "model_index.json").write_text(json.dumps({
                "_class_name": "ZImagePipeline",
                "missing_component": ["diffusers", "MissingModel"],
            }), encoding="utf-8")
            self.assertTrue(nas.inventory()[0]["partial"])

    async def test_inventory_accepts_tokenizer_free_diffusers_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            hub = Path(directory) / "hub"
            snapshot = hub / "models--org--ddpm" / "snapshots" / RESOLVED_REVISION
            (snapshot / "unet").mkdir(parents=True)
            (snapshot / "scheduler").mkdir()
            (snapshot / "model_index.json").write_text(json.dumps({
                "_class_name": "DDPMPipeline",
                "unet": ["diffusers", "UNet2DModel"],
                "scheduler": ["diffusers", "DDPMScheduler"],
            }), encoding="utf-8")
            (snapshot / "unet" / "config.json").write_text("{}")
            (snapshot / "unet" / "diffusion_pytorch_model.safetensors").write_bytes(b"weights")
            (snapshot / "scheduler" / "scheduler_config.json").write_text("{}")
            nas = VirtualNAS(
                Path(directory), lambda: hub, FakeRegistry(), lambda: False,
            )

            model = nas.inventory()[0]

            self.assertEqual(model["model_id"], "org/ddpm")
            self.assertFalse(model["partial"])

    async def test_inventory_resolves_diffusers_shards_relative_to_component_index(self):
        with tempfile.TemporaryDirectory() as directory:
            hub = Path(directory) / "hub"
            snapshot = hub / "models--org--sharded" / "snapshots" / RESOLVED_REVISION
            transformer = snapshot / "transformer"
            scheduler = snapshot / "scheduler"
            transformer.mkdir(parents=True)
            scheduler.mkdir()
            (snapshot / "model_index.json").write_text(json.dumps({
                "_class_name": "ShardedPipeline",
                "transformer": ["diffusers", "Transformer2DModel"],
                "scheduler": ["diffusers", "DDPMScheduler"],
            }), encoding="utf-8")
            shard_names = [
                "diffusion_pytorch_model-00001-of-00002.safetensors",
                "diffusion_pytorch_model-00002-of-00002.safetensors",
            ]
            for shard_name in shard_names:
                (transformer / shard_name).write_bytes(b"weights")
            (transformer / "config.json").write_text("{}")
            (transformer / "diffusion_pytorch_model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {
                    "layer.0": shard_names[0],
                    "layer.1": shard_names[1],
                }}),
                encoding="utf-8",
            )
            (scheduler / "scheduler_config.json").write_text("{}")
            nas = VirtualNAS(
                Path(directory), lambda: hub, FakeRegistry(), lambda: False,
            )

            self.assertFalse(nas.inventory()[0]["partial"])

            (transformer / shard_names[1]).unlink()
            self.assertTrue(nas.inventory()[0]["partial"])

    async def test_inventory_reports_complete_external_comfyui_bundle_as_manageable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "ComfyUI" / "models"
            files = (
                "diffusion_models/minimax_music3_dit_fp16.safetensors",
                "text_encoders/minimax_music3_text_encoder_pruned_int8_convrot.safetensors",
                "vae/minimax_music3_dav.safetensors",
            )
            for relative in files:
                target = root.joinpath(*relative.split("/"))
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"weights")
            nas = VirtualNAS(
                Path(directory), lambda: Path(directory) / "missing-hub",
                FakeRegistry(), lambda: False,
                external_model_roots_provider=lambda: [root],
            )

            models = nas.inventory()
            self.assertEqual(models, [{
                "model_id": "Comfy-Org/MiniMax-Music-3",
                "size_bytes": 21,
                "transfer_size_bytes": 21,
                "file_count": 3,
                "transfer_entry_count": 12,
                "partial": False,
                "has_partial_download": False,
                "partial_size_bytes": 0,
                "partial_revision_size_bytes": {},
                "partial_revisions": [],
                "partial_revision_refs": {},
                "revision": "ComfyUI",
                "revisions": [],
                "revision_refs": {},
                "last_modified": models[0]["last_modified"],
                "source": "ComfyUI",
                "externally_managed": True,
            }])

    async def test_inventory_counts_installed_ltx_variants(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "ComfyUI" / "models"
            files = (
                "diffusion_models/LTX2.5/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
                "text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
                "text_encoders/gemma4_e2b_it_bf16.safetensors",
                "vae/LTX2.5/ltx-2.5-video-vae-bf16.safetensors",
                "vae/LTX2.5/ltx-2.5-audio-vae-bf16.safetensors",
                "diffusion_models/LTX2.5/LTX-2.5-Distilled-Q8_0.gguf",
            )
            for relative in files:
                target = root.joinpath(*relative.split("/"))
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"weights")
            nas = VirtualNAS(
                Path(directory), lambda: Path(directory) / "missing-hub",
                FakeRegistry(), lambda: False,
                external_model_roots_provider=lambda: [root],
            )

            model = nas.inventory()[0]

            self.assertEqual(model["model_id"], "Lightricks/LTX-2.5")
            self.assertEqual(model["file_count"], 6)
            self.assertEqual(model["size_bytes"], 42)
            self.assertFalse(model["partial"])
            self.assertNotIn("transferable", model)
            self.assertNotIn("deletable", model)

    async def test_inventory_accepts_each_supported_ltx_transformer_format(self):
        transformers = (
            "diffusion_models/LTX2.5/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
            "diffusion_models/LTX2.5/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors",
            "diffusion_models/LTX2.5/LTX-2.5-Distilled-Q8_0.gguf",
        )
        required = (
            "text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
            "text_encoders/gemma4_e2b_it_bf16.safetensors",
            "vae/LTX2.5/ltx-2.5-video-vae-bf16.safetensors",
            "vae/LTX2.5/ltx-2.5-audio-vae-bf16.safetensors",
        )
        for transformer in transformers:
            with self.subTest(transformer=transformer), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "ComfyUI" / "models"
                for relative in (*required, transformer):
                    target = root.joinpath(*relative.split("/"))
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(b"weights")
                nas = VirtualNAS(
                    Path(directory), lambda: Path(directory) / "missing-hub",
                    FakeRegistry(), lambda: False,
                    external_model_roots_provider=lambda: [root],
                )

                model = nas.inventory()[0]

                self.assertEqual(model["model_id"], "Lightricks/LTX-2.5")
                self.assertEqual(model["file_count"], 5)
                self.assertFalse(model["partial"])

    async def test_inventory_ignores_ltx_bundle_without_a_transformer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "ComfyUI" / "models"
            required = (
                "text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
                "text_encoders/gemma4_e2b_it_bf16.safetensors",
                "vae/LTX2.5/ltx-2.5-video-vae-bf16.safetensors",
                "vae/LTX2.5/ltx-2.5-audio-vae-bf16.safetensors",
            )
            for relative in required:
                target = root.joinpath(*relative.split("/"))
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"weights")
            nas = VirtualNAS(
                Path(directory), lambda: Path(directory) / "missing-hub",
                FakeRegistry(), lambda: False,
                external_model_roots_provider=lambda: [root],
            )

            self.assertEqual(nas.inventory(), [])

    async def test_inventory_ignores_incomplete_external_comfyui_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "ComfyUI" / "models"
            target = root / "vae" / "minimax_music3_dav.safetensors"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"weights")
            nas = VirtualNAS(
                Path(directory), lambda: Path(directory) / "missing-hub",
                FakeRegistry(), lambda: False,
                external_model_roots_provider=lambda: [root],
            )

            self.assertEqual(nas.inventory(), [])

    async def test_external_comfyui_model_export_import_roundtrip(self):
        with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as target_dir:
            root = Path(source_dir) / "ComfyUI" / "models"
            bundle_files = {
                "diffusion_models/minimax_music3_dit_fp16.safetensors": b"dit-weights",
                "text_encoders/minimax_music3_text_encoder_pruned_int8_convrot.safetensors": b"encoder-weights",
                "vae/minimax_music3_dav.safetensors": b"vae-weights",
            }
            for relative, content in bundle_files.items():
                path = root.joinpath(*relative.split("/"))
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            source = VirtualNAS(
                Path(source_dir), lambda: Path(source_dir) / "missing-hub",
                FakeRegistry(), lambda: True,
                external_model_roots_provider=lambda: [root],
            )
            target_hub = Path(target_dir) / "hub"
            target = VirtualNAS(
                Path(target_dir), lambda: target_hub, FakeRegistry(), lambda: True,
            )

            result = await target.import_model(
                "Comfy-Org/MiniMax-Music-3",
                source.export_model("Comfy-Org/MiniMax-Music-3"),
            )

            self.assertTrue(result["ok"])
            # The export copies; the ComfyUI install is left untouched.
            for relative, content in bundle_files.items():
                self.assertEqual(
                    root.joinpath(*relative.split("/")).read_bytes(), content,
                )
            snapshot = (
                target_hub / "models--Comfy-Org--MiniMax-Music-3"
                / "snapshots" / "comfyui"
            )
            for relative, content in bundle_files.items():
                self.assertEqual(
                    snapshot.joinpath(*relative.split("/")).read_bytes(), content,
                )
            self.assertEqual(
                (snapshot / ".sparkdeck-external").read_bytes(), b"comfyui\n",
            )
            self.assertEqual(
                (target_hub / "models--Comfy-Org--MiniMax-Music-3"
                 / "refs" / "main").read_bytes(),
                b"comfyui",
            )
            models = target.inventory()
            self.assertEqual(len(models), 1)
            model = models[0]
            self.assertEqual(model["model_id"], "Comfy-Org/MiniMax-Music-3")
            self.assertFalse(model["partial"])
            self.assertFalse(model.get("externally_managed"))
            self.assertNotEqual(model.get("deletable"), False)
            # The synthetic refs/main makes the default revision resolvable.
            self.assertIn("main", model["revisions"])
            self.assertEqual(model["revision_refs"], {"main": "comfyui"})
            self.assertTrue(VirtualNAS._has_revision(model, "main"))
            # The imported copy is a normal managed model: deletable.
            self.assertTrue(target.delete_model("Comfy-Org/MiniMax-Music-3")["ok"])
            self.assertEqual(target.inventory(), [])

    async def test_delete_external_comfyui_model_unlinks_only_bundle_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "ComfyUI" / "models"
            bundle_files = (
                "diffusion_models/minimax_music3_dit_fp16.safetensors",
                "text_encoders/minimax_music3_text_encoder_pruned_int8_convrot.safetensors",
                "vae/minimax_music3_dav.safetensors",
            )
            for relative in bundle_files:
                path = root.joinpath(*relative.split("/"))
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"weights")
            keep = root / "loras" / "keep.safetensors"
            keep.parent.mkdir(parents=True)
            keep.write_bytes(b"keep")
            nas = VirtualNAS(
                Path(directory), lambda: Path(directory) / "missing-hub",
                FakeRegistry(), lambda: True,
                external_model_roots_provider=lambda: [root],
            )

            self.assertTrue(nas.delete_model("Comfy-Org/MiniMax-Music-3")["ok"])

            for relative in bundle_files:
                self.assertFalse(root.joinpath(*relative.split("/")).exists())
            self.assertEqual(keep.read_bytes(), b"keep")
            # Directories are left in place for ComfyUI.
            self.assertTrue((root / "diffusion_models").is_dir())
            with self.assertRaises(LookupError):
                nas.delete_model("Comfy-Org/MiniMax-Music-3")

    async def test_delete_external_comfyui_model_refuses_symlinked_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "ComfyUI" / "models"
            outside = Path(directory) / "outside.safetensors"
            outside.write_bytes(b"outside")
            real_files = (
                "diffusion_models/minimax_music3_dit_fp16.safetensors",
                "text_encoders/minimax_music3_text_encoder_pruned_int8_convrot.safetensors",
            )
            for relative in real_files:
                path = root.joinpath(*relative.split("/"))
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"weights")
            link = root / "vae" / "minimax_music3_dav.safetensors"
            link.parent.mkdir(parents=True)
            link.symlink_to(outside)
            nas = VirtualNAS(
                Path(directory), lambda: Path(directory) / "missing-hub",
                FakeRegistry(), lambda: True,
                external_model_roots_provider=lambda: [root],
            )

            with self.assertRaises(LookupError):
                nas.delete_model("Comfy-Org/MiniMax-Music-3")

            self.assertEqual(outside.read_bytes(), b"outside")
            self.assertTrue(link.is_symlink())

    async def test_delete_external_comfyui_model_is_blocked_during_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "ComfyUI" / "models"
            for relative in (
                "diffusion_models/minimax_music3_dit_fp16.safetensors",
                "text_encoders/minimax_music3_text_encoder_pruned_int8_convrot.safetensors",
                "vae/minimax_music3_dav.safetensors",
            ):
                path = root.joinpath(*relative.split("/"))
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"weights")
            nas = VirtualNAS(
                Path(directory), lambda: Path(directory) / "missing-hub",
                FakeRegistry(), lambda: True,
                external_model_roots_provider=lambda: [root],
            )

            stream = nas.export_model("Comfy-Org/MiniMax-Music-3")
            with self.assertRaisesRegex(RuntimeError, "transfer"):
                nas.delete_model("Comfy-Org/MiniMax-Music-3")
            await stream.aclose()

    async def test_delete_prefers_external_bundle_over_partial_hub_residue(self):
        with tempfile.TemporaryDirectory() as directory:
            hub = Path(directory) / "hub"
            residue = (
                hub / "models--Comfy-Org--MiniMax-Music-3"
                / "blobs" / "weights.incomplete"
            )
            residue.parent.mkdir(parents=True)
            residue.write_bytes(b"partial")
            (hub / "models--Comfy-Org--MiniMax-Music-3"
             / "snapshots" / RESOLVED_REVISION).mkdir(parents=True)
            root = Path(directory) / "ComfyUI" / "models"
            bundle_files = (
                "diffusion_models/minimax_music3_dit_fp16.safetensors",
                "text_encoders/minimax_music3_text_encoder_pruned_int8_convrot.safetensors",
                "vae/minimax_music3_dav.safetensors",
            )
            for relative in bundle_files:
                path = root.joinpath(*relative.split("/"))
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"weights")
            nas = VirtualNAS(
                Path(directory), lambda: hub, FakeRegistry(), lambda: True,
                external_model_roots_provider=lambda: [root],
            )

            # Inventory displays the complete external entry, so delete
            # targets the external files, not the partial hub residue.
            self.assertTrue(nas.delete_model("Comfy-Org/MiniMax-Music-3")["ok"])
            for relative in bundle_files:
                self.assertFalse(root.joinpath(*relative.split("/")).exists())
            self.assertEqual(residue.read_bytes(), b"partial")

            # With no external copy left, delete removes the hub residue.
            self.assertTrue(nas.delete_model("Comfy-Org/MiniMax-Music-3")["ok"])
            self.assertFalse(
                (hub / "models--Comfy-Org--MiniMax-Music-3").exists(),
            )
            with self.assertRaises(LookupError):
                nas.delete_model("Comfy-Org/MiniMax-Music-3")

    async def test_delete_prefers_complete_hub_copy_over_external_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            hub = Path(directory) / "hub"
            create_cached_model(hub, "Comfy-Org/MiniMax-Music-3")
            root = Path(directory) / "ComfyUI" / "models"
            bundle_files = (
                "diffusion_models/minimax_music3_dit_fp16.safetensors",
                "text_encoders/minimax_music3_text_encoder_pruned_int8_convrot.safetensors",
                "vae/minimax_music3_dav.safetensors",
            )
            for relative in bundle_files:
                path = root.joinpath(*relative.split("/"))
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"weights")
            nas = VirtualNAS(
                Path(directory), lambda: hub, FakeRegistry(), lambda: True,
                external_model_roots_provider=lambda: [root],
            )

            # Inventory displays the complete hub copy in this case.
            self.assertTrue(nas.delete_model("Comfy-Org/MiniMax-Music-3")["ok"])
            self.assertFalse(
                (hub / "models--Comfy-Org--MiniMax-Music-3").exists(),
            )
            for relative in bundle_files:
                self.assertTrue(root.joinpath(*relative.split("/")).is_file())

    async def test_export_reads_largest_bundle_copy_across_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            small_root = Path(directory) / "small" / "models"
            large_root = Path(directory) / "large" / "models"
            bundle_files = (
                "diffusion_models/minimax_music3_dit_fp16.safetensors",
                "text_encoders/minimax_music3_text_encoder_pruned_int8_convrot.safetensors",
                "vae/minimax_music3_dav.safetensors",
            )
            for root, content in ((small_root, b"small"), (large_root, b"large-weights")):
                for relative in bundle_files:
                    path = root.joinpath(*relative.split("/"))
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(content)
            nas = VirtualNAS(
                Path(directory), lambda: Path(directory) / "missing-hub",
                FakeRegistry(), lambda: True,
                external_model_roots_provider=lambda: [small_root, large_root],
            )

            # Inventory keeps the largest copy; export must ship those bytes.
            self.assertEqual(nas.inventory()[0]["size_bytes"], 3 * len(b"large-weights"))
            target_hub = Path(directory) / "target-hub"
            target = VirtualNAS(
                Path(directory) / "target", lambda: target_hub,
                FakeRegistry(), lambda: True,
            )
            await target.import_model(
                "Comfy-Org/MiniMax-Music-3",
                nas.export_model("Comfy-Org/MiniMax-Music-3"),
            )

            snapshot = (
                target_hub / "models--Comfy-Org--MiniMax-Music-3"
                / "snapshots" / "comfyui"
            )
            for relative in bundle_files:
                self.assertEqual(
                    snapshot.joinpath(*relative.split("/")).read_bytes(),
                    b"large-weights",
                )
            for root in (small_root, large_root):
                for relative in bundle_files:
                    self.assertTrue(root.joinpath(*relative.split("/")).is_file())

    async def test_delete_external_comfyui_model_removes_every_root_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            roots = [
                Path(directory) / "first" / "models",
                Path(directory) / "second" / "models",
            ]
            bundle_files = (
                "diffusion_models/minimax_music3_dit_fp16.safetensors",
                "text_encoders/minimax_music3_text_encoder_pruned_int8_convrot.safetensors",
                "vae/minimax_music3_dav.safetensors",
            )
            for root in roots:
                for relative in bundle_files:
                    path = root.joinpath(*relative.split("/"))
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"weights")
                keep = root / "loras" / "keep.safetensors"
                keep.parent.mkdir(parents=True, exist_ok=True)
                keep.write_bytes(b"keep")
            nas = VirtualNAS(
                Path(directory), lambda: Path(directory) / "missing-hub",
                FakeRegistry(), lambda: True,
                external_model_roots_provider=lambda: list(roots),
            )

            self.assertTrue(nas.delete_model("Comfy-Org/MiniMax-Music-3")["ok"])

            for root in roots:
                for relative in bundle_files:
                    self.assertFalse(root.joinpath(*relative.split("/")).exists())
                self.assertEqual(
                    (root / "loras" / "keep.safetensors").read_bytes(), b"keep",
                )
            with self.assertRaises(LookupError):
                nas.delete_model("Comfy-Org/MiniMax-Music-3")

    async def test_delete_external_comfyui_model_rolls_back_when_staging_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "ComfyUI" / "models"
            bundle_files = (
                "diffusion_models/minimax_music3_dit_fp16.safetensors",
                "text_encoders/minimax_music3_text_encoder_pruned_int8_convrot.safetensors",
                "vae/minimax_music3_dav.safetensors",
            )
            for relative in bundle_files:
                path = root.joinpath(*relative.split("/"))
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"weights")
            nas = VirtualNAS(
                Path(directory), lambda: Path(directory) / "missing-hub",
                FakeRegistry(), lambda: True,
                external_model_roots_provider=lambda: [root],
            )
            real_replace = os.replace
            renames = 0

            def failing_replace(src, dst):
                nonlocal renames
                renames += 1
                if renames == 2:
                    raise OSError("simulated staging failure")
                return real_replace(src, dst)

            with patch("sparkdeck.virtual_nas.os.replace", failing_replace):
                with self.assertRaisesRegex(RuntimeError, "stage"):
                    nas.delete_model("Comfy-Org/MiniMax-Music-3")

            for relative in bundle_files:
                self.assertEqual(
                    root.joinpath(*relative.split("/")).read_bytes(), b"weights",
                )
            self.assertEqual(
                list(root.rglob("*.sparkdeck-deleting")), [],
            )

    async def test_inventory_accepts_weights_only_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            hub = Path(directory) / "hub"
            snapshot = hub / "models--org--raw" / "snapshots" / RESOLVED_REVISION
            snapshot.mkdir(parents=True)
            (snapshot / "model.safetensors").write_bytes(b"weights")
            nas = VirtualNAS(
                Path(directory), lambda: hub, FakeRegistry(), lambda: False,
            )

            model = nas.inventory()[0]

            self.assertEqual(model["model_id"], "org/raw")
            self.assertFalse(model["partial"])

    async def test_inventory_marks_config_without_tokenizer_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            hub = Path(directory) / "hub"
            snapshot = hub / "models--org--raw" / "snapshots" / RESOLVED_REVISION
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text("{}", encoding="utf-8")
            (snapshot / "model.safetensors").write_bytes(b"weights")
            nas = VirtualNAS(
                Path(directory), lambda: hub, FakeRegistry(), lambda: False,
            )

            self.assertTrue(nas.inventory()[0]["partial"])

    async def test_inventory_reports_unrecognized_comfyui_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "ComfyUI" / "models"
            bundle_files = (
                "diffusion_models/minimax_music3_dit_fp16.safetensors",
                "text_encoders/minimax_music3_text_encoder_pruned_int8_convrot.safetensors",
                "vae/minimax_music3_dav.safetensors",
            )
            extra_files = (
                "checkpoints/Minimax Video/minimax_video.safetensors",
                "loras/foo.safetensors",
            )
            for relative in (*bundle_files, *extra_files):
                target = root.joinpath(*relative.split("/"))
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"weights")
            nas = VirtualNAS(
                Path(directory), lambda: Path(directory) / "missing-hub",
                FakeRegistry(), lambda: False,
                external_model_roots_provider=lambda: [root],
            )

            models = nas.inventory()
            by_id = {model["model_id"]: model for model in models}

            self.assertEqual(
                [m["model_id"] for m in models].count("Comfy-Org/MiniMax-Music-3"),
                1,
            )
            self.assertEqual(by_id["Comfy-Org/MiniMax-Music-3"]["file_count"], 3)
            for name in ("checkpoints/Minimax Video", "loras/foo"):
                entry = by_id[name]
                self.assertEqual(entry["size_bytes"], 7)
                self.assertEqual(entry["file_count"], 1)
                self.assertFalse(entry["partial"])
                self.assertEqual(entry["revision"], "ComfyUI")
                self.assertEqual(entry["source"], "ComfyUI")
                self.assertTrue(entry["externally_managed"])
                self.assertFalse(entry["transferable"])
                self.assertFalse(entry["deletable"])

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
            registered = []

            result = await target.import_model(
                "org/model", source.export_model("org/model"),
                registered=registered.append,
            )

            self.assertTrue(result["ok"])
            copied = target_hub / "models--org--model"
            copied_stat = copied.lstat()
            self.assertEqual(registered, [(copied_stat.st_dev, copied_stat.st_ino)])
            self.assertEqual((copied / "blobs" / "weights").read_bytes(), b"model-weights")
            self.assertEqual(list(target_hub.glob(".sparkdeck-vnas-*.tar")), [])
            with self.assertRaises(FileExistsError):
                await target.import_model("org/model", bytes_stream(b"unused"))

    async def test_streamed_import_rejects_corrupted_file_payload(self):
        with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as target_dir:
            source_hub = Path(source_dir) / "hub"
            target_hub = Path(target_dir) / "hub"
            create_cached_model(source_hub)
            source = VirtualNAS(
                Path(source_dir), lambda: source_hub, FakeRegistry(), lambda: True,
            )
            target = VirtualNAS(
                Path(target_dir), lambda: target_hub, FakeRegistry(), lambda: True,
            )
            body = bytearray(b"".join([
                chunk async for chunk in source.export_model("org/model")
            ]))
            payload_offset = body.index(b"model-weights")
            body[payload_offset] ^= 1

            with self.assertRaisesRegex(ValueError, "integrity verification"):
                await target.import_model("org/model", bytes_stream(bytes(body)))

            self.assertFalse((target_hub / "models--org--model").exists())

    async def test_streamed_import_caps_declared_payload_to_admitted_space(self):
        with tempfile.TemporaryDirectory() as directory:
            hub = Path(directory) / "hub"
            nas = VirtualNAS(
                Path(directory), lambda: hub, FakeRegistry(), lambda: True,
            )
            required = virtual_nas.transfer_required_free_bytes(1)
            nas.free_bytes = Mock(return_value=required)
            body = b"".join([
                virtual_nas._FILE_STREAM_MAGIC,
                virtual_nas._file_stream_header({
                    "path": "models--org--model", "type": "directory",
                }),
                virtual_nas._file_stream_header({
                    "path": "models--org--model/blobs/weights",
                    "type": "file", "size": required + 1,
                }),
            ])

            with self.assertRaisesRegex(ValueError, "admitted transfer size"):
                await nas.import_model(
                    "org/model", bytes_stream(body), required_model_bytes=1,
                )

            self.assertFalse((hub / "models--org--model").exists())

    async def test_file_stream_entry_budget_is_covered_by_capacity_reserve(self):
        payload = 164 * 1024**3
        entry_budget = (
            virtual_nas._FILE_STREAM_ENTRY_ALLOCATION_BYTES
            * virtual_nas._FILE_STREAM_MAX_ENTRIES
        )

        self.assertEqual(
            virtual_nas.TRANSFER_STAGING_RESERVE_BYTES,
            virtual_nas._FILE_STREAM_BASE_RESERVE_BYTES + entry_budget,
        )
        self.assertEqual(
            virtual_nas.transfer_required_free_bytes(payload),
            payload + virtual_nas.TRANSFER_STAGING_RESERVE_BYTES,
        )
        self.assertLess(
            virtual_nas.transfer_required_free_bytes(payload), payload * 2,
        )

    async def test_streamed_import_rejects_entry_overflow_before_registration(self):
        with tempfile.TemporaryDirectory() as directory:
            hub = Path(directory) / "hub"
            nas = VirtualNAS(Path(directory), lambda: hub, FakeRegistry(), lambda: True)
            with patch.object(virtual_nas, "_FILE_STREAM_MAX_ENTRIES", 2):
                body = b"".join([
                    virtual_nas._FILE_STREAM_MAGIC,
                    virtual_nas._file_stream_header({
                        "path": "models--org--model", "type": "directory",
                    }),
                    virtual_nas._file_stream_header({
                        "path": "models--org--model/blobs", "type": "directory",
                    }),
                    virtual_nas._file_stream_header({
                        "path": "models--org--model/snapshots", "type": "directory",
                    }),
                ])

                with self.assertRaisesRegex(ValueError, "too much metadata"):
                    await nas.import_model("org/model", bytes_stream(body))

            self.assertFalse((hub / "models--org--model").exists())
            self.assertEqual(list(hub.glob(".sparkdeck-vnas-stage-*")), [])

    async def test_inventory_and_export_reject_over_budget_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            hub = Path(directory) / "hub"
            repository = create_cached_model(hub)
            nas = VirtualNAS(Path(directory), lambda: hub, FakeRegistry(), lambda: True)
            exact_entries = nas._whole_model_stream_entries(repository, None)

            with patch.object(
                virtual_nas, "_FILE_STREAM_MAX_ENTRIES", len(exact_entries) - 1,
            ):
                model = nas.inventory()[0]
                self.assertEqual(
                    model["transfer_entry_count"], len(exact_entries),
                )
                self.assertFalse(model["partial"])
                self.assertIs(model["transferable"], False)
                with self.assertRaisesRegex(LookupError, "complete requested"):
                    nas.issue_direct_export_capability("org/model", None)
                with self.assertRaisesRegex(RuntimeError, "cannot be transferred"):
                    await nas.queue_transfer(
                        "org/model", "local", ["worker-a"],
                    )
                self.assertEqual(nas.jobs, [])
                stream = nas.export_model("org/model")
                with self.assertRaisesRegex(ValueError, "entry limit"):
                    await anext(stream)
                await stream.aclose()

            self.assertFalse(nas.model_in_transfer("org/model", "local"))

    async def test_streamed_import_cancellation_never_registers_repository(self):
        for cancel_phase in ("syncing", "registering"):
            with self.subTest(cancel_phase=cancel_phase), tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as target_dir:
                source_hub = Path(source_dir) / "hub"
                target_hub = Path(target_dir) / "hub"
                create_cached_model(source_hub)
                source = VirtualNAS(
                    Path(source_dir), lambda: source_hub,
                    FakeRegistry(), lambda: True,
                )
                target = VirtualNAS(
                    Path(target_dir), lambda: target_hub,
                    FakeRegistry(), lambda: True,
                )
                canceled = asyncio.Event()

                def phase(value: str) -> None:
                    if value == cancel_phase:
                        canceled.set()

                with self.assertRaises(virtual_nas.TransferCanceled):
                    await target.import_model(
                        "org/model", source.export_model("org/model"),
                        phase=phase, cancel=canceled,
                    )

                self.assertFalse(
                    (target_hub / "models--org--model").exists(),
                )

    async def test_streamed_import_uses_hardlinks_without_symlink_privilege(self):
        with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as target_dir:
            source_hub = Path(source_dir) / "hub"
            target_hub = Path(target_dir) / "hub"
            repository = create_cached_model(source_hub)
            blob = repository / "blobs" / "weights"
            snapshot_file = (
                repository / "snapshots" / "revision-1" / "model.safetensors"
            )
            snapshot_file.unlink()
            try:
                snapshot_file.symlink_to(blob)
            except OSError as exc:
                self.skipTest(f"source filesystem cannot create test symlink: {exc}")
            source = VirtualNAS(
                Path(source_dir), lambda: source_hub, FakeRegistry(), lambda: True,
            )
            target = VirtualNAS(
                Path(target_dir), lambda: target_hub, FakeRegistry(), lambda: True,
            )
            stream = source.export_model("org/model")

            with patch(
                "sparkdeck.virtual_nas.os.symlink",
                side_effect=PermissionError("symlinks disabled"),
            ):
                await target.import_model("org/model", stream)

            copied = target_hub / "models--org--model"
            self.assertTrue(os.path.samefile(
                copied / "blobs" / "weights",
                copied / "snapshots" / "revision-1" / "model.safetensors",
            ))
            onward_hub = Path(target_dir) / "onward-hub"
            onward = VirtualNAS(
                Path(target_dir) / "onward", lambda: onward_hub,
                FakeRegistry(), lambda: True,
            )
            await onward.import_model(
                "org/model", target.export_model("org/model"),
            )
            onward_copy = onward_hub / "models--org--model"
            self.assertTrue(os.path.samefile(
                onward_copy / "blobs" / "weights",
                onward_copy / "snapshots" / "revision-1" / "model.safetensors",
            ))

    async def test_streamed_export_preserves_safe_dangling_residue_link(self):
        with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as target_dir:
            source_hub = Path(source_dir) / "hub"
            target_hub = Path(target_dir) / "hub"
            repository = create_cached_model(source_hub)
            residue = repository / "snapshots" / "interrupted"
            residue.mkdir()
            dangling = residue / "missing.safetensors"
            try:
                dangling.symlink_to("../../blobs/missing")
            except OSError as exc:
                self.skipTest(f"source filesystem cannot create test symlink: {exc}")
            source = VirtualNAS(Path(source_dir), lambda: source_hub, FakeRegistry(), lambda: True)
            target = VirtualNAS(Path(target_dir), lambda: target_hub, FakeRegistry(), lambda: True)

            await target.import_model("org/model", source.export_model("org/model"))

            copied = (
                target_hub / "models--org--model" / "snapshots"
                / "interrupted" / "missing.safetensors"
            )
            self.assertTrue(copied.is_symlink())
            self.assertFalse(copied.exists())

    async def test_streamed_export_rejects_dangling_link_outside_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            hub = Path(directory) / "hub"
            repository = create_cached_model(hub)
            residue = repository / "snapshots" / "interrupted"
            residue.mkdir()
            dangling = residue / "outside"
            try:
                dangling.symlink_to("../../../outside")
            except OSError as exc:
                self.skipTest(f"source filesystem cannot create test symlink: {exc}")
            nas = VirtualNAS(Path(directory), lambda: hub, FakeRegistry(), lambda: True)

            with self.assertRaisesRegex(ValueError, "link escapes"):
                _ = b"".join([
                    chunk async for chunk in nas.export_model("org/model")
                ])

    async def test_streamed_export_batches_files_for_high_throughput(self):
        with tempfile.TemporaryDirectory() as directory:
            hub = Path(directory) / "hub"
            repository = create_cached_model(hub)
            payload_size = virtual_nas.FILE_STREAM_CHUNK_BYTES * 2 + 123
            (repository / "snapshots" / "revision-1" / "model.safetensors").write_bytes(
                b"x" * payload_size
            )
            nas = VirtualNAS(
                Path(directory), lambda: hub, FakeRegistry(), lambda: True,
            )

            chunk_sizes = [len(chunk) async for chunk in nas.export_model("org/model")]

            self.assertGreaterEqual(chunk_sizes.count(virtual_nas.FILE_STREAM_CHUNK_BYTES), 2)
            self.assertLessEqual(max(chunk_sizes), virtual_nas.FILE_STREAM_CHUNK_BYTES)
            self.assertGreater(sum(chunk_sizes), payload_size)

    async def test_whole_model_entry_walk_does_not_block_event_loop(self):
        with tempfile.TemporaryDirectory() as directory:
            hub = Path(directory) / "hub"
            create_cached_model(hub)
            nas = VirtualNAS(Path(directory), lambda: hub, FakeRegistry(), lambda: True)
            entered = threading.Event()
            release = threading.Event()
            original = nas._whole_model_stream_entries

            def blocked_entries(*args, **kwargs):
                entered.set()
                release.wait(2)
                return original(*args, **kwargs)

            with patch.object(nas, "_whole_model_stream_entries", side_effect=blocked_entries):
                stream = nas.export_model("org/model")
                first = await anext(stream)
                self.assertEqual(first, virtual_nas._FILE_STREAM_MAGIC)
                next_chunk = asyncio.create_task(anext(stream))
                self.assertTrue(await asyncio.to_thread(entered.wait, 2))
                heartbeat = asyncio.create_task(asyncio.sleep(0))
                await asyncio.wait_for(heartbeat, 0.5)
                release.set()
                await next_chunk
                await stream.aclose()

    async def test_late_peer_import_cancel_rolls_back_completed_model(self):
        with tempfile.TemporaryDirectory() as directory:
            hub = Path(directory) / "hub"
            repository = create_cached_model(hub)
            nas = VirtualNAS(Path(directory), lambda: hub, FakeRegistry(), lambda: True)
            repository_stat = repository.lstat()
            record = {
                "status": "completed", "phase": "completed",
                "phase_started_at": 0, "bytes_transferred": 1,
                "cancel": asyncio.Event(), "model_id": "org/model",
                "destination_identity": (
                    repository_stat.st_dev, repository_stat.st_ino,
                ),
            }
            nas._peer_imports["transfer-1"] = record

            result = await nas.cancel_peer_import("org/model", "transfer-1")
            repeated = await nas.cancel_peer_import("org/model", "transfer-1")

            self.assertEqual(result["status"], "canceled")
            self.assertTrue(result["rollback_confirmed"])
            self.assertEqual(repeated["status"], "canceled")
            self.assertTrue(repeated["rollback_confirmed"])
            self.assertFalse((hub / "models--org--model").exists())

    async def test_running_peer_cancel_waits_for_rollback_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            hub = Path(directory) / "hub"
            nas = VirtualNAS(Path(directory), lambda: hub, FakeRegistry(), lambda: True)
            done = asyncio.Event()
            record = {
                "status": "running", "phase": "receiving",
                "phase_started_at": 0, "bytes_transferred": 1,
                "cancel": asyncio.Event(), "done": done,
                "model_id": "org/model", "destination_identity": None,
            }
            nas._peer_imports["transfer-1"] = record

            cancellation = asyncio.create_task(
                nas.cancel_peer_import("org/model", "transfer-1"),
            )
            await asyncio.sleep(0)
            self.assertTrue(record["cancel"].is_set())
            self.assertFalse(cancellation.done())
            repository = create_cached_model(hub)
            repository_stat = repository.lstat()
            record["status"] = "completed"
            record["destination_identity"] = (
                repository_stat.st_dev, repository_stat.st_ino,
            )
            done.set()

            result = await cancellation

            self.assertEqual(result, {
                "status": "canceled", "rollback_confirmed": True,
            })
            self.assertFalse((hub / "models--org--model").exists())

    async def test_failed_peer_cancel_preserves_unowned_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            hub = Path(directory) / "hub"
            repository = create_cached_model(hub)
            nas = VirtualNAS(Path(directory), lambda: hub, FakeRegistry(), lambda: True)
            done = asyncio.Event()
            done.set()
            nas._peer_imports["transfer-1"] = {
                "status": "failed", "phase": "registering",
                "phase_started_at": 0, "bytes_transferred": 1,
                "cancel": asyncio.Event(), "done": done,
                "model_id": "org/model", "destination_identity": None,
            }

            result = await nas.cancel_peer_import("org/model", "transfer-1")

            self.assertEqual(result, {
                "status": "canceled", "rollback_confirmed": True,
            })
            self.assertTrue(repository.is_dir())

    async def test_peer_cancel_refuses_replaced_owned_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            hub = Path(directory) / "hub"
            repository = create_cached_model(hub)
            repository_stat = repository.lstat()
            repository.rename(hub / "original-model")
            replacement = create_cached_model(hub)
            sentinel = replacement / "concurrent-download"
            sentinel.write_text("keep", encoding="utf-8")
            nas = VirtualNAS(Path(directory), lambda: hub, FakeRegistry(), lambda: True)
            done = asyncio.Event()
            done.set()
            nas._peer_imports["transfer-1"] = {
                "status": "completed", "phase": "completed",
                "phase_started_at": 0, "bytes_transferred": 1,
                "cancel": asyncio.Event(), "done": done,
                "model_id": "org/model",
                "destination_identity": (
                    repository_stat.st_dev, repository_stat.st_ino,
                ),
            }

            with self.assertRaisesRegex(
                RuntimeError, "rollback cannot be confirmed",
            ):
                await nas.cancel_peer_import("org/model", "transfer-1")

            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    @unittest.skipIf(os.name == "nt", "POSIX permission bits are required")
    async def test_streamed_import_preserves_private_and_executable_modes(self):
        with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as target_dir:
            source_hub = Path(source_dir) / "hub"
            target_hub = Path(target_dir) / "hub"
            repository = create_cached_model(source_hub)
            private = repository / "snapshots" / "revision-1" / "config.json"
            executable = repository / "snapshots" / "revision-1" / "helper.sh"
            executable.write_text("#!/bin/sh\n", encoding="utf-8")
            private.chmod(0o600)
            executable.chmod(0o751)
            source = VirtualNAS(Path(source_dir), lambda: source_hub, FakeRegistry(), lambda: True)
            target = VirtualNAS(Path(target_dir), lambda: target_hub, FakeRegistry(), lambda: True)

            await target.import_model("org/model", source.export_model("org/model"))

            copied = target_hub / "models--org--model" / "snapshots" / "revision-1"
            self.assertEqual(stat.S_IMODE((copied / "config.json").stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE((copied / "helper.sh").stat().st_mode), 0o751)

    async def test_streamed_import_rejects_unsafe_file_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            hub = Path(directory) / "hub"
            nas = VirtualNAS(Path(directory), lambda: hub, FakeRegistry(), lambda: True)
            for mode in (True, -1, 0o1000):
                with self.subTest(mode=mode):
                    body = b"".join([
                        virtual_nas._FILE_STREAM_MAGIC,
                        virtual_nas._file_stream_header({
                            "path": "models--org--model", "type": "directory",
                        }),
                        virtual_nas._file_stream_header({
                            "path": "models--org--model/file", "type": "file",
                            "size": 0, "mode": mode,
                        }),
                    ])
                    with self.assertRaisesRegex(ValueError, "mode is invalid"):
                        await nas.import_model("org/model", bytes_stream(body))

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

    def test_delete_retries_after_making_cached_tree_owner_writable(self):
        with tempfile.TemporaryDirectory() as directory:
            hub = Path(directory) / "hub"
            repository = create_cached_model(hub)
            nas = VirtualNAS(Path(directory), lambda: hub, FakeRegistry(), lambda: True)
            real_rmtree = shutil.rmtree
            calls = 0

            def readonly_once(path):
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise PermissionError("simulated read-only cache")
                return real_rmtree(path)

            with patch("sparkdeck.virtual_nas.shutil.rmtree", readonly_once):
                result = nas.delete_model("org/model")

            self.assertTrue(result["ok"])
            self.assertEqual(calls, 2)
            self.assertFalse(repository.exists())

    def test_delete_reports_actionable_error_for_cache_permission_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            hub = Path(directory) / "hub"
            repository = create_cached_model(hub)
            nas = VirtualNAS(Path(directory), lambda: hub, FakeRegistry(), lambda: True)

            with patch(
                "sparkdeck.virtual_nas.shutil.rmtree",
                side_effect=PermissionError("simulated owner mismatch"),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "check cache ownership and permissions",
                ):
                    nas.delete_model("org/model")

            self.assertTrue(repository.exists())

    async def test_import_rejects_traversal_and_escaping_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            hub = Path(directory) / "hub"
            nas = VirtualNAS(Path(directory), lambda: hub, FakeRegistry(), lambda: True)
            for target, error in (
                ("../../../../outside", "link escapes"),
                ("C:/outside", "unsafe link"),
            ):
                with self.subTest(target=target):
                    stream_bytes = b"".join([
                        virtual_nas._FILE_STREAM_MAGIC,
                        virtual_nas._file_stream_header({
                            "path": "models--org--model", "type": "directory",
                        }),
                        virtual_nas._file_stream_header({
                            "path": "models--org--model/snapshots/rev/weights",
                            "type": "symlink", "target": target,
                        }),
                        struct.pack(">I", 0),
                    ])

                    with self.assertRaisesRegex(ValueError, error):
                        await nas.import_model(
                            "org/model", bytes_stream(stream_bytes),
                        )

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

        self.assertEqual(canceled["status"], "running")
        self.assertEqual(canceled["phase"], "canceling")
        self.assertEqual(nas.jobs[0]["status"], "canceled")
        await nas.stop()

    async def test_target_failure_is_recorded(self):
        registry = FakeRegistry(fail_target="worker-a")
        nas = VirtualNAS(Path(self.temp.name), lambda: self.hub, registry, lambda: True)
        nas.start = Mock()
        await nas.queue_transfer("org/model", "local", ["worker-a"])
        job = nas.jobs[0]
        job["bytes_transferred"] = 5

        await nas._run_transfer(job)

        self.assertEqual(job["status"], "failed")
        self.assertEqual(job["bytes_transferred"], 0)
        self.assertIn("simulated target outage", job["error"])
        await nas.stop()

    async def test_remote_nodes_transfer_directly_over_the_fabric(self):
        registry = DirectTransferRegistry()
        registry.remote_models["worker-a"] = [{
            "model_id": "org/model", "size_bytes": 123,
            "partial": False, "transferable": True, "revisions": ["revision-1"],
        }]
        nas = VirtualNAS(Path(self.temp.name), lambda: self.hub, registry, lambda: True)
        job = {
            "id": "direct-job", "kind": "transfer", "model_id": "org/model",
            "source_node_id": "worker-a", "target_node_id": "worker-b",
            "revision": "revision-1", "requested_revision": "revision-1",
            "status": "queued", "bytes_total": 123, "bytes_transferred": 0,
            "created_at": 0, "started_at": None, "completed_at": None, "error": None,
        }
        nas.jobs = [job]

        await nas._run_transfer(job)

        self.assertEqual(job["status"], "completed")
        self.assertEqual(job["bytes_transferred"], 123)
        self.assertEqual(registry.received, {})
        self.assertEqual(registry.direct_resolutions, ["worker-a", "worker-b"])
        self.assertEqual(
            [path for _, _, path, _ in registry.direct_requests if path.endswith(("export-capability", "import-from-peer"))],
            [
                "/api/agent/virtual-nas/models/org%2Fmodel/export-capability",
                "/api/agent/virtual-nas/models/org%2Fmodel/import-from-peer",
            ],
        )

    async def test_direct_cancel_after_peer_completion_requires_rollback_confirmation(self):
        for confirms, expected_status in ((True, "canceled"), (False, "failed")):
            with self.subTest(confirms=confirms):
                registry = DirectTransferRegistry()
                registry.remote_models["worker-a"] = [{
                    "model_id": "org/model", "size_bytes": 123,
                    "partial": False, "transferable": True,
                    "revisions": ["revision-1"],
                }]
                original_request = registry.request

                async def request(node_id, method, path, **kwargs):
                    if method == "DELETE" and "/peer-imports/" in path:
                        registry.direct_requests.append((node_id, method, path, {}))
                        if not confirms:
                            raise RuntimeError("rollback endpoint unavailable")
                        registry.remote_models[node_id] = []
                        return {
                            "status": "canceled", "rollback_confirmed": True,
                        }
                    return await original_request(node_id, method, path, **kwargs)

                registry.request = request
                nas = VirtualNAS(
                    Path(self.temp.name), lambda: self.hub,
                    registry, lambda: True,
                )
                job = {
                    "id": f"direct-cancel-{confirms}", "kind": "transfer",
                    "model_id": "org/model", "source_node_id": "worker-a",
                    "target_node_id": "worker-b", "revision": "revision-1",
                    "requested_revision": "revision-1", "status": "queued",
                    "bytes_total": 123, "bytes_transferred": 0,
                    "created_at": 0, "started_at": None,
                    "completed_at": None, "error": None,
                }
                nas.jobs = [job]
                original_storage = nas._node_storage

                async def storage(node_id):
                    value = await original_storage(node_id)
                    if (
                        node_id == "worker-b"
                        and registry.remote_models.get("worker-b")
                    ):
                        nas._cancel_events[job["id"]].set()
                    return value

                nas._node_storage = storage

                await nas._run_transfer(job)

                self.assertEqual(job["status"], expected_status)
                delete_calls = [
                    call for call in registry.direct_requests
                    if call[1] == "DELETE"
                ]
                self.assertEqual(len(delete_calls), 1)
                if confirms:
                    self.assertEqual(registry.remote_models["worker-b"], [])
                else:
                    self.assertIn("rollback could not be confirmed", job["error"])

    async def test_local_source_whole_model_upload_uses_fabric_stream(self):
        registry = FakeRegistry()
        nas = VirtualNAS(Path(self.temp.name), lambda: self.hub, registry, lambda: True)
        job = {
            "id": "local-source-job", "kind": "transfer", "model_id": "org/model",
            "source_node_id": "local", "target_node_id": "worker-a",
            "revision": "revision-1", "requested_revision": "revision-1",
            "status": "queued", "bytes_total": 0, "bytes_transferred": 0,
            "created_at": 0, "started_at": None, "completed_at": None, "error": None,
        }
        nas.jobs = [job]

        await nas._run_transfer(job)

        self.assertEqual(job["status"], "completed")
        upload = next(call for call in registry.stream_calls if call[1] == "PUT")
        self.assertTrue(upload[3]["use_fabric"])

    async def test_remote_source_whole_model_download_uses_fabric_stream(self):
        registry = RemoteSourceRegistry()
        registry.remote_models["worker-a"] = [{
            "model_id": "org/model", "size_bytes": 7,
            "partial": False, "transferable": True, "revisions": ["revision-1"],
        }]
        nas = VirtualNAS(Path(self.temp.name), lambda: self.hub, registry, lambda: True)
        local_models: list[dict] = []

        async def storage(node_id):
            if node_id == "worker-a":
                return {"models": registry.remote_models["worker-a"], "free_size": 10**12}
            return {"models": local_models, "free_size": 10**12}

        async def imported(model_id, chunks, **_kwargs):
            received = b"".join([chunk async for chunk in chunks])
            local_models.append({
                "model_id": model_id, "size_bytes": len(received),
                "partial": False, "revisions": ["revision-1"],
            })
            return {"bytes_received": len(received)}

        nas._node_storage = storage
        nas.import_model = imported
        job = {
            "id": "remote-source-job", "kind": "transfer", "model_id": "org/model",
            "source_node_id": "worker-a", "target_node_id": "local",
            "revision": "revision-1", "requested_revision": "revision-1",
            "status": "queued", "bytes_total": 7, "bytes_transferred": 0,
            "created_at": 0, "started_at": None, "completed_at": None, "error": None,
        }
        nas.jobs = [job]

        await nas._run_transfer(job)

        self.assertEqual(job["status"], "completed")
        download = next(call for call in registry.stream_calls if call[1] == "GET")
        self.assertTrue(download[3]["use_fabric"])
        self.assertEqual(
            registry.last_source_response.chunk_size,
            virtual_nas.FILE_STREAM_CHUNK_BYTES,
        )

    async def test_local_source_selective_file_upload_uses_fabric_stream(self):
        registry = FakeRegistry()
        nas = VirtualNAS(Path(self.temp.name), lambda: self.hub, registry, lambda: True)

        result = await nas.transfer_model_files(
            "org/model", "revision-1", ["model.safetensors"], "local", "worker-a",
        )

        self.assertTrue(result["ok"])
        upload = next(call for call in registry.stream_calls if call[1] == "PUT")
        self.assertTrue(upload[3]["use_fabric"])

    async def test_remote_source_selective_file_download_uses_fabric_stream(self):
        registry = RemoteSourceRegistry()
        nas = VirtualNAS(Path(self.temp.name), lambda: self.hub, registry, lambda: True)

        async def imported(_model_id, chunks, **_kwargs):
            return {
                "ok": True,
                "bytes_received": len(b"".join([chunk async for chunk in chunks])),
            }

        nas.import_model_files = imported
        result = await nas.transfer_model_files(
            "org/model", "revision-1", ["model.safetensors"], "worker-a", "local",
        )

        self.assertTrue(result["ok"])
        download = next(call for call in registry.stream_calls if call[1] == "GET")
        self.assertTrue(download[3]["use_fabric"])
        self.assertEqual(
            registry.last_source_response.chunk_size,
            virtual_nas.FILE_STREAM_CHUNK_BYTES,
        )

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

    async def test_nontransferable_source_is_rejected_without_persisting_jobs(self):
        nas = VirtualNAS(
            Path(self.temp.name), lambda: self.hub, FakeRegistry(), lambda: True,
        )

        async def storage(node_id):
            if node_id == "local":
                return {"models": [{
                    "model_id": "org/model", "size_bytes": 13,
                    "partial": False, "transferable": False,
                }]}
            return {"models": [], "free_size": 10 * 1024 * 1024 * 1024}

        nas._node_storage = AsyncMock(side_effect=storage)

        with self.assertRaisesRegex(RuntimeError, "current storage location"):
            await nas.queue_transfer("org/model", "local", ["worker-a"])

        self.assertEqual(nas.jobs, [])

    async def test_worker_revalidates_source_transferability_before_export(self):
        nas = VirtualNAS(
            Path(self.temp.name), lambda: self.hub, FakeRegistry(), lambda: True,
        )
        nas.start = Mock()
        result = await nas.queue_transfer("org/model", "local", ["worker-a"])
        job = nas.jobs[0]

        async def storage(node_id):
            if node_id == "local":
                return {"models": [{
                    "model_id": "org/model", "size_bytes": 13,
                    "partial": False, "transferable": False,
                }]}
            return {"models": [], "free_size": 10 * 1024 * 1024 * 1024}

        nas._node_storage = AsyncMock(side_effect=storage)
        nas.export_model = Mock()

        await nas._run_transfer(job)

        self.assertEqual(result["job_ids"], [job["id"]])
        self.assertEqual(job["status"], "failed")
        self.assertIn("current storage location", job["error"])
        nas.export_model.assert_not_called()

    async def test_external_comfyui_source_model_transfers_to_worker(self):
        root = Path(self.temp.name) / "ComfyUI" / "models"
        bundle_files = {
            "diffusion_models/minimax_music3_dit_fp16.safetensors": b"dit-weights",
            "text_encoders/minimax_music3_text_encoder_pruned_int8_convrot.safetensors": b"encoder-weights",
            "vae/minimax_music3_dav.safetensors": b"vae-weights",
        }
        for relative, content in bundle_files.items():
            path = root.joinpath(*relative.split("/"))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        registry = FakeRegistry()

        async def request(node_id, method, path, **kwargs):
            if "worker-a" in registry.received:
                return {"models": [{
                    "model_id": "Comfy-Org/MiniMax-Music-3",
                    "size_bytes": len(registry.received["worker-a"]),
                    "partial": False,
                }], "free_size": 10 * 1024 * 1024 * 1024}
            return {"models": [], "free_size": 10 * 1024 * 1024 * 1024}

        registry.request = request
        nas = VirtualNAS(
            Path(self.temp.name), lambda: self.hub, registry, lambda: True,
            external_model_roots_provider=lambda: [root],
        )

        result = await nas.queue_transfer(
            "Comfy-Org/MiniMax-Music-3", "local", ["worker-a"],
        )
        final = await self.wait_final(nas, 1)

        self.assertEqual(len(result["job_ids"]), 1)
        self.assertEqual(final[0]["status"], "completed")
        target_hub = Path(self.temp.name) / "worker-hub"
        target = VirtualNAS(
            Path(self.temp.name) / "worker", lambda: target_hub,
            FakeRegistry(), lambda: True,
        )
        await target.import_model(
            "Comfy-Org/MiniMax-Music-3",
            bytes_stream(registry.received["worker-a"]),
        )
        copied = target_hub / "models--Comfy-Org--MiniMax-Music-3"
        self.assertEqual((copied / "refs" / "main").read_bytes(), b"comfyui")
        self.assertEqual(
            (copied / "snapshots" / "comfyui" / ".sparkdeck-external").read_bytes(),
            b"comfyui\n",
        )
        for relative, content in bundle_files.items():
            self.assertEqual(
                (copied / "snapshots" / "comfyui").joinpath(
                    *relative.split("/"),
                ).read_bytes(),
                content,
            )
        # The transfer copies; the ComfyUI install is left untouched.
        for relative, content in bundle_files.items():
            self.assertEqual(
                root.joinpath(*relative.split("/")).read_bytes(), content,
            )
        await nas.stop()

    async def test_transfer_sizes_external_bundle_without_partial_residue(self):
        root = Path(self.temp.name) / "ComfyUI" / "models"
        bundle_files = {
            "diffusion_models/minimax_music3_dit_fp16.safetensors": b"dit-weights",
            "text_encoders/minimax_music3_text_encoder_pruned_int8_convrot.safetensors": b"encoder-weights",
            "vae/minimax_music3_dav.safetensors": b"vae-weights",
        }
        for relative, content in bundle_files.items():
            path = root.joinpath(*relative.split("/"))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        bundle_size = sum(len(content) for content in bundle_files.values())
        available = virtual_nas.transfer_required_free_bytes(bundle_size)
        # Partial Hugging Face cache residue merges into the same inventory
        # entry, inflating size_bytes far beyond the transferable payload.
        residue = (
            self.hub / "models--Comfy-Org--MiniMax-Music-3"
            / "blobs" / "weights.incomplete"
        )
        residue.parent.mkdir(parents=True)
        residue.write_bytes(b"x" * 10_000_000)
        (self.hub / "models--Comfy-Org--MiniMax-Music-3"
         / "snapshots" / RESOLVED_REVISION).mkdir(parents=True)
        registry = FakeRegistry()

        async def request(node_id, method, path, **kwargs):
            if "worker-a" in registry.received:
                return {"models": [{
                    "model_id": "Comfy-Org/MiniMax-Music-3",
                    "size_bytes": len(registry.received["worker-a"]),
                    "partial": False,
                }], "free_size": available}
            return {"models": [], "free_size": available}

        registry.request = request
        nas = VirtualNAS(
            Path(self.temp.name), lambda: self.hub, registry, lambda: True,
            external_model_roots_provider=lambda: [root],
        )
        merged = next(
            item for item in nas.inventory()
            if item["model_id"] == "Comfy-Org/MiniMax-Music-3"
        )
        self.assertEqual(merged["size_bytes"], 10_000_000 + bundle_size)
        self.assertEqual(merged["transfer_size_bytes"], bundle_size)

        # Capacity fits the external payload plus its bounded reserve but not
        # the residue-inflated size; size against the external bundle alone.
        result = await nas.queue_transfer(
            "Comfy-Org/MiniMax-Music-3", "local", ["worker-a"],
        )
        final = await self.wait_final(nas, 1)

        self.assertEqual(len(result["job_ids"]), 1)
        # The queued job is sized by the external bundle alone; completion
        # with 80 MB free proves both capacity checks used that size, since
        # the residue-inflated sum would require ~87 MB.
        self.assertEqual(result["jobs"][0]["bytes_total"], bundle_size)
        self.assertEqual(final[0]["status"], "completed")
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
        required = virtual_nas.transfer_required_free_bytes(20)
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
        required = 20 + 64 * 1024 * 1024
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
        job = {
            "id": "download-1", "kind": "download", "model_id": "org/model",
            "source_node_id": "huggingface", "target_node_id": "worker-a",
            "revision": RESOLVED_REVISION, "status": "running",
            "bytes_total": 1000, "bytes_transferred": 970,
            "download_cache_baseline_bytes": 1000,
            "download_attempt_start_bytes": 970,
            "download_attempted_at": 100, "started_at": 99, "created_at": 1,
        }
        manager.virtual_nas.list_transfers.return_value = {"items": [job]}
        manager.node_registry = Mock()
        manager.node_registry.get.return_value = {
            "id": "worker-a", "name": "Worker A",
        }
        inventory = {
            "id": "worker-a", "name": "Worker A", "online": True,
            "models": [{
                "model_id": "org/model", "size_bytes": 1970,
                "partial": False, "has_partial_download": True,
                "partial_revision_size_bytes": {RESOLVED_REVISION: 970},
            }],
        }
        manager.model_cache_inventory = AsyncMock(return_value=[inventory])

        with patch("manager._monotonic", side_effect=[100, 102]):
            first = await manager.virtual_nas_inventory()
            inventory["models"][0]["size_bytes"] = 1975
            inventory["models"][0]["partial_revision_size_bytes"][RESOLVED_REVISION] = 975
            cached = await manager.virtual_nas_inventory()
            manager._invalidate_virtual_nas_nodes()
            result = await manager.virtual_nas_inventory()

        self.assertIsNone(first["jobs"][0]["bytes_per_second"])
        self.assertIsNone(cached["jobs"][0]["bytes_per_second"])
        self.assertEqual(result["jobs"][0]["bytes_transferred"], 975)
        self.assertEqual(result["jobs"][0]["progress"], 0.975)
        self.assertEqual(result["jobs"][0]["bytes_per_second"], 2.5)

    def test_public_job_only_reports_sampled_live_transfer_rate(self):
        manager = Manager.__new__(Manager)
        manager.settings = {"cluster_node_name": "Coordinator"}
        manager.node_registry = Mock()
        manager.node_registry.get.return_value = {
            "id": "worker-a", "name": "Worker A",
        }

        result = manager._public_virtual_nas_job({
            "id": "transfer-1", "kind": "transfer", "model_id": "org/model",
            "source_node_id": "local", "target_node_id": "worker-a",
            "status": "running", "bytes_total": 10_000_000_000,
            "bytes_transferred": 5_000_000_000,
            "bytes_per_second": 625_000_000,
            "created_at": 99, "started_at": 100, "completed_at": None,
        })
        unsampled = manager._public_virtual_nas_job({
            "id": "download-1", "kind": "download", "model_id": "org/model",
            "source_node_id": "huggingface", "target_node_id": "worker-a",
            "status": "completed", "bytes_total": 5_000_000_000,
            "bytes_transferred": 5_000_000_000,
            "download_attempt_start_bytes": 0,
            "download_attempted_at": 100, "completed_at": 101,
            "created_at": 99,
        })

        self.assertEqual(result["bytes_per_second"], 625_000_000)
        self.assertIsNone(unsampled["bytes_per_second"])


if __name__ == "__main__":
    unittest.main()
