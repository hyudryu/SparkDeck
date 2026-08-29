import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import httpx

from manager import Manager
from sparkdeck.service import SparkDeckService
from sparkdeck.virtual_nas import VirtualNAS, _is_complete_repository


class FakeManager:
    def __init__(self):
        self.http = httpx.AsyncClient()
        self.list_containers = AsyncMock(return_value=[])
        self.remove_container = AsyncMock(return_value={"ok": True})
        self.start_container = AsyncMock(return_value={"status": "running"})
        self.stop_container = AsyncMock(return_value={"status": "exited"})
        self._vllm_chat = AsyncMock()
        self._vllm_completions = AsyncMock()

    async def selected_cluster_nodes(self, node_ids):
        return [{"id": node_id, "name": node_id, "online": True} for node_id in node_ids]

    def public_target_node(self, node):
        return {"id": node["id"], "name": node["name"]}

    @property
    def settings(self):
        if not hasattr(self, "_settings"):
            self._settings = {"cluster_node_name": "Coordinator"}
        return self._settings


class GgufDistributionTests(unittest.IsolatedAsyncioTestCase):
    """The controller-first GGUF homes flow: one Hub seed, file-scoped fan-out."""

    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.manager = FakeManager()
        self.service = SparkDeckService(self.manager, Path(self.temp.name))
        self.revision = "a" * 40
        self.model_root = Path(self.temp.name) / "cache" / "models--org--model"
        self.artifact = (
            self.model_root / "snapshots" / self.revision / "UD" / "model-Q8.gguf"
        )
        self.artifact.parent.mkdir(parents=True)
        self.artifact.write_bytes(b"gguf")
        virtual_nas = Mock()
        virtual_nas.resolve_download_revision = AsyncMock(return_value={
            "resolved_revision": self.revision,
        })
        virtual_nas.download_model_files_checked = AsyncMock(return_value={"ok": True})
        virtual_nas._model_path = Mock(return_value=self.model_root)
        virtual_nas.enabled = True
        self.virtual_nas = virtual_nas
        self.manager.virtual_nas = virtual_nas
        self.manager.node_has_model_files = AsyncMock(return_value=False)
        self.manager.node_download_model_files = AsyncMock(return_value={"ok": True})
        self.manager.node_transfer_model_files = AsyncMock(return_value={"ok": True})
        self.manager.node_supports_selective_downloads = AsyncMock(return_value=True)

    async def asyncTearDown(self):
        await self.manager.http.aclose()
        await self.service.close()
        self.temp.cleanup()

    async def prepare(self, **kwargs):
        return await self.service._prepare_public_gguf_artifact(
            "org/model", "UD/model-Q8.gguf", "release-gguf", None, **kwargs,
        )

    async def test_home_without_artifact_receives_stream_from_seeded_controller(self):
        prepared = await self.prepare(home_node_ids=["local", "worker-1"])

        self.assertEqual(prepared, str(self.artifact))
        self.manager.node_download_model_files.assert_awaited_once_with(
            "local", "org/model", self.revision,
            ["UD/model-Q8.gguf"], "release-gguf",
        )
        self.manager.node_transfer_model_files.assert_awaited_once_with(
            "local", "worker-1", "org/model", self.revision,
            ["UD/model-Q8.gguf"], "release-gguf",
        )

    async def test_existing_remote_copy_streams_to_the_controller_without_hub(self):
        self.manager.node_has_model_files = AsyncMock(
            side_effect=lambda node_id, *args, **kwargs: node_id == "worker-1"
        )

        prepared = await self.prepare(home_node_ids=["local", "worker-1"])

        self.assertEqual(prepared, str(self.artifact))
        self.manager.node_download_model_files.assert_not_awaited()
        self.manager.node_transfer_model_files.assert_awaited_once_with(
            "worker-1", "local", "org/model", self.revision,
            ["UD/model-Q8.gguf"], "release-gguf",
        )

    async def test_all_homes_complete_skips_downloads_and_transfers(self):
        self.manager.node_has_model_files = AsyncMock(return_value=True)

        prepared = await self.prepare(home_node_ids=["local", "worker-1"])

        self.assertEqual(prepared, str(self.artifact))
        self.manager.node_download_model_files.assert_not_awaited()
        self.manager.node_transfer_model_files.assert_not_awaited()

    async def test_explicit_seed_node_is_used_for_the_hub_download(self):
        await self.prepare(home_node_ids=["local", "worker-1"], download_node_id="worker-1")

        self.manager.node_download_model_files.assert_awaited_once_with(
            "worker-1", "org/model", self.revision,
            ["UD/model-Q8.gguf"], "release-gguf",
        )
        self.manager.node_transfer_model_files.assert_awaited_once_with(
            "worker-1", "local", "org/model", self.revision,
            ["UD/model-Q8.gguf"], "release-gguf",
        )

    async def test_seed_outside_homes_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "download_node_id must be one of"):
            await self.prepare(home_node_ids=["local", "worker-1"], download_node_id="worker-9")

    async def test_failed_seed_download_reports_every_attempt(self):
        self.manager.node_download_model_files = AsyncMock(
            side_effect=RuntimeError("hub unreachable")
        )

        with self.assertRaisesRegex(RuntimeError, "no selected node could download"):
            await self.prepare(home_node_ids=["local", "worker-1"])

    async def test_unsupported_agent_blocks_distribution_before_downloading(self):
        self.manager.node_supports_selective_downloads = AsyncMock(
            side_effect=lambda node_id: node_id == "local"
        )

        with self.assertRaisesRegex(RuntimeError, "worker-1"):
            await self.prepare(home_node_ids=["local", "worker-1"])

        self.manager.node_download_model_files.assert_not_awaited()

    async def test_distribution_requires_the_virtual_nas(self):
        self.virtual_nas.enabled = False

        with self.assertRaisesRegex(RuntimeError, "Virtual NAS is required"):
            await self.prepare(home_node_ids=["local", "worker-1"])

    async def test_background_failure_keeps_a_visible_error_card(self):
        self.manager.node_has_model_files = AsyncMock(return_value=False)
        self.manager.node_download_model_files = AsyncMock(
            side_effect=RuntimeError("hub unreachable")
        )
        prepared = str(self.artifact)
        with (
            patch("sparkdeck.service.launch_managed_container", AsyncMock()),
        ):
            created = await self.service.create_deployment({
                "model": "org/model", "alias": "failed-pull",
                "runtime": "llama.cpp", "revision": "release-gguf",
                "settings": {"artifact": "UD/model-Q8.gguf"},
                "node_ids": ["local", "worker-1"],
            })

        self.assertEqual(created["status"], "starting")
        await self.service._deployment_launch_tasks[created["id"]]

        stored = self.service.store.deployment("failed-pull")
        self.assertIn("hub unreachable", stored["settings"]["launch_error"])
        listed = next(
            item for item in await self.service.deployments()
            if item["id"] == created["id"]
        )
        self.assertEqual(listed["status"], "error")
        self.assertIn("hub unreachable", listed["last_error"])

    async def test_remote_homes_are_rejected_for_absolute_local_artifacts(self):
        artifact = Path(self.temp.name) / "controller.gguf"
        artifact.write_bytes(b"gguf")

        with self.assertRaisesRegex(ValueError, "cannot be distributed"):
            await self.service.create_deployment({
                "model": "local/model", "alias": "local-gguf",
                "runtime": "llama.cpp",
                "settings": {"artifact": str(artifact)},
                "node_ids": ["local", "worker-1"],
            })


class ManagerSelectiveDownloadTests(unittest.IsolatedAsyncioTestCase):
    async def test_remote_seed_forwards_the_controller_credential(self):
        manager = Manager.__new__(Manager)
        manager.node_registry = Mock()
        manager.node_registry.get = Mock(return_value={
            "id": "worker-1", "capabilities": ["virtual-nas-files-download-v1"],
        })
        manager.node_registry.request = AsyncMock(return_value={"ok": True})
        manager._resolved_hf_token = Mock(return_value="hf-controller-token")

        await manager.node_download_model_files(
            "worker-1", "org/model", "a" * 40, ["q8/model.gguf"], "main",
        )

        self.assertEqual(
            manager.node_registry.request.await_args.kwargs["json_body"]["hf_token"],
            "hf-controller-token",
        )

    async def test_remote_seed_without_capability_fails_instead_of_whole_repo(self):
        manager = Manager.__new__(Manager)
        manager.node_registry = Mock()
        manager.node_registry.get = Mock(return_value={
            "id": "worker-1", "capabilities": [],
        })

        with self.assertRaisesRegex(RuntimeError, "does not support selective"):
            await manager.node_download_model_files(
                "worker-1", "org/model", "a" * 40, ["q8/model.gguf"], "main",
            )


class ManagerNodeFileCheckTests(unittest.IsolatedAsyncioTestCase):
    async def test_local_file_check_inspects_the_controller_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            hub = Path(directory) / "hub"
            nas = VirtualNAS(Path(directory), lambda: hub, Mock(), lambda: True)
            revision = "a" * 40
            snapshot = hub / "models--org--model" / "snapshots" / revision
            snapshot.mkdir(parents=True)
            (snapshot / "model.gguf").write_bytes(b"gguf")
            manager = Manager.__new__(Manager)
            manager.virtual_nas = nas

            present = await manager.node_has_model_files(
                "local", "org/model", revision, ["model.gguf"],
            )
            missing = await manager.node_has_model_files(
                "local", "org/model", revision, ["other.gguf"],
            )

            self.assertTrue(present)
            self.assertFalse(missing)


class SelectiveSnapshotRoundTripTests(unittest.IsolatedAsyncioTestCase):
    """Real cache-to-cache round trips for the file-scoped transfer."""

    def build_nodes(self, directory: Path):
        source_hub = directory / "source" / "hub"
        target_hub = directory / "target" / "hub"
        source = VirtualNAS(
            directory / "source", lambda: source_hub, Mock(), lambda: True,
        )
        target = VirtualNAS(
            directory / "target", lambda: target_hub, Mock(), lambda: True,
        )
        return source, target, source_hub, target_hub

    @staticmethod
    def seed_selective_source(hub: Path, revision: str) -> None:
        """Cache one UD quantization as a selective (marker-flagged) snapshot."""
        root = hub / "models--org--model"
        blob = root / "blobs" / "blob-q8"
        blob.parent.mkdir(parents=True)
        blob.write_bytes(b"q8-weights")
        snapshot = root / "snapshots" / revision
        (snapshot / "UD").mkdir(parents=True)
        entry = snapshot / "UD" / "model-Q8.gguf"
        if os.name != "nt":
            # Mirror the real POSIX Hugging Face cache: a relative symlink
            # into blobs/ (three levels up from a nested snapshot entry).
            os.symlink("../../../blobs/blob-q8", entry)
        else:
            # Windows without developer mode stores real files instead of
            # symlinks; the transfer path must support both cache layouts.
            entry.write_bytes(b"q8-weights")
        (snapshot / ".sparkdeck-selective.incomplete").write_text("selective\n")
        refs = root / "refs"
        refs.mkdir()
        (refs / "main").write_text(revision, encoding="utf-8")

    @staticmethod
    def stub_remote_target(source: VirtualNAS, target: VirtualNAS, target_hub: Path) -> None:
        """Wire source->target through a fake HTTP hop backed by real code.

        Inventory requests are answered from the target directory so the
        capacity gate and post-checks see real state, and import streams
        are delivered into the target's real import implementation.
        """

        async def fake_request(node_id, method, path, **kwargs):
            models = []
            if (target_hub / "models--org--model").exists():
                models = [{"model_id": "org/model", "size_bytes": 10}]
            return {"models": models, "free_size": 1 << 30}

        async def fake_open_stream(node_id, method, path, *, content=None, headers=None, timeout=600):
            if method == "PUT" and path.endswith("/files/import"):
                model_bytes = int(headers["X-SparkDeck-Model-Bytes"])
                await target.import_model_files(
                    "org/model", content, required_model_bytes=model_bytes,
                )
                response = Mock()
                response.status_code = 200

                async def aread():
                    return b'{"ok": true}'

                async def aclose():
                    return None

                response.aread = aread
                response.aclose = aclose
                return response
            raise AssertionError(f"unexpected agent stream: {method} {path}")

        source.node_registry.request = AsyncMock(side_effect=fake_request)
        source.node_registry.open_stream = fake_open_stream

    async def test_transfer_places_a_selective_snapshot_on_an_empty_target(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target, source_hub, target_hub = self.build_nodes(Path(directory))
            revision = "a" * 40
            self.seed_selective_source(source_hub, revision)
            self.stub_remote_target(source, target, target_hub)

            result = await source.transfer_model_files(
                "org/model", revision, ["UD/model-Q8.gguf"],
                "local", "local-2", "main",
            )

            self.assertTrue(result["ok"])
            target_file = (
                target_hub / "models--org--model"
                / "snapshots" / revision / "UD" / "model-Q8.gguf"
            )
            self.assertTrue(target_file.is_file())
            self.assertEqual(target_file.read_bytes(), b"q8-weights")
            marker = target_file.parent.parent / ".sparkdeck-selective.incomplete"
            self.assertTrue(marker.is_file())
            self.assertTrue(target.has_model_files(
                "org/model", revision, ["UD/model-Q8.gguf"],
            )["complete"])

    async def test_transfer_merges_into_a_target_caching_another_quantization(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target, source_hub, target_hub = self.build_nodes(Path(directory))
            revision = "a" * 40
            self.seed_selective_source(source_hub, revision)
            self.stub_remote_target(source, target, target_hub)
            target_root = target_hub / "models--org--model"
            q4_blob = target_root / "blobs" / "blob-q4"
            q4_blob.parent.mkdir(parents=True)
            q4_blob.write_bytes(b"q4-weights")
            q4_snapshot = target_root / "snapshots" / revision / "Q4"
            q4_snapshot.mkdir(parents=True)
            (q4_snapshot / "model-Q4.gguf").write_bytes(b"q4-weights")

            result = await source.transfer_model_files(
                "org/model", revision, ["UD/model-Q8.gguf"],
                "local", "local-2", "main",
            )

            self.assertTrue(result["ok"])
            self.assertEqual(
                q4_blob.read_bytes(), b"q4-weights",
                "merging must not disturb the target's existing quantization",
            )
            q8 = (
                target_root / "snapshots" / revision / "UD" / "model-Q8.gguf"
            )
            self.assertTrue(q8.is_file())
            self.assertEqual(q8.read_bytes(), b"q8-weights")
            self.assertTrue(target.has_model_files(
                "org/model", revision, ["UD/model-Q8.gguf"],
            )["complete"])

    async def test_stream_round_trip_between_two_real_cache_instances(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target, source_hub, _ = self.build_nodes(Path(directory))
            revision = "a" * 40
            self.seed_selective_source(source_hub, revision)

            export = source.export_model_files(
                "org/model", revision, ["UD/model-Q8.gguf"], "main",
            )
            result = await target.import_model_files(
                "org/model", export,
            )

            self.assertTrue(result["ok"])
            self.assertTrue(target.has_model_files(
                "org/model", revision, ["UD/model-Q8.gguf"],
            )["complete"])

    async def test_subset_of_a_complete_source_marks_the_target_selective(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target, source_hub, _ = self.build_nodes(Path(directory))
            revision = "c" * 40
            root = source_hub / "models--org--model"
            snapshot = root / "snapshots" / revision
            (snapshot / "UD").mkdir(parents=True)
            (snapshot / "UD" / "model-Q8.gguf").write_bytes(b"q8-weights")
            (snapshot / "tokenizer.json").write_text("{}")
            # No selective marker: the source cache holds a complete snapshot.

            export = source.export_model_files(
                "org/model", revision, ["UD/model-Q8.gguf"], "main",
            )
            await target.import_model_files("org/model", export)

            target_root = (
                Path(directory) / "target" / "hub" / "models--org--model"
            )
            marker = (
                target_root / "snapshots" / revision
                / ".sparkdeck-selective.incomplete"
            )
            self.assertTrue(marker.is_file())
            self.assertFalse(_is_complete_repository(target_root))
            self.assertFalse(target.has_model_files(
                "org/model", revision, ["tokenizer.json"],
            )["complete"])

    async def test_merge_keeps_a_complete_target_snapshot_unmarked(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target, source_hub, target_hub = self.build_nodes(Path(directory))
            revision = "a" * 40
            self.seed_selective_source(source_hub, revision)
            self.stub_remote_target(source, target, target_hub)
            target_root = target_hub / "models--org--model"
            q4_blob = target_root / "blobs" / "blob-q4"
            q4_blob.parent.mkdir(parents=True)
            q4_blob.write_bytes(b"q4-weights")
            q4_snapshot = target_root / "snapshots" / revision / "Q4"
            q4_snapshot.mkdir(parents=True)
            (q4_snapshot / "model-Q4.gguf").write_bytes(b"q4-weights")

            await source.transfer_model_files(
                "org/model", revision, ["UD/model-Q8.gguf"],
                "local", "local-2", "main",
            )

            marker = target_root / "snapshots" / revision / ".sparkdeck-selective.incomplete"
            self.assertFalse(marker.exists())
            self.assertTrue(_is_complete_repository(target_root))

    async def test_merge_updates_a_stale_revision_ref(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target, source_hub, target_hub = self.build_nodes(Path(directory))
            revision = "a" * 40
            self.seed_selective_source(source_hub, revision)
            self.stub_remote_target(source, target, target_hub)
            target_root = target_hub / "models--org--model"
            target_root.mkdir(parents=True)
            stale_ref = target_root / "refs" / "main"
            stale_ref.parent.mkdir(parents=True)
            stale_ref.write_text("b" * 40, encoding="utf-8")

            await source.transfer_model_files(
                "org/model", revision, ["UD/model-Q8.gguf"],
                "local", "local-2", "main",
            )

            self.assertEqual(
                stale_ref.read_text(encoding="utf-8").strip(), revision,
            )

    async def test_merge_refuses_a_destination_that_is_not_a_real_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target, source_hub, target_hub = self.build_nodes(Path(directory))
            revision = "a" * 40
            self.seed_selective_source(source_hub, revision)
            decoy = target_hub / "models--org--model"
            decoy.parent.mkdir(parents=True)
            decoy.write_bytes(b"not a directory")

            export = source.export_model_files(
                "org/model", revision, ["UD/model-Q8.gguf"], "main",
            )

            with self.assertRaisesRegex(RuntimeError, "not a safe directory"):
                await target.import_model_files("org/model", export)

    async def test_multi_shard_transfer_carries_every_selected_file(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target, _, _ = self.build_nodes(Path(directory))
            revision = "b" * 40
            root = Path(directory) / "source" / "hub" / "models--org--model"
            snapshot = root / "snapshots" / revision
            for index in (1, 2):
                shard = snapshot / f"model-{index:05d}-of-00002.gguf"
                shard.parent.mkdir(parents=True, exist_ok=True)
                shard.write_bytes(f"shard-{index}".encode())

            export = source.export_model_files(
                "org/model", revision,
                ["model-00001-of-00002.gguf", "model-00002-of-00002.gguf"],
            )
            await target.import_model_files("org/model", export)

            complete = target.has_model_files("org/model", revision, [
                "model-00001-of-00002.gguf", "model-00002-of-00002.gguf",
            ])
            self.assertTrue(complete["complete"])


class LlamaCppHomesContractTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.manager = FakeManager()
        self.service = SparkDeckService(self.manager, Path(self.temp.name))

    async def asyncTearDown(self):
        await self.manager.http.aclose()
        await self.service.close()
        self.temp.cleanup()

    async def test_homes_require_the_controller_first(self):
        with self.assertRaisesRegex(ValueError, "first selected node must be the controller"):
            self.service._llama_cpp_artifact_homes(
                ["worker-1", "local"], {"download_node_id": "worker-1"},
            )

    async def test_homes_default_to_controller_only_when_no_nodes_requested(self):
        homes, seed = self.service._llama_cpp_artifact_homes(None, {})

        self.assertIsNone(homes)
        self.assertIsNone(seed)

    async def test_remote_seed_is_rejected_for_the_controller_only_selection(self):
        with self.assertRaisesRegex(ValueError, "download_node_id must be one of"):
            self.service._llama_cpp_artifact_homes(
                None, {"download_node_id": "worker-1"},
            )

    async def test_controller_seed_without_explicit_nodes_is_accepted(self):
        homes, seed = self.service._llama_cpp_artifact_homes(
            None, {"download_node_id": "local"},
        )

        self.assertIsNone(homes)
        self.assertEqual(seed, "local")

    async def test_homes_keep_controller_order_and_report_the_requested_seed(self):
        homes, seed = self.service._llama_cpp_artifact_homes(
            ["local", "worker-1", "worker-2"], {"download_node_id": "worker-2"},
        )

        self.assertEqual(homes, ["local", "worker-1", "worker-2"])
        self.assertEqual(seed, "worker-2")


if __name__ == "__main__":
    unittest.main()
