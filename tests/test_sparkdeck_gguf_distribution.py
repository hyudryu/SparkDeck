import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import httpx

from manager import Manager
from sparkdeck.service import SparkDeckService
from sparkdeck.virtual_nas import VirtualNAS


class FakeManager:
    def __init__(self):
        self.http = httpx.AsyncClient()
        self.list_containers = AsyncMock(return_value=["local"])
        self.remove_container = AsyncMock(return_value={"ok": True})
        self.start_container = AsyncMock(return_value={"status": "running"})
        self.stop_container = AsyncMock(return_value={"status": "exited"})
        self._vllm_chat = AsyncMock()
        self._vllm_completions = AsyncMock()

    async def selected_cluster_nodes(self, node_ids):
        return [{"id": node_id, "name": node_id, "online": True} for node_id in node_ids]


class GgufDistributionTests(unittest.IsolatedAsyncioTestCase):
    """The controller-first GGUF homes flow: one Hub seed, NAS fan-out."""

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
        virtual_nas.list_transfers = Mock(return_value={"items": [
            {"id": "job-1", "status": "completed"},
        ]})
        virtual_nas.enabled = True
        self.virtual_nas = virtual_nas
        self.manager.virtual_nas = virtual_nas
        self.manager.node_has_model_files = AsyncMock(return_value=False)
        self.manager.node_download_model_files = AsyncMock(return_value={"ok": True})
        self.manager.node_supports_selective_downloads = AsyncMock(return_value=True)
        self.manager.queue_virtual_nas_transfer = AsyncMock(return_value={
            "job_ids": ["job-1"],
            "jobs": [{"id": "job-1", "status": "queued"}],
        })

    async def asyncTearDown(self):
        await self.manager.http.aclose()
        await self.service.close()
        self.temp.cleanup()

    async def prepare(self, **kwargs):
        return await self.service._prepare_public_gguf_artifact(
            "org/model", "UD/model-Q8.gguf", "release-gguf", None, **kwargs,
        )

    async def test_home_without_artifact_receives_transfer_from_seeded_controller(self):
        prepared = await self.prepare(home_node_ids=["local", "worker-1"])

        self.assertEqual(prepared, str(self.artifact))
        self.manager.node_download_model_files.assert_awaited_once_with(
            "local", "org/model", self.revision,
            ["UD/model-Q8.gguf"], "release-gguf",
        )
        self.manager.queue_virtual_nas_transfer.assert_awaited_once_with(
            "org/model", "local", ["worker-1"], self.revision,
            requested_revision="release-gguf",
        )

    async def test_existing_remote_copy_is_transferred_without_a_hub_download(self):
        self.manager.node_has_model_files = AsyncMock(
            side_effect=lambda node_id, *args, **kwargs: node_id == "worker-1"
        )

        prepared = await self.prepare(home_node_ids=["local", "worker-1"])

        self.assertEqual(prepared, str(self.artifact))
        self.manager.node_download_model_files.assert_not_awaited()
        self.manager.queue_virtual_nas_transfer.assert_awaited_once_with(
            "org/model", "worker-1", ["local"], self.revision,
            requested_revision="release-gguf",
        )

    async def test_all_homes_complete_skips_downloads_and_transfers(self):
        self.manager.node_has_model_files = AsyncMock(return_value=True)

        prepared = await self.prepare(home_node_ids=["local", "worker-1"])

        self.assertEqual(prepared, str(self.artifact))
        self.manager.node_download_model_files.assert_not_awaited()
        self.manager.queue_virtual_nas_transfer.assert_not_awaited()

    async def test_explicit_seed_node_is_used_for_the_hub_download(self):
        await self.prepare(home_node_ids=["local", "worker-1"], download_node_id="worker-1")

        self.manager.node_download_model_files.assert_awaited_once_with(
            "worker-1", "org/model", self.revision,
            ["UD/model-Q8.gguf"], "release-gguf",
        )
        self.manager.queue_virtual_nas_transfer.assert_awaited_once_with(
            "org/model", "worker-1", ["local"], self.revision,
            requested_revision="release-gguf",
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

    async def test_transfers_require_the_virtual_nas(self):
        self.virtual_nas.enabled = False

        with self.assertRaisesRegex(RuntimeError, "Virtual NAS is required"):
            await self.prepare(home_node_ids=["local", "worker-1"])

    async def test_failed_transfer_job_fails_preparation(self):
        self.virtual_nas.list_transfers = Mock(return_value={"items": [
            {"id": "job-1", "status": "failed", "error": "target offline"},
        ]})

        with self.assertRaisesRegex(RuntimeError, "target offline"):
            await self.prepare(home_node_ids=["local", "worker-1"])


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

    async def test_homes_keep_controller_order_and_report_the_requested_seed(self):
        homes, seed = self.service._llama_cpp_artifact_homes(
            ["local", "worker-1", "worker-2"], {"download_node_id": "worker-2"},
        )

        self.assertEqual(homes, ["local", "worker-1", "worker-2"])
        self.assertEqual(seed, "worker-2")


if __name__ == "__main__":
    unittest.main()
