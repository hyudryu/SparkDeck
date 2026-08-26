import asyncio
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock

from manager import DEFAULT_SETTINGS, Manager
from sparkdeck.virtual_nas import VirtualNAS, validate_model_id


def create_cached_model(hub: Path, model_id: str = "org/model") -> Path:
    owner, repository = model_id.split("/", 1)
    root = hub / f"models--{owner}--{repository}"
    (root / "blobs").mkdir(parents=True)
    (root / "snapshots" / "revision-1").mkdir(parents=True)
    (root / "blobs" / "weights").write_bytes(b"model-weights")
    (root / "snapshots" / "revision-1" / "config.json").write_text("{}")
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

    def get(self, node_id):
        return self.nodes.get(node_id)

    async def probe(self, node, force=False):
        return {**node, "online": True}

    async def request(self, node_id, method, path, **_kwargs):
        return {"models": []}

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
        return FakeResponse()


class InventoryAndArchiveTests(unittest.IsolatedAsyncioTestCase):
    def test_virtual_nas_is_disabled_by_default(self):
        self.assertIs(DEFAULT_SETTINGS["virtual_nas_enabled"], False)

    async def test_inventory_lists_only_complete_models_without_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            hub = Path(directory) / "hub"
            complete = create_cached_model(hub)
            (hub / "models--partial--repo" / "snapshots").mkdir(parents=True)
            nas = VirtualNAS(Path(directory), lambda: hub, FakeRegistry(), lambda: False)

            models = nas.inventory()

            self.assertEqual([item["model_id"] for item in models], ["org/model"])
            self.assertGreater(models[0]["size_bytes"], 0)
            self.assertNotIn(str(complete), json.dumps(models))
            self.assertNotIn("path", models[0])

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


class DeleteGuardTests(unittest.IsolatedAsyncioTestCase):
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


if __name__ == "__main__":
    unittest.main()
