import asyncio
from pathlib import Path
import sys
import tempfile
import unittest
import uuid
from unittest.mock import AsyncMock, Mock, patch

from sparkdeck.virtual_nas import VirtualNAS, TransferCanceled, VIRTUAL_NAS_DOWNLOAD_CANCEL_CAPABILITY


class ManagedDownloadCancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_running_local_writer_dies_before_canceled_and_cache_stops_growing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nas = VirtualNAS(root, lambda: root / "hub", Mock(), lambda: True)
            nas._save = Mock()
            nas.estimate_download_size = AsyncMock(return_value=100)
            nas._node_storage = AsyncMock(return_value={"models": [], "free_size": 10**12})
            job = {"id": str(uuid.uuid4()), "model_id": "org/model", "revision": "a" * 40,
                   "requested_revision": "main", "source_node_id": "huggingface", "target_node_id": "local",
                   "status": "queued", "kind": "download", "bytes_total": 100}
            dependent = {"id": str(uuid.uuid4()), "depends_on_job_id": job["id"], "status": "queued"}
            nas.jobs = [job, dependent]
            payload_file = root / "partial.incomplete"
            worker_script = "import sys,json,time; p=json.load(sys.stdin); f=open(sys.argv[1],'ab',buffering=0);\nwhile True: f.write(b'x'*1024); time.sleep(.01)"
            launch = asyncio.create_subprocess_exec
            processes = []

            async def fake_hf_worker(*_args, **kwargs):
                process = await launch(sys.executable, "-c", worker_script, str(payload_file), **kwargs)
                processes.append(process)
                return process

            with patch("sparkdeck.virtual_nas.asyncio.create_subprocess_exec", side_effect=fake_hf_worker):
                running = asyncio.create_task(nas._run_download(job))
                for _ in range(200):
                    if payload_file.exists() and payload_file.stat().st_size:
                        break
                    await asyncio.sleep(.01)
                self.assertTrue(payload_file.exists())
                stopping = await nas.cancel_transfer(job["id"])
                self.assertEqual(stopping["phase"], "canceling")
                await asyncio.wait_for(running, 5)
            self.assertIsNotNone(processes[0].returncode)
            self.assertEqual(job["status"], "canceled")
            self.assertEqual(dependent["status"], "canceled")
            size = payload_file.stat().st_size
            await asyncio.sleep(.1)
            self.assertEqual(payload_file.stat().st_size, size)

    async def test_cancel_before_delayed_request_prevents_any_worker_start(self):
        with tempfile.TemporaryDirectory() as directory:
            nas = VirtualNAS(Path(directory), lambda: Path(directory) / "hub", Mock(), lambda: True)
            operation_id = str(uuid.uuid4())
            result = await nas.cancel_download_operation(operation_id)
            self.assertEqual(result["status"], "canceled")
            with patch("sparkdeck.virtual_nas.asyncio.create_subprocess_exec", AsyncMock()) as launch:
                with self.assertRaises(TransferCanceled):
                    await nas._download_model_process("org/model", "a" * 40, operation_id=operation_id)
            launch.assert_not_awaited()

    async def test_remote_cancel_failure_retains_running_ownership_and_can_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            nas = VirtualNAS(Path(directory), lambda: Path(directory) / "hub", Mock(), lambda: True)
            nas._save = Mock()
            event = asyncio.Event()
            writer = asyncio.Event()
            stopped = asyncio.Event()
            job = {"status": "running", "phase": "downloading"}
            attempts = 0

            async def cancel_writer():
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise RuntimeError("offline")
                stopped.set()
                writer.set()

            task = asyncio.create_task(nas._await_download_with_cancel(writer.wait(), event, cancel_writer, job))
            event.set()
            for _ in range(100):
                if job.get("error"):
                    break
                await asyncio.sleep(.01)
            self.assertEqual(job["status"], "running")
            self.assertEqual(job["phase"], "canceling")
            self.assertFalse(task.done())
            event.set()
            with self.assertRaises(TransferCanceled):
                await asyncio.wait_for(task, 2)
            self.assertTrue(stopped.is_set())

    async def test_remote_job_sends_operation_id_and_confirms_stop_before_terminal_state(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = Mock()
            nas = VirtualNAS(Path(directory), lambda: Path(directory) / "hub", registry, lambda: True)
            nas._save = Mock()
            nas.estimate_download_size = AsyncMock(return_value=100)
            nas._node_storage = AsyncMock(return_value={"models": [], "free_size": 10**12})
            nas._validate_download_node = AsyncMock(return_value={"capabilities": [VIRTUAL_NAS_DOWNLOAD_CANCEL_CAPABILITY]})
            job = {"id": str(uuid.uuid4()), "model_id": "org/model", "revision": "a" * 40,
                   "requested_revision": "main", "source_node_id": "huggingface", "target_node_id": "ws1",
                   "status": "queued", "kind": "download", "bytes_total": 100}
            nas.jobs = [job]
            started, writer_stopped, allow_stop = asyncio.Event(), asyncio.Event(), asyncio.Event()

            async def request(node, method, path, **kwargs):
                if path.endswith("/cancel"):
                    self.assertEqual(path, f"/api/agent/virtual-nas/downloads/{job['id']}/cancel")
                    self.assertEqual(job["status"], "running")
                    await allow_stop.wait()
                    writer_stopped.set()
                    return {"operation_id": job["id"], "status": "canceled"}
                self.assertEqual(kwargs["json_body"]["operation_id"], job["id"])
                started.set()
                await writer_stopped.wait()
                return {"status": "canceled"}

            registry.request = AsyncMock(side_effect=request)
            active = asyncio.create_task(nas._run_download(job))
            await asyncio.wait_for(started.wait(), 2)
            await nas.cancel_transfer(job["id"])
            await asyncio.sleep(.02)
            self.assertEqual(job["status"], "running")
            self.assertEqual(job["phase"], "canceling")
            allow_stop.set()
            await asyncio.wait_for(active, 2)
            self.assertTrue(writer_stopped.is_set())
            self.assertEqual(job["status"], "canceled")
