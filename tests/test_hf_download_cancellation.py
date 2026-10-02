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
            failed_stop, allow_stop = asyncio.Event(), asyncio.Event()
            actual_stop = nas._stop_download_process

            async def guarded_stop(process):
                if not allow_stop.is_set():
                    failed_stop.set()
                    raise PermissionError("simulated transient stop failure")
                await actual_stop(process)

            async def fake_hf_worker(*_args, **kwargs):
                process = await launch(sys.executable, "-c", worker_script, str(payload_file), **kwargs)
                processes.append(process)
                return process

            with (
                patch("sparkdeck.virtual_nas.asyncio.create_subprocess_exec", side_effect=fake_hf_worker),
                patch.object(nas, "_stop_download_process", side_effect=guarded_stop),
            ):
                running = asyncio.create_task(nas._run_download(job))
                for _ in range(200):
                    if payload_file.exists() and payload_file.stat().st_size:
                        break
                    await asyncio.sleep(.01)
                self.assertTrue(payload_file.exists())
                stopping = await nas.cancel_transfer(job["id"])
                self.assertEqual(stopping["phase"], "canceling")
                await asyncio.wait_for(failed_stop.wait(), 2)
                self.assertFalse(running.done())
                self.assertIsNone(processes[0].returncode)
                self.assertTrue(nas._process_download_locks["org/model"].locked())
                self.assertEqual(job["status"], "running")
                allow_stop.set()
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
            # A delayed request must also remain rejected after agent restart.
            nas = VirtualNAS(Path(directory), lambda: Path(directory) / "hub", Mock(), lambda: True)
            with patch("sparkdeck.virtual_nas.asyncio.create_subprocess_exec", AsyncMock()) as launch:
                with self.assertRaises(TransferCanceled):
                    await nas._download_model_process("org/model", "a" * 40, operation_id=operation_id)
            launch.assert_not_awaited()

    async def test_cancellation_during_spawn_reaps_new_process_without_starting_download(self):
        with tempfile.TemporaryDirectory() as directory:
            nas = VirtualNAS(Path(directory), lambda: Path(directory) / "hub", Mock(), lambda: True)
            launch_started, release_launch = asyncio.Event(), asyncio.Event()
            launch = asyncio.create_subprocess_exec
            processes = []

            async def delayed_launch(*_args, **kwargs):
                launch_started.set()
                await release_launch.wait()
                process = await launch(sys.executable, "-c", "import time; time.sleep(60)", **kwargs)
                processes.append(process)
                return process

            with patch("sparkdeck.virtual_nas.asyncio.create_subprocess_exec", side_effect=delayed_launch):
                task = asyncio.create_task(nas._download_model_process("org/model", "a" * 40))
                await asyncio.wait_for(launch_started.wait(), 2)
                task.cancel()
                release_launch.set()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 3)
            self.assertIsNotNone(processes[0].returncode)

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

    async def test_dropped_http_request_retains_ownership_until_remote_stop_confirmed(self):
        with tempfile.TemporaryDirectory() as directory:
            nas = VirtualNAS(Path(directory), lambda: Path(directory) / "hub", Mock(), lambda: True)
            nas._save = Mock()
            event, allow_stop = asyncio.Event(), asyncio.Event()
            job = {"status": "running"}

            async def dropped_request():
                raise RuntimeError("HTTP connection dropped")

            async def cancel_writer():
                if not allow_stop.is_set():
                    raise RuntimeError("stop acknowledgment unavailable")

            active = asyncio.create_task(nas._await_download_with_cancel(dropped_request(), event, cancel_writer, job))
            for _ in range(100):
                if job.get("error"):
                    break
                await asyncio.sleep(.01)
            self.assertFalse(active.done())
            self.assertEqual(job["status"], "running")
            allow_stop.set()
            event.set()
            with self.assertRaisesRegex(RuntimeError, "HTTP connection dropped"):
                await asyncio.wait_for(active, 2)

    async def test_shutdown_waits_for_remote_exit_ack_and_closes_hung_http_waiter(self):
        with tempfile.TemporaryDirectory() as directory:
            nas = VirtualNAS(Path(directory), lambda: Path(directory) / "hub", Mock(), lambda: True)
            nas._save = Mock()
            event, allow_stop, attempted, http_started = (asyncio.Event() for _ in range(4))
            job = {"status": "running"}

            async def hung_request():
                http_started.set()
                await asyncio.Event().wait()

            async def cancel_writer():
                attempted.set()
                if not allow_stop.is_set():
                    raise RuntimeError("offline")

            active = asyncio.create_task(nas._await_download_with_cancel(hung_request(), event, cancel_writer, job))
            await http_started.wait()
            active.cancel()
            await attempted.wait()
            self.assertFalse(active.done())
            self.assertEqual(job["status"], "running")
            allow_stop.set()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(active, 3)

    async def test_requeued_job_rejected_by_agent_tombstone_never_completes_partial_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = Mock()
            nas = VirtualNAS(Path(directory), lambda: Path(directory) / "hub", registry, lambda: True)
            nas._save = Mock()
            nas.estimate_download_size = AsyncMock(return_value=100)
            nas._node_storage = AsyncMock(return_value={"models": [], "free_size": 10**12})
            nas._validate_download_node = AsyncMock(return_value={"capabilities": [VIRTUAL_NAS_DOWNLOAD_CANCEL_CAPABILITY]})
            job = {"id": str(uuid.uuid4()), "model_id": "org/model", "revision": "a" * 40,
                   "requested_revision": "main", "source_node_id": "huggingface", "target_node_id": "ws1",
                   "status": "queued", "kind": "download", "bytes_total": 100, "bytes_transferred": 25}
            nas.jobs = [job]
            registry.request = AsyncMock(return_value={"operation_id": job["id"], "status": "canceled"})
            await nas._run_download(job)
            self.assertEqual(job["status"], "canceled")
            self.assertEqual(job["bytes_transferred"], 25)

    async def test_worker_stops_when_its_agent_parent_exits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            partial = root / "partial.incomplete"
            # The parent holds a child process writing simulated cache bytes.
            # It exits without cooperative cancellation, as an agent crash can.
            child_script = (
                "from sparkdeck.hf_download_worker import _guard_parent_lifetime; "
                "import sys,os,time; _guard_parent_lifetime(int(sys.argv[1])); "
                "f=open(sys.argv[2],'ab',buffering=0);\n"
                "while True: f.write(b'x'*1024); time.sleep(.01)"
            )
            parent_script = (
                "import subprocess,sys,os,time; "
                "p=subprocess.Popen([sys.executable,'-c',sys.argv[1],str(os.getpid()),sys.argv[2]]); "
                "time.sleep(60)"
            )
            parent = await asyncio.create_subprocess_exec(
                sys.executable, "-c", parent_script, child_script, str(partial),
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                for _ in range(300):
                    if partial.exists() and partial.stat().st_size:
                        break
                    await asyncio.sleep(.01)
                if not partial.exists():
                    parent.kill()
                    _out, error = await parent.communicate()
                    self.fail(error.decode())
                parent.kill()
                await asyncio.wait_for(parent.wait(), 3)
                await asyncio.sleep(.1)
                size = partial.stat().st_size
                await asyncio.sleep(.2)
                self.assertEqual(partial.stat().st_size, size)
            finally:
                if parent.returncode is None:
                    parent.kill()
                    await parent.wait()
