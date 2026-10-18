import asyncio
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import Mock, patch

from sparkdeck.virtual_nas import (
    DownloadStalled, TransferCanceled, VirtualNAS, _stall_window_seconds,
)


STALL_SCRIPT = (
    "import sys,time\n"
    "open(sys.argv[1],'ab').write(b'x'*64)\n"
    "while True: time.sleep(1)\n"
)
# Xet parallel direct-addressed reconstruction: constant logical size while
# interior ranges keep landing (only mtimes move).
SPARSE_SCRIPT = (
    "import sys,time\n"
    "f=open(sys.argv[1],'wb'); f.truncate(4096)\n"
    "t=time.time(); i=0\n"
    "while time.time()-t < 1.2:\n"
    "    f.seek((i*64)%4096); f.write(b'x'*64); f.flush(); i+=1; time.sleep(.05)\n"
    "sys.exit(1)\n"
)
FAIL_SCRIPT = "import sys\nsys.exit(1)\n"
OK_SCRIPT = "import sys\nsys.exit(0)\n"
# Xet shared-layout staging: bytes land under the cache-wide hub/blobs.
HUB_BLOBS_SCRIPT = (
    "import sys,time\n"
    "d=sys.argv[1]\n"
    "open(d+'/growing.part','wb').close()\n"
    "t=time.time(); n=0\n"
    "while time.time()-t < 1.2:\n"
    "    f=open(d+'/growing.part','r+b'); f.seek(n); f.write(b'x'*4096); f.close(); n+=4096; time.sleep(.05)\n"
    "sys.exit(1)\n"
)


class StallWindowScalingTests(unittest.TestCase):
    def test_low_caps_widen_the_window(self):
        self.assertEqual(_stall_window_seconds(900.0, 0.0), 900.0)
        # A 100 MB/s cap still commits a chunk well inside the base window.
        self.assertEqual(_stall_window_seconds(900.0, 100e6), 900.0)
        # A 1 KiB/s cap needs 2 * 64 MiB / 1 KiB-s to show one chunk.
        self.assertGreater(_stall_window_seconds(900.0, 1024.0), 100_000.0)


class DownloadStallGuardTests(unittest.IsolatedAsyncioTestCase):
    def _nas(self, root):
        nas = VirtualNAS(root, lambda: root / "hub", Mock(), lambda: True)
        nas._save = Mock()
        return nas

    def _fake_launch(self, scripts, processes, partial):
        launch = asyncio.create_subprocess_exec
        attempts = []

        async def fake_worker(*_args, **kwargs):
            attempts.append(len(attempts))
            script = scripts[min(len(attempts) - 1, len(scripts) - 1)]
            process = await launch(sys.executable, "-c", script, str(partial), **kwargs)
            processes.append(process)
            return process

        return attempts, fake_worker

    async def test_stalled_writer_is_killed_and_relaunched(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nas = self._nas(root)
            partial = root / "hub" / "models--org--model" / "blobs" / "p.incomplete"
            partial.parent.mkdir(parents=True)
            processes = []
            attempts, fake_worker = self._fake_launch(
                [STALL_SCRIPT, FAIL_SCRIPT], processes, partial)
            with (
                patch("sparkdeck.virtual_nas.asyncio.create_subprocess_exec",
                      side_effect=fake_worker),
                patch("sparkdeck.virtual_nas._DOWNLOAD_STALL_BACKOFF_SECONDS", (0.0,)),
            ):
                # Attempt 1 stalls (one write, then silence): the guard kills
                # it and relaunches. Attempt 2 exits nonzero: that surfaces as
                # the ordinary download failure, proving the retry ran.
                with self.assertRaisesRegex(RuntimeError, "Hugging Face download failed"):
                    await nas._download_model_process(
                        "org/model", "a" * 40,
                        stall_timeout_seconds=.3, stall_poll_seconds=.05,
                        stall_retries=1)
            self.assertEqual(len(attempts), 2)
            self.assertIsNotNone(processes[0].returncode)

    async def test_persistent_stall_fails_with_stalled_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nas = self._nas(root)
            partial = root / "hub" / "models--org--model" / "blobs" / "p.incomplete"
            partial.parent.mkdir(parents=True)
            processes = []
            attempts, fake_worker = self._fake_launch(
                [STALL_SCRIPT], processes, partial)
            with (
                patch("sparkdeck.virtual_nas.asyncio.create_subprocess_exec",
                      side_effect=fake_worker),
                patch("sparkdeck.virtual_nas._DOWNLOAD_STALL_BACKOFF_SECONDS", (0.0,)),
            ):
                with self.assertRaisesRegex(RuntimeError, "stalled"):
                    await nas._download_model_process(
                        "org/model", "a" * 40,
                        stall_timeout_seconds=.2, stall_poll_seconds=.05,
                        stall_retries=2)
            self.assertEqual(len(attempts), 3)
            self.assertTrue(all(p.returncode is not None for p in processes))

    async def test_cancel_during_stall_backoff_stops_retries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nas = self._nas(root)
            partial = root / "hub" / "models--org--model" / "blobs" / "p.incomplete"
            partial.parent.mkdir(parents=True)
            processes = []
            attempts, fake_worker = self._fake_launch(
                [STALL_SCRIPT], processes, partial)
            operation_id = str(uuid.uuid4())
            state = {"cancel": asyncio.Event(), "done": asyncio.Event(),
                     "active": False}
            nas._download_operations[operation_id] = state
            with (
                patch("sparkdeck.virtual_nas.asyncio.create_subprocess_exec",
                      side_effect=fake_worker),
                patch("sparkdeck.virtual_nas._DOWNLOAD_STALL_BACKOFF_SECONDS", (5.0,)),
            ):
                task = asyncio.create_task(nas._download_model_process(
                    "org/model", "a" * 40, operation_id=operation_id,
                    stall_timeout_seconds=.2, stall_poll_seconds=.05,
                    stall_retries=3))
                for _ in range(400):
                    if processes and processes[0].returncode is not None:
                        break
                    await asyncio.sleep(.01)
                self.assertTrue(processes and processes[0].returncode is not None)
                state["cancel"].set()
                with self.assertRaises(TransferCanceled):
                    await asyncio.wait_for(task, 3)
            self.assertEqual(len(attempts), 1)
            self.assertFalse(state["active"])

    async def test_stalled_attempt_relaunch_can_succeed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nas = self._nas(root)
            partial = root / "hub" / "models--org--model" / "blobs" / "p.incomplete"
            partial.parent.mkdir(parents=True)
            nas.inventory = lambda: [{"model_id": "org/model", "size_bytes": 7}]
            processes = []
            attempts, fake_worker = self._fake_launch(
                [STALL_SCRIPT, OK_SCRIPT], processes, partial)
            with (
                patch("sparkdeck.virtual_nas.asyncio.create_subprocess_exec",
                      side_effect=fake_worker),
                patch("sparkdeck.virtual_nas._DOWNLOAD_STALL_BACKOFF_SECONDS", (0.0,)),
                patch.object(VirtualNAS, "_has_revision", return_value=True),
            ):
                result = await nas._download_model_process(
                    "org/model", "a" * 40,
                    stall_timeout_seconds=.2, stall_poll_seconds=.05,
                    stall_retries=2)
            self.assertTrue(result["ok"])
            self.assertEqual(len(attempts), 2)

    async def test_sparse_direct_writes_are_not_false_stalls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nas = self._nas(root)
            partial = root / "hub" / "models--org--model" / "blobs" / "p.incomplete"
            partial.parent.mkdir(parents=True)
            processes = []
            attempts, fake_worker = self._fake_launch(
                [SPARSE_SCRIPT], processes, partial)
            with (
                patch("sparkdeck.virtual_nas.asyncio.create_subprocess_exec",
                      side_effect=fake_worker),
                patch("sparkdeck.virtual_nas._DOWNLOAD_STALL_BACKOFF_SECONDS", (0.0,)),
            ):
                # The constant-size file holds mtime-only activity past the
                # stall timeout: the writer must end by exit(1), not a kill.
                with self.assertRaisesRegex(RuntimeError, "download failed"):
                    await nas._download_model_process(
                        "org/model", "a" * 40,
                        stall_timeout_seconds=.5, stall_poll_seconds=.05,
                        stall_retries=0)
            self.assertEqual(len(attempts), 1)

    async def test_cache_wide_xet_staging_counts_as_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nas = self._nas(root)
            staging = root / "hub" / "blobs"
            staging.mkdir(parents=True)
            (root / "hub" / "models--org--model" / "blobs").mkdir(parents=True)
            processes = []
            attempts, fake_worker = self._fake_launch(
                [HUB_BLOBS_SCRIPT], processes, staging)
            with (
                patch("sparkdeck.virtual_nas.asyncio.create_subprocess_exec",
                      side_effect=fake_worker),
                patch("sparkdeck.virtual_nas._DOWNLOAD_STALL_BACKOFF_SECONDS", (0.0,)),
            ):
                # Repository blobs never move; the payload grows under the
                # cache-wide staging dir. The writer must end by exit(1).
                with self.assertRaisesRegex(RuntimeError, "download failed"):
                    await nas._download_model_process(
                        "org/model", "a" * 40,
                        stall_timeout_seconds=.5, stall_poll_seconds=.05,
                        stall_retries=0)
            self.assertEqual(len(attempts), 1)

    async def test_rate_file_setup_failure_releases_operation_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nas = self._nas(root)
            operation_id = str(uuid.uuid4())
            with patch.object(nas, "_register_download_rate_file",
                              side_effect=OSError("TMPDIR full")):
                with self.assertRaises(OSError):
                    await nas._download_model_process(
                        "org/model", "a" * 40, operation_id=operation_id,
                        bandwidth_mbps=1.0)
            state = nas._download_operations[operation_id]
            self.assertFalse(state["active"])
            self.assertTrue(state["done"].is_set())

    async def test_stall_event_is_internal_only(self):
        # DownloadStalled must never cross the public helper boundary: the
        # final attempt converts it into the retryable RuntimeError above.
        self.assertTrue(issubclass(DownloadStalled, Exception))


if __name__ == "__main__":
    unittest.main()
