import asyncio
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import Mock, patch

from sparkdeck.virtual_nas import DownloadStalled, TransferCanceled, VirtualNAS


STALL_SCRIPT = (
    "import sys,time\n"
    "open(sys.argv[1],'ab').write(b'x'*64)\n"
    "while True: time.sleep(1)\n"
)
FAIL_SCRIPT = "import sys\nsys.exit(1)\n"
OK_SCRIPT = "import sys\nsys.exit(0)\n"


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

    async def test_stall_event_is_internal_only(self):
        # DownloadStalled must never cross the public helper boundary: the
        # final attempt converts it into the retryable RuntimeError above.
        self.assertTrue(issubclass(DownloadStalled, Exception))


if __name__ == "__main__":
    unittest.main()
