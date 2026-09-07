import asyncio
import tempfile
import unittest
from pathlib import Path

from sparkdeck.virtual_nas import VirtualNAS


class ShieldedTransferAdmissionTests(unittest.IsolatedAsyncioTestCase):
    async def test_admitted_parent_can_start_shielded_work_during_update(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            nas = VirtualNAS(root, lambda: root / "hub", None, lambda: True)
            for admission in ("queue", "endpoint"):
                with self.subTest(admission=admission):
                    parent = asyncio.current_task()
                    if admission == "queue":
                        nas._active["job"] = parent
                    else:
                        nas._transfer_tasks[parent] = 1
                    nas.reserve_update()

                    async def child_work():
                        nas._reserve_stream("org/model")
                        try:
                            await asyncio.sleep(0)
                            return "completed"
                        finally:
                            nas._release_stream("org/model")

                    result = await nas._await_uncancelable(child_work())
                    self.assertEqual(result, "completed")
                    self.assertFalse(nas._streaming_models)
                    self.assertEqual(set(nas._transfer_tasks), {parent} if admission == "endpoint" else set())
                    nas._active.clear()
                    nas._transfer_tasks.clear()
                    nas.end_update()

    async def test_shielding_does_not_admit_new_work_during_update(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            nas = VirtualNAS(root, lambda: root / "hub", None, lambda: True)
            nas.reserve_update()

            async def child_work():
                nas._reserve_stream("org/model")

            with self.assertRaisesRegex(RuntimeError, "update is pending"):
                await nas._await_uncancelable(child_work())
            self.assertFalse(nas._transfer_tasks)
            self.assertFalse(nas._streaming_models)
