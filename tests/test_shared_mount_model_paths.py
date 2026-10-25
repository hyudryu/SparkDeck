"""Model paths on shared network mounts are not controller-local disk.

A hub-cache snapshot served from an NFS/CIFS/FUSE mount is readable on every
node that mounts it at the same absolute path, so a multi-node launch must not
be refused as a "controller-local model path".  A path on this host's own disk
keeps the old refusal.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import httpx

from sparkdeck.models import Deployment, DeploymentKind, ModelIdentity, RuntimeKind
from sparkdeck.service import (
    SparkDeckService,
    _is_shared_network_path,
    _path_mount_fstype,
)

SNAPSHOT = (
    "/mnt/sparknest/hf-home/hub/"
    "models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e10"
)
REPO_ROOT = "/mnt/sparknest/hf-home/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017"

MOUNTINFO = (
    "22 1 8:1 / / rw,relatime shared:1 - ext4 /dev/sda1 rw\n"
    "36 35 98:0 / /home rw,relatime shared:12 - ext4 /dev/sda2 rw\n"
    "817 34 0:55 / /mnt/sparknest rw,nosuid,nodev,noatime shared:615"
    " - fuse.sparknest sparknest rw,user_id=1000,allow_other\n"
    "900 34 0:60 / /mnt/nfs-cache rw,relatime shared:700"
    " - nfs4 storage.example:/cache rw,vers=4.1\n"
)


class FakeManager:
    def __init__(self):
        self.http = httpx.AsyncClient()


class _Readable:
    """Minimal file-like wrapper so a patched ``open`` supports ``with``."""

    def __init__(self, text):
        self._lines = text.splitlines(keepends=True)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def readlines(self):
        return list(self._lines)


def _patched_mountinfo(text=MOUNTINFO):
    """Patch mountinfo + realpath so the parse is host-independent."""
    return (
        patch("builtins.open", Mock(return_value=_Readable(text))),
        patch("sparkdeck.service.os.path.realpath", side_effect=lambda value: value),
    )


class SharedMountPathTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.manager = FakeManager()
        self.service = SparkDeckService(self.manager, Path(self.temp.name))

    async def asyncTearDown(self):
        await self.manager.http.aclose()
        await self.service.close()
        self.temp.cleanup()

    def test_fstype_resolves_longest_mount(self):
        open_patch, realpath_patch = _patched_mountinfo()
        with open_patch, realpath_patch:
            self.assertEqual(_path_mount_fstype(SNAPSHOT), "fuse.sparknest")
            self.assertEqual(_path_mount_fstype("/home/user/weights"), "ext4")
            self.assertEqual(
                _path_mount_fstype("/mnt/nfs-cache/hub/model"), "nfs4",
            )

    def test_shared_mount_detection(self):
        open_patch, realpath_patch = _patched_mountinfo()
        with open_patch, realpath_patch:
            self.assertTrue(_is_shared_network_path(SNAPSHOT))
            self.assertTrue(_is_shared_network_path("/mnt/nfs-cache/hub/model"))
            self.assertFalse(_is_shared_network_path("/home/user/weights"))
            self.assertFalse(_is_shared_network_path("org/model"))
            self.assertFalse(_is_shared_network_path(None))

    def test_unreadable_mountinfo_is_not_shared(self):
        with patch("builtins.open", side_effect=OSError("no mountinfo")):
            self.assertFalse(_is_shared_network_path(SNAPSHOT))

    def test_local_directory_is_not_shared(self):
        self.assertFalse(_is_shared_network_path(self.temp.name))

    async def test_start_selection_allows_remote_nodes_for_shared_mount(self):
        deployment = {
            "runtime": "vllm",
            "model": {"repository": SNAPSHOT},
            "settings": {},
        }
        self.manager._resolve_local_path = Mock(return_value=REPO_ROOT)
        with patch("sparkdeck.service._is_shared_network_path", return_value=True):
            await self.service._validate_start_selection(
                deployment, ["local", "remote-1"], None,
            )
        with patch("sparkdeck.service._is_shared_network_path", return_value=False):
            with self.assertRaisesRegex(ValueError, "controller-local model paths"):
                await self.service._validate_start_selection(
                    deployment, ["local", "remote-1"], None,
                )

    async def test_start_of_shared_mount_deployment_reaches_manager(self):
        self.service.store.add_deployment(Deployment(
            id="dep-nas", alias="nas weights", runtime=RuntimeKind.VLLM,
            kind=DeploymentKind.MANAGED, model=ModelIdentity(SNAPSHOT),
            settings={"manager_deployment_id": "cluster-1"},
        ))
        self.manager.deployments = [{
            "id": "cluster-1", "status": "stopped", "engine": "vllm",
            "model": SNAPSHOT,
        }]
        self.manager._resolve_local_path = Mock(return_value=REPO_ROOT)
        self.manager.deployment_action = AsyncMock(
            return_value={"ok": True, "errors": []},
        )
        with patch("sparkdeck.service._is_shared_network_path", return_value=True):
            await self.service.deployment_action(
                "dep-nas", "start", node_ids=["local", "remote-1"],
            )
        self.manager.deployment_action.assert_awaited()


if __name__ == "__main__":
    unittest.main()
