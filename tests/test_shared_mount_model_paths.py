"""Model paths reachable from several nodes are not controller-local disk.

A shared mount (NFS/CIFS/FUSE/SparkNest) exposes the same absolute path on
every node that mounts it, so a multi-node launch must not be refused as a
"controller-local model path".  Reachability is decided per node by the
agent path probe: a selected node that exposes the directory may join the
launch, and one that does not is rejected before any topology mutates.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import httpx

from sparkdeck.models import Deployment, DeploymentKind, ModelIdentity, RuntimeKind
from sparkdeck.service import SparkDeckService

SNAPSHOT = (
    "/mnt/sparknest/hf-home/hub/"
    "models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e10"
)
REPO_ROOT = "/mnt/sparknest/hf-home/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017"


class FakeManager:
    """Only the members the path-validated start flow touches."""

    def __init__(self):
        self.http = httpx.AsyncClient()
        self._resolve_local_path = Mock(return_value=REPO_ROOT)
        self._remote_path_available = True
        self.deployments = []
        self.deployment_action = AsyncMock(return_value={"ok": True, "errors": []})

    async def node_has_local_path(self, node_id, path):
        return self._remote_path_available

    def _deployment_served_models(self, payload):
        return []

    def _without_sensitive_cli_credentials(self, settings):
        return settings


class SharedMountPathTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.manager = FakeManager()
        self.service = SparkDeckService(self.manager, Path(self.temp.name))

    async def asyncTearDown(self):
        await self.manager.http.aclose()
        await self.service.close()
        self.temp.cleanup()

    async def test_start_selection_allows_remote_nodes_that_expose_the_path(self):
        deployment = {
            "runtime": "vllm",
            "model": {"repository": SNAPSHOT},
            "settings": {},
        }
        await self.service._validate_start_selection(
            deployment, ["local", "remote-1"], None,
        )

    async def test_start_selection_rejects_remote_nodes_without_the_path(self):
        self.manager._remote_path_available = False
        deployment = {
            "runtime": "vllm",
            "model": {"repository": SNAPSHOT},
            "settings": {},
        }
        with self.assertRaisesRegex(ValueError, "local model path is not available"):
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
        self.manager.deployment_action = AsyncMock(
            return_value={"ok": True, "errors": []},
        )
        await self.service.deployment_action(
            "dep-nas", "start", node_ids=["local", "remote-1"],
        )
        self.manager.deployment_action.assert_awaited()


if __name__ == "__main__":
    unittest.main()
