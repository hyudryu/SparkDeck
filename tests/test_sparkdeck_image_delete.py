import unittest
from unittest.mock import AsyncMock, Mock

from manager import Manager


def cluster_node(node_id: str, *, local: bool = False) -> dict:
    return {
        "id": node_id,
        "name": "Controller" if local else node_id,
        "local": local,
        "enabled": True,
        "online": True,
        "docker_ready": True,
    }


class ClusterImageDeleteTests(unittest.IsolatedAsyncioTestCase):
    async def test_delete_targets_every_owner_and_reports_node_failures(self):
        manager = Manager.__new__(Manager)
        manager.cluster_nodes = AsyncMock(return_value=[
            cluster_node("local", local=True),
            cluster_node("worker-1"),
            cluster_node("worker-2"),
        ])
        manager.remove_image = AsyncMock(return_value={"ok": True})
        manager.node_registry = Mock()

        async def remote(node_id, method, path, **_kwargs):
            self.assertEqual(method, "DELETE")
            self.assertEqual(path, "/api/agent/images/sha256%3Aa")
            if node_id == "worker-2":
                raise RuntimeError("worker offline")
            return {"ok": True}

        manager.node_registry.request = AsyncMock(side_effect=remote)

        result = await manager.remove_image_on_nodes(
            "sha256:a", ["local", "worker-1", "worker-2"],
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["node_ids"], ["local", "worker-1", "worker-2"])
        self.assertEqual(
            [(item["node_id"], item["ok"]) for item in result["results"]],
            [("local", True), ("worker-1", True), ("worker-2", False)],
        )
        self.assertIn("worker offline", result["results"][2]["error"])
        manager.remove_image.assert_awaited_once_with("sha256:a")

if __name__ == "__main__":
    unittest.main()
