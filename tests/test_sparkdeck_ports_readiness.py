import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import httpx

from manager import Manager
from sparkdeck.models import Deployment, DeploymentKind, ModelIdentity, RuntimeKind
from sparkdeck.service import SparkDeckService


class PerNodePortTests(unittest.IsolatedAsyncioTestCase):
    async def test_remote_first_replicas_allocate_ports_on_their_own_nodes(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = Manager.__new__(Manager)
            manager.settings = {
                "cluster_fabric_ip": "169.254.10.1",
                "cluster_fabric_interface": "cx7-local",
            }
            manager.deployments_path = Path(directory) / "deployments.json"
            manager.deployments = []
            manager._allocate_port = AsyncMock(return_value=8007)
            manager.cluster_nodes = AsyncMock(return_value=[
                {
                    "id": "local", "name": "Controller", "local": True,
                    "online": True, "docker_ready": True,
                    "fabric_ip": "169.254.10.1", "fabric_interface": "cx7-local",
                    "interfaces": [],
                },
                {
                    "id": "remote-1", "name": "Worker", "local": False,
                    "online": True, "docker_ready": True,
                    "fabric_ip": "169.254.10.2", "fabric_interface": "cx7-remote",
                    "interfaces": [],
                },
            ])
            payloads = {}

            async def create_member(node_id, payload):
                payloads[node_id] = payload
                # The worker agent chooses 9101 from its own port namespace.
                port = 9101 if node_id == "remote-1" else payload["port"]
                return {
                    "id": f"container-{node_id}", "status": "running", "port": port,
                }

            manager._create_member = create_member

            deployment = await manager.create_deployment({
                "model": "org/model", "engine": "vllm",
                "deployment_mode": "replicated",
                "node_ids": ["remote-1", "local"],
            })

            self.assertIsNone(payloads["remote-1"]["port"])
            self.assertIsNone(payloads["remote-1"]["cluster_member"]["serve_port"])
            self.assertEqual(payloads["local"]["port"], 8007)
            self.assertEqual(deployment["members"][0]["port"], 9101)
            self.assertEqual(deployment["members"][1]["port"], 8007)
            self.assertEqual(deployment["api_port"], 9101)
            self.assertIsNone(deployment["launch_settings"]["port"])


class LocalReadinessTests(unittest.IsolatedAsyncioTestCase):
    async def test_running_container_is_model_visible_only_after_ready_phase(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = Mock()
            manager.http = httpx.AsyncClient()
            manager.deployments = []
            manager.list_containers = AsyncMock(return_value=[{
                "name": "local-model", "model": "org/model", "runtime": "vllm",
                "managed": True, "status": "running", "port": 8000,
                "phase": {"phase": "loading_model"},
            }])
            service = SparkDeckService(manager, Path(directory))
            service.store.add_deployment(Deployment(
                id="dep-1", alias="friendly", runtime=RuntimeKind.VLLM,
                kind=DeploymentKind.MANAGED, model=ModelIdentity("org/model"),
                container_name="local-model",
            ))

            self.assertEqual((await service.deployments())[0]["status"], "starting")
            self.assertEqual((await service.models())["data"], [])

            manager.list_containers.return_value[0]["phase"] = {"phase": "ready"}
            self.assertEqual((await service.deployments())[0]["status"], "running")
            self.assertEqual((await service.models())["data"][0]["id"], "friendly")

            await manager.http.aclose()
            await service.close()


if __name__ == "__main__":
    unittest.main()
