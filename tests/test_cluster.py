import asyncio
import json
import tempfile
import unittest
from pathlib import Path

import httpx

from cluster import AgentCredentials, NodeRegistry, normalize_agent_url
from manager import Manager


class AgentCredentialsTests(unittest.TestCase):
    def test_pairing_code_is_one_time_and_token_is_durable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            credentials = AgentCredentials(Path(directory))
            code = credentials.data["pairing_code"]

            paired = credentials.pair(code)

            self.assertTrue(credentials.accepts_token(paired["agent_token"]))
            self.assertNotEqual(credentials.data["pairing_code"], code)
            with self.assertRaisesRegex(ValueError, "invalid pairing code"):
                credentials.pair(code)
            self.assertEqual(
                oct((Path(directory) / "agent.json").stat().st_mode & 0o777),
                "0o600",
            )

    def test_agent_url_rejects_embedded_credentials(self) -> None:
        self.assertEqual(normalize_agent_url("http://spark-2:7878/"), "http://spark-2:7878")
        with self.assertRaises(ValueError):
            normalize_agent_url("http://user:secret@spark-2:7878")


class NodeRegistryTests(unittest.IsolatedAsyncioTestCase):
    async def test_pairing_persists_secret_but_returns_public_config(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.path, "/api/agent/pair")
            return httpx.Response(200, json={
                "node_id": "remote-1",
                "agent_token": "durable-secret",
                "name": "Spark 2",
                "protocol_version": 1,
            })

        with tempfile.TemporaryDirectory() as directory:
            client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            try:
                registry = NodeRegistry(Path(directory), client)
                public = await registry.pair_remote(
                    "http://spark-2:7878", "123456", fabric_ip="169.254.10.2"
                )
            finally:
                await client.aclose()

            self.assertNotIn("agent_token", public)
            saved = json.loads((Path(directory) / "nodes.json").read_text())
            self.assertEqual(saved[0]["agent_token"], "durable-secret")
            self.assertEqual(saved[0]["fabric_ip"], "169.254.10.2")

    async def test_probe_distinguishes_online_and_degraded_nodes(self) -> None:
        docker_ready = True

        async def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.headers["authorization"], "Bearer secret")
            return httpx.Response(200, json={
                "node_id": "remote-1",
                "protocol_version": 1,
                "docker_ready": docker_ready,
                "fabric_ready": True,
                "interfaces": [],
                "stats": {},
                "containers": [],
            })

        with tempfile.TemporaryDirectory() as directory:
            client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            try:
                registry = NodeRegistry(Path(directory), client)
                node = {
                    "id": "remote-1", "name": "Spark 2",
                    "agent_url": "http://spark-2:7878", "agent_token": "secret",
                    "enabled": True,
                }
                registry.nodes = [node]
                online = await registry.probe(node, force=True)
                self.assertEqual(online["status"], "online")

                docker_ready = False
                degraded = await registry.probe(node, force=True)
                self.assertEqual(degraded["status"], "degraded")
                self.assertTrue(degraded["online"])
                self.assertIn("Docker", degraded["status_message"])
            finally:
                await client.aclose()


class DistributedLaunchTests(unittest.IsolatedAsyncioTestCase):
    async def test_members_are_persisted_before_slow_node_launches_finish(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            instance = Manager.__new__(Manager)
            instance.settings = {
                "cluster_fabric_ip": "169.254.10.1",
                "cluster_fabric_interface": "cx7-local",
            }
            instance.deployments_path = Path(directory) / "deployments.json"
            instance.deployments = []
            entered = 0
            both_entered = asyncio.Event()
            release = asyncio.Event()

            async def cluster_nodes(local_stats=None):
                return [
                    {
                        "id": "local", "name": "Spark 1", "online": True,
                        "docker_ready": True, "fabric_ip": "169.254.10.1",
                        "fabric_interface": "cx7-local", "interfaces": [],
                    },
                    {
                        "id": "remote-1", "name": "Spark 2", "online": True,
                        "docker_ready": True, "fabric_ip": "169.254.10.2",
                        "fabric_interface": "cx7-remote", "interfaces": [],
                    },
                ]

            async def allocate_port():
                return 8008

            async def create_member(node_id, payload):
                nonlocal entered
                entered += 1
                if entered == 2:
                    both_entered.set()
                await release.wait()
                return {"id": f"container-{node_id}", "status": "running"}

            instance.cluster_nodes = cluster_nodes
            instance._allocate_port = allocate_port
            instance._create_member = create_member

            task = asyncio.create_task(instance.create_deployment({
                "model": "deepseek-ai/DeepSeek-V4-Flash",
                "engine": "vllm",
                "deployment_mode": "sharded",
                "node_ids": ["local", "remote-1"],
            }))
            await asyncio.wait_for(both_entered.wait(), 1)

            self.assertEqual(len(instance.deployments[0]["members"]), 2)
            self.assertTrue(all(
                member["status"] == "queued"
                for member in instance.deployments[0]["members"]
            ))
            persisted = json.loads(instance.deployments_path.read_text())
            self.assertEqual(len(persisted[0]["members"]), 2)
            self.assertEqual(persisted[0]["api_port"], 8008)

            release.set()
            await task

    async def test_cluster_logs_show_progress_before_container_exists(self) -> None:
        instance = Manager.__new__(Manager)
        instance.cluster_member_launches = {}
        instance._cluster_launch_update(
            "cluster-test-r0", "pulling_image",
            "Downloading Docker image example/vllm:latest; this can take several minutes",
            model="example/model",
            cluster_member={"deployment_id": "test", "node_id": "local", "rank": 0},
        )

        async def is_managed_container(name):
            return False

        instance.is_managed_container = is_managed_container
        logs = await instance.get_cluster_member_logs("cluster-test-r0")

        self.assertIn("Controller launch progress", logs)
        self.assertIn("Downloading Docker image", logs)
        self.assertIn("Container has not been created yet", logs)

    async def test_combined_logs_fall_back_to_coordinator_status(self) -> None:
        instance = Manager.__new__(Manager)
        instance.deployments = [{
            "id": "test",
            "members": [{
                "node_id": "remote-1", "node_name": "Spark 2", "rank": 1,
                "container_name": "cluster-test-r1", "status": "queued",
                "phase": {"phase": "queued", "message": "Waiting for image pull"},
            }],
        }]

        async def member_action(member, action):
            raise RuntimeError("older agent has not created the container")

        instance._member_action = member_action
        result = await instance.deployment_logs("test")

        self.assertIn("Coordinator launch status", result["members"][0]["logs"])
        self.assertIn("Waiting for image pull", result["members"][0]["logs"])
        self.assertIn("older agent", result["members"][0]["logs"])
        self.assertIsNone(result["members"][0]["error"])

    async def test_vllm_sharded_launch_generates_coordinated_rank_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            instance = Manager.__new__(Manager)
            instance.settings = {
                "cluster_fabric_ip": "169.254.10.1",
                "cluster_fabric_interface": "cx7-local",
            }
            instance.deployments_path = Path(directory) / "deployments.json"
            instance.deployments = []
            captured = []

            async def cluster_nodes(local_stats=None):
                return [
                    {
                        "id": "local", "name": "Spark 1", "online": True,
                        "docker_ready": True,
                        "fabric_ip": "169.254.10.1", "fabric_interface": "cx7-local",
                        "interfaces": [],
                    },
                    {
                        "id": "remote-1", "name": "Spark 2", "online": True,
                        "docker_ready": True,
                        "fabric_ip": "169.254.10.2", "fabric_interface": "cx7-remote",
                        "interfaces": [],
                    },
                ]

            async def allocate_port():
                return 8007

            async def create_member(node_id, payload):
                captured.append((node_id, payload))
                return {"id": f"container-{node_id}", "status": "running"}

            instance.cluster_nodes = cluster_nodes
            instance._allocate_port = allocate_port
            instance._create_member = create_member

            deployment = await instance.create_deployment({
                "model": "deepseek-ai/DeepSeek-V4-Flash",
                "engine": "vllm",
                "deployment_mode": "sharded",
                "node_ids": ["local", "remote-1"],
                "extra_args": [
                    "--max-model-len", "65536", "--tensor-parallel-size", "99",
                    "--pipeline-parallel-size", "99", "--headless",
                ],
            })

            self.assertEqual(deployment["status"], "starting")
            self.assertEqual(len(captured), 2)
            for rank, (node_id, payload) in enumerate(captured):
                args = payload["extra_args"]
                self.assertEqual(args[args.index("--node-rank") + 1], str(rank))
                self.assertEqual(args[args.index("--nnodes") + 1], "2")
                self.assertEqual(args[args.index("--tensor-parallel-size") + 1], "1")
                self.assertEqual(args[args.index("--pipeline-parallel-size") + 1], "2")
                self.assertEqual(args[args.index("--master-addr") + 1], "169.254.10.1")
                self.assertEqual("--headless" in args, rank > 0)
                self.assertEqual(args.count("--tensor-parallel-size"), 1)
                self.assertEqual(args.count("--pipeline-parallel-size"), 1)
                self.assertEqual(payload["cluster_member"]["fabric_interface"],
                                 "cx7-local" if node_id == "local" else "cx7-remote")

    async def test_idle_monitor_never_stops_one_cluster_member(self) -> None:
        instance = Manager.__new__(Manager)
        instance.settings = {"idle_timeout_seconds": 1}
        instance._activity = {
            "cluster-worker": {"counter": 0, "last_active": 0},
        }
        stopped = []

        async def list_containers():
            return [{
                "name": "cluster-worker",
                "managed": True,
                "status": "running",
                "deployment_id": "deployment-1",
                "phase": {"phase": "ready"},
                "port": 8000,
            }]

        async def read_activity_counter(port):
            self.fail("cluster members should not be polled independently")

        async def stop_container(name):
            stopped.append(name)

        instance.list_containers = list_containers
        instance._read_activity_counter = read_activity_counter
        instance.stop_container = stop_container

        await instance._idle_tick()

        self.assertEqual(stopped, [])

    def test_node_in_use_by_deployment_cannot_be_removed(self) -> None:
        instance = Manager.__new__(Manager)
        instance.deployments = [{
            "id": "deployment-1",
            "name": "DeepSeek cluster",
            "node_ids": ["local", "remote-1"],
            "members": [],
        }]

        with self.assertRaisesRegex(ValueError, "remove that deployment first"):
            instance.remove_cluster_node("remote-1")


if __name__ == "__main__":
    unittest.main()
