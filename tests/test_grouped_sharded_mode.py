"""Tests for the ``grouped_sharded`` deployment mode.

A grouped-sharded deployment runs N independent sharded engine groups, each on
its own tensor-parallel-sized node set, behind one served name.
"""

import unittest
from unittest import mock

from manager import Manager


def four_node_manager() -> Manager:
    instance = Manager.__new__(Manager)
    instance.settings = {
        "cluster_fabric_ip": "169.254.10.1",
        "cluster_fabric_interface": "cx7-local",
    }
    instance.deployments = []

    async def cluster_nodes(local_stats=None):
        return [
            {
                "id": node_id,
                "name": f"Spark {index}",
                "online": True,
                "docker_ready": True,
                "fabric_ip": f"169.254.10.{index}",
                "fabric_interface": f"cx7-node-{index}",
                "interfaces": [],
            }
            for index, node_id in enumerate(
                ["local", "remote-1", "remote-2", "remote-3"], start=1,
            )
        ]

    instance.cluster_nodes = cluster_nodes
    instance._allocate_port = mock.AsyncMock(return_value=8000)
    return instance


class GroupedShardedModeValidationTests(unittest.IsolatedAsyncioTestCase):
    async def test_mode_allowlist_accepts_grouped_sharded(self) -> None:
        from sparkdeck.service import _MODE_ALLOWLIST

        self.assertIn("grouped_sharded", _MODE_ALLOWLIST)

    async def test_preflight_accepts_exact_instance_times_tp_nodes(self) -> None:
        instance = four_node_manager()
        plan = await instance._preflight_deployment_launch({
            "model": "org/model",
            "engine": "vllm",
            "deployment_mode": "grouped_sharded",
            "tensor_parallel_size": 2,
            "instances": 2,
            "node_ids": ["local", "remote-1", "remote-2", "remote-3"],
        })
        self.assertEqual(plan["mode"], "grouped_sharded")
        self.assertEqual(len(plan["node_ids"]), 4)

    async def test_preflight_rejects_tensor_parallel_below_two(self) -> None:
        instance = four_node_manager()
        with self.assertRaisesRegex(ValueError, "tensor_parallel_size >= 2"):
            await instance._preflight_deployment_launch({
                "model": "org/model",
                "engine": "vllm",
                "deployment_mode": "grouped_sharded",
                "tensor_parallel_size": 1,
                "instances": 4,
                "node_ids": ["local", "remote-1", "remote-2", "remote-3"],
            })

    async def test_preflight_rejects_node_count_not_instances_times_tp(self) -> None:
        instance = four_node_manager()
        with self.assertRaisesRegex(ValueError, "2 instance\\(s\\) of TP2"):
            await instance._preflight_deployment_launch({
                "model": "org/model",
                "engine": "vllm",
                "deployment_mode": "grouped_sharded",
                "tensor_parallel_size": 2,
                "instances": 2,
                "node_ids": ["local", "remote-1", "remote-2"],
            })

    async def test_preflight_rejects_llama_cpp_grouped_sharded(self) -> None:
        instance = four_node_manager()
        with self.assertRaisesRegex(ValueError, "llama.cpp"):
            await instance._preflight_deployment_launch({
                "model": "org/model",
                "engine": "llama.cpp",
                "deployment_mode": "grouped_sharded",
                "tensor_parallel_size": 2,
                "instances": 2,
                "node_ids": ["local", "remote-1", "remote-2", "remote-3"],
            })

    async def test_preflight_defaults_instances_to_one(self) -> None:
        instance = four_node_manager()
        plan = await instance._preflight_deployment_launch({
            "model": "org/model",
            "engine": "vllm",
            "deployment_mode": "grouped_sharded",
            "tensor_parallel_size": 2,
            "node_ids": ["local", "remote-1"],
        })
        self.assertEqual(plan["mode"], "grouped_sharded")

    async def test_preflight_requires_fabric_for_every_group(self) -> None:
        instance = four_node_manager()

        async def cluster_nodes(local_stats=None):
            return [
                {
                    "id": "local", "name": "Spark 1", "online": True,
                    "docker_ready": True, "fabric_ip": "169.254.10.1",
                    "fabric_interface": "cx7-local", "interfaces": [],
                },
                {
                    "id": "remote-1", "name": "Spark 2", "online": True,
                    "docker_ready": True,
                    # No fabric identity: a coordinator cannot be elected here.
                    "fabric_ip": None, "fabric_interface": None,
                    "interfaces": [],
                },
            ]

        instance.cluster_nodes = cluster_nodes
        with self.assertRaisesRegex(ValueError, "fabric IP"):
            await instance._preflight_deployment_launch({
                "model": "org/model",
                "engine": "vllm",
                "deployment_mode": "grouped_sharded",
                "tensor_parallel_size": 2,
                "node_ids": ["local", "remote-1"],
            })


if __name__ == "__main__":
    unittest.main()
