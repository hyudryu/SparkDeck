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


class GroupedShardedMemberBuildTests(unittest.IsolatedAsyncioTestCase):
    async def test_vllm_members_partition_into_independent_groups(self) -> None:
        instance = four_node_manager()
        instance.deployments_path = None
        captured = []

        async def create_member(node_id, payload):
            captured.append((node_id, payload))
            return {
                "id": f"container-{node_id}", "status": "running",
                "model_source": "public_repository",
            }

        instance._create_member = create_member
        instance._save_deployments = lambda: None

        deployment = await instance.create_deployment({
            "model": "org/model",
            "engine": "vllm",
            "deployment_mode": "grouped_sharded",
            "tensor_parallel_size": 2,
            "instances": 2,
            "node_ids": ["local", "remote-1", "remote-2", "remote-3"],
            "extra_args": ["--max-model-len", "4096", "--headless"],
        })

        self.assertEqual(deployment["status"], "starting")
        self.assertEqual(len(captured), 4)
        members = deployment["members"]
        self.assertEqual([m["instance_id"] for m in members], [0, 0, 1, 1])
        self.assertEqual([m["rank"] for m in members], [0, 1, 0, 1])
        # Consecutive partition: the first T nodes form group 0, the next T
        # form group 1.
        self.assertEqual(
            [m["node_id"] for m in members],
            ["local", "remote-1", "remote-2", "remote-3"],
        )
        # Container names stay unique across groups via the global rank.
        names = [m["container_name"] for m in members]
        self.assertEqual(len(set(names)), 4)

        # Per-group vLLM rendezvous: each group gets its own coordinator
        # fabric IP and port, and its own TP-sized --nnodes.
        expected_masters = ["169.254.10.1", "169.254.10.1", "169.254.10.3", "169.254.10.3"]
        for index, (_, payload) in enumerate(captured):
            args = payload["extra_args"]
            group = index // 2
            local_rank = index % 2
            self.assertEqual(args[args.index("--nnodes") + 1], "2")
            self.assertEqual(args[args.index("--node-rank") + 1], str(local_rank))
            self.assertEqual(args[args.index("--master-addr") + 1], expected_masters[index])
            self.assertEqual(
                args[args.index("--master-port") + 1], str(29501 + group),
            )
            self.assertEqual(args[args.index("--tensor-parallel-size") + 1], "2")
            self.assertEqual(args[args.index("--pipeline-parallel-size") + 1], "1")
            self.assertEqual("--headless" in args, local_rank > 0)
            self.assertEqual(payload["cluster_member"]["instance_id"], group)
            self.assertEqual(payload["cluster_member"]["nnodes"], 2)

    async def test_sglang_members_use_per_group_dist_init(self) -> None:
        instance = four_node_manager()
        instance.deployments_path = None
        captured = []

        async def create_member(node_id, payload):
            captured.append((node_id, payload))
            return {
                "id": f"container-{node_id}", "status": "running",
                "model_source": "public_repository",
            }

        instance._create_member = create_member
        instance._save_deployments = lambda: None

        deployment = await instance.create_deployment({
            "model": "org/model",
            "engine": "sglang",
            "deployment_mode": "grouped_sharded",
            "tensor_parallel_size": 2,
            "instances": 2,
            "node_ids": ["local", "remote-1", "remote-2", "remote-3"],
        })

        self.assertEqual(len(captured), 4)
        for index, (_, payload) in enumerate(captured):
            args = payload["extra_args"]
            group = index // 2
            self.assertEqual(args[args.index("--dist-init-addr") + 1],
                             f"169.254.10.{1 + group * 2}:{29501 + group}")
            self.assertEqual(payload["sg_tp_size"], 2)
        self.assertEqual(
            [m["instance_id"] for m in deployment["members"]], [0, 0, 1, 1],
        )


def grouped_member(group: int, rank: int, status: str = "running") -> dict:
    return {
        "instance_id": group,
        "rank": rank,
        "node_id": f"node-{group}-{rank}",
        "node_name": f"Node {group}-{rank}",
        "container_name": f"cluster-d1-r{group * 2 + rank}-model",
        "status": status,
    }


def grouped_deployment() -> dict:
    return {
        "id": "d1",
        "mode": "grouped_sharded",
        "members": [
            grouped_member(0, 0), grouped_member(0, 1),
            grouped_member(1, 0), grouped_member(1, 1),
        ],
    }


class GroupedShardedRoutingTests(unittest.TestCase):
    def test_route_order_only_considers_group_coordinators(self) -> None:
        manager = Manager.__new__(Manager)
        order = manager._cluster_route_order(grouped_deployment())
        self.assertEqual(
            [member["container_name"] for member in order],
            ["cluster-d1-r0-model", "cluster-d1-r2-model"],
        )

    def test_route_order_round_robins_across_coordinators(self) -> None:
        manager = Manager.__new__(Manager)
        deployment = grouped_deployment()
        first = manager._cluster_route_order(deployment)[0]
        second = manager._cluster_route_order(deployment)[0]
        self.assertNotEqual(
            first["container_name"], second["container_name"],
            "an idle grouped cluster must alternate between group coordinators",
        )

    def test_route_order_excludes_stopped_groups(self) -> None:
        manager = Manager.__new__(Manager)
        deployment = grouped_deployment()
        deployment["members"][2]["status"] = "stopped"
        deployment["members"][3]["status"] = "stopped"
        order = manager._cluster_route_order(deployment)
        self.assertEqual(
            [member["container_name"] for member in order],
            ["cluster-d1-r0-model"],
        )

    def test_route_order_error_group_is_excluded(self) -> None:
        manager = Manager.__new__(Manager)
        deployment = grouped_deployment()
        deployment["members"][0]["status"] = "error"
        deployment["members"][1]["status"] = "error"
        order = manager._cluster_route_order(deployment)
        self.assertEqual(
            [member["container_name"] for member in order],
            ["cluster-d1-r2-model"],
        )

    def test_sharded_members_do_not_collide_with_instance_keys(self) -> None:
        # instance_id 0 must key per instance, not fall through to the
        # container-name identity (``0 or default`` is falsy in Python).
        key = Manager._cluster_member_key("d1", grouped_member(0, 0))
        self.assertEqual(key, "d1:instance:0")


class GroupedShardedLifecycleTests(unittest.IsolatedAsyncioTestCase):
    def lifecycle_manager(self) -> tuple[Manager, dict, list]:
        deployment = {
            **grouped_deployment(),
            "desired_state": "running",
            "status": "running",
            "instances": 2,
        }
        for member in deployment["members"]:
            member["status"] = "running"
        instance = Manager.__new__(Manager)
        instance.deployments = [deployment]
        instance._save_deployments = lambda: None
        actions = []

        async def member_action(member, action, *, log_tail=300):
            actions.append((member["instance_id"], action))
            member["status"] = "starting" if action == "start" else "stopped"
            return {"ok": True}

        instance._member_action = member_action
        return instance, deployment, actions

    async def test_per_instance_stop_derives_degraded_state(self) -> None:
        instance, deployment, actions = self.lifecycle_manager()
        result = await instance.deployment_action("d1", "stop", instance=1)
        self.assertTrue(result["ok"])
        # Only the targeted group's ranks are stopped.
        self.assertEqual(actions, [(1, "stop"), (1, "stop")])
        self.assertEqual(deployment["members"][2]["desired_state"], "stopped")
        self.assertEqual(deployment["members"][3]["desired_state"], "stopped")
        # Untargeted groups stay expected-running for the health monitor.
        self.assertEqual(deployment["members"][0]["desired_state"], "running")
        self.assertEqual(deployment["desired_state"], "running")
        self.assertEqual(deployment["status"], "degraded")

    async def test_per_instance_start_of_stopped_group_is_degraded(self) -> None:
        instance, deployment, actions = self.lifecycle_manager()
        deployment["desired_state"] = "stopped"
        deployment["status"] = "stopped"
        for member in deployment["members"]:
            member["status"] = "stopped"
        result = await instance.deployment_action("d1", "start", instance=0)
        self.assertTrue(result["ok"])
        self.assertEqual(actions, [(0, "start"), (0, "start")])
        self.assertEqual(deployment["members"][0]["desired_state"], "running")
        self.assertEqual(deployment["members"][2]["desired_state"], "stopped")
        self.assertEqual(deployment["desired_state"], "running")
        self.assertEqual(deployment["status"], "degraded")

    async def test_full_stop_unchanged_by_grouped_mode(self) -> None:
        instance, deployment, actions = self.lifecycle_manager()
        await instance.deployment_action("d1", "stop")
        self.assertEqual(actions, [(0, "stop"), (0, "stop"), (1, "stop"), (1, "stop")])
        self.assertEqual(deployment["desired_state"], "stopped")
        self.assertEqual(deployment["status"], "stopped")

    async def test_per_instance_action_rejects_unknown_instance(self) -> None:
        instance, _, _ = self.lifecycle_manager()
        with self.assertRaisesRegex(ValueError, "instance"):
            await instance.deployment_action("d1", "stop", instance=5)

    async def test_health_ignores_targeted_stopped_group(self) -> None:
        instance, deployment, _ = self.lifecycle_manager()
        await instance.deployment_action("d1", "stop", instance=1)

        def nodes_with_group1_gone():
            return [
                {
                    "id": f"node-{group}-{rank}", "online": True,
                    "docker_ready": True,
                    "containers": (
                        [{"name": member["container_name"], "status": "running"}]
                        if member.get("desired_state") != "stopped"
                        else []
                    ),
                }
                for member in deployment["members"]
                for group, rank in [(member["instance_id"], member["rank"])]
            ]

        self.assertIsNone(instance._cluster_health_issue(
            deployment, nodes_with_group1_gone(),
        ))
        # The same missing containers on an expected-running group are a
        # recoverable split.
        nodes = nodes_with_group1_gone()
        for member in deployment["members"]:
            member.pop("desired_state", None)
        issue = instance._cluster_health_issue(deployment, nodes)
        self.assertIsNotNone(issue)
        self.assertIn("instance 1", issue)


class GroupedShardedRecipeContractTests(unittest.TestCase):
    def test_contract_reads_persisted_grouped_topology(self) -> None:
        # A grouped recipe persists the per-group TP and instance count as
        # standalone fields, not as a whole-world argv flag.
        instance = Manager.__new__(Manager)
        contract = instance.recipe_deployment_contract({
            "engine": "vllm",
            "deployment_mode": "grouped_sharded",
            "instances": 2,
            "tensor_parallel_size": 2,
            "node_ids": ["local", "remote-1", "remote-2", "remote-3"],
        })
        self.assertTrue(contract["supported"])
        self.assertEqual(contract["deployment_mode"], "grouped_sharded")
        self.assertEqual(contract["required_node_count"], 4)
        self.assertEqual(contract["instances"], 2)
        self.assertEqual(contract["tensor_parallel_size"], 2)

    def test_contract_rejects_grouped_topology_below_two_ranks(self) -> None:
        instance = Manager.__new__(Manager)
        contract = instance.recipe_deployment_contract({
            "engine": "vllm",
            "deployment_mode": "grouped_sharded",
            "instances": 4,
            "tensor_parallel_size": 1,
            "node_ids": ["local"],
        })
        self.assertFalse(contract["supported"])
        self.assertIn("tensor_parallel_size >= 2", contract["error"])


class GroupedShardedInstanceValidationTests(unittest.TestCase):
    def test_target_instance_rejects_bools_and_fractions(self) -> None:
        deployment = {**grouped_deployment(), "instances": 2}
        with self.assertRaises(ValueError):
            Manager._grouped_target_instance(deployment, "stop", True)
        with self.assertRaises(ValueError):
            Manager._grouped_target_instance(deployment, "stop", 1.5)

    def test_topology_helper_rejects_fractional_fields(self) -> None:
        from manager import _grouped_sharded_topology

        with self.assertRaises(ValueError):
            _grouped_sharded_topology(
                {"tensor_parallel_size": 2.5, "instances": 2}, 5,
            )


if __name__ == "__main__":
    unittest.main()
