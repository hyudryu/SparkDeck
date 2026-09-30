"""SGLang managed launches preserve CPU affinity and operator environment."""
import asyncio
import unittest
from unittest.mock import AsyncMock, Mock

from manager import Manager
from sparkdeck.runtime_environment import normalize_runtime_environment
from sparkdeck.runtimes import RuntimeRegistry, launch_managed_container


class SglangAffinityTests(unittest.IsolatedAsyncioTestCase):
    def manager(self):
        instance = Manager.__new__(Manager)
        instance.lock = asyncio.Lock()
        instance.recipe_launches = {}
        instance.settings = {"hf_cache": "/cache", "shm_size": "1g"}
        instance.client = Mock()
        instance.evict_other_backends = AsyncMock()
        instance._run_managed_container = Mock(return_value=Mock())
        instance._container_summary = Mock(return_value={"name": "sglang-test"})
        instance._created_container_model_source = Mock(return_value="public_repository")
        instance._cluster_launch_update = Mock()
        instance._build_volumes = Mock(return_value={})
        instance._container_hf_environment = Mock(return_value={"HF_TOKEN": "managed"})
        instance._distributed_network_environment = Mock(return_value={
            "NCCL_SOCKET_IFNAME": "default0", "NCCL_IB_HCA": "default-hca",
        })
        return instance

    def test_affinity_rejects_invalid_or_unbounded_ranges(self):
        for value in (True, [1], " 1", "1;echo", "3-1", "1,1", "1-4,3-5", "0-9999999999", "1,,2"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Manager._normalized_sg_cpu_affinity(value)
        self.assertEqual(Manager._normalized_sg_cpu_affinity("5-9,15-19"), "5-9,15-19")
        self.assertIsNone(Manager._normalized_sg_cpu_affinity(None))

    async def test_invalid_affinity_fails_before_eviction_or_docker(self):
        instance = self.manager()
        for kwargs in ({"sg_cpu_affinity": "9-5"}, {"sg_cpu_affinity": "5", "engine": "vllm"}):
            with self.assertRaises(ValueError):
                await instance.create_container("org/model", port=30000, engine=kwargs.pop("engine", "sglang"), **kwargs)
        instance.evict_other_backends.assert_not_awaited()
        instance._run_managed_container.assert_not_called()

    async def test_distributed_launch_preserves_affinity_and_environment_precedence(self):
        instance = self.manager()
        environment = {
            "NCCL_SOCKET_IFNAME": "enp1s0f1np1", "NCCL_IB_HCA": "rocep1s0f1,roceP2p1s0f1",
            "NCCL_IB_GID_INDEX": "3", "HF_HUB_OFFLINE": "1", "SGLANG_ENABLE_JIT_DEEPGEMM": "0",
        }
        await instance.create_container(
            "org/model", port=30000, engine="sglang", sg_cpu_affinity="5-9,15-19",
            environment=environment, infiniband_device=False,
            extra_args=["--max-total-tokens", "1000000"],
            sg_context_length=262144, sg_max_running_requests=10,
            cluster_member={"deployment_id": "dep", "node_id": "worker", "rank": 1,
                            "nnodes": 2, "mode": "sharded", "fabric_interface": "fabric0"},
        )
        options = instance._run_managed_container.call_args.args[0]
        self.assertEqual(options["cpuset_cpus"], "5-9,15-19")
        self.assertEqual(options["network_mode"], "host")
        self.assertEqual(instance._cli_option(options["command"], {"--max-total-tokens"}), "1000000")
        self.assertEqual(options["environment"], {"HF_TOKEN": "managed", **environment})

    async def test_unset_affinity_retains_existing_default(self):
        instance = self.manager()
        await instance.create_container("org/model", port=30000, engine="sglang")
        self.assertNotIn("cpuset_cpus", instance._run_managed_container.call_args.args[0])

    def test_durable_launch_settings_keep_affinity(self):
        settings = Manager._deployment_launch_settings({
            "model": "org/model", "engine": "sglang", "sg_cpu_affinity": "5-9,15-19",
            "environment": {"NCCL_IB_GID_INDEX": "3"},
        })
        self.assertEqual(settings["sg_cpu_affinity"], "5-9,15-19")
        self.assertEqual(settings["environment"], {"NCCL_IB_GID_INDEX": "3"})

    def test_invalid_settings_edit_does_not_change_stopped_deployment(self):
        instance = self.manager()
        deployment = {"id": "dep", "status": "stopped", "engine": "sglang",
                      "launch_settings": {"model": "org/model", "engine": "sglang",
                                          "sg_cpu_affinity": "5-9", "extra_args": []}}
        instance.deployments = [deployment]
        instance._save_deployments = Mock()
        for changes in ({"sg_cpu_affinity": "9-5"}, {"engine": "vllm", "sg_cpu_affinity": "5-9"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                instance.update_deployment_settings("dep", changes)
        self.assertEqual(deployment["launch_settings"]["sg_cpu_affinity"], "5-9")
        self.assertEqual(deployment["launch_settings"]["engine"], "sglang")
        instance._save_deployments.assert_not_called()

    def test_sglang_environment_still_rejects_secrets(self):
        with self.assertRaisesRegex(ValueError, "managed by SparkDeck"):
            normalize_runtime_environment({"HF_TOKEN": "secret"}, "sglang")

    async def test_adapter_forwards_affinity_and_environment(self):
        instance = Mock()
        instance.create_container = AsyncMock(return_value={"name": "created"})
        await launch_managed_container(
            instance, RuntimeRegistry().get("sglang"), "org/model", "alias", "dep",
            {"sg_cpu_affinity": "5-9,15-19", "environment": {"HF_HUB_OFFLINE": "1"}},
        )
        self.assertEqual(instance.create_container.call_args.kwargs["sg_cpu_affinity"], "5-9,15-19")
        self.assertEqual(instance.create_container.call_args.kwargs["environment"], {"HF_HUB_OFFLINE": "1"})
