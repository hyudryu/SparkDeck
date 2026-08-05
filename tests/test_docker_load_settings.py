import asyncio
import copy
import unittest
from types import SimpleNamespace
from unittest import mock

from manager import Manager


class FakeContainer:
    def __init__(self, collection, cid, name, config, host_config, status="running"):
        self.collection = collection
        self.id = cid
        self.short_id = cid[:12]
        self.name = name
        self.status = status
        self.labels = config.get("Labels") or {}
        self.ports = {}
        self.attrs = {
            "Config": copy.deepcopy(config),
            "HostConfig": copy.deepcopy(host_config),
            "Created": "2026-08-05T00:00:00Z",
        }
        self.removed = False

    def reload(self):
        return None

    def stop(self, timeout=30):
        self.status = "exited"

    def start(self):
        self.status = "running"

    def rename(self, name):
        self.collection.items.pop(self.name, None)
        self.name = name
        self.collection.items[name] = self

    def remove(self, force=False):
        self.removed = True
        self.collection.items.pop(self.name, None)
        self.collection.items.pop(self.id, None)


class FakeContainers:
    def __init__(self):
        self.items = {}

    def add(self, container):
        self.items[container.name] = container
        self.items[container.id] = container

    def get(self, identifier):
        if identifier not in self.items:
            raise RuntimeError(f"container {identifier} not found")
        return self.items[identifier]


class FakeAPI:
    def __init__(self, containers, fail=False):
        self.containers = containers
        self.fail = fail
        self.created_config = None

    def create_container_from_config(self, config, name=None):
        if self.fail:
            raise RuntimeError("create failed")
        self.created_config = copy.deepcopy(config)
        cid = "replacement-container-id"
        container = FakeContainer(
            self.containers,
            cid,
            name,
            config,
            config.get("HostConfig") or {},
            status="created",
        )
        self.containers.add(container)
        return {"Id": cid}


class DockerLoadSettingsTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.manager = Manager.__new__(Manager)

    def test_managed_vllm_command_round_trips_common_and_custom_flags(self):
        model = "example/Model"
        command = [
            "vllm", "serve", model,
            "--host", "0.0.0.0", "--port", "8000",
            "--gpu-memory-utilization", "0.85",
            "--max-num-seqs", "12",
            "--max-model-len=1048576",
            "--kv-cache-dtype", "fp8",
            "--speculative-config", '{"method":"mtp","num_speculative_tokens":3}',
            "--enable-prefix-caching",
        ]

        settings = self.manager._container_load_settings(command, "vllm", model)

        self.assertEqual(settings["max_concurrency"], 12)
        self.assertEqual(settings["context_window"], 1048576)
        self.assertEqual(settings["kv_cache_dtype"], "fp8")
        self.assertEqual(settings["gpu_memory_utilization"], 0.85)
        self.assertIn("--speculative-config", settings["command_flags"])
        updated = self.manager._updated_container_command(
            command,
            "vllm",
            model,
            {**settings, "max_concurrency": 4, "kv_cache_dtype": "fp8_e4m3"},
        )
        self.assertEqual(self.manager._cli_option(updated, {"--max-num-seqs"}), "4")
        self.assertEqual(
            self.manager._cli_option(updated, {"--kv-cache-dtype"}), "fp8_e4m3"
        )
        self.assertIn('{"method":"mtp","num_speculative_tokens":3}', updated)

    def test_shell_wrapped_vllm_preserves_preamble_and_expressions(self):
        script = (
            'export CUDA_HOME=/opt/cuda; SPEC="{\\"method\\":\\"dspark\\"}"; '
            "exec /opt/env/bin/vllm serve deepseek-ai/DeepSeek-V4-Flash "
            "--served-model-name deepseek-v4-flash --host 0.0.0.0 --port 8889 "
            "--max-num-seqs 12 --max-cudagraph-capture-size $(( 12 * (5 + 1) )) "
            '--kv-cache-dtype nvfp4_ds_mla --speculative-config "${SPEC}"\n'
        )
        command = ["bash", "-lc", script]

        settings = self.manager._container_load_settings(
            command, "vllm", "deepseek-v4-flash"
        )
        updated = self.manager._updated_container_command(
            command,
            "vllm",
            "deepseek-v4-flash",
            {**settings, "max_concurrency": 6, "context_window": 524288},
        )

        self.assertEqual(settings["max_concurrency"], 12)
        self.assertEqual(settings["kv_cache_dtype"], "nvfp4_ds_mla")
        self.assertIn("export CUDA_HOME=/opt/cuda", updated[-1])
        self.assertIn('$(( 12 * (5 + 1) ))', updated[-1])
        self.assertIn('"${SPEC}"', updated[-1])
        self.assertIn("--max-num-seqs 6", updated[-1])
        self.assertIn("--max-model-len 524288", updated[-1])
        self.assertIn("--port 8889", updated[-1])

    def test_sglang_settings_use_engine_specific_flags(self):
        model = "example/SGLang-Model"
        command = [
            "-m", "sglang.launch_server", "--model-path", model,
            "--host", "0.0.0.0", "--port", "8000",
            "--context-length", "131072",
            "--max-running-requests", "8",
            "--mem-fraction-static", "0.9",
            "--kv-cache-dtype", "fp8_e4m3",
            "--enable-cache-report",
        ]

        settings = self.manager._container_load_settings(command, "sglang", model)
        updated = self.manager._updated_container_command(
            command,
            "sglang",
            model,
            {**settings, "max_concurrency": 3, "context_window": 65536},
        )

        self.assertEqual(settings["max_concurrency"], 8)
        self.assertEqual(settings["context_window"], 131072)
        self.assertEqual(settings["gpu_memory_utilization"], 0.9)
        self.assertIn("--enable-cache-report", settings["command_flags"])
        self.assertEqual(
            self.manager._cli_option(updated, {"--max-running-requests"}), "3"
        )
        self.assertEqual(
            self.manager._cli_option(updated, {"--context-length"}), "65536"
        )
        self.assertIn("--enable-cache-report", updated)

    async def test_update_clones_full_config_and_restores_running_state(self):
        name = "external-vllm"
        model = "example/Model"
        command = [
            "vllm", "serve", model, "--host", "0.0.0.0", "--port", "8000",
            "--max-num-seqs", "8",
        ]
        config = {
            "Image": "example/vllm:latest",
            "Cmd": command,
            "Env": ["SPECIAL_RUNTIME_FLAG=1"],
            "Labels": {"vllm-model": model},
        }
        host_config = {
            "NetworkMode": "host",
            "Binds": ["/models:/models:ro"],
            "ShmSize": 68719476736,
        }
        containers = FakeContainers()
        original = FakeContainer(
            containers, "original-container-id", name, config, host_config
        )
        containers.add(original)
        api = FakeAPI(containers)
        self.manager.client = SimpleNamespace(containers=containers, api=api)
        self.manager.lock = asyncio.Lock()
        self.manager._container_summary = lambda container: {
            "name": container.name, "status": container.status
        }

        inline_thread = mock.AsyncMock(
            side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)
        )
        with mock.patch("manager.asyncio.to_thread", inline_thread):
            result = await self.manager.update_container_settings(
                name,
                {
                    **self.manager._container_load_settings(command, "vllm", model),
                    "max_concurrency": 3,
                },
            )

        replacement = containers.get(name)
        self.assertTrue(result["ok"])
        self.assertEqual(replacement.status, "running")
        self.assertTrue(original.removed)
        self.assertEqual(api.created_config["Env"], ["SPECIAL_RUNTIME_FLAG=1"])
        self.assertEqual(api.created_config["HostConfig"], host_config)
        self.assertEqual(
            self.manager._cli_option(
                api.created_config["Cmd"], {"--max-num-seqs"}
            ),
            "3",
        )

    async def test_failed_replacement_restores_original_container(self):
        name = "external-vllm"
        model = "example/Model"
        command = ["vllm", "serve", model, "--max-num-seqs", "8"]
        config = {
            "Image": "example/vllm:latest",
            "Cmd": command,
            "Labels": {"vllm-model": model},
        }
        containers = FakeContainers()
        original = FakeContainer(
            containers, "original-container-id", name, config, {}, status="running"
        )
        containers.add(original)
        self.manager.client = SimpleNamespace(
            containers=containers, api=FakeAPI(containers, fail=True)
        )
        self.manager.lock = asyncio.Lock()

        inline_thread = mock.AsyncMock(
            side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)
        )
        with mock.patch("manager.asyncio.to_thread", inline_thread):
            with self.assertRaisesRegex(RuntimeError, "create failed"):
                await self.manager.update_container_settings(
                    name,
                    {
                        **self.manager._container_load_settings(command, "vllm", model),
                        "max_concurrency": 2,
                    },
                )

        self.assertIs(containers.get(name), original)
        self.assertEqual(original.status, "running")
        self.assertFalse(original.removed)


if __name__ == "__main__":
    unittest.main()
