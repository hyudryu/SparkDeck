import asyncio
import copy
import json
import tempfile
import unittest
from pathlib import Path
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
        self.attrs["State"] = {"Health": {"Status": "healthy"}}

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
        self.disconnected = []
        self.connected = []

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

    def disconnect_container_from_network(self, container, network, force=False):
        self.disconnected.append((container, network, force))

    def connect_container_to_network(self, container, network, **kwargs):
        self.connected.append((container, network, kwargs))


class DockerLoadSettingsTests(unittest.IsolatedAsyncioTestCase):
    def test_external_summary_combines_entrypoint_and_cmd_and_exposes_safe_environment(self):
        containers = FakeContainers()
        container = FakeContainer(
            containers,
            "external-container-id",
            "external-vllm",
            {
                "Image": "example/vllm:latest",
                "Entrypoint": ["vllm", "serve"],
                "Cmd": [
                    "/cache/models/org--model/snapshots/rev",
                    "--max-model-len", "65536",
                    "--enable-prefix-caching",
                ],
                "Env": [
                    "VLLM_CACHE_ROOT=/cache/vllm",
                    "NCCL_DEBUG=INFO",
                    'SPECULATIVE_CONFIG={"method":"dspark"}',
                    "HF_TOKEN=must-not-leak",
                    "SERVICE_API_KEY=must-not-leak",
                    "DATABASE_URL=postgres://user:password@host/db",
                    "SENTRY_DSN=https://secret@example.invalid/1",
                    "FEATURE_FLAG=must-not-be-discovered",
                ],
                "Labels": {"io.sparkdeck.model": "served-name"},
            },
            {},
        )

        summary = self.manager._container_summary(container)

        self.assertIsNotNone(summary)
        settings = summary["load_settings"]
        self.assertEqual(summary["model"], "served-name")
        self.assertEqual(
            settings["model"], "/cache/models/org--model/snapshots/rev",
        )
        self.assertEqual(settings["context_window"], 65536)
        self.assertIn("--enable-prefix-caching", settings["extra_args"])
        self.assertEqual(settings["environment"], {
            "VLLM_CACHE_ROOT": "/cache/vllm",
            "NCCL_DEBUG": "INFO",
            "SPECULATIVE_CONFIG": '{"method":"dspark"}',
        })

    def test_shell_wrapped_summary_retains_launch_model_beside_served_label(self):
        containers = FakeContainers()
        container = FakeContainer(
            containers, "wrapped-container-id", "wrapped-vllm",
            {
                "Image": "example/vllm:latest",
                "Cmd": [
                    "bash", "-lc",
                    "exec vllm serve /models/actual-model "
                    "--served-model-name public-name --max-model-len 8192",
                ],
                "Labels": {"io.sparkdeck.model": "public-name"},
            },
            {},
        )

        summary = self.manager._container_summary(container)

        self.assertEqual(summary["model"], "public-name")
        self.assertEqual(summary["load_settings"]["model"], "/models/actual-model")

    def test_credential_bearing_discovered_command_is_read_only(self):
        containers = FakeContainers()
        container = FakeContainer(
            containers, "protected-container-id", "protected-vllm",
            {
                "Image": "example/vllm:latest",
                "Entrypoint": ["vllm", "serve"],
                "Cmd": ["org/model", "--api-key", "do-not-expose"],
                "Labels": {},
            },
            {},
        )

        summary = self.manager._container_summary(container)

        self.assertIsNotNone(summary)
        self.assertFalse(summary["load_settings"]["editable"])

    def _hooked_container(self, labels):
        containers = FakeContainers()
        return FakeContainer(
            containers, "hooked-container-id", "hooked-vllm",
            {
                "Image": "example/vllm:latest",
                "Entrypoint": ["vllm", "serve"],
                "Cmd": ["org/model"],
                "Labels": labels,
            },
            {},
        )

    def _stub_image_labels(self, labels):
        image = mock.Mock()
        image.attrs = {"Config": {"Labels": labels}}
        self.manager.client = mock.Mock()
        self.manager.client.images.get.return_value = image

    def test_summary_exposes_external_lifecycle_hook_labels(self):
        self._stub_image_labels({})
        container = self._hooked_container({
            "io.sparkdeck.start-command": "  /opt/stack/start.sh  ",
            "io.sparkdeck.stop-command": "/opt/stack/stop.sh --all",
        })

        summary = self.manager._container_summary(container)

        self.assertEqual(summary["start_command"], "/opt/stack/start.sh")
        self.assertEqual(summary["stop_command"], "/opt/stack/stop.sh --all")

    def test_summary_honors_container_override_of_image_hook_label(self):
        self._stub_image_labels({
            "io.sparkdeck.start-command": "/image/baked-start.sh",
        })
        container = self._hooked_container({
            "io.sparkdeck.start-command": "/opt/stack/start.sh",
        })

        summary = self.manager._container_summary(container)

        self.assertEqual(summary["start_command"], "/opt/stack/start.sh")

    def test_summary_ignores_hook_labels_inherited_from_the_image(self):
        self._stub_image_labels({
            "io.sparkdeck.start-command": "/image/baked-start.sh",
            "io.sparkdeck.stop-command": "/image/baked-stop.sh",
        })
        container = self._hooked_container({
            # Identical key+value pairs are inherited from the image, not
            # supplied by the operator at container-creation time.
            "io.sparkdeck.start-command": "/image/baked-start.sh",
            "io.sparkdeck.stop-command": "/image/baked-stop.sh",
        })

        summary = self.manager._container_summary(container)

        self.assertNotIn("start_command", summary)
        self.assertNotIn("stop_command", summary)

    def test_summary_ignores_hooks_when_the_image_cannot_be_inspected(self):
        self.manager.client = mock.Mock()
        self.manager.client.images.get.side_effect = RuntimeError("gone")
        container = self._hooked_container({
            "io.sparkdeck.start-command": "/opt/stack/start.sh",
        })

        summary = self.manager._container_summary(container)

        self.assertNotIn("start_command", summary)

    def test_image_label_lookup_is_cached_per_image_reference(self):
        self._stub_image_labels({})
        first = self._hooked_container({
            "io.sparkdeck.start-command": "/opt/stack/start.sh",
        })
        second = self._hooked_container({
            "io.sparkdeck.stop-command": "/opt/stack/stop.sh",
        })

        self.manager._container_summary(first)
        self.manager._container_summary(second)

        self.manager.client.images.get.assert_called_once_with(
            "example/vllm:latest",
        )

    def test_summary_omits_absent_or_blank_lifecycle_hook_labels(self):
        self._stub_image_labels({})
        container = self._hooked_container({
            "io.sparkdeck.start-command": "   ",
        })

        summary = self.manager._container_summary(container)

        self.assertNotIn("start_command", summary)
        self.assertNotIn("stop_command", summary)

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
            "--default-chat-template-kwargs", '{"enable_thinking":true,"tools":true}',
            "--enable-prefix-caching",
        ]

        settings = self.manager._container_load_settings(command, "vllm", model)

        self.assertEqual(settings["max_concurrency"], 12)
        self.assertEqual(settings["context_window"], 1048576)
        self.assertEqual(settings["kv_cache_dtype"], "fp8")
        self.assertEqual(settings["gpu_memory_utilization"], 0.85)
        self.assertEqual(settings["thinking_mode"], "enabled")
        self.assertIn("--speculative-config", settings["command_flags"])
        updated = self.manager._updated_container_command(
            command,
            "vllm",
            model,
            {
                **settings,
                "max_concurrency": 4,
                "kv_cache_dtype": "fp8_e4m3",
                "thinking_mode": "disabled",
            },
        )
        self.assertEqual(self.manager._cli_option(updated, {"--max-num-seqs"}), "4")
        self.assertEqual(
            self.manager._cli_option(updated, {"--kv-cache-dtype"}), "fp8_e4m3"
        )
        self.assertIn('{"method":"mtp","num_speculative_tokens":3}', updated)
        template_kwargs = json.loads(
            self.manager._cli_option(updated, {"--default-chat-template-kwargs"})
        )
        self.assertFalse(template_kwargs["enable_thinking"])
        self.assertTrue(template_kwargs["tools"])

    async def test_container_alias_is_persisted_without_renaming_docker(self):
        with tempfile.TemporaryDirectory() as directory:
            name = "external-vllm"
            containers = FakeContainers()
            container = FakeContainer(
                containers,
                "container-id",
                name,
                {
                    "Image": "example/vllm:latest",
                    "Cmd": ["vllm", "serve", "example/Model"],
                    "Labels": {},
                },
                {},
            )
            containers.add(container)
            self.manager.client = SimpleNamespace(containers=containers)
            self.manager.container_aliases_path = Path(directory) / "container_aliases.json"
            self.manager.container_aliases = {}

            result = await self.manager.update_container_alias(name, "Fast benchmark")

            self.assertEqual(result["alias"], "Fast benchmark")
            self.assertEqual(container.name, name)
            self.assertEqual(
                json.loads(self.manager.container_aliases_path.read_text()),
                {name: "Fast benchmark"},
            )
            summary = self.manager._container_summary(container)
            self.assertEqual(summary["alias"], "Fast benchmark")

            cleared = await self.manager.update_container_alias(name, "")
            self.assertIsNone(cleared["alias"])
            self.assertEqual(
                json.loads(self.manager.container_aliases_path.read_text()), {}
            )

    def test_shell_wrapped_vllm_preserves_preamble_and_expressions(self):
        script = (
            'export CUDA_HOME=/opt/cuda; SPEC="{\\"method\\":\\"dspark\\"}"; '
            "exec /opt/env/bin/vllm serve deepseek-ai/DeepSeek-V4-Flash "
            "--served-model-name deepseek-v4-flash --host 0.0.0.0 --port 8889 "
            "--max-num-seqs 12 --max-cudagraph-capture-size $(( 12 * (5 + 1) )) "
            "--kv-cache-dtype nvfp4_ds_mla "
            "--default-chat-template-kwargs '{\"thinking\":false}' "
            '--speculative-config "${SPEC}"\n'
        )
        command = ["bash", "-lc", script]

        settings = self.manager._container_load_settings(
            command, "vllm", "deepseek-v4-flash"
        )
        updated = self.manager._updated_container_command(
            command,
            "vllm",
            "deepseek-v4-flash",
            {
                **settings,
                "max_concurrency": 6,
                "context_window": 524288,
                "thinking_mode": "enabled",
            },
        )

        self.assertEqual(settings["max_concurrency"], 12)
        self.assertEqual(settings["kv_cache_dtype"], "nvfp4_ds_mla")
        self.assertEqual(settings["thinking_mode"], "disabled")
        self.assertIn("export CUDA_HOME=/opt/cuda", updated[-1])
        self.assertIn('$(( 12 * (5 + 1) ))', updated[-1])
        self.assertIn('"${SPEC}"', updated[-1])
        self.assertIn("--max-num-seqs 6", updated[-1])
        self.assertIn("--max-model-len 524288", updated[-1])
        self.assertIn("--port 8889", updated[-1])
        self.assertEqual(
            self.manager._container_load_settings(
                updated, "vllm", "deepseek-v4-flash"
            )["thinking_mode"],
            "enabled",
        )

    def test_sglang_settings_use_engine_specific_flags(self):
        model = "example/SGLang-Model"
        command = [
            "-m", "sglang.launch_server", "--model-path", model,
            "--host", "0.0.0.0", "--port", "8000",
            "--context-length", "131072",
            "--max-running-requests", "8",
            "--max-total-tokens", "2097152",
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
        self.assertIn("--max-total-tokens", settings["extra_args"])
        self.assertEqual(
            self.manager._cli_option(updated, {"--max-running-requests"}), "3"
        )
        self.assertEqual(
            self.manager._cli_option(updated, {"--context-length"}), "65536"
        )
        self.assertIn("--enable-cache-report", updated)
        self.assertEqual(
            self.manager._cli_option(updated, {"--max-total-tokens"}), "2097152"
        )

    async def test_update_clones_full_config_and_restores_running_state(self):
        name = "external-vllm"
        model = "example/Model"
        command = [
            "vllm", "serve", model, "--host", "0.0.0.0", "--port", "8000",
            "--max-num-seqs", "8",
        ]
        config = {
            "Image": "example/vllm:latest",
            "Entrypoint": command[:2],
            "Cmd": command[2:],
            "Env": [
                "SPECIAL_RUNTIME_FLAG=1", "NCCL_DEBUG=INFO",
                "IMAGE_DEFAULT=keep-private",
            ],
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
                    "environment": {"NCCL_DEBUG": "WARN", "VLLM_USE_V1": "1"},
                },
            )

        replacement = containers.get(name)
        self.assertTrue(result["ok"])
        self.assertEqual(replacement.status, "running")
        self.assertTrue(original.removed)
        self.assertEqual(api.created_config["Env"], [
            "SPECIAL_RUNTIME_FLAG=1",
            "IMAGE_DEFAULT=keep-private",
            "NCCL_DEBUG=WARN",
            "VLLM_USE_V1=1",
        ])
        self.assertEqual(api.created_config["HostConfig"], host_config)
        self.assertEqual(api.created_config["Entrypoint"], ["vllm", "serve"])
        self.assertEqual(api.created_config["Cmd"][0], model)
        self.assertEqual(
            self.manager._cli_option(
                api.created_config["Cmd"], {"--max-num-seqs"}
            ),
            "3",
        )

    async def test_update_preserves_all_docker_network_attachments(self):
        name = "external-vllm"
        model = "example/Model"
        command = ["vllm", "serve", model]
        config = {
            "Image": "example/vllm:latest", "Cmd": command,
            "Labels": {"vllm-model": model},
        }
        containers = FakeContainers()
        original = FakeContainer(
            containers, "original-container-id", name, config,
            {"NetworkMode": "frontend"},
        )
        original.attrs["NetworkSettings"] = {"Networks": {
            "frontend": {
                "Aliases": [original.short_id, name, "model-api"],
                "IPAMConfig": None,
            },
            "metrics": {
                "Aliases": ["model-metrics"],
                "IPAMConfig": {
                    "IPv4Address": "172.30.0.20",
                    "IPv6Address": "fd00::20",
                },
                "DriverOpts": {"com.example.option": "value"},
            },
        }}
        containers.add(original)
        api = FakeAPI(containers)
        self.manager.client = SimpleNamespace(containers=containers, api=api)
        self.manager.lock = asyncio.Lock()
        self.manager._container_summary = lambda container: {"name": container.name}

        inline_thread = mock.AsyncMock(
            side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)
        )
        with mock.patch("manager.asyncio.to_thread", inline_thread):
            await self.manager.update_container_settings(name, {
                **self.manager._container_load_settings(command, "vllm", model),
                "max_concurrency": 2,
            })

        self.assertEqual(
            api.created_config["NetworkingConfig"]["EndpointsConfig"],
            {
                "frontend": {"Aliases": [name, "model-api"]},
                "metrics": {
                    "Aliases": ["model-metrics"],
                    "DriverOpts": {"com.example.option": "value"},
                    "IPAMConfig": {
                        "IPv4Address": "172.30.0.20",
                        "IPv6Address": "fd00::20",
                    },
                },
            },
        )
        self.assertEqual(api.disconnected, [
            (original.id, "frontend", True),
            (original.id, "metrics", True),
        ])

    async def test_update_infers_unlabelled_sglang_from_image(self):
        name = "external-sglang"
        model = "example/SGLang-Model"
        entrypoint = ["python", "-m", "sglang.launch_server"]
        command = [
            *entrypoint, "--model-path", model,
            "--host", "0.0.0.0", "--port", "8000",
            "--context-length", "131072", "--enable-metrics",
        ]
        config = {
            "Image": "lmsysorg/sglang:latest",
            "Entrypoint": entrypoint,
            "Cmd": command[len(entrypoint):],
            "Env": [], "Labels": {},
        }
        containers = FakeContainers()
        original = FakeContainer(containers, "sglang-id", name, config, {})
        containers.add(original)
        api = FakeAPI(containers)
        self.manager.client = SimpleNamespace(containers=containers, api=api)
        self.manager.lock = asyncio.Lock()
        self.manager._container_summary = lambda container: {
            "name": container.name, "status": container.status,
        }

        inline_thread = mock.AsyncMock(
            side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)
        )
        with mock.patch("manager.asyncio.to_thread", inline_thread):
            await self.manager.update_container_settings(name, {
                **self.manager._container_load_settings(command, "sglang", model),
                "context_window": 262144,
            })

        self.assertEqual(api.created_config["Entrypoint"], entrypoint)
        self.assertEqual(api.created_config["Cmd"][:2], ["--model-path", model])
        self.assertEqual(
            self.manager._cli_option(
                api.created_config["Cmd"], {"--context-length"}, int,
            ),
            262144,
        )

    async def test_update_rejects_hidden_credential_flags_without_replacement(self):
        name = "protected-vllm"
        model = "example/Model"
        command = ["vllm", "serve", model, "--api-key", "do-not-expose"]
        config = {
            "Image": "example/vllm:latest", "Cmd": command,
            "Env": [], "Labels": {"vllm-model": model},
        }
        containers = FakeContainers()
        original = FakeContainer(containers, "protected-id", name, config, {})
        containers.add(original)
        api = FakeAPI(containers)
        self.manager.client = SimpleNamespace(containers=containers, api=api)
        self.manager.lock = asyncio.Lock()

        inline_thread = mock.AsyncMock(
            side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)
        )
        with mock.patch("manager.asyncio.to_thread", inline_thread):
            with self.assertRaisesRegex(ValueError, "credential-bearing"):
                await self.manager.update_container_settings(name, {
                    "command_flags": "--max-num-seqs 4",
                })

        self.assertIsNone(api.created_config)
        self.assertIs(containers.get(name), original)

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
        original.attrs["NetworkSettings"] = {"Networks": {
            "inference": {
                "Aliases": [name, "model-api"],
                "IPAMConfig": {"IPv4Address": "172.31.0.10"},
            },
        }}
        containers.add(original)
        api = FakeAPI(containers, fail=True)
        self.manager.client = SimpleNamespace(
            containers=containers, api=api
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
        self.assertEqual(api.disconnected, [(original.id, "inference", True)])
        self.assertEqual(api.connected, [(
            original.id,
            "inference",
            {
                "ipv4_address": "172.31.0.10",
                "aliases": [name, "model-api"],
            },
        )])

    async def test_replacement_that_exits_after_start_rolls_back(self):
        class DelayedExitContainer(FakeContainer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.reload_count = 0
                self.started = False

            def start(self):
                self.started = True
                self.status = "running"

            def reload(self):
                if self.started:
                    self.reload_count += 1
                    if self.reload_count >= 2:
                        self.status = "exited"

        class DelayedExitAPI(FakeAPI):
            def create_container_from_config(self, config, name=None):
                self.created_config = copy.deepcopy(config)
                cid = "exiting-replacement-id"
                replacement = DelayedExitContainer(
                    self.containers, cid, name, config,
                    config.get("HostConfig") or {}, status="created",
                )
                self.containers.add(replacement)
                self.replacement = replacement
                return {"Id": cid}

        name = "external-vllm"
        model = "example/Model"
        command = ["vllm", "serve", model]
        config = {
            "Image": "example/vllm:latest", "Cmd": command,
            "Labels": {"vllm-model": model},
        }
        containers = FakeContainers()
        original = FakeContainer(
            containers, "original-container-id", name, config, {}, status="running"
        )
        containers.add(original)
        api = DelayedExitAPI(containers)
        self.manager.client = SimpleNamespace(containers=containers, api=api)
        self.manager.lock = asyncio.Lock()

        inline_thread = mock.AsyncMock(
            side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)
        )
        with mock.patch("manager.asyncio.to_thread", inline_thread):
            with self.assertRaisesRegex(RuntimeError, "exited during startup"):
                await self.manager.update_container_settings(name, {
                    **self.manager._container_load_settings(command, "vllm", model),
                    "max_concurrency": 2,
                })

        self.assertGreaterEqual(api.replacement.reload_count, 2)
        self.assertTrue(api.replacement.removed)
        self.assertIs(containers.get(name), original)
        self.assertEqual(original.status, "running")
        self.assertFalse(original.removed)

    async def test_failed_ownership_confirmation_restores_original_container(self):
        name = "managed-vllm"
        model = "example/Model"
        command = ["vllm", "serve", model, "--max-num-seqs", "8"]
        config = {
            "Image": "example/vllm:latest",
            "Cmd": command,
            "Labels": {"vllm-model": model, "io.sparkdeck.managed": "1"},
        }
        containers = FakeContainers()
        original = FakeContainer(
            containers, "original-container-id", name, config, {}, status="running"
        )
        containers.add(original)
        self.manager.client = SimpleNamespace(
            containers=containers, api=FakeAPI(containers)
        )
        self.manager.lock = asyncio.Lock()
        self.manager._container_summary = lambda container: {
            "name": container.name, "status": container.status
        }
        ledger = mock.MagicMock()
        ledger.confirm.side_effect = OSError("ledger write failed")
        self.manager.managed_workload_ledger = ledger

        inline_thread = mock.AsyncMock(
            side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)
        )
        with mock.patch("manager.asyncio.to_thread", inline_thread):
            with self.assertRaisesRegex(OSError, "ledger write failed"):
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


class ContainerToRecipeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.manager = Manager.__new__(Manager)

    def _attach(self, containers) -> tempfile.TemporaryDirectory:
        self.manager.client = SimpleNamespace(containers=containers)
        self.manager.lock = asyncio.Lock()
        self.manager.recipes = []
        directory = tempfile.TemporaryDirectory()
        self.manager.recipes_path = Path(directory.name) / "recipes.json"
        return directory

    async def test_sglang_container_imports_scalars_and_preserves_flags(self):
        model = "RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead"
        command = [
            "python3", "-m", "sglang.launch_server",
            "--model-path", model,
            "--host", "0.0.0.0", "--port", "8888",
            "--tp-size", "1",
            "--context-length", "262144",
            "--max-running-requests", "10",
            "--mem-fraction-static", "0.90",
            "--kv-cache-dtype", "fp8_e4m3",
            "--reasoning-parser", "qwen3",
            "--mamba-full-memory-ratio", "4.21",
            "--enable-metrics",
        ]
        containers = FakeContainers()
        containers.add(FakeContainer(
            containers, "sglang-cid", "qwen3.8-27b-sglang",
            {"Image": "lmsysorg/sglang:latest", "Cmd": command, "Labels": {}},
            {},
        ))
        with self._attach(containers):
            recipe = await self.manager.container_to_recipe("qwen3.8-27b-sglang")

        self.assertEqual(recipe["engine"], "sglang")
        self.assertEqual(recipe["model"], model)
        self.assertEqual(recipe["sg_context_length"], 262144)
        self.assertEqual(recipe["sg_max_running_requests"], 10)
        self.assertEqual(recipe["sg_mem_fraction"], 0.9)
        self.assertEqual(recipe["sg_tp_size"], 1)
        args = recipe["extra_args"]
        self.assertEqual(args[args.index("--kv-cache-dtype") + 1], "fp8_e4m3")
        self.assertIn("--mamba-full-memory-ratio", args)
        self.assertIn("--reasoning-parser", args)
        self.assertIn("--enable-metrics", args)
        self.assertNotIn("--context-length", args)
        self.assertNotIn(model, args)

    async def test_vllm_container_imports_controls_into_extra_args(self):
        model = "example/Model"
        command = [
            "vllm", "serve", model,
            "--host", "0.0.0.0", "--port", "8000",
            "--gpu-memory-utilization", "0.85",
            "--max-num-seqs", "12",
            "--max-model-len", "1048576",
            "--kv-cache-dtype", "fp8",
            "--max-cudagraph-capture-size", "512",
            "--enable-prefix-caching",
        ]
        containers = FakeContainers()
        containers.add(FakeContainer(
            containers, "vllm-cid", "vllm-example",
            {"Image": "example/vllm:latest", "Cmd": command, "Labels": {}},
            {},
        ))
        with self._attach(containers):
            recipe = await self.manager.container_to_recipe("vllm-example")

        self.assertEqual(recipe["engine"], "vllm")
        self.assertEqual(recipe["model"], model)
        self.assertEqual(recipe["gpu_memory_utilization"], 0.85)
        args = recipe["extra_args"]
        self.assertEqual(
            self.manager._cli_option(args, {"--max-model-len"}, int), 1048576
        )
        self.assertEqual(self.manager._cli_option(args, {"--max-num-seqs"}, int), 12)
        self.assertEqual(self.manager._cli_option(args, {"--kv-cache-dtype"}), "fp8")
        self.assertEqual(
            self.manager._cli_option(args, {"--max-cudagraph-capture-size"}, int), 512
        )
        self.assertIn("--enable-prefix-caching", args)

    async def test_vllm_entrypoint_and_cmd_import_as_one_command(self):
        containers = FakeContainers()
        containers.add(FakeContainer(
            containers, "entrypoint-cid", "entrypoint-vllm",
            {
                "Image": "example/vllm:latest",
                "Entrypoint": ["vllm", "serve"],
                "Cmd": ["org/model", "--max-model-len", "65536", "--enable-prefix-caching"],
                "Labels": {},
            },
            {},
        ))
        with self._attach(containers):
            recipe = await self.manager.container_to_recipe("entrypoint-vllm")

        self.assertEqual(recipe["model"], "org/model")
        self.assertEqual(
            self.manager._cli_option(recipe["extra_args"], {"--max-model-len"}, int),
            65536,
        )
        self.assertIn("--enable-prefix-caching", recipe["extra_args"])

    async def test_missing_container_raises_lookup_error(self):
        with self._attach(FakeContainers()):
            with self.assertRaises(LookupError):
                await self.manager.container_to_recipe("gone")

    async def test_command_without_model_raises_value_error(self):
        containers = FakeContainers()
        containers.add(FakeContainer(
            containers, "cid", "mystery",
            {"Image": "lmsysorg/sglang:latest", "Cmd": ["--help"], "Labels": {}},
            {},
        ))
        with self._attach(containers):
            with self.assertRaises(ValueError):
                await self.manager.container_to_recipe("mystery")

    async def test_shell_wrapped_vllm_container_imports_model(self):
        containers = FakeContainers()
        containers.add(FakeContainer(
            containers, "cid", "wrapped-vllm",
            {
                "Image": "example/vllm:latest",
                "Cmd": ["sh", "-c", "vllm serve org/model --max-num-seqs 8 --enable-prefix-caching"],
                "Labels": {},
            },
            {},
        ))
        with self._attach(containers):
            recipe = await self.manager.container_to_recipe("wrapped-vllm")

        self.assertEqual(recipe["model"], "org/model")
        self.assertEqual(recipe["engine"], "vllm")
        self.assertIn("--enable-prefix-caching", recipe["extra_args"])

    async def test_sglang_inline_model_path_imports(self):
        command = [
            "python3", "-m", "sglang.launch_server",
            "--model-path=org/inline-model",
            "--host", "0.0.0.0", "--port", "8888",
        ]
        containers = FakeContainers()
        containers.add(FakeContainer(
            containers, "cid", "inline-sglang",
            {"Image": "lmsysorg/sglang:latest", "Cmd": command, "Labels": {}},
            {},
        ))
        with self._attach(containers):
            recipe = await self.manager.container_to_recipe("inline-sglang")

        self.assertEqual(recipe["model"], "org/inline-model")
        self.assertEqual(recipe["engine"], "sglang")

    async def test_reimport_clears_removed_scalars(self):
        base_cmd = [
            "python3", "-m", "sglang.launch_server",
            "--model-path", "org/model", "--host", "0.0.0.0", "--port", "8888",
            "--context-length", "262144",
        ]
        trimmed_cmd = [
            "python3", "-m", "sglang.launch_server",
            "--model-path", "org/model", "--host", "0.0.0.0", "--port", "8888",
        ]
        containers = FakeContainers()
        containers.add(FakeContainer(
            containers, "cid", "sglang-a",
            {"Image": "lmsysorg/sglang:latest", "Cmd": base_cmd, "Labels": {}},
            {},
        ))
        with self._attach(containers):
            first = await self.manager.container_to_recipe("sglang-a")
            self.assertEqual(first["sg_context_length"], 262144)

            containers.items.pop("sglang-a")
            replacement = FakeContainer(
                containers, "cid2", "sglang-a",
                {"Image": "lmsysorg/sglang:latest", "Cmd": trimmed_cmd, "Labels": {}},
                {},
            )
            containers.add(replacement)
            second = await self.manager.container_to_recipe("sglang-a")

        self.assertEqual(second["id"], first["id"])
        self.assertIsNone(second["sg_context_length"])

    async def test_summary_preserves_llama_runtime_label(self):
        containers = FakeContainers()
        container = FakeContainer(
            containers, "llama-cid", "llama-box",
            {
                "Image": "ghcr.io/ggml/llama.cpp:server",
                "Cmd": ["llama-server", "--port", "8080"],
                "Labels": {
                    "io.sparkdeck.managed": "1",
                    "io.sparkdeck.runtime": "llama.cpp",
                },
            },
            {},
        )

        summary = self.manager._container_summary(container)

        self.assertIsNotNone(summary)
        self.assertEqual(summary["engine"], "llama.cpp")

    async def test_add_recipe_rejects_invalid_sg_scalars(self):
        with self._attach(FakeContainers()):
            with self.assertRaises(ValueError):
                await self.manager.add_recipe("org/model", sg_tp_size=1.5)
            with self.assertRaises(ValueError):
                await self.manager.add_recipe("org/model", sg_mem_fraction=1.2)
            recipe = await self.manager.add_recipe(
                "org/model", sg_tp_size="2", sg_mem_fraction="0.9",
            )

        self.assertEqual(recipe["sg_tp_size"], 2)
        self.assertEqual(recipe["sg_mem_fraction"], 0.9)

    async def test_update_recipe_validates_sg_scalars(self):
        with self._attach(FakeContainers()):
            recipe = await self.manager.add_recipe("org/model", sg_tp_size=1)
            with self.assertRaises(ValueError):
                await self.manager.update_recipe(
                    recipe["id"], {"sg_mem_fraction": 1.2},
                )
            with self.assertRaises(ValueError):
                await self.manager.update_recipe(
                    recipe["id"], {"sg_tp_size": 1.5},
                )
            updated = await self.manager.update_recipe(
                recipe["id"], {"sg_tp_size": 2},
            )

        self.assertEqual(updated["sg_tp_size"], 2)

    async def test_update_deployment_settings_validates_sg_scalars(self):
        with self.assertRaises(ValueError):
            await self.manager.update_deployment_settings(
                "dep-1", {"sg_mem_fraction": 1.2},
            )
        with self.assertRaises(ValueError):
            await self.manager.update_deployment_settings(
                "dep-1", {"sg_tp_size": 0},
            )

    def test_summary_infers_sglang_engine_from_image(self):
        model = "RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead"
        command = [
            "python3", "-m", "sglang.launch_server",
            "--model-path", model, "--host", "0.0.0.0", "--port", "8888",
            "--context-length", "262144", "--max-running-requests", "10",
        ]
        containers = FakeContainers()
        container = FakeContainer(
            containers, "sglang-cid", "qwen3.8-27b-sglang",
            {"Image": "lmsysorg/sglang:latest", "Cmd": command, "Labels": {}},
            {},
        )

        summary = self.manager._container_summary(container)

        self.assertIsNotNone(summary)
        self.assertEqual(summary["engine"], "sglang")
        self.assertEqual(summary["model"], model)
        self.assertEqual(summary["load_settings"]["context_window"], 262144)


if __name__ == "__main__":
    unittest.main()
