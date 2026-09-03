import asyncio
import json
import os
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, call, patch

import httpx

from cluster import NodeRegistry
from manager import Manager, PERSISTED_DEPLOYMENT_ARGS_ERROR
from sparkdeck.onboarding import resolve_agent_connection


with patch("docker.from_env", return_value=Mock()):
    import server


class SettingsApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.assignment_patch = patch.object(
            server.onboarding.assignment, "load", return_value=None,
        )
        self.assignment_patch.start()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app),
            base_url="http://test",
        )

    async def asyncTearDown(self):
        await self.client.aclose()
        self.assignment_patch.stop()

    async def test_get_returns_only_masked_credential_status_and_live_ui_settings(self):
        sentinel = "hf_get_sentinel_secret"
        stored = {
            "theme": "dark",
            "default_runtime": "sglang",
            "default_context_length": 24576,
        }
        with (
            patch.object(
                server.sparkdeck.store,
                "get_setting",
                side_effect=lambda key, default: stored.get(key, default),
            ),
            patch.object(server.manager, "_resolved_hf_token", return_value=sentinel),
            patch.dict(server.manager.settings, {"cluster_node_name": "gx10-node-1"}),
        ):
            response = await self.client.get("/api/v1/settings")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "theme": "dark",
            "default_runtime": "sglang",
            "default_context_length": 24576,
            "vllm_image": server.manager.settings["vllm_image"],
            "hf_token_configured": True,
        })
        # The node name must never ride the settings response: on joined
        # workers it is forwarded to the controller and would name the wrong
        # machine. The unforwarded onboarding status carries it instead.
        self.assertNotIn("cluster_node_name", response.json())
        self.assertNotIn(sentinel, response.text)

    async def test_put_stores_credential_on_controller_without_echoing_it(self):
        sentinel = "hf_put_sentinel_secret"
        update_settings = AsyncMock()
        with (
            patch.object(server.sparkdeck.store, "set_setting") as set_setting,
            patch.object(server.manager, "update_settings", update_settings),
            patch.object(server.manager, "_resolved_hf_token", return_value=sentinel),
        ):
            response = await self.client.put("/api/v1/settings", json={
                "theme": "light",
                "community_api_url": "https://community.example/",
                "hf_token": sentinel,
                "default_runtime": "sglang",
                "default_context_length": 32768,
            })

        # A legacy community_api_url field is ignored, not stored or echoed.
        self.assertNotIn("https://community.example", response.text)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "theme": "light",
            "default_runtime": "sglang",
            "default_context_length": 32768,
            "vllm_image": server.manager.settings["vllm_image"],
            "hf_token_configured": True,
        })
        self.assertNotIn(sentinel, response.text)
        update_settings.assert_awaited_once_with({"hf_token": sentinel})
        self.assertEqual(set_setting.call_args_list, [
            call("theme", "light"),
            call("default_runtime", "sglang"),
            call("default_context_length", 32768),
        ])

    async def test_invalid_deployment_defaults_are_rejected_without_saving(self):
        with patch.object(server.sparkdeck.store, "set_setting") as set_setting:
            bad_runtime = await self.client.put(
                "/api/v1/settings", json={"default_runtime": "tensorrt"},
            )
            bad_context = await self.client.put(
                "/api/v1/settings", json={"default_context_length": 32},
            )

        self.assertEqual(bad_runtime.status_code, 400)
        self.assertEqual(bad_context.status_code, 400)
        set_setting.assert_not_called()

    async def test_invalid_credential_does_not_partially_save_other_settings(self):
        sentinel = "hf invalid sentinel"
        update_settings = AsyncMock()
        with (
            patch.object(server.sparkdeck.store, "set_setting") as set_setting,
            patch.object(server.manager, "update_settings", update_settings),
        ):
            response = await self.client.put("/api/v1/settings", json={
                "theme": "dark",
                "hf_token": sentinel,
            })

        self.assertEqual(response.status_code, 400)
        self.assertNotIn(sentinel, response.text)
        set_setting.assert_not_called()
        update_settings.assert_not_awaited()

    async def test_blank_credential_preserves_the_configured_value(self):
        update_settings = AsyncMock()
        with (
            patch.object(server.sparkdeck.store, "set_setting"),
            patch.object(server.manager, "update_settings", update_settings),
            patch.object(
                server.manager, "_resolved_hf_token", return_value="hf_existing_secret",
            ),
        ):
            response = await self.client.put("/api/v1/settings", json={
                "theme": "system",
                "hf_token": "   ",
            })

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["hf_token_configured"])
        update_settings.assert_not_awaited()

    async def test_explicit_clear_removes_the_saved_credential(self):
        clear_hf_token = AsyncMock()
        with (
            patch.object(
                server.sparkdeck.store, "get_setting",
                side_effect=lambda key, default: default,
            ),
            patch.object(server.manager, "clear_hf_token", clear_hf_token),
            patch.object(server.manager, "_resolved_hf_token", return_value=""),
        ):
            response = await self.client.delete("/api/v1/settings/hf-token")

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["hf_token_configured"])
        clear_hf_token.assert_awaited_once_with()

    async def test_agent_launch_errors_redact_forwarded_credential(self):
        sentinel = "hf_agent_sentinel_secret"
        with (
            patch.object(server, "_require_agent"),
            patch.object(
                server.manager,
                "create_container",
                AsyncMock(side_effect=RuntimeError(f"launch rejected {sentinel}")),
            ),
            patch.object(server.manager, "_resolved_hf_token", return_value=""),
        ):
            response = await self.client.post("/api/agent/containers", json={
                "model": "org/private-model",
                "cluster_member": {"deployment_id": "dep-1", "node_id": "worker-1"},
                "hf_token": sentinel,
            })

        self.assertEqual(response.status_code, 500)
        self.assertNotIn(sentinel, response.text)
        self.assertIn("REDACTED", response.text)

    async def test_agent_launch_forwards_promoted_shared_memory_size(self):
        create_container = AsyncMock(return_value={"status": "running"})
        source_shm_size = 64 * 1024 ** 3
        with (
            patch.object(server, "_require_agent"),
            patch.object(server.manager, "create_container", create_container),
        ):
            response = await self.client.post("/api/agent/containers", json={
                "model": "org/model",
                "cluster_member": {
                    "deployment_id": "dep-1", "node_id": "worker-1",
                },
                "shm_size": source_shm_size,
                "infiniband_device": True,
            })

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            create_container.await_args.kwargs["shm_size"], source_shm_size,
        )
        self.assertIs(
            create_container.await_args.kwargs["infiniband_device"], True,
        )

    def test_log_redaction_handles_json_keys_and_raw_configured_token(self):
        sentinel = "hf_log_sentinel_secret"
        with patch.object(server.manager, "_resolved_hf_token", return_value=sentinel):
            redacted = server._redact_log(json.dumps({
                "hf_token": sentinel,
                "HUGGING_FACE_HUB_TOKEN": sentinel,
            }))

        self.assertNotIn(sentinel, redacted)
        self.assertGreaterEqual(redacted.count("REDACTED"), 2)


class ClusterCredentialTests(unittest.IsolatedAsyncioTestCase):
    def test_positional_token_suffix_is_not_treated_as_a_credential(self):
        args = ["--served-model-name", "my-token", "--dtype", "auto"]

        self.assertEqual(Manager._without_sensitive_cli_credentials(args), args)
        Manager._reject_sensitive_cli_credentials(args)

    def test_legacy_cli_credentials_are_removed_and_new_ones_are_rejected(self):
        instance = Manager.__new__(Manager)
        instance.recipes = [{
            "id": "legacy-recipe",
            "extra_args": ["--dtype", "auto", "--hf-token=hf_recipe_secret"],
        }]
        instance.deployments = [{
            "id": "legacy-deployment",
            "launch_settings": {
                "extra_args": ["--hf_token", "hf_deployment_secret", "--revision", "main"],
            },
        }]

        self.assertTrue(instance._migrate_recipe_hf_credentials())
        self.assertTrue(instance._migrate_deployment_hf_credentials())
        self.assertEqual(instance.recipes[0]["extra_args"], ["--dtype", "auto"])
        self.assertEqual(
            instance.deployments[0]["launch_settings"]["extra_args"],
            ["--revision", "main"],
        )
        self.assertTrue(instance.deployments[0]["settings_dirty"])
        with self.assertRaisesRegex(ValueError, "credentials in Settings"):
            instance._reject_hf_cli_credentials(["--hf-token", "hf_new_secret"])

    def test_legacy_stopped_deployment_remains_stopped_after_load(self):
        with tempfile.TemporaryDirectory() as directory:
            instance = Manager.__new__(Manager)
            instance.deployments_path = Path(directory) / "deployments.json"
            instance.deployments_path.write_text(json.dumps([{
                "id": "legacy-stopped", "status": "stopped",
            }]), encoding="utf-8")

            loaded = instance._load_deployments()

            self.assertEqual(loaded[0]["desired_state"], "stopped")

    def test_malformed_recipe_args_are_isolated_and_durable_during_startup(self):
        class StartupContinued(Exception):
            pass

        with tempfile.TemporaryDirectory() as directory:
            recipes_path = Path(directory) / "recipes.json"
            recipes_path.write_text(json.dumps([
                {
                    "id": "malformed", "model": "org/bad", "engine": "vllm",
                    "extra_args": 42,
                },
                {
                    "id": "valid", "model": "org/good", "engine": "vllm",
                    "extra_args": ["--dtype", "auto", "--hf-token", "hf_secret"],
                },
            ]), encoding="utf-8")

            with (
                patch.object(Manager, "_load_settings", return_value={}),
                patch.object(
                    Manager, "_load_unsloth_settings",
                    side_effect=StartupContinued,
                ),
            ):
                with self.assertRaises(StartupContinued):
                    Manager(Path(directory))

            persisted = json.loads(recipes_path.read_text(encoding="utf-8"))
            malformed, valid = persisted
            self.assertEqual(malformed["extra_args"], [])
            self.assertFalse(malformed["supported"])
            self.assertIn("extra_args", malformed["error"])
            self.assertEqual(valid["extra_args"], ["--dtype", "auto"])
            self.assertNotIn("hf_secret", recipes_path.read_text(encoding="utf-8"))
            self.assertNotEqual(valid.get("supported"), False)
            with patch.object(server.manager, "recipe_launches", {}):
                malformed_item = server._public_recipe(malformed)
                valid_item = server._public_recipe(valid)
            self.assertFalse(malformed_item["supported"])
            self.assertIn("extra_args", malformed_item["error"])
            self.assertTrue(valid_item["supported"])

            restarted = Manager.__new__(Manager)
            restarted.recipes_path = recipes_path
            restarted.recipes = restarted._load_recipes()
            self.assertFalse(restarted._migrate_recipe_hf_credentials())
            self.assertFalse(restarted.recipes[0]["supported"])

    async def test_malformed_deployment_args_are_isolated_across_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            deployments_path = path / "deployments.json"
            deployments_path.write_text(json.dumps([
                {
                    "id": "malformed", "engine": "vllm", "status": "registered",
                    "launch_settings": {"engine": "vllm", "extra_args": False},
                },
                {
                    "id": "valid", "engine": "vllm", "status": "registered",
                    "launch_settings": {
                        "engine": "vllm",
                        "extra_args": ["--dtype", "auto", "--hf-token", "hf_secret"],
                    },
                },
            ]), encoding="utf-8")

            with patch("manager.docker.from_env", return_value=Mock()):
                first = Manager(path)
                await first.http.aclose()

            malformed, valid = first.deployments
            self.assertEqual(malformed["launch_settings"]["extra_args"], [])
            self.assertEqual(malformed["status"], "error")
            self.assertIn("launch_settings.extra_args", malformed["error"])
            self.assertEqual(
                malformed["launch_settings_error"],
                PERSISTED_DEPLOYMENT_ARGS_ERROR,
            )
            self.assertNotEqual(valid.get("status"), "error")
            self.assertNotIn(
                "hf_secret", deployments_path.read_text(encoding="utf-8"),
            )
            first_persisted = deployments_path.read_text(encoding="utf-8")

            with patch("manager.docker.from_env", return_value=Mock()):
                restarted = Manager(path)
                await restarted.http.aclose()

            self.assertEqual(
                deployments_path.read_text(encoding="utf-8"), first_persisted,
            )
            self.assertEqual(restarted.deployments[0]["status"], "error")
            self.assertEqual(
                restarted.deployments[0]["launch_settings_error"],
                PERSISTED_DEPLOYMENT_ARGS_ERROR,
            )
            self.assertNotEqual(restarted.deployments[1].get("status"), "error")

    def test_explicit_empty_controller_credential_disables_worker_fallback(self):
        instance = Manager.__new__(Manager)
        instance.settings = {
            "hf_token": "hf_worker_local_secret",
            "hf_cache": "",
        }

        self.assertEqual(instance._container_hf_environment(""), {})
        self.assertEqual(
            instance._container_hf_environment()["HF_TOKEN"],
            "hf_worker_local_secret",
        )

    def test_settings_are_written_atomically_to_a_private_file(self):
        with tempfile.TemporaryDirectory() as directory:
            instance = Manager.__new__(Manager)
            instance.settings_path = Path(directory) / "settings.json"
            instance.settings = {"hf_token": "hf_private_secret"}

            instance._save_settings()

            self.assertEqual(
                json.loads(instance.settings_path.read_text(encoding="utf-8")),
                instance.settings,
            )
            self.assertFalse(Path(f"{instance.settings_path}.tmp").exists())
            if os.name != "nt":
                self.assertEqual(instance.settings_path.stat().st_mode & 0o777, 0o600)

    def test_private_json_temporary_file_is_created_with_mode_0600(self):
        with tempfile.TemporaryDirectory() as directory:
            instance = Manager.__new__(Manager)
            instance.settings_path = Path(directory) / "settings.json"
            instance.settings = {"hf_token": "hf_private_secret"}

            with patch("sparkdeck.private_json.os.open", wraps=os.open) as secure_open:
                instance._save_settings()

            self.assertEqual(secure_open.call_args.args[2], 0o600)
            self.assertTrue(secure_open.call_args.args[1] & os.O_EXCL)

    async def test_manager_explicitly_clears_saved_credential(self):
        with tempfile.TemporaryDirectory() as directory:
            instance = Manager.__new__(Manager)
            instance.lock = asyncio.Lock()
            instance.settings_path = Path(directory) / "settings.json"
            instance.settings = {"hf_token": "hf_private_secret"}
            instance.virtual_nas = Mock()
            instance.virtual_nas.stop = AsyncMock()

            result = await instance.clear_hf_token()

            self.assertEqual(instance.settings["hf_token"], "")
            self.assertFalse(result["hf_token_configured"])

    async def test_remote_credential_destination_is_revalidated_before_send(self):
        requests = []
        resolutions = []

        def resolve(host, port, **kwargs):
            resolutions.append((host, port))
            address = "100.100.20.30" if len(resolutions) == 1 else "198.51.100.20"
            return [(
                socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "",
                (address, port),
            )]

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json={"name": "container"}, request=request)

        with tempfile.TemporaryDirectory() as directory:
            client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            registry = NodeRegistry(
                Path(directory), client, "controller",
                connection_resolver=resolve_agent_connection,
            )
            registry.nodes = [{
                "id": "worker-1", "name": "Worker",
                "agent_url": "http://rebound.example:7878",
                "agent_token": "agent-secret", "enabled": True,
            }]
            instance = Manager.__new__(Manager)
            instance.node_registry = registry
            try:
                with patch(
                    "sparkdeck.onboarding.socket.getaddrinfo", side_effect=resolve,
                ):
                    result = await instance._create_member("worker-1", {
                        "model": "org/private-model",
                        "hf_token": "hf_forwarded_secret",
                    })
            finally:
                await client.aclose()

        self.assertEqual(result, {"name": "container"})
        self.assertEqual(resolutions, [("rebound.example", 7878)])
        self.assertEqual(
            str(requests[0].url),
            "http://100.100.20.30:7878/api/agent/containers",
        )
        self.assertEqual(requests[0].headers["host"], "rebound.example:7878")
        self.assertEqual(requests[0].headers["authorization"], "Bearer agent-secret")
        self.assertEqual(
            json.loads(requests[0].content)["hf_token"], "hf_forwarded_secret",
        )
