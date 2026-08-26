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
from manager import Manager
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
        stored = {"theme": "dark", "community_api_url": "https://example.test"}
        with (
            patch.object(
                server.sparkdeck.store,
                "get_setting",
                side_effect=lambda key, default: stored.get(key, default),
            ),
            patch.object(server.manager, "_resolved_hf_token", return_value=sentinel),
        ):
            response = await self.client.get("/api/v1/settings")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "theme": "dark",
            "community_api_url": "https://example.test",
            "hf_token_configured": True,
        })
        self.assertNotIn(sentinel, response.text)
        self.assertNotIn("default_runtime", response.text)
        self.assertNotIn("default_context_length", response.text)

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

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "theme": "light",
            "community_api_url": "https://community.example",
            "hf_token_configured": True,
        })
        self.assertNotIn(sentinel, response.text)
        update_settings.assert_awaited_once_with({"hf_token": sentinel})
        self.assertEqual(set_setting.call_args_list, [
            call("theme", "light"),
            call("community_api_url", "https://community.example"),
        ])

    async def test_invalid_credential_does_not_partially_save_other_settings(self):
        sentinel = "hf invalid sentinel"
        update_settings = AsyncMock()
        with (
            patch.object(server.sparkdeck.store, "set_setting") as set_setting,
            patch.object(server.manager, "update_settings", update_settings),
        ):
            response = await self.client.put("/api/v1/settings", json={
                "theme": "dark",
                "community_api_url": "",
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
                "community_api_url": "",
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
