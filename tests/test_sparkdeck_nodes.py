import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import httpx

from cluster import AGENT_PROTOCOL_VERSION, NodeRegistry
from manager import Manager
from sparkdeck.onboarding import is_forwardable_path
from sparkdeck.service import SparkDeckService


def node(node_id: str, name: str, *, local: bool = False) -> dict:
    return {
        "id": node_id,
        "name": name,
        "local": local,
        "enabled": True,
        "status": "online",
        "online": True,
        "docker_ready": True,
        "fabric_ready": True,
        "agent_url": f"https://{node_id}.private.example",
        "fabric_ip": "169.254.10.2",
        "stats": {"gpus": [{"name": "GB10"}]},
        "disk": {"free_bytes": 1000},
    }


class NodePullTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.manager = Manager.__new__(Manager)
        self.nodes = [node("local", "Coordinator", local=True), node("remote-1", "Worker")]
        self.manager.cluster_nodes = AsyncMock(return_value=self.nodes)
        self.manager.pull_image_result = AsyncMock(return_value={"ok": True})
        self.manager.node_registry = Mock()
        self.manager.node_registry.request = AsyncMock(return_value={"ok": True})

    async def test_pull_dispatches_only_to_selected_nodes_and_returns_safe_metadata(self):
        result = await self.manager.pull_image_on_nodes(
            "ghcr.io/example/vllm:latest", ["remote-1", "local", "remote-1"],
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["node_ids"], ["remote-1", "local"])
        self.manager.pull_image_result.assert_awaited_once_with("ghcr.io/example/vllm:latest")
        self.manager.node_registry.request.assert_awaited_once_with(
            "remote-1", "POST", "/api/agent/images/pull",
            json_body={"image": "ghcr.io/example/vllm:latest"}, timeout=1800,
        )
        self.assertNotIn("agent_url", result["selected_nodes"][0])
        self.assertNotIn("fabric_ip", result["selected_nodes"][0])

    async def test_pull_reports_per_node_partial_failure(self):
        self.manager.node_registry.request.side_effect = RuntimeError("agent unavailable")

        result = await self.manager.pull_image_on_nodes("example/image", ["local", "remote-1"])

        self.assertFalse(result["ok"])
        self.assertTrue(result["results"][0]["ok"])
        self.assertEqual(result["results"][1]["error"], "agent unavailable")

    async def test_unknown_node_is_rejected_before_pull(self):
        with self.assertRaisesRegex(ValueError, "unknown cluster node"):
            await self.manager.pull_image_on_nodes("example/image", ["missing"])
        self.manager.pull_image_result.assert_not_awaited()


class NodeRenameTests(unittest.IsolatedAsyncioTestCase):
    async def test_local_rename_is_normalized_and_persisted(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = Manager.__new__(Manager)
            manager.settings = {"cluster_node_name": "Old name"}
            manager.settings_path = Path(directory) / "settings.json"
            manager.lock = asyncio.Lock()

            result = await manager.rename_cluster_node("local", "  Main   Spark  ")

            self.assertEqual(result["name"], "Main Spark")
            self.assertEqual(result["name_sync"], "local")
            saved = json.loads(manager.settings_path.read_text())
            self.assertEqual(saved["cluster_node_name"], "Main Spark")

    async def test_remote_rename_persists_registry_and_syncs_authenticated_worker(self):
        manager = Manager.__new__(Manager)
        manager.node_registry = Mock()
        manager.node_registry.get.return_value = {
            "id": "remote-1", "name": "Old", "agent_token": "secret",
        }
        manager.node_registry.update.return_value = {
            "id": "remote-1", "name": "Compute 1", "agent_url": "https://private",
        }
        manager.node_registry.request = AsyncMock(return_value={"name": "Compute 1"})
        manager.node_registry._status_cache = {"remote-1": (1, {})}

        result = await manager.rename_cluster_node("remote-1", "Compute 1")

        manager.node_registry.update.assert_called_once_with(
            "remote-1", {"name": "Compute 1"},
        )
        manager.node_registry.request.assert_awaited_once_with(
            "remote-1", "PATCH", "/api/agent/node",
            json_body={"name": "Compute 1"}, timeout=10,
        )
        self.assertEqual(result["name_sync"], "synchronized")
        self.assertNotIn("agent_url", result)
        self.assertNotIn("agent_token", result)

    async def test_offline_remote_rename_remains_durable_and_reports_pending_sync(self):
        manager = Manager.__new__(Manager)
        manager.node_registry = Mock()
        manager.node_registry.get.return_value = {"id": "remote-1", "name": "Old"}
        manager.node_registry.update.return_value = {"id": "remote-1", "name": "New"}
        manager.node_registry.request = AsyncMock(side_effect=RuntimeError("offline"))
        manager.node_registry._status_cache = {}

        result = await manager.rename_cluster_node("remote-1", "New")

        manager.node_registry.update.assert_called_once()
        self.assertEqual(result["name"], "New")
        self.assertEqual(result["name_sync"], "pending")
        self.assertFalse(result["online"])

    async def test_invalid_names_are_rejected_before_any_write(self):
        manager = Manager.__new__(Manager)
        for value in ("", "   ", "bad\nname", "x" * 81, None):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    await manager.rename_cluster_node("local", value)

    async def test_registry_alias_remains_authoritative_over_agent_status(self):
        with tempfile.TemporaryDirectory() as directory:
            client = httpx.AsyncClient()
            registry = NodeRegistry(Path(directory), client)
            remote = {
                "id": "remote-1", "name": "Controller alias", "enabled": True,
                "agent_url": "https://private", "agent_token": "secret",
            }
            registry.nodes = [remote]
            registry.request = AsyncMock(return_value={
                "name": "Stale worker name", "protocol_version": AGENT_PROTOCOL_VERSION,
                "docker_ready": True,
            })

            result = await registry.probe(remote, force=True)

            self.assertEqual(result["name"], "Controller alias")
            self.assertEqual(registry.request.await_args_list[-1].args, (
                "remote-1", "PATCH", "/api/agent/node",
            ))
            self.assertEqual(
                registry.request.await_args_list[-1].kwargs["json_body"],
                {"name": "Controller alias"},
            )
            await client.aclose()

    async def test_later_probe_reconciles_an_offline_rename_to_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            client = httpx.AsyncClient()
            registry = NodeRegistry(Path(directory), client)
            remote = {
                "id": "remote-1", "name": "New durable alias", "enabled": True,
                "agent_url": "https://private", "agent_token": "secret",
            }
            registry.nodes = [remote]
            registry.request = AsyncMock(side_effect=[
                {
                    "name": "Old worker name",
                    "protocol_version": AGENT_PROTOCOL_VERSION,
                    "docker_ready": True,
                },
                {"name": "New durable alias"},
            ])

            result = await registry.probe(remote, force=True)

            self.assertTrue(result["online"])
            self.assertEqual(result["name"], "New durable alias")
            registry.request.assert_awaited_with(
                "remote-1", "PATCH", "/api/agent/node",
                json_body={"name": "New durable alias"}, timeout=10,
            )
            await client.aclose()

    def test_versioned_rename_is_forwarded_but_agent_sync_stays_local(self):
        self.assertTrue(is_forwardable_path("/api/v1/nodes/remote-1"))
        self.assertFalse(is_forwardable_path("/api/agent/node"))


class NodeDashboardVisibilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_registry_visibility_persists_and_overrides_agent_status(self):
        with tempfile.TemporaryDirectory() as directory:
            client = httpx.AsyncClient()
            registry = NodeRegistry(Path(directory), client)
            remote = {
                "id": "remote-1", "name": "Inference PC", "enabled": True,
                "agent_url": "https://private", "agent_token": "secret",
            }
            registry.nodes = [remote]
            registry.update("remote-1", {"hidden_from_dashboard": True})
            registry.request = AsyncMock(return_value={
                "name": "Inference PC",
                "protocol_version": AGENT_PROTOCOL_VERSION,
                "docker_ready": True,
                "hidden_from_dashboard": False,
            })

            result = await registry.probe(remote, force=True)

            self.assertTrue(result["hidden_from_dashboard"])
            saved = json.loads((Path(directory) / "nodes.json").read_text())
            self.assertTrue(saved[0]["hidden_from_dashboard"])
            await client.aclose()

    async def test_local_visibility_is_persisted_without_changing_membership(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = Manager.__new__(Manager)
            manager.settings = {"cluster_node_name": "Coordinator"}
            manager.settings_path = Path(directory) / "settings.json"
            manager.lock = asyncio.Lock()

            result = await manager.set_cluster_node_dashboard_hidden(
                "local", True,
            )

            self.assertEqual(result, {
                "id": "local", "hidden_from_dashboard": True,
            })
            saved = json.loads(manager.settings_path.read_text())
            self.assertTrue(saved["cluster_node_hidden_from_dashboard"])

    async def test_remote_visibility_updates_only_controller_registry(self):
        manager = Manager.__new__(Manager)
        manager.node_registry = Mock()
        manager.node_registry.get.return_value = {
            "id": "remote-1", "name": "Inference PC",
        }
        manager.node_registry.update.return_value = {
            "id": "remote-1", "hidden_from_dashboard": True,
        }

        result = await manager.set_cluster_node_dashboard_hidden(
            "remote-1", True,
        )

        manager.node_registry.update.assert_called_once_with(
            "remote-1", {"hidden_from_dashboard": True},
        )
        self.assertEqual(result, {
            "id": "remote-1", "hidden_from_dashboard": True,
        })

    async def test_visibility_requires_a_boolean(self):
        manager = Manager.__new__(Manager)
        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            await manager.set_cluster_node_dashboard_hidden(
                "local", "true",
            )


class FakeServiceManager:
    def __init__(self):
        self.http = httpx.AsyncClient()
        self.nodes = [node("local", "Coordinator", local=True), node("remote-1", "Worker")]
        self.selected_cluster_nodes = AsyncMock(side_effect=self._selected)
        self.cluster_nodes = AsyncMock(return_value=self.nodes)
        self.create_deployment = AsyncMock(return_value={
            "id": "cluster-1",
            "status": "starting",
            "api_port": 8008,
            "node_ids": ["local", "remote-1"],
            "members": [
                {"node_id": "local", "node_name": "Coordinator", "container_name": "rank-0"},
                {"node_id": "remote-1", "node_name": "Worker", "container_name": "rank-1"},
            ],
        })
        self.deployment_action = AsyncMock(return_value={"ok": True, "errors": []})
        self.proxy_cluster_inference = AsyncMock(return_value={
            "choices": [{"message": {"content": "remote"}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 20},
        })
        self.list_containers = AsyncMock(return_value=[])
        self.remove_container = AsyncMock(return_value={"ok": True})

    async def _selected(self, node_ids):
        by_id = {item["id"]: item for item in self.nodes}
        return [by_id[node_id] for node_id in node_ids]

    @staticmethod
    def public_target_node(value):
        return Manager.public_target_node(value)


class SelectedDeploymentTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.manager = FakeServiceManager()
        self.service = SparkDeckService(self.manager, Path(self.temp.name))

    async def asyncTearDown(self):
        await self.manager.http.aclose()
        await self.service.close()
        self.temp.cleanup()

    async def test_selected_nodes_use_cluster_launch_and_persist_safe_configuration(self):
        result = await self.service.create_deployment({
            "model": "org/model",
            "alias": "selected-model",
            "runtime": "vllm",
            "node_ids": ["local", "remote-1"],
            "revision": "release-2",
            "settings": {"context_length": 8192, "image": "private/image"},
        })

        launch = self.manager.create_deployment.await_args.args[0]
        self.assertEqual(launch["node_ids"], ["local", "remote-1"])
        self.assertEqual(launch["deployment_mode"], "replicated")
        self.assertEqual(launch["extra_args"][-2:], ["--revision", "release-2"])
        self.assertEqual(result["node_ids"], ["local", "remote-1"])
        self.assertEqual([item["name"] for item in result["selected_nodes"]], ["Coordinator", "Worker"])

        stored = self.service.store.deployment("selected-model")
        self.assertEqual(stored["settings"]["node_ids"], ["local", "remote-1"])
        self.assertEqual(stored["settings"]["deployment_mode"], "replicated")
        self.assertEqual(stored["settings"]["manager_deployment_id"], "cluster-1")
        self.assertEqual(stored["settings"]["context_length"], 8192)
        self.assertNotIn("image", stored["settings"])

        self.manager.deployments = [self.manager.create_deployment.return_value]
        listed = (await self.service.deployments())[0]
        self.assertEqual(listed["node_ids"], ["local", "remote-1"])
        self.assertEqual([item["id"] for item in listed["selected_nodes"]], ["local", "remote-1"])

    async def test_remote_only_run_launches_and_uses_agent_inference_tunnel(self):
        self.manager.create_deployment.return_value = {
            "id": "remote-cluster", "status": "starting", "api_port": 8010,
            "node_ids": ["remote-1"],
            "members": [{
                "node_id": "remote-1", "node_name": "Worker",
                "container_name": "remote-rank-0",
            }],
        }
        created = await self.service.create_deployment({
            "model": "org/model", "alias": "remote-model",
            "runtime": "sglang", "node_id": "remote-1",
        })

        response = await self.service.proxy({
            "model": "remote-model", "messages": [], "stream": False,
        }, "chat/completions")

        self.assertEqual(created["node_ids"], ["remote-1"])
        self.assertEqual(created["selected_nodes"][0]["id"], "remote-1")
        self.assertEqual(response["model"], "remote-model")
        self.manager.proxy_cluster_inference.assert_awaited_once()
        self.assertEqual(
            self.manager.proxy_cluster_inference.await_args.args[:2],
            ("remote-cluster", "org/model"),
        )

    async def test_cluster_lifecycle_dispatches_to_owning_manager_deployment(self):
        await self.service.create_deployment({
            "model": "org/model", "alias": "clustered", "runtime": "vllm",
            "node_ids": ["local", "remote-1"],
        })

        stopped = await self.service.deployment_action(result_id := result_id_for(self.service, "clustered"), "stop")
        removed = await self.service.delete_deployment(result_id)

        self.assertEqual(stopped["status"], "stopped")
        self.assertEqual(removed, {"ok": True, "id": result_id})
        self.assertEqual(
            [call.args for call in self.manager.deployment_action.await_args_list],
            [("cluster-1", "stop"), ("cluster-1", "remove")],
        )


def result_id_for(service: SparkDeckService, alias: str) -> str:
    return service.store.deployment(alias)["id"]


class RemoteInferenceTunnelTests(unittest.IsolatedAsyncioTestCase):
    def manager(self):
        manager = Manager.__new__(Manager)
        manager.deployments = [{
            "id": "remote-cluster", "model": "org/model",
            "launch_settings": {},
            "members": [{
                "rank": 0, "node_id": "remote-1", "container_name": "rank-0",
            }],
        }]
        manager.node_registry = Mock()
        manager.node_registry.request = AsyncMock(return_value={"choices": [], "usage": {}})
        manager._acquire_inference_slot = AsyncMock(return_value="remote-cluster")
        manager._release_inference_slot = Mock()
        return manager

    async def test_remote_nonstream_uses_authenticated_agent_and_admission(self):
        manager = self.manager()
        body = {"model": "org/model", "messages": [], "stream": False}

        result = await manager.proxy_cluster_inference(
            "remote-cluster", "org/model", body, "chat/completions",
        )

        self.assertEqual(result, {"choices": [], "usage": {}})
        manager._acquire_inference_slot.assert_awaited_once()
        manager.node_registry.request.assert_awaited_once_with(
            "remote-1", "POST", "/api/agent/inference/chat/completions",
            json_body={
                **body,
                "_sparkdeck_container_name": "rank-0",
                "_sparkdeck_deployment_id": "remote-cluster",
            }, timeout=600,
        )
        manager._release_inference_slot.assert_called_once_with("remote-cluster")

    async def test_remote_health_uses_selected_primary_agent(self):
        manager = self.manager()
        manager.node_registry.request.return_value = {"ready": True}

        ready = await manager.cluster_deployment_health("remote-cluster", "org/model")

        self.assertTrue(ready)
        manager.node_registry.request.assert_awaited_once_with(
            "remote-1", "POST", "/api/agent/inference/health",
            json_body={
                "model": "org/model",
                "_sparkdeck_container_name": "rank-0",
                "_sparkdeck_deployment_id": "remote-cluster",
            }, timeout=10,
        )

    async def test_remote_stream_releases_admission_after_agent_stream(self):
        class Response:
            status_code = 200

            def __init__(self):
                self.closed = False

            async def aiter_lines(self):
                yield 'data: {"choices": []}'
                yield "data: [DONE]"

            async def aclose(self):
                self.closed = True

        manager = self.manager()
        response = Response()
        manager.node_registry.open_stream = AsyncMock(return_value=response)
        body = {"model": "org/model", "messages": [], "stream": True}

        stream = await manager.proxy_cluster_inference(
            "remote-cluster", "org/model", body, "chat/completions",
        )
        chunks = [chunk async for chunk in stream]

        self.assertEqual(chunks[-1], "data: [DONE]\n\n")
        self.assertTrue(response.closed)
        manager._release_inference_slot.assert_called_once_with("remote-cluster")


class NetworkInterfaceDiscoveryTests(unittest.TestCase):
    def test_tailscale_status_supplies_ipv4_when_linux_ip_is_unavailable(self):
        tailscale = Mock(stdout=json.dumps({
            "BackendState": "Running",
            "TailscaleIPs": ["100.101.102.103", "fd7a:115c:a1e0::1"],
        }))
        with patch(
            "manager.subprocess.run",
            side_effect=[FileNotFoundError("ip"), tailscale],
        ) as run:
            interfaces = Manager._network_interfaces()

        self.assertEqual(interfaces, [{
            "name": "tailscale0",
            "ipv4": ["100.101.102.103"],
            "up": True,
            "rdma": False,
        }])
        self.assertEqual(run.call_count, 2)

    def test_existing_linux_tailscale_interface_avoids_cli_fallback(self):
        linux = Mock(stdout=json.dumps([{
            "ifname": "tailscale0",
            "operstate": "UNKNOWN",
            "addr_info": [{"family": "inet", "local": "100.64.0.9"}],
        }]))
        with patch("manager.subprocess.run", return_value=linux) as run:
            interfaces = Manager._network_interfaces()

        self.assertEqual(interfaces[0]["ipv4"], ["100.64.0.9"])
        run.assert_called_once()


class RemovedOllamaTests(unittest.IsolatedAsyncioTestCase):
    async def test_proxy_models_ignores_legacy_ollama_state(self):
        manager = Manager.__new__(Manager)
        manager.get_state = AsyncMock(return_value={
            "containers": [], "ollama": {"models": [{"name": "legacy/model"}]},
            "unsloth": {}, "sparkrun_targets": {},
        })

        result = await manager.proxy_models()

        self.assertEqual(result["data"], [])

    async def test_removed_ollama_prefix_is_not_routed_to_ollama(self):
        manager = Manager.__new__(Manager)
        manager._unsloth_loaded_model = AsyncMock(return_value=None)
        manager._vllm_chat = AsyncMock(return_value={"ok": True})

        await manager.proxy_chat_completions({"model": "CLOUD legacy", "messages": []})

        manager._vllm_chat.assert_awaited_once()

    def test_legacy_ollama_setting_is_discarded(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = Manager.__new__(Manager)
            manager.settings_path = Path(directory) / "settings.json"
            manager.settings_path.write_text(json.dumps({"ollama_base_url": "http://localhost:11434"}))

            settings = manager._load_settings()

        self.assertNotIn("ollama_base_url", settings)

    def test_legacy_ollama_deployment_is_retained_but_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = Manager.__new__(Manager)
            manager.deployments_path = Path(directory) / "deployments.json"
            manager.deployments_path.write_text(json.dumps([
                {"id": "legacy", "engine": "ollama", "status": "running"},
            ]))

            deployments = manager._load_deployments()

        self.assertEqual(deployments[0]["status"], "error")
        self.assertIn("unsupported persisted runtime", deployments[0]["error"])

    async def test_ollama_container_launch_is_rejected(self):
        manager = Manager.__new__(Manager)
        with self.assertRaisesRegex(ValueError, "engine must be vllm or sglang"):
            await manager.create_container("legacy/model", engine="ollama")


if __name__ == "__main__":
    unittest.main()
