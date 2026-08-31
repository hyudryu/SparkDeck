import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import httpx

from manager import ClientAbort, Manager
from sparkdeck.models import Deployment, DeploymentKind, ModelIdentity, RuntimeKind
from sparkdeck.service import SparkDeckService


class FakeManager:
    def __init__(self):
        self.http = httpx.AsyncClient()
        self.list_containers = AsyncMock(return_value=[])
        self.remove_container = AsyncMock(return_value={"ok": True})
        self.start_container = AsyncMock(return_value={"ok": True})
        self.stop_container = AsyncMock(return_value={"ok": True})
        self.get_cluster_member_logs = AsyncMock(return_value="container output")
        self.get_logs = AsyncMock(return_value="container output")
        self._vllm_chat = AsyncMock()
        self._vllm_completions = AsyncMock()
        self.proxy_chat_completions = AsyncMock()
        self.proxy_completions = AsyncMock()
        self._unsloth_loaded_model = AsyncMock(return_value=None)


class ManagedIdentityTests(unittest.IsolatedAsyncioTestCase):
    async def test_explicitly_stopped_registered_deployment_cannot_auto_wake(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = FakeManager()
            service = SparkDeckService(manager, Path(directory))
            service.store.add_deployment(Deployment(
                id="stopped", alias="sleeping-model", runtime=RuntimeKind.VLLM,
                kind=DeploymentKind.MANAGED, model=ModelIdentity("org/model"),
                container_name="sleeping-container", desired_state="stopped",
            ))

            with self.assertRaisesRegex(RuntimeError, "deployment is stopped"):
                await service.proxy(
                    {"model": "sleeping-model", "messages": [], "stream": False},
                    "chat/completions",
                )

            manager._vllm_chat.assert_not_awaited()
            await manager.http.aclose()
            await service.close()

    async def test_registered_alias_passes_exact_container_and_deployment_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = FakeManager()
            service = SparkDeckService(manager, Path(directory))
            manager._vllm_chat.return_value = {"choices": [], "usage": {}}
            service.store.add_deployment(Deployment(
                id="revision-b", alias="model-b", runtime=RuntimeKind.VLLM,
                kind=DeploymentKind.MANAGED, model=ModelIdentity("org/shared", "rev-b"),
                container_name="sparkdeck-model-b",
            ))

            await service.proxy(
                {"model": "model-b", "messages": [], "stream": False},
                "chat/completions",
            )

            self.assertEqual(manager._vllm_chat.await_args.kwargs, {
                "container_name": "sparkdeck-model-b",
                "deployment_id": "revision-b",
            })
            await manager.http.aclose()
            await service.close()

    async def test_manager_resolves_duplicate_repository_by_exact_identity(self):
        manager = Manager.__new__(Manager)
        manager._capacity_redeploying_models = set()
        manager.list_containers = AsyncMock(return_value=[
            {"name": "model-a", "deployment_id": "revision-a", "model": "org/shared",
             "served_models": [], "status": "running", "port": 8001},
            {"name": "model-b", "deployment_id": "revision-b", "model": "org/shared",
             "served_models": [], "status": "running", "port": 8002},
        ])
        manager._check_ready = AsyncMock(return_value=True)
        manager._mark_active = Mock()

        selected = await manager._resolve_vllm_target(
            "org/shared", container_name="model-b", deployment_id="revision-b",
        )

        self.assertEqual(selected["port"], 8002)
        manager._mark_active.assert_called_once_with("model-b")


class DeletionAndCancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_unmanaged_discovered_container_has_lifecycle_logs_and_remove(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = FakeManager()
            manager.list_containers.return_value = [{
                "name": "user-container", "model": "org/model", "engine": "vllm",
                "managed": False, "status": "running", "port": 8000,
            }]
            service = SparkDeckService(manager, Path(directory))

            stopped = await service.deployment_action(
                "container:user-container", "stop",
            )
            self.assertEqual(stopped["status"], "stopped")
            manager.stop_container.assert_awaited_once_with(
                "user-container", explicit=True, managed=False,
            )

            started = await service.deployment_action(
                "container:user-container", "start",
            )
            self.assertEqual(started["status"], "running")
            manager.start_container.assert_awaited_once_with(
                "user-container", explicit=True, managed=False,
            )

            logs = await service.deployment_logs("container:user-container")
            self.assertEqual(logs, {"logs": "container output"})
            manager.get_logs.assert_awaited_once_with("user-container", 300)
            manager.get_cluster_member_logs.assert_not_awaited()

            result = await service.delete_deployment("container:user-container")
            self.assertTrue(result["ok"])
            manager.remove_container.assert_awaited_once_with("user-container")

            await manager.http.aclose()
            await service.close()

    async def test_unmanaged_created_container_is_stopped_and_startable(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = FakeManager()
            service = SparkDeckService(manager, Path(directory))

            deployment = service._discovered_deployment({
                "name": "created-container", "model": "org/model",
                "managed": False, "status": "created", "load_settings": {},
                "started_at": "2026-08-30T12:34:56Z",
            }, "vllm", "org/model")

            self.assertEqual(deployment["status"], "stopped")
            self.assertTrue(deployment["controllable"])
            self.assertEqual(deployment["required_node_count"], 1)
            self.assertEqual(
                deployment["last_deployed_at"], "2026-08-30T12:34:56Z",
            )
            await manager.http.aclose()
            await service.close()

    async def test_discovered_tp_runtime_promotes_to_selected_managed_nodes(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = FakeManager()
            container = {
                "name": "external-tp2", "model": "served-name",
                "engine": "vllm", "managed": False, "status": "exited",
                "image": "example/vllm:latest", "port": 8214,
                "load_settings": {
                    "model": "/cache/models/org--model/snapshots/rev",
                    "tensor_parallel_size": 2,
                    "command_flags": "--tensor-parallel-size 2 --max-model-len 8192",
                    "environment": {"NCCL_DEBUG": "WARN"},
                },
            }
            manager.list_containers.return_value = [container]
            manager._recovered_deployment_launch_settings = Mock(return_value={
                "extra_args": ["--tensor-parallel-size", "2"], "port": 8214,
            })
            manager.create_deployment = AsyncMock(return_value={
                "id": "cluster-1", "name": "served-name", "status": "starting",
                "node_ids": ["local", "worker-1"], "api_port": 8214,
                "members": [{"container_name": "cluster-1-r0"}],
                "launch_settings": {
                    "deployment_mode": "sharded",
                    "node_ids": ["local", "worker-1"],
                    "extra_args": ["--tensor-parallel-size", "2"],
                },
            })
            manager._save_deployments = Mock()
            service = SparkDeckService(manager, Path(directory))

            listed = service._discovered_deployment(
                container, "vllm", "served-name",
            )
            self.assertEqual(listed["deployment_mode"], "sharded")
            self.assertEqual(listed["required_node_count"], 2)

            result = await service.deployment_action(
                "container:external-tp2", "start", ["worker-1", "local"],
            )

            launch = manager.create_deployment.await_args.args[0]
            self.assertEqual(launch["node_ids"], ["worker-1", "local"])
            self.assertNotIn("port", launch)
            self.assertEqual(launch["deployment_mode"], "sharded")
            self.assertEqual(
                launch["model"], "/cache/models/org--model/snapshots/rev",
            )
            self.assertEqual(result["node_ids"], ["local", "worker-1"])
            stored = service.store.deployment(result["id"], include_private=True)
            self.assertEqual(
                stored["model"]["repository"],
                "/cache/models/org--model/snapshots/rev",
            )
            self.assertEqual(
                stored["settings"]["source_container_name"], "external-tp2",
            )
            manager.model_cache_inventory = AsyncMock(return_value=[
                {
                    "id": node_id,
                    "models": [{
                        "model_id": "/cache/models/org--model/snapshots/rev",
                        "revisions": ["main"],
                    }],
                }
                for node_id in ("local", "worker-1")
            ])
            await service._validate_start_selection(
                stored, ["local", "worker-1"], launch,
            )
            await manager.http.aclose()
            await service.close()

    async def test_discovered_tp_runtime_rejects_wrong_node_count(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = FakeManager()
            manager.list_containers.return_value = [{
                "name": "external-tp2", "model": "org/model", "engine": "vllm",
                "managed": False, "status": "exited",
                "load_settings": {"tensor_parallel_size": 2},
            }]
            service = SparkDeckService(manager, Path(directory))

            with self.assertRaisesRegex(
                ValueError, "tensor parallel size 2 requires exactly 2 node",
            ):
                await service.deployment_action(
                    "container:external-tp2", "start", ["local"],
                )

            await manager.http.aclose()
            await service.close()

    async def test_explicit_promotion_converts_single_node_discovered_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = FakeManager()
            container = {
                "name": "external-single", "model": "org/model",
                "engine": "vllm", "managed": False, "status": "exited",
                "image": "example/vllm:latest", "port": 8214,
                "load_settings": {
                    "model": "org/model", "tensor_parallel_size": 1,
                    "extra_args": [],
                },
            }
            manager.list_containers.return_value = [container]
            manager._recovered_deployment_launch_settings = Mock(return_value={
                "port": 8214,
            })
            manager.create_deployment = AsyncMock(return_value={
                "id": "cluster-single", "name": "org/model", "status": "starting",
                "node_ids": ["local"], "api_port": 8215,
                "members": [{"container_name": "cluster-single-r0"}],
                "launch_settings": {"deployment_mode": "single", "node_ids": ["local"]},
            })
            manager._save_deployments = Mock()
            service = SparkDeckService(manager, Path(directory))

            result = await service.deployment_action(
                "container:external-single", "start", ["local"], promote=True,
            )

            manager.create_deployment.assert_awaited_once()
            self.assertNotIn("port", manager.create_deployment.await_args.args[0])
            self.assertEqual(
                result["id"],
                manager.create_deployment.await_args.args[0]["sparkdeck_record_id"],
            )
            self.assertEqual(result["kind"], "managed")
            self.assertEqual(result["node_ids"], ["local"])
            await manager.http.aclose()
            await service.close()

    async def test_promotion_rolls_back_manager_record_when_adoption_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = FakeManager()
            container = {
                "name": "external-single", "model": "org/model",
                "engine": "vllm", "managed": False, "status": "exited",
                "image": "example/vllm:latest", "port": 8214,
                "load_settings": {"model": "org/model", "tensor_parallel_size": 1},
            }
            manager.list_containers.return_value = [container]
            manager._recovered_deployment_launch_settings = Mock(return_value={})
            replacement = {
                "id": "cluster-single", "name": "org/model", "status": "starting",
                "node_ids": ["local"], "api_port": 8215,
                "members": [{"container_name": "cluster-single-r0"}],
                "launch_settings": {"deployment_mode": "single", "node_ids": ["local"]},
            }
            manager.create_deployment = AsyncMock(return_value=replacement)
            manager.deployment_action = AsyncMock(return_value={"ok": True, "errors": []})
            service = SparkDeckService(manager, Path(directory))
            service._adopt_manager_replacement = Mock(side_effect=RuntimeError("SQLite failed"))

            with self.assertRaisesRegex(RuntimeError, "SQLite failed"):
                await service.deployment_action(
                    "container:external-single", "start", ["local"], promote=True,
                )

            manager.deployment_action.assert_awaited_once_with(
                "cluster-single", "remove",
            )
            await manager.http.aclose()
            await service.close()

    async def test_promotion_rolls_back_persisted_manager_launch_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = FakeManager()
            container = {
                "name": "external-single", "model": "org/model",
                "engine": "vllm", "managed": False, "status": "exited",
                "image": "example/vllm:latest", "port": 8214,
                "load_settings": {"model": "org/model", "tensor_parallel_size": 1},
            }
            manager.list_containers.return_value = [container]
            manager._recovered_deployment_launch_settings = Mock(return_value={})
            manager.deployments = []

            async def fail_after_persisting(body):
                manager.deployments.append({
                    "id": "failed-cluster",
                    "sparkdeck_record_id": body["sparkdeck_record_id"],
                })
                raise RuntimeError("member launch failed")

            manager.create_deployment = AsyncMock(side_effect=fail_after_persisting)
            manager.deployment_action = AsyncMock(return_value={"ok": True, "errors": []})
            service = SparkDeckService(manager, Path(directory))

            with self.assertRaisesRegex(RuntimeError, "member launch failed"):
                await service.deployment_action(
                    "container:external-single", "start", ["local"], promote=True,
                )

            manager.deployment_action.assert_awaited_once_with(
                "failed-cluster", "remove",
            )
            await manager.http.aclose()
            await service.close()

    async def test_discovered_local_model_rejects_remote_promotion(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = FakeManager()
            manager.list_containers.return_value = [{
                "name": "external-local", "model": "served-name",
                "engine": "vllm", "managed": False, "status": "exited",
                "load_settings": {
                    "model": "C:\\models\\actual-model",
                    "tensor_parallel_size": 2,
                },
            }]
            manager._resolve_local_path = Mock(return_value=Path("C:/models/actual-model"))
            manager._recovered_deployment_launch_settings = Mock(return_value={})
            manager.create_deployment = AsyncMock()
            service = SparkDeckService(manager, Path(directory))

            with self.assertRaisesRegex(ValueError, "controller-local model paths"):
                await service.deployment_action(
                    "container:external-local", "start", ["local", "worker-1"],
                )

            manager.create_deployment.assert_not_awaited()
            await manager.http.aclose()
            await service.close()

    async def test_credential_bearing_discovered_runtime_rejects_promotion(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = FakeManager()
            manager.list_containers.return_value = [{
                "name": "external-protected", "model": "org/model",
                "engine": "vllm", "managed": False, "status": "exited",
                "load_settings": {
                    "editable": False,
                    "tensor_parallel_size": 2,
                    "extra_args": ["--api-key", "do-not-drop"],
                },
            }]
            manager._recovered_deployment_launch_settings = Mock()
            manager.create_deployment = AsyncMock()
            service = SparkDeckService(manager, Path(directory))

            listed = service._discovered_deployment(
                manager.list_containers.return_value[0], "vllm", "org/model",
            )
            self.assertFalse(listed["promotable"])

            with self.assertRaisesRegex(ValueError, "cannot be promoted safely"):
                await service.deployment_action(
                    "container:external-protected", "start", ["local", "worker-1"],
                )

            manager._recovered_deployment_launch_settings.assert_not_called()
            manager.create_deployment.assert_not_awaited()
            await manager.http.aclose()
            await service.close()

    async def test_promoted_source_container_does_not_mask_missing_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = FakeManager()
            manager.deployments = []
            manager.list_containers.return_value = [{
                "name": "external-source", "model": "served-name",
                "engine": "vllm", "managed": False, "status": "exited",
                "port": None,
            }]
            service = SparkDeckService(manager, Path(directory))
            service.store.add_deployment(Deployment(
                id="promoted", alias="served-name", runtime=RuntimeKind.VLLM,
                kind=DeploymentKind.MANAGED, model=ModelIdentity("served-name"),
                container_name="replacement-r0",
                settings={
                    "manager_deployment_id": "missing-cluster",
                    "source_container_name": "external-source",
                },
            ))

            listed = await service.deployments()

            self.assertEqual(len(listed), 1)
            self.assertEqual(listed[0]["id"], "promoted")
            self.assertEqual(listed[0]["status"], "missing")
            await manager.http.aclose()
            await service.close()

    async def test_promoted_source_container_does_not_overwrite_replacement_port(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = FakeManager()
            manager.deployments = [{
                "id": "cluster-1", "status": "running", "api_port": 9000,
                "created_at": 1_787_918_400,
                "launch_controls": {
                    "context_window": 131072, "max_concurrency": 32,
                },
                "node_ids": ["local"], "launch_settings": {
                    "deployment_mode": "single", "node_ids": ["local"],
                },
                "members": [{
                    "node_id": "local", "container_name": "replacement-r0",
                }],
            }]
            manager.list_containers.return_value = [{
                "name": "external-source", "model": "served-name",
                "engine": "vllm", "managed": False, "status": "exited",
                "port": None,
            }]
            service = SparkDeckService(manager, Path(directory))
            service.store.add_deployment(Deployment(
                id="promoted", alias="served-name", runtime=RuntimeKind.VLLM,
                kind=DeploymentKind.MANAGED, model=ModelIdentity("served-name"),
                container_name="replacement-r0",
                settings={
                    "manager_deployment_id": "cluster-1",
                    "source_container_name": "external-source",
                },
            ))

            listed = await service.deployments()

            self.assertEqual(len(listed), 1)
            self.assertEqual(listed[0]["status"], "running")
            self.assertEqual(listed[0]["port"], 9000)
            self.assertEqual(listed[0]["settings"]["context_length"], 131072)
            self.assertEqual(listed[0]["settings"]["max_concurrency"], 32)
            self.assertEqual(
                listed[0]["last_deployed_at"], "2026-08-28T12:00:00+00:00",
            )
            await manager.http.aclose()
            await service.close()

    async def test_registered_standalone_uses_container_start_for_recency(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = FakeManager()
            manager.list_containers.return_value = [{
                "name": "legacy-managed", "model": "org/model",
                "engine": "vllm", "managed": True, "status": "exited",
                "port": 8000, "started_at": "2026-08-30T12:34:56Z",
            }]
            service = SparkDeckService(manager, Path(directory))
            service.store.add_deployment(Deployment(
                id="legacy", alias="Legacy", runtime=RuntimeKind.VLLM,
                kind=DeploymentKind.MANAGED, model=ModelIdentity("org/model"),
                container_name="legacy-managed",
            ))

            listed = await service.deployments()

            self.assertEqual(len(listed), 1)
            self.assertEqual(
                listed[0]["last_deployed_at"], "2026-08-30T12:34:56Z",
            )
            await manager.http.aclose()
            await service.close()

    async def test_failed_first_launch_has_no_deployment_recency(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = FakeManager()
            manager.deployments = [{
                "id": "failed-cluster", "status": "error",
                "created_at": 1_787_918_400, "sparkdeck_record_id": "failed",
                "node_ids": ["local"], "members": [],
                "launch_settings": {"node_ids": ["local"]},
            }]
            service = SparkDeckService(manager, Path(directory))
            service.store.add_deployment(Deployment(
                id="failed", alias="Failed", runtime=RuntimeKind.VLLM,
                kind=DeploymentKind.MANAGED, model=ModelIdentity("org/model"),
                settings={"manager_deployment_id": "failed-cluster"},
            ))

            listed = await service.deployments()

            self.assertEqual(len(listed), 1)
            self.assertNotIn("last_deployed_at", listed[0])
            await manager.http.aclose()
            await service.close()


class ExternalContainerLifecycleTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.manager = Manager.__new__(Manager)
        self.container = Mock()
        self.container.status = "running"
        self.manager.client = Mock()
        self.manager.client.containers.get.return_value = self.container
        self.manager._try_fit_new_model = Mock()
        self.manager._explicitly_stopped_containers = set()

    async def test_external_start_preserves_restart_policy_and_skips_eviction(self):
        self.container.status = "exited"

        await self.manager.start_container(
            "external-container", explicit=True, managed=False,
        )

        self.manager._try_fit_new_model.assert_not_called()
        self.container.update.assert_not_called()
        self.container.start.assert_called_once_with()

    async def test_external_stop_preserves_restart_policy(self):
        def stopped(*, timeout):
            self.container.status = "exited"

        self.container.stop.side_effect = stopped

        await self.manager.stop_container(
            "external-container", explicit=True, managed=False,
        )

        self.container.update.assert_not_called()
        self.container.stop.assert_called_once_with(timeout=10)

    async def test_missing_container_is_still_deleted_from_local_store(self):
        class NotFound(Exception):
            pass

        with tempfile.TemporaryDirectory() as directory:
            manager = FakeManager()
            manager.remove_container.side_effect = NotFound("gone")
            service = SparkDeckService(manager, Path(directory))
            service.store.add_deployment(Deployment(
                id="dep-1", alias="gone", runtime=RuntimeKind.VLLM,
                kind=DeploymentKind.MANAGED, model=ModelIdentity("org/model"),
                container_name="missing-container",
            ))

            result = await service.delete_deployment("dep-1")

            self.assertTrue(result["ok"])
            self.assertIsNone(service.store.deployment("dep-1"))
            await manager.http.aclose()
            await service.close()

    async def test_external_nonstream_request_is_cancelled_on_disconnect(self):
        closed = asyncio.Event()
        blocker = asyncio.Event()

        async def post(*_args, **_kwargs):
            try:
                await blocker.wait()
            finally:
                closed.set()

        with tempfile.TemporaryDirectory() as directory:
            manager = FakeManager()
            await manager.http.aclose()
            manager.http = Mock(post=post)
            manager._await_or_cancel = Manager._await_or_cancel
            service = SparkDeckService(manager, Path(directory))
            service.store.add_deployment(Deployment(
                id="external", alias="external", runtime=RuntimeKind.LLAMA_CPP,
                kind=DeploymentKind.EXTERNAL, model=ModelIdentity("local/model"),
                base_url_set=True,
            ), "http://127.0.0.1:8080")
            cancel = asyncio.Event()
            request = asyncio.create_task(service.proxy(
                {"model": "external", "messages": [], "stream": False},
                "chat/completions", cancel,
            ))
            await asyncio.sleep(0)
            cancel.set()

            with self.assertRaises(ClientAbort):
                await request
            self.assertTrue(closed.is_set())
            await service.close()

    async def test_external_stream_status_is_raised_before_iteration(self):
        def unavailable(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, text="backend unavailable")

        with tempfile.TemporaryDirectory() as directory:
            manager = FakeManager()
            await manager.http.aclose()
            manager.http = httpx.AsyncClient(transport=httpx.MockTransport(unavailable))
            service = SparkDeckService(manager, Path(directory))
            service.store.add_deployment(Deployment(
                id="external-stream", alias="external-stream",
                runtime=RuntimeKind.LLAMA_CPP, kind=DeploymentKind.EXTERNAL,
                model=ModelIdentity("local/model"), base_url_set=True,
            ), "http://127.0.0.1:8080")

            with self.assertRaises(httpx.HTTPStatusError) as raised:
                await service.proxy(
                    {"model": "external-stream", "messages": [], "stream": True},
                    "chat/completions",
                )

            self.assertEqual(raised.exception.response.status_code, 503)
            await manager.http.aclose()
            await service.close()


class NativeLlamaRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_adopted_native_llama_is_listed_and_uses_manager_router(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = FakeManager()
            manager._unsloth_loaded_model.return_value = "org/native-gguf"
            manager.proxy_chat_completions.return_value = {
                "model": "org/native-gguf", "choices": [], "usage": {},
            }
            service = SparkDeckService(manager, Path(directory))

            models = await service.models()
            response = await service.proxy(
                {"model": "org/native-gguf", "messages": [], "stream": False},
                "chat/completions",
            )

            native = next(item for item in models["data"] if item["id"] == "org/native-gguf")
            self.assertEqual(native["runtime"], "llama.cpp")
            self.assertEqual(native["owned_by"], "llama.cpp")
            manager.proxy_chat_completions.assert_awaited_once()
            manager._vllm_chat.assert_not_awaited()
            self.assertEqual(response["model"], "org/native-gguf")
            await manager.http.aclose()
            await service.close()


class ManagedStreamingAliasTests(unittest.IsolatedAsyncioTestCase):
    async def test_managed_sse_model_is_rewritten_to_requested_alias(self):
        async def chunks():
            yield b'data: {"model":"org/shared","choices":[{"delta":{"content":"x"}}]}\n\n'
            yield b'data: [DONE]\n\n'

        with tempfile.TemporaryDirectory() as directory:
            manager = FakeManager()
            manager._vllm_chat.return_value = chunks()
            service = SparkDeckService(manager, Path(directory))
            service.store.add_deployment(Deployment(
                id="dep-stream", alias="friendly", runtime=RuntimeKind.SGLANG,
                kind=DeploymentKind.MANAGED, model=ModelIdentity("org/shared"),
                container_name="exact-container",
            ))

            stream = await service.proxy(
                {"model": "friendly", "messages": [], "stream": True},
                "chat/completions",
            )
            body = b"".join([chunk async for chunk in stream])

            self.assertIn(b'"model":"friendly"', body)
            self.assertNotIn(b'"model":"org/shared"', body)
            await manager.http.aclose()
            await service.close()


if __name__ == "__main__":
    unittest.main()
