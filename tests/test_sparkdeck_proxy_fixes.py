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
    async def test_unmanaged_discovered_container_remove_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = FakeManager()
            manager.list_containers.return_value = [{
                "name": "user-container", "model": "org/model", "engine": "vllm",
                "managed": False, "status": "running", "port": 8000,
            }]
            service = SparkDeckService(manager, Path(directory))

            with self.assertRaisesRegex(ValueError, "unmanaged discovered containers"):
                await service.delete_deployment("container:user-container")

            manager.remove_container.assert_not_awaited()
            await manager.http.aclose()
            await service.close()

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
