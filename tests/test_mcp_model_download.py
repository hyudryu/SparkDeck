import asyncio
import unittest
from unittest.mock import AsyncMock, Mock, patch

import httpx
from mcp.server.mcpserver.exceptions import UnexpectedToolError

from mcp_server import ControllerClient, ControllerError, build_server
from manager import Manager
from sparkdeck.virtual_nas import VirtualNAS


with patch("docker.from_env", return_value=Mock()):
    import server as application


class MCPModelDownloadTests(unittest.IsolatedAsyncioTestCase):
    async def test_download_uses_app_queue_and_job_progress_is_visible_to_storage(self):
        jobs = []

        async def queue(model_id, revision, node_ids, download_node_id):
            jobs.append({
                "id": "download-1", "kind": "download", "status": "queued",
                "model_id": model_id, "target_node_id": node_ids[0],
                "revision": "a" * 40, "requested_revision": revision,
                "bytes_transferred": 0, "bytes_total": 22_000_000_000,
            })
            return {"workflow_id": "workflow-1", "job_ids": ["download-1"], "jobs": jobs,
                    "plan": {"action": "download", "resolved_revision": "a" * 40}}

        queue_mock = AsyncMock(side_effect=queue)
        client = ControllerClient("http://test", transport=httpx.ASGITransport(app=application.app))
        mcp = build_server(client)
        with (
            patch.object(application.manager, "virtual_nas_enabled", return_value=True),
            patch.object(application.manager, "selected_cluster_nodes", AsyncMock(return_value=[{"id": "ws1"}])),
            patch.object(application.manager, "queue_recipe_model_preparation", queue_mock),
            patch.object(application.manager, "virtual_nas_transfer_preflight", AsyncMock(return_value={"sources": []})),
            patch.object(application.manager, "virtual_nas_inventory", AsyncMock(side_effect=lambda: {
                "enabled": True, "nodes": [{"id": "ws1", "models": []}], "jobs": jobs,
            })),
        ):
            queued = await mcp.call_tool("download_huggingface_model", {
                "model_id": "nvidia/Qwen3.8-27B-NVFP4", "node_id": "ws1", "revision": "release-1",
            })
            queue_mock.assert_awaited_once_with("nvidia/Qwen3.8-27B-NVFP4", "release-1", ["ws1"], "ws1")
            job_id = queued.structured_content["job_ids"][0]
            self.assertEqual(queued.structured_content["jobs"][0]["status"], "queued")
            jobs[0].update(status="running", bytes_transferred=11_000_000_000, progress=0.5,
                           bytes_per_second=100_000_000, phase="downloading")
            status = await mcp.call_tool("get_storage_transfer", {"job_id": job_id})
            self.assertEqual(status.structured_content["bytes_transferred"], 11_000_000_000)
            self.assertEqual(status.structured_content["progress"], 0.5)
            self.assertEqual((await client.storage())["jobs"][0]["id"], job_id)
            jobs[0].update(status="completed", bytes_transferred=22_000_000_000, progress=1.0)
            completed = await mcp.call_tool("get_storage_transfer", {"job_id": job_id})
            self.assertEqual(completed.structured_content["status"], "completed")

    async def test_already_ready_preserves_no_jobs_and_main_default(self):
        prepared = {"workflow_id": None, "job_ids": [], "jobs": [], "plan": {"action": "ready"}}
        client = ControllerClient()
        with (
            patch.object(client, "_request", AsyncMock(return_value={"sources": []})),
            patch.object(client, "pull_storage_weights", AsyncMock(return_value=prepared)) as pull,
        ):
            result = await client.download_huggingface_model("org/model", "local")
        pull.assert_awaited_once_with("org/model", ["local"], revision="main", download_node_id="local")
        self.assertEqual(result, prepared)

    async def test_disabled_storage_uses_app_guard_and_does_not_queue(self):
        mcp = build_server(ControllerClient("http://test", transport=httpx.ASGITransport(app=application.app)))
        queue = AsyncMock()
        with (
            patch.object(application.manager, "virtual_nas_enabled", return_value=False),
            patch.object(application.manager, "queue_recipe_model_preparation", queue),
            patch.object(application.manager, "virtual_nas_transfer_preflight", AsyncMock(return_value={"sources": []})),
        ):
            with self.assertRaises(UnexpectedToolError) as raised:
                await mcp.call_tool("download_huggingface_model", {"model_id": "org/model", "node_id": "ws1"})
        self.assertIsInstance(raised.exception.__cause__, ControllerError)
        self.assertIn("Enable it in Storage", str(raised.exception.__cause__))
        queue.assert_not_awaited()

    async def test_cached_peer_transfer_and_active_retry_use_pinned_workflow(self):
        model, sha = "org/model", "a" * 40
        jobs = []
        preflight = {
            "enabled": True, "model_id": model, "revision": sha, "resolved_revision": sha,
            "sources": [{"node_id": "spark", "size_bytes": 100}],
            "targets": [
                {"node_id": "spark", "has_required_weights": True},
                {"node_id": "ws1", "has_required_weights": False,
                 "has_model_cache": False, "free_bytes": 100_000_000_000,
                 "download_eligible": False, "download_reason": "Agent lacks Hub download support"},
            ],
        }
        manager = Manager.__new__(Manager)
        manager.virtual_nas = VirtualNAS.__new__(VirtualNAS)
        nas = manager.virtual_nas
        nas.jobs = jobs
        nas._enabled_provider = lambda: True
        nas._queue_lock = asyncio.Lock()
        nas._validate_online_node = AsyncMock(return_value={"online": True})
        nas._node_storage = AsyncMock(side_effect=lambda node: {
            "models": [{"model_id": model, "revision": sha, "revisions": [sha], "size_bytes": 100}]
            if node == "spark" else [], "free_size": 100_000_000_000,
        })
        nas._save = Mock()
        nas.start = Mock()
        nas._wake = asyncio.Event()
        manager.virtual_nas.queue_download_and_transfer = AsyncMock()
        manager.virtual_nas_transfer_preflight = AsyncMock(return_value=preflight)
        manager._public_virtual_nas_job = lambda job: dict(job)

        transfer = AsyncMock(wraps=manager.queue_virtual_nas_transfer)
        preparation = AsyncMock()
        mcp = build_server(ControllerClient("http://test", transport=httpx.ASGITransport(app=application.app)))
        with (
            patch.object(application.manager, "virtual_nas_enabled", return_value=True),
            patch.object(application.manager, "selected_cluster_nodes", AsyncMock()),
            patch.object(application.manager, "virtual_nas_transfer_preflight", manager.virtual_nas_transfer_preflight),
            patch.object(application.manager, "queue_recipe_model_preparation", preparation),
            patch.object(application.manager, "queue_virtual_nas_transfer", transfer),
            patch.object(application.manager, "virtual_nas_inventory", AsyncMock(side_effect=lambda: {"jobs": jobs})),
        ):
            result = await mcp.call_tool("download_huggingface_model", {"model_id": model, "node_id": "ws1"})
            self.assertNotIn("plan", result.structured_content)
            self.assertEqual(transfer.await_args.args[:4],
                             (model, "spark", ["ws1"], sha))
            # The original peer disappearing must not change the active workflow's selected set.
            preflight["sources"] = [{"node_id": "new-peer", "size_bytes": 50}]
            # Global branch preflight reports SHA-requested work as a conflict,
            # so retry cannot rely on active_job_id being populated.
            preflight["targets"][1]["has_preparation_conflict"] = True
            retry = await mcp.call_tool("download_huggingface_model", {"model_id": model, "node_id": "ws1"})
            self.assertEqual(retry.structured_content["job_ids"], result.structured_content["job_ids"])
            self.assertNotIn("plan", retry.structured_content)
            self.assertEqual(transfer.await_count, 1)
            preparation.assert_not_awaited()
            manager.virtual_nas.queue_download_and_transfer.assert_not_awaited()
            # Simulate a source disappearing after discovery but before the
            # guarded transfer queues. Real VirtualNAS must reject it, with no
            # preparation path that might populate the unrequested peer.
            jobs.clear()
            preflight["sources"] = [{"node_id": "spark", "size_bytes": 100}]
            nas._node_storage.side_effect = lambda node: {"models": [], "free_size": 100_000_000_000}
            with self.assertRaises(UnexpectedToolError) as raised:
                await mcp.call_tool("download_huggingface_model", {"model_id": model, "node_id": "ws1"})
            self.assertIn("cached source model revision not found", str(raised.exception.__cause__))
            self.assertEqual(jobs, [])
            preparation.assert_not_awaited()
            nas.queue_download_and_transfer.assert_not_awaited()

    async def test_invalid_repository_revision_or_empty_node_never_posts(self):
        client = ControllerClient()
        with patch.object(client, "_request", AsyncMock()) as request:
            for values in [("../model", "ws1", "main"), ("org/model", "", "main"),
                           ("org/model", "ws1", "../bad")]:
                with self.subTest(values=values), self.assertRaises(ControllerError):
                    await client.download_huggingface_model(values[0], values[1], revision=values[2])
        request.assert_not_awaited()

    async def test_partial_destination_download_never_selects_cached_peer(self):
        client = ControllerClient()
        preflight = {"resolved_revision": "a" * 40,
                     "sources": [{"node_id": "spark"}],
                     "targets": [{"node_id": "ws1", "has_model_cache": True}]}
        with (
            patch.object(client, "_request", AsyncMock(return_value=preflight)),
            patch.object(client, "storage", AsyncMock(return_value={"jobs": []})),
            patch.object(client, "pull_storage_weights", AsyncMock(return_value={"job_ids": ["resume"]})) as pull,
            patch.object(client, "transfer_storage_weights", AsyncMock()) as transfer,
        ):
            result = await client.download_huggingface_model("org/model", "ws1")
        pull.assert_awaited_once_with("org/model", ["ws1"], revision="main", download_node_id="ws1")
        transfer.assert_not_awaited()
        self.assertEqual(result["job_ids"], ["resume"])

    async def test_tool_schema_requires_explicit_node_and_exposes_no_credentials(self):
        tools = {tool.name: tool for tool in await build_server(ControllerClient()).list_tools()}
        download = tools["download_huggingface_model"]
        self.assertEqual(set(download.input_schema["required"]), {"model_id", "node_id"})
        self.assertEqual(set(download.input_schema["properties"]), {"model_id", "node_id", "revision"})
        self.assertIn("Storage", download.description)
        self.assertIn("get_storage_transfer", download.description)
