import asyncio
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx

with patch("docker.from_env", return_value=Mock()):
    import server

from sparkdeck.image_patch_jobs import CAPABILITY, ImagePatchJobs
from cluster import NodeAgentResponseError


CONTENT = "# private patch source\nVALUE = 42\n"
FILES = [{"target": "/opt/runtime/patch.py", "content": CONTENT}]
HASHES = [{"target": FILES[0]["target"], "sha256": hashlib.sha256(CONTENT.encode()).hexdigest()}]
BASE_ID = "sha256:" + "a" * 64


def request(nodes=None):
    return {
        "base_image": "runtime/base:stable", "image": "runtime/patched:v1",
        "files": FILES, "node_ids": nodes or ["local"],
    }


def result(base=BASE_ID, image="sha256:patched"):
    return {"base_id": base, "image_id": image, "files": HASHES}


class ImagePatchJobTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.manager = SimpleNamespace(
            client=Mock(), _images_cache=[{"stale": True}], _images_ts=100,
            selected_cluster_nodes=AsyncMock(side_effect=lambda ids: [
                {"id": node, "name": node, "capabilities": [CAPABILITY]} for node in ids
            ]), node_registry=SimpleNamespace(request=AsyncMock()),
        )
        self.jobs = ImagePatchJobs(self.manager, self.directory.name)

    async def asyncTearDown(self):
        for task in list(self.jobs.tasks):
            task.cancel()
        await asyncio.gather(*self.jobs.tasks, return_exceptions=True)
        self.directory.cleanup()

    async def finish(self, job):
        await asyncio.gather(*self.jobs.tasks)
        return self.jobs.get(job["id"])

    async def test_local_build_persists_hashes_and_invalidates_inventory_without_source(self):
        def build(client, payload, log):
            self.assertIs(client, self.manager.client)
            self.assertEqual(payload["files"], FILES)
            self.assertNotIn("node_ids", payload)
            log("Copied 1 patch file")
            return result()

        with patch("sparkdeck.image_patch_jobs.build_patched_image", side_effect=build):
            job = await self.finish(await self.jobs.start(request()))
        self.assertEqual(job["status"], "succeeded")
        self.assertEqual(job["nodes"][0]["image_id"], "sha256:patched")
        self.assertIn("Copied 1 patch file", job["nodes"][0]["logs"])
        self.assertEqual(self.manager._images_cache, [])
        self.assertEqual(self.manager._images_ts, 0)
        persisted = Path(self.jobs.path).read_text(encoding="utf-8")
        self.assertNotIn("private patch source", persisted)
        self.assertNotIn('"content"', persisted)
        self.assertEqual(json.loads(persisted)[0]["files"], HASHES)
        job["nodes"][0]["logs"].append("mutated")
        self.assertNotIn("mutated", self.jobs.get(job["id"])["nodes"][0]["logs"])

    async def test_remote_builds_are_sequential_and_pin_first_resolved_base(self):
        calls = []

        async def remote(node_id, method, path, **kwargs):
            calls.append((node_id, method, kwargs.get("json_body")))
            if method == "POST":
                self.assertNotIn("node_ids", kwargs["json_body"])
                if node_id == "node-3":
                    self.assertEqual(kwargs["json_body"]["expected_base_id"], BASE_ID)
                else:
                    self.assertNotIn("expected_base_id", kwargs["json_body"])
                return {"id": node_id, "status": "building", "nodes": [{"logs": ["Building"]}]}
            return {"id": node_id, "status": "succeeded", "files": HASHES,
                    "nodes": [{**result(), "logs": ["Build complete"]}]}

        self.manager.node_registry.request.side_effect = remote
        with patch("sparkdeck.image_patch_jobs.asyncio.sleep", new=AsyncMock()):
            job = await self.finish(await self.jobs.start(request(["node-4", "node-3"])))
        self.assertEqual(job["status"], "succeeded")
        self.assertEqual([(node, method) for node, method, _ in calls], [
            ("node-4", "POST"), ("node-4", "GET"), ("node-3", "POST"), ("node-3", "GET"),
        ])
        self.assertEqual(job["nodes"][1]["logs"], ["Build complete"])

    async def test_remote_failure_keeps_completed_node_and_skips_remaining_nodes(self):
        with patch.object(self.jobs, "_remote", new=AsyncMock(side_effect=[result(), RuntimeError("base unavailable")])) as remote:
            job = await self.finish(await self.jobs.start(request(["node-4", "node-3", "node-2"])))
        self.assertEqual(job["status"], "failed")
        self.assertEqual([n["status"] for n in job["nodes"]], ["succeeded", "failed", "failed"])
        self.assertEqual(job["nodes"][0]["image_id"], "sha256:patched")
        self.assertEqual(job["nodes"][1]["error"], "base unavailable")
        self.assertIn("Not built", job["nodes"][2]["error"])
        self.assertEqual(remote.await_count, 2)

    async def test_remote_poll_recovers_without_restarting_accepted_build(self):
        for failure in (RuntimeError("could not contact worker"), NodeAgentResponseError("worker", 503, "restarting")):
            with self.subTest(failure=failure):
                self.manager.node_registry.request.reset_mock()
                self.manager.node_registry.request.side_effect = [
                    {"id": "accepted", "status": "building", "nodes": [{"logs": ["Building"]}]},
                    failure,
                    {"id": "accepted", "status": "succeeded", "files": HASHES,
                     "nodes": [{**result(), "logs": ["Build complete"]}]},
                ]
                with patch("sparkdeck.image_patch_jobs.asyncio.sleep", new=AsyncMock()):
                    job = await self.finish(await self.jobs.start(request(["node-3"])))
                self.assertEqual(job["status"], "succeeded")
                calls = self.manager.node_registry.request.await_args_list
                self.assertEqual([call.args[1] for call in calls], ["POST", "GET", "GET"])
                self.assertEqual(calls[1].args[2], calls[2].args[2])
                self.assertIn("Build complete", job["nodes"][0]["logs"])
                self.assertTrue(any("retrying the existing build" in line for line in job["nodes"][0]["logs"]))

    async def test_persistent_poll_failure_stops_at_deadline_with_bounded_warnings(self):
        elapsed = [0]

        async def sleep(seconds):
            elapsed[0] += 240

        async def remote(node_id, method, path, **kwargs):
            if method == "POST":
                return {"id": "accepted", "status": "building", "nodes": [{"logs": ["Building"]}]}
            raise RuntimeError("lost connection " + "x" * 3000)

        self.manager.node_registry.request.side_effect = remote
        with patch("sparkdeck.image_patch_jobs.time", new=SimpleNamespace(monotonic=lambda: elapsed[0])), patch("sparkdeck.image_patch_jobs.asyncio.sleep", side_effect=sleep):
            job = await self.finish(await self.jobs.start(request(["node-3", "node-4"])))
        self.assertEqual(job["status"], "failed")
        self.assertIn("may still be running", job["error"])
        self.assertEqual(elapsed[0], 7200)
        calls = self.manager.node_registry.request.await_args_list
        self.assertEqual(sum(call.args[1] == "POST" for call in calls), 1)
        self.assertTrue(all(call.args[0] == "node-3" for call in calls))
        logs = job["nodes"][0]["logs"]
        self.assertLessEqual(len(logs), 21)
        self.assertTrue(all(len(line) <= 2000 for line in logs))

    async def test_terminal_worker_failure_after_connection_recovers_is_not_retried(self):
        self.manager.node_registry.request.side_effect = [
            {"id": "accepted", "status": "building", "nodes": [{"logs": []}]},
            RuntimeError("connection lost"),
            {"id": "accepted", "status": "failed", "nodes": [{"error": "verification failed", "logs": []}]},
        ]
        with patch("sparkdeck.image_patch_jobs.asyncio.sleep", new=AsyncMock()):
            job = await self.finish(await self.jobs.start(request(["node-3"])))
        self.assertEqual(job["status"], "failed")
        self.assertEqual(job["error"], "verification failed")
        self.assertEqual(self.manager.node_registry.request.await_count, 3)

    async def test_mismatched_remote_base_or_patch_hash_fails_job(self):
        for mismatched in (result(base="sha256:different"), {**result(), "files": []}):
            with self.subTest(result=mismatched):
                with patch.object(self.jobs, "_remote", new=AsyncMock(side_effect=[result(), mismatched])):
                    job = await self.finish(await self.jobs.start(request(["node-4", "node-3"])))
                self.assertEqual(job["status"], "failed")
                self.assertEqual(job["nodes"][0]["status"], "succeeded")
                self.assertEqual(job["nodes"][1]["status"], "failed")

    async def test_concurrent_starts_accept_only_one_build(self):
        gate = asyncio.Event()

        async def build(*args):
            await gate.wait()
            return result()

        with patch.object(self.jobs, "_local", side_effect=build):
            starts = await asyncio.gather(self.jobs.start(request()), self.jobs.start(request()), return_exceptions=True)
            self.assertEqual(sum(isinstance(item, ValueError) for item in starts), 1)
            self.assertEqual(len(self.jobs.list()["items"]), 1)
            gate.set()
            await asyncio.gather(*self.jobs.tasks)

    async def test_unsupported_worker_is_rejected_before_any_build_is_queued(self):
        self.manager.selected_cluster_nodes.return_value = [{"id": "node-3", "name": "Node 3", "capabilities": []}]
        self.manager.selected_cluster_nodes.side_effect = None
        with self.assertRaisesRegex(ValueError, "Update SparkDeck.*Node 3"):
            await self.jobs.start(request(["node-3"]))
        self.assertEqual(self.jobs.list(), {"items": []})

    async def test_public_request_cannot_supply_internal_base_constraint(self):
        with self.assertRaisesRegex(ValueError, "reserved"):
            await self.jobs.start({**request(), "expected_base_id": "sha256:" + "a" * 64})
        self.assertEqual(self.jobs.list(), {"items": []})

    async def test_agent_build_uses_local_node_and_preserves_coordinator_base_constraint(self):
        payload = {key: value for key, value in request().items() if key != "node_ids"}
        payload["expected_base_id"] = BASE_ID
        with patch("sparkdeck.image_patch_jobs.build_patched_image", return_value=result()) as build:
            job = await self.finish(await self.jobs.start(payload, agent=True))
        self.assertEqual(job["status"], "succeeded")
        self.assertEqual([node["node_id"] for node in job["nodes"]], ["local"])
        self.manager.selected_cluster_nodes.assert_not_awaited()
        self.assertEqual(build.call_args.args[1]["expected_base_id"], BASE_ID)

    async def test_remote_terminal_failure_preserves_worker_logs_and_error(self):
        self.manager.node_registry.request.return_value = {
            "id": "worker-build", "status": "failed",
            "nodes": [{"logs": ["Checking patch destination"], "error": "destination is a symlink"}],
        }
        job = await self.finish(await self.jobs.start(request(["node-3"])))
        self.assertEqual(job["status"], "failed")
        self.assertEqual(job["nodes"][0]["logs"], ["Checking patch destination"])
        self.assertEqual(job["nodes"][0]["error"], "destination is a symlink")

    async def test_restart_marks_active_nodes_failed_preserving_completed_nodes(self):
        self.jobs.path.write_text(json.dumps([{
            "id": "interrupted", "status": "building", "nodes": [
                {"node_id": "node-4", "status": "succeeded", "image_id": "done"},
                {"node_id": "node-3", "status": "building"},
                {"node_id": "node-2", "status": "queued"},
            ],
        }]), encoding="utf-8")
        restarted = ImagePatchJobs(self.manager, self.directory.name)
        job = restarted.get("interrupted")
        self.assertEqual(job["status"], "failed")
        self.assertEqual([node["status"] for node in job["nodes"]], ["succeeded", "failed", "failed"])
        self.assertEqual(job["nodes"][0]["image_id"], "done")
        self.assertIn("restarted", job["nodes"][1]["error"])


class ImagePatchApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://test")
        self.assignment = patch.object(server.onboarding.assignment, "load", return_value=None)
        self.assignment.start()

    async def asyncTearDown(self):
        await self.client.aclose()
        self.assignment.stop()

    async def test_gui_route_accepts_patch_payload_and_returns_queued_job(self):
        start = AsyncMock(return_value={"id": "build-1", "status": "queued"})
        with patch.object(server.image_patch_jobs, "start", start):
            response = await self.client.post("/api/v1/images/patch-builds", json=request())
        self.assertEqual(response.status_code, 202)
        start.assert_awaited_once_with(request())

    async def test_agent_create_and_status_require_agent_credentials(self):
        with patch.object(server.manager.agent_credentials, "authorize_controller", return_value=False):
            for method, path in (("POST", "/api/agent/images/patch-builds"), ("GET", "/api/agent/images/patch-builds/job")):
                response = await self.client.request(method, path, json=request())
                self.assertEqual(response.status_code, 401)

    async def test_authorized_agent_build_sets_agent_mode_and_missing_job_is_404(self):
        payload = {key: value for key, value in request().items() if key != "node_ids"}
        with patch.object(server, "_require_agent"), patch.object(server.image_patch_jobs, "start", new=AsyncMock(return_value={"id": "job"})) as start:
            response = await self.client.post("/api/agent/images/patch-builds", json=payload)
        self.assertEqual(response.status_code, 202)
        start.assert_awaited_once_with(payload, agent=True)
        with patch.object(server, "_require_agent"), patch.object(server.image_patch_jobs, "get", side_effect=LookupError("not found")):
            response = await self.client.get("/api/agent/images/patch-builds/missing")
        self.assertEqual(response.status_code, 404)

    async def test_malformed_json_and_non_object_bodies_never_start_a_build(self):
        with patch.object(server.image_patch_jobs, "start", new=AsyncMock()) as start:
            for body in (b"{bad", b"[]", b"null", b"\xff"):
                response = await self.client.post("/api/v1/images/patch-builds", content=body)
                self.assertEqual(response.status_code, 400)
        start.assert_not_awaited()

    async def test_oversized_declared_and_chunked_uploads_never_start_a_build(self):
        async def chunks():
            for _ in range(26):
                yield b"x" * 1024 * 1024

        with patch.object(server.image_patch_jobs, "start", new=AsyncMock()) as start:
            declared = await self.client.post("/api/v1/images/patch-builds", content=b"{}", headers={"Content-Length": str(26 * 1024 * 1024)})
            streamed = await self.client.post("/api/v1/images/patch-builds", content=chunks())
        self.assertEqual(declared.status_code, 413)
        self.assertEqual(streamed.status_code, 413)
        start.assert_not_awaited()
