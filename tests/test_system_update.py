import asyncio
import os
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call, patch

import httpx

from sparkdeck.node_toolchain import (
    NODE_ENGINE,
    NodeToolchain,
    _supported,
    frontend_build_environment,
    main as node_toolchain_main,
    resolve_node_toolchain,
)

from sparkdeck.updater import (
    CAPABILITY,
    CONFIRMATION,
    MAIN_BRANCH,
    MAIN_COMMIT_API,
    UpdateService,
    _helper_alive,
    _spawn_update_helper,
    assert_checkout_safe,
    local_blockers,
)
from sparkdeck.update_helper import (
    _prepare_frontend_bundle,
    _publish_frontend_bundle,
    _restore_frontend_bundle,
    apply as apply_update,
    fetch_update_target,
    install_release_revision,
    install_revision,
    npm_executable,
    publish_windows_frontend_stamp,
    restart_service,
)


class FakeManager:
    def __init__(self):
        self.http = SimpleNamespace(get=AsyncMock())
        self.node_registry = SimpleNamespace(request=AsyncMock())
        self.cluster_nodes = AsyncMock(return_value=[{
            "id": "local", "name": "Controller", "local": True,
            "online": True, "enabled": True, "capabilities": [CAPABILITY],
            "app_revision": "a" * 40,
        }])


def response(status, value):
    return httpx.Response(status, json=value, request=httpx.Request("GET", "https://api.github.com/test"))


def git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def initialize_git_repository(root: Path) -> None:
    git(root, "init")
    git(root, "config", "user.name", "SparkDeck Tests")
    git(root, "config", "user.email", "sparkdeck-tests@example.invalid")
    git(root, "checkout", "-b", "main")
    (root / "state.txt").write_text("base", encoding="utf-8")
    git(root, "add", "state.txt")
    git(root, "commit", "-m", "base")


class UpdateServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.manager = FakeManager()
        self.service = UpdateService(self.manager, self.root, self.root / "data")

    async def asyncTearDown(self):
        self.temp.cleanup()

    async def test_unavailable_main_is_an_honest_blocker(self):
        self.manager.http.get.return_value = response(404, {"message": "Not Found"})
        with patch("sparkdeck.updater.current_revision", return_value="a" * 40), \
             patch("sparkdeck.updater.local_blockers", return_value=[]):
            overview = await self.service.overview()
        self.assertFalse(overview["can_update"])
        self.assertIn("Could not check origin/main", overview["blockers"][0])

    async def test_overview_resolves_immutable_main_target(self):
        self.manager.http.get.return_value = response(200, {"sha": "b" * 40})
        with patch("sparkdeck.updater.current_revision", return_value="a" * 40), \
             patch("sparkdeck.updater.local_blockers", return_value=[]):
            overview = await self.service.overview()
        self.assertEqual(overview["target"], {
            "branch": "main",
            "revision": "b" * 40,
            "url": "https://github.com/hyudryu/SparkDeck/tree/main",
        })
        self.assertFalse(overview["up_to_date"])
        self.assertTrue(overview["can_update"])
        self.assertEqual(self.manager.http.get.await_args.args[0], MAIN_COMMIT_API)

    async def test_overview_reports_cluster_up_to_date_at_main_revision(self):
        self.manager.http.get.return_value = response(200, {"sha": "a" * 40})
        with patch("sparkdeck.updater.current_revision", return_value="a" * 40), \
             patch("sparkdeck.updater.local_blockers", return_value=[]):
            overview = await self.service.overview()
        self.assertTrue(overview["up_to_date"])
        self.assertFalse(overview["can_update"])

    async def test_overview_allows_eligible_nodes_when_another_node_is_blocked(self):
        self.manager.http.get.return_value = response(200, {"sha": "b" * 40})
        self.manager.cluster_nodes.return_value = [
            {
                "id": "local", "name": "Controller", "local": True,
                "online": True, "enabled": True, "capabilities": [CAPABILITY],
                "app_revision": "a" * 40,
            },
            {
                "id": "worker", "name": "Worker", "local": False,
                "online": True, "enabled": True, "capabilities": [CAPABILITY],
                "app_revision": "a" * 40,
            },
        ]
        with patch("sparkdeck.updater.current_revision", return_value="a" * 40), \
             patch("sparkdeck.updater.local_blockers", return_value=["Launcher unavailable"]):
            overview = await self.service.overview()

        self.assertTrue(overview["can_update"])
        self.assertIn("Controller: Launcher unavailable", overview["blockers"])
        self.assertEqual(overview["nodes"][1]["blockers"], [])

    async def test_release_resolves_to_immutable_commit(self):
        self.manager.http.get.side_effect = [
            response(200, [{"tag_name": "v1.2.3", "name": "Release 1.2.3", "html_url": "https://github.com/hyudryu/SparkDeck/releases/tag/v1.2.3", "draft": False}]),
            response(200, {"sha": "b" * 40}),
        ]
        release, error = await self.service.latest_release()
        self.assertIsNone(error)
        self.assertEqual(release["tag"], "v1.2.3")
        self.assertEqual(release["revision"], "b" * 40)

    async def test_start_requires_explicit_confirmation(self):
        with self.assertRaisesRegex(ValueError, "confirmation"):
            await self.service.start_cluster("yes", "b" * 40)
        self.assertEqual(CONFIRMATION, "update-entire-cluster")

    async def test_start_rejects_when_main_moved_after_confirmation(self):
        self.manager.http.get.side_effect = [
            response(200, {"sha": "b" * 40}),
            response(200, {"sha": "c" * 40}),
        ]
        with patch("sparkdeck.updater.current_revision", return_value="a" * 40), \
             patch("sparkdeck.updater.local_blockers", return_value=[]):
            with self.assertRaisesRegex(RuntimeError, "origin/main changed"):
                await self.service.start_cluster(CONFIRMATION, "b" * 40)
        self.assertFalse(self.service._read(self.service.cluster_path).get("active", False))

    async def test_release_list_filters_drafts_and_keeps_prereleases(self):
        self.manager.http.get.return_value = response(200, [
            {"tag_name": "v2.0.0-rc1", "name": "2.0 RC", "draft": False, "prerelease": True},
            {"tag_name": "v1.9.0", "name": "Hidden", "draft": True},
            {"tag_name": "bad tag", "name": "Invalid", "draft": False},
            {"tag_name": "v1.8.0", "name": "1.8", "draft": False},
        ])
        releases, error = await self.service.published_releases()
        self.assertIsNone(error)
        self.assertEqual([item["tag"] for item in releases], ["v2.0.0-rc1", "v1.8.0"])
        self.assertTrue(releases[0]["prerelease"])

    async def test_non_latest_published_release_resolves_exact_commit(self):
        self.manager.http.get.side_effect = [
            response(200, [
                {"tag_name": "v2.0.0", "draft": False},
                {"tag_name": "v1.0.0", "draft": False},
            ]),
            response(200, {"sha": "c" * 40}),
        ]
        release, error = await self.service.resolve_release("v1.0.0", force=True)
        self.assertIsNone(error)
        self.assertEqual(release["tag"], "v1.0.0")
        self.assertEqual(release["revision"], "c" * 40)

    async def test_worker_preflight_failure_does_not_stop_controller(self):
        state = {
            "id": "job", "active": True, "phase": "preflight", "target_branch": "main",
            "target_revision": "b" * 40,
            "nodes": [
                {"id": "worker", "name": "Worker", "local": False, "phase": "pending"},
                {"id": "local", "name": "Controller", "local": True, "phase": "pending"},
            ],
        }
        self.manager.node_registry.request.side_effect = RuntimeError("worker unavailable")
        self.service.preflight_local = AsyncMock(return_value={"ok": True})
        self.service.start_local = AsyncMock()
        await self.service._run_cluster(state)
        self.assertEqual(state["phase"], "updating_controller")
        self.assertEqual(state["nodes"][0]["phase"], "failed")
        self.assertIn("worker unavailable", state["nodes"][0]["error"])
        self.service.start_local.assert_awaited_once_with("main", "b" * 40)

    async def test_tracked_changes_on_one_worker_do_not_skip_later_nodes(self):
        state = {
            "id": "job", "active": True, "phase": "preflight",
            "target_branch": "main", "target_revision": "b" * 40,
            "nodes": [
                {
                    "id": "desktop", "name": "Mark's Desktop",
                    "local": False, "phase": "pending",
                },
                {
                    "id": "worker", "name": "Worker 2",
                    "local": False, "phase": "pending",
                },
                {
                    "id": "local", "name": "Controller",
                    "local": True, "phase": "pending",
                },
            ],
        }
        self.manager.node_registry.request.side_effect = [
            RuntimeError(
                "Mark's Desktop agent error: HTTP 409: "
                '{"detail":"Tracked files have local changes"}'
            ),
            {"capability": CAPABILITY, "blockers": []},
            {"phase": "accepted"},
            {"phase": "succeeded", "current_revision": "b" * 40},
        ]
        self.service.preflight_local = AsyncMock(return_value={"ok": True})
        self.service.start_local = AsyncMock()

        with patch("sparkdeck.updater.asyncio.sleep", new=AsyncMock()):
            await self.service._run_cluster(state)

        self.assertEqual(state["nodes"][0]["phase"], "failed")
        self.assertIn("Tracked files have local changes", state["nodes"][0]["error"])
        self.assertEqual(state["nodes"][1]["phase"], "succeeded")
        self.assertEqual(state["nodes"][1]["current_revision"], "b" * 40)
        self.assertEqual(state["phase"], "updating_controller")
        self.assertEqual(
            [request.args[0] for request in self.manager.node_registry.request.await_args_list],
            ["desktop", "worker", "worker", "worker"],
        )
        self.service.start_local.assert_awaited_once_with("main", "b" * 40)

    async def test_blocked_controller_does_not_stop_eligible_worker(self):
        state = {
            "id": "job", "active": True, "phase": "preflight", "target_branch": "main",
            "target_revision": "b" * 40,
            "nodes": [
                {
                    "id": "worker", "name": "Worker", "local": False,
                    "phase": "pending", "blockers": [],
                },
                {
                    "id": "local", "name": "Controller", "local": True,
                    "phase": "pending", "blockers": ["Launcher unavailable"],
                },
            ],
        }
        self.manager.node_registry.request.side_effect = [
            {"capability": CAPABILITY, "blockers": []},
            {"phase": "accepted"},
            {"phase": "succeeded", "current_revision": "b" * 40},
        ]
        self.service.preflight_local = AsyncMock()
        self.service.start_local = AsyncMock()

        with patch("sparkdeck.updater.asyncio.sleep", new=AsyncMock()):
            await self.service._run_cluster(state)

        self.assertFalse(state["active"])
        self.assertEqual(state["phase"], "partial")
        self.assertEqual(state["nodes"][0]["phase"], "succeeded")
        self.assertEqual(state["nodes"][1]["phase"], "failed")
        self.assertIn("Launcher unavailable", state["nodes"][1]["error"])
        self.service.preflight_local.assert_not_awaited()
        self.service.start_local.assert_not_awaited()

    async def test_worker_runtime_failure_does_not_stop_later_nodes(self):
        state = {
            "id": "job", "active": True, "phase": "preflight", "target_branch": "main",
            "target_revision": "b" * 40,
            "nodes": [
                {"id": "bad", "name": "Bad worker", "local": False, "phase": "pending"},
                {"id": "good", "name": "Good worker", "local": False, "phase": "pending"},
                {"id": "local", "name": "Controller", "local": True, "phase": "pending"},
            ],
        }
        self.manager.node_registry.request.side_effect = [
            {"capability": CAPABILITY, "blockers": []},
            {"capability": CAPABILITY, "blockers": []},
            RuntimeError("update request failed"),
            {"phase": "accepted"},
            {"phase": "succeeded", "current_revision": "b" * 40},
        ]
        self.service.preflight_local = AsyncMock(return_value={"ok": True})
        self.service.start_local = AsyncMock()

        with patch("sparkdeck.updater.asyncio.sleep", new=AsyncMock()):
            await self.service._run_cluster(state)

        self.assertEqual(state["nodes"][0]["phase"], "failed")
        self.assertEqual(state["nodes"][1]["phase"], "succeeded")
        self.assertEqual(state["phase"], "updating_controller")
        self.service.start_local.assert_awaited_once()

    async def test_worker_requests_use_main_branch_and_immutable_revision(self):
        state = {
            "id": "job", "active": True, "phase": "preflight", "target_branch": "main",
            "target_revision": "b" * 40,
            "nodes": [
                {"id": "worker", "name": "Worker", "local": False, "phase": "pending"},
                {"id": "local", "name": "Controller", "local": True, "phase": "pending"},
            ],
        }
        self.service.preflight_local = AsyncMock(return_value={"ok": True})
        self.service.start_local = AsyncMock()
        self.manager.node_registry.request.side_effect = [
            {"capability": CAPABILITY, "blockers": []},
            {"phase": "accepted"},
            {"phase": "succeeded", "current_revision": "b" * 40},
        ]
        with patch("sparkdeck.updater.asyncio.sleep", new=AsyncMock()):
            await self.service._run_cluster(state)
        expected = {"branch": "main", "revision": "b" * 40}
        self.assertEqual(self.manager.node_registry.request.await_args_list[0].kwargs["json_body"], expected)
        self.assertEqual(self.manager.node_registry.request.await_args_list[1].kwargs["json_body"], expected)
        self.service.start_local.assert_awaited_once_with("main", "b" * 40)

    async def test_preflight_accepts_divergent_clean_checkout(self):
        commands = []

        def command(_root, *args, **_kwargs):
            commands.append(args)
            if args[:2] == ("git", "rev-parse"):
                return "b" * 40
            if args[:2] == ("git", "show"):
                return '{"update_protocol": 1, "data_schema": 1}'
            return ""

        with patch("sparkdeck.updater.local_blockers", return_value=[]), \
             patch("sparkdeck.updater._run", side_effect=command), \
             patch(
                 "sparkdeck.updater.subprocess.run", return_value=Mock(returncode=0)
             ) as process_run:
            result = await self.service.preflight_local("main", "b" * 40)
        self.assertTrue(result["ok"])
        self.assertIn((
            "git", "fetch", "--force", "origin",
            "refs/heads/main:refs/remotes/origin/main",
        ), commands)
        self.assertEqual(process_run.call_count, 2)
        process_run.assert_called_with(
            ["git", "read-tree", "--dry-run", "-m", "-u", "HEAD", "b" * 40],
            cwd=self.root, capture_output=True, text=True, check=False,
        )

    async def test_preflight_rejects_approved_commit_removed_from_main(self):
        def command(_root, *args, **_kwargs):
            if args[:2] == ("git", "rev-parse"):
                return "c" * 40
            return ""

        with patch("sparkdeck.updater.local_blockers", return_value=[]), \
             patch("sparkdeck.updater._run", side_effect=command), \
             patch(
                 "sparkdeck.updater.subprocess.run",
                 side_effect=[Mock(returncode=1), Mock(returncode=0)],
             ):
            with self.assertRaisesRegex(RuntimeError, "no longer in origin/main"):
                await self.service.preflight_local("main", "b" * 40)

    async def test_preflight_accepts_pinned_commit_after_main_advances(self):
        def command(_root, *args, **_kwargs):
            if args[:2] == ("git", "rev-parse"):
                return "c" * 40
            if args[:2] == ("git", "show"):
                return '{"update_protocol": 1, "data_schema": 1}'
            return ""

        with patch("sparkdeck.updater.local_blockers", return_value=[]), \
             patch("sparkdeck.updater._run", side_effect=command), \
             patch("sparkdeck.updater.subprocess.run", side_effect=[
                 Mock(returncode=0), Mock(returncode=0),
             ]) as process_run:
            result = await self.service.preflight_local("main", "b" * 40)

        self.assertTrue(result["ok"])
        self.assertEqual(process_run.call_args_list[0].args[0], [
            "git", "merge-base", "--is-ancestor", "b" * 40, "c" * 40,
        ])

    async def test_preflight_keeps_event_loop_responsive_during_local_checks(self):
        check_started = threading.Event()
        event_loop_progressed = threading.Event()

        def blocking_local_check(_root):
            check_started.set()
            if not event_loop_progressed.wait(timeout=1):
                return ["event loop stalled during local installation checks"]
            return ["expected local blocker"]

        async def prove_event_loop_progress():
            while not check_started.is_set():
                await asyncio.sleep(0)
            event_loop_progressed.set()

        with patch("sparkdeck.updater.local_blockers", side_effect=blocking_local_check):
            progress = asyncio.create_task(prove_event_loop_progress())
            with self.assertRaisesRegex(RuntimeError, "expected local blocker"):
                await self.service.preflight_local("main", "b" * 40)
            await progress

    async def test_preflight_rejects_checkout_collision(self):
        def command(_root, *args, **_kwargs):
            if args[:2] == ("git", "rev-parse"):
                return "b" * 40
            if args[:2] == ("git", "show"):
                return '{"update_protocol": 1, "data_schema": 1}'
            return ""

        with patch("sparkdeck.updater.local_blockers", return_value=[]), \
             patch("sparkdeck.updater._run", side_effect=command), \
             patch("sparkdeck.updater.subprocess.run", side_effect=[
                 Mock(returncode=0),
                 Mock(returncode=1, stderr="untracked file would be overwritten"),
             ]):
            with self.assertRaisesRegex(RuntimeError, "overwriting local files"):
                await self.service.preflight_local("main", "b" * 40)

    async def test_interrupted_local_helper_becomes_retryable(self):
        self.service._write(self.service.agent_path, {
            "phase": "staging", "target_revision": "b" * 40,
            "helper_pid": 999999, "boot_id": "old-boot",
        })
        with patch("sparkdeck.updater.current_revision", return_value="a" * 40), \
             patch("sparkdeck.updater.local_blockers", return_value=[]):
            status = self.service.agent_status()
        self.assertEqual(status["phase"], "failed")
        self.assertIn("interrupted", status["error"].lower())

    async def test_target_checkout_does_not_finish_before_helper_verification(self):
        self.service._write(self.service.agent_path, {
            "phase": "restarting", "target_revision": "b" * 40,
            "helper_pid": 4321,
        })
        with patch("sparkdeck.updater.current_revision", return_value="b" * 40), \
             patch("sparkdeck.updater._helper_alive", return_value=True), \
             patch("sparkdeck.updater.local_blockers", return_value=[]):
            status = self.service.agent_status()

        self.assertEqual(status["phase"], "restarting")

    async def test_interrupted_controller_job_is_unblocked(self):
        self.service._write(self.service.cluster_path, {
            "id": "stale", "active": True, "phase": "preflight",
            "target_revision": "b" * 40, "nodes": [],
        })
        self.manager.http.get.return_value = response(404, {"message": "Not Found"})
        with patch("sparkdeck.updater.current_revision", return_value="a" * 40), \
             patch("sparkdeck.updater.local_blockers", return_value=[]), \
             patch("sparkdeck.updater._run", return_value=""):
            overview = await self.service.overview()
        self.assertFalse(overview["job"]["active"])
        self.assertEqual(overview["job"]["phase"], "failed")

    async def test_controller_restart_preserves_worker_failure_as_partial(self):
        self.service._write(self.service.cluster_path, {
            "id": "stale", "active": True, "phase": "updating_controller",
            "target_revision": "b" * 40,
            "nodes": [
                {
                    "id": "worker", "name": "Worker", "local": False,
                    "phase": "failed", "error": "Docker unavailable",
                },
                {
                    "id": "local", "name": "Controller", "local": True,
                    "phase": "updating",
                },
            ],
        })
        self.service._write(self.service.agent_path, {
            "phase": "succeeded", "target_revision": "b" * 40,
        })
        self.manager.http.get.return_value = response(200, {"sha": "b" * 40})
        with patch("sparkdeck.updater.current_revision", return_value="b" * 40), \
             patch("sparkdeck.updater.local_blockers", return_value=[]):
            overview = await self.service.overview()

        self.assertFalse(overview["job"]["active"])
        self.assertEqual(overview["job"]["phase"], "partial")
        self.assertEqual(overview["job"]["nodes"][0]["error"], "Docker unavailable")
        self.assertEqual(overview["job"]["nodes"][1]["phase"], "succeeded")

    async def test_restarted_controller_reconciles_local_node_success(self):
        self.service._write(self.service.cluster_path, {
            "id": "stale", "active": True, "phase": "updating_controller",
            "target_revision": "b" * 40,
            "nodes": [
                {
                    "id": "local", "name": "Controller", "local": True,
                    "phase": "updating", "current_revision": "a" * 40,
                    "error": "stale error",
                },
                {
                    "id": "worker", "name": "Worker", "local": False,
                    "phase": "succeeded", "current_revision": "b" * 40,
                },
            ],
        })
        self.service._write(self.service.agent_path, {
            "phase": "succeeded", "target_revision": "b" * 40,
        })
        self.manager.http.get.return_value = response(200, {"sha": "b" * 40})

        with patch("sparkdeck.updater.current_revision", return_value="b" * 40), \
             patch("sparkdeck.updater.local_blockers", return_value=[]):
            overview = await self.service.overview()

        local = next(node for node in overview["job"]["nodes"] if node["local"])
        self.assertFalse(overview["job"]["active"])
        self.assertEqual(overview["job"]["phase"], "succeeded")
        self.assertEqual(local["phase"], "succeeded")
        self.assertEqual(local["current_revision"], "b" * 40)
        self.assertNotIn("error", local)

    async def test_completed_controller_job_self_heals_stale_local_node(self):
        self.service._write(self.service.cluster_path, {
            "id": "completed", "active": False, "phase": "succeeded",
            "target_revision": "b" * 40,
            "nodes": [{
                "id": "local", "name": "Controller", "local": True,
                "phase": "updating", "current_revision": "a" * 40,
            }],
        })
        self.service._write(self.service.agent_path, {
            "phase": "succeeded", "target_revision": "b" * 40,
        })
        self.manager.http.get.return_value = response(200, {"sha": "b" * 40})

        with patch("sparkdeck.updater.current_revision", return_value="b" * 40), \
             patch("sparkdeck.updater.local_blockers", return_value=[]):
            overview = await self.service.overview()

        local = overview["job"]["nodes"][0]
        self.assertEqual(local["phase"], "succeeded")
        self.assertEqual(local["current_revision"], "b" * 40)
        self.assertEqual(self.service._read(self.service.cluster_path), overview["job"])

    async def test_controller_checkout_waits_for_helper_success_before_finishing(self):
        self.service._write(self.service.cluster_path, {
            "id": "active", "active": True, "phase": "updating_controller",
            "target_revision": "b" * 40,
            "nodes": [{
                "id": "local", "name": "Controller", "local": True,
                "phase": "updating",
            }],
        })
        self.service._write(self.service.agent_path, {
            "phase": "restarting", "target_revision": "b" * 40,
            "helper_pid": 4321,
        })
        self.manager.http.get.return_value = response(200, {"sha": "b" * 40})
        with patch("sparkdeck.updater.current_revision", return_value="b" * 40), \
             patch("sparkdeck.updater._helper_alive", return_value=True), \
             patch("sparkdeck.updater.local_blockers", return_value=[]):
            overview = await self.service.overview()

        self.assertTrue(overview["job"]["active"])
        self.assertEqual(overview["job"]["phase"], "updating_controller")
        self.assertEqual(overview["job"]["nodes"][0]["phase"], "updating")

    async def test_windows_start_records_verified_helper_identity(self):
        self.service.preflight_local = AsyncMock(return_value={"ok": True})
        with patch("sparkdeck.updater.current_revision", return_value="a" * 40), \
             patch("sparkdeck.updater.platform.system", return_value="Windows"), \
             patch("sparkdeck.updater._spawn_update_helper", return_value=4321), \
             patch("sparkdeck.updater._windows_process_started", return_value=987654321):
            state = await self.service.start_local("main", "b" * 40)

        self.assertEqual(state["helper_pid"], 4321)
        self.assertEqual(state["helper_started_at"], 987654321)


class UpdateHelperProcessTests(unittest.TestCase):
    @patch("sparkdeck.updater.os.kill")
    @patch("sparkdeck.updater._windows_process_started", return_value=123456)
    @patch("sparkdeck.updater.platform.system", return_value="Windows")
    def test_windows_liveness_uses_creation_time_without_signaling(
        self, _system, _started, process_signal,
    ):
        self.assertTrue(_helper_alive({
            "helper_pid": 4321,
            "helper_started_at": 123456,
        }))
        process_signal.assert_not_called()

    @patch("sparkdeck.updater._windows_process_started", return_value=654321)
    @patch("sparkdeck.updater.platform.system", return_value="Windows")
    def test_windows_liveness_rejects_reused_pid(self, _system, _started):
        self.assertFalse(_helper_alive({
            "helper_pid": 4321,
            "helper_started_at": 123456,
        }))

    @patch("sparkdeck.updater.subprocess.run")
    @patch("sparkdeck.updater.platform.system", return_value="Windows")
    def test_windows_helper_is_spawned_through_detaching_bootstrap(self, _system, process_run):
        process_run.return_value = Mock(returncode=0, stdout="4321\n", stderr="")
        root = Path("C:/SparkDeck")
        command = ["python.exe", "-m", "sparkdeck.update_helper"]

        pid = _spawn_update_helper(root, command)

        self.assertEqual(pid, 4321)
        invocation = process_run.call_args.args[0]
        self.assertEqual(invocation[0:2], [os.sys.executable, "-c"])
        self.assertIn("DETACHED_PROCESS", invocation[2])
        self.assertEqual(invocation[-1], str(root))
        self.assertEqual(process_run.call_args.kwargs["cwd"], root)


class LocalUpdatePreflightTests(unittest.TestCase):
    @patch("sparkdeck.updater.resolve_node_toolchain")
    @patch("sparkdeck.updater.platform.system", return_value="Windows")
    @patch("sparkdeck.updater._run")
    def test_windows_preflight_uses_bundled_launcher_process_status(
        self, command_run, _system, _toolchain,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            launcher = root / "scripts" / "windows" / "sparkdeck.ps1"
            launcher.parent.mkdir(parents=True)
            launcher.write_text("# launcher", encoding="utf-8")
            command_run.side_effect = ["https://github.com/hyudryu/SparkDeck.git", "", ""]

            self.assertEqual(local_blockers(root), [])

        self.assertEqual(command_run.call_args_list[-1].args, (
            root,
            "powershell.exe",
            "-NoLogo",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(launcher),
            "process-status",
        ))
        self.assertEqual(command_run.call_args_list[-1].kwargs, {"timeout": 30})

    @patch("sparkdeck.updater.resolve_node_toolchain")
    @patch("sparkdeck.updater.platform.system", return_value="Windows")
    @patch("sparkdeck.updater._run")
    def test_windows_preflight_requires_bundled_launcher(
        self, command_run, _system, _toolchain,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command_run.side_effect = ["https://github.com/hyudryu/SparkDeck.git", ""]

            blockers = local_blockers(root)

        self.assertEqual(len(blockers), 1)
        self.assertIn("bundled Windows launcher", blockers[0])

    @patch("sparkdeck.updater.resolve_node_toolchain")
    @patch("sparkdeck.updater._service_preflight")
    @patch("sparkdeck.updater._run")
    def test_frontend_toolchain_failure_is_a_node_local_blocker(
        self, command_run, _service, toolchain,
    ):
        command_run.side_effect = ["https://github.com/hyudryu/SparkDeck.git", ""]
        toolchain.side_effect = RuntimeError(
            "Node.js and npm are not available to the SparkDeck service"
        )

        blockers = local_blockers(Path("/sparkdeck"))

        self.assertEqual(len(blockers), 1)
        self.assertIn("Frontend build preflight failed", blockers[0])
        self.assertIn("not available to the SparkDeck service", blockers[0])


class NodeToolchainTests(unittest.TestCase):
    def test_supported_versions_match_frontend_engine(self):
        expected = {
            (20, 18, 9): False,
            (20, 19, 0): True,
            (21, 7, 3): False,
            (22, 11, 0): False,
            (22, 12, 0): True,
            (23, 0, 0): True,
        }

        self.assertEqual(NODE_ENGINE, "^20.19.0 || >=22.12.0")
        for version, supported in expected.items():
            with self.subTest(version=version):
                self.assertEqual(_supported(version), supported)

    @patch("sparkdeck.node_toolchain.subprocess.run")
    def test_discovers_nvm_install_when_service_path_omits_node(self, process_run):
        process_run.return_value = Mock(returncode=0, stdout="v22.12.0\n", stderr="")
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            node_bin = home / ".nvm" / "versions" / "node" / "v22.12.0" / "bin"
            node_bin.mkdir(parents=True)
            (node_bin / "node").write_text("node", encoding="utf-8")
            (node_bin / "npm").write_text("npm", encoding="utf-8")

            toolchain = resolve_node_toolchain(
                {"HOME": str(home), "PATH": "/usr/empty"}, home=home, system="Linux",
            )

        self.assertEqual(toolchain.npm, node_bin / "npm")
        self.assertEqual(toolchain.node, node_bin / "node")
        self.assertEqual(toolchain.version, "22.12.0")
        self.assertTrue(
            process_run.call_args.kwargs["env"]["PATH"].startswith(str(node_bin))
        )

    @patch("sparkdeck.node_toolchain.subprocess.run")
    def test_explicit_node_bin_takes_precedence(self, process_run):
        process_run.return_value = Mock(returncode=0, stdout="v20.19.0\n", stderr="")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            explicit = root / "custom-node" / "bin"
            explicit.mkdir(parents=True)
            (explicit / "node").write_text("node", encoding="utf-8")
            (explicit / "npm").write_text("npm", encoding="utf-8")

            toolchain = resolve_node_toolchain({
                "HOME": str(root),
                "PATH": "/usr/empty",
                "SPARKDECK_NODE_BIN": str(explicit),
            }, home=root, system="Linux")

        self.assertEqual(toolchain.npm.parent, explicit)

    @patch("sparkdeck.node_toolchain.subprocess.run")
    def test_skips_unsupported_path_node_for_supported_nvm_install(self, process_run):
        def version_for(command, **_kwargs):
            version = "v18.20.0" if "system" in command[0] else "v22.12.0"
            return Mock(returncode=0, stdout=version, stderr="")

        process_run.side_effect = version_for
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            system_bin = home / "system-bin"
            nvm_bin = home / ".nvm" / "versions" / "node" / "v22.12.0" / "bin"
            for node_bin in (system_bin, nvm_bin):
                node_bin.mkdir(parents=True)
                (node_bin / "node").write_text("node", encoding="utf-8")
                (node_bin / "npm").write_text("npm", encoding="utf-8")

            with patch(
                "sparkdeck.node_toolchain.shutil.which",
                return_value=str(system_bin / "npm"),
            ):
                toolchain = resolve_node_toolchain(
                    {"HOME": str(home), "PATH": str(system_bin)},
                    home=home,
                    system="Linux",
                )

        self.assertEqual(toolchain.npm, nvm_bin / "npm")

    def test_build_environment_prepends_every_toolchain_path(self):
        toolchain = NodeToolchain(
            npm=Path("/home/test/.asdf/shims/npm"),
            node=Path("/home/test/.asdf/shims/node"),
            version="22.12.0",
            path_entries=(Path("/home/test/.asdf/shims"), Path("/home/test/.asdf/bin")),
        )

        environment = frontend_build_environment(toolchain, {"PATH": "/usr/bin"})

        self.assertEqual(
            environment["PATH"],
            os.pathsep.join([
                str(Path("/home/test/.asdf/shims")),
                str(Path("/home/test/.asdf/bin")),
                "/usr/bin",
            ]),
        )

    @patch("sparkdeck.node_toolchain.subprocess.run")
    @patch("sparkdeck.node_toolchain.resolve_node_toolchain")
    def test_cli_runs_npm_with_the_resolved_environment(self, resolve, process_run):
        node_bin = Path("/home/test/.nvm/versions/node/v22.12.0/bin")
        resolve.return_value = NodeToolchain(
            npm=node_bin / "npm",
            node=node_bin / "node",
            version="22.12.0",
            path_entries=(node_bin,),
        )
        process_run.return_value = Mock(returncode=0)

        with patch("sparkdeck.node_toolchain.sys.argv", [
            "node_toolchain", "--prefix", "frontend", "ci",
        ]), self.assertRaises(SystemExit) as exited:
            node_toolchain_main()

        self.assertEqual(exited.exception.code, 0)
        self.assertEqual(
            process_run.call_args.args[0],
            [str(node_bin / "npm"), "--prefix", "frontend", "ci"],
        )
        self.assertTrue(
            process_run.call_args.kwargs["env"]["PATH"].startswith(
                str(node_bin) + os.pathsep
            )
        )


class UpdateHelperTests(unittest.TestCase):
    @patch("sparkdeck.update_helper.resolve_node_toolchain")
    def test_windows_npm_resolves_cmd_shim(self, resolve):
        resolve.return_value = NodeToolchain(
            npm=Path("C:/Node/npm.cmd"), node=Path("C:/Node/node.exe"),
            version="22.12.0", path_entries=(Path("C:/Node"),),
        )
        self.assertEqual(npm_executable(), "C:\\Node\\npm.cmd")

    def test_update_passes_toolchain_path_to_both_npm_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "root"
            stage = Path(directory) / "stage"
            state_path = root / "data" / "system-update-agent.json"
            state_path.parent.mkdir(parents=True)
            state_path.write_text("{}", encoding="utf-8")
            (stage / "sparkdeck").mkdir(parents=True)
            (stage / "frontend").mkdir()
            (stage / "sparkdeck-update.json").write_text(
                '{"update_protocol": 1, "data_schema": 1}', encoding="utf-8",
            )
            (stage / "sparkdeck" / "updater.py").write_text(
                CAPABILITY, encoding="utf-8",
            )
            (stage / "frontend" / "package-lock.json").write_text(
                "{}", encoding="utf-8",
            )
            node_bin = Path(directory) / "node" / "bin"
            toolchain = NodeToolchain(
                npm=node_bin / "npm",
                node=node_bin / "node",
                version="22.12.0",
                path_entries=(node_bin,),
            )

            def command_result(_root, *args, **_kwargs):
                if args == ("git", "remote", "get-url", "origin"):
                    return "https://github.com/hyudryu/SparkDeck.git"
                if args == ("git", "rev-parse", "HEAD"):
                    return "a" * 40
                return ""

            with patch("sparkdeck.update_helper.run", side_effect=command_result) as command_run, \
                 patch("sparkdeck.update_helper.resolve_node_toolchain", return_value=toolchain), \
                 patch("sparkdeck.update_helper.fetch_update_target"), \
                 patch("sparkdeck.update_helper.install_revision"), \
                 patch("sparkdeck.update_helper._prepare_frontend_bundle", return_value=None), \
                 patch("sparkdeck.update_helper.restart_service"), \
                 patch("sparkdeck.update_helper.wait_for_revision", return_value=True), \
                 patch("sparkdeck.update_helper.tempfile.mkdtemp", return_value=str(stage)), \
                 patch("sparkdeck.update_helper.time.sleep"):
                apply_update(root, state_path, MAIN_BRANCH, "b" * 40)

        npm_calls = [
            command for command in command_run.call_args_list
            if len(command.args) > 1 and command.args[1] == str(toolchain.npm)
        ]
        self.assertEqual(len(npm_calls), 2)
        for command in npm_calls:
            self.assertTrue(
                command.kwargs["env"]["PATH"].startswith(str(node_bin) + os.pathsep)
            )

    @patch("sparkdeck.update_helper.run", return_value="a" * 64)
    @patch("sparkdeck.update_helper.platform.system", return_value="Windows")
    def test_windows_frontend_stamp_matches_launcher_fingerprint(self, _system, command_run):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            module = root / "scripts" / "windows" / "SparkDeck.Windows.psm1"
            module.parent.mkdir(parents=True)
            module.write_text("# launcher module", encoding="utf-8")
            dist = root / "frontend" / "dist"
            dist.mkdir(parents=True)
            environment = {"SPARKDECK_VERSION": "main-12345678"}

            publish_windows_frontend_stamp(root, environment)

            self.assertEqual(
                (dist / ".sparkdeck-source.stamp").read_text(encoding="utf-8"),
                "a" * 64,
            )
            self.assertEqual(
                command_run.call_args.kwargs["env"]["SPARKDECK_FINGERPRINT_ROOT"],
                str(root),
            )
            self.assertEqual(
                command_run.call_args.kwargs["env"]["SPARKDECK_VERSION"],
                "main-12345678",
            )

    def test_divergent_checkout_installs_target_without_moving_feature_branch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initialize_git_repository(root)
            git(root, "branch", "feature/work")
            (root / "state.txt").write_text("main", encoding="utf-8")
            git(root, "commit", "-am", "main update")
            target = git(root, "rev-parse", "HEAD")
            git(root, "checkout", "feature/work")
            (root / "feature.txt").write_text("preserve me", encoding="utf-8")
            git(root, "add", "feature.txt")
            git(root, "commit", "-m", "feature work")
            feature_tip = git(root, "rev-parse", "HEAD")

            mode = install_revision(root, target)

            self.assertEqual(mode, "detached")
            self.assertEqual(git(root, "rev-parse", "HEAD"), target)
            self.assertEqual(git(root, "rev-parse", "feature/work"), feature_tip)
            branch = subprocess.run(
                ["git", "symbolic-ref", "--quiet", "--short", "HEAD"],
                cwd=root, capture_output=True, text=True, check=False,
            )
            self.assertNotEqual(branch.returncode, 0)

    def test_main_branch_detaches_without_moving_main_pointer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initialize_git_repository(root)
            git(root, "checkout", "-b", "target")
            (root / "main.txt").write_text("target", encoding="utf-8")
            git(root, "add", "main.txt")
            git(root, "commit", "-m", "target")
            target = git(root, "rev-parse", "HEAD")
            git(root, "checkout", "main")
            local_main = git(root, "rev-parse", "main")

            mode = install_revision(root, target)

            self.assertEqual(mode, "detached")
            self.assertEqual(git(root, "rev-parse", "HEAD"), target)
            self.assertEqual(git(root, "branch", "--show-current"), "")
            self.assertEqual(git(root, "rev-parse", "main"), local_main)

    def test_divergent_main_detaches_without_moving_main_pointer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initialize_git_repository(root)
            git(root, "checkout", "-b", "target")
            (root / "target.txt").write_text("target", encoding="utf-8")
            git(root, "add", "target.txt")
            git(root, "commit", "-m", "target")
            target = git(root, "rev-parse", "HEAD")
            git(root, "checkout", "main")
            (root / "local.txt").write_text("local", encoding="utf-8")
            git(root, "add", "local.txt")
            git(root, "commit", "-m", "local main")
            local_main = git(root, "rev-parse", "HEAD")

            mode = install_revision(root, target)

            self.assertEqual(mode, "detached")
            self.assertEqual(git(root, "rev-parse", "HEAD"), target)
            self.assertEqual(git(root, "rev-parse", "main"), local_main)

    def test_untracked_collision_blocks_detach_without_moving_head(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initialize_git_repository(root)
            git(root, "checkout", "-b", "target")
            (root / "collision.txt").write_text("target", encoding="utf-8")
            git(root, "add", "collision.txt")
            git(root, "commit", "-m", "target")
            target = git(root, "rev-parse", "HEAD")
            git(root, "checkout", "main")
            (root / "collision.txt").write_text("local untracked", encoding="utf-8")
            original = git(root, "rev-parse", "HEAD")

            with self.assertRaisesRegex(RuntimeError, "would be overwritten"):
                install_revision(root, target)

            self.assertEqual(git(root, "rev-parse", "HEAD"), original)
            self.assertEqual(git(root, "branch", "--show-current"), "main")
            self.assertEqual(
                (root / "collision.txt").read_text(encoding="utf-8"),
                "local untracked",
            )

    def test_preflight_and_install_refuse_ignored_file_collision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initialize_git_repository(root)
            (root / ".gitignore").write_text("secret.env\n", encoding="utf-8")
            git(root, "add", ".gitignore")
            git(root, "commit", "-m", "ignore local secret")
            git(root, "checkout", "-b", "target")
            (root / "secret.env").write_text("target", encoding="utf-8")
            git(root, "add", "--force", "secret.env")
            git(root, "commit", "-m", "target tracks path")
            target = git(root, "rev-parse", "HEAD")
            git(root, "checkout", "main")
            (root / "secret.env").write_text("local secret", encoding="utf-8")
            original = git(root, "rev-parse", "HEAD")

            with self.assertRaisesRegex(RuntimeError, "overwriting ignored local files"):
                assert_checkout_safe(root, target)
            with self.assertRaisesRegex(RuntimeError, "would be overwritten"):
                install_revision(root, target)

            self.assertEqual(git(root, "rev-parse", "HEAD"), original)
            self.assertEqual(
                (root / "secret.env").read_text(encoding="utf-8"),
                "local secret",
            )

    @patch("sparkdeck.update_helper.platform.system", return_value="Windows")
    @patch("sparkdeck.update_helper.run")
    def test_windows_restart_uses_bundled_launcher(self, command_run, _system):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            launcher = root / "scripts" / "windows" / "sparkdeck.ps1"
            launcher.parent.mkdir(parents=True)
            launcher.write_text("# launcher", encoding="utf-8")

            restart_service(root)

        command_run.assert_called_once_with(
            root,
            "powershell.exe",
            "-NoLogo",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(launcher),
            "restart",
            timeout=180,
        )

    @patch("sparkdeck.update_helper.platform.system", return_value="Linux")
    @patch("sparkdeck.update_helper.run")
    def test_linux_restart_still_uses_systemd(self, command_run, _system):
        root = Path("/sparkdeck")

        restart_service(root)

        command_run.assert_called_once_with(
            root, "systemctl", "--user", "restart", "sparkdeck.service", timeout=60,
        )

    def test_frontend_bundle_publish_and_rollback_preserve_correct_build(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            live_dist = root / "frontend" / "dist"
            staged_dist = root / "staged-dist"
            live_dist.mkdir(parents=True)
            staged_dist.mkdir()
            (live_dist / "index.html").write_text("old release", encoding="utf-8")
            (live_dist / "old.js").write_text("old asset", encoding="utf-8")
            (staged_dist / "index.html").write_text("new release", encoding="utf-8")
            (staged_dist / "new.js").write_text("new asset", encoding="utf-8")
            source_file = root / "frontend" / "src" / "App.tsx"
            source_file.parent.mkdir()
            source_file.write_text("new source", encoding="utf-8")
            now = time.time()
            os.utime(staged_dist / "index.html", (now - 120, now - 120))
            os.utime(source_file, (now - 60, now - 60))

            swap_root = _prepare_frontend_bundle(staged_dist, live_dist)
            had_previous = _publish_frontend_bundle(live_dist, swap_root)

            self.assertTrue(had_previous)
            self.assertEqual((live_dist / "index.html").read_text(encoding="utf-8"), "new release")
            self.assertFalse((live_dist / "old.js").exists())
            self.assertGreater(
                (live_dist / "index.html").stat().st_mtime_ns,
                source_file.stat().st_mtime_ns,
            )

            _restore_frontend_bundle(live_dist, swap_root, had_previous)

            self.assertEqual((live_dist / "index.html").read_text(encoding="utf-8"), "old release")
            self.assertTrue((live_dist / "old.js").exists())
            self.assertFalse((live_dist / "new.js").exists())

    @patch("sparkdeck.update_helper.run")
    @patch("sparkdeck.update_helper.subprocess.run")
    def test_dormant_release_downgrade_uses_detached_checkout_without_reset(self, process_run, command_run):
        process_run.side_effect = [Mock(returncode=1), Mock(returncode=0)]

        direction = install_release_revision(Path("/sparkdeck"), "b" * 40)

        self.assertEqual(direction, "downgrade")
        command_run.assert_called_once_with(
            Path("/sparkdeck"), "git", "checkout", "--detach", "b" * 40,
        )
        self.assertNotIn("reset", " ".join(str(part) for part in command_run.call_args.args))

    @patch("sparkdeck.update_helper.subprocess.run")
    @patch("sparkdeck.update_helper.run")
    def test_main_fetch_uses_exact_remote_tracking_ref(self, command_run, process_run):
        command_run.side_effect = ["", "b" * 40]
        process_run.return_value = Mock(returncode=0)

        fetch_update_target(Path("/sparkdeck"), MAIN_BRANCH, "b" * 40)

        self.assertEqual(command_run.call_args_list[0].args, (
            Path("/sparkdeck"), "git", "fetch", "--force", "origin",
            "refs/heads/main:refs/remotes/origin/main",
        ))
        self.assertEqual(command_run.call_args_list[1].args, (
            Path("/sparkdeck"), "git", "rev-parse", "refs/remotes/origin/main^{commit}",
        ))
        process_run.assert_called_once_with(
            ["git", "merge-base", "--is-ancestor", "b" * 40, "b" * 40],
            cwd=Path("/sparkdeck"), capture_output=True, text=True, check=False,
        )

    @patch("sparkdeck.update_helper.run")
    @patch("sparkdeck.update_helper.subprocess.run")
    def test_main_fetch_rejects_target_removed_from_main(self, process_run, command_run):
        command_run.side_effect = ["", "c" * 40]
        process_run.return_value = Mock(returncode=1)

        with self.assertRaisesRegex(RuntimeError, "no longer in origin/main"):
            fetch_update_target(Path("/sparkdeck"), MAIN_BRANCH, "b" * 40)

    @patch("sparkdeck.update_helper.run")
    def test_main_install_detaches_divergent_checkout(self, command_run):
        command_run.return_value = ""

        mode = install_revision(Path("/sparkdeck"), "b" * 40)

        self.assertEqual(mode, "detached")
        command_run.assert_called_once_with(
            Path("/sparkdeck"), "git", "checkout", "--no-overwrite-ignore",
            "--detach", "b" * 40,
        )
