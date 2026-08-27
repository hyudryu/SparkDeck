import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx

from sparkdeck.updater import CAPABILITY, CONFIRMATION, MAIN_BRANCH, MAIN_COMMIT_API, UpdateService
from sparkdeck.update_helper import (
    _prepare_frontend_bundle,
    _publish_frontend_bundle,
    _restore_frontend_bundle,
    fetch_update_target,
    install_release_revision,
    install_revision,
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

    async def test_worker_failure_stops_before_controller(self):
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
        self.assertEqual(state["phase"], "failed")
        self.service.start_local.assert_not_awaited()

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

    async def test_preflight_fetches_exact_main_ref_and_rejects_non_forward_target(self):
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
             patch("sparkdeck.updater.subprocess.run", side_effect=[
                 Mock(returncode=0), Mock(returncode=1),
             ]):
            with self.assertRaisesRegex(RuntimeError, "forward-only"):
                await self.service.preflight_local("main", "b" * 40)
        self.assertIn((
            "git", "fetch", "--force", "origin",
            "refs/heads/main:refs/remotes/origin/main",
        ), commands)

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


class UpdateHelperTests(unittest.TestCase):
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
    def test_main_install_rejects_backward_revision(self, process_run, command_run):
        process_run.return_value = Mock(returncode=1)

        with self.assertRaisesRegex(RuntimeError, "forward-only"):
            install_revision(Path("/sparkdeck"), "b" * 40)

        command_run.assert_not_called()
