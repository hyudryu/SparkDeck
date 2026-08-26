import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx

from sparkdeck.updater import CAPABILITY, CONFIRMATION, UpdateService
from sparkdeck.update_helper import install_revision


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

    async def test_no_release_is_an_honest_blocker(self):
        self.manager.http.get.return_value = response(404, {"message": "Not Found"})
        with patch("sparkdeck.updater.current_revision", return_value="a" * 40), \
             patch("sparkdeck.updater.local_blockers", return_value=[]):
            overview = await self.service.overview()
        self.assertFalse(overview["can_update"])
        self.assertIn("No published GitHub release", overview["blockers"][0])

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
            await self.service.start_cluster("yes", "v1.2.3")
        self.assertEqual(CONFIRMATION, "update-entire-cluster")

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
            "id": "job", "active": True, "phase": "preflight", "target_tag": "v1",
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


class UpdateHelperTests(unittest.TestCase):
    @patch("sparkdeck.update_helper.run")
    @patch("sparkdeck.update_helper.subprocess.run")
    def test_downgrade_uses_detached_checkout_without_reset(self, process_run, command_run):
        process_run.side_effect = [Mock(returncode=1), Mock(returncode=0)]

        direction = install_revision(Path("/sparkdeck"), "b" * 40)

        self.assertEqual(direction, "downgrade")
        command_run.assert_called_once_with(
            Path("/sparkdeck"), "git", "checkout", "--detach", "b" * 40,
        )
        self.assertNotIn("reset", " ".join(str(part) for part in command_run.call_args.args))
