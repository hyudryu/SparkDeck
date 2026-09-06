import copy
import unittest
from unittest.mock import AsyncMock, Mock

from manager import Manager
from sparkdeck.service import SparkDeckService


A, B = "a" * 40, "b" * 40
MODEL = "RadixArk/Qwen3.8-27B-NVFP4"


def cached(node_id, revisions, *, main=None, partial=False):
    return {"id": node_id, "models": [{
        "model_id": MODEL, "partial": partial, "revisions": list(revisions),
        "revision_refs": {"main": main} if main else {},
    }]}


class CachedStartSelectionTests(unittest.IsolatedAsyncioTestCase):
    def service(self, inventory):
        manager = Manager.__new__(Manager)
        manager._resolve_local_path = Mock(return_value=None)
        manager.model_cache_inventory = AsyncMock(return_value=inventory)
        service = SparkDeckService.__new__(SparkDeckService)
        service.manager = manager
        return service

    async def select(self, inventory, *, revision=None, settings=None):
        return await self.service(inventory)._validate_start_selection(
            {"runtime": "vllm", "model": {"repository": MODEL, "revision": revision}},
            [item["id"] for item in inventory], settings,
        )

    async def test_unpinned_cached_snapshot_without_main_ref_is_selected(self):
        self.assertEqual(await self.select([cached("node-4", [A])]), A)
        self.assertEqual(await self.select([cached("node-4", [A, B]), cached("node-3", [A])]), A)

    async def test_unanimous_main_ref_wins_over_other_complete_snapshots(self):
        self.assertEqual(await self.select([
            cached("node-4", [A, B, "main"], main=B),
            cached("node-3", [A, B, "main"], main=B),
        ]), B)

    async def test_ambiguous_disjoint_and_unversioned_caches_do_not_guess(self):
        for inventory in ([cached("node-4", [A, B])],
                          [cached("node-4", [A]), cached("node-3", [B])],
                          [cached("node-4", [])],
                          [cached("node-4", ["main"])],
                          [cached("node-4", [A, B], main=A), cached("node-3", [A, B], main=B)]):
            with self.subTest(inventory=inventory), self.assertRaisesRegex(ValueError, "set --revision"):
                await self.select(inventory)

    async def test_partial_or_missing_weights_are_not_accepted(self):
        with self.assertRaisesRegex(ValueError, "not available.*node-4"):
            await self.select([cached("node-4", [A], partial=True)])
        service = self.service([cached("node-4", [A])])
        with self.assertRaisesRegex(ValueError, "not available.*node-3"):
            await service._validate_start_selection(
                {"runtime": "vllm", "model": {"repository": MODEL}}, ["node-4", "node-3"], None,
            )

    async def test_explicit_identity_and_cli_revisions_are_never_replaced(self):
        for revision, settings in ((A, None), (None, {"extra_args": ["--revision", A]}),
                                   (None, {"extra_args": ["--revision=" + A]})):
            with self.subTest(revision=revision, settings=settings):
                self.assertIsNone(await self.select([cached("node-4", [A, B])], revision=revision, settings=settings))
                with self.assertRaisesRegex(ValueError, "not available"):
                    await self.select([cached("node-4", [B])], revision=revision, settings=settings)
        self.assertIsNone(await self.select([cached("node-4", [A, "main"], main=A)], revision="main"))


class CachedRevisionRelaunchTests(unittest.IsolatedAsyncioTestCase):
    def manager(self):
        manager = Manager.__new__(Manager)
        manager.deployments = [{
            "id": "old", "engine": "vllm", "status": "stopped", "desired_state": "stopped",
            "launch_settings": {"model": MODEL, "engine": "vllm", "extra_args": ["--max-model-len", "256000"]},
            "members": [{"node_id": "node-4", "container_name": "old-container"}],
        }]
        manager._save_deployments = Mock()
        manager._preflight_deployment_launch = AsyncMock()
        manager._member_action = AsyncMock(return_value={"ok": True})
        async def create(body):
            return {"id": "new", "launch_settings": body}
        manager.create_deployment = AsyncMock(side_effect=create)
        return manager

    async def test_pin_reaches_preflight_and_persisted_replacement_without_mutating_original_settings(self):
        manager = self.manager()
        old_settings = manager.deployments[0]["launch_settings"]
        original = copy.deepcopy(old_settings)
        result = await manager.deployment_action("old", "start", ["node-4"], model_revision=A)
        self.assertTrue(result["ok"])
        args = result["deployment"]["launch_settings"]["extra_args"]
        self.assertEqual(args, ["--max-model-len", "256000", "--revision", A])
        self.assertEqual(manager._preflight_deployment_launch.await_args.args[0]["extra_args"], args)
        self.assertEqual(old_settings, original)

    async def test_failed_preflight_keeps_previous_deployment_and_settings_untouched(self):
        manager = self.manager()
        original = copy.deepcopy(manager.deployments)
        manager._preflight_deployment_launch.side_effect = ValueError("node unavailable")
        with self.assertRaisesRegex(ValueError, "node unavailable"):
            await manager.deployment_action("old", "start", ["node-4"], model_revision=A)
        self.assertEqual(manager.deployments, original)
        manager._member_action.assert_not_called()
        manager._save_deployments.assert_not_called()

    async def test_internal_pin_cannot_override_explicit_pin_or_apply_to_other_actions(self):
        manager = self.manager()
        manager.deployments[0]["launch_settings"]["extra_args"] += ["--revision", B]
        with self.assertRaisesRegex(ValueError, "cannot replace"):
            await manager.deployment_action("old", "start", ["node-4"], model_revision=A)
        manager._preflight_deployment_launch.assert_not_called()
        for action, nodes in (("stop", ["node-4"]), ("start", None)):
            with self.assertRaisesRegex(ValueError, "explicit full-deployment"):
                await manager.deployment_action("old", action, nodes, model_revision=A)
