import copy
import unittest
from unittest.mock import AsyncMock, Mock, patch

from fastapi import HTTPException
from manager import Manager

with patch("docker.from_env", return_value=Mock()):
    import server


TAG = "local/vllm:patched"
IDS = ["sha256:" + char * 64 for char in "ab"]
BASE = "sha256:" + "c" * 64


def fixture():
    job = {"id": "build1", "status": "succeeded", "image": TAG,
           "files": [{"target": "/opt/patch.py", "sha256": "d" * 64}],
           "nodes": [{"node_id": node, "status": "succeeded", "image_id": image_id, "base_id": BASE}
                     for node, image_id in zip(["local", "worker"], IDS)]}
    inventory = {"partial": False, "errors": [], "results": [
        {"node": {"id": node, "name": node}, "containers": [],
         "images": [{"id": image_id[:19], "full_id": image_id, "tags": [TAG], "size": 100}]}
        for node, image_id in zip(["local", "worker"], IDS)]}
    return job, inventory


class PatchedImageInventoryTests(unittest.IsolatedAsyncioTestCase):
    async def inventory(self, job, inventory):
        with patch.object(server.manager, "cluster_image_inventory", AsyncMock(return_value=inventory)), \
                patch.object(server.image_patch_jobs, "verified_images", return_value=[job]):
            return await server._v1_image_inventory()

    async def test_verified_outputs_group_and_delete_each_nodes_actual_image(self):
        job, raw = fixture()
        result = await self.inventory(job, raw)
        self.assertEqual(len(result["items"]), 1)
        item = result["items"][0]
        self.assertEqual(item["id"], "patch-build:build1")
        self.assertEqual(item["node_image_ids"], dict(zip(["local", "worker"], IDS)))
        self.assertEqual(item["tags"], [TAG])
        remove = AsyncMock(return_value={"ok": True})
        with patch.object(server, "_v1_image_items", AsyncMock(return_value=[item])), \
                patch.object(server.manager, "remove_image_on_nodes", remove):
            await server.v1_remove_image(item["id"])
        remove.assert_awaited_once_with(item["id"], ["local", "worker"], node_image_ids=item["node_image_ids"])

    async def test_retagged_or_short_id_only_images_never_join_verified_group(self):
        for change in ("id", "tag", "missing_full_id", "failed_job", "different_base"):
            with self.subTest(change=change):
                job, raw = fixture()
                image = raw["results"][1]["images"][0]
                if change == "id":
                    image["full_id"] = "sha256:" + "e" * 64
                elif change == "tag":
                    image["tags"] = ["local/vllm:other"]
                elif change == "missing_full_id":
                    image.pop("full_id")
                elif change == "failed_job":
                    job["status"] = "failed"
                else:
                    job["nodes"][1]["base_id"] = "sha256:" + "f" * 64
                result = await self.inventory(job, raw)
                self.assertEqual(len(result["items"]), 2)

    async def test_cross_store_verified_outputs_group_using_common_runtime_identity(self):
        job, raw = fixture()
        for index, node in enumerate(job["nodes"]):
            node["base_id"] = IDS[index]
            node["base_identity"] = "runtime-v1:sha256:" + "f" * 64
        item = (await self.inventory(job, raw))["items"][0]
        self.assertEqual(item["node_ids"], ["local", "worker"])
        self.assertEqual(item["node_image_ids"], dict(zip(["local", "worker"], IDS)))
        for invalid in (None, "runtime-v1:sha256:" + "e" * 64, "malformed"):
            with self.subTest(invalid=invalid):
                job["nodes"][1]["base_identity"] = invalid
                self.assertEqual(len((await self.inventory(job, raw))["items"]), 2)

    async def test_in_use_on_either_node_protects_entire_group(self):
        for identity in (IDS[1], TAG, IDS[1][:19]):
            job, raw = fixture()
            raw["results"][1]["containers"] = [{"image": identity}]
            item = (await self.inventory(job, raw))["items"][0]
            self.assertTrue(item["in_use"])
            remove = AsyncMock()
            with patch.object(server, "_v1_image_items", AsyncMock(return_value=[item])), \
                    patch.object(server.manager, "remove_image_on_nodes", remove):
                with self.assertRaises(HTTPException) as raised:
                    await server.v1_remove_image(item["id"])
            self.assertEqual(raised.exception.status_code, 409)
            remove.assert_not_called()

    async def test_unrelated_images_keep_existing_identity_and_partial_inventory(self):
        job, raw = fixture()
        normal = {"id": "sha256:normal", "tags": ["base:v1"]}
        for result in raw["results"]:
            result["images"].append(copy.deepcopy(normal))
        raw.update(partial=True, errors=[{"error": "another node offline"}])
        result = await self.inventory(job, raw)
        self.assertTrue(result["partial"])
        self.assertEqual(result["errors"], raw["errors"])
        normal_item = next(item for item in result["items"] if item["id"] == normal["id"])
        self.assertEqual(normal_item["node_ids"], ["local", "worker"])
        self.assertNotIn("node_image_ids", normal_item)

    async def test_manager_removes_node_specific_immutable_ids(self):
        manager = Manager.__new__(Manager)
        manager.cluster_nodes = AsyncMock(return_value=[{"id": "local", "name": "local"}, {"id": "worker", "name": "worker"}])
        manager.remove_image = AsyncMock(return_value={"ok": True})
        manager.node_registry = Mock(request=AsyncMock(return_value={"ok": True}))
        result = await manager.remove_image_on_nodes("patch-build:build1", ["local", "worker"],
                                                     node_image_ids=dict(zip(["local", "worker"], IDS)))
        self.assertTrue(result["ok"])
        manager.remove_image.assert_awaited_once_with(IDS[0])
        manager.node_registry.request.assert_awaited_once_with(
            "worker", "DELETE", "/api/agent/images/" + IDS[1].replace(":", "%3A"), timeout=120)
        with self.assertRaisesRegex(ValueError, "every owner"):
            await manager.remove_image_on_nodes("patch-build:build1", ["local", "worker"], node_image_ids={"local": IDS[0]})

    async def test_manager_inventory_includes_full_id_for_verification(self):
        manager = Manager.__new__(Manager)
        manager._images_ts, manager._images_cache = 0, []
        manager.client = Mock()
        manager.client.images.list.return_value = [Mock(id=IDS[0], short_id=IDS[0][:19], tags=[TAG], attrs={})]
        self.assertEqual((await manager.list_images())[0]["full_id"], IDS[0])
