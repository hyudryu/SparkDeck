import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock

from manager import Manager
from sparkdeck.virtual_nas import (
    DOWNLOAD_STAGING_RESERVE_BYTES,
    TRANSFER_STAGING_RESERVE_BYTES,
    VirtualNAS,
)


MODEL_ID = "org/model"
REVISION = "release-2026-08-27"
MODEL_BYTES = 100
AMPLE_BYTES = 10 * 1024 * 1024 * 1024


def target(
    node_id: str,
    *,
    free_bytes: int | None = AMPLE_BYTES,
    has_required_weights: bool = False,
    has_model_cache: bool = False,
    active_job_id: str | None = None,
) -> dict:
    return {
        "node_id": node_id,
        "node_name": node_id,
        "free_bytes": free_bytes,
        "has_required_weights": has_required_weights,
        "has_model_cache": has_model_cache,
        "active_job_id": active_job_id,
        "reason": "Model preparation is already active" if active_job_id else None,
    }


def preparation_preflight(
    targets: list[dict], *, sources: list[dict] | None = None,
) -> dict:
    sources = list(sources or [])
    return {
        "enabled": True,
        "model_id": MODEL_ID,
        "revision": REVISION,
        "source": sources[0] if sources else None,
        "sources": sources,
        "download": {
            "size_bytes": MODEL_BYTES,
            "required_free_bytes": (
                MODEL_BYTES * 2 + DOWNLOAD_STAGING_RESERVE_BYTES
            ),
        },
        "download_error": None,
        "targets": targets,
        "staging_reserve_bytes": TRANSFER_STAGING_RESERVE_BYTES,
    }


def planning_manager(preflight: dict) -> Manager:
    manager = Manager.__new__(Manager)
    manager.virtual_nas_transfer_preflight = AsyncMock(return_value=preflight)
    manager.virtual_nas = Mock()
    manager.virtual_nas.estimate_download_size = AsyncMock(
        return_value=MODEL_BYTES
    )
    manager.virtual_nas.list_transfers.return_value = {"items": []}
    manager.settings = {
        "virtual_nas_enabled": True, "cluster_node_name": "Coordinator",
    }
    manager.node_registry = Mock()
    manager.node_registry.get.side_effect = lambda node_id: {
        "id": node_id, "name": node_id,
    }
    return manager


class RecipePreparationPlanningTests(unittest.IsolatedAsyncioTestCase):
    async def test_no_selected_source_seeds_first_node_then_fans_out(self):
        manager = planning_manager(preparation_preflight([
            target("node-b"), target("node-a"), target("node-c"),
        ]))

        plan = await manager.recipe_model_preparation_preflight(
            MODEL_ID, REVISION, ["node-b", "node-a", "node-c"],
        )

        self.assertTrue(plan["eligible"])
        self.assertEqual(plan["action"], "download")
        self.assertEqual(plan["download_node_id"], "node-b")
        self.assertEqual(
            plan["transfer_target_node_ids"], ["node-a", "node-c"],
        )
        self.assertEqual(plan["revision"], REVISION)

    async def test_source_outside_selection_does_not_replace_hf_seed(self):
        manager = planning_manager(preparation_preflight(
            [target("node-b"), target("node-a")],
            sources=[{
                "node_id": "outside", "node_name": "Outside",
                "size_bytes": MODEL_BYTES,
            }],
        ))

        plan = await manager.recipe_model_preparation_preflight(
            MODEL_ID, REVISION, ["node-b", "node-a"],
        )

        self.assertEqual(plan["action"], "download")
        self.assertEqual(plan["download_node_id"], "node-b")
        self.assertEqual(plan["transfer_target_node_ids"], ["node-a"])

    async def test_single_node_preparation_is_one_hf_seed_without_fanout(self):
        manager = planning_manager(preparation_preflight([target("solo")]))

        plan = await manager.recipe_model_preparation_preflight(
            MODEL_ID, REVISION, ["solo"],
        )

        self.assertTrue(plan["eligible"])
        self.assertEqual(plan["download_node_id"], "solo")
        self.assertEqual(plan["transfer_target_node_ids"], [])

    async def test_selected_exact_revision_source_fans_out_to_missing_nodes(self):
        source = {
            "node_id": "source", "node_name": "Source",
            "size_bytes": MODEL_BYTES,
        }
        manager = planning_manager(preparation_preflight(
            [
                target("source", has_required_weights=True, has_model_cache=True),
                target("target-a"),
                target("target-b"),
            ],
            sources=[source],
        ))

        plan = await manager.recipe_model_preparation_preflight(
            MODEL_ID, REVISION, ["source", "target-a", "target-b"],
        )

        self.assertTrue(plan["eligible"])
        self.assertEqual(plan["action"], "transfer")
        self.assertEqual(plan["source"]["node_id"], "source")
        self.assertEqual(
            plan["transfer_target_node_ids"], ["target-a", "target-b"],
        )
        self.assertEqual(plan["revision"], REVISION)

    async def test_insufficient_or_unknown_authoritative_capacity_blocks(self):
        required = MODEL_BYTES * 2 + TRANSFER_STAGING_RESERVE_BYTES
        source = {
            "node_id": "source", "node_name": "Source",
            "size_bytes": MODEL_BYTES,
        }
        for free_bytes, expected in (
            (required - 1, "Not enough free cache space"),
            (None, "Free cache capacity is unavailable"),
        ):
            with self.subTest(free_bytes=free_bytes):
                manager = planning_manager(preparation_preflight(
                    [
                        target(
                            "source", has_required_weights=True,
                            has_model_cache=True,
                        ),
                        target("target", free_bytes=free_bytes),
                    ],
                    sources=[source],
                ))

                plan = await manager.recipe_model_preparation_preflight(
                    MODEL_ID, REVISION, ["source", "target"],
                )

                self.assertFalse(plan["eligible"])
                self.assertIn(expected, plan["reason"])

    async def test_generic_disk_capacity_never_substitutes_for_cache_capacity(self):
        manager = Manager.__new__(Manager)
        manager.settings = {"virtual_nas_enabled": True}
        manager.cluster_nodes = AsyncMock(return_value=[{
            "id": "worker", "name": "Worker", "online": True,
            "disk": {"free": AMPLE_BYTES},
        }])
        manager.node_registry = Mock()
        manager.node_registry.request = AsyncMock(return_value={
            "models": [],
            # Deliberately no cache-mount free_size.
        })
        manager.virtual_nas = Mock()
        manager.virtual_nas.list_transfers.return_value = {"items": []}
        manager.virtual_nas.estimate_download_size = AsyncMock(
            return_value=MODEL_BYTES
        )

        plan = await manager.recipe_model_preparation_preflight(
            MODEL_ID, REVISION, ["worker"],
        )

        self.assertFalse(plan["eligible"])
        self.assertEqual(plan["reason"], "Free cache capacity is unavailable")


class RecipePreparationQueueTests(unittest.IsolatedAsyncioTestCase):
    async def test_duplicate_node_ids_are_rejected_before_planning(self):
        manager = planning_manager(preparation_preflight([target("node-a")]))

        with self.assertRaisesRegex(ValueError, "must not contain duplicates"):
            await manager.queue_recipe_model_preparation(
                MODEL_ID, REVISION, ["node-a", "node-a"],
            )

        manager.virtual_nas_transfer_preflight.assert_not_awaited()

    async def test_matching_active_workflow_is_idempotent(self):
        manager = planning_manager(preparation_preflight([target("node-a")]))
        jobs = [
            {
                "id": "download", "kind": "download", "model_id": MODEL_ID,
                "source_node_id": "huggingface", "target_node_id": "node-a",
                "revision": REVISION, "depends_on_job_id": None,
                "workflow_id": "workflow-1",
                "workflow_node_ids": ["node-a", "node-b"],
                "status": "running", "bytes_total": MODEL_BYTES,
                "bytes_transferred": 10, "created_at": 1,
                "started_at": 2, "completed_at": None, "error": None,
            },
            {
                "id": "fanout", "kind": "transfer", "model_id": MODEL_ID,
                "source_node_id": "node-a", "target_node_id": "node-b",
                "revision": REVISION, "depends_on_job_id": "download",
                "workflow_id": "workflow-1",
                "workflow_node_ids": ["node-a", "node-b"],
                "status": "queued", "bytes_total": MODEL_BYTES,
                "bytes_transferred": 0, "created_at": 1,
                "started_at": None, "completed_at": None, "error": None,
            },
        ]
        manager.virtual_nas.list_transfers.return_value = {"items": jobs}
        manager.recipe_model_preparation_preflight = AsyncMock()

        result = await manager.queue_recipe_model_preparation(
            MODEL_ID, REVISION, ["node-a", "node-b"],
        )

        self.assertEqual(result["workflow_id"], "workflow-1")
        self.assertEqual(result["job_ids"], ["download", "fanout"])
        self.assertTrue(all(job["revision"] == REVISION for job in result["jobs"]))
        manager.recipe_model_preparation_preflight.assert_not_awaited()

    async def test_transfer_queue_retains_exact_revision_and_workflow_scope(self):
        source = {
            "node_id": "source", "node_name": "Source",
            "size_bytes": MODEL_BYTES,
        }
        manager = planning_manager(preparation_preflight(
            [
                target("source", has_required_weights=True, has_model_cache=True),
                target("target"),
            ],
            sources=[source],
        ))
        manager.queue_virtual_nas_transfer = AsyncMock(return_value={
            "job_ids": ["job-1"], "jobs": [],
        })

        result = await manager.queue_recipe_model_preparation(
            MODEL_ID, REVISION, ["source", "target"],
        )

        call = manager.queue_virtual_nas_transfer.await_args
        self.assertEqual(call.args[:4], (
            MODEL_ID, "source", ["target"], REVISION,
        ))
        self.assertEqual(call.args[5], ["source", "target"])
        self.assertEqual(result["workflow_id"], call.args[4])


class Registry:
    def __init__(self):
        self.nodes = {
            node_id: {"id": node_id, "name": node_id, "enabled": True}
            for node_id in ("seed", "target")
        }
        self.request = AsyncMock()
        self.open_stream = AsyncMock()

    def get(self, node_id):
        return self.nodes.get(node_id)

    async def probe(self, node, force=False):
        return {**node, "online": True}


def queued_job(**overrides) -> dict:
    job = {
        "id": "job-1", "kind": "transfer", "model_id": MODEL_ID,
        "source_node_id": "seed", "target_node_id": "target",
        "revision": REVISION, "depends_on_job_id": None,
        "workflow_id": "workflow-1", "workflow_node_ids": ["seed", "target"],
        "status": "queued", "bytes_total": 10, "bytes_transferred": 0,
        "created_at": 1, "started_at": None, "completed_at": None,
        "error": None,
    }
    job.update(overrides)
    return job


class RecipePreparationExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_transfer_revalidates_capacity_against_actual_source_size(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = Registry()
            nas = VirtualNAS(
                Path(directory), lambda: Path(directory) / "hub", registry,
                lambda: True,
            )
            job = queued_job()
            nas.jobs = [job]
            actual_size = 1000
            actual_required = (
                actual_size * 2 + TRANSFER_STAGING_RESERVE_BYTES
            )
            nas._node_storage = AsyncMock(side_effect=[
                {
                    "models": [{
                        "model_id": MODEL_ID, "partial": False,
                        "revisions": [REVISION], "size_bytes": actual_size,
                    }],
                    "free_size": AMPLE_BYTES,
                },
                {"models": [], "free_size": actual_required - 1},
            ])

            await nas._run_transfer(job)

            self.assertEqual(job["status"], "failed")
            self.assertEqual(job["bytes_total"], actual_size)
            self.assertIn("insufficient free cache space", job["error"])
            registry.open_stream.assert_not_awaited()

    async def test_download_revalidates_execution_time_free_space(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = Registry()
            nas = VirtualNAS(
                Path(directory), lambda: Path(directory) / "hub", registry,
                lambda: True,
            )
            job = queued_job(
                kind="download", source_node_id="huggingface",
                target_node_id="seed", bytes_total=MODEL_BYTES,
            )
            nas.jobs = [job]
            required = MODEL_BYTES * 2 + DOWNLOAD_STAGING_RESERVE_BYTES
            nas._node_storage = AsyncMock(return_value={
                "models": [], "free_size": required - 1,
            })

            await nas._run_download(job)

            self.assertEqual(job["status"], "failed")
            self.assertIn("insufficient free cache space", job["error"])
            registry.request.assert_not_awaited()

    async def test_exact_revision_is_persisted_but_hf_token_is_not(self):
        secret = "hf_recipe_secret_value"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = Registry()
            nas = VirtualNAS(
                root, lambda: root / "hub", registry, lambda: True,
                token_provider=lambda: secret,
            )
            nas.start = Mock()
            nas._node_storage = AsyncMock(return_value={
                "models": [], "free_size": AMPLE_BYTES,
            })

            result = await nas.queue_download_and_transfer(
                MODEL_ID, REVISION, "seed", ["target"], MODEL_BYTES,
                "workflow-1", ["seed", "target"],
            )

            self.assertEqual(len(result["jobs"]), 2)
            self.assertTrue(
                all(job["revision"] == REVISION for job in result["jobs"])
            )
            persisted = nas.path.read_text(encoding="utf-8")
            self.assertNotIn(secret, persisted)
            self.assertNotIn("hf_token", persisted)
            self.assertEqual(
                json.loads(persisted)[0]["workflow_node_ids"],
                ["seed", "target"],
            )

    async def test_download_error_redacts_ephemeral_hf_token(self):
        secret = "hf_ephemeral_secret"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = Registry()
            registry.request.side_effect = RuntimeError(
                f"remote rejected {secret}"
            )
            nas = VirtualNAS(
                root, lambda: root / "hub", registry, lambda: True,
                token_provider=lambda: secret,
            )
            job = queued_job(
                kind="download", source_node_id="huggingface",
                target_node_id="seed", bytes_total=MODEL_BYTES,
            )
            nas.jobs = [job]
            nas._node_storage = AsyncMock(return_value={
                "models": [], "free_size": AMPLE_BYTES,
            })

            await nas._run_download(job)

            self.assertEqual(job["status"], "failed")
            self.assertNotIn(secret, job["error"])
            self.assertIn("[REDACTED]", job["error"])
            self.assertNotIn(secret, nas.path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
