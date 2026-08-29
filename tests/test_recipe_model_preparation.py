import asyncio
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock

from manager import Manager
from sparkdeck.virtual_nas import (
    DOWNLOAD_STAGING_RESERVE_BYTES,
    TRANSFER_STAGING_RESERVE_BYTES,
    VIRTUAL_NAS_DOWNLOAD_CAPABILITY,
    VIRTUAL_NAS_DOWNLOAD_BASELINE_CAPABILITY,
    VirtualNAS,
)


MODEL_ID = "org/model"
REVISION = "release-2026-08-27"
RESOLVED_REVISION = "a" * 40
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
        "download_eligible": not has_required_weights,
        "download_reason": (
            "Required model weights are already available"
            if has_required_weights else None
        ),
    }


def preparation_preflight(
    targets: list[dict], *, sources: list[dict] | None = None,
) -> dict:
    sources = list(sources or [])
    return {
        "enabled": True,
        "model_id": MODEL_ID,
        "revision": REVISION,
        "resolved_revision": RESOLVED_REVISION,
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
    manager.virtual_nas.resolve_download_revision = AsyncMock(return_value={
        "requested_revision": REVISION,
        "resolved_revision": RESOLVED_REVISION,
        "size_bytes": MODEL_BYTES,
    })
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
    async def test_disabled_virtual_nas_blocks_missing_weights_without_hub_lookup(self):
        manager = Manager.__new__(Manager)
        manager.settings = {"virtual_nas_enabled": False}
        manager.model_cache_inventory = AsyncMock(return_value=[{
            "id": "worker", "name": "Worker", "online": True,
            "cache_free_size": AMPLE_BYTES,
            "virtual_nas_download_capable": True, "models": [],
        }])
        manager.virtual_nas_transfers = Mock(return_value={"items": []})
        manager.virtual_nas = Mock()
        manager.virtual_nas.resolve_download_revision = AsyncMock()
        manager.virtual_nas.estimate_download_size = AsyncMock()

        plan = await manager.recipe_model_preparation_preflight(
            MODEL_ID, REVISION, ["worker"],
        )

        self.assertFalse(plan["eligible"])
        self.assertEqual(plan["reason"], "Virtual NAS is disabled")
        manager.virtual_nas.resolve_download_revision.assert_not_awaited()
        manager.virtual_nas.estimate_download_size.assert_not_awaited()

    async def test_disabled_virtual_nas_allows_already_cached_recipe(self):
        manager = Manager.__new__(Manager)
        manager.settings = {"virtual_nas_enabled": False}
        manager.model_cache_inventory = AsyncMock(return_value=[{
            "id": "worker", "name": "Worker", "online": True,
            "cache_free_size": AMPLE_BYTES,
            "models": [{
                "model_id": MODEL_ID, "partial": False,
                "revisions": [REVISION], "size_bytes": MODEL_BYTES,
            }],
        }])
        manager.virtual_nas_transfers = Mock(return_value={"items": []})
        manager.virtual_nas = Mock()
        manager.virtual_nas.resolve_download_revision = AsyncMock()

        plan = await manager.recipe_model_preparation_preflight(
            MODEL_ID, REVISION, ["worker"],
        )

        self.assertTrue(plan["eligible"])
        self.assertEqual(plan["action"], "ready")
        manager.virtual_nas.resolve_download_revision.assert_not_awaited()

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

    async def test_explicit_seed_replaces_the_default_choice(self):
        manager = planning_manager(preparation_preflight([
            target("node-b"), target("node-a"),
        ]))

        plan = await manager.recipe_model_preparation_preflight(
            MODEL_ID, REVISION, ["node-b", "node-a"],
            download_node_id="node-a",
        )

        self.assertTrue(plan["eligible"])
        self.assertEqual(plan["download_node_id"], "node-a")
        self.assertEqual(plan["transfer_target_node_ids"], ["node-b"])

    async def test_explicit_seed_that_cannot_download_blocks_instead_of_falling_back(self):
        preflight = preparation_preflight([target("node-b"), target("node-a")])
        preflight["targets"][1]["download_eligible"] = False
        preflight["targets"][1]["download_reason"] = "Node cannot download from Hugging Face"
        preflight["download"] = {
            "size_bytes": MODEL_BYTES,
            "required_free_bytes": MODEL_BYTES * 2 + 1,
        }
        manager = planning_manager(preflight)

        plan = await manager.recipe_model_preparation_preflight(
            MODEL_ID, REVISION, ["node-b", "node-a"],
            download_node_id="node-a",
        )

        self.assertFalse(plan["eligible"])
        self.assertIsNone(plan["download_node_id"])

    async def test_seed_outside_the_selected_set_is_rejected(self):
        manager = planning_manager(preparation_preflight([target("node-b")]))

        with self.assertRaisesRegex(
            ValueError, "download_node_id must be one of the selected nodes",
        ):
            await manager.recipe_model_preparation_preflight(
                MODEL_ID, REVISION, ["node-b"], download_node_id="elsewhere",
            )

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

    async def test_cached_missing_revisions_resume_from_hf_while_empty_nodes_transfer(self):
        source = {
            "node_id": "source", "node_name": "Source",
            "size_bytes": MODEL_BYTES,
        }
        manager = planning_manager(preparation_preflight(
            [
                target("source", has_required_weights=True, has_model_cache=True),
                target("partial", has_model_cache=True),
                target("wrong-revision", has_model_cache=True),
                target("empty"),
            ],
            sources=[source],
        ))

        plan = await manager.recipe_model_preparation_preflight(
            MODEL_ID, REVISION,
            ["source", "partial", "wrong-revision", "empty"],
        )

        self.assertTrue(plan["eligible"])
        self.assertEqual(plan["action"], "download")
        self.assertEqual(plan["download_node_id"], "partial")
        self.assertEqual(
            plan["download_node_ids"], ["partial", "wrong-revision"],
        )
        self.assertEqual(plan["transfer_target_node_ids"], ["empty"])
        self.assertEqual(plan["source"]["node_id"], "source")

    async def test_no_source_uses_empty_seed_and_resumes_cached_nodes_independently(self):
        manager = planning_manager(preparation_preflight([
            target("empty-a"),
            target("partial", has_model_cache=True),
            target("wrong-revision", has_model_cache=True),
            target("empty-b"),
        ]))

        plan = await manager.recipe_model_preparation_preflight(
            MODEL_ID, REVISION,
            ["empty-a", "partial", "wrong-revision", "empty-b"],
        )

        self.assertTrue(plan["eligible"])
        self.assertEqual(plan["download_node_id"], "empty-a")
        self.assertEqual(
            plan["download_node_ids"], [
                "empty-a", "partial", "wrong-revision",
            ],
        )
        self.assertEqual(
            plan["transfer_target_node_ids"], ["empty-b"],
        )

    async def test_other_revision_job_blocks_without_being_adopted(self):
        manager = Manager.__new__(Manager)
        manager.settings = {"virtual_nas_enabled": True}
        manager.model_cache_inventory = AsyncMock(return_value=[{
            "id": "worker", "name": "Worker", "online": True,
            "virtual_nas_download_capable": True,
            "cache_free_size": AMPLE_BYTES, "models": [],
        }])
        manager.virtual_nas = Mock()
        manager.virtual_nas.estimate_download_size = AsyncMock(
            return_value=MODEL_BYTES,
        )
        manager.virtual_nas.resolve_download_revision = AsyncMock(
            side_effect=lambda _model, requested: {
                "requested_revision": requested,
                "resolved_revision": RESOLVED_REVISION,
                "size_bytes": MODEL_BYTES,
            },
        )
        manager.virtual_nas_transfers = Mock(return_value={"items": [{
            "id": "revision-a-job", "model_id": MODEL_ID,
            "target_node_id": "worker", "revision": RESOLVED_REVISION,
            "requested_revision": "revision-a",
            "status": "running",
        }]})

        conflict = await manager.virtual_nas_transfer_preflight(
            MODEL_ID, "revision-b",
        )
        exact = await manager.virtual_nas_transfer_preflight(
            MODEL_ID, "revision-a",
        )
        plan = await manager.recipe_model_preparation_preflight(
            MODEL_ID, "revision-b", ["worker"],
        )

        conflicting_target = conflict["targets"][0]
        self.assertFalse(conflicting_target["download_eligible"])
        self.assertIn("Another revision", conflicting_target["download_reason"])
        self.assertIsNone(conflicting_target["active_job_id"])
        self.assertEqual(exact["targets"][0]["active_job_id"], "revision-a-job")
        self.assertFalse(plan["eligible"])
        self.assertIn("Another revision", plan["reason"])

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
            "capabilities": [
                VIRTUAL_NAS_DOWNLOAD_CAPABILITY,
                VIRTUAL_NAS_DOWNLOAD_BASELINE_CAPABILITY,
            ],
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
        manager.virtual_nas.resolve_download_revision = AsyncMock(return_value={
            "requested_revision": REVISION,
            "resolved_revision": RESOLVED_REVISION,
            "size_bytes": MODEL_BYTES,
        })

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
                "revision": RESOLVED_REVISION,
                "requested_revision": REVISION, "depends_on_job_id": None,
                "workflow_id": "workflow-1",
                "workflow_node_ids": ["node-a", "node-b"],
                "status": "running", "bytes_total": MODEL_BYTES,
                "bytes_transferred": 10, "created_at": 1,
                "started_at": 2, "completed_at": None, "error": None,
            },
            {
                "id": "fanout", "kind": "transfer", "model_id": MODEL_ID,
                "source_node_id": "node-a", "target_node_id": "node-b",
                "revision": RESOLVED_REVISION,
                "requested_revision": REVISION,
                "depends_on_job_id": "download",
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
        self.assertTrue(
            all(job["revision"] == RESOLVED_REVISION for job in result["jobs"])
        )
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
            MODEL_ID, "source", ["target"], RESOLVED_REVISION,
        ))
        self.assertEqual(call.args[5], ["source", "target"])
        self.assertEqual(result["workflow_id"], call.args[4])

    async def test_mixed_queue_batches_resumable_downloads_and_clean_transfers(self):
        source = {
            "node_id": "source", "node_name": "Source",
            "size_bytes": MODEL_BYTES,
        }
        manager = planning_manager(preparation_preflight(
            [
                target("source", has_required_weights=True, has_model_cache=True),
                target("partial", has_model_cache=True),
                target("wrong-revision", has_model_cache=True),
                target("empty"),
            ],
            sources=[source],
        ))
        manager.virtual_nas.queue_download_and_transfer = AsyncMock(return_value={
            "job_ids": ["download-a", "download-b", "transfer"], "jobs": [],
        })

        result = await manager.queue_recipe_model_preparation(
            MODEL_ID, REVISION,
            ["source", "partial", "wrong-revision", "empty"],
        )

        call = manager.virtual_nas.queue_download_and_transfer.await_args
        self.assertEqual(call.args[:5], (
            MODEL_ID, RESOLVED_REVISION, "partial", ["empty"], MODEL_BYTES,
        ))
        self.assertEqual(
            call.kwargs["additional_download_node_ids"], ["wrong-revision"],
        )
        self.assertEqual(call.kwargs["source_node_id"], "source")
        self.assertEqual(call.kwargs["requested_revision"], REVISION)
        manager.virtual_nas.resolve_download_revision.assert_awaited_once_with(
            MODEL_ID, REVISION,
        )
        self.assertIs(
            manager.virtual_nas_transfer_preflight.await_args.args[2],
            manager.virtual_nas.resolve_download_revision.return_value,
        )
        self.assertEqual(
            result["job_ids"], ["download-a", "download-b", "transfer"],
        )


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
        return {
            **node, "online": True,
            "capabilities": [
                VIRTUAL_NAS_DOWNLOAD_CAPABILITY,
                VIRTUAL_NAS_DOWNLOAD_BASELINE_CAPABILITY,
            ],
        }


def queued_job(**overrides) -> dict:
    job = {
        "id": "job-1", "kind": "transfer", "model_id": MODEL_ID,
        "source_node_id": "seed", "target_node_id": "target",
        "revision": RESOLVED_REVISION, "requested_revision": REVISION,
        "depends_on_job_id": None,
        "workflow_id": "workflow-1", "workflow_node_ids": ["seed", "target"],
        "status": "queued", "bytes_total": 10, "bytes_transferred": 0,
        "created_at": 1, "started_at": None, "completed_at": None,
        "error": None,
    }
    job.update(overrides)
    return job


class RecipePreparationExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_transfer_rejects_target_with_wrong_requested_ref_alias(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = Registry()
            nas = VirtualNAS(
                Path(directory), lambda: Path(directory) / "hub", registry,
                lambda: True,
            )
            job = queued_job(target_node_id="local")
            nas.jobs = [job]
            exact_source = {
                "model_id": MODEL_ID, "partial": False,
                "revisions": [REVISION, RESOLVED_REVISION],
                "revision_refs": {REVISION: RESOLVED_REVISION},
                "size_bytes": MODEL_BYTES,
            }
            wrong_alias = {
                "model_id": MODEL_ID, "partial": False,
                "revisions": [REVISION, RESOLVED_REVISION, "b" * 40],
                "revision_refs": {REVISION: "b" * 40},
                "size_bytes": MODEL_BYTES,
            }
            nas._node_storage = AsyncMock(side_effect=[
                {"models": [exact_source], "free_size": AMPLE_BYTES},
                {"models": [], "free_size": AMPLE_BYTES},
                {"models": [wrong_alias], "free_size": AMPLE_BYTES},
            ])

            async def source_bytes():
                yield b"archive"

            source_response = Mock(status_code=200)
            source_response.aiter_bytes = source_bytes
            source_response.aclose = AsyncMock()
            registry.open_stream.return_value = source_response
            nas.import_model = AsyncMock(return_value={"ok": True})

            await nas._run_transfer(job)

            self.assertEqual(job["status"], "failed")
            self.assertIn("complete requested revision", job["error"])

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
                        "revisions": [REVISION, RESOLVED_REVISION],
                        "revision_refs": {REVISION: RESOLVED_REVISION},
                        "size_bytes": actual_size,
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
                target_node_id="seed", bytes_total=1,
            )
            nas.jobs = [job]
            required = MODEL_BYTES * 2 + DOWNLOAD_STAGING_RESERVE_BYTES
            nas.estimate_download_size = AsyncMock(return_value=MODEL_BYTES)
            nas._node_storage = AsyncMock(return_value={
                "models": [], "free_size": required - 1,
            })

            await nas._run_download(job)

            self.assertEqual(job["status"], "failed")
            self.assertEqual(job["bytes_total"], MODEL_BYTES)
            self.assertIn("insufficient free cache space", job["error"])
            nas.estimate_download_size.assert_awaited_once_with(
                MODEL_ID, RESOLVED_REVISION, "", force_refresh=True,
            )
            registry.request.assert_not_awaited()

    async def test_resumed_download_revalidates_with_attempt_baseline_credit(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = Registry()
            nas = VirtualNAS(
                Path(directory), lambda: Path(directory) / "hub", registry,
                lambda: True,
            )
            cached = 25
            required = MODEL_BYTES * 2 + DOWNLOAD_STAGING_RESERVE_BYTES - cached
            job = queued_job(
                kind="download", source_node_id="huggingface",
                target_node_id="seed", bytes_total=MODEL_BYTES,
                require_partial_cache=True,
                download_cache_baseline_bytes=100,
            )
            nas.jobs = [job]
            nas.estimate_download_size = AsyncMock(return_value=MODEL_BYTES)
            nas._node_storage = AsyncMock(return_value={
                "models": [{
                    "model_id": MODEL_ID, "size_bytes": 125,
                    "partial": False, "has_partial_download": True,
                    "partial_size_bytes": cached,
                }],
                "free_size": required,
            })
            registry.request.return_value = {"ok": True, "size_bytes": MODEL_BYTES}

            await nas._run_download(job)

            self.assertEqual(job["status"], "completed")
            self.assertEqual(job["download_attempt_start_bytes"], cached)
            request = registry.request.await_args
            self.assertEqual(
                request.kwargs["json_body"]["download_cache_baseline_bytes"], 100,
            )

    async def test_remote_download_omits_baseline_for_legacy_capability(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = Registry()

            async def legacy_probe(node, force=False):
                return {
                    **node, "online": True,
                    "capabilities": [VIRTUAL_NAS_DOWNLOAD_CAPABILITY],
                }

            registry.probe = legacy_probe
            registry.request.return_value = {"ok": True, "size_bytes": MODEL_BYTES}
            nas = VirtualNAS(
                Path(directory), lambda: Path(directory) / "hub", registry,
                lambda: True,
            )
            job = queued_job(
                kind="download", source_node_id="huggingface",
                target_node_id="seed", bytes_total=MODEL_BYTES,
                download_cache_baseline_bytes=0,
            )
            nas.jobs = [job]
            nas.estimate_download_size = AsyncMock(return_value=MODEL_BYTES)
            nas._node_storage = AsyncMock(return_value={
                "models": [], "free_size": AMPLE_BYTES,
            })

            await nas._run_download(job)

            self.assertEqual(job["status"], "completed")
            self.assertNotIn(
                "download_cache_baseline_bytes",
                registry.request.await_args.kwargs["json_body"],
            )

    async def test_finish_job_preserves_alias_for_revision_completed_while_queued(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = Registry()
            registry.request.return_value = {
                "ok": True, "size_bytes": MODEL_BYTES,
            }
            nas = VirtualNAS(
                Path(directory), lambda: Path(directory) / "hub", registry,
                lambda: True,
            )
            job = queued_job(
                kind="download", source_node_id="huggingface",
                target_node_id="seed", bytes_total=MODEL_BYTES,
                require_partial_cache=True,
            )
            nas.jobs = [job]
            nas._node_storage = AsyncMock(return_value={
                "models": [{
                    "model_id": MODEL_ID, "size_bytes": MODEL_BYTES,
                    "partial": False, "revisions": [RESOLVED_REVISION],
                }],
                "free_size": AMPLE_BYTES,
            })
            nas.estimate_download_size = AsyncMock(return_value=MODEL_BYTES)

            await nas._run_download(job)

            self.assertEqual(job["status"], "completed")
            self.assertEqual(job["bytes_transferred"], MODEL_BYTES)
            nas.estimate_download_size.assert_awaited_once()
            registry.request.assert_awaited_once()
            self.assertEqual(
                registry.request.await_args.kwargs["json_body"]["requested_revision"],
                REVISION,
            )

    async def test_finish_job_rechecks_partial_cache_after_metadata_refresh(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = Registry()
            nas = VirtualNAS(
                Path(directory), lambda: Path(directory) / "hub", registry,
                lambda: True,
            )
            job = queued_job(
                kind="download", source_node_id="huggingface",
                target_node_id="seed", bytes_total=MODEL_BYTES,
                require_partial_cache=True,
            )
            nas.jobs = [job]
            nas._node_storage = AsyncMock(side_effect=[
                {
                    "models": [{
                        "model_id": MODEL_ID, "size_bytes": 25,
                        "partial": True, "revisions": [],
                    }],
                    "free_size": AMPLE_BYTES,
                },
                {"models": [], "free_size": AMPLE_BYTES},
            ])
            nas.estimate_download_size = AsyncMock(return_value=MODEL_BYTES)

            await nas._run_download(job)

            self.assertEqual(job["status"], "failed")
            self.assertIn("partial model cache no longer exists", job["error"])
            registry.request.assert_not_awaited()

    async def test_stop_waits_for_uncancelable_local_download_without_requeue(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = Registry()
            nas = VirtualNAS(
                root, lambda: root / "hub", registry, lambda: True,
            )
            job = queued_job(
                kind="download", source_node_id="huggingface",
                target_node_id="local", bytes_total=MODEL_BYTES,
            )
            nas.jobs = [job]
            nas.estimate_download_size = AsyncMock(return_value=MODEL_BYTES)
            nas._node_storage = AsyncMock(return_value={
                "models": [], "free_size": AMPLE_BYTES,
            })
            started = threading.Event()
            release = threading.Event()

            def blocking_download(*_args):
                started.set()
                if not release.wait(5):
                    raise RuntimeError("test download timed out")
                return {"ok": True, "size_bytes": MODEL_BYTES}

            nas.download_model = Mock(side_effect=blocking_download)
            active = asyncio.create_task(nas._run_download(job))
            nas._active["local"] = active
            nas._dispatcher = asyncio.create_task(asyncio.Event().wait())
            self.assertTrue(await asyncio.to_thread(started.wait, 2))

            stopping = asyncio.create_task(nas.stop())
            await asyncio.sleep(0.05)
            self.assertFalse(stopping.done())
            self.assertEqual(job["status"], "running")
            release.set()
            await asyncio.wait_for(stopping, 2)

            self.assertEqual(job["status"], "completed")
            self.assertEqual(nas.download_model.call_count, 1)
            nas.start()
            await asyncio.sleep(0.05)
            self.assertEqual(nas.download_model.call_count, 1)
            await nas.stop()

    async def test_stop_waits_for_remote_agent_download_without_requeue(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = Registry()
            nas = VirtualNAS(
                root, lambda: root / "hub", registry, lambda: True,
            )
            job = queued_job(
                kind="download", source_node_id="huggingface",
                target_node_id="seed", bytes_total=MODEL_BYTES,
            )
            nas.jobs = [job]
            nas.estimate_download_size = AsyncMock(return_value=MODEL_BYTES)
            nas._node_storage = AsyncMock(return_value={
                "models": [], "free_size": AMPLE_BYTES,
            })
            started = asyncio.Event()
            release = asyncio.Event()

            async def blocking_request(*_args, **_kwargs):
                started.set()
                await release.wait()
                return {"ok": True, "size_bytes": MODEL_BYTES}

            registry.request.side_effect = blocking_request
            active = asyncio.create_task(nas._run_download(job))
            nas._active["seed"] = active
            nas._dispatcher = asyncio.create_task(asyncio.Event().wait())
            await asyncio.wait_for(started.wait(), 2)

            stopping = asyncio.create_task(nas.stop())
            await asyncio.sleep(0.05)
            self.assertFalse(stopping.done())
            self.assertEqual(job["status"], "running")
            release.set()
            await asyncio.wait_for(stopping, 2)

            self.assertEqual(job["status"], "completed")
            self.assertEqual(registry.request.await_count, 1)
            nas.start()
            await asyncio.sleep(0.05)
            self.assertEqual(registry.request.await_count, 1)
            await nas.stop()

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
                MODEL_ID, RESOLVED_REVISION, "seed", ["target"], MODEL_BYTES,
                "workflow-1", ["seed", "target"],
                requested_revision=REVISION,
            )

            self.assertEqual(len(result["jobs"]), 2)
            self.assertTrue(
                all(job["revision"] == RESOLVED_REVISION for job in result["jobs"])
            )
            self.assertTrue(
                all(job["requested_revision"] == REVISION for job in result["jobs"])
            )
            persisted = nas.path.read_text(encoding="utf-8")
            self.assertNotIn(secret, persisted)
            self.assertNotIn("hf_token", persisted)
            self.assertEqual(
                json.loads(persisted)[0]["workflow_node_ids"],
                ["seed", "target"],
            )

    async def test_mixed_batch_persists_downloads_and_source_transfers_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = Registry()
            registry.nodes.update({
                node_id: {"id": node_id, "name": node_id, "enabled": True}
                for node_id in ("source", "partial", "wrong", "empty")
            })
            nas = VirtualNAS(
                root, lambda: root / "hub", registry, lambda: True,
            )
            nas.start = Mock()
            nas._node_storage = AsyncMock(side_effect=[
                {"models": [], "free_size": AMPLE_BYTES},
                {"models": [], "free_size": AMPLE_BYTES},
                {"models": [{
                    "model_id": MODEL_ID, "partial": False,
                    "revisions": [REVISION, RESOLVED_REVISION],
                    "revision_refs": {REVISION: RESOLVED_REVISION},
                    "size_bytes": MODEL_BYTES,
                }], "free_size": AMPLE_BYTES},
                {"models": [], "free_size": AMPLE_BYTES},
            ])

            result = await nas.queue_download_and_transfer(
                MODEL_ID, RESOLVED_REVISION, "partial", ["empty"], MODEL_BYTES,
                "workflow-mixed", ["source", "partial", "wrong", "empty"],
                additional_download_node_ids=["wrong"],
                source_node_id="source",
                requested_revision=REVISION,
            )

            self.assertEqual(
                [job["kind"] for job in result["jobs"]],
                ["download", "download", "transfer"],
            )
            self.assertEqual(
                [job["target_node_id"] for job in result["jobs"]],
                ["partial", "wrong", "empty"],
            )
            transfer = result["jobs"][-1]
            self.assertEqual(transfer["source_node_id"], "source")
            self.assertIsNone(transfer["depends_on_job_id"])
            persisted = json.loads(nas.path.read_text(encoding="utf-8"))
            self.assertEqual(len(persisted), 3)

    async def test_download_only_batch_does_not_validate_unused_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = Registry()
            nas = VirtualNAS(
                root, lambda: root / "hub", registry, lambda: True,
            )
            nas.start = Mock()
            nas._node_storage = AsyncMock(return_value={
                "models": [{
                    "model_id": MODEL_ID, "partial": True,
                    "revisions": [], "size_bytes": 1,
                }],
                "free_size": AMPLE_BYTES,
            })

            result = await nas.queue_download_and_transfer(
                MODEL_ID, RESOLVED_REVISION, "seed", [], MODEL_BYTES,
                source_node_id="offline-unused-source",
                requested_revision=REVISION,
            )

            self.assertEqual(len(result["jobs"]), 1)
            self.assertEqual(result["jobs"][0]["kind"], "download")
            self.assertEqual(result["jobs"][0]["target_node_id"], "seed")

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
            nas.estimate_download_size = AsyncMock(return_value=MODEL_BYTES)

            await nas._run_download(job)

            self.assertEqual(job["status"], "failed")
            self.assertNotIn(secret, job["error"])
            self.assertIn("[REDACTED]", job["error"])
            self.assertNotIn(secret, nas.path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
