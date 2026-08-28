import os
import sqlite3
import tempfile
import threading
import unittest
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from sparkdeck.models import BenchmarkSample, Deployment, DeploymentKind, ModelIdentity, RuntimeKind
from sparkdeck.storage import (
    COMMUNITY_CONSENT_CONTRACT_VERSION,
    COMMUNITY_EVIDENCE_POLICY,
    COMMUNITY_UPLOAD_FIELDS,
    SparkDeckStore,
    _COMMUNITY_AGGREGATE_BATCH_SIZE,
    _community_prompt_bucket,
)


class SparkDeckStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = SparkDeckStore(Path(self.temp.name) / "sparkdeck.sqlite3")

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def test_prompt_bucket_contract_uses_explicit_half_up_rounding(self):
        self.assertEqual(_community_prompt_bucket(1), 400)
        self.assertEqual(_community_prompt_bucket(800), 400)
        self.assertEqual(_community_prompt_bucket(801), 1000)
        self.assertEqual(_community_prompt_bucket(2_499), 2000)
        self.assertEqual(_community_prompt_bucket(2_500), 3000)
        self.assertEqual(_community_prompt_bucket(9_999), 9000)
        self.assertIsNone(_community_prompt_bucket(10_000))

    def test_consent_defaults_off_and_deployment_hides_endpoint(self):
        self.assertFalse(self.store.sync_status()["consent"])
        deployment = Deployment(
            id="dep-1", alias="chat", runtime=RuntimeKind.SGLANG,
            kind=DeploymentKind.EXTERNAL,
            model=ModelIdentity("org/model"), base_url_set=True,
        )
        self.store.add_deployment(deployment, "http://private-host:8000", None)
        public = self.store.deployment("chat")
        self.assertTrue(public["base_url_set"])
        self.assertNotIn("_base_url", public)
        self.assertNotIn("private-host", str(public))

    def test_deployment_creation_timestamps_persist_for_all_kinds(self):
        cases = (
            ("external", RuntimeKind.VLLM, DeploymentKind.EXTERNAL),
            ("managed-llama", RuntimeKind.LLAMA_CPP, DeploymentKind.MANAGED),
        )
        created_at = {}
        for deployment_id, runtime, kind in cases:
            self.store.add_deployment(Deployment(
                id=deployment_id,
                alias=deployment_id,
                runtime=runtime,
                kind=kind,
                model=ModelIdentity(f"org/{deployment_id}"),
            ))
            value = self.store.deployment(deployment_id)["created_at"]
            self.assertIsNotNone(datetime.fromisoformat(value))
            created_at[deployment_id] = value

        database = self.store.path
        self.store.close()
        self.store = SparkDeckStore(database)

        self.assertEqual(
            {item["id"]: item["created_at"] for item in self.store.deployments()},
            created_at,
        )

    def test_migration_backfills_created_at_for_legacy_deployments(self):
        database = Path(self.temp.name) / "legacy.sqlite3"
        connection = sqlite3.connect(database)
        try:
            connection.execute(
                """CREATE TABLE deployments (
                    id TEXT PRIMARY KEY,
                    alias TEXT NOT NULL UNIQUE,
                    runtime TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    repository TEXT NOT NULL,
                    revision TEXT,
                    artifact TEXT,
                    quantization TEXT,
                    base_url TEXT,
                    container_name TEXT,
                    settings_json TEXT NOT NULL DEFAULT '{}',
                    credential_ref TEXT
                )"""
            )
            connection.execute(
                """INSERT INTO deployments(
                    id, alias, runtime, kind, repository, settings_json
                ) VALUES (?, ?, ?, ?, ?, ?)""",
                ("legacy", "Legacy", "llama.cpp", "managed", "org/model", "{}"),
            )
            connection.commit()
        finally:
            connection.close()

        migrated = SparkDeckStore(database)
        try:
            created_at = migrated.deployment("legacy")["created_at"]
            self.assertIsNotNone(datetime.fromisoformat(created_at))
            stored = migrated._connection.execute(
                "SELECT created_at FROM deployments WHERE id = 'legacy'"
            ).fetchone()[0]
            self.assertEqual(stored, created_at)
        finally:
            migrated.close()

    def test_only_consented_eligible_samples_enter_outbox(self):
        sample = BenchmarkSample(
            id="sample-1", created_at="2026-08-25T00:00:00+00:00",
            deployment_id=None, model=ModelIdentity(
                "org/model", quantization="NVFP4",
            ),
            runtime=RuntimeKind.VLLM, runtime_version=None,
            hardware={"architecture": "x86_64"},
            configuration={"context_length": 4096},
            input_tokens=20, output_tokens=30, latency_ms=100,
            ttft_ms=10, generation_tokens_per_second=300,
            prompt_tokens_per_second=200, cold_start=False,
            eligible_for_community=True,
        )
        initial = self.store.community_consent_snapshot()
        self.assertEqual(initial, {
            "enabled": False, "generation": 0, "telemetry_cluster_id": None,
        })
        self.assertFalse(self.store.add_benchmark_if_consented(
            sample, initial["generation"],
        ))
        self.assertEqual(self.store.benchmarks()[1], 0)
        self.assertEqual(self.store.sync_status()["outbox"]["waiting_for_account"], 0)
        consent = self.store.set_community_consent(True)
        self.assertRegex(consent["telemetry_cluster_id"], r"^[0-9a-f-]{36}$")
        second = replace(sample, id="sample-2")
        self.assertTrue(self.store.add_benchmark_if_consented(
            second, consent["generation"],
        ))
        self.assertEqual(self.store.sync_status()["outbox"]["waiting_for_account"], 1)
        self.store.set_setting("device_pairing", {"status": "paired", "device_id": "device-1"})
        self.assertEqual(self.store.retry_outbox(), 1)
        self.assertEqual(self.store.outbox_batch(), [{
            "model_id": "org/model",
            "quantization": "NVFP4",
            "prompt_tokens_bucket": 400,
            "inference_tokens_per_second": 300.0,
            "telemetry_cluster_id": consent["telemetry_cluster_id"],
            "concurrency": 1,
        }])
        self.assertEqual(self.store.mark_outbox_failed(["sample-2"], "offline"), 1)
        self.assertEqual(self.store.outbox_batch(), [])
        self.assertEqual(self.store.retry_outbox(), 1)
        self.assertEqual(self.store.mark_outbox_synced(["sample-2"]), 1)
        self.assertEqual(self.store.sync_status()["outbox"]["synced"], 1)

    def test_store_files_are_owner_only(self):
        if os.name == "nt":
            self.skipTest("POSIX modes do not exist on Windows")

        self.assertEqual(
            oct(self.store.path.stat().st_mode & 0o777), "0o600")
        self.assertEqual(
            oct(self.store.path.parent.stat().st_mode & 0o777), "0o700")
        for suffix in ("-wal", "-shm"):
            sidecar = Path(f"{self.store.path}{suffix}")
            if sidecar.exists():
                self.assertEqual(
                    oct(sidecar.stat().st_mode & 0o777), "0o600")

    def test_sync_status_redacts_pairing_claims(self):
        self.store.set_setting("device_pairing", {
            "status": "paired", "sub": "stable-sub", "email": "person@example.com",
            "credential": "secret", "refresh_token": "refresh-secret-1",
        })

        status = self.store.sync_status()

        self.assertEqual(status["pairing"], {"status": "paired", "token_invalid": False})
        self.assertNotIn("stable-sub", str(status))
        self.assertNotIn("person@example.com", str(status))
        self.assertNotIn("secret", str(status))
        self.assertNotIn("refresh-secret-1", str(status))

    def test_pairing_promotes_waiting_uploads_to_pending(self):
        sample = BenchmarkSample(
            id="sample-waiting", created_at="2026-08-25T00:00:00+00:00",
            deployment_id=None, model=ModelIdentity("org/model"),
            runtime=RuntimeKind.VLLM, runtime_version=None,
            hardware={"architecture": "x86_64"},
            configuration={"context_length": 4096},
            input_tokens=20, output_tokens=30, latency_ms=100,
            ttft_ms=10, generation_tokens_per_second=300,
            prompt_tokens_per_second=200, cold_start=False,
            eligible_for_community=True,
        )
        consent = self.store.set_community_consent(True)
        self.assertTrue(self.store.add_benchmark_if_consented(
            sample, consent["generation"],
        ))
        self.assertEqual(self.store.sync_status()["outbox"]["waiting_for_account"], 1)

        self.assertEqual(self.store.promote_outbox_for_pairing(), 1)

        self.assertEqual(self.store.sync_status()["outbox"]["waiting_for_account"], 0)
        self.assertEqual(self.store.sync_status()["outbox"]["pending"], 1)
        self.assertEqual(self.store.promote_outbox_for_pairing(), 0)

    def test_enabling_consent_does_not_queue_existing_samples(self):
        sample = BenchmarkSample(
            id="sample-private", created_at="2026-08-25T00:00:00+00:00",
            deployment_id="private-deployment",
            model=ModelIdentity(
                "org/model", revision="abc123", artifact="C:/private/model.gguf"
            ),
            runtime=RuntimeKind.LLAMA_CPP, runtime_version="registry.local/team/image:1",
            hardware={"architecture": "aarch64"}, configuration={"context_length": 4096},
            input_tokens=20, output_tokens=30, latency_ms=100, ttft_ms=10,
            generation_tokens_per_second=300, prompt_tokens_per_second=200,
            cold_start=False, eligible_for_community=True,
        )
        self.store.add_benchmark(sample, queue=False)
        self.store.set_community_consent(True)
        self.store.set_setting("device_pairing", {"status": "paired"})
        self.store.retry_outbox()
        self.assertEqual(self.store.outbox_batch(), [])
        local, _ = self.store.benchmarks()
        self.assertEqual(local[0]["model"]["artifact"], "C:/private/model.gguf")
        self.assertEqual(local[0]["model"]["revision"], "abc123")
        self.assertEqual(local[0]["runtime_version"], "registry.local/team/image:1")
        self.assertEqual(local[0]["hardware"], {"architecture": "aarch64"})
        self.assertEqual(local[0]["sync_state"], "local")

    def test_withdrawing_consent_removes_unsent_uploads_but_keeps_samples(self):
        base = BenchmarkSample(
            id="pending", created_at="2026-08-25T00:00:00+00:00",
            deployment_id=None, model=ModelIdentity("org/model"),
            runtime=RuntimeKind.VLLM, runtime_version=None,
            hardware={}, configuration={"context_length": 4096},
            input_tokens=20, output_tokens=30,
            latency_ms=100, ttft_ms=10, generation_tokens_per_second=300,
            prompt_tokens_per_second=200, cold_start=False,
            eligible_for_community=True,
        )
        self.store.set_setting("device_pairing", {"status": "paired"})
        self.store.set_community_consent(True)
        self.store.add_benchmark(base, queue=True)
        self.store.add_benchmark(replace(base, id="failed"), queue=True)
        self.store.mark_outbox_failed(["failed"], "offline")

        self.store.set_community_consent(False)

        self.assertFalse(self.store.sync_status()["consent"])
        self.assertEqual(self.store.outbox_batch(), [])
        self.assertEqual(self.store.retry_outbox(), 0)
        local, total = self.store.benchmarks()
        self.assertEqual(total, 2)
        self.assertTrue(all(item["sync_state"] == "local" for item in local))

        reenabled = self.store.set_community_consent(True)
        self.assertEqual(reenabled["telemetry_cluster_id"], self.store.get_setting(
            "telemetry_cluster_id",
        ))
        self.assertEqual(self.store.outbox_batch(), [])

    def test_consent_generation_rejects_request_across_disable_reenable(self):
        sample = BenchmarkSample(
            id="stale-generation", created_at="2026-08-25T00:00:00+00:00",
            deployment_id=None,
            model=ModelIdentity("RadixArk/Qwen3.8-27B", quantization="NVFP4"),
            runtime=RuntimeKind.SGLANG, runtime_version=None, hardware={},
            configuration={}, input_tokens=400, output_tokens=64,
            latency_ms=4_000, ttft_ms=100,
            generation_tokens_per_second=80,
            prompt_tokens_per_second=None, cold_start=False,
            eligible_for_community=True,
        )
        first = self.store.set_community_consent(
            True, "55555555-5555-4555-8555-555555555555",
        )
        disabled = self.store.set_community_consent(False)
        second = self.store.set_community_consent(True)

        self.assertEqual(disabled["generation"], first["generation"] + 1)
        self.assertEqual(second["generation"], disabled["generation"] + 1)
        self.assertEqual(
            second["telemetry_cluster_id"], first["telemetry_cluster_id"],
        )
        self.assertFalse(self.store.add_benchmark_if_consented(
            sample, first["generation"],
        ))
        self.assertEqual(self.store.benchmarks()[1], 0)

        self.assertTrue(self.store.add_benchmark_if_consented(
            replace(sample, id="current-generation"), second["generation"],
        ))
        self.assertEqual(self.store.benchmarks()[1], 1)

    def test_controller_cluster_id_is_validated_and_shared_without_rotation(self):
        cluster_id = "66666666-6666-4666-8666-666666666666"
        first = self.store.set_community_consent(True, cluster_id.upper())
        repeated = self.store.set_community_consent(True, cluster_id)

        self.assertEqual(first["telemetry_cluster_id"], cluster_id)
        self.assertEqual(repeated, first)
        with self.assertRaisesRegex(ValueError, "must be a UUID"):
            self.store.set_community_consent(True, "host-or-account-name")

    def test_cluster_identity_change_invalidates_epoch_and_unsent_rows(self):
        sample = BenchmarkSample(
            id="old-cluster", created_at="2026-08-25T00:00:00+00:00",
            deployment_id=None, model=ModelIdentity("org/model"),
            runtime=RuntimeKind.VLLM, runtime_version=None, hardware={},
            configuration={}, input_tokens=400, output_tokens=32,
            latency_ms=3_000, ttft_ms=50,
            generation_tokens_per_second=64,
            prompt_tokens_per_second=None, cold_start=False,
            eligible_for_community=True,
        )
        first = self.store.set_community_consent(
            True, "77777777-7777-4777-8777-777777777777",
        )
        self.assertTrue(self.store.add_benchmark_if_consented(
            sample, first["generation"],
        ))
        self.assertEqual(self.store.sync_status()["outbox"]["waiting_for_account"], 1)

        changed = self.store.set_community_consent(
            True, "88888888-8888-4888-8888-888888888888",
        )

        self.assertEqual(changed["generation"], first["generation"] + 1)
        self.assertEqual(self.store.outbox_batch(), [])
        self.assertFalse(self.store.add_benchmark_if_consented(
            replace(sample, id="stale-cluster"), first["generation"],
        ))

    def test_membership_revocation_clears_cluster_identity_and_unsent_rows(self):
        first = self.store.set_community_consent(
            True, "99999999-9999-4999-8999-999999999999",
        )
        sample = BenchmarkSample(
            id="former-cluster", created_at="2026-08-25T00:00:00+00:00",
            deployment_id=None, model=ModelIdentity("org/model"),
            runtime=RuntimeKind.VLLM, runtime_version=None, hardware={},
            configuration={}, input_tokens=400, output_tokens=64,
            latency_ms=4_000, ttft_ms=100,
            generation_tokens_per_second=80,
            prompt_tokens_per_second=None, cold_start=False,
            eligible_for_community=True,
        )
        self.assertTrue(self.store.add_benchmark_if_consented(
            sample, first["generation"],
        ))

        revoked = self.store.revoke_community_membership()

        self.assertEqual(revoked, {
            "enabled": False,
            "generation": first["generation"] + 1,
            "telemetry_cluster_id": None,
        })
        self.assertIsNone(self.store.get_setting("telemetry_cluster_id"))
        self.assertEqual(self.store.outbox_batch(), [])
        reenabled = self.store.set_community_consent(True)
        self.assertNotEqual(
            reenabled["telemetry_cluster_id"], first["telemetry_cluster_id"],
        )

    def test_migration_requires_fresh_consent_for_expanded_payload_contract(self):
        sample = BenchmarkSample(
            id="legacy-consent", created_at="2026-08-25T00:00:00+00:00",
            deployment_id=None, model=ModelIdentity("org/model"),
            runtime=RuntimeKind.VLLM, runtime_version=None,
            hardware={}, configuration={"context_length": 4096},
            input_tokens=20, output_tokens=30, latency_ms=100, ttft_ms=10,
            generation_tokens_per_second=300, prompt_tokens_per_second=200,
            cold_start=False, eligible_for_community=True,
        )
        self.store.set_setting("device_pairing", {"status": "paired"})
        self.store.set_community_consent(True)
        self.store.add_benchmark(sample, queue=True)
        with self.store._connection:
            self.store._connection.execute(
                "DELETE FROM settings WHERE key = ?",
                ("community_consent_contract_version",),
            )

        database = self.store.path
        self.store.close()
        self.store = SparkDeckStore(database)

        self.assertFalse(self.store.sync_status()["consent"])
        self.assertEqual(self.store.outbox_batch(), [])
        self.assertEqual(self.store.benchmarks()[1], 1)
        self.assertEqual(
            self.store.get_setting("community_consent_contract_version"),
            COMMUNITY_CONSENT_CONTRACT_VERSION,
        )
        self.assertGreater(
            self.store.community_consent_snapshot()["generation"], 1,
        )

    def test_migration_adds_cluster_id_column_to_legacy_benchmark_table(self):
        database = Path(self.temp.name) / "legacy-benchmarks.sqlite3"
        connection = sqlite3.connect(database)
        try:
            connection.execute(
                """CREATE TABLE benchmark_samples (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    deployment_id TEXT,
                    model_json TEXT NOT NULL,
                    runtime TEXT NOT NULL,
                    runtime_version TEXT,
                    hardware_json TEXT NOT NULL,
                    configuration_json TEXT NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    latency_ms REAL NOT NULL,
                    ttft_ms REAL,
                    generation_tps REAL,
                    prompt_tps REAL,
                    cold_start INTEGER,
                    eligible INTEGER NOT NULL
                )"""
            )
            connection.execute(
                """INSERT INTO benchmark_samples VALUES (
                    'legacy-row', '2026-08-25T00:00:00+00:00', NULL,
                    '{"repository":"org/model"}', 'vllm', NULL, '{}', '{}',
                    400, 32, 4000, 100, 80, NULL, 0, 0
                )"""
            )
            connection.commit()
        finally:
            connection.close()

        migrated = SparkDeckStore(database)
        try:
            columns = {
                row[1] for row in migrated._connection.execute(
                    "PRAGMA table_info(benchmark_samples)"
                )
            }
            self.assertIn("telemetry_cluster_id", columns)
            self.assertEqual(migrated.benchmarks()[1], 1)
            stored = migrated._connection.execute(
                "SELECT telemetry_cluster_id FROM benchmark_samples "
                "WHERE id = 'legacy-row'"
            ).fetchone()[0]
            self.assertIsNone(stored)
        finally:
            migrated.close()

    def test_legacy_device_class_is_normalized_for_local_and_upload_records(self):
        sample = BenchmarkSample(
            id="legacy-hardware", created_at="2026-08-25T00:00:00+00:00",
            deployment_id=None, model=ModelIdentity("org/model"),
            runtime=RuntimeKind.VLLM, runtime_version=None,
            hardware={"device_class": "dgx-spark"},
            configuration={"context_length": 4096},
            input_tokens=20, output_tokens=30, latency_ms=100, ttft_ms=10,
            generation_tokens_per_second=300, prompt_tokens_per_second=200,
            cold_start=False, eligible_for_community=True,
        )
        self.store.set_setting("device_pairing", {"status": "paired"})
        self.store.set_community_consent(True)
        self.store.add_benchmark(sample, queue=True)

        local, _ = self.store.benchmarks()
        upload = self.store.outbox_batch()

        self.assertEqual(local[0]["hardware"], {"hardware_class": "dgx-spark"})
        self.assertLessEqual(set(upload[0]), COMMUNITY_UPLOAD_FIELDS)
        self.assertNotIn("hardware", upload[0])

    def test_context_and_speed_are_required_at_the_upload_boundary(self):
        base = BenchmarkSample(
            id="missing-context", created_at="2026-08-25T00:00:00+00:00",
            deployment_id="dep-private", model=ModelIdentity("org/model"),
            runtime=RuntimeKind.VLLM, runtime_version="1.2.3",
            hardware={"hardware_class": "dgx-spark"}, configuration={},
            input_tokens=0, output_tokens=30, latency_ms=100, ttft_ms=10,
            generation_tokens_per_second=300, prompt_tokens_per_second=200,
            cold_start=False, eligible_for_community=True,
        )
        self.store.set_setting("device_pairing", {"status": "paired"})
        self.store.set_community_consent(True)
        self.store.add_benchmark(base, queue=True)
        self.store.add_benchmark(replace(
            base,
            id="missing-speed",
            input_tokens=400,
            generation_tokens_per_second=None,
        ), queue=True)

        local, total = self.store.benchmarks()
        self.assertEqual(total, 2)
        self.assertTrue(all(not row["eligible_for_community"] for row in local))
        self.assertEqual(self.store.outbox_batch(), [])
        self.assertEqual(self.store.sync_status()["outbox"]["pending"], 0)

    def test_community_evidence_policy_uses_only_upload_dimensions(self):
        self.assertEqual(COMMUNITY_EVIDENCE_POLICY, {
            "minimum_samples": 10,
            "exact_match_dimensions": [
                "model_id", "quantization", "prompt_tokens_bucket",
            ],
            "metric": "inference_tokens_per_second",
        })

    def test_coordinated_series_groups_exact_dimensions_without_filling_gaps(self):
        base = {
            "id": "run-1", "created_at": "2026-08-27T00:00:00+00:00",
            "deployment_id": "dep-1", "model_id": "org/model",
            "context_window_size": 4096, "concurrency": 1,
            "tensor_parallel_size": 2, "prompt_tokens_per_second": 1000.0,
            "generation_tokens_per_second": 80.0, "request_count": 4,
        }
        self.store.add_benchmark_series_point(base)
        self.store.add_benchmark_series_point({
            **base, "id": "run-2", "prompt_tokens_per_second": 1200.0,
            "generation_tokens_per_second": 100.0,
        })
        self.store.add_benchmark_series_point({
            **base, "id": "run-c5", "concurrency": 5,
            "prompt_tokens_per_second": 900.0,
            "generation_tokens_per_second": 150.0,
        })

        summaries = self.store.benchmark_model_summaries()
        detail = self.store.benchmark_model_detail("org/model")

        self.assertEqual(summaries[0]["run_count"], 3)
        self.assertEqual(summaries[0]["context_windows"], [4096])
        self.assertEqual(summaries[0]["tensor_parallel_sizes"], [2])
        self.assertEqual([point["concurrency"] for point in detail["points"]], [1, 5])
        self.assertEqual(detail["points"][0]["prompt_tokens_per_second"], 1100.0)
        self.assertEqual(detail["points"][0]["sample_count"], 2)

    def test_deleting_coordinated_history_removes_linked_series_point(self):
        created_at = "2026-08-27T00:00:00+00:00"
        point = {
            "id": "series-delete", "created_at": created_at,
            "deployment_id": "dep-1", "model_id": "org/model",
            "context_window_size": 4096, "concurrency": 2,
            "tensor_parallel_size": 2, "prompt_tokens_per_second": 1000.0,
            "generation_tokens_per_second": 80.0, "request_count": 4,
        }
        self.store.add_benchmark_series_point(point)
        self.store.add_benchmark_series_point({
            **point, "id": "series-keep",
            "created_at": "2026-08-27T00:01:00+00:00",
            "generation_tokens_per_second": 160.0,
        })
        sample = BenchmarkSample(
            id="history-delete", created_at=created_at,
            deployment_id="dep-1", model=ModelIdentity("org/model"),
            runtime=RuntimeKind.VLLM, runtime_version=None, hardware={},
            configuration={
                "context_length": 4096, "benchmark_concurrency": 2,
                "tensor_parallel_size": 2,
            },
            input_tokens=100, output_tokens=50, latency_ms=1000, ttft_ms=None,
            generation_tokens_per_second=80, prompt_tokens_per_second=1000,
            cold_start=False, eligible_for_community=True,
        )
        self.store.add_benchmark(sample, queue=False)

        linked_sample = self.store._connection.execute(
            "SELECT sample_id FROM benchmark_series_points WHERE id = ?",
            (point["id"],),
        ).fetchone()[0]
        self.assertEqual(linked_sample, sample.id)
        self.assertEqual(self.store.benchmark_model_summaries()[0]["run_count"], 2)
        self.assertEqual(
            self.store.benchmark_model_detail("org/model")["points"][0]["sample_count"],
            2,
        )

        self.assertTrue(self.store.delete_benchmark(sample.id))

        local, total = self.store.benchmarks()
        detail = self.store.benchmark_model_detail("org/model")
        self.assertEqual((local, total), ([], 0))
        self.assertEqual(self.store.benchmark_model_summaries()[0]["run_count"], 1)
        self.assertEqual(detail["points"][0]["sample_count"], 1)
        self.assertEqual(detail["points"][0]["generation_tokens_per_second"], 160.0)
        self.assertIsNone(self.store._connection.execute(
            "SELECT id FROM benchmark_series_points WHERE id = ?",
            (point["id"],),
        ).fetchone())

    def test_coordinated_point_and_history_roll_back_together(self):
        point = {
            "id": "duplicate-series", "created_at": "2026-08-27T00:00:00+00:00",
            "deployment_id": "dep-1", "model_id": "org/model",
            "context_window_size": 4096, "concurrency": 2,
            "tensor_parallel_size": 2, "prompt_tokens_per_second": 1000.0,
            "generation_tokens_per_second": 80.0, "request_count": 4,
        }
        self.store.add_benchmark_series_point(point)
        sample = BenchmarkSample(
            id="atomic-history", created_at=point["created_at"],
            deployment_id="dep-1", model=ModelIdentity("org/model"),
            runtime=RuntimeKind.VLLM, runtime_version=None, hardware={},
            configuration={
                "context_length": 4096, "benchmark_concurrency": 2,
                "tensor_parallel_size": 2,
            },
            input_tokens=100, output_tokens=50, latency_ms=1000, ttft_ms=None,
            generation_tokens_per_second=80, prompt_tokens_per_second=1000,
            cold_start=False, eligible_for_community=True,
        )

        with self.assertRaises(sqlite3.IntegrityError):
            self.store.add_coordinated_benchmark(point, sample, queue=True)

        local, total = self.store.benchmarks()
        self.assertEqual((local, total), ([], 0))
        self.assertEqual(self.store.benchmark_model_summaries()[0]["run_count"], 1)

    def test_migration_backfills_legacy_coordinated_history_link(self):
        database = Path(self.temp.name) / "legacy-series.sqlite3"
        connection = sqlite3.connect(database)
        try:
            connection.executescript(
                """
                CREATE TABLE benchmark_samples (
                    id TEXT PRIMARY KEY, created_at TEXT NOT NULL,
                    deployment_id TEXT, model_json TEXT NOT NULL,
                    runtime TEXT NOT NULL, runtime_version TEXT,
                    hardware_json TEXT NOT NULL, configuration_json TEXT NOT NULL,
                    input_tokens INTEGER NOT NULL, output_tokens INTEGER NOT NULL,
                    latency_ms REAL NOT NULL, ttft_ms REAL, generation_tps REAL,
                    prompt_tps REAL, cold_start INTEGER, eligible INTEGER NOT NULL
                );
                CREATE TABLE benchmark_series_points (
                    id TEXT PRIMARY KEY, created_at TEXT NOT NULL,
                    deployment_id TEXT, model_id TEXT NOT NULL,
                    context_window_size INTEGER NOT NULL, concurrency INTEGER NOT NULL,
                    tensor_parallel_size INTEGER NOT NULL, prompt_tps REAL NOT NULL,
                    generation_tps REAL NOT NULL, request_count INTEGER NOT NULL
                );
                """
            )
            values = (
                "legacy-history", "2026-08-27T00:00:00+00:00", "dep-1",
                '{"repository":"org/model"}', "vllm", None, "{}",
                '{"context_length":4096,"benchmark_concurrency":2,'
                '"tensor_parallel_size":2}',
                100, 50, 1000.0, None, 80.0, 1000.0, 0, 0,
            )
            connection.execute(
                "INSERT INTO benchmark_samples VALUES "
                "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                values,
            )
            connection.execute(
                "INSERT INTO benchmark_series_points VALUES "
                "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "legacy-series", values[1], "dep-1", "org/model",
                    4096, 2, 2, 1000.0, 80.0, 4,
                ),
            )
            connection.commit()
        finally:
            connection.close()

        migrated = SparkDeckStore(database)
        try:
            linked_sample = migrated._connection.execute(
                "SELECT sample_id FROM benchmark_series_points WHERE id = ?",
                ("legacy-series",),
            ).fetchone()[0]
            self.assertEqual(linked_sample, "legacy-history")
            self.assertTrue(migrated.delete_benchmark("legacy-history"))
            self.assertEqual(migrated.benchmark_model_summaries(), [])
        finally:
            migrated.close()

    def test_manual_non_c1_benchmarks_stay_local_and_are_not_queued(self):
        sample = BenchmarkSample(
            id="series-upload-c5", created_at="2026-08-27T00:00:00+00:00",
            deployment_id="dep-1", model=ModelIdentity(
                "org/model", quantization="Q4_K_M",
            ),
            runtime=RuntimeKind.VLLM, runtime_version=None, hardware={},
            configuration={
                "context_length": 16384, "benchmark_concurrency": 5,
                "tensor_parallel_size": 2, "private": "drop-me",
            },
            input_tokens=100, output_tokens=50, latency_ms=1000, ttft_ms=None,
            generation_tokens_per_second=50, prompt_tokens_per_second=100,
            cold_start=False, eligible_for_community=True,
        )
        self.store.set_setting("device_pairing", {"status": "paired"})
        consent = self.store.set_community_consent(
            True, "11111111-1111-4111-8111-111111111111",
        )
        for concurrency in (2, 5, 10):
            self.store.add_benchmark(replace(
                sample,
                id=f"series-upload-c{concurrency}",
                configuration={
                    **sample.configuration,
                    "benchmark_concurrency": concurrency,
                },
            ), queue=True)
        self.assertFalse(self.store.add_benchmark_if_consented(
            replace(sample, id="atomic-c5"), consent["generation"],
        ))

        local, total = self.store.benchmarks()
        self.assertTrue(consent["enabled"])
        self.assertEqual(total, 3)
        self.assertTrue(all(not row["eligible_for_community"] for row in local))
        self.assertEqual(self.store.outbox_batch(), [])

    def test_upload_omits_untrusted_benchmark_dimensions_for_ordinary_sample(self):
        sample = BenchmarkSample(
            id="ordinary-tp", created_at="2026-08-27T00:00:00+00:00",
            deployment_id="dep-1", model=ModelIdentity("org/model"),
            runtime=RuntimeKind.VLLM, runtime_version=None, hardware={},
            configuration={
                "context_length": 16384, "tensor_parallel_size": 4,
            },
            input_tokens=100, output_tokens=50, latency_ms=1000, ttft_ms=None,
            generation_tokens_per_second=50, prompt_tokens_per_second=100,
            cold_start=False, eligible_for_community=True,
        )
        self.store.set_setting("device_pairing", {"status": "paired"})
        consent = self.store.set_community_consent(True)
        self.assertTrue(self.store.add_benchmark_if_consented(
            sample, consent["generation"],
        ))

        local, _ = self.store.benchmarks()
        self.assertEqual(local[0]["configuration"]["tensor_parallel_size"], 4)
        self.assertEqual(self.store.outbox_batch(), [{
            "model_id": "org/model", "quantization": "UNKNOWN",
            "prompt_tokens_bucket": 400,
            "inference_tokens_per_second": 50.0,
            "telemetry_cluster_id": consent["telemetry_cluster_id"],
            "concurrency": 1,
        }])

    def test_local_community_aggregates_group_only_privacy_eligible_rows(self):
        cluster_id = "22222222-2222-4222-8222-222222222222"
        consent = self.store.set_community_consent(True, cluster_id)
        sample = BenchmarkSample(
            id="eligible-1", created_at="2026-08-25T00:00:00+00:00",
            deployment_id="private-deployment", model=ModelIdentity(
                "org/model", revision="private-revision",
                artifact="C:/private/model.gguf", quantization="NVFP4",
            ),
            runtime=RuntimeKind.VLLM, runtime_version="private/image:latest",
            hardware={"hardware_class": "private-device"},
            configuration={"context_length": 4096, "api_key": "private"},
            input_tokens=20, output_tokens=30, latency_ms=100, ttft_ms=10,
            generation_tokens_per_second=80, prompt_tokens_per_second=200,
            cold_start=False, eligible_for_community=True,
        )
        candidates = [sample, replace(
            sample, id="eligible-2", generation_tokens_per_second=100,
        ), replace(
            sample, id="invalid-coordinated-marker",
            configuration={"context_length": 4096, "benchmark_concurrency": 3},
            generation_tokens_per_second=90,
        ), replace(
            sample, id="ineligible", eligible_for_community=False,
            generation_tokens_per_second=1000,
        ), replace(
            sample, id="other-context", input_tokens=2200,
            generation_tokens_per_second=40,
        ), replace(
            sample, id="coordinated-c1", configuration={
                "context_length": 4096, "benchmark_concurrency": 1,
                "tensor_parallel_size": 1,
            }, generation_tokens_per_second=500,
        ), replace(
            sample, id="coordinated-c10", configuration={
                "context_length": 4096, "benchmark_concurrency": 10,
                "tensor_parallel_size": 2,
            }, generation_tokens_per_second=5,
        )]
        for candidate in candidates:
            self.store.add_benchmark_if_consented(
                candidate, consent["generation"],
            )
        with self.store._connection:
            self.store._connection.execute(
                "UPDATE benchmark_samples SET telemetry_cluster_id = ? "
                "WHERE id = 'eligible-2'",
                ("44444444-4444-4444-8444-444444444444",),
            )

        aggregates = self.store.community_aggregates()

        self.assertEqual(aggregates, [
            {
                "model_id": "org/model",
                "quantization": "NVFP4",
                "prompt_tokens_bucket": 400,
                "inference_tokens_per_second": 226.66666666666666,
                "sample_count": 3,
                "unique_cluster_count": 2,
            },
            {
                "model_id": "org/model",
                "quantization": "NVFP4",
                "prompt_tokens_bucket": 2000,
                "inference_tokens_per_second": 40.0,
                "sample_count": 1,
                "unique_cluster_count": 1,
            },
        ])
        serialized = str(aggregates)
        for private_value in (
            "private-deployment", "private-revision", "private/image",
            "private-device", "C:/private", "api_key",
        ):
            self.assertNotIn(private_value, serialized)

    def test_local_community_aggregates_scan_large_history_in_bounded_batches(self):
        row_count = _COMMUNITY_AGGREGATE_BATCH_SIZE * 20 + 17
        rows = [{
            "model_json": '{"repository":"org/model","quantization":"NVFP4"}',
            "configuration_json": '{"context_length":4096}',
            "input_tokens": 400,
            "generation_tps": 80.0,
            "telemetry_cluster_id": "33333333-3333-4333-8333-333333333333",
        } for _ in range(row_count)]

        class BatchCursor:
            def __init__(self):
                self.offset = 0
                self.batch_sizes = []

            def fetchmany(self, size):
                self.batch_sizes.append(size)
                batch = rows[self.offset:self.offset + size]
                self.offset += len(batch)
                return batch

            def fetchall(self):
                raise AssertionError("community aggregation must not buffer all rows")

        cursor = BatchCursor()
        store = SparkDeckStore.__new__(SparkDeckStore)
        store._lock = threading.RLock()
        store._connection = type("Connection", (), {
            "execute": lambda self, _query: cursor,
        })()

        aggregates = store.community_aggregates()

        self.assertEqual(aggregates, [{
            "model_id": "org/model",
            "quantization": "NVFP4",
            "prompt_tokens_bucket": 400,
            "inference_tokens_per_second": 80.0,
            "sample_count": row_count,
            "unique_cluster_count": 1,
        }])
        self.assertGreater(len(cursor.batch_sizes), 20)
        self.assertEqual(set(cursor.batch_sizes), {_COMMUNITY_AGGREGATE_BATCH_SIZE})

    def test_migration_removes_invalid_legacy_upload_but_keeps_local_sample(self):
        sample = BenchmarkSample(
            id="legacy-invalid", created_at="2026-08-25T00:00:00+00:00",
            deployment_id=None, model=ModelIdentity("org/model"),
            runtime=RuntimeKind.VLLM, runtime_version=None,
            hardware={}, configuration={}, input_tokens=20, output_tokens=30,
            latency_ms=100, ttft_ms=10, generation_tokens_per_second=300,
            prompt_tokens_per_second=200, cold_start=False,
            eligible_for_community=True,
        )
        self.store.add_benchmark(sample, queue=False)
        database = self.store.path
        self.store.close()
        connection = sqlite3.connect(database)
        try:
            connection.execute(
                "UPDATE benchmark_samples SET eligible = 1 WHERE id = ?",
                (sample.id,),
            )
            connection.execute(
                "INSERT INTO upload_outbox(sample_id, status, created_at) "
                "VALUES (?, 'pending', ?)",
                (sample.id, sample.created_at),
            )
            connection.commit()
        finally:
            connection.close()

        self.store = SparkDeckStore(database)

        local, total = self.store.benchmarks()
        self.assertEqual(total, 1)
        self.assertFalse(local[0]["eligible_for_community"])
        self.assertEqual(local[0]["configuration"], {})
        self.assertEqual(self.store.sync_status()["outbox"]["pending"], 0)
