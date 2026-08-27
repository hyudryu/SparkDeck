import sqlite3
import tempfile
import threading
import unittest
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from sparkdeck.models import BenchmarkSample, Deployment, DeploymentKind, ModelIdentity, RuntimeKind
from sparkdeck.storage import (
    COMMUNITY_EVIDENCE_POLICY,
    COMMUNITY_UPLOAD_FIELDS,
    SparkDeckStore,
    _COMMUNITY_AGGREGATE_BATCH_SIZE,
)


class SparkDeckStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = SparkDeckStore(Path(self.temp.name) / "sparkdeck.sqlite3")

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

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
            deployment_id=None, model=ModelIdentity("org/model"),
            runtime=RuntimeKind.VLLM, runtime_version=None,
            hardware={"architecture": "x86_64"},
            configuration={"context_length": 4096},
            input_tokens=20, output_tokens=30, latency_ms=100,
            ttft_ms=10, generation_tokens_per_second=300,
            prompt_tokens_per_second=200, cold_start=False,
            eligible_for_community=True,
        )
        self.store.add_benchmark(sample, queue=False)
        self.assertEqual(self.store.sync_status()["outbox"]["waiting_for_account"], 0)
        self.store.set_setting("community_consent", True)
        second = replace(sample, id="sample-2")
        self.store.add_benchmark(second, queue=True)
        self.assertEqual(self.store.sync_status()["outbox"]["waiting_for_account"], 1)
        self.store.set_setting("device_pairing", {"status": "paired", "device_id": "device-1"})
        self.assertEqual(self.store.retry_outbox(), 1)
        self.assertEqual(self.store.outbox_batch(), [{
            "model_id": "org/model",
            "context_window_size": 4096,
            "inference_tokens_per_second": 300.0,
        }])
        self.assertEqual(self.store.mark_outbox_failed(["sample-2"], "offline"), 1)
        self.assertEqual(self.store.outbox_batch(), [])
        self.assertEqual(self.store.retry_outbox(), 1)
        self.assertEqual(self.store.mark_outbox_synced(["sample-2"]), 1)
        self.assertEqual(self.store.sync_status()["outbox"]["synced"], 1)

    def test_sync_status_redacts_pairing_claims(self):
        self.store.set_setting("device_pairing", {
            "status": "paired", "sub": "stable-sub", "email": "person@example.com",
            "credential": "secret",
        })

        status = self.store.sync_status()

        self.assertEqual(status["pairing"], {"status": "paired"})
        self.assertNotIn("stable-sub", str(status))
        self.assertNotIn("person@example.com", str(status))
        self.assertNotIn("secret", str(status))

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
        self.store.set_setting("community_consent", True)
        self.store.add_benchmark(sample, queue=True)
        self.assertEqual(self.store.sync_status()["outbox"]["waiting_for_account"], 1)

        self.assertEqual(self.store.promote_outbox_for_pairing(), 1)

        self.assertEqual(self.store.sync_status()["outbox"]["waiting_for_account"], 0)
        self.assertEqual(self.store.sync_status()["outbox"]["pending"], 1)
        self.assertEqual(self.store.promote_outbox_for_pairing(), 0)

    def test_consent_queues_existing_samples_and_upload_drops_artifact(self):
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
        rows = self.store.outbox_batch()
        self.assertEqual(set(rows[0]), COMMUNITY_UPLOAD_FIELDS)
        self.assertEqual(rows, [{
            "model_id": "org/model",
            "context_window_size": 4096,
            "inference_tokens_per_second": 300.0,
        }])
        local, _ = self.store.benchmarks()
        self.assertEqual(local[0]["model"]["artifact"], "C:/private/model.gguf")
        self.assertEqual(local[0]["model"]["revision"], "abc123")
        self.assertEqual(local[0]["runtime_version"], "registry.local/team/image:1")
        self.assertEqual(local[0]["hardware"], {"architecture": "aarch64"})
        self.assertEqual(local[0]["sync_state"], "pending")

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

        self.store.set_community_consent(True)
        uploads = self.store.outbox_batch()
        self.assertEqual(len(uploads), 2)
        self.assertTrue(all(set(item) == COMMUNITY_UPLOAD_FIELDS for item in uploads))

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
        self.assertEqual(set(upload[0]), COMMUNITY_UPLOAD_FIELDS)
        self.assertNotIn("hardware", upload[0])

    def test_context_and_speed_are_required_at_the_upload_boundary(self):
        base = BenchmarkSample(
            id="missing-context", created_at="2026-08-25T00:00:00+00:00",
            deployment_id="dep-private", model=ModelIdentity("org/model"),
            runtime=RuntimeKind.VLLM, runtime_version="1.2.3",
            hardware={"hardware_class": "dgx-spark"}, configuration={},
            input_tokens=20, output_tokens=30, latency_ms=100, ttft_ms=10,
            generation_tokens_per_second=300, prompt_tokens_per_second=200,
            cold_start=False, eligible_for_community=True,
        )
        self.store.set_setting("device_pairing", {"status": "paired"})
        self.store.set_community_consent(True)
        self.store.add_benchmark(base, queue=True)
        self.store.add_benchmark(replace(
            base,
            id="missing-speed",
            configuration={"max_model_len": 8192},
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
            "exact_match_dimensions": ["model_id", "context_window_size"],
            "metric": "inference_tokens_per_second",
        })

    def test_local_community_aggregates_group_only_privacy_eligible_rows(self):
        sample = BenchmarkSample(
            id="eligible-1", created_at="2026-08-25T00:00:00+00:00",
            deployment_id="private-deployment", model=ModelIdentity(
                "org/model", revision="private-revision",
                artifact="C:/private/model.gguf",
            ),
            runtime=RuntimeKind.VLLM, runtime_version="private/image:latest",
            hardware={"hardware_class": "private-device"},
            configuration={"context_length": 4096, "api_key": "private"},
            input_tokens=20, output_tokens=30, latency_ms=100, ttft_ms=10,
            generation_tokens_per_second=80, prompt_tokens_per_second=200,
            cold_start=False, eligible_for_community=True,
        )
        self.store.add_benchmark(sample, queue=False)
        self.store.add_benchmark(replace(
            sample, id="eligible-2", generation_tokens_per_second=100,
        ), queue=False)
        self.store.add_benchmark(replace(
            sample, id="ineligible", eligible_for_community=False,
            generation_tokens_per_second=1000,
        ), queue=False)
        self.store.add_benchmark(replace(
            sample, id="other-context", configuration={"max_model_len": 8192},
            generation_tokens_per_second=40,
        ), queue=False)

        aggregates = self.store.community_aggregates()

        self.assertEqual(aggregates, [
            {
                "model_id": "org/model",
                "context_window_size": 4096,
                "inference_tokens_per_second": 90.0,
                "sample_count": 2,
            },
            {
                "model_id": "org/model",
                "context_window_size": 8192,
                "inference_tokens_per_second": 40.0,
                "sample_count": 1,
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
            "model_json": '{"repository":"org/model"}',
            "configuration_json": '{"context_length":4096}',
            "generation_tps": 80.0,
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
            "context_window_size": 4096,
            "inference_tokens_per_second": 80.0,
            "sample_count": row_count,
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
