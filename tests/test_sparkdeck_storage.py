import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from sparkdeck.models import BenchmarkSample, Deployment, DeploymentKind, ModelIdentity, RuntimeKind
from sparkdeck.storage import SparkDeckStore


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

    def test_only_consented_eligible_samples_enter_outbox(self):
        sample = BenchmarkSample(
            id="sample-1", created_at="2026-08-25T00:00:00+00:00",
            deployment_id=None, model=ModelIdentity("org/model"),
            runtime=RuntimeKind.VLLM, runtime_version=None,
            hardware={"architecture": "x86_64"}, configuration={},
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
        self.assertEqual([item["id"] for item in self.store.outbox_batch()], ["sample-2"])
        self.assertEqual(self.store.mark_outbox_failed(["sample-2"], "offline"), 1)
        self.assertEqual(self.store.outbox_batch(), [])
        self.assertEqual(self.store.retry_outbox(), 1)
        self.assertEqual(self.store.mark_outbox_synced(["sample-2"]), 1)
        self.assertEqual(self.store.sync_status()["outbox"]["synced"], 1)

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
        self.assertEqual([item["id"] for item in rows], ["sample-private"])
        self.assertNotIn("artifact", rows[0]["model"])
        self.assertIsNone(rows[0]["runtime_version"])
        local, _ = self.store.benchmarks()
        self.assertEqual(local[0]["sync_state"], "pending")
