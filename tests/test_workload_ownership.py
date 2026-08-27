import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import docker

from manager import CONTROLLER_LABEL, DEPLOYMENT_LABEL, Manager
from sparkdeck.workload_ownership import ManagedWorkloadLedger


class ManagedWorkloadLedgerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.ledger = ManagedWorkloadLedger(Path(self.temp.name))

    def tearDown(self):
        self.temp.cleanup()

    def test_reconcile_backfills_observed_and_prunes_unobserved_claims(self):
        self.ledger.claim("pending", "deployment-pending")
        self.ledger.confirm("stale", "deployment-stale")

        self.ledger.reconcile({"observed": "deployment-observed"})

        claims = self.ledger.snapshot()
        self.assertEqual(set(claims), {"observed"})
        self.assertEqual(claims["observed"]["state"], "active")

    def test_corrupt_ledger_fails_closed(self):
        self.ledger.path.write_text("not-json", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "cannot verify managed workload ownership"):
            self.ledger.snapshot()


class ManagerManagedWorkloadLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.manager = Manager.__new__(Manager)
        self.manager.managed_workload_ledger = ManagedWorkloadLedger(Path(self.temp.name))
        self.manager.client = Mock()

    async def asyncTearDown(self):
        self.temp.cleanup()

    @staticmethod
    def run_options(name="sparkdeck-model"):
        return {
            "name": name,
            "image": "example/runtime:latest",
            "labels": {
                CONTROLLER_LABEL: "1",
                DEPLOYMENT_LABEL: "deployment-1",
            },
        }

    async def test_claim_is_durable_before_docker_create(self):
        container = Mock(labels={
            CONTROLLER_LABEL: "1", DEPLOYMENT_LABEL: "deployment-1",
        })

        def run(**_options):
            claim = self.manager.managed_workload_ledger.snapshot()["sparkdeck-model"]
            self.assertEqual(claim["state"], "pending")
            return container

        self.manager.client.containers.run.side_effect = run

        result = self.manager._run_managed_container(self.run_options())

        self.assertIs(result, container)
        self.assertEqual(
            self.manager.managed_workload_ledger.snapshot()["sparkdeck-model"]["state"],
            "active",
        )

    async def test_ambiguous_create_failure_retains_pending_claim(self):
        self.manager.client.containers.run.side_effect = docker.errors.DockerException(
            "daemon disconnected"
        )
        self.manager.client.containers.get.side_effect = docker.errors.DockerException(
            "daemon unavailable"
        )

        with self.assertRaises(docker.errors.DockerException):
            self.manager._run_managed_container(self.run_options())

        self.assertEqual(
            self.manager.managed_workload_ledger.snapshot()["sparkdeck-model"]["state"],
            "pending",
        )

    async def test_successful_remove_releases_claim(self):
        self.manager.managed_workload_ledger.confirm("sparkdeck-model", "deployment-1")
        self.manager.client.containers.get.return_value = Mock()

        await self.manager.remove_container("sparkdeck-model")

        self.assertEqual(self.manager.managed_workload_ledger.snapshot(), {})

    async def test_successful_inventory_backfills_and_prunes_claims(self):
        self.manager._container_summary = Mock(return_value={
            "name": "legacy-managed", "managed": True, "status": "exited",
            "deployment_id": "legacy-deployment",
        })
        self.manager._get_container_phase = AsyncMock(return_value={"phase": "exited"})
        self.manager.client.containers.list.return_value = [Mock()]

        await self.manager.list_containers()

        self.assertEqual(
            set(self.manager.managed_workload_ledger.snapshot()), {"legacy-managed"},
        )
        self.manager.client.containers.list.return_value = []
        await self.manager.list_containers()
        self.assertEqual(self.manager.managed_workload_ledger.snapshot(), {})

    async def test_not_found_remove_releases_claim_but_still_reports_absent(self):
        self.manager.managed_workload_ledger.confirm("sparkdeck-model", "deployment-1")
        self.manager.client.containers.get.side_effect = docker.errors.NotFound("missing")

        with self.assertRaises(docker.errors.NotFound):
            await self.manager.remove_container("sparkdeck-model")

        self.assertEqual(self.manager.managed_workload_ledger.snapshot(), {})

    async def test_docker_outage_during_member_removal_is_not_reported_absent(self):
        self.manager.client.containers.get.side_effect = docker.errors.DockerException(
            "daemon unavailable"
        )
        self.manager.cluster_member_launches = {}

        with self.assertRaises(docker.errors.DockerException):
            await self.manager.remove_cluster_member("sparkdeck-model")
