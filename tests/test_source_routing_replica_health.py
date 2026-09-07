import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from manager import Manager, SourceRoutingUnavailable
from sparkdeck.models import Deployment, DeploymentKind, ModelIdentity, RuntimeKind
from sparkdeck.service import SparkDeckService, _replica_summary


def cluster():
    return {
        "id": "runtime", "sparkdeck_record_id": "saved", "mode": "replicated",
        "status": "degraded", "desired_state": "running",
        "members": [
            {"node_id": "a", "node_name": "Node A", "rank": 0,
             "status": "running", "node_status": "online", "node_docker_ready": True},
            {"node_id": "b", "node_name": "Node B", "rank": 1,
             "status": "exited", "node_status": "online", "node_docker_ready": True},
        ],
    }


def rule(node="b", enabled=True):
    return {"source_ip": "192.0.2.1", "requested_model": "model", "enabled": enabled,
            "deployment_id": "saved", "instance_id": None, "node_ids": [node]}


class ReplicaHealthTests(unittest.IsolatedAsyncioTestCase):
    async def test_live_restarted_replica_overrides_saved_exited_status_for_save_and_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            saved = cluster()
            saved["members"][0]["status"] = "exited"
            observed = deepcopy(saved)
            observed["members"][0]["status"] = "running"
            manager = Manager.__new__(Manager)
            manager.http = None
            manager.deployments = [saved]
            manager.source_ip_routing_rules = {}
            manager.source_ip_routing_rules_path = Path(directory) / "rules.json"
            manager._proxy_cluster_member = AsyncMock(return_value={"choices": [], "usage": {}})
            service = SparkDeckService(manager, Path(directory))
            try:
                service.store.add_deployment(Deployment(
                    id="saved", alias="model", runtime=RuntimeKind.VLLM,
                    kind=DeploymentKind.MANAGED, model=ModelIdentity("org/model"),
                    settings={"manager_deployment_id": "runtime"},
                ))
                live = {**service.store.deployment("saved"), "status": "degraded",
                        "desired_state": "running", "deployment_mode": "replicated",
                        "replicas": _replica_summary(observed)}
                service.deployments = AsyncMock(return_value=[live])
                service._record_response = Mock()
                service._managed_hardware_snapshot = AsyncMock(return_value=({}, True))
                await service.upsert_source_ip_routing_rule(rule("a"))
                self.assertNotIn("_observed_replicas", manager.list_source_ip_routing_rules()[0])
                await service.proxy({"model": "model"}, "completions", caller_ip="192.0.2.1")
                self.assertEqual(manager._proxy_cluster_member.await_count, 1)
                # Explicit stopped intent remains authoritative even when
                # the last observed container was running.
                saved["members"][0]["desired_state"] = "stopped"
                with self.assertRaises(SourceRoutingUnavailable):
                    await service.proxy({"model": "model"}, "completions", caller_ip="192.0.2.1")
                saved["members"][0].pop("desired_state")
                observed["members"][0]["status"] = "exited"
                live["replicas"] = _replica_summary(observed)
                with self.assertRaises(LookupError):
                    await service.upsert_source_ip_routing_rule(rule("a"))
                with self.assertRaises(SourceRoutingUnavailable):
                    await service.proxy({"model": "model"}, "completions", caller_ip="192.0.2.1")
                self.assertEqual(manager._proxy_cluster_member.await_count, 1)
            finally:
                await service.close()

    async def test_deployment_list_exposes_observed_replica_health(self):
        with tempfile.TemporaryDirectory() as directory:
            observed = cluster()
            saved = deepcopy(observed)
            saved["members"][0]["status"] = "starting"
            manager = SimpleNamespace(
                http=None, deployments=[saved],
                get_state=AsyncMock(return_value={"deployments": [observed], "containers": [], "docker_ready": True}),
                cluster_nodes=AsyncMock(return_value=[]),
                list_containers=AsyncMock(return_value=[]),
            )
            service = SparkDeckService(manager, Path(directory))
            try:
                service.store.add_deployment(Deployment(
                    id="saved", alias="model", runtime=RuntimeKind.VLLM,
                    kind=DeploymentKind.MANAGED, model=ModelIdentity("org/model"),
                    settings={"manager_deployment_id": "runtime"},
                ))
                result = await service.deployments()
                self.assertEqual([replica["available"] for replica in result[0]["replicas"]], [True, False])
                self.assertEqual(result[0]["replicas"][0]["status"], "running")
            finally:
                await service.close()

    def test_public_replica_summary_preserves_individual_health_and_hides_private_fields(self):
        deployment = cluster()
        deployment["members"][0]["agent_token"] = "private"
        result = _replica_summary(deployment)
        self.assertEqual([item["available"] for item in result], [True, False])
        self.assertEqual(result[1]["status"], "exited")
        self.assertEqual(result[1]["node_id"], "b")
        self.assertNotIn("agent_token", result[0])
        for update in ({"node_status": "offline"}, {"node_status": "unknown"},
                       {"node_status": "disconnected"},
                       {"node_docker_ready": False}, {"desired_state": "stopped"},
                       {"status": "starting"}):
            changed = deepcopy(deployment)
            changed["members"][0].update(update)
            with self.subTest(update=update):
                self.assertFalse(_replica_summary(changed)[0]["available"])

    def test_enabled_save_rejects_failed_replica_but_stale_disable_is_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = Manager.__new__(Manager)
            manager.deployments = [cluster()]
            manager.source_ip_routing_rules = {}
            manager.source_ip_routing_rules_path = Path(directory) / "rules.json"
            with self.assertRaisesRegex(ValueError, "unavailable"):
                manager.upsert_source_ip_routing_rule(rule())
            self.assertEqual(manager.source_ip_routing_rules, {})
            manager.upsert_source_ip_routing_rule(rule("a"))
            manager.deployments = []
            manager.upsert_source_ip_routing_rule(rule("a", enabled=False))
            self.assertIsNone(manager.source_ip_routing_rule("192.0.2.1", "model"))

    def test_persisted_startup_phase_does_not_override_live_routing_probe(self):
        manager = Manager.__new__(Manager)
        deployment = cluster()
        deployment["members"][0].update(status="starting")
        deployment["members"][0].pop("node_status")
        manager.deployments = [deployment]
        self.assertEqual(manager._source_route_candidates(deployment, rule("a"))[0]["node_id"], "a")
        deployment["members"][0]["node_status"] = "disconnected"
        with self.assertRaises(SourceRoutingUnavailable):
            manager._source_route_candidates(deployment, rule("a"))

    async def test_service_uses_observed_replica_health_in_degraded_deployment(self):
        deployment = cluster()
        stored = {"id": "saved", "alias": "model", "kind": "managed", "runtime": "vllm",
                  "model": {"repository": "org/model"}, "served_models": ["model"],
                  "settings": {"manager_deployment_id": "runtime"}}
        live = {**stored, "status": "degraded", "desired_state": "running",
                "deployment_mode": "replicated", "replicas": _replica_summary(deployment)}
        service = SparkDeckService.__new__(SparkDeckService)
        service.store = SimpleNamespace(deployment=Mock(return_value=stored))
        service.manager = SimpleNamespace(deployments=[deployment])
        service.deployments = AsyncMock(return_value=[live])
        selected = await service._source_routed_deployment(rule("a"), "model", validating=True)
        self.assertEqual(selected["id"], "saved")
        with self.assertRaisesRegex(LookupError, "replica is unavailable"):
            await service._source_routed_deployment(rule("b"), "model", validating=True)
        with self.assertRaisesRegex(SourceRoutingUnavailable, "replica is unavailable"):
            await service._source_routed_deployment(rule("b"), "model")
