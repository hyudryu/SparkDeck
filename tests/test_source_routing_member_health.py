import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from manager import Manager, SourceRoutingUnavailable
from sparkdeck.models import Deployment, DeploymentKind, ModelIdentity, RuntimeKind
from sparkdeck.service import SparkDeckService


class MemberHealthRoutingTests(unittest.IsolatedAsyncioTestCase):
    def fixture(self, directory, mode):
        nodes = ["local"] if mode == "single" else ["local", "worker"]
        manager = Manager.__new__(Manager)
        manager.http = None
        manager.source_ip_routing_rules = {}
        manager.source_ip_routing_rules_path = Path(directory) / "rules.json"
        manager.deployments = [{
            "id": "runtime", "sparkdeck_record_id": "saved", "model": "org/model",
            "mode": mode, "status": "stopped", "desired_state": "running",
            "members": [{"node_id": node, "rank": rank, "container_name": f"rank-{rank}",
                         "status": "exited", "agent_token": "private"}
                        for rank, node in enumerate(nodes)],
        }]
        manager._container_by_name = AsyncMock(return_value={"name": "rank-0", "status": "running"})
        manager._check_ready = AsyncMock(return_value=True)
        manager.node_registry = SimpleNamespace(request=AsyncMock(return_value={"status": "running", "ready": None}))
        manager._proxy_cluster_member = AsyncMock(return_value={"choices": [], "usage": {}})
        service = SparkDeckService(manager, Path(directory))
        service.store.add_deployment(Deployment(
            id="saved", alias="model", runtime=RuntimeKind.VLLM,
            kind=DeploymentKind.MANAGED, model=ModelIdentity("org/model"),
            settings={"manager_deployment_id": "runtime"},
        ))
        service._record_response = Mock()
        service._managed_hardware_snapshot = AsyncMock(return_value=({}, True))
        rule = {"source_ip": "192.0.2.1", "requested_model": "model", "enabled": True,
                "deployment_id": "saved", "instance_id": None, "node_ids": nodes}
        return manager, service, rule

    async def test_live_health_allows_enable_and_inference_despite_saved_exited_in_both_modes(self):
        for mode in ("single", "sharded"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                manager, service, rule = self.fixture(directory, mode)
                try:
                    saved = await service.upsert_source_ip_routing_rule(rule)
                    self.assertEqual(saved, rule)
                    snapshot = await service._source_routed_deployment(rule, "model")
                    self.assertNotIn("agent_token", str(snapshot["_source_routing_members"]))
                    self.assertNotIn("observed", manager.source_ip_routing_rules_path.read_text())
                    await service.proxy({"model": "model"}, "completions", caller_ip="192.0.2.1")
                    self.assertEqual(manager._proxy_cluster_member.await_count, 1)
                    service._source_routing_cache.clear()
                    manager._check_ready.return_value = False
                    with self.assertRaises(SourceRoutingUnavailable):
                        await service.proxy({"model": "model"}, "completions", caller_ip="192.0.2.1")
                    manager._check_ready.return_value = True
                    service._source_routing_cache.clear()
                    manager.deployments[0]["members"][0]["desired_state"] = "stopped"
                    with self.assertRaises(SourceRoutingUnavailable):
                        await service.proxy({"model": "model"}, "completions", caller_ip="192.0.2.1")
                    manager.deployments[0]["members"][0].pop("desired_state")
                    manager.deployments[0]["members"][0]["node_id"] = "moved"
                    with self.assertRaises(SourceRoutingUnavailable):
                        await service.proxy({"model": "model"}, "completions", caller_ip="192.0.2.1")
                    manager.deployments[0]["members"][0]["node_id"] = "local"
                    manager.deployments[0]["id"] = "replacement-runtime"
                    with self.assertRaises(SourceRoutingUnavailable):
                        await service.proxy({"model": "model"}, "completions", caller_ip="192.0.2.1")
                    self.assertEqual(manager._proxy_cluster_member.await_count, 1)
                finally:
                    await service.close()

    async def test_missing_or_unhealthy_observation_blocks_even_if_saved_running(self):
        for mode in ("single", "sharded"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                manager, service, rule = self.fixture(directory, mode)
                try:
                    target = await service._source_routed_deployment(rule, "model")
                    observed = target["_source_routing_members"]
                    deployment = manager.deployments[0]
                    for member in deployment["members"]:
                        member["status"] = "running"
                    malformed = [None, [], deepcopy(observed)]
                    malformed[-1][0]["status"] = "unknown"
                    if mode == "sharded":
                        malformed.extend([observed[:-1], list(reversed(observed)), [observed[0], observed[0]]])
                    for members in malformed:
                        with self.subTest(observation=members), self.assertRaises(SourceRoutingUnavailable):
                            manager._source_route_candidates(deployment, {**rule, "_observed_members": members})
                    service._source_routing_cache.clear()
                    manager._check_ready.return_value = False
                    with self.assertRaises((SourceRoutingUnavailable, LookupError)):
                        await service.upsert_source_ip_routing_rule(rule)
                    manager._proxy_cluster_member.assert_not_awaited()
                finally:
                    await service.close()
