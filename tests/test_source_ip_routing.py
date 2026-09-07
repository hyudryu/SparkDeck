import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import httpx

from manager import Manager, SourceRoutingUnavailable
from sparkdeck.models import Deployment, DeploymentKind, ModelIdentity, RuntimeKind
from sparkdeck.service import SparkDeckService


def _replicated_deployment():
    return {
        "id": "manager-1",
        "sparkdeck_record_id": "record-1",
        "mode": "replicated",
        "desired_state": "running",
        "members": [
            {"node_id": "node-a", "rank": 0, "status": "running"},
            {"node_id": "node-b", "rank": 1, "status": "running"},
        ],
    }


def _rule(**updates):
    value = {
        "source_ip": "2001:0db8::1",
        "requested_model": "shared-model",
        "enabled": True,
        "deployment_id": "record-1",
        "instance_id": None,
        "node_ids": ["node-b"],
    }
    value.update(updates)
    return value


def _grouped_deployment(manager_id="manager-1", record_id="record-1"):
    members = []
    for instance_id, nodes in enumerate((("node-a", "node-b"), ("node-c", "node-d"))):
        for rank, node_id in enumerate(nodes):
            members.append({
                "node_id": node_id,
                "rank": rank,
                "instance_id": instance_id,
                "status": "running",
                "desired_state": "running",
            })
    return {
        "id": manager_id,
        "sparkdeck_record_id": record_id,
        "mode": "grouped_sharded",
        "desired_state": "running",
        "status": "running",
        "model": "org/shared",
        "members": members,
    }


class SourceRoutingPersistenceTests(unittest.TestCase):
    def manager(self, directory):
        manager = Manager.__new__(Manager)
        manager.source_ip_routing_rules_path = Path(directory) / "rules.json"
        manager.source_ip_routing_rules = {}
        manager.deployments = [_replicated_deployment()]
        return manager

    def test_versioned_round_trip_is_canonical_and_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = self.manager(directory)
            saved = manager.upsert_source_ip_routing_rule(_rule())
            again = manager.upsert_source_ip_routing_rule(_rule(
                source_ip="2001:db8:0:0:0:0:0:1",
            ))

            self.assertEqual(saved, again)
            self.assertEqual(saved["source_ip"], "2001:db8::1")
            payload = json.loads(
                manager.source_ip_routing_rules_path.read_text(encoding="utf-8")
            )
            self.assertEqual(payload, {"version": 1, "rules": [saved]})
            reloaded = self.manager(directory)
            reloaded.source_ip_routing_rules = reloaded._load_source_ip_routing_rules()
            self.assertEqual(reloaded.list_source_ip_routing_rules(), [saved])

    def test_rejects_network_zone_unknown_and_wrong_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = self.manager(directory)
            for source in ("10.0.0.0/24", "fe80::1%eth0", "unknown"):
                with self.subTest(source=source), self.assertRaises(ValueError):
                    manager.upsert_source_ip_routing_rule(_rule(source_ip=source))
            with self.assertRaisesRegex(LookupError, "replica was not found"):
                manager.upsert_source_ip_routing_rule(_rule(node_ids=["node-x"]))

    def test_failed_atomic_write_does_not_publish_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = self.manager(directory)
            with patch(
                "manager._atomic_private_json_write",
                side_effect=OSError("disk full"),
            ), self.assertRaises(OSError):
                manager.upsert_source_ip_routing_rule(_rule())
            self.assertEqual(manager.source_ip_routing_rules, {})

    def test_disabled_stale_rule_can_be_saved_and_deleted(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = self.manager(directory)
            stale = _rule(
                enabled=False, deployment_id="deleted", node_ids=["gone"],
            )
            manager.upsert_source_ip_routing_rule(stale)
            self.assertIsNone(manager.source_ip_routing_rule(
                "2001:db8::1", "shared-model",
            ))
            self.assertTrue(manager.delete_source_ip_routing_rule(
                "2001:0db8::1", "shared-model",
            ))


class SourceRoutingPlacementTests(unittest.IsolatedAsyncioTestCase):
    async def test_pins_secondary_replica_and_skips_balancing_and_affinity(self):
        manager = Manager.__new__(Manager)
        manager.deployments = [_replicated_deployment()]
        manager._proxy_cluster_member = AsyncMock(return_value={"choices": []})
        manager._cluster_route_order = Mock(
            side_effect=AssertionError("load balancing must not run")
        )
        manager._cluster_affinity_context = Mock(
            side_effect=AssertionError("affinity must not run")
        )

        result = await manager.proxy_cluster_inference(
            "manager-1", "org/model", {"stream": False}, "chat/completions",
            source_route=_rule(source_ip="10.0.0.8"),
        )

        self.assertEqual(result, {"choices": []})
        selected = manager._proxy_cluster_member.await_args.args[1]
        self.assertEqual(selected["node_id"], "node-b")

    async def test_changed_fingerprint_fails_closed_without_proxying(self):
        manager = Manager.__new__(Manager)
        deployment = _replicated_deployment()
        deployment["members"][1]["node_id"] = "replacement"
        manager.deployments = [deployment]
        manager._proxy_cluster_member = AsyncMock()

        with self.assertRaisesRegex(SourceRoutingUnavailable, "topology"):
            await manager.proxy_cluster_inference(
                "manager-1", "org/model", {"stream": False},
                "chat/completions", source_route=_rule(),
            )
        manager._proxy_cluster_member.assert_not_awaited()

    async def test_pinned_transport_failure_never_fails_over(self):
        manager = Manager.__new__(Manager)
        manager.deployments = [_replicated_deployment()]
        manager._proxy_cluster_member = AsyncMock(
            side_effect=httpx.ConnectError("offline")
        )

        with self.assertRaisesRegex(SourceRoutingUnavailable, "unavailable"):
            await manager.proxy_cluster_inference(
                "manager-1", "org/model", {"stream": False},
                "chat/completions", source_route=_rule(),
            )
        self.assertEqual(manager._proxy_cluster_member.await_count, 1)


class SourceRoutingServiceIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def _service(self, directory, deployments):
        manager = Manager.__new__(Manager)
        manager.http = httpx.AsyncClient()
        manager.deployments = deployments
        manager.source_ip_routing_rules_path = Path(directory) / "routes.json"
        manager.source_ip_routing_rules = {}
        manager._proxy_cluster_member = AsyncMock(
            side_effect=lambda deployment, member, *args, **kwargs: {
                "choices": [], "usage": {}, "selected_node": member["node_id"],
            }
        )
        service = SparkDeckService(manager, Path(directory))
        service._record_response = Mock()
        service._managed_hardware_snapshot = AsyncMock(return_value=({}, True))
        live = []
        for cluster in deployments:
            record_id = cluster["sparkdeck_record_id"]
            service.store.add_deployment(Deployment(
                id=record_id,
                alias=f"alias-{record_id}",
                runtime=RuntimeKind.VLLM,
                kind=DeploymentKind.MANAGED,
                model=ModelIdentity("org/shared"),
                settings={"manager_deployment_id": cluster["id"]},
            ))
            live.append({
                "id": record_id,
                "alias": f"alias-{record_id}",
                "runtime": RuntimeKind.VLLM.value,
                "kind": DeploymentKind.MANAGED.value,
                "status": "running",
                "desired_state": "running",
                "deployment_mode": "grouped_sharded",
                "model": {"repository": "org/shared"},
                "served_models": ["shared-model"],
                "settings": {"manager_deployment_id": cluster["id"]},
                "instances": [
                    {"instance_id": 0, "status": "running"},
                    {"instance_id": 1, "status": "running"},
                ],
            })
        service.deployments = AsyncMock(return_value=live)
        return manager, service

    async def test_two_source_ips_pin_distinct_groups_before_ambiguous_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            first = _grouped_deployment("manager-1", "record-1")
            second = _grouped_deployment("manager-2", "record-2")
            manager, service = await self._service(directory, [first, second])
            manager.upsert_source_ip_routing_rule(_rule(
                source_ip="10.0.0.1", deployment_id="record-1",
                instance_id=1, node_ids=["node-c", "node-d"],
            ))
            manager.upsert_source_ip_routing_rule(_rule(
                source_ip="10.0.0.2", deployment_id="record-2",
                instance_id=0, node_ids=["node-a", "node-b"],
            ))

            one = await service.proxy(
                {"model": "shared-model", "stream": False},
                "chat/completions", caller_ip="10.0.0.1",
            )
            two = await service.proxy(
                {"model": "shared-model", "stream": False},
                "chat/completions", caller_ip="10.0.0.2",
            )

            self.assertEqual(one["selected_node"], "node-c")
            self.assertEqual(two["selected_node"], "node-a")
            selected_deployments = [
                call.args[0]["sparkdeck_record_id"]
                for call in manager._proxy_cluster_member.await_args_list
            ]
            self.assertEqual(selected_deployments, ["record-1", "record-2"])
            await manager.http.aclose()
            await service.close()

    async def test_two_source_ips_pin_two_groups_in_one_deployment(self):
        with tempfile.TemporaryDirectory() as directory:
            manager, service = await self._service(
                directory, [_grouped_deployment()],
            )
            for source_ip, instance_id, nodes in (
                ("10.0.0.1", 0, ["node-a", "node-b"]),
                ("10.0.0.2", 1, ["node-c", "node-d"]),
            ):
                manager.upsert_source_ip_routing_rule(_rule(
                    source_ip=source_ip, instance_id=instance_id, node_ids=nodes,
                ))
            first = await service.proxy(
                {"model": "shared-model", "stream": False},
                "chat/completions", caller_ip="10.0.0.1",
            )
            second = await service.proxy(
                {"model": "shared-model", "stream": False},
                "chat/completions", caller_ip="10.0.0.2",
            )
            self.assertEqual(
                (first["selected_node"], second["selected_node"]),
                ("node-a", "node-c"),
            )
            await manager.http.aclose()
            await service.close()

    async def test_stopped_group_fails_closed_and_disabled_rule_uses_default(self):
        with tempfile.TemporaryDirectory() as directory:
            cluster = _grouped_deployment()
            manager, service = await self._service(directory, [cluster])
            active = _rule(
                source_ip="10.0.0.1", instance_id=1,
                node_ids=["node-c", "node-d"],
            )
            manager.upsert_source_ip_routing_rule(active)
            for member in cluster["members"]:
                if member["instance_id"] == 1:
                    member["desired_state"] = "stopped"
            with self.assertRaises(SourceRoutingUnavailable):
                await service.proxy(
                    {"model": "shared-model", "stream": False},
                    "chat/completions", caller_ip="10.0.0.1",
                )
            manager._proxy_cluster_member.assert_not_awaited()

            # A disabled exact match returns to ordinary group selection.
            manager.upsert_source_ip_routing_rule({**active, "enabled": False})
            fallback = await service.proxy(
                {"model": "shared-model", "stream": False},
                "chat/completions", caller_ip="10.0.0.1",
            )
            self.assertEqual(fallback["selected_node"], "node-a")
            await manager.http.aclose()
            await service.close()

    async def test_streaming_pin_never_retries_another_group(self):
        async def broken_stream():
            raise httpx.ConnectError("group offline")
            yield ""  # pragma: no cover

        with tempfile.TemporaryDirectory() as directory:
            manager, service = await self._service(
                directory, [_grouped_deployment()],
            )
            manager.upsert_source_ip_routing_rule(_rule(
                source_ip="10.0.0.1", instance_id=1,
                node_ids=["node-c", "node-d"],
            ))
            manager._proxy_cluster_member = AsyncMock(return_value=broken_stream())
            stream = await service.proxy(
                {"model": "shared-model", "stream": True},
                "chat/completions", caller_ip="10.0.0.1",
            )
            chunks = [chunk async for chunk in stream]
            self.assertEqual(manager._proxy_cluster_member.await_count, 1)
            self.assertIn("upstream_error", "".join(chunks))
            await manager.http.aclose()
            await service.close()

    async def test_runtime_change_fails_closed_before_transport(self):
        with tempfile.TemporaryDirectory() as directory:
            manager, service = await self._service(
                directory, [_grouped_deployment()],
            )
            manager.upsert_source_ip_routing_rule(_rule(
                source_ip="10.0.0.1", instance_id=0,
                node_ids=["node-a", "node-b"],
            ))
            live = await service.deployments()
            live[0]["runtime"] = RuntimeKind.LLAMA_CPP.value
            service.deployments = AsyncMock(return_value=live)

            with self.assertRaisesRegex(
                SourceRoutingUnavailable, "supported managed runtime",
            ):
                await service.proxy(
                    {"model": "shared-model", "stream": False},
                    "chat/completions", caller_ip="10.0.0.1",
                )
            manager._proxy_cluster_member.assert_not_awaited()
            await manager.http.aclose()
            await service.close()


class SourceRoutingApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        import server

        self.server = server
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app),
            base_url="http://test",
        )

    async def asyncTearDown(self):
        await self.client.aclose()

    async def test_get_put_and_ipv6_query_delete_contract(self):
        expected = _rule(source_ip="2001:db8::1")
        with patch.object(
            self.server.sparkdeck, "upsert_source_ip_routing_rule",
            AsyncMock(return_value=expected),
        ) as upsert, patch.object(
            self.server.sparkdeck, "source_ip_routing_rules",
            return_value=[expected],
        ), patch.object(
            self.server.sparkdeck, "delete_source_ip_routing_rule",
            return_value=True,
        ) as delete:
            put = await self.client.put(
                "/api/v1/inference-routing-rules", json=expected,
            )
            listing = await self.client.get("/api/v1/inference-routing-rules")
            deleted = await self.client.delete(
                "/api/v1/inference-routing-rules",
                params={
                    "source_ip": "2001:db8::1",
                    "requested_model": "shared-model",
                },
            )

        self.assertEqual(put.status_code, 200)
        self.assertEqual(listing.json(), {"items": [expected]})
        self.assertEqual(deleted.json(), {"ok": True})
        upsert.assert_awaited_once_with(expected)
        delete.assert_called_once_with("2001:db8::1", "shared-model")

    async def test_inference_maps_unavailable_pin_to_503(self):
        with patch.object(
            self.server.sparkdeck, "proxy",
            AsyncMock(side_effect=SourceRoutingUnavailable("pinned group offline")),
        ):
            response = await self.client.post(
                "/v1/chat/completions",
                json={"model": "shared-model", "messages": []},
            )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], "pinned group offline")
