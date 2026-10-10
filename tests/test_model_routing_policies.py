import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

import httpx

from sparkdeck.models import Deployment, DeploymentKind, ModelIdentity, RuntimeKind
from sparkdeck.service import SparkDeckService


class FakeManager:
    def __init__(self):
        self.http = httpx.AsyncClient()
        self.list_containers = AsyncMock(return_value=[])
        self.proxy_chat_completions = AsyncMock()
        self.proxy_completions = AsyncMock()
        self.proxy_cluster_inference = AsyncMock()


RECORDS = (
    ("record-one", "model-one", "cluster-one"),
    ("record-two", "model-two", "cluster-two"),
)


def live_rows():
    return [
        {
            "id": record_id, "alias": alias, "runtime": "vllm",
            "kind": "managed", "status": "running",
            "served_models": ["shared-name"],
            "model": {"repository": f"org/{alias}"},
            "settings": {"manager_deployment_id": manager_id},
            "last_deployed_at": "2026-10-01T00:00:00Z",
        }
        for record_id, alias, manager_id in RECORDS
    ]


def build_service(directory):
    manager = FakeManager()
    service = SparkDeckService(manager, Path(directory))
    for record_id, alias, manager_id in RECORDS:
        service.store.add_deployment(Deployment(
            id=record_id, alias=alias, runtime=RuntimeKind.VLLM,
            kind=DeploymentKind.MANAGED,
            model=ModelIdentity(f"org/{alias}"),
            settings={"manager_deployment_id": manager_id},
        ))
    service.deployments = AsyncMock(return_value=live_rows())
    return manager, service


class ModelRoutingPolicyValidationTests(unittest.IsolatedAsyncioTestCase):
    async def test_upsert_validates_members_and_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            manager, service = build_service(directory)
            try:
                with self.assertRaisesRegex(ValueError, "non-empty"):
                    await service.upsert_model_routing_policy(
                        {"model": "", "members": [{"deployment_id": "record-one"}]},
                    )
                with self.assertRaisesRegex(ValueError, "between 1 and"):
                    await service.upsert_model_routing_policy(
                        {"model": "shared-name", "members": []},
                    )
                with self.assertRaisesRegex(ValueError, "duplicate"):
                    await service.upsert_model_routing_policy({
                        "model": "shared-name", "members": [
                            {"deployment_id": "record-one"},
                            {"deployment_id": "record-one"},
                        ],
                    })
                with self.assertRaisesRegex(ValueError, "max_concurrency"):
                    await service.upsert_model_routing_policy({
                        "model": "shared-name",
                        "members": [{"deployment_id": "record-one", "max_concurrency": 0}],
                    })
                with self.assertRaisesRegex(LookupError, "not found"):
                    await service.upsert_model_routing_policy({
                        "model": "shared-name",
                        "members": [{"deployment_id": "missing"}],
                    })
                self.assertEqual(await service.model_routing_policies(), [])
            finally:
                await manager.http.aclose()
                await service.close()

    async def test_policy_persists_and_reports_member_status(self):
        with tempfile.TemporaryDirectory() as directory:
            manager, service = build_service(directory)
            try:
                saved = await service.upsert_model_routing_policy({
                    "model": "shared-name", "members": [
                        {"deployment_id": "record-two", "max_concurrency": 3},
                        {"deployment_id": "record-one", "max_concurrency": None},
                    ],
                })
                self.assertEqual(saved["model"], "shared-name")
                self.assertEqual(
                    [member["deployment_id"] for member in saved["members"]],
                    ["record-two", "record-one"],
                )
                self.assertTrue(all(member["live"] for member in saved["members"]))
                self.assertEqual(saved["members"][0]["alias"], "model-two")

                reloaded = SparkDeckService(FakeManager(), Path(directory))
                items = await reloaded.model_routing_policies()
                self.assertEqual(len(items), 1)
                self.assertEqual(items[0]["members"][0]["max_concurrency"], 3)
                self.assertTrue(reloaded.delete_model_routing_policy("shared-name"))
                self.assertFalse(reloaded.delete_model_routing_policy("shared-name"))
                await reloaded.close()
            finally:
                await manager.http.aclose()
                await service.close()


class ModelRoutingProxyTests(unittest.IsolatedAsyncioTestCase):
    async def test_priority_instance_overflows_at_its_concurrency_cap(self):
        """Requests fill the priority instance up to its cap; the next
        request lands on the following instance, like an ALB fallback."""
        with tempfile.TemporaryDirectory() as directory:
            manager, service = build_service(directory)
            started = []
            release = asyncio.Event()

            async def cluster_inference(manager_id, *args, **kwargs):
                started.append(manager_id)
                if len(started) >= 4:
                    release.set()
                else:
                    await release.wait()
                return {"model": "shared-name", "choices": [], "usage": {}}

            manager.proxy_cluster_inference = cluster_inference
            try:
                await service.upsert_model_routing_policy({
                    "model": "shared-name", "members": [
                        {"deployment_id": "record-two", "max_concurrency": 3},
                        {"deployment_id": "record-one", "max_concurrency": None},
                    ],
                })
                responses = await asyncio.gather(*(
                    service.proxy(
                        {"model": "shared-name", "messages": [], "stream": False},
                        "chat/completions",
                    )
                    for _ in range(4)
                ))
                self.assertTrue(all("choices" in response for response in responses))
                self.assertEqual(started, [
                    "cluster-two", "cluster-two", "cluster-two", "cluster-one",
                ])
                # Completed requests release their slots.
                for _ in range(200):
                    if not service._model_routing_inflight:
                        break
                    await asyncio.sleep(0.01)
                self.assertEqual(service._model_routing_inflight, {})
            finally:
                release.set()
                await manager.http.aclose()
                await service.close()

    async def test_all_saturated_members_take_the_least_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            manager, service = build_service(directory)
            started = []
            release = asyncio.Event()

            async def cluster_inference(manager_id, *args, **kwargs):
                started.append(manager_id)
                if len(started) >= 3:
                    release.set()
                else:
                    await release.wait()
                return {"model": "shared-name", "choices": [], "usage": {}}

            manager.proxy_cluster_inference = cluster_inference
            try:
                await service.upsert_model_routing_policy({
                    "model": "shared-name", "members": [
                        {"deployment_id": "record-two", "max_concurrency": 1},
                        {"deployment_id": "record-one", "max_concurrency": 1},
                    ],
                })
                await asyncio.gather(*(
                    service.proxy(
                        {"model": "shared-name", "messages": [], "stream": False},
                        "chat/completions",
                    )
                    for _ in range(3)
                ))
                # Both caps are hit; the tie breaks toward the first member
                # in priority order, and no request is failed.
                self.assertEqual(sorted(started), [
                    "cluster-one", "cluster-two", "cluster-two",
                ])
            finally:
                release.set()
                await manager.http.aclose()
                await service.close()

    async def test_exact_alias_bypasses_the_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            manager, service = build_service(directory)
            manager.proxy_cluster_inference = AsyncMock(return_value={
                "model": "shared-name", "choices": [], "usage": {},
            })
            try:
                await service.upsert_model_routing_policy({
                    "model": "shared-name", "members": [
                        {"deployment_id": "record-two", "max_concurrency": 1},
                    ],
                })
                await service.proxy(
                    {"model": "model-one", "messages": [], "stream": False},
                    "chat/completions",
                )
                self.assertEqual(
                    manager.proxy_cluster_inference.await_args.args[0],
                    "cluster-one",
                )
                self.assertEqual(service._model_routing_inflight, {})
            finally:
                await manager.http.aclose()
                await service.close()

    async def test_streaming_requests_release_their_slot_when_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            manager, service = build_service(directory)
            closed = asyncio.Event()

            async def cluster_inference(manager_id, *args, **kwargs):
                async def chunks():
                    try:
                        yield "data: one\n\n"
                        yield "data: [DONE]\n\n"
                    finally:
                        closed.set()
                return chunks()

            manager.proxy_cluster_inference = cluster_inference
            try:
                await service.upsert_model_routing_policy({
                    "model": "shared-name", "members": [
                        {"deployment_id": "record-two", "max_concurrency": 1},
                        {"deployment_id": "record-one", "max_concurrency": None},
                    ],
                })
                stream = await service.proxy(
                    {"model": "shared-name", "messages": [], "stream": True},
                    "chat/completions",
                )
                self.assertEqual(
                    service._model_routing_inflight.get("record-two"), 1,
                )
                # A saturated priority instance sends the next request to the
                # overflow instance while the stream is still open.
                manager.proxy_cluster_inference = AsyncMock(return_value={
                    "model": "shared-name", "choices": [], "usage": {},
                })
                await service.proxy(
                    {"model": "shared-name", "messages": [], "stream": False},
                    "chat/completions",
                )
                self.assertEqual(
                    manager.proxy_cluster_inference.await_args.args[0],
                    "cluster-one",
                )
                chunks = [chunk async for chunk in stream]
                self.assertTrue(any("[DONE]" in chunk for chunk in chunks))
                self.assertTrue(closed.is_set())
                self.assertEqual(service._model_routing_inflight, {})
            finally:
                await manager.http.aclose()
                await service.close()

    async def test_failed_upstream_releases_the_slot(self):
        with tempfile.TemporaryDirectory() as directory:
            manager, service = build_service(directory)
            manager.proxy_cluster_inference = AsyncMock(
                side_effect=RuntimeError("upstream exploded"),
            )
            try:
                await service.upsert_model_routing_policy({
                    "model": "shared-name", "members": [
                        {"deployment_id": "record-two", "max_concurrency": 1},
                        {"deployment_id": "record-one", "max_concurrency": None},
                    ],
                })
                with self.assertRaises(RuntimeError):
                    await service.proxy(
                        {"model": "shared-name", "messages": [], "stream": False},
                        "chat/completions",
                    )
                self.assertEqual(service._model_routing_inflight, {})
            finally:
                await manager.http.aclose()
                await service.close()

    async def test_without_policy_the_newest_owner_still_answers(self):
        with tempfile.TemporaryDirectory() as directory:
            manager, service = build_service(directory)
            manager.proxy_cluster_inference = AsyncMock(return_value={
                "model": "shared-name", "choices": [], "usage": {},
            })
            try:
                rows = live_rows()
                rows[0]["last_deployed_at"] = "2026-10-10T00:00:00Z"
                service.deployments = AsyncMock(return_value=rows)
                await service.proxy(
                    {"model": "shared-name", "messages": [], "stream": False},
                    "chat/completions",
                )
                self.assertEqual(
                    manager.proxy_cluster_inference.await_args.args[0],
                    "cluster-one",
                )
                self.assertEqual(service._model_routing_inflight, {})
            finally:
                await manager.http.aclose()
                await service.close()


class ModelRoutingReviewFixTests(unittest.IsolatedAsyncioTestCase):
    async def test_members_may_be_live_discovered_deployments(self):
        with tempfile.TemporaryDirectory() as directory:
            manager, service = build_service(directory)
            try:
                rows = live_rows() + [{
                    "id": "container:one", "alias": "container:one",
                    "runtime": "vllm", "kind": "discovered", "status": "running",
                    "served_models": ["shared-name"],
                    "model": {"repository": "org/discovered"},
                }]
                service.deployments = AsyncMock(return_value=rows)
                saved = await service.upsert_model_routing_policy({
                    "model": "shared-name", "members": [
                        {"deployment_id": "container:one", "max_concurrency": 2},
                        {"deployment_id": "record-one", "max_concurrency": None},
                    ],
                })
                self.assertEqual(
                    [member["deployment_id"] for member in saved["members"]],
                    ["container:one", "record-one"],
                )
                self.assertTrue(saved["members"][0]["live"])
            finally:
                await manager.http.aclose()
                await service.close()

    async def test_failed_persistence_leaves_active_policies_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            manager, service = build_service(directory)
            try:
                await service.upsert_model_routing_policy({
                    "model": "shared-name", "members": [
                        {"deployment_id": "record-one", "max_concurrency": None},
                    ],
                })

                def fail(_policies=None):
                    raise OSError("disk full")

                service._save_model_routing_policies = fail
                with self.assertRaises(OSError):
                    await service.upsert_model_routing_policy({
                        "model": "shared-name", "members": [
                            {"deployment_id": "record-two", "max_concurrency": 5},
                        ],
                    })
                self.assertEqual(
                    [
                        member["deployment_id"]
                        for member in service._model_routing_policies["shared-name"]["members"]
                    ],
                    ["record-one"],
                )
                with self.assertRaises(OSError):
                    service.delete_model_routing_policy("shared-name")
                self.assertIn("shared-name", service._model_routing_policies)
            finally:
                await manager.http.aclose()
                await service.close()

    async def test_stream_cleanup_failure_still_releases_the_slot(self):
        with tempfile.TemporaryDirectory() as directory:
            manager, service = build_service(directory)
            try:
                async def chunks():
                    try:
                        yield "data: one\n\n"
                    finally:
                        raise RuntimeError("cleanup failed")

                service._model_routing_inflight["record-two"] = 1
                wrapped = service._release_routing_slot_stream(chunks(), "record-two")
                collected = []
                with self.assertRaises(RuntimeError):
                    async for chunk in wrapped:
                        collected.append(chunk)
                self.assertEqual(collected, ["data: one\n\n"])
                self.assertEqual(service._model_routing_inflight, {})
            finally:
                await manager.http.aclose()
                await service.close()

    async def test_catalog_publishes_shared_name_on_policy_priority_member(self):
        with tempfile.TemporaryDirectory() as directory:
            manager, service = build_service(directory)
            try:
                rows = live_rows()
                rows[1]["last_deployed_at"] = "2026-10-10T00:00:00Z"
                service._model_routing_policies = {
                    "shared-name": {
                        "model": "shared-name",
                        "members": [
                            {"deployment_id": "record-one", "max_concurrency": None},
                            {"deployment_id": "record-two", "max_concurrency": None},
                        ],
                        "updated_at": None,
                    },
                }
                owners = service._shared_selector_owners(
                    rows, {"shared-name": {"record-one", "record-two"}},
                )
                # record-two is the newest deployment, but the policy's first
                # live member publishes the shared name in the catalog.
                self.assertEqual(owners, {"shared-name": "record-one"})
            finally:
                await manager.http.aclose()
                await service.close()


if __name__ == "__main__":
    unittest.main()
