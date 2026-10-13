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
        self.proxy_cluster_inference = AsyncMock(return_value={
            "id": "chatcmpl-1", "model": "ignored", "choices": [], "usage": {},
        })
        self.embedding_models = AsyncMock(return_value={"models": []})


def live_rows():
    # One request id per deployment so the default alias resolves to exactly
    # one owner, plus distinct aliases to prove names still route directly.
    return [
        {
            "id": "record-one", "alias": "model-one", "runtime": "vllm",
            "kind": "managed", "status": "running",
            "served_models": ["alpha-model"],
            "model": {"repository": "org/model-one"},
            "settings": {"manager_deployment_id": "cluster-one"},
            "last_deployed_at": "2026-10-01T00:00:00Z",
        },
        {
            "id": "record-two", "alias": "model-two", "runtime": "vllm",
            "kind": "managed", "status": "running",
            "served_models": ["beta-model"],
            "model": {"repository": "org/model-two"},
            "settings": {"manager_deployment_id": "cluster-two"},
            "last_deployed_at": "2026-10-01T00:00:00Z",
        },
    ]


def build_service(directory, default_model=None):
    manager = FakeManager()
    service = SparkDeckService(manager, Path(directory))
    for row in live_rows():
        service.store.add_deployment(Deployment(
            id=row["id"], alias=row["alias"], runtime=RuntimeKind.VLLM,
            kind=DeploymentKind.MANAGED,
            model=ModelIdentity(row["model"]["repository"]),
            settings={"manager_deployment_id": row["settings"]["manager_deployment_id"]},
        ))
    service.deployments = AsyncMock(return_value=live_rows())
    if default_model is not None:
        service.store.set_setting("default_model", default_model)
    return manager, service


class DefaultModelProxyTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_routes_to_the_configured_model(self):
        with tempfile.TemporaryDirectory() as directory:
            manager, service = build_service(directory, default_model="beta-model")
            try:
                result = await service.proxy(
                    {"model": "default", "messages": [], "stream": False},
                    "chat/completions",
                )
            finally:
                await manager.http.aclose()
                await service.close()
        call = manager.proxy_cluster_inference.await_args
        # The alias is indistinguishable from a request that named the target:
        # the target's Manager id and repository reach the upstream body.
        self.assertEqual(call.args[0], "cluster-two")
        self.assertEqual(call.args[1], "org/model-two")
        self.assertEqual(call.args[2]["model"], "org/model-two")
        self.assertEqual(result["model"], "beta-model")

    async def test_default_alias_matches_any_casing(self):
        for requested in ("default", "Default", "DEFAULT"):
            with tempfile.TemporaryDirectory() as directory:
                manager, service = build_service(directory, default_model="beta-model")
                try:
                    await service.proxy(
                        {"model": requested, "messages": [], "stream": False},
                        "chat/completions",
                    )
                finally:
                    await manager.http.aclose()
                    await service.close()
            self.assertEqual(
                manager.proxy_cluster_inference.await_args.args[0], "cluster-two",
                requested,
            )

    async def test_without_setting_the_alias_takes_the_legacy_path(self):
        with tempfile.TemporaryDirectory() as directory:
            manager, service = build_service(directory)
            manager.proxy_chat_completions.side_effect = LookupError(
                "model not found",
            )
            try:
                with self.assertRaises(LookupError):
                    await service.proxy(
                        {"model": "default", "messages": [], "stream": False},
                        "chat/completions",
                    )
            finally:
                await manager.http.aclose()
                await service.close()
            manager.proxy_cluster_inference.assert_not_awaited()

    async def test_unserved_target_falls_through_like_a_named_model(self):
        with tempfile.TemporaryDirectory() as directory:
            manager, service = build_service(directory, default_model="ghost-model")
            manager.proxy_chat_completions.side_effect = LookupError(
                "model not found",
            )
            try:
                with self.assertRaises(LookupError):
                    await service.proxy(
                        {"model": "default", "messages": [], "stream": False},
                        "chat/completions",
                    )
            finally:
                await manager.http.aclose()
                await service.close()
            # The redirect rewrote the request to the configured target, so
            # the legacy forward carries that name rather than the alias.
            self.assertEqual(
                manager.proxy_chat_completions.await_args.args[0]["model"],
                "ghost-model",
            )

    async def test_target_naming_a_deployment_alias_routes_to_its_owner(self):
        # The setting stores a deployment id/compat alias rather than a served
        # request id: store resolution handles it, exactly as a direct request
        # naming that alias would.
        with tempfile.TemporaryDirectory() as directory:
            manager, service = build_service(directory, default_model="model-two")
            try:
                await service.proxy(
                    {"model": "default", "messages": [], "stream": False},
                    "chat/completions",
                )
            finally:
                await manager.http.aclose()
                await service.close()
        self.assertEqual(
            manager.proxy_cluster_inference.await_args.args[0], "cluster-two",
        )

    async def test_stopped_exact_record_falls_through_to_the_live_owner(self):
        # The target names a stopped persisted record's compat alias while a
        # live discovered container uses that alias: proxy() keeps resolving
        # through the candidate set, so the listing must advertise the same
        # owner instead of dropping the alias.
        with tempfile.TemporaryDirectory() as directory:
            manager, service = build_service(directory, default_model="model-two")
            live = [live_rows()[0], {
                "id": "discovered-1", "alias": "model-two", "runtime": "vllm",
                "kind": "managed", "status": "running",
                "served_models": ["gamma-model"],
                "model": {"repository": "org/model-gamma"},
                "settings": {"manager_deployment_id": "cluster-gamma"},
                "last_deployed_at": "2026-10-02T00:00:00Z",
            }]
            service.deployments = AsyncMock(return_value=live)
            try:
                await service.proxy(
                    {"model": "default", "messages": [], "stream": False},
                    "chat/completions",
                )
                listing = await service.models()
            finally:
                await manager.http.aclose()
                await service.close()
        self.assertEqual(
            manager.proxy_cluster_inference.await_args.args[0], "cluster-gamma",
        )
        entries = {entry["id"]: entry for entry in listing["data"]}
        self.assertIn("default", entries)
        self.assertEqual(entries["default"]["deployment_id"], "discovered-1")
        self.assertEqual(entries["default"]["alias_of"], "model-two")

    async def test_named_models_route_exactly_as_before(self):
        with tempfile.TemporaryDirectory() as directory:
            manager, service = build_service(directory, default_model="beta-model")
            try:
                await service.proxy(
                    {"model": "alpha-model", "messages": [], "stream": False},
                    "chat/completions",
                )
            finally:
                await manager.http.aclose()
                await service.close()
        self.assertEqual(
            manager.proxy_cluster_inference.await_args.args[0], "cluster-one",
        )


class DefaultModelListingTests(unittest.IsolatedAsyncioTestCase):
    async def test_models_advertise_the_default_alias_of_the_target(self):
        with tempfile.TemporaryDirectory() as directory:
            manager, service = build_service(directory, default_model="beta-model")
            try:
                listing = await service.models()
            finally:
                await manager.http.aclose()
                await service.close()
        entries = {entry["id"]: entry for entry in listing["data"]}
        self.assertIn("default", entries)
        self.assertEqual(entries["default"]["alias_of"], "beta-model")
        self.assertEqual(entries["default"]["deployment_id"], "record-two")
        self.assertIn(
            "default",
            [model["slug"] for model in listing["models"]],
        )

    async def test_alias_id_target_is_advertised_off_its_owner(self):
        # A target naming a deployment through its compat alias is not a
        # listed request id, so the entry is built from the deployment that
        # would actually serve it, mirroring proxy resolution.
        with tempfile.TemporaryDirectory() as directory:
            manager, service = build_service(directory, default_model="model-two")
            try:
                listing = await service.models()
            finally:
                await manager.http.aclose()
                await service.close()
        entries = {entry["id"]: entry for entry in listing["data"]}
        self.assertIn("default", entries)
        self.assertEqual(entries["default"]["deployment_id"], "record-two")
        self.assertEqual(entries["default"]["alias_of"], "model-two")

    async def test_deployment_serving_the_reserved_name_is_replaced(self):
        # Routing claims the reserved name case-insensitively, so a
        # deployment that publishes "Default" itself keeps its other request
        # ids while its reserved entry is replaced by the alias entry.
        with tempfile.TemporaryDirectory() as directory:
            manager, service = build_service(directory, default_model="beta-model")
            rows = [dict(row) for row in live_rows()]
            rows[0]["served_models"] = ["alpha-model", "Default"]
            service.deployments = AsyncMock(return_value=rows)
            try:
                listing = await service.models()
            finally:
                await manager.http.aclose()
                await service.close()
        reserved = [
            entry for entry in listing["data"]
            if entry["id"].casefold() == "default"
        ]
        self.assertEqual(len(reserved), 1)
        self.assertEqual(reserved[0]["deployment_id"], "record-two")
        self.assertEqual(reserved[0]["alias_of"], "beta-model")
        entry_ids = [entry["id"] for entry in listing["data"]]
        self.assertNotIn("Default", entry_ids)
        self.assertIn("alpha-model", entry_ids)

    async def test_winner_entry_survives_reserved_removal_for_metadata(self):
        # The winning deployment publishes only a case variant of the reserved
        # name: its row is removed from the listing, but it must still supply
        # the metadata for the alias entry because inference routes the
        # configured alias to it.
        with tempfile.TemporaryDirectory() as directory:
            manager, service = build_service(directory, default_model="model-two")
            live = [live_rows()[0], {**live_rows()[1], "served_models": ["Default"]}]
            service.deployments = AsyncMock(return_value=live)
            try:
                await service.proxy(
                    {"model": "default", "messages": [], "stream": False},
                    "chat/completions",
                )
                listing = await service.models()
            finally:
                await manager.http.aclose()
                await service.close()
        self.assertEqual(
            manager.proxy_cluster_inference.await_args.args[0], "cluster-two",
        )
        reserved = [
            entry for entry in listing["data"]
            if entry["id"].casefold() == "default"
        ]
        self.assertEqual(len(reserved), 1)
        self.assertEqual(reserved[0]["deployment_id"], "record-two")
        self.assertEqual(reserved[0]["alias_of"], "model-two")
        self.assertNotIn("Default", [entry["id"] for entry in listing["data"]])

    async def test_embedding_id_collision_does_not_hide_the_chat_owner(self):
        # An embedding cache entry may share the target's compat-alias id on
        # the separate embedding surface; the alias must still advertise the
        # chat deployment inference routes to.
        with tempfile.TemporaryDirectory() as directory:
            manager, service = build_service(directory, default_model="model-two")
            manager.embedding_models = AsyncMock(return_value={"models": [{
                "model_id": "model-two", "node_ids": ["node-1"], "revision": "abc",
            }]})
            try:
                listing = await service.models()
            finally:
                await manager.http.aclose()
                await service.close()
        reserved = [
            entry for entry in listing["data"]
            if entry["id"].casefold() == "default"
        ]
        self.assertEqual(len(reserved), 1)
        self.assertEqual(reserved[0]["deployment_id"], "record-two")
        self.assertEqual(reserved[0]["alias_of"], "model-two")

    async def test_models_omit_the_alias_when_unset_or_unserved(self):
        with tempfile.TemporaryDirectory() as directory:
            manager, service = build_service(directory)
            try:
                listing = await service.models()
            finally:
                await manager.http.aclose()
                await service.close()
        self.assertNotIn(
            "default", [entry["id"] for entry in listing["data"]],
        )

        with tempfile.TemporaryDirectory() as directory:
            manager, service = build_service(directory, default_model="beta-model")
            # The target stops serving its request id: the gateway would
            # reject "default", so the listing must not promise it either.
            stopped = [dict(row) for row in live_rows()]
            stopped[1]["served_models"] = ["renamed-model"]
            service.deployments = AsyncMock(return_value=stopped)
            try:
                listing = await service.models()
            finally:
                await manager.http.aclose()
                await service.close()
        self.assertNotIn(
            "default", [entry["id"] for entry in listing["data"]],
        )


if __name__ == "__main__":
    unittest.main()
