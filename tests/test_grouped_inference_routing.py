import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from sparkdeck.models import Deployment, DeploymentKind, ModelIdentity, RuntimeKind
from sparkdeck.service import SparkDeckService


def grouped_service(tmp_path):
    manager = SimpleNamespace(
        http=Mock(),
        deployments=[],
        list_containers=AsyncMock(return_value=[]),
        proxy_cluster_inference=AsyncMock(return_value={"choices": [], "usage": {}}),
    )
    service = SparkDeckService(manager, tmp_path)
    # A stopped record's alias collides with the surviving deployment's served name.
    for record_id, alias, desired in (
        ("stopped", "vision-public", "stopped"),
        ("surviving", "vision-groups", "running"),
    ):
        service.store.add_deployment(Deployment(
            id=record_id, alias=alias, runtime=RuntimeKind.VLLM,
            kind=DeploymentKind.MANAGED, model=ModelIdentity("org/vision"),
            settings={"manager_deployment_id": f"cluster-{record_id}"},
            desired_state=desired,
        ))
    live = {
        **service.store.deployment("surviving", include_private=True),
        "status": "degraded", "deployment_mode": "grouped_sharded",
        "served_models": ["vision-public"],
        "instances": [
            {"instance_id": 0, "status": "stopped", "desired_state": "stopped"},
            {"instance_id": 1, "status": "running", "desired_state": "running"},
        ],
    }
    service.deployments = AsyncMock(return_value=[
        {**service.store.deployment("stopped", include_private=True), "status": "stopped"},
        live,
    ])
    return service, manager, live


@pytest.mark.parametrize("endpoint", ["chat/completions", "completions"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("selector", ["vision-public", "vision-groups"])
def test_degraded_grouped_deployment_remains_listed_and_routes_to_manager(
    tmp_path, endpoint, stream, selector,
):
    async def run():
        service, manager, _ = grouped_service(tmp_path)
        try:
            if stream:
                async def chunks():
                    yield 'data: {"choices": [{"delta": {"content": "OK"}}]}\n\n'
                    yield "data: [DONE]\n\n"
                manager.proxy_cluster_inference.return_value = chunks()
            models = await service.models()
            assert [(row["id"], row["deployment_id"]) for row in models["data"]] == [
                ("vision-public", "surviving"),
            ]
            body = {"model": selector, "stream": stream}
            body.update({"messages": []} if endpoint == "chat/completions" else {"prompt": "Hi"})
            result = await service.proxy(body, endpoint)
            if stream:
                output = "".join([chunk async for chunk in result])
                assert "OK" in output
                assert "[DONE]" in output
            else:
                assert result["model"] == selector
            manager.proxy_cluster_inference.assert_awaited_once()
            args = manager.proxy_cluster_inference.await_args.args
            assert args[:2] == ("cluster-surviving", "org/vision")
            assert args[2]["stream"] is stream
            assert args[3] == endpoint
        finally:
            await service.close()
    asyncio.run(run())


@pytest.mark.parametrize("state", ["stopped", "starting", "error"])
def test_degraded_grouped_deployment_without_ready_group_is_not_advertised(tmp_path, state):
    async def run():
        service, manager, live = grouped_service(tmp_path)
        live["instances"][1]["status"] = state
        try:
            assert (await service.models())["data"] == []
            assert await service._live_deployment_for_model_id("vision-public") is None
            with pytest.raises(RuntimeError, match="deployment is stopped"):
                await service.proxy({"model": "vision-public", "messages": []}, "chat/completions")
            manager.proxy_cluster_inference.assert_not_awaited()
        finally:
            await service.close()
    asyncio.run(run())


@pytest.mark.parametrize("change", ["whole_stop", "group_stop", "not_grouped"])
def test_degraded_deployment_requires_running_group_and_running_intent(tmp_path, change):
    async def run():
        service, _, live = grouped_service(tmp_path)
        if change == "whole_stop":
            live["desired_state"] = "stopped"
        elif change == "group_stop":
            live["instances"][1]["desired_state"] = "stopped"
        else:
            live["deployment_mode"] = "sharded"
        try:
            assert (await service.models())["data"] == []
            assert await service._live_deployment_for_model_id("vision-public") is None
        finally:
            await service.close()
    asyncio.run(run())
