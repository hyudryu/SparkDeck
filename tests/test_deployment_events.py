import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sparkdeck.service import SparkDeckService


def observer():
    service = SparkDeckService.__new__(SparkDeckService)
    service._deployment_log_states = {}
    service._deployment_log_errors = {}
    return service


def deployment(status, **kwargs):
    return {"id": "model-1", "alias": "My model", "status": status, **kwargs}


def events(caplog):
    return [record for record in caplog.records if record.name == "sparkdeck.lifecycle"]


def test_launch_stop_restart_logged_once_with_inventory_outages(caplog):
    caplog.set_level(logging.INFO)
    service = observer()
    for status in ["saved", "stopped", "starting", "running", "running",
                   "unknown", "missing", "unreachable", "running", "stopping",
                   "stopped", "stopped", "starting", "running"]:
        service._observe_deployment_events([deployment(status, desired_state="stopped")])
    assert [record.deployment_event for record in events(caplog)] == [
        "launched", "stopped", "launched",
    ]


def test_unexpected_exit_and_error_changes_are_reported_without_repeated_polls(caplog):
    caplog.set_level(logging.INFO)
    service = observer()
    for status, error in [("running", ""), ("stopped", ""), ("stopped", ""),
                          ("error", "Out of memory"), ("error", "Out of memory"),
                          ("error", "Worker failed")]:
        service._observe_deployment_events([
            deployment(status, desired_state="running", last_error=error),
        ])
    records = events(caplog)
    assert [record.deployment_event for record in records] == [
        "launched", "crashed", "error", "error",
    ]
    assert all(record.levelno == logging.ERROR for record in records[1:])
    assert "Out of memory" in records[2].getMessage()


def test_group_lifecycle_retains_group_identity(caplog):
    caplog.set_level(logging.INFO)
    service = observer()
    first = {"instance_id": 0, "status": "running", "desired_state": "running"}
    second = {"instance_id": 1, "status": "running", "desired_state": "running"}
    service._observe_deployment_events([deployment("running", instances=[first, second])])
    second.update(status="stopped", desired_state="stopped")
    service._observe_deployment_events([deployment("degraded", instances=[first, second])])
    service._observe_deployment_events([deployment("degraded", instances=[first, second])])
    records = events(caplog)
    assert [record.deployment_event for record in records] == ["launched", "launched", "stopped"]
    assert "engine group 1" in records[-1].getMessage()


def test_initial_error_visible_but_initial_stopped_deployments_are_quiet(caplog):
    caplog.set_level(logging.INFO)
    service = observer()
    service._observe_deployment_events([deployment("stopped")])
    service._observe_deployment_events([deployment("error", last_error="Launch failed")])
    service._observe_deployment_events([deployment("error", last_error="Launch failed")])
    assert [record.getMessage() for record in events(caplog)] == [
        "Deployment My model error: Launch failed",
    ]


def test_readiness_fluctuations_do_not_announce_another_launch(caplog):
    caplog.set_level(logging.INFO)
    service = observer()
    for status in ["running", "starting", "running", "degraded", "running",
                   "stopping", "running", "stopped", "starting", "running"]:
        service._observe_deployment_events([deployment(status, desired_state="stopped")])
    assert [record.deployment_event for record in events(caplog)] == [
        "launched", "stopped", "launched",
    ]


@pytest.mark.parametrize("cancelled", [False, True])
def test_background_launch_failure_is_logged_but_cancellation_is_not(caplog, cancelled):
    async def scenario():
        service = observer()
        service._deployment_launches = {}
        service._deployment_launch_tasks = {}
        service.store = Mock()
        service.store.deployment.return_value = {"id": "model-1"}
        service._link_cluster_record = Mock()

        async def launch(body, *, launch_persisted):
            launch_persisted.set_result({"id": "cluster-1"})
            if cancelled:
                raise asyncio.CancelledError()
            raise RuntimeError("Worker launch failed")

        service.manager = SimpleNamespace(create_deployment=launch, deployments=[])
        record = SimpleNamespace(id="model-1", alias="My model")
        await service._begin_cluster_deployment(record, {}, "replicated", [], [], {})
        await asyncio.sleep(0)

    asyncio.run(scenario())
    failures = [record for record in caplog.records if "launch failed" in record.getMessage()]
    assert len(failures) == (0 if cancelled else 1)
    if failures:
        assert failures[0].levelno == logging.ERROR
        assert "Worker launch failed" in failures[0].getMessage()


def test_external_endpoint_health_reports_errors_without_inventing_lifecycle(caplog):
    caplog.set_level(logging.INFO)
    service = observer()
    for status in ["running", "error", "error", "running", "error"]:
        service._observe_deployment_events([
            deployment(status, kind="external", last_error="Endpoint health check failed"),
        ])
    records = events(caplog)
    assert len(records) == 2
    assert all(record.levelno == logging.ERROR for record in records)
    assert all(not hasattr(record, "deployment_event") for record in records)


def test_inventory_errors_are_deduplicated_without_losing_known_running_state(caplog):
    caplog.set_level(logging.INFO)
    service = observer()
    for status, error in [("running", ""), ("missing", "Docker is unavailable"),
                          ("missing", "Docker is unavailable"), ("running", ""),
                          ("unreachable", "Node is unavailable"), ("running", "")]:
        service._observe_deployment_events([deployment(status, last_error=error)])
    records = events(caplog)
    assert [getattr(record, "deployment_event", "error") for record in records] == [
        "launched", "error", "error",
    ]
    assert "Docker is unavailable" in records[1].getMessage()


def test_removed_deployment_is_retired_only_after_confirmed_inventory(caplog):
    caplog.set_level(logging.INFO)
    service = observer()
    row = {"id": "container:my-model", "status": "running"}
    service._observe_deployment_events([row], inventory_complete=True)
    service._observe_deployment_events([], inventory_complete=False)
    service._observe_deployment_events([row], inventory_complete=True)
    key = (row["id"], None)
    service._deployment_log_errors[key] = "Prior discovery error"
    service._observe_deployment_events([], inventory_complete=True)
    assert not service._deployment_log_states
    assert not service._deployment_log_errors
    service._observe_deployment_events([], inventory_complete=True)
    service._observe_deployment_events([row], inventory_complete=True)
    assert [record.deployment_event for record in events(caplog)] == [
        "launched", "stopped", "launched",
    ]


def test_degraded_recovery_errors_are_reported_once_and_reset_when_healthy(caplog):
    caplog.set_level(logging.INFO)
    service = observer()
    for status, error in [
        ("running", ""), ("degraded", "Automatic recovery could not stop every rank"),
        ("degraded", "Automatic recovery could not stop every rank"),
        ("degraded", "Automatic recovery failed to start every rank"),
        ("running", "Automatic recovery failed to start every rank"),
        ("degraded", "Automatic recovery failed to start every rank"),
    ]:
        service._observe_deployment_events([deployment(status, last_error=error)])
    records = events(caplog)
    assert len(records) == 4
    assert records[0].deployment_event == "launched"
    assert all(record.levelno == logging.ERROR for record in records[1:])


def test_grouped_parent_error_is_reported_even_while_groups_keep_serving(caplog):
    caplog.set_level(logging.INFO)
    service = observer()
    groups = [{"instance_id": 0, "status": "running"}, {"instance_id": 1, "status": "starting"}]
    row = deployment("degraded", instances=groups, last_error="Automatic recovery failed")
    service._observe_deployment_events([row])
    service._observe_deployment_events([row])
    errors = [record for record in events(caplog) if record.levelno == logging.ERROR]
    assert len(errors) == 1
    assert errors[0].getMessage() == "Deployment My model error: Automatic recovery failed"


def test_health_recovery_ends_generation_and_success_announces_new_launch(caplog):
    from sparkdeck.service import _deployment_status

    caplog.set_level(logging.INFO)
    service = observer()
    for raw_status in ["ready", "recovering", "recovering", "starting", "ready"]:
        runtime = {"status": raw_status, "desired_state": "running", "health_issue": "rank 1 exited"}
        service._observe_deployment_events(
            [deployment(_deployment_status(raw_status))],
            runtime_deployments={"model-1": runtime},
        )
    assert [record.deployment_event for record in events(caplog)] == ["launched", "crashed", "launched"]


def test_group_rank_loss_reports_only_affected_engine_and_relaunch(caplog):
    from sparkdeck.service import _grouped_instance_summary

    caplog.set_level(logging.INFO)
    service = observer()
    for rank_status in ["running", "missing", "missing", "running"]:
        cluster = {
            "status": "degraded" if rank_status == "missing" else "running",
            "desired_state": "running",
            "members": [
                {"instance_id": group, "rank": rank, "desired_state": "running",
                 "node_id": f"node-{group}", "node_docker_ready": True,
                 "status": rank_status if group == 1 and rank == 1 else "running",
                 "phase": {"phase": "ready"}}
                for group in (0, 1) for rank in (0, 1)
            ],
        }
        service._observe_deployment_events(
            [deployment(cluster["status"], instances=_grouped_instance_summary(cluster))],
            runtime_deployments={"model-1": cluster},
        )
    records = events(caplog)
    assert [record.deployment_event for record in records] == ["launched", "launched", "crashed", "launched"]
    assert all("engine group 1" in record.getMessage() for record in records[2:])


@pytest.mark.parametrize("reason", ["readiness", "unreachable", "intentional_stop", "startup_recovery"])
def test_readiness_outage_and_intentional_stop_do_not_invent_crashes(caplog, reason):
    from sparkdeck.service import _grouped_instance_summary

    caplog.set_level(logging.INFO)
    service = observer()
    member = {"instance_id": 0, "rank": 0, "status": "running", "phase": {"phase": "ready"}}
    cluster = {"status": "running", "members": [member]}
    service._observe_deployment_events([deployment("running", instances=_grouped_instance_summary(cluster))])
    if reason == "readiness":
        member["phase"] = {"phase": "starting"}
    elif reason == "unreachable":
        member["status"] = "unreachable"
    elif reason == "intentional_stop":
        member.update(status="exited", desired_state="stopped")
    else:
        cluster["status"] = "recovering"
        member["status"] = "starting"
    service._observe_deployment_events(
        [deployment("degraded", instances=_grouped_instance_summary(cluster))],
        runtime_deployments={"model-1": cluster},
    )
    assert all(record.deployment_event != "crashed" for record in events(caplog))


def test_inventory_passes_raw_recovery_evidence_to_observer(caplog):
    from unittest.mock import AsyncMock

    caplog.set_level(logging.INFO)

    async def scenario():
        service = observer()
        service.store = Mock()
        service.registry = SimpleNamespace(kinds=[])
        service._probe_external_endpoint = AsyncMock()
        service._adopt_unlinked_manager_deployments = AsyncMock(side_effect=lambda *args, **kwargs: [
            deployment("running", kind="managed", desired_state="running",
                       settings={"manager_deployment_id": "cluster-1"}),
        ])
        runtime = {"id": "cluster-1", "sparkdeck_record_id": "model-1",
                   "desired_state": "running", "status": "ready"}
        service.manager = SimpleNamespace(
            deployments=[runtime],
            get_state=AsyncMock(side_effect=lambda: {
                "deployments": [runtime], "docker_ready": True, "containers": [],
            }),
            cluster_nodes=AsyncMock(return_value=[]),
        )
        await service.deployments(observe_events=True)
        runtime.update(status="recovering", health_issue="rank 1 exited")
        await service.deployments(observe_events=True)
        runtime.update(status="ready", health_issue=None)
        await service.deployments(observe_events=True)

    asyncio.run(scenario())
    assert [record.deployment_event for record in events(caplog)] == ["launched", "crashed", "launched"]


def test_missing_rank_requires_reliable_node_inventory_to_be_crash():
    from sparkdeck.service import _deployment_process_lost

    cluster = {"desired_state": "running", "members": [
        {"instance_id": 0, "rank": 0, "desired_state": "running",
         "status": "missing",
         # A reachable node reporting docker_ready=false advertises no
         # container summary, so absence is an inventory outage, not a crash.
         "node_docker_ready": False},
    ]}
    assert _deployment_process_lost(cluster) is False
    cluster["members"][0]["node_docker_ready"] = True
    assert _deployment_process_lost(cluster) is True
    # Legacy members without node-docker evidence are treated as unreliable too.
    cluster["members"][0].pop("node_docker_ready")
    assert _deployment_process_lost(cluster) is False


def test_recreation_records_new_launch(caplog):
    from sparkdeck.service import _grouped_instance_summary

    caplog.set_level(logging.INFO)
    service = observer()
    member = {"instance_id": 0, "rank": 0, "desired_state": "running",
              "node_id": "node-0", "node_docker_ready": True,
              "status": "running", "phase": {"phase": "ready"}}
    cluster = {"status": "running", "desired_state": "running", "members": [member]}
    service._observe_deployment_events(
        [deployment("running", instances=_grouped_instance_summary(cluster))],
    )
    # A deliberate recreation queues replacement ranks with recreate_pending.
    member.update(status="creating", recreate_pending=True)
    cluster["status"] = "starting"
    service._observe_deployment_events(
        [deployment("starting", instances=_grouped_instance_summary(cluster))],
        runtime_deployments={"model-1": cluster},
    )
    # The replacement reaches ready: the new generation announces a launch.
    member.update(status="running", recreate_pending=False, phase={"phase": "ready"})
    cluster["status"] = "running"
    service._observe_deployment_events(
        [deployment("running", instances=_grouped_instance_summary(cluster))],
        runtime_deployments={"model-1": cluster},
    )
    assert [record.deployment_event for record in events(caplog)] == ["launched", "launched"]


def test_discovered_container_carries_explicit_stop_intent():
    from types import SimpleNamespace

    service = observer()
    service.manager = SimpleNamespace(_explicitly_stopped_containers={"legacy-vllm"})
    stopped = service._discovered_deployment(
        {"name": "legacy-vllm", "status": "exited"}, "vllm", "repo/model",
    )
    assert stopped["desired_state"] == "stopped"
    service.manager._explicitly_stopped_containers = set()
    crashed = service._discovered_deployment(
        {"name": "legacy-vllm", "status": "exited"}, "vllm", "repo/model",
    )
    assert crashed["desired_state"] == "running"


def test_removed_discovered_container_retired_when_only_images_unavailable(caplog):
    from unittest.mock import AsyncMock

    caplog.set_level(logging.INFO)

    async def scenario():
        service = observer()
        service.store = Mock()
        service.registry = SimpleNamespace(kinds=["vllm"])
        service._probe_external_endpoint = AsyncMock()
        # A managed card is registered so get_state() is invoked every poll.
        service._adopt_unlinked_manager_deployments = AsyncMock(side_effect=lambda *a, **k: [
            deployment("running", kind="managed", desired_state="running",
                       settings={"manager_deployment_id": "cluster-1"}),
        ])
        runtime = {"id": "cluster-1", "sparkdeck_record_id": "model-1",
                   "desired_state": "running", "status": "ready"}
        container_a = {"name": "legacy-vllm", "status": "running", "runtime": "vllm",
                       "model": "repo/model", "served_model": "repo/model"}
        service.manager = SimpleNamespace(
            deployments=[runtime],
            get_state=AsyncMock(side_effect=[
                {"deployments": [runtime], "docker_ready": True,
                 "containers_ready": True, "containers": [container_a]},
                # Container absence is authoritative even though the separate
                # image inventory failed and docker_ready is therefore false.
                {"deployments": [runtime], "docker_ready": False,
                 "containers_ready": True, "containers": []},
            ]),
            cluster_nodes=AsyncMock(return_value=[]),
        )
        await service.deployments(observe_events=True)
        await service.deployments(observe_events=True)

    asyncio.run(scenario())
    retirements = [
        record for record in events(caplog)
        if "removed from inventory" in record.getMessage()
    ]
    assert len(retirements) == 1
    assert retirements[0].deployment_event == "stopped"
