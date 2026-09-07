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
