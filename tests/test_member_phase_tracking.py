"""Regression tests for member phase tracking and error reconciliation.

Covers three production failure modes observed on a live grouped_sharded
cluster:

- a running member whose logs contain a handled/transient error frame (for
  example a torch/c10 backtrace from a recovered NCCL fault) must not latch
  ``phase: error`` and degrade the whole deployment;
- grouped_sharded rank > 0 members run headless (no HTTP API, no uvicorn
  startup marker) and must leave the ``starting…`` phase once their group's
  rank-0 coordinator is ready;
- stopping or inspecting a deployment whose container was never created
  (launch failed at image pull) must not persist Docker's 404 as the
  deployment's persistent error.
"""

import asyncio
from unittest.mock import AsyncMock

from manager import Manager
from sparkdeck.service import (
    _missing_container_error,
    _stale_missing_container_error,
)


def _phase_for(logs, *, status="running", ready=False, **container):
    manager = Manager.__new__(Manager)
    manager._check_ready = AsyncMock(return_value=ready)
    manager.get_logs = AsyncMock(return_value=logs)
    target = {"name": "member-1", "status": status, **container}
    return asyncio.run(manager._get_container_phase(target))


# --- (a) log-scraped error latching ---------------------------------------

TRANSIENT_C10_BACKTRACE = "\n".join([
    "INFO:     Started server process [1]",
    "Traceback (most recent call last):",
    "frame #0: c10::Error::Error(...) + 0xc8 (0x7f1a2b in /usr/local/lib/python3.12/dist-packages/torch/lib/libc10.so)",
    "frame #1: c10d::ProcessGroupNCCL::heartbeatMonitor(...) + 0x1f0 (0x7f1a3c in libtorch_cuda.so)",
    "INFO:     NCCL watchdog recovered; continuing",
    'INFO:     10.0.0.2:51234 - "POST /v1/chat/completions HTTP/1.1" 200 OK',
])


def test_transient_backtrace_with_continued_output_is_not_an_error():
    phase = _phase_for(TRANSIENT_C10_BACKTRACE)

    assert phase["phase"] != "error"


def test_transient_backtrace_does_not_degrade_grouped_deployment():
    worker_phase = _phase_for(TRANSIENT_C10_BACKTRACE)
    deployment = {
        "mode": "grouped_sharded",
        "launch_settings": {"tensor_parallel_size": 2},
        "members": [
            {
                "instance_id": 0, "rank": 0, "status": "running",
                "phase": {"phase": "ready", "progress": 1.0},
            },
            {"instance_id": 0, "rank": 1, "status": "running", "phase": worker_phase},
        ],
    }

    assert Manager._grouped_deployment_status(deployment) == "running"


def test_latest_output_error_still_marks_the_phase_failed():
    logs = "\n".join([
        "INFO:     Started server process [1]",
        "Traceback (most recent call last):",
        '  File "/srv/vllm/engine.py", line 12, in <module>',
        "RuntimeError: CUDA error: no kernel image is available",
    ])

    phase = _phase_for(logs)

    assert phase["phase"] == "error"
    assert "RuntimeError: CUDA error" in phase["message"]


def test_repeated_distinct_errors_in_recent_tail_mark_the_phase_failed():
    logs = "\n".join([
        "INFO: boot",
        "RuntimeError: first engine fault",
        "INFO: retrying",
        "RuntimeError: second engine fault",
        "INFO: retrying again",
    ])

    assert _phase_for(logs)["phase"] == "error"


def test_ready_probe_still_wins_over_error_like_logs():
    phase = _phase_for(TRANSIENT_C10_BACKTRACE, ready=True)

    assert phase["phase"] == "ready"


# --- (b) stale starting phase on headless ranks ----------------------------

def _group_members():
    return [
        {
            "instance_id": 0, "rank": 0, "status": "running",
            "phase": {"phase": "ready", "progress": 1.0, "message": "vLLM API ready"},
        },
        {
            "instance_id": 0, "rank": 1, "status": "running",
            "phase": {"phase": "starting", "progress": None, "message": "starting…"},
        },
    ]


def test_headless_rank_leaves_starting_once_group_leader_is_ready():
    members = _group_members()

    Manager._fold_headless_member_phases(members)

    assert members[1]["phase"]["phase"] == "ready"
    assert members[0]["phase"]["message"] == "vLLM API ready"


def test_headless_rank_stays_starting_until_group_leader_is_ready():
    members = _group_members()
    members[0]["phase"] = {"phase": "loading", "progress": 0.5}

    Manager._fold_headless_member_phases(members)

    assert members[1]["phase"]["phase"] == "starting"


def test_headless_rank_honest_failure_is_never_folded_into_ready():
    members = _group_members()
    members[1]["phase"] = {"phase": "error", "message": "NCCL init failed"}

    Manager._fold_headless_member_phases(members)

    assert members[1]["phase"]["phase"] == "error"


def test_headless_rank_without_running_container_is_not_folded():
    members = _group_members()
    members[1]["status"] = "restarting"

    Manager._fold_headless_member_phases(members)

    assert members[1]["phase"]["phase"] == "starting"


# --- (c) phantom missing-container error ----------------------------------

DOCKER_404 = (
    "404 Client Error for http+docker://localhost/v1.43/containers/"
    "cluster-abc123-r0-model: Not Found "
    '("No such container: cluster-abc123-r0-model")'
)


def test_missing_container_error_marker_detection():
    assert _missing_container_error(DOCKER_404)
    assert _missing_container_error("managed container not found")
    assert not _missing_container_error("image pull failed: manifest unknown")
    assert not _missing_container_error(None)


def test_stop_of_never_created_member_is_not_an_actionable_error():
    results = [RuntimeError(DOCKER_404)]

    assert Manager._member_action_errors(results, "stop") == []
    assert Manager._member_action_errors(results, "remove") == []
    # A start against a missing container is a genuine failure.
    assert Manager._member_action_errors(results, "start") == [DOCKER_404]


def test_stop_of_never_created_group_does_not_persist_the_404():
    deployment = {
        "id": "d1",
        "mode": "grouped_sharded",
        "status": "error",
        "desired_state": "running",
        "error": "image pull failed",
        "members": [
            {
                "instance_id": 0, "rank": rank, "node_id": f"node-{rank}",
                "container_name": f"cluster-d1-r{rank}-model", "status": "error",
            }
            for rank in (0, 1)
        ],
    }
    manager = Manager.__new__(Manager)
    manager.deployments = [deployment]
    manager._save_deployments = lambda: None

    async def missing_member(member, action, *, log_tail=300):
        raise RuntimeError(DOCKER_404)

    manager._member_action = missing_member

    result = asyncio.run(manager.deployment_action("d1", "stop"))

    assert result["ok"]
    assert deployment["status"] == "stopped"
    assert deployment["error"] is None
    for member in deployment["members"]:
        assert member["status"] == "stopped"
        assert "failed_stop_error" not in member


def test_stale_missing_container_error_is_suppressed_only_when_never_launched():
    cluster = {
        "desired_state": "stopped",
        "last_deployed_at": None,
        "error": DOCKER_404,
    }
    assert _stale_missing_container_error(cluster, DOCKER_404)

    launched = {**cluster, "last_deployed_at": 1_700_000_000}
    assert not _stale_missing_container_error(launched, DOCKER_404)

    running = {**cluster, "desired_state": "running"}
    assert not _stale_missing_container_error(running, DOCKER_404)

    assert not _stale_missing_container_error(cluster, "image pull failed")
