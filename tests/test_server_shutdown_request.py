"""Regression tests for PID-scoped launcher shutdown requests."""

import os

import server


def test_current_process_request_is_consumed(tmp_path):
    shutdown_file = tmp_path / "shutdown.request"
    shutdown_file.write_text(str(os.getpid()), encoding="utf-8")

    assert server._consume_shutdown_request(
        shutdown_file, frozenset({os.getpid()})
    )
    assert not shutdown_file.exists()


def test_foreign_process_request_is_discarded(tmp_path):
    shutdown_file = tmp_path / "shutdown.request"
    shutdown_file.write_text("99999999", encoding="utf-8")

    assert not server._consume_shutdown_request(
        shutdown_file, frozenset({os.getpid()})
    )
    assert not shutdown_file.exists()


def test_malformed_request_is_discarded(tmp_path):
    shutdown_file = tmp_path / "shutdown.request"
    shutdown_file.write_text("stop now", encoding="utf-8")

    assert not server._consume_shutdown_request(
        shutdown_file, frozenset({os.getpid()})
    )
    assert not shutdown_file.exists()


def test_startup_discards_stale_request_before_new_request(tmp_path):
    shutdown_file = tmp_path / "shutdown.request"
    shutdown_file.write_text(str(os.getpid()), encoding="utf-8")

    server._discard_shutdown_request(shutdown_file)
    assert not shutdown_file.exists()

    shutdown_file.write_text(str(os.getpid()), encoding="utf-8")
    assert server._consume_shutdown_request(
        shutdown_file, frozenset({os.getpid()})
    )


def test_server_pid_is_always_an_accepted_shutdown_identity():
    assert os.getpid() in server._shutdown_request_process_ids()
