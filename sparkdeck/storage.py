"""SQLite persistence for local settings, deployments, benchmarks and sync."""

from __future__ import annotations

import json
import math
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .models import BenchmarkSample, Deployment, DeploymentKind, ModelIdentity, RuntimeKind


COMMUNITY_UPLOAD_FIELDS = frozenset({
    "model_id", "context_window_size", "inference_tokens_per_second",
})
COMMUNITY_EVIDENCE_POLICY = {
    "minimum_samples": 10,
    "exact_match_dimensions": ["model_id", "context_window_size"],
    "metric": "inference_tokens_per_second",
}
# The hosted community API works out of the box; clearing the setting keeps
# the installation fully local.
DEFAULT_COMMUNITY_API_URL = os.environ.get(
    "SPARKDECK_COMMUNITY_API_URL",
    "https://oqft567ar3.execute-api.us-east-2.amazonaws.com",
)
_COMMUNITY_AGGREGATE_BATCH_SIZE = 256


class SparkDeckStore:
    """Small synchronous store; operations are short and serialized locally."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._migrate()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    @contextmanager
    def locked(self):
        """Hold the store's re-entrant lock across a compound state change."""
        with self._lock:
            yield

    def _migrate(self) -> None:
        with self._lock, self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS deployments (
                    id TEXT PRIMARY KEY,
                    alias TEXT NOT NULL UNIQUE,
                    runtime TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    repository TEXT NOT NULL,
                    revision TEXT,
                    artifact TEXT,
                    quantization TEXT,
                    base_url TEXT,
                    container_name TEXT,
                    settings_json TEXT NOT NULL DEFAULT '{}',
                    credential_ref TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS benchmark_samples (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    deployment_id TEXT,
                    model_json TEXT NOT NULL,
                    runtime TEXT NOT NULL,
                    runtime_version TEXT,
                    hardware_json TEXT NOT NULL,
                    configuration_json TEXT NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    latency_ms REAL NOT NULL,
                    ttft_ms REAL,
                    generation_tps REAL,
                    prompt_tps REAL,
                    cold_start INTEGER,
                    eligible INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS benchmark_samples_created_at
                    ON benchmark_samples(created_at DESC);
                CREATE TABLE IF NOT EXISTS upload_outbox (
                    sample_id TEXT PRIMARY KEY REFERENCES benchmark_samples(id) ON DELETE CASCADE,
                    status TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    next_attempt_at TEXT,
                    last_error TEXT,
                    created_at TEXT NOT NULL
                );
                """
            )
            self._connection.execute(
                "INSERT OR IGNORE INTO settings(key, value_json) VALUES (?, ?)",
                ("community_consent", "false"),
            )
            self._connection.execute(
                "INSERT OR IGNORE INTO settings(key, value_json) VALUES (?, ?)",
                ("device_pairing", json.dumps({"status": "not_paired"})),
            )
            self._connection.execute(
                "INSERT OR IGNORE INTO settings(key, value_json) VALUES (?, ?)",
                ("community_api_url", json.dumps(DEFAULT_COMMUNITY_API_URL)),
            )
            deployment_columns = {
                row[1] for row in self._connection.execute(
                    "PRAGMA table_info(deployments)"
                )
            }
            if "created_at" not in deployment_columns:
                self._connection.execute(
                    "ALTER TABLE deployments ADD COLUMN created_at TEXT"
                )
            self._connection.execute(
                "UPDATE deployments SET created_at = ? "
                "WHERE created_at IS NULL OR TRIM(created_at) = ''",
                (datetime.now(timezone.utc).isoformat(),),
            )
            # Older versions considered samples uploadable without the two
            # measurements required by the public aggregate. Fail closed when
            # opening an existing database and discard only their unsent upload
            # instructions; the full benchmark rows remain local.
            invalid_sample_ids: list[str] = []
            for row in self._connection.execute(
                "SELECT id, configuration_json, generation_tps "
                "FROM benchmark_samples WHERE eligible = 1"
            ).fetchall():
                try:
                    configuration = json.loads(row["configuration_json"] or "{}")
                except (TypeError, ValueError, json.JSONDecodeError):
                    configuration = {}
                if (
                    not isinstance(configuration, dict)
                    or community_context_window(configuration) is None
                    or _positive_speed(row["generation_tps"]) is None
                ):
                    invalid_sample_ids.append(row["id"])
            if invalid_sample_ids:
                self._connection.executemany(
                    "UPDATE benchmark_samples SET eligible = 0 WHERE id = ?",
                    ((sample_id,) for sample_id in invalid_sample_ids),
                )
                marks = ",".join("?" for _ in invalid_sample_ids)
                self._connection.execute(
                    f"DELETE FROM upload_outbox WHERE sample_id IN ({marks}) "
                    "AND status IN ('pending', 'failed', 'waiting_for_account')",
                    tuple(invalid_sample_ids),
                )

    def get_setting(self, key: str, default: Any = None) -> Any:
        with self._lock:
            row = self._connection.execute(
                "SELECT value_json FROM settings WHERE key = ?", (key,)
            ).fetchone()
        return default if row is None else json.loads(row["value_json"])

    def set_setting(self, key: str, value: Any) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT INTO settings(key, value_json) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json",
                (key, json.dumps(value)),
            )

    def add_deployment(self, deployment: Deployment, base_url: str | None = None,
                       credential_ref: str | None = None) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """INSERT INTO deployments(
                    id, alias, runtime, kind, repository, revision, artifact,
                    quantization, base_url, container_name, settings_json, credential_ref,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    deployment.id, deployment.alias, deployment.runtime.value,
                    deployment.kind.value, deployment.model.repository,
                    deployment.model.revision, deployment.model.artifact,
                    deployment.model.quantization, base_url, deployment.container_name,
                    json.dumps(deployment.settings), credential_ref,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def update_container(self, deployment_id: str, container_name: str) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "UPDATE deployments SET container_name = ? WHERE id = ?",
                (container_name, deployment_id),
            )

    def update_alias(self, deployment_id: str, alias: str) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "UPDATE deployments SET alias = ? WHERE id = ?",
                (alias, deployment_id),
            )

    def update_runtime_location(self, deployment_id: str, container_name: str,
                                base_url: str) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "UPDATE deployments SET container_name = ?, base_url = ? WHERE id = ?",
                (container_name, base_url, deployment_id),
            )

    def update_managed_routing(
        self, deployment_id: str, settings: dict[str, Any],
        container_name: str | None, base_url: str | None,
    ) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "UPDATE deployments SET settings_json = ?, container_name = ?, base_url = ? "
                "WHERE id = ?",
                (json.dumps(settings), container_name, base_url, deployment_id),
            )

    def delete_deployment(self, deployment_id: str) -> None:
        with self._lock, self._connection:
            self._connection.execute("DELETE FROM deployments WHERE id = ?", (deployment_id,))

    def deployments(self, include_private: bool = False) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM deployments ORDER BY alias COLLATE NOCASE"
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            deployment = Deployment(
                id=row["id"], alias=row["alias"], runtime=RuntimeKind(row["runtime"]),
                kind=DeploymentKind(row["kind"]),
                model=ModelIdentity(
                    repository=row["repository"], revision=row["revision"],
                    artifact=row["artifact"], quantization=row["quantization"],
                ),
                container_name=row["container_name"],
                settings=json.loads(row["settings_json"] or "{}"),
                base_url_set=bool(row["base_url"]),
            ).to_dict()
            deployment["created_at"] = row["created_at"]
            if include_private:
                deployment["_base_url"] = row["base_url"]
                deployment["_credential_ref"] = row["credential_ref"]
            result.append(deployment)
        return result

    def deployment(self, deployment_id_or_alias: str, include_private: bool = False) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM deployments WHERE id = ? OR alias = ?",
                (deployment_id_or_alias, deployment_id_or_alias),
            ).fetchone()
        if row is None:
            return None
        items = [d for d in self.deployments(include_private=include_private) if d["id"] == row["id"]]
        return items[0] if items else None

    def add_benchmark(self, sample: BenchmarkSample, queue: bool) -> None:
        value = sample.to_dict()
        community_eligible = bool(
            sample.eligible_for_community
            and community_context_window(sample.configuration) is not None
            and _positive_speed(sample.generation_tokens_per_second) is not None
        )
        with self._lock, self._connection:
            self._connection.execute(
                """INSERT INTO benchmark_samples VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )""",
                (
                    sample.id, sample.created_at, sample.deployment_id,
                    json.dumps(value["model"]), sample.runtime.value,
                    sample.runtime_version, json.dumps(sample.hardware),
                    json.dumps(sample.configuration), sample.input_tokens,
                    sample.output_tokens, sample.latency_ms, sample.ttft_ms,
                    sample.generation_tokens_per_second,
                    sample.prompt_tokens_per_second,
                    None if sample.cold_start is None else int(sample.cold_start),
                    int(community_eligible),
                ),
            )
            if queue and community_eligible:
                pairing = self.get_setting("device_pairing", {"status": "not_paired"})
                status = "pending" if pairing.get("status") == "paired" else "waiting_for_account"
                self._connection.execute(
                    "INSERT INTO upload_outbox(sample_id, status, created_at) VALUES (?, ?, ?)",
                    (sample.id, status, sample.created_at),
                )

    def benchmarks(self, limit: int = 100, offset: int = 0) -> tuple[list[dict[str, Any]], int]:
        limit = min(500, max(1, int(limit)))
        offset = max(0, int(offset))
        with self._lock:
            total = self._connection.execute("SELECT COUNT(*) FROM benchmark_samples").fetchone()[0]
            rows = self._connection.execute(
                """SELECT benchmark_samples.*, upload_outbox.status AS sync_state
                   FROM benchmark_samples
                   LEFT JOIN upload_outbox ON upload_outbox.sample_id = benchmark_samples.id
                   ORDER BY benchmark_samples.created_at DESC LIMIT ? OFFSET ?""",
                (limit, offset),
            ).fetchall()
        items = [_benchmark_row(row) for row in rows]
        return items, total

    def community_aggregates(self) -> list[dict[str, Any]]:
        """Aggregate only the fields that are eligible for community sharing.

        This provides useful evidence for fully local installations while
        keeping the public aggregate boundary identical to the upload payload.
        Rows are grouped by the exact dimensions declared in
        ``COMMUNITY_EVIDENCE_POLICY`` and no private benchmark metadata leaves
        this method.
        """
        grouped: dict[tuple[str, int], tuple[float, int]] = {}
        with self._lock:
            cursor = self._connection.execute(
                "SELECT model_json, configuration_json, generation_tps "
                "FROM benchmark_samples WHERE eligible = 1"
            )
            while True:
                rows = cursor.fetchmany(_COMMUNITY_AGGREGATE_BATCH_SIZE)
                if not rows:
                    break
                for row in rows:
                    try:
                        model = json.loads(row["model_json"] or "{}")
                        configuration = json.loads(
                            row["configuration_json"] or "{}"
                        )
                    except (TypeError, ValueError, json.JSONDecodeError):
                        continue
                    if not isinstance(model, dict) or not isinstance(configuration, dict):
                        continue
                    model_id = str(model.get("repository") or "").strip()
                    context_window = community_context_window(configuration)
                    speed = _positive_speed(row["generation_tps"])
                    if not model_id or context_window is None or speed is None:
                        continue
                    key = (model_id, context_window)
                    total, count = grouped.get(key, (0.0, 0))
                    grouped[key] = (total + speed, count + 1)

        items = [
            {
                "model_id": model_id,
                "context_window_size": context_window,
                "inference_tokens_per_second": total / count,
                "sample_count": count,
            }
            for (model_id, context_window), (total, count) in grouped.items()
        ]
        return sorted(
            items,
            key=lambda item: (
                -item["sample_count"], item["model_id"].casefold(),
                item["context_window_size"],
            ),
        )

    def delete_benchmark(self, sample_id: str) -> bool:
        with self._lock, self._connection:
            cursor = self._connection.execute("DELETE FROM benchmark_samples WHERE id = ?", (sample_id,))
        return bool(cursor.rowcount)

    def retry_outbox(self) -> int:
        with self._lock, self._connection:
            pairing = self.get_setting("device_pairing", {"status": "not_paired"})
            status = "pending" if pairing.get("status") == "paired" else "waiting_for_account"
            cursor = self._connection.execute(
                "UPDATE upload_outbox SET status = ?, next_attempt_at = NULL, last_error = NULL "
                "WHERE status IN ('failed', 'waiting_for_account')",
                (status,),
            )
        return cursor.rowcount

    def promote_outbox_for_pairing(self) -> int:
        """Move account-waiting uploads to pending after pairing succeeds."""
        with self._lock, self._connection:
            cursor = self._connection.execute(
                "UPDATE upload_outbox SET status = 'pending' "
                "WHERE status = 'waiting_for_account'",
            )
        return cursor.rowcount

    def outbox_batch(self, limit: int = 50, now: str | None = None) -> list[dict[str, Any]]:
        """Return an idempotent upload batch without exposing private deployment data."""
        return [item["payload"] for item in self.outbox_entries(limit, now)]

    def outbox_entries(self, limit: int = 50, now: str | None = None) -> list[dict[str, Any]]:
        """Return private queue identifiers alongside privacy-safe upload payloads."""
        limit = min(200, max(1, int(limit)))
        now = now or datetime.now(timezone.utc).isoformat()
        with self._lock:
            rows = self._connection.execute(
                """SELECT sample_id FROM upload_outbox
                   WHERE status = 'pending'
                      OR (status = 'failed' AND (next_attempt_at IS NULL OR next_attempt_at <= ?))
                   ORDER BY created_at LIMIT ?""",
                (now, limit),
            ).fetchall()
        if not rows:
            return []
        wanted = [row["sample_id"] for row in rows]
        marks = ",".join("?" for _ in wanted)
        with self._lock:
            sample_rows = self._connection.execute(
                f"SELECT * FROM benchmark_samples WHERE id IN ({marks})",
                tuple(wanted),
            ).fetchall()
        by_id = {
            row["id"]: payload
            for row in sample_rows
            if (payload := _upload_row(row)) is not None
        }
        return [
            {"sample_id": sample_id, "payload": by_id[sample_id]}
            for sample_id in wanted if sample_id in by_id
        ]

    def outbox_entry(
        self, sample_id: str, now: str | None = None,
    ) -> dict[str, Any] | None:
        """Re-read one currently uploadable sample at the outbound boundary."""
        now = now or datetime.now(timezone.utc).isoformat()
        with self._lock:
            row = self._connection.execute(
                """SELECT benchmark_samples.*
                   FROM upload_outbox
                   JOIN benchmark_samples
                     ON benchmark_samples.id = upload_outbox.sample_id
                   WHERE upload_outbox.sample_id = ?
                     AND (upload_outbox.status = 'pending'
                       OR (upload_outbox.status = 'failed'
                         AND (upload_outbox.next_attempt_at IS NULL
                           OR upload_outbox.next_attempt_at <= ?)))""",
                (sample_id, now),
            ).fetchone()
        if row is None or (payload := _upload_row(row)) is None:
            return None
        return {"sample_id": sample_id, "payload": payload}

    def mark_outbox_synced(self, sample_ids: list[str]) -> int:
        if not sample_ids:
            return 0
        marks = ",".join("?" for _ in sample_ids)
        with self._lock, self._connection:
            cursor = self._connection.execute(
                f"UPDATE upload_outbox SET status = 'synced', last_error = NULL, "
                f"next_attempt_at = NULL WHERE sample_id IN ({marks})",
                tuple(sample_ids),
            )
        return cursor.rowcount

    def mark_outbox_failed(self, sample_ids: list[str], error: str) -> int:
        """Record bounded exponential retry state for a failed batch."""
        if not sample_ids:
            return 0
        updated = 0
        with self._lock, self._connection:
            for sample_id in sample_ids:
                row = self._connection.execute(
                    "SELECT attempts FROM upload_outbox WHERE sample_id = ?", (sample_id,)
                ).fetchone()
                if row is None:
                    continue
                attempts = int(row["attempts"]) + 1
                delay = min(3600, 15 * (2 ** min(attempts - 1, 8)))
                next_at = (datetime.now(timezone.utc) + timedelta(seconds=delay)).isoformat()
                self._connection.execute(
                    "UPDATE upload_outbox SET status = 'failed', attempts = ?, "
                    "next_attempt_at = ?, last_error = ? WHERE sample_id = ?",
                    (attempts, next_at, str(error)[:500], sample_id),
                )
                updated += 1
        return updated

    def sync_status(self) -> dict[str, Any]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT status, COUNT(*) AS count FROM upload_outbox GROUP BY status"
            ).fetchall()
        counts = {row["status"]: row["count"] for row in rows}
        pairing = self.get_setting("device_pairing", {"status": "not_paired"})
        return {
            "consent": bool(self.get_setting("community_consent", False)),
            # Account claims and upload credentials are private service state.
            "pairing": {
                "status": "paired" if pairing.get("status") == "paired" else "not_paired",
            },
            "outbox": {
                "pending": counts.get("pending", 0),
                "waiting_for_account": counts.get("waiting_for_account", 0),
                "failed": counts.get("failed", 0),
                "synced": counts.get("synced", 0),
            },
            "cloud_endpoint_configured": bool(self.get_setting("community_api_url", None)),
        }

    def set_community_consent(self, enabled: bool) -> None:
        """Persist consent and queue eligible local samples only after opt-in."""
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT INTO settings(key, value_json) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json",
                ("community_consent", json.dumps(bool(enabled))),
            )
            if not enabled:
                # Withdrawing consent removes every unsent upload instruction,
                # while the benchmark_samples rows remain available locally.
                self._connection.execute(
                    "DELETE FROM upload_outbox WHERE status IN "
                    "('pending', 'failed', 'waiting_for_account')"
                )
                return
            pairing = self.get_setting("device_pairing", {"status": "not_paired"})
            status = "pending" if pairing.get("status") == "paired" else "waiting_for_account"
            self._connection.execute(
                """INSERT OR IGNORE INTO upload_outbox(sample_id, status, created_at)
                   SELECT id, ?, created_at FROM benchmark_samples WHERE eligible = 1""",
                (status,),
            )


def _benchmark_row(row: sqlite3.Row) -> dict[str, Any]:
    hardware = json.loads(row["hardware_json"])
    # Normalize records written before the public benchmark contract settled on
    # ``hardware_class`` so local history remains useful without leaking any
    # additional machine identity.
    if "hardware_class" not in hardware and "device_class" in hardware:
        hardware["hardware_class"] = hardware.pop("device_class")
    value = {
        "id": row["id"], "created_at": row["created_at"],
        "deployment_id": row["deployment_id"], "model": json.loads(row["model_json"]),
        "runtime": row["runtime"], "runtime_version": row["runtime_version"],
        "hardware": hardware,
        "configuration": json.loads(row["configuration_json"]),
        "input_tokens": row["input_tokens"], "output_tokens": row["output_tokens"],
        "latency_ms": row["latency_ms"], "ttft_ms": row["ttft_ms"],
        "generation_tokens_per_second": row["generation_tps"],
        "prompt_tokens_per_second": row["prompt_tps"],
        "cold_start": None if row["cold_start"] is None else bool(row["cold_start"]),
        "eligible_for_community": bool(row["eligible"]),
    }
    if "sync_state" in row.keys():
        value["sync_state"] = row["sync_state"] or "local"
    return value


def _upload_row(row: sqlite3.Row) -> dict[str, Any] | None:
    """Build the exact three-field cloud payload, separate from local history."""
    value = _benchmark_row(row)
    context_window = community_context_window(value["configuration"])
    speed = _positive_speed(value["generation_tokens_per_second"])
    model_id = str(value.get("model", {}).get("repository") or "").strip()
    if not model_id or context_window is None or speed is None:
        return None
    payload = {
        "model_id": model_id,
        "context_window_size": context_window,
        "inference_tokens_per_second": speed,
    }
    assert set(payload) == COMMUNITY_UPLOAD_FIELDS
    return payload


def community_context_window(configuration: dict[str, Any]) -> int | None:
    for key in ("max_model_len", "context_length", "context_size"):
        value = configuration.get(key)
        if isinstance(value, bool):
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return None


def _positive_speed(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 and math.isfinite(parsed) else None
