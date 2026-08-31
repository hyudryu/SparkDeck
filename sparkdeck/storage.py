"""SQLite persistence for local settings, deployments, benchmarks and sync."""

from __future__ import annotations

import json
import math
import os
import sqlite3
import statistics
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .catalog import canonical_quantization
from .models import BenchmarkSample, Deployment, DeploymentKind, ModelIdentity, RuntimeKind


COMMUNITY_UPLOAD_FIELDS = frozenset({
    "model_id", "quantization", "prompt_tokens_bucket",
    "inference_tokens_per_second", "telemetry_cluster_id",
    "concurrency",
})
COMMUNITY_CONSENT_CONTRACT_VERSION = 3
COMMUNITY_EVIDENCE_POLICY = {
    "minimum_samples": 10,
    "exact_match_dimensions": [
        "model_id", "quantization", "prompt_tokens_bucket",
    ],
    "metric": "inference_tokens_per_second",
}
# The community backend is built in — every installation talks to the hosted
# SparkDeck community API. Not user-configurable; the env override exists for
# development and forks (set it empty to stay fully local while developing).
COMMUNITY_API_URL = os.environ.get(
    "SPARKDECK_COMMUNITY_API_URL",
    "https://oqft567ar3.execute-api.us-east-2.amazonaws.com",
)
_COMMUNITY_AGGREGATE_BATCH_SIZE = 256
_LOCAL_HISTORY_MODEL_LIMIT = 500


def _outlier_filtered_mean(values: list[float]) -> float:
    """Return a Tukey-IQR mean without discarding small sample sets."""
    if len(values) < 4:
        return statistics.fmean(values)
    q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
    spread = q3 - q1
    lower, upper = q1 - 1.5 * spread, q3 + 1.5 * spread
    retained = [value for value in values if lower <= value <= upper]
    return statistics.fmean(retained or values)


def _restrict_permissions(path: Path, mode: int) -> None:
    """Best-effort owner-only permissions; Windows has no POSIX modes."""
    try:
        path.chmod(mode)
    except OSError:
        pass


class SparkDeckStore:
    """Small synchronous store; operations are short and serialized locally."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _restrict_permissions(self.path.parent, 0o700)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._migrate()
        # The store holds the community refresh token; keep it owner-only.
        _restrict_permissions(self.path, 0o600)
        for suffix in ("-wal", "-shm"):
            sidecar = Path(f"{self.path}{suffix}")
            if sidecar.exists():
                _restrict_permissions(sidecar, 0o600)

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
                    desired_state TEXT NOT NULL DEFAULT 'running',
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
                    eligible INTEGER NOT NULL,
                    telemetry_cluster_id TEXT
                );
                CREATE INDEX IF NOT EXISTS benchmark_samples_created_at
                    ON benchmark_samples(created_at DESC);
                CREATE TABLE IF NOT EXISTS benchmark_series_points (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    deployment_id TEXT,
                    model_id TEXT NOT NULL,
                    context_window_size INTEGER NOT NULL,
                    concurrency INTEGER NOT NULL,
                    tensor_parallel_size INTEGER NOT NULL,
                    prompt_tps REAL NOT NULL,
                    generation_tps REAL NOT NULL,
                    request_count INTEGER NOT NULL,
                    sample_id TEXT REFERENCES benchmark_samples(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS benchmark_series_model
                    ON benchmark_series_points(model_id, tensor_parallel_size, context_window_size, concurrency);
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
                ("community_consent_generation", "0"),
            )
            consent_enabled = bool(self.get_setting("community_consent", False))
            consent_version_row = self._connection.execute(
                "SELECT value_json FROM settings WHERE key = ?",
                ("community_consent_contract_version",),
            ).fetchone()
            try:
                consent_version = (
                    json.loads(consent_version_row["value_json"])
                    if consent_version_row is not None else None
                )
            except (TypeError, ValueError, json.JSONDecodeError):
                consent_version = None
            if consent_version != COMMUNITY_CONSENT_CONTRACT_VERSION:
                # Quantization and the opaque cluster identifier widen the
                # public payload. Existing opt-ins must not silently authorize
                # that new contract. Invalidate any in-flight consent snapshot
                # when migration actually transitions sharing from on to off.
                self._connection.execute(
                    "UPDATE settings SET value_json = 'false' "
                    "WHERE key = 'community_consent'"
                )
                if consent_enabled:
                    generation = int(self.get_setting(
                        "community_consent_generation", 0,
                    )) + 1
                    self._connection.execute(
                        "UPDATE settings SET value_json = ? WHERE key = ?",
                        (json.dumps(generation), "community_consent_generation"),
                    )
                self._connection.execute(
                    "DELETE FROM upload_outbox WHERE status IN "
                    "('pending', 'failed', 'waiting_for_account')"
                )
                self._connection.execute(
                    "INSERT INTO settings(key, value_json) VALUES (?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json",
                    (
                        "community_consent_contract_version",
                        json.dumps(COMMUNITY_CONSENT_CONTRACT_VERSION),
                    ),
                )
            self._connection.execute(
                "INSERT OR IGNORE INTO settings(key, value_json) VALUES (?, ?)",
                ("device_pairing", json.dumps({"status": "not_paired"})),
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
            if "desired_state" not in deployment_columns:
                self._connection.execute(
                    "ALTER TABLE deployments ADD COLUMN desired_state TEXT "
                    "NOT NULL DEFAULT 'running'"
                )
            self._connection.execute(
                "UPDATE deployments SET created_at = ? "
                "WHERE created_at IS NULL OR TRIM(created_at) = ''",
                (datetime.now(timezone.utc).isoformat(),),
            )
            series_columns = {
                row[1] for row in self._connection.execute(
                    "PRAGMA table_info(benchmark_series_points)"
                )
            }
            if "sample_id" not in series_columns:
                self._connection.execute(
                    "ALTER TABLE benchmark_series_points ADD COLUMN sample_id TEXT "
                    "REFERENCES benchmark_samples(id) ON DELETE CASCADE"
                )
            self._connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS benchmark_series_sample "
                "ON benchmark_series_points(sample_id) WHERE sample_id IS NOT NULL"
            )
            benchmark_columns = {
                row[1] for row in self._connection.execute(
                    "PRAGMA table_info(benchmark_samples)"
                )
            }
            if "telemetry_cluster_id" not in benchmark_columns:
                self._connection.execute(
                    "ALTER TABLE benchmark_samples "
                    "ADD COLUMN telemetry_cluster_id TEXT"
                )
            self._backfill_benchmark_series_links()
            # Older versions considered samples uploadable without the two
            # measurements required by the public aggregate. Fail closed when
            # opening an existing database and discard only their unsent upload
            # instructions; the full benchmark rows remain local.
            invalid_sample_ids: list[str] = []
            for row in self._connection.execute(
                "SELECT id, input_tokens, generation_tps, telemetry_cluster_id "
                "FROM benchmark_samples WHERE eligible = 1"
            ).fetchall():
                if (
                    _community_prompt_tokens(row["input_tokens"]) is None
                    or _positive_speed(row["generation_tps"]) is None
                    or not _valid_telemetry_cluster_id(
                        row["telemetry_cluster_id"]
                    )
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

    def _backfill_benchmark_series_links(self) -> None:
        """Link legacy coordinated rows when their local sample is unambiguous."""
        points = self._connection.execute(
            """SELECT id, created_at, deployment_id, model_id,
                      context_window_size, concurrency, tensor_parallel_size
               FROM benchmark_series_points WHERE sample_id IS NULL"""
        ).fetchall()
        point_matches: dict[str, list[str]] = {}
        sample_match_counts: dict[str, int] = {}
        for point in points:
            candidates = self._connection.execute(
                """SELECT id, model_json, configuration_json
                   FROM benchmark_samples
                   WHERE created_at = ? AND deployment_id IS ?
                     AND NOT EXISTS (
                         SELECT 1 FROM benchmark_series_points linked
                         WHERE linked.sample_id = benchmark_samples.id
                     )""",
                (point["created_at"], point["deployment_id"]),
            ).fetchall()
            matches = [
                row["id"] for row in candidates
                if _benchmark_matches_series_point(row, point)
            ]
            point_matches[point["id"]] = matches
            for sample_id in matches:
                sample_match_counts[sample_id] = sample_match_counts.get(sample_id, 0) + 1
        for point_id, matches in point_matches.items():
            if len(matches) == 1 and sample_match_counts[matches[0]] == 1:
                self._connection.execute(
                    "UPDATE benchmark_series_points SET sample_id = ? WHERE id = ?",
                    (matches[0], point_id),
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
                    desired_state, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    deployment.id, deployment.alias, deployment.runtime.value,
                    deployment.kind.value, deployment.model.repository,
                    deployment.model.revision, deployment.model.artifact,
                    deployment.model.quantization, base_url, deployment.container_name,
                    json.dumps(deployment.settings), credential_ref,
                    deployment.desired_state,
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

    def update_deployment_model(
        self, deployment_id: str, model: dict[str, Any],
    ) -> None:
        """Persist edited model-identity fields of a saved deployment."""
        with self._lock, self._connection:
            self._connection.execute(
                "UPDATE deployments SET repository = ?, revision = ?, artifact = ?, "
                "quantization = ? WHERE id = ?",
                (
                    str(model.get("repository") or ""),
                    model.get("revision"),
                    model.get("artifact"),
                    model.get("quantization"),
                    deployment_id,
                ),
            )

    def update_desired_state(self, deployment_id: str, desired_state: str) -> None:
        if desired_state not in {"running", "stopped"}:
            raise ValueError("desired_state must be running or stopped")
        with self._lock, self._connection:
            self._connection.execute(
                "UPDATE deployments SET desired_state = ? WHERE id = ?",
                (desired_state, deployment_id),
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

    def update_saved_deployment_settings(
        self, deployment_id: str, settings: dict[str, Any],
        alias: str | None = None,
    ) -> None:
        """Persist a bookmark's settings and optional rename atomically."""
        with self._lock, self._connection:
            self._connection.execute(
                "UPDATE deployments SET settings_json = ? WHERE id = ?",
                (json.dumps(settings), deployment_id),
            )
            if alias is not None:
                self._connection.execute(
                    "UPDATE deployments SET alias = ? WHERE id = ?",
                    (alias, deployment_id),
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
                desired_state=row["desired_state"] or "running",
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
        with self._lock, self._connection:
            consent = self._community_consent_snapshot_locked()
            cluster_id = (
                consent["telemetry_cluster_id"]
                if queue and consent["enabled"] else None
            )
            community_eligible = self._insert_benchmark_locked(
                sample, cluster_id,
            )
            self._link_benchmark_series_point(sample)
            if queue and community_eligible:
                self._queue_benchmark_locked(sample)

    def add_benchmark_if_consented(
        self, sample: BenchmarkSample, expected_generation: int,
    ) -> bool:
        """Atomically persist and queue telemetry under one consent snapshot.

        The caller captures ``community_consent_snapshot`` when measurement
        begins. Turning sharing off (or changing cluster identity) increments
        the generation, so a request that finishes after that boundary cannot
        write telemetry even if sharing was subsequently enabled again.
        """
        with self._lock, self._connection:
            consent = self._community_consent_snapshot_locked()
            if (
                not consent["enabled"]
                or consent["generation"] != expected_generation
                or not consent["telemetry_cluster_id"]
                or not _community_sample_eligible(sample)
            ):
                return False
            if not self._insert_benchmark_locked(
                sample, consent["telemetry_cluster_id"],
            ):
                return False
            self._link_benchmark_series_point(sample)
            self._queue_benchmark_locked(sample)
            return True

    def _insert_benchmark_locked(
        self, sample: BenchmarkSample, telemetry_cluster_id: str | None,
    ) -> bool:
        value = sample.to_dict()
        community_eligible = bool(
            telemetry_cluster_id and _community_sample_eligible(sample)
        )
        self._connection.execute(
            """INSERT INTO benchmark_samples(
                id, created_at, deployment_id, model_json, runtime,
                runtime_version, hardware_json, configuration_json,
                input_tokens, output_tokens, latency_ms, ttft_ms,
                generation_tps, prompt_tps, cold_start, eligible,
                telemetry_cluster_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                sample.id, sample.created_at, sample.deployment_id,
                json.dumps(value["model"]), sample.runtime.value,
                sample.runtime_version, json.dumps(sample.hardware),
                json.dumps(sample.configuration), sample.input_tokens,
                sample.output_tokens, sample.latency_ms, sample.ttft_ms,
                sample.generation_tokens_per_second,
                sample.prompt_tokens_per_second,
                None if sample.cold_start is None else int(sample.cold_start),
                int(community_eligible), telemetry_cluster_id,
            ),
        )
        return community_eligible

    def _queue_benchmark_locked(self, sample: BenchmarkSample) -> None:
        pairing = self.get_setting("device_pairing", {"status": "not_paired"})
        status = (
            "pending" if pairing.get("status") == "paired"
            else "waiting_for_account"
        )
        self._connection.execute(
            "INSERT INTO upload_outbox(sample_id, status, created_at) "
            "VALUES (?, ?, ?)",
            (sample.id, status, sample.created_at),
        )

    def _link_benchmark_series_point(self, sample: BenchmarkSample) -> None:
        """Connect a coordinated chart point to its Local history sample."""
        value = sample.to_dict()
        candidates = self._connection.execute(
            """SELECT id, created_at, deployment_id, model_id,
                      context_window_size, concurrency, tensor_parallel_size
               FROM benchmark_series_points
               WHERE sample_id IS NULL AND created_at = ? AND deployment_id IS ?""",
            (sample.created_at, sample.deployment_id),
        ).fetchall()
        matches = [
            point["id"] for point in candidates
            if _benchmark_matches_series_point({
                "id": sample.id,
                "model_json": json.dumps(value["model"]),
                "configuration_json": json.dumps(value["configuration"]),
            }, point)
        ]
        if len(matches) == 1:
            self._connection.execute(
                "UPDATE benchmark_series_points SET sample_id = ? WHERE id = ?",
                (sample.id, matches[0]),
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

    def benchmark_history_models(
        self, limit: int = _LOCAL_HISTORY_MODEL_LIMIT,
    ) -> list[dict[str, Any]]:
        """Return one latest local-history row for each identified model."""
        limit = min(_LOCAL_HISTORY_MODEL_LIMIT, max(1, int(limit)))
        with self._lock:
            rows = self._connection.execute(
                """WITH samples AS (
                       SELECT benchmark_samples.*,
                              upload_outbox.status AS sync_state,
                              json_extract(model_json, '$.repository') AS model_id
                       FROM benchmark_samples
                       LEFT JOIN upload_outbox
                         ON upload_outbox.sample_id = benchmark_samples.id
                       WHERE json_valid(model_json)
                   ), ranked AS (
                       SELECT samples.*,
                              COUNT(*) OVER (PARTITION BY model_id) AS sample_count,
                              ROW_NUMBER() OVER (
                                  PARTITION BY model_id
                                  ORDER BY created_at DESC, id DESC
                              ) AS model_rank
                       FROM samples
                       WHERE model_id IS NOT NULL
                         AND TRIM(model_id) != ''
                         AND LOWER(model_id) != 'local-model'
                         AND LOWER(model_id) NOT LIKE 'local-model-%'
                   )
                   SELECT * FROM ranked
                   WHERE model_rank = 1
                   ORDER BY created_at DESC, id DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()
        items = []
        for row in rows:
            item = _benchmark_row(row)
            item["sample_count"] = int(row["sample_count"])
            items.append(item)
        return items

    def add_benchmark_series_point(self, point: dict[str, Any]) -> None:
        """Persist one coordinated benchmark run without prompts or outputs."""
        with self._lock, self._connection:
            self._connection.execute(
                """INSERT INTO benchmark_series_points(
                    id, created_at, deployment_id, model_id, context_window_size,
                    concurrency, tensor_parallel_size, prompt_tps, generation_tps,
                    request_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    point["id"], point["created_at"], point.get("deployment_id"),
                    point["model_id"], point["context_window_size"],
                    point["concurrency"], point["tensor_parallel_size"],
                    point["prompt_tokens_per_second"],
                    point["generation_tokens_per_second"], point["request_count"],
                ),
            )

    def add_coordinated_benchmark(
        self, point: dict[str, Any], sample: BenchmarkSample, queue: bool,
    ) -> None:
        """Atomically persist one chart point, Local-history row, and outbox row."""
        with self._lock, self._connection:
            consent = self._community_consent_snapshot_locked()
            cluster_id = (
                consent["telemetry_cluster_id"]
                if queue and consent["enabled"] else None
            )
            community_eligible = self._insert_benchmark_locked(
                sample, cluster_id,
            )
            self._insert_benchmark_series_point_locked(point, sample.id)
            if queue and community_eligible:
                self._queue_benchmark_locked(sample)

    def add_coordinated_benchmark_if_consented(
        self, point: dict[str, Any], sample: BenchmarkSample,
        expected_generation: int,
    ) -> bool:
        """Consent-aware atomic variant for coordinated telemetry."""
        with self._lock, self._connection:
            consent = self._community_consent_snapshot_locked()
            if (
                not consent["enabled"]
                or consent["generation"] != expected_generation
                or not consent["telemetry_cluster_id"]
                or not _community_sample_eligible(sample)
            ):
                return False
            if not self._insert_benchmark_locked(
                sample, consent["telemetry_cluster_id"],
            ):
                return False
            self._insert_benchmark_series_point_locked(point, sample.id)
            self._queue_benchmark_locked(sample)
            return True

    def _insert_benchmark_series_point_locked(
        self, point: dict[str, Any], sample_id: str,
    ) -> None:
        self._connection.execute(
            """INSERT INTO benchmark_series_points(
                id, created_at, deployment_id, model_id, context_window_size,
                concurrency, tensor_parallel_size, prompt_tps, generation_tps,
                request_count, sample_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                point["id"], point["created_at"], point.get("deployment_id"),
                point["model_id"], point["context_window_size"],
                point["concurrency"], point["tensor_parallel_size"],
                point["prompt_tokens_per_second"],
                point["generation_tokens_per_second"], point["request_count"],
                sample_id,
            ),
        )

    def benchmark_model_summaries(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                """SELECT model_id, COUNT(*) AS run_count,
                          MAX(prompt_tps) AS best_prompt_tps,
                          MAX(generation_tps) AS best_generation_tps,
                          MAX(created_at) AS latest_at
                   FROM benchmark_series_points
                   GROUP BY model_id ORDER BY latest_at DESC, model_id COLLATE NOCASE"""
            ).fetchall()
            dimensions = self._connection.execute(
                """SELECT DISTINCT model_id, context_window_size, tensor_parallel_size
                   FROM benchmark_series_points"""
            ).fetchall()
        by_model: dict[str, dict[str, set[int]]] = {}
        for row in dimensions:
            values = by_model.setdefault(row["model_id"], {"windows": set(), "tp_sizes": set()})
            values["windows"].add(int(row["context_window_size"]))
            values["tp_sizes"].add(int(row["tensor_parallel_size"]))
        return [{
            "model_id": row["model_id"],
            "run_count": int(row["run_count"]),
            "best_prompt_tokens_per_second": float(row["best_prompt_tps"]),
            "best_generation_tokens_per_second": float(row["best_generation_tps"]),
            "context_windows": sorted(by_model[row["model_id"]]["windows"]),
            "tensor_parallel_sizes": sorted(by_model[row["model_id"]]["tp_sizes"]),
            "latest_at": row["latest_at"],
        } for row in rows]

    def benchmark_model_detail(self, model_id: str) -> dict[str, Any] | None:
        with self._lock:
            rows = self._connection.execute(
                """SELECT context_window_size, concurrency, tensor_parallel_size,
                          AVG(prompt_tps) AS prompt_tps,
                          AVG(generation_tps) AS generation_tps,
                          COUNT(*) AS sample_count
                   FROM benchmark_series_points WHERE model_id = ?
                   GROUP BY context_window_size, concurrency, tensor_parallel_size
                   ORDER BY tensor_parallel_size, context_window_size, concurrency""",
                (model_id,),
            ).fetchall()
        if not rows:
            return None
        return {
            "model_id": model_id,
            "points": [{
                "context_window_size": int(row["context_window_size"]),
                "concurrency": int(row["concurrency"]),
                "tensor_parallel_size": int(row["tensor_parallel_size"]),
                "prompt_tokens_per_second": float(row["prompt_tps"]),
                "generation_tokens_per_second": float(row["generation_tps"]),
                "sample_count": int(row["sample_count"]),
            } for row in rows],
        }

    def community_aggregates(self) -> list[dict[str, Any]]:
        """Aggregate only the fields that are eligible for community sharing.

        This provides useful evidence for fully local installations while
        keeping the public aggregate boundary identical to the upload payload.
        Rows are grouped by the exact dimensions declared in
        ``COMMUNITY_EVIDENCE_POLICY`` and no private benchmark metadata leaves
        this method. Explicit non-C1 runs are excluded because concurrency
        dimensions cannot be represented by this community contract. Manual
        benchmark detail remains available from ``benchmark_series_points``.
        """
        grouped: dict[tuple[str, str, int], dict[str, list[float]]] = {}
        with self._lock:
            cursor = self._connection.execute(
                "SELECT model_json, configuration_json, input_tokens, "
                "generation_tps, telemetry_cluster_id "
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
                    raw_concurrency = configuration.get("benchmark_concurrency")
                    if (
                        raw_concurrency is not None
                        and _positive_integer(raw_concurrency) != 1
                    ):
                        continue
                    model_id = str(model.get("repository") or "").strip()
                    quantization = _community_quantization(model)
                    prompt_bucket = _community_prompt_bucket(
                        row["input_tokens"]
                    )
                    speed = _positive_speed(row["generation_tps"])
                    cluster_id = row["telemetry_cluster_id"]
                    if (
                        not model_id or prompt_bucket is None or speed is None
                        or not _valid_telemetry_cluster_id(cluster_id)
                    ):
                        continue
                    key = (model_id, quantization, prompt_bucket)
                    grouped.setdefault(key, {}).setdefault(
                        str(cluster_id), []
                    ).append(speed)

        items = []
        for (model_id, quantization, prompt_bucket), contributors in grouped.items():
            contributor_means = [
                _outlier_filtered_mean(values)
                for values in contributors.values()
            ]
            items.append({
                "model_id": model_id,
                "quantization": quantization,
                "prompt_tokens_bucket": prompt_bucket,
                "inference_tokens_per_second": statistics.fmean(contributor_means),
                # The evidence threshold counts equal-weight contributors, not
                # raw inference requests from a potentially busy installation.
                "sample_count": len(contributor_means),
                "unique_cluster_count": len(contributor_means),
            })
        return sorted(
            items,
            key=lambda item: (
                -item["sample_count"], item["model_id"].casefold(),
                item["quantization"].casefold(), item["prompt_tokens_bucket"],
            ),
        )

    def delete_benchmark(self, sample_id: str) -> bool:
        with self._lock, self._connection:
            cursor = self._connection.execute("DELETE FROM benchmark_samples WHERE id = ?", (sample_id,))
        return bool(cursor.rowcount)

    def delete_benchmark_model(self, model_id: str) -> int:
        """Delete every local sample for one exact displayed model identity."""
        with self._lock, self._connection:
            rows = self._connection.execute(
                "SELECT id, model_json FROM benchmark_samples"
            ).fetchall()
            sample_ids = []
            for row in rows:
                try:
                    repository = str(
                        (json.loads(row["model_json"]) or {}).get("repository") or ""
                    ).strip()
                except (TypeError, ValueError, json.JSONDecodeError):
                    continue
                if repository == model_id:
                    sample_ids.append(row["id"])
            if not sample_ids:
                return 0
            self._connection.executemany(
                "DELETE FROM benchmark_samples WHERE id = ?",
                ((sample_id,) for sample_id in sample_ids),
            )
            return len(sample_ids)

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
        contribution_averages = self._community_contribution_averages()
        wanted = [row["sample_id"] for row in rows]
        marks = ",".join("?" for _ in wanted)
        with self._lock:
            sample_rows = self._connection.execute(
                f"SELECT * FROM benchmark_samples WHERE id IN ({marks})",
                tuple(wanted),
            ).fetchall()
        by_id = {}
        for row in sample_rows:
            payload = _upload_row(row)
            if payload is None:
                continue
            key = (
                payload["model_id"], payload["quantization"],
                payload["prompt_tokens_bucket"],
            )
            payload["inference_tokens_per_second"] = contribution_averages[key]
            by_id[row["id"]] = payload
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
        key = (
            payload["model_id"], payload["quantization"],
            payload["prompt_tokens_bucket"],
        )
        payload["inference_tokens_per_second"] = (
            self._community_contribution_averages()[key]
        )
        return {"sample_id": sample_id, "payload": payload}

    def _community_contribution_averages(
        self,
    ) -> dict[tuple[str, str, int], float]:
        """Build this user's robust C1 mean for each upload dimension."""
        grouped: dict[tuple[str, str, int], list[float]] = {}
        with self._lock:
            cursor = self._connection.execute(
                "SELECT * FROM benchmark_samples WHERE eligible = 1"
            )
            while True:
                rows = cursor.fetchmany(_COMMUNITY_AGGREGATE_BATCH_SIZE)
                if not rows:
                    break
                for row in rows:
                    payload = _upload_row(row)
                    if payload is None:
                        continue
                    key = (
                        payload["model_id"], payload["quantization"],
                        payload["prompt_tokens_bucket"],
                    )
                    grouped.setdefault(key, []).append(
                        payload["inference_tokens_per_second"]
                    )
        return {
            key: _outlier_filtered_mean(values)
            for key, values in grouped.items()
        }

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
            # Account claims and upload credentials are private service state;
            # only the status and a re-authentication flag leave the store.
            "pairing": {
                "status": "paired" if pairing.get("status") == "paired" else "not_paired",
                "token_invalid": bool(pairing.get("token_invalid")),
            },
            "outbox": {
                "pending": counts.get("pending", 0),
                "waiting_for_account": counts.get("waiting_for_account", 0),
                "failed": counts.get("failed", 0),
                "synced": counts.get("synced", 0),
            },
            "cloud_endpoint_configured": bool(COMMUNITY_API_URL),
        }

    def community_consent_snapshot(self) -> dict[str, Any]:
        """Return the epoch-bound state required to begin telemetry capture."""
        with self._lock:
            return self._community_consent_snapshot_locked()

    def _community_consent_snapshot_locked(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.get_setting("community_consent", False)),
            "generation": int(self.get_setting(
                "community_consent_generation", 0,
            )),
            "telemetry_cluster_id": self.get_setting("telemetry_cluster_id"),
        }

    def set_community_consent(
        self, enabled: bool, telemetry_cluster_id: str | None = None,
    ) -> dict[str, Any]:
        """Persist consent without retroactively queueing local history.

        A cluster identifier is generated only on the first opt-in. Controllers
        pass that same opaque UUID to agents so one SparkDeck cluster contributes
        one identity to the hosted aggregate, regardless of node count.
        """
        enabled = bool(enabled)
        with self._lock, self._connection:
            current = self._community_consent_snapshot_locked()
            cluster_id = current["telemetry_cluster_id"]
            if enabled:
                if telemetry_cluster_id is not None:
                    cluster_id = _telemetry_cluster_id(telemetry_cluster_id)
                elif cluster_id is None:
                    cluster_id = str(uuid.uuid4())
                else:
                    cluster_id = _telemetry_cluster_id(cluster_id)

            identity_changed = (
                enabled and cluster_id != current["telemetry_cluster_id"]
            )
            state_changed = enabled != current["enabled"]
            generation = current["generation"] + int(
                state_changed or identity_changed
            )
            self._connection.execute(
                "INSERT INTO settings(key, value_json) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json",
                ("community_consent", json.dumps(enabled)),
            )
            self._connection.execute(
                "INSERT INTO settings(key, value_json) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json",
                ("community_consent_generation", json.dumps(generation)),
            )
            self._connection.execute(
                "INSERT INTO settings(key, value_json) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json",
                (
                    "community_consent_contract_version",
                    json.dumps(COMMUNITY_CONSENT_CONTRACT_VERSION),
                ),
            )
            if enabled and cluster_id is not None:
                self._connection.execute(
                    "INSERT INTO settings(key, value_json) VALUES (?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json",
                    ("telemetry_cluster_id", json.dumps(cluster_id)),
                )
            if not enabled or identity_changed:
                # Withdrawing consent or changing cluster identity invalidates
                # every unsent instruction. Synced rows remain local history;
                # enabling never resurrects historical telemetry.
                self._connection.execute(
                    "DELETE FROM upload_outbox WHERE status IN "
                    "('pending', 'failed', 'waiting_for_account')"
                )
            return {
                "enabled": enabled,
                "generation": generation,
                "telemetry_cluster_id": cluster_id,
            }

    def revoke_community_membership(self) -> dict[str, Any]:
        """Atomically sever worker consent and its former cluster identity."""
        with self._lock, self._connection:
            current = self._community_consent_snapshot_locked()
            generation = current["generation"] + 1
            for key, value in (
                ("community_consent", False),
                ("community_consent_generation", generation),
            ):
                self._connection.execute(
                    "INSERT INTO settings(key, value_json) VALUES (?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json",
                    (key, json.dumps(value)),
                )
            self._connection.execute(
                "DELETE FROM settings WHERE key = 'telemetry_cluster_id'"
            )
            self._connection.execute(
                "DELETE FROM upload_outbox WHERE status IN "
                "('pending', 'failed', 'waiting_for_account')"
            )
            return {
                "enabled": False,
                "generation": generation,
                "telemetry_cluster_id": None,
            }


def _benchmark_matches_series_point(
    sample_row: sqlite3.Row | dict[str, Any],
    point_row: sqlite3.Row,
) -> bool:
    """Return whether a local sample is the series row created for the same run."""
    try:
        model = json.loads(sample_row["model_json"] or "{}")
        configuration = json.loads(sample_row["configuration_json"] or "{}")
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    if not isinstance(model, dict) or not isinstance(configuration, dict):
        return False
    return (
        str(model.get("repository") or "").strip() == point_row["model_id"]
        and community_context_window(configuration) == point_row["context_window_size"]
        and _positive_integer(configuration.get("benchmark_concurrency"))
        == point_row["concurrency"]
        and _positive_integer(configuration.get("tensor_parallel_size"))
        == point_row["tensor_parallel_size"]
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
    """Build the strict cloud payload allowlist, separate from local history."""
    value = _benchmark_row(row)
    prompt_bucket = _community_prompt_bucket(value["input_tokens"])
    speed = _positive_speed(value["generation_tokens_per_second"])
    model = value.get("model", {})
    model_id = str(model.get("repository") or "").strip()
    quantization = _community_quantization(model)
    cluster_id = row["telemetry_cluster_id"]
    raw_concurrency = value.get("configuration", {}).get(
        "benchmark_concurrency"
    )
    if (
        not model_id or prompt_bucket is None or speed is None
        or not _valid_telemetry_cluster_id(cluster_id)
        or (
            raw_concurrency is not None
            and _positive_integer(raw_concurrency) != 1
        )
    ):
        return None
    payload = {
        "model_id": model_id,
        "quantization": quantization,
        "prompt_tokens_bucket": prompt_bucket,
        "inference_tokens_per_second": speed,
        "telemetry_cluster_id": cluster_id,
        "concurrency": 1,
    }
    if not set(payload) <= COMMUNITY_UPLOAD_FIELDS:
        raise ValueError("community upload payload exceeds the public field contract")
    return payload


def _community_sample_eligible(sample: BenchmarkSample) -> bool:
    raw_concurrency = sample.configuration.get("benchmark_concurrency")
    return bool(
        sample.eligible_for_community
        and str(sample.model.repository or "").strip()
        and _community_prompt_tokens(sample.input_tokens) is not None
        and _positive_speed(sample.generation_tokens_per_second) is not None
        and (
            raw_concurrency is None
            or _positive_integer(raw_concurrency) == 1
        )
    )


def _community_quantization(model: dict[str, Any]) -> str:
    """Return one explicit aggregate dimension without guessing locally."""
    return canonical_quantization(model.get("quantization")) or "UNKNOWN"


def _community_prompt_tokens(value: Any) -> int | None:
    """Validate near-empty prompt occupancy used by passive C1 samples."""
    parsed = _positive_integer(value)
    return parsed if parsed is not None and parsed < 10_000 else None


def _community_prompt_bucket(value: Any) -> int | None:
    """Normalize prompt occupancy for useful local aggregate cohorts."""
    tokens = _community_prompt_tokens(value)
    if tokens is None:
        return None
    if tokens <= 800:
        return 400
    return min(9_000, max(1_000, ((tokens + 500) // 1_000) * 1_000))


def _telemetry_cluster_id(value: Any) -> str:
    try:
        return str(uuid.UUID(str(value)))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("telemetry_cluster_id must be a UUID") from exc


def _valid_telemetry_cluster_id(value: Any) -> bool:
    try:
        _telemetry_cluster_id(value)
    except ValueError:
        return False
    return True


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


def _coordinated_concurrency(value: Any) -> int | None:
    parsed = _positive_integer(value)
    return parsed if parsed in {1, 2, 5, 10} else None


def _positive_speed(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 and math.isfinite(parsed) else None


def _positive_integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if 0 < parsed <= 100_000_000 else None
