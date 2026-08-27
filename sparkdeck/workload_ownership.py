"""Durable per-node ownership claims for managed Docker containers."""

from __future__ import annotations

import copy
import json
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from sparkdeck.private_json import atomic_private_json_write


LEDGER_FILENAME = "managed-containers.json"
LEDGER_VERSION = 1


class ManagedWorkloadLedger:
    """Keep container ownership durable across Docker and process failures."""

    def __init__(self, data_dir: Path):
        self.path = Path(data_dir) / LEDGER_FILENAME
        self._lock = threading.RLock()

    @contextmanager
    def locked(self) -> Iterator[None]:
        with self._lock:
            yield

    def _load(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            containers = value.get("containers") if isinstance(value, dict) else None
            if (
                not isinstance(value, dict)
                or value.get("version") != LEDGER_VERSION
                or not isinstance(containers, dict)
            ):
                raise ValueError("unsupported ledger shape")
            result: dict[str, dict[str, Any]] = {}
            for raw_name, raw_claim in containers.items():
                name = str(raw_name or "").strip()
                if not name or not isinstance(raw_claim, dict):
                    raise ValueError("invalid container claim")
                state = raw_claim.get("state")
                if state not in {"pending", "active"}:
                    raise ValueError("invalid container claim state")
                result[name] = {
                    "state": state,
                    "deployment_id": str(raw_claim.get("deployment_id") or "") or None,
                    "claimed_at": float(raw_claim.get("claimed_at") or 0.0),
                }
            return result
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"cannot verify managed workload ownership; repair {LEDGER_FILENAME} and retry"
            ) from exc

    def _save(self, claims: dict[str, dict[str, Any]]) -> None:
        atomic_private_json_write(self.path, {
            "version": LEDGER_VERSION,
            "containers": claims,
        })

    def snapshot(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return copy.deepcopy(self._load())

    def claim(self, name: str, deployment_id: str | None = None) -> None:
        name = str(name or "").strip()
        if not name:
            raise ValueError("managed container name is required")
        with self._lock:
            claims = self._load()
            claims[name] = {
                "state": "pending",
                "deployment_id": str(deployment_id or "") or None,
                "claimed_at": time.time(),
            }
            self._save(claims)

    def confirm(self, name: str, deployment_id: str | None = None) -> None:
        name = str(name or "").strip()
        with self._lock:
            claims = self._load()
            existing = claims.get(name, {})
            claims[name] = {
                "state": "active",
                "deployment_id": (
                    str(deployment_id or existing.get("deployment_id") or "") or None
                ),
                "claimed_at": float(existing.get("claimed_at") or time.time()),
            }
            self._save(claims)

    def release(self, name: str) -> None:
        name = str(name or "").strip()
        with self._lock:
            claims = self._load()
            if claims.pop(name, None) is not None:
                self._save(claims)

    def reconcile(self, observed: dict[str, str | None]) -> None:
        """Replace claims after a successful, lock-serialized full inventory."""
        normalized = {
            str(name): str(deployment_id or "") or None
            for name, deployment_id in observed.items() if str(name or "").strip()
        }
        with self._lock:
            current = self._load()
            updated = {}
            now = time.time()
            for name, deployment_id in normalized.items():
                previous = current.get(name, {})
                updated[name] = {
                    "state": "active",
                    "deployment_id": deployment_id or previous.get("deployment_id"),
                    "claimed_at": float(previous.get("claimed_at") or now),
                }
            if updated != current:
                self._save(updated)
