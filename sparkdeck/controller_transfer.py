"""Orchestrated controller-role transfer to a joined worker node.

The transfer is driven by the outgoing controller and proceeds in one
direction: preflight, bundle, stage on the target, promote the target
(restart), re-bind every worker to the promoted controller, then demote the
outgoing controller by rejoining it as a worker. Before the promote commit
the flow is fully reversible; after it, only the outgoing controller can
still authenticate to un-rebound workers, so it stays alive until every
worker moved or the operator resumes the flow manually.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import ipaddress
import json
import os
import secrets
import shutil
import sqlite3
import sys
import tempfile
import time
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Any

import httpx

from cluster import (
    AGENT_PROTOCOL_VERSION,
    CONTROLLER_TRANSFER_CAPABILITY,
    LOCAL_NODE_ID,
)
from sparkdeck.private_json import atomic_private_json_write

TRANSFER_STATE_FILENAME = "controller-transfer.json"
TRANSFER_STAGING_DIRNAME = "controller-transfer-incoming"
TRANSFER_CONFIRMATION = "transfer-controller"
TRANSFER_BUNDLE_VERSION = 1
MAX_STORE_BYTES = 256 * 1024 * 1024
NEW_CONTROLLER_BOOT_TIMEOUT_SECONDS = 300.0
REBIND_ATTEMPTS = 3
COMPLETION_POLL_SECONDS = 5.0

# Settings that describe this machine rather than the cluster stay with the
# node; every other controller setting transfers with the role.
NODE_LOCAL_SETTING_KEYS = {
    "hf_cache",
    "cluster_node_name",
    "cluster_node_hidden_from_dashboard",
    "cluster_fabric_ip",
    "cluster_fabric_interface",
}

# Controller-owned JSON state that moves with the role. Every entry is
# remapped for node references before it is bundled.
TRANSFERRED_JSON_FILENAMES = (
    "deployments.json",  # legacy runtime records
    "recipes.json",
    "unsloth_models.json",
    "spark_runs.json",
    "model_routing_policies.json",
    "usage_aliases.json",
    "usage_merge_groups.json",
    "usage_routing_rules.json",
    "usage_cache_estimates.json",
    "source_ip_routing_rules.json",
    "speed_samples.json",
    "token_stats.json",
    "hourly_token_stats.json",
    "token_usage_sync.json",
)

# JSON keys whose values are cluster node ids. ``node_ids`` covers saved
# deployment settings, replicas, and recipes; the scalars cover deployment
# members and the RouterOS gateway pin. ``selected_nodes`` entries carry the
# same sentinel under a generic ``id`` key.
_NODE_REF_LIST_KEYS = {"node_ids"}
_NODE_REF_SCALAR_KEYS = {"node_id", "routeros_gateway_node_id"}

_TAILSCALE_V4 = ipaddress.ip_network("100.64.0.0/10")


def remap_node_refs(value: Any, replacement: str) -> Any:
    """Deep-copy ``value`` with ``local`` node references moved to a node id."""
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if key in _NODE_REF_LIST_KEYS and isinstance(item, list):
                result[key] = [
                    replacement if entry == LOCAL_NODE_ID else entry
                    for entry in item
                ]
            elif key in _NODE_REF_SCALAR_KEYS and item == LOCAL_NODE_ID:
                result[key] = replacement
            elif key == "selected_nodes" and isinstance(item, list):
                result[key] = [
                    {**entry, "id": replacement}
                    if isinstance(entry, dict) and entry.get("id") == LOCAL_NODE_ID
                    else entry
                    for entry in item
                ]
            else:
                result[key] = remap_node_refs(item, replacement)
        return result
    if isinstance(value, list):
        return [remap_node_refs(entry, replacement) for entry in value]
    return value


def _read_store_deployment_rows(data_dir: Path) -> list[dict]:
    """Read deployment rows read-only for preflight warnings."""
    path = data_dir / "sparkdeck.sqlite3"
    if not path.exists():
        return []
    try:
        uri = f"file:{path.resolve().as_posix()}?mode=ro"
        connection = sqlite3.connect(uri, uri=True, timeout=2.0)
        try:
            exists = connection.execute(
                "SELECT 1 FROM sqlite_master"
                " WHERE type = 'table' AND name = 'deployments'"
            ).fetchone()
            if not exists:
                return []
            rows = []
            for runtime, settings_json, credential_ref in connection.execute(
                "SELECT runtime, settings_json, credential_ref FROM deployments"
            ):
                try:
                    settings = json.loads(settings_json or "")
                except ValueError:
                    settings = {}
                rows.append({
                    "runtime": str(runtime or ""),
                    "settings": settings if isinstance(settings, dict) else {},
                    "credential_ref": credential_ref,
                })
            return rows
        finally:
            connection.close()
    except sqlite3.Error:
        return []


def _read_store_setting(data_dir: Path, key: str) -> Any:
    path = data_dir / "sparkdeck.sqlite3"
    if not path.exists():
        return None
    try:
        uri = f"file:{path.resolve().as_posix()}?mode=ro"
        connection = sqlite3.connect(uri, uri=True, timeout=2.0)
        try:
            exists = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'settings'"
            ).fetchone()
            if not exists:
                return None
            row = connection.execute(
                "SELECT value_json FROM settings WHERE key = ?", (key,)
            ).fetchone()
            return json.loads(row[0]) if row else None
        finally:
            connection.close()
    except (sqlite3.Error, ValueError):
        return None


def snapshot_store(store_path: Path, destination: Path, replacement: str) -> None:
    """Write a standalone, node-remapped copy of the deployment store."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    # VACUUM INTO folds the live WAL into one consistent standalone file
    # without interrupting writers on the service's own connection.
    connection = sqlite3.connect(str(store_path), timeout=30.0)
    try:
        connection.execute("VACUUM INTO ?", (str(destination),))
    finally:
        connection.close()
    output = sqlite3.connect(str(destination))
    try:
        integrity = output.execute("PRAGMA integrity_check").fetchone()
        if not integrity or integrity[0] != "ok":
            raise ValueError("deployment store failed its integrity check")
        for deployment_id, settings_json in output.execute(
            "SELECT id, settings_json FROM deployments"
        ).fetchall():
            if not settings_json:
                continue
            try:
                settings = json.loads(settings_json)
            except ValueError:
                continue
            remapped = remap_node_refs(settings, replacement)
            if remapped != settings:
                output.execute(
                    "UPDATE deployments SET settings_json = ? WHERE id = ?",
                    (json.dumps(remapped), deployment_id),
                )
        output.commit()
    finally:
        output.close()
    try:
        destination.chmod(0o600)
    except OSError:
        pass


class TransferService:
    """Owns one durable controller transfer, on either side of it."""

    def __init__(self, manager: Any, onboarding: Any, root: Path, data_dir: Path):
        self.manager = manager
        self.onboarding = onboarding
        self.root = Path(root).resolve()
        self.data_dir = Path(data_dir)
        self.state_path = self.data_dir / TRANSFER_STATE_FILENAME
        self._task: asyncio.Task | None = None
        self._completion_task: asyncio.Task | None = None
        self._closing = False

    # ---------- durable state ----------

    def _read(self) -> dict:
        try:
            value = json.loads(self.state_path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError):
            return {}

    def _write(self, state: dict) -> None:
        state["updated_at"] = time.time()
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
        os.replace(temporary, self.state_path)
        try:
            self.state_path.chmod(0o600)
        except OSError:
            pass

    def _staging_dir(self) -> Path:
        return self.data_dir / TRANSFER_STAGING_DIRNAME

    def _staged_job(self) -> dict | None:
        try:
            manifest = json.loads(
                (self._staging_dir() / "manifest.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            return None
        if not isinstance(manifest, dict) or not manifest.get("job_id"):
            return None
        return manifest

    # ---------- controller-side public surface ----------

    def status(self) -> dict:
        state = self._read()
        if not state:
            return {
                "active": False,
                "phase": "idle",
                "capability": CONTROLLER_TRANSFER_CAPABILITY,
                "confirmation": TRANSFER_CONFIRMATION,
            }
        state.pop("old_controller_forward_token", None)
        state.setdefault("active", False)
        state["capability"] = CONTROLLER_TRANSFER_CAPABILITY
        state["confirmation"] = TRANSFER_CONFIRMATION
        return state

    def _cluster_update_active(self) -> bool:
        try:
            value = json.loads(
                (self.data_dir / "system-update.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            return False
        return isinstance(value, dict) and bool(value.get("active"))

    async def preflight(self, target_node_id: str) -> dict:
        """Collect every gate that must hold before a transfer may start."""
        blockers: list[str] = []
        warnings: list[str] = []
        target: dict | None = None
        if self.manager.is_joined_worker():
            blockers.append("only the controller can transfer the controller role")
        elif self._read().get("active"):
            blockers.append("a controller transfer is already running")
        elif self._cluster_update_active():
            blockers.append("a cluster system update is running")
        else:
            target = self.manager.node_registry.get(str(target_node_id))
            if not target or str(target_node_id) == LOCAL_NODE_ID:
                blockers.append("target is not a joined worker node")
                target = None
            elif not target.get("enabled", True):
                blockers.append(f"{target.get('name', target_node_id)} is disabled")
            if target is not None:
                for node in list(self.manager.node_registry.nodes):
                    name = str(node.get("name") or node.get("id"))
                    try:
                        probe = await self.manager.node_registry.probe(
                            node, details=False
                        )
                        if not probe.get("online"):
                            blockers.append(f"{name} is offline")
                    except Exception as exc:
                        blockers.append(f"{name} is unreachable: {exc}")
                    try:
                        info = await self.manager.node_registry.request(
                            node["id"], "GET", "/api/agent/info",
                            timeout=10, allow_disabled=True,
                        )
                        if CONTROLLER_TRANSFER_CAPABILITY not in (
                            info.get("capabilities") or []
                        ):
                            blockers.append(
                                f"{name} needs the latest SparkDeck build"
                                " before a controller transfer"
                            )
                    except Exception as exc:
                        blockers.append(
                            f"{name} did not report transfer support: {exc}"
                        )
                if not self._controller_agent_urls():
                    blockers.append(
                        "this controller has no cluster-reachable (Tailscale)"
                        " URL to rejoin the new controller with"
                    )
        if target is not None and not self.manager.is_joined_worker():
            try:
                remote = await self.manager.node_registry.request(
                    str(target_node_id),
                    "POST",
                    "/api/agent/controller-transfer/preflight",
                    json_body={},
                    timeout=30,
                )
                blockers.extend(
                    f"{target.get('name', target_node_id)}: {item}"
                    for item in remote.get("blockers") or []
                )
            except Exception as exc:
                blockers.append(f"target preflight failed: {exc}")
        rows = _read_store_deployment_rows(self.data_dir)
        keyring = [
            row for row in rows
            if str(row["credential_ref"] or "").startswith("keyring:")
        ]
        if keyring:
            warnings.append(
                f"{len(keyring)} saved deployment(s) use API keys stored in this"
                " machine's credential manager; re-enter them on the new"
                " controller after the transfer"
            )
        local_llama = [
            row for row in rows
            if row["runtime"] == "llama.cpp"
            and (
                not row["settings"].get("node_ids")
                or LOCAL_NODE_ID in (row["settings"].get("node_ids") or [])
            )
        ]
        if local_llama:
            warnings.append(
                f"{len(local_llama)} llama.cpp deployment(s) run on the"
                " controller; llama.cpp must stay on a controller, so move or"
                " stop them after the transfer"
            )
        if _read_store_setting(self.data_dir, "community_consent"):
            warnings.append(
                "the community account sign-in may need to be repeated on the"
                " new controller"
            )
        estimated = 64 * 1024
        for name in TRANSFERRED_JSON_FILENAMES:
            try:
                estimated += (self.data_dir / name).stat().st_size
            except OSError:
                pass
        try:
            estimated += (self.data_dir / "sparkdeck.sqlite3").stat().st_size
        except OSError:
            pass
        return {
            "ok": not blockers,
            "blockers": blockers,
            "warnings": warnings,
            "target_node_id": str(target_node_id),
            "target_name": (target or {}).get("name"),
            "estimated_bytes": estimated,
        }

    async def start(self, target_node_id: str, confirmation: str) -> dict:
        if self._closing:
            raise RuntimeError("Service is shutting down")
        if confirmation != TRANSFER_CONFIRMATION:
            raise ValueError("Explicit controller transfer confirmation is required")
        if self.manager.is_joined_worker():
            raise RuntimeError("only the controller can transfer the controller role")
        preflight = await self.preflight(target_node_id)
        if not preflight["ok"]:
            raise RuntimeError("; ".join(preflight["blockers"]))
        target = self.manager.node_registry.get(str(target_node_id))
        if not target:
            raise RuntimeError("target is not a joined worker node")
        target_url = str(target.get("agent_url") or "").rstrip("/")
        own_urls = {url.rstrip("/") for url in self._controller_agent_urls()}
        if not target_url or target_url in own_urls:
            raise RuntimeError("target does not advertise a distinct cluster URL")
        state = {
            "id": uuid.uuid4().hex,
            "active": True,
            "committed": False,
            "incoming": False,
            "phase": "preflight",
            "message": "Preparing the controller transfer",
            "error": None,
            "target_node_id": str(target_node_id),
            "target_name": target.get("name"),
            "target_url": target_url,
            "old_controller_node_id": self.manager.agent_credentials.node_id,
            "old_controller_name": self.manager.settings.get("cluster_node_name"),
            "controller_url": target_url,
            "nodes": [
                {
                    "id": node.get("id"),
                    "name": node.get("name"),
                    "status": "pending",
                }
                for node in self.manager.node_registry.nodes
                if node.get("id") != str(target_node_id)
            ],
            "warnings": list(preflight["warnings"]),
            "started_at": time.time(),
            "completed_at": None,
        }
        self._write(state)
        self._task = asyncio.create_task(self._run_transfer(state))
        return self.status()

    async def cancel(self) -> dict:
        state = self._read()
        if not state.get("active"):
            raise RuntimeError("no active controller transfer")
        if state.get("committed"):
            raise RuntimeError(
                "the target is already the controller; the transfer can only"
                " be resumed, not cancelled"
            )
        if self._task is not None and not self._task.done():
            self._task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await self._task
        if not self.manager.is_joined_worker():
            try:
                await self.manager.node_registry.request(
                    state["target_node_id"],
                    "POST",
                    "/api/agent/controller-transfer/discard",
                    json_body={},
                    timeout=15,
                )
            except Exception:
                pass
        state.update(
            active=False,
            phase="cancelled",
            message="Transfer cancelled before the target was promoted",
        )
        self._write(state)
        return self.status()

    async def resume(self) -> dict:
        state = self._read()
        if not state.get("active") or not state.get("committed"):
            raise RuntimeError("no committed controller transfer to resume")
        if self._task is not None and not self._task.done():
            raise RuntimeError("the transfer is already running")
        if self.manager.is_joined_worker():
            raise RuntimeError("only the outgoing controller can resume the transfer")
        self._task = asyncio.create_task(self._resume_committed(state))
        return self.status()

    # ---------- outgoing-controller orchestration ----------

    async def _run_transfer(self, state: dict) -> None:
        try:
            state.update(
                phase="bundling",
                message="Capturing controller state",
                error=None,
            )
            self._write(state)
            bundle = await asyncio.to_thread(self._build_bundle, state)
            # The raw forward token only ever lives in private state files:
            # here for the demote step, hashed inside the transferred registry.
            state["old_controller_forward_token"] = bundle["old_controller"][
                "forward_token"
            ]
            self._write(state)

            state.update(
                phase="staging",
                message=f"Uploading controller state to {state.get('target_name')}",
            )
            self._write(state)
            await self.manager.node_registry.request(
                state["target_node_id"],
                "POST",
                "/api/agent/controller-transfer/receive",
                json_body=bundle,
                timeout=600,
            )

            state.update(
                phase="promoting",
                message="Promoting the target node to controller",
            )
            self._write(state)
            await self.manager.node_registry.request(
                state["target_node_id"],
                "POST",
                "/api/agent/controller-transfer/promote",
                json_body={"job_id": state["id"]},
                timeout=60,
            )
            await self._wait_for_new_controller(state)

            # Commit point: the promoted controller is live. Quiesce this
            # process's cluster actors so only the new controller acts.
            state.update(
                committed=True,
                phase="rebinding",
                message="Re-pointing workers to the new controller",
            )
            self._write(state)
            adopt_worker = getattr(self.manager, "adopt_worker_role", None)
            if adopt_worker is not None:
                await adopt_worker()
            await self._rebind_nodes(state)
            if any(
                node.get("status") == "failed" for node in state.get("nodes", [])
            ):
                state.update(
                    message=(
                        "Some workers could not be re-bound; bring them back"
                        " online and resume the transfer"
                    )
                )
                self._write(state)
                return
            await self._demote_self(state)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            state.update(
                phase="failed",
                active=not state.get("committed"),
                error=str(exc)[:500],
                message=(
                    "Transfer failed"
                    + (
                        " after the target was promoted; resolve the issue and"
                        " resume it from the Cluster page"
                        if state.get("committed")
                        else ""
                    )
                ),
            )
            self._write(state)

    def _controller_agent_urls(self) -> list[str]:
        port = getattr(self.onboarding, "port", 7878)
        urls = []
        for interface in self.manager._network_interfaces():
            for address in interface.get("ipv4") or []:
                try:
                    if ipaddress.ip_address(str(address)) in _TAILSCALE_V4:
                        urls.append(f"http://{address}:{port}")
                except ValueError:
                    continue
        return list(dict.fromkeys(urls))

    def _build_bundle(self, state: dict) -> dict:
        registry = self.manager.node_registry
        target = registry.get(state["target_node_id"])
        if not target:
            raise RuntimeError("target node is no longer registered")
        old_node_id = self.manager.agent_credentials.node_id
        old_urls = self._controller_agent_urls()
        if not old_urls:
            raise RuntimeError("controller has no cluster-reachable URL")
        old_forward_token = secrets.token_urlsafe(32)

        entries = [
            {
                key: value
                for key, value in node.items()
                if key not in {"forward_token"}
            }
            for node in registry.nodes
            if node.get("id") != state["target_node_id"]
        ]
        # The outgoing controller rejoins as a worker, so the transferred
        # registry must already contain its future entry, including the
        # bearer the promoted controller will use to reach it.
        entries.append({
            "id": old_node_id,
            "name": str(
                self.manager.settings.get("cluster_node_name") or "Controller"
            ),
            "agent_url": old_urls[0],
            "agent_token": self.manager.agent_credentials.data.get("agent_token"),
            "fabric_ip": self.manager.settings.get("cluster_fabric_ip") or None,
            "fabric_interface": (
                self.manager.settings.get("cluster_fabric_interface") or None
            ),
            "enabled": True,
            "hidden_from_dashboard": bool(
                self.manager.settings.get("cluster_node_hidden_from_dashboard")
            ),
            "protocol_version": AGENT_PROTOCOL_VERSION,
            "paired_at": time.time(),
            "usage_reconciled": False,
            "forward_token_hash": hashlib.sha256(
                old_forward_token.encode("utf-8")
            ).hexdigest(),
        })

        with tempfile.TemporaryDirectory() as temporary:
            store_copy = Path(temporary) / "store.sqlite3"
            snapshot_store(
                self.data_dir / "sparkdeck.sqlite3", store_copy, old_node_id
            )
            store_bytes = store_copy.read_bytes()

        json_files: dict[str, Any] = {}
        for name in TRANSFERRED_JSON_FILENAMES:
            path = self.data_dir / name
            if not path.exists():
                continue
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            json_files[name] = remap_node_refs(value, old_node_id)

        overlay = {
            key: value
            for key, value in self.manager.settings.items()
            if key not in NODE_LOCAL_SETTING_KEYS
        }

        transfer_state = {
            "id": state["id"],
            "active": True,
            "committed": True,
            "incoming": True,
            "phase": "promoted",
            "message": "Promoted to controller; restarting with the transferred state",
            "error": None,
            "target_node_id": state["target_node_id"],
            "target_name": state["target_name"],
            "target_url": state["target_url"],
            "old_controller_node_id": old_node_id,
            "old_controller_name": self.manager.settings.get("cluster_node_name"),
            "controller_url": state["target_url"],
            "nodes": [
                {"id": entry["id"], "name": entry.get("name"), "status": "pending"}
                for entry in entries
            ],
            "warnings": list(state.get("warnings") or []),
            "started_at": state.get("started_at") or time.time(),
            "completed_at": None,
        }
        return {
            "bundle_version": TRANSFER_BUNDLE_VERSION,
            "job_id": state["id"],
            "created_at": time.time(),
            "old_controller": {
                "node_id": old_node_id,
                "name": self.manager.settings.get("cluster_node_name"),
                "agent_url": old_urls[0],
                "agent_token": self.manager.agent_credentials.data.get("agent_token"),
                "forward_token": old_forward_token,
            },
            "target": {
                "node_id": state["target_node_id"],
                "name": state["target_name"],
            },
            "controller_url": state["target_url"],
            "new_controller_node_id": state["target_node_id"],
            "nodes": entries,
            "settings_overlay": overlay,
            "json_files": json_files,
            "store_sha256": hashlib.sha256(store_bytes).hexdigest(),
            "store_b64": base64.b64encode(store_bytes).decode("ascii"),
            "transfer_state": transfer_state,
        }

    async def _wait_for_new_controller(self, state: dict) -> None:
        url = str(state["target_url"]).rstrip("/")
        deadline = time.monotonic() + NEW_CONTROLLER_BOOT_TIMEOUT_SECONDS
        last = "controller did not answer"
        while time.monotonic() < deadline:
            try:
                response = await self.manager.http.get(
                    f"{url}/api/v1/onboarding", timeout=5
                )
                if response.status_code == 200:
                    identity = response.json()
                    if (
                        identity.get("role") == "controller"
                        and str((identity.get("node") or {}).get("id") or "")
                        == state["target_node_id"]
                    ):
                        return
                    last = "target did not report the promoted controller role"
            except httpx.HTTPError as exc:
                last = str(exc)
            except ValueError:
                last = "target returned an invalid identity"
            state["message"] = f"Waiting for the new controller to boot: {last}"
            self._write(state)
            await asyncio.sleep(3)
        raise RuntimeError(f"the new controller did not come up: {last}")

    async def _rebind_nodes(self, state: dict) -> None:
        for entry in state.get("nodes", []):
            if entry.get("status") in {"rebound", "skipped"}:
                continue
            node = self.manager.node_registry.get(entry.get("id"))
            if node is None:
                entry["status"] = "skipped"
                entry["message"] = "removed from the cluster"
                continue
            error: str | None = None
            for _ in range(REBIND_ATTEMPTS):
                try:
                    await self.manager.node_registry.request(
                        entry["id"],
                        "POST",
                        "/api/agent/onboarding/rebind",
                        json_body={
                            "controller_url": state["target_url"],
                            "controller_node_id": state["target_node_id"],
                        },
                        timeout=15,
                    )
                    entry["status"] = "rebound"
                    entry.pop("error", None)
                    entry.pop("message", None)
                    error = None
                    break
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    error = str(exc)[:300]
                    await asyncio.sleep(2)
            if error is not None:
                entry["status"] = "failed"
                entry["error"] = error
            self._write(state)

    async def _demote_self(self, state: dict) -> None:
        state.update(
            phase="demoting",
            message="Rejoining this machine as a worker under the new controller",
        )
        self._write(state)
        archive = self.data_dir / f"controller-transfer-archive-{int(time.time())}"
        archive.mkdir(parents=True, exist_ok=True)
        moved: list[str] = []
        try:
            for name in (
                "sparkdeck.sqlite3",
                "sparkdeck.sqlite3-wal",
                "sparkdeck.sqlite3-shm",
                "nodes.json",
                *TRANSFERRED_JSON_FILENAMES,
            ):
                source = self.data_dir / name
                if source.exists():
                    shutil.move(str(source), str(archive / name))
                    moved.append(name)
            self.state_path.replace(archive / TRANSFER_STATE_FILENAME)
            self.onboarding.assignment.save({
                "controller_url": state["target_url"],
                "forward_token": state["old_controller_forward_token"],
                "node_id": self.manager.agent_credentials.node_id,
                "controller_node_id": state["target_node_id"],
                "protocol_version": AGENT_PROTOCOL_VERSION,
                "joined_at": time.time(),
                "rebound_at": time.time(),
            })
            try:
                self.manager.agent_credentials.rebind_controller(
                    state["target_node_id"]
                )
            except OSError:
                # The new controller's first authorized call claims ownership
                # durably after the restart, so a failed pre-claim is safe.
                pass
        except Exception:
            # A half-archived controller must stay coherent: put everything
            # back and let the operator resume the transfer.
            for name in moved:
                with suppress(OSError):
                    shutil.move(str(archive / name), str(self.data_dir / name))
            with suppress(OSError):
                (archive / TRANSFER_STATE_FILENAME).replace(self.state_path)
            raise
        self._spawn_helper("demote")

    def _spawn_helper(self, action: str) -> int:
        from sparkdeck.updater import _spawn_update_helper

        command = [
            sys.executable,
            "-m",
            "sparkdeck.transfer_helper",
            "--root",
            str(self.root),
            "--state",
            str(self.data_dir / TRANSFER_STATE_FILENAME),
            "--action",
            action,
        ]
        return _spawn_update_helper(self.root, command)

    async def _resume_committed(self, state: dict) -> None:
        try:
            await self._wait_for_new_controller(state)
            adopt_worker = getattr(self.manager, "adopt_worker_role", None)
            if adopt_worker is not None:
                await adopt_worker()
            state["phase"] = "rebinding"
            self._write(state)
            await self._rebind_nodes(state)
            if any(
                node.get("status") == "failed" for node in state.get("nodes", [])
            ):
                state.update(
                    message=(
                        "Some workers could not be re-bound; bring them back"
                        " online and resume the transfer"
                    )
                )
                self._write(state)
                return
            await self._demote_self(state)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            state.update(
                phase="failed",
                error=str(exc)[:500],
                message=(
                    "Committed transfer needs attention: resolve the reported"
                    " issue and resume it from the Cluster page"
                ),
            )
            self._write(state)

    # ---------- incoming-controller completion ----------

    async def _completion_loop(self, state: dict) -> None:
        while True:
            await asyncio.sleep(COMPLETION_POLL_SECONDS)
            registry = list(self.manager.node_registry.nodes)
            entries = {entry["id"]: entry for entry in state.get("nodes", [])}
            for node in registry:
                entries.setdefault(
                    node["id"],
                    {"id": node["id"], "name": node.get("name"), "status": "pending"},
                )
            all_online = True
            for node in registry:
                entry = entries[node["id"]]
                try:
                    probe = await self.manager.node_registry.probe(
                        node, details=False
                    )
                    online = bool(probe.get("online"))
                    entry["name"] = probe.get("name") or entry.get("name")
                except Exception:
                    online = False
                entry["online"] = online
                entry["status"] = "online" if online else "pending"
                if not online:
                    all_online = False
            state["nodes"] = list(entries.values())
            if registry and all_online:
                state.update(
                    phase="completed",
                    active=False,
                    completed_at=time.time(),
                    message=(
                        "Controller role transferred; every registered node is"
                        " online under the new controller"
                    ),
                )
                self._write(state)
                return
            self._write(state)

    def defers_cluster_actions(self) -> bool:
        """Gate controller schedulers until the transferred cluster re-binds."""
        state = self._read()
        return bool(
            state.get("active")
            and state.get("incoming")
            and state.get("phase") in {"promoting", "promoted", "completing"}
        )

    # ---------- target-side (agent) surface ----------

    def agent_preflight(self) -> dict:
        """Target-side gates for receiving a controller state bundle."""
        blockers: list[str] = []
        if not self.onboarding.assignment.load():
            blockers.append("node is not a joined worker")
        saved = self.onboarding._saved_deployment_count()
        if saved:
            blockers.append(
                f"{saved} saved deployment record"
                f"{'s' if saved != 1 else ''} must be removed first"
            )
        legacy = len([
            deployment
            for deployment in getattr(self.manager, "deployments", [])
            if isinstance(deployment, dict)
        ])
        if legacy:
            blockers.append(
                f"{legacy} legacy deployment record"
                f"{'s' if legacy != 1 else ''} must be removed first"
            )
        state = self._read()
        if state.get("active") and state.get("phase") not in {"failed", "cancelled"}:
            blockers.append("a controller transfer is already active on this node")
        return {"ok": not blockers, "blockers": blockers}

    async def agent_receive(self, body: dict[str, Any]) -> dict:
        if not self.onboarding.assignment.load():
            raise RuntimeError("only a joined worker can receive controller state")
        if (
            not isinstance(body, dict)
            or body.get("bundle_version") != TRANSFER_BUNDLE_VERSION
        ):
            raise ValueError("unsupported controller transfer bundle version")
        job_id = str(body.get("job_id") or "")
        if not job_id:
            raise ValueError("bundle job id is required")
        target = body.get("target") or {}
        if target.get("node_id") != self.manager.agent_credentials.node_id:
            raise ValueError("bundle was staged for a different node")
        try:
            store_bytes = base64.b64decode(body.get("store_b64") or "", validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError("bundle store payload is not valid base64") from exc
        if not store_bytes:
            raise ValueError("bundle store payload is empty")
        if len(store_bytes) > MAX_STORE_BYTES:
            raise ValueError("bundle store payload exceeds the size limit")
        if hashlib.sha256(store_bytes).hexdigest() != body.get("store_sha256"):
            raise ValueError("bundle store payload failed its checksum")
        # Validate the store before it is allowed near live state.
        with tempfile.TemporaryDirectory() as temporary:
            candidate = Path(temporary) / "store.sqlite3"
            candidate.write_bytes(store_bytes)
            connection = sqlite3.connect(str(candidate))
            try:
                row = connection.execute("PRAGMA integrity_check").fetchone()
            finally:
                connection.close()
        if not row or row[0] != "ok":
            raise ValueError("bundle store payload failed its integrity check")

        staging = self._staging_dir()
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        payload = {key: value for key, value in body.items() if key != "store_b64"}
        atomic_private_json_write(staging / "bundle.json", payload)
        store_path = staging / "store.sqlite3"
        store_path.write_bytes(store_bytes)
        try:
            store_path.chmod(0o600)
        except OSError:
            pass
        bundle_bytes = (staging / "bundle.json").read_bytes()
        atomic_private_json_write(staging / "manifest.json", {
            "job_id": job_id,
            "received_at": time.time(),
            "bundle_sha256": hashlib.sha256(bundle_bytes).hexdigest(),
            "store_sha256": body.get("store_sha256"),
            "bytes": len(store_bytes) + len(bundle_bytes),
        })
        return {"ok": True, "job_id": job_id, "staged_bytes": len(store_bytes) + len(bundle_bytes)}

    async def agent_promote(self, body: dict[str, Any]) -> dict:
        if not self.onboarding.assignment.load():
            raise RuntimeError("only a joined worker can be promoted to controller")
        if not isinstance(body, dict) or set(body) != {"job_id"}:
            raise ValueError("request must contain only job_id")
        staging = self._staging_dir()
        manifest = self._staged_job()
        if manifest is None or manifest.get("job_id") != body.get("job_id"):
            raise RuntimeError("no staged controller state for this transfer")
        bundle_bytes = (staging / "bundle.json").read_bytes()
        if (
            hashlib.sha256(bundle_bytes).hexdigest()
            != manifest.get("bundle_sha256")
        ):
            raise RuntimeError("staged bundle failed its checksum")
        store_bytes = (staging / "store.sqlite3").read_bytes()
        if hashlib.sha256(store_bytes).hexdigest() != manifest.get("store_sha256"):
            raise RuntimeError("staged controller store failed its checksum")
        state = self._read()
        if state.get("active") and state.get("phase") not in {"failed", "cancelled"}:
            raise RuntimeError("a controller transfer is already active on this node")
        self._write({
            "id": str(body["job_id"]),
            "active": True,
            "committed": False,
            "incoming": True,
            "phase": "promoting",
            "message": "Installing controller state",
            "started_at": time.time(),
        })
        helper_pid = self._spawn_helper("promote")
        return {"ok": True, "helper_pid": helper_pid}

    def agent_status(self) -> dict:
        state = self._read()
        state.pop("old_controller_forward_token", None)
        return {
            "capability": CONTROLLER_TRANSFER_CAPABILITY,
            "state": state or None,
            "staged": self._staged_job(),
        }

    def agent_discard(self) -> dict:
        staging = self._staging_dir()
        existed = staging.exists()
        if existed:
            shutil.rmtree(staging)
        state = self._read()
        if state.get("active") and not state.get("committed"):
            state.update(
                active=False,
                phase="cancelled",
                message="Staged controller transfer discarded",
            )
            self._write(state)
        return {"ok": True, "discarded": existed}

    # ---------- lifecycle ----------

    def start(self) -> None:
        """Reconcile durable transfer state with this process's role."""
        state = self._read()
        if not state.get("active"):
            return
        if self.manager.is_joined_worker():
            if (
                state.get("incoming")
                and state.get("phase") in {"promoting", "promoted"}
            ):
                # The promote helper never removed the worker assignment;
                # this node stayed a worker and the outgoing controller still
                # owns the job (its staged bundle remains for a retry).
                state.update(
                    active=False,
                    phase="failed",
                    message="Promotion did not complete; this node stayed a worker",
                )
                self._write(state)
            return
        if state.get("incoming"):
            if state.get("phase") in {"promoting", "promoted", "completing"}:
                if state.get("phase") != "completing":
                    state.update(
                        phase="completing",
                        message="Waiting for every node to re-bind to this controller",
                    )
                    self._write(state)
                self._completion_task = asyncio.create_task(
                    self._completion_loop(state)
                )
                return
            state.update(
                active=False,
                phase="failed",
                error=state.get("error") or "interrupted incoming transfer",
            )
            self._write(state)
            return
        if state.get("committed"):
            state.update(
                phase="rebinding",
                message="Resuming the committed controller transfer",
            )
            self._write(state)
            self._task = asyncio.create_task(self._resume_committed(state))
            return
        state.update(
            active=False,
            phase="failed",
            error="Transfer interrupted before the target was promoted; start it again",
        )
        self._write(state)

    async def close(self) -> None:
        self._closing = True
        for task in (self._task, self._completion_task):
            if task is not None and not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
