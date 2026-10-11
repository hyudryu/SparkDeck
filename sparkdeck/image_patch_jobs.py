"""Durable, bounded image patch build history and cluster coordination."""
import asyncio
import copy
import hashlib
import json
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from cluster import NodeAgentResponseError
from sparkdeck.download_throttle import (
    IMAGE_PULL_BANDWIDTH_CAPABILITY,
    acquire_dispatch_slot,
    capped_transfer_timeout_seconds,
    parse_bandwidth_mbps,
)
from sparkdeck.image_patches import build_patched_image, normalize_patch_request, valid_base_image_identity

CAPABILITY = "patched-images-v1"
ACTIVE = {"queued", "building"}


class ImagePatchJobs:
    def __init__(self, manager, data_dir):
        self.manager = manager
        self.path = Path(data_dir) / "image-patch-builds.json"
        self.tasks = set()
        self._log_flush_handle = None
        self._log_flush_task = None
        self._write_lock = threading.Lock()
        self._snapshot_generation = 0
        self._written_generation = 0
        try:
            self.jobs = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.jobs = []
        for job in self.jobs:
            if job["status"] in ACTIVE:
                job["status"] = "failed"
                job["error"] = "SparkDeck restarted during this build. Inspect Images before retrying with a new tag."
                for node in job["nodes"]:
                    if node["status"] in ACTIVE:
                        node.update(status="failed", error=job["error"])

    def _save(self):
        snapshot, generation = self._snapshot()
        self._write_snapshot(snapshot, generation)

    def _snapshot(self):
        self._snapshot_generation += 1
        return copy.deepcopy(self.jobs), self._snapshot_generation

    def _write_snapshot(self, snapshot, generation):
        with self._write_lock:
            # A delayed background log flush must never overwrite a newer
            # initial/terminal state saved by the event loop.
            if generation <= self._written_generation:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps(snapshot), encoding="utf-8")
            temporary.replace(self.path)
            self._written_generation = generation

    def list(self):
        return {"items": copy.deepcopy(self.jobs)}

    def verified_images(self):
        """Small provenance records for inventory, without copying build logs."""
        return [{
            "id": job["id"], "image": job["image"], "status": job["status"],
            "files": job["files"],
            "nodes": [{key: node.get(key) for key in ("node_id", "image_id", "base_id", "base_identity", "status")} for node in job["nodes"]],
        } for job in self.jobs if job["status"] == "succeeded"]

    def _save_progress(self):
        # Once a build is accepted, a history write failure must not cancel an
        # already running Docker operation or turn a published image into a
        # reported failure. Keep its actual state available in memory.
        try:
            self._save()
        except OSError as exc:
            self._persistence_warning(exc)

    def _persistence_warning(self, exc):
        for job in self.jobs:
            if job["status"] in ACTIVE or job is self.jobs[0]:
                job["persistence_warning"] = ("Build history could not be saved: " + str(exc))[:2000]

    def _begin_log_flush(self):
        self._log_flush_handle = None
        if self._log_flush_task is not None and not self._log_flush_task.done():
            self._schedule_log_flush()
            return
        self._log_flush_task = asyncio.create_task(self._flush_logs())

    async def _flush_logs(self):
        try:
            snapshot, generation = self._snapshot()
            await asyncio.to_thread(self._write_snapshot, snapshot, generation)
        except OSError as exc:
            self._persistence_warning(exc)

    def _schedule_log_flush(self):
        if self._log_flush_handle is None:
            # The snapshot/write is outside the Docker stream callback. The
            # callback only updates bounded in-memory logs for immediate UI use.
            self._log_flush_handle = asyncio.get_running_loop().call_later(1, self._begin_log_flush)

    def get(self, job_id):
        for job in self.jobs:
            if job["id"] == job_id:
                return copy.deepcopy(job)
        raise LookupError("image patch build not found")

    async def start(self, payload, *, agent=False):
        if not isinstance(payload, dict):
            raise ValueError("request body must be an object")
        request = normalize_patch_request(
            {k: v for k, v in payload.items()
             if k not in {"node_ids", "job_id", "max_download_bandwidth_mbps"}}
        )
        bandwidth = payload.get("max_download_bandwidth_mbps")
        if bandwidth is not None:
            if not agent:
                raise ValueError(
                    "max_download_bandwidth_mbps is reserved for cluster coordination"
                )
            # The controller's forwarded cap is authoritative on this node:
            # its own setting may still be the unlimited default.
            request["max_download_bandwidth_mbps"] = parse_bandwidth_mbps(bandwidth)
        job_id = payload.get("job_id")
        if "job_id" in payload:
            if not agent:
                raise ValueError("job_id is reserved for cluster coordination")
            if not isinstance(job_id, str) or not re.fullmatch(r"[0-9a-f]{32}", job_id):
                raise ValueError("job_id must be a 32-character hexadecimal ID")
        fingerprint = hashlib.sha256(json.dumps(request, sort_keys=True).encode("utf-8")).hexdigest()
        if job_id:
            for existing in self.jobs:
                if existing["id"] == job_id:
                    if existing.get("request_fingerprint") != fingerprint:
                        raise ValueError("job_id already belongs to a different patch request")
                    return copy.deepcopy(existing)
        if not agent and ("expected_base_id" in payload or "expected_base_identity" in payload):
            raise ValueError("expected_base_id and expected_base_identity are reserved for cluster coordination")
        missing_bandwidth: list = []
        cap_mbps = 0.0
        if agent:
            selected = [{"id": "local", "name": "This node"}]
        else:
            ids = payload.get("node_ids")
            if not isinstance(ids, list) or not ids or any(not isinstance(n, str) or not n for n in ids):
                raise ValueError("select at least one build node")
            selected = await self.manager.selected_cluster_nodes(ids)
            unsupported = [n["name"] for n in selected if n["id"] != "local" and CAPABILITY not in n.get("capabilities", [])]
            if unsupported:
                raise ValueError("Update SparkDeck before building patched images on: " + ", ".join(unsupported))
            # A remote build's base pull happens on that node with its local
            # settings; forward this controller's cap instead and refuse the
            # dispatch to agents that could not honor it.
            try:
                cap_mbps = parse_bandwidth_mbps(
                    (getattr(self.manager, "settings", None) or {}).get(
                        "max_download_bandwidth_mbps", 0,
                    )
                )
            except (AttributeError, ValueError):
                cap_mbps = 0.0
            missing_bandwidth = [
                n["name"] for n in selected
                if n["id"] != "local"
                and IMAGE_PULL_BANDWIDTH_CAPABILITY not in n.get("capabilities", [])
            ]
            if cap_mbps > 0 and missing_bandwidth:
                raise ValueError(
                    "Update SparkDeck before enforcing the download limit on: "
                    + ", ".join(missing_bandwidth)
                )
            if not missing_bandwidth:
                # Explicit 0 overrides a stale node-local cap; omit the field
                # for old agents that would reject it.
                request["max_download_bandwidth_mbps"] = cap_mbps
        if any(j["status"] in ACTIVE for j in self.jobs):
            raise ValueError("An image patch build is already active. Wait for it to finish.")
        node_records = []
        for n in selected:
            record = {
                "node_id": n["id"], "node_name": n["name"],
                "status": "queued", "logs": [],
            }
            # A mixed fleet is addressed per node: the shared request stays
            # free of the coordination field so legacy agents accept the
            # POST, while each bandwidth-capable node records the resolved
            # cap so a stale node-local cap cannot keep throttling its base
            # pull (the dispatch attaches the field to that node's payload).
            if missing_bandwidth and IMAGE_PULL_BANDWIDTH_CAPABILITY in n.get(
                "capabilities", [],
            ):
                record["max_download_bandwidth_mbps"] = cap_mbps
            node_records.append(record)
        job = {
            "id": job_id or uuid.uuid4().hex, "base_image": request["base_image"], "image": request["image"],
            "request_fingerprint": fingerprint,
            "created_at": datetime.now(timezone.utc).isoformat(), "status": "queued",
            "files": [{"target": f["target"], "sha256": hashlib.sha256(f["content"].encode("utf-8")).hexdigest()} for f in request["files"]],
            "nodes": node_records,
        }
        previous_jobs = self.jobs
        self.jobs = [job, *previous_jobs[:99]]
        try:
            self._save()
        except Exception:
            self.jobs = previous_jobs
            raise
        task = asyncio.create_task(self._run(job, request, agent=agent))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return copy.deepcopy(job)

    def _log(self, node, line):
        node["logs"] = [*node["logs"], str(line)[:2000]][-200:]
        self._schedule_log_flush()

    async def _prefetch_base_image(self, node, request):
        """Pull the build's base image under the cluster dispatch slot.

        The reservation covers ONLY this pull phase: the capped base pull is
        the sole bandwidth consumer of a capped patch build, and a cached
        base comes back from the daemon pull as a manifest check in seconds.
        A node-targeted pull is used (not the Images fan-out) so a mixed
        fleet pulls at this node's recorded cap, which is the authority the
        build was dispatched with.
        """
        base = str(request.get("base_image") or "").strip()
        if not base:
            return
        cap_mbps = float(request.get("max_download_bandwidth_mbps") or 0)
        async with acquire_dispatch_slot(
            getattr(self.manager, "dispatched_download_lock", None), True,
        ):
            if node["node_id"] == "local":
                await self.manager.pull_image_result(base, cap_mbps)
            else:
                await self.manager.node_registry.request(
                    node["node_id"], "POST", "/api/agent/images/pull",
                    json_body={
                        "image": base,
                        "max_download_bandwidth_mbps": cap_mbps,
                    },
                    timeout=capped_transfer_timeout_seconds(
                        cap_mbps, default=1800,
                    ),
                )

    async def _local(self, node, request):
        loop = asyncio.get_running_loop()
        def log(line):
            loop.call_soon_threadsafe(self._log, node, line)
        # The build runs in a worker thread with a synchronous docker
        # client, and its images.pull ignores the bandwidth setting. Fetch
        # a missing base through the capped pull API first when capped.
        forwarded = request.get("max_download_bandwidth_mbps")
        pull_kwargs: dict = {}
        if forwarded is not None:
            # The controller forwarded its cap: it is authoritative here
            # because this node's own setting may still be unlimited.
            capped = float(forwarded) > 0
            pull_kwargs = {"bandwidth_mbps": float(forwarded)}
        else:
            resolver = getattr(self.manager, "_image_pull_bytes_per_second", None)
            try:
                capped = bool(resolver) and resolver(None) > 0
            except TypeError:
                capped = False
        if capped:
            try:
                def _base_missing():
                    try:
                        self.manager.client.images.get(request["base_image"])
                        return False
                    except Exception as exc:
                        if exc.__class__.__name__ != "ImageNotFound":
                            raise
                        return True
                if await asyncio.to_thread(_base_missing):
                    log("Pulling base image (bandwidth-limited)…")
                    await self.manager.pull_image_result(
                        request["base_image"], **pull_kwargs,
                    )
            except Exception as exc:
                self._log(node, f"Base image pull failed: {exc}")
                raise
        # max_download_bandwidth_mbps is a coordination field the controller
        # (or a forwarded agent dispatch) adds to the request; the patch
        # builder re-validates its payload against the strict field set, so
        # strip it before handing the request over.
        build_request = {
            key: value for key, value in request.items()
            if key != "max_download_bandwidth_mbps"
        }
        result = await asyncio.to_thread(build_patched_image, self.manager.client, build_request, log)
        self.manager._images_cache = []
        self.manager._images_ts = 0
        return result

    async def _remote(self, node, request):
        registry = self.manager.node_registry
        remote_id = uuid.uuid4().hex
        remote_request = {**request, "job_id": remote_id}
        # The agent-side base pull honors the forwarded cap, so a large base
        # image at a low cap legitimately runs past the historical fixed two
        # hours. Scale the polling window with the cap (unknown archive size
        # uses the capped-transfer allowance), and re-arm it whenever the
        # remote reports new progress: slow work is not a stalled job.
        window = capped_transfer_timeout_seconds(
            request.get("max_download_bandwidth_mbps"), default=7200.0,
            floor=7200.0,
        )
        deadline = time.monotonic() + window
        seen_remote_logs: list | None = None
        poll_warnings = []
        remote = None
        method = "POST"
        accepted = False
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("Timed out waiting for the build. It may still be running on this node; inspect Images before retrying.")
            try:
                if method == "POST":
                    remote = await registry.request(node["node_id"], "POST", "/api/agent/images/patch-builds", json_body=remote_request, timeout=min(30, remaining))
                else:
                    remote = await registry.request(node["node_id"], "GET", f"/api/agent/images/patch-builds/{remote_id}", timeout=min(30, remaining))
                accepted = True
                method = "GET"
            except (RuntimeError, OSError) as exc:
                if (method == "POST" and "expected_base_identity" in request
                        and isinstance(exc, NodeAgentResponseError) and exc.status_code == 400
                        and "unsupported" in str(exc).casefold()):
                    raise RuntimeError(
                        f"Update SparkDeck on {node.get('node_name') or node['node_id']} "
                        "before building this patched image; this node cannot verify the base runtime identity"
                    ) from exc
                if method == "GET" and not accepted and isinstance(exc, NodeAgentResponseError) and exc.status_code == 404:
                    # A lost POST response is ambiguous. Reconcile by known ID;
                    # if absent, resubmit the same idempotent request, never a
                    # new build that might race the first accepted POST.
                    method = "POST"
                elif isinstance(exc, NodeAgentResponseError) and exc.status_code < 500 and exc.status_code not in {408, 429}:
                    raise
                else:
                    method = "GET"
                    poll_warnings = [*poll_warnings, ("Status temporarily unavailable; retrying the existing build: " + str(exc))[:2000]][-20:]
            detail = remote["nodes"][0] if remote else {}
            # Remote progress (not transient poll warnings) is what proves the
            # build is still working, so the polling window re-arms only on it.
            detail_logs = [str(line)[:2000] for line in detail.get("logs", [])[-200:]]
            if detail.get("logs") is not None and detail_logs != seen_remote_logs:
                seen_remote_logs = detail_logs
                deadline = time.monotonic() + window
            logs = [*detail_logs, *poll_warnings][-200:]
            if logs != node["logs"]:
                node["logs"] = logs
                self._save_progress()
            if remote and remote["status"] == "succeeded":
                result = {"image_id": detail["image_id"], "base_id": detail["base_id"], "files": remote["files"]}
                if detail.get("base_identity") is not None:
                    result["base_identity"] = detail["base_identity"]
                return result
            if remote and remote["status"] == "failed":
                raise RuntimeError(detail.get("error") or remote.get("error") or "Image build failed")
            await asyncio.sleep(2)

    async def _run(self, job, request, *, agent):
        try:
            job["status"] = "building"
            self._save_progress()
            for node in job["nodes"]:
                node["status"] = "building"
                self._save_progress()
                try:
                    node_request = request
                    if (
                        "max_download_bandwidth_mbps" not in request
                        and "max_download_bandwidth_mbps" in node
                    ):
                        # Mixed-fleet dispatch: this node advertises the
                        # bandwidth capability, so attach its recorded cap
                        # (legacy nodes keep the field-free shared request).
                        node_request = {
                            **request,
                            "max_download_bandwidth_mbps":
                                node["max_download_bandwidth_mbps"],
                        }
                    build_capped = float(
                        node_request.get("max_download_bandwidth_mbps") or 0,
                    ) > 0
                    # The bandwidth budget is consumed ONLY while pulling an
                    # uncached base image; the build itself is disk/CPU work.
                    # Pre-pull the base under the cluster dispatch slot, then
                    # release the slot before building: holding the budget
                    # across a long docker build (or the entire remote job
                    # poll) would block every other capped transfer for work
                    # that moves no registry bytes. A cached base makes the
                    # pull a quick manifest round-trip, and the builder then
                    # finds the image present and pulls nothing. Uncapped
                    # builds keep the pre-cap parallel behavior.
                    if build_capped:
                        await self._prefetch_base_image(node, node_request)
                    result = await (self._local(node, node_request) if agent or node["node_id"] == "local" else self._remote(node, node_request))
                    if result["files"] != job["files"]:
                        raise RuntimeError("Built image file hashes do not match the submitted patch")
                    if request.get("expected_base_id") and request["expected_base_id"] != result["base_id"]:
                        raise RuntimeError("Selected nodes resolved different base images")
                    identity = result.get("base_identity")
                    if identity is not None and not valid_base_image_identity(identity):
                        raise RuntimeError("Node returned an invalid base image runtime identity")
                    if request.get("expected_base_identity") and identity is None:
                        raise RuntimeError(f"Update SparkDeck on {node['node_name']} to verify the base runtime identity")
                    if request.get("expected_base_identity") and request["expected_base_identity"] != identity:
                        raise RuntimeError("Selected nodes resolved different base image contents")
                    if identity is not None and "expected_base_id" not in request:
                        request = {**request, "expected_base_identity": identity}
                    else:
                        # Older agents/controllers only understand Docker IDs.
                        # Keep their strict comparison; never silently weaken
                        # an existing constraint or compare tags as identity.
                        request = {**request, "expected_base_id": result["base_id"]}
                    node.update(status="succeeded", image_id=result["image_id"], base_id=result["base_id"])
                    if identity is not None and "expected_base_id" not in request:
                        # A legacy-ID coordinated build has no runtime proof
                        # for its first node. Keep one consistent provenance
                        # scheme even when later agents support fingerprints.
                        node["base_identity"] = identity
                except Exception as exc:
                    node.update(status="failed", error=str(exc)[:2000])
                    raise
                finally:
                    self._save_progress()
            job["status"] = "succeeded"
        except Exception as exc:
            job.update(status="failed", error=str(exc)[:2000])
            for node in job["nodes"]:
                if node["status"] == "queued":
                    node.update(status="failed", error="Not built because an earlier node failed. Use a new tag to retry.")
        finally:
            if self._log_flush_handle is not None:
                self._log_flush_handle.cancel()
                self._log_flush_handle = None
            if self._log_flush_task is not None:
                await self._log_flush_task
                self._log_flush_task = None
            self._save_progress()
