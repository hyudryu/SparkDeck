"""Durable, bounded image patch build history and cluster coordination."""
import asyncio
import copy
import hashlib
import json
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from cluster import NodeAgentResponseError
from sparkdeck.image_patches import build_patched_image, normalize_patch_request

CAPABILITY = "patched-images-v1"
ACTIVE = {"queued", "building"}


class ImagePatchJobs:
    def __init__(self, manager, data_dir):
        self.manager = manager
        self.path = Path(data_dir) / "image-patch-builds.json"
        self.tasks = set()
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
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.jobs), encoding="utf-8")
        temporary.replace(self.path)

    def list(self):
        return {"items": copy.deepcopy(self.jobs)}

    def _save_progress(self):
        # Once a build is accepted, a history write failure must not cancel an
        # already running Docker operation or turn a published image into a
        # reported failure. Keep its actual state available in memory.
        try:
            self._save()
        except OSError as exc:
            for job in self.jobs:
                if job["status"] in ACTIVE or job is self.jobs[0]:
                    job["persistence_warning"] = ("Build history could not be saved: " + str(exc))[:2000]

    def get(self, job_id):
        for job in self.jobs:
            if job["id"] == job_id:
                return copy.deepcopy(job)
        raise LookupError("image patch build not found")

    async def start(self, payload, *, agent=False):
        if not isinstance(payload, dict):
            raise ValueError("request body must be an object")
        request = normalize_patch_request({k: v for k, v in payload.items() if k not in {"node_ids", "job_id"}})
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
        if not agent and payload.get("expected_base_id"):
            raise ValueError("expected_base_id is reserved for cluster coordination")
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
        if any(j["status"] in ACTIVE for j in self.jobs):
            raise ValueError("An image patch build is already active. Wait for it to finish.")
        job = {
            "id": job_id or uuid.uuid4().hex, "base_image": request["base_image"], "image": request["image"],
            "request_fingerprint": fingerprint,
            "created_at": datetime.now(timezone.utc).isoformat(), "status": "queued",
            "files": [{"target": f["target"], "sha256": hashlib.sha256(f["content"].encode("utf-8")).hexdigest()} for f in request["files"]],
            "nodes": [{"node_id": n["id"], "node_name": n["name"], "status": "queued", "logs": []} for n in selected],
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
        self._save_progress()

    async def _local(self, node, request):
        loop = asyncio.get_running_loop()
        def log(line):
            loop.call_soon_threadsafe(self._log, node, line)
        result = await asyncio.to_thread(build_patched_image, self.manager.client, request, log)
        self.manager._images_cache = []
        self.manager._images_ts = 0
        return result

    async def _remote(self, node, request):
        registry = self.manager.node_registry
        remote_id = uuid.uuid4().hex
        remote_request = {**request, "job_id": remote_id}
        deadline = time.monotonic() + 7200
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
            logs = [str(line)[:2000] for line in [*detail.get("logs", [])[-200:], *poll_warnings]][-200:]
            if logs != node["logs"]:
                node["logs"] = logs
                self._save_progress()
            if remote and remote["status"] == "succeeded":
                return {"image_id": detail["image_id"], "base_id": detail["base_id"], "files": remote["files"]}
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
                    result = await (self._local(node, request) if agent or node["node_id"] == "local" else self._remote(node, request))
                    if result["files"] != job["files"]:
                        raise RuntimeError("Built image file hashes do not match the submitted patch")
                    if request.get("expected_base_id") and request["expected_base_id"] != result["base_id"]:
                        raise RuntimeError("Selected nodes resolved different base images")
                    request = {**request, "expected_base_id": result["base_id"]}
                    node.update(status="succeeded", image_id=result["image_id"], base_id=result["base_id"])
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
            self._save_progress()
