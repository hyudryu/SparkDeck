"""
VLLMController - container + queue + telemetry manager.
"""
import asyncio
import json
import re
import subprocess
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any

import docker
import httpx

DEFAULT_SETTINGS = {
    "max_concurrent_models": 1,
    "vllm_image": "nvcr.io/nvidia/vllm:26.03.post1-py3",
    "hf_cache": "/home/hyudryu/.cache/huggingface",
    "port_range_start": 8000,
    "port_range_end": 8099,
    "shm_size": "16g",
    "default_gpu_memory_utilization": 0.9,
}

CONTROLLER_LABEL = "vllm-controller"
MODEL_LABEL = "vllm-model"

PENDING = "pending"          # waiting for capacity
DISPATCHING = "dispatching"  # model starting, will run when ready
RUNNING = "running"          # inference active
DONE = "done"
ERROR = "error"
CANCELED = "canceled"


def _is_vllm_image(tag: str) -> bool:
    return "vllm" in (tag or "").lower()


class Manager:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.settings_path = self.data_dir / "settings.json"
        self.settings = self._load_settings()
        self.client = docker.from_env()
        self.jobs: dict[str, dict] = {}
        self.queue: deque[str] = deque()
        self.lock = asyncio.Lock()
        self.worker_task: asyncio.Task | None = None
        self.http = httpx.AsyncClient(timeout=600)
        self._cpu_prev: tuple[int, int] | None = None
        self._stats_cache: dict[str, Any] = {}
        self._stats_ts: float = 0.0

    # ---------- lifecycle ----------
    async def start(self):
        self.worker_task = asyncio.create_task(self._worker_loop())

    async def stop(self):
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
        await self.http.aclose()

    # ---------- settings ----------
    def _load_settings(self) -> dict:
        if self.settings_path.exists():
            try:
                data = json.loads(self.settings_path.read_text())
                return {**DEFAULT_SETTINGS, **data}
            except Exception:
                pass
        return dict(DEFAULT_SETTINGS)

    def _save_settings(self):
        self.settings_path.write_text(json.dumps(self.settings, indent=2))

    async def update_settings(self, data: dict) -> dict:
        async with self.lock:
            for k, v in data.items():
                if k in DEFAULT_SETTINGS:
                    self.settings[k] = v
            self._save_settings()
        return self.settings

    # ---------- containers ----------
    def _container_summary(self, c) -> dict | None:
        labels = c.labels or {}
        try:
            image_tag = c.image.tags[0] if c.image.tags else c.image.short_id
        except Exception:
            image_tag = ""

        is_managed = labels.get(CONTROLLER_LABEL) == "1"
        if not is_managed and not _is_vllm_image(image_tag):
            return None

        # parse model from cmd or label
        model = labels.get(MODEL_LABEL, "")
        if not model:
            cmd = c.attrs.get("Config", {}).get("Cmd") or []
            if "serve" in cmd:
                i = cmd.index("serve")
                if i + 1 < len(cmd):
                    model = cmd[i + 1]

        host_port = None
        ports = c.ports or {}
        for _, bindings in ports.items():
            if bindings:
                try:
                    host_port = int(bindings[0]["HostPort"])
                    break
                except Exception:
                    pass

        return {
            "id": c.short_id,
            "name": c.name,
            "image": image_tag,
            "status": c.status,
            "model": model,
            "port": host_port,
            "managed": is_managed,
            "created": c.attrs.get("Created"),
        }

    async def list_containers(self) -> list[dict]:
        def _run():
            out = []
            for c in self.client.containers.list(all=True):
                summary = self._container_summary(c)
                if summary:
                    out.append(summary)
            out.sort(key=lambda x: (x["status"] != "running", x["name"]))
            return out
        containers = await asyncio.to_thread(_run)
        # Attach setup phase (health/log-derived) in parallel.
        if containers:
            phases = await asyncio.gather(
                *[self._get_container_phase(c) for c in containers],
                return_exceptions=True,
            )
            for c, ph in zip(containers, phases):
                c["phase"] = ph if isinstance(ph, dict) else {
                    "phase": "unknown", "progress": None, "message": str(ph),
                }
        return containers

    # ---------- setup phase ----------
    _RE_LOAD_SHARDS = re.compile(
        r"(?:Loading|Downloading)\s+(?:safetensors\s+)?checkpoint\s+shards.*?(\d+)\s*/\s*(\d+)",
        re.IGNORECASE,
    )
    _RE_HF_DOWNLOAD = re.compile(
        r"(?:safetensors|pytorch_model|\.bin|Fetching\s+\d+\s+files)[^\n]{0,30}?(\d{1,3})\s*%",
        re.IGNORECASE,
    )
    _RE_LOAD_PCT = re.compile(r"Loading.*?(\d{1,3})\s*%")
    _RE_ERROR = re.compile(r"\b(Error|Traceback|OutOfMemoryError|CUDA out of memory|RuntimeError)\b")

    def _parse_phase(self, logs: str) -> dict:
        if not logs:
            return {"phase": "starting", "progress": None, "message": "starting…"}
        # Strip ANSI escapes and tqdm carriage returns so progress lines parse cleanly.
        logs = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", logs)
        logs = logs.replace("\r", "\n")
        lines = [l for l in logs.split("\n") if l.strip()]
        tail = "\n".join(lines[-80:])

        # Ready signals
        if "Application startup complete" in tail or "Uvicorn running on" in tail:
            return {"phase": "ready", "progress": 1.0, "message": "vLLM API ready"}

        # Loading checkpoint shards (most common phase before ready)
        m = list(self._RE_LOAD_SHARDS.finditer(tail))
        if m:
            cur, tot = int(m[-1].group(1)), int(m[-1].group(2))
            pct = cur / tot if tot else None
            return {
                "phase": "loading",
                "progress": pct,
                "message": f"loading checkpoint shards {cur}/{tot}",
            }

        # HuggingFace download progress
        m = list(self._RE_HF_DOWNLOAD.finditer(tail))
        if m:
            pct = int(m[-1].group(1))
            return {
                "phase": "downloading",
                "progress": pct / 100.0,
                "message": f"downloading model files {pct}%",
            }

        # Generic "Loading X%" fallback
        m = list(self._RE_LOAD_PCT.finditer(tail))
        if m:
            pct = int(m[-1].group(1))
            return {
                "phase": "loading",
                "progress": pct / 100.0,
                "message": f"loading {pct}%",
            }

        # Error detection
        if self._RE_ERROR.search(tail):
            err_line = next(
                (l for l in reversed(lines) if self._RE_ERROR.search(l)),
                "",
            )
            return {
                "phase": "error",
                "progress": None,
                "message": err_line.strip()[:240] or "error in startup logs",
            }

        # Pre-load init signals
        if "Initializing" in tail or "Starting vLLM" in tail or "engine" in tail.lower():
            return {"phase": "initializing", "progress": None, "message": "initializing engine…"}

        return {"phase": "starting", "progress": None, "message": "starting…"}

    async def _get_container_phase(self, c: dict) -> dict:
        if c["status"] != "running":
            return {"phase": c["status"], "progress": None, "message": c["status"]}
        if await self._check_ready(c):
            return {"phase": "ready", "progress": 1.0, "message": "vLLM API ready"}
        try:
            logs = await self.get_logs(c["name"], tail=150)
        except Exception as e:
            return {"phase": "starting", "progress": None, "message": f"starting… ({e})"}
        return self._parse_phase(logs)

    async def _allocate_port(self) -> int:
        def _scan():
            used = set()
            for c in self.client.containers.list(all=True):
                for _, bindings in (c.ports or {}).items():
                    for b in (bindings or []):
                        try:
                            used.add(int(b["HostPort"]))
                        except Exception:
                            pass
            return used
        used = await asyncio.to_thread(_scan)
        for p in range(self.settings["port_range_start"], self.settings["port_range_end"] + 1):
            if p not in used:
                return p
        raise RuntimeError("No available ports in configured range")

    async def create_container(
        self,
        model: str,
        port: int | None = None,
        gpu_memory_utilization: float | None = None,
        extra_args: list[str] | None = None,
        name: str | None = None,
        image: str | None = None,
    ) -> dict:
        image = image or self.settings["vllm_image"]
        if port is None:
            port = await self._allocate_port()
        if gpu_memory_utilization is None:
            gpu_memory_utilization = self.settings["default_gpu_memory_utilization"]
        if name is None:
            safe = model.replace("/", "-").replace("_", "-").lower()
            name = f"vllm-{safe}-{port}"
        extra = list(extra_args or [])

        cmd = [
            "vllm", "serve", model,
            "--host", "0.0.0.0",
            "--port", "8000",
            "--gpu-memory-utilization", str(gpu_memory_utilization),
            *extra,
        ]

        def _create():
            container = self.client.containers.run(
                image=image,
                command=cmd,
                name=name,
                detach=True,
                ports={"8000/tcp": port},
                volumes={
                    self.settings["hf_cache"]: {
                        "bind": "/root/.cache/huggingface",
                        "mode": "rw",
                    }
                },
                ipc_mode="host",
                shm_size=self.settings["shm_size"],
                device_requests=[
                    docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
                ],
                labels={CONTROLLER_LABEL: "1", MODEL_LABEL: model},
                restart_policy={"Name": "unless-stopped"},
            )
            container.reload()
            return self._container_summary(container)
        return await asyncio.to_thread(_create)

    async def start_container(self, name: str) -> dict:
        def _do():
            self.client.containers.get(name).start()
        await asyncio.to_thread(_do)
        return {"ok": True}

    async def stop_container(self, name: str) -> dict:
        def _do():
            self.client.containers.get(name).stop(timeout=30)
        await asyncio.to_thread(_do)
        return {"ok": True}

    async def remove_container(self, name: str) -> dict:
        def _do():
            self.client.containers.get(name).remove(force=True)
        await asyncio.to_thread(_do)
        return {"ok": True}

    async def get_logs(self, name: str, tail: int = 200) -> str:
        def _do():
            return self.client.containers.get(name).logs(tail=tail).decode("utf-8", errors="replace")
        return await asyncio.to_thread(_do)

    # ---------- images ----------
    async def list_images(self) -> list[dict]:
        def _do():
            out = []
            for img in self.client.images.list():
                tags = img.tags or []
                out.append({
                    "id": img.short_id,
                    "tags": tags,
                    "size": img.attrs.get("Size", 0),
                    "created": img.attrs.get("Created"),
                    "is_vllm": any(_is_vllm_image(t) for t in tags),
                })
            out.sort(key=lambda x: (not x["is_vllm"], x["tags"][0] if x["tags"] else ""))
            return out
        return await asyncio.to_thread(_do)

    async def remove_image(self, image_id: str) -> dict:
        def _do():
            self.client.images.remove(image_id, force=True)
        await asyncio.to_thread(_do)
        return {"ok": True}

    async def pull_image_stream(self, image: str):
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()

        def producer():
            try:
                for line in self.client.api.pull(image, stream=True, decode=True):
                    asyncio.run_coroutine_threadsafe(q.put(line), loop)
            except Exception as e:
                asyncio.run_coroutine_threadsafe(q.put({"error": str(e)}), loop)
            finally:
                asyncio.run_coroutine_threadsafe(q.put(None), loop)

        threading.Thread(target=producer, daemon=True).start()

        while True:
            item = await q.get()
            if item is None:
                yield "data: {\"done\": true}\n\n"
                break
            yield f"data: {json.dumps(item)}\n\n"

    # ---------- system telemetry ----------
    def _read_cpu_pct(self) -> float | None:
        try:
            with open("/proc/stat") as f:
                line = f.readline()
            parts = [int(x) for x in line.split()[1:]]
            idle = parts[3] + parts[4]  # idle + iowait
            total = sum(parts)
            if self._cpu_prev is None:
                self._cpu_prev = (idle, total)
                return None
            d_idle = idle - self._cpu_prev[0]
            d_total = total - self._cpu_prev[1]
            self._cpu_prev = (idle, total)
            if d_total <= 0:
                return None
            return round(100.0 * (1 - d_idle / d_total), 1)
        except Exception:
            return None

    def _read_mem(self) -> dict:
        info: dict[str, int] = {}
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    k, v = line.split(":", 1)
                    info[k] = int(v.strip().split()[0]) * 1024
        except Exception:
            return {}
        total = info.get("MemTotal", 0)
        available = info.get("MemAvailable", 0)
        used = total - available
        return {
            "total": total,
            "used": used,
            "available": available,
            "pct": round(100.0 * used / total, 1) if total else 0.0,
        }

    def _read_cpu_temp(self) -> float | None:
        # Take max of acpitz/coretemp/k10temp/cpu-thermal zones
        temps = []
        try:
            for p in Path("/sys/class/thermal").glob("thermal_zone*"):
                try:
                    t = (p / "type").read_text().strip()
                    if any(k in t for k in ("acpitz", "coretemp", "k10temp", "cpu", "x86_pkg_temp", "soc")):
                        v = int((p / "temp").read_text().strip()) / 1000.0
                        temps.append(v)
                except Exception:
                    continue
        except Exception:
            pass
        return round(max(temps), 1) if temps else None

    def _read_gpu(self) -> list[dict]:
        try:
            r = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True, text=True, timeout=4,
            )
            out = []
            for line in r.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 8:
                    continue
                def _num(s):
                    try:
                        return float(s)
                    except Exception:
                        return None
                out.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "util": _num(parts[2]),
                    "mem_used_mib": _num(parts[3]),
                    "mem_total_mib": _num(parts[4]),
                    "temp": _num(parts[5]),
                    "power_draw_w": _num(parts[6]),
                    "power_limit_w": _num(parts[7]),
                })
            return out
        except Exception as e:
            return [{"error": str(e)}]

    async def get_stats(self) -> dict:
        # Throttle to 1Hz so multiple clients don't hammer nvidia-smi
        now = time.time()
        if now - self._stats_ts < 0.8 and self._stats_cache:
            return self._stats_cache

        def _gather():
            return {
                "cpu_pct": self._read_cpu_pct(),
                "cpu_temp_c": self._read_cpu_temp(),
                "mem": self._read_mem(),
                "gpus": self._read_gpu(),
                "ts": time.time(),
            }
        stats = await asyncio.to_thread(_gather)
        self._stats_cache = stats
        self._stats_ts = now
        return stats

    # ---------- jobs ----------
    async def submit_job(
        self,
        model: str,
        messages: list,
        params: dict | None = None,
        container: str | None = None,
    ) -> dict:
        params = params or {}
        job = {
            "id": uuid.uuid4().hex[:8],
            "model": model,
            "container": container,
            "messages": messages,
            "params": params,
            "requested_at": time.time(),
            "started_at": None,
            "completed_at": None,
            "status": PENDING,
            "result": None,
            "error": None,
            "override": False,
        }
        async with self.lock:
            self.jobs[job["id"]] = job
            self.queue.append(job["id"])
        return self._public_job(job)

    async def force_run(self, job_id: str) -> dict:
        async with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return {"error": "not found"}
            job["override"] = True
        return self._public_job(job)

    async def cancel_job(self, job_id: str) -> dict:
        async with self.lock:
            job = self.jobs.get(job_id)
            if job and job["status"] in (PENDING, DISPATCHING):
                job["status"] = CANCELED
                job["completed_at"] = time.time()
                if job_id in self.queue:
                    try:
                        self.queue.remove(job_id)
                    except ValueError:
                        pass
        return {"ok": True}

    async def clear_finished(self) -> dict:
        async with self.lock:
            keep = {}
            for jid, j in self.jobs.items():
                if j["status"] not in (DONE, ERROR, CANCELED):
                    keep[jid] = j
            self.jobs = keep
        return {"ok": True}

    async def get_job(self, job_id: str) -> dict | None:
        return self.jobs.get(job_id)

    def _public_job(self, j: dict) -> dict:
        msgs = j.get("messages") or []
        last = msgs[-1] if msgs else {}
        preview = last.get("content", "") if isinstance(last, dict) else str(last)
        result_text = None
        result = j.get("result") or {}
        try:
            choices = result.get("choices") or []
            if choices:
                msg = choices[0].get("message") or {}
                result_text = msg.get("content") or choices[0].get("text")
        except Exception:
            pass
        return {
            "id": j["id"],
            "model": j["model"],
            "container": j.get("container"),
            "status": j["status"],
            "requested_at": j["requested_at"],
            "started_at": j.get("started_at"),
            "completed_at": j.get("completed_at"),
            "override": j.get("override", False),
            "preview": (preview or "")[:240],
            "result_text": result_text,
            "error": j.get("error"),
        }

    # ---------- worker / scheduler ----------
    async def _worker_loop(self):
        while True:
            try:
                await self._tick()
            except Exception as e:
                print(f"[worker] error: {e}")
            await asyncio.sleep(1.0)

    async def _tick(self):
        async with self.lock:
            pending_ids = [jid for jid in list(self.queue)
                           if self.jobs.get(jid, {}).get("status") in (PENDING, DISPATCHING)]
        if not pending_ids:
            return

        containers = await self.list_containers()
        running = [c for c in containers if c["status"] == "running"]
        starting = [c for c in containers if c["status"] in ("created", "restarting")]
        max_c = int(self.settings.get("max_concurrent_models", 1))

        # Index containers by model and by name
        by_name = {c["name"]: c for c in containers}
        running_by_model: dict[str, list[dict]] = {}
        for c in running:
            running_by_model.setdefault(c["model"], []).append(c)
        any_by_model: dict[str, list[dict]] = {}
        for c in containers:
            any_by_model.setdefault(c["model"], []).append(c)

        for jid in pending_ids:
            job = self.jobs.get(jid)
            if not job or job["status"] not in (PENDING, DISPATCHING):
                continue

            # Resolve target container
            target = None
            if job.get("container") and job["container"] in by_name:
                target = by_name[job["container"]]
            elif job["model"] in running_by_model:
                target = running_by_model[job["model"]][0]
            elif job["model"] in any_by_model:
                # exists but stopped
                stopped = [c for c in any_by_model[job["model"]] if c["status"] != "running"]
                if stopped:
                    target = stopped[0]

            if target and target["status"] == "running":
                ready = await self._check_ready(target)
                if ready:
                    job["status"] = RUNNING
                    job["started_at"] = time.time()
                    asyncio.create_task(self._run_inference(jid, target))
                else:
                    job["status"] = DISPATCHING
                continue

            # Need to start or create
            active_count = len(running) + len(starting)
            can_start = bool(job.get("override")) or active_count < max_c
            if not can_start:
                continue

            try:
                if target:
                    await self.start_container(target["name"])
                else:
                    info = await self.create_container(model=job["model"])
                    job["container"] = info["name"]
                job["status"] = DISPATCHING
                # Reflect in local counters so we don't over-spawn this tick
                starting.append({"name": job.get("container") or "", "status": "created"})
            except Exception as e:
                job["status"] = ERROR
                job["error"] = f"start failed: {e}"
                job["completed_at"] = time.time()

    async def _check_ready(self, container: dict) -> bool:
        port = container.get("port")
        if not port:
            return False
        try:
            r = await self.http.get(f"http://localhost:{port}/health", timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        # Fall back to /v1/models
        try:
            r = await self.http.get(f"http://localhost:{port}/v1/models", timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    async def _run_inference(self, jid: str, container: dict):
        job = self.jobs[jid]
        port = container["port"]
        try:
            url = f"http://localhost:{port}/v1/chat/completions"
            payload: dict = {
                "model": job["model"],
                "messages": job["messages"],
            }
            payload.update(job.get("params") or {})
            r = await self.http.post(url, json=payload, timeout=600)
            r.raise_for_status()
            job["result"] = r.json()
            job["status"] = DONE
        except httpx.HTTPStatusError as e:
            job["status"] = ERROR
            job["error"] = f"HTTP {e.response.status_code}: {e.response.text[:500]}"
        except Exception as e:
            job["status"] = ERROR
            job["error"] = str(e)
        finally:
            job["completed_at"] = time.time()
            async with self.lock:
                if jid in self.queue:
                    try:
                        self.queue.remove(jid)
                    except ValueError:
                        pass

    # ---------- aggregate state ----------
    async def get_state(self) -> dict:
        containers, images, stats = await asyncio.gather(
            self.list_containers(),
            self.list_images(),
            self.get_stats(),
        )
        running_models = [c for c in containers if c["status"] == "running"]
        return {
            "containers": containers,
            "images": images,
            "stats": stats,
            "settings": self.settings,
            "queue": [self._public_job(j) for j in self.jobs.values()],
            "summary": {
                "running_models": len(running_models),
                "max_concurrent_models": int(self.settings.get("max_concurrent_models", 1)),
                "total_containers": len(containers),
            },
        }
