"""llama-benchy wrapper: install detection, benchmark runs, CSV export.

llama-benchy (https://github.com/eugr/llama-benchy) benchmarks any
OpenAI-compatible endpoint with llama-bench style measurements. This service
detects the tool, runs it against models currently served by SparkDeck,
captures its JSON report, stores a meaningful CSV per run, and keeps run
history on disk under ``<data_dir>/benchy/runs/<run_id>/``.
"""

from __future__ import annotations

import asyncio
import csv
import json
import re
import shutil
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RUN_ACTIVE_STATES = ("pending", "running")
RUN_TERMINAL_STATES = ("completed", "failed", "cancelled")
MAX_RUN_SECONDS = 4 * 60 * 60
MAX_LIST_ITEMS = 8
_MAX_POSITIVE = 2_000_000


class BenchyError(ValueError):
    """Raised for invalid benchmark requests; maps to HTTP 400."""


class BenchyService:
    """Run and record llama-benchy benchmarks against served models."""

    def __init__(self, manager: Any, sparkdeck: Any, data_dir: Path):
        self.manager = manager
        self.sparkdeck = sparkdeck
        self.runs_dir = Path(data_dir) / "benchy" / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.runs: dict[str, dict[str, Any]] = {}
        self._processes: dict[str, asyncio.subprocess.Process] = {}
        self._install_lock = asyncio.Lock()
        self._detect_cache: dict[str, Any] | None = None
        for state_path in sorted(self.runs_dir.glob("*/state.json")):
            run = self._load_state(state_path)
            if run is None:
                continue
            if run.get("status") in RUN_ACTIVE_STATES:
                # A controller restart orphans the subprocess; mark it failed
                # rather than leaving a permanently "running" history entry.
                run["status"] = "failed"
                run["error"] = "Benchmark interrupted by a SparkDeck restart"
                run["finished_at"] = _utcnow()
                self._save_state(run)
            self.runs[run["id"]] = run

    # ---------- tool detection & install ----------

    async def detect(self, refresh: bool = False) -> dict[str, Any]:
        if self._detect_cache is not None and not refresh:
            return self._detect_cache
        on_path = shutil.which("llama-benchy")
        version = None
        if on_path:
            version = await self._probe_version([on_path, "--version"])
        mode = "path" if version else None
        if version is None:
            # No usable console script: fall back to the Python module (the
            # pip install below makes it importable in the server's venv).
            version = await self._probe_version(
                [sys.executable, "-m", "llama_benchy", "--version"]
            )
            if version is not None:
                mode = "python_module"
                on_path = None
        self._detect_cache = {
            "installed": version is not None,
            "version": version,
            "launch_mode": mode,
            "path_on_host": bool(on_path),
        }
        return self._detect_cache

    async def install(self) -> dict[str, Any]:
        async with self._install_lock:
            try:
                process = await asyncio.create_subprocess_exec(
                    sys.executable, "-m", "pip", "install", "--upgrade", "llama-benchy",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
            except OSError as exc:
                raise BenchyError(f"could not start pip: {exc}") from exc
            try:
                stdout, _ = await asyncio.wait_for(process.communicate(), timeout=900)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise BenchyError("llama-benchy installation timed out")
        if process.returncode != 0:
            raise BenchyError(_output_tail(stdout))
        return await self.detect(refresh=True)

    def _argv_prefix(self, detection: dict[str, Any]) -> list[str]:
        if detection.get("path_on_host"):
            return [shutil.which("llama-benchy") or "llama-benchy"]
        return [sys.executable, "-m", "llama_benchy"]

    @staticmethod
    async def _probe_version(argv: list[str]) -> str | None:
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except (OSError, ValueError):
            return None
        try:
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=30)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return None
        if process.returncode != 0:
            return None
        match = re.search(r"(\d+\.\d+\.\d+[^\s]*)", stdout.decode(errors="replace"))
        return match.group(1) if match else None

    # ---------- served model discovery ----------

    async def served_models(self) -> list[dict[str, Any]]:
        """Models currently served, with the endpoint llama-benchy should hit."""
        models: list[dict[str, Any]] = []
        seen: set[str] = set()
        llama = await self._native_llama_target()
        if llama:
            models.append(llama)
            seen.add(llama["id"])
        try:
            served = await self.sparkdeck.models()
        except Exception:
            served = {"data": []}
        for entry in served.get("data", []):
            model_id = str(entry.get("id") or "")
            if not model_id or model_id in seen:
                continue
            deployment_id = entry.get("deployment_id")
            target = self._deployment_target(entry, deployment_id)
            if target is None:
                continue
            models.append(target)
            seen.add(model_id)
        return models

    async def _native_llama_target(self) -> dict[str, Any] | None:
        manager = self.manager
        if not manager._llama_running():
            return None
        model = getattr(manager, "_llama_model", None)
        port = getattr(manager, "_llama_port", None)
        if not model or not port:
            return None
        host = manager.settings.get("llama_server_host") or "127.0.0.1"
        variant = manager._unsloth_variant(model)
        return {
            "id": model,
            "label": model.split("/")[-1],
            "runtime": "llama.cpp",
            "deployment_id": None,
            "model": model,
            "quantization": variant or None,
            "base_url": f"http://{host}:{int(port)}",
        }

    def _deployment_target(
        self, entry: dict[str, Any], deployment_id: str | None,
    ) -> dict[str, Any] | None:
        if not deployment_id:
            return None
        stored = self.sparkdeck.store.deployment(deployment_id, include_private=True)
        if stored is None:
            return None
        base_url = str(stored.get("_base_url") or "").rstrip("/")
        if stored.get("kind") == "managed":
            port = entry.get("port") or stored.get("port")
            if port:
                base_url = f"http://127.0.0.1:{int(port)}"
        if not base_url:
            return None
        identity = entry.get("model") or {}
        return {
            "id": entry["id"],
            "label": entry.get("id"),
            "runtime": entry.get("runtime"),
            "deployment_id": deployment_id,
            "model": identity.get("repository") or entry["id"],
            "quantization": identity.get("quantization"),
            "base_url": base_url,
        }

    # ---------- run lifecycle ----------

    async def start_run(self, body: dict[str, Any]) -> dict[str, Any]:
        detection = await self.detect()
        if not detection.get("installed"):
            raise BenchyError("llama-benchy is not installed")
        active = self.active_run()
        if active:
            raise BenchyError(f"benchmark run {active['id']} is already in progress")
        config = self._validate_config(body)
        target = next(
            (item for item in await self.served_models() if item["id"] == config["model_id"]),
            None,
        )
        if target is None:
            raise BenchyError(
                f"model {config['model_id']} is not currently served; load it first"
            )

        run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        run: dict[str, Any] = {
            "id": run_id,
            "status": "running",
            "created_at": _utcnow(),
            "started_at": _utcnow(),
            "finished_at": None,
            "duration_seconds": None,
            "model": target["model"],
            "model_id": target["id"],
            "quantization": target["quantization"],
            "runtime": target["runtime"],
            "base_url": target["base_url"],
            "deployment_id": target["deployment_id"],
            "config": config,
            "benchy_version": detection.get("version"),
            "error": None,
            "progress": {"requests_done": 0, "requests_failed": 0, "current": None},
            "results": [],
            "result_count": 0,
            "csv_filename": None,
            "report": None,
        }
        self.runs[run_id] = run
        self._save_state(run)
        argv = self._build_argv(run, target, run_dir, detection)
        run["_argv"] = argv
        run["_run_dir"] = str(run_dir)
        process = await self._spawn(run)
        asyncio.get_running_loop().create_task(self._monitor_run(run, run_dir))
        return self.public_run(run)

    def _build_argv(
        self, run: dict[str, Any], target: dict[str, Any],
        run_dir: Path, detection: dict[str, Any],
    ) -> list[str]:
        config = run["config"]
        argv = self._argv_prefix(detection) + [
            "--base-url", target["base_url"],
            "--model", target["model"],
            "--pp", *(str(value) for value in config["prompt_sizes"]),
            "--tg", *(str(value) for value in config["response_sizes"]),
            "--concurrency", *(str(value) for value in config["concurrency_levels"]),
            "--depth", *(str(value) for value in config["context_depths"]),
            "--runs", str(config["runs"]),
            "--warmup-runs", str(config["warmup_runs"]),
            "--format", "json",
            "--save-result", str(run_dir / "report.json"),
            "--emit-progress", str(run_dir / "progress.jsonl"),
        ]
        if config.get("exact_tg"):
            argv.append("--exact-tg")
        return argv

    async def _spawn(self, run: dict[str, Any]) -> asyncio.subprocess.Process:
        log_path = Path(run["_run_dir"]) / "output.log"
        argv = run.pop("_argv")
        try:
            with open(log_path, "wb") as log_file:
                process = await asyncio.create_subprocess_exec(
                    *argv,
                    stdout=log_file,
                    stderr=log_file,
                    stdin=asyncio.subprocess.DEVNULL,
                )
        except OSError as exc:
            run["status"] = "failed"
            run["error"] = f"could not launch llama-benchy: {exc}"
            run["finished_at"] = _utcnow()
            self._save_state(run)
            raise BenchyError(run["error"])
        self._processes[run["id"]] = process
        return process

    async def _monitor_run(
        self, run: dict[str, Any], run_dir: Path,
    ) -> None:
        run_id = run["id"]
        process = self._processes.get(run_id)
        progress_path = run_dir / "progress.jsonl"
        report_path = run_dir / "report.json"
        offset = 0
        deadline = time.monotonic() + MAX_RUN_SECONDS
        try:
            while True:
                offset = self._consume_progress(run, progress_path, offset)
                if process is None or process.returncode is not None:
                    break
                if time.monotonic() > deadline:
                    run["error"] = "benchmark exceeded the maximum allowed duration"
                    process.kill()
                await asyncio.sleep(0.5)
            returncode = await process.wait() if process else -1
            self._processes.pop(run_id, None)
            self._consume_progress(run, progress_path, offset)
            run["finished_at"] = _utcnow()
            started = _parse_ts(run["started_at"])
            finished = _parse_ts(run["finished_at"])
            if started and finished:
                run["duration_seconds"] = round(finished - started, 1)
            cancelled = bool(run.pop("_cancel_requested", False))
            if cancelled:
                run["status"] = "cancelled"
            elif returncode == 0 and report_path.is_file():
                try:
                    report = json.loads(report_path.read_text(encoding="utf-8"))
                except (OSError, ValueError) as exc:
                    run["status"] = "failed"
                    run["error"] = f"could not read llama-benchy report: {exc}"
                else:
                    run["status"] = "completed"
                    run["report"] = {
                        "benchy_version": report.get("version"),
                        "latency_mode": report.get("latency_mode"),
                        "latency_ms": report.get("latency_ms"),
                        "prefix_caching_enabled": report.get("prefix_caching_enabled"),
                    }
                    run["results"] = _flatten_report(run, report)
                    run["result_count"] = len(run["results"])
                    _write_csv(run_dir / "results.csv", run)
                    run["csv_filename"] = "results.csv"
            else:
                run["status"] = "failed"
                if not run.get("error"):
                    run["error"] = (
                        f"llama-benchy exited with code {returncode}; "
                        f"{_log_tail(run_dir / 'output.log')}"
                    )
            run.pop("_run_dir", None)
            self._save_state(run)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # defensive: monitoring must never crash the server
            run["status"] = "failed"
            run["error"] = str(exc)
            run["finished_at"] = _utcnow()
            self._save_state(run)

    def _consume_progress(
        self, run: dict[str, Any], progress_path: Path, offset: int,
    ) -> int:
        try:
            with open(progress_path, "rb") as handle:
                handle.seek(offset)
                raw = handle.read()
                offset += len(raw)
        except OSError:
            return offset
        progress = run.get("progress") or {}
        for line in raw.decode(errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("type") == "request_start":
                progress["current"] = {
                    "prompt_size": event.get("prompt_size"),
                    "response_size": event.get("response_size"),
                    "context_depth": event.get("context_size"),
                    "concurrency": event.get("concurrency"),
                }
            elif event.get("type") == "request_end":
                if event.get("error"):
                    progress["requests_failed"] = int(progress.get("requests_failed") or 0) + 1
                else:
                    progress["requests_done"] = int(progress.get("requests_done") or 0) + 1
        run["progress"] = progress
        return offset

    async def cancel_run(self, run_id: str) -> dict[str, Any]:
        run = self.runs.get(run_id)
        if run is None:
            raise LookupError("benchmark run not found")
        if run["status"] not in RUN_ACTIVE_STATES:
            raise BenchyError("benchmark run is not active")
        run["_cancel_requested"] = True
        process = self._processes.get(run_id)
        if process and process.returncode is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
        return self.public_run(run)

    def delete_run(self, run_id: str) -> None:
        run = self.runs.get(run_id)
        if run is None:
            raise LookupError("benchmark run not found")
        if run["status"] in RUN_ACTIVE_STATES:
            raise BenchyError("cancel the running benchmark before deleting it")
        self.runs.pop(run_id, None)
        shutil.rmtree(self.runs_dir / run_id, ignore_errors=True)

    def csv_path(self, run_id: str) -> Path:
        run = self.runs.get(run_id)
        if run is None:
            raise LookupError("benchmark run not found")
        path = self.runs_dir / run_id / "results.csv"
        if not path.is_file():
            raise LookupError("benchmark run has no CSV results")
        return path

    # ---------- querying ----------

    def active_run(self) -> dict[str, Any] | None:
        for run in self.runs.values():
            if run["status"] in RUN_ACTIVE_STATES:
                return run
        return None

    def list_runs(self) -> list[dict[str, Any]]:
        items = []
        for run in sorted(
            self.runs.values(), key=lambda item: item["created_at"], reverse=True,
        ):
            item = self.public_run(run)
            item.pop("results", None)  # details are fetched per run on demand
            items.append(item)
        return items

    def get_run(self, run_id: str) -> dict[str, Any]:
        run = self.runs.get(run_id)
        if run is None:
            raise LookupError("benchmark run not found")
        return self.public_run(run)

    def public_run(self, run: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in run.items() if not key.startswith("_")}

    # ---------- validation & persistence ----------

    def _validate_config(self, body: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(body, dict):
            raise BenchyError("request body must be an object")
        model_id = str(body.get("model_id") or "").strip()
        if not model_id:
            raise BenchyError("model_id is required")
        prompt_sizes = _positive_list(body.get("prompt_sizes"), "prompt_sizes", [2048])
        response_sizes = _positive_list(body.get("response_sizes"), "response_sizes", [128])
        concurrency = _positive_list(
            body.get("concurrency_levels"), "concurrency_levels", [1], maximum=256,
        )
        depths = _positive_list(
            body.get("context_depths"), "context_depths", [0], allow_zero=True,
        )
        runs = _bounded_int(body.get("runs"), "runs", default=3, maximum=10)
        warmup = _bounded_int(body.get("warmup_runs"), "warmup_runs", default=1, maximum=10)
        exact_tg = bool(body.get("exact_tg", False))
        shapes = len(prompt_sizes) * len(response_sizes) * len(concurrency) * len(depths)
        if shapes > 256:
            raise BenchyError(
                "the requested combination would run more than 256 test shapes"
            )
        return {
            "model_id": model_id,
            "prompt_sizes": prompt_sizes,
            "response_sizes": response_sizes,
            "concurrency_levels": concurrency,
            "context_depths": depths,
            "runs": runs,
            "warmup_runs": warmup,
            "exact_tg": exact_tg,
        }

    def _load_state(self, state_path: Path) -> dict[str, Any] | None:
        try:
            run = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(run, dict) or not run.get("id"):
            return None
        return run

    def _save_state(self, run: dict[str, Any]) -> None:
        run_dir = self.runs_dir / run["id"]
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "state.json"
        payload = self.public_run(run)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(path)


# ---------- pure helpers (module level for testability) ----------


def _flatten_report(run: dict[str, Any], report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in report.get("benchmarks") or []:
        if not isinstance(item, dict):
            continue
        row = {
            "prompt_size": item.get("prompt_size"),
            "response_size": item.get("response_size"),
            "context_depth": item.get("context_size"),
            "concurrency": item.get("concurrency"),
            "is_context_prefill_phase": bool(item.get("is_context_prefill_phase")),
            "pp_tokens_per_second": _metric_mean(item.get("pp_throughput")),
            "pp_tokens_per_second_std": _metric_std(item.get("pp_throughput")),
            "pp_tokens_per_second_request": _metric_mean(item.get("pp_req_throughput")),
            "pp_tokens_per_second_request_std": _metric_std(item.get("pp_req_throughput")),
            "tg_tokens_per_second": _metric_mean(item.get("tg_throughput")),
            "tg_tokens_per_second_std": _metric_std(item.get("tg_throughput")),
            "tg_tokens_per_second_request": _metric_mean(item.get("tg_req_throughput")),
            "tg_tokens_per_second_request_std": _metric_std(item.get("tg_req_throughput")),
            "peak_tg_tokens_per_second": _metric_mean(item.get("peak_throughput")),
            "peak_tg_tokens_per_second_request": _metric_mean(item.get("peak_req_throughput")),
            "ttfr_ms": _metric_mean(item.get("ttfr")),
            "est_ppt_ms": _metric_mean(item.get("est_ppt")),
            "e2e_ttft_ms": _metric_mean(item.get("e2e_ttft")),
        }
        rows.append(row)
    return rows


CSV_COLUMNS = [
    "run_id", "created_at", "model", "model_id", "quantization", "runtime",
    "base_url", "benchy_version", "latency_mode", "latency_ms",
    "prompt_size", "response_size", "context_depth", "concurrency",
    "runs", "warmup_runs", "exact_tg", "is_context_prefill_phase",
    "pp_tokens_per_second", "pp_tokens_per_second_std",
    "pp_tokens_per_second_request", "pp_tokens_per_second_request_std",
    "tg_tokens_per_second", "tg_tokens_per_second_std",
    "tg_tokens_per_second_request", "tg_tokens_per_second_request_std",
    "peak_tg_tokens_per_second", "peak_tg_tokens_per_second_request",
    "ttfr_ms", "est_ppt_ms", "e2e_ttft_ms",
]


def _write_csv(path: Path, run: dict[str, Any]) -> None:
    config = run.get("config") or {}
    report = run.get("report") or {}
    header = {column: None for column in CSV_COLUMNS}
    rows = []
    for result in run.get("results") or []:
        row = {
            **header,
            "run_id": run["id"],
            "created_at": run.get("started_at"),
            "model": run.get("model"),
            "model_id": run.get("model_id"),
            "quantization": run.get("quantization"),
            "runtime": run.get("runtime"),
            "base_url": run.get("base_url"),
            "benchy_version": run.get("benchy_version"),
            "latency_mode": report.get("latency_mode"),
            "latency_ms": _round(report.get("latency_ms")),
            "runs": config.get("runs"),
            "warmup_runs": config.get("warmup_runs"),
            "exact_tg": config.get("exact_tg"),
            **{key: _round(value) for key, value in result.items()},
        }
        rows.append(row)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _metric_mean(metric: Any) -> float | None:
    if not isinstance(metric, dict):
        return None
    value = metric.get("mean")
    return float(value) if isinstance(value, (int, float)) else None


def _metric_std(metric: Any) -> float | None:
    if not isinstance(metric, dict):
        return None
    value = metric.get("std")
    return float(value) if isinstance(value, (int, float)) else None


def _round(value: Any) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    if isinstance(value, int):
        return value
    return round(float(value), 2)


def _positive_list(
    value: Any, name: str, default: list[int], maximum: int = _MAX_POSITIVE,
    allow_zero: bool = False,
) -> list[int]:
    if value is None:
        return list(default)
    if not isinstance(value, list) or not value:
        raise BenchyError(f"{name} must be a non-empty list of integers")
    numbers: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise BenchyError(f"{name} must contain integers only")
        if item < 0 or (item == 0 and not allow_zero) or item > maximum:
            raise BenchyError(f"{name} values must be between "
                              f"{0 if allow_zero else 1} and {maximum}")
        if item not in numbers:
            numbers.append(item)
    if len(numbers) > MAX_LIST_ITEMS:
        raise BenchyError(f"{name} accepts at most {MAX_LIST_ITEMS} values")
    return numbers


def _bounded_int(value: Any, name: str, default: int, maximum: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise BenchyError(f"{name} must be an integer")
    if not 0 <= value <= maximum:
        raise BenchyError(f"{name} must be between 0 and {maximum}")
    return value


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_ts(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None


def _output_tail(stdout: bytes | None, limit: int = 800) -> str:
    text = (stdout or b"").decode(errors="replace").strip()
    return text[-limit:] if text else "llama-benchy installation failed"


def _log_tail(path: Path, limit: int = 800) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return "no output was captured"
    return f"output tail: {text[-limit:]}" if text else "no output was captured"
