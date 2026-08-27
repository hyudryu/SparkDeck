"""SparkDeck container, queue, cluster, and telemetry manager."""
import asyncio
import codecs
import copy
import ipaddress
import json
import logging
import math
import os
import re
import socket
import shlex
import subprocess
import tempfile
import threading
import time
import unicodedata
import uuid
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import docker
import httpx
import shutil

from cluster import (
    AGENT_PROTOCOL_VERSION,
    LOCAL_NODE_ID,
    AgentCredentials,
    NodeRegistry,
)
from sparkdeck.onboarding import resolve_agent_connection
from sparkdeck.private_json import atomic_private_json_write as _atomic_private_json_write
from sparkdeck.virtual_nas import VirtualNAS, validate_model_id
from sparkdeck.updater import CAPABILITY, current_revision
from sparkdeck.routeros import ROUTEROS_TIMEOUT_SECONDS, RouterOSService

DEFAULT_SETTINGS = {
    "max_concurrent_models": 2,
    "max_retries": 2,
    "idle_timeout_seconds": 30,  # 0 disables auto-stop
    "vllm_image": "nvcr.io/nvidia/vllm:26.03.post1-py3",
    "hf_cache": "/home/hyudryu/.cache/huggingface",
    # Stored server-side and never returned by the settings/state APIs. An
    # empty value falls back to the process environment or the HF cache token.
    "hf_token": "",
    "port_range_start": 8000,
    "port_range_end": 8099,
    "shm_size": "16g",
    "default_gpu_memory_utilization": 0.9,
    # If concurrent vLLM streams stop making progress, temporarily serialize
    # the deployment. Streams that have not emitted anything can be aborted
    # upstream and transparently replayed from the controller queue.
    "vllm_nudger_enabled": False,
    "vllm_nudger_rate_threshold": 5.0,
    "vllm_nudger_stall_seconds": 3.0,
    # Trust vLLM's startup KV-capacity report and lower --max-num-seqs with a
    # coordinated redeploy when the configured full-context concurrency is unsafe.
    "vllm_auto_adjust_concurrency": True,
    # Opt-in model cache replication across authenticated cluster nodes.
    "virtual_nas_enabled": False,
    # Cluster management uses the normal LAN/Tailscale address while model
    # collectives use this ConnectX/RDMA interface. Blank values are inferred
    # from the local interfaces advertised by the node agent.
    "cluster_node_name": socket.gethostname(),
    "cluster_fabric_ip": "",
    "cluster_fabric_interface": "",
    # llama-server launcher (GGUF models from the local HF cache). One server
    # at a time, bound to localhost and proxied via /v1.
    "llama_server_bin": "~/.local/share/llama.cpp/llama-server",
    "llama_server_host": "127.0.0.1",
    "llama_server_port": 8100,
    "llama_rpc_server_bin": "~/.local/share/llama.cpp/ggml-rpc-server",
    "llama_rpc_port": 50052,
    # Flagship model pricing: input/output cost per 1M tokens (USD).
    # Editable from the Usage tab; used for opportunity-cost comparison
    # against locally-served models.
    "flagship_pricing": {
        "Fable 5":           {"input": 0.0, "output": 0.0},
        "Opus 4.8":          {"input": 0.0, "output": 0.0},
        "GPT 5.6 Sol":       {"input": 0.0, "output": 0.0},
        "Kimi K3":           {"input": 0.0, "output": 0.0},
        "GLM 5.2":           {"input": 0.0, "output": 0.0},
    },
}

logger = logging.getLogger(__name__)
HF_CREDENTIAL_CLI_OPTIONS = {"--hf-token", "--hf_token"}
PERSISTED_RECIPE_ARGS_ERROR = (
    "unsupported persisted extra_args: expected an array of strings"
)
PERSISTED_DEPLOYMENT_ARGS_ERROR = (
    "unsupported persisted launch_settings.extra_args: expected an array of strings"
)


class _RetryingDockerClient:
    """Reconnect to Docker lazily after the daemon was unavailable at startup.

    Docker's ``from_env`` constructor negotiates an API version immediately,
    so constructing the process-wide manager used to prevent controller-only
    installations from starting at all. Defer even the first negotiation until
    an operation actually needs Docker. Failed attempts are cached briefly and
    only one caller may negotiate at a time, so aggregate state polling cannot
    fill the thread pool while a daemon is hung. A later access retries and
    permanently caches the first working client.
    """

    RETRY_INTERVAL_SECONDS = 5.0
    CONNECT_WAIT_SECONDS = 0.25

    def __init__(
        self,
        initial_error=None,
        *,
        retry_interval=None,
        connect_wait_timeout=None,
        clock=None,
    ):
        self._client = None
        self._condition = threading.Condition()
        self._connecting = False
        self._clock = clock or time.monotonic
        self._retry_interval = (
            self.RETRY_INTERVAL_SECONDS
            if retry_interval is None
            else max(0.0, float(retry_interval))
        )
        self._connect_wait_timeout = (
            self.CONNECT_WAIT_SECONDS
            if connect_wait_timeout is None
            else max(0.0, float(connect_wait_timeout))
        )
        self._last_error = str(initial_error) if initial_error is not None else ""
        self._retry_after = (
            self._clock() + self._retry_interval
            if initial_error is not None
            else 0.0
        )

    def _unavailable(self, detail: str | None = None):
        message = detail or self._last_error or "reconnect is already in progress"
        return docker.errors.DockerException(f"Docker is unavailable: {message}")

    def _connect(self):
        with self._condition:
            if self._client is not None:
                return self._client
            if self._clock() < self._retry_after:
                raise self._unavailable()
            if self._connecting:
                # Healthy API negotiation is normally quick. Let concurrent
                # inventory/status callers share that first result rather than
                # reporting a transient outage, but cap the wait so a hung
                # Docker daemon cannot tie up the controller thread pool.
                finished = self._condition.wait_for(
                    lambda: self._client is not None or not self._connecting,
                    timeout=self._connect_wait_timeout,
                )
                if self._client is not None:
                    return self._client
                if finished:
                    raise self._unavailable()
                raise self._unavailable("reconnect is already in progress")
            self._connecting = True
        try:
            try:
                client = docker.from_env()
            except docker.errors.DockerException as exc:
                with self._condition:
                    self._last_error = str(exc)
                    self._retry_after = self._clock() + self._retry_interval
                raise self._unavailable() from exc
            with self._condition:
                self._client = client
                self._last_error = ""
                self._retry_after = 0.0
            return client
        finally:
            with self._condition:
                self._connecting = False
                self._condition.notify_all()

    def __getattr__(self, name: str):
        return getattr(self._connect(), name)


def _append_persisted_error(existing: Any, marker: str) -> str:
    current = str(existing or "").strip()
    return current if marker in current else "; ".join(filter(None, [current, marker]))


def _remove_persisted_error(existing: Any, marker: str) -> str:
    return "; ".join(
        part.strip() for part in str(existing or "").split(";")
        if part.strip() and part.strip() != marker
    )


# The official SGLang image exposes ``python3`` rather than a ``python``
# executable. Keep this separate from the vLLM setting because SGLang recipes
# should not fall back to the configured vLLM image.
DEFAULT_SGLANG_IMAGE = "lmsysorg/sglang:latest"

CONTROLLER_LABEL = "io.sparkdeck.managed"
MODEL_LABEL = "io.sparkdeck.model"
ENGINE_LABEL = "io.sparkdeck.runtime"
DEPLOYMENT_LABEL = "io.sparkdeck.deployment"
NODE_LABEL = "io.sparkdeck.node"
RANK_LABEL = "io.sparkdeck.rank"
SERVICE_PORT_LABEL = "io.sparkdeck.service-port"
MODE_LABEL = "io.sparkdeck.deployment-mode"
NNODES_LABEL = "io.sparkdeck.nnodes"

# Containers created by earlier releases remain discoverable and manageable.
LEGACY_LABELS = {
    CONTROLLER_LABEL: "vllm-controller",
    MODEL_LABEL: "vllm-model",
    ENGINE_LABEL: "vllm-controller.engine",
    DEPLOYMENT_LABEL: "vllm-controller.deployment",
    NODE_LABEL: "vllm-controller.node",
    RANK_LABEL: "vllm-controller.rank",
    SERVICE_PORT_LABEL: "vllm-controller.service-port",
    MODE_LABEL: "vllm-controller.deployment-mode",
    NNODES_LABEL: "vllm-controller.nnodes",
}


def _label_value(labels: dict, key: str, default: Any = None) -> Any:
    value = labels.get(key)
    if value is None:
        value = labels.get(LEGACY_LABELS[key])
    return default if value is None else value

# Distributed workers form one rendezvous generation.  Checking every two
# minutes catches a split rank well before the default 601-second TCPStore
# timeout.  Start times farther apart than one interval indicate that Docker
# restarted only part of the deployment.
CLUSTER_HEALTH_INTERVAL_SECONDS = 120.0
# A first launch may legitimately wait up to the distributed store's 601s
# join timeout while another node finishes pulling its image. Once a restart
# counter baseline exists, any single-rank increment is detected immediately.
CLUSTER_START_SKEW_SECONDS = 660.0
CLUSTER_RECOVERY_ALIGNMENT_SECONDS = CLUSTER_HEALTH_INTERVAL_SECONDS

# vLLM prints its allocated KV capacity after engine initialization. Poll only
# deployments that have not produced a usable capacity report yet; once found,
# the result is persisted with the deployment.
VLLM_CAPACITY_SCAN_INTERVAL_SECONDS = 5.0
VLLM_GPU_KV_CACHE_RE = re.compile(
    r"GPU KV cache size:\s*([\d,]+)\s+tokens",
    re.IGNORECASE,
)
VLLM_MAX_CONCURRENCY_RE = re.compile(
    r"Maximum concurrency for\s+([\d,]+)\s+tokens per request:\s*"
    r"([0-9]+(?:\.[0-9]+)?)x",
    re.IGNORECASE,
)

# FanController consumes a short-lived cluster temperature override. Remote
# node probes are cached for four seconds, so a two-second publisher interval
# keeps the file fresh without increasing agent traffic. If this controller
# stops, FanController ignores the override after the TTL and continues from
# its own local sensors.
FAN_CLUSTER_SYNC_INTERVAL_SECONDS = 2.0
FAN_TEMPERATURE_OVERRIDE_TTL_SECONDS = 12.0
FAN_TEMPERATURE_MAX_SAMPLE_AGE_SECONDS = 15.0

# Remote controller calls must cover each worker-side RouterOS request phase
# plus agent transport/serialization overhead. An overview has three phases
# (resource, gathered details, then traffic); connection setup adds device
# validation; and a fan update wraps its write in two complete overviews.
ROUTEROS_CONTROLLER_TIMEOUT_MARGIN_SECONDS = 10.0
ROUTEROS_OVERVIEW_TIMEOUT_SECONDS = (
    ROUTEROS_TIMEOUT_SECONDS * 3 + ROUTEROS_CONTROLLER_TIMEOUT_MARGIN_SECONDS
)
ROUTEROS_CONNECT_TIMEOUT_SECONDS = (
    ROUTEROS_TIMEOUT_SECONDS * 4 + ROUTEROS_CONTROLLER_TIMEOUT_MARGIN_SECONDS
)
ROUTEROS_FAN_UPDATE_TIMEOUT_SECONDS = (
    ROUTEROS_TIMEOUT_SECONDS * 7 + ROUTEROS_CONTROLLER_TIMEOUT_MARGIN_SECONDS
)

# CPU/GPU chart history.  Thirty-second samples keep the payload small while
# retaining enough detail for a useful two-hour view (including both ends of
# the window, hence the extra point).
TEMPERATURE_HISTORY_INTERVAL_SECONDS = 30.0
TEMPERATURE_HISTORY_WINDOW_SECONDS = 2 * 60 * 60
TEMPERATURE_HISTORY_MAX_SAMPLES = (
    int(TEMPERATURE_HISTORY_WINDOW_SECONDS / TEMPERATURE_HISTORY_INTERVAL_SECONDS) + 1
)

# On-demand benchmark recordings use a finer cadence than the compact header
# history. A run is armed at a target temperature, starts once the hottest
# available CPU/GPU sensor reaches the configured margin above that target,
# and stops automatically after the node cools below the target.
TEMPERATURE_RUN_SAMPLE_INTERVAL_SECONDS = 1.0
# A disconnected node must not leave an armed or recording run alive forever.
# Five consecutive misses tolerate brief agent restarts while bounding the
# amount of time an unusable recording remains active.
TEMPERATURE_RUN_MAX_TELEMETRY_FAILURES = 5

# Safety margin (GB) kept free on the GPU even when running multiple models.
# EarlyOom is the safety net, but this avoids triggering it in the common case.
GPU_VRAM_BUFFER_GB = 10.0

FAN_MODE_DEFAULTS = {
    "curve": {
        "curve_points": [[40.0, 0.0], [60.0, 30.0], [75.0, 60.0], [90.0, 100.0]],
        "curve_min_temp": 30.0,
        "curve_max_temp": 100.0,
        "min_floor_pct": 0.0,
    },
    "pid": {
        "setpoint": 65.0,
        "kp": 4.0,
        "ki": 0.2,
        "kd": 1.0,
        "min_floor_pct": 0.0,
    },
    "hysteresis": {
        "hyst_on_temp": 75.0,
        "hyst_off_temp": 65.0,
    },
    "manual": {"manual_duty_pct": 100.0},
}


class ClientAbort(Exception):
    """Raised when the downstream /v1 client disconnects mid-request, so the
    proxy can unwind without recording usage or returning a response."""


class StreamNudge(Exception):
    """Raised internally to close and replay a zero-output upstream stream."""


class FanSettingsConflict(Exception):
    """Raised when live FanController state cannot safely accept an update."""

# Default per-model llama-server launch settings. Persisted per-model in
# data/unsloth_models.json and edited from the Models tab. Only fields the
# launcher knows how to map to llama-server flags are stored.
UNSLOTH_DEFAULT_SETTINGS = {
    "cache_type_kv": "bf16",
    "max_seq_length": 262144,   # max tokens — very large context window
    "gguf_variant": "Q8_0",
    "parallel": 4,              # llama-server --parallel decode slots
    "kv_unified": False,        # each slot gets its own KV pool (no sharing)
    "load_in_4bit": False,
    "tensor_parallel": False,
    "tensor_parallel_size": 2,
    "split_mode": "tensor",      # llama-server: tensor (experimental), layer, or row
    "trust_remote_code": False,
    # Multi-token prediction uses the model's built-in MTP head.  It only
    # applies to MTP-capable GGUFs (for example Qwen MTP variants).
    "mtp_enabled": False,
    # DSpark uses a separate draft GGUF stored alongside the target model.
    # It is mutually exclusive with the built-in MTP draft mode.
    "dspark_enabled": False,
    "mtp_predict_tokens": 3,
    "speculative_type": "auto",
    "spec_draft_n_max": None,
    # Pricing per 1 million tokens in USD (set 0 to disable).
    "input_cost_per_1m": 0.0,
    "output_cost_per_1m": 0.0,
    # Cache (KV-prefix-hit) input tokens are charged at this rate per 1M.
    # Defaults to 0 (free) — cached tokens are served from the KV cache.
    "cache_cost_per_1m": 0.0,
}

# Known model pricing per 1M tokens (USD). Used when no per-model pricing
# is set — the UI falls back to these built-in rates.
MODEL_PRICING = {
    # OpenAI
    "gpt-4o":              {"input": 2.50, "output": 10.00, "cache": 1.25},
    "gpt-4o-mini":         {"input": 0.15, "output": 0.60, "cache": 0.075},
    "gpt-4-turbo":         {"input": 10.00, "output": 30.00, "cache": 5.00},
    "gpt-4":               {"input": 30.00, "output": 60.00, "cache": 15.00},
    # Google Gemini
    "gemini-2.5-pro":      {"input": 1.25, "output": 15.00, "cache": 0.31},
    "gemini-2.5-flash":    {"input": 0.10, "output": 0.40, "cache": 0.025},
    "gemini-2.0-flash":    {"input": 0.10, "output": 0.40, "cache": 0.025},
    # Anthropic Claude
    "claude-sonnet-4":     {"input": 3.00, "output": 15.00, "cache": 0.30},
    "claude-opus-4":       {"input": 15.00, "output": 75.00, "cache": 1.50},
    "claude-haiku-3-5":    {"input": 0.80, "output": 4.00, "cache": 0.08},
    # Qwen (Alibaba)
    "qwen3.5-max":         {"input": 0.60, "output": 1.80, "cache": 0.15},
    "qwen3-coder-plus":    {"input": 0.30, "output": 0.60, "cache": 0.075},
    # Groq (hosted open weights)
    "llama-3.3-70b":       {"input": 0.59, "output": 0.79, "cache": 0.29},
}

USAGE_SPEED_WINDOW_TOKENS = 1_000_000
USAGE_HOURLY_RETENTION_DAYS = 370

PENDING = "pending"          # waiting for capacity
DISPATCHING = "dispatching"  # model starting, will run when ready
ADMISSION_WAITING = "admission_waiting"  # ready, queued for a proxy slot
RUNNING = "running"          # inference active
DONE = "done"
ERROR = "error"
CANCELED = "canceled"


def _is_vllm_image(tag: str) -> bool:
    return "vllm" in (tag or "").lower()


def _normalize_node_name(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("name must be a string")
    normalized = unicodedata.normalize("NFKC", value).strip()
    if any(unicodedata.category(character).startswith("C") for character in normalized):
        raise ValueError("name must not contain control characters")
    normalized = re.sub(r" {2,}", " ", normalized)
    if not normalized:
        raise ValueError("name must not be empty")
    if len(normalized) > 80:
        raise ValueError("name must be at most 80 characters")
    return normalized


def _is_atlas_serving_container(name: str, image: str) -> bool:
    """Return whether a container is an Atlas Serving run started by SparkRun.

    Atlas Serving uses the ``avarok/atlas-*`` image and SparkRun names its
    transient runner containers ``sparkrun_*``.  Those containers deliberately
    do not carry the controller's labels and their image does not include
    "vllm", so they must be identified before the normal external-vLLM filter.
    """
    return (
        (name or "").startswith("sparkrun_")
        and "atlas" in (image or "").lower()
    )


def _community_pairing_fanout(nodes: list[dict], results: list) -> dict:
    """Merge per-node community pairing fan-out results for reporting."""
    applied: list[str] = []
    conflicts: list[dict] = []
    errors: list[str] = []
    for node, result in zip(nodes, results):
        name = node.get("name", node["id"])
        if isinstance(result, Exception):
            errors.append(f"{name}: {result}")
        elif isinstance(result, dict) and result.get("applied"):
            applied.append(name)
        elif isinstance(result, dict):
            conflicts.append({
                "node": name,
                "email": (result.get("existing") or {}).get("email"),
            })
    return {"applied": applied, "conflicts": conflicts, "errors": errors}


def _community_consent_fanout(nodes: list[dict], results: list) -> dict:
    """Merge per-node consent updates without inventing pairing conflicts."""
    applied: list[str] = []
    errors: list[str] = []
    for node, result in zip(nodes, results):
        name = node.get("name", node["id"])
        if isinstance(result, Exception):
            errors.append(f"{name}: {result}")
        elif isinstance(result, dict) and result.get("applied") is True:
            applied.append(name)
        else:
            errors.append(f"{name}: consent update was not applied")
    return {"applied": applied, "conflicts": [], "errors": errors}


class Manager:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.settings_path = self.data_dir / "settings.json"
        self.settings = self._load_settings()
        self.recipes_path = self.data_dir / "recipes.json"
        self.recipes: list[dict] = self._load_recipes()
        if self._migrate_recipe_hf_credentials():
            self._save_recipes()
        # Ephemeral launch phase for Saved Models.  This is returned with
        # /api/state so an image pull remains visible after a browser refresh.
        self.recipe_launches: dict[str, dict] = {}
        # Ephemeral per-container launch history used by cluster agents. Docker
        # has no container logs while an image is still downloading, so this
        # fills the otherwise silent gap between the launch request and the
        # container being created.
        self.cluster_member_launches: dict[str, dict] = {}
        self.unsloth_settings_path = self.data_dir / "unsloth_models.json"
        self.unsloth_settings: dict[str, dict] = self._load_unsloth_settings()
        # Saved SparkRun targets are references understood by the SparkRun CLI
        # (for example @spark-arena/<uuid>), not copied recipe YAML.
        self.spark_launches_path = self.data_dir / "spark_runs.json"
        self.spark_launches: list[dict] = self._load_spark_launches()
        # Lifetime token counters keyed by model id, persisted across
        # restarts. Only cleared via the reset endpoint.
        self.token_stats_path = self.data_dir / "token_stats.json"
        self.token_stats: dict[str, dict] = self._load_token_stats()
        self.usage_aliases_path = self.data_dir / "usage_aliases.json"
        self.usage_aliases: dict[str, str] = self._load_usage_aliases()
        self.usage_merge_groups_path = self.data_dir / "usage_merge_groups.json"
        self.usage_merge_groups: dict[str, str] = self._load_usage_merge_groups()
        self.usage_routing_rules_path = self.data_dir / "usage_routing_rules.json"
        self.usage_routing_rules: dict[str, str] = self._load_usage_routing_rules()
        self.usage_cache_estimates_path = (
            self.data_dir / "usage_cache_estimates.json"
        )
        self.usage_cache_estimates: dict[str, dict] = (
            self._load_usage_cache_estimates()
        )
        self.speed_samples_path = self.data_dir / "speed_samples.json"
        self.speed_samples: dict[str, list[dict]] = self._load_speed_samples()
        self._speed_samples_version = 0
        self._speed_samples_flush_task: asyncio.Task | None = None
        # Session token counters — same shape as token_stats but NOT
        # persisted. Tracks tokens served since the controller started (or
        # since the last session reset). The topbar widget shows these; the
        # Usage tab shows the lifetime token_stats.
        self.session_token_stats: dict[str, dict] = {}
        # Hourly token counters — same shape as token_stats but keyed by
        # hour timestamp (e.g. "2024-01-15T10"). Persisted to disk so the
        # Usage → Analysis charts survive restarts. Used for the hourly/daily
        # charts and the GitHub-style activity grid.
        self.hourly_token_stats_path = self.data_dir / "hourly_token_stats.json"
        self.hourly_token_stats: dict[str, dict] = self._load_hourly_token_stats()
        # In-flight Spark Run executions (run_id -> run dict). Not persisted.
        self.spark_runs: dict[str, dict] = {}
        # docker.from_env() negotiates the daemon API version synchronously.
        # Keep construction non-blocking so a hung Docker Desktop cannot delay
        # the controller UI from binding and serving its liveness endpoint.
        self.client = _RetryingDockerClient()
        self.jobs: dict[str, dict] = {}
        self.queue: deque[str] = deque()
        self.lock = asyncio.Lock()
        self.worker_task: asyncio.Task | None = None
        self.idle_task: asyncio.Task | None = None
        self.cluster_health_task: asyncio.Task | None = None
        self.deployment_capacity_task: asyncio.Task | None = None
        self.fan_cluster_task: asyncio.Task | None = None
        self.temperature_history_task: asyncio.Task | None = None
        self.temperature_recording_task: asyncio.Task | None = None
        self.inference_nudger_task: asyncio.Task | None = None
        self.token_usage_sync_task: asyncio.Task | None = None
        self._deployment_action_lock = asyncio.Lock()
        # A controller restart can attach to an older deployment whose vLLM
        # capacity line has already rolled beyond the normal short log tail.
        # Permit one bounded deep-history lookup per deployment, then resume
        # lightweight polling while new engines are still initializing.
        self._capacity_deep_scan_attempted: set[str] = set()
        self._capacity_redeploying_models: set[str] = set()
        self.http = httpx.AsyncClient(timeout=600)
        self.agent_credentials = AgentCredentials(self.data_dir)
        self.node_registry = NodeRegistry(
            self.data_dir, self.http, self.agent_credentials.node_id,
            connection_resolver=resolve_agent_connection,
        )
        self.routeros = RouterOSService(self.data_dir)
        self.virtual_nas = VirtualNAS(
            self.data_dir,
            lambda: Path(self.settings.get("hf_cache") or "") / "hub",
            self.node_registry,
            lambda: bool(self.settings.get("virtual_nas_enabled", False)),
        )
        self.token_usage_sync_path = self.data_dir / "token_usage_sync.json"
        self.token_usage_sync = self._load_token_usage_sync()
        self._token_usage_sync_status: dict[str, Any] = {
            "enabled": True,
            "last_sync_at": None,
            "peers": 0,
            "error": None,
        }
        self._rebuild_synced_token_usage()
        self.deployments_path = self.data_dir / "deployments.json"
        self.deployments: list[dict] = self._load_deployments()
        deployments_changed = self._migrate_deployment_hf_credentials()
        deployments_changed = (
            self._migrate_vllm_prompt_token_details() or deployments_changed
        )
        if deployments_changed:
            self._save_deployments()
        self.container_aliases_path = self.data_dir / "container_aliases.json"
        self.container_aliases: dict[str, str] = self._load_container_aliases()
        self._cpu_prev: tuple[int, int] | None = None
        self._stats_cache: dict[str, Any] = {}
        self._stats_ts: float = 0.0
        self._temperature_history: deque[dict[str, float | None]] = deque(
            maxlen=TEMPERATURE_HISTORY_MAX_SAMPLES,
        )
        self._remote_temperature_histories: dict[
            str, deque[dict[str, float | None]]
        ] = {}
        self.temperature_runs_path = self.data_dir / "temperature_runs.json"
        self.temperature_runs: dict[str, dict] = self._load_temperature_runs()
        self._active_temperature_run_id: str | None = None
        self._temperature_recording_lock = asyncio.Lock()
        self._temperature_runs_last_saved_at = 0.0
        interrupted_runs = False
        for run in self.temperature_runs.values():
            if run.get("status") not in {"armed", "recording"}:
                continue
            run["status"] = "interrupted"
            run["stopped_at"] = run.get("last_sample_at") or time.time()
            interrupted_runs = True
        if interrupted_runs:
            self._save_temperature_runs()
        # container_name -> {"last_active": ts, "counter": int}
        self._activity: dict[str, dict] = {}
        # timestamp, {iface: (rx_bytes, tx_bytes)}
        self._net_prev: tuple[float, dict[str, tuple[int, int]]] | None = None
        # Image list is slow on large Docker installs; cache it briefly.
        self._images_cache: list[dict] = []
        self._images_ts: float = 0.0
        # Serializes /v1 proxy swaps so two requests for different models
        # don't thrash the GPU. Concurrent requests for the *same* model
        # serialize on the lock, then short-circuit when they see it ready.
        self._swap_lock = asyncio.Lock()
        # llama-server launcher state. We spawn the server ourselves; the
        # tracked process plus a state file (data/llama_server.json) let a
        # restarted controller re-adopt a still-running server.
        self._llama_proc: asyncio.subprocess.Process | None = None
        self._llama_pid: int | None = None
        self._llama_model: str | None = None
        self._llama_port: int = 0
        self._llama_lock = asyncio.Lock()
        self._llama_state_path = self.data_dir / "llama_server.json"
        self._llama_log_dir = self.data_dir / "logs"
        self._llama_current_log: Path | None = None
        self._llama_current_log_model: str | None = None
        self._llama_log_index_path = self.data_dir / "llama_logs.json"
        self._llama_model_logs: dict[str, str] = self._load_llama_log_index()
        # llama.cpp tensor parallelism uses its RPC backend: rank 0 runs the
        # public llama-server locally and each additional Spark exposes one
        # CUDA device through an authenticated agent-managed RPC worker.
        self._llama_rpc_proc: asyncio.subprocess.Process | None = None
        self._llama_rpc_pid: int | None = None
        self._llama_rpc_host: str | None = None
        self._llama_rpc_port: int = 0
        self._llama_rpc_lock = asyncio.Lock()
        self._llama_rpc_state_path = self.data_dir / "llama_rpc_server.json"
        self._llama_rpc_log_path = self._llama_log_dir / "llama-rpc-server.log"
        self._llama_rpc_adopt_tried = False
        # Remote workers used by the currently launched local llama-server.
        # Persisted in llama_server.json so controller restarts can clean them
        # up when the model is unloaded.
        self._llama_rpc_nodes: list[str] = []
        self._llama_rpc_endpoints: list[str] = []
        # In-flight load_unsloth_model task, so the UI's Cancel-load button
        # can abort a launch that is still waiting for readiness.
        self._llama_load_task: asyncio.Task | None = None
        # In-flight /v1 streams, keyed by request id. Output timestamps are
        # retained in a short rolling window so SSE batching does not make the
        # displayed decode rate jump between zero and a burst value.
        # Feeds the live tok/s + connection count in the Tokens widget.
        self._trailing_window = 5.0  # seconds
        self._active_reqs: dict[int, dict] = {}
        self._req_seq = 0
        # Controller-side vLLM admission control. vLLM's --max-num-seqs is an
        # engine scheduler limit, but forwarding more HTTP requests than that
        # still lets them enter the engine and influence batching/prefill. Keep
        # excess requests in a FIFO here so only the configured number reaches
        # each deployment at once.
        #
        # target id -> {limit, running, model, waiters: deque[waiter]}
        self._inference_admission: dict[str, dict] = {}
        self._nudger_slow_since: dict[str, float] = {}
        # Set while a launch is in flight (model, variant, started_at, log
        # path); surfaced in /api/state so the UI can show load progress.
        self._llama_launching: dict | None = None
        # True only after the server answered /health — loaded_model stays
        # None during the load window so the UI shows "Loading", not "Loaded".
        self._llama_ready: bool = False
        # Memory-bandwidth monitor (Grace/GB10 only). A background `perf stat`
        # reads the SoC's SCF PMU counters every second; the latest read/write
        # bandwidth is cached here. None when unsupported or not yet seen.
        self._mem_bw_thread: threading.Thread | None = None
        self._mem_bw_proc: subprocess.Popen | None = None
        self._mem_bw: dict[str, float] = {}  # {"read_bps":..., "write_bps":..., "ts":...}
        self._mem_bw_lock = threading.Lock()
        self._fan_settings_lock = threading.Lock()
        self._fan_control_lock = threading.Lock()
        self._online_users_cache: dict[str, Any] = {
            "count": None, "names": [], "sessions": 0,
        }
        self._online_users_ts = 0.0

    # ---------- lifecycle ----------
    def is_joined_worker(self) -> bool:
        """Return whether this process has a durable controller assignment."""
        path = self.data_dir / "controller.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return bool(
                value.get("controller_url")
                and value.get("forward_token")
                and value.get("node_id")
            )
        except (OSError, json.JSONDecodeError, AttributeError):
            return False

    def _start_controller_tasks(self) -> None:
        task_factories = (
            ("worker_task", self._worker_loop),
            ("idle_task", self._idle_monitor_loop),
            ("cluster_health_task", self._cluster_health_monitor_loop),
            ("deployment_capacity_task", self._deployment_capacity_monitor_loop),
            ("fan_cluster_task", self._fan_cluster_monitor_loop),
            ("token_usage_sync_task", self._token_usage_sync_loop),
        )
        for field, factory in task_factories:
            current = getattr(self, field, None)
            if current is None or current.done():
                setattr(self, field, asyncio.create_task(factory()))

    async def adopt_worker_role(self) -> None:
        """Stop controller-only schedulers after a successful live join."""
        await self.virtual_nas.stop()
        for field in (
            "worker_task", "idle_task", "cluster_health_task",
            "deployment_capacity_task", "fan_cluster_task",
            "token_usage_sync_task",
        ):
            task = getattr(self, field, None)
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            setattr(self, field, None)

    def adopt_controller_role(self) -> None:
        """Resume controller schedulers after leaving a controller."""
        self._start_controller_tasks()
        self.virtual_nas.start()

    async def start(self):
        routeros = getattr(self, "routeros", None)
        if routeros is not None:
            await routeros.start()
        if not self.is_joined_worker():
            self._start_controller_tasks()
            self.virtual_nas.start()
        self.inference_nudger_task = asyncio.create_task(
            self._inference_nudger_loop()
        )
        self.temperature_history_task = asyncio.create_task(
            self._temperature_history_monitor_loop()
        )
        self._start_mem_bw_monitor()

    async def stop(self):
        routeros = getattr(self, "routeros", None)
        if routeros is not None:
            await routeros.stop()
        await self.virtual_nas.stop()
        for t in (
            self.worker_task,
            self.idle_task,
            self.cluster_health_task,
            self.deployment_capacity_task,
            self.fan_cluster_task,
            self.temperature_history_task,
            self.temperature_recording_task,
            self.inference_nudger_task,
            self.token_usage_sync_task,
        ):
            if t:
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass
        active_run_id = getattr(self, "_active_temperature_run_id", None)
        active_run = getattr(self, "temperature_runs", {}).get(active_run_id)
        if active_run and active_run.get("status") in {"armed", "recording"}:
            active_run["status"] = "interrupted"
            active_run["stopped_at"] = time.time()
            self._save_temperature_runs()
        self._active_temperature_run_id = None
        self.temperature_recording_task = None
        self._stop_mem_bw_monitor()
        await self.http.aclose()

    # ---------- cluster agent / node discovery ----------
    @staticmethod
    def _network_interfaces() -> list[dict]:
        """Return IPv4 interface details without adding a runtime dependency."""
        interfaces = []
        try:
            proc = subprocess.run(
                ["ip", "-j", "-4", "addr", "show"],
                capture_output=True,
                text=True,
                timeout=3,
                check=True,
            )
            raw = json.loads(proc.stdout)
        except Exception:
            raw = []
        for item in raw:
            name = item.get("ifname")
            if not name or name == "lo":
                continue
            addresses = [
                a.get("local") for a in item.get("addr_info", [])
                if a.get("family") == "inet" and a.get("local")
            ]
            if not addresses:
                continue
            rdma = Path(f"/sys/class/net/{name}/device/infiniband").exists()
            interfaces.append({
                "name": name,
                "ipv4": addresses,
                "up": item.get("operstate") == "UP",
                "rdma": rdma,
            })

        # Windows does not provide Linux's ``ip`` utility. Tailscale's own
        # status contract is cross-platform and also covers installations
        # where the tunnel adapter is hidden from ordinary interface tools.
        if not any("tailscale" in str(item.get("name") or "").casefold()
                   for item in interfaces):
            try:
                proc = subprocess.run(
                    ["tailscale", "status", "--json"],
                    capture_output=True,
                    text=True,
                    timeout=3,
                    check=True,
                )
                status = json.loads(proc.stdout)
                addresses = []
                for address in status.get("TailscaleIPs") or []:
                    try:
                        parsed = ipaddress.ip_address(address)
                    except ValueError:
                        continue
                    if parsed.version == 4:
                        addresses.append(str(parsed))
                if addresses:
                    interfaces.append({
                        "name": "tailscale0",
                        "ipv4": list(dict.fromkeys(addresses)),
                        "up": str(status.get("BackendState") or "").casefold() == "running",
                        "rdma": False,
                    })
            except Exception:
                pass
        return interfaces

    @staticmethod
    def _distributed_network_environment(
        interface: str,
        sys_class_net: Path = Path("/sys/class/net"),
    ) -> dict[str, str]:
        """Build one consistent NCCL/Gloo/UCX transport configuration.

        Container images may contain host-specific RDMA defaults. Always
        replace them using the interface selected for this cluster member so
        socket bootstrap and collective traffic cannot target different ports.

        Images (e.g. the Aiden sparkrun builds) also bake in a hard-coded
        NCCL_IB_GID_INDEX that only matches the build host's GID table. With
        NCCL 2.21+ RoCE GID selection is dynamic, so a stale index (which can
        point at an empty slot after reboot/link flaps) must be *unset*, not
        renumbered: ``None`` lets docker-py emit a bare ``KEY`` entry, which
        the Docker API interprets as "remove from the image environment".
        """
        environment = {
            "NCCL_SOCKET_IFNAME": interface,
            "GLOO_SOCKET_IFNAME": interface,
            # Unset stale image-baked GID index; NCCL >= 2.21 selects the
            # IPv4 RoCEv2 GID dynamically.
            "NCCL_IB_GID_INDEX": None,
        }
        infiniband_dir = sys_class_net / interface / "device" / "infiniband"
        try:
            hcas = sorted(path.name for path in infiniband_dir.iterdir())
        except OSError:
            hcas = []
        if hcas:
            environment.update({
                "NCCL_NET": "IB",
                "NCCL_IB_DISABLE": "0",
                "NCCL_IB_HCA": ",".join(hcas),
                "UCX_NET_DEVICES": ",".join(f"{hca}:1" for hca in hcas),
            })
        else:
            # Override stale RDMA defaults inherited from an image. Socket is
            # slower but remains a valid, predictable fallback.
            environment.update({
                "NCCL_NET": "Socket",
                "NCCL_IB_DISABLE": "1",
                "NCCL_IB_HCA": "",
                "UCX_NET_DEVICES": "",
            })
        return environment

    async def agent_status(self, stats: dict | None = None) -> dict:
        if stats is None:
            stats = await self.get_stats()
        disk = await self.get_disk()
        docker_ready, docker_status_message = await self._docker_runtime_status()
        try:
            containers = await self.list_containers()
            containers = [
                {
                    "name": c.get("name"),
                    "model": c.get("model"),
                    "status": c.get("status"),
                    "port": c.get("port"),
                    "deployment_id": c.get("deployment_id"),
                    "rank": c.get("rank"),
                    "started_at": c.get("started_at"),
                    "restart_count": c.get("restart_count", 0),
                    "phase": c.get("phase"),
                }
                for c in containers if c.get("managed")
            ]
        except Exception:
            containers = []
        existing_names = {c.get("name") for c in containers}
        # Launch updates can arrive from a Docker worker thread while the
        # status endpoint is being serialized.
        for name, launch in list(self.cluster_member_launches.items()):
            if name in existing_names:
                continue
            phase = launch.get("phase") or "queued"
            containers.append({
                "name": name,
                "model": launch.get("model"),
                "status": "error" if phase == "error" else "creating",
                "port": None,
                "deployment_id": launch.get("deployment_id"),
                "rank": launch.get("rank"),
                "phase": {
                    "phase": phase,
                    "message": launch.get("message"),
                    "updated_at": launch.get("updated_at"),
                },
            })
        interfaces = self._network_interfaces()
        configured_iface = self.settings.get("cluster_fabric_interface") or ""
        fabric_ready = any(
            i.get("up") and i.get("rdma") and
            (not configured_iface or i.get("name") == configured_iface)
            for i in interfaces
        )
        return {
            "node_id": self.agent_credentials.node_id,
            "name": self.settings.get("cluster_node_name") or socket.gethostname(),
            "hostname": socket.gethostname(),
            "protocol_version": AGENT_PROTOCOL_VERSION,
            "capabilities": [CAPABILITY],
            "update_protocol": 1,
            "app_revision": current_revision(Path(__file__).parent),
            "status": "online" if docker_ready else "degraded",
            "online": True,
            "status_message": docker_status_message,
            "docker_ready": docker_ready,
            "fabric_ready": fabric_ready,
            "interfaces": interfaces,
            "routeros": self.routeros.presence(),
            "stats": stats,
            "disk": disk,
            "containers": containers,
            "llama_rpc": self.llama_rpc_status(),
        }

    async def _docker_runtime_status(self) -> tuple[bool, str | None]:
        """Return whether Docker can run SparkDeck's Linux GPU workloads."""

        def probe() -> tuple[bool, str | None]:
            if not self.client.ping():
                return False, "Docker is unavailable"
            daemon_info = self.client.info()
            os_type = str((daemon_info or {}).get("OSType") or "").strip().lower()
            if os_type != "linux":
                return False, "Docker must be configured to use Linux containers"
            return True, None

        try:
            return await asyncio.to_thread(probe)
        except Exception:
            return False, "Docker is unavailable"

    def _cluster_launch_update(
        self,
        name: str | None,
        phase: str,
        message: str,
        *,
        model: str | None = None,
        cluster_member: dict | None = None,
        error: str | None = None,
    ) -> None:
        """Record controller-side progress before Docker logs are available."""
        if not name or not cluster_member:
            return
        now = time.time()
        launches = getattr(self, "cluster_member_launches", None)
        if launches is None:
            launches = self.cluster_member_launches = {}
        launch = launches.setdefault(name, {
            "name": name,
            "model": model,
            "deployment_id": cluster_member.get("deployment_id"),
            "node_id": cluster_member.get("node_id"),
            "rank": cluster_member.get("rank"),
            "created_at": now,
            "events": [],
        })
        launch.update({
            "phase": phase,
            "message": message,
            "updated_at": now,
            "error": error,
        })
        events = launch.setdefault("events", [])
        if not events or events[-1].get("message") != message:
            events.append({"at": now, "phase": phase, "message": message})
            del events[:-100]

    def _cluster_launch_text(self, name: str) -> str:
        launch = getattr(self, "cluster_member_launches", {}).get(name)
        if not launch:
            return ""
        lines = ["=== Controller launch progress ==="]
        for event in launch.get("events", []):
            stamp = datetime.fromtimestamp(event.get("at", time.time())).strftime("%H:%M:%S")
            lines.append(f"[{stamp}] {event.get('message') or event.get('phase') or 'Working'}")
        return "\n".join(lines)

    @staticmethod
    def _inferred_fabric(status: dict, configured_ip: str | None = None,
                         configured_interface: str | None = None) -> tuple[str | None, str | None]:
        if configured_ip:
            return configured_ip, configured_interface
        interfaces = status.get("interfaces") or []
        candidates = [i for i in interfaces if i.get("up") and i.get("rdma")]
        if configured_interface:
            candidates = [i for i in candidates if i.get("name") == configured_interface]
        if not candidates:
            candidates = [i for i in interfaces if i.get("up")]
        if not candidates:
            return None, configured_interface
        selected = candidates[0]
        addresses = selected.get("ipv4") or []
        return (addresses[0] if addresses else None), selected.get("name")

    async def cluster_nodes(self, local_stats: dict | None = None) -> list[dict]:
        local = await self.agent_status(local_stats)
        local["fabric_ip"], local["fabric_interface"] = self._inferred_fabric(
            local,
            self.settings.get("cluster_fabric_ip"),
            self.settings.get("cluster_fabric_interface"),
        )
        nodes = await self.node_registry.public_nodes(local)

        async def add_legacy_disk(node: dict) -> None:
            # Agents from before disk telemetry was added still expose the
            # regular /api/disk endpoint. This compatibility request makes the
            # selector useful before every node has pulled the latest build.
            if node.get("local") or not node.get("online") or node.get("disk"):
                return
            try:
                node["disk"] = await self.node_registry.request(
                    node["id"], "GET", "/api/disk", timeout=3
                )
            except Exception:
                pass

        await asyncio.gather(*(add_legacy_disk(node) for node in nodes))
        for node in nodes:
            if node.get("local") or not node.get("online"):
                continue
            self._record_remote_temperature_sample(node)
        return nodes

    @staticmethod
    def public_target_node(node: dict) -> dict:
        """Return the stable, credential-free node selector contract."""
        return {
            key: node.get(key)
            for key in (
                "id", "name", "local", "enabled", "status", "online",
                "last_seen", "protocol_version", "docker_ready", "fabric_ready",
                "stats", "disk",
                "routeros",
            )
        } | {
            "selectable": bool(
                node.get("enabled", True)
                and node.get("online")
                and node.get("docker_ready")
            )
        }

    async def selected_cluster_nodes(self, node_ids: list[str] | None = None) -> list[dict]:
        """Resolve and validate an ordered target set, defaulting to this node."""
        raw = node_ids or [LOCAL_NODE_ID]
        if not raw or any(not isinstance(value, str) or not value.strip() for value in raw):
            raise ValueError("node_ids must contain non-empty node IDs")
        requested = list(dict.fromkeys(value.strip() for value in raw))
        available = {node["id"]: node for node in await self.cluster_nodes()}
        missing = [node_id for node_id in requested if node_id not in available]
        if missing:
            raise ValueError(f"unknown cluster node(s): {', '.join(missing)}")
        offline = [
            node_id for node_id in requested
            if not available[node_id].get("enabled", True)
            or not available[node_id].get("online")
        ]
        if offline:
            names = [available[node_id].get("name", node_id) for node_id in offline]
            raise ValueError(f"cluster node(s) are offline: {', '.join(names)}")
        docker_unready = [
            node_id for node_id in requested
            if not available[node_id].get("docker_ready")
        ]
        if docker_unready:
            names = [available[node_id].get("name", node_id) for node_id in docker_unready]
            raise ValueError(f"Docker is unavailable on: {', '.join(names)}")
        return [available[node_id] for node_id in requested]

    # ---------- RouterOS switch management ----------
    async def routeros_cluster_presence(self) -> dict:
        nodes = await self.cluster_nodes()
        summaries = []
        for node in nodes:
            state = node.get("routeros") if isinstance(node.get("routeros"), dict) else {}
            # Presence is intentionally local and lightweight. Connectivity is
            # unknown until routeros_cluster_overview authenticates to the switch.
            summaries.append({
                "node_id": node.get("id"),
                "node_name": node.get("name"),
                "online": bool(node.get("online")),
                "detected": bool(state.get("detected")),
                "configured": bool(state.get("configured")),
            })
        return {
            "detected": any(item["detected"] for item in summaries),
            "nodes": summaries,
        }

    async def routeros_cluster_overview(self) -> dict:
        nodes = await self.cluster_nodes()

        async def load(node: dict) -> dict:
            node_id = str(node.get("id") or "")
            node_name = str(node.get("name") or node_id)
            presence = node.get("routeros") if isinstance(node.get("routeros"), dict) else {}
            if not node.get("online"):
                return {
                    **presence,
                    "node_id": node_id,
                    "node_name": node_name,
                    "connected": False,
                    "error": node.get("status_message") or "SparkDeck node is offline",
                    "health": [],
                    "interfaces": [],
                }
            if not presence.get("detected") and not presence.get("configured"):
                return {
                    **presence,
                    "node_id": node_id,
                    "node_name": node_name,
                    "connected": False,
                    "health": [],
                    "interfaces": [],
                }
            try:
                if node_id == LOCAL_NODE_ID:
                    result = await self.routeros.overview()
                else:
                    result = await self.node_registry.request(
                        node_id, "GET", "/api/agent/routeros",
                        timeout=ROUTEROS_OVERVIEW_TIMEOUT_SECONDS,
                    )
                return {**result, "node_id": node_id, "node_name": node_name}
            except Exception as exc:
                return {
                    **presence,
                    "node_id": node_id,
                    "node_name": node_name,
                    "connected": False,
                    "error": str(exc),
                    "health": [],
                    "interfaces": [],
                }

        overviews = await asyncio.gather(*(load(node) for node in nodes))
        return {
            "detected": any(bool(item.get("detected")) for item in overviews),
            "nodes": overviews,
        }

    async def _routeros_target(self, node_id: str) -> dict:
        normalized = str(node_id or "").strip()
        available = {node["id"]: node for node in await self.cluster_nodes()}
        node = available.get(normalized)
        if not node:
            raise ValueError("cluster node not found")
        if not node.get("online"):
            raise RuntimeError(f"{node.get('name', normalized)} is offline")
        return node

    async def connect_routeros(self, node_id: str, body: dict) -> dict:
        node = await self._routeros_target(node_id)
        if node["id"] == LOCAL_NODE_ID:
            return await self.routeros.connect(body)
        result = await self.node_registry.request(
            node["id"], "PUT", "/api/agent/routeros/connection",
            json_body=body, timeout=ROUTEROS_CONNECT_TIMEOUT_SECONDS,
        )
        # _routeros_target() just populated the four-second agent-status cache.
        # Drop that pre-connection snapshot so the immediate UI reload probes
        # the worker and observes its newly configured RouterOS state.
        self.node_registry._status_cache.pop(node["id"], None)
        return result

    async def disconnect_routeros(self, node_id: str) -> dict:
        node = await self._routeros_target(node_id)
        if node["id"] == LOCAL_NODE_ID:
            return self.routeros.disconnect()
        return await self.node_registry.request(
            node["id"], "DELETE", "/api/agent/routeros/connection", timeout=10,
        )

    async def update_routeros_fan_settings(self, node_id: str, body: dict) -> dict:
        node = await self._routeros_target(node_id)
        if node["id"] == LOCAL_NODE_ID:
            return await self.routeros.update_fan_settings(body)
        return await self.node_registry.request(
            node["id"], "PATCH", "/api/agent/routeros/fan-settings",
            json_body=body, timeout=ROUTEROS_FAN_UPDATE_TIMEOUT_SECONDS,
        )

    async def rename_cluster_node(self, node_id: str, name: Any) -> dict:
        """Durably rename a local or paired node without exposing credentials.

        The controller registry is authoritative for remote display names. An
        online worker is also updated through its authenticated agent endpoint;
        if that second write fails, the controller rename remains durable and
        the response explicitly reports that worker synchronization is pending.
        """
        normalized = _normalize_node_name(name)
        node_id = str(node_id or "").strip()
        if node_id == LOCAL_NODE_ID:
            async with self.lock:
                self.settings["cluster_node_name"] = normalized
                self._save_settings()
            return {
                **self.public_target_node({
                    "id": LOCAL_NODE_ID, "name": normalized, "local": True,
                    "enabled": True, "status": "online", "online": True,
                    "docker_ready": True,
                }),
                "name_sync": "local",
            }

        if not self.node_registry.get(node_id):
            raise LookupError("node not found")
        updated = self.node_registry.update(node_id, {"name": normalized})
        sync = "synchronized"
        try:
            await self.node_registry.request(
                node_id, "PATCH", "/api/agent/node",
                json_body={"name": normalized}, timeout=10,
            )
        except Exception:
            # The controller-side name is already durable. Offline workers can
            # be renamed safely without rolling that write back; their local UI
            # keeps its prior name until an authenticated synchronization works.
            sync = "pending"
        self.node_registry._status_cache.pop(node_id, None)
        return {
            **self.public_target_node({
                **updated, "name": normalized,
                "status": "online" if sync == "synchronized" else "unreachable",
                "online": sync == "synchronized",
            }),
            "name_sync": sync,
        }

    def virtual_nas_enabled(self) -> bool:
        return bool(self.settings.get("virtual_nas_enabled", False))

    async def virtual_nas_inventory(self) -> dict:
        instructions = [
            "Enable Virtual NAS to copy complete Hugging Face model caches between paired nodes.",
            "Transfers are serialized and remain local to your authenticated SparkDeck cluster.",
        ]
        if not self.virtual_nas_enabled():
            return {
                "enabled": False, "nodes": [], "jobs": [],
                "instructions": instructions,
            }
        nodes = await self.model_cache_inventory()
        return {
            "enabled": True, "nodes": nodes,
            "jobs": self.virtual_nas_transfers()["items"],
            "instructions": instructions,
        }

    async def model_cache_inventory(self) -> list[dict]:
        """Return safe Hugging Face cache sizes even when transfers are off.

        The Models page needs read-only disk accounting independently of the
        Virtual NAS transfer feature.  Inventory responses contain model IDs
        and byte counts only; filesystem paths remain agent-private.
        """
        cluster_nodes = await self.cluster_nodes()

        async def inventory_for(node: dict) -> dict:
            models: list[dict] = []
            online = bool(node.get("online"))
            if online:
                try:
                    if node.get("id") == LOCAL_NODE_ID:
                        models = await asyncio.to_thread(self.virtual_nas.inventory)
                    else:
                        payload = await self.node_registry.request(
                            node["id"], "GET", "/api/agent/virtual-nas/inventory",
                            timeout=30,
                        )
                        models = list((payload or {}).get("models") or [])
                except Exception:
                    online = False
            return {
                "id": node.get("id"), "name": node.get("name"),
                "online": online,
                "total_size": max(0, int(
                    (node.get("disk") or {}).get("total")
                    or (node.get("disk") or {}).get("total_bytes")
                    or 0
                )),
                "free_size": max(0, int(
                    (node.get("disk") or {}).get("free")
                    or (node.get("disk") or {}).get("free_bytes")
                    or 0
                )),
                "models": models,
            }

        nodes = await asyncio.gather(*(inventory_for(node) for node in cluster_nodes))
        return list(nodes)

    async def queue_virtual_nas_transfer(
        self, model_id: str, source_node_id: str,
        target_node_ids: list[str],
    ) -> dict:
        result = await self.virtual_nas.queue_transfer(
            model_id, source_node_id, target_node_ids,
        )
        jobs = [self._public_virtual_nas_job(job) for job in result["jobs"]]
        return {"job_ids": result["job_ids"], "jobs": jobs}

    def virtual_nas_transfers(self) -> dict:
        return {
            "items": [
                self._public_virtual_nas_job(job)
                for job in self.virtual_nas.list_transfers()["items"]
            ]
        }

    async def cancel_virtual_nas_transfer(self, job_id: str) -> dict:
        return self._public_virtual_nas_job(
            await self.virtual_nas.cancel_transfer(job_id)
        )

    def _public_virtual_nas_job(self, job: dict) -> dict:
        def node_name(node_id: str) -> str:
            if node_id == LOCAL_NODE_ID:
                return str(self.settings.get("cluster_node_name") or "This node")
            node = self.node_registry.get(node_id) or {}
            return str(node.get("name") or node_id)

        total = max(0, int(job.get("bytes_total") or 0))
        transferred = max(0, int(job.get("bytes_transferred") or 0))
        progress = (
            min(1.0, transferred / total) if total
            else (1.0 if job.get("status") == "completed" else 0.0)
        )
        return {
            **job,
            "source_node_name": node_name(job["source_node_id"]),
            "target_node_name": node_name(job["target_node_id"]),
            "progress": progress,
            "finished_at": job.get("completed_at"),
        }

    async def delete_virtual_nas_model(self, node_id: str, model_id: str) -> dict:
        model_id = validate_model_id(model_id)
        node_id = str(node_id or "")
        if self.virtual_nas.model_in_transfer(model_id, node_id):
            raise RuntimeError("model is in use by a virtual NAS transfer")
        if node_id != LOCAL_NODE_ID:
            node = self.node_registry.get(node_id)
            if not node:
                raise ValueError(f"unknown node_id '{node_id}'")
            status = await self.node_registry.probe(node, force=True)
            if not status.get("online"):
                raise RuntimeError(f"node '{node.get('name', node_id)}' is offline")
            result = await self.node_registry.request(
                node_id, "DELETE",
                f"/api/agent/virtual-nas/models/{quote(model_id, safe='')}",
                timeout=600,
            )
            return {**(result or {}), "node_id": node_id, "model_id": model_id}
        if await self._local_model_uses_cache(model_id):
            raise RuntimeError("model is in use by a local deployment")
        return {
            **self.virtual_nas.delete_model(model_id),
            "node_id": LOCAL_NODE_ID,
        }

    async def _local_model_uses_cache(self, model_id: str) -> bool:
        active_container_states = {"created", "restarting", "running", "paused"}
        try:
            for container in await self.list_containers():
                if str(container.get("status") or "").lower() not in active_container_states:
                    continue
                ids = set(self._container_model_ids(container))
                ids.add(str(container.get("model") or ""))
                if model_id in ids:
                    return True
        except Exception:
            # A Docker outage is not evidence that a model is unused. Fail
            # closed so cache deletion cannot break an opaque running process.
            raise RuntimeError("cannot verify whether the model is in use")
        if await self._unsloth_loaded_model() == model_id:
            return True
        for deployment in self.deployments:
            if deployment.get("model") != model_id:
                continue
            if deployment.get("status") in {"stopped", "error", "removed"}:
                continue
            members = deployment.get("members") or []
            if not members or any(member.get("node_id") == LOCAL_NODE_ID for member in members):
                return True
        return False

    async def pair_node(self, body: dict) -> dict:
        return await self.node_registry.pair_remote(
            body.get("agent_url") or "",
            body.get("pairing_code") or "",
            name=body.get("name"),
            fabric_ip=body.get("fabric_ip"),
            fabric_interface=body.get("fabric_interface"),
        )

    async def sync_token_usage_once(self) -> dict:
        """Pull per-origin counters from peers, merge, then fan out the union."""
        status = getattr(self, "_token_usage_sync_status", {})
        status.update({"enabled": True, "error": None})
        self._token_usage_sync_status = status

        nodes = [
            node for node in self.node_registry.nodes
            if node.get("enabled", True)
        ]
        pulls = await asyncio.gather(*(
            self.node_registry.request(
                node["id"], "GET", "/api/agent/token-usage", timeout=5
            )
            for node in nodes
        ), return_exceptions=True)
        participating: list[dict] = []
        reconciling: set[str] = set()
        errors: list[str] = []
        for node, result in zip(nodes, pulls):
            if isinstance(result, Exception):
                errors.append(f"{node.get('name', node['id'])}: {result}")
                continue
            if not isinstance(result, dict):
                continue
            try:
                if node.get("usage_reconciled") is True:
                    self.merge_token_usage_sync(result)
                else:
                    self.reconcile_token_usage_sync(result, node["id"])
                    reconciling.add(node["id"])
                participating.append(node)
            except (OSError, ValueError) as exc:
                errors.append(f"{node.get('name', node['id'])}: {exc}")

        snapshot = self.token_usage_sync_snapshot()
        pushes = await asyncio.gather(*(
            self.node_registry.request(
                node["id"], "POST", "/api/agent/token-usage",
                json_body={
                    **snapshot,
                    **(
                        {"reconcile_origin": node["id"]}
                        if node["id"] in reconciling else {}
                    ),
                },
                timeout=5,
            )
            for node in participating
        ), return_exceptions=True)
        for node, result in zip(participating, pushes):
            if isinstance(result, Exception):
                errors.append(f"{node.get('name', node['id'])}: {result}")
            elif node["id"] in reconciling:
                self.node_registry.mark_usage_reconciled(node["id"])

        status.update({
            "last_sync_at": time.time(),
            "peers": len(participating),
            "error": "; ".join(errors) if errors else None,
        })
        return dict(status)

    async def push_community_pairing(self, sub: str, email: str | None) -> dict:
        """Best-effort fan-out of a community sign-in to every enabled peer."""
        nodes = [
            node for node in self.node_registry.nodes
            if node.get("enabled", True)
        ]
        results = await asyncio.gather(*(
            self.node_registry.request(
                node["id"], "PUT", "/api/agent/community-pairing",
                json_body={"sub": sub, "email": email},
                timeout=5,
            )
            for node in nodes
        ), return_exceptions=True)
        return _community_pairing_fanout(nodes, results)

    async def push_community_unpair(self, sub: str) -> dict:
        """Best-effort fan-out of a community sign-out to every enabled peer."""
        nodes = [
            node for node in self.node_registry.nodes
            if node.get("enabled", True)
        ]
        results = await asyncio.gather(*(
            self.node_registry.request(
                node["id"], "DELETE", "/api/agent/community-pairing",
                json_body={"sub": sub},
                timeout=5,
            )
            for node in nodes
        ), return_exceptions=True)
        return _community_pairing_fanout(nodes, results)

    async def push_community_consent(self, enabled: bool) -> dict:
        """Best-effort fan-out of consent to every joined peer.

        Disabled nodes still own their local upload workers, so privacy state
        must reach them even though normal workload operations skip them.
        """
        nodes = list(self.node_registry.nodes)
        results = await asyncio.gather(*(
            self.node_registry.request(
                node["id"], "PUT", "/api/agent/community-consent",
                json_body={"enabled": enabled},
                timeout=20,
                allow_disabled=True,
            )
            for node in nodes
        ), return_exceptions=True)
        return _community_consent_fanout(nodes, results)

    async def _token_usage_sync_loop(self) -> None:
        while True:
            try:
                await self.sync_token_usage_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("token usage synchronization failed")
                self._token_usage_sync_status.update({
                    "enabled": True,
                    "last_sync_at": time.time(),
                    "error": str(exc),
                })
            await asyncio.sleep(5.0)

    async def refresh_node(self, node_id: str) -> dict:
        node = self.node_registry.get(node_id)
        if not node or node_id == LOCAL_NODE_ID:
            local = await self.agent_status()
            local["id"] = LOCAL_NODE_ID
            local["status"] = "online"
            local["online"] = True
            return local
        return await self.node_registry.probe(node, force=True)

    def remove_cluster_node(self, node_id: str) -> bool:
        self._assert_cluster_node_removable(node_id)
        return self.node_registry.remove(node_id)

    def _assert_cluster_node_removable(self, node_id: str) -> None:
        for deployment in self.deployments:
            member_node_ids = {
                member.get("node_id")
                for member in deployment.get("members", [])
            }
            if node_id in set(deployment.get("node_ids", [])) | member_node_ids:
                name = deployment.get("name") or deployment.get("id") or "deployment"
                raise ValueError(
                    f"node is still used by {name}; remove that deployment first"
                )

    async def detach_cluster_node(self, node_id: str, *, force: bool = False) -> bool:
        """Disconnect one worker before forgetting its controller-side record.

        A force removal is intentionally controller-local recovery for an
        unreachable node. The worker may retain a stale assignment until its
        operator uses Leave cluster or pairs it again.
        """
        if node_id == LOCAL_NODE_ID:
            raise ValueError("the controller node cannot be removed")
        node = self.node_registry.get(node_id)
        if not node:
            return False
        self._assert_cluster_node_removable(node_id)
        if not force:
            try:
                await self.node_registry.request(
                    node_id, "POST", "/api/agent/onboarding/detach", timeout=10,
                )
            except RuntimeError as exc:
                cause = exc.__cause__
                protocol_version = node.get("protocol_version")
                if not (
                    protocol_version == AGENT_PROTOCOL_VERSION
                    and isinstance(cause, httpx.HTTPStatusError)
                    and cause.response.status_code == 404
                ):
                    raise
                # Protocol v1 predates authenticated detach but remains the
                # advertised compatible version. Preserve the legacy removal
                # behavior when that one capability is absent; a later local
                # Leave cluster clears the worker's stale assignment.
        return self.node_registry.remove(node_id)

    # ---------- clustered deployments ----------
    def _load_deployments(self) -> list[dict]:
        if not self.deployments_path.exists():
            return []
        try:
            value = json.loads(self.deployments_path.read_text(encoding="utf-8"))
            if not isinstance(value, list):
                return []
            for deployment in value:
                engine = str(deployment.get("engine") or "vllm")
                if engine not in {"vllm", "sglang"}:
                    deployment["status"] = "error"
                    deployment["error"] = f"unsupported persisted runtime: {engine}"
            return value
        except Exception:
            return []

    def _save_deployments(self) -> None:
        _atomic_private_json_write(self.deployments_path, self.deployments)

    @staticmethod
    def _with_vllm_prompt_token_details(args: list[Any]) -> list[Any]:
        """Enable cached-token details unless the user explicitly chose."""
        result = list(args)
        names = {str(arg).split("=", 1)[0] for arg in result}
        if not names.intersection({
            "--enable-prompt-tokens-details",
            "--no-enable-prompt-tokens-details",
        }):
            result.append("--enable-prompt-tokens-details")
        return result

    def _migrate_vllm_prompt_token_details(self) -> bool:
        """Mark saved vLLM deployments for a reporting-capable rebuild."""
        changed = False
        for deployment in self.deployments:
            if (
                deployment.get("status") == "error"
                and PERSISTED_DEPLOYMENT_ARGS_ERROR
                in str(deployment.get("error") or "")
            ):
                continue
            settings = deployment.get("launch_settings")
            if not isinstance(settings, dict):
                continue
            if (settings.get("engine") or deployment.get("engine") or "vllm") != "vllm":
                continue
            original = list(settings.get("extra_args") or [])
            updated = self._with_vllm_prompt_token_details(original)
            if updated == original:
                continue
            settings["extra_args"] = updated
            deployment["settings_dirty"] = True
            changed = True
        return changed

    def _migrate_deployment_hf_credentials(self) -> bool:
        """Discard legacy CLI credentials from durable deployment records."""
        changed = False
        for deployment in self.deployments:
            if (
                PERSISTED_DEPLOYMENT_ARGS_ERROR
                in str(deployment.get("error") or "")
                and deployment.get("launch_settings_error")
                != PERSISTED_DEPLOYMENT_ARGS_ERROR
            ):
                deployment["launch_settings_error"] = (
                    PERSISTED_DEPLOYMENT_ARGS_ERROR
                )
                changed = True
            settings = deployment.get("launch_settings")
            if not isinstance(settings, dict):
                continue
            raw_args = settings.get("extra_args")
            if raw_args is None:
                original = []
            elif not isinstance(raw_args, list) or any(
                not isinstance(value, str) for value in raw_args
            ):
                settings["extra_args"] = []
                deployment["status"] = "error"
                deployment["error"] = _append_persisted_error(
                    deployment.get("error"), PERSISTED_DEPLOYMENT_ARGS_ERROR,
                )
                deployment["launch_settings_error"] = (
                    PERSISTED_DEPLOYMENT_ARGS_ERROR
                )
                deployment["settings_dirty"] = True
                changed = True
                continue
            else:
                original = list(raw_args)
            sanitized = self._without_hf_cli_credentials(original)
            if sanitized != original:
                settings["extra_args"] = sanitized
                deployment["settings_dirty"] = True
                changed = True
        return changed

    def _load_container_aliases(self) -> dict[str, str]:
        if not self.container_aliases_path.exists():
            return {}
        try:
            value = json.loads(self.container_aliases_path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                return {}
            return {
                str(name): str(alias)
                for name, alias in value.items()
                if str(name).strip() and str(alias).strip()
            }
        except Exception:
            return {}

    def _save_container_aliases(self) -> None:
        tmp = self.container_aliases_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.container_aliases, indent=2), encoding="utf-8")
        tmp.replace(self.container_aliases_path)

    def _load_usage_aliases(self) -> dict[str, str]:
        if not self.usage_aliases_path.exists():
            return {}
        try:
            value = json.loads(self.usage_aliases_path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                return {}
            return {
                str(model): str(alias)
                for model, alias in value.items()
                if str(model).strip() and str(alias).strip()
            }
        except Exception:
            return {}

    def _save_usage_aliases(self) -> None:
        tmp = self.usage_aliases_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.usage_aliases, indent=2), encoding="utf-8")
        tmp.replace(self.usage_aliases_path)

    def _load_usage_merge_groups(self) -> dict[str, str]:
        if not self.usage_merge_groups_path.exists():
            return {}
        try:
            value = json.loads(
                self.usage_merge_groups_path.read_text(encoding="utf-8")
            )
            if not isinstance(value, dict):
                return {}
            return {
                str(model): str(group)
                for model, group in value.items()
                if str(model).strip() and str(group).strip()
            }
        except Exception:
            return {}

    def _save_usage_merge_groups(self) -> None:
        tmp = self.usage_merge_groups_path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(self.usage_merge_groups, indent=2), encoding="utf-8"
        )
        tmp.replace(self.usage_merge_groups_path)

    def _load_usage_routing_rules(self) -> dict[str, str]:
        if not self.usage_routing_rules_path.exists():
            return {}
        try:
            value = json.loads(
                self.usage_routing_rules_path.read_text(encoding="utf-8")
            )
            if not isinstance(value, dict):
                return {}
            return {
                str(source).strip(): str(destination).strip()
                for source, destination in value.items()
                if str(source).strip() and str(destination).strip()
            }
        except Exception:
            return {}

    def _save_usage_routing_rules(self) -> None:
        tmp = self.usage_routing_rules_path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(self.usage_routing_rules, indent=2), encoding="utf-8"
        )
        tmp.replace(self.usage_routing_rules_path)

    def _load_usage_cache_estimates(self) -> dict[str, dict]:
        if not self.usage_cache_estimates_path.exists():
            return {}
        try:
            value = json.loads(
                self.usage_cache_estimates_path.read_text(encoding="utf-8")
            )
        except Exception:
            return {}
        if not isinstance(value, dict):
            return {}
        estimates: dict[str, dict] = {}
        for row_key, raw in value.items():
            if not isinstance(raw, dict):
                continue
            try:
                rate_pct = min(100.0, max(0.0, float(raw["rate_pct"])))
                legacy_input = max(0, int(raw["legacy_input"]))
                measured_cached = min(
                    legacy_input, max(0, int(raw.get("measured_cached") or 0))
                )
                estimated_cached = min(
                    legacy_input - measured_cached,
                    max(0, int(raw.get("estimated_cached") or 0)),
                )
            except (KeyError, TypeError, ValueError):
                continue
            estimates[str(row_key)] = {
                "rate_pct": rate_pct,
                "legacy_input": legacy_input,
                "measured_cached": measured_cached,
                "estimated_cached": estimated_cached,
            }
        return estimates

    def _save_usage_cache_estimates(self) -> None:
        tmp = self.usage_cache_estimates_path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(self.usage_cache_estimates, indent=2), encoding="utf-8"
        )
        tmp.replace(self.usage_cache_estimates_path)

    @staticmethod
    def _usage_route_target(model: str, rules: dict[str, str]) -> str:
        """Resolve a model through a validated, possibly chained rule map."""
        target = model
        seen: set[str] = set()
        while target in rules and target not in seen:
            seen.add(target)
            target = rules[target]
        return target

    def update_usage_routing_rule(self, source: Any, destination: Any) -> dict:
        source_key = str(source or "").strip()
        destination_key = str(destination or "").strip()
        if not source_key or not destination_key:
            raise ValueError("source and destination are required")
        if len(source_key) > 512 or len(destination_key) > 512:
            raise ValueError("model names must be 512 characters or fewer")
        if source_key == destination_key:
            raise ValueError("source and destination must be different")

        candidate = dict(getattr(self, "usage_routing_rules", {}))
        candidate[source_key] = destination_key
        for model in candidate:
            current = model
            seen: set[str] = set()
            while current in candidate:
                if current in seen:
                    raise ValueError("routing rules cannot contain a cycle")
                seen.add(current)
                current = candidate[current]

        self.usage_routing_rules = candidate
        self._save_usage_routing_rules()
        return {
            "ok": True,
            "source": source_key,
            "destination": destination_key,
        }

    def delete_usage_routing_rule(self, source: Any) -> dict:
        source_key = str(source or "").strip()
        if not source_key or source_key not in self.usage_routing_rules:
            raise ValueError("routing rule not found")
        self.usage_routing_rules.pop(source_key)
        self._save_usage_routing_rules()
        return {"ok": True, "source": source_key}

    def update_usage_alias(
        self,
        model: Any,
        alias: Any,
        *,
        merge_group: Any = None,
        update_merge_group: bool = False,
    ) -> dict:
        model_key = str(model or "").strip()
        if not model_key or model_key not in self.token_stats:
            raise ValueError("usage model not found")
        # Validate the entire request before changing either map or file.  A
        # bad merge group must not leave an otherwise-valid alias half-saved.
        normalized = self._normalized_alias(alias)
        normalized_group = (
            self._normalized_alias(merge_group)
            if update_merge_group
            else getattr(self, "usage_merge_groups", {}).get(model_key, "")
        )
        if normalized:
            self.usage_aliases[model_key] = normalized
        else:
            self.usage_aliases.pop(model_key, None)
        self._save_usage_aliases()
        if update_merge_group:
            if normalized_group:
                self.usage_merge_groups[model_key] = normalized_group
            else:
                self.usage_merge_groups.pop(model_key, None)
            self._save_usage_merge_groups()
        return {
            "ok": True,
            "model": model_key,
            "alias": normalized or None,
            "merge_group": normalized_group or None,
        }

    @staticmethod
    def _normalized_alias(value: Any, fallback: str = "") -> str:
        alias = str(value or "").strip()
        if len(alias) > 120:
            raise ValueError("alias must be 120 characters or fewer")
        return alias or fallback

    def update_deployment_alias(self, deployment_id: str, alias: Any) -> dict:
        deployment = self._deployment(deployment_id)
        if not deployment:
            raise ValueError("deployment not found")
        deployment["name"] = self._normalized_alias(alias, deployment.get("model") or deployment_id)
        if deployment.get("launch_settings"):
            deployment["launch_settings"]["deployment_name"] = deployment["name"]
        self._save_deployments()
        return deployment

    async def update_container_alias(self, name: str, alias: Any) -> dict:
        try:
            self.client.containers.get(name)
        except Exception as exc:
            raise ValueError("container not found") from exc
        normalized = self._normalized_alias(alias)
        if normalized:
            self.container_aliases[name] = normalized
        else:
            self.container_aliases.pop(name, None)
        self._save_container_aliases()
        return {"ok": True, "name": name, "alias": normalized or None}

    def _deployment(self, deployment_id: str) -> dict | None:
        return next(
            (d for d in self.deployments if d.get("id") == deployment_id), None
        )

    @staticmethod
    def _pricing_value(value: Any, field: str) -> float | None:
        if value in (None, ""):
            return None
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be a non-negative number") from exc
        if not math.isfinite(result) or result < 0:
            raise ValueError(f"{field} must be a non-negative number")
        return result

    @classmethod
    def _deployment_launch_settings(cls, body: dict) -> dict:
        """Return the durable, credential-free inputs for a cluster launch."""
        engine = body.get("engine") or "vllm"
        extra_args = cls._without_hf_cli_credentials(
            list(body.get("extra_args") or [])
        )
        if engine == "vllm":
            extra_args = cls._with_vllm_prompt_token_details(extra_args)
        return {
            "deployment_name": body.get("deployment_name") or body.get("name"),
            "model": body.get("model") or "",
            "engine": engine,
            "image": body.get("image") or None,
            "extra_args": extra_args,
            "gpu_memory_utilization": body.get("gpu_memory_utilization"),
            "gpu_memory_gb": body.get("gpu_memory_gb"),
            "sg_tp_size": body.get("sg_tp_size"),
            "sg_context_length": body.get("sg_context_length"),
            "sg_max_running_requests": body.get("sg_max_running_requests"),
            "sg_mem_fraction": body.get("sg_mem_fraction"),
            "sg_image": body.get("sg_image") or None,
            "deployment_mode": body.get("deployment_mode") or body.get("mode") or "single",
            "node_ids": list(dict.fromkeys(body.get("node_ids") or [LOCAL_NODE_ID])),
            "port": body.get("port"),
            "sparkdeck_record_id": body.get("sparkdeck_record_id"),
            "input_cost_per_1m": cls._pricing_value(
                body.get("input_cost_per_1m"), "input_cost_per_1m"
            ),
            "cache_cost_per_1m": cls._pricing_value(
                body.get("cache_cost_per_1m"), "cache_cost_per_1m"
            ),
            "output_cost_per_1m": cls._pricing_value(
                body.get("output_cost_per_1m"), "output_cost_per_1m"
            ),
        }

    @staticmethod
    def _deployment_pricing(settings: dict) -> dict:
        return {
            "input_cost_per_1m": settings.get("input_cost_per_1m"),
            "cache_cost_per_1m": settings.get("cache_cost_per_1m"),
            "output_cost_per_1m": settings.get("output_cost_per_1m"),
        }

    @classmethod
    def _deployment_launch_controls(cls, settings: dict) -> dict:
        """Parse common cluster controls without removing image-specific args."""
        args = list(settings.get("extra_args") or [])
        engine = settings.get("engine") or "vllm"
        speculative = {}
        raw_speculative = cls._cli_option(args, {"--speculative-config"})
        if raw_speculative:
            try:
                parsed = json.loads(raw_speculative)
                if isinstance(parsed, dict):
                    speculative = parsed
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
        thinking_mode, _, _ = cls._thinking_config(args)
        context_window = cls._cli_option(
            args,
            {"--context-length"} if engine == "sglang"
            else {"--max-model-len", "--max-model-length"},
            int,
        )
        max_concurrency = cls._cli_option(
            args,
            {"--max-running-requests"} if engine == "sglang"
            else {"--max-num-seqs"},
            int,
        )
        if engine == "sglang":
            context_window = settings.get("sg_context_length") or context_window
            max_concurrency = settings.get("sg_max_running_requests") or max_concurrency
        return {
            "context_window": context_window,
            "max_concurrency": max_concurrency,
            "kv_cache_dtype": cls._cli_option(args, {"--kv-cache-dtype"}),
            "thinking_mode": thinking_mode,
            "dspark_num_speculative_tokens": (
                speculative.get("num_speculative_tokens")
                if isinstance(speculative.get("num_speculative_tokens"), int)
                else None
            ),
            "max_cudagraph_capture_size": cls._cli_option(
                args, {"--max-cudagraph-capture-size"}, int
            ),
            "max_num_batched_tokens": cls._cli_option(
                args, {"--max-num-batched-tokens"}, int
            ),
        }

    def _apply_deployment_launch_controls(
        self, args: list[str], engine: str, controls: dict
    ) -> list[str]:
        """Merge structured editor fields back into the complete argv."""
        flags = shlex.join([str(value) for value in args])

        def positive_int(key: str) -> int | None:
            value = controls.get(key)
            if value in (None, ""):
                return None
            try:
                parsed = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{key} must be a positive integer") from exc
            if parsed <= 0:
                raise ValueError(f"{key} must be a positive integer")
            return parsed

        flags = self._replace_command_option(
            flags,
            {"--context-length"} if engine == "sglang"
            else {"--max-model-len", "--max-model-length"},
            None if engine == "sglang" else positive_int("context_window"),
        )
        flags = self._replace_command_option(
            flags,
            {"--max-running-requests"} if engine == "sglang"
            else {"--max-num-seqs"},
            None if engine == "sglang" else positive_int("max_concurrency"),
        )
        kv_dtype = controls.get("kv_cache_dtype")
        kv_dtype = str(kv_dtype).strip() if kv_dtype not in (None, "") else None
        flags = self._replace_command_option(
            flags, {"--kv-cache-dtype"}, kv_dtype
        )
        flags = self._replace_thinking_config(
            flags, str(controls.get("thinking_mode") or "default")
        )

        if engine == "vllm":
            flags = self._replace_command_option(
                flags,
                {"--max-cudagraph-capture-size"},
                positive_int("max_cudagraph_capture_size"),
            )
            flags = self._replace_command_option(
                flags,
                {"--max-num-batched-tokens"},
                positive_int("max_num_batched_tokens"),
            )

            speculative_tokens = positive_int("dspark_num_speculative_tokens")
            try:
                current_tokens = shlex.split(flags)
            except ValueError as exc:
                raise ValueError("launch arguments have invalid shell quoting") from exc
            raw_speculative = self._cli_option(
                current_tokens, {"--speculative-config"}
            )
            speculative: dict[str, Any] = {}
            if raw_speculative:
                try:
                    parsed = json.loads(raw_speculative)
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError("--speculative-config must contain valid JSON") from exc
                if not isinstance(parsed, dict):
                    raise ValueError("--speculative-config must contain a JSON object")
                speculative = parsed
            if speculative_tokens is None:
                speculative.pop("num_speculative_tokens", None)
            else:
                speculative["num_speculative_tokens"] = speculative_tokens
            speculative_value = None
            if speculative:
                speculative_value = shlex.quote(
                    json.dumps(speculative, separators=(",", ":"))
                )
            flags = self._replace_command_option(
                flags, {"--speculative-config"}, speculative_value
            )

        try:
            return shlex.split(flags)
        except ValueError as exc:
            raise ValueError("launch arguments have invalid shell quoting") from exc

    def update_deployment_settings(self, deployment_id: str, body: dict) -> dict:
        """Save the inputs used to rebuild a stopped clustered deployment."""
        if "extra_args" in body:
            self._reject_hf_cli_credentials(body.get("extra_args"))
        deployment = self._deployment(deployment_id)
        if not deployment:
            raise ValueError("deployment not found")
        persisted_args_error = bool(
            deployment.get("launch_settings_error")
            or PERSISTED_DEPLOYMENT_ARGS_ERROR
            in str(deployment.get("error") or "")
        )
        if deployment.get("status") != "stopped" and not (
            deployment.get("status") == "error" and persisted_args_error
        ):
            raise ValueError("stop the cluster before changing its launch settings")
        if persisted_args_error and (
            "extra_args" not in body
            or not isinstance(body.get("extra_args"), list)
        ):
            raise ValueError(
                "extra_args must be explicitly repaired with an array before "
                "this deployment can start"
            )

        current = deployment.get("launch_settings") or {
            "deployment_name": deployment.get("name"),
            "model": deployment.get("model"),
            "engine": deployment.get("engine", "vllm"),
            "deployment_mode": deployment.get("mode", "single"),
            "node_ids": deployment.get("node_ids") or [LOCAL_NODE_ID],
            "port": deployment.get("api_port"),
        }
        merged = {**current, **body}
        controls = body.get("launch_controls")
        if isinstance(controls, dict) and (merged.get("engine") or "vllm") == "sglang":
            merged["sg_context_length"] = controls.get("context_window")
            merged["sg_max_running_requests"] = controls.get("max_concurrency")
        settings = self._deployment_launch_settings(merged)
        if "launch_controls" in body:
            if not isinstance(controls, dict):
                raise ValueError("launch_controls must be an object")
            settings["extra_args"] = self._apply_deployment_launch_controls(
                settings["extra_args"], settings["engine"], controls
            )
        if not settings["model"]:
            raise ValueError("model is required")
        if settings["engine"] not in {"vllm", "sglang"}:
            raise ValueError("engine must be vllm or sglang")
        mode = settings["deployment_mode"]
        if mode not in {"single", "sharded", "replicated"}:
            raise ValueError("deployment_mode must be single, sharded, or replicated")
        if mode == "single":
            settings["node_ids"] = settings["node_ids"][:1] or [LOCAL_NODE_ID]
        elif len(settings["node_ids"]) < 2:
            raise ValueError(f"{mode} deployment requires at least two nodes")
        if mode == "sharded" and settings["node_ids"][0] != LOCAL_NODE_ID:
            raise ValueError("the coordinator must be the first node in a sharded deployment")
        if mode == "sharded" and settings["engine"] == "vllm":
            requested_tp = self._cli_option(
                settings["extra_args"], {"--tensor-parallel-size", "-tp"}, int
            )
            requested_pp = self._cli_option(
                settings["extra_args"], {"--pipeline-parallel-size", "-pp"}, int
            )
            if requested_tp is not None or requested_pp is not None:
                tp = requested_tp or 1
                pp = requested_pp or 1
                if tp < 1 or pp < 1 or tp * pp != len(settings["node_ids"]):
                    raise ValueError(
                        "explicit tensor/pipeline parallel sizes must be positive "
                        f"and multiply to the {len(settings['node_ids'])} selected nodes"
                    )

        # Preserve the assigned API port unless the editor explicitly changes it.
        if settings.get("port") is None:
            settings["port"] = deployment.get("api_port")
        remaining_error = (
            _remove_persisted_error(
                deployment.get("error"), PERSISTED_DEPLOYMENT_ARGS_ERROR,
            )
            if persisted_args_error else ""
        )
        deployment.update({
            "name": settings.get("deployment_name") or settings["model"],
            "model": settings["model"],
            "engine": settings["engine"],
            "mode": mode,
            "node_ids": settings["node_ids"],
            "launch_settings": settings,
            "settings_dirty": True,
            "status": (
                "error" if remaining_error else "stopped"
            ) if persisted_args_error else deployment.get("status"),
            "error": remaining_error or None,
        })
        deployment.pop("launch_settings_error", None)
        deployment.pop("kv_capacity", None)
        deployment.pop("auto_concurrency_adjustment", None)
        self._save_deployments()
        return deployment

    def _recovered_deployment_launch_settings(
        self, deployment: dict, primary_container: dict | None = None,
    ) -> dict:
        load_settings = (primary_container or {}).get("load_settings") or {}
        try:
            recovered_args = shlex.split(load_settings.get("command_flags") or "")
        except ValueError:
            recovered_args = list(load_settings.get("extra_args") or [])
        recovered_args = self._without_cli_options(
            recovered_args,
            {"--host", "--port", "--gpu-memory-utilization",
             "--gpu_memory_utilization"},
        )
        return self._deployment_launch_settings({
            "deployment_name": deployment.get("name"),
            "model": deployment.get("model"),
            "engine": deployment.get("engine", "vllm"),
            "image": (primary_container or {}).get("image"),
            "extra_args": recovered_args,
            "gpu_memory_utilization": load_settings.get(
                "gpu_memory_utilization"
            ),
            "deployment_mode": deployment.get("mode", "single"),
            "node_ids": deployment.get("node_ids") or [LOCAL_NODE_ID],
            "port": deployment.get("api_port"),
        })

    @classmethod
    def _deployment_pricing_model_key(cls, deployment: dict) -> str:
        explicit = str(deployment.get("pricing_model_key") or "").strip()
        if explicit:
            return explicit
        settings = deployment.get("launch_settings") or {}
        model = str(deployment.get("model") or settings.get("model") or "")
        return cls._stats_key(
            model, cls._variant_from_cmd(settings.get("extra_args") or [])
        ) if model else ""

    async def update_deployment_pricing(
        self, deployment_id: str, body: dict
    ) -> dict:
        """Update accounting metadata without restarting a running cluster."""
        deployment = self._deployment(deployment_id)
        if not deployment:
            raise ValueError("deployment not found")
        pricing = {
            field: self._pricing_value(body.get(field), field)
            for field in (
                "input_cost_per_1m", "cache_cost_per_1m", "output_cost_per_1m",
            )
        }
        launch_settings = deployment.get("launch_settings")
        if not isinstance(launch_settings, dict) or not launch_settings:
            containers = await self.list_containers()
            candidates = [
                container for container in containers
                if container.get("deployment_id") == deployment_id
            ]
            primary = next(
                (container for container in candidates
                 if container.get("rank") == 0),
                candidates[0] if candidates else None,
            )
            launch_settings = self._recovered_deployment_launch_settings(
                deployment, primary
            )
            deployment["launch_settings"] = launch_settings
            if primary and primary.get("stats_key"):
                deployment["pricing_model_key"] = primary["stats_key"]
        launch_settings.update(pricing)
        if not deployment.get("pricing_model_key"):
            deployment["pricing_model_key"] = (
                self._deployment_pricing_model_key(deployment)
            )
        deployment["pricing"] = pricing
        self._save_deployments()
        return {"ok": True, "deployment_id": deployment_id, **pricing}

    @staticmethod
    def _without_cli_options(args: list[str], names: set[str]) -> list[str]:
        """Drop scalar distributed flags so coordinator-owned values win."""
        result: list[str] = []
        i = 0
        while i < len(args):
            value = args[i]
            key = value.split("=", 1)[0]
            if key in names:
                if "=" not in value and i + 1 < len(args) and not args[i + 1].startswith("--"):
                    i += 2
                else:
                    i += 1
                continue
            result.append(value)
            i += 1
        return result

    @classmethod
    def _without_hf_cli_credentials(cls, args: list[Any]) -> list[str]:
        if not isinstance(args, list):
            return []
        return cls._without_cli_options(
            [str(value) for value in (args or [])], HF_CREDENTIAL_CLI_OPTIONS,
        )

    @classmethod
    def _reject_hf_cli_credentials(cls, args: list[Any] | None) -> None:
        if args is not None and (
            not isinstance(args, list)
            or any(not isinstance(value, str) for value in args)
        ):
            raise ValueError("extra_args must be an array of strings")
        original = [str(value) for value in (args or [])]
        if cls._without_hf_cli_credentials(original) != original:
            raise ValueError(
                "configure Hugging Face credentials in Settings, not launch arguments"
            )

    @staticmethod
    def _cli_option(args: list[str], names: set[str], cast=None):
        """Return the last value supplied for one of ``names``.

        Docker images may persist options as either ``--flag value`` or
        ``--flag=value``.  Load-settings discovery needs to understand both
        forms so editing a container never silently resets an existing flag.
        """
        found = None
        for i, raw in enumerate(args):
            token = str(raw)
            key, separator, inline = token.partition("=")
            if key not in names:
                continue
            if separator:
                found = inline
            elif i + 1 < len(args) and not str(args[i + 1]).startswith("-"):
                found = args[i + 1]
        if found is None or cast is None:
            return found
        try:
            return cast(found)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _shell_vllm_command(script: str):
        """Locate a vLLM invocation inside a ``sh -c``/``bash -lc`` script."""
        return re.search(
            r"(?:^|;\s*|\bexec\s+)(?:[^\s;]*/)?vllm\s+serve\s+"
            r"(?P<model>(?:\"[^\"]*\"|'[^']*'|[^\s;]+))(?P<flags>.*)$",
            script or "",
            re.DOTALL,
        )

    @classmethod
    def _thinking_config(cls, args: list[str]) -> tuple[str, dict, str | None]:
        """Return mode, full chat-template kwargs, and its thinking key."""
        raw = cls._cli_option(args, {"--default-chat-template-kwargs"})
        if raw is None:
            return "default", {}, None
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            return "default", {}, None
        if not isinstance(payload, dict):
            return "default", {}, None
        for key in ("enable_thinking", "thinking"):
            if isinstance(payload.get(key), bool):
                return ("enabled" if payload[key] else "disabled"), payload, key
        return "default", payload, None

    def _container_load_settings(self, cmd: list[str], engine: str, model: str) -> dict:
        """Extract editable launch settings from a Docker command.

        Common performance controls get dedicated UI fields. Every remaining
        flag is also exposed as text so new engine options do not require a
        controller release before users can edit them. Shell-wrapped commands
        retain their original quoting and variable expressions.
        """
        cmd = [str(value) for value in (cmd or [])]
        engine = engine if engine == "sglang" else "vllm"
        command_flags = ""
        analysis_cmd = cmd
        if engine == "vllm" and len(cmd) >= 3 and cmd[-2] in {"-c", "-lc"}:
            match = self._shell_vllm_command(cmd[-1])
            if match:
                command_flags = match.group("flags").strip()
                try:
                    analysis_cmd = [
                        "vllm", "serve", shlex.split(match.group("model"))[0],
                        *shlex.split(command_flags),
                    ]
                except (ValueError, IndexError):
                    analysis_cmd = cmd
        if engine == "sglang":
            managed = {
                "--model-path", "--host", "--port", "--tp-size",
                "--context-length", "--max-running-requests",
                "--mem-fraction-static", "--max-total-tokens",
                "--kv-cache-dtype",
            }
            skip_tokens = {"-m", "python", "python3", "sglang.launch_server", model}
            extra_args = []
            i = 0
            while i < len(analysis_cmd):
                token = analysis_cmd[i]
                key = token.split("=", 1)[0]
                if key in managed:
                    if "=" not in token and i + 1 < len(analysis_cmd):
                        i += 2
                    else:
                        i += 1
                    continue
                if token in skip_tokens:
                    i += 1
                    continue
                extra_args.append(token)
                i += 1
            if not command_flags:
                editable = []
                i = 0
                while i < len(analysis_cmd):
                    token = analysis_cmd[i]
                    if token in {"python", "python3", "-m", "sglang.launch_server"}:
                        i += 1
                        continue
                    if token == "--model-path" and i + 1 < len(analysis_cmd):
                        i += 2
                        continue
                    editable.append(token)
                    i += 1
                command_flags = shlex.join(editable)
            thinking_mode, _, _ = self._thinking_config(analysis_cmd)
            return {
                "editable": "--model-path" in analysis_cmd,
                "engine": engine,
                "gpu_memory_utilization": self._cli_option(
                    analysis_cmd, {"--mem-fraction-static"}, float
                ),
                "max_concurrency": self._cli_option(
                    analysis_cmd, {"--max-running-requests"}, int
                ),
                "kv_cache_dtype": self._cli_option(analysis_cmd, {"--kv-cache-dtype"}),
                "context_window": self._cli_option(
                    analysis_cmd, {"--context-length"}, int
                ),
                "tensor_parallel_size": self._cli_option(
                    analysis_cmd, {"--tp-size"}, int
                ),
                "thinking_mode": thinking_mode,
                "extra_args": extra_args,
                "command_flags": command_flags,
            }

        managed = {
            "--host", "--port", "--gpu-memory-utilization",
            "--gpu_memory_utilization", "--max-model-len", "--max-model-length",
            "--max-num-seqs", "--kv-cache-dtype",
        }
        try:
            model_index = analysis_cmd.index("serve") + 2
        except ValueError:
            try:
                model_index = analysis_cmd.index(model) + 1
            except ValueError:
                model_index = len(analysis_cmd)
        if not command_flags:
            command_flags = shlex.join(analysis_cmd[model_index:])
        extra_args = []
        i = model_index
        while i < len(analysis_cmd):
            token = analysis_cmd[i]
            key = token.split("=", 1)[0]
            if key in managed:
                if "=" not in token and i + 1 < len(analysis_cmd):
                    i += 2
                else:
                    i += 1
                continue
            extra_args.append(token)
            i += 1
        thinking_mode, _, _ = self._thinking_config(analysis_cmd)
        return {
            "editable": "serve" in analysis_cmd,
            "engine": engine,
            "gpu_memory_utilization": self._cli_option(
                analysis_cmd,
                {"--gpu-memory-utilization", "--gpu_memory_utilization"},
                float,
            ),
            "max_concurrency": self._cli_option(
                analysis_cmd, {"--max-num-seqs"}, int
            ),
            "kv_cache_dtype": self._cli_option(
                analysis_cmd, {"--kv-cache-dtype"}
            ),
            "context_window": self._cli_option(
                analysis_cmd, {"--max-model-len", "--max-model-length"}, int
            ),
            "tensor_parallel_size": self._cli_option(
                analysis_cmd, {"--tensor-parallel-size", "-tp"}, int
            ),
            "thinking_mode": thinking_mode,
            "extra_args": extra_args,
            "command_flags": command_flags,
        }

    async def _create_member(self, node_id: str, payload: dict) -> dict:
        if node_id == LOCAL_NODE_ID:
            return await self.create_container(**payload)
        return await self.node_registry.request(
            node_id,
            "POST",
            "/api/agent/containers",
            json_body=payload,
            timeout=1800,
        )

    def _cluster_primary_member(self, deployment_id: str) -> tuple[dict, dict]:
        deployment = self._deployment(deployment_id)
        if not deployment:
            raise LookupError("cluster deployment not found")
        members = sorted(
            deployment.get("members") or [], key=lambda member: int(member.get("rank") or 0)
        )
        if not members:
            raise LookupError("cluster deployment has no inference member")
        return deployment, members[0]

    async def proxy_cluster_inference(
        self,
        deployment_id: str,
        model: str,
        body: dict,
        endpoint: str,
        cancel: asyncio.Event | None = None,
    ):
        """Proxy through the selected primary member without bypassing admission."""
        deployment, primary = self._cluster_primary_member(deployment_id)
        node_id = primary.get("node_id")
        if node_id == LOCAL_NODE_ID:
            return (
                await self._vllm_chat(
                    model, body, bool(body.get("stream")), cancel,
                    container_name=primary.get("container_name"),
                    deployment_id=deployment_id,
                )
                if endpoint == "chat/completions"
                else await self._vllm_completions(
                    model, body, bool(body.get("stream")), cancel,
                    container_name=primary.get("container_name"),
                    deployment_id=deployment_id,
                )
            )
        if not node_id:
            raise LookupError("cluster primary node is unavailable")

        controls = self._deployment_launch_controls(deployment.get("launch_settings") or {})
        admission_container = {
            "deployment_id": deployment_id,
            "name": primary.get("container_name"),
            "stats_key": model,
            "load_settings": {"max_concurrency": controls.get("max_concurrency")},
        }
        admission = await self._acquire_inference_slot(admission_container, model, cancel)
        path = f"/api/agent/inference/{endpoint}"
        remote_body = {
            **body,
            "_sparkdeck_container_name": primary.get("container_name"),
            "_sparkdeck_deployment_id": deployment_id,
        }
        if body.get("stream"):
            try:
                response = await self._await_or_cancel(
                    self.node_registry.open_stream(
                        node_id, "POST", path, json_body=remote_body, timeout=600,
                    ),
                    cancel,
                )
                if response.status_code >= 400:
                    detail = (await response.aread()).decode("utf-8", errors="replace")
                    await response.aclose()
                    raise RuntimeError(f"remote inference failed: HTTP {response.status_code}: {detail[:500]}")
            except BaseException:
                self._release_inference_slot(admission)
                raise

            async def stream_remote():
                try:
                    async for line in self._aiter_lines_cancellable(response, cancel):
                        if line:
                            yield f"{line}\n\n"
                finally:
                    await response.aclose()
                    self._release_inference_slot(admission)

            return stream_remote()

        try:
            return await self._await_or_cancel(
                self.node_registry.request(
                    node_id, "POST", path, json_body=remote_body, timeout=600,
                ),
                cancel,
            )
        finally:
            self._release_inference_slot(admission)

    async def cluster_deployment_health(self, deployment_id: str, model: str) -> bool:
        """Check the selected primary member through its authenticated agent."""
        _, primary = self._cluster_primary_member(deployment_id)
        node_id = primary.get("node_id")
        if node_id == LOCAL_NODE_ID:
            container = await self._resolve_vllm_target(
                model, container_name=primary.get("container_name"),
                deployment_id=deployment_id,
            )
            return await self._check_ready(container)
        result = await self.node_registry.request(
            node_id, "POST", "/api/agent/inference/health",
            json_body={
                "model": model,
                "_sparkdeck_container_name": primary.get("container_name"),
                "_sparkdeck_deployment_id": deployment_id,
            }, timeout=10,
        )
        return bool((result or {}).get("ready"))

    async def create_deployment(self, body: dict) -> dict:
        body = dict(body)
        self._reject_hf_cli_credentials(body.get("extra_args"))
        engine = str(body.get("engine") or "vllm")
        if engine not in {"vllm", "sglang"}:
            raise ValueError("engine must be vllm or sglang")
        controls = body.get("launch_controls")
        if controls is not None:
            if not isinstance(controls, dict):
                raise ValueError("launch_controls must be an object")
            controls = {
                **self._deployment_launch_controls(
                    self._deployment_launch_settings(body)
                ),
                **controls,
            }
            if engine == "sglang":
                if "context_window" in controls:
                    body["sg_context_length"] = controls.get("context_window")
                if "max_concurrency" in controls:
                    body["sg_max_running_requests"] = controls.get("max_concurrency")
            body["extra_args"] = self._apply_deployment_launch_controls(
                list(body.get("extra_args") or []), engine, controls
            )
        mode = body.get("deployment_mode") or "single"
        if mode not in {"single", "sharded", "replicated"}:
            raise ValueError("deployment_mode must be single, sharded, or replicated")
        node_ids = list(dict.fromkeys(body.get("node_ids") or [LOCAL_NODE_ID]))
        if mode == "single":
            node_ids = node_ids[:1] or [LOCAL_NODE_ID]
        elif len(node_ids) < 2:
            raise ValueError(f"{mode} deployment requires at least two nodes")
        if mode == "sharded" and node_ids[0] != LOCAL_NODE_ID:
            raise ValueError("the coordinator must be the first node in a sharded deployment")

        available = {n["id"]: n for n in await self.cluster_nodes()}
        missing = [nid for nid in node_ids if nid not in available]
        offline = [nid for nid in node_ids if nid in available and not available[nid].get("online")]
        if missing:
            raise ValueError(f"unknown cluster node(s): {', '.join(missing)}")
        if offline:
            names = [available[n].get("name", n) for n in offline]
            raise ValueError(f"cluster node(s) are offline: {', '.join(names)}")
        docker_unready = [
            nid for nid in node_ids if not available[nid].get("docker_ready")
        ]
        if docker_unready:
            names = [available[n].get("name", n) for n in docker_unready]
            raise ValueError(f"Docker is unavailable on: {', '.join(names)}")

        model = body.get("model") or ""
        if not model:
            raise ValueError("model is required")
        vllm_parallel_layout: tuple[int, int] | None = None
        if mode == "sharded" and engine == "vllm":
            requested_args = list(body.get("extra_args") or [])
            requested_tp = self._cli_option(
                requested_args, {"--tensor-parallel-size", "-tp"}, int
            )
            requested_pp = self._cli_option(
                requested_args, {"--pipeline-parallel-size", "-pp"}, int
            )
            if requested_tp is not None or requested_pp is not None:
                tp = requested_tp or 1
                pp = requested_pp or 1
                if tp < 1 or pp < 1 or tp * pp != len(node_ids):
                    raise ValueError(
                        "explicit tensor/pipeline parallel sizes must be positive "
                        f"and multiply to the {len(node_ids)} selected nodes"
                    )
                vllm_parallel_layout = (tp, pp)
            else:
                # Default to one local tensor-parallel rank per node and
                # pipeline parallelism across nodes. Models that do not
                # implement SupportsPP can explicitly request TP=nnodes, PP=1.
                vllm_parallel_layout = (1, len(node_ids))
        requested_port = body.get("port")
        local_port = requested_port
        if LOCAL_NODE_ID in node_ids and local_port is None:
            # A controller port is meaningful only for the controller member.
            # Remote agents allocate against their own Docker/host namespace.
            local_port = await self._allocate_port()
        master_ip, _ = self._inferred_fabric(
            available[LOCAL_NODE_ID],
            self.settings.get("cluster_fabric_ip"),
            self.settings.get("cluster_fabric_interface"),
        )
        if mode == "sharded" and not master_ip:
            raise ValueError("could not determine the coordinator ConnectX/fabric IP")
        fabrics: dict[str, tuple[str | None, str | None]] = {}
        for node_id in node_ids:
            node = available[node_id]
            fabrics[node_id] = self._inferred_fabric(
                node, node.get("fabric_ip"), node.get("fabric_interface")
            )
            if mode == "sharded" and not fabrics[node_id][0]:
                raise ValueError(
                    f"could not determine fabric IP for {node.get('name', node_id)}"
                )

        # Persist only after every preflight check succeeds. A failed launch
        # remains visible for diagnosis, but invalid input does not leave a
        # phantom deployment card behind.
        deployment_id = uuid.uuid4().hex[:12]
        deployment = {
            "id": deployment_id,
            "name": body.get("deployment_name") or model,
            "model": model,
            "engine": engine,
            "mode": mode,
            "node_ids": node_ids,
            "status": "launching",
            "members": [],
            "created_at": time.time(),
            "recipe_id": body.get("recipe_id"),
            "managed_by": body.get("managed_by"),
            "sparkdeck_record_id": body.get("sparkdeck_record_id"),
            "automation_run_id": body.get("automation_run_id"),
            "settings_dirty": False,
        }
        self.deployments.append(deployment)
        self._save_deployments()

        base = {
            "model": model,
            "engine": engine,
            # Agent requests are authenticated and the deployment payload is
            # not persisted. Forward the coordinator credential so every
            # selected node downloads as the same HF account.
            "hf_token": self._resolved_hf_token(),
            "gpu_memory_utilization": body.get("gpu_memory_utilization"),
            "gpu_memory_gb": body.get("gpu_memory_gb"),
            "extra_args": (
                self._with_vllm_prompt_token_details(
                    list(body.get("extra_args") or [])
                )
                if engine == "vllm"
                else list(body.get("extra_args") or [])
            ),
            "image": body.get("image"),
            "sg_tp_size": body.get("sg_tp_size"),
            "sg_context_length": body.get("sg_context_length"),
            "sg_max_running_requests": body.get("sg_max_running_requests"),
            "sg_mem_fraction": body.get("sg_mem_fraction"),
            "sg_image": body.get("sg_image"),
        }

        tasks = []
        member_specs = []
        for rank, node_id in enumerate(node_ids):
            node = available[node_id]
            member_port = local_port if node_id == LOCAL_NODE_ID else None
            fabric_ip, fabric_interface = fabrics[node_id]
            safe_model = re.sub(r"[^a-zA-Z0-9_.-]+", "-", model).strip("-").lower()
            name = f"cluster-{deployment_id}-r{rank}-{safe_model[:36]}"
            payload = dict(base)
            payload.update({
                "port": member_port,
                "name": name,
                "cluster_member": {
                    "deployment_id": deployment_id,
                    "node_id": node_id,
                    "rank": rank,
                    "nnodes": len(node_ids),
                    "mode": mode,
                    "serve_port": member_port,
                    "fabric_ip": fabric_ip,
                    "fabric_interface": fabric_interface,
                },
            })
            if mode == "sharded":
                if engine == "vllm":
                    tp_size, pp_size = vllm_parallel_layout or (1, len(node_ids))
                    vllm_args = self._without_cli_options(
                        payload["extra_args"],
                        {"--distributed-executor-backend", "--nnodes", "--node-rank",
                         "--master-addr", "--master-port", "--tensor-parallel-size", "-tp",
                         "--pipeline-parallel-size", "-pp"},
                    )
                    # ``--headless`` is a valueless switch, unlike the scalar
                    # options handled above.
                    vllm_args = [arg for arg in vllm_args if arg != "--headless"]
                    payload["extra_args"] = vllm_args + [
                        "--distributed-executor-backend", "mp",
                        "--nnodes", str(len(node_ids)),
                        "--node-rank", str(rank),
                        "--master-addr", master_ip,
                        "--master-port", "29501",
                        "--tensor-parallel-size", str(tp_size),
                        "--pipeline-parallel-size", str(pp_size),
                    ]
                    if rank > 0:
                        payload["extra_args"].append("--headless")
                else:
                    payload["extra_args"] = self._without_cli_options(
                        payload["extra_args"],
                        {"--nnodes", "--node-rank", "--dist-init-addr", "--tp-size"},
                    ) + [
                        "--nnodes", str(len(node_ids)),
                        "--node-rank", str(rank),
                        "--dist-init-addr", f"{master_ip}:29501",
                    ]
                    payload["sg_tp_size"] = len(node_ids)
            member_specs.append({
                "node_id": node_id,
                "node_name": node.get("name", node_id),
                "rank": rank,
                "container_name": name,
                "fabric_ip": fabric_ip,
                "port": member_port,
                "status": "queued",
                "phase": {
                    "phase": "queued",
                    "message": "Waiting for the node agent to begin launch",
                },
            })
            tasks.append(self._create_member(node_id, payload))

        # Save member identities before awaiting image pulls. Those requests
        # can take many minutes, and the logs UI needs the names immediately
        # so it can ask every node for controller-side launch progress.
        deployment["members"] = member_specs
        deployment["api_port"] = member_specs[0].get("port")
        deployment["launch_settings"] = self._deployment_launch_settings({
            **body,
            "deployment_name": deployment["name"],
            "node_ids": node_ids,
            # Persist only a user-requested fixed port. Automatically chosen
            # ports must be reallocated by each owning node on rebuild.
            "port": requested_port,
        })
        self._save_deployments()

        created = await asyncio.gather(*tasks, return_exceptions=True)
        errors = []
        model_sources = []
        for spec, result in zip(member_specs, created):
            if isinstance(result, Exception):
                spec["status"] = "error"
                spec["error"] = str(result)
                errors.append(f"{spec['node_name']}: {result}")
            else:
                spec["status"] = result.get("status", "starting")
                spec["container_id"] = result.get("id")
                spec["port"] = result.get("port") or spec.get("port")
                model_source = str(result.get("model_source") or "unknown")
                if model_source not in {"local", "public_repository", "unknown"}:
                    model_source = "unknown"
                spec["model_source"] = model_source
                model_sources.append(model_source)
        deployment["model_source"] = (
            "local"
            if "local" in model_sources
            else (
                "public_repository"
                if model_sources
                and all(source == "public_repository" for source in model_sources)
                else "unknown"
            )
        )
        deployment["launch_settings"]["model_source"] = deployment["model_source"]
        deployment["api_port"] = member_specs[0].get("port") if member_specs else None
        deployment["status"] = "error" if errors else "starting"
        if errors:
            deployment["error"] = "; ".join(errors)
            # Best-effort rollback of every member that was created.
            await asyncio.gather(
                *[
                    self._member_action(spec, "remove")
                    for spec in member_specs if spec.get("container_id")
                ],
                return_exceptions=True,
            )
        self._save_deployments()
        if errors:
            raise RuntimeError(deployment["error"])
        return {
            **deployment,
            "selected_nodes": [
                self.public_target_node(available[node_id]) for node_id in node_ids
            ],
        }

    async def _member_action(
        self, member: dict, action: str, *, log_tail: int = 300,
    ) -> Any:
        node_id = member["node_id"]
        name = member["container_name"]
        if node_id == LOCAL_NODE_ID:
            if action == "start":
                return await self.start_container(name)
            if action == "stop":
                return await self.stop_container(name)
            if action == "remove":
                return await self.remove_cluster_member(name)
            if action == "logs":
                return {"logs": await self.get_cluster_member_logs(name, log_tail)}
        method = "GET" if action == "logs" else ("DELETE" if action == "remove" else "POST")
        suffix = (
            f"/logs?tail={max(1, min(int(log_tail), 100_000))}"
            if action == "logs"
            else ("" if action == "remove" else f"/{action}")
        )
        return await self.node_registry.request(
            node_id, method, f"/api/agent/containers/{name}{suffix}", timeout=120
        )

    def _cluster_action_lock(self) -> asyncio.Lock:
        """Return the lifecycle lock, including on lightweight test instances."""
        lock = getattr(self, "_deployment_action_lock", None)
        if lock is None:
            lock = self._deployment_action_lock = asyncio.Lock()
        return lock

    @staticmethod
    def _member_action_errors(results: list[Any], action: str) -> list[str]:
        """Return actionable member errors, keeping DELETE idempotent.

        Older node agents return 404 after a cluster member has already been
        removed. That is the desired end state for a remove action, so it must
        not leave an undeletable deployment card behind.
        """
        errors = []
        for result in results:
            if not isinstance(result, Exception):
                continue
            message = str(result)
            already_absent = action == "remove" and any(
                marker in message.lower()
                for marker in (
                    "cluster member not found",
                    "managed container not found",
                    "no such container",
                )
            )
            if not already_absent:
                errors.append(message)
        return errors

    async def deployment_action(self, deployment_id: str, action: str) -> dict:
        # A health recovery and a user action must never interleave their
        # per-rank stop/start requests.
        async with self._cluster_action_lock():
            return await self._deployment_action_locked(deployment_id, action)

    async def _deployment_action_locked(self, deployment_id: str, action: str) -> dict:
        deployment = self._deployment(deployment_id)
        if not deployment:
            raise ValueError("deployment not found")
        if action not in {"start", "stop", "remove"}:
            raise ValueError("invalid deployment action")
        if action == "start" and (
            deployment.get("launch_settings_error")
            or PERSISTED_DEPLOYMENT_ARGS_ERROR
            in str(deployment.get("error") or "")
        ):
            raise ValueError(
                "saved launch arguments are invalid; edit extra_args before starting"
            )
        if (
            action == "start"
            and str(deployment.get("engine") or "vllm") not in {"vllm", "sglang"}
        ):
            raise ValueError("persisted deployment runtime is no longer supported")

        if action == "start" and deployment.get("settings_dirty"):
            # The stopped containers still contain the old argv. Remove them,
            # then use the normal fully validated launch path with the saved
            # settings so every node/rank receives a coherent replacement.
            removed = await asyncio.gather(
                *[
                    self._member_action(member, "remove")
                    for member in deployment.get("members", [])
                ],
                return_exceptions=True,
            )
            remove_errors = self._member_action_errors(removed, "remove")
            if remove_errors:
                deployment["status"] = "stopped"
                deployment["error"] = "; ".join(remove_errors)
                self._save_deployments()
                return {"ok": False, "errors": remove_errors}

            deployment["members"] = []
            deployment["error"] = None
            self._save_deployments()
            launch_body = dict(deployment.get("launch_settings") or {})
            launch_body["recipe_id"] = deployment.get("recipe_id")
            try:
                replacement = await self.create_deployment(launch_body)
            except Exception:
                # Keep the saved card so its settings can be corrected and
                # retried. create_deployment also leaves its failed attempt
                # visible with the node-specific diagnostic.
                deployment["status"] = "stopped"
                self._save_deployments()
                raise
            self.deployments = [
                item for item in self.deployments if item.get("id") != deployment_id
            ]
            self._save_deployments()
            return {
                "ok": True,
                "errors": [],
                "deployment": replacement,
                "replaced_deployment_id": deployment_id,
            }

        results = await asyncio.gather(
            *[self._member_action(m, action) for m in deployment.get("members", [])],
            return_exceptions=True,
        )
        errors = self._member_action_errors(results, action)
        if action == "remove" and not errors:
            self.deployments = [d for d in self.deployments if d.get("id") != deployment_id]
        else:
            if errors:
                # A partial start is still intended to be running; keep it in
                # the health monitor's candidate set so the successful rank is
                # stopped and the complete cluster is retried atomically.
                deployment["status"] = "degraded" if action == "start" else "error"
            else:
                deployment["status"] = "starting" if action == "start" else "stopped"
            deployment["error"] = "; ".join(errors) if errors else None
        self._save_deployments()
        return {"ok": not errors, "errors": errors}

    @staticmethod
    def _container_started_epoch(value: Any) -> float | None:
        if not value or str(value).startswith("0001-"):
            return None
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
        except (TypeError, ValueError, OverflowError):
            return None

    def _cluster_health_issue(self, deployment: dict, nodes: list[dict]) -> str | None:
        """Describe a recoverable split deployment, or return ``None``.

        An unreachable agent is deliberately not called unhealthy here: the
        coordinator cannot guarantee that its worker was stopped, so starting
        a new rendezvous generation would risk preserving the exact zombie we
        are trying to prevent.
        """
        members = deployment.get("members") or []
        if len(members) < 2:
            return None
        if deployment.get("status") in {"stopped", "error", "launching"}:
            return None

        node_by_id = {node.get("id"): node for node in nodes}
        member_containers = []
        for member in members:
            node = node_by_id.get(member.get("node_id"))
            if not node or not node.get("online") or not node.get("docker_ready"):
                return None
            containers = {
                container.get("name"): container
                for container in node.get("containers") or []
            }
            member_containers.append(containers.get(member.get("container_name")))

        states = [
            container.get("status", "unknown") if container else "missing"
            for container in member_containers
        ]
        # A launch still represented by a synthetic agent row is not a failed
        # runtime. create_deployment changes the saved state from launching
        # only after every slow image pull/container creation has returned.
        if any(state in {"queued", "creating"} for state in states):
            return None

        if not all(state == "running" for state in states):
            detail = ", ".join(
                f"rank {member.get('rank')}: {state}"
                for member, state in zip(members, states)
            )
            return f"cluster ranks are split ({detail})"

        started = [
            self._container_started_epoch((container or {}).get("started_at"))
            for container in member_containers
        ]
        known_started = [value for value in started if value is not None]
        start_skew = (
            max(known_started) - min(known_started)
            if len(known_started) == len(members)
            else None
        )
        restart_counts = {
            member.get("container_name"): int((container or {}).get("restart_count") or 0)
            for member, container in zip(members, member_containers)
        }
        baseline = deployment.get("health_restart_counts")
        accept_recovery = deployment.get("health_accept_restart_counts")
        if accept_recovery and start_skew is not None:
            if start_skew <= CLUSTER_RECOVERY_ALIGNMENT_SECONDS:
                deployment["health_restart_counts"] = restart_counts
                deployment.pop("health_accept_restart_counts", None)
                baseline = restart_counts
        elif not isinstance(baseline, dict):
            if start_skew is not None and start_skew <= CLUSTER_RECOVERY_ALIGNMENT_SECONDS:
                # Adopt an already-coordinated deployment after a controller
                # upgrade, even if its historical restart counters differ.
                deployment["health_restart_counts"] = restart_counts
                baseline = restart_counts
            elif len(set(restart_counts.values())) > 1:
                return "cluster ranks have different Docker restart generations"
            else:
                deployment["health_restart_counts"] = restart_counts
                baseline = restart_counts

        if isinstance(baseline, dict):
            changed = [
                name for name, count in restart_counts.items()
                if int(baseline.get(name, count)) != count
            ]
            if changed:
                return "Docker restarted only part of the cluster"

        if len(known_started) == len(members):
            if start_skew is not None and start_skew > CLUSTER_START_SKEW_SECONDS:
                return "cluster ranks were started in different rendezvous generations"
            return None
        return None

    async def _recover_cluster_deployment(self, deployment_id: str, issue: str) -> None:
        """Stop every rank, then start every rank as one atomic generation."""
        async with self._cluster_action_lock():
            deployment = self._deployment(deployment_id)
            if not deployment or deployment.get("status") in {"stopped", "error", "launching"}:
                return
            members = list(deployment.get("members") or [])
            if len(members) < 2:
                return

            checked_at = time.time()
            deployment["status"] = "recovering"
            deployment["health_issue"] = issue
            deployment["health_checked_at"] = checked_at
            self._save_deployments()
            print(f"[cluster-health] {deployment_id}: {issue}; stopping every rank")

            stopped = await asyncio.gather(
                *(self._member_action(member, "stop") for member in members),
                return_exceptions=True,
            )
            stop_errors = [
                f"rank {member.get('rank')}: {result}"
                for member, result in zip(members, stopped)
                if isinstance(result, Exception)
            ]
            if stop_errors:
                deployment["status"] = "degraded"
                deployment["error"] = (
                    "Automatic recovery could not stop every rank; no rank was restarted: "
                    + "; ".join(stop_errors)
                )
                self._save_deployments()
                print(f"[cluster-health] {deployment_id}: {deployment['error']}")
                return

            started = await asyncio.gather(
                *(self._member_action(member, "start") for member in members),
                return_exceptions=True,
            )
            start_errors = [
                f"rank {member.get('rank')}: {result}"
                for member, result in zip(members, started)
                if isinstance(result, Exception)
            ]
            deployment["health_restarted_at"] = time.time()
            if start_errors:
                deployment["status"] = "degraded"
                deployment["error"] = "Automatic recovery failed to start every rank: " + "; ".join(start_errors)
            else:
                deployment["status"] = "starting"
                deployment["error"] = None
                deployment["health_issue"] = None
                # Manual Docker start does not clear cumulative RestartCount.
                # Adopt the post-recovery counters on the next aligned check.
                deployment["health_accept_restart_counts"] = True
            self._save_deployments()
            print(
                f"[cluster-health] {deployment_id}: "
                + (deployment.get("error") or "all ranks restarted together")
            )

    async def _cluster_health_tick(self) -> None:
        candidates = [
            deployment for deployment in list(self.deployments)
            if len(deployment.get("members") or []) >= 2
            and deployment.get("status") not in {"stopped", "error", "launching"}
        ]
        if not candidates:
            return
        nodes = await self.cluster_nodes()
        for deployment in candidates:
            old_health = (
                deployment.get("health_restart_counts"),
                deployment.get("health_accept_restart_counts"),
            )
            issue = self._cluster_health_issue(deployment, nodes)
            new_health = (
                deployment.get("health_restart_counts"),
                deployment.get("health_accept_restart_counts"),
            )
            if new_health != old_health:
                self._save_deployments()
            if issue:
                await self._recover_cluster_deployment(deployment.get("id"), issue)

    @staticmethod
    def _parse_vllm_capacity_log(text: str) -> list[dict]:
        """Extract vLLM's full-context KV concurrency reports from logs."""
        reports = []
        for match in VLLM_MAX_CONCURRENCY_RE.finditer(text or ""):
            try:
                context_tokens = int(match.group(1).replace(",", ""))
                maximum = float(match.group(2))
            except (TypeError, ValueError):
                continue
            if context_tokens > 0 and math.isfinite(maximum) and maximum >= 0:
                cache_matches = list(
                    VLLM_GPU_KV_CACHE_RE.finditer((text or "")[:match.start()])
                )
                gpu_kv_cache_tokens = None
                if cache_matches:
                    try:
                        gpu_kv_cache_tokens = int(
                            cache_matches[-1].group(1).replace(",", "")
                        )
                    except (TypeError, ValueError):
                        pass
                reports.append({
                    "context_tokens": context_tokens,
                    "maximum_concurrency": maximum,
                    "safe_concurrency": math.floor(maximum),
                    "gpu_kv_cache_tokens": gpu_kv_cache_tokens,
                })
        return reports

    async def _deployment_capacity_reports(
        self, deployment: dict,
    ) -> list[dict]:
        members = list(deployment.get("members") or [])
        values = await asyncio.gather(*(
            self._member_action(member, "logs", log_tail=3000)
            for member in members
        ), return_exceptions=True)
        parsed_by_member: list[list[dict]] = []
        for value in values:
            if isinstance(value, Exception) or not isinstance(value, dict):
                parsed_by_member.append([])
            else:
                parsed_by_member.append(
                    self._parse_vllm_capacity_log(value.get("logs") or "")
                )

        deployment_id = str(deployment.get("id") or "")
        attempted = getattr(self, "_capacity_deep_scan_attempted", None)
        if attempted is None:
            attempted = self._capacity_deep_scan_attempted = set()
        if deployment_id not in attempted and not all(parsed_by_member):
            attempted.add(deployment_id)
            missing = [
                index for index, parsed in enumerate(parsed_by_member)
                if not parsed
            ]
            deep_values = await asyncio.gather(*(
                self._member_action(members[index], "logs", log_tail=100_000)
                for index in missing
            ), return_exceptions=True)
            for index, value in zip(missing, deep_values):
                if isinstance(value, Exception) or not isinstance(value, dict):
                    continue
                parsed_by_member[index] = self._parse_vllm_capacity_log(
                    value.get("logs") or ""
                )

        reports = []
        for member, parsed in zip(members, parsed_by_member):
            # A restarted container can only have one current Docker log, but
            # tolerate duplicate lines by keeping its most recent report.
            if parsed:
                reports.append({
                    **parsed[-1],
                    "rank": member.get("rank"),
                    "node_id": member.get("node_id"),
                    "node_name": member.get("node_name"),
                })
        return reports

    async def _auto_reduce_vllm_concurrency(
        self, deployment_id: str, safe_limit: int, observation: dict,
    ) -> None:
        """Persist a lower --max-num-seqs and rebuild every rank once."""
        async with self._cluster_action_lock():
            deployment = self._deployment(deployment_id)
            if (
                not deployment
                or deployment.get("engine") != "vllm"
                or deployment.get("status") in {"stopped", "error", "launching"}
            ):
                return
            settings = copy.deepcopy(deployment.get("launch_settings") or {})
            if not settings:
                observation["status"] = "could_not_adjust"
                observation["error"] = "deployment has no saved launch settings"
                deployment["kv_capacity"] = observation
                self._save_deployments()
                return
            controls = self._deployment_launch_controls(settings)
            configured = controls.get("max_concurrency")
            if configured is None or int(configured) <= safe_limit:
                observation["status"] = "within_limit"
                deployment["kv_capacity"] = observation
                self._save_deployments()
                return

            previous_limit = int(configured)
            controls["max_concurrency"] = safe_limit
            settings["extra_args"] = self._apply_deployment_launch_controls(
                list(settings.get("extra_args") or []), "vllm", controls
            )
            adjustment = {
                "from": previous_limit,
                "to": safe_limit,
                "reason": "vLLM reported lower full-context KV capacity",
                "started_at": time.time(),
            }
            observation.update({
                "status": "redeploying",
                "configured_concurrency": previous_limit,
                "adjusted_concurrency": safe_limit,
            })
            deployment.update({
                "launch_settings": settings,
                "settings_dirty": True,
                "status": "capacity_redeploying",
                "kv_capacity": observation,
                "auto_concurrency_adjustment": adjustment,
                "error": None,
            })
            self._save_deployments()
            print(
                f"[vllm-capacity] {deployment_id}: reducing max concurrency "
                f"from {previous_limit} to {safe_limit}; stopping every rank"
            )

            members = list(deployment.get("members") or [])
            member_names = {
                member.get("container_name") for member in members
                if member.get("container_name")
            }
            protected_models = {str(deployment.get("model") or "")}
            try:
                containers = await self.list_containers()
                for container in containers:
                    if container.get("name") in member_names:
                        protected_models.update(self._container_model_ids(container))
            except Exception:
                # The deployment's backing model still protects normal calls;
                # aliases are best-effort if Docker inspection is unavailable.
                pass
            protected_models.discard("")
            guard = getattr(self, "_capacity_redeploying_models", None)
            if guard is None:
                guard = self._capacity_redeploying_models = set()
            guard.update(protected_models)
            try:
                stopped = await asyncio.gather(*(
                    self._member_action(member, "stop") for member in members
                ), return_exceptions=True)
                stop_errors = [
                    f"rank {member.get('rank')}: {result}"
                    for member, result in zip(members, stopped)
                    if isinstance(result, Exception)
                ]
                if stop_errors:
                    observation["status"] = "redeploy_failed"
                    observation["error"] = "could not stop every rank"
                    deployment["status"] = "error"
                    deployment["error"] = (
                        "Automatic KV-capacity adjustment could not stop every rank; "
                        "no rank was restarted: " + "; ".join(stop_errors)
                    )
                    self._save_deployments()
                    return

                deployment["status"] = "stopped"
                self._save_deployments()
                result = await self._deployment_action_locked(deployment_id, "start")
                replacement = (
                    result.get("deployment") if isinstance(result, dict) else None
                )
                if (
                    not isinstance(result, dict)
                    or not result.get("ok")
                    or not isinstance(replacement, dict)
                ):
                    observation["status"] = "redeploy_failed"
                    deployment["kv_capacity"] = observation
                    self._save_deployments()
                    return

                adjustment["completed_at"] = time.time()
                replacement["auto_concurrency_adjustment"] = adjustment
                replacement["kv_capacity"] = {
                    **observation,
                    "status": "awaiting_recheck",
                    "configured_concurrency": safe_limit,
                }
                self._save_deployments()
                print(
                    f"[vllm-capacity] {replacement.get('id')}: redeployed with "
                    f"max concurrency {safe_limit}"
                )
            finally:
                guard.difference_update(protected_models)

    async def _deployment_capacity_tick(self) -> None:
        if not self.settings.get("vllm_auto_adjust_concurrency", True):
            return
        candidates = [
            deployment for deployment in list(self.deployments)
            if deployment.get("engine") == "vllm"
            and deployment.get("members")
            and deployment.get("status") not in {
                "stopped", "error", "launching", "recovering",
                "capacity_redeploying",
            }
            and (deployment.get("kv_capacity") or {}).get("status") not in {
                "within_limit", "reported", "insufficient_for_one",
                "could_not_adjust", "redeploy_failed",
            }
        ]
        for deployment in candidates:
            reports = await self._deployment_capacity_reports(deployment)
            if not reports:
                continue
            settings = deployment.get("launch_settings") or {}
            controls = self._deployment_launch_controls(settings)
            context_window = controls.get("context_window")
            matching = [
                report for report in reports
                if context_window is None
                or report["context_tokens"] == int(context_window)
            ]
            if not matching:
                continue
            maximum = min(
                report["maximum_concurrency"] for report in matching
            )
            safe_limit = math.floor(maximum)
            configured = controls.get("max_concurrency")
            observation = {
                "status": "reported",
                "observed_at": time.time(),
                "context_tokens": matching[0]["context_tokens"],
                "maximum_concurrency": maximum,
                "safe_concurrency": safe_limit,
                "configured_concurrency": configured,
                "reports": matching,
            }
            if configured is None:
                deployment["kv_capacity"] = observation
                self._save_deployments()
                continue
            if safe_limit < 1:
                observation["status"] = "insufficient_for_one"
                observation["error"] = (
                    "vLLM reports insufficient KV cache for one full-context request"
                )
                deployment["kv_capacity"] = observation
                self._save_deployments()
                continue
            if safe_limit >= int(configured):
                observation["status"] = "within_limit"
                deployment["kv_capacity"] = observation
                self._save_deployments()
                continue
            await self._auto_reduce_vllm_concurrency(
                deployment.get("id"), safe_limit, observation
            )

    async def _deployment_capacity_monitor_loop(self) -> None:
        while True:
            await asyncio.sleep(VLLM_CAPACITY_SCAN_INTERVAL_SECONDS)
            try:
                await self._deployment_capacity_tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("vLLM KV-capacity monitor failed")

    async def _cluster_health_monitor_loop(self) -> None:
        # Let startup finish before the first probe, then reconcile at the
        # requested two-minute cadence.
        while True:
            await asyncio.sleep(CLUSTER_HEALTH_INTERVAL_SECONDS)
            try:
                await self._cluster_health_tick()
            except Exception as exc:
                print(f"[cluster-health] check failed: {exc}")

    async def deployment_logs(self, deployment_id: str) -> dict:
        deployment = self._deployment(deployment_id)
        if not deployment:
            raise ValueError("deployment not found")
        values = await asyncio.gather(
            *[self._member_action(m, "logs") for m in deployment.get("members", [])],
            return_exceptions=True,
        )
        def member_logs(member: dict, value: Any) -> str:
            if isinstance(value, dict) and value.get("logs"):
                return value["logs"]
            phase = member.get("phase") or {}
            message = phase.get("message") or (
                "Waiting for the node agent to begin launch"
                if member.get("status") in {"queued", "creating"}
                else "No output has been reported yet"
            )
            fallback = f"=== Coordinator launch status ===\n{message}"
            # An older remote agent returns 404 until Docker has created the
            # container. The coordinator status is the useful signal in that
            # case, so keep the transport detail below it instead of replacing
            # the entire member panel with an error.
            if isinstance(value, Exception):
                fallback += f"\n\nAgent log request: {value}"
            return fallback

        return {
            "members": [
                {
                    **member,
                    "logs": member_logs(member, value),
                    "error": None,
                }
                for member, value in zip(deployment.get("members", []), values)
            ]
        }

    async def get_cluster_member_logs(self, name: str, tail: int = 300) -> str:
        """Combine pre-container launch progress with Docker output."""
        launch_text = self._cluster_launch_text(name)
        managed = await self.is_managed_container(name)
        if not launch_text and not managed:
            raise ValueError("cluster member not found")
        sections = [launch_text] if launch_text else []
        if managed:
            try:
                container_logs = await self.get_logs(name, tail)
            except Exception as exc:
                container_logs = f"Could not read container logs: {exc}"
            sections.append("=== Container logs ===\n" + (container_logs or "(no output yet)"))
        elif launch_text:
            sections.append("=== Container logs ===\nContainer has not been created yet.")
        return "\n\n".join(sections)

    async def remove_cluster_member(self, name: str) -> dict:
        if await self.is_managed_container(name):
            return await self.remove_container(name)
        launches = getattr(self, "cluster_member_launches", {})
        if name in launches:
            launches.pop(name, None)
            return {"ok": True}
        return {"ok": True, "already_absent": True}

    # ---------- memory bandwidth (Grace/GB10 SCF PMU) ----------
    # Grace exposes memory traffic on the Coresight SCF PMU
    # (nvidia_scf_pmu_0). Per NVIDIA's measuring-performance guide:
    #   read  BW = CMEM_RD_DATA * 32 / interval   (bytes/s)
    #   write BW = CMEM_WR_TOTAL_BYTES / interval  (bytes/s)
    # We tail `perf stat -a -I 1000` for those two events, parse each
    # 1-second line, and cache the latest values. Best-effort: if perf is
    # absent, lacks permission (perf_event_paranoid), or the events don't
    # exist on this hardware, the monitor stays inactive and the stat
    # surfaces None — the UI then shows "—".
    def _start_mem_bw_monitor(self):
        if self._mem_bw_thread is not None:
            return
        try:
            proc = subprocess.Popen(
                [
                    "perf", "stat", "-a", "-I", "1000",
                    "-e", "nvidia_scf_pmu_0/CMEM_RD_DATA/",
                    "-e", "nvidia_scf_pmu_0/CMEM_WR_TOTAL_BYTES/",
                ],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True,
            )
        except FileNotFoundError:
            # perf not installed — leave the monitor inactive.
            self._stash_bw("error", "perf not installed")
            return
        except Exception as e:
            self._stash_bw("error", f"perf spawn failed: {e}")
            return
        self._mem_bw_proc = proc

        def _reader():
            try:
                if proc.stdout is None:
                    raise RuntimeError("perf provided no stdout stream")
                first = True
                for line in proc.stdout:
                    if first:
                        first = False
                        # perf prints error context to stdout/stderr on
                        # startup (e.g. "<not supported>", event-not-found).
                        # Capture it so the UI can say *why* it's unavailable.
                        ls = line.strip()
                        if ls.startswith("#") or "not supported" in ls.lower() or "not found" in ls.lower():
                            self._stash_bw("error", ls.lstrip("# ").strip()[:160])
                    self._parse_perf_bw_line(line)
            except Exception:
                pass

        t = threading.Thread(target=_reader, daemon=True)
        self._mem_bw_thread = t
        t.start()

    def _parse_perf_bw_line(self, line: str):
        # perf -I 1000 emits lines like:
        #   1.001234567  123456789  nvidia_scf_pmu_0/CMEM_RD_DATA/
        # A leading '#' marks an error/comment line (e.g. "<not counted>").
        line = line.strip()
        if not line or line.startswith("#"):
            return
        parts = line.split(None, 2)
        if len(parts) < 3:
            return
        try:
            # parts[0] = timestamp, parts[1] = count, parts[2] = event name
            count = float(parts[1])
        except ValueError:
            return
        ev = parts[2]
        # perf emits a count over a ~1s interval; the bandwidth is count * scale / 1s.
        if "CMEM_RD_DATA" in ev:
            bps = count * 32.0
            self._stash_bw("read_bps", bps)
        elif "CMEM_WR_TOTAL_BYTES" in ev:
            bps = count * 1.0
            self._stash_bw("write_bps", bps)

    def _stash_bw(self, key: str, bps: float):
        with self._mem_bw_lock:
            rec = self._mem_bw
            rec[key] = bps
            rec["ts"] = time.time()

    def _stop_mem_bw_monitor(self):
        proc = self._mem_bw_proc
        if proc and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass
        self._mem_bw_proc = None
        self._mem_bw_thread = None

    def _read_mem_bw(self) -> dict:
        with self._mem_bw_lock:
            rec = dict(self._mem_bw)
        return {
            "read_bps": rec.get("read_bps"),
            "write_bps": rec.get("write_bps"),
            "ts": rec.get("ts"),
            "error": rec.get("error"),
        }

    @staticmethod
    def _parse_online_users(who_output: str) -> dict:
        """Return unique account names and session count from ``who`` output."""
        names: list[str] = []
        seen: set[str] = set()
        sessions = 0
        for line in who_output.splitlines():
            parts = line.split()
            if not parts:
                continue
            sessions += 1
            username = parts[0]
            if username not in seen:
                seen.add(username)
                names.append(username)
        return {"count": len(names), "names": names, "sessions": sessions}

    def _read_online_users(self) -> dict:
        """Read interactive local/SSH logins, cached briefly for stats polling."""
        now = time.monotonic()
        if now - getattr(self, "_online_users_ts", 0.0) < 5.0:
            return dict(self._online_users_cache)
        try:
            result = subprocess.run(
                ["who"], capture_output=True, text=True, timeout=2, check=True,
            )
            users = self._parse_online_users(result.stdout)
        except Exception as exc:
            users = {
                "count": None,
                "names": [],
                "sessions": 0,
                "error": str(exc),
            }
        self._online_users_cache = users
        self._online_users_ts = now
        return dict(users)

    def _read_fan_state(self) -> dict | None:
        """Read the external Noctua fan state published by FanController.

        Returns None when the daemon is not running or the state file is stale
        (>30 s old), otherwise {rpm, duty_byte, duty_pct, temp, mode, status, ts}.
        """
        try:
            state_root = Path(
                os.environ.get("XDG_STATE_HOME")
                or Path.home() / ".local" / "state"
            )
            path = state_root / "fancontroller" / "state.json"
            if not path.exists():
                return None
            data = json.loads(path.read_text(encoding="utf-8"))
            ts = data.get("ts")
            if ts is None or time.time() - float(ts) > 30:
                return None
            return {
                "rpm": data.get("rpm"),
                "duty_byte": data.get("duty_byte"),
                "duty_pct": data.get("duty_pct"),
                "temp": data.get("temp"),
                "local_temp": data.get("local_temp"),
                "temperature_override": data.get("temperature_override", {}),
                "temperature_override_active": data.get(
                    "temperature_override_active", False,
                ),
                "mode": data.get("mode"),
                "active_settings": data.get("active_settings", {}),
                "status": data.get("status"),
                "max_speed": data.get("max_speed", False),
                "ts": ts,
            }
        except Exception:
            return None

    def _fan_control_path(self) -> Path:
        state_root = Path(
            os.environ.get("XDG_STATE_HOME")
            or Path.home() / ".local" / "state"
        )
        return state_root / "fancontroller" / "control.json"

    def _fan_config_path(self) -> Path:
        return Path.home() / ".config" / "fancontroller" / "config.json"

    def _read_fan_control(self) -> dict:
        try:
            data = json.loads(self._fan_control_path().read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _update_fan_control(
        self,
        updates: dict[str, Any] | None = None,
        remove: tuple[str, ...] = (),
    ) -> dict:
        """Atomically merge fields without clobbering another control type."""
        with self._fan_control_lock:
            path = self._fan_control_path()
            data = self._read_fan_control()
            for key in remove:
                data.pop(key, None)
            data.update(updates or {})
            path.parent.mkdir(parents=True, exist_ok=True)
            if not data:
                path.unlink(missing_ok=True)
                return {}
            tmp = path.with_suffix(".json.tmp")
            try:
                tmp.write_text(json.dumps(data), encoding="utf-8")
                os.replace(tmp, path)
            finally:
                tmp.unlink(missing_ok=True)
            return data

    @staticmethod
    def _cluster_temperature_override(
        nodes: list[dict], now: float | None = None,
    ) -> dict | None:
        """Return metadata for the hottest fresh CPU/GPU sensor in a cluster."""
        current_time = time.time() if now is None else float(now)
        hottest: tuple[float, dict] | None = None
        for node in nodes:
            if not node.get("online"):
                continue
            stats = node.get("stats") or {}
            observed_at = stats.get("ts")
            if isinstance(observed_at, bool) or not isinstance(observed_at, (int, float)):
                continue
            observed_at = float(observed_at)
            if (not math.isfinite(observed_at)
                    or current_time - observed_at > FAN_TEMPERATURE_MAX_SAMPLE_AGE_SECONDS):
                continue
            samples: list[tuple[float, str]] = []

            def add(value: Any, sensor: str) -> None:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    return
                value = float(value)
                if math.isfinite(value) and -40 <= value <= 150:
                    samples.append((value, sensor))

            add(stats.get("cpu_temp_c"), "cpu")
            for index, gpu in enumerate(stats.get("gpus") or []):
                if isinstance(gpu, dict):
                    add(gpu.get("temp_c"), f"gpu:{index}")
            if not samples:
                continue
            temperature_c, sensor = max(samples, key=lambda sample: sample[0])
            metadata = {
                "temperature_c": temperature_c,
                "source": "vllm-cluster-max",
                "sensor": sensor,
                "node_id": str(node.get("id") or node.get("node_id") or ""),
                "node_name": str(node.get("name") or node.get("hostname") or "unknown"),
                "observed_at": observed_at,
                "expires_at": current_time + FAN_TEMPERATURE_OVERRIDE_TTL_SECONDS,
            }
            if hottest is None or temperature_c > hottest[0]:
                hottest = (temperature_c, metadata)
        return hottest[1] if hottest else None

    def _set_fan_temperature_override(self, override: dict | None) -> None:
        if override is None:
            self._update_fan_control(remove=("temperature_override",))
        else:
            self._update_fan_control({"temperature_override": override})

    async def _fan_cluster_temperature_tick(self) -> None:
        # Only the controller attached to a live FanController should publish.
        if self._read_fan_state() is None:
            return
        local_stats = await self.get_stats()
        nodes = [{
            "id": LOCAL_NODE_ID,
            "name": self.settings.get("cluster_node_name") or socket.gethostname(),
            "online": True,
            "stats": local_stats,
        }]
        remote = await asyncio.gather(
            *(self.node_registry.probe(node) for node in self.node_registry.nodes),
            return_exceptions=True,
        )
        nodes.extend(status for status in remote if isinstance(status, dict))
        for node in nodes:
            if node.get("id") != LOCAL_NODE_ID:
                self._record_remote_temperature_sample(node)
        self._set_fan_temperature_override(
            self._cluster_temperature_override(nodes),
        )

    async def _fan_cluster_monitor_loop(self) -> None:
        while True:
            try:
                await self._fan_cluster_temperature_tick()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(f"[fan-cluster] temperature sync failed: {exc}")
            await asyncio.sleep(FAN_CLUSTER_SYNC_INTERVAL_SECONDS)

    # ---------- local path auto-mount ----------
    @staticmethod
    def _resolve_local_path(model_path: str) -> str | None:
        """Return the expanded absolute path if *model_path* is a local path,
        otherwise return None.  Handles ``~`` expansion and normalises
        ``//`` to ``/``.
        """
        if not model_path or not model_path.startswith(("/", "~")):
            return None
        expanded = str(Path(model_path).expanduser().resolve())
        return expanded if Path(expanded).is_dir() else None

    def _created_container_model_source(self, container: Any, model: str) -> str:
        """Classify the launched model without exposing its filesystem path."""
        if self._resolve_local_path(model):
            return "local"
        if not str(model or "").strip():
            return "unknown"
        try:
            result = container.exec_run(["test", "-e", "--", str(model)])
            exit_code = getattr(result, "exit_code", None)
            if exit_code is None and isinstance(result, (tuple, list)) and result:
                exit_code = result[0]
        except Exception:
            return "unknown"
        return "local" if exit_code == 0 else "public_repository"

    def _image_hf_cache_target(self, image: str | None) -> str:
        """Return the Hugging Face cache directory declared by *image*.

        Most vLLM images use ``/root/.cache/huggingface``, but custom runtime
        images can set ``HF_HOME`` to another absolute path.  Mounting the
        host cache at the image's declared location keeps offline images able
        to resolve snapshots without overriding their runtime environment.
        """
        fallback = "/root/.cache/huggingface"
        if not image:
            return fallback
        try:
            docker_image = self.client.images.get(image)
            env = ((docker_image.attrs or {}).get("Config") or {}).get("Env") or []
        except Exception:
            return fallback
        for entry in env:
            if not isinstance(entry, str) or not entry.startswith("HF_HOME="):
                continue
            target = entry.partition("=")[2].strip().rstrip("/")
            if target.startswith("/") and target != "":
                return target
        return fallback

    def _build_volumes(
        self,
        model_path: str,
        hf_cache: str | None = None,
        image: str | None = None,
    ) -> dict:
        """Build the ``volumes`` dict for a Docker container.

        Always mounts the HuggingFace cache (if provided).  Additionally, when
        *model_path* is a local directory, mounts it into the container at the
        **same** absolute path so the container can read the model weights
        directly — no downloading, no HF repo ID guessing.
        """
        volumes: dict[str, dict] = {}
        # HF cache (always)
        if hf_cache:
            resolved = str(Path(hf_cache).expanduser().resolve())
            volumes[resolved] = {
                "bind": self._image_hf_cache_target(image),
                "mode": "rw",
            }
        # Local model path
        local = self._resolve_local_path(model_path)
        if local:
            volumes[local] = {"bind": local, "mode": "rw"}
        return volumes

    def _resolved_hf_token(self, explicit: str | None = None) -> str:
        """Resolve an HF credential without exposing it through public state."""
        # A controller-supplied value is authoritative, including an empty
        # value. Remote workers must never silently substitute a different
        # local account when the controller has no credential configured.
        if explicit is not None:
            return explicit.strip() if isinstance(explicit, str) else ""
        candidates = [
            self.settings.get("hf_token"),
            os.environ.get("HF_TOKEN"),
            os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        ]
        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        cache = self.settings.get("hf_cache")
        if cache:
            try:
                token = (Path(cache).expanduser() / "token").read_text(encoding="utf-8").strip()
                if token:
                    return token
            except OSError:
                pass
        return ""

    def _redact_hf_secret(self, value: Any, explicit: str | None = None) -> str:
        redacted = str(value)
        token = self._resolved_hf_token(explicit)
        if token:
            redacted = redacted.replace(token, "[REDACTED]")
        return redacted

    def _container_hf_environment(self, explicit: str | None = None) -> dict[str, str]:
        token = self._resolved_hf_token(explicit)
        if not token:
            return {}
        # New huggingface_hub versions prefer HF_TOKEN; older Transformers
        # integrations still inspect HUGGING_FACE_HUB_TOKEN.
        return {"HF_TOKEN": token, "HUGGING_FACE_HUB_TOKEN": token}

    def _read_fan_config(self) -> dict:
        path = self._fan_config_path()
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise FanSettingsConflict("FanController config is unavailable") from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("FanController config is invalid") from exc
        if not isinstance(current, dict):
            raise ValueError("FanController config is not an object")
        return current

    def get_fan_settings(self) -> dict:
        """Return the safe, mode-specific subset used by the web editor."""
        with self._fan_settings_lock:
            state = self._read_fan_state()
            if state is None:
                raise FanSettingsConflict("FanController state is unavailable")
            current = self._read_fan_config()
            settings: dict[str, dict] = {}
            for mode, defaults in FAN_MODE_DEFAULTS.items():
                settings[mode] = {
                    key: current.get(key, default)
                    for key, default in defaults.items()
                }
            return {"mode": state.get("mode"), "settings": settings}

    @staticmethod
    def _validate_fan_settings(mode: str, updates: Any) -> dict:
        fields = {
            "curve": {"curve_points", "curve_min_temp", "curve_max_temp", "min_floor_pct"},
            "pid": {"setpoint", "kp", "ki", "kd", "min_floor_pct"},
            "hysteresis": {"hyst_on_temp", "hyst_off_temp"},
            "manual": {"manual_duty_pct"},
        }
        if not isinstance(mode, str) or mode not in fields:
            raise ValueError("unknown fan mode")
        if not isinstance(updates, dict):
            raise ValueError("active_settings must be an object")
        supplied = set(updates)
        expected = fields[mode]
        if supplied != expected:
            unknown = sorted(supplied - expected)
            missing = sorted(expected - supplied)
            details = []
            if unknown:
                details.append("unknown: " + ", ".join(unknown))
            if missing:
                details.append("missing: " + ", ".join(missing))
            raise ValueError("invalid active settings (" + "; ".join(details) + ")")

        def number(name: str, low: float, high: float) -> float:
            value = updates[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a number")
            value = float(value)
            if not math.isfinite(value) or value < low or value > high:
                raise ValueError(f"{name} must be between {low:g} and {high:g}")
            return value

        if mode == "curve":
            lo = number("curve_min_temp", 0, 90)
            hi = number("curve_max_temp", 40, 120)
            floor = number("min_floor_pct", 0, 100)
            if hi <= lo:
                raise ValueError("curve_max_temp must be greater than curve_min_temp")
            raw_points = updates["curve_points"]
            if not isinstance(raw_points, list) or len(raw_points) < 2:
                raise ValueError("curve_points must contain at least two points")
            points: list[list[float]] = []
            previous_temp: float | None = None
            for index, point in enumerate(raw_points):
                if not isinstance(point, (list, tuple)) or len(point) != 2:
                    raise ValueError(f"curve point {index + 1} must be [temperature, duty]")
                temp, duty = point
                if (isinstance(temp, bool) or isinstance(duty, bool)
                        or not isinstance(temp, (int, float))
                        or not isinstance(duty, (int, float))):
                    raise ValueError(f"curve point {index + 1} must contain numbers")
                temp, duty = float(temp), float(duty)
                if not math.isfinite(temp):
                    raise ValueError(f"curve point {index + 1} temperature must be finite")
                if not math.isfinite(duty) or duty < 0 or duty > 100:
                    raise ValueError(f"curve point {index + 1} duty must be between 0 and 100")
                if previous_temp is not None and temp <= previous_temp:
                    raise ValueError("curve point temperatures must be unique and increasing")
                points.append([temp, duty])
                previous_temp = temp
            return {
                "curve_points": points,
                "curve_min_temp": lo,
                "curve_max_temp": hi,
                "min_floor_pct": floor,
            }
        if mode == "pid":
            return {
                "setpoint": number("setpoint", 30, 100),
                "kp": number("kp", 0, 50),
                "ki": number("ki", 0, 10),
                "kd": number("kd", 0, 20),
                "min_floor_pct": number("min_floor_pct", 0, 100),
            }
        if mode == "hysteresis":
            on_temp = number("hyst_on_temp", 30, 110)
            off_temp = number("hyst_off_temp", 20, 100)
            if off_temp >= on_temp:
                raise ValueError("hyst_off_temp must be lower than hyst_on_temp")
            return {"hyst_on_temp": on_temp, "hyst_off_temp": off_temp}
        return {"manual_duty_pct": number("manual_duty_pct", 0, 100)}

    def update_fan_settings(
        self, mode: str, updates: Any, expected_mode: str | None = None,
    ) -> dict:
        """Atomically update one mode and optionally make it the active mode."""
        validated = self._validate_fan_settings(mode, updates)
        expected_mode = mode if expected_mode is None else expected_mode
        if expected_mode not in FAN_MODE_DEFAULTS:
            raise ValueError("unknown expected fan mode")
        with self._fan_settings_lock:
            state = self._read_fan_state()
            if state is None:
                raise FanSettingsConflict("FanController state is unavailable")
            if state.get("mode") != expected_mode:
                raise FanSettingsConflict("fan mode changed; refresh and try again")

            path = self._fan_config_path()
            current = self._read_fan_config()
            current.update(validated)
            current["mode"] = mode
            tmp = path.with_suffix(".json.tmp")
            try:
                with tmp.open("w", encoding="utf-8") as handle:
                    json.dump(current, handle, indent=2)
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp, path)
            finally:
                tmp.unlink(missing_ok=True)
        return {
            "mode": mode,
            "previous_mode": expected_mode,
            "active_settings": validated,
        }

    def get_fan_max_speed(self) -> dict:
        """Return whether the external fan max-speed override is active."""
        try:
            path = self._fan_control_path()
            if not path.exists():
                return {"enabled": False}
            data = json.loads(path.read_text(encoding="utf-8"))
            return {"enabled": bool(data.get("max_speed", False))}
        except Exception:
            return {"enabled": False}

    def set_fan_max_speed(self, enabled: bool) -> dict:
        """Write/clear max speed without discarding cluster temperature."""
        try:
            if enabled:
                self._update_fan_control({"max_speed": True})
            else:
                self._update_fan_control(remove=("max_speed",))
            return {"enabled": enabled}
        except Exception as e:
            return {"error": str(e)}

    # ---------- settings ----------
    def _load_settings(self) -> dict:
        if self.settings_path.exists():
            try:
                data = json.loads(self.settings_path.read_text())
                # Removed runtimes must not remain advertised through a stale
                # persisted setting after upgrade.
                data.pop("ollama_base_url", None)
                # Cluster usage is always combined. Discard the retired
                # opt-out so an older settings file cannot silently leave a
                # node out of Usage totals.
                data.pop("sync_token_usage", None)
                return {**DEFAULT_SETTINGS, **data}
            except Exception:
                pass
        return dict(DEFAULT_SETTINGS)

    def _save_settings(self):
        _atomic_private_json_write(self.settings_path, self.settings)

    def public_settings(self) -> dict:
        public = {k: v for k, v in self.settings.items() if k != "hf_token"}
        public["hf_token"] = ""
        public["hf_token_configured"] = bool(self._resolved_hf_token())
        return public

    async def clear_hf_token(self) -> dict:
        async with self.lock:
            self.settings["hf_token"] = ""
            self._save_settings()
        return self.public_settings()

    async def update_settings(self, data: dict) -> dict:
        async with self.lock:
            for k, v in data.items():
                if k in DEFAULT_SETTINGS:
                    # The UI sends an empty password field when an existing
                    # token should remain unchanged.
                    if k == "hf_token" and not str(v or "").strip():
                        continue
                    self.settings[k] = v
            self._save_settings()
        if self.settings.get("virtual_nas_enabled") and not self.is_joined_worker():
            self.virtual_nas.start()
        elif not self.settings.get("virtual_nas_enabled"):
            await self.virtual_nas.stop()
        return self.public_settings()

    # ---------- lifetime token stats ----------
    # Counters are keyed by model id plus a variant tag (quant/dtype) so the
    # same repo served at Q4 vs bf16 gets separate entries, e.g.
    # "Qwen/Qwen3-8B [awq]" vs "Qwen/Qwen3-8B [bfloat16]". Persisted on every update so they survive
    # restarts; only the reset endpoint clears them.
    #
    # Speed fields (gen_tokens / gen_time_s) are retained for compatibility.
    # The Usage table uses persisted request intervals instead: its rolling
    # 1M-output-token rate divides output tokens by the union of decode intervals,
    # so overlapping streams contribute their combined throughput.
    def _load_token_stats(self) -> dict[str, dict]:
        if self.token_stats_path.exists():
            try:
                data = json.loads(self.token_stats_path.read_text())
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
        return {}

    def _save_token_stats(self):
        self.token_stats_path.write_text(json.dumps(self.token_stats, indent=2))

    # Token-usage synchronization stores one authoritative component per
    # controller node. Peers exchange the whole bounded ledger and select the
    # newest revision for each origin, rather than adding aggregate totals.
    # This makes pull/push fan-out idempotent and prevents replication loops.
    @staticmethod
    def _usage_version_key(value: Any) -> tuple[int, str]:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            return (0, "")
        try:
            counter = max(0, int(value[0]))
        except (TypeError, ValueError):
            counter = 0
        return counter, str(value[1] or "")

    def _local_usage_node_id(self) -> str:
        credentials = getattr(self, "agent_credentials", None)
        return str(getattr(credentials, "node_id", None) or LOCAL_NODE_ID)

    @staticmethod
    def _prune_hourly_usage(
        hourly: Any, now: datetime | None = None,
    ) -> bool:
        """Retain the UTC history needed by the one-year activity view."""
        if not isinstance(hourly, dict):
            return False
        current = now or datetime.now(timezone.utc)
        cutoff = (current - timedelta(days=USAGE_HOURLY_RETENTION_DAYS)).replace(
            minute=0, second=0, microsecond=0, tzinfo=None,
        )
        changed = False
        for hour_key in list(hourly):
            try:
                hour = datetime.strptime(str(hour_key), "%Y-%m-%dT%H")
            except ValueError:
                hourly.pop(hour_key, None)
                changed = True
                continue
            if hour < cutoff:
                hourly.pop(hour_key, None)
                changed = True
        return changed

    @classmethod
    def _prune_usage_ledger(cls, ledger: Any) -> bool:
        if not isinstance(ledger, dict):
            return False
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=USAGE_HOURLY_RETENTION_DAYS)
        ).replace(minute=0, second=0, microsecond=0).strftime("%Y-%m-%dT%H")
        if ledger.get("hourly_retention_cutoff") == cutoff:
            return False
        for component in (ledger.get("origins") or {}).values():
            if not isinstance(component, dict):
                continue
            cls._prune_hourly_usage(component.get("hourly_token_stats"))
        ledger["hourly_retention_cutoff"] = cutoff
        return True

    def _load_token_usage_sync(self) -> dict:
        path = self.token_usage_sync_path
        if path.exists():
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                if (
                    isinstance(value, dict)
                    and value.get("version") == 1
                    and isinstance(value.get("origins"), dict)
                ):
                    value.setdefault("epoch", [0, ""])
                    value.setdefault("model_epochs", {})
                    local = (value.get("origins") or {}).get(
                        self._local_usage_node_id()
                    )
                    if isinstance(local, dict):
                        local.setdefault(
                            "speed_samples",
                            copy.deepcopy(getattr(self, "speed_samples", {})),
                        )
                    self._prune_usage_ledger(value)
                    return value
            except Exception:
                pass

        # Attribute an existing installation's counters to this node exactly
        # once. Subsequent startups load this source ledger instead of treating
        # the already-aggregated compatibility files as new local usage.
        node_id = self._local_usage_node_id()
        models = copy.deepcopy(getattr(self, "token_stats", {}))
        hourly = copy.deepcopy(getattr(self, "hourly_token_stats", {}))
        self._prune_hourly_usage(hourly)
        model_epochs = {model: [0, ""] for model in models}
        return {
            "version": 1,
            "epoch": [0, ""],
            "model_epochs": copy.deepcopy(model_epochs),
            "origins": {
                node_id: {
                    "revision": 0,
                    "token_stats": models,
                    "hourly_token_stats": hourly,
                    "model_epochs": model_epochs,
                    "speed_samples": copy.deepcopy(
                        getattr(self, "speed_samples", {})
                    ),
                },
            },
        }

    def _save_token_usage_sync(self) -> None:
        path = self.token_usage_sync_path
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(self.token_usage_sync, separators=(",", ":")),
            encoding="utf-8",
        )
        temporary.replace(path)

    @staticmethod
    def _sum_usage_records(target: dict, source: Any) -> None:
        if not isinstance(source, dict):
            return
        for field in (
            "input", "output", "cached", "requests",
            "gen_tokens", "gen_time_s", "pp_tokens", "pp_time_s",
        ):
            value = source.get(field)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                target[field] = target.get(field, 0) + max(0, value)

    def _rebuild_synced_token_usage(self) -> None:
        ledger = getattr(self, "token_usage_sync", None)
        if not isinstance(ledger, dict):
            return
        aggregate: dict[str, dict] = {}
        hourly: dict[str, dict[str, dict]] = {}
        current_model_epochs = ledger.get("model_epochs") or {}
        origins = ledger.get("origins") or {}
        for component in origins.values():
            if not isinstance(component, dict):
                continue
            component_epochs = component.get("model_epochs") or {}
            for model, stats in (component.get("token_stats") or {}).items():
                if self._usage_version_key(component_epochs.get(model)) != (
                    self._usage_version_key(current_model_epochs.get(model))
                ):
                    continue
                self._sum_usage_records(aggregate.setdefault(str(model), {}), stats)
            for hour_key, models in (
                component.get("hourly_token_stats") or {}
            ).items():
                if not isinstance(models, dict):
                    continue
                for model, stats in models.items():
                    if self._usage_version_key(component_epochs.get(model)) != (
                        self._usage_version_key(current_model_epochs.get(model))
                    ):
                        continue
                    hour = hourly.setdefault(str(hour_key), {})
                    self._sum_usage_records(hour.setdefault(str(model), {}), stats)
        self.token_stats = aggregate
        self.hourly_token_stats = hourly
        local_component = origins.get(self._local_usage_node_id())
        if isinstance(local_component, dict):
            self.speed_samples = copy.deepcopy(
                local_component.get("speed_samples") or {}
            )

    def _record_local_synced_tokens(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int,
        gen_time_s: float | None,
        pp_time_s: float | None,
        hour_key: str,
        ended_at: float | None = None,
    ) -> bool:
        ledger = getattr(self, "token_usage_sync", None)
        if not isinstance(ledger, dict):
            return False
        node_id = self._local_usage_node_id()
        origins = ledger.setdefault("origins", {})
        component = origins.setdefault(node_id, {
            "revision": 0,
            "token_stats": {},
            "hourly_token_stats": {},
            "model_epochs": {},
            "speed_samples": {},
        })
        global_model_epochs = ledger.setdefault("model_epochs", {})
        model_version = global_model_epochs.setdefault(model, [0, ""])
        component_epochs = component.setdefault("model_epochs", {})
        if self._usage_version_key(component_epochs.get(model)) != (
            self._usage_version_key(model_version)
        ):
            component.setdefault("token_stats", {})[model] = {}
            for models in component.setdefault("hourly_token_stats", {}).values():
                if isinstance(models, dict):
                    models.pop(model, None)
            component.setdefault("speed_samples", {}).pop(model, None)
            component_epochs[model] = list(model_version)

        def add(rec: dict) -> None:
            rec["input"] = rec.get("input", 0) + prompt_tokens
            rec["output"] = rec.get("output", 0) + completion_tokens
            rec["cached"] = rec.get("cached", 0) + cached_tokens
            rec["requests"] = rec.get("requests", 0) + 1
            if gen_time_s and gen_time_s > 0 and completion_tokens > 0:
                rec["gen_tokens"] = rec.get("gen_tokens", 0) + completion_tokens
                rec["gen_time_s"] = rec.get("gen_time_s", 0.0) + gen_time_s
            pp_tokens = max(0, prompt_tokens - cached_tokens)
            if pp_time_s and pp_time_s > 0 and pp_tokens > 0:
                rec["pp_tokens"] = rec.get("pp_tokens", 0) + pp_tokens
                rec["pp_time_s"] = rec.get("pp_time_s", 0.0) + pp_time_s

        add(component.setdefault("token_stats", {}).setdefault(model, {}))
        hour = component.setdefault("hourly_token_stats", {}).setdefault(
            hour_key, {}
        )
        add(hour.setdefault(model, {}))
        self._append_speed_sample(
            component.setdefault("speed_samples", {}),
            model,
            completion_tokens,
            gen_time_s,
            ended_at,
        )
        prune_aggregate = self._prune_usage_ledger(ledger)
        component["revision"] = int(component.get("revision") or 0) + 1
        try:
            self._save_token_usage_sync()
        except Exception:
            pass
        return prune_aggregate

    def token_usage_sync_snapshot(self) -> dict:
        if self._prune_usage_ledger(self.token_usage_sync):
            self._rebuild_synced_token_usage()
            self._save_token_usage_sync()
            self._save_hourly_token_stats()
        return {
            "enabled": True,
            "ledger": copy.deepcopy(self.token_usage_sync),
        }

    def _visible_usage_component(self, ledger: dict, component: Any) -> dict | None:
        if not isinstance(component, dict):
            return None
        visible = copy.deepcopy(component)
        visible.setdefault("token_stats", {})
        visible.setdefault("hourly_token_stats", {})
        visible.setdefault("speed_samples", {})
        component_epochs = visible.setdefault("model_epochs", {})
        global_epochs = ledger.get("model_epochs") or {}
        models = set(visible["token_stats"]) | set(visible["speed_samples"])
        for hourly_models in visible["hourly_token_stats"].values():
            if isinstance(hourly_models, dict):
                models.update(str(model) for model in hourly_models)
        for model in models:
            if self._usage_version_key(component_epochs.get(model)) == (
                self._usage_version_key(global_epochs.get(model))
            ):
                continue
            visible["token_stats"].pop(model, None)
            visible["speed_samples"].pop(model, None)
            for hourly_models in visible["hourly_token_stats"].values():
                if isinstance(hourly_models, dict):
                    hourly_models.pop(model, None)
            component_epochs.pop(model, None)
        for hour_key in list(visible["hourly_token_stats"]):
            if not visible["hourly_token_stats"][hour_key]:
                visible["hourly_token_stats"].pop(hour_key, None)
        return visible

    @staticmethod
    def _usage_component_models(component: dict) -> set[str]:
        models = set(str(model) for model in (component.get("token_stats") or {}))
        models.update(str(model) for model in (component.get("speed_samples") or {}))
        for hourly_models in (component.get("hourly_token_stats") or {}).values():
            if isinstance(hourly_models, dict):
                models.update(str(model) for model in hourly_models)
        return models

    @staticmethod
    def _usage_component_revision(component: Any) -> int:
        try:
            return max(0, int((component or {}).get("revision") or 0))
        except (AttributeError, TypeError, ValueError):
            return 0

    def reconcile_token_usage_sync(
        self, payload: Any, remote_origin_id: str,
    ) -> None:
        """Join an existing peer without honoring its pre-cluster resets."""
        remote = payload.get("ledger") if isinstance(payload, dict) else None
        if (
            not isinstance(remote, dict)
            or remote.get("version") != 1
            or not isinstance(remote.get("origins"), dict)
        ):
            raise ValueError("invalid token usage sync payload")
        remote = copy.deepcopy(remote)
        self._prune_usage_ledger(remote)
        local = self.token_usage_sync
        self._prune_usage_ledger(local)
        origins: dict[str, dict] = {}
        for origin, component in (local.get("origins") or {}).items():
            visible = self._visible_usage_component(local, component)
            if visible is not None:
                origins[str(origin)] = visible
        remote_component = self._visible_usage_component(
            remote, (remote.get("origins") or {}).get(remote_origin_id)
        )
        if remote_component is None:
            raise ValueError("peer usage ledger is missing its local origin")
        current = origins.get(remote_origin_id)
        if current is None or self._usage_component_revision(
            remote_component
        ) >= self._usage_component_revision(current):
            origins[str(remote_origin_id)] = remote_component

        node_id = self._local_usage_node_id()
        epoch_counter = max(
            self._usage_version_key(local.get("epoch"))[0],
            self._usage_version_key(remote.get("epoch"))[0],
        ) + 1
        model_epochs = {
            str(model): list(self._usage_version_key(version))
            for model, version in (local.get("model_epochs") or {}).items()
        }
        component_models = {
            origin: self._usage_component_models(component)
            for origin, component in origins.items()
        }
        active_models: set[str] = set()
        for models in component_models.values():
            active_models.update(models)
        for model in active_models:
            max_counter = self._usage_version_key(model_epochs.get(model))[0]
            for component in origins.values():
                max_counter = max(
                    max_counter,
                    self._usage_version_key(
                        (component.get("model_epochs") or {}).get(model)
                    )[0],
                )
            model_epochs[model] = [max_counter + 1, node_id]
            for origin, component in origins.items():
                if model in component_models[origin]:
                    component.setdefault("model_epochs", {})[model] = list(
                        model_epochs[model]
                    )
        self.token_usage_sync = {
            "version": 1,
            "epoch": [epoch_counter, node_id],
            "model_epochs": model_epochs,
            "origins": origins,
        }
        self._prune_usage_ledger(self.token_usage_sync)
        self._rebuild_synced_token_usage()
        self._save_token_usage_sync()
        self._save_token_stats()
        self._save_hourly_token_stats()

    def _preserve_local_origin_during_reconciliation(self, remote: dict) -> dict:
        local_node_id = self._local_usage_node_id()
        local_component = self._visible_usage_component(
            self.token_usage_sync,
            (self.token_usage_sync.get("origins") or {}).get(local_node_id),
        )
        if local_component is None:
            return remote
        remote_epochs = remote.setdefault("model_epochs", {})
        for model in self._usage_component_models(local_component):
            remote_epochs.setdefault(model, [0, ""])
            local_component.setdefault("model_epochs", {})[model] = list(
                self._usage_version_key(remote_epochs[model])
            )
        remote.setdefault("origins", {})[local_node_id] = local_component
        return remote

    def merge_token_usage_sync(self, payload: Any) -> bool:
        remote = payload.get("ledger") if isinstance(payload, dict) else None
        if (
            not isinstance(remote, dict)
            or remote.get("version") != 1
            or not isinstance(remote.get("origins"), dict)
        ):
            raise ValueError("invalid token usage sync payload")
        remote = copy.deepcopy(remote)
        if payload.get("reconcile_origin") == self._local_usage_node_id():
            remote = self._preserve_local_origin_during_reconciliation(remote)
        self._prune_usage_ledger(remote)
        local = self.token_usage_sync
        changed = self._prune_usage_ledger(local)
        local_epoch = self._usage_version_key(local.get("epoch"))
        remote_epoch = self._usage_version_key(remote.get("epoch"))
        if remote_epoch > local_epoch:
            self.token_usage_sync = copy.deepcopy(remote)
            local = self.token_usage_sync
            changed = True
        elif remote_epoch < local_epoch:
            if changed:
                self._rebuild_synced_token_usage()
                self._save_token_usage_sync()
                self._save_hourly_token_stats()
            return changed
        else:
            local_model_epochs = local.setdefault("model_epochs", {})
            for model, version in (remote.get("model_epochs") or {}).items():
                if self._usage_version_key(version) > self._usage_version_key(
                    local_model_epochs.get(model)
                ):
                    local_model_epochs[str(model)] = list(version)
                    changed = True

            local_origins = local.setdefault("origins", {})
            for origin, component in remote["origins"].items():
                if not isinstance(component, dict):
                    continue
                try:
                    remote_revision = max(0, int(component.get("revision") or 0))
                except (TypeError, ValueError):
                    continue
                current = local_origins.get(origin)
                try:
                    local_revision = max(
                        0, int((current or {}).get("revision") or 0)
                    )
                except (TypeError, ValueError):
                    local_revision = 0
                if current is None or remote_revision > local_revision:
                    local_origins[str(origin)] = copy.deepcopy(component)
                    changed = True

        # After adopting a cluster reset, retain an empty component for this
        # node so its next local request joins the new epoch cleanly.
        local_node_id = self._local_usage_node_id()
        origins = local.setdefault("origins", {})
        if local_node_id not in origins:
            origins[local_node_id] = {
                "revision": 0,
                "token_stats": {},
                "hourly_token_stats": {},
                "model_epochs": {},
                "speed_samples": {},
            }
            changed = True
        if changed:
            self._rebuild_synced_token_usage()
            self._save_token_usage_sync()
            self._save_token_stats()
            self._save_hourly_token_stats()
        return changed

    def _load_speed_samples(self) -> dict[str, list[dict]]:
        if self.speed_samples_path.exists():
            try:
                data = json.loads(self.speed_samples_path.read_text())
                if isinstance(data, dict):
                    return {
                        str(model): samples
                        for model, samples in data.items()
                        if isinstance(samples, list)
                    }
            except Exception:
                pass
        return {}

    def _write_speed_samples_snapshot(self, snapshot: dict) -> None:
        tmp = self.speed_samples_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(snapshot, separators=(",", ":")))
        tmp.replace(self.speed_samples_path)

    def _save_speed_samples(self) -> None:
        self._write_speed_samples_snapshot(copy.deepcopy(self.speed_samples))

    async def _flush_speed_samples_later(self) -> None:
        """Persist request intervals in bounded batches off the event loop."""
        await asyncio.sleep(1.0)
        while True:
            version = self._speed_samples_version
            snapshot = copy.deepcopy(self.speed_samples)
            await asyncio.to_thread(self._write_speed_samples_snapshot, snapshot)
            if version == self._speed_samples_version:
                return
            # Coalesce requests that arrived while the previous snapshot was
            # being encoded/written instead of starting one write per request.
            await asyncio.sleep(0.1)

    def _queue_speed_samples_save(self) -> None:
        self._speed_samples_version = getattr(
            self, "_speed_samples_version", 0
        ) + 1
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # Synchronous maintenance/tests still get durable writes.
            self._save_speed_samples()
            return
        task = getattr(self, "_speed_samples_flush_task", None)
        if task is None or task.done():
            self._speed_samples_flush_task = loop.create_task(
                self._flush_speed_samples_later()
            )

    @staticmethod
    def _append_speed_sample(
        target: dict[str, list[dict]],
        model: str,
        completion_tokens: int,
        gen_time_s: float | None,
        ended_at: float | None = None,
    ) -> None:
        if not gen_time_s or gen_time_s <= 0 or completion_tokens <= 0:
            return
        sample_ended_at = ended_at if ended_at is not None else time.time()
        samples = target.setdefault(model, [])
        samples.append({
            "tokens": completion_tokens,
            "started_at": sample_ended_at - float(gen_time_s),
            "ended_at": sample_ended_at,
        })
        # Each model retains at least the newest 1M output tokens. The
        # event cap prevents pathological one-token requests from growing the
        # metadata file without bound.
        retained = []
        retained_tokens = 0.0
        for sample in reversed(samples):
            retained.append(sample)
            retained_tokens += max(0.0, float(sample.get("tokens") or 0))
            if retained_tokens >= USAGE_SPEED_WINDOW_TOKENS or len(retained) >= 250_000:
                break
        target[model] = list(reversed(retained))

    def _record_speed_sample(
        self,
        model: str,
        completion_tokens: int,
        gen_time_s: float | None,
        ended_at: float | None = None,
    ) -> None:
        self._append_speed_sample(
            self.speed_samples, model, completion_tokens, gen_time_s, ended_at
        )

    def rolling_generation_speed(self, models: list[str]) -> dict:
        """Aggregate per-origin decode rates without comparing host clocks.

        Intervals are unioned within each node's own clock domain. The node
        rates are then summed, so wall-clock skew between hosts cannot turn
        concurrent cluster work into apparently sequential requests.
        """
        sample_maps: list[dict] = []
        ledger = getattr(self, "token_usage_sync", None)
        if isinstance(ledger, dict):
            global_epochs = ledger.get("model_epochs") or {}
            for component in (ledger.get("origins") or {}).values():
                if not isinstance(component, dict):
                    continue
                component_epochs = component.get("model_epochs") or {}
                eligible = {
                    model: values
                    for model, values in (component.get("speed_samples") or {}).items()
                    if self._usage_version_key(component_epochs.get(model))
                    == self._usage_version_key(global_epochs.get(model))
                }
                sample_maps.append(eligible)
        if not any(mapping.get(model) for mapping in sample_maps for model in models):
            sample_maps = [getattr(self, "speed_samples", {})]
        origin_speeds = []
        for sample_map in sample_maps:
            samples = []
            for model in models:
                for raw in sample_map.get(model, []):
                    try:
                        tokens = float(raw.get("tokens") or 0)
                        started_at = float(raw.get("started_at"))
                        ended_at = float(raw.get("ended_at"))
                    except (AttributeError, TypeError, ValueError):
                        continue
                    if tokens <= 0 or ended_at <= started_at:
                        continue
                    samples.append((ended_at, started_at, tokens))
            speed = self._rolling_speed_for_samples(samples)
            if speed["tok_s"] is not None:
                origin_speeds.append(speed)

        if not origin_speeds:
            return {"tokens": 0, "active_time_s": 0.0, "tok_s": None}
        selected_tokens = sum(speed["tokens"] for speed in origin_speeds)
        aggregate_rate = sum(speed["tok_s"] for speed in origin_speeds)
        return {
            "tokens": selected_tokens,
            "active_time_s": (
                selected_tokens / aggregate_rate if aggregate_rate > 0 else 0.0
            ),
            "tok_s": aggregate_rate or None,
        }

    @staticmethod
    def _rolling_speed_for_samples(samples: list[tuple[float, float, float]]) -> dict:
        if not samples:
            return {"tokens": 0, "active_time_s": 0.0, "tok_s": None}

        # Sweep backward through the piecewise-constant aggregate decode
        # rate.  This cuts the 1M-token boundary through all streams active at
        # that instant, rather than selecting whole requests serially (which
        # under-counted concurrent throughput at the cutoff).
        deltas: dict[float, float] = {}
        for ended_at, started_at, tokens in samples:
            rate = tokens / (ended_at - started_at)
            deltas[started_at] = deltas.get(started_at, 0.0) + rate
            deltas[ended_at] = deltas.get(ended_at, 0.0) - rate
        boundaries = sorted(deltas, reverse=True)
        selected_tokens = 0.0
        active_time = 0.0
        aggregate_rate = 0.0
        for index, upper in enumerate(boundaries[:-1]):
            aggregate_rate -= deltas[upper]
            lower = boundaries[index + 1]
            if aggregate_rate <= 0 or upper <= lower:
                continue
            interval_tokens = aggregate_rate * (upper - lower)
            remaining = USAGE_SPEED_WINDOW_TOKENS - selected_tokens
            used_tokens = min(interval_tokens, remaining)
            selected_tokens += used_tokens
            active_time += used_tokens / aggregate_rate
            if selected_tokens >= USAGE_SPEED_WINDOW_TOKENS:
                break
        return {
            "tokens": int(selected_tokens),
            "active_time_s": active_time,
            "tok_s": selected_tokens / active_time if active_time > 0 else None,
        }

    def _load_hourly_token_stats(self) -> dict[str, dict]:
        if self.hourly_token_stats_path.exists():
            try:
                data = json.loads(self.hourly_token_stats_path.read_text())
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
        return {}

    def _save_hourly_token_stats(self):
        self.hourly_token_stats_path.write_text(
            json.dumps(self.hourly_token_stats, indent=2)
        )

    @staticmethod
    def _stats_key(model: str, variant: str = "") -> str:
        return f"{model} [{variant}]" if variant else model

    @staticmethod
    def _variant_from_cmd(cmd: list) -> str:
        """Short quant/dtype descriptor from a vLLM or SGLang command line.
        Explicit "auto" values are dropped — they mean "no override" and
        would otherwise split stats for identical configs."""
        parts = []
        for flag in ("--quantization", "--dtype", "--kv-cache-dtype"):
            if flag in cmd:
                i = cmd.index(flag)
                if i + 1 < len(cmd) and cmd[i + 1].lower() != "auto":
                    parts.append(cmd[i + 1])
        return "+".join(parts)

    @staticmethod
    def _served_models_from_cmd(cmd: list, fallback: str = "") -> list[str]:
        """Return the public model ids configured on an inference server.

        vLLM accepts one or more values after ``--served-model-name``.  The
        model path/repository used to load weights is not a valid request id
        when that flag is present, so container discovery must keep the two
        concepts separate.
        """
        served: list[str] = []
        for i, token in enumerate(cmd):
            token = str(token)
            if token.startswith("--served-model-name="):
                value = token.split("=", 1)[1]
                if value:
                    served.append(value)
                continue
            if token != "--served-model-name":
                continue
            j = i + 1
            while j < len(cmd) and not str(cmd[j]).startswith("-"):
                served.append(str(cmd[j]))
                j += 1
        # Preserve command-line order while dropping accidental duplicates.
        served = list(dict.fromkeys(served))
        return served or ([fallback] if fallback else [])

    @classmethod
    def _deployment_served_models(
        cls, deployment: dict, primary_container: dict | None = None,
    ) -> list[str]:
        """Return the API-facing ids for a saved cluster deployment."""
        if primary_container and primary_container.get("status") == "running":
            live = primary_container.get("served_models") or []
            if live:
                return list(dict.fromkeys(str(value) for value in live if value))
        settings = deployment.get("launch_settings") or {}
        return cls._served_models_from_cmd(
            list(settings.get("extra_args") or []),
            str(deployment.get("model") or ""),
        )

    @staticmethod
    def _container_model_ids(container: dict) -> list[str]:
        """All request ids that may be used to select a container."""
        ids = [container.get("model")]
        ids.extend(container.get("served_models") or [])
        return [model_id for model_id in dict.fromkeys(ids) if model_id]

    @staticmethod
    def _upstream_model_id(container: dict, requested_model: str) -> str:
        """Translate a backing model id to the id accepted by its server."""
        served = container.get("served_models") or []
        if requested_model in served:
            return requested_model
        return served[0] if served else requested_model

    def _unsloth_variant(self, model: str) -> str:
        """Quant descriptor from the saved Unsloth load settings for a model."""
        st = self.unsloth_settings.get(model) or {}
        parts = []
        if model.upper().endswith("GGUF") and st.get("gguf_variant"):
            parts.append(str(st["gguf_variant"]))
        if st.get("load_in_4bit"):
            parts.append("4bit")
        return "+".join(parts)

    def _record_tokens(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        gen_time_s: float | None = None,
        cached_tokens: int = 0,
        pp_time_s: float | None = None,
    ):
        if not model:
            return
        try:
            prompt_tokens = int(prompt_tokens or 0)
            completion_tokens = int(completion_tokens or 0)
            cached_tokens = int(cached_tokens or 0)
        except (TypeError, ValueError):
            return
        # Clamp cached to ≤ prompt (defensive: some upstreams report
        # cached_tokens > prompt_tokens in edge cases).
        if cached_tokens > prompt_tokens:
            cached_tokens = prompt_tokens
        hour_key = self._usage_hour_key()
        ended_at = time.time()
        prune_aggregate = self._record_local_synced_tokens(
            model,
            prompt_tokens,
            completion_tokens,
            cached_tokens,
            gen_time_s,
            pp_time_s,
            hour_key,
            ended_at,
        )
        self._record_speed_sample(model, completion_tokens, gen_time_s, ended_at)
        # Update both lifetime and session counters with the same values.
        for stats in (self.token_stats, self.session_token_stats):
            rec = stats.setdefault(model, {})
            rec["input"] = rec.get("input", 0) + prompt_tokens
            rec["output"] = rec.get("output", 0) + completion_tokens
            rec["cached"] = rec.get("cached", 0) + cached_tokens
            rec["requests"] = rec.get("requests", 0) + 1
            # Only count requests with a measured decode window towards the
            # average speed, so prefill time never dilutes it.
            if gen_time_s and gen_time_s > 0 and completion_tokens > 0:
                rec["gen_tokens"] = rec.get("gen_tokens", 0) + completion_tokens
                rec["gen_time_s"] = rec.get("gen_time_s", 0.0) + gen_time_s
            pp_tokens = max(0, prompt_tokens - cached_tokens)
            if pp_time_s and pp_time_s > 0 and pp_tokens > 0:
                rec["pp_tokens"] = rec.get("pp_tokens", 0) + pp_tokens
                rec["pp_time_s"] = rec.get("pp_time_s", 0.0) + pp_time_s
        # Record hourly stats for the Analysis charts.
        hrec = self.hourly_token_stats.setdefault(hour_key, {}).setdefault(model, {})
        hrec["input"] = hrec.get("input", 0) + prompt_tokens
        hrec["output"] = hrec.get("output", 0) + completion_tokens
        hrec["cached"] = hrec.get("cached", 0) + cached_tokens
        hrec["requests"] = hrec.get("requests", 0) + 1
        if gen_time_s and gen_time_s > 0 and completion_tokens > 0:
            hrec["gen_tokens"] = hrec.get("gen_tokens", 0) + completion_tokens
            hrec["gen_time_s"] = hrec.get("gen_time_s", 0.0) + gen_time_s
        pp_tokens = max(0, prompt_tokens - cached_tokens)
        if pp_time_s and pp_time_s > 0 and pp_tokens > 0:
            hrec["pp_tokens"] = hrec.get("pp_tokens", 0) + pp_tokens
            hrec["pp_time_s"] = hrec.get("pp_time_s", 0.0) + pp_time_s
        if prune_aggregate:
            self._prune_hourly_usage(self.hourly_token_stats)
        try:
            self._save_token_stats()
        except Exception:
            pass
        try:
            self._save_hourly_token_stats()
        except Exception:
            pass
        try:
            self._queue_speed_samples_save()
        except Exception:
            pass

    @staticmethod
    def _usage_hour_key() -> str:
        """Return a timezone-independent cluster history bucket."""
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H")

    @staticmethod
    def _usage_prompt_counts(usage: dict) -> tuple[int, int]:
        try:
            prompt_tokens = max(0, int(usage.get("prompt_tokens") or 0))
        except (TypeError, ValueError):
            prompt_tokens = 0
        details = usage.get("prompt_tokens_details")
        try:
            cached_tokens = max(
                0, int((details or {}).get("cached_tokens") or 0)
            ) if isinstance(details, dict) else 0
        except (TypeError, ValueError):
            cached_tokens = 0
        return prompt_tokens, min(prompt_tokens, cached_tokens)

    @staticmethod
    def _usage_has_cached_prompt_tokens(usage: dict) -> bool:
        """Whether usage contains an authoritative cached-prompt count.

        vLLM continuous usage snapshots omit ``prompt_tokens_details`` until
        the terminal chunk. Missing detail must not be interpreted as zero
        cache hits or a mostly-cached long context produces an impossible PP
        rate from the full prompt token count.
        """
        details = usage.get("prompt_tokens_details")
        if not isinstance(details, dict) or "cached_tokens" not in details:
            return False
        try:
            return int(details["cached_tokens"]) >= 0
        except (TypeError, ValueError):
            return False

    def _record_usage(
        self,
        model: str,
        usage: dict | None,
        gen_time_s: float | None = None,
        pp_time_s: float | None = None,
    ):
        """Record an OpenAI-style usage object {prompt_tokens, completion_tokens}.

        Also extracts cached_tokens from prompt_tokens_details when present
        (OpenAI / vLLM prefix-cache hits).
        """
        if not usage:
            return
        prompt_tokens, cached = self._usage_prompt_counts(usage)
        self._record_tokens(
            model,
            prompt_tokens,
            usage.get("completion_tokens") or 0,
            gen_time_s,
            cached_tokens=cached,
            pp_time_s=pp_time_s,
        )

    @staticmethod
    def _usage_from_sse_line(line: str) -> dict | None:
        """Extract a usage object from an SSE `data:` line, or None.
        Cheap string pre-check so we don't JSON-parse every streamed chunk."""
        if not line.startswith("data:") or '"usage"' not in line:
            return None
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            return None
        try:
            obj = json.loads(payload)
        except Exception:
            return None
        return obj.get("usage") if isinstance(obj, dict) else None

    @staticmethod
    def _sse_chunk_token_counts(line: str) -> tuple[int, int]:
        """Return exact ``(thinking, output)`` token counts for an SSE chunk.

        vLLM's MTP/speculative decoding can emit several token ids in one
        event.  Counting events therefore under-reports throughput by the
        speculative acceptance length.  Older servers may omit ``token_ids``;
        for those, fall back to one token per non-empty delta.
        """
        if not line.startswith("data:") or '"choices"' not in line:
            return 0, 0
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            return 0, 0
        try:
            obj = json.loads(payload)
        except Exception:
            return 0, 0
        if not isinstance(obj, dict):
            return 0, 0
        thinking_tokens = output_tokens = 0
        for ch in obj.get("choices") or []:
            if not isinstance(ch, dict):
                continue
            delta = ch.get("delta") or {}
            thinking = bool(
                delta.get("reasoning_content") or delta.get("reasoning")
            )
            output = bool(delta.get("content") or ch.get("text"))
            token_ids = ch.get("token_ids")
            count = len(token_ids) if isinstance(token_ids, list) else 0
            if count == 0 and (thinking or output):
                count = 1
            # Reasoning and visible content normally arrive in separate
            # chunks. If a backend combines them, count the ids once.
            if thinking:
                thinking_tokens += count
            elif output:
                output_tokens += count
        return thinking_tokens, output_tokens

    # ----- client-disconnect propagation -----
    # uvicorn does not cancel a running handler when the HTTP client goes
    # away, so without these helpers an aborted client leaves llama-server /
    # vLLM generating until the token budget is exhausted. server.py watches
    # req.is_disconnected() and sets an asyncio.Event; the helpers below race
    # upstream work against that event. Aborting the httpx request closes the
    # upstream connection, which is what makes the inference server cancel
    # the task on its side.
    @staticmethod
    async def _await_or_cancel(
        coro,
        cancel: asyncio.Event | None,
        interrupt: asyncio.Event | None = None,
    ):
        """Await a coroutine, aborting it on disconnect or a stream nudge."""
        if cancel is None and interrupt is None:
            return await coro
        t = asyncio.create_task(coro)
        cw = asyncio.create_task(cancel.wait()) if cancel is not None else None
        iw = (
            asyncio.create_task(interrupt.wait())
            if interrupt is not None else None
        )
        watchers = {task for task in (cw, iw) if task is not None}
        done, _ = await asyncio.wait(
            {t, *watchers}, return_when=asyncio.FIRST_COMPLETED,
        )
        for watcher in watchers:
            watcher.cancel()
        if t in done:
            return t.result()
        t.cancel()
        try:
            await t
        except BaseException:
            pass
        if cw is not None and cw in done:
            raise ClientAbort("client disconnected")
        if iw is not None and iw in done:
            raise StreamNudge("stream selected for transparent replay")
        raise ClientAbort("client disconnected")

    @staticmethod
    async def _aiter_lines_cancellable(
        r,
        cancel: asyncio.Event | None,
        interrupt: asyncio.Event | None = None,
    ):
        """Yield lines from an httpx streaming response, stopping promptly
        when `cancel` fires (even during a long prefill with no output). The
        caller's `async with` then closes the response, and closing that
        connection makes the upstream server abort the generation."""
        aiter = r.aiter_lines()
        pending: asyncio.Task | None = None
        try:
            while True:
                if pending is None:
                    pending = asyncio.create_task(aiter.__anext__())
                if cancel is None and interrupt is None:
                    try:
                        line = await pending
                    except StopAsyncIteration:
                        return
                    pending = None
                    yield line
                    continue
                cw = (
                    asyncio.create_task(cancel.wait())
                    if cancel is not None else None
                )
                iw = (
                    asyncio.create_task(interrupt.wait())
                    if interrupt is not None else None
                )
                watchers = {task for task in (cw, iw) if task is not None}
                done, _ = await asyncio.wait(
                    {pending, *watchers}, return_when=asyncio.FIRST_COMPLETED
                )
                for watcher in watchers:
                    watcher.cancel()
                # Events win a simultaneous race with an upstream line. This
                # prevents a first token from slipping downstream after the
                # nudger selected a zero-output stream for safe replay.
                if cw is not None and cw in done:
                    if not pending.done():
                        pending.cancel()
                        try:
                            await pending
                        except BaseException:
                            pass
                    return
                if iw is not None and iw in done:
                    if not pending.done():
                        pending.cancel()
                        try:
                            await pending
                        except BaseException:
                            pass
                    raise StreamNudge(
                        "stream selected for transparent replay"
                    )
                if pending not in done:
                    pending.cancel()
                    try:
                        await pending
                    except BaseException:
                        pass
                    return
                t, pending = pending, None
                try:
                    line = t.result()
                except StopAsyncIteration:
                    return
                yield line
        finally:
            if pending is not None and not pending.done():
                pending.cancel()

    # ----- live request tracking (Tokens widget) -----
    def _track_start(
        self,
        key: str,
        streaming: bool = False,
        admission_target: str | None = None,
        nudge_event: asyncio.Event | None = None,
        deployment_id: str | None = None,
    ) -> int:
        self._req_seq += 1
        rid = self._req_seq
        self._active_reqs[rid] = {
            "key": key, "thinking": deque(), "output": deque(),
            "streaming": streaming,
            "started_at": time.monotonic(),
            "pp_tokens": 0,
            "pp_time_s": 0.0,
            "total_tokens": 0,
            "forwarded_chunks": 0,
            "admission_target": admission_target,
            "nudge_event": nudge_event,
            "deployment_id": deployment_id,
            "paused": False,
        }
        self._mark_deployment_used(deployment_id)
        return rid

    def _track_prompt_processing(
        self, rid: int, prompt_tokens: int, pp_time_s: float,
    ) -> None:
        rec = self._active_reqs.get(rid)
        if rec is None or prompt_tokens <= 0 or pp_time_s <= 0:
            return
        rec["pp_tokens"] = int(prompt_tokens)
        rec["pp_time_s"] = float(pp_time_s)

    def _track_output(
        self, rid: int, ts: float, kind: str = "output", count: int = 1,
    ):
        rec = self._active_reqs.get(rid)
        if rec is not None and count > 0:
            rec["total_tokens"] = rec.get("total_tokens", 0) + count
            timestamps = rec[kind]
            timestamps.extend([ts] * count)
            cutoff = ts - self._trailing_window
            while timestamps and timestamps[0] < cutoff:
                timestamps.popleft()

    def _track_end(self, rid: int):
        rec = self._active_reqs.pop(rid, None)
        if rec:
            self._mark_deployment_used(rec.get("deployment_id"))

    def _mark_deployment_used(self, deployment_id: str | None) -> None:
        """Record cluster inference activity, persisting at a bounded rate."""
        if not deployment_id:
            return
        now = time.time()
        for deployment in getattr(self, "deployments", []):
            if deployment.get("id") != deployment_id:
                continue
            deployment["last_used_at"] = now
            saved_at = getattr(self, "_deployment_last_used_saved_at", None)
            if saved_at is None:
                saved_at = self._deployment_last_used_saved_at = {}
            # Preserve the latest in memory for the live UI, while avoiding a
            # deployments.json write for every chunk or concurrent request.
            if now - float(saved_at.get(deployment_id) or 0) >= 5.0:
                try:
                    self._save_deployments()
                    saved_at[deployment_id] = now
                except Exception:
                    pass
            return

    def active_requests(self) -> dict:
        """Per-model, five-second rolling thinking/output stream rates."""
        now = time.monotonic()
        out: dict[str, dict] = {}
        for rid, rec in list(self._active_reqs.items()):
            # A zero-output stream selected for transparent replay stays in
            # _active_reqs so its downstream connection remains open, but it
            # has released its admission slot and is waiting in the FIFO.
            # Do not report that paused request as running.
            if rec.get("paused"):
                continue
            e = out.setdefault(rec["key"], {
                "connections": 0, "decoded_tokens": 0,
                "thinking_tok_s": 0.0, "output_tok_s": 0.0,
                "pp_tokens": 0, "pp_time_s": 0.0, "pp_measuring": 0,
            })
            e["connections"] += 1
            e["decoded_tokens"] += int(rec.get("total_tokens") or 0)
            if rec.get("pp_tokens") and rec.get("pp_time_s"):
                e["pp_tokens"] += int(rec["pp_tokens"])
                e["pp_time_s"] += float(rec["pp_time_s"])
            else:
                e["pp_measuring"] += 1
            for kind, field in (("thinking", "thinking_tok_s"), ("output", "output_tok_s")):
                timestamps = rec[kind]
                cutoff = now - self._trailing_window
                while timestamps and timestamps[0] < cutoff:
                    timestamps.popleft()
                if timestamps:
                    observed = min(self._trailing_window, max(1.0, now - timestamps[0]))
                    e[field] += len(timestamps) / observed
        # clean up per-model entries when all their streams ended
        active_keys = {rec["key"] for rec in self._active_reqs.values()}
        for key in list(out.keys()):
            if key not in active_keys:
                del out[key]
        for e in out.values():
            e["thinking_tok_s"] = round(e["thinking_tok_s"], 1)
            e["output_tok_s"] = round(e["output_tok_s"], 1)
            e["pp_tok_s"] = (
                round(e["pp_tokens"] / e["pp_time_s"], 1)
                if e["pp_time_s"] > 0
                else None
            )
        admission_running: dict[str, int] = {}
        for admission in self.inference_admission().values():
            model = admission.get("model")
            if not model:
                continue
            e = out.setdefault(model, {
                "connections": 0, "decoded_tokens": 0,
                "thinking_tok_s": 0.0, "output_tok_s": 0.0,
                "pp_tokens": 0, "pp_time_s": 0.0, "pp_measuring": 0,
                "pp_tok_s": None,
            })
            e["queued"] = e.get("queued", 0) + admission["queued"]
            e["admission_limit"] = admission.get(
                "effective_limit", admission["limit"]
            )
            admission_running[model] = (
                admission_running.get(model, 0) + admission["running"]
            )
        # Admission owns the authoritative running count from slot grant until
        # release.  max() includes non-streaming/prefill work that has no live
        # token timestamps without double-counting streams tracked above.
        for model, running in admission_running.items():
            out[model]["connections"] = max(
                out[model]["connections"], running
            )
        return out

    # ----- controller-side inference admission queue -----
    @staticmethod
    def _inference_admission_config(container: dict, model: str) -> tuple[str, int | None, str]:
        """Return a stable target id, configured limit, and telemetry key."""
        target = str(
            container.get("deployment_id")
            or container.get("name")
            or f"port:{container.get('port')}"
        )
        stats_key = str(container.get("stats_key") or model)
        raw_limit = (container.get("load_settings") or {}).get("max_concurrency")
        try:
            limit = int(raw_limit)
        except (TypeError, ValueError):
            limit = None
        if limit is not None and limit <= 0:
            limit = None
        return target, limit, stats_key

    def _admission_store(self) -> dict[str, dict]:
        # Several focused tests construct Manager with __new__. Keep this
        # state lazily initializable so those tests remain lightweight.
        store = getattr(self, "_inference_admission", None)
        if store is None:
            store = self._inference_admission = {}
        return store

    @staticmethod
    def _drain_inference_waiters(state: dict) -> None:
        waiters = state["waiters"]
        limit = state.get("effective_limit", state["limit"])
        while state["running"] < limit and waiters:
            waiter = waiters.popleft()
            future = waiter["future"]
            if future.cancelled() or future.done():
                continue
            waiter["granted"] = True
            state["running"] += 1
            future.set_result(None)

    async def _acquire_inference_slot(
        self, container: dict, model: str, cancel: asyncio.Event | None,
    ) -> str | None:
        """Wait for one FIFO proxy slot without contacting vLLM first."""
        target, limit, stats_key = self._inference_admission_config(container, model)
        if limit is None:
            return None
        if cancel is not None and cancel.is_set():
            raise ClientAbort("client disconnected while queued")

        store = self._admission_store()
        state = store.setdefault(target, {
            "limit": limit,
            "running": 0,
            "model": stats_key,
            "waiters": deque(),
        })
        state["limit"] = limit
        state["model"] = stats_key

        loop = asyncio.get_running_loop()
        waiter = {
            "future": loop.create_future(),
            "created_at": time.monotonic(),
            "granted": False,
        }
        state["waiters"].append(waiter)
        self._drain_inference_waiters(state)

        cancel_waiter: asyncio.Task | None = None
        try:
            if cancel is None:
                await waiter["future"]
            else:
                cancel_waiter = asyncio.create_task(cancel.wait())
                done, _ = await asyncio.wait(
                    {waiter["future"], cancel_waiter},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if cancel_waiter in done:
                    raise ClientAbort("client disconnected while queued")
                await waiter["future"]
            return target
        except BaseException:
            if waiter["granted"]:
                self._release_inference_slot(target)
            else:
                try:
                    state["waiters"].remove(waiter)
                except ValueError:
                    pass
                if not waiter["future"].done():
                    waiter["future"].cancel()
            raise
        finally:
            if cancel_waiter is not None:
                cancel_waiter.cancel()

    def _release_inference_slot(self, target: str | None) -> None:
        if target is None:
            return
        state = self._admission_store().get(target)
        if state is None:
            return
        state["running"] = max(0, state["running"] - 1)
        self._drain_inference_waiters(state)
        if not state["running"] and not state["waiters"]:
            self._admission_store().pop(target, None)
            getattr(self, "_nudger_slow_since", {}).pop(target, None)

    async def _admit_vllm_target(
        self,
        model: str,
        cancel: asyncio.Event | None,
        container: dict | None = None,
        *,
        container_name: str | None = None,
        deployment_id: str | None = None,
    ) -> tuple[dict, str | None]:
        """Acquire admission and re-resolve runtime state before forwarding.

        A queued request may outlive a container restart or port/config
        change. Re-resolving after every grant makes the forwarded URL fresh;
        if the admission identity or limit changed, release the old grant and
        queue against the new target instead.
        """
        current = container or await self._resolve_vllm_target(
            model, container_name=container_name, deployment_id=deployment_id,
        )
        while True:
            key = current.get("stats_key") or model
            admission = await self._acquire_inference_slot(current, key, cancel)
            try:
                fresh = await self._resolve_vllm_target(
                    model, container_name=container_name,
                    deployment_id=deployment_id,
                )
            except BaseException:
                self._release_inference_slot(admission)
                raise
            old_target, old_limit, _ = self._inference_admission_config(
                current, key
            )
            new_target, new_limit, _ = self._inference_admission_config(
                fresh, fresh.get("stats_key") or model
            )
            if old_target == new_target and old_limit == new_limit:
                return fresh, admission
            self._release_inference_slot(admission)
            current = fresh

    def inference_admission(self) -> dict[str, dict]:
        """Public snapshot of running and controller-queued inference."""
        now = time.monotonic()
        out = {}
        for target, state in self._admission_store().items():
            queued = [
                waiter for waiter in state["waiters"]
                if not waiter["future"].done()
            ]
            if not state["running"] and not queued:
                continue
            snapshot = {
                "model": state.get("model"),
                "limit": state["limit"],
                "running": state["running"],
                "queued": len(queued),
                "oldest_wait_seconds": round(
                    max(0.0, now - queued[0]["created_at"]), 1
                ) if queued else 0.0,
            }
            effective_limit = state.get("effective_limit")
            if effective_limit is not None and effective_limit != state["limit"]:
                snapshot["effective_limit"] = effective_limit
            if state.get("nudger"):
                snapshot["nudger"] = dict(state["nudger"])
            out[target] = snapshot
        return out

    def _inference_nudger_tick(self, now: float | None = None) -> None:
        """Detect stalled streams, serialize, then gradually restore capacity.

        Only streams with zero emitted tokens are interrupted and replayed.
        Once a stream has produced any reasoning or visible output, replaying
        it could duplicate or change content, so it is never interrupted.
        A serialized target is held at one slot for ``stall_seconds``, then
        gains one slot per interval until its configured limit is restored.
        Low throughput during recovery sends it back to one slot.
        """
        settings = getattr(self, "settings", DEFAULT_SETTINGS)
        if not settings.get("vllm_nudger_enabled", True):
            return
        try:
            threshold = max(
                0.0, float(settings.get("vllm_nudger_rate_threshold", 5.0))
            )
            stall_seconds = max(
                0.5, float(settings.get("vllm_nudger_stall_seconds", 3.0))
            )
        except (TypeError, ValueError):
            threshold, stall_seconds = 5.0, 3.0
        now = time.monotonic() if now is None else now
        slow_since = getattr(self, "_nudger_slow_since", None)
        if slow_since is None:
            slow_since = self._nudger_slow_since = {}

        groups: dict[str, list[tuple[int, dict]]] = {}
        for rid, rec in list(getattr(self, "_active_reqs", {}).items()):
            target = rec.get("admission_target")
            if (
                not target
                or not rec.get("streaming")
                or rec.get("paused")
                or rec.get("nudge_event") is None
            ):
                continue
            groups.setdefault(target, []).append((rid, rec))

        recovering = {
            target for target, state in self._admission_store().items()
            if state.get("effective_limit") is not None
        }
        for target in set(slow_since) | set(groups) | recovering:
            streams = groups.get(target, [])
            state = self._admission_store().get(target)
            if state is None:
                slow_since.pop(target, None)
                continue

            try:
                configured_limit = max(1, int(state["limit"]))
            except (KeyError, TypeError, ValueError):
                configured_limit = 1
            effective_limit = state.get("effective_limit")
            if effective_limit is not None:
                try:
                    effective_limit = max(
                        1, min(configured_limit, int(effective_limit))
                    )
                except (TypeError, ValueError):
                    effective_limit = 1
                state["effective_limit"] = effective_limit

            if len(streams) < 2:
                slow_since.pop(target, None)
                if (state.get("nudger") or {}).get("status") == "watching":
                    state.pop("nudger", None)

            # A limit of one is the recovery baseline. There cannot be a
            # concurrent-stream signal until the next slot is admitted, so
            # hold it for one interval and then probe with one more slot.
            monitor_concurrency = len(streams) >= 2 and effective_limit != 1
            if monitor_concurrency:
                cutoff = now - stall_seconds
                recent_tokens = 0
                for _, rec in streams:
                    for kind in ("thinking", "output"):
                        timestamps = rec.get(kind, ())
                        recent_tokens += sum(ts >= cutoff for ts in timestamps)
                rate = recent_tokens / stall_seconds
                if rate >= threshold:
                    slow_since.pop(target, None)
                    if (state.get("nudger") or {}).get("status") == "watching":
                        state.pop("nudger", None)
                else:
                    # The stall clock cannot predate the newest concurrent
                    # stream, including one admitted by a recovery step.
                    newest_start = max(
                        rec.get("started_at", now) for _, rec in streams
                    )
                    began = slow_since.setdefault(target, newest_start)
                    if now - began < stall_seconds:
                        state["nudger"] = {
                            **(state.get("nudger") or {}),
                            "status": "watching",
                            "rate_tok_s": round(rate, 1),
                            "threshold_tok_s": threshold,
                            "effective_limit": effective_limit,
                            "configured_limit": configured_limit,
                        }
                        # Do not raise the limit while a low-rate interval is
                        # still being evaluated.
                        continue

                    streams.sort(
                        key=lambda item: item[1].get("started_at", now)
                    )
                    survivor_id = streams[0][0]
                    victims = [
                        (rid, rec) for rid, rec in streams[1:]
                        if rec.get("total_tokens", 0) == 0
                        and rec.get("forwarded_chunks", 0) == 0
                    ]
                    state["effective_limit"] = 1
                    state["_nudger_next_step_at"] = now + stall_seconds
                    nudger = state.get("nudger") or {}
                    nudger.update({
                        "status": (
                            "nudging" if victims else "blocked_partial_output"
                        ),
                        "rate_tok_s": round(rate, 1),
                        "threshold_tok_s": threshold,
                        "effective_limit": 1,
                        "configured_limit": configured_limit,
                        "survivor_request": survivor_id,
                        "replayed_requests": len(victims),
                        "trigger_count": int(
                            nudger.get("trigger_count", 0)
                        ) + 1,
                        "triggered_at": time.time(),
                    })
                    if not victims:
                        nudger["reason"] = (
                            "all newer streams already emitted output; "
                            "unsafe to replay"
                        )
                    state["nudger"] = nudger
                    for _, rec in victims:
                        rec["nudge_event"].set()
                    slow_since.pop(target, None)
                    continue

            effective_limit = state.get("effective_limit")
            if effective_limit is None:
                continue
            if effective_limit >= configured_limit:
                state.pop("effective_limit", None)
                state.pop("_nudger_next_step_at", None)
                state.pop("nudger", None)
                self._drain_inference_waiters(state)
                continue

            next_step_at = state.setdefault(
                "_nudger_next_step_at", now + stall_seconds
            )
            if now < next_step_at:
                continue

            next_limit = min(configured_limit, effective_limit + 1)
            if next_limit >= configured_limit:
                state.pop("effective_limit", None)
                state.pop("_nudger_next_step_at", None)
                state.pop("nudger", None)
            else:
                state["effective_limit"] = next_limit
                state["_nudger_next_step_at"] = now + stall_seconds
                nudger = state.get("nudger") or {}
                nudger.update({
                    "status": "recovering",
                    "effective_limit": next_limit,
                    "configured_limit": configured_limit,
                })
                state["nudger"] = nudger
            self._drain_inference_waiters(state)

    async def _inference_nudger_loop(self) -> None:
        while True:
            try:
                self._inference_nudger_tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("vLLM inference nudger tick failed")
            await asyncio.sleep(0.5)

    def get_token_stats(self) -> dict:
        total_in = sum(r.get("input", 0) for r in self.token_stats.values())
        total_out = sum(r.get("output", 0) for r in self.token_stats.values())
        total_cached = sum(r.get("cached", 0) for r in self.token_stats.values())
        total_req = sum(r.get("requests", 0) for r in self.token_stats.values())
        return {
            "models": self.token_stats,
            "groups": self.usage_rows(),
            "total": {"input": total_in, "output": total_out, "cached": total_cached, "requests": total_req},
        }

    def usage_rows(self) -> list[dict]:
        """Return reversible display groups over the raw lifetime counters.

        Directional routing rules are applied first, then the destination's
        optional merge group.  The raw counters remain keyed by the model name
        originally recorded, so changing a rule is always reversible.
        """
        rows: dict[str, dict] = {}
        numeric_fields = (
            "input", "output", "cached", "requests", "gen_tokens", "gen_time_s",
        )
        merge_groups = getattr(self, "usage_merge_groups", {})
        aliases = getattr(self, "usage_aliases", {})
        routing_rules = getattr(self, "usage_routing_rules", {})
        cache_estimates = getattr(self, "usage_cache_estimates", {})
        for model, model_stats in self.token_stats.items():
            destination = self._usage_route_target(model, routing_rules)
            merge_group = str(merge_groups.get(destination) or "").strip()
            row_key = (
                f"group:{merge_group}"
                if merge_group
                else f"model:{destination}"
            )
            row = rows.setdefault(row_key, {
                "key": row_key,
                "label": merge_group or aliases.get(destination) or destination,
                "merge_group": merge_group or None,
                "route_target": None if merge_group else destination,
                "models": [],
                "members": [],
                "stats": {field: 0 for field in numeric_fields},
                "total_cost": 0.0,
                "_estimated_cached": 0,
                "_applied_estimate_keys": set(),
            })
            measured_cached = max(0, int(model_stats.get("cached", 0) or 0))
            input_tokens = max(0, int(model_stats.get("input", 0) or 0))
            source_estimate_key = f"model:{model}"
            source_estimate = cache_estimates.get(source_estimate_key)
            source_estimated_cached = 0
            if isinstance(source_estimate, dict):
                source_estimated_cached = min(
                    max(0, input_tokens - measured_cached),
                    max(0, int(source_estimate.get("estimated_cached") or 0)),
                )
                row["_applied_estimate_keys"].add(source_estimate_key)
                row.setdefault("_cache_estimates", []).append(source_estimate)
            row["_estimated_cached"] += source_estimated_cached
            row["models"].append(model)
            row["members"].append({
                "model": model,
                "alias": aliases.get(model),
                "merge_group": merge_group or None,
                "routed_to": destination if destination != model else None,
                # Private values are removed before returning the public row.
                # They let merge groups reprice estimated cache hits at each
                # member's actual (or routed destination's) rates.
                "_pricing_model": destination,
                "_cost_stats": {
                    "input": input_tokens,
                    "cached": measured_cached + source_estimated_cached,
                    "output": max(0, int(model_stats.get("output", 0) or 0)),
                },
            })
            for field in numeric_fields:
                row["stats"][field] += model_stats.get(field, 0) or 0

        for row in rows.values():
            row["routed_sources"] = [
                member["model"]
                for member in row["members"]
                if member.get("routed_to")
            ]
            speed = self.rolling_generation_speed(row["models"])
            # Existing installations have lifetime speed totals but no
            # interval samples until this version records its first request.
            # Preserve a useful fallback during that migration period.
            if speed["tok_s"] is None:
                legacy_tokens = row["stats"].get("gen_tokens", 0) or 0
                legacy_time = row["stats"].get("gen_time_s", 0) or 0
                if legacy_tokens > 0 and legacy_time > 0:
                    speed = {
                        "tokens": legacy_tokens,
                        "active_time_s": legacy_time,
                        "tok_s": legacy_tokens / legacy_time,
                        "legacy": True,
                    }
            row["speed"] = speed
            measured_cached = max(0, int(row["stats"].get("cached", 0) or 0))
            estimated_cached = int(row.pop("_estimated_cached", 0) or 0)
            applied_estimate_keys = row.pop("_applied_estimate_keys", set())
            estimate = cache_estimates.get(row["key"])
            if (
                isinstance(estimate, dict)
                and row["key"] not in applied_estimate_keys
            ):
                extra_estimated_cached = min(
                    max(
                        0,
                        int(row["stats"].get("input", 0) or 0)
                        - measured_cached
                        - estimated_cached,
                    ),
                    max(0, int(estimate.get("estimated_cached") or 0)),
                )
                estimated_cached += extra_estimated_cached
                row.setdefault("_cache_estimates", []).append(estimate)

                # A group-level estimate has no single source model. Spread
                # it across each member's remaining misses so mixed-price
                # merge groups can still be repriced consistently.
                remaining = extra_estimated_cached
                capacities = [
                    max(
                        0,
                        int(member["_cost_stats"]["input"])
                        - int(member["_cost_stats"]["cached"]),
                    )
                    for member in row["members"]
                ]
                remaining_capacity = sum(capacities)
                for member, capacity in zip(row["members"], capacities):
                    if not remaining or not remaining_capacity:
                        break
                    allocation = min(
                        capacity,
                        round(remaining * capacity / remaining_capacity),
                    )
                    member["_cost_stats"]["cached"] += allocation
                    remaining -= allocation
                    remaining_capacity -= capacity

            estimates = row.pop("_cache_estimates", [])
            if estimates:
                row["cache_estimate"] = dict(estimates[0])
            effective_cached = measured_cached + estimated_cached
            row["stats"]["measured_cached"] = measured_cached
            row["stats"]["estimated_cached"] = estimated_cached
            row["stats"]["cached"] = effective_cached
            row["stats"]["input_miss"] = max(
                0, (row["stats"].get("input", 0) or 0) - effective_cached
            )

            # Routed rows use one set of rates for the entire logical model.
            # Prefer the destination, but historical aliases may outlive the
            # deployment that supplied their pricing. In that case, fall back
            # to a priced member (normally the currently deployed model).
            # Plain merge groups may intentionally mix prices, so calculate
            # them per member after applying estimated cache hits.
            if row.get("route_target"):
                pricing_candidates = [row["route_target"]]
                pricing_candidates.extend(
                    model for model in row["models"]
                    if model != row["route_target"]
                )
                pricing_model = row["route_target"]
                for candidate in pricing_candidates:
                    candidate_cost = self.calculate_cost(candidate, {
                        "input": 1, "cached": 0, "output": 1,
                    })
                    if any(candidate_cost.get(rate, 0) for rate in (
                        "input_cost_per_1m",
                        "output_cost_per_1m",
                        "cache_cost_per_1m",
                    )):
                        pricing_model = candidate
                        break

                row["pricing_model"] = pricing_model
                row["total_cost"] = self.calculate_cost(pricing_model, {
                    "input": row["stats"].get("input", 0),
                    "cached": effective_cached,
                    "output": row["stats"].get("output", 0),
                })["total_cost"]
                if estimated_cached:
                    row["cost_estimated"] = True
            else:
                row["total_cost"] = sum(
                    self.calculate_cost(
                        member["_pricing_model"], member["_cost_stats"],
                    ).get("total_cost", 0.0)
                    for member in row["members"]
                )
                if estimated_cached:
                    row["cost_estimated"] = True
            row["members"].sort(key=lambda member: (
                member["model"] != row.get("route_target"), member["model"]
            ))
            for member in row["members"]:
                member.pop("_pricing_model", None)
                member.pop("_cost_stats", None)
            row["total_cost"] = round(max(0.0, row["total_cost"]), 2)
        return list(rows.values())

    def calculate_cost(self, model: str, stats: dict | None = None) -> dict:
        """Calculate the total cost for a model's token usage.

        Uses per-model pricing from unsloth_settings when available, otherwise
        falls back to the built-in MODEL_PRICING map.  When *stats* is
        supplied it is used instead of the lifetime ``token_stats`` so the
        caller can compute a session-scoped cost.

        Cached (KV-prefix-hit) tokens are charged at ``cache_cost_per_1m``
        (or the built-in ``cache`` rate), which is typically lower than the
        regular input rate.  When the cache rate is 0, cached tokens are
        free.
        """
        if stats is None:
            stats = self.token_stats.get(model)
        if not stats:
            return {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0,
                    "input_cost": 0.0, "output_cost": 0.0, "cache_cost": 0.0,
                    "total_cost": 0.0}

        input_tokens = stats.get("input", 0)
        output_tokens = stats.get("output", 0)
        cached_tokens = stats.get("cached", 0)

        # Resolve pricing: cluster deployment metadata first, then standalone
        # per-model settings, then the built-in map.
        input_cost_per_1m = 0.0
        output_cost_per_1m = 0.0
        cache_cost_per_1m = 0.0
        deployment_pricing: dict[str, Any] | None = None

        model_lower = model.casefold()
        for deployment in getattr(self, "deployments", []):
            settings = deployment.get("launch_settings") or {}
            pricing_model_key = self._deployment_pricing_model_key(deployment)
            if not pricing_model_key or pricing_model_key.casefold() != model_lower:
                continue
            values = {
                "input": settings.get("input_cost_per_1m"),
                "output": settings.get("output_cost_per_1m"),
                "cache": settings.get("cache_cost_per_1m"),
            }
            if any(value is not None for value in values.values()):
                deployment_pricing = values
                break

        # Check per-model unsloth settings
        for key, settings in getattr(self, "unsloth_settings", {}).items():
            if key.lower() in model.lower() or model.lower() in key.lower():
                inp = settings.get("input_cost_per_1m")
                out = settings.get("output_cost_per_1m")
                cch = settings.get("cache_cost_per_1m")
                if inp and inp > 0:
                    input_cost_per_1m = float(inp)
                if out and out > 0:
                    output_cost_per_1m = float(out)
                if cch and cch > 0:
                    cache_cost_per_1m = float(cch)
                break

        # Fallback to built-in pricing map (prefix match)
        if input_cost_per_1m == 0 and output_cost_per_1m == 0:
            for prefix, rates in MODEL_PRICING.items():
                if prefix.lower() in model_lower or model_lower in prefix.lower():
                    input_cost_per_1m = float(rates.get("input", 0))
                    output_cost_per_1m = float(rates.get("output", 0))
                    cache_cost_per_1m = float(rates.get("cache", 0))
                    break

        # Cluster fields override their matching defaults independently.
        # An explicit zero therefore means free, while a blank field keeps
        # the standalone/built-in rate for that token class.
        if deployment_pricing is not None:
            if deployment_pricing["input"] is not None:
                input_cost_per_1m = float(deployment_pricing["input"])
            if deployment_pricing["output"] is not None:
                output_cost_per_1m = float(deployment_pricing["output"])
            if deployment_pricing["cache"] is not None:
                cache_cost_per_1m = float(deployment_pricing["cache"])

        # Non-cached input tokens are charged at the regular input rate;
        # cached tokens are charged at the (typically lower) cache rate.
        non_cached_input = max(0, input_tokens - cached_tokens)
        input_cost = round((non_cached_input / 1_000_000) * input_cost_per_1m, 2)
        cache_cost = round((cached_tokens / 1_000_000) * cache_cost_per_1m, 2)
        output_cost = round((output_tokens / 1_000_000) * output_cost_per_1m, 2)
        total_cost = round(input_cost + cache_cost + output_cost, 2)

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_tokens": cached_tokens,
            "input_cost_per_1m": input_cost_per_1m,
            "output_cost_per_1m": output_cost_per_1m,
            "cache_cost_per_1m": cache_cost_per_1m,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "cache_cost": cache_cost,
            "total_cost": total_cost,
        }

    def _reset_synced_token_usage(self) -> None:
        ledger = getattr(self, "token_usage_sync", None)
        if not isinstance(ledger, dict):
            return
        node_id = self._local_usage_node_id()
        counter = self._usage_version_key(ledger.get("epoch"))[0] + 1
        origins = copy.deepcopy(ledger.get("origins") or {})
        for component in origins.values():
            if not isinstance(component, dict):
                continue
            component["token_stats"] = {}
            component["speed_samples"] = {}
            component["revision"] = int(component.get("revision") or 0) + 1
        origins.setdefault(node_id, {
            "revision": 0,
            "token_stats": {},
            "hourly_token_stats": {},
            "model_epochs": {},
            "speed_samples": {},
        })
        self.token_usage_sync = {
            "version": 1,
            "epoch": [counter, node_id],
            "model_epochs": copy.deepcopy(ledger.get("model_epochs") or {}),
            "origins": origins,
        }
        if ledger.get("hourly_retention_cutoff"):
            self.token_usage_sync["hourly_retention_cutoff"] = ledger[
                "hourly_retention_cutoff"
            ]
        self._save_token_usage_sync()

    def _erase_synced_usage_model(self, model: str) -> None:
        ledger = getattr(self, "token_usage_sync", None)
        if not isinstance(ledger, dict):
            return
        node_id = self._local_usage_node_id()
        model_epochs = ledger.setdefault("model_epochs", {})
        counter = self._usage_version_key(model_epochs.get(model))[0] + 1
        model_epochs[model] = [counter, node_id]
        for component in (ledger.get("origins") or {}).values():
            if not isinstance(component, dict):
                continue
            (component.get("token_stats") or {}).pop(model, None)
            for models in (component.get("hourly_token_stats") or {}).values():
                if isinstance(models, dict):
                    models.pop(model, None)
            (component.get("speed_samples") or {}).pop(model, None)
            component["revision"] = int(component.get("revision") or 0) + 1
        self._save_token_usage_sync()

    def reset_token_stats(self) -> dict:
        self._reset_synced_token_usage()
        self.token_stats = {}
        self.speed_samples = {}
        self.usage_cache_estimates = {}
        # Ensure an in-flight batched snapshot notices the reset and follows
        # up with the empty state after its current write completes.
        self._speed_samples_version = getattr(
            self, "_speed_samples_version", 0
        ) + 1
        try:
            self._save_token_stats()
        except Exception:
            pass
        try:
            self._save_speed_samples()
        except Exception:
            pass
        try:
            self._save_usage_cache_estimates()
        except Exception:
            pass
        return self.get_token_stats()

    def erase_usage_model(self, model: Any) -> dict:
        """Erase every persisted usage record for one exact model key."""
        model_key = str(model or "").strip()
        if not model_key or model_key not in self.token_stats:
            raise ValueError("usage model not found")

        self._erase_synced_usage_model(model_key)
        self.token_stats.pop(model_key, None)
        self.session_token_stats.pop(model_key, None)
        self.speed_samples.pop(model_key, None)
        self.usage_aliases.pop(model_key, None)
        self.usage_merge_groups.pop(model_key, None)
        cache_estimates = getattr(self, "usage_cache_estimates", None)
        if isinstance(cache_estimates, dict):
            cache_estimates.pop(f"model:{model_key}", None)
        for hour_key in list(self.hourly_token_stats):
            models = self.hourly_token_stats.get(hour_key)
            if not isinstance(models, dict):
                continue
            models.pop(model_key, None)
            if not models:
                self.hourly_token_stats.pop(hour_key, None)

        # Invalidate an older in-flight speed snapshot before persisting the
        # erased state, so it cannot permanently restore this model.
        self._speed_samples_version = getattr(
            self, "_speed_samples_version", 0
        ) + 1
        self._save_token_stats()
        self._save_hourly_token_stats()
        self._save_speed_samples()
        self._save_usage_aliases()
        self._save_usage_merge_groups()
        if hasattr(self, "usage_cache_estimates_path"):
            self._save_usage_cache_estimates()
        return {"ok": True, "model": model_key}

    def reset_session_token_stats(self) -> dict:
        """Clear only the session counters (not the persisted lifetime stats)."""
        self.session_token_stats = {}
        return self.get_token_stats()

    # ─── hourly token stats (for Usage → Analysis charts) ──────
    @staticmethod
    def _parse_date(s: str) -> datetime:
        """Parse a YYYY-MM-DD string into a midnight datetime."""
        return datetime.strptime(s, "%Y-%m-%d")

    def get_hourly_token_stats(self, start: str | None = None,
                               end: str | None = None) -> list[dict]:
        """Return per-hour token usage as a list of {hour, input, output,
        cached, requests} sorted chronologically.  *start* / *end* are optional
        YYYY-MM-DD strings that bound the date range (inclusive)."""
        start_dt = self._parse_date(start) if start else None
        end_dt = (self._parse_date(end) + timedelta(days=1)
                  if end else None)
        result = []
        for hour_key, models in sorted(self.hourly_token_stats.items()):
            # hour_key looks like "2024-01-15T10"
            try:
                dt = datetime.strptime(hour_key, "%Y-%m-%dT%H")
            except ValueError:
                continue
            if start_dt and dt < start_dt:
                continue
            if end_dt and dt >= end_dt:
                continue
            total_in = sum(m.get("input", 0) for m in models.values())
            total_out = sum(m.get("output", 0) for m in models.values())
            total_cached = sum(m.get("cached", 0) for m in models.values())
            total_req = sum(m.get("requests", 0) for m in models.values())
            result.append({
                "hour": hour_key,
                "input": total_in,
                "output": total_out,
                "cached": total_cached,
                "requests": total_req,
            })
        return result

    def get_daily_token_stats(self, start: str | None = None,
                              end: str | None = None,
                              weeks: int = 12) -> list[dict]:
        """Return per-day token usage as a list of {date, input, output,
        cached, requests} sorted chronologically.  When *start* / *end* are
        omitted, defaults to the last *weeks* weeks (or all data if less)."""
        if start or end:
            start_dt = self._parse_date(start) if start else None
            end_dt = (self._parse_date(end) + timedelta(days=1)
                      if end else None)
        else:
            # Default: last *weeks* weeks.
            end_dt = (
                datetime.now(timezone.utc).replace(tzinfo=None)
                + timedelta(days=1)
            )
            start_dt = end_dt - timedelta(weeks=weeks)
        daily: dict[str, dict] = {}
        for hour_key, models in self.hourly_token_stats.items():
            try:
                dt = datetime.strptime(hour_key, "%Y-%m-%dT%H")
            except ValueError:
                continue
            if start_dt and dt < start_dt:
                continue
            if end_dt and dt >= end_dt:
                continue
            date_str = dt.strftime("%Y-%m-%d")
            d = daily.setdefault(date_str, {
                "input": 0, "output": 0, "cached": 0, "requests": 0,
                "models": {},
            })
            d["input"] += sum(m.get("input", 0) for m in models.values())
            d["output"] += sum(m.get("output", 0) for m in models.values())
            d["cached"] += sum(m.get("cached", 0) for m in models.values())
            d["requests"] += sum(m.get("requests", 0) for m in models.values())
            for model, counters in models.items():
                model_total = d["models"].setdefault(model, {
                    "input": 0, "output": 0, "cached": 0, "requests": 0,
                })
                for key in ("input", "output", "cached", "requests"):
                    model_total[key] += counters.get(key, 0)
        return [
            {"date": d, "input": v["input"], "output": v["output"],
             "cached": v["cached"], "requests": v["requests"],
             "models": v["models"]}
            for d, v in sorted(daily.items())
        ]

    # ---------- recipes ----------
    def recipe_deployment_contract(self, recipe: dict) -> dict:
        """Return the node-count contract for a persisted launch recipe."""
        def positive_parallelism(value) -> int:
            try:
                return max(1, int(value or 1))
            except (TypeError, ValueError):
                return 1

        engine = str(recipe.get("engine") or "vllm")
        raw_args = recipe.get("extra_args")
        args_error = None
        if raw_args is None:
            args = []
        elif not isinstance(raw_args, list) or any(
            not isinstance(value, str) for value in raw_args
        ):
            args = []
            args_error = PERSISTED_RECIPE_ARGS_ERROR
        else:
            args = list(raw_args)
        if engine == "sglang":
            tensor_parallel = recipe.get("sg_tp_size")
            if tensor_parallel is not None:
                try:
                    tensor_parallel = int(tensor_parallel)
                except (TypeError, ValueError):
                    tensor_parallel = None
            if tensor_parallel is None:
                tensor_parallel = self._cli_option(args, {"--tp-size"}, int)
            pipeline_parallel = 1
        else:
            tensor_parallel = self._cli_option(
                args, {"--tensor-parallel-size", "-tp"}, int
            )
            pipeline_parallel = self._cli_option(
                args, {"--pipeline-parallel-size", "-pp"}, int
            )
        tensor_parallel = positive_parallelism(tensor_parallel)
        pipeline_parallel = positive_parallelism(pipeline_parallel)
        parallel_nodes = tensor_parallel * pipeline_parallel
        persisted_mode = recipe.get("deployment_mode")
        mode = str(persisted_mode or "single")
        mode_error = args_error
        raw_saved_nodes = recipe.get("node_ids")
        saved_nodes = []
        if raw_saved_nodes is None or raw_saved_nodes == []:
            saved_nodes = [LOCAL_NODE_ID]
        elif not isinstance(raw_saved_nodes, list):
            mode_error = "persisted node_ids must be an array of non-empty node IDs"
            saved_nodes = [LOCAL_NODE_ID]
        else:
            seen_nodes = set()
            for value in raw_saved_nodes:
                if not isinstance(value, str) or not value.strip():
                    mode_error = "persisted node_ids must contain only non-empty node IDs"
                    continue
                node_id = value.strip()
                if node_id not in seen_nodes:
                    seen_nodes.add(node_id)
                    saved_nodes.append(node_id)
            if not saved_nodes:
                saved_nodes = [LOCAL_NODE_ID]
        if mode == "replicated":
            required_nodes = max(2, len(saved_nodes))
        elif mode == "sharded":
            required_nodes = parallel_nodes if parallel_nodes > 1 else max(2, len(saved_nodes))
        elif persisted_mode is None and parallel_nodes > 1:
            # A legacy TP/PP recipe created before deployment modes existed is
            # a distributed launch even if its persisted mode defaulted to single.
            mode = "sharded"
            required_nodes = parallel_nodes
        elif mode == "single":
            required_nodes = 1
        else:
            required_nodes = 1
            invalid_mode = f"unsupported persisted deployment mode: {mode}"
            mode_error = f"{mode_error}; {invalid_mode}" if mode_error else invalid_mode
        model_revision = self._cli_option(args, {"--revision"})
        return {
            "required_node_count": required_nodes,
            "deployment_mode": mode,
            "tensor_parallel_size": tensor_parallel,
            "pipeline_parallel_size": pipeline_parallel,
            "model_revision": str(model_revision).strip() if model_revision else None,
            "supported": mode_error is None,
            "error": mode_error,
        }

    def _load_recipes(self) -> list[dict]:
        if self.recipes_path.exists():
            try:
                value = json.loads(self.recipes_path.read_text())
                if not isinstance(value, list):
                    return []
                for recipe in value:
                    engine = str(recipe.get("engine") or "vllm")
                    if engine not in {"vllm", "sglang"}:
                        recipe["supported"] = False
                        recipe["error"] = f"unsupported persisted runtime: {engine}"
                return value
            except Exception:
                pass
        return []

    def _migrate_recipe_hf_credentials(self) -> bool:
        """Discard legacy CLI credentials so they cannot re-enter public state."""
        changed = False
        for recipe in self.recipes:
            raw_args = recipe.get("extra_args")
            if raw_args is None:
                original = []
            elif not isinstance(raw_args, list) or any(
                not isinstance(value, str) for value in raw_args
            ):
                recipe["supported"] = False
                recipe["error"] = _append_persisted_error(
                    recipe.get("error"), PERSISTED_RECIPE_ARGS_ERROR,
                )
                # Replace the corrupt launch input so every public/listing
                # path remains safe while the durable unsupported marker keeps
                # this recipe from being launched until the user edits it.
                recipe["extra_args"] = []
                changed = True
                continue
            else:
                original = list(raw_args)
            sanitized = self._without_hf_cli_credentials(original)
            if sanitized != original:
                recipe["extra_args"] = sanitized
                changed = True
        return changed

    def _save_recipes(self):
        _atomic_private_json_write(self.recipes_path, self.recipes)

    @staticmethod
    def _recipe_key(model: str, image: str | None, extra_args: list | None,
                    engine: str = "vllm", deployment_mode: str = "single",
                    node_ids: list[str] | None = None) -> tuple:
        return (
            model or "", image or "", tuple(extra_args or []), engine or "vllm",
            deployment_mode or "single", tuple(node_ids or [LOCAL_NODE_ID]),
        )

    @staticmethod
    def _normalize_recipe_node_ids(node_ids: list[str] | None) -> list[str]:
        if node_ids is None:
            return [LOCAL_NODE_ID]
        if not isinstance(node_ids, list):
            raise ValueError("node_ids must be an array")
        if not node_ids:
            raise ValueError("node_ids must contain at least one node ID")
        normalized: list[str] = []
        seen: set[str] = set()
        for value in node_ids:
            if not isinstance(value, str) or not value.strip():
                raise ValueError("node_ids must contain only non-empty strings")
            node_id = value.strip()
            if node_id not in seen:
                seen.add(node_id)
                normalized.append(node_id)
        return normalized

    async def add_recipe(
        self,
        model: str,
        name: str | None = None,
        image: str | None = None,
        extra_args: list | None = None,
        gpu_memory_utilization: float | None = None,
        gpu_memory_gb: float | None = None,
        engine: str = "vllm",
        # SGLang-specific fields
        sg_tp_size: int | None = None,
        sg_context_length: int | None = None,
        sg_max_running_requests: int | None = None,
        sg_mem_fraction: float | None = None,
        sg_image: str | None = None,
        deployment_mode: str = "single",
        node_ids: list[str] | None = None,
        launch_controls: dict | None = None,
        force_new: bool = False,
    ) -> dict:
        self._reject_hf_cli_credentials(extra_args)
        if not model:
            raise ValueError("model is required")
        if engine not in {"vllm", "sglang"}:
            raise ValueError("engine must be vllm or sglang")
        deployment_mode = deployment_mode or "single"
        if deployment_mode not in {"single", "sharded", "replicated"}:
            raise ValueError("deployment_mode must be single, sharded, or replicated")
        node_ids = self._normalize_recipe_node_ids(node_ids)
        if deployment_mode == "single":
            node_ids = node_ids[:1]
        elif len(node_ids) < 2:
            raise ValueError(f"{deployment_mode} deployment requires at least two nodes")
        if launch_controls is not None:
            if not isinstance(launch_controls, dict):
                raise ValueError("launch_controls must be an object")
            launch_controls = {
                **self._deployment_launch_controls({
                    "engine": engine,
                    "extra_args": list(extra_args or []),
                    "sg_context_length": sg_context_length,
                    "sg_max_running_requests": sg_max_running_requests,
                }),
                **launch_controls,
            }
            extra_args = self._apply_deployment_launch_controls(
                list(extra_args or []), engine, launch_controls
            )
            if engine == "sglang":
                if "context_window" in launch_controls:
                    sg_context_length = launch_controls.get("context_window")
                if "max_concurrency" in launch_controls:
                    sg_max_running_requests = launch_controls.get("max_concurrency")
        async with self.lock:
            key = self._recipe_key(model, image, extra_args, engine, deployment_mode, node_ids)
            for r in [] if force_new else self.recipes:
                if self._recipe_key(
                    r.get("model", ""), r.get("image"), r.get("extra_args"),
                    r.get("engine", "vllm"), r.get("deployment_mode", "single"),
                    r.get("node_ids"),
                ) == key:
                    # Update name/gpu_mem/engine if provided so explicit re-saves overwrite metadata.
                    if name:
                        r["name"] = name
                    if engine != r.get("engine"):
                        r["engine"] = engine
                    if gpu_memory_utilization is not None:
                        r["gpu_memory_utilization"] = gpu_memory_utilization
                    if gpu_memory_gb is not None:
                        r["gpu_memory_gb"] = gpu_memory_gb
                    # Update SGLang fields if provided
                    if sg_tp_size is not None:
                        r["sg_tp_size"] = sg_tp_size
                    if sg_context_length is not None:
                        r["sg_context_length"] = sg_context_length
                    if sg_max_running_requests is not None:
                        r["sg_max_running_requests"] = sg_max_running_requests
                    if sg_mem_fraction is not None:
                        r["sg_mem_fraction"] = sg_mem_fraction
                    if sg_image is not None:
                        r["sg_image"] = sg_image
                    r["deployment_mode"] = deployment_mode or "single"
                    r["node_ids"] = list(node_ids or [LOCAL_NODE_ID])
                    self._save_recipes()
                    return r
            recipe = {
                "id": uuid.uuid4().hex[:8],
                "name": name or model,
                "model": model,
                "engine": engine,
                "image": image or None,
                "extra_args": list(extra_args or []),
                "gpu_memory_utilization": gpu_memory_utilization,
                "gpu_memory_gb": gpu_memory_gb,
                # SGLang-specific fields
                "sg_tp_size": sg_tp_size,
                "sg_context_length": sg_context_length,
                "sg_max_running_requests": sg_max_running_requests,
                "sg_mem_fraction": sg_mem_fraction,
                "sg_image": sg_image,
                "deployment_mode": deployment_mode or "single",
                "node_ids": list(node_ids or [LOCAL_NODE_ID]),
                "created_at": time.time(),
            }
            self.recipes.append(recipe)
            self._save_recipes()
            return recipe

    async def delete_recipe(self, rid: str) -> bool:
        async with self.lock:
            before = len(self.recipes)
            self.recipes = [r for r in self.recipes if r.get("id") != rid]
            if len(self.recipes) < before:
                self._save_recipes()
                return True
        return False

    async def update_recipe(self, rid: str, changes: dict) -> dict:
        """Update the durable launch inputs for an existing recipe."""
        if not isinstance(changes, dict):
            raise ValueError("recipe changes must be an object")
        allowed = {
            "name", "model", "engine", "image", "extra_args",
            "gpu_memory_utilization", "gpu_memory_gb", "sg_tp_size",
            "sg_context_length", "sg_max_running_requests", "sg_mem_fraction",
            "sg_image", "deployment_mode", "node_ids", "launch_controls",
        }
        unknown = sorted(set(changes) - allowed)
        if unknown:
            raise ValueError(f"unsupported recipe field(s): {', '.join(unknown)}")
        if "extra_args" in changes:
            self._reject_hf_cli_credentials(changes.get("extra_args"))
        async with self.lock:
            recipe = next((r for r in self.recipes if r.get("id") == rid), None)
            if not recipe:
                raise ValueError("recipe not found")
            merged = {**recipe, **changes}
            if not str(merged.get("model") or "").strip():
                raise ValueError("model is required")
            engine = merged.get("engine") or "vllm"
            if engine not in {"vllm", "sglang"}:
                raise ValueError("engine must be vllm or sglang")
            mode = merged.get("deployment_mode") or "single"
            if mode not in {"single", "sharded", "replicated"}:
                raise ValueError("deployment_mode must be single, sharded, or replicated")
            nodes = self._normalize_recipe_node_ids(merged.get("node_ids"))
            if mode == "single":
                nodes = nodes[:1]
            elif len(nodes) < 2:
                raise ValueError(f"{mode} deployment requires at least two nodes")
            controls = changes.get("launch_controls")
            if controls is not None:
                if not isinstance(controls, dict):
                    raise ValueError("launch_controls must be an object")
                controls = {
                    **self._deployment_launch_controls(
                        self._deployment_launch_settings(merged)
                    ),
                    **controls,
                }
                merged["extra_args"] = self._apply_deployment_launch_controls(
                    list(merged.get("extra_args") or []), engine, controls
                )
                if engine == "sglang":
                    if "context_window" in controls:
                        merged["sg_context_length"] = controls.get("context_window")
                    if "max_concurrency" in controls:
                        merged["sg_max_running_requests"] = controls.get("max_concurrency")
            merged.pop("launch_controls", None)
            merged["node_ids"] = nodes
            merged["engine"] = engine
            merged["deployment_mode"] = mode
            merged["extra_args"] = list(merged.get("extra_args") or [])
            if "extra_args" in changes:
                had_args_error = PERSISTED_RECIPE_ARGS_ERROR in str(
                    merged.get("error") or ""
                )
                remaining_error = _remove_persisted_error(
                    merged.get("error"), PERSISTED_RECIPE_ARGS_ERROR,
                )
                if remaining_error:
                    merged["error"] = remaining_error
                    merged["supported"] = False
                elif had_args_error:
                    merged.pop("error", None)
                    merged.pop("supported", None)
            merged["updated_at"] = time.time()
            recipe.clear()
            recipe.update(merged)
            self._save_recipes()
            return dict(recipe)

    async def get_recipe(self, rid: str) -> dict | None:
        for r in self.recipes:
            if r.get("id") == rid:
                return r
        return None

    # ---------- saved SparkRun targets ----------
    def _load_spark_launches(self) -> list[dict]:
        if self.spark_launches_path.exists():
            try:
                data = json.loads(self.spark_launches_path.read_text())
                if isinstance(data, list):
                    return data
            except Exception:
                pass
        return []

    def _save_spark_launches(self):
        self.spark_launches_path.write_text(json.dumps(self.spark_launches, indent=2))

    @staticmethod
    def _normalize_spark_reference(reference: str) -> str:
        """Accept the reference formats users commonly paste into the portal."""
        value = (reference or "").strip()
        if value.lower().startswith("sparkrun run "):
            try:
                tokens = shlex.split(value)
            except ValueError as exc:
                raise ValueError(f"invalid SparkRun command: {exc}") from exc
            if len(tokens) < 3:
                raise ValueError("SparkRun command is missing a recipe reference")
            value = tokens[2]

        uuid_match = re.fullmatch(
            r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
            r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
            value,
        )
        if uuid_match:
            return f"@spark-arena/{value.lower()}"

        if re.match(r"https?://(?:www\.)?spark-arena\.com/", value, re.IGNORECASE):
            match = re.search(
                r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
                r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
                value,
            )
            if match:
                return f"@spark-arena/{match.group(0).lower()}"
        return value

    @staticmethod
    def _sparkrun_error(stderr: bytes, reference: str) -> str:
        """Return the useful CLI error without exposing a Python traceback."""
        text = stderr.decode("utf-8", errors="replace").strip()
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        detail = lines[-1] if lines else f"SparkRun could not resolve {reference}"
        detail = re.sub(r"^[\w.]+(?:Error|Exception):\s*", "", detail)
        return detail[:1000]

    @staticmethod
    def _sparkrun_executable() -> str:
        """Find SparkRun in interactive shells and minimal systemd PATHs."""
        executable = shutil.which("sparkrun")
        if executable:
            return executable
        local_executable = Path.home() / ".local" / "bin" / "sparkrun"
        if local_executable.is_file() and os.access(local_executable, os.X_OK):
            return str(local_executable)
        raise ValueError("`sparkrun` is not installed or is not on PATH")

    @staticmethod
    def _project_spark_config(data: dict) -> dict:
        """Keep display details from a resolved SparkRun config, never YAML.

        The command is important here: most runtime arguments live in the
        command template rather than ``defaults``.  Saved launches still run
        by reference; this projection is only the resolved, read-only view
        shown before launch.
        """
        meta = data.get("metadata") or {}
        defaults = data.get("defaults") or {}
        env = data.get("env") or {}
        mods = data.get("mods") or []
        if not isinstance(meta, dict):
            meta = {}
        if not isinstance(defaults, dict):
            defaults = {}
        if not isinstance(env, dict):
            env = {}
        if isinstance(mods, str):
            mods = [mods]
        if not isinstance(mods, list):
            mods = []
        tags = meta.get("tags") or []
        if isinstance(tags, str):
            tags = [tags]
        return {
            "model": data.get("model"), "runtime": data.get("runtime"),
            "container": data.get("container"), "description": meta.get("description") or data.get("description"),
            "author": meta.get("author") or meta.get("maintainer"),
            "tags": [str(tag) for tag in tags] if isinstance(tags, list) else [],
            "min_nodes": data.get("min_nodes"), "max_nodes": data.get("max_nodes"),
            "defaults": defaults,
            "command": data.get("command") if isinstance(data.get("command"), str) else "",
            "env": env,
            "mods": [str(mod) for mod in mods],
            "builder": data.get("builder"),
            "recipe_version": data.get("recipe_version"),
        }

    async def _export_spark_reference(self, reference: str) -> dict:
        """Ask SparkRun to resolve a target and return its normalized config."""
        executable = self._sparkrun_executable()
        try:
            proc = await asyncio.create_subprocess_exec(
                executable, "export", "recipe", reference, "--json",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as e:
            raise ValueError("`sparkrun` is not installed or is not on PATH") from e
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=90)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            raise ValueError("SparkRun took too long to resolve this run")
        if proc.returncode:
            raise ValueError(self._sparkrun_error(stderr, reference))
        try:
            data = json.loads(stdout)
        except json.JSONDecodeError as e:
            raise ValueError("SparkRun returned invalid recipe data") from e
        if not isinstance(data, dict):
            raise ValueError("SparkRun returned an invalid recipe")
        return data

    async def _resolve_spark_reference(self, reference: str) -> tuple[str, dict]:
        data = await self._export_spark_reference(reference)
        metadata = data.get("metadata") or {}
        display_name = data.get("name") or (metadata.get("name") if isinstance(metadata, dict) else None)
        return display_name or reference, self._project_spark_config(data)

    @staticmethod
    def _normalize_spark_command_json(command: str) -> tuple[str, int]:
        """Repair doubled braces only when removing them yields valid JSON."""
        repairs = 0

        def replace(match: re.Match) -> str:
            nonlocal repairs
            candidate = match.group(0)[1:-1]
            try:
                json.loads(candidate)
            except (json.JSONDecodeError, TypeError):
                return match.group(0)
            repairs += 1
            return candidate

        return re.sub(r"\{\{[^\n]*?\}\}", replace, command or ""), repairs

    async def _spark_run_target(self, launch: dict) -> tuple[str, str | None, int]:
        """Use a temporary normalized recipe only for known malformed JSON."""
        reference = launch.get("reference") or ""
        saved_command = (launch.get("metadata") or {}).get("command") or ""
        _, possible_repairs = self._normalize_spark_command_json(saved_command)
        if not possible_repairs:
            return reference, None, 0

        data = await self._export_spark_reference(reference)
        normalized, repairs = self._normalize_spark_command_json(data.get("command") or "")
        if not repairs:
            return reference, None, 0
        data["command"] = normalized
        temp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", prefix="sparkrun-normalized-", delete=False,
        )
        try:
            json.dump(data, temp)
        finally:
            temp.close()
        return temp.name, temp.name, repairs

    async def add_spark_launch(self, reference: str) -> dict:
        reference = self._normalize_spark_reference(reference)
        if not reference or len(reference) > 500 or any(c in reference for c in "\r\n\x00"):
            raise ValueError("a valid SparkRun reference is required")
        name, metadata = await self._resolve_spark_reference(reference)
        async with self.lock:
            for launch in self.spark_launches:
                if launch.get("reference") == reference:
                    launch.update({"name": name, "metadata": metadata, "updated_at": time.time(), "resolve_error": None})
                    self._save_spark_launches()
                    return launch
            launch = {"id": uuid.uuid4().hex[:8], "reference": reference, "name": name,
                      "metadata": metadata, "created_at": time.time(), "updated_at": time.time(),
                      "resolve_error": None}
            self.spark_launches.append(launch)
            self._save_spark_launches()
            return launch

    async def refresh_spark_launch(self, rid: str) -> dict | None:
        launch = await self.get_spark_launch(rid)
        if not launch:
            return None
        try:
            name, metadata = await self._resolve_spark_reference(launch.get("reference") or "")
            error = None
        except ValueError as e:
            name, metadata, error = launch.get("name"), launch.get("metadata"), str(e)
        async with self.lock:
            launch.update({"name": name, "metadata": metadata, "updated_at": time.time(), "resolve_error": error})
            self._save_spark_launches()
        return launch

    async def delete_spark_launch(self, rid: str) -> bool:
        async with self.lock:
            before = len(self.spark_launches)
            self.spark_launches = [r for r in self.spark_launches if r.get("id") != rid]
            if len(self.spark_launches) < before:
                self._save_spark_launches()
                return True
        return False

    async def get_spark_launch(self, rid: str) -> dict | None:
        return next((r for r in self.spark_launches if r.get("id") == rid), None)

    # ---- spark run execution ----
    def _public_spark_runs(self) -> list[dict]:
        now = time.time()
        out = []
        for run_id, run in self.spark_runs.items():
            lines = run.get("log_lines") or []
            out.append({
                "id": run_id,
                "recipe_id": run.get("recipe_id"),
                "recipe_name": run.get("recipe_name"),
                "status": run.get("status"),
                "started_at": run.get("started_at"),
                "elapsed_s": round(now - run["started_at"], 1) if run.get("started_at") else None,
                "last_line": lines[-1] if lines else None,
                "error": run.get("error"),
                "max_concurrency": run.get("max_concurrency"),
            })
        # Most recent first.
        out.sort(key=lambda r: r.get("started_at") or 0, reverse=True)
        return out

    async def run_spark_launch_stream(self, rid: str, solo: bool = True,
                                      overrides: dict | None = None):
        """Spawn `sparkrun run <reference> [overrides]` and stream
        stdout+stderr as SSE `data: {...}` events. Records the run in
        self.spark_runs. Best-effort: a missing `sparkrun` binary yields an
        error event rather than raising."""
        launch = await self.get_spark_launch(rid)
        if not launch:
            yield f"data: {json.dumps({'error': f'SparkRun {rid} not found'})}\n\n"
            yield "data: {\"done\": true}\n\n"
            return

        # sparkrun launches a model container — free the GPU first.
        yield f"data: {json.dumps({'status': 'evicting other backends…'})}\n\n"
        try:
            await self.evict_other_backends(protect="vllm")
        except Exception as e:
            yield f"data: {json.dumps({'warning': f'eviction skipped: {e}'})}\n\n"

        run_id = uuid.uuid4().hex[:8]
        run = {
            "id": run_id,
            "recipe_id": rid,
            "recipe_name": launch.get("name"),
            "reference": launch.get("reference"),
            "status": "starting",
            "started_at": time.time(),
            "log_lines": deque(maxlen=2000),
            "error": None,
            "proc": None,
        }
        self.spark_runs[run_id] = run
        yield f"data: {json.dumps({'run_id': run_id, 'status': 'starting', 'recipe': launch.get('name')})}\n\n"

        try:
            executable = self._sparkrun_executable()
        except ValueError as e:
            run["status"] = "error"
            run["error"] = str(e)
            run["log_lines"].append(run["error"])
            yield f"data: {json.dumps({'error': run['error']})}\n\n"
            yield "data: {\"done\": true}\n\n"
            return
        cmd = [executable, "run", launch.get("reference") or ""]
        if solo:
            cmd.append("--solo")
        overrides = overrides or {}
        flag_map = {"tensor_parallel": "--tp", "pipeline_parallel": "--pp", "port": "--port",
                    "gpu_memory_utilization": "--gpu-mem", "max_model_len": "--max-model-len",
                    "served_model_name": "--served-model-name"}
        for key, flag in flag_map.items():
            value = overrides.get(key)
            if value not in (None, ""):
                cmd.extend([flag, str(value)])
        parallel_streams = overrides.get("parallel_streams")
        if parallel_streams not in (None, ""):
            try:
                parallel_streams = int(parallel_streams)
                if parallel_streams < 1:
                    raise ValueError
            except (TypeError, ValueError):
                yield f"data: {json.dumps({'error': 'Parallel streams must be a positive integer'})}\n\n"
                yield "data: {\"done\": true}\n\n"
                return
            cmd.extend(["-o", f"max_num_seqs={parallel_streams}"])
        else:
            recipe_default = (
                (launch.get("metadata") or {}).get("defaults") or {}
            ).get("max_num_seqs")
            try:
                parallel_streams = int(recipe_default)
            except (TypeError, ValueError):
                parallel_streams = None
        run["max_concurrency"] = parallel_streams
        for option in overrides.get("options") or []:
            option = str(option).strip()
            if option and "=" in option and "\n" not in option and "\r" not in option:
                cmd.extend(["-o", option])
        temp_recipe: str | None = None
        try:
            target, temp_recipe, repairs = await self._spark_run_target(launch)
        except ValueError as e:
            run["status"] = "error"
            run["error"] = f"SparkRun recipe could not be prepared: {e}"
            run["log_lines"].append(run["error"])
            yield f"data: {json.dumps({'error': run['error']})}\n\n"
            yield "data: {\"done\": true}\n\n"
            return
        cmd[2] = target
        if repairs:
            warning = f"normalized {repairs} doubled-brace JSON argument from the resolved recipe"
            run["log_lines"].append(warning)
            yield f"data: {json.dumps({'warning': warning})}\n\n"
        yield f"data: {json.dumps({'status': 'running', 'cmd': ' '.join(cmd)})}\n\n"

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except FileNotFoundError:
            run["status"] = "error"
            run["error"] = "`sparkrun` not found on PATH. Install it with `uvx sparkrun setup install`."
            run["log_lines"].append(run["error"])
            yield f"data: {json.dumps({'error': run['error']})}\n\n"
            yield "data: {\"done\": true}\n\n"
            if temp_recipe:
                Path(temp_recipe).unlink(missing_ok=True)
            return

        run["proc"] = proc
        run["status"] = "running"
        try:
            if proc.stdout is None:
                raise RuntimeError("sparkrun provided no stdout stream")
            async for raw in proc.stdout:
                line = raw.decode("utf-8", errors="replace").rstrip()
                run["log_lines"].append(line)
                yield f"data: {json.dumps({'line': line})}\n\n"
            await proc.wait()
            if proc.returncode == 0:
                run["status"] = "done"
                yield f"data: {json.dumps({'status': 'done', 'returncode': 0})}\n\n"
            else:
                run["status"] = "error"
                run["error"] = f"sparkrun exited with code {proc.returncode}"
                yield f"data: {json.dumps({'status': 'error', 'returncode': proc.returncode})}\n\n"
        except asyncio.CancelledError:
            run["status"] = "canceled"
            try: proc.terminate()
            except Exception: pass
            yield f"data: {json.dumps({'status': 'canceled'})}\n\n"
        except Exception as e:
            run["status"] = "error"
            run["error"] = str(e)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        if temp_recipe:
            Path(temp_recipe).unlink(missing_ok=True)
        yield "data: {\"done\": true}\n\n"

    async def cancel_spark_run(self, run_id: str) -> dict:
        run = self.spark_runs.get(run_id)
        if not run:
            return {"error": "run not found"}
        proc = run.get("proc")
        if proc and proc.returncode is None:
            try:
                proc.terminate()
            except Exception as e:
                return {"error": f"terminate failed: {e}"}
            run["status"] = "canceled"
        return {"ok": True, "status": run.get("status")}

    def get_spark_run_logs(self, run_id: str) -> list[str] | None:
        run = self.spark_runs.get(run_id)
        if not run:
            return None
        return list(run.get("log_lines") or [])

    async def container_to_recipe(self, name: str) -> dict:
        """Inspect a container's config and create a recipe from it."""
        def _inspect():
            try:
                c = self.client.containers.get(name)
            except Exception:
                return None
            return c

        c = await asyncio.to_thread(_inspect)
        if c is None:
            raise ValueError(f"Container '{name}' not found")

        labels = c.labels or {}
        model = _label_value(labels, MODEL_LABEL, "")
        engine = _label_value(labels, ENGINE_LABEL, "vllm")
        attrs = c.attrs or {}
        image_tag = attrs.get("Config", {}).get("Image") or attrs.get("Image") or None

        # Parse command to extract gpu_memory_utilization and extra_args
        cmd = c.attrs.get("Config", {}).get("Cmd") or []
        gpu_mem = None
        extra_args = []
        sg_tp_size = sg_context_length = sg_max_running_requests = sg_mem_fraction = None

        if cmd and engine == "sglang":
            # SGLang containers run: python -m sglang.launch_server
            # --model-path <model> --host … --port … [sg flags] [extra args].
            # Map the controller-managed flags into recipe fields; keep the
            # rest as extra args.
            def _sg_flag(flag, cast):
                try:
                    i = cmd.index(flag)
                    return cast(cmd[i + 1])
                except (ValueError, IndexError):
                    return None

            sg_tp_size = _sg_flag("--tp-size", int)
            sg_context_length = _sg_flag("--context-length", int)
            sg_max_running_requests = _sg_flag("--max-running-requests", int)
            sg_mem_fraction = _sg_flag("--mem-fraction-static", float)
            managed = {
                "--model-path", "--host", "--port", "--tp-size",
                "--context-length", "--max-running-requests",
                "--mem-fraction-static", "--max-total-tokens",
            }
            skip_tokens = {"-m", "python", "python3", "sglang.launch_server", model}
            i = 0
            while i < len(cmd):
                tok = cmd[i]
                if tok in managed:
                    i += 2  # skip flag + value
                    continue
                if tok in skip_tokens:
                    i += 1
                    continue
                extra_args.append(tok)
                i += 1

        elif cmd:
            # Find --gpu-memory-utilization value
            try:
                gmu_idx = cmd.index("--gpu-memory-utilization")
                if gmu_idx + 1 < len(cmd):
                    gpu_mem = float(cmd[gmu_idx + 1])
            except (ValueError, IndexError):
                pass

            # Collect extra args: everything after the model that isn't
            # --host, --port, --gpu-memory-utilization (which we manage)
            # or the model itself, or "vllm"/"serve"
            skip_flags = {"--host", "--port", "--gpu-memory-utilization"}
            # Find model position in cmd
            model_idx = None
            if "serve" in cmd:
                si = cmd.index("serve")
                if si + 1 < len(cmd):
                    model_idx = si + 1
            elif model and model in cmd:
                model_idx = cmd.index(model)

            if model_idx is not None:
                # Walk tokens after the model
                i = model_idx + 1
                while i < len(cmd):
                    tok = cmd[i]
                    if tok in skip_flags:
                        i += 2  # skip flag + value
                        continue
                    extra_args.append(tok)
                    i += 1

        if not model:
            # Fallback: use container name as model hint
            model = name.replace("vllm-", "", 1).replace("sglang-", "", 1)

        return await self.add_recipe(
            model=model,
            name=model,
            image=image_tag,
            extra_args=extra_args if extra_args else None,
            gpu_memory_utilization=gpu_mem,
            engine=engine,
            sg_tp_size=sg_tp_size,
            sg_context_length=sg_context_length,
            sg_max_running_requests=sg_max_running_requests,
            sg_mem_fraction=sg_mem_fraction,
            sg_image=image_tag if engine == "sglang" else None,
        )

    # ---------- cross-backend mutual exclusion ----------
    async def evict_other_backends(self, protect: str = "") -> dict:
        """Stop other supported managed runtimes before an exclusive launch."""
        # SparkDeck only coordinates its supported managed runtimes. Legacy
        # local backends are deliberately not probed, stopped, or unloaded.
        supported_actions: dict[str, list] = {"stopped": []}
        try:
            for container in await self.list_containers():
                runtime = container.get("engine") or "vllm"
                if (
                    container.get("managed")
                    and container.get("status") == "running"
                    and runtime != protect
                ):
                    await self.stop_container(container["name"])
                    self._activity.pop(container["name"], None)
                    supported_actions["stopped"].append(container["name"])
        except Exception as exc:
            print(f"[evict] managed runtime stop failed: {exc}")
        return supported_actions

    # ---------- containers ----------
    def _container_summary(self, c) -> dict | None:
        labels = c.labels or {}

        # Avoid lazy-loading c.image (it triggers a Docker API call per
        # container). The tag/name is already present in the container attrs.
        attrs = c.attrs or {}
        image_tag = (
            attrs.get("Config", {}).get("Image", "")
            or attrs.get("Image", "")
        )

        is_managed = _label_value(labels, CONTROLLER_LABEL) == "1"
        engine_label = _label_value(labels, ENGINE_LABEL, "vllm")
        is_atlas_serving = _is_atlas_serving_container(c.name, image_tag)
        if not is_managed and not _is_vllm_image(image_tag) and not is_atlas_serving:
            return None

        # parse model from cmd or label
        model = _label_value(labels, MODEL_LABEL, "")
        cmd = c.attrs.get("Config", {}).get("Cmd") or []
        if not model:
            if engine_label == "sglang":
                # SGLang uses --model-path <model>
                if "--model-path" in cmd:
                    i = cmd.index("--model-path")
                    if i + 1 < len(cmd):
                        model = cmd[i + 1]
            elif "serve" in cmd:
                i = cmd.index("serve")
                if i + 1 < len(cmd):
                    model = cmd[i + 1]

        # Quant/dtype tag so token stats can tell apart the same model
        # served at different precisions (e.g. Q4 vs bf16).
        variant = self._variant_from_cmd(cmd)
        served_models = self._served_models_from_cmd(cmd, model)

        host_port = None
        ports = c.ports or {}
        for _, bindings in ports.items():
            if bindings:
                try:
                    host_port = int(bindings[0]["HostPort"])
                    break
                except Exception:
                    pass
        if host_port is None and _label_value(labels, SERVICE_PORT_LABEL):
            try:
                host_port = int(_label_value(labels, SERVICE_PORT_LABEL))
            except (TypeError, ValueError):
                pass

        # Estimate VRAM from model name + quant flags in the command.
        try:
            params_b, bpp = self._estimate_params_and_quant(model, cmd)
            vram_gb = round(params_b * 1e9 * bpp * 1.2 / (1024 ** 3), 1) if params_b > 0 else None
        except Exception:
            vram_gb = None

        summary = {
            "id": c.short_id,
            "name": c.name,
            "alias": getattr(self, "container_aliases", {}).get(c.name),
            "image": image_tag,
            "status": c.status,
            "model": model,
            "served_model": served_models[0] if served_models else model,
            "served_models": served_models,
            "variant": variant,
            "stats_key": self._stats_key(model, variant),
            "port": host_port,
            "managed": is_managed,
            "engine": engine_label,
            "load_settings": self._container_load_settings(cmd, engine_label, model),
            "vram_gb": vram_gb,
            "created": c.attrs.get("Created"),
            "started_at": (c.attrs.get("State") or {}).get("StartedAt"),
            "restart_count": c.attrs.get("RestartCount", 0),
        }
        if _label_value(labels, DEPLOYMENT_LABEL):
            summary.update({
                "deployment_id": _label_value(labels, DEPLOYMENT_LABEL),
                "node_id": _label_value(labels, NODE_LABEL),
                "rank": int(_label_value(labels, RANK_LABEL, "0")),
                "deployment_mode": _label_value(labels, MODE_LABEL, "single"),
                "nnodes": int(_label_value(labels, NNODES_LABEL, "1")),
            })
        if is_atlas_serving:
            summary["source"] = "atlas-serving"
        return summary

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
        # sparkrun containers hide vLLM logs inside the container.
        logs = ""
        if c.get("name", "").startswith("sparkrun_"):
            try:
                logs = await self.get_sparkrun_logs(c["name"], tail=150)
            except Exception:
                pass
        if not logs:
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
        engine: str = "vllm",
        gpu_memory_utilization: float | None = None,
        gpu_memory_gb: float | None = None,
        extra_args: list[str] | None = None,
        name: str | None = None,
        image: str | None = None,
        # SGLang-specific fields
        sg_tp_size: int | None = None,
        sg_context_length: int | None = None,
        sg_max_running_requests: int | None = None,
        sg_mem_fraction: float | None = None,
        sg_image: str | None = None,
        recipe_id: str | None = None,
        cluster_member: dict | None = None,
        hf_token: str | None = None,
        sparkdeck_deployment_id: str | None = None,
    ) -> dict:
        self._reject_hf_cli_credentials(extra_args)
        if engine not in {"vllm", "sglang"}:
            raise ValueError("engine must be vllm or sglang")
        distributed_member = bool(
            cluster_member and cluster_member.get("mode") == "sharded"
        )
        if engine == "sglang":
            if recipe_id:
                self.recipe_launches[recipe_id] = {
                    "phase": "Releasing GPU memory", "started_at": time.time(),
                }
            # SGLang has CUDA context conflicts with other engines — always evict.
            await self.evict_other_backends(protect="sglang")
            # Older recipes may have saved their image in ``image`` before
            # SGLang gained a dedicated image field, so retain it as a
            # compatibility fallback. A new recipe always gets the SGLang
            # image rather than the configured vLLM image.
            image = sg_image or image or DEFAULT_SGLANG_IMAGE
            if port is None:
                port = await self._allocate_port()
            if name is None:
                safe = model.replace("/", "-").replace("_", "-").lower()
                name = f"sglang-{safe}-{port}"
            self._cluster_launch_update(
                name, "preparing", "Preparing SGLang launch",
                model=model, cluster_member=cluster_member,
            )
            extra = list(extra_args or [])
            sg_cmd = ["-m", "sglang.launch_server", "--model-path", model]
            sg_cmd += ["--host", "0.0.0.0"]
            serve_port = int(cluster_member.get("serve_port") or port or 8000) if distributed_member else 8000
            sg_cmd += ["--port", str(serve_port)]
            if sg_tp_size and sg_tp_size > 0:
                sg_cmd += ["--tp-size", str(sg_tp_size)]
            if sg_context_length and sg_context_length > 0:
                sg_cmd += ["--context-length", str(sg_context_length)]
            if sg_mem_fraction and sg_mem_fraction > 0:
                sg_cmd += ["--mem-fraction-static", str(sg_mem_fraction)]
            if sg_max_running_requests and sg_max_running_requests > 0:
                # Calculate max-total-tokens as max_running_requests * context_length * 2
                ctx = sg_context_length or 32768
                max_total = sg_max_running_requests * ctx * 2
                sg_cmd += ["--max-running-requests", str(sg_max_running_requests)]
                sg_cmd += ["--max-total-tokens", str(max_total)]
            else:
                if sg_context_length and sg_context_length > 0:
                    # Default to 8 running requests if not specified
                    default_running = 8
                    sg_cmd += ["--max-running-requests", str(default_running)]
                    max_total = default_running * sg_context_length * 2
                    sg_cmd += ["--max-total-tokens", str(max_total)]
            sg_cmd += extra

            def _create():
                # docker-py does not pull a missing image as part of
                # containers.run().  SGLang recipes use a separate default
                # image, so make the first launch self-contained.
                try:
                    self._cluster_launch_update(
                        name, "checking_image", f"Checking Docker image {image}",
                        model=model, cluster_member=cluster_member,
                    )
                    if recipe_id:
                        self.recipe_launches[recipe_id].update({
                            "phase": "Checking SGLang image", "image": image,
                        })
                    self.client.images.get(image)
                except docker.errors.ImageNotFound:
                    self._cluster_launch_update(
                        name, "pulling_image",
                        f"Downloading Docker image {image}; this can take several minutes",
                        model=model, cluster_member=cluster_member,
                    )
                    if recipe_id:
                        self.recipe_launches[recipe_id].update({
                            "phase": "Downloading SGLang image", "image": image,
                        })
                    print(f"[sglang] pulling missing image: {image}")
                    self.client.images.pull(image)
                self._cluster_launch_update(
                    name, "creating_container", "Creating Docker container",
                    model=model, cluster_member=cluster_member,
                )
                if recipe_id:
                    self.recipe_launches[recipe_id].update({
                        "phase": "Creating container", "image": image,
                    })
                labels = {
                    CONTROLLER_LABEL: "1", MODEL_LABEL: model,
                    ENGINE_LABEL: "sglang",
                }
                if sparkdeck_deployment_id:
                    labels[DEPLOYMENT_LABEL] = sparkdeck_deployment_id
                if cluster_member:
                    labels.update({
                        DEPLOYMENT_LABEL: cluster_member["deployment_id"],
                        NODE_LABEL: cluster_member["node_id"],
                        RANK_LABEL: str(cluster_member["rank"]),
                        SERVICE_PORT_LABEL: (
                            str(serve_port)
                            if distributed_member and (
                                cluster_member.get("mode") != "sharded"
                                or int(cluster_member.get("rank", 0)) == 0
                            )
                            else ""
                        ),
                        MODE_LABEL: cluster_member.get("mode", "single"),
                        NNODES_LABEL: str(cluster_member.get("nnodes", 1)),
                    })
                run_options = {
                    "image": image,
                    "command": sg_cmd,
                    "entrypoint": ["python3"],
                    "name": name,
                    "detach": True,
                    "volumes": self._build_volumes(
                        model, self.settings["hf_cache"], image
                    ),
                    "ipc_mode": "host",
                    "shm_size": self.settings["shm_size"],
                    "device_requests": [
                        docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
                    ],
                    "labels": labels,
                    "restart_policy": {"Name": "unless-stopped"},
                }
                hf_environment = self._container_hf_environment(hf_token)
                if hf_environment:
                    run_options["environment"] = hf_environment
                if distributed_member:
                    run_options["network_mode"] = "host"
                    run_options["ulimits"] = [
                        docker.types.Ulimit(name="memlock", soft=-1, hard=-1)
                    ]
                    iface = cluster_member.get("fabric_interface")
                    if iface:
                        run_options.setdefault("environment", {}).update(
                            self._distributed_network_environment(iface)
                        )
                    if Path("/dev/infiniband").exists():
                        run_options["devices"] = ["/dev/infiniband:/dev/infiniband"]
                else:
                    run_options["ports"] = {"8000/tcp": port}
                container = self.client.containers.run(**run_options)
                container.reload()
                self._cluster_launch_update(
                    name, "starting", "Container created; starting the model server",
                    model=model, cluster_member=cluster_member,
                )
                if recipe_id:
                    self.recipe_launches[recipe_id].update({"phase": "Starting model"})
                summary = self._container_summary(container)
                if summary is not None:
                    summary["model_source"] = self._created_container_model_source(
                        container, model
                    )
                return summary
            try:
                return await asyncio.to_thread(_create)
            except Exception as exc:
                safe_error = self._redact_hf_secret(exc, hf_token)
                self._cluster_launch_update(
                    name, "error", f"Launch failed: {safe_error}",
                    model=model, cluster_member=cluster_member, error=safe_error,
                )
                raise RuntimeError(safe_error) from exc
        else:
            # vLLM path — VRAM-aware multi-model
            image = image or self.settings["vllm_image"]
            if port is None:
                port = await self._allocate_port()
            if gpu_memory_gb is not None and gpu_memory_gb > 0:
                total_gb = await self._gpu_total_gb()
                if total_gb:
                    gpu_memory_utilization = min(0.98, max(0.1, gpu_memory_gb / total_gb))
            if gpu_memory_utilization is None:
                gpu_memory_utilization = self.settings["default_gpu_memory_utilization"]
            if name is None:
                safe = model.replace("/", "-").replace("_", "-").lower()
                name = f"vllm-{safe}-{port}"
            self._cluster_launch_update(
                name, "preparing", "Preparing vLLM launch",
                model=model, cluster_member=cluster_member,
            )
            extra = self._with_vllm_prompt_token_details(
                list(extra_args or [])
            )

            serve_port = int(cluster_member.get("serve_port") or port or 8000) if distributed_member else 8000
            cmd = [
                "vllm", "serve", model,
                "--host", "0.0.0.0",
                "--port", str(serve_port),
                "--gpu-memory-utilization", str(gpu_memory_utilization),
                *extra,
            ]

            # ── VRAM estimation ────────────────────────────────
            params_b, bpp = self._estimate_params_and_quant(model, cmd)
            if params_b > 0:
                need_gb = params_b * 1e9 * bpp * 1.2 / (1024 ** 3)
                if cluster_member and cluster_member.get("mode") == "sharded":
                    need_gb /= max(1, int(cluster_member.get("nnodes", 1)))
            else:
                total_gb = await self._gpu_total_gb() or 122.0
                need_gb = total_gb * gpu_memory_utilization

            # Evict LRU containers only if the new model won't fit
            self._try_fit_new_model(need_gb, protect_name=name)

            # Cap --gpu-memory-utilization to actual free memory so
            # the new container doesn't OOM when another model already
            # occupies GPU memory (vLLM sees the full GPU, not free MiB).
            total_gb, used_gb = self._read_gpu_memory_gb()
            if total_gb > 0:
                free_gb = total_gb - used_gb - GPU_VRAM_BUFFER_GB
                free_fraction = max(0.1, free_gb / total_gb)
                gpu_memory_utilization = min(gpu_memory_utilization, free_fraction)
                # cmd layout: [vllm, serve, model, --host, IP, --port, PORT, --gpu-memory-utilization, VALUE, *extra]
                cmd[8] = str(gpu_memory_utilization)  # --gpu-memory-utilization value

            def _create():
                try:
                    self._cluster_launch_update(
                        name, "checking_image", f"Checking Docker image {image}",
                        model=model, cluster_member=cluster_member,
                    )
                    self.client.images.get(image)
                except docker.errors.ImageNotFound:
                    self._cluster_launch_update(
                        name, "pulling_image",
                        f"Downloading Docker image {image}; this can take several minutes",
                        model=model, cluster_member=cluster_member,
                    )
                    print(f"[vllm] pulling missing image: {image}")
                    self.client.images.pull(image)
                self._cluster_launch_update(
                    name, "creating_container", "Creating Docker container",
                    model=model, cluster_member=cluster_member,
                )
                labels = {
                    CONTROLLER_LABEL: "1", MODEL_LABEL: model,
                    ENGINE_LABEL: "vllm",
                }
                if sparkdeck_deployment_id:
                    labels[DEPLOYMENT_LABEL] = sparkdeck_deployment_id
                if cluster_member:
                    labels.update({
                        DEPLOYMENT_LABEL: cluster_member["deployment_id"],
                        NODE_LABEL: cluster_member["node_id"],
                        RANK_LABEL: str(cluster_member["rank"]),
                        SERVICE_PORT_LABEL: (
                            str(serve_port)
                            if distributed_member and (
                                cluster_member.get("mode") != "sharded"
                                or int(cluster_member.get("rank", 0)) == 0
                            )
                            else ""
                        ),
                        MODE_LABEL: cluster_member.get("mode", "single"),
                        NNODES_LABEL: str(cluster_member.get("nnodes", 1)),
                    })
                run_options = {
                    "image": image,
                    "command": cmd,
                    # Neutralize the image's ENTRYPOINT so our `command` is the
                    # full invocation. Works for both nvcr.io/nvidia/vllm
                    # (entrypoint is a thin shim) and vllm/vllm-openai
                    # (entrypoint is `vllm serve` and would double-prefix).
                    "entrypoint": [],
                    "name": name,
                    "detach": True,
                    "volumes": self._build_volumes(
                        model, self.settings["hf_cache"], image
                    ),
                    "ipc_mode": "host",
                    "shm_size": self.settings["shm_size"],
                    "device_requests": [
                        docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
                    ],
                    "labels": labels,
                    "restart_policy": {"Name": "unless-stopped"},
                }
                hf_environment = self._container_hf_environment(hf_token)
                if hf_environment:
                    run_options["environment"] = hf_environment
                if distributed_member:
                    run_options["network_mode"] = "host"
                    run_options["ulimits"] = [
                        docker.types.Ulimit(name="memlock", soft=-1, hard=-1)
                    ]
                    iface = cluster_member.get("fabric_interface")
                    if iface:
                        run_options.setdefault("environment", {}).update(
                            self._distributed_network_environment(iface)
                        )
                    if Path("/dev/infiniband").exists():
                        run_options["devices"] = ["/dev/infiniband:/dev/infiniband"]
                else:
                    run_options["ports"] = {"8000/tcp": port}
                container = self.client.containers.run(**run_options)
                container.reload()
                self._cluster_launch_update(
                    name, "starting", "Container created; starting the model server",
                    model=model, cluster_member=cluster_member,
                )
                summary = self._container_summary(container)
                if summary is not None:
                    summary["model_source"] = self._created_container_model_source(
                        container, model
                    )
                return summary
            try:
                return await asyncio.to_thread(_create)
            except Exception as exc:
                safe_error = self._redact_hf_secret(exc, hf_token)
                self._cluster_launch_update(
                    name, "error", f"Launch failed: {safe_error}",
                    model=model, cluster_member=cluster_member, error=safe_error,
                )
                raise RuntimeError(safe_error) from exc

    async def start_container(self, name: str) -> dict:
        # VRAM-aware: evict only if the GPU is full.
        # Estimate VRAM from the container's model label.
        try:
            c = self.client.containers.get(name)
            model = _label_value(c.labels or {}, MODEL_LABEL, "")
            params_b, bpp = self._estimate_params_and_quant(model)
            if params_b > 0:
                need_gb = params_b * 1e9 * bpp * 1.2 / (1024 ** 3)
                labels = c.labels or {}
                if _label_value(labels, MODE_LABEL) == "sharded":
                    need_gb /= max(1, int(_label_value(labels, NNODES_LABEL, "1")))
            else:
                need_gb = 30.0  # conservative fallback
            self._try_fit_new_model(need_gb, protect_name=name)
        except Exception:
            pass  # best-effort; if we can't read the container, proceed anyway
        def _do():
            container = self.client.containers.get(name)
            # A manual Stop disables automatic restarts. Re-enable them only
            # when the user explicitly starts this managed container again.
            container.update(restart_policy={"Name": "unless-stopped"})
            container.start()
        await asyncio.to_thread(_do)
        return {"ok": True}

    @staticmethod
    def _replace_command_option(
        flags: str, names: set[str], value: str | int | float | None
    ) -> str:
        """Replace scalar CLI options without disturbing other shell text."""
        option = "(?:" + "|".join(
            re.escape(name) for name in sorted(names, key=len, reverse=True)
        ) + ")"
        scalar = r'''(?:"(?:\\.|[^"])*"|'(?:\\.|[^'])*'|[^\s]+)'''
        pattern = rf"(?<!\S){option}(?:={scalar}|\s+{scalar})"
        updated = re.sub(pattern, "", flags or "")
        # Do not normalize internal whitespace: image-specific flags may carry
        # shell expressions or quoted templates whose contents are significant.
        updated = updated.strip()
        if value is None or str(value).strip() == "":
            return updated
        canonical = sorted(names, key=lambda name: (len(name), name))[0]
        return f"{updated} {canonical} {value}".strip()

    def _replace_thinking_config(self, flags: str, mode: str) -> str:
        """Update thinking without discarding unrelated chat-template kwargs."""
        if mode not in {"default", "enabled", "disabled"}:
            raise ValueError("thinking_mode must be default, enabled, or disabled")
        try:
            tokens = shlex.split(flags)
        except ValueError as exc:
            raise ValueError("command flags have invalid shell quoting") from exc
        _, payload, key = self._thinking_config(tokens)
        if mode == "default":
            if key is None:
                return flags
            payload.pop(key, None)
        else:
            key = key or "enable_thinking"
            payload[key] = mode == "enabled"
        value = None
        if payload:
            value = shlex.quote(json.dumps(payload, separators=(",", ":")))
        return self._replace_command_option(
            flags, {"--default-chat-template-kwargs"}, value
        )

    def _updated_container_command(
        self, cmd: list[str], engine: str, model: str, settings: dict
    ) -> list[str]:
        """Apply an edited settings form to an existing Docker command."""
        existing = self._container_load_settings(cmd, engine, model)
        flags = str(settings.get("command_flags", existing["command_flags"]) or "")
        if len(flags) > 65536:
            raise ValueError("command flags must be 65536 characters or fewer")

        def positive_int(key: str):
            value = settings.get(key, existing.get(key))
            if value in (None, ""):
                return None
            try:
                value = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{key} must be a positive integer") from exc
            if value <= 0:
                raise ValueError(f"{key} must be a positive integer")
            return value

        gpu = settings.get(
            "gpu_memory_utilization", existing.get("gpu_memory_utilization")
        )
        if gpu not in (None, ""):
            try:
                gpu = float(gpu)
            except (TypeError, ValueError) as exc:
                raise ValueError("gpu_memory_utilization must be a number") from exc
            if not 0 < gpu <= 1:
                raise ValueError("gpu_memory_utilization must be between 0 and 1")
        else:
            gpu = None
        concurrency = positive_int("max_concurrency")
        context_window = positive_int("context_window")
        kv_dtype = settings.get("kv_cache_dtype", existing.get("kv_cache_dtype"))
        kv_dtype = str(kv_dtype).strip() if kv_dtype not in (None, "") else None
        thinking_mode = settings.get(
            "thinking_mode", existing.get("thinking_mode", "default")
        )

        if engine == "sglang":
            flags = self._replace_command_option(
                flags, {"--mem-fraction-static"}, gpu
            )
            flags = self._replace_command_option(
                flags, {"--max-running-requests"}, concurrency
            )
            flags = self._replace_command_option(
                flags, {"--context-length"}, context_window
            )
        else:
            flags = self._replace_command_option(
                flags,
                {"--gpu-memory-utilization", "--gpu_memory_utilization"},
                gpu,
            )
            flags = self._replace_command_option(
                flags, {"--max-num-seqs"}, concurrency
            )
            flags = self._replace_command_option(
                flags, {"--max-model-len", "--max-model-length"}, context_window
            )
        flags = self._replace_command_option(flags, {"--kv-cache-dtype"}, kv_dtype)
        flags = self._replace_thinking_config(flags, str(thinking_mode))

        original = [str(value) for value in (cmd or [])]
        # Preserve host/port because Docker's network bindings and controller
        # routing still point at the existing values.
        if engine == "vllm" and len(original) >= 3 and original[-2] in {"-c", "-lc"}:
            match = self._shell_vllm_command(original[-1])
            if not match:
                raise ValueError("could not locate the vLLM command in the shell script")
            old_flags = match.group("flags")
            try:
                old_tokens = shlex.split(old_flags)
            except ValueError as exc:
                raise ValueError("the existing vLLM command has invalid shell quoting") from exc
            for names in ({"--host"}, {"--port"}):
                flags = self._replace_command_option(
                    flags, names, self._cli_option(old_tokens, names)
                )
            script = original[-1]
            trailing = "\n" if script.endswith("\n") else ""
            script = script[:match.start("flags")].rstrip() + " " + flags + trailing
            return [*original[:-1], script]

        try:
            flag_tokens = shlex.split(flags)
        except ValueError as exc:
            raise ValueError("command flags have invalid shell quoting") from exc
        if engine == "sglang":
            try:
                model_index = original.index("--model-path")
                prefix = original[:model_index + 2]
            except ValueError as exc:
                raise ValueError("could not locate --model-path in the SGLang command") from exc
        else:
            try:
                model_index = original.index("serve") + 1
                prefix = original[:model_index + 1]
            except ValueError as exc:
                raise ValueError("could not locate the model in the vLLM command") from exc
        # Keep the original network endpoint even if it was edited in the raw
        # flags field; changing it would make the existing port mapping stale.
        old_flag_tokens = original[len(prefix):]
        for names in ({"--host"}, {"--port"}):
            flag_tokens = self._without_cli_options(flag_tokens, names)
            old_value = self._cli_option(old_flag_tokens, names)
            if old_value is not None:
                flag_tokens += [sorted(names)[0], str(old_value)]
        return [*prefix, *flag_tokens]

    async def update_container_settings(self, name: str, settings: dict) -> dict:
        """Transactionally recreate a Docker model with an edited command.

        Docker cannot mutate a container command in place. The original is
        stopped and renamed, then retained until the exact-config clone has
        been created (and started when appropriate). A failed replacement is
        removed and the original name/state is restored.
        """
        if not isinstance(settings, dict):
            raise ValueError("settings must be an object")

        async with self.lock:
            container = await asyncio.to_thread(self.client.containers.get, name)
            container.reload()
            labels = container.labels or {}
            if _label_value(labels, DEPLOYMENT_LABEL):
                raise ValueError("edit the deployment recipe instead of one cluster member")
            attrs = copy.deepcopy(container.attrs or {})
            config = copy.deepcopy(attrs.get("Config") or {})
            cmd = config.get("Cmd") or []
            engine = _label_value(labels, ENGINE_LABEL, "vllm")
            model = _label_value(labels, MODEL_LABEL, "")
            new_cmd = self._updated_container_command(cmd, engine, model, settings)
            was_running = container.status in {"running", "restarting", "paused"}
            backup_name = f"{name}.settings-backup-{uuid.uuid4().hex[:8]}"

            create_config = config
            create_config["Cmd"] = new_cmd
            create_config["HostConfig"] = copy.deepcopy(attrs.get("HostConfig") or {})

            def _replace():
                backup = container
                if was_running:
                    backup.stop(timeout=30)
                backup.rename(backup_name)
                try:
                    created = self.client.api.create_container_from_config(
                        create_config, name=name
                    )
                    replacement = self.client.containers.get(created["Id"])
                    if was_running:
                        replacement.start()
                    replacement.reload()
                    summary = self._container_summary(replacement)
                    backup.remove(force=True)
                    return summary
                except Exception:
                    try:
                        self.client.containers.get(name).remove(force=True)
                    except Exception:
                        pass
                    backup = self.client.containers.get(backup_name)
                    backup.rename(name)
                    if was_running:
                        backup.start()
                    raise

            summary = await asyncio.to_thread(_replace)
            return {"ok": True, "container": summary}

    async def is_managed_container(self, name: str) -> bool:
        def _check():
            container = self.client.containers.get(name)
            return _label_value(container.labels or {}, CONTROLLER_LABEL) == "1"
        try:
            return await asyncio.to_thread(_check)
        except Exception:
            return False

    async def stop_container(self, name: str) -> dict:
        def _do():
            container = self.client.containers.get(name)
            # Prevent Docker's restart policy from resurrecting a model that
            # the user explicitly stopped. Do this before signalling SGLang,
            # whose shutdown can end in SIGKILL under GPU memory pressure.
            container.update(restart_policy={"Name": "no"})
            container.reload()
            if container.status in ("running", "restarting", "paused"):
                container.stop(timeout=10)
                container.reload()
            if container.status not in ("exited", "dead", "created"):
                container.kill()
                container.reload()
            if container.status not in ("exited", "dead", "created"):
                raise RuntimeError(f"container remained {container.status} after stop")
        try:
            await asyncio.wait_for(asyncio.to_thread(_do), timeout=30)
        except asyncio.TimeoutError:
            raise RuntimeError(f"container stop timed out after 30s")
        return {"ok": True}

    async def remove_container(self, name: str) -> dict:
        def _do():
            self.client.containers.get(name).remove(force=True)
        await asyncio.to_thread(_do)
        getattr(self, "cluster_member_launches", {}).pop(name, None)
        aliases = getattr(self, "container_aliases", {})
        if name in aliases:
            aliases.pop(name, None)
            self._save_container_aliases()
        return {"ok": True}

    async def get_logs(self, name: str, tail: int = 200) -> str:
        def _do():
            return self.client.containers.get(name).logs(tail=tail).decode("utf-8", errors="replace")
        return await asyncio.to_thread(_do)

    async def get_sparkrun_logs(self, name: str, tail: int = 150) -> str:
        """Read the internal vLLM log that sparkrun writes inside the container.
        sparkrun containers run 'sleep infinity' as the docker entrypoint and
        spawn vLLM in the background, logging to /tmp/sparkrun_serve.log."""
        def _do():
            try:
                c = self.client.containers.get(name)
                result = c.exec_run(
                    f"tail -n {tail} /tmp/sparkrun_serve.log",
                    stdout=True, stderr=True,
                )
                if getattr(result, "exit_code", None) != 0:
                    return ""
                output = getattr(result, "output", result)
                if isinstance(output, bytes):
                    return output.decode("utf-8", errors="replace")
                return str(output)
            except Exception:
                return ""
        return await asyncio.to_thread(_do)

    # ---------- images ----------
    async def list_images(self) -> list[dict]:
        now = time.time()
        if now - self._images_ts < 15 and self._images_cache:
            return self._images_cache

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
        self._images_cache = await asyncio.to_thread(_do)
        self._images_ts = now
        return self._images_cache

    async def cluster_image_inventory(self) -> dict:
        """Return image/container inventory from each enabled cluster node."""
        nodes = [
            node for node in await self.cluster_nodes()
            if node.get("local") or node.get("enabled", True)
        ]

        async def fetch(node: dict) -> dict:
            if node.get("local"):
                images, containers = await asyncio.gather(
                    self.list_images(), self.list_containers(),
                )
                payload = {"images": images, "containers": containers}
            else:
                payload = await self.node_registry.request(
                    node["id"], "GET", "/api/agent/images", timeout=15,
                )
            if not isinstance(payload, dict):
                raise RuntimeError("node returned an invalid image inventory")
            return {
                "node": self.public_target_node(node),
                "images": payload.get("images") or [],
                "containers": payload.get("containers") or [],
            }

        fetched = await asyncio.gather(*(fetch(node) for node in nodes), return_exceptions=True)
        results = []
        errors = []
        for node, result in zip(nodes, fetched):
            if isinstance(result, Exception):
                errors.append({
                    "node": self.public_target_node(node),
                    "error": str(result)[:500],
                })
            else:
                results.append(result)
        return {"results": results, "errors": errors, "partial": bool(errors)}

    async def remove_image(self, image_id: str) -> dict:
        def _do():
            self.client.images.remove(image_id, force=True)
        await asyncio.to_thread(_do)
        self._images_cache = []
        self._images_ts = 0
        return {"ok": True, "image": image_id}

    async def remove_image_on_nodes(
        self, image_id: str, node_ids: list[str],
    ) -> dict:
        """Remove an image from every recorded owner without failing fast."""
        image_id = str(image_id or "").strip()
        requested = list(dict.fromkeys(str(value).strip() for value in node_ids))
        if not image_id:
            raise ValueError("image ID is required")
        if not requested or any(not value for value in requested):
            raise ValueError("image owners are required")

        available = {node["id"]: node for node in await self.cluster_nodes()}

        async def remove(node_id: str) -> Any:
            node = available.get(node_id)
            if not node:
                raise RuntimeError("owning node is no longer registered")
            if node_id == LOCAL_NODE_ID:
                return await self.remove_image(image_id)
            return await self.node_registry.request(
                node_id, "DELETE",
                f"/api/agent/images/{quote(image_id, safe='')}", timeout=120,
            )

        removed = await asyncio.gather(
            *(remove(node_id) for node_id in requested), return_exceptions=True,
        )
        results = []
        for node_id, result in zip(requested, removed):
            node = available.get(node_id) or {"id": node_id, "name": node_id}
            item = {
                "node_id": node_id,
                "node_name": node.get("name") or node_id,
                "ok": not isinstance(result, Exception),
            }
            if isinstance(result, Exception):
                item["error"] = str(result)
            results.append(item)
        selected = [available[node_id] for node_id in requested if node_id in available]
        return {
            "ok": all(item["ok"] for item in results),
            "image": image_id,
            "node_ids": requested,
            "selected_nodes": [self.public_target_node(node) for node in selected],
            "results": results,
        }

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

    async def pull_image_result(self, image: str) -> dict:
        """Pull one image locally and collapse Docker progress into a result."""
        error = None
        async for event in self.pull_image_stream(image):
            try:
                payload = json.loads(event.removeprefix("data:").strip())
            except (AttributeError, json.JSONDecodeError):
                continue
            if payload.get("error"):
                error = str(payload["error"])
        if error:
            raise RuntimeError(error)
        return {"ok": True, "image": image}

    async def pull_image_on_nodes(
        self, image: str, node_ids: list[str] | None = None,
    ) -> dict:
        """Pull a Docker image concurrently on exactly the selected nodes."""
        image = str(image or "").strip()
        if not image:
            raise ValueError("image is required")
        selected = await self.selected_cluster_nodes(node_ids)

        async def pull(node: dict) -> dict:
            if node["id"] == LOCAL_NODE_ID:
                return await self.pull_image_result(image)
            return await self.node_registry.request(
                node["id"], "POST", "/api/agent/images/pull",
                json_body={"image": image}, timeout=1800,
            )

        pulled = await asyncio.gather(*(pull(node) for node in selected), return_exceptions=True)
        results = []
        for node, result in zip(selected, pulled):
            item = {
                "node_id": node["id"],
                "node_name": node.get("name") or node["id"],
                "ok": not isinstance(result, Exception),
            }
            if isinstance(result, Exception):
                item["error"] = str(result)
            results.append(item)
        return {
            "ok": all(item["ok"] for item in results),
            "image": image,
            "node_ids": [node["id"] for node in selected],
            "selected_nodes": [self.public_target_node(node) for node in selected],
            "results": results,
        }

    # ---------- llama-server (GGUF) launcher ----------
    # The controller scans the local HF cache for GGUF repos and launches
    # llama-server itself, one model at a time, bound to localhost. The /v1
    # proxy routes matching chat/completions requests to it. (Section kept
    # under the historical "unsloth" name to avoid renaming data files and
    # API paths.)

    def _load_unsloth_settings(self) -> dict:
        if self.unsloth_settings_path.exists():
            try:
                data = json.loads(self.unsloth_settings_path.read_text())
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
        return {}

    def _save_unsloth_settings(self):
        self.unsloth_settings_path.write_text(
            json.dumps(self.unsloth_settings, indent=2)
        )

    def get_unsloth_settings(self, model_path: str) -> dict:
        """Saved per-model launch settings, merged over defaults."""
        saved = self.unsloth_settings.get(model_path, {})
        merged = {**UNSLOTH_DEFAULT_SETTINGS, **saved}
        if merged.get("split_mode") not in {"tensor", "layer", "row"}:
            merged["split_mode"] = UNSLOTH_DEFAULT_SETTINGS["split_mode"]
        return merged

    async def set_unsloth_settings(self, model_path: str, settings: dict) -> dict:
        async with self.lock:
            merged = {**UNSLOTH_DEFAULT_SETTINGS}
            for k, v in (settings or {}).items():
                if k in UNSLOTH_DEFAULT_SETTINGS:
                    merged[k] = v
            merged["parallel"] = max(1, min(16, int(merged.get("parallel") or 1)))
            merged["tensor_parallel_size"] = max(
                2, min(16, int(merged.get("tensor_parallel_size") or 2))
            )
            if merged.get("split_mode") not in {"tensor", "layer", "row"}:
                merged["split_mode"] = UNSLOTH_DEFAULT_SETTINGS["split_mode"]
            merged["mtp_predict_tokens"] = max(
                1, min(64, int(merged.get("mtp_predict_tokens") or 1))
            )
            if merged.get("mtp_enabled") and merged.get("dspark_enabled"):
                raise ValueError("MTP and DSPARK cannot be enabled at the same time")
            self.unsloth_settings[model_path] = merged
            self._save_unsloth_settings()
        return self.get_unsloth_settings(model_path)

    # ----- GGUF discovery (local HF cache) -----

    def _gguf_cache_root(self) -> Path:
        return Path(self.settings.get("hf_cache") or "") / "hub"

    @staticmethod
    def _quant_from_filename(repo_name: str, f: Path) -> str | None:
        """Quant tag for a .gguf file: 'BF16' for files under a BF16/ dir,
        otherwise the filename stem minus the model-name prefix and any
        shard suffix. None for non-model files (mmproj, non-first shards)."""
        if f.name.lower().startswith("mmproj"):
            return None
        if f.parent.name.upper() == "BF16":
            quant = "BF16"
        else:
            stem = f.stem
            # Strip the model-name prefix, dropping trailing segments
            # (e.g. "-GGUF", "-MTP") until the stem starts with it.
            prefix = repo_name
            quant = ""
            while prefix:
                if stem.startswith(prefix + "-"):
                    quant = stem[len(prefix) + 1:]
                    break
                if "-" not in prefix:
                    break
                prefix = prefix.rsplit("-", 1)[0]
            if not quant:
                quant = stem
        quant = re.sub(r"-\d{5}-of-\d{5}$", "", quant)
        return quant or None

    def _scan_gguf_models(self) -> list[dict]:
        """Scan the HF hub cache for GGUF repos. Sync; call via to_thread."""
        root = self._gguf_cache_root()
        models = []
        if not root.is_dir():
            return models
        for repo_dir in sorted(root.glob("models--*--*")):
            parts = repo_dir.name[len("models--"):].split("--", 1)
            if len(parts) != 2:
                continue
            repo_id = f"{parts[0]}/{parts[1]}"
            snaps = sorted((repo_dir / "snapshots").glob("*")) if (repo_dir / "snapshots").is_dir() else []
            if not snaps:
                continue
            snap = snaps[-1]
            quants: dict[str, dict] = {}
            dspark_drafts: list[dict] = []
            mmproj = None
            total = 0
            latest = 0.0
            for f in snap.rglob("*.gguf"):
                try:
                    st = f.stat()
                except OSError:
                    continue
                total += st.st_size
                latest = max(latest, st.st_mtime)
                if f.name.lower().startswith("mmproj"):
                    mmproj = mmproj or str(f)
                    continue
                # Unsloth publishes DSpark as a separate draft GGUF, usually
                # under dspark/ and/or with a dspark-* filename.  Do not show
                # it as a target-model quantization in the main variant list.
                is_dspark_draft = (
                    f.parent.name.lower() == "dspark"
                    or f.name.lower().startswith("dspark-")
                )
                if is_dspark_draft:
                    name_upper = f.stem.upper()
                    if "Q8_0" in name_upper:
                        draft_quant = "Q8_0"
                    elif "BF16" in name_upper:
                        draft_quant = "BF16"
                    else:
                        draft_quant = f.stem
                    dspark_drafts.append({
                        "quant": draft_quant,
                        "filename": f.name,
                        "path": str(f),
                        "size_bytes": st.st_size,
                    })
                    continue
                quant = self._quant_from_filename(parts[1], f)
                if not quant:
                    continue
                q = quants.setdefault(quant, {
                    "quant": quant, "filename": f.name,
                    "path": None, "size_bytes": 0, "files": [],
                })
                q["size_bytes"] += st.st_size
                q["files"].append(str(f))
                is_later_shard = (
                    re.search(r"-\d{5}-of-\d{5}$", f.stem) is not None
                    and "-00001-of-" not in f.stem
                )
                if not is_later_shard:
                    q["filename"] = f.name
                    q["path"] = str(f)
            if not quants:
                continue
            variants = sorted(
                (q for q in quants.values() if q.get("path")),
                key=lambda v: -v["size_bytes"],
            )
            if not variants:
                continue
            models.append({
                "id": repo_id,
                "name": parts[1],
                "is_gguf": True,
                "is_vision": mmproj is not None,
                "size_bytes": total,
                "last_modified": latest or None,
                "cache_path": str(repo_dir),
                "mmproj": mmproj,
                "dspark_drafts": sorted(
                    dspark_drafts,
                    key=lambda d: (
                        0 if d["quant"] == "Q8_0" else
                        1 if d["quant"] == "BF16" else 2,
                        d["filename"].lower(),
                    ),
                ),
                "variants": variants,
            })
        models.sort(key=lambda x: x["name"].lower())
        return models

    async def _find_gguf(self, model_path: str, variant: str) -> tuple[dict, dict]:
        """Resolve (model entry, variant entry) for a repo id + quant tag."""
        def normalized_quant(value: str) -> str:
            value = str(value or "").strip()
            return re.sub(r"^(?:ggml-model-|model-)", "", value, flags=re.I).upper()

        models = await asyncio.to_thread(self._scan_gguf_models)
        for m in models:
            if m["id"] != model_path:
                continue
            for v in m["variants"]:
                if (
                    v["quant"] == variant
                    or normalized_quant(v["quant"]) == normalized_quant(variant)
                ):
                    return m, v
            raise LookupError(
                f"variant '{variant}' not downloaded for '{model_path}' "
                f"(have: {', '.join(v['quant'] for v in m['variants'])})"
            )
        raise LookupError(f"GGUF model '{model_path}' not found in HF cache")

    async def list_unsloth_models(self) -> dict:
        """
        Returns {"reachable": bool, "models": [...], "loaded_model": str|None,
                 "settings": {path: {...}}, "error": str|None}.
        Models come from scanning the local HF cache; loaded_model from the
        launcher state. reachable=False when the llama-server binary is
        missing, so the UI can show that instead of crashing the state poll.
        """
        loaded = await self._unsloth_loaded_model()
        bin_path = Path(self.settings["llama_server_bin"]).expanduser()
        reachable = bin_path.is_file()
        error = None if reachable else f"llama-server binary not found at {bin_path}"
        try:
            models = await asyncio.to_thread(self._scan_gguf_models)
        except Exception as e:
            return {"reachable": reachable, "models": [], "loaded_model": loaded,
                    "launching": self._llama_launch_status(),
                    "settings": dict(self.unsloth_settings), "error": str(e)}
        # Ensure the loaded model always has a card even if it's not in the
        # cache list (so it can be unloaded). Minimal metadata in that case.
        if loaded and not any(m["id"] == loaded for m in models):
            models.append({
                "id": loaded,
                "name": loaded.split("/")[-1],
                "is_gguf": loaded.upper().endswith("GGUF"),
                "is_vision": False,
                "size_bytes": None,
                "last_modified": None,
                "cache_path": None,
                "dspark_drafts": [],
                "variants": [],
                "loaded_not_cached": True,
            })
            models.sort(key=lambda x: x["name"].lower())
        return {
            "reachable": reachable,
            "models": models,
            "loaded_model": loaded,
            "launching": self._llama_launch_status(),
            "settings": dict(self.unsloth_settings),
            "error": error,
        }

    async def list_unsloth_gguf_variants(self, model_path: str) -> dict:
        """GGUF quant variants present in the local HF cache for a model."""
        if not model_path:
            raise ValueError("model_path is required")
        models = await asyncio.to_thread(self._scan_gguf_models)
        for m in models:
            if m["id"] == model_path:
                return {"model_path": model_path, "variants": m["variants"]}
        return {"model_path": model_path, "variants": []}

    # ----- llama-server process management -----

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def _llama_rpc_binary(self) -> Path:
        """Resolve the ggml RPC worker next to the configured llama-server."""
        configured = Path(
            self.settings.get("llama_rpc_server_bin")
            or "~/.local/share/llama.cpp/ggml-rpc-server"
        ).expanduser()
        llama_bin = Path(self.settings["llama_server_bin"]).expanduser()
        candidates = [
            configured,
            llama_bin.parent / "ggml-rpc-server",
            llama_bin.parent / "rpc-server",
            llama_bin.parent / "build" / "bin" / "ggml-rpc-server",
        ]
        for candidate in candidates:
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate
        return configured

    def _llama_rpc_adopt(self) -> None:
        if self._llama_rpc_adopt_tried:
            return
        self._llama_rpc_adopt_tried = True
        try:
            state = json.loads(self._llama_rpc_state_path.read_text())
            pid = int(state.get("pid") or 0)
            host = str(state.get("host") or "")
            port = int(state.get("port") or 0)
            if pid and host and port and self._pid_alive(pid):
                cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().decode(
                    errors="replace"
                )
                if "rpc-server" in cmdline and host in cmdline and str(port) in cmdline:
                    self._llama_rpc_pid = pid
                    self._llama_rpc_host = host
                    self._llama_rpc_port = port
                    return
        except Exception:
            pass
        try:
            self._llama_rpc_state_path.unlink(missing_ok=True)
        except Exception:
            pass

    def llama_rpc_status(self) -> dict:
        self._llama_rpc_adopt()
        running = bool(
            self._llama_rpc_pid and self._pid_alive(self._llama_rpc_pid)
        )
        return {
            "available": self._llama_rpc_binary().is_file(),
            "running": running,
            "host": self._llama_rpc_host if running else None,
            "port": self._llama_rpc_port if running else None,
            "pid": self._llama_rpc_pid if running else None,
        }

    async def start_llama_rpc_worker(self, host: str, port: int | None = None) -> dict:
        """Expose this node's CUDA device to a trusted coordinator fabric."""
        host = str(host or "").strip()
        port = int(port or self.settings.get("llama_rpc_port") or 50052)
        if not host:
            raise ValueError("fabric host is required")
        if port < 1024 or port > 65535:
            raise ValueError("llama RPC port must be between 1024 and 65535")
        local_addresses = {
            address
            for interface in self._network_interfaces()
            for address in interface.get("ipv4") or []
        }
        if host not in local_addresses:
            raise ValueError(f"{host} is not an address on this node")
        binary = self._llama_rpc_binary()
        if not binary.is_file():
            raise RuntimeError(
                f"ggml-rpc-server binary not found at {binary}; rebuild llama.cpp "
                "with -DGGML_RPC=ON"
            )

        async with self._llama_rpc_lock:
            current = self.llama_rpc_status()
            if current["running"] and current["host"] == host and current["port"] == port:
                return current
            await self.stop_llama_rpc_worker()
            await self.evict_other_backends(protect="unsloth")
            self._llama_log_dir.mkdir(parents=True, exist_ok=True)
            log_file = open(self._llama_rpc_log_path, "ab")
            try:
                proc = await asyncio.create_subprocess_exec(
                    str(binary), "--host", host, "--port", str(port), "--cache",
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )
            finally:
                log_file.close()
            self._llama_rpc_proc = proc
            self._llama_rpc_pid = proc.pid
            self._llama_rpc_host = host
            self._llama_rpc_port = port
            self._llama_rpc_state_path.write_text(json.dumps({
                "pid": proc.pid,
                "host": host,
                "port": port,
                "started_at": time.time(),
            }, indent=2))

            deadline = time.time() + 30
            while time.time() < deadline:
                if proc.returncode is not None:
                    break
                try:
                    reader, writer = await asyncio.open_connection(host, port)
                    writer.close()
                    await writer.wait_closed()
                    return self.llama_rpc_status()
                except Exception:
                    await asyncio.sleep(0.5)
            tail = ""
            try:
                tail = self._llama_rpc_log_path.read_text(errors="replace")[-1200:]
            except Exception:
                pass
            await self.stop_llama_rpc_worker()
            raise RuntimeError(f"ggml-rpc-server failed to start. Log tail:\n{tail}")

    async def stop_llama_rpc_worker(self) -> dict:
        self._llama_rpc_adopt()
        pid = self._llama_rpc_pid
        if pid and self._pid_alive(pid):
            try:
                os.kill(pid, 15)
            except OSError:
                pass
            deadline = time.time() + 10
            while time.time() < deadline and self._pid_alive(pid):
                await asyncio.sleep(0.25)
            if self._pid_alive(pid):
                try:
                    os.kill(pid, 9)
                except OSError:
                    pass
        self._llama_rpc_proc = None
        self._llama_rpc_pid = None
        self._llama_rpc_host = None
        self._llama_rpc_port = 0
        try:
            self._llama_rpc_state_path.unlink(missing_ok=True)
        except Exception:
            pass
        return {"ok": True, **self.llama_rpc_status()}

    def _llama_adopt(self):
        """One-shot: re-adopt a llama-server we launched before a controller
        restart, using the state file. Validates pid + cmdline."""
        if getattr(self, "_llama_adopt_tried", False):
            return
        self._llama_adopt_tried = True
        try:
            st = json.loads(self._llama_state_path.read_text())
            pid = int(st.get("pid") or 0)
            gguf = str(st.get("gguf_path") or "")
            if pid and self._pid_alive(pid):
                cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace")
                if "llama-server" in cmdline and gguf and gguf in cmdline:
                    self._llama_pid = pid
                    self._llama_model = st.get("model")
                    self._llama_port = int(st.get("port") or 0)
                    self._llama_rpc_nodes = list(st.get("rpc_node_ids") or [])
                    self._llama_rpc_endpoints = list(st.get("rpc_endpoints") or [])
                    log_path = str(st.get("log_path") or "")
                    if log_path:
                        self._llama_current_log = Path(log_path)
                        self._llama_current_log_model = st.get("model")
                    self._llama_ready = True  # adopted servers were ready
                    return
        except Exception:
            pass
        # Nothing to adopt — drop a stale state file.
        try:
            self._llama_state_path.unlink(missing_ok=True)
        except Exception:
            pass

    def _llama_running(self) -> bool:
        self._llama_adopt()
        pid = getattr(self, "_llama_pid", None)
        if not pid or not self._pid_alive(pid):
            return False
        if self._llama_proc is not None and self._llama_proc.returncode is not None:
            return False
        return True

    def _llama_write_state(self, pid: int, model: str, port: int, gguf_path: str):
        try:
            self._llama_state_path.write_text(json.dumps({
                "pid": pid, "model": model, "port": port,
                "gguf_path": gguf_path, "started_at": time.time(),
                "log_path": str(self._llama_current_log or ""),
                "rpc_node_ids": self._llama_rpc_nodes,
                "rpc_endpoints": self._llama_rpc_endpoints,
            }, indent=2))
        except Exception as e:
            print(f"[llama] failed to write state file: {e}")

    async def _unsloth_loaded_model(self) -> str | None:
        """Model currently served by our llama-server, or None. Pure launcher
        state — no HTTP probe, so a busy server can't cause misrouting.
        Returns None during the load window (process spawned but not yet
        answering /health)."""
        if not self._llama_running() or not self._llama_ready:
            return None
        return self._llama_model

    # Log markers → user-facing launch phase. The last marker present in the
    # log tail wins, so phases advance as the server writes more lines.
    _LLAMA_PHASES = [
        ("load_model: loading model", "Loading model weights"),
        ("load_model: loaded metadata", "Reading metadata"),
        ("load_tensors", "Mapping tensors"),
        ("loaded multimodal", "Loading vision encoder"),
        ("load_model: initializing", "Initializing context"),
        ("llama_server: model loaded", "Starting server"),
        ("CUDA error", "CUDA error"),
    ]

    @staticmethod
    def _proc_read_bytes(pid: int | None) -> int | None:
        if not pid:
            return None
        try:
            for line in Path(f"/proc/{pid}/io").read_text().splitlines():
                if line.startswith("read_bytes:"):
                    return int(line.split(":", 1)[1].strip())
        except (OSError, ValueError):
            pass
        return None

    @staticmethod
    def _interface_tx_counter(interfaces: list[str]) -> int | None:
        total = 0
        try:
            for interface in set(interfaces):
                total += int(Path(
                    f"/sys/class/net/{interface}/statistics/tx_bytes"
                ).read_text().strip())
        except (OSError, ValueError):
            return None
        return total

    @classmethod
    def _rpc_tx_counter(cls, endpoints: list[str]) -> tuple[int | None, list[str]]:
        """Return bytes sent on the interfaces routing to RPC workers."""
        interfaces: set[str] = set()
        for endpoint in endpoints:
            host = endpoint.rsplit(":", 1)[0].strip("[]")
            try:
                result = subprocess.run(
                    ["ip", "route", "get", host], capture_output=True,
                    text=True, timeout=2, check=False,
                )
                match = re.search(r"\bdev\s+(\S+)", result.stdout)
                if match:
                    interfaces.add(match.group(1))
            except (OSError, subprocess.SubprocessError):
                continue
        if not interfaces:
            return None, []
        values = sorted(interfaces)
        return cls._interface_tx_counter(values), values

    @staticmethod
    def _format_progress_bytes(value: float) -> str:
        for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
            if value < 1024 or unit == "TiB":
                return f"{value:.1f} {unit}"
            value /= 1024
        return f"{value:.1f} TiB"

    def _llama_transfer_progress(self, info: dict) -> tuple[int | None, str]:
        total = int(info.get("size_bytes") or 0)
        if total <= 0:
            return None, ""

        endpoints = list(info.get("rpc_endpoints") or [])
        if endpoints and info.get("rpc_tx_start") is not None:
            interfaces = list(info.get("rpc_interfaces") or [])
            if not interfaces:
                return None, ""
            current = self._interface_tx_counter(interfaces)
            if current is None:
                return None, ""
            start = int(info["rpc_tx_start"])
            transferred = max(0, current - start)
            tp_size = max(1, int(info.get("tensor_parallel_size") or 1))
            expected = total * max(1, tp_size - 1) / tp_size
            source = f"RPC transfer over {', '.join(interfaces)}"
        else:
            current = self._proc_read_bytes(self._llama_pid)
            start = int(info.get("read_bytes_start") or 0)
            if current is None:
                return None, ""
            transferred = max(0, current - start)
            # read_bytes counts physical storage I/O only. A page-cached GGUF
            # can load normally while this remains zero, so do not publish a
            # determinate 0% that looks stalled.
            if transferred == 0:
                return None, "Loading model from filesystem cache"
            expected = total
            source = "Model data read"

        fraction = min(1.0, transferred / expected) if expected else 0.0
        # Weight transfer is most of startup. Reserve the final 5% for model
        # initialization, context/KV allocation, and the health transition.
        percent = min(95, round(fraction * 95))
        now = time.monotonic()
        previous_bytes = info.get("progress_sample_bytes")
        previous_time = info.get("progress_sample_time")
        rate = None
        if previous_bytes is not None and previous_time is not None and now > previous_time:
            rate = max(0.0, transferred - previous_bytes) / (now - previous_time)
        info["progress_sample_bytes"] = transferred
        info["progress_sample_time"] = now

        detail = (
            f"{source}: {self._format_progress_bytes(transferred)} / "
            f"~{self._format_progress_bytes(expected)}"
        )
        if rate and rate > 0:
            detail += f" · {self._format_progress_bytes(rate)}/s"
        if fraction >= 1.0:
            detail += " · initializing"
        return percent, detail

    def _llama_launch_status(self) -> dict | None:
        """Launch-in-flight status for /api/state: model, variant, start time,
        current phase, last log line, and load percent. Standalone
        llama-server does not expose its internal callback, so progress uses
        native log percentages when present and observable file/RPC transfer
        counters otherwise."""
        info = self._llama_launching
        if not info:
            return None
        phase = "Starting"
        log_line = ""
        tail = ""
        try:
            tail = Path(info["log"]).read_bytes()[-4096:].decode(errors="replace")
            for marker, label in self._LLAMA_PHASES:
                if marker in tail:
                    phase = label
            lines = [l.strip() for l in tail.strip().splitlines() if l.strip()]
            if lines:
                log_line = lines[-1][:160]
        except Exception:
            pass
        percent = None
        progress_detail = ""
        # Prefer a native percentage if a llama.cpp build emits one. Current
        # standalone server builds keep their progress callback internal, so
        # fall back to observable file/RPC transfer counters.
        native = re.findall(
            r"(?:load(?:ing)?(?: model)?|progress)[^%\r\n]{0,80}"
            r"\b(\d{1,3}(?:\.\d+)?)\s*%",
            tail, flags=re.I,
        )
        if native:
            percent = min(99, max(0, round(float(native[-1]))))
            progress_detail = "llama.cpp reported load progress"
        else:
            percent, progress_detail = self._llama_transfer_progress(info)
        if percent is not None and percent >= 95 and phase == "Loading model weights":
            phase = "Initializing model"
        return {
            "model": info["model"],
            "variant": info["variant"],
            "started_at": info["started_at"],
            "phase": phase,
            "log_line": log_line,
            "percent": percent,
            "progress_detail": progress_detail,
            "log_available": bool(info.get("log")),
            "tensor_parallel_size": info.get("tensor_parallel_size", 1),
            "rpc_endpoints": list(info.get("rpc_endpoints") or []),
        }

    def _load_llama_log_index(self) -> dict[str, str]:
        try:
            value = json.loads(self._llama_log_index_path.read_text())
            if isinstance(value, dict):
                return {
                    str(model): str(path) for model, path in value.items()
                    if str(model).strip() and str(path).strip()
                }
        except (OSError, json.JSONDecodeError):
            pass
        return {}

    def _remember_llama_log(self, model_path: str, path: Path) -> None:
        self._llama_model_logs[model_path] = str(path)
        temporary = self._llama_log_index_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self._llama_model_logs, indent=2))
        temporary.replace(self._llama_log_index_path)

    @staticmethod
    def _new_llama_log_path(log_dir: Path) -> Path:
        return log_dir / (
            f"llama-server-{time.time_ns()}-{uuid.uuid4().hex[:8]}.log"
        )

    def get_llama_server_logs(
        self, model_path: str | None = None, since: int = 0,
        limit_bytes: int = 262144,
    ) -> dict:
        """Read one model's managed llama-server log incrementally."""
        requested_model = str(model_path or "").strip() or None
        launching = self._llama_launching or {}
        log_value = launching.get("log") if (
            not requested_model or launching.get("model") == requested_model
        ) else None
        path = Path(log_value) if log_value else None
        if path is None and requested_model:
            indexed = getattr(self, "_llama_model_logs", {}).get(requested_model)
            if indexed:
                path = Path(indexed)
            elif requested_model in {
                self._llama_model,
                getattr(self, "_llama_current_log_model", None),
            }:
                path = self._llama_current_log
        elif path is None:
            path = self._llama_current_log
        if path is None and (
            not requested_model or requested_model == self._llama_model
        ):
            candidates = sorted(
                self._llama_log_dir.glob("llama-server-*.log"),
                key=lambda item: item.stat().st_mtime,
                reverse=True,
            ) if self._llama_log_dir.is_dir() else []
            path = candidates[0] if candidates else None
        if path is None or not path.is_file():
            return {
                "log_id": None, "text": "", "offset": 0,
                "next_offset": 0, "size_bytes": 0, "complete": True,
            }
        try:
            if path.resolve().parent != self._llama_log_dir.resolve():
                raise ValueError("invalid llama-server log path")
        except OSError as exc:
            raise ValueError("invalid llama-server log path") from exc

        size = path.stat().st_size
        offset = max(0, int(since or 0))
        if offset > size:
            offset = 0
        limit = max(4096, min(1024 * 1024, int(limit_bytes or 262144)))
        with path.open("rb") as handle:
            handle.seek(offset)
            content = handle.read(limit)
        next_offset = offset + len(content)
        at_eof = next_offset >= size
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        text = decoder.decode(content, final=at_eof)
        pending, _ = decoder.getstate()
        if pending and not at_eof:
            next_offset -= len(pending)
        return {
            "log_id": path.name,
            "text": text,
            "offset": offset,
            "next_offset": next_offset,
            "size_bytes": size,
            "complete": next_offset >= size,
            "running": self._llama_running(),
            "model": (
                requested_model
                or getattr(self, "_llama_current_log_model", None)
                or self._llama_model
            ),
        }

    async def _stop_running_cluster_deployments(self) -> None:
        """Free both sides of a sharded Docker deployment before llama RPC."""
        active_ids = {
            container.get("deployment_id")
            for container in await self.list_containers()
            if container.get("status") == "running" and container.get("deployment_id")
        }
        for deployment in list(self.deployments):
            if deployment.get("status") == "stopped":
                continue
            if deployment.get("status") == "error" and deployment.get("id") not in active_ids:
                continue
            await self.deployment_action(deployment["id"], "stop")

    async def _start_llama_rpc_cluster(self, tp_size: int) -> list[str]:
        """Start one authenticated RPC worker for every non-local TP rank."""
        nodes = await self.cluster_nodes()
        remotes = [
            node for node in nodes
            if not node.get("local") and node.get("online")
        ]
        needed = tp_size - 1
        if len(remotes) < needed:
            raise RuntimeError(
                f"llama.cpp TP={tp_size} needs {needed} online remote node(s); "
                f"only {len(remotes)} available"
            )
        selected = remotes[:needed]
        port = int(self.settings.get("llama_rpc_port") or 50052)
        started: list[str] = []
        endpoints: list[str] = []
        try:
            for node in selected:
                fabric_ip, _ = self._inferred_fabric(
                    node, node.get("fabric_ip"), node.get("fabric_interface")
                )
                if not fabric_ip:
                    raise RuntimeError(
                        f"could not determine fabric IP for {node.get('name', node['id'])}"
                    )
                await self.node_registry.request(
                    node["id"], "POST", "/api/agent/llama-rpc",
                    json_body={"host": fabric_ip, "port": port}, timeout=45,
                )
                started.append(node["id"])
                endpoints.append(f"{fabric_ip}:{port}")
        except Exception:
            await asyncio.gather(*(
                self.node_registry.request(
                    node_id, "DELETE", "/api/agent/llama-rpc", timeout=15
                )
                for node_id in started
            ), return_exceptions=True)
            raise
        self._llama_rpc_nodes = started
        self._llama_rpc_endpoints = endpoints
        return endpoints

    @staticmethod
    def _llama_tensor_parallel_args(
        argv: list[str], endpoints: list[str], tp_size: int,
        split_mode: str = "tensor",
    ) -> list[str]:
        if tp_size < 2 or len(endpoints) != tp_size - 1:
            raise ValueError("llama.cpp multi-GPU endpoints do not match GPU count")
        if split_mode not in {"tensor", "layer", "row"}:
            raise ValueError(f"unsupported llama.cpp split mode: {split_mode}")
        return [
            *argv,
            "--rpc", ",".join(endpoints),
            "--split-mode", split_mode,
            "--tensor-split", ",".join("1" for _ in range(tp_size)),
            "--fit", "off",
        ]

    @staticmethod
    def _llama_speculative_args(
        argv: list[str], settings: dict, model_entry: dict,
    ) -> tuple[list[str], dict | None]:
        """Add one supported speculative mode to a llama-server command."""
        mtp_enabled = bool(settings.get("mtp_enabled"))
        dspark_enabled = bool(settings.get("dspark_enabled"))
        if mtp_enabled and dspark_enabled:
            raise ValueError("MTP and DSPARK cannot be enabled at the same time")

        draft_tokens = str(settings["mtp_predict_tokens"])
        if mtp_enabled:
            return [
                *argv,
                "--spec-type", "draft-mtp",
                "--spec-draft-n-max", draft_tokens,
            ], None

        if not dspark_enabled:
            return argv, None

        drafts = list(model_entry.get("dspark_drafts") or [])
        if not drafts:
            model_id = model_entry.get("id") or "the model repository"
            raise LookupError(
                f"DSPARK is enabled, but no DSPARK draft GGUF is downloaded for "
                f"'{model_id}'. Download the repository's dspark/*.gguf file "
                "with `hf download`, then refresh and retry."
            )
        draft = min(
            drafts,
            key=lambda d: (
                0 if str(d.get("quant")).upper() == "Q8_0" else
                1 if str(d.get("quant")).upper() == "BF16" else 2,
                str(d.get("filename") or d.get("path") or "").lower(),
            ),
        )
        return [
            *argv,
            "--spec-draft-model", draft["path"],
            "--spec-type", "draft-dspark",
            "--spec-draft-n-max", draft_tokens,
            "--spec-draft-ngl", "99",
        ], draft

    async def _stop_remote_llama_rpc_workers(self) -> None:
        node_ids = list(self._llama_rpc_nodes)
        self._llama_rpc_nodes = []
        self._llama_rpc_endpoints = []
        if not node_ids:
            return
        results = await asyncio.gather(*(
            self.node_registry.request(
                node_id, "DELETE", "/api/agent/llama-rpc", timeout=15
            )
            for node_id in node_ids
        ), return_exceptions=True)
        for node_id, result in zip(node_ids, results):
            if isinstance(result, Exception):
                print(f"[llama-rpc] failed to stop {node_id}: {result}")

    async def load_unsloth_model(self, model_path: str, overrides: dict | None = None) -> dict:
        """Launch llama-server for a cached GGUF model. Settings come from the
        per-model store (defaults if unsaved), optionally overridden for this
        call. Blocks until the server answers /health (large models can take
        minutes), then persists the effective settings."""
        if not model_path:
            raise ValueError("model_path is required")
        settings = self.get_unsloth_settings(model_path)
        if overrides:
            for k, v in overrides.items():
                if k in settings:
                    settings[k] = v
        settings["parallel"] = max(1, min(16, int(settings.get("parallel") or 1)))
        settings["tensor_parallel_size"] = max(
            2, min(16, int(settings.get("tensor_parallel_size") or 2))
        )
        if settings.get("split_mode") not in {"tensor", "layer", "row"}:
            settings["split_mode"] = UNSLOTH_DEFAULT_SETTINGS["split_mode"]
        settings["mtp_predict_tokens"] = max(
            1, min(64, int(settings.get("mtp_predict_tokens") or 1))
        )
        if settings.get("mtp_enabled") and settings.get("dspark_enabled"):
            raise ValueError("MTP and DSPARK cannot be enabled at the same time")
        model_entry, variant = await self._find_gguf(model_path, settings["gguf_variant"])

        bin_path = Path(self.settings["llama_server_bin"]).expanduser()
        if not bin_path.is_file():
            raise RuntimeError(f"llama-server binary not found at {bin_path}")
        host = self.settings.get("llama_server_host") or "127.0.0.1"

        parallel = settings["parallel"]
        tensor_parallel = bool(settings.get("tensor_parallel"))
        tp_size = int(settings.get("tensor_parallel_size") or 2) if tensor_parallel else 1
        max_seq = int(settings.get("max_seq_length") or 0)
        # When multiple parallel slots are active, scale context per slot so each
        # slot gets a full max_seq_length window (llama-server shares -c across slots).
        ctx = max_seq * parallel if parallel > 1 else max_seq
        argv = [
            str(bin_path),
            "-m", variant["path"],
            "--alias", model_path,
            "--host", host,
            "--port", "0",  # placeholder, replaced below
            "-c", str(ctx),
            "--parallel", str(parallel),
            "--flash-attn", "on",
            "--no-context-shift",
            "-ngl", "-1",
            "--metrics",
            "--jinja",
        ]
        if settings.get("kv_unified"):
            argv.append("--kv-unified")
        else:
            argv.append("--no-kv-unified")
        kv = (settings.get("cache_type_kv") or "").strip()
        if kv:
            argv += ["--cache-type-k", kv, "--cache-type-v", kv]
        argv, speculative_draft = self._llama_speculative_args(
            argv, settings, model_entry
        )
        if model_entry.get("mmproj"):
            argv += ["--mmproj", model_entry["mmproj"]]

        async with self._llama_lock:
            # A different model may be running — stop it first. Then free the
            # Release other managed GPU runtimes while retaining the
            # llama-server process this path is about to replace.
            await self._stop_llama_server()
            if tensor_parallel:
                await self._stop_running_cluster_deployments()
            await self.evict_other_backends(protect="unsloth")
            rpc_endpoints: list[str] = []
            if tensor_parallel:
                rpc_endpoints = await self._start_llama_rpc_cluster(tp_size)
                argv = self._llama_tensor_parallel_args(
                    argv, rpc_endpoints, tp_size, settings["split_mode"]
                )

            base_port = int(self.settings.get("llama_server_port") or 8100)
            port = None
            for p in range(base_port, base_port + 10):
                s = socket.socket()
                try:
                    s.bind((host, p))
                    port = p
                    break
                except OSError:
                    continue
                finally:
                    s.close()
            if port is None:
                await self._stop_remote_llama_rpc_workers()
                raise RuntimeError(f"no free port in {base_port}..{base_port + 9}")
            argv[argv.index("--port") + 1] = str(port)

            self._llama_log_dir.mkdir(parents=True, exist_ok=True)
            log_path = self._new_llama_log_path(self._llama_log_dir)
            self._llama_current_log = log_path
            self._llama_current_log_model = model_path
            self._remember_llama_log(model_path, log_path)
            rpc_tx_start, rpc_interfaces = await asyncio.to_thread(
                self._rpc_tx_counter, rpc_endpoints
            )
            log_file = open(log_path, "wb")
            try:
                proc = await asyncio.create_subprocess_exec(
                    *argv, stdout=log_file, stderr=subprocess.STDOUT,
                )
            except Exception:
                log_file.close()
                await self._stop_remote_llama_rpc_workers()
                raise
            log_file.close()
            self._llama_proc = proc
            self._llama_pid = proc.pid
            self._llama_model = model_path
            self._llama_port = port
            self._llama_ready = False
            self._llama_launching = {
                "model": model_path, "variant": variant["quant"],
                "started_at": time.time(), "log": str(log_path),
                "size_bytes": variant.get("size_bytes") or 0,
                "model_files": list(variant.get("files") or [variant["path"]]),
                "speculative_draft": (
                    speculative_draft.get("path") if speculative_draft else None
                ),
                "read_bytes_start": self._proc_read_bytes(proc.pid) or 0,
                "tensor_parallel_size": tp_size,
                "rpc_endpoints": rpc_endpoints,
                "rpc_tx_start": rpc_tx_start,
                "rpc_interfaces": rpc_interfaces,
            }
            self._llama_write_state(proc.pid, model_path, port, variant["path"])
            print(f"[llama] launched pid={proc.pid} port={port} model={model_path} "
                  f"variant={variant['quant']} tp={tp_size} log={log_path}")

            # Wait for readiness; on failure include the log tail.
            try:
                deadline = time.time() + 600
                ready = False
                while time.time() < deadline:
                    if proc.returncode is not None:
                        break
                    try:
                        r = await self.http.get(f"http://{host}:{port}/health", timeout=5)
                        if r.status_code == 200:
                            ready = True
                            break
                    except Exception:
                        pass
                    await asyncio.sleep(2.0)
                if not ready:
                    tail = ""
                    try:
                        tail = log_path.read_text(errors="replace")[-800:]
                    except Exception:
                        pass
                    await self._stop_llama_server()
                    if proc.returncode is not None:
                        if "LLAMA_SPLIT_MODE_TENSOR not implemented" in tail:
                            raise RuntimeError(
                                "llama-server tensor split does not support this model "
                                "architecture. In Load settings, select Layer split and "
                                f"retry. Log tail:\n{tail}"
                            )
                        raise RuntimeError(
                            f"llama-server exited during load (code {proc.returncode}). "
                            f"Log tail:\n{tail}"
                        )
                    raise RuntimeError(f"llama-server not ready after 600s. Log tail:\n{tail}")
                self._llama_ready = True
            except asyncio.CancelledError:
                # Cancel load button: kill the half-started server so it
                # doesn't come up orphaned, then let the task unwind.
                await self._stop_llama_server()
                raise
            finally:
                self._llama_launching = None

        # Persist the effective settings so the next launch defaults to what
        # was last launched, even when it was a one-off override.
        self.unsloth_settings[model_path] = settings
        self._save_unsloth_settings()
        return {"ok": True, "model_path": model_path, "port": port}

    async def _stop_llama_server(self):
        """SIGTERM the tracked llama-server (10s grace, then SIGKILL)."""
        self._llama_adopt()
        pid = getattr(self, "_llama_pid", None)
        if pid and self._pid_alive(pid):
            try:
                os.kill(pid, 15)
            except OSError:
                pass
            deadline = time.time() + 10
            while time.time() < deadline and self._pid_alive(pid):
                await asyncio.sleep(0.5)
            if self._pid_alive(pid):
                try:
                    os.kill(pid, 9)
                except OSError:
                    pass
        self._llama_proc = None
        self._llama_pid = None
        self._llama_model = None
        self._llama_port = 0
        self._llama_ready = False
        await self._stop_remote_llama_rpc_workers()
        try:
            self._llama_state_path.unlink(missing_ok=True)
        except Exception:
            pass

    async def unload_unsloth_model(self, model_path: str) -> dict:
        if not model_path:
            raise ValueError("model_path is required")
        if self._llama_running() and self._llama_model != model_path:
            raise RuntimeError(
                f"'{model_path}' is not running (current: '{self._llama_model}')"
            )
        await self._stop_llama_server()
        return {"ok": True, "model_path": model_path}

    async def cancel_unsloth_load(self) -> dict:
        """Cancel an in-flight llama-server launch (UI Cancel-load button).
        The task's CancelledError path stops the half-started server."""
        t = self._llama_load_task
        if t is None or t.done():
            return {"ok": False, "detail": "no load in progress"}
        t.cancel()
        return {"ok": True}

    # ---------- /v1 proxy ----------
    async def proxy_models(self) -> dict:
        """Return OpenAI-compatible models from supported local runtimes."""
        data = await self.get_state()
        models = []
        for c in data.get("containers", []):
            model_ids = list(c.get("served_models") or [c.get("model")])
            for model_id in dict.fromkeys(model_ids):
                if not model_id or any(m["id"] == model_id for m in models):
                    continue
                models.append({
                    "id": model_id,
                    "object": "model",
                    "owned_by": "vllm",
                    "type": "local",
                })
        # llama-server runs one model at a time; surface the loaded one so
        # OpenAI clients can target it. The controller forwards requests to
        # the llama-server process it launched on localhost.
        loaded = await self._unsloth_loaded_model()
        if loaded:
            models.append({
                "id": loaded,
                "object": "model",
                "owned_by": "llama.cpp",
                "type": "local",
            })
        # sparkrun containers run vLLM in host-network mode; surface their
        # served models so the Chat/A/B dropdowns can target them.
        for model_id in (data.get("sparkrun_targets") or {}):
            if not any(m["id"] == model_id for m in models):
                models.append({
                    "id": model_id,
                    "object": "model",
                    "owned_by": "sparkrun",
                    "type": "local",
                })
        return {"object": "list", "data": models}

    async def proxy_chat_completions(self, body: dict, cancel: asyncio.Event | None = None):
        """Route /v1/chat/completions to a supported managed backend."""
        model = body.get("model", "")
        stream = body.get("stream", False)

        loaded_unsloth = await self._unsloth_loaded_model()
        if loaded_unsloth and loaded_unsloth == model:
            return await self._unsloth_chat(model, body, stream, cancel)

        return await self._vllm_chat(model, body, stream, cancel)

    async def proxy_completions(self, body: dict, cancel: asyncio.Event | None = None):
        """Route /v1/completions to a supported managed backend."""
        model = body.get("model", "")
        stream = body.get("stream", False)

        loaded_unsloth = await self._unsloth_loaded_model()
        if loaded_unsloth and loaded_unsloth == model:
            return await self._unsloth_completions(model, body, stream, cancel)

        return await self._vllm_completions(model, body, stream, cancel)

    async def _vllm_chat(self, model: str, body: dict, stream: bool,
                         cancel: asyncio.Event | None = None, *,
                         container_name: str | None = None,
                         deployment_id: str | None = None):
        """Route /v1/chat/completions to the appropriate vLLM container."""
        container = await self._resolve_vllm_target(
            model, container_name=container_name, deployment_id=deployment_id,
        )
        port = container["port"]
        key = container.get("stats_key") or model
        body = {**body, "model": self._upstream_model_id(container, model)}
        url = f"http://localhost:{port}/v1/chat/completions"
        if stream:
            return self._vllm_stream(
                url, body, key, cancel, container, requested_model=model,
                container_name=container_name, deployment_id=deployment_id,
            )
        else:
            admission = None
            container, admission = await self._admit_vllm_target(
                model, cancel, container, container_name=container_name,
                deployment_id=deployment_id,
            )
            key = container.get("stats_key") or model
            body = {**body, "model": self._upstream_model_id(container, model)}
            url = f"http://localhost:{container['port']}/v1/chat/completions"
            rid = self._track_start(
                key, deployment_id=container.get("deployment_id")
            )
            try:
                r = await self._await_or_cancel(
                    self.http.post(url, json=body, timeout=600), cancel
                )
                r.raise_for_status()
                data = r.json()
                self._record_usage(key, data.get("usage"))
                return data
            finally:
                self._track_end(rid)
                self._release_inference_slot(admission)

    async def _vllm_completions(self, model: str, body: dict, stream: bool,
                                cancel: asyncio.Event | None = None, *,
                                container_name: str | None = None,
                                deployment_id: str | None = None):
        """Route /v1/completions to the appropriate vLLM container."""
        container = await self._resolve_vllm_target(
            model, container_name=container_name, deployment_id=deployment_id,
        )
        port = container["port"]
        key = container.get("stats_key") or model
        body = {**body, "model": self._upstream_model_id(container, model)}
        url = f"http://localhost:{port}/v1/completions"
        if stream:
            return self._vllm_stream(
                url, body, key, cancel, container, requested_model=model,
                container_name=container_name, deployment_id=deployment_id,
            )
        else:
            admission = None
            container, admission = await self._admit_vllm_target(
                model, cancel, container, container_name=container_name,
                deployment_id=deployment_id,
            )
            key = container.get("stats_key") or model
            body = {**body, "model": self._upstream_model_id(container, model)}
            url = f"http://localhost:{container['port']}/v1/completions"
            rid = self._track_start(
                key, deployment_id=container.get("deployment_id")
            )
            try:
                r = await self._await_or_cancel(
                    self.http.post(url, json=body, timeout=600), cancel
                )
                r.raise_for_status()
                data = r.json()
                self._record_usage(key, data.get("usage"))
                return data
            finally:
                self._track_end(rid)
                self._release_inference_slot(admission)

    def _llama_base_url(self) -> str:
        host = self.settings.get("llama_server_host") or "127.0.0.1"
        return f"http://{host}:{self._llama_port}"

    async def _unsloth_chat(self, model: str, body: dict, stream: bool,
                            cancel: asyncio.Event | None = None):
        """Route /v1/chat/completions to the running llama-server."""
        url = f"{self._llama_base_url()}/v1/chat/completions"
        key = self._stats_key(model, self._unsloth_variant(model))
        if stream:
            return self._unsloth_stream(url, body, key, cancel)
        else:
            rid = self._track_start(key)
            try:
                r = await self._await_or_cancel(
                    self.http.post(url, json=body, timeout=600), cancel
                )
                r.raise_for_status()
                data = r.json()
                self._record_usage(key, data.get("usage"))
                return data
            finally:
                self._track_end(rid)

    async def _unsloth_completions(self, model: str, body: dict, stream: bool,
                                   cancel: asyncio.Event | None = None):
        """Route /v1/completions to the running llama-server."""
        url = f"{self._llama_base_url()}/v1/completions"
        key = self._stats_key(model, self._unsloth_variant(model))
        if stream:
            return self._unsloth_stream(url, body, key, cancel)
        else:
            rid = self._track_start(key)
            try:
                r = await self._await_or_cancel(
                    self.http.post(url, json=body, timeout=600), cancel
                )
                r.raise_for_status()
                data = r.json()
                self._record_usage(key, data.get("usage"))
                return data
            finally:
                self._track_end(rid)

    async def _unsloth_stream(self, url: str, body: dict, key: str,
                              cancel: asyncio.Event | None = None):
        """Stream an OpenAI-compatible response from llama-server, passing
        chunks through. `key` is the token-stats key (model id plus variant
        tag). Decode speed is measured from the first to the last output
        chunk, excluding prefill. We ask for stream_options.include_usage so
        token counts can be recorded; if the upstream rejects that option
        (400), the request is retried without it so streaming never breaks."""
        body = {
            **body,
            "stream_options": {
                **(body.get("stream_options") or {}),
                "include_usage": True,
            },
        }
        rid = self._track_start(key, streaming=True)
        retried = False
        try:
            while True:
                attempt_started_at = time.monotonic()
                try:
                    async with self.http.stream(
                        "POST", url, json=body, timeout=None,
                    ) as r:
                        if r.status_code != 200:
                            detail = (await r.aread()).decode("utf-8", errors="replace")
                            if r.status_code == 400 and not retried:
                                # Upstream may not know stream_options — drop it and retry.
                                retried = True
                                body = {k: v for k, v in body.items() if k != "stream_options"}
                                continue
                            yield f"data: {json.dumps({'error': {'message': f'HTTP {r.status_code}: {detail[:300]}', 'type': 'upstream_error', 'code': r.status_code}})}\n\n"
                            yield "data: [DONE]\n\n"
                            return
                        first_out_ts = None
                        last_out_ts = None
                        async for line in self._aiter_lines_cancellable(r, cancel):
                            if not line:
                                continue
                            thinking_tokens, output_tokens = (
                                self._sse_chunk_token_counts(line)
                            )
                            if thinking_tokens or output_tokens:
                                now = time.monotonic()
                                self._track_output(
                                    rid, now, "thinking", thinking_tokens,
                                )
                                self._track_output(
                                    rid, now, "output", output_tokens,
                                )
                                if first_out_ts is None:
                                    first_out_ts = now
                                last_out_ts = now
                            usage = self._usage_from_sse_line(line)
                            if usage:
                                gen_time = (
                                    last_out_ts - first_out_ts
                                    if first_out_ts is not None
                                    else None
                                )
                                pp_time = (
                                    first_out_ts - attempt_started_at
                                    if first_out_ts is not None
                                    else None
                                )
                                if pp_time:
                                    prompt_tokens, cached_tokens = (
                                        self._usage_prompt_counts(usage)
                                    )
                                    self._track_prompt_processing(
                                        rid,
                                        prompt_tokens - cached_tokens,
                                        pp_time,
                                    )
                                self._record_usage(
                                    key, usage, gen_time, pp_time
                                )
                            yield f"{line}\n\n"
                    return
                except Exception as e:
                    yield f"data: {json.dumps({'error': {'message': str(e), 'type': 'upstream_error'}})}\n\n"
                    yield "data: [DONE]\n\n"
                    return
        finally:
            self._track_end(rid)

    async def _sparkrun_targets(self) -> dict[str, dict]:
        """Discover models served by running sparkrun containers.
        sparkrun containers run in host network mode, so their internal
        vLLM port is reachable on localhost. Runtime load settings are kept so
        controller admission honors SparkRun's --max-num-seqs as well."""
        out: dict[str, dict] = {}
        active_run_limit = next((
            run.get("max_concurrency")
            for run in sorted(
                getattr(self, "spark_runs", {}).values(),
                key=lambda item: item.get("started_at") or 0,
                reverse=True,
            )
            if run.get("status") == "running" and run.get("max_concurrency")
        ), None)
        try:
            for c in self.client.containers.list(all=False):
                if not c.name.startswith("sparkrun_"):
                    continue
                summary = self._container_summary(c) or {}
                load_settings = dict(summary.get("load_settings") or {})
                if not load_settings.get("max_concurrency") and active_run_limit:
                    load_settings["max_concurrency"] = active_run_limit
                # Host-network sparkrun containers expose vLLM directly on localhost.
                # Default recipe port is 8000; inspect the recipe command to find it.
                port = 8000
                try:
                    r = await self.http.get(
                        f"http://localhost:{port}/v1/models", timeout=2
                    )
                    if r.status_code != 200:
                        continue
                    for m in (r.json() or {}).get("data", []):
                        mid = m.get("id") if isinstance(m, dict) else None
                        if mid:
                            out[mid] = {
                                **summary,
                                "name": summary.get("name") or c.name,
                                "model": summary.get("model") or mid,
                                "served_models": [mid],
                                "stats_key": summary.get("stats_key") or mid,
                                "port": port,
                                "load_settings": load_settings,
                            }
                except Exception:
                    pass
        except Exception:
            pass
        return out

    async def _resolve_vllm_target(
        self, model: str, *, container_name: str | None = None,
        deployment_id: str | None = None,
    ) -> dict:
        """Find or start a running vLLM container for the given model."""
        # A capacity-triggered replacement briefly has no container for the
        # intended deployment. Do not let ensure_loaded wake an older stopped
        # copy of the same model during that gap; wait until the coherent
        # replacement generation has at least been created.
        while model in getattr(self, "_capacity_redeploying_models", set()):
            await asyncio.sleep(0.25)
        containers = await self.list_containers()
        # Look for a running container with this model
        for c in containers:
            if container_name and c.get("name") != container_name:
                continue
            if (
                deployment_id and c.get("deployment_id")
                and c.get("deployment_id") != deployment_id
            ):
                continue
            if model in self._container_model_ids(c) and c["status"] == "running":
                ready = await self._check_ready(c)
                if ready:
                    self._mark_active(c["name"])
                    return c
        # Try ensure_loaded to start/swap the container
        try:
            return await self.ensure_loaded(
                model, container_name=container_name,
                deployment_id=deployment_id,
            )
        except LookupError:
            raise LookupError(f"No managed container found for model '{model}'")
        except TimeoutError:
            raise TimeoutError(f"Timeout waiting for model '{model}' to become ready")

    async def _vllm_stream(self, url: str, body: dict, key: str,
                           cancel: asyncio.Event | None = None,
                           container: dict | None = None,
                           requested_model: str | None = None, *,
                           container_name: str | None = None,
                           deployment_id: str | None = None):
        """Stream vLLM SSE response, passing through chunks as-is.
        Forces continuous usage stats so prompt counts are available as soon
        as generation begins, allowing the live prefill rate to be populated
        before the terminal usage chunk. `key` is the token-stats key (model
        id plus variant tag). Decode speed is measured from the first to the
        last output chunk, excluding prefill."""
        body = {
            **body,
            # vLLM otherwise emits token_ids=null. Exact ids are required for
            # correct rates when speculative decoding batches several tokens
            # into one SSE event.
            "return_token_ids": True,
            "stream_options": {
                **(body.get("stream_options") or {}),
                "include_usage": True,
                "continuous_usage_stats": True,
            },
        }
        admission = None
        rid = None
        endpoint = None
        nudge_event = asyncio.Event()
        retried_without_token_ids = False
        retried_without_continuous_usage = False
        try:
            if container is not None:
                requested_model = requested_model or body.get("model") or key
                endpoint = (
                    "/v1/chat/completions"
                    if url.endswith("/v1/chat/completions")
                    else "/v1/completions"
                )
                container, admission = await self._admit_vllm_target(
                    requested_model, cancel, container,
                    container_name=container_name, deployment_id=deployment_id,
                )
                key = container.get("stats_key") or requested_model
                body["model"] = self._upstream_model_id(
                    container, requested_model
                )
                url = f"http://localhost:{container['port']}{endpoint}"
            rid = self._track_start(
                key,
                streaming=True,
                admission_target=admission,
                nudge_event=nudge_event,
                deployment_id=(container or {}).get("deployment_id"),
            )
            while True:
                attempt_started_at = time.monotonic()
                stream_context = self.http.stream(
                    "POST", url, json=body, timeout=None,
                )
                entered = False
                nudged = False
                try:
                    r = await self._await_or_cancel(
                        stream_context.__aenter__(), cancel, nudge_event,
                    )
                    entered = True
                    if r.status_code != 200:
                        detail = (await r.aread()).decode("utf-8", errors="replace")
                        if (
                            r.status_code == 400
                            and body.get("return_token_ids")
                            and not retried_without_token_ids
                        ):
                            # Older OpenAI-compatible servers may reject this
                            # vLLM extension. Preserve streaming with the
                            # one-event fallback rather than failing the chat.
                            retried_without_token_ids = True
                            body = {
                                k: v for k, v in body.items()
                                if k != "return_token_ids"
                            }
                            continue
                        stream_options = body.get("stream_options") or {}
                        if (
                            r.status_code == 400
                            and stream_options.get("continuous_usage_stats")
                            and not retried_without_continuous_usage
                        ):
                            # Keep compatibility with older vLLM releases:
                            # live PP speed is unavailable there, but the
                            # terminal include_usage chunk still records totals.
                            retried_without_continuous_usage = True
                            body = {
                                **body,
                                "stream_options": {
                                    k: v for k, v in stream_options.items()
                                    if k != "continuous_usage_stats"
                                },
                            }
                            continue
                        yield f"data: {json.dumps({'error': {'message': f'HTTP {r.status_code}: {detail[:300]}', 'type': 'upstream_error', 'code': r.status_code}})}\n\n"
                        yield "data: [DONE]\n\n"
                        return
                    first_out_ts = None
                    last_out_ts = None
                    latest_usage = None
                    async for line in self._aiter_lines_cancellable(
                        r, cancel, nudge_event,
                    ):
                        if not line:
                            continue
                        thinking_tokens, output_tokens = (
                            self._sse_chunk_token_counts(line)
                        )
                        if thinking_tokens or output_tokens:
                            now = time.monotonic()
                            self._track_output(
                                rid, now, "thinking", thinking_tokens,
                            )
                            self._track_output(
                                rid, now, "output", output_tokens,
                            )
                            if first_out_ts is None:
                                first_out_ts = now
                            last_out_ts = now
                        usage = self._usage_from_sse_line(line)
                        if usage:
                            # Continuous usage is cumulative. Keep only the
                            # latest snapshot for lifetime accounting so each
                            # request is counted exactly once.
                            latest_usage = usage
                            pp_time = (
                                first_out_ts - attempt_started_at
                                if first_out_ts is not None
                                else None
                            )
                            if (
                                pp_time
                                and self._usage_has_cached_prompt_tokens(usage)
                            ):
                                prompt_tokens, cached_tokens = (
                                    self._usage_prompt_counts(usage)
                                )
                                self._track_prompt_processing(
                                    rid,
                                    prompt_tokens - cached_tokens,
                                    pp_time,
                                )
                        rec = self._active_reqs.get(rid)
                        if rec is not None:
                            rec["forwarded_chunks"] = (
                                rec.get("forwarded_chunks", 0) + 1
                            )
                        yield f"{line}\n\n"
                    if latest_usage:
                        gen_time = (
                            last_out_ts - first_out_ts
                            if first_out_ts is not None
                            else None
                        )
                        pp_time = (
                            first_out_ts - attempt_started_at
                            if first_out_ts is not None
                            else None
                        )
                        # Do not contaminate the completed-request PP average
                        # when an older backend omits authoritative cache-hit
                        # detail from its terminal usage payload.
                        measured_pp_time = (
                            pp_time
                            if self._usage_has_cached_prompt_tokens(latest_usage)
                            else None
                        )
                        self._record_usage(
                            key, latest_usage, gen_time, measured_pp_time
                        )
                    return
                except StreamNudge:
                    nudged = True
                finally:
                    if entered:
                        await stream_context.__aexit__(None, None, None)
                if nudged:
                    rec = self._active_reqs.get(rid)
                    if rec is None:
                        return
                    if (
                        rec.get("total_tokens", 0) > 0
                        or rec.get("forwarded_chunks", 0) > 0
                    ):
                        yield f"data: {json.dumps({'error': {'message': 'Stream stalled after output began and cannot be safely replayed; retry the request', 'type': 'nudger_retry', 'code': 'stream_nudge_retry', 'retryable': True}})}\n\n"
                        yield "data: [DONE]\n\n"
                        return
                    # This request has sent nothing downstream, so closing its
                    # upstream response and replaying the exact body is
                    # transparent to the client. Release before re-acquiring:
                    # the nudger has reduced this target to one active slot.
                    rec["paused"] = True
                    rec["admission_target"] = None
                    self._release_inference_slot(admission)
                    admission = None
                    nudge_event.clear()
                    if container is None or requested_model is None:
                        return
                    container, admission = await self._admit_vllm_target(
                        requested_model, cancel, container,
                        container_name=container_name,
                        deployment_id=deployment_id,
                    )
                    key = container.get("stats_key") or requested_model
                    body["model"] = self._upstream_model_id(
                        container, requested_model
                    )
                    url = f"http://localhost:{container['port']}{endpoint}"
                    rec["key"] = key
                    rec["admission_target"] = admission
                    rec["paused"] = False
        except ClientAbort:
            return
        except Exception as e:
            yield f"data: {json.dumps({'error': {'message': str(e), 'type': 'upstream_error'}})}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            if rid is not None:
                self._track_end(rid)
            self._release_inference_slot(admission)

    # ---------- system telemetry ----------
    def _read_disk(self) -> dict:
        try:
            usage = shutil.disk_usage("/")
            total = usage.total
            free = usage.free
            used = total - free
            return {
                "total": total,
                "used": used,
                "free": free,
                "pct": round(100.0 * used / total, 1) if total else 0.0,
            }
        except Exception:
            return {}

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

    def _read_net(self) -> dict:
        """Return aggregate network throughput (bytes/s) across physical/tunnel interfaces.

        Docker bridges (docker0, br-*) and their veth peers are excluded because
        the same traffic appears on both, which would double-count container I/O.
        """
        def _is_virtual(iface: str) -> bool:
            if iface == "lo":
                return True
            if iface.startswith(("veth", "virbr", "docker", "br-")):
                return True
            # Linux creates a /sys/class/net/<iface>/bridge dir for bridge devices.
            return (Path("/sys/class/net") / iface / "bridge").exists()

        try:
            info: dict[str, tuple[int, int]] = {}
            with open("/proc/net/dev") as f:
                for line in f:
                    if "|" in line:
                        continue
                    parts = line.split()
                    if len(parts) < 9:
                        continue
                    iface = parts[0].rstrip(":")
                    if _is_virtual(iface):
                        continue
                    rx = int(parts[1])
                    tx = int(parts[9])
                    info[iface] = (rx, tx)

            now = time.time()
            prev = self._net_prev
            self._net_prev = (now, info)
            if prev is None:
                return {"download_bps": 0.0, "upload_bps": 0.0, "interfaces": {}}

            prev_ts, prev_info = prev
            elapsed = now - prev_ts
            if elapsed <= 0:
                return {"download_bps": 0.0, "upload_bps": 0.0, "interfaces": {}}

            total_rx_diff = 0
            total_tx_diff = 0
            ifaces: dict[str, dict] = {}
            for iface, (rx, tx) in info.items():
                p = prev_info.get(iface)
                if not p:
                    continue
                rx_diff = max(0, rx - p[0])
                tx_diff = max(0, tx - p[1])
                total_rx_diff += rx_diff
                total_tx_diff += tx_diff
                ifaces[iface] = {
                    "download_bps": round(rx_diff / elapsed, 1),
                    "upload_bps": round(tx_diff / elapsed, 1),
                }

            return {
                "download_bps": round(total_rx_diff / elapsed, 1),
                "upload_bps": round(total_tx_diff / elapsed, 1),
                "interfaces": ifaces,
            }
        except Exception:
            return {"download_bps": 0.0, "upload_bps": 0.0, "interfaces": {}}

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

    def _read_cpu_clock_mhz(self) -> dict:
        """Current and max CPU frequency across online cores, in MHz."""
        cur_freqs = []
        max_freqs = []
        try:
            for p in sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*")):
                try:
                    cur_path = p / "cpufreq" / "scaling_cur_freq"
                    max_path = p / "cpufreq" / "scaling_max_freq"
                    if cur_path.exists():
                        cur_freqs.append(int(cur_path.read_text().strip()) / 1000.0)
                    if max_path.exists():
                        max_freqs.append(int(max_path.read_text().strip()) / 1000.0)
                except Exception:
                    continue
        except Exception:
            pass
        # Fallback to /proc/cpuinfo if cpufreq is unavailable
        if not cur_freqs:
            try:
                for line in Path("/proc/cpuinfo").read_text().splitlines():
                    if line.startswith("cpu MHz"):
                        try:
                            cur_freqs.append(float(line.split(":", 1)[1].strip()))
                        except Exception:
                            continue
            except Exception:
                pass
        # Base (non-boost) frequency from the hardware; used as the reference
        # for the clock delta in the UI. Read from the first core's
        # base_frequency — it's uniform across cores.
        base = None
        try:
            bf = Path("/sys/devices/system/cpu/cpu0/cpufreq/base_frequency")
            if bf.exists():
                base = round(int(bf.read_text().strip()) / 1000.0, 0)
        except Exception:
            pass
        return {
            "current": round(sum(cur_freqs) / len(cur_freqs), 0) if cur_freqs else None,
            "max": round(sum(max_freqs) / len(max_freqs), 0) if max_freqs else None,
            "base": base,
        }

    def _read_gpu(self) -> list[dict]:
        try:
            r = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit,clocks.gr,clocks.max.gr,clocks.default_applications.gr",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True, text=True, timeout=4,
            )
            out = []
            for line in r.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 10:
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
                    "clock_mhz": _num(parts[8]),
                    "clock_max_mhz": _num(parts[9]),
                    # Default app-graphics clock = factory base. Used as the
                    # reference for the clock delta. "[N/A]" on some SoCs
                    # (GB10) → None, UI falls back to raw display.
                    "clock_base_mhz": _num(parts[10]) if len(parts) > 10 else None,
                })
            return out
        except Exception as e:
            return [{"error": str(e)}]

    async def _gpu_total_gb(self) -> float | None:
        """Total GPU memory of the first GPU in GB, or None if unavailable."""
        try:
            stats = await self.get_stats()
            gpu = (stats.get("gpus") or [{}])[0]
            total_mib = gpu.get("mem_total_mib")
            if total_mib:
                return total_mib / 1024.0
        except Exception:
            pass
        return None

    # ---------- VRAM-aware multi-model ----------
    def _read_gpu_memory_gb(self) -> tuple[float, float]:
        """Return (total_gb, used_gb) from nvidia-smi for the first GPU."""
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total,memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3,
            )
            parts = [p.strip() for p in r.stdout.strip().split(",")]
            if len(parts) >= 2:
                total = float(parts[0]) / 1024.0
                used = float(parts[1]) / 1024.0
                return (total, used)
        except Exception:
            pass
        return (0.0, 0.0)

    @staticmethod
    def _estimate_params_and_quant(
        model: str, cmd: list[str] | None = None
    ) -> tuple[float, float]:
        """Estimate (params_in_billions, bytes_per_param) from model name + cmd.

        Returns (0, 0) when the model name doesn't contain a recognizable
        size hint.  bytes_per_param defaults to 2 (bf16) unless the command
        line declares a quantization or lower dtype.
        """
        # --- bytes/param from command flags ---
        bpp = 2.0  # bf16 default
        if cmd:
            for i, tok in enumerate(cmd):
                if tok == "--quantization" and i + 1 < len(cmd):
                    q = cmd[i + 1].lower()
                    if any(k in q for k in ("awq", "gptq", "sqqp")):
                        bpp = 0.5  # 4-bit
                    elif any(k in q for k in ("fp8", "e4m3")):
                        bpp = 1.0
                    break
                if tok == "--dtype" and i + 1 < len(cmd):
                    d = cmd[i + 1].lower()
                    if "int8" in d:
                        bpp = 1.0
                    elif "float8" in d or "fp8" in d:
                        bpp = 1.0
                    elif "bfloat16" in d or "float16" in d or d == "auto":
                        bpp = 2.0
                    break

        # --- param count from model name ---
        m = re.search(r"(\d+(?:\.\d+)?)\s*b\b", model, re.IGNORECASE)
        params = float(m.group(1)) if m else 0.0

        return (params, bpp)

    def _try_fit_new_model(
        self,
        need_gb: float,
        *,
        protect_name: str | None = None,
    ) -> None:
        """Stop the least-recently-active managed containers until *need_gb*
        fits in the GPU's free memory (total − used − 10 GB buffer).

        Containers whose name matches ``protect_name`` are never evicted.
        Best-effort — failures are logged but never raised.
        """
        total, used = self._read_gpu_memory_gb()
        if total <= 0:
            return  # can't read GPU info; let the launch proceed
        free = total - used - GPU_VRAM_BUFFER_GB
        if free >= need_gb:
            return  # fits already

        # Need to evict — find managed running containers, sorted by LRU.
        try:
            containers = [
                container for container in self.client.containers.list()
                if _label_value(container.labels or {}, CONTROLLER_LABEL) == "1"
            ]
            running = []
            for c in containers:
                if c.status == "running" and c.name != protect_name:
                    running.append(c)
            running.sort(
                key=lambda c: self._activity.get(c.name, {}).get(
                    "last_active", 0
                )
            )

            for c in running:
                if free >= need_gb:
                    break
                try:
                    print(f"[fit] evicting {c.name} to make room "
                          f"(need={need_gb:.1f} GB, free={free:.1f} GB)")
                    c.update(restart_policy={"Name": "no"})
                    c.stop(timeout=10)
                    self._activity.pop(c.name, None)
                    # Re-check free memory after stop
                    _, used = self._read_gpu_memory_gb()
                    free = total - used - GPU_VRAM_BUFFER_GB
                except Exception as e:
                    print(f"[fit] failed to stop {c.name}: {e}")
        except Exception as e:
            print(f"[fit] container listing failed: {e}")

    async def get_stats(self) -> dict:
        # Light cache so several browser tabs don't pile on nvidia-smi.
        # Frontend polls at 1 Hz; serve fresh data at slightly under that.
        # Live request stats skip the cache so the Tokens widget really
        # updates once per second while streams are active.
        now = time.time()
        if now - self._stats_ts < 0.8 and self._stats_cache:
            self._record_temperature_sample(self._stats_cache, now)
            return {**self._stats_cache, "active_requests": self.active_requests()}

        def _gather():
            cpu_clock = self._read_cpu_clock_mhz()
            return {
                "cpu_pct": self._read_cpu_pct(),
                "cpu_temp_c": self._read_cpu_temp(),
                "cpu_clock_mhz": cpu_clock.get("current"),
                "cpu_clock_max_mhz": cpu_clock.get("max"),
                "cpu_clock_base_mhz": cpu_clock.get("base"),
                "mem": self._read_mem(),
                "gpus": self._read_gpu(),
                "net": self._read_net(),
                "mem_bw": self._read_mem_bw(),
                "online_users": self._read_online_users(),
                "fan": self._read_fan_state(),
                "ts": time.time(),
            }
        stats = await asyncio.to_thread(_gather)
        self._stats_cache = stats
        self._stats_ts = now
        self._record_temperature_sample(stats, stats.get("ts", now))
        return {**stats, "active_requests": self.active_requests()}

    def _record_temperature_sample(
        self,
        stats: dict,
        observed_at: float | None = None,
        history: deque[dict[str, float | None]] | None = None,
    ) -> bool:
        """Append at most one CPU/GPU temperature point every 30 seconds."""
        now = float(observed_at if observed_at is not None else time.time())
        history = history if history is not None else self._temperature_history
        if history and now - float(history[-1]["ts"]) < TEMPERATURE_HISTORY_INTERVAL_SECONDS:
            return False

        def finite_temperature(value: Any) -> float | None:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return None
            value = float(value)
            return round(value, 1) if math.isfinite(value) else None

        gpu = (stats.get("gpus") or [{}])[0]
        history.append({
            "ts": now,
            "cpu_temp_c": finite_temperature(stats.get("cpu_temp_c")),
            "gpu_temp_c": finite_temperature(gpu.get("temp")),
        })
        cutoff = now - TEMPERATURE_HISTORY_WINDOW_SECONDS
        while history and float(history[0]["ts"]) < cutoff:
            history.popleft()
        return True

    def _record_remote_temperature_sample(self, node: dict) -> bool:
        node_id = str(node.get("id") or "")
        stats = node.get("stats") or {}
        if not node_id or not stats:
            return False
        history = self._remote_temperature_histories.setdefault(
            node_id,
            deque(maxlen=TEMPERATURE_HISTORY_MAX_SAMPLES),
        )
        return self._record_temperature_sample(
            stats,
            stats.get("ts"),
            history,
        )

    def temperature_history(
        self, history: deque[dict[str, float | None]] | None = None,
    ) -> dict:
        history = history if history is not None else self._temperature_history
        return {
            "sample_interval_seconds": TEMPERATURE_HISTORY_INTERVAL_SECONDS,
            "window_seconds": TEMPERATURE_HISTORY_WINDOW_SECONDS,
            "samples": list(history),
        }

    async def temperature_history_for_node(self, node_id: str) -> dict:
        if not node_id or node_id == LOCAL_NODE_ID:
            return {"node_id": LOCAL_NODE_ID, **self.temperature_history()}
        history = self._remote_temperature_histories.get(node_id)
        if history:
            return {"node_id": node_id, **self.temperature_history(history)}
        # A newly selected updated agent may already have samples even before
        # this coordinator has observed its first 30-second point. Older
        # agents do not expose this endpoint, so return an empty graph rather
        # than failing the whole dashboard.
        try:
            result = await self.node_registry.request(
                node_id, "GET", "/api/agent/temperature-history", timeout=5,
            )
        except RuntimeError:
            result = self.temperature_history(deque())
        return {"node_id": node_id, **(result or {})}

    def _load_temperature_runs(self) -> dict[str, dict]:
        try:
            raw = json.loads(self.temperature_runs_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return {}
        entries = raw.get("runs") if isinstance(raw, dict) else raw
        if not isinstance(entries, list):
            return {}
        runs: dict[str, dict] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            run_id = str(entry.get("id") or "").strip()
            if not run_id:
                continue
            run = copy.deepcopy(entry)
            run["id"] = run_id
            run["name"] = str(run.get("name") or f"Temperature run {run_id}")[:120]
            run["samples"] = [
                sample for sample in (run.get("samples") or [])
                if isinstance(sample, dict)
            ]
            runs[run_id] = run
        return runs

    def _save_temperature_runs(self) -> None:
        path = self.temperature_runs_path
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({
            "version": 1,
            "runs": list(self.temperature_runs.values()),
        }, indent=2), encoding="utf-8")
        temporary.replace(path)
        self._temperature_runs_last_saved_at = time.time()

    @staticmethod
    def _public_temperature_run(run: dict, *, include_samples: bool = False) -> dict:
        public = {
            key: copy.deepcopy(value)
            for key, value in run.items()
            if key != "samples"
        }
        samples = run.get("samples") or []
        public["sample_count"] = len(samples)
        if samples:
            public["duration_seconds"] = round(
                max(0.0, float(samples[-1].get("elapsed_seconds") or 0.0)), 3,
            )
        else:
            public["duration_seconds"] = 0.0
        if include_samples:
            public["samples"] = copy.deepcopy(samples)
        return public

    def temperature_runs_state(self) -> dict:
        runs = sorted(
            (
                self._public_temperature_run(run)
                for run in self.temperature_runs.values()
            ),
            key=lambda run: float(run.get("armed_at") or 0.0),
            reverse=True,
        )
        return {
            "sample_interval_seconds": TEMPERATURE_RUN_SAMPLE_INTERVAL_SECONDS,
            "active_run_id": self._active_temperature_run_id,
            "runs": runs,
        }

    def temperature_run(self, run_id: str) -> dict:
        run = self.temperature_runs.get(str(run_id or ""))
        if not run:
            raise ValueError("temperature run not found")
        return self._public_temperature_run(run, include_samples=True)

    @staticmethod
    def _temperature_run_values(stats: dict) -> tuple[float | None, float | None]:
        def finite(value: Any) -> float | None:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return None
            value = float(value)
            return round(value, 1) if math.isfinite(value) else None

        gpu = (stats.get("gpus") or [{}])[0]
        return finite(stats.get("cpu_temp_c")), finite(gpu.get("temp"))

    async def _temperature_stats_for_node(self, node_id: str) -> dict:
        if not node_id or node_id == LOCAL_NODE_ID:
            return await self.get_stats()
        result = await self.node_registry.request(
            node_id, "GET", "/api/agent/stats", timeout=5,
        )
        if not isinstance(result, dict):
            raise RuntimeError("node returned invalid temperature telemetry")
        return result

    def _temperature_node_name(self, node_id: str) -> str:
        if not node_id or node_id == LOCAL_NODE_ID:
            return str(self.settings.get("cluster_node_name") or socket.gethostname())
        node = self.node_registry.get(node_id)
        if not node:
            raise ValueError("target node not found")
        if not node.get("enabled", True):
            raise ValueError(f"node {node.get('name', node_id)} is disabled")
        return str(node.get("name") or node.get("hostname") or node_id)

    def _process_temperature_run_sample(
        self, run: dict, stats: dict, observed_at: float,
    ) -> str:
        cpu_temp, gpu_temp = self._temperature_run_values(stats)
        available = [value for value in (cpu_temp, gpu_temp) if value is not None]
        hottest = max(available) if available else None
        if hottest is not None:
            run.pop("last_error", None)
            run.pop("telemetry_failures", None)
        status = run.get("status")
        if status == "armed":
            if hottest is None or hottest < float(run["trigger_temp_c"]):
                return "waiting"
            run["status"] = "recording"
            run["started_at"] = observed_at
            status = "recording"

        if status != "recording":
            return str(status or "idle")

        started_at = float(run.get("started_at") or observed_at)
        run.setdefault("samples", []).append({
            "elapsed_seconds": round(max(0.0, observed_at - started_at), 3),
            "cpu_temp_c": cpu_temp,
            "gpu_temp_c": gpu_temp,
        })
        run["last_sample_at"] = observed_at
        if hottest is not None and hottest < float(run["target_temp_c"]):
            run["status"] = "complete"
            run["stopped_at"] = observed_at
            return "complete"
        return "recording"

    @staticmethod
    def _record_temperature_telemetry_failure(
        run: dict, error: Exception | str, observed_at: float,
    ) -> bool:
        """Record a telemetry miss and interrupt a run after a bounded streak."""
        failures = int(run.get("telemetry_failures") or 0) + 1
        run["telemetry_failures"] = failures
        run["last_error"] = str(error)
        if failures < TEMPERATURE_RUN_MAX_TELEMETRY_FAILURES:
            return False
        run["status"] = "interrupted"
        run["stopped_at"] = observed_at
        run["interruption_reason"] = (
            f"Temperature telemetry unavailable for {failures} consecutive polls"
        )
        return True

    async def _temperature_recording_loop(self, run_id: str) -> None:
        try:
            while self._active_temperature_run_id == run_id:
                await asyncio.sleep(TEMPERATURE_RUN_SAMPLE_INTERVAL_SECONDS)
                run = self.temperature_runs.get(run_id)
                if not run or run.get("status") not in {"armed", "recording"}:
                    break
                observed_at = time.time()
                try:
                    stats = await self._temperature_stats_for_node(run["node_id"])
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    interrupted = self._record_temperature_telemetry_failure(
                        run, exc, observed_at,
                    )
                    if interrupted:
                        self._active_temperature_run_id = None
                        self._save_temperature_runs()
                        break
                    if time.time() - self._temperature_runs_last_saved_at >= 5.0:
                        self._save_temperature_runs()
                    continue
                if not any(
                    value is not None for value in self._temperature_run_values(stats)
                ):
                    interrupted = self._record_temperature_telemetry_failure(
                        run, "temperature telemetry unavailable", observed_at,
                    )
                    if interrupted:
                        self._active_temperature_run_id = None
                        self._save_temperature_runs()
                        break
                    if time.time() - self._temperature_runs_last_saved_at >= 5.0:
                        self._save_temperature_runs()
                    continue
                result = self._process_temperature_run_sample(run, stats, observed_at)
                if result == "complete":
                    self._active_temperature_run_id = None
                    self._save_temperature_runs()
                    break
                if time.time() - self._temperature_runs_last_saved_at >= 5.0:
                    self._save_temperature_runs()
        except asyncio.CancelledError:
            raise
        finally:
            if self.temperature_recording_task is asyncio.current_task():
                self.temperature_recording_task = None

    async def arm_temperature_recording(
        self,
        node_id: Any,
        target_temp_c: Any,
        trigger_margin_pct: Any = 5.0,
    ) -> dict:
        node_id = str(node_id or LOCAL_NODE_ID).strip() or LOCAL_NODE_ID
        try:
            target = float(target_temp_c)
            margin = float(trigger_margin_pct)
        except (TypeError, ValueError) as exc:
            raise ValueError("target temperature and trigger margin must be numbers") from exc
        if not math.isfinite(target) or target < 1 or target > 120:
            raise ValueError("target temperature must be between 1 and 120°C")
        if not math.isfinite(margin) or margin < 0 or margin > 100:
            raise ValueError("trigger margin must be between 0 and 100%")

        async with self._temperature_recording_lock:
            if self._active_temperature_run_id:
                raise ValueError("a temperature run is already armed or recording")
            node_name = self._temperature_node_name(node_id)
            stats = await self._temperature_stats_for_node(node_id)
            now = time.time()
            run_id = uuid.uuid4().hex[:10]
            run = {
                "id": run_id,
                "name": f"{node_name} · {datetime.fromtimestamp(now).strftime('%Y-%m-%d %H:%M:%S')}",
                "node_id": node_id,
                "node_name": node_name,
                "status": "armed",
                "armed_at": now,
                "started_at": None,
                "stopped_at": None,
                "target_temp_c": round(target, 1),
                "trigger_margin_pct": round(margin, 1),
                "trigger_temp_c": round(target * (1.0 + margin / 100.0), 1),
                "samples": [],
            }
            self.temperature_runs[run_id] = run
            self._active_temperature_run_id = run_id
            self._process_temperature_run_sample(run, stats, now)
            self._save_temperature_runs()
            self.temperature_recording_task = asyncio.create_task(
                self._temperature_recording_loop(run_id)
            )
            return self._public_temperature_run(run, include_samples=True)

    async def cancel_temperature_recording(self) -> dict:
        async with self._temperature_recording_lock:
            run_id = self._active_temperature_run_id
            run = self.temperature_runs.get(run_id)
            if not run:
                raise ValueError("no temperature run is active")
            task = self.temperature_recording_task
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            run["status"] = "cancelled"
            run["stopped_at"] = time.time()
            self._active_temperature_run_id = None
            self.temperature_recording_task = None
            self._save_temperature_runs()
            return self._public_temperature_run(run, include_samples=True)

    def rename_temperature_run(self, run_id: str, name: Any) -> dict:
        run = self.temperature_runs.get(str(run_id or ""))
        if not run:
            raise ValueError("temperature run not found")
        clean_name = str(name or "").strip()
        if not clean_name:
            raise ValueError("run name is required")
        if len(clean_name) > 120:
            raise ValueError("run name must be 120 characters or fewer")
        run["name"] = clean_name
        self._save_temperature_runs()
        return self._public_temperature_run(run, include_samples=True)

    async def _temperature_history_monitor_loop(self) -> None:
        while True:
            try:
                await self.get_stats()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(f"[telemetry] temperature history sample failed: {exc}")
            await asyncio.sleep(TEMPERATURE_HISTORY_INTERVAL_SECONDS)

    async def get_disk(self) -> dict:
        """Return disk usage separately; polled infrequently by the UI."""
        return self._read_disk()

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
            "attempts": 0,
            "_cancel_event": asyncio.Event(),
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
            if job and job["status"] in (
                PENDING, DISPATCHING, ADMISSION_WAITING,
            ):
                event = job.get("_cancel_event")
                if event is not None:
                    event.set()
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
            "attempts": j.get("attempts", 0),
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

        # Index containers by model and by name
        by_name = {c["name"]: c for c in containers}
        running_by_model: dict[str, list[dict]] = {}
        for c in running:
            for model_id in self._container_model_ids(c):
                running_by_model.setdefault(model_id, []).append(c)
        any_by_model: dict[str, list[dict]] = {}
        for c in containers:
            for model_id in self._container_model_ids(c):
                any_by_model.setdefault(model_id, []).append(c)

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
                    # Keep the job cancelable while it waits in the separate
                    # controller admission queue. _run_inference marks it
                    # RUNNING only after a slot is actually granted.
                    job["status"] = ADMISSION_WAITING
                    asyncio.create_task(self._run_inference(jid, target))
                else:
                    job["status"] = DISPATCHING
                continue

            # Need to start or create — VRAM-aware check
            can_start = bool(job.get("override"))
            if not can_start:
                # Estimate how much VRAM this model needs; skip this tick
                # if the GPU is full and we can't evict anything.
                params_b, bpp = self._estimate_params_and_quant(job["model"])
                if params_b > 0:
                    need_gb = params_b * 1e9 * bpp * 1.2 / (1024 ** 3)
                else:
                    need_gb = 30.0  # conservative fallback
                total, used = self._read_gpu_memory_gb()
                free = total - used - GPU_VRAM_BUFFER_GB
                can_start = free >= need_gb or len(running) == 0
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

    # ---------- idle monitor / auto-stop ----------
    _RE_VLLM_COUNTER = re.compile(
        r"^vllm:(?:request_success_total|e2e_request_latency_seconds_count|"
        r"request_prompt_tokens_total|request_generation_tokens_total)\b[^\s]*\s+(\d+(?:\.\d+)?)",
        re.MULTILINE,
    )

    async def _read_activity_counter(self, port: int) -> int | None:
        """Sum of vLLM request-counter metrics; None if /metrics is unavailable."""
        try:
            r = await self.http.get(f"http://localhost:{port}/metrics", timeout=2)
            if r.status_code != 200:
                return None
            total = 0
            for m in self._RE_VLLM_COUNTER.finditer(r.text):
                try:
                    total += int(float(m.group(1)))
                except ValueError:
                    continue
            return total
        except Exception:
            return None

    def _mark_active(self, container_name: str | None, counter: int | None = None):
        if not container_name:
            return
        rec = self._activity.setdefault(container_name, {})
        rec["last_active"] = time.time()
        if counter is not None:
            rec["counter"] = counter

    async def _idle_monitor_loop(self):
        while True:
            try:
                await self._idle_tick()
            except Exception as e:
                print(f"[idle-monitor] error: {e}")
            await asyncio.sleep(5.0)

    async def _idle_tick(self):
        timeout = int(self.settings.get("idle_timeout_seconds", 0) or 0)
        if timeout <= 0:
            return  # disabled
        try:
            containers = await self.list_containers()
        except Exception:
            return
        now = time.time()
        for c in containers:
            # Only auto-stop containers we manage and that are healthy.
            if not c.get("managed") or c.get("status") != "running":
                continue
            # A distributed worker does not necessarily expose the public API
            # or see request counters.  Stopping one member independently would
            # tear down the entire deployment, so cluster lifecycle is always
            # coordinated at the deployment level.
            if c.get("deployment_id") and (
                c.get("deployment_mode") != "single"
                or int(c.get("nnodes") or 1) > 1
            ):
                continue
            phase = (c.get("phase") or {}).get("phase")
            if phase != "ready":
                continue
            name = c["name"]
            counter = await self._read_activity_counter(c["port"])
            rec = self._activity.get(name)
            prev_counter = rec.get("counter") if rec else None
            if counter is not None and (prev_counter is None or counter > prev_counter):
                # Activity detected (counter rose, or first read).
                self._mark_active(name, counter)
                continue
            if counter is not None and prev_counter is not None and counter < prev_counter:
                # Counter went backwards = container restarted; reset baseline.
                self._mark_active(name, counter)
                continue
            # No new activity. If we've never seen any, seed last_active to now.
            if rec is None or "last_active" not in rec:
                self._mark_active(name, counter or 0)
                continue
            if now - rec["last_active"] >= timeout:
                try:
                    print(f"[idle-monitor] stopping {name} (idle for {int(now - rec['last_active'])}s)")
                    await self.stop_container(name)
                    self._activity.pop(name, None)
                except Exception as e:
                    print(f"[idle-monitor] failed to stop {name}: {e}")

    async def _container_by_name(self, name: str) -> dict | None:
        def _do():
            try:
                c = self.client.containers.get(name)
            except Exception:
                return None
            return self._container_summary(c)
        return await asyncio.to_thread(_do)

    async def ensure_loaded(
        self, model: str, timeout: float = 300.0, *,
        container_name: str | None = None,
        deployment_id: str | None = None,
    ) -> dict:
        """
        Make a managed container for `model` running and ready. Uses VRAM-aware
        eviction: if the GPU is full (total − used − 10 GB buffer < estimated
        model size), the least-recently-active running managed container is
        evicted to make room. Held under _swap_lock so concurrent callers
        serialize. Raises LookupError if no managed container exists for this
        model, TimeoutError if it never becomes ready.
        """
        async with self._swap_lock:
            containers = await self.list_containers()
            target = None
            for c in containers:
                if container_name and c.get("name") != container_name:
                    continue
                if (
                    deployment_id and c.get("deployment_id")
                    and c.get("deployment_id") != deployment_id
                ):
                    continue
                if not c.get("managed") or model not in self._container_model_ids(c):
                    continue
                # Prefer a running candidate if there are duplicates.
                if target is None or c["status"] == "running":
                    target = c
            if target is None:
                raise LookupError(model)

            # If the target is already running, just wait for it to be ready.
            if target["status"] == "running":
                if await self._check_ready(target):
                    self._mark_active(target["name"])
                    return target
                deadline = time.time() + timeout
                while time.time() < deadline:
                    if await self._check_ready(target):
                        self._mark_active(target["name"])
                        return target
                    await asyncio.sleep(2.0)
                raise TimeoutError(f"{model} not ready after {int(timeout)}s")

            # Need to start the target — evict LRU containers only if
            # the GPU doesn't have enough free memory.
            params_b, bpp = self._estimate_params_and_quant(target.get("model") or model)
            if params_b > 0:
                need_gb = params_b * 1e9 * bpp * 1.2 / (1024 ** 3)
            else:
                need_gb = 30.0  # conservative fallback
            self._try_fit_new_model(need_gb, protect_name=target["name"])

            await self.start_container(target["name"])

            deadline = time.time() + timeout
            while time.time() < deadline:
                fresh = await self._container_by_name(target["name"])
                if fresh and fresh["status"] == "running" and await self._check_ready(fresh):
                    self._mark_active(fresh["name"])
                    return fresh
                await asyncio.sleep(2.0)
            raise TimeoutError(f"{model} not ready after {int(timeout)}s")

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
        admission = None
        cancel = job.get("_cancel_event")
        try:
            container, admission = await self._admit_vllm_target(
                job["model"], cancel, container,
            )
            if cancel is not None and cancel.is_set():
                raise ClientAbort("job canceled while queued")
            job["status"] = RUNNING
            job["started_at"] = time.time()
            job["attempts"] = job.get("attempts", 0) + 1
            self._mark_deployment_used(container.get("deployment_id"))
            # Refresh idle clock only when the job is actually admitted to the
            # backend, not while it is waiting in the controller queue.
            self._mark_active(container.get("name"))
            url = f"http://localhost:{container['port']}/v1/chat/completions"
            payload: dict = {
                "messages": job["messages"],
            }
            payload.update(job.get("params") or {})
            payload["model"] = self._upstream_model_id(container, job["model"])
            r = await self.http.post(url, json=payload, timeout=600)
            r.raise_for_status()
            job["result"] = r.json()
            self._record_usage(
                container.get("stats_key") or job["model"],
                (job["result"] or {}).get("usage"),
            )
            job["error"] = None
            job["status"] = DONE
        except ClientAbort:
            job["status"] = CANCELED
            job["error"] = None
        except Exception as e:
            if isinstance(e, httpx.HTTPStatusError):
                err_msg = f"HTTP {e.response.status_code}: {e.response.text[:500]}"
            else:
                err_msg = str(e)
            max_retries = int(self.settings.get("max_retries", 2))
            attempts = job.get("attempts", 1)
            if attempts <= max_retries:
                # Re-queue for another attempt
                job["error"] = f"{err_msg} (attempt {attempts}/{max_retries + 1}, retrying…)"
                job["status"] = PENDING
                async with self.lock:
                    if jid not in self.queue:
                        self.queue.append(jid)
                return
            job["status"] = ERROR
            job["error"] = f"{err_msg} (after {attempts} attempt{'s' if attempts != 1 else ''})"
        finally:
            if job.get("started_at"):
                self._mark_deployment_used(container.get("deployment_id"))
            self._release_inference_slot(admission)
            if job["status"] in (DONE, ERROR, CANCELED):
                job["completed_at"] = time.time()
                async with self.lock:
                    if jid in self.queue:
                        try:
                            self.queue.remove(jid)
                        except ValueError:
                            pass

    # ---------- aggregate state ----------
    async def get_state(self) -> dict:
        containers_result, images_result, stats_result = await asyncio.gather(
            self.list_containers(),
            self.list_images(),
            self.get_stats(),
            return_exceptions=True,
        )
        for result in (containers_result, images_result):
            if isinstance(result, BaseException) and not isinstance(
                result, docker.errors.DockerException
            ):
                raise result
        if isinstance(stats_result, BaseException):
            raise stats_result
        containers_unavailable = isinstance(
            containers_result, docker.errors.DockerException
        )
        images_unavailable = isinstance(images_result, docker.errors.DockerException)
        containers = [] if containers_unavailable else containers_result
        images = [] if images_unavailable else images_result
        stats = stats_result
        nodes = await self.cluster_nodes(stats)
        local_docker_ready = next(
            (
                bool(node.get("docker_ready"))
                for node in nodes
                if node.get("local") or node.get("id") == LOCAL_NODE_ID
            ),
            False,
        )
        containers_by_node: dict[str, dict[str, dict]] = {
            LOCAL_NODE_ID: {c.get("name"): c for c in containers},
        }
        for node in nodes:
            node_containers = containers_by_node.setdefault(node["id"], {})
            # Real local Docker summaries win over the synthetic launch rows;
            # remote nodes only have their advertised summaries here.
            for container in node.get("containers") or []:
                node_containers.setdefault(container.get("name"), container)
        node_by_id = {n["id"]: n for n in nodes}
        public_deployments = []
        for saved in self.deployments:
            deployment = {**saved, "members": []}
            member_states = []
            member_inventory_unknown = False
            runtime_flags = []
            primary_container = None
            for original in saved.get("members", []):
                member = dict(original)
                node = node_by_id.get(member.get("node_id"), {})
                container = containers_by_node.get(member.get("node_id"), {}).get(
                    member.get("container_name")
                )
                if not node.get("online"):
                    member["status"] = "unreachable"
                elif container:
                    member["status"] = container.get("status", "unknown")
                    member["phase"] = container.get("phase")
                    member["container_id"] = container.get("id", member.get("container_id"))
                    load_settings = container.get("load_settings") or {}
                    command_flags = load_settings.get("command_flags")
                    if command_flags is not None:
                        runtime_flags.append({
                            "node_id": member.get("node_id"),
                            "node_name": member.get("node_name"),
                            "rank": member.get("rank"),
                            "command_flags": command_flags,
                        })
                    if primary_container is None or member.get("rank") == 0:
                        primary_container = container
                elif (
                    containers_unavailable
                    and member.get("node_id") == LOCAL_NODE_ID
                ):
                    member["status"] = "unknown"
                    member["status_message"] = "Docker is unavailable"
                    member_inventory_unknown = True
                elif saved.get("status") not in {"error", "launching"}:
                    member["status"] = "missing"
                member["node_status"] = node.get("status", "unknown")
                deployment["members"].append(member)
                member_states.append(member.get("status"))
            if saved.get("status") != "error":
                if member_inventory_unknown:
                    deployment["status"] = "unknown"
                    deployment["status_message"] = "Docker is unavailable"
                elif any(s in {"unreachable", "missing", "dead", "error"} for s in member_states):
                    deployment["status"] = "degraded"
                elif member_states and all(s == "exited" for s in member_states):
                    deployment["status"] = "stopped"
                elif member_states and all(s == "running" for s in member_states):
                    primary = deployment["members"][0]
                    phase = primary.get("phase") or {}
                    deployment["status"] = "ready" if phase.get("phase") == "ready" else "starting"
            if not deployment.get("launch_settings"):
                deployment["launch_settings"] = (
                    self._recovered_deployment_launch_settings(
                        saved, primary_container
                    )
                )
            deployment["launch_controls"] = self._deployment_launch_controls(
                deployment["launch_settings"]
            )
            deployment["pricing"] = self._deployment_pricing(
                deployment["launch_settings"]
            )
            deployment["runtime_flags"] = runtime_flags
            deployment["served_models"] = self._deployment_served_models(
                deployment, primary_container
            )
            deployment["served_model"] = (
                deployment["served_models"][0]
                if deployment["served_models"]
                else deployment.get("model")
            )
            deployment["selected_nodes"] = [
                self.public_target_node(node_by_id[node_id])
                for node_id in deployment.get("node_ids") or []
                if node_id in node_by_id
            ]
            deployment["stats_key"] = (
                primary_container.get("stats_key")
                if primary_container
                else None
            )
            if not deployment.get("last_used_at") and deployment["stats_key"]:
                prior_samples = self.speed_samples.get(
                    deployment["stats_key"], []
                )
                prior_last_used = max(
                    (
                        float(sample.get("ended_at") or 0)
                        for sample in prior_samples
                        if isinstance(sample, dict)
                    ),
                    default=0.0,
                )
                if prior_last_used > 0:
                    deployment["last_used_at"] = prior_last_used
            public_deployments.append(deployment)
        running_models = [c for c in containers if c["status"] == "running"]
        # Token-stats key of the model currently holding the GPU, so the UI
        # can auto-select the right per-variant entry on model switches.
        loaded_stats_key = next(
            (c.get("stats_key") or c["model"]
             for c in running_models if c.get("model")),
            None,
        )
        return {
            "containers": containers,
            "images": images,
            "docker_ready": bool(
                local_docker_ready
                and not containers_unavailable
                and not images_unavailable
            ),
            "stats": stats,
            "settings": self.public_settings(),
            "token_usage_sync": dict(self._token_usage_sync_status),
            "nodes": nodes,
            "deployments": public_deployments,
            "token_stats": self.token_stats,
            "token_costs": {
                model: self.calculate_cost(model, model_stats)
                for model, model_stats in self.token_stats.items()
            },
            "usage_aliases": dict(self.usage_aliases),
            "usage_merge_groups": dict(self.usage_merge_groups),
            "usage_routing_rules": dict(self.usage_routing_rules),
            "usage_cache_estimates": copy.deepcopy(self.usage_cache_estimates),
            "usage_rows": self.usage_rows(),
            "session_token_stats": self.session_token_stats,
            "active_requests": self.active_requests(),
            "inference_admission": self.inference_admission(),
            "queue": [self._public_job(j) for j in self.jobs.values()],
            "summary": {
                "running_models": len(running_models),
                "max_concurrent_models": int(self.settings.get("max_concurrent_models", 1)),
                # GPU VRAM for multi-model planning
                "gpu_vram": {
                    "total_gb": round(((stats.get("gpus") or [{}])[0].get("mem_total_mib") or 0) / 1024.0, 1),
                    "used_gb": round(((stats.get("gpus") or [{}])[0].get("mem_used_mib") or 0) / 1024.0, 1),
                },
                "total_containers": len(containers),
                "loaded_stats_key": loaded_stats_key,
                "cluster_nodes": len(nodes),
                "cluster_nodes_online": sum(1 for n in nodes if n.get("online")),
                "cluster_deployments": len(public_deployments),
            },
        }
