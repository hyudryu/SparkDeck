"""Validation for non-secret container environment variables."""

from __future__ import annotations

import re
from typing import Any


_ENVIRONMENT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SECRET_NAME = re.compile(
    r"(?:^|_)(?:TOKEN|SECRET|PASSWORD|PASSWD|API_KEY|PRIVATE_KEY|ACCESS_KEY)(?:$|_)",
    re.IGNORECASE,
)
_PROTECTED_NAMES = {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"}

# Docker inspection may expose arbitrary application credentials. Only these
# known runtime-tuning inputs are safe and useful to carry into a recipe saved
# from an externally created vLLM container.
_DISCOVERED_RUNTIME_ENVIRONMENT_NAMES = frozenset({
    "HF_HUB_OFFLINE",
    "NCCL_DEBUG",
    "NCCL_IB_DISABLE",
    "NCCL_IB_GID_INDEX",
    "NCCL_IB_HCA",
    "NCCL_NET",
    "NCCL_SOCKET_IFNAME",
    "PYTORCH_CUDA_ALLOC_CONF",
    "SPECULATIVE_CONFIG",
    "UCX_NET_DEVICES",
    "VLLM_ASSETS_CACHE",
    "VLLM_ATTENTION_BACKEND",
    "VLLM_CACHE_ROOT",
    "VLLM_CONFIG_ROOT",
    "VLLM_LOGGING_LEVEL",
    "VLLM_LOG_STATS_INTERVAL",
    "VLLM_NO_USAGE_STATS",
    "VLLM_TARGET_DEVICE",
    "VLLM_USE_V1",
    "VLLM_WORKER_MULTIPROC_METHOD",
})


def normalize_runtime_environment(
    value: Any, engine: str = "vllm",
) -> dict[str, str]:
    """Return a bounded, credential-free environment map for vLLM."""
    if value is None:
        return {}
    if engine != "vllm" and value:
        raise ValueError("runtime environment variables are only supported for vLLM")
    if not isinstance(value, dict):
        raise ValueError("environment must be an object of string values")
    if len(value) > 128:
        raise ValueError("environment cannot contain more than 128 variables")
    result: dict[str, str] = {}
    total_size = 0
    for name, raw_value in value.items():
        if not isinstance(name, str) or not _ENVIRONMENT_NAME.fullmatch(name):
            raise ValueError(f"invalid environment variable name: {name!r}")
        if name in _PROTECTED_NAMES:
            raise ValueError(
                f"{name} is managed by SparkDeck; configure Hugging Face credentials in Settings"
            )
        if _SECRET_NAME.search(name):
            raise ValueError(
                f"{name} looks secret and cannot be stored in deployment environment variables"
            )
        if not isinstance(raw_value, str):
            raise ValueError(f"environment variable {name} must have a string value")
        if "\x00" in raw_value:
            raise ValueError(f"environment variable {name} cannot contain a NUL character")
        if "\n" in raw_value or "\r" in raw_value:
            raise ValueError(f"environment variable {name} cannot contain a newline")
        total_size += len(name.encode("utf-8")) + len(raw_value.encode("utf-8"))
        if total_size > 65536:
            raise ValueError("environment cannot exceed 64 KiB")
        result[name] = raw_value
    return result


def discovered_runtime_environment(
    value: Any, engine: str = "vllm",
) -> dict[str, str]:
    """Return only allowlisted tuning values found through Docker inspection."""
    if engine != "vllm" or not isinstance(value, dict):
        return {}
    result: dict[str, str] = {}
    for name, raw_value in value.items():
        if name not in _DISCOVERED_RUNTIME_ENVIRONMENT_NAMES:
            continue
        try:
            result = normalize_runtime_environment(
                {**result, name: raw_value}, engine,
            )
        except ValueError:
            # Malformed image defaults should not hide the container itself.
            continue
    return result
