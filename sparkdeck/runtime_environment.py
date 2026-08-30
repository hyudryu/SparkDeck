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
