"""Public GPU provenance and exact hardware cohorts for benchmark evidence."""

import hashlib
import json
import math
import re
from collections import Counter
from typing import Any


def benchmark_hardware(hardware: Any) -> dict[str, Any]:
    """Derive hardware from GPU names, never an asserted device-class label.

    Unknown historical rows remain unknown. Architecture and memory are useful
    metadata, but do not split identical GPUs when drivers reserve different
    memory or older agents omit their host architecture.
    """
    raw = hardware if isinstance(hardware, dict) else {}
    gpus = []
    raw_gpus = raw.get("gpus")
    invalid_inventory = isinstance(raw_gpus, list) and len(raw_gpus) > 1024
    for gpu in (raw_gpus if isinstance(raw_gpus, list) and not invalid_inventory else []):
        if not isinstance(gpu, dict):
            invalid_inventory = True
            continue
        model = gpu.get("model", gpu.get("name"))
        if not isinstance(model, str) or not model.strip() or len(model) > 200:
            invalid_inventory = True
            continue
        if any(ord(char) < 32 for char in model):
            invalid_inventory = True
            continue
        model = re.sub(r"\s+", " ", model.strip())
        memory = gpu.get("memory_mib", gpu.get("mem_total_mib"))
        if (
            isinstance(memory, bool) or not isinstance(memory, (int, float))
            or not math.isfinite(memory) or not 0 < memory <= 10_000_000
        ):
            memory = None
        gpus.append({"model": model, "memory_mib": memory})
    if invalid_inventory:
        gpus = []
    gpus.sort(key=lambda gpu: (gpu["model"].casefold(), gpu["memory_mib"] or 0))
    architecture = raw.get("architecture")
    if architecture not in {"aarch64", "arm64", "x86_64", "AMD64", "mixed"}:
        architecture = "unknown"
    if architecture == "AMD64":
        architecture = "x86_64"
    if architecture == "arm64":
        architecture = "aarch64"
    spark = [bool(re.search(r"\bgb10\b|\bdgx spark\b", gpu["model"], re.I)) for gpu in gpus]
    hardware_class = (
        "unknown" if not gpus else "dgx-spark" if all(spark)
        else "mixed" if any(spark) else "workstation"
    )
    public = {
        "hardware_class": hardware_class, "architecture": architecture,
        "gpu_count": len(gpus) if gpus else None, "gpus": gpus,
    }
    display_names = {
        "nvidia gb10" if is_spark else gpu["model"].casefold():
        "DGX Spark" if is_spark else gpu["model"]
        for gpu, is_spark in zip(gpus, spark)
    }
    names = sorted(
        "nvidia gb10" if is_spark else gpu["model"].casefold()
        for gpu, is_spark in zip(gpus, spark)
    )
    key = (
        hashlib.sha256(json.dumps(names, separators=(",", ":")).encode()).hexdigest()[:24]
        if names else "unknown"
    )
    label = (
        " + ".join(f"{count} × {display_names[name]}" for name, count in Counter(names).items())
        if names else "Unknown hardware"
    )
    return {"hardware": public, "hardware_key": key, "hardware_label": label}


def hardware_cohort(payload: dict[str, Any]) -> tuple[str, str, int, int, str]:
    return (payload["model_id"], payload["quantization"], payload["prompt_tokens_bucket"],
            payload["tensor_parallel_size"], payload["hardware_key"])
