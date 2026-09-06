"""Read-only, node-local runtime patch files for managed vLLM launches."""
from pathlib import Path, PurePosixPath
from typing import Any

MAX_RUNTIME_FILE_MOUNTS = 16


def _overlap(first: str, second: str) -> bool:
    a, b = PurePosixPath(first), PurePosixPath(second)
    return a == b or a in b.parents or b in a.parents


def normalize_runtime_file_mounts(value: Any, engine: str = "vllm") -> list[dict[str, str]]:
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > MAX_RUNTIME_FILE_MOUNTS:
        raise ValueError("runtime_file_mounts must be an array of at most 16 files")
    if value and engine != "vllm":
        raise ValueError("runtime_file_mounts are only supported for vLLM")
    result = []
    for entry in value:
        if not isinstance(entry, dict) or set(entry) != {"source", "target"}:
            raise ValueError("each runtime file mount must have only source and target")
        normalized = {}
        for key in ("source", "target"):
            path = entry[key]
            if (not isinstance(path, str) or not path.startswith("/")
                    or path.startswith("//") or str(PurePosixPath(path)) == "/" or len(path) > 4096
                    or any(char in path for char in ("\\", ":", "\x00", "\n", "\r"))
                    or ".." in PurePosixPath(path).parts or path.endswith("/")):
                raise ValueError(f"runtime file mount {key} must be an absolute Linux file path")
            normalized[key] = str(PurePosixPath(path))
        if any(_overlap(normalized["target"], old["target"]) for old in result):
            raise ValueError("runtime file mount targets must not overlap")
        if any(normalized["source"] == old["source"] for old in result):
            raise ValueError("runtime file mount sources must be unique")
        result.append(normalized)
    return result


def runtime_file_volumes(value: Any, managed_volumes: dict) -> dict:
    """Validate on the executing node before Docker can create missing paths."""
    volumes = dict(managed_volumes)
    for entry in normalize_runtime_file_mounts(value):
        source, target = entry["source"], entry["target"]
        if any(_overlap(target, bind["bind"]) for bind in managed_volumes.values()):
            raise ValueError("runtime file mount target conflicts with a managed cache or model mount")
        local = Path(source)
        if not local.is_file():
            raise ValueError(f"runtime file mount source must be an existing regular file on this node: {source}")
        # Reject aliases of a managed source rather than replacing its bind.
        if any(local.resolve() == Path(path).resolve() for path in managed_volumes):
            raise ValueError("runtime file mount source conflicts with a managed mount")
        if source in volumes:
            raise ValueError("runtime file mount sources must be unique")
        volumes[source] = {"bind": target, "mode": "ro"}
    return volumes
