"""Durable private JSON persistence helpers."""

from __future__ import annotations

import json
import os
import secrets
from pathlib import Path
from typing import Any


def atomic_private_json_write(path: Path, value: Any) -> None:
    """Atomically write JSON with owner-only permissions from file creation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    descriptor = -1
    try:
        # Supplying 0600 to O_CREAT avoids the disclosure window caused by
        # creating a normal file and tightening its mode only after writing.
        descriptor = os.open(temporary, flags, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            json.dump(value, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        try:
            path.chmod(0o600)
        except OSError:
            # Windows does not implement POSIX modes. The restricted creation
            # mode is still supplied on platforms that enforce it.
            pass
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
