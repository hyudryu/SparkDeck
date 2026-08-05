"""Temperature sources: nvidia-smi for the GPU."""
from __future__ import annotations

import logging
import subprocess
from typing import Optional

log = logging.getLogger(__name__)


class TempSource:
    def __init__(self, key: str, label: str) -> None:
        self.key = key
        self.label = label

    def read(self) -> Optional[float]:
        raise NotImplementedError


class NvidiaSmiSource(TempSource):
    def __init__(self) -> None:
        super().__init__("gpu", "GPU (nvidia-smi)")

    def read(self) -> Optional[float]:
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=2,
            )
        except (subprocess.SubprocessError, FileNotFoundError):
            return None
        if out.returncode != 0:
            return None
        temps: list[float] = []
        for line in out.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                temps.append(float(line))
            except ValueError:
                log.warning("ignoring non-numeric nvidia-smi output: %r", line)
        return max(temps) if temps else None


def discover() -> list[TempSource]:
    """Discover available temperature sources.

    Currently only the NVIDIA GPU (via nvidia-smi) is supported.  Returns an
    empty list if no sources are available.
    """
    sources: list[TempSource] = []
    nv = NvidiaSmiSource()
    if nv.read() is not None:
        sources.append(nv)
    return sources


def aggregate_max(sources: list[TempSource]) -> Optional[float]:
    vals = [v for v in (s.read() for s in sources) if v is not None]
    return max(vals) if vals else None
