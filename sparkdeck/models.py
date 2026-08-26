"""Typed public contracts shared by the SparkDeck backend services."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class RuntimeKind(StrEnum):
    VLLM = "vllm"
    LLAMA_CPP = "llama.cpp"
    SGLANG = "sglang"


class DeploymentKind(StrEnum):
    MANAGED = "managed"
    EXTERNAL = "external"


@dataclass(slots=True)
class ModelIdentity:
    repository: str
    revision: str | None = None
    artifact: str | None = None
    quantization: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Deployment:
    id: str
    alias: str
    runtime: RuntimeKind
    kind: DeploymentKind
    model: ModelIdentity
    status: str = "registered"
    container_name: str | None = None
    settings: dict[str, Any] = field(default_factory=dict)
    base_url_set: bool = False

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["runtime"] = self.runtime.value
        value["kind"] = self.kind.value
        return value


@dataclass(slots=True)
class BenchmarkSample:
    id: str
    created_at: str
    deployment_id: str | None
    model: ModelIdentity
    runtime: RuntimeKind
    runtime_version: str | None
    hardware: dict[str, Any]
    configuration: dict[str, Any]
    input_tokens: int
    output_tokens: int
    latency_ms: float
    ttft_ms: float | None
    generation_tokens_per_second: float | None
    prompt_tokens_per_second: float | None
    cold_start: bool | None
    eligible_for_community: bool

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["runtime"] = self.runtime.value
        return value
