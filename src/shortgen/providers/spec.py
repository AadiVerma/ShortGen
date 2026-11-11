"""Registry entry schema: what ``configs/models.yaml`` declares for each model."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import Field

from shortgen.device import DeviceKind
from shortgen.domain.models import FrozenModel


class ProviderKind(StrEnum):
    """The five roles a model can play. Values match keys under ``providers:``."""

    PLANNER = "planner"
    SPEECH = "speech"
    TRANSCRIBER = "transcriber"
    IMAGE = "image"
    VIDEO = "video"


class WeightRef(FrozenModel):
    """An auxiliary set of weights an adapter downloads, with its own license."""

    repo: str
    purpose: str
    license: str
    commercial_use: bool


class ModelSpec(FrozenModel):
    """One registry entry: how to build an adapter and what its weights permit."""

    name: str
    kind: ProviderKind
    adapter: str = Field(pattern=r"^[A-Za-z_][\w.]*:[A-Za-z_]\w*$", description="module:Class")
    model_id: str | None = None
    license: str = Field(min_length=1)
    commercial_use: bool | None = Field(
        description="None means not determined: the provider's own terms decide (hosted services)"
    )
    hosted: bool = Field(
        default=False,
        description="True if requests leave this machine for a remote, usually paid, service",
    )
    source: str | None = None
    weights_host: Literal["huggingface", "ollama", "none"] = "none"
    extra_weights: list[WeightRef] = Field(default_factory=list)
    requires: list[str] = Field(default_factory=list, description="importable module names")
    install_hint: str | None = None
    devices: list[DeviceKind] = Field(default_factory=lambda: list(DeviceKind))
    min_memory_gb: float | None = None
    notes: str = ""
    options: dict[str, Any] = Field(default_factory=dict)


class ProviderFingerprint(FrozenModel):
    """What makes a provider's output differ; part of every stage's input hash."""

    name: str
    adapter: str
    model_id: str | None
    options: dict[str, Any]
