"""Compute device and memory detection.

PyTorch is optional at import time: the core, the CLI and ``doctor`` all work
without it, and report plainly that model adapters will not.
"""

from __future__ import annotations

import importlib
import platform
from enum import StrEnum
from typing import Any

import psutil
from pydantic import Field

from shortgen.domain.models import FrozenModel

_BYTES_PER_GB = 1024**3
AUTO = "auto"


class DeviceKind(StrEnum):
    """Where tensors live."""

    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"


class DeviceInfo(FrozenModel):
    """The device adapters should use, plus what is known about memory."""

    kind: DeviceKind
    name: str
    total_memory_gb: float
    available_memory_gb: float
    accelerator_memory_gb: float | None = None
    torch_version: str | None = None
    warnings: list[str] = Field(default_factory=list)


def detect_device(preference: str = AUTO) -> DeviceInfo:
    """Pick CUDA, then MPS, then CPU, honouring an explicit ``preference`` if usable.

    Falling back to CPU is never silent: the reason is recorded in ``warnings``
    for the caller to log.
    """
    torch = _import_torch()
    usable = _usable_kinds(torch)
    warnings: list[str] = []
    if torch is None:
        warnings.append("PyTorch is not installed; adapters that need it cannot run.")
    kind = _choose(preference, usable, warnings)
    memory = psutil.virtual_memory()
    return DeviceInfo(
        kind=kind,
        name=_device_name(kind, torch),
        total_memory_gb=round(memory.total / _BYTES_PER_GB, 1),
        available_memory_gb=round(memory.available / _BYTES_PER_GB, 1),
        accelerator_memory_gb=_accelerator_memory_gb(kind, torch, memory.total),
        torch_version=str(torch.__version__) if torch is not None else None,
        warnings=warnings,
    )


def _import_torch() -> Any | None:
    try:
        return importlib.import_module("torch")
    except ImportError:
        return None


def _usable_kinds(torch: Any | None) -> list[DeviceKind]:
    """Devices available here, best first. CPU is always last and always present."""
    usable: list[DeviceKind] = []
    if torch is not None and torch.cuda.is_available():
        usable.append(DeviceKind.CUDA)
    if torch is not None and torch.backends.mps.is_available():
        usable.append(DeviceKind.MPS)
    usable.append(DeviceKind.CPU)
    return usable


def _choose(preference: str, usable: list[DeviceKind], warnings: list[str]) -> DeviceKind:
    if preference != AUTO:
        wanted = DeviceKind(preference)
        if wanted in usable:
            return wanted
        warnings.append(f"Requested device '{preference}' is not available; falling back to CPU.")
        return DeviceKind.CPU
    best = usable[0]
    if best is DeviceKind.CPU and len(warnings) == 0:
        warnings.append(
            "No CUDA or MPS accelerator found; falling back to CPU. "
            "Image and video models will be extremely slow."
        )
    return best


def _device_name(kind: DeviceKind, torch: Any | None) -> str:
    if kind is DeviceKind.CUDA and torch is not None:
        return str(torch.cuda.get_device_name(0))
    if kind is DeviceKind.MPS:
        return "Apple Silicon GPU (MPS)"
    return platform.processor() or platform.machine() or "CPU"


def _accelerator_memory_gb(kind: DeviceKind, torch: Any | None, system_total: int) -> float | None:
    """Dedicated VRAM on CUDA; on MPS the GPU shares the system's unified memory."""
    if kind is DeviceKind.CUDA and torch is not None:
        total = float(torch.cuda.get_device_properties(0).total_memory)
        return round(total / _BYTES_PER_GB, 1)
    if kind is DeviceKind.MPS:
        return round(system_total / _BYTES_PER_GB, 1)
    return None
