"""Helpers shared by adapters with heavy, optional dependencies."""

from __future__ import annotations

import gc
import importlib
from typing import Any

from shortgen.device import DeviceKind
from shortgen.errors import ProviderUnavailableError


def require(module: str, install_hint: str | None) -> Any:
    """Import an optional dependency, or explain how to install it."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        hint = f" Install it with: {install_hint}" if install_hint else ""
        raise ProviderUnavailableError(f"Python module '{module}' is not installed.{hint}") from exc


def release_memory() -> None:
    """Return freed model memory to the system (CUDA and MPS keep caches otherwise)."""
    gc.collect()
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def half_precision(torch: Any, kind: DeviceKind) -> Any:
    """float16 on accelerators, float32 on CPU where half precision is slow or unsupported."""
    return torch.float32 if kind is DeviceKind.CPU else torch.float16


def seeded_generator(torch: Any, seed: int) -> Any:
    """A CPU generator: the same seed then gives the same noise on every device."""
    return torch.Generator(device="cpu").manual_seed(seed)


def place_pipeline(pipeline: Any, kind: DeviceKind, cpu_offload: bool) -> None:
    """Move a diffusers pipeline to its device.

    Model CPU offload keeps only the active sub-model on the GPU. It relies on
    accelerate hooks that target CUDA, so it is only used there.
    """
    if cpu_offload and kind is DeviceKind.CUDA:
        pipeline.enable_model_cpu_offload()
    else:
        pipeline.to(kind.value)
