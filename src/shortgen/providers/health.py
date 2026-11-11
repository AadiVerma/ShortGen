"""Availability checks shared by every adapter and reported by ``shortgen doctor``."""

from __future__ import annotations

import importlib
import importlib.util
import logging
from enum import StrEnum

from shortgen.device import DeviceInfo, DeviceKind
from shortgen.domain.models import FrozenModel
from shortgen.providers.spec import ModelSpec

log = logging.getLogger(__name__)


class HealthStatus(StrEnum):
    """Outcome of one check."""

    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


class HealthCheck(FrozenModel):
    """One line of a doctor report."""

    name: str
    status: HealthStatus
    detail: str
    hint: str | None = None


def default_health(spec: ModelSpec, device: DeviceInfo) -> list[HealthCheck]:
    """Dependency, device, memory and weight-cache checks derived from the registry entry."""
    return [
        *check_imports(spec),
        *check_device(spec, device),
        *check_cached_weights(spec),
    ]


def check_imports(spec: ModelSpec) -> list[HealthCheck]:
    """Fail for each Python module the adapter needs but cannot import."""
    missing = [module for module in spec.requires if importlib.util.find_spec(module) is None]
    if missing:
        return [
            HealthCheck(
                name=f"{spec.name}: dependencies",
                status=HealthStatus.FAIL,
                detail=f"missing Python module(s): {', '.join(missing)}",
                hint=spec.install_hint,
            )
        ]
    if not spec.requires:
        return []
    detail = f"{', '.join(spec.requires)} importable"
    return [HealthCheck(name=f"{spec.name}: dependencies", status=HealthStatus.OK, detail=detail)]


def check_device(spec: ModelSpec, device: DeviceInfo) -> list[HealthCheck]:
    """Warn when the model is not known to run on this device or needs more memory."""
    checks: list[HealthCheck] = []
    # A model that can run on the CPU runs everywhere, so only models that need
    # a specific accelerator are flagged.
    if device.kind not in spec.devices and DeviceKind.CPU not in spec.devices:
        supported = ", ".join(kind.value for kind in spec.devices)
        checks.append(
            HealthCheck(
                name=f"{spec.name}: device",
                status=HealthStatus.WARN,
                detail=f"not expected to work on {device.kind.value} (supported: {supported})",
            )
        )
    budget = device.accelerator_memory_gb or device.total_memory_gb
    if spec.min_memory_gb is not None and budget < spec.min_memory_gb:
        checks.append(
            HealthCheck(
                name=f"{spec.name}: memory",
                status=HealthStatus.WARN,
                detail=f"wants about {spec.min_memory_gb:g} GB, this device has {budget:g} GB",
                hint="See docs/hardware.md for lighter alternatives.",
            )
        )
    return checks


def check_cached_weights(spec: ModelSpec) -> list[HealthCheck]:
    """Report whether Hugging Face weights are already in the local cache."""
    if spec.weights_host != "huggingface":
        return []
    repos = [repo for repo in (spec.model_id, *(ref.repo for ref in spec.extra_weights)) if repo]
    cached = _cached_hf_repos()
    if cached is None:
        return [
            HealthCheck(
                name=f"{spec.name}: weights",
                status=HealthStatus.WARN,
                detail="cannot inspect the Hugging Face cache (huggingface_hub not installed)",
                hint=spec.install_hint,
            )
        ]
    absent = [repo for repo in repos if repo not in cached]
    if absent:
        return [
            HealthCheck(
                name=f"{spec.name}: weights",
                status=HealthStatus.WARN,
                detail=f"not downloaded yet: {', '.join(absent)}",
                hint="They download automatically on first use.",
            )
        ]
    return [HealthCheck(name=f"{spec.name}: weights", status=HealthStatus.OK, detail="cached")]


def _cached_hf_repos() -> set[str] | None:
    """Repo IDs in the local Hugging Face cache; ``None`` if it cannot be inspected."""
    try:
        hub = importlib.import_module("huggingface_hub")
    except ImportError:
        return None
    try:
        return {str(repo.repo_id) for repo in hub.scan_cache_dir().repos}
    except Exception:  # a missing cache directory raises a library-specific error
        log.debug("could not scan the Hugging Face cache", exc_info=True)
        return set()
