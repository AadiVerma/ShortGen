"""Frame geometry and clip fitting. Pure arithmetic, no ffmpeg calls."""

from __future__ import annotations

import math

from pydantic import Field

from shortgen.domain.models import FrozenModel

_EPSILON = 1e-6


class Size(FrozenModel):
    """A frame size in pixels."""

    width: int = Field(gt=0)
    height: int = Field(gt=0)


class ClipFit(FrozenModel):
    """How to make a generated clip last exactly as long as its scene.

    ``slowdown`` multiplies presentation timestamps (1.0 = untouched). Whatever
    time is still missing after slowing down is filled by holding the last
    frame; a clip that is too long is simply cut.
    """

    slowdown: float = Field(ge=1)
    hold_s: float = Field(ge=0)


def fit_area(target: Size, max_area: int, multiple: int) -> Size:
    """Largest size with ``target``'s aspect ratio within a pixel budget.

    Diffusion models have a native pixel budget and need dimensions divisible
    by some factor. Each side is rounded down to ``multiple``, as in the
    reference pipelines.
    """
    width = _floor_to(math.sqrt(max_area * target.width / target.height), multiple)
    height = _floor_to(math.sqrt(max_area * target.height / target.width), multiple)
    return Size(width=width, height=height)


def fit_clip(clip_s: float, target_s: float, max_slowdown: float) -> ClipFit:
    """Decide how a clip of ``clip_s`` seconds fills a ``target_s`` second scene."""
    if clip_s <= 0 or clip_s >= target_s:
        return ClipFit(slowdown=1.0, hold_s=0.0)
    slowdown = min(max_slowdown, target_s / clip_s)
    return ClipFit(slowdown=slowdown, hold_s=max(0.0, target_s - clip_s * slowdown))


def cover_filter(size: Size) -> str:
    """ffmpeg filter chain that fills ``size`` exactly, cropping any overflow."""
    return (
        f"scale={size.width}:{size.height}:force_original_aspect_ratio=increase,"
        f"crop={size.width}:{size.height}"
    )


def _floor_to(value: float, multiple: int) -> int:
    """Round down to a multiple; the epsilon absorbs float error on exact roots."""
    return max(multiple, int(value + _EPSILON) // multiple * multiple)
