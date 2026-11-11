"""FLUX.1-schnell (Apache-2.0) via diffusers.

API used, as read from diffusers 0.42: ``FluxPipeline.from_pretrained(model,
torch_dtype=torch.bfloat16)`` and a call with ``num_inference_steps=4,
guidance_scale=0.0, max_sequence_length=256``.

Only the schnell variant is registered. FLUX.1-dev uses the same pipeline but
its weights are under a non-commercial license.

Not yet run against real weights by the shortgen authors; see docs/hardware.md.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import Field

from shortgen.adapters._torch import place_pipeline, release_memory, require, seeded_generator
from shortgen.domain.models import FrozenModel
from shortgen.providers.interfaces import ImageGenerator, ImageRequest
from shortgen.render.layout import fit_area

log = logging.getLogger(__name__)


class FluxOptions(FrozenModel):
    """Sampling and sizing settings; values come from the registry entry."""

    steps: int = Field(ge=1)
    guidance_scale: float = Field(ge=0)
    max_sequence_length: int = Field(ge=1, le=512)
    max_area: int = Field(gt=0)
    size_multiple: int = Field(gt=0)
    cpu_offload: bool


class FluxImage(ImageGenerator):
    """Fast few-step illustrations. No negative prompt and no reference images."""

    _pipeline: Any = None

    def load(self) -> None:
        options = self.parse_options(FluxOptions)
        torch = require("torch", self.spec.install_hint)
        diffusers = require("diffusers", self.spec.install_hint)
        pipeline = diffusers.FluxPipeline.from_pretrained(
            self.spec.model_id, torch_dtype=torch.bfloat16
        )
        place_pipeline(pipeline, self.context.device.kind, options.cpu_offload)
        self._pipeline = pipeline

    def unload(self) -> None:
        self._pipeline = None
        release_memory()

    def generate(self, request: ImageRequest) -> None:
        options = self.parse_options(FluxOptions)
        torch = require("torch", self.spec.install_hint)
        if request.reference_images:
            log.warning(
                "%s does not support reference images; ignoring %d for this scene",
                self.name,
                len(request.reference_images),
            )
        size = fit_area(request.target, options.max_area, options.size_multiple)
        result = self._pipeline(
            prompt=request.prompt,
            width=size.width,
            height=size.height,
            num_inference_steps=options.steps,
            guidance_scale=options.guidance_scale,
            max_sequence_length=options.max_sequence_length,
            generator=seeded_generator(torch, request.seed),
        )
        result.images[0].save(request.output_path, format="PNG")
