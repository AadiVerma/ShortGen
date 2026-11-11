"""Stable Diffusion XL base 1.0 (CreativeML Open RAIL++-M) via diffusers.

API used, as read from diffusers 0.42: ``StableDiffusionXLPipeline
.from_pretrained``, ``AutoencoderKL.from_pretrained``, ``load_ip_adapter`` /
``set_ip_adapter_scale`` / ``unload_ip_adapter`` and the ``ip_adapter_image``
call argument.

The IP-Adapter repository, subfolder and weight file name are the ones used
in the diffusers IP-Adapter guide. That guide also says model CPU offload
must be enabled after the adapter is loaded, which ``_reference_kwargs`` does.

Run for real on a 16 GB Apple Silicon Mac (text-to-image only; see
docs/hardware.md). The IP-Adapter path and CUDA have not been run.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import Field

from shortgen.adapters._torch import (
    half_precision,
    place_pipeline,
    release_memory,
    require,
    seeded_generator,
)
from shortgen.device import DeviceKind
from shortgen.domain.models import FrozenModel
from shortgen.providers.interfaces import ImageGenerator, ImageRequest
from shortgen.render.layout import fit_area

log = logging.getLogger(__name__)


class SdxlOptions(FrozenModel):
    """Sampling, sizing and IP-Adapter settings; values come from the registry entry."""

    vae_id: str | None
    steps: int = Field(ge=1)
    guidance_scale: float = Field(ge=0)
    max_area: int = Field(gt=0)
    size_multiple: int = Field(gt=0)
    cpu_offload: bool
    ip_adapter_repo: str
    ip_adapter_subfolder: str
    ip_adapter_weight: str
    ip_adapter_scale: float = Field(ge=0, le=1)


class SdxlImage(ImageGenerator):
    """Illustrations with optional reference-image conditioning (IP-Adapter)."""

    _pipeline: Any = None
    _ip_adapter_loaded = False

    def load(self) -> None:
        options = self.parse_options(SdxlOptions)
        torch = require("torch", self.spec.install_hint)
        diffusers = require("diffusers", self.spec.install_hint)
        kind = self.context.device.kind
        dtype = half_precision(torch, kind)
        kwargs: dict[str, Any] = {"torch_dtype": dtype, "use_safetensors": True}
        if kind is not DeviceKind.CPU:
            kwargs["variant"] = "fp16"
        if options.vae_id:
            kwargs["vae"] = diffusers.AutoencoderKL.from_pretrained(
                options.vae_id, torch_dtype=dtype
            )
        pipeline = diffusers.StableDiffusionXLPipeline.from_pretrained(self.spec.model_id, **kwargs)
        place_pipeline(pipeline, kind, options.cpu_offload)
        if kind is DeviceKind.MPS:
            # Recommended by the diffusers MPS guide for machines under 64 GB.
            pipeline.enable_attention_slicing()
        pipeline.vae.enable_tiling()
        self._pipeline = pipeline

    def unload(self) -> None:
        self._pipeline = None
        self._ip_adapter_loaded = False
        release_memory()

    def generate(self, request: ImageRequest) -> None:
        options = self.parse_options(SdxlOptions)
        torch = require("torch", self.spec.install_hint)
        size = fit_area(request.target, options.max_area, options.size_multiple)
        result = self._pipeline(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=size.width,
            height=size.height,
            num_inference_steps=options.steps,
            guidance_scale=options.guidance_scale,
            generator=seeded_generator(torch, request.seed),
            **self._reference_kwargs(request, options),
        )
        result.images[0].save(request.output_path, format="PNG")

    def _reference_kwargs(self, request: ImageRequest, options: SdxlOptions) -> dict[str, Any]:
        """Attach or detach IP-Adapter so each scene gets exactly what it asked for.

        Once IP-Adapter is attached the pipeline requires a reference on every
        call, so it is unloaded again for scenes without one.
        """
        if not request.reference_images:
            if self._ip_adapter_loaded:
                self._pipeline.unload_ip_adapter()
                self._ip_adapter_loaded = False
            return {}
        if len(request.reference_images) > 1:
            log.warning(
                "scene has %d reference images; SDXL IP-Adapter uses only the first (%s)",
                len(request.reference_images),
                request.reference_images[0].name,
            )
        if not self._ip_adapter_loaded:
            self._pipeline.load_ip_adapter(
                options.ip_adapter_repo,
                subfolder=options.ip_adapter_subfolder,
                weight_name=options.ip_adapter_weight,
            )
            self._pipeline.set_ip_adapter_scale(options.ip_adapter_scale)
            # Re-place the pipeline so the adapter's image encoder is on the
            # device, or covered by the offload hooks, like everything else.
            place_pipeline(self._pipeline, self.context.device.kind, options.cpu_offload)
            self._ip_adapter_loaded = True
        image_module = require("PIL.Image", self.spec.install_hint)
        reference = image_module.open(request.reference_images[0]).convert("RGB")
        return {"ip_adapter_image": reference}
