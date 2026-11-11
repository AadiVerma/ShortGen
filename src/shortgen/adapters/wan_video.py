"""Wan 2.2 TI2V-5B image-to-video (Apache-2.0) via diffusers.

API used, as read from diffusers 0.42 (``pipelines/wan/pipeline_wan_i2v.py``):
``AutoencoderKLWan.from_pretrained(model, subfolder="vae",
torch_dtype=torch.float32)``, ``WanImageToVideoPipeline.from_pretrained(model,
vae=vae, torch_dtype=torch.bfloat16)`` and a call with ``image, prompt,
negative_prompt, height, width, num_frames, num_inference_steps,
guidance_scale, generator, output_type``; frames are at ``.frames[0]``.

Uncertain and flagged:

* The Wan 2.2 TI2V-5B model card only shows a text-to-video diffusers example
  (``WanPipeline``). Using the same checkpoint for image-to-video relies on
  ``WanImageToVideoPipeline``'s ``expand_timesteps`` code path, which exists in
  the diffusers source for this model family but was not executed here.
* The model card asks for a 24 GB GPU. Nothing here has been measured.

Frames are written as PNGs and encoded by ffmpeg, keeping ffmpeg the only
renderer and avoiding an imageio/OpenCV dependency.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Any

from pydantic import Field

from shortgen.adapters._motion import motion_prompt
from shortgen.adapters._torch import place_pipeline, release_memory, require, seeded_generator
from shortgen.providers.interfaces import VideoGenerator, VideoRequest
from shortgen.render.ffmpeg import FramesSpec, frames_command
from shortgen.render.layout import fit_area
from shortgen.render.settings import VideoEncode

_FRAME_GROUP = 4  # Wan's VAE compresses time 4x, so clips have 4k + 1 frames
_FRAME_PATTERN = "%05d.png"


class WanOptions(VideoEncode):
    """Sampling, clip length and encoding settings; values come from the registry entry."""

    steps: int = Field(ge=1)
    guidance_scale: float = Field(ge=0)
    max_area: int = Field(gt=0)
    native_fps: int = Field(ge=1)
    max_frames: int = Field(ge=_FRAME_GROUP + 1)
    cpu_offload: bool
    negative_prompt: str


def snap_frame_count(duration_s: float, fps: int, max_frames: int) -> int:
    """Smallest valid Wan frame count (4k + 1) that covers ``duration_s``, up to the cap.

    A scene longer than the cap gets the longest clip the model can make; the
    compose stage then slows it and holds the last frame.
    """
    cap = (max_frames - 1) // _FRAME_GROUP * _FRAME_GROUP + 1
    wanted = max(_FRAME_GROUP + 1, math.ceil(duration_s * fps))
    covering = math.ceil((wanted - 1) / _FRAME_GROUP) * _FRAME_GROUP + 1
    return min(cap, covering)


class WanVideo(VideoGenerator):
    """Neural image-to-video. Heavy: see the module docstring and docs/hardware.md."""

    _pipeline: Any = None

    def load(self) -> None:
        options = self.parse_options(WanOptions)
        torch = require("torch", self.spec.install_hint)
        diffusers = require("diffusers", self.spec.install_hint)
        vae = diffusers.AutoencoderKLWan.from_pretrained(
            self.spec.model_id, subfolder="vae", torch_dtype=torch.float32
        )
        pipeline = diffusers.WanImageToVideoPipeline.from_pretrained(
            self.spec.model_id, vae=vae, torch_dtype=torch.bfloat16
        )
        place_pipeline(pipeline, self.context.device.kind, options.cpu_offload)
        self._pipeline = pipeline

    def unload(self) -> None:
        self._pipeline = None
        release_memory()

    def generate(self, request: VideoRequest) -> None:
        options = self.parse_options(WanOptions)
        torch = require("torch", self.spec.install_hint)
        pipeline = self._pipeline
        multiple = pipeline.vae_scale_factor_spatial * pipeline.transformer.config.patch_size[1]
        size = fit_area(request.target, options.max_area, int(multiple))
        frames = pipeline(
            image=self._load_image(request.image_path, (size.width, size.height)),
            prompt=motion_prompt(request.prompt, request.camera),
            negative_prompt=options.negative_prompt,
            height=size.height,
            width=size.width,
            num_frames=snap_frame_count(request.duration_s, options.native_fps, options.max_frames),
            num_inference_steps=options.steps,
            guidance_scale=options.guidance_scale,
            generator=seeded_generator(torch, request.seed),
            output_type="pil",
        ).frames[0]
        self._encode(frames, request.output_path, options)

    def _load_image(self, path: Path, size: tuple[int, int]) -> Any:
        """Open the still and centre-crop it to the exact generation size."""
        image_module = require("PIL.Image", self.spec.install_hint)
        ops = require("PIL.ImageOps", self.spec.install_hint)
        return ops.fit(image_module.open(path).convert("RGB"), size)

    def _encode(self, frames: Any, output: Path, options: WanOptions) -> None:
        """Write frames to a temp directory and let ffmpeg turn them into the clip."""
        with tempfile.TemporaryDirectory(dir=output.parent, prefix=".frames-") as directory:
            for index, frame in enumerate(frames):
                frame.save(Path(directory) / (_FRAME_PATTERN % index))
            spec = FramesSpec(
                pattern=Path(directory) / _FRAME_PATTERN,
                output=output,
                fps=options.native_fps,
                encode=options,
            )
            command = frames_command(self.context.ffmpeg_bin, spec)
            self.context.commands.run(command.args, cwd=command.cwd)
