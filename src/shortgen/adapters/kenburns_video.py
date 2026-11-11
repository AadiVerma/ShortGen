"""Ken Burns animation: a slow zoom or pan over the still, rendered by ffmpeg.

This is the default "video model" because it needs no weights and no GPU, so
the full pipeline works on any machine. Swap in a neural image-to-video
adapter (see ``wan_video.py``) by changing ``providers.video``.
"""

from __future__ import annotations

import shutil

from pydantic import Field

from shortgen.providers.health import HealthCheck, HealthStatus
from shortgen.providers.interfaces import VideoGenerator, VideoRequest
from shortgen.render.ffmpeg import KenBurnsSpec, kenburns_command
from shortgen.render.settings import VideoEncode


class KenBurnsOptions(VideoEncode):
    """Motion strength and clip encoding; values come from the registry entry."""

    zoom: float = Field(gt=1, le=2)
    supersample: int = Field(ge=1, le=4)


class KenBurnsVideo(VideoGenerator):
    """Turns the scene's camera move into an ffmpeg ``zoompan`` render."""

    def generate(self, request: VideoRequest) -> None:
        options = self.parse_options(KenBurnsOptions)
        spec = KenBurnsSpec(
            image=request.image_path,
            output=request.output_path,
            frames=max(1, round(request.duration_s * request.fps)),
            fps=request.fps,
            size=request.target,
            camera=request.camera,
            zoom=options.zoom,
            supersample=options.supersample,
            encode=options,
        )
        command = kenburns_command(self.context.ffmpeg_bin, spec)
        self.context.commands.run(command.args, cwd=command.cwd)

    def health(self) -> list[HealthCheck]:
        found = shutil.which(self.context.ffmpeg_bin)
        return [
            HealthCheck(
                name=f"{self.name}: ffmpeg",
                status=HealthStatus.OK if found else HealthStatus.FAIL,
                detail=found or f"'{self.context.ffmpeg_bin}' not found on PATH",
            )
        ]
