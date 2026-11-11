"""Image-to-video on Higgsfield: OPTIONAL, HOSTED and PAID. Off by default.

Unlike every default in shortgen, this does not run on your machine. Each
scene's picture is uploaded to Higgsfield together with its motion prompt, and
each clip spends credits on your account (video costs far more than images).

NOT VERIFIED against the live service; see ``shortgen.adapters._higgsfield``.
This adapter is the least certain part of shortgen: the model ids, the argument
names (``image_url``, ``prompt``, ``duration``) and the ``video.url`` result
field come from examples in Higgsfield's API documentation, not from anything
that was run. The argument names are options so they can be corrected in
configuration.

Hosted models make clips of a few fixed lengths. The adapter asks for the
shortest one that covers the scene; if none does, the compose stage slows the
clip slightly and holds its last frame, as it does for every video model.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import Field

from shortgen.adapters._higgsfield import (
    HiggsfieldJobs,
    JobOptions,
    credentials_check,
    video_url,
)
from shortgen.adapters._motion import motion_prompt
from shortgen.errors import ProviderError
from shortgen.pipeline.hashing import file_sha256, stable_hash
from shortgen.providers.health import HealthCheck, check_imports
from shortgen.providers.interfaces import VideoGenerator, VideoRequest

log = logging.getLogger(__name__)


class HiggsfieldVideoOptions(JobOptions):
    """Clip settings and argument names; values come from the registry entry."""

    durations_s: list[int] = Field(min_length=1, description="clip lengths the model offers")
    image_argument: str = Field(min_length=1)
    duration_argument: str | None


def pick_duration(wanted_s: float, offered_s: list[int]) -> int:
    """The shortest offered clip that covers the scene, else the longest one."""
    covering = [duration for duration in sorted(offered_s) if duration >= wanted_s]
    return covering[0] if covering else max(offered_s)


class HiggsfieldVideo(VideoGenerator):
    """One hosted, paid clip per scene."""

    _jobs: HiggsfieldJobs | None = None

    def load(self) -> None:
        options = self.parse_options(HiggsfieldVideoOptions)
        self._jobs = HiggsfieldJobs(self.spec, options, self.context.env)
        log.warning(
            "%s is a hosted, paid service: pictures and prompts are sent to %s and every "
            "clip costs credits",
            self.name,
            self.spec.source,
        )

    def unload(self) -> None:
        self._jobs = None

    def generate(self, request: VideoRequest) -> None:
        options = self.parse_options(HiggsfieldVideoOptions)
        if self._jobs is None:
            raise ProviderError(f"{self.name} was used before load()")
        jobs = self._jobs
        prompt = motion_prompt(request.prompt, request.camera)
        duration = pick_duration(request.duration_s, options.durations_s)
        key = stable_hash(
            [
                self.spec.model_id,
                file_sha256(request.image_path),
                prompt,
                duration,
                options.model_dump(mode="json"),
            ]
        )

        def arguments() -> dict[str, Any]:
            built: dict[str, Any] = {
                options.image_argument: jobs.upload(request.image_path),
                "prompt": prompt,
            }
            if options.duration_argument:
                built[options.duration_argument] = duration
            return built

        result = jobs.run(key, arguments)
        jobs.download(video_url(result), request.output_path)
        jobs.finish(key)

    def health(self) -> list[HealthCheck]:
        return [*check_imports(self.spec), credentials_check(self.name, self.context.env)]
