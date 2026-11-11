"""Text-to-image on Higgsfield: OPTIONAL, HOSTED and PAID. Off by default.

Unlike every default in shortgen, this does not run on your machine. Each
scene's prompt (and any character reference image, if you enable that) is sent
to Higgsfield, and each image spends credits on your account.

NOT VERIFIED against the live service; see ``shortgen.adapters._higgsfield``
for what was read from the client's source and what comes from examples only.
The arguments sent are ``prompt``, ``aspect_ratio`` and ``resolution``, as in
the client's README example for ``bytedance/seedream/v4/text-to-image``.

Not sent, because no documented argument for them was found: the negative
prompt and the seed. Add them under ``arguments`` in your configuration if the
model you choose documents them.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from PIL import Image, UnidentifiedImageError

from shortgen.adapters._higgsfield import (
    KEPT,
    HiggsfieldJobs,
    JobOptions,
    credentials_check,
    image_url,
)
from shortgen.errors import ProviderError
from shortgen.pipeline.hashing import file_sha256, stable_hash
from shortgen.providers.health import HealthCheck, check_imports
from shortgen.providers.interfaces import ImageGenerator, ImageRequest
from shortgen.render.layout import Size

log = logging.getLogger(__name__)

_DOWNLOAD_SUFFIX = ".download"


class HiggsfieldImageOptions(JobOptions):
    """Image settings; values come from the registry entry."""

    resolution: str | None
    reference_argument: str | None


def aspect_ratio(size: Size) -> str:
    """The frame's shape as the smallest whole-number ratio, e.g. ``9:16``."""
    divisor = math.gcd(size.width, size.height)
    return f"{size.width // divisor}:{size.height // divisor}"


class HiggsfieldImage(ImageGenerator):
    """One hosted, paid image per scene."""

    _jobs: HiggsfieldJobs | None = None

    def load(self) -> None:
        options = self.parse_options(HiggsfieldImageOptions)
        self._jobs = HiggsfieldJobs(self.spec, options, self.context.env)
        log.warning(
            "%s is a hosted, paid service: prompts are sent to %s and every image costs credits",
            self.name,
            self.spec.source,
        )

    def unload(self) -> None:
        self._jobs = None

    def generate(self, request: ImageRequest) -> None:
        options = self.parse_options(HiggsfieldImageOptions)
        jobs = self._ready()
        references = request.reference_images if options.reference_argument else []
        if request.reference_images and not references:
            log.warning("%s: reference images are ignored (reference_argument unset)", self.name)
        key = stable_hash(
            [
                self.spec.model_id,
                request.prompt,
                aspect_ratio(request.target),
                options.model_dump(mode="json"),
                [file_sha256(path) for path in references],
            ]
        )
        result = jobs.run(key, lambda: self._arguments(request, options, jobs))
        download = request.output_path.with_name(request.output_path.name + _DOWNLOAD_SUFFIX)
        jobs.download(image_url(result), download)
        try:
            with Image.open(download) as image:
                image.convert("RGB").save(request.output_path, format="PNG")
        except (UnidentifiedImageError, OSError) as exc:
            raise ProviderError(
                f"Higgsfield's output is not a readable image ({exc}). {KEPT}"
            ) from exc
        finally:
            download.unlink(missing_ok=True)
        jobs.finish(key)

    def health(self) -> list[HealthCheck]:
        return [*check_imports(self.spec), credentials_check(self.name, self.context.env)]

    def _ready(self) -> HiggsfieldJobs:
        if self._jobs is None:
            raise ProviderError(f"{self.name} was used before load()")
        return self._jobs

    def _arguments(
        self, request: ImageRequest, options: HiggsfieldImageOptions, jobs: HiggsfieldJobs
    ) -> dict[str, Any]:
        """The job's arguments; uploads the reference images when that is enabled."""
        arguments: dict[str, Any] = {
            "prompt": request.prompt,
            "aspect_ratio": aspect_ratio(request.target),
        }
        if options.resolution:
            arguments["resolution"] = options.resolution
        if options.reference_argument and request.reference_images:
            arguments[options.reference_argument] = [
                jobs.upload(path) for path in request.reference_images
            ]
        return arguments
