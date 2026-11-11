"""Bring-your-own images: one picture per scene, read from a folder.

Not a model. It lets you use illustrations from any source (your own drawings,
another tool) with the rest of the pipeline, and it makes a complete demo
possible with no image model at all.

Name each file after its scene id: ``s001.png``, ``s002.jpg``, ...
(`shortgen plan` then `shortgen status` shows how many scenes there are.)
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from shortgen.domain.models import FrozenModel
from shortgen.errors import ProviderError
from shortgen.pipeline.hashing import file_sha256
from shortgen.pipeline.workspace import IMAGE_SUFFIXES
from shortgen.providers.interfaces import ImageGenerator, ImageRequest


class FolderImageOptions(FrozenModel):
    """Where the per-scene images are; relative paths are inside the project directory."""

    directory: str


class FolderImage(ImageGenerator):
    """Uses ``<directory>/<scene id>.<png|jpg|jpeg|webp>`` as the scene's illustration."""

    def generate(self, request: ImageRequest) -> None:
        source = self._find(request.scene_id)
        if source is None:
            raise ProviderError(
                f"no image for scene {request.scene_id}: expected "
                f"{self._directory() / request.scene_id}.png (or .jpg, .jpeg, .webp)"
            )
        with Image.open(source) as image:
            image.convert("RGB").save(request.output_path, format="PNG")

    def source_fingerprint(self, scene_id: str) -> str | None:
        """Hash of the supplied file, so replacing a picture regenerates that scene."""
        source = self._find(scene_id)
        return file_sha256(source) if source is not None else None

    def _directory(self) -> Path:
        options = self.parse_options(FolderImageOptions)
        return self.context.project_dir / Path(options.directory).expanduser()

    def _find(self, scene_id: str) -> Path | None:
        for suffix in sorted(IMAGE_SUFFIXES):
            candidate = self._directory() / f"{scene_id}{suffix}"
            if candidate.is_file():
                return candidate
        return None
