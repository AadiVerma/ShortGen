"""The on-disk layout of a project and the only code that builds paths into it."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Collection
from pathlib import Path

from shortgen.errors import PathValidationError

MANIFEST_NAME = "manifest.json"
SCRIPT_NAME = "script.md"
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp"})
_PARTIAL_TAG = ".partial"


class Workspace:
    """A project directory. Relative paths are POSIX strings stored in the manifest."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    @property
    def manifest_path(self) -> Path:
        """The single source of truth for the project."""
        return self.root / MANIFEST_NAME

    @property
    def script_path(self) -> Path:
        """A readable copy of the script; the manifest holds the authoritative text."""
        return self.root / SCRIPT_NAME

    def resolve(self, relative: str | Path) -> Path:
        """Absolute path for a project-relative one, refusing to leave the project."""
        return resolve_inside(self.root, relative)

    @staticmethod
    def audio(scene_id: str) -> str:
        """Narration audio of a scene."""
        return f"audio/{scene_id}.wav"

    @staticmethod
    def words(scene_id: str) -> str:
        """Word timings the speech model produced itself, if any."""
        return f"audio/{scene_id}.words.json"

    @staticmethod
    def alignment(scene_id: str) -> str:
        """Script words timed against the scene's audio."""
        return f"alignment/{scene_id}.json"

    @staticmethod
    def image(scene_id: str) -> str:
        """The scene's illustration."""
        return f"images/{scene_id}.png"

    @staticmethod
    def clip(scene_id: str) -> str:
        """The scene's animated clip, before it is fitted to the narration."""
        return f"clips/{scene_id}.mp4"

    @staticmethod
    def segment(scene_id: str) -> str:
        """The scene's rendered segment. QuickTime holds PCM audio, so lengths are sample-exact."""
        return f"build/{scene_id}.mov"

    subtitles_ass = "subtitles/captions.ass"
    subtitles_srt = "subtitles/captions.srt"
    subtitles_cues = "subtitles/captions.json"
    final_video = "output/final.mp4"


def resolve_inside(root: Path, candidate: str | Path) -> Path:
    """Resolve ``candidate`` against ``root`` and reject anything outside it."""
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise PathValidationError(f"path escapes the project directory: {candidate}")
    return resolved


def validate_input_file(path: Path, allowed_suffixes: Collection[str]) -> Path:
    """Check a user-supplied input is an existing regular file of an expected type."""
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise PathValidationError(f"not a file: {path}")
    if resolved.suffix.lower() not in allowed_suffixes:
        expected = ", ".join(sorted(allowed_suffixes))
        raise PathValidationError(f"unsupported file type for {path} (expected one of {expected})")
    return resolved


def partial_path(final: Path) -> Path:
    """Sibling temp path that keeps the extension (ffmpeg picks formats by it)."""
    return final.with_name(f"{final.stem}{_PARTIAL_TAG}{final.suffix}")


def atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` so readers see either the old file or the new one, never half.

    The data is flushed and fsynced to a temp file in the same directory, then
    renamed over the target; a crash or Ctrl+C at any point leaves the previous
    file intact.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temp = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temp.replace(path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
