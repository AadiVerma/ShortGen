"""The data contracts every other module speaks.

These models are deliberately dumb: validation only, no behaviour that needs a
model, a GPU or a filesystem. They serialise to the project manifest as JSON.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Base for all contracts: unknown keys are errors, not silently dropped."""

    model_config = ConfigDict(extra="forbid")


class FrozenModel(StrictModel):
    """Base for value objects that must not change after creation."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class AspectRatio(StrEnum):
    """Supported output frame shapes."""

    PORTRAIT = "9:16"
    LANDSCAPE = "16:9"


class CameraMove(StrEnum):
    """A simple camera instruction every video adapter can interpret."""

    STATIC = "static"
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    PAN_LEFT = "pan_left"
    PAN_RIGHT = "pan_right"
    PAN_UP = "pan_up"
    PAN_DOWN = "pan_down"


class Character(FrozenModel):
    """A recurring character, optionally pinned to a reference image."""

    name: str = Field(min_length=1)
    description: str = ""
    reference_image: str | None = None


class Shot(FrozenModel):
    """What the camera shows for one scene and how it moves."""

    image_prompt: str = Field(min_length=1)
    motion_prompt: str
    camera: CameraMove
    #: What this one picture must not show. Empty means the project's ``negative_prompt``.
    negative_prompt: str = ""


class Scene(FrozenModel):
    """One narrated beat of the story with its single shot."""

    id: str = Field(pattern=r"^s\d{3,}$")
    index: int = Field(ge=0)
    narration: str = Field(min_length=1)
    shot: Shot
    characters: list[str] = Field(default_factory=list)


class Timing(FrozenModel):
    """A half-open interval ``[start, end)`` in seconds."""

    start: float = Field(ge=0)
    end: float = Field(ge=0)

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.end < self.start:
            raise ValueError(f"end ({self.end}) is before start ({self.start})")
        return self

    @property
    def duration(self) -> float:
        """Length of the interval in seconds."""
        return self.end - self.start


class WordTiming(Timing):
    """One spoken word and when it is heard, relative to its audio file."""

    word: str = Field(min_length=1)


class AssetKind(StrEnum):
    """Every kind of file a stage can produce."""

    AUDIO = "audio"
    WORDS = "words"
    ALIGNMENT = "alignment"
    IMAGE = "image"
    CLIP = "clip"
    SUBTITLES_ASS = "subtitles_ass"
    SUBTITLES_SRT = "subtitles_srt"
    SUBTITLES_CUES = "subtitles_cues"
    SEGMENT = "segment"
    VIDEO = "video"


class Asset(FrozenModel):
    """A file inside the project directory, identified by content hash."""

    kind: AssetKind
    path: str
    sha256: str
    scene_id: str | None = None


class JobStatus(StrEnum):
    """Lifecycle of one unit of work."""

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class Job(StrictModel):
    """The manifest record of one unit of work (one stage, optionally one scene)."""

    stage: str
    scene_id: str | None = None
    status: JobStatus = JobStatus.PENDING
    inputs_hash: str
    outputs: list[Asset] = Field(default_factory=list)
    attempts: int = 0
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None


class Project(StrictModel):
    """A script (or a topic to write one from) and the plan derived from it."""

    name: str = Field(min_length=1)
    script: str = ""
    topic: str | None = None
    created_at: datetime
    scenes: list[Scene] = Field(default_factory=list)

    @model_validator(mode="after")
    def _has_source(self) -> Self:
        if not self.script.strip() and not (self.topic or "").strip():
            raise ValueError("a project needs a script or a topic")
        return self


def scene_id(index: int) -> str:
    """Return the stable identifier for the scene at ``index`` (``0`` -> ``s001``)."""
    return f"s{index + 1:03d}"
