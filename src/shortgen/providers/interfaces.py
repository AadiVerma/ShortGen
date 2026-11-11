"""Provider interfaces: the only thing pipeline stages know about models.

Every provider has the same lifecycle. ``load()`` is called lazily, once,
right before the first request of a stage; ``unload()`` is called when the
stage ends. The runner never keeps two providers loaded at the same time,
which is what keeps peak memory at one model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, TypeVar

from pydantic import BaseModel, ValidationError

from shortgen.config import format_validation_error
from shortgen.device import DeviceInfo
from shortgen.domain.models import CameraMove, Character, FrozenModel, Scene, WordTiming
from shortgen.domain.scene_planner import PlanningRules
from shortgen.errors import ConfigError
from shortgen.process import CommandRunner
from shortgen.providers.health import HealthCheck, default_health
from shortgen.providers.spec import ModelSpec, ProviderKind
from shortgen.render.layout import Size

OptionsT = TypeVar("OptionsT", bound=BaseModel)


@dataclass(frozen=True)
class ProviderContext:
    """Runtime facts handed to every adapter instead of global state."""

    device: DeviceInfo
    commands: CommandRunner
    ffmpeg_bin: str
    project_dir: Path = Path()
    #: The process environment (plus ``./.env``). Only hosted adapters read it, for credentials.
    env: Mapping[str, str] = field(default_factory=dict)


class PlanRequest(FrozenModel):
    """A script to cut into scenes."""

    script: str
    language: str
    rules: PlanningRules
    characters: list[Character]
    seed: int


class SpeechRequest(FrozenModel):
    """Narration to synthesise into ``output_path`` as 16-bit PCM WAV."""

    text: str
    voice: str
    speed: float
    language: str
    output_path: Path


class SpeechResult(FrozenModel):
    """Word timings the speech model produced itself; empty if it has none."""

    words: list[WordTiming]


class ImageRequest(FrozenModel):
    """One illustration. ``target`` is the final frame; adapters pick a native size."""

    scene_id: str
    prompt: str
    negative_prompt: str
    target: Size
    seed: int
    reference_images: list[Path]
    output_path: Path


class VideoRequest(FrozenModel):
    """Animate ``image_path`` into a clip of roughly ``duration_s`` seconds."""

    image_path: Path
    prompt: str
    camera: CameraMove
    duration_s: float
    fps: int
    target: Size
    seed: int
    output_path: Path


class Provider(ABC):
    """Base class for all adapters: construction is cheap, ``load()`` is not."""

    kind: ClassVar[ProviderKind]

    def __init__(
        self, spec: ModelSpec, options: Mapping[str, Any], context: ProviderContext
    ) -> None:
        self.spec = spec
        self.options = dict(options)
        self.context = context

    @property
    def name(self) -> str:
        """The registry name this provider was created from."""
        return self.spec.name

    def parse_options(self, schema: type[OptionsT]) -> OptionsT:
        """Validate this provider's merged options against the adapter's schema."""
        try:
            return schema.model_validate(self.options)
        except ValidationError as exc:
            raise ConfigError(format_validation_error(f"options for '{self.name}'", exc)) from exc

    def load(self) -> None:  # noqa: B027 - optional hook, stateless adapters need nothing
        """Load weights onto the device. Called once before a stage's first request."""

    def unload(self) -> None:  # noqa: B027 - optional hook
        """Drop every large object so the next stage starts with free memory."""

    def health(self) -> list[HealthCheck]:
        """Checks for ``shortgen doctor``. Must not load the model."""
        return default_health(self.spec, self.context.device)


class ScriptPlanner(Provider):
    """Turns a script into scenes, and optionally a topic into a script."""

    kind = ProviderKind.PLANNER

    @abstractmethod
    def write_script(self, topic: str, target_words: int, language: str) -> str:
        """Write a narration script about ``topic``."""

    @abstractmethod
    def plan(self, request: PlanRequest) -> list[Scene]:
        """Split the script into scenes, each with one shot."""


class SpeechSynthesizer(Provider):
    """Text to speech."""

    kind = ProviderKind.SPEECH

    @abstractmethod
    def synthesize(self, request: SpeechRequest) -> SpeechResult:
        """Write the narration audio and return native word timings, if any."""


class Transcriber(Provider):
    """Speech to timed words, used to align narration when TTS gives no timings."""

    kind = ProviderKind.TRANSCRIBER

    @abstractmethod
    def transcribe(self, audio_path: Path, language: str) -> list[WordTiming]:
        """Return the words heard in ``audio_path`` with their timings."""


class ImageGenerator(Provider):
    """Text (and optional reference images) to one still image."""

    kind = ProviderKind.IMAGE

    @abstractmethod
    def generate(self, request: ImageRequest) -> None:
        """Write a PNG to ``request.output_path``."""

    def source_fingerprint(self, scene_id: str) -> str | None:
        """Identity of any external input this provider reads for a scene.

        Models generate from the request alone and return ``None``. A provider
        that reads a file per scene returns its hash, so that replacing the
        file invalidates exactly that scene. Must not load the model.
        """
        return None


class VideoGenerator(Provider):
    """Image to a short video clip."""

    kind = ProviderKind.VIDEO

    @abstractmethod
    def generate(self, request: VideoRequest) -> None:
        """Write an MP4 to ``request.output_path``."""
