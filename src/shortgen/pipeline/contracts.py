"""Stage contracts: the typed inputs each stage hashes, and its JSON artefacts.

A unit of work is re-run exactly when the hash of its inputs model changes
(or an output file is missing). Upstream files appear here as content hashes,
so regenerating an image automatically invalidates its clip, and so on down.
"""

from __future__ import annotations

from shortgen.domain.models import CameraMove, Character, FrozenModel, WordTiming
from shortgen.domain.scene_planner import PlanningRules
from shortgen.domain.subtitles import SubtitleRules, SubtitleStyle
from shortgen.domain.timing import TimingRules
from shortgen.providers.spec import ProviderFingerprint
from shortgen.render.layout import Size
from shortgen.render.settings import EncodeSettings


class FileRef(FrozenModel):
    """A project file an upstream stage produced, pinned by content hash."""

    path: str
    sha256: str


class PlanInputs(FrozenModel):
    """What the scene plan depends on."""

    script: str
    topic: str | None
    language: str
    seed: int
    rules: PlanningRules
    characters: list[Character]
    provider: ProviderFingerprint


class SpeechInputs(FrozenModel):
    """What one scene's narration audio depends on."""

    text: str
    voice: str
    speed: float
    language: str
    provider: ProviderFingerprint


class AlignInputs(FrozenModel):
    """What one scene's word timings depend on."""

    narration: str
    audio: FileRef
    native_words: FileRef
    language: str
    provider: ProviderFingerprint


class ImageInputs(FrozenModel):
    """What one scene's illustration depends on."""

    prompt: str
    negative_prompt: str
    target: Size
    seed: int
    references: list[FileRef]
    source: str | None
    provider: ProviderFingerprint


class VideoInputs(FrozenModel):
    """What one scene's clip depends on."""

    image: FileRef
    prompt: str
    camera: CameraMove
    duration_s: float
    fps: int
    target: Size
    seed: int
    provider: ProviderFingerprint


class SubtitleInputs(FrozenModel):
    """What the caption files depend on."""

    alignments: list[FileRef]
    timing: TimingRules
    rules: SubtitleRules
    style: SubtitleStyle
    target: Size
    fps: int


class ComposeInputs(FrozenModel):
    """What one scene's rendered segment depends on."""

    clip: FileRef
    audio: FileRef
    duration_s: float
    audio_offset_s: float
    target: Size
    fps: int
    max_slowdown: float
    encode: EncodeSettings


class RenderInputs(FrozenModel):
    """What the final video depends on."""

    segments: list[FileRef]
    captions: FileRef | None
    caption_style: SubtitleStyle | None
    encode: EncodeSettings


class NativeWords(FrozenModel):
    """``audio/<scene>.words.json``: timings from the speech model, possibly empty."""

    words: list[WordTiming]


class SceneAlignment(FrozenModel):
    """``alignment/<scene>.json``: script words timed against the scene's audio."""

    duration_s: float
    source: str
    words: list[WordTiming]
