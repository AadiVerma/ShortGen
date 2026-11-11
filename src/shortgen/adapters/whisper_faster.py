"""faster-whisper (CTranslate2) transcriber, MIT.

API used, as read from faster-whisper 1.2.1: ``WhisperModel(size, device,
compute_type)`` and ``model.transcribe(path, language, beam_size,
word_timestamps=True)`` returning ``(segments, info)`` where each segment has
``.words`` with ``.word``, ``.start`` and ``.end``.

CTranslate2 has no MPS backend, so on Apple Silicon this runs on the CPU;
use the ``mlx_whisper`` entry there for GPU speed.

Not yet run against real weights by the shortgen authors; see docs/hardware.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field

from shortgen.adapters._torch import release_memory, require
from shortgen.device import DeviceKind
from shortgen.domain.models import FrozenModel, WordTiming
from shortgen.providers.interfaces import Transcriber


class FasterWhisperOptions(FrozenModel):
    """Model size and per-device precision; values come from the registry entry."""

    model: str = Field(min_length=1)
    beam_size: int = Field(ge=1)
    cuda_compute_type: str
    cpu_compute_type: str


class FasterWhisperTranscriber(Transcriber):
    """Word-level timestamps from Whisper via CTranslate2."""

    _model: Any = None

    def load(self) -> None:
        options = self.parse_options(FasterWhisperOptions)
        faster_whisper = require("faster_whisper", self.spec.install_hint)
        on_cuda = self.context.device.kind is DeviceKind.CUDA
        self._model = faster_whisper.WhisperModel(
            options.model,
            device="cuda" if on_cuda else "cpu",
            compute_type=options.cuda_compute_type if on_cuda else options.cpu_compute_type,
        )

    def unload(self) -> None:
        self._model = None
        release_memory()

    def transcribe(self, audio_path: Path, language: str) -> list[WordTiming]:
        options = self.parse_options(FasterWhisperOptions)
        segments, _info = self._model.transcribe(
            str(audio_path),
            language=language,
            beam_size=options.beam_size,
            word_timestamps=True,
        )
        return [
            WordTiming(word=word.word.strip(), start=word.start, end=max(word.start, word.end))
            for segment in segments
            for word in (segment.words or [])
            if word.word.strip()
        ]
