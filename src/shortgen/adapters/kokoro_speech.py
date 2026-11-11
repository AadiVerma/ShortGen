"""Kokoro-82M text to speech (Apache-2.0) via the ``kokoro`` package.

API used, as read from kokoro 0.9.4's ``pipeline.py``:
``KPipeline(lang_code, repo_id, device)`` and ``pipeline(text, voice, speed)``,
which yields results with ``.audio`` (a torch tensor at 24 kHz) and, for
English, ``.tokens`` carrying ``start_ts``/``end_ts`` in seconds relative to
that result's audio.

Not yet run against real weights by the shortgen authors; see docs/hardware.md.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from shortgen.adapters._torch import release_memory, require
from shortgen.device import DeviceKind
from shortgen.domain.models import FrozenModel, WordTiming
from shortgen.errors import ProviderError
from shortgen.media import write_wav
from shortgen.providers.interfaces import SpeechRequest, SpeechResult, SpeechSynthesizer

_INT16_MAX = 32767


class KokoroOptions(FrozenModel):
    """Language, output sample rate and device preference; values come from the registry entry."""

    lang_code: str = Field(min_length=1)
    sample_rate: int = Field(gt=0)
    use_gpu: bool


class KokoroSpeech(SpeechSynthesizer):
    """Narration with native word timestamps (English)."""

    _pipeline: Any = None

    def load(self) -> None:
        options = self.parse_options(KokoroOptions)
        kokoro = require("kokoro", self.spec.install_hint)
        on_cuda = options.use_gpu and self.context.device.kind is DeviceKind.CUDA
        self._pipeline = kokoro.KPipeline(
            lang_code=options.lang_code,
            repo_id=self.spec.model_id,
            device="cuda" if on_cuda else "cpu",
        )

    def unload(self) -> None:
        self._pipeline = None
        release_memory()

    def synthesize(self, request: SpeechRequest) -> SpeechResult:
        options = self.parse_options(KokoroOptions)
        numpy = require("numpy", self.spec.install_hint)
        chunks: list[Any] = []
        words: list[WordTiming] = []
        offset = 0.0
        for result in self._pipeline(request.text, voice=request.voice, speed=request.speed):
            if result.audio is None:
                continue
            audio = result.audio.detach().cpu().numpy()
            words.extend(timed_words(result.tokens, offset))
            chunks.append(audio)
            offset += len(audio) / options.sample_rate
        if not chunks:
            raise ProviderError(f"Kokoro produced no audio for: {request.text[:60]!r}")
        samples = numpy.clip(numpy.concatenate(chunks), -1.0, 1.0)
        pcm = (samples * _INT16_MAX).astype("<i2").tobytes()
        write_wav(request.output_path, pcm, options.sample_rate)
        return SpeechResult(words=words)


def timed_words(tokens: Any, offset: float) -> list[WordTiming]:
    """Kokoro tokens with usable timestamps, shifted by ``offset`` seconds.

    Punctuation tokens and tokens without timestamps (non-English voices) are
    skipped; an empty result makes the pipeline fall back to a transcriber.
    """
    words: list[WordTiming] = []
    for token in tokens or []:
        text = str(getattr(token, "text", "")).strip()
        start, end = getattr(token, "start_ts", None), getattr(token, "end_ts", None)
        if start is None or end is None or not any(char.isalnum() for char in text):
            continue
        begin = offset + float(start)
        words.append(WordTiming(word=text, start=begin, end=max(begin, offset + float(end))))
    return words
