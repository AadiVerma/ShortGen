"""Kokoro-82M text to speech on ONNX Runtime (no PyTorch), via ``kokoro-onnx``.

Verified by running it: on a 2-vCPU Linux container this synthesises speech
faster than real time and returns per-phoneme timings, which this adapter
folds into per-word timings for captions.

API used (kokoro-onnx 0.6.1): ``Kokoro(model_path, voices_path)`` and
``create_timed(text, voice, speed, lang)`` returning ``(audio, sample_rate,
timings)`` where each timing has ``phoneme``, ``start`` and ``end`` in seconds.
Phoneme timings are only produced by model files that expose a duration
output; the files in the project's ``model-files-v1.1`` release do, the older
``model-files-v1.0`` ones do not (the adapter then returns no word timings and
the align stage falls back to the configured transcriber).

The model weights are Apache-2.0 (hexgrad/Kokoro-82M); the ONNX export is
published by the kokoro-onnx project (MIT code) on GitHub releases. The two
files are downloaded on first use and checked against pinned SHA-256 hashes.
"""

from __future__ import annotations

import hashlib
import logging
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from pydantic import Field

from shortgen.adapters._torch import release_memory, require
from shortgen.domain.models import FrozenModel, WordTiming
from shortgen.domain.timing import SpokenSpan, assign_spans, script_tokens
from shortgen.errors import ProviderError, ProviderUnavailableError
from shortgen.media import write_wav
from shortgen.providers.health import HealthCheck, HealthStatus, check_imports
from shortgen.providers.interfaces import SpeechRequest, SpeechResult, SpeechSynthesizer

log = logging.getLogger(__name__)

_INT16_MAX = 32767
_CHUNK_BYTES = 1 << 20
_PROGRESS_STEP = 0.2
_PARTIAL_SUFFIX = ".partial"


class WeightFile(FrozenModel):
    """A file the adapter needs, where to get it and what it must hash to."""

    path: str = Field(min_length=1)
    url: str = Field(pattern=r"^https://")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @property
    def location(self) -> Path:
        """The local path with ``~`` expanded."""
        return Path(self.path).expanduser()


class KokoroOnnxOptions(FrozenModel):
    """Model files, phonemiser language and sample rate; values come from the registry entry."""

    model: WeightFile
    voices: WeightFile
    lang: str = Field(min_length=2)
    sample_rate: int = Field(gt=0)


class KokoroOnnxSpeech(SpeechSynthesizer):
    """Narration with word timings, on the CPU, without PyTorch."""

    _kokoro: Any = None

    def load(self) -> None:
        options = self.parse_options(KokoroOnnxOptions)
        module = require("kokoro_onnx", self.spec.install_hint)
        for weights in (options.model, options.voices):
            ensure_weight_file(weights)
        self._kokoro = module.Kokoro(str(options.model.location), str(options.voices.location))

    def unload(self) -> None:
        self._kokoro = None
        release_memory()

    def synthesize(self, request: SpeechRequest) -> SpeechResult:
        options = self.parse_options(KokoroOnnxOptions)
        numpy = require("numpy", self.spec.install_hint)
        audio, sample_rate, spoken = self._kokoro.create_timed(
            request.text, voice=request.voice, speed=request.speed, lang=options.lang
        )
        if len(audio) == 0:
            raise ProviderError(f"Kokoro produced no audio for: {request.text[:60]!r}")
        if sample_rate != options.sample_rate:
            raise ProviderError(
                f"Kokoro returned {sample_rate} Hz audio, the registry says {options.sample_rate}"
            )
        pcm = (numpy.clip(audio, -1.0, 1.0) * _INT16_MAX).astype("<i2").tobytes()
        write_wav(request.output_path, pcm, sample_rate)
        return SpeechResult(words=words_from_phonemes(request.text, spoken))

    def health(self) -> list[HealthCheck]:
        options = self.parse_options(KokoroOnnxOptions)
        missing = [w for w in (options.model, options.voices) if not w.location.is_file()]
        weights = HealthCheck(
            name=f"{self.name}: weights",
            status=HealthStatus.WARN if missing else HealthStatus.OK,
            detail=f"not downloaded yet: {', '.join(w.location.name for w in missing)}"
            if missing
            else str(options.model.location.parent),
            hint="They download from GitHub on first use (about 350 MB)." if missing else None,
        )
        return [*check_imports(self.spec), weights]


def words_from_phonemes(text: str, spoken: Sequence[Any]) -> list[WordTiming]:
    """Turn per-phoneme timings into per-word timings for the script's words.

    Phonemes arrive in speaking order with spaces between pronounced units.
    Those units are not always the written words (espeak fuses "on the" and
    expands numbers), so they are paired up by ``assign_spans``.
    """
    spans: list[SpokenSpan] = []
    start, end, size = 0.0, 0.0, 0
    for timing in [*spoken, None]:
        phoneme = " " if timing is None else str(timing.phoneme)
        if phoneme.isspace():
            if size:
                spans.append(SpokenSpan(start=start, end=max(start, end), size=size))
            size = 0
        elif any(char.isalpha() for char in phoneme):  # skips punctuation marks
            if not size:
                start = float(timing.start)
            end = float(timing.end)
            size += 1
    return assign_spans(script_tokens(text), spans)


def ensure_weight_file(
    weights: WeightFile, opener: Callable[..., Any] = urllib.request.urlopen
) -> Path:
    """Return the local weight file, downloading and verifying it if it is missing.

    The download goes to a temp name and is renamed only after its SHA-256
    matches, so an interrupted or tampered download never becomes the model.
    """
    target = weights.location
    if target.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + _PARTIAL_SUFFIX)
    log.info("downloading %s to %s", weights.url, target)
    digest = hashlib.sha256()
    try:
        with opener(weights.url) as response, partial.open("wb") as handle:
            total = int(response.headers.get("Content-Length") or 0)
            received, reported = 0, 0.0
            while chunk := response.read(_CHUNK_BYTES):
                handle.write(chunk)
                digest.update(chunk)
                received += len(chunk)
                if total and received / total >= reported + _PROGRESS_STEP:
                    reported = received / total
                    log.info("  %s: %d%%", target.name, round(reported * 100))
    except (urllib.error.URLError, OSError) as exc:
        partial.unlink(missing_ok=True)
        raise ProviderUnavailableError(
            f"could not download {weights.url} ({exc}). Download it yourself and save it as "
            f"{target}"
        ) from exc
    if digest.hexdigest() != weights.sha256:
        partial.unlink(missing_ok=True)
        raise ProviderUnavailableError(
            f"{weights.url} does not match its pinned SHA-256; refusing to use it"
        )
    partial.replace(target)
    return target
