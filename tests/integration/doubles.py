"""Misbehaving providers, registered through a user models file like any third-party adapter."""

from __future__ import annotations

from shortgen.adapters.fake import FakeImage, FakeSpeech
from shortgen.errors import ProviderError, ProviderUnavailableError, TransientError
from shortgen.providers.interfaces import ImageRequest, SpeechRequest, SpeechResult

#: Lifecycle events in order, e.g. ``["load tracked_speech", "unload tracked_speech", ...]``.
EVENTS: list[str] = []

#: How ``MoodyImage`` should misbehave: "", "fail", "interrupt" or "flaky".
mood = ""
_FLAKY_FAILURES = 2

MODELS_YAML = """
models:
  tracked_speech:
    kind: speech
    adapter: tests.integration.doubles:TrackedSpeech
    license: none
    commercial_use: true
    options: {sample_rate: 8000, seconds_per_word: 0.3, tone_hz: 220, native_timings: false}
  moody_image:
    kind: image
    adapter: tests.integration.doubles:MoodyImage
    license: none
    commercial_use: true
    options: {max_area: 4096, size_multiple: 16}
  missing_image:
    kind: image
    adapter: tests.integration.doubles:MissingImage
    license: none
    commercial_use: true
    options: {max_area: 4096, size_multiple: 16}
"""


class _Tracked:
    """Mixin recording load/unload so tests can assert on model lifecycle."""

    name: str

    def load(self) -> None:
        EVENTS.append(f"load {self.name}")

    def unload(self) -> None:
        EVENTS.append(f"unload {self.name}")


class TrackedSpeech(_Tracked, FakeSpeech):
    def synthesize(self, request: SpeechRequest) -> SpeechResult:
        EVENTS.append(f"use {self.name}")
        return super().synthesize(request)


class MoodyImage(_Tracked, FakeImage):
    """Misbehaves according to the module-level ``mood`` set by the test.

    ``fail``/``interrupt`` hit the scene whose prompt mentions "storm";
    ``flaky`` raises a transient error twice before succeeding.
    """

    _transient_failures = 0

    def generate(self, request: ImageRequest) -> None:
        stormy = "storm" in request.prompt
        if mood == "fail" and stormy:
            raise ProviderError("the storm scene cannot be drawn")
        if mood == "interrupt" and stormy:
            request.output_path.write_bytes(b"half an image")
            raise KeyboardInterrupt
        if mood == "flaky" and self._transient_failures < _FLAKY_FAILURES:
            self._transient_failures += 1
            raise TransientError("model server busy")
        super().generate(request)


class MissingImage(FakeImage):
    """An adapter whose dependency is not installed."""

    def load(self) -> None:
        raise ProviderUnavailableError("needs 'imaginary-lib'; pip install imaginary-lib")
