"""Fake providers that write tiny, valid placeholder files.

They make the whole pipeline runnable in seconds without a GPU or a download:
in CI, in ``make smoke``, and when developing a new stage.
"""

from __future__ import annotations

import hashlib
import math
import struct
import textwrap
import zlib
from array import array
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pydantic import Field

from shortgen.adapters.heuristic_planner import HeuristicPlanner
from shortgen.domain.models import FrozenModel
from shortgen.domain.timing import count_words, proportional_word_timings
from shortgen.media import write_wav
from shortgen.providers.interfaces import (
    ImageGenerator,
    ImageRequest,
    SpeechRequest,
    SpeechResult,
    SpeechSynthesizer,
    VideoGenerator,
    VideoRequest,
)
from shortgen.render.layout import Size, fit_area

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_TONE_AMPLITUDE = 2000  # quiet: about 6% of 16-bit full scale
_RGB_BYTES = 3
_LABEL_MIN_PX = 10
_LABEL_WIDTH_DIVISOR = 22
_LABEL_CHARS = 26


class FakePlanner(HeuristicPlanner):
    """Heuristic planning plus a canned script, so topic projects are testable."""

    def write_script(self, topic: str, target_words: int, language: str) -> str:
        return (
            f"This is a short story about {topic}. It begins on a quiet morning.\n\n"
            f"Something unexpected happens, and everything about {topic} changes.\n\n"
            "By nightfall, the world feels a little different. The end."
        )


class FakeSpeechOptions(FrozenModel):
    """How long the placeholder tone is and whether it pretends to have word timings."""

    sample_rate: int = Field(gt=0)
    seconds_per_word: float = Field(gt=0)
    tone_hz: float = Field(gt=0)
    native_timings: bool


class FakeSpeech(SpeechSynthesizer):
    """Writes a quiet tone whose length is proportional to the word count."""

    def synthesize(self, request: SpeechRequest) -> SpeechResult:
        options = self.parse_options(FakeSpeechOptions)
        duration = count_words(request.text) * options.seconds_per_word / request.speed
        step = 2 * math.pi * options.tone_hz / options.sample_rate
        samples = array(
            "h",
            (
                int(_TONE_AMPLITUDE * math.sin(step * n))
                for n in range(int(duration * options.sample_rate))
            ),
        )
        write_wav(request.output_path, samples.tobytes(), options.sample_rate)
        words = proportional_word_timings(request.text, duration) if options.native_timings else []
        return SpeechResult(words=words)


class FakeImageOptions(FrozenModel):
    """Size budget of the placeholder image."""

    max_area: int = Field(gt=0)
    size_multiple: int = Field(gt=0)


class FakeImage(ImageGenerator):
    """Writes a vertical colour gradient that depends on the prompt and the seed."""

    def generate(self, request: ImageRequest) -> None:
        options = self.parse_options(FakeImageOptions)
        size = fit_area(request.target, options.max_area, options.size_multiple)
        digest = hashlib.sha256(f"{request.seed}:{request.prompt}".encode()).digest()
        write_gradient_png(request.output_path, size, digest[:_RGB_BYTES], digest[-_RGB_BYTES:])
        label_placeholder(request.output_path, f"PLACEHOLDER {request.scene_id}: {request.prompt}")


class FakeVideo(VideoGenerator):
    """Writes a small text placeholder. Not playable: pair it with a recording runner."""

    def generate(self, request: VideoRequest) -> None:
        summary = f"fake clip: {request.duration_s:.3f}s of {request.image_path.name}\n"
        request.output_path.write_text(summary, encoding="utf-8")


def label_placeholder(path: Path, text: str) -> None:
    """Write the prompt onto the placeholder so nobody mistakes it for model output."""
    with Image.open(path) as source:
        image = source.convert("RGB")
    font_px = max(_LABEL_MIN_PX, image.width // _LABEL_WIDTH_DIVISOR)
    font = ImageFont.load_default(size=font_px)
    wrapped = textwrap.fill(text, width=_LABEL_CHARS)
    ImageDraw.Draw(image).multiline_text(
        (image.width / 2, image.height / 3),
        wrapped,
        font=font,
        fill="#FFFFFF",
        stroke_width=max(1, font_px // 12),
        stroke_fill="#000000",
        anchor="mm",
        align="center",
    )
    image.save(path, format="PNG")


def write_gradient_png(path: Path, size: Size, top: bytes, bottom: bytes) -> None:
    """Encode an RGB top-to-bottom gradient as a PNG using only the standard library."""
    rows = bytearray()
    last_row = max(1, size.height - 1)
    for y in range(size.height):
        blend = y / last_row
        pixel = bytes(round(a + (b - a) * blend) for a, b in zip(top, bottom, strict=True))
        rows += b"\x00" + pixel * size.width  # filter type 0, then the scanline
    header = struct.pack(">IIBBBBB", size.width, size.height, 8, 2, 0, 0, 0)  # 8-bit RGB
    data = (
        _PNG_SIGNATURE
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(bytes(rows)))
        + _png_chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _png_chunk(tag: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(tag + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", checksum)
