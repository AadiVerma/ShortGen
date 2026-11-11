"""Small WAV helpers built on the standard library.

The pipeline's audio contract is 16-bit PCM WAV, which lets durations be
measured without ffprobe or any audio dependency.
"""

from __future__ import annotations

import wave
from pathlib import Path

from shortgen.errors import ProviderError

SAMPLE_WIDTH_BYTES = 2  # 16-bit PCM


def wav_duration(path: Path) -> float:
    """Length of a PCM WAV file in seconds."""
    try:
        with wave.open(str(path), "rb") as audio:
            frames, rate = audio.getnframes(), audio.getframerate()
    except (wave.Error, EOFError, OSError) as exc:
        raise ProviderError(f"not a readable PCM WAV file: {path} ({exc})") from exc
    if frames == 0 or rate == 0:
        raise ProviderError(f"audio file is empty: {path}")
    return frames / rate


def write_wav(path: Path, pcm: bytes, sample_rate: int, channels: int = 1) -> None:
    """Write raw little-endian 16-bit PCM samples as a WAV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(channels)
        audio.setsampwidth(SAMPLE_WIDTH_BYTES)
        audio.setframerate(sample_rate)
        audio.writeframes(pcm)
