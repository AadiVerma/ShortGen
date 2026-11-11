"""Encoder settings shared by configuration and the ffmpeg command builders."""

from __future__ import annotations

from pydantic import Field

from shortgen.domain.models import FrozenModel

# Codec, preset and pixel-format names are plain identifiers. Rejecting anything
# else means a config value can never smuggle in an extra ffmpeg option.
_IDENTIFIER = r"^[A-Za-z0-9_]+$"


class VideoEncode(FrozenModel):
    """How a video stream is encoded."""

    video_codec: str = Field(pattern=_IDENTIFIER)
    preset: str = Field(pattern=_IDENTIFIER)
    crf: int = Field(ge=0, le=51)
    pixel_format: str = Field(pattern=_IDENTIFIER)


class EncodeSettings(VideoEncode):
    """How one ffmpeg output with video and audio is encoded."""

    audio_codec: str = Field(pattern=_IDENTIFIER)
    audio_bitrate: str | None = Field(default=None, pattern=r"^\d+[kM]?$")
    audio_sample_rate: int = Field(gt=0)
