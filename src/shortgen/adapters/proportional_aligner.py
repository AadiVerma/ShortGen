"""A transcriber that hears nothing, for machines without a Whisper install."""

from __future__ import annotations

from pathlib import Path

from shortgen.domain.models import WordTiming
from shortgen.providers.interfaces import Transcriber


class ProportionalAligner(Transcriber):
    """Returns no words, so the align stage spreads the script by word length.

    That fallback lives in :func:`shortgen.domain.timing.align_to_script`; this
    class exists so "no recogniser" is an explicit, configurable choice.
    """

    def transcribe(self, audio_path: Path, language: str) -> list[WordTiming]:
        return []
