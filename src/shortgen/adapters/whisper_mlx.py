"""mlx-whisper transcriber for Apple Silicon, MIT.

API used, as read from mlx-whisper 0.4.3: ``mlx_whisper.transcribe(path,
path_or_hf_repo=..., word_timestamps=True, language=...)`` returning a dict
whose ``segments`` each carry ``words`` with ``word``, ``start`` and ``end``.

Not yet run on a Mac by the shortgen authors; see docs/hardware.md.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from shortgen.adapters._torch import release_memory, require
from shortgen.domain.models import WordTiming
from shortgen.providers.interfaces import Transcriber

_HOLDER_MODULE = "mlx_whisper.transcribe"


class MlxWhisperTranscriber(Transcriber):
    """Word-level timestamps from Whisper on MLX."""

    _module: Any = None

    def load(self) -> None:
        # mlx_whisper loads weights itself on the first transcribe() call and
        # caches them, so there is nothing to load eagerly.
        self._module = require("mlx_whisper", self.spec.install_hint)

    def unload(self) -> None:
        # mlx_whisper has no public unload. Version 0.4.3 keeps the model on a
        # module-level ``ModelHolder``; clearing it is the only way to free the
        # weights before the next stage. If that private detail changes, this
        # becomes a no-op and the weights stay loaded until the process exits.
        holder = getattr(sys.modules.get(_HOLDER_MODULE), "ModelHolder", None)
        if holder is not None:
            holder.model = None
            holder.model_path = None
        self._module = None
        release_memory()

    def transcribe(self, audio_path: Path, language: str) -> list[WordTiming]:
        result = self._module.transcribe(
            str(audio_path),
            path_or_hf_repo=self.spec.model_id,
            word_timestamps=True,
            language=language,
        )
        return [
            WordTiming(
                word=str(word["word"]).strip(),
                start=float(word["start"]),
                end=max(float(word["start"]), float(word["end"])),
            )
            for segment in result.get("segments", [])
            for word in segment.get("words", [])
            if str(word["word"]).strip()
        ]
