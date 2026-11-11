"""The ONNX Kokoro adapter, phoneme-to-word pairing and the bring-your-own-images provider."""

from __future__ import annotations

import hashlib
import io
import sys
import urllib.error
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from shortgen.adapters.folder_image import FolderImage
from shortgen.adapters.kokoro_onnx_speech import (
    KokoroOnnxSpeech,
    WeightFile,
    ensure_weight_file,
    words_from_phonemes,
)
from shortgen.config import default_config_dir
from shortgen.domain.timing import SpokenSpan, assign_spans
from shortgen.errors import ProviderError, ProviderUnavailableError
from shortgen.media import wav_duration
from shortgen.providers.health import HealthStatus
from shortgen.providers.interfaces import ImageRequest, ProviderContext, SpeechRequest
from shortgen.providers.spec import ProviderKind
from shortgen.registry import ModelRegistry
from shortgen.render.layout import Size
from tests.helpers import RecordingRunner
from tests.unit.test_registry import CPU

REGISTRY = ModelRegistry.from_files(default_config_dir() / "models.yaml")


def phonemes(units: str, step: float = 0.1) -> list[Any]:
    """Timings for a phoneme string: every character lasts ``step`` seconds."""
    return [
        SimpleNamespace(phoneme=char, start=round(i * step, 3), end=round((i + 1) * step, 3))
        for i, char in enumerate(units)
    ]


def spans(*sizes: int) -> list[SpokenSpan]:
    """Back-to-back spans, one second per phoneme."""
    result, cursor = [], 0.0
    for size in sizes:
        result.append(SpokenSpan(start=cursor, end=cursor + size, size=size))
        cursor += size
    return result


# ------------------------------------------------- pairing spoken units with written words


def test_one_to_one_pairing_keeps_measured_times() -> None:
    words = assign_spans(["Meet", "Odysseus,"], spans(4, 8))
    assert [(w.word, w.start, w.end) for w in words] == [("Meet", 0, 4), ("Odysseus,", 4, 12)]


def test_fused_function_words_share_their_span_by_length() -> None:
    # espeak says "on the" as one unit: 3 pronounced units for 4 written words.
    words = assign_spans(["out", "on", "the", "water"], spans(3, 5, 5))
    assert [w.word for w in words] == ["out", "on", "the", "water"]
    assert (words[1].start, words[2].end) == (3, 8)  # "on" + "the" fill the fused span
    assert words[1].end == words[2].start == pytest.approx(5.0)  # split 2:3 by letters
    assert (words[3].start, words[3].end) == (8, 13)


def test_numbers_and_symbols_absorb_the_units_they_expand_into() -> None:
    # "In 2026, it" is spoken as: in / two / thousand / twenty / six / it
    words = assign_spans(["In", "2026,", "it"], spans(2, 3, 7, 6, 4, 2))
    assert [(w.word, w.start, w.end) for w in words] == [
        ("In", 0, 2),
        ("2026,", 2, 22),
        ("it", 22, 24),
    ]
    percent = assign_spans(["uses", "20%", "of"], spans(5, 6, 6, 2))
    assert (percent[1].start, percent[1].end) == (5, 17)  # "twenty" + "percent"


def test_impossible_pairings_return_nothing_rather_than_guess() -> None:
    assert assign_spans(["one"], spans(1, 1, 1, 1, 1, 1)) == []  # needs more than 4 units
    assert assign_spans(["a", "b", "c", "d"], spans(4)) == []  # more than 3 words in one unit
    assert assign_spans([], spans(1)) == []
    assert assign_spans(["a"], []) == []


def test_phoneme_timings_become_word_timings_skipping_punctuation() -> None:
    words = words_from_phonemes("Meet Odysseus, hero.", phonemes("miit adisies, hiro."))
    assert [w.word for w in words] == ["Meet", "Odysseus,", "hero."]
    assert (words[0].start, words[0].end) == (0.0, 0.4)
    assert words[1].end == pytest.approx(1.2)  # the comma is not part of the word
    assert words[2].start == pytest.approx(1.4)
    assert words_from_phonemes("anything", []) == []  # a model without duration output


# ------------------------------------------------------------------------ weight downloads


class Download:
    """Stands in for ``urllib.request.urlopen``."""

    def __init__(self, payload: bytes | Exception) -> None:
        self.payload = payload
        self.urls: list[str] = []

    def __call__(self, url: str) -> Any:
        self.urls.append(url)
        if isinstance(self.payload, Exception):
            raise self.payload
        body = io.BytesIO(self.payload)
        body.headers = {"Content-Length": str(len(self.payload))}  # type: ignore[attr-defined]
        return body


def weights(tmp_path: Path, payload: bytes) -> WeightFile:
    return WeightFile(
        path=str(tmp_path / "cache" / "model.onnx"),
        url="https://example.com/model.onnx",
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def test_missing_weights_are_downloaded_verified_and_cached(tmp_path: Path) -> None:
    spec = weights(tmp_path, b"weights")
    download = Download(b"weights")
    assert ensure_weight_file(spec, download).read_bytes() == b"weights"
    ensure_weight_file(spec, download)  # second call: already there
    assert download.urls == ["https://example.com/model.onnx"]
    assert [p.name for p in spec.location.parent.iterdir()] == ["model.onnx"]


def test_a_download_with_the_wrong_hash_is_refused(tmp_path: Path) -> None:
    spec = weights(tmp_path, b"weights")
    with pytest.raises(ProviderUnavailableError, match="pinned SHA-256"):
        ensure_weight_file(spec, Download(b"tampered"))
    assert not spec.location.exists()
    assert list(spec.location.parent.iterdir()) == []


def test_a_failed_download_says_where_to_put_the_file(tmp_path: Path) -> None:
    spec = weights(tmp_path, b"weights")
    with pytest.raises(ProviderUnavailableError, match=r"save it as .*model\.onnx"):
        ensure_weight_file(spec, Download(urllib.error.URLError("offline")))


def test_weight_urls_must_be_https_and_hashes_well_formed() -> None:
    with pytest.raises(ValueError, match="url"):
        WeightFile(path="m", url="http://example.com/m", sha256="0" * 64)
    with pytest.raises(ValueError, match="sha256"):
        WeightFile(path="m", url="https://example.com/m", sha256="abc")


# --------------------------------------------------------------------- the adapter itself


def onnx_speech(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kokoro: Any) -> KokoroOnnxSpeech:
    module = ModuleType("kokoro_onnx")
    module.__spec__ = ModuleSpec("kokoro_onnx", None)
    module.Kokoro = kokoro  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kokoro_onnx", module)
    files = {}
    for name in ("model", "voices"):
        path = tmp_path / f"{name}.bin"
        path.write_bytes(name.encode())
        digest = hashlib.sha256(name.encode()).hexdigest()
        files[name] = {"path": str(path), "url": f"https://example.com/{name}", "sha256": digest}
    context = ProviderContext(device=CPU, commands=RecordingRunner(), ffmpeg_bin="ffmpeg")
    provider = REGISTRY.create("kokoro_onnx", ProviderKind.SPEECH, context, files)
    assert isinstance(provider, KokoroOnnxSpeech)
    return provider


def test_onnx_adapter_writes_wav_and_returns_word_timings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    numpy = pytest.importorskip("numpy")
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    class Kokoro:
        def __init__(self, model_path: str, voices_path: str) -> None:
            calls.append(((model_path, voices_path), {}))

        def create_timed(self, text: str, **kwargs: Any) -> Any:
            calls.append(((text,), kwargs))
            return numpy.full(24000, 0.5, dtype="float32"), 24000, phonemes("helou wurld", 0.05)

    speech = onnx_speech(tmp_path, monkeypatch, Kokoro)
    speech.load()
    request = SpeechRequest(
        text="Hello world",
        voice="af_heart",
        speed=1.1,
        language="en",
        output_path=tmp_path / "a.wav",
    )
    result = speech.synthesize(request)
    assert calls[0][0] == (str(tmp_path / "model.bin"), str(tmp_path / "voices.bin"))
    assert calls[1] == (("Hello world",), {"voice": "af_heart", "speed": 1.1, "lang": "en-us"})
    assert wav_duration(request.output_path) == pytest.approx(1.0)
    assert [(w.word, w.start) for w in result.words] == [("Hello", 0.0), ("world", 0.3)]
    assert [check.status for check in speech.health()][-1] is HealthStatus.OK
    speech.unload()
    assert speech._kokoro is None


def test_onnx_adapter_rejects_silence_and_reports_missing_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    numpy = pytest.importorskip("numpy")

    class Silent:
        def __init__(self, *_: str) -> None: ...

        def create_timed(self, text: str, **_: Any) -> Any:
            return numpy.zeros(0, dtype="float32"), 24000, []

    speech = onnx_speech(tmp_path, monkeypatch, Silent)
    speech.load()
    request = SpeechRequest(
        text="...", voice="af_heart", speed=1.0, language="en", output_path=tmp_path / "a.wav"
    )
    with pytest.raises(ProviderError, match="produced no audio"):
        speech.synthesize(request)
    (tmp_path / "voices.bin").unlink()
    weights_check = speech.health()[-1]
    assert weights_check.status is HealthStatus.WARN
    assert "voices.bin" in weights_check.detail


# ------------------------------------------------------------------ bring your own images


def folder(project: Path) -> FolderImage:
    context = ProviderContext(
        device=CPU, commands=RecordingRunner(), ffmpeg_bin="ffmpeg", project_dir=project
    )
    provider = REGISTRY.create("folder", ProviderKind.IMAGE, context, {})
    assert isinstance(provider, FolderImage)
    return provider


def request(project: Path, scene_id: str) -> ImageRequest:
    return ImageRequest(
        scene_id=scene_id,
        prompt="ignored",
        negative_prompt="",
        target=Size(width=1080, height=1920),
        seed=1,
        reference_images=[],
        output_path=project / "out.partial.png",
    )


def test_folder_provider_converts_the_scene_picture_to_png(tmp_path: Path) -> None:
    (tmp_path / "art").mkdir()
    Image.new("RGBA", (40, 30), (255, 0, 0, 128)).save(tmp_path / "art" / "s002.webp")
    provider = folder(tmp_path)
    provider.generate(request(tmp_path, "s002"))
    with Image.open(tmp_path / "out.partial.png") as result:
        assert (result.format, result.mode, result.size) == ("PNG", "RGB", (40, 30))


def test_folder_provider_names_the_file_it_wanted(tmp_path: Path) -> None:
    with pytest.raises(ProviderError, match=r"no image for scene s009: expected .*art/s009\.png"):
        folder(tmp_path).generate(request(tmp_path, "s009"))


def test_replacing_a_picture_changes_its_fingerprint(tmp_path: Path) -> None:
    (tmp_path / "art").mkdir()
    provider = folder(tmp_path)
    assert provider.source_fingerprint("s001") is None
    Image.new("RGB", (8, 8), "red").save(tmp_path / "art" / "s001.png")
    first = provider.source_fingerprint("s001")
    Image.new("RGB", (8, 8), "blue").save(tmp_path / "art" / "s001.png")
    assert first is not None and provider.source_fingerprint("s001") != first
