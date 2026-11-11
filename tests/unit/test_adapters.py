"""Adapter glue, exercised against stand-ins for the heavy libraries.

These tests prove the adapters call the libraries the way their docstrings
say and handle the results correctly. They do NOT prove the real models
work: that needs real weights and hardware (see docs/hardware.md).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from shortgen.adapters._motion import motion_prompt
from shortgen.adapters._torch import require
from shortgen.adapters.flux_image import FluxImage
from shortgen.adapters.kokoro_speech import KokoroSpeech, timed_words
from shortgen.adapters.sdxl_image import SdxlImage
from shortgen.adapters.wan_video import WanVideo, snap_frame_count
from shortgen.adapters.whisper_faster import FasterWhisperTranscriber
from shortgen.adapters.whisper_mlx import MlxWhisperTranscriber
from shortgen.config import default_config_dir
from shortgen.device import DeviceInfo, DeviceKind
from shortgen.domain.models import CameraMove, WordTiming
from shortgen.errors import ProviderError, ProviderUnavailableError
from shortgen.media import wav_duration
from shortgen.providers.interfaces import (
    ImageRequest,
    Provider,
    ProviderContext,
    SpeechRequest,
    VideoRequest,
)
from shortgen.providers.spec import ProviderKind
from shortgen.registry import ModelRegistry
from shortgen.render.layout import Size
from tests.helpers import RecordingRunner

PORTRAIT = Size(width=1080, height=1920)


def device(kind: DeviceKind) -> DeviceInfo:
    return DeviceInfo(kind=kind, name="test", total_memory_gb=16, available_memory_gb=8)


def build(name: str, kind: ProviderKind, on: DeviceKind, runner: Any = None) -> Provider:
    """Create a shipped adapter exactly as the pipeline would, on a pretend device."""
    registry = ModelRegistry.from_files(default_config_dir() / "models.yaml")
    context = ProviderContext(
        device=device(on), commands=runner or RecordingRunner(), ffmpeg_bin="ffmpeg"
    )
    return registry.create(name, kind, context, {})


def install(monkeypatch: pytest.MonkeyPatch, name: str, **attributes: Any) -> ModuleType:
    """Put a stand-in module into ``sys.modules`` for the duration of a test."""
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


class Recorder:
    """Callable that remembers every call's keyword arguments."""

    def __init__(self, result: Any = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.args: list[tuple[Any, ...]] = []
        self._result = result

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.args.append(args)
        self.calls.append(kwargs)
        return self._result


class StubImage:
    """Stands in for a PIL image."""

    def save(self, path: Path, format: str | None = None) -> None:
        Path(path).write_bytes(b"image")

    def convert(self, _mode: str) -> StubImage:
        return self


class StubPipeline:
    """Stands in for a diffusers pipeline; records how it was configured and called."""

    def __init__(self, frames: int = 0) -> None:
        self.calls: list[dict[str, Any]] = []
        self.events: list[str] = []
        self._frames = frames
        self.vae = SimpleNamespace(enable_tiling=lambda: self.events.append("vae tiling"))
        self.vae_scale_factor_spatial = 16
        self.transformer = SimpleNamespace(config=SimpleNamespace(patch_size=(1, 2, 2)))

    def to(self, target: str) -> StubPipeline:
        self.events.append(f"to {target}")
        return self

    def enable_model_cpu_offload(self) -> None:
        self.events.append("cpu offload")

    def enable_attention_slicing(self) -> None:
        self.events.append("attention slicing")

    def load_ip_adapter(self, repo: str, subfolder: str, weight_name: str) -> None:
        self.events.append(f"load ip {repo}/{subfolder}/{weight_name}")

    def set_ip_adapter_scale(self, scale: float) -> None:
        self.events.append(f"ip scale {scale}")

    def unload_ip_adapter(self) -> None:
        self.events.append("unload ip")

    def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self._frames:
            return SimpleNamespace(frames=[[StubImage() for _ in range(kwargs["num_frames"])]])
        return SimpleNamespace(images=[StubImage()])


def fake_torch(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    generator = SimpleNamespace(manual_seed=lambda seed: f"generator({seed})")
    return install(
        monkeypatch,
        "torch",
        float16="float16",
        float32="float32",
        bfloat16="bfloat16",
        Generator=lambda device: generator,
        cuda=SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
    )


def image_request(tmp_path: Path, references: list[Path] | None = None) -> ImageRequest:
    return ImageRequest(
        scene_id="s001",
        prompt="a lighthouse",
        negative_prompt="blurry",
        target=PORTRAIT,
        seed=7,
        reference_images=references or [],
        output_path=tmp_path / "out.partial.png",
    )


def test_missing_dependency_names_the_install_command() -> None:
    with pytest.raises(ProviderUnavailableError, match=r"'surely_missing_xyz'.*pip install x"):
        require("surely_missing_xyz", "pip install x")


# ---------------------------------------------------------------- Kokoro


def test_kokoro_tokens_become_offset_word_timings() -> None:
    tokens = [
        SimpleNamespace(text="Hello", start_ts=0.1, end_ts=0.4),
        SimpleNamespace(text=",", start_ts=0.4, end_ts=0.45),  # punctuation is skipped
        SimpleNamespace(text="world", start_ts=None, end_ts=None),  # untimed is skipped
        SimpleNamespace(text=" again ", start_ts=0.5, end_ts=0.45),  # reversed is clamped
    ]
    assert timed_words(tokens, offset=2.0) == [
        WordTiming(word="Hello", start=2.1, end=2.4),
        WordTiming(word="again", start=2.5, end=2.5),
    ]
    assert timed_words(None, 0.0) == []


def test_kokoro_concatenates_chunks_and_offsets_later_timestamps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    numpy = pytest.importorskip("numpy")

    def chunk(seconds: float, word: str) -> Any:
        samples = numpy.full(int(24000 * seconds), 0.25, dtype="float32")
        tensor = SimpleNamespace(
            detach=lambda: SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: samples))
        )
        token = SimpleNamespace(text=word, start_ts=0.1, end_ts=0.3)
        return SimpleNamespace(audio=tensor, tokens=[token])

    pipeline = Recorder(result=[chunk(1.0, "One"), SimpleNamespace(audio=None), chunk(0.5, "two")])
    constructor = Recorder(result=pipeline)
    install(monkeypatch, "kokoro", KPipeline=constructor)

    speech = build("kokoro", ProviderKind.SPEECH, DeviceKind.MPS)
    assert isinstance(speech, KokoroSpeech)
    speech.load()
    request = SpeechRequest(
        text="One two", voice="af_heart", speed=1.1, language="en", output_path=tmp_path / "a.wav"
    )
    result = speech.synthesize(request)

    assert constructor.calls == [
        {"lang_code": "a", "repo_id": "hexgrad/Kokoro-82M", "device": "cpu"}
    ]
    assert pipeline.args == [("One two",)]
    assert pipeline.calls == [{"voice": "af_heart", "speed": 1.1}]
    assert wav_duration(request.output_path) == pytest.approx(1.5)
    assert [(w.word, w.start) for w in result.words] == [("One", 0.1), ("two", 1.1)]
    speech.unload()
    assert speech._pipeline is None


def test_kokoro_uses_cuda_when_present_and_reports_silence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("numpy")
    constructor = Recorder(result=Recorder(result=[]))
    install(monkeypatch, "kokoro", KPipeline=constructor)
    speech = build("kokoro", ProviderKind.SPEECH, DeviceKind.CUDA)
    assert isinstance(speech, KokoroSpeech)
    speech.load()
    assert constructor.calls[0]["device"] == "cuda"
    request = SpeechRequest(
        text="...", voice="af_heart", speed=1.0, language="en", output_path=tmp_path / "a.wav"
    )
    with pytest.raises(ProviderError, match="produced no audio"):
        speech.synthesize(request)


# --------------------------------------------------------------- Whisper


def test_faster_whisper_picks_device_and_flattens_words(monkeypatch: pytest.MonkeyPatch) -> None:
    segments = [
        SimpleNamespace(words=[SimpleNamespace(word=" The", start=0.0, end=0.2)]),
        SimpleNamespace(words=None),
        SimpleNamespace(words=[SimpleNamespace(word=" ", start=0.2, end=0.2)]),
    ]
    model = SimpleNamespace(transcribe=Recorder(result=(iter(segments), object())))
    constructor = Recorder(result=model)
    install(monkeypatch, "faster_whisper", WhisperModel=constructor)

    transcriber = build("faster_whisper", ProviderKind.TRANSCRIBER, DeviceKind.MPS)
    assert isinstance(transcriber, FasterWhisperTranscriber)
    transcriber.load()
    assert constructor.args == [("small",)]
    assert constructor.calls == [{"device": "cpu", "compute_type": "int8"}]  # no MPS in CTranslate2
    words = transcriber.transcribe(Path("a.wav"), "en")
    assert words == [WordTiming(word="The", start=0.0, end=0.2)]
    assert model.transcribe.calls == [{"language": "en", "beam_size": 5, "word_timestamps": True}]

    on_gpu = build("faster_whisper", ProviderKind.TRANSCRIBER, DeviceKind.CUDA)
    on_gpu.load()
    assert constructor.calls[-1] == {"device": "cuda", "compute_type": "float16"}


def test_mlx_whisper_reads_word_dicts_and_clears_the_cached_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reply = {"segments": [{"words": [{"word": " Hi", "start": 0.0, "end": 0.3}]}, {}]}
    transcribe = Recorder(result=reply)
    install(monkeypatch, "mlx_whisper", transcribe=transcribe)
    holder = SimpleNamespace(model="weights", model_path="repo")
    install(monkeypatch, "mlx_whisper.transcribe", ModelHolder=holder)

    transcriber = build("mlx_whisper", ProviderKind.TRANSCRIBER, DeviceKind.MPS)
    assert isinstance(transcriber, MlxWhisperTranscriber)
    transcriber.load()
    assert transcriber.transcribe(Path("a.wav"), "en") == [
        WordTiming(word="Hi", start=0.0, end=0.3)
    ]
    assert transcribe.calls == [
        {
            "path_or_hf_repo": "mlx-community/whisper-large-v3-turbo",
            "word_timestamps": True,
            "language": "en",
        }
    ]
    transcriber.unload()
    assert holder.model is None


# ------------------------------------------------------------------ SDXL


def sdxl(monkeypatch: pytest.MonkeyPatch, on: DeviceKind) -> tuple[SdxlImage, StubPipeline, Any]:
    fake_torch(monkeypatch)
    pipeline = StubPipeline()
    loader = Recorder(result=pipeline)
    vae_loader = Recorder(result="vae")
    install(
        monkeypatch,
        "diffusers",
        StableDiffusionXLPipeline=SimpleNamespace(from_pretrained=loader),
        AutoencoderKL=SimpleNamespace(from_pretrained=vae_loader),
    )
    adapter = build("sdxl", ProviderKind.IMAGE, on)
    assert isinstance(adapter, SdxlImage)
    adapter.load()
    return adapter, pipeline, loader


def test_sdxl_loads_half_precision_with_the_fixed_vae_on_mps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter, pipeline, loader = sdxl(monkeypatch, DeviceKind.MPS)
    assert loader.args == [("stabilityai/stable-diffusion-xl-base-1.0",)]
    assert loader.calls == [
        {"torch_dtype": "float16", "use_safetensors": True, "variant": "fp16", "vae": "vae"}
    ]
    assert pipeline.events == ["to mps", "attention slicing", "vae tiling"]

    request = image_request(tmp_path)
    adapter.generate(request)
    call = pipeline.calls[0]
    assert (call["width"], call["height"]) == (768, 1344)
    assert (call["prompt"], call["negative_prompt"]) == ("a lighthouse", "blurry")
    assert (call["num_inference_steps"], call["guidance_scale"]) == (30, 6.5)
    assert call["generator"] == "generator(7)"
    assert "ip_adapter_image" not in call
    assert request.output_path.read_bytes() == b"image"


def test_sdxl_uses_full_precision_on_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    _, pipeline, loader = sdxl(monkeypatch, DeviceKind.CPU)
    assert loader.calls[0]["torch_dtype"] == "float32"
    assert "variant" not in loader.calls[0]
    assert pipeline.events == ["to cpu", "vae tiling"]


def test_sdxl_attaches_ip_adapter_only_for_scenes_with_references(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter, pipeline, _ = sdxl(monkeypatch, DeviceKind.CUDA)
    install(monkeypatch, "PIL")
    install(monkeypatch, "PIL.Image", open=lambda path: StubImage())
    references = [tmp_path / "mira.png", tmp_path / "boat.png"]

    adapter.generate(image_request(tmp_path, references))
    adapter.generate(image_request(tmp_path, references))
    adapter.generate(image_request(tmp_path))

    assert pipeline.events == [
        "to cuda",
        "vae tiling",
        "load ip h94/IP-Adapter/sdxl_models/ip-adapter_sdxl.bin",  # once, lazily
        "ip scale 0.6",
        "to cuda",  # the adapter's image encoder is placed like the rest
        "unload ip",  # the scene without a reference must not require one
    ]
    assert isinstance(pipeline.calls[0]["ip_adapter_image"], StubImage)
    assert "ip_adapter_image" not in pipeline.calls[2]


# ------------------------------------------------------------------ FLUX


def test_flux_schnell_call_shape_and_cpu_offload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_torch(monkeypatch)
    pipeline = StubPipeline()
    loader = Recorder(result=pipeline)
    install(monkeypatch, "diffusers", FluxPipeline=SimpleNamespace(from_pretrained=loader))
    adapter = build("flux_schnell", ProviderKind.IMAGE, DeviceKind.CUDA)
    assert isinstance(adapter, FluxImage)
    adapter.load()
    adapter.generate(image_request(tmp_path))

    assert loader.args == [("black-forest-labs/FLUX.1-schnell",)]
    assert loader.calls == [{"torch_dtype": "bfloat16"}]
    assert pipeline.events == ["cpu offload"]
    call = pipeline.calls[0]
    assert (call["num_inference_steps"], call["guidance_scale"]) == (4, 0.0)
    assert call["max_sequence_length"] == 256
    assert "negative_prompt" not in call  # schnell is guidance-distilled
    assert (call["width"] % 16, call["height"] % 16) == (0, 0)


# ------------------------------------------------------------------- Wan


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(5.0, 121), (3.0, 73), (2.0, 49), (0.05, 5), (60.0, 121)],
)
def test_wan_frame_counts_are_4k_plus_1_and_capped(seconds: float, expected: int) -> None:
    frames = snap_frame_count(seconds, fps=24, max_frames=121)
    assert frames == expected
    assert snap_frame_count(seconds, fps=24, max_frames=120) <= 117  # the cap is snapped too
    assert (frames - 1) % 4 == 0


def test_wan_motion_prompt_includes_the_camera_move() -> None:
    assert motion_prompt("Waves roll in.", CameraMove.ZOOM_IN) == (
        "Waves roll in, the camera slowly pushes in"
    )


def test_wan_generates_frames_and_encodes_them_with_ffmpeg(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_torch(monkeypatch)
    pipeline = StubPipeline(frames=1)
    loader = Recorder(result=pipeline)
    vae_loader = Recorder(result="wan-vae")
    install(
        monkeypatch,
        "diffusers",
        WanImageToVideoPipeline=SimpleNamespace(from_pretrained=loader),
        AutoencoderKLWan=SimpleNamespace(from_pretrained=vae_loader),
    )
    install(monkeypatch, "PIL")
    install(monkeypatch, "PIL.Image", open=lambda path: StubImage())
    fit = Recorder(result="fitted-image")
    install(monkeypatch, "PIL.ImageOps", fit=fit)
    runner = RecordingRunner()

    adapter = build("wan", ProviderKind.VIDEO, DeviceKind.CUDA, runner)
    assert isinstance(adapter, WanVideo)
    adapter.load()
    request = VideoRequest(
        image_path=tmp_path / "s001.png",
        prompt="gentle waves",
        camera=CameraMove.PAN_LEFT,
        duration_s=3.0,
        fps=30,
        target=PORTRAIT,
        seed=3,
        output_path=tmp_path / "s001.partial.mp4",
    )
    adapter.generate(request)

    model = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    assert vae_loader.args == [(model,)]
    assert vae_loader.calls == [{"subfolder": "vae", "torch_dtype": "float32"}]
    assert loader.calls == [{"vae": "wan-vae", "torch_dtype": "bfloat16"}]
    assert pipeline.events == ["cpu offload"]

    call = pipeline.calls[0]
    assert (call["width"], call["height"]) == (704, 1248)  # 9:16 within 704x1280, multiples of 32
    assert fit.args == [(fit.args[0][0], (704, 1248))]
    assert call["image"] == "fitted-image"
    assert call["num_frames"] == 73
    assert call["prompt"] == "gentle waves, the camera pans slowly to the left"
    assert call["output_type"] == "pil"

    (command,) = runner.commands
    assert command[command.index("-framerate") + 1] == "24"  # the model's native rate
    assert command[command.index("-i") + 1].endswith("%05d.png")
    assert command[-1] == str(request.output_path)
    assert not list(tmp_path.glob(".frames-*")), "temporary frames must be cleaned up"
