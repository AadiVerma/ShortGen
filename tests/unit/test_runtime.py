"""Device detection, subprocess handling, media helpers, logging and the fake adapters."""

from __future__ import annotations

import json
import logging
import struct
import sys
import wave
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from shortgen import device as device_module
from shortgen.adapters.fake import write_gradient_png
from shortgen.device import DeviceInfo, DeviceKind, detect_device
from shortgen.errors import CommandError, ProviderError, RenderError
from shortgen.logging import JsonFormatter, configure_logging
from shortgen.media import wav_duration, write_wav
from shortgen.process import SubprocessRunner
from shortgen.providers.health import HealthStatus, default_health
from shortgen.providers.spec import ModelSpec, ProviderKind
from shortgen.render.layout import Size


def fake_torch(*, cuda: bool, mps: bool) -> Any:
    """The slice of the torch API that device detection touches."""
    return SimpleNamespace(
        __version__="9.9.9",
        cuda=SimpleNamespace(
            is_available=lambda: cuda,
            get_device_name=lambda _index: "Test GPU",
            get_device_properties=lambda _index: SimpleNamespace(total_memory=24 * 1024**3),
        ),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
    )


def use_torch(monkeypatch: pytest.MonkeyPatch, torch: Any | None) -> None:
    monkeypatch.setattr(device_module, "_import_torch", lambda: torch)


def test_cuda_is_preferred_over_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    use_torch(monkeypatch, fake_torch(cuda=True, mps=True))
    info = detect_device()
    assert (info.kind, info.name, info.accelerator_memory_gb) == (DeviceKind.CUDA, "Test GPU", 24.0)
    assert info.warnings == []


def test_mps_shares_system_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    use_torch(monkeypatch, fake_torch(cuda=False, mps=True))
    info = detect_device()
    assert info.kind is DeviceKind.MPS
    assert info.accelerator_memory_gb == info.total_memory_gb


def test_cpu_fallback_warns_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    use_torch(monkeypatch, fake_torch(cuda=False, mps=False))
    info = detect_device()
    assert info.kind is DeviceKind.CPU
    assert "falling back to CPU" in info.warnings[0]


def test_unavailable_explicit_device_falls_back_with_a_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_torch(monkeypatch, fake_torch(cuda=False, mps=True))
    info = detect_device("cuda")
    assert info.kind is DeviceKind.CPU
    assert "'cuda' is not available" in info.warnings[0]
    assert detect_device("cpu").warnings == []  # an explicit, satisfiable choice is silent


def test_missing_torch_is_reported_not_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    use_torch(monkeypatch, None)
    info = detect_device()
    assert info.kind is DeviceKind.CPU
    assert info.torch_version is None
    assert info.warnings == ["PyTorch is not installed; adapters that need it cannot run."]


def test_subprocess_runner_captures_output_and_raises_on_failure() -> None:
    runner = SubprocessRunner()
    assert runner.run([sys.executable, "-c", "print('hi')"]).stdout == "hi\n"
    with pytest.raises(CommandError) as error:
        runner.run([sys.executable, "-c", "import sys; sys.stderr.write('bad\\n'); sys.exit(3)"])
    assert error.value.returncode == 3
    assert "bad" in str(error.value)
    with pytest.raises(RenderError, match="executable not found"):
        runner.run(["definitely-not-a-real-binary-xyz"])


def test_subprocess_arguments_are_never_interpreted_by_a_shell(tmp_path: Path) -> None:
    canary = tmp_path / "canary"
    payload = f"; touch {canary}"
    result = SubprocessRunner().run(
        [sys.executable, "-c", "import sys; print(sys.argv[1])", payload]
    )
    assert result.stdout.strip() == payload
    assert not canary.exists()


def test_wav_round_trip_and_validation(tmp_path: Path) -> None:
    path = tmp_path / "a.wav"
    write_wav(path, struct.pack("<4h", 0, 1, 2, 3) * 2000, sample_rate=8000)
    assert wav_duration(path) == pytest.approx(1.0)
    empty = tmp_path / "empty.wav"
    write_wav(empty, b"", sample_rate=8000)
    with pytest.raises(ProviderError, match="empty"):
        wav_duration(empty)
    (tmp_path / "junk.wav").write_bytes(b"not audio")
    with pytest.raises(ProviderError, match="not a readable PCM WAV"):
        wav_duration(tmp_path / "junk.wav")


def test_fake_image_is_a_real_png(tmp_path: Path) -> None:
    path = tmp_path / "g.png"
    write_gradient_png(path, Size(width=32, height=48), b"\xff\x00\x00", b"\x00\x00\xff")
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert struct.unpack(">II", data[16:24]) == (32, 48)
    assert data[-8:-4] == b"IEND"


def test_fake_speech_is_a_real_wav(tmp_path: Path) -> None:
    from shortgen.adapters.fake import FakeSpeech
    from shortgen.providers.interfaces import SpeechRequest
    from tests.unit.test_registry import CONTEXT

    spec = ModelSpec(
        name="fake_speech",
        kind=ProviderKind.SPEECH,
        adapter="shortgen.adapters.fake:FakeSpeech",
        license="none",
        commercial_use=True,
    )
    options = {"sample_rate": 8000, "seconds_per_word": 0.5, "tone_hz": 220, "native_timings": True}
    request = SpeechRequest(
        text="four short words here",
        voice="v",
        speed=2.0,
        language="en",
        output_path=tmp_path / "s.wav",
    )
    result = FakeSpeech(spec, options, CONTEXT).synthesize(request)
    with wave.open(str(request.output_path)) as audio:
        assert (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) == (1, 2, 8000)
    assert wav_duration(request.output_path) == pytest.approx(1.0)  # 4 words * 0.5 s / speed 2
    assert [word.word for word in result.words] == ["four", "short", "words", "here"]


def test_health_flags_missing_modules_and_small_devices() -> None:
    spec = ModelSpec(
        name="big",
        kind=ProviderKind.VIDEO,
        adapter="x.y:Z",
        license="Apache-2.0",
        commercial_use=True,
        requires=["json", "surely_not_installed_module"],
        install_hint="pip install it",
        devices=[DeviceKind.CUDA],
        min_memory_gb=24,
    )
    from tests.unit.test_registry import CPU

    checks = {check.name: check for check in default_health(spec, CPU)}
    assert checks["big: dependencies"].status is HealthStatus.FAIL
    assert "surely_not_installed_module" in checks["big: dependencies"].detail
    assert "json" not in checks["big: dependencies"].detail
    assert checks["big: dependencies"].hint == "pip install it"
    assert checks["big: device"].status is HealthStatus.WARN
    assert "24 GB" in checks["big: memory"].detail


def test_cpu_capable_models_are_not_flagged_on_other_devices() -> None:
    spec = ModelSpec(
        name="small",
        kind=ProviderKind.SPEECH,
        adapter="x.y:Z",
        license="Apache-2.0",
        commercial_use=True,
        devices=[DeviceKind.CUDA, DeviceKind.CPU],
    )
    mac = DeviceInfo(kind=DeviceKind.MPS, name="m4", total_memory_gb=16, available_memory_gb=9)
    assert default_health(spec, mac) == []


def test_json_logs_carry_extra_fields() -> None:
    record = logging.LogRecord("shortgen.x", logging.INFO, __file__, 1, "stage %s", ("plan",), None)
    record.stage = "plan"
    payload = json.loads(JsonFormatter().format(record))
    assert payload["message"] == "stage plan"
    assert payload["level"] == "INFO"
    assert payload["stage"] == "plan"


def test_configure_logging_is_idempotent() -> None:
    configure_logging("DEBUG", "json")
    configure_logging("WARNING", "console")
    logger = logging.getLogger("shortgen")
    assert len(logger.handlers) == 1
    assert logger.level == logging.WARNING
    logger.handlers.clear()
