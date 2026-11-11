"""Environment diagnosis: ffmpeg, device, memory and the configured models."""

from __future__ import annotations

import platform
import shutil
import sys
from collections.abc import Mapping

from shortgen.config import AppConfig
from shortgen.device import DeviceInfo, DeviceKind
from shortgen.errors import ShortgenError
from shortgen.process import CommandRunner
from shortgen.providers.health import HealthCheck, HealthStatus
from shortgen.providers.interfaces import ProviderContext
from shortgen.providers.spec import ModelSpec, ProviderKind
from shortgen.registry import ModelRegistry
from shortgen.render.captions import resolve_font_file
from shortgen.render.ffmpeg import REQUIRED_FILTERS

MIN_PYTHON = (3, 11)
_FILTER_NAME_COLUMN = 1


def run_doctor(
    config: AppConfig,
    registry: ModelRegistry,
    device: DeviceInfo,
    commands: CommandRunner,
    env: Mapping[str, str] | None = None,
) -> list[HealthCheck]:
    """Every check, in the order a user should fix them."""
    return [
        _python_check(),
        *_ffmpeg_checks(config, commands),
        *_caption_checks(config),
        _device_check(device),
        _memory_check(device),
        *_provider_checks(
            config,
            registry,
            ProviderContext(
                device=device,
                commands=commands,
                ffmpeg_bin=config.render.ffmpeg_bin,
                env=env or {},
            ),
        ),
    ]


def has_failures(checks: list[HealthCheck]) -> bool:
    """Whether anything is broken badly enough that the pipeline cannot run."""
    return any(check.status is HealthStatus.FAIL for check in checks)


def _python_check() -> HealthCheck:
    version = sys.version_info
    ok = (version.major, version.minor) >= MIN_PYTHON
    return HealthCheck(
        name="python",
        status=HealthStatus.OK if ok else HealthStatus.FAIL,
        detail=f"{platform.python_version()} on {platform.system()} {platform.machine()}",
        hint=None if ok else "shortgen needs Python 3.11 or newer.",
    )


def _ffmpeg_checks(config: AppConfig, commands: CommandRunner) -> list[HealthCheck]:
    render = config.render
    checks = [_binary_check("ffprobe", render.ffprobe_bin)]
    ffmpeg = _binary_check("ffmpeg", render.ffmpeg_bin)
    checks.insert(0, ffmpeg)
    if ffmpeg.status is not HealthStatus.OK:
        return checks
    try:
        filters = _listed_names(
            commands.run([render.ffmpeg_bin, "-hide_banner", "-filters"]).stdout
        )
        encoders = _listed_names(
            commands.run([render.ffmpeg_bin, "-hide_banner", "-encoders"]).stdout
        )
    except ShortgenError as exc:
        return [
            *checks,
            HealthCheck(name="ffmpeg: capabilities", status=HealthStatus.FAIL, detail=str(exc)),
        ]
    checks.append(_filter_check(filters))
    checks.append(_encoder_check(encoders, config))
    return checks


def _binary_check(name: str, binary: str) -> HealthCheck:
    path = shutil.which(binary)
    return HealthCheck(
        name=name,
        status=HealthStatus.OK if path else HealthStatus.FAIL,
        detail=path or f"'{binary}' not found on PATH",
        hint=None
        if path
        else "Install ffmpeg (macOS: brew install ffmpeg, Debian/Ubuntu: apt install ffmpeg).",
    )


def _listed_names(listing: str) -> set[str]:
    """Names from ``ffmpeg -filters`` / ``-encoders`` output (second column)."""
    names: set[str] = set()
    for line in listing.splitlines():
        columns = line.split()
        if len(columns) > _FILTER_NAME_COLUMN:
            names.add(columns[_FILTER_NAME_COLUMN])
    return names


def _filter_check(filters: set[str]) -> HealthCheck:
    missing = [name for name in REQUIRED_FILTERS if name not in filters]
    if not missing:
        detail = f"{', '.join(REQUIRED_FILTERS)} available"
        return HealthCheck(name="ffmpeg: filters", status=HealthStatus.OK, detail=detail)
    return HealthCheck(
        name="ffmpeg: filters",
        status=HealthStatus.FAIL,
        detail=f"missing filter(s): {', '.join(missing)}",
        hint="These are core ffmpeg filters; install a standard ffmpeg build.",
    )


def _caption_checks(config: AppConfig) -> list[HealthCheck]:
    """Say which font captions will be drawn with. Needs no ffmpeg text filter."""
    if not config.subtitles.enabled:
        return []
    style = config.subtitles.style
    font = resolve_font_file(style)
    if font is not None and font.is_file():
        return [HealthCheck(name="captions: font", status=HealthStatus.OK, detail=str(font))]
    wanted = style.font_path or ", ".join([style.font_name, *style.font_fallbacks])
    return [
        HealthCheck(
            name="captions: font",
            status=HealthStatus.FAIL if style.font_path else HealthStatus.WARN,
            detail=f"not found: {wanted}"
            + ("" if style.font_path else "; captions will use Pillow's built-in font"),
            hint="Set subtitles.style.font_name to an installed font, or font_path to a font file.",
        )
    ]


def _encoder_check(encoders: set[str], config: AppConfig) -> HealthCheck:
    settings = (config.render.intermediate, config.render.final)
    wanted = {s.video_codec for s in settings} | {s.audio_codec for s in settings}
    missing = sorted(wanted - encoders)
    return HealthCheck(
        name="ffmpeg: encoders",
        status=HealthStatus.FAIL if missing else HealthStatus.OK,
        detail=f"missing encoder(s): {', '.join(missing)}"
        if missing
        else f"{', '.join(sorted(wanted))} available",
        hint="Change render.*.video_codec/audio_codec or install a fuller ffmpeg."
        if missing
        else None,
    )


def _device_check(device: DeviceInfo) -> HealthCheck:
    torch = f"torch {device.torch_version}" if device.torch_version else "torch not installed"
    return HealthCheck(
        name="device",
        status=HealthStatus.WARN if device.warnings else HealthStatus.OK,
        detail=f"{device.kind.value}: {device.name} ({torch})",
        hint=" ".join(device.warnings) or None,
    )


def _memory_check(device: DeviceInfo) -> HealthCheck:
    detail = f"{device.total_memory_gb:g} GB total, {device.available_memory_gb:g} GB available"
    if device.kind is DeviceKind.CUDA and device.accelerator_memory_gb is not None:
        detail += f", {device.accelerator_memory_gb:g} GB VRAM"
    return HealthCheck(name="memory", status=HealthStatus.OK, detail=detail)


def _provider_checks(
    config: AppConfig, registry: ModelRegistry, context: ProviderContext
) -> list[HealthCheck]:
    checks: list[HealthCheck] = []
    for kind in ProviderKind:
        name = str(getattr(config.providers, kind.value))
        label = f"{kind.value}: {name}"
        try:
            spec = registry.spec(name, kind)
            provider = registry.create(name, kind, context, config.provider_options.get(name, {}))
            health = provider.health()
        except ShortgenError as exc:
            checks.append(HealthCheck(name=label, status=HealthStatus.FAIL, detail=str(exc)))
            continue
        checks.append(_license_check(label, spec))
        checks.extend(health)
    return checks


def _license_check(label: str, spec: ModelSpec) -> HealthCheck:
    """One line saying what the selected model is and what its terms allow."""
    if spec.hosted:
        return HealthCheck(
            name=label,
            status=HealthStatus.WARN,
            detail=f"{spec.model_id}: HOSTED, PAID service, not local ({spec.license})",
            hint=f"Prompts and images are sent to {spec.source}. Read its terms and pricing.",
        )
    usage = {
        True: "commercial use allowed",
        False: "NON-COMMERCIAL license",
        None: "commercial use not determined, read the license",
    }[spec.commercial_use]
    weights = f"{spec.model_id}, license {spec.license}, {usage}" if spec.model_id else None
    return HealthCheck(
        name=label,
        status=HealthStatus.OK if spec.commercial_use else HealthStatus.WARN,
        detail=weights or "no model weights needed",
    )
