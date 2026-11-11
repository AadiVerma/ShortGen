"""Configuration: layered YAML plus environment overrides, validated once.

Nothing here keeps global state. :func:`load_config` is a pure function of
the files and the environment mapping it is given.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from importlib import resources
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, ValidationError, model_validator

from shortgen.domain.models import AspectRatio, Character, FrozenModel
from shortgen.domain.scene_planner import PlanningRules
from shortgen.domain.subtitles import SubtitleRules, SubtitleStyle
from shortgen.domain.timing import TimingRules
from shortgen.errors import ConfigError
from shortgen.render.layout import Size
from shortgen.render.settings import EncodeSettings

ENV_PREFIX = "SHORTGEN__"
ENV_SEPARATOR = "__"
DEFAULT_CONFIG_NAME = "default.yaml"
MODELS_FILE_NAME = "models.yaml"
PROJECT_CONFIG_NAME = "config.yaml"
DOTENV_NAME = ".env"


class ProjectSettings(FrozenModel):
    """Creative defaults that apply to the whole video."""

    aspect: AspectRatio
    fps: int = Field(ge=1, le=120)
    seed: int = Field(ge=0)
    language: str = Field(min_length=2)
    style: str
    negative_prompt: str


class ProviderSelection(FrozenModel):
    """Which registry entry fills each provider role."""

    planner: str
    speech: str
    transcriber: str
    image: str
    video: str


class SpeechSettings(FrozenModel):
    """Voice selection, interpreted by the speech adapter."""

    voice: str = Field(min_length=1)
    speed: float = Field(gt=0, le=3)


class SubtitleSettings(FrozenModel):
    """Whether captions are burned in, and how."""

    enabled: bool
    rules: SubtitleRules
    style: SubtitleStyle


class RenderSettings(FrozenModel):
    """ffmpeg binaries, output sizes and encoder settings."""

    ffmpeg_bin: str = Field(min_length=1)
    ffprobe_bin: str = Field(min_length=1)
    resolutions: dict[AspectRatio, Size]
    max_slowdown: float = Field(ge=1)
    intermediate: EncodeSettings
    final: EncodeSettings

    @model_validator(mode="after")
    def _all_aspects(self) -> RenderSettings:
        missing = [aspect.value for aspect in AspectRatio if aspect not in self.resolutions]
        if missing:
            raise ValueError(f"missing resolution for aspect ratio(s): {', '.join(missing)}")
        return self


class RuntimeSettings(FrozenModel):
    """Device choice, retry policy and logging."""

    device: Literal["auto", "cuda", "mps", "cpu"]
    max_attempts: int = Field(ge=1)
    backoff_base_s: float = Field(ge=0)
    backoff_max_s: float = Field(ge=0)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"]
    log_format: Literal["console", "json"]
    models_file: Path | None = None


class AppConfig(FrozenModel):
    """The fully resolved configuration for one invocation."""

    project: ProjectSettings
    providers: ProviderSelection
    provider_options: dict[str, dict[str, Any]] = Field(default_factory=dict)
    characters: list[Character] = Field(default_factory=list)
    planning: PlanningRules
    speech: SpeechSettings
    timing: TimingRules
    subtitles: SubtitleSettings
    render: RenderSettings
    runtime: RuntimeSettings

    @property
    def resolution(self) -> Size:
        """Output frame size for the configured aspect ratio."""
        return self.render.resolutions[self.project.aspect]


def default_config_dir() -> Path:
    """Directory holding ``default.yaml`` and ``models.yaml``.

    Installed wheels carry them as package data; a source checkout reads the
    repository's ``configs/`` directory so there is a single copy to edit.
    """
    packaged = resources.files("shortgen").joinpath("_configs")
    if packaged.is_dir():
        return Path(str(packaged))
    checkout = Path(__file__).resolve().parents[2] / "configs"
    if checkout.is_dir():
        return checkout
    raise ConfigError("default configuration files are missing from this installation")


def read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML mapping, with errors that name the file."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"cannot read config file {path}: {exc.strerror}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a YAML mapping at the top level")
    return data


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Return ``base`` with ``override`` applied; nested mappings merge, the rest replace."""
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def env_overrides(env: Mapping[str, str]) -> dict[str, Any]:
    """Turn ``SHORTGEN__SECTION__KEY=value`` variables into a nested mapping.

    Values are parsed as YAML scalars, so ``20`` is an int and ``true`` a bool.
    """
    overrides: dict[str, Any] = {}
    for name, raw in env.items():
        if not name.startswith(ENV_PREFIX):
            continue
        keys = [part.lower() for part in name[len(ENV_PREFIX) :].split(ENV_SEPARATOR) if part]
        if not keys:
            continue
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            raise ConfigError(f"environment variable {name} is not a valid YAML value") from exc
        node = overrides
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = value
    return overrides


def read_dotenv(path: Path) -> dict[str, str]:
    """Parse a minimal ``.env`` file (``KEY=value`` lines, ``#`` comments)."""
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        values[key.strip().removeprefix("export ").strip()] = value.strip().strip("'\"")
    return values


def process_environment(cwd: Path | None = None) -> dict[str, str]:
    """The real environment, with ``./.env`` as a lower-priority fallback."""
    return {**read_dotenv((cwd or Path.cwd()) / DOTENV_NAME), **os.environ}


def load_config(
    *,
    project_dir: Path | None = None,
    config_file: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> AppConfig:
    """Resolve and validate configuration from all layers.

    Raises :class:`ConfigError` listing every invalid field at once.
    """
    data = read_yaml(default_config_dir() / DEFAULT_CONFIG_NAME)
    project_config = project_dir / PROJECT_CONFIG_NAME if project_dir else None
    for layer in (project_config, config_file):
        if layer is None:
            continue
        if layer is config_file and not layer.is_file():
            raise ConfigError(f"config file not found: {layer}")
        if layer.is_file():
            data = deep_merge(data, read_yaml(layer))
    data = deep_merge(data, env_overrides(env if env is not None else {}))
    try:
        return AppConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(format_validation_error("configuration", exc)) from exc


def format_validation_error(subject: str, error: ValidationError) -> str:
    """One line per invalid field: ``render.final.crf: Input should be ...``."""
    lines = [
        f"  {'.'.join(str(part) for part in issue['loc']) or '<root>'}: {issue['msg']}"
        for issue in error.errors()
    ]
    return f"invalid {subject}:\n" + "\n".join(lines)
