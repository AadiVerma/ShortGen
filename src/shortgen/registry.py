"""Config-driven model registry.

``configs/models.yaml`` maps a short name to an adapter class and the facts
about its weights. Adding a model is one adapter module plus one entry here;
the pipeline never changes.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from shortgen.config import (
    MODELS_FILE_NAME,
    AppConfig,
    deep_merge,
    default_config_dir,
    format_validation_error,
    read_yaml,
)
from shortgen.errors import ConfigError
from shortgen.providers.interfaces import (
    ImageGenerator,
    Provider,
    ProviderContext,
    ScriptPlanner,
    SpeechSynthesizer,
    Transcriber,
    VideoGenerator,
)
from shortgen.providers.spec import ModelSpec, ProviderFingerprint, ProviderKind

INTERFACES: dict[ProviderKind, type[Provider]] = {
    ProviderKind.PLANNER: ScriptPlanner,
    ProviderKind.SPEECH: SpeechSynthesizer,
    ProviderKind.TRANSCRIBER: Transcriber,
    ProviderKind.IMAGE: ImageGenerator,
    ProviderKind.VIDEO: VideoGenerator,
}
_MODELS_KEY = "models"


class ModelRegistry:
    """Looks up registry entries and builds adapters from them."""

    def __init__(self, specs: Mapping[str, ModelSpec]) -> None:
        self._specs = dict(specs)

    @classmethod
    def from_files(cls, *paths: Path) -> ModelRegistry:
        """Load one or more registry files; later files override earlier ones per key."""
        entries: dict[str, Any] = {}
        for path in paths:
            models = read_yaml(path).get(_MODELS_KEY, {})
            if not isinstance(models, dict):
                raise ConfigError(f"{path}: '{_MODELS_KEY}' must be a mapping of name to entry")
            entries = deep_merge(entries, models)
        specs: dict[str, ModelSpec] = {}
        for name, entry in entries.items():
            if not isinstance(entry, dict):
                raise ConfigError(f"registry entry '{name}' must be a mapping")
            try:
                specs[name] = ModelSpec.model_validate({"name": name, **entry})
            except ValidationError as exc:
                raise ConfigError(format_validation_error(f"registry entry '{name}'", exc)) from exc
        return cls(specs)

    def specs(self, kind: ProviderKind | None = None) -> list[ModelSpec]:
        """All entries, optionally only those of one kind, in file order."""
        return [spec for spec in self._specs.values() if kind is None or spec.kind is kind]

    def spec(self, name: str, kind: ProviderKind) -> ModelSpec:
        """The entry called ``name``, which must be of ``kind``."""
        spec = self._specs.get(name)
        if spec is None or spec.kind is not kind:
            available = ", ".join(s.name for s in self.specs(kind)) or "none"
            problem = "is not registered" if spec is None else f"is a {spec.kind.value} model"
            raise ConfigError(
                f"providers.{kind.value}: '{name}' {problem}; available {kind.value} models: "
                f"{available}"
            )
        return spec

    def fingerprint(
        self, name: str, kind: ProviderKind, overrides: Mapping[str, Any]
    ) -> ProviderFingerprint:
        """Identity of a configured provider, for stage input hashes."""
        spec = self.spec(name, kind)
        return ProviderFingerprint(
            name=spec.name,
            adapter=spec.adapter,
            model_id=spec.model_id,
            options=deep_merge(spec.options, overrides),
        )

    def create(
        self,
        name: str,
        kind: ProviderKind,
        context: ProviderContext,
        overrides: Mapping[str, Any],
    ) -> Provider:
        """Instantiate the adapter for ``name``. Cheap: weights load later, on demand."""
        spec = self.spec(name, kind)
        adapter_class = load_adapter_class(spec)
        return adapter_class(spec, deep_merge(spec.options, overrides), context)


def load_adapter_class(spec: ModelSpec) -> type[Provider]:
    """Import ``module:Class`` from a registry entry and check it fits its role."""
    module_name, _, class_name = spec.adapter.partition(":")
    try:
        adapter = getattr(importlib.import_module(module_name), class_name)
    except (ImportError, AttributeError) as exc:
        raise ConfigError(
            f"cannot import adapter '{spec.adapter}' for '{spec.name}': {exc}"
        ) from exc
    interface = INTERFACES[spec.kind]
    if not (isinstance(adapter, type) and issubclass(adapter, interface)):
        raise ConfigError(
            f"adapter '{spec.adapter}' for '{spec.name}' must subclass {interface.__name__}"
        )
    return adapter


def load_registry(config: AppConfig) -> ModelRegistry:
    """The shipped registry, plus the user's ``runtime.models_file`` if set."""
    paths = [default_config_dir() / MODELS_FILE_NAME]
    if config.runtime.models_file is not None:
        paths.append(config.runtime.models_file)
    return ModelRegistry.from_files(*paths)
