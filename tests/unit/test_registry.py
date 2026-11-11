from __future__ import annotations

from pathlib import Path

import pytest

from shortgen.adapters.fake import FakeImage, FakeImageOptions
from shortgen.config import AppConfig, default_config_dir, load_config
from shortgen.device import DeviceInfo, DeviceKind
from shortgen.errors import ConfigError
from shortgen.providers.interfaces import ProviderContext
from shortgen.providers.spec import ProviderKind
from shortgen.registry import INTERFACES, ModelRegistry, load_adapter_class, load_registry
from tests.helpers import RecordingRunner

CPU = DeviceInfo(kind=DeviceKind.CPU, name="test", total_memory_gb=16, available_memory_gb=8)
CONTEXT = ProviderContext(device=CPU, commands=RecordingRunner(), ffmpeg_bin="ffmpeg")


@pytest.fixture(scope="module")
def registry() -> ModelRegistry:
    return ModelRegistry.from_files(default_config_dir() / "models.yaml")


def test_every_shipped_entry_records_a_license_and_a_valid_adapter(registry: ModelRegistry) -> None:
    specs = registry.specs()
    assert {spec.kind for spec in specs} == set(ProviderKind)
    for spec in specs:
        assert spec.license
        assert issubclass(load_adapter_class(spec), INTERFACES[spec.kind])
        if spec.weights_host == "huggingface":
            assert spec.model_id and spec.source, f"{spec.name} must say where its weights are"


def test_default_providers_exist_and_allow_commercial_use(
    registry: ModelRegistry, config: AppConfig
) -> None:
    for kind in ProviderKind:
        spec = registry.spec(getattr(config.providers, kind.value), kind)
        assert spec.commercial_use, f"default {kind.value} model must not be non-commercial"


def test_no_default_provider_is_hosted(registry: ModelRegistry, config: AppConfig) -> None:
    for kind in ProviderKind:
        spec = registry.spec(getattr(config.providers, kind.value), kind)
        assert not spec.hosted, f"default {kind.value} model must run locally"


def test_hosted_entries_are_marked_and_claim_nothing_about_commercial_use(
    registry: ModelRegistry,
) -> None:
    hosted = [spec for spec in registry.specs() if spec.hosted]
    assert {spec.name for spec in hosted} == {"higgsfield_image", "higgsfield_video"}
    for spec in hosted:
        assert spec.commercial_use is None
        assert spec.source and spec.install_hint


def test_the_hosted_profile_changes_only_the_image_and_video_providers(config: AppConfig) -> None:
    profile = load_config(config_file=default_config_dir() / "higgsfield.yaml")
    assert profile.providers.image == "higgsfield_image"
    assert profile.providers.video == "higgsfield_video"
    for local in ("planner", "speech", "transcriber"):
        assert getattr(profile.providers, local) == getattr(config.providers, local)


def test_create_builds_the_adapter_with_merged_options(registry: ModelRegistry) -> None:
    provider = registry.create("fake_image", ProviderKind.IMAGE, CONTEXT, {"max_area": 64})
    assert isinstance(provider, FakeImage)
    assert provider.options == {"max_area": 64, "size_multiple": 16}
    fingerprint = registry.fingerprint("fake_image", ProviderKind.IMAGE, {"max_area": 64})
    assert fingerprint.options == provider.options


def test_unknown_or_wrong_kind_names_list_the_alternatives(registry: ModelRegistry) -> None:
    with pytest.raises(
        ConfigError, match=r"'nope' is not registered; available image models: sdxl"
    ):
        registry.spec("nope", ProviderKind.IMAGE)
    with pytest.raises(ConfigError, match="'kokoro' is a speech model"):
        registry.spec("kokoro", ProviderKind.IMAGE)


def test_invalid_options_are_reported_against_the_provider(registry: ModelRegistry) -> None:
    provider = registry.create("fake_image", ProviderKind.IMAGE, CONTEXT, {"max_area": "big"})
    with pytest.raises(ConfigError, match=r"options for 'fake_image'[\s\S]*max_area"):
        provider.parse_options(FakeImageOptions)


def test_user_registry_file_adds_and_overrides_entries(tmp_path: Path, config: AppConfig) -> None:
    extra = tmp_path / "models.yaml"
    extra.write_text(
        "models:\n"
        "  my_image:\n"
        "    kind: image\n"
        "    adapter: shortgen.adapters.fake:FakeImage\n"
        "    license: none\n"
        "    commercial_use: true\n"
        "  sdxl:\n"
        "    options: {steps: 12}\n"
    )
    custom = config.model_copy(
        update={"runtime": config.runtime.model_copy(update={"models_file": extra})}
    )
    registry = load_registry(custom)
    assert registry.spec("my_image", ProviderKind.IMAGE).adapter.endswith(":FakeImage")
    sdxl = registry.spec("sdxl", ProviderKind.IMAGE)
    assert sdxl.options["steps"] == 12
    assert sdxl.options["guidance_scale"] == 6.5  # untouched options survive the merge


@pytest.mark.parametrize(
    ("adapter", "message"),
    [
        ("shortgen.adapters.missing:Nope", "cannot import adapter"),
        ("shortgen.adapters.fake:Missing", "cannot import adapter"),
        ("shortgen.adapters.fake:FakeSpeech", "must subclass ImageGenerator"),
    ],
)
def test_bad_adapter_paths_fail_clearly(tmp_path: Path, adapter: str, message: str) -> None:
    path = tmp_path / "models.yaml"
    path.write_text(
        f"models:\n  broken:\n    kind: image\n    adapter: {adapter}\n"
        "    license: none\n    commercial_use: true\n"
    )
    registry = ModelRegistry.from_files(path)
    with pytest.raises(ConfigError, match=message):
        registry.create("broken", ProviderKind.IMAGE, CONTEXT, {})


def test_malformed_entries_are_config_errors(tmp_path: Path) -> None:
    path = tmp_path / "models.yaml"
    path.write_text("models:\n  x:\n    kind: image\n    adapter: not-a-path\n")
    with pytest.raises(ConfigError, match="registry entry 'x'"):
        ModelRegistry.from_files(path)
    path.write_text("models:\n  x: 3\n")
    with pytest.raises(ConfigError, match="must be a mapping"):
        ModelRegistry.from_files(path)
