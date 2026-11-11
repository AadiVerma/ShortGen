from __future__ import annotations

from pathlib import Path

import pytest

from shortgen.config import (
    AppConfig,
    deep_merge,
    env_overrides,
    load_config,
    read_dotenv,
)
from shortgen.domain.models import AspectRatio
from shortgen.errors import ConfigError


def test_defaults_are_valid_and_free_by_default(config: AppConfig) -> None:
    assert config.project.aspect is AspectRatio.PORTRAIT
    assert (config.resolution.width, config.resolution.height) == (1080, 1920)
    assert config.providers.video == "kenburns"
    assert config.providers.planner == "heuristic"


def test_deep_merge_merges_mappings_and_replaces_scalars_and_lists() -> None:
    base = {"a": {"x": 1, "y": [1, 2]}, "b": 1}
    assert deep_merge(base, {"a": {"y": [3]}, "c": 2}) == {"a": {"x": 1, "y": [3]}, "b": 1, "c": 2}
    assert base["a"]["y"] == [1, 2]  # inputs are not mutated


def test_env_overrides_parse_nested_typed_values() -> None:
    env = {
        "SHORTGEN__RENDER__FINAL__CRF": "20",
        "SHORTGEN__SUBTITLES__ENABLED": "false",
        "SHORTGEN__PROVIDERS__VIDEO": "wan",
        "UNRELATED": "1",
    }
    assert env_overrides(env) == {
        "render": {"final": {"crf": 20}},
        "subtitles": {"enabled": False},
        "providers": {"video": "wan"},
    }


def test_layers_apply_in_order(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "config.yaml").write_text("project: {aspect: '16:9', fps: 24}\n")
    extra = tmp_path / "extra.yaml"
    extra.write_text("project: {fps: 25}\n")
    config = load_config(
        project_dir=project, config_file=extra, env={"SHORTGEN__PROJECT__SEED": "7"}
    )
    assert config.project.aspect is AspectRatio.LANDSCAPE  # from the project file
    assert config.project.fps == 25  # --config beats the project file
    assert config.project.seed == 7  # the environment beats both
    assert config.resolution.width == 1920


def test_invalid_values_are_reported_by_dotted_path() -> None:
    env = {"SHORTGEN__RENDER__FINAL__CRF": "99", "SHORTGEN__PROJECT__ASPECT": "4:3"}
    with pytest.raises(ConfigError) as error:
        load_config(env=env)
    message = str(error.value)
    assert "render.final.crf" in message
    assert "project.aspect" in message


def test_unknown_keys_are_errors_not_typos_waiting_to_happen() -> None:
    with pytest.raises(ConfigError, match=r"project\.aspekt"):
        load_config(env={"SHORTGEN__PROJECT__ASPEKT": "9:16"})


def test_codec_names_cannot_carry_ffmpeg_options() -> None:
    with pytest.raises(ConfigError, match=r"render\.final\.video_codec"):
        load_config(env={"SHORTGEN__RENDER__FINAL__VIDEO_CODEC": "'libx264 -f null'"})


def test_missing_explicit_config_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="config file not found"):
        load_config(config_file=tmp_path / "nope.yaml")


def test_malformed_yaml_names_the_file(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("project: [unclosed\n")
    with pytest.raises(ConfigError, match=r"bad\.yaml"):
        load_config(config_file=bad)


def test_dotenv_parsing(tmp_path: Path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("# comment\nexport SHORTGEN__PROJECT__FPS=24\nQUOTED='a b'\nnot a pair\n")
    assert read_dotenv(dotenv) == {"SHORTGEN__PROJECT__FPS": "24", "QUOTED": "a b"}
    assert read_dotenv(tmp_path / "missing") == {}
