"""The CLI and the MCP server: thin wrappers that must agree with each other."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from shortgen import app, mcp_server
from shortgen.cli import EXIT_FAILURE, cli
from shortgen.doctor import has_failures
from shortgen.errors import ShortgenError
from shortgen.pipeline.stage import StageName
from shortgen.process import CommandResult
from shortgen.providers.health import HealthStatus
from tests.helpers import FAKE_ENV, STORY

runner = CliRunner()


@pytest.fixture(autouse=True)
def fake_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Fake models via real environment variables, and a clean working directory."""
    for key, value in {**FAKE_ENV, "SHORTGEN__PROVIDERS__VIDEO": "kenburns"}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.chdir(tmp_path)
    yield


@pytest.fixture
def script(tmp_path: Path) -> Path:
    path = tmp_path / "lighthouse.md"
    path.write_text(STORY)
    return path


def invoke(*args: str) -> Any:
    return runner.invoke(cli, list(args))


# ------------------------------------------------------------------- CLI


@pytest.mark.ffmpeg
def test_run_goes_from_script_to_video_in_one_command_and_resumes(script: Path) -> None:
    first = invoke("run", str(script))
    assert first.exit_code == 0, first.output
    video = Path("projects/lighthouse/output/final.mp4").resolve()
    assert video.is_file()
    assert f"Video: {video}" in first.output

    again = invoke("run", str(script))
    assert again.exit_code == 0
    assert "done," not in again.output  # nothing was regenerated

    script.write_text(STORY.replace("sea was calm", "sky was clear"))
    edited = invoke("run", str(script))
    assert "speech     1 done, 2 skipped" in edited.output  # only the edited scene


@pytest.mark.ffmpeg
def test_staged_commands_match_the_single_command(script: Path) -> None:
    assert invoke("init", "demo", "--script", str(script), "--aspect", "16:9").exit_code == 0
    assert invoke("plan", "demo").exit_code == 0
    assert invoke("generate", "demo").exit_code == 0

    status = json.loads(invoke("status", "demo", "--json").output)
    by_stage = {stage["stage"]: stage for stage in status["stages"]}
    assert by_stage["video"]["pending"] == 0
    assert by_stage["compose"]["pending"] == 3
    assert status["final_video"] is None

    assert invoke("render", "demo").exit_code == 0
    assert "│ render    │ 1/1" in invoke("status", "demo").output
    assert (
        invoke("generate", "demo", "--stage", "image", "--scene", "s002", "--force").exit_code == 0
    )
    assert "aspect: '16:9'" in Path("demo/config.yaml").read_text()


def test_expected_errors_are_one_line_and_exit_1(script: Path) -> None:
    missing = invoke("status", "nowhere")
    assert missing.exit_code == EXIT_FAILURE
    assert "error: no project at" in missing.output
    assert "Traceback" not in missing.output

    invoke("init", "demo", "--script", str(script))
    assert "already exists" in invoke("init", "demo", "--script", str(script)).output
    assert "exactly one of" in invoke("init", "other").output
    assert invoke("run").exit_code == EXIT_FAILURE


def test_bad_config_is_reported_before_any_command_runs(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("render: {final: {crf: 999}}\n")
    result = invoke("--config", str(bad), "doctor")
    assert result.exit_code == EXIT_FAILURE
    assert "render.final.crf" in result.output


def test_version_and_help() -> None:
    assert invoke("--version").output.startswith("shortgen ")
    listing = invoke("--help").output
    for command in ("init", "plan", "generate", "render", "run", "status", "doctor"):
        assert command in listing


# ---------------------------------------------------------------- doctor


class FfmpegWithout:
    """Pretends to be an ffmpeg build lacking some filters."""

    def __init__(self, missing: set[str]) -> None:
        self._missing = missing

    def run(self, args: list[str], *, cwd: Path | None = None) -> CommandResult:
        names = ["overlay", "zoompan", "tpad", "apad", "adelay", "libx264", "aac", "pcm_s16le"]
        lines = [f" ... {name} V->V description" for name in names if name not in self._missing]
        return CommandResult(stdout="\n".join(lines), stderr="")


def doctor_checks(env: dict[str, str], commands: Any = None) -> dict[str, Any]:
    checks = app.doctor(env={**FAKE_ENV, **env}, commands=commands or FfmpegWithout(set()))
    return {check.name: check for check in checks}


@pytest.mark.ffmpeg
def test_doctor_passes_with_free_local_providers() -> None:
    checks = doctor_checks({"SHORTGEN__PROVIDERS__VIDEO": "kenburns"})
    assert not has_failures(list(checks.values()))
    assert checks["ffmpeg: filters"].status is HealthStatus.OK
    assert checks["video: kenburns"].detail == "no model weights needed"
    assert "GB total" in checks["memory"].detail


@pytest.mark.ffmpeg
def test_doctor_accepts_an_ffmpeg_without_any_text_rendering_filter() -> None:
    """Homebrew's default ffmpeg has no ass/subtitles/drawtext; captions must not need them."""
    checks = doctor_checks({})  # FfmpegWithout lists neither ass nor drawtext
    assert checks["ffmpeg: filters"].status is HealthStatus.OK
    assert "ass" not in checks["ffmpeg: filters"].detail
    assert checks["captions: font"].status in (HealthStatus.OK, HealthStatus.WARN)


@pytest.mark.ffmpeg
def test_doctor_fails_on_a_missing_core_filter_and_a_missing_font_file() -> None:
    checks = doctor_checks(
        {"SHORTGEN__SUBTITLES__STYLE__FONT_PATH": "/no/such/font.ttf"}, FfmpegWithout({"zoompan"})
    )
    assert checks["ffmpeg: filters"].status is HealthStatus.FAIL
    assert "zoompan" in checks["ffmpeg: filters"].detail
    assert checks["captions: font"].status is HealthStatus.FAIL
    without = doctor_checks({"SHORTGEN__SUBTITLES__ENABLED": "false"})
    assert "captions: font" not in without


def test_doctor_flags_a_hosted_provider_and_checks_its_credentials() -> None:
    hosted = {"SHORTGEN__PROVIDERS__IMAGE": "higgsfield_image"}
    checks = doctor_checks(hosted)
    assert checks["image: higgsfield_image"].status is HealthStatus.WARN
    assert "HOSTED, PAID" in checks["image: higgsfield_image"].detail
    assert "cloud.higgsfield.ai" in str(checks["image: higgsfield_image"].hint)
    assert checks["higgsfield_image: credentials"].status is HealthStatus.FAIL
    with_key = doctor_checks({**hosted, "HF_KEY": "key:secret"})
    assert with_key["higgsfield_image: credentials"].status is HealthStatus.OK
    assert "secret" not in " ".join(f"{c.detail} {c.hint}" for c in with_key.values())


def test_doctor_reports_missing_binaries_and_dependencies_without_crashing() -> None:
    checks = doctor_checks(
        {
            "SHORTGEN__RENDER__FFMPEG_BIN": "no-such-ffmpeg",
            "SHORTGEN__PROVIDERS__IMAGE": "sdxl",
            "SHORTGEN__PROVIDERS__VIDEO": "nonsense",
        }
    )
    assert checks["ffmpeg"].status is HealthStatus.FAIL
    assert "ffmpeg: filters" not in checks  # not probed when the binary is absent
    assert checks["image: sdxl"].detail.startswith("stabilityai/stable-diffusion-xl-base-1.0")
    assert checks["sdxl: dependencies"].status is HealthStatus.FAIL  # torch is not installed in CI
    assert "shortgen[image]" in (checks["sdxl: dependencies"].hint or "")
    assert checks["video: nonsense"].status is HealthStatus.FAIL
    assert has_failures(list(checks.values()))


def test_doctor_cli_exit_code_reflects_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHORTGEN__RENDER__FFMPEG_BIN", "no-such-ffmpeg")
    result = invoke("doctor", "--json")
    assert result.exit_code == EXIT_FAILURE
    assert any(check["status"] == "fail" for check in json.loads(result.output))


# ------------------------------------------------------------------- MCP


def call(server: Any, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    result = asyncio.run(server.call_tool(name, arguments))
    assert not result.is_error, result.content
    return json.loads(result.content[0].text)


def test_mcp_exposes_exactly_the_cli_commands() -> None:
    server = mcp_server.create_server()
    tools = {tool.name: tool for tool in asyncio.run(server.list_tools())}
    assert set(tools) == {"init_project", "plan", "generate", "render", "run", "status", "doctor"}
    assert all(tool.description for tool in tools.values())
    assert tools["generate"].input_schema["required"] == ["project_dir"]


@pytest.mark.ffmpeg
def test_mcp_tools_drive_the_same_pipeline_as_the_cli(script: Path, tmp_path: Path) -> None:
    server = mcp_server.create_server()
    project = str(tmp_path / "mcp-project")
    created = call(server, "init_project", {"project_dir": project, "script_path": str(script)})
    assert created["name"] == "mcp-project"

    planned = call(server, "plan", {"project_dir": project})
    assert planned["stages"] == [
        {"stage": "plan", "total": 1, "skipped": 0, "completed": 1, "failed": []}
    ]
    generated = call(server, "generate", {"project_dir": project, "stages": ["speech", "align"]})
    assert [stage["stage"] for stage in generated["stages"]] == ["speech", "align"]

    finished = call(server, "run", {"project_dir": project})
    assert finished["final_video"] == str(Path(project) / "output/final.mp4")
    skipped = {s["stage"]: s["skipped"] for s in finished["stages"]}
    assert (skipped["plan"], skipped["speech"]) == (1, 3)  # earlier tool calls were reused

    # The CLI sees exactly what the MCP tools did.
    cli_status = json.loads(invoke("status", project, "--json").output)
    assert cli_status == call(server, "status", {"project_dir": project})
    assert call(server, "doctor", {})["ok"] is True


def test_mcp_tool_errors_carry_the_reason_to_the_assistant(tmp_path: Path) -> None:
    server = mcp_server.create_server()
    with pytest.raises(Exception, match=r"no project at .*create one with `shortgen init`"):
        asyncio.run(server.call_tool("status", {"project_dir": str(tmp_path / "missing")}))
    with pytest.raises(Exception, match="'nonsense' is not a valid StageName"):
        asyncio.run(server.call_tool("generate", {"project_dir": "x", "stages": ["nonsense"]}))


# ------------------------------------------------------------ app helpers


def test_prepare_project_resolves_every_kind_of_source(script: Path, tmp_path: Path) -> None:
    created = app.prepare_project(script)
    assert created == Path("projects/lighthouse")
    assert app.prepare_project(script) == created  # second call adopts, does not re-init
    assert app.prepare_project(created) == created  # a directory is used as is
    topic = app.prepare_project(None, topic="A day in the life of a lighthouse keeper!")
    assert topic == Path("projects/a-day-in-the-life-of-a-lighthouse-keeper")
    with pytest.raises(ShortgenError, match="script file, a project directory or --topic"):
        app.prepare_project(None)


def test_a_folder_with_a_script_becomes_a_project_in_place(tmp_path: Path) -> None:
    folder = tmp_path / "my-short"
    folder.mkdir()
    (folder / "script.md").write_text(STORY)
    (folder / "config.yaml").write_text("project: {aspect: '16:9'}\n")
    assert app.prepare_project(folder) == folder
    assert (folder / "manifest.json").is_file()
    assert (folder / "config.yaml").read_text() == "project: {aspect: '16:9'}\n"  # kept as is
    ctx = app.build_context(folder, env=FAKE_ENV)
    assert ctx.config.resolution.width == 320  # the folder's own config applies

    # Editing the project's own script.md is picked up by the next run.
    edited = STORY.replace("sea was calm", "sky was clear")
    (folder / "script.md").write_text(edited)
    app.prepare_project(folder)
    assert "sky was clear" in ctx.store.load().project.script

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ShortgenError, match="is not a project"):
        app.prepare_project(empty)


def directed(picture: str) -> str:
    """STORY with a picture description under its second paragraph."""
    first, second, third = STORY.strip().split("\n\n")[1:]
    return f"# The Lighthouse\n\n{first}\n\n{second}\n<!-- image: {picture} -->\n\n{third}\n"


def test_every_command_sees_an_edited_script_not_only_run(tmp_path: Path) -> None:
    folder = tmp_path / "my-short"
    folder.mkdir()
    (folder / "script.md").write_text(directed("a small boat"))
    app.prepare_project(folder)
    app.execute(folder, [StageName.PLAN, StageName.IMAGE], env=FAKE_ENV)

    # Change one picture description, then ask for images only, as `generate` does.
    (folder / "script.md").write_text(directed("a red sail at dawn"))
    status = app.project_status(app.build_context(folder, env=FAKE_ENV))
    plan = next(stage for stage in status.stages if stage.stage == "plan")
    assert plan.fresh == 0  # status already knows the plan is out of date
    image = next(stage for stage in status.stages if stage.stage == "image")
    assert image.pending == 0 and "old plan" in str(image.note)  # and says its counts are not final

    summary = app.execute(folder, [StageName.IMAGE], env=FAKE_ENV)

    done = {stage.stage: stage.completed for stage in summary.stages}
    assert done == {"plan": 1, "image": 1}  # planned again, then only the edited scene
    scenes = app.build_context(folder, env=FAKE_ENV).store.load().project.scenes
    assert scenes[1].shot.image_prompt == "a red sail at dawn"
    settled = app.project_status(app.build_context(folder, env=FAKE_ENV))
    assert all(stage.note is None for stage in settled.stages)


def test_a_negative_line_redraws_only_its_own_scene(tmp_path: Path) -> None:
    folder = tmp_path / "my-short"
    folder.mkdir()
    (folder / "script.md").write_text(directed("a dark bedroom"))
    app.execute(folder, [StageName.PLAN, StageName.IMAGE], env=FAKE_ENV)

    with_negative = directed("a dark bedroom -->\n<!-- negative: lamp, ceiling light")
    (folder / "script.md").write_text(with_negative)
    summary = app.execute(folder, [StageName.IMAGE], env=FAKE_ENV)

    assert {stage.stage: stage.completed for stage in summary.stages} == {"plan": 1, "image": 1}
    ctx = app.build_context(folder, env=FAKE_ENV)
    scenes = ctx.store.load().project.scenes
    assert [scene.shot.negative_prompt for scene in scenes] == ["", "lamp, ceiling light", ""]
    assert "lamp" not in ctx.config.project.negative_prompt  # the project's own list is untouched


def test_commands_other_than_run_work_on_a_brand_new_story_folder(tmp_path: Path) -> None:
    folder = tmp_path / "new-story"
    folder.mkdir()
    (folder / "script.md").write_text(STORY)

    status = app.project_status(app.build_context(folder, env=FAKE_ENV))
    assert status.stages[0].pending == 1  # set up in place, nothing run yet

    summary = app.execute(folder, [StageName.IMAGE], scene_ids=["s001", "s002"], env=FAKE_ENV)
    done = {stage.stage: stage.completed for stage in summary.stages}
    assert done == {"plan": 1, "image": 2}  # planned first, then only the two scenes asked for
    assert (folder / "images" / "s001.png").is_file()
    assert not (folder / "images" / "s003.png").exists()


def test_a_script_that_differs_only_in_trailing_blank_lines_is_not_an_edit(tmp_path: Path) -> None:
    folder = tmp_path / "my-short"
    folder.mkdir()
    (folder / "script.md").write_text(STORY)
    app.prepare_project(folder)
    (folder / "script.md").write_text(STORY + "\n\n")
    assert app.adopt_script_edits(folder) is False


def test_init_rejects_unusable_scripts(tmp_path: Path) -> None:
    empty = tmp_path / "empty.md"
    empty.write_text("  \n")
    with pytest.raises(ShortgenError, match="script file is empty"):
        app.init_project(tmp_path / "p1", script_file=empty)
    binary = tmp_path / "story.pdf"
    binary.write_bytes(b"%PDF")
    with pytest.raises(ShortgenError, match="unsupported file type"):
        app.init_project(tmp_path / "p2", script_file=binary)
    assert not (tmp_path / "p1").exists()
