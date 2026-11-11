"""End-to-end pipeline runs with fake providers: no GPU, no models, no ffmpeg."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shortgen import app
from shortgen.domain.models import JobStatus
from shortgen.errors import ConfigError, StageError, StageFailedError
from shortgen.pipeline.runner import StageReport
from shortgen.pipeline.stage import STAGE_ORDER, StageName
from tests.helpers import STORY, RecordingRunner, make_context, make_project
from tests.integration import doubles

SCENES = 3  # paragraphs in STORY


def completed(reports: list[StageReport]) -> dict[str, int]:
    return {report.stage.value: report.completed for report in reports if report.completed}


@pytest.fixture
def project(tmp_path: Path) -> Path:
    return make_project(tmp_path)


@pytest.fixture
def doubles_env(tmp_path: Path) -> dict[str, str]:
    """Environment selecting the misbehaving test providers via a user registry file."""
    doubles.EVENTS.clear()
    doubles.mood = ""
    models = tmp_path / "test-models.yaml"
    models.write_text(doubles.MODELS_YAML)
    return {
        "SHORTGEN__RUNTIME__MODELS_FILE": str(models),
        "SHORTGEN__PROVIDERS__IMAGE": "moody_image",
        "SHORTGEN__PROVIDERS__SPEECH": "tracked_speech",
    }


def test_full_pipeline_produces_every_artifact(project: Path) -> None:
    runner = RecordingRunner()
    reports = app.run_pipeline(make_context(project, commands=runner))

    assert [report.stage for report in reports] == list(STAGE_ORDER)
    assert all(not report.failed for report in reports)
    for relative in (
        "audio/s001.wav",
        "audio/s003.words.json",
        "alignment/s002.json",
        "images/s001.png",
        "clips/s003.mp4",
        "subtitles/captions.ass",
        "subtitles/captions.srt",
        "subtitles/captions.json",
        "build/s002.mov",
        "output/final.mp4",
    ):
        assert (project / relative).is_file(), relative
    assert not list(project.rglob("*.partial.*")), "temp files must not survive a clean run"

    manifest = json.loads((project / "manifest.json").read_text())
    assert len(manifest["project"]["scenes"]) == SCENES
    assert {job["status"] for job in manifest["jobs"].values()} == {"done"}
    assert len(manifest["jobs"]) == 2 + 5 * SCENES + 1  # plan, subtitles, render + per-scene

    # One compose command per scene and one final render, all argument lists.
    assert len(runner.ffmpeg_calls()) == SCENES + 1
    final = runner.ffmpeg_calls()[-1]
    graph = final[final.index("-filter_complex") + 1]
    assert graph.startswith("[0:v][1:v]overlay=")
    assert "ass=" not in graph and "subtitles=" not in graph  # works without libass
    assert not list(project.rglob(".render-*")), "caption scratch files must be cleaned up"

    srt = " ".join((project / "subtitles/captions.srt").read_text().split())
    assert "Mira climbed the stairs" in srt
    assert "By morning the sea was calm." in srt


def test_second_run_does_no_work(project: Path) -> None:
    app.run_pipeline(make_context(project))
    runner = RecordingRunner()
    reports = app.run_pipeline(make_context(project, commands=runner))
    assert completed(reports) == {}
    assert all(report.skipped == report.total for report in reports)
    assert runner.commands == []


def test_editing_one_scene_regenerates_only_what_depends_on_it(tmp_path: Path) -> None:
    project = make_project(tmp_path)
    app.run_pipeline(make_context(project))
    edited = tmp_path / "story.md"
    edited.write_text(STORY.replace("By morning the sea was calm.", "By morning she was asleep."))
    assert app.sync_script(project, edited)

    reports = app.run_pipeline(make_context(project))
    assert completed(reports) == {
        "plan": 1,
        "speech": 1,
        "align": 1,
        "image": 1,
        "video": 1,
        "subtitles": 1,
        "compose": 1,
        "render": 1,
    }


def test_restyling_subtitles_only_redoes_subtitles_and_final_render(project: Path) -> None:
    app.run_pipeline(make_context(project))
    restyled = make_context(project, env={"SHORTGEN__SUBTITLES__STYLE__UPPERCASE": "true"})
    assert completed(app.run_pipeline(restyled)) == {"subtitles": 1, "render": 1}
    assert "MIRA CLIMBED" in (project / "subtitles/captions.ass").read_text()


def test_disabling_subtitles_skips_the_stage_and_the_burn_in(project: Path) -> None:
    runner = RecordingRunner()
    env = {"SHORTGEN__SUBTITLES__ENABLED": "false"}
    reports = app.run_pipeline(make_context(project, env=env, commands=runner))
    assert next(r for r in reports if r.stage is StageName.SUBTITLES).total == 0
    assert "-filter_complex" not in runner.ffmpeg_calls()[-1]
    assert not (project / "subtitles").exists()


def test_deleted_output_is_regenerated(project: Path) -> None:
    app.run_pipeline(make_context(project))
    (project / "images/s002.png").unlink()
    reports = app.run_pipeline(make_context(project))
    # The fake image is deterministic, so its hash is unchanged and nothing downstream reruns.
    assert completed(reports) == {"image": 1}


def test_replacing_a_supplied_picture_regenerates_only_that_scene(project: Path) -> None:
    from PIL import Image

    art = project / "art"
    art.mkdir()
    for number in range(1, SCENES + 1):
        Image.new("RGB", (64, 64), "navy").save(art / f"s{number:03d}.png")
    env = {"SHORTGEN__PROVIDERS__IMAGE": "folder"}
    app.run_pipeline(make_context(project, env=env))
    Image.new("RGB", (64, 64), "gold").save(art / "s002.jpg")
    (art / "s002.png").unlink()
    reports = completed(app.run_pipeline(make_context(project, env=env)))
    assert (reports["image"], reports["video"]) == (1, 1)
    assert "speech" not in reports


def test_force_and_scene_filter_rerun_exactly_the_selection(project: Path) -> None:
    ctx = make_context(project)
    app.run_pipeline(ctx)
    reports = app.run_pipeline(ctx, [StageName.IMAGE], force=True, scene_ids={"s002"})
    assert completed(reports) == {"image": 1}
    assert reports[0].total == 1


def test_stage_cannot_run_before_its_inputs_exist(project: Path) -> None:
    ctx = make_context(project)
    with pytest.raises(StageError, match="no scenes yet; run the plan stage first"):
        app.run_pipeline(ctx, [StageName.SPEECH])
    app.run_pipeline(ctx, [StageName.PLAN])
    with pytest.raises(StageError, match="has not completed for s001; run it first"):
        app.run_pipeline(ctx, [StageName.VIDEO])


def test_one_failing_scene_does_not_stop_the_others_and_is_retried_alone(
    project: Path, doubles_env: dict[str, str]
) -> None:
    doubles.mood = "fail"
    with pytest.raises(StageFailedError) as failure:
        app.run_pipeline(make_context(project, env=doubles_env))
    assert failure.value.stage == "image"
    assert failure.value.failures == ["s001"]

    manifest = make_context(project, env=doubles_env).store.load()
    failed = manifest.jobs["image:s001"]
    assert failed.status is JobStatus.FAILED
    assert failed.error == "ProviderError: the storm scene cannot be drawn"
    assert manifest.jobs["image:s002"].status is JobStatus.DONE
    assert "video:s001" not in manifest.jobs  # the pipeline stopped after the failed stage

    doubles.mood = ""
    reports = app.run_pipeline(make_context(project, env=doubles_env))
    assert completed(reports)["image"] == 1  # only the scene that failed
    assert (project / "output/final.mp4").is_file()


def test_transient_errors_are_retried_within_the_unit(
    project: Path, doubles_env: dict[str, str]
) -> None:
    doubles.mood = "flaky"
    ctx = make_context(project, env=doubles_env)
    app.run_pipeline(ctx, [StageName.PLAN, StageName.IMAGE])
    assert ctx.store.load().jobs["image:s001"].attempts == 3
    assert ctx.store.load().jobs["image:s002"].attempts == 1


def test_interrupt_leaves_a_consistent_resumable_project(
    project: Path, doubles_env: dict[str, str]
) -> None:
    doubles.mood = "interrupt"
    with pytest.raises(KeyboardInterrupt):
        app.run_pipeline(make_context(project, env=doubles_env))

    manifest = make_context(project, env=doubles_env).store.load()  # still valid JSON
    assert manifest.jobs["image:s001"].status is JobStatus.INTERRUPTED
    assert not (project / "images/s001.png").exists()
    assert not list(project.rglob("*.partial.*")), "the half-written image must be removed"
    assert doubles.EVENTS[-1] == "unload moody_image", "the model must be unloaded on Ctrl+C"

    doubles.mood = ""
    reports = app.run_pipeline(make_context(project, env=doubles_env))
    assert completed(reports)["image"] == SCENES
    assert "speech" not in completed(reports)  # finished work is not repeated


def test_only_one_model_is_loaded_at_a_time(project: Path, doubles_env: dict[str, str]) -> None:
    app.run_pipeline(make_context(project, env=doubles_env))
    loaded: set[str] = set()
    for event in doubles.EVENTS:
        action, _, name = event.partition(" ")
        if action == "load":
            assert not loaded, f"{name} loaded while {loaded} still in memory"
            loaded.add(name)
        elif action == "unload":
            loaded.discard(name)
    assert doubles.EVENTS.count("load tracked_speech") == 1
    assert doubles.EVENTS.count("use tracked_speech") == SCENES
    assert not loaded


def test_up_to_date_stage_never_loads_its_model(project: Path, doubles_env: dict[str, str]) -> None:
    app.run_pipeline(make_context(project, env=doubles_env))
    doubles.EVENTS.clear()
    app.run_pipeline(make_context(project, env=doubles_env))
    assert doubles.EVENTS == []


def test_unavailable_provider_stops_its_stage_immediately(
    project: Path, doubles_env: dict[str, str]
) -> None:
    env = {**doubles_env, "SHORTGEN__PROVIDERS__IMAGE": "missing_image"}
    with pytest.raises(StageFailedError) as failure:
        app.run_pipeline(make_context(project, env=env))
    assert failure.value.failures == ["s001"]  # not retried for every scene
    job = make_context(project, env=env).store.load().jobs["image:s001"]
    assert "pip install imaginary-lib" in (job.error or "")


def test_topic_projects_get_a_script_from_the_planner(tmp_path: Path) -> None:
    project = tmp_path / "topic-project"
    app.init_project(project, topic="a lighthouse keeper")
    ctx = make_context(project)
    app.run_pipeline(ctx, [StageName.PLAN])
    manifest = ctx.store.load()
    assert "a lighthouse keeper" in manifest.project.script
    assert (project / "script.md").read_text().strip() == manifest.project.script
    assert len(manifest.project.scenes) == SCENES
    assert completed(app.run_pipeline(ctx, [StageName.PLAN])) == {}  # and it is stable


def test_heuristic_planner_refuses_topics_with_a_useful_message(tmp_path: Path) -> None:
    project = tmp_path / "topic-project"
    app.init_project(project, topic="anything")
    ctx = make_context(project, env={"SHORTGEN__PROVIDERS__PLANNER": "heuristic"})
    with pytest.raises(StageFailedError):
        app.run_pipeline(ctx, [StageName.PLAN])
    assert "cannot write a script from a topic" in (ctx.store.load().jobs["plan:_"].error or "")


def test_misconfigured_provider_fails_before_any_work(project: Path) -> None:
    ctx = make_context(project, env={"SHORTGEN__PROVIDERS__VIDEO": "sora"})
    with pytest.raises(ConfigError, match=r"providers\.video: 'sora' is not registered"):
        app.run_pipeline(ctx)
    assert ctx.store.load().jobs == {}


def test_status_reports_progress_without_running_anything(project: Path) -> None:
    ctx = make_context(project)
    before = app.project_status(ctx)
    assert before.scenes == 0
    assert before.final_video is None
    assert before.stages[0].pending == 1
    assert before.stages[1].blocked_reason is not None

    app.run_pipeline(ctx)
    after = app.project_status(ctx)
    assert all(stage.pending == 0 and stage.failed == 0 for stage in after.stages)
    assert after.final_video == str((project / "output/final.mp4").resolve())
