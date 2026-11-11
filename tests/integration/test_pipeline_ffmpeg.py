"""The real renderer end to end: fake models, real ffmpeg, a playable MP4.

Skipped automatically when ffmpeg/ffprobe are not installed.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from shortgen import app
from shortgen.domain.models import AssetKind
from shortgen.pipeline.stage import StageName
from shortgen.pipeline.stages import read_alignment, scene_seconds, upstream
from shortgen.process import SubprocessRunner
from tests.helpers import make_context, make_project

pytestmark = pytest.mark.ffmpeg

REAL_VIDEO = {"SHORTGEN__PROVIDERS__VIDEO": "kenburns"}
BRIGHT = 235  # luma above this only occurs in white caption text, not in the fake gradients


def probe(path: Path) -> dict[str, dict[str, str]]:
    """Stream properties of a media file, keyed by codec type."""
    output = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-show_streams", "-of", "json", str(path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {stream["codec_type"]: stream for stream in json.loads(output)["streams"]}


@pytest.fixture(scope="module")
def rendered(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A project rendered once with real ffmpeg, in a directory with an awkward name."""
    root = tmp_path_factory.mktemp("it's a [test]: dir")
    project = make_project(root)
    app.run_pipeline(make_context(project, env=REAL_VIDEO, commands=SubprocessRunner()))
    return project


def test_final_video_has_the_configured_shape(rendered: Path) -> None:
    streams = probe(rendered / "output/final.mp4")
    video, audio = streams["video"], streams["audio"]
    assert (video["codec_name"], video["width"], video["height"]) == ("h264", 180, 320)
    assert video["pix_fmt"] == "yuv420p"
    assert video["r_frame_rate"] == "30/1"
    assert (audio["codec_name"], audio["sample_rate"], audio["channels"]) == ("aac", "48000", 2)


def test_final_video_is_exactly_as_long_as_the_timeline(rendered: Path) -> None:
    ctx = make_context(rendered, env=REAL_VIDEO)
    manifest = ctx.store.load()
    fps = ctx.config.project.fps
    expected_frames = sum(
        round(
            scene_seconds(ctx, upstream(manifest, StageName.ALIGN, scene.id, AssetKind.ALIGNMENT))
            * fps
        )
        for scene in manifest.project.scenes
    )
    video = probe(rendered / "output/final.mp4")["video"]
    assert int(video["nb_read_frames"]) == expected_frames


def test_each_segment_matches_its_scene_length(rendered: Path) -> None:
    ctx = make_context(rendered, env=REAL_VIDEO)
    manifest = ctx.store.load()
    for scene in manifest.project.scenes:
        seconds = scene_seconds(
            ctx, upstream(manifest, StageName.ALIGN, scene.id, AssetKind.ALIGNMENT)
        )
        streams = probe(rendered / f"build/{scene.id}.mov")
        assert int(streams["video"]["nb_read_frames"]) == round(seconds * 30)
        assert float(streams["audio"]["duration"]) == pytest.approx(seconds, abs=0.001)


def test_subtitles_cover_the_narration_and_stay_inside_the_video(rendered: Path) -> None:
    ctx = make_context(rendered, env=REAL_VIDEO)
    manifest = ctx.store.load()
    last = manifest.project.scenes[-1]
    alignment = read_alignment(
        ctx.workspace, upstream(manifest, StageName.ALIGN, last.id, AssetKind.ALIGNMENT)
    )
    assert [word.word for word in alignment.words] == last.narration.split()
    video_seconds = float(probe(rendered / "output/final.mp4")["video"]["duration"])
    final_cue_end = (rendered / "subtitles/captions.srt").read_text().split(" --> ")[-1][:12]
    hours, minutes, seconds = final_cue_end.replace(",", ".").split(":")
    assert int(hours) * 3600 + int(minutes) * 60 + float(seconds) <= video_seconds


def bright_pixels(video: Path, at: float) -> int:
    """Near-white pixels in the caption band of the frame at ``at`` seconds."""
    band = "crop=iw:ih/3:0:ih/2,format=gray"
    seek = ["ffmpeg", "-v", "error", "-ss", f"{at:.3f}", "-i", str(video), "-frames:v", "1"]
    raw = subprocess.run(
        [*seek, "-vf", band, "-f", "rawvideo", "-"], check=True, capture_output=True
    ).stdout
    return sum(1 for value in raw if value > BRIGHT)


def test_captions_are_burned_in_only_while_their_cue_is_on_screen(rendered: Path) -> None:
    ctx = make_context(rendered, env=REAL_VIDEO)
    cues = json.loads((rendered / "subtitles/captions.json").read_text())["cues"]
    first = cues[0]
    assert first["start"] == pytest.approx(ctx.config.timing.pad_start_s)
    video = rendered / "output/final.mp4"
    during = bright_pixels(video, (first["start"] + first["end"]) / 2)
    before = bright_pixels(video, first["start"] / 3)
    assert before == 0, "nothing should be drawn before the first word is spoken"
    assert during > 40, "the caption text is missing from the frame"


def test_rerun_with_real_ffmpeg_is_a_no_op(rendered: Path) -> None:
    before = (rendered / "output/final.mp4").stat().st_mtime_ns
    reports = app.run_pipeline(make_context(rendered, env=REAL_VIDEO, commands=SubprocessRunner()))
    assert all(report.completed == 0 for report in reports)
    assert (rendered / "output/final.mp4").stat().st_mtime_ns == before
