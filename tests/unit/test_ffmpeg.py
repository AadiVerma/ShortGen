"""ffmpeg command building: argument lists, never shell strings."""

from __future__ import annotations

from pathlib import Path

import pytest

from shortgen.config import AppConfig
from shortgen.domain.models import CameraMove
from shortgen.errors import RenderError
from shortgen.process import CommandResult
from shortgen.render.ffmpeg import (
    ConcatSpec,
    FramesSpec,
    KenBurnsSpec,
    SegmentSpec,
    concat_command,
    concat_list,
    frames_command,
    kenburns_command,
    probe_duration,
    segment_command,
)
from shortgen.render.layout import ClipFit, Size, cover_filter, fit_area, fit_clip
from shortgen.render.settings import VideoEncode

CLIP_ENCODE = VideoEncode(video_codec="libx264", preset="veryfast", crf=14, pixel_format="yuv420p")
HOSTILE = Path("/tmp/it's a; rm -rf $(dir) && `x`/clip one.mp4")  # noqa: S108


def option(args: list[str], flag: str) -> str:
    """The value following ``flag`` in an argument list."""
    return args[args.index(flag) + 1]


def kenburns(camera: CameraMove) -> list[str]:
    spec = KenBurnsSpec(
        image=Path("in.png"),
        output=Path("out.mp4"),
        frames=90,
        fps=30,
        size=Size(width=1080, height=1920),
        camera=camera,
        zoom=1.2,
        supersample=2,
        encode=CLIP_ENCODE,
    )
    return kenburns_command("ffmpeg", spec).args


def test_fit_area_matches_known_model_sizes() -> None:
    portrait = Size(width=1080, height=1920)
    assert fit_area(portrait, 1024 * 1024, 64) == Size(width=768, height=1344)  # SDXL bucket
    assert fit_area(Size(width=1920, height=1080), 1024 * 1024, 64) == Size(width=1344, height=768)
    wan = fit_area(portrait, 704 * 1280, 32)
    assert wan.width * wan.height <= 704 * 1280
    assert (wan.width % 32, wan.height % 32) == (0, 0)
    assert fit_area(portrait, 1, 16) == Size(width=16, height=16)  # never collapses to zero


def test_fit_clip_slows_then_holds_then_cuts() -> None:
    assert fit_clip(5.0, 5.0, 2.0) == ClipFit(slowdown=1.0, hold_s=0.0)
    assert fit_clip(8.0, 5.0, 2.0) == ClipFit(slowdown=1.0, hold_s=0.0)  # too long: cut
    assert fit_clip(4.0, 6.0, 2.0) == ClipFit(slowdown=1.5, hold_s=0.0)
    assert fit_clip(2.0, 6.0, 2.0) == ClipFit(slowdown=2.0, hold_s=2.0)
    assert fit_clip(0.0, 6.0, 2.0) == ClipFit(slowdown=1.0, hold_s=0.0)


def test_cover_filter_fills_and_crops() -> None:
    assert cover_filter(Size(width=1080, height=1920)) == (
        "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920"
    )


def test_kenburns_zoom_in_command() -> None:
    args = kenburns(CameraMove.ZOOM_IN)
    assert args[0] == "ffmpeg"
    assert option(args, "-i") == "in.png"
    assert option(args, "-frames:v") == "90"
    assert args[-1] == "out.mp4"
    assert "-an" in args
    assert option(args, "-vf") == (
        "scale=2160:3840:force_original_aspect_ratio=increase,crop=2160:3840,"
        "zoompan=z='1+0.2000*on/89':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
        ":d=90:s=1080x1920:fps=30,format=yuv420p"
    )


@pytest.mark.parametrize("camera", list(CameraMove))
def test_every_camera_move_has_comma_free_expressions(camera: CameraMove) -> None:
    zoompan = option(kenburns(camera), "-vf").split("zoompan=")[1].split(":d=")[0]
    assert "," not in zoompan
    assert zoompan.count("'") == 6  # exactly the three quoted expressions


def test_opposite_pans_differ() -> None:
    assert option(kenburns(CameraMove.PAN_LEFT), "-vf") != option(
        kenburns(CameraMove.PAN_RIGHT), "-vf"
    )


def test_segment_command_is_frame_exact(config: AppConfig) -> None:
    spec = SegmentSpec(
        clip=Path("clip.mp4"),
        audio=Path("voice.wav"),
        output=Path("seg.mov"),
        frames=76,
        fps=30,
        audio_offset_s=0.15,
        size=Size(width=1080, height=1920),
        fit=ClipFit(slowdown=1.5, hold_s=0.0),
        encode=config.render.intermediate,
    )
    args = segment_command("ffmpeg", spec).args
    graph = option(args, "-filter_complex")
    video, audio = graph.split(";")
    assert video.startswith("[0:v]setpts=1.500000*PTS,scale=1080:1920")
    assert "tpad=stop_mode=clone:stop_duration=2.533333" in video
    assert audio == (
        "[1:a]aresample=48000,adelay=150:all=1,apad=whole_dur=2.533333,atrim=duration=2.533333[a]"
    )
    assert option(args, "-frames:v") == "76"
    assert option(args, "-c:a") == "pcm_s16le"
    assert "-b:a" not in args  # PCM has no bitrate setting
    assert [args[i + 1] for i, a in enumerate(args) if a == "-map"] == ["[v]", "[a]"]


def test_segment_without_slowdown_leaves_timestamps_alone(config: AppConfig) -> None:
    spec = SegmentSpec(
        clip=Path("c.mp4"),
        audio=Path("a.wav"),
        output=Path("o.mov"),
        frames=30,
        fps=30,
        audio_offset_s=0.0,
        size=Size(width=320, height=180),
        fit=ClipFit(slowdown=1.0, hold_s=0.0),
        encode=config.render.intermediate,
    )
    assert "setpts" not in option(segment_command("ffmpeg", spec).args, "-filter_complex")


def test_concat_overlays_the_caption_timeline_without_text_filters(config: AppConfig) -> None:
    spec = ConcatSpec(
        list_file=Path("/p/output/.render-x/segments.ffconcat"),
        output=Path("/p/output/final.partial.mp4"),
        caption_track=Path("/odd: path's [dir]/captions.ffconcat"),
        frames=354,
        encode=config.render.final,
    )
    command = concat_command("ffmpeg", spec)
    args = command.args
    assert command.cwd is None
    inputs = [args[i + 1] for i, arg in enumerate(args) if arg == "-i"]
    assert inputs == [str(spec.list_file), "/odd: path's [dir]/captions.ffconcat"]
    graph = option(args, "-filter_complex")
    assert graph == "[0:v][1:v]overlay=format=auto:eof_action=pass,format=yuv420p[v]"
    assert "path" not in graph  # no file name ever enters the filter graph
    assert [args[i + 1] for i, a in enumerate(args) if a == "-map"] == ["[v]", "0:a"]
    assert option(args, "-frames:v") == "354"
    assert option(args, "-b:a") == "192k"
    assert option(args, "-movflags") == "+faststart"


def test_concat_without_captions_has_no_filter(config: AppConfig) -> None:
    spec = ConcatSpec(
        list_file=Path("l.txt"),
        output=Path("o.mp4"),
        caption_track=None,
        encode=config.render.final,
    )
    args = concat_command("ffmpeg", spec).args
    assert "-filter_complex" not in args
    assert "-frames:v" not in args
    assert args.count("-i") == 1


def test_hostile_paths_stay_single_arguments(config: AppConfig) -> None:
    spec = SegmentSpec(
        clip=HOSTILE,
        audio=HOSTILE,
        output=HOSTILE,
        frames=1,
        fps=30,
        audio_offset_s=0,
        size=Size(width=16, height=16),
        fit=ClipFit(slowdown=1.0, hold_s=0.0),
        encode=config.render.intermediate,
    )
    args = segment_command("ffmpeg", spec).args
    assert args.count(str(HOSTILE)) == 3
    assert str(HOSTILE) not in option(args, "-filter_complex")


def test_concat_list_quotes_single_quotes() -> None:
    assert concat_list([Path("/a/b.mov"), Path("/it's/c.mov")]) == (
        "ffconcat version 1.0\nfile '/a/b.mov'\nfile '/it'\\''s/c.mov'\n"
    )


def test_frames_command_reads_a_numbered_sequence() -> None:
    spec = FramesSpec(
        pattern=Path("frames/%05d.png"), output=Path("clip.mp4"), fps=24, encode=CLIP_ENCODE
    )
    args = frames_command("ffmpeg", spec).args
    assert option(args, "-framerate") == "24"
    assert option(args, "-i") == "frames/%05d.png"
    assert args.index("-framerate") < args.index("-i")  # an input option, so it comes first


class StubRunner:
    def __init__(self, stdout: str) -> None:
        self.stdout = stdout
        self.seen: list[str] = []

    def run(self, args: list[str], *, cwd: Path | None = None) -> CommandResult:
        self.seen = list(args)
        return CommandResult(stdout=self.stdout, stderr="")


def test_probe_duration_parses_ffprobe_output() -> None:
    runner = StubRunner("4.966667\n")
    assert probe_duration(runner, "ffprobe", Path("clip.mp4")) == pytest.approx(4.966667)
    assert runner.seen[0] == "ffprobe"
    assert runner.seen[-1] == "clip.mp4"


@pytest.mark.parametrize("output", ["", "N/A\n", "0.000000\n"])
def test_probe_duration_rejects_unusable_media(output: str) -> None:
    with pytest.raises(RenderError):
        probe_duration(StubRunner(output), "ffprobe", Path("clip.mp4"))
