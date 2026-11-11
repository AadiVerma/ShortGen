"""ffmpeg command builders. ffmpeg is the only renderer shortgen uses.

Every function here is pure: it turns a typed spec into an argument list.
Nothing is ever joined into a shell string, and no user-controlled text is
placed inside a filter graph: paths are only ever passed as separate arguments.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import Field

from shortgen.domain.models import CameraMove, FrozenModel
from shortgen.errors import RenderError
from shortgen.process import CommandRunner
from shortgen.render.layout import ClipFit, Size, cover_filter
from shortgen.render.settings import EncodeSettings, VideoEncode

REQUIRED_FILTERS = ("overlay", "zoompan", "tpad", "apad", "adelay")
_QUIET = ("-hide_banner", "-loglevel", "error", "-nostdin")
_OUTPUT_CHANNELS = "2"


@dataclass(frozen=True)
class Command:
    """An argument list plus the directory to run it in."""

    args: list[str]
    cwd: Path | None = None


class KenBurnsSpec(FrozenModel):
    """Animate a still image with a slow zoom or pan."""

    image: Path
    output: Path
    frames: int = Field(ge=1)
    fps: int = Field(ge=1)
    size: Size
    camera: CameraMove
    zoom: float = Field(gt=1)
    supersample: int = Field(ge=1)
    encode: VideoEncode


class SegmentSpec(FrozenModel):
    """Fit one scene's clip to its narration and mux them, frame-exact."""

    clip: Path
    audio: Path
    output: Path
    frames: int = Field(ge=1)
    fps: int = Field(ge=1)
    audio_offset_s: float = Field(ge=0)
    size: Size
    fit: ClipFit
    encode: EncodeSettings


class ConcatSpec(FrozenModel):
    """Join scene segments and optionally overlay a caption timeline."""

    list_file: Path
    output: Path
    caption_track: Path | None
    frames: int | None = Field(default=None, ge=1)
    encode: EncodeSettings


class FramesSpec(FrozenModel):
    """Encode a numbered image sequence (``%05d.png``) into a clip."""

    pattern: Path
    output: Path
    fps: int = Field(ge=1)
    encode: VideoEncode


def kenburns_command(ffmpeg: str, spec: KenBurnsSpec) -> Command:
    """Build the zoompan command for one still image."""
    canvas = Size(
        width=spec.size.width * spec.supersample, height=spec.size.height * spec.supersample
    )
    zoom, x, y = _motion(spec.camera, spec.zoom, max(1, spec.frames - 1))
    video_filter = (
        f"{cover_filter(canvas)},"
        f"zoompan=z='{zoom}':x='{x}':y='{y}':d={spec.frames}"
        f":s={spec.size.width}x{spec.size.height}:fps={spec.fps},"
        f"format={spec.encode.pixel_format}"
    )
    return Command(
        [
            ffmpeg,
            *_QUIET,
            "-y",
            "-i",
            str(spec.image),
            "-vf",
            video_filter,
            "-frames:v",
            str(spec.frames),
            "-an",
            *_video_args(spec.encode),
            str(spec.output),
        ]
    )


def segment_command(ffmpeg: str, spec: SegmentSpec) -> Command:
    """Build the per-scene command: retime, fill the frame, pad the audio, mux.

    The video is capped by frame count and the audio padded to exactly the same
    length, so every segment is as long as the timeline says it is.
    """
    duration = spec.frames / spec.fps
    video_steps = [
        cover_filter(spec.size),
        f"fps={spec.fps}",
        f"tpad=stop_mode=clone:stop_duration={duration:.6f}",
        "setsar=1",
        f"format={spec.encode.pixel_format}",
    ]
    if spec.fit.slowdown > 1:
        video_steps.insert(0, f"setpts={spec.fit.slowdown:.6f}*PTS")
    audio_steps = [
        f"aresample={spec.encode.audio_sample_rate}",
        f"adelay={round(spec.audio_offset_s * 1000)}:all=1",
        f"apad=whole_dur={duration:.6f}",
        f"atrim=duration={duration:.6f}",
    ]
    graph = f"[0:v]{','.join(video_steps)}[v];[1:a]{','.join(audio_steps)}[a]"
    return Command(
        [
            ffmpeg,
            *_QUIET,
            "-y",
            "-i",
            str(spec.clip),
            "-i",
            str(spec.audio),
            "-filter_complex",
            graph,
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-frames:v",
            str(spec.frames),
            *_video_args(spec.encode),
            *_audio_args(spec.encode),
            str(spec.output),
        ]
    )


def concat_command(ffmpeg: str, spec: ConcatSpec) -> Command:
    """Build the final command: concatenate segments, overlay captions, encode.

    Captions arrive as a timeline of transparent PNGs (see ``render.captions``)
    and are composited with the ``overlay`` filter, which every ffmpeg build
    has. No text-rendering filter (``ass``, ``subtitles``, ``drawtext``) is used.
    """
    args = [ffmpeg, *_QUIET, "-y", "-f", "concat", "-safe", "0", "-i", str(spec.list_file)]
    if spec.caption_track is not None:
        # eof_action=pass: once the caption timeline ends, frames pass through
        # untouched (repeat would append one extra frame to the output).
        graph = (
            f"[0:v][1:v]overlay=format=auto:eof_action=pass,format={spec.encode.pixel_format}[v]"
        )
        args += ["-i", str(spec.caption_track), "-filter_complex", graph]
        args += ["-map", "[v]", "-map", "0:a"]
    if spec.frames is not None:
        args += ["-frames:v", str(spec.frames)]
    args += [
        *_video_args(spec.encode),
        *_audio_args(spec.encode),
        "-movflags",
        "+faststart",
        str(spec.output),
    ]
    return Command(args)


def frames_command(ffmpeg: str, spec: FramesSpec) -> Command:
    """Build the command that turns generated frames into a silent clip."""
    return Command(
        [
            ffmpeg,
            *_QUIET,
            "-y",
            "-framerate",
            str(spec.fps),
            "-i",
            str(spec.pattern),
            "-an",
            *_video_args(spec.encode),
            str(spec.output),
        ]
    )


def probe_duration_command(ffprobe: str, path: Path) -> Command:
    """Build the ffprobe command that prints a file's duration in seconds."""
    return Command(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
    )


def probe_duration(runner: CommandRunner, ffprobe: str, path: Path) -> float:
    """Duration of a media file in seconds, as reported by ffprobe."""
    output = runner.run(probe_duration_command(ffprobe, path).args).stdout.strip()
    try:
        duration = float(output)
    except ValueError as exc:
        raise RenderError(f"ffprobe returned no duration for {path}: {output!r}") from exc
    if duration <= 0:
        raise RenderError(f"{path} has no playable duration")
    return duration


def concat_list(paths: Sequence[Path]) -> str:
    """Contents of an ffmpeg concat-demuxer list file for ``paths``."""
    lines = ["ffconcat version 1.0"]
    for path in paths:
        quoted = str(path).replace("'", "'\\''")
        lines.append(f"file '{quoted}'")
    return "\n".join(lines) + "\n"


def _motion(camera: CameraMove, zoom: float, last_frame: int) -> tuple[str, str, str]:
    """zoompan ``z``, ``x`` and ``y`` expressions for a camera move.

    ``on`` is zoompan's output frame counter. The expressions contain no
    commas or quotes, so they are safe inside the quoted filter options.
    """
    progress = f"on/{last_frame}"
    travel = f"{zoom - 1:.4f}"
    centre_x, centre_y = "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
    span_x, span_y = "(iw-iw/zoom)", "(ih-ih/zoom)"
    held = f"{zoom:.4f}"
    moves: dict[CameraMove, tuple[str, str, str]] = {
        CameraMove.STATIC: ("1", "0", "0"),
        CameraMove.ZOOM_IN: (f"1+{travel}*{progress}", centre_x, centre_y),
        CameraMove.ZOOM_OUT: (f"{held}-{travel}*{progress}", centre_x, centre_y),
        CameraMove.PAN_RIGHT: (held, f"{span_x}*{progress}", centre_y),
        CameraMove.PAN_LEFT: (held, f"{span_x}*(1-{progress})", centre_y),
        CameraMove.PAN_DOWN: (held, centre_x, f"{span_y}*{progress}"),
        CameraMove.PAN_UP: (held, centre_x, f"{span_y}*(1-{progress})"),
    }
    return moves[camera]


def _video_args(encode: VideoEncode) -> list[str]:
    return [
        "-c:v",
        encode.video_codec,
        "-preset",
        encode.preset,
        "-crf",
        str(encode.crf),
        "-pix_fmt",
        encode.pixel_format,
    ]


def _audio_args(encode: EncodeSettings) -> list[str]:
    args = ["-c:a", encode.audio_codec]
    if encode.audio_bitrate is not None:
        args += ["-b:a", encode.audio_bitrate]
    return [*args, "-ar", str(encode.audio_sample_rate), "-ac", _OUTPUT_CHANNELS]
