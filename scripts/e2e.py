"""End-to-end test harness: render every example project and verify the result.

For each example folder it copies the project, runs the pipeline one stage at
a time (timing each), then inspects what came out: stream properties, exact
frame count, audible audio, caption text, caption burn-in, and that a second
run does no work. It writes a Markdown report, a JSON report and one contact
sheet per video.

    python scripts/e2e.py                      # all examples, configured (real) models
    python scripts/e2e.py --images demo        # no image model: bundled art or placeholders
    python scripts/e2e.py --only odyssey-cartoon,octopus-facts --out e2e-runs

It uses the same ``shortgen.app`` functions as the CLI, so what passes here is
what a user gets.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from shortgen import __version__, app
from shortgen.config import process_environment
from shortgen.domain.models import AssetKind
from shortgen.domain.subtitles import CueSheet
from shortgen.domain.timing import script_tokens
from shortgen.errors import ShortgenError
from shortgen.pipeline.context import PipelineContext
from shortgen.pipeline.contracts import SceneAlignment
from shortgen.pipeline.stage import STAGE_ORDER, StageName
from shortgen.pipeline.stages import scene_seconds, upstream

AUDIBLE_DB = -50.0  # mean volume below this is effectively silence
BURN_IN_DIFF = 60  # grey-level change that only drawn text causes
BURN_IN_PIXELS = 30
SHEET_FRAMES = 8
THUMB_WIDTH = 320
PROJECT_OUTPUTS = (
    "manifest.json",
    "audio",
    "alignment",
    "images",
    "clips",
    "subtitles",
    "build",
    "output",
)


@dataclass
class Check:
    """One verified property of a rendered video."""

    name: str
    ok: bool
    detail: str = ""


@dataclass
class Result:
    """Everything measured for one example."""

    name: str
    aspect: str = ""
    providers: dict[str, str] = field(default_factory=dict)
    scenes: int = 0
    words: int = 0
    video_seconds: float = 0.0
    stage_seconds: dict[str, float] = field(default_factory=dict)
    total_seconds: float = 0.0
    rerun_seconds: float = 0.0
    size_mb: float = 0.0
    error: str | None = None
    checks: list[Check] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.error is None and all(check.ok for check in self.checks)


def run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=True, capture_output=True, text=True)


def probe(path: Path) -> dict[str, dict[str, Any]]:
    """Stream properties keyed by codec type, with frames actually counted."""
    out = run(
        ["ffprobe", "-v", "error", "-count_frames", "-show_streams", "-of", "json", str(path)]
    ).stdout
    return {stream["codec_type"]: stream for stream in json.loads(out)["streams"]}


def mean_volume_db(path: Path) -> float:
    log = run(
        ["ffmpeg", "-hide_banner", "-i", str(path), "-af", "volumedetect", "-vn", "-f", "null", "-"]
    ).stderr
    for line in log.splitlines():
        if "mean_volume:" in line:
            return float(line.split("mean_volume:")[1].split()[0])
    return float("-inf")


def grey_frame(path: Path, at: float, crop: str) -> bytes:
    """A small greyscale crop of the frame at ``at`` seconds."""
    vf = f"{crop},scale={THUMB_WIDTH}:-2,format=gray"
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-ss",
        f"{at:.3f}",
        "-i",
        str(path),
        "-frames:v",
        "1",
        "-vf",
        vf,
        "-f",
        "rawvideo",
        "-",
    ]
    return subprocess.run(cmd, check=True, capture_output=True).stdout


def scene_starts(ctx: PipelineContext) -> list[tuple[str, float, float]]:
    """(scene id, start, length) for every scene on the final timeline."""
    manifest = ctx.store.load()
    spans, cursor = [], 0.0
    for scene in manifest.project.scenes:
        length = scene_seconds(
            ctx, upstream(manifest, StageName.ALIGN, scene.id, AssetKind.ALIGNMENT)
        )
        spans.append((scene.id, cursor, length))
        cursor += length
    return spans


def check_streams(
    ctx: PipelineContext, video: Path, sheet: CueSheet | None, result: Result
) -> None:
    streams = probe(video)
    size, fps = ctx.config.resolution, ctx.config.project.fps
    v, a = streams.get("video", {}), streams.get("audio", {})
    shape = (v.get("width"), v.get("height"), v.get("r_frame_rate"))
    result.checks.append(
        Check(
            "resolution and frame rate", shape == (size.width, size.height, f"{fps}/1"), str(shape)
        )
    )
    expected = round(sum(length for _, _, length in scene_starts(ctx)) * fps)
    frames = int(v.get("nb_read_frames", 0))
    result.checks.append(
        Check(
            "frame count matches the timeline exactly",
            frames == expected,
            f"{frames} frames, expected {expected}",
        )
    )
    result.video_seconds = frames / fps
    drift = abs(float(a.get("duration", 0)) - result.video_seconds)
    volume = mean_volume_db(video)
    result.checks.append(
        Check(
            "audio present, audible and as long as the video",
            bool(a) and volume > AUDIBLE_DB and drift < 0.1,
            f"mean {volume:.1f} dB, length differs by {drift * 1000:.0f} ms",
        )
    )
    if sheet is not None:
        inside = all(0 <= cue.start < cue.end <= result.video_seconds + 1e-6 for cue in sheet.cues)
        result.checks.append(
            Check("every caption lies inside the video", inside, f"{len(sheet.cues)} cues")
        )


def check_assets(ctx: PipelineContext, result: Result) -> None:
    manifest = ctx.store.load()
    missing = [
        f"{kind}/{scene.id}"
        for scene in manifest.project.scenes
        for kind, path in (
            ("audio", f"audio/{scene.id}.wav"),
            ("image", f"images/{scene.id}.png"),
            ("clip", f"clips/{scene.id}.mp4"),
            ("segment", f"build/{scene.id}.mov"),
        )
        if not (ctx.workspace.root / path).is_file()
    ]
    result.checks.append(
        Check("every scene has audio, image, clip and segment", not missing, ", ".join(missing))
    )
    sources: dict[str, int] = {}
    for scene in manifest.project.scenes:
        text = (ctx.workspace.root / f"alignment/{scene.id}.json").read_text(encoding="utf-8")
        source = SceneAlignment.model_validate_json(text).source
        sources[source] = sources.get(source, 0) + 1
    detail = ", ".join(f"{count} scene(s) {name}" for name, count in sorted(sources.items()))
    result.checks.append(
        Check(
            "word timings available for every scene",
            sum(sources.values()) == len(manifest.project.scenes),
            detail,
        )
    )


def check_captions(ctx: PipelineContext, video: Path, sheet: CueSheet, result: Result) -> None:
    manifest = ctx.store.load()
    script_words = [
        word for scene in manifest.project.scenes for word in script_tokens(scene.narration)
    ]
    caption_words = " ".join(" ".join(cue.lines) for cue in sheet.cues).split()
    result.checks.append(
        Check(
            "captions carry every script word, in order",
            caption_words == " ".join(script_words).split(),
            f"{len(caption_words)} words",
        )
    )
    rules = ctx.config.subtitles.rules
    tidy = all(len(cue.lines) <= rules.max_lines for cue in sheet.cues) and all(
        earlier.end <= later.start + 1e-6
        for earlier, later in zip(sheet.cues, sheet.cues[1:], strict=False)
    )
    result.checks.append(Check("captions respect max_lines and never overlap", tidy))
    changed = [
        burned_in(ctx, video, cue.start, cue.end)
        for cue in sheet.cues[:: max(1, len(sheet.cues) // 4)]
    ]
    result.checks.append(
        Check(
            "captions are visibly burned into the frames",
            all(count > BURN_IN_PIXELS for count in changed),
            f"changed pixels per sampled cue: {changed}",
        )
    )


def burned_in(ctx: PipelineContext, video: Path, start: float, end: float) -> int:
    """Pixels that differ between the final frame and the same frame without captions."""
    at = (start + end) / 2
    scene_id, scene_start, _ = next(span for span in reversed(scene_starts(ctx)) if span[1] <= at)
    segment = ctx.workspace.root / f"build/{scene_id}.mov"
    final = grey_frame(video, at, "crop=iw:ih/2:0:ih/2")
    clean = grey_frame(segment, at - scene_start, "crop=iw:ih/2:0:ih/2")
    return sum(1 for a, b in zip(final, clean, strict=False) if abs(a - b) > BURN_IN_DIFF)


def contact_sheet(video: Path, sheet: CueSheet | None, seconds: float, out: Path) -> None:
    """A strip of frames taken while captions are on screen, for a quick visual review."""
    if sheet is not None and sheet.cues:
        step = max(1, len(sheet.cues) // SHEET_FRAMES)
        times = [(cue.start + cue.end) / 2 for cue in sheet.cues[::step]][:SHEET_FRAMES]
    else:
        times = [seconds * (i + 0.5) / SHEET_FRAMES for i in range(SHEET_FRAMES)]
    frames = []
    for index, at in enumerate(times):
        frame = out.with_name(f".frame{index}.png")
        run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-ss",
                f"{at:.3f}",
                "-i",
                str(video),
                "-frames:v",
                "1",
                "-vf",
                f"scale={THUMB_WIDTH}:-2",
                str(frame),
            ]
        )
        frames.append(frame)
    images = [Image.open(frame) for frame in frames]
    width, height = images[0].size
    columns = 4 if width > height else SHEET_FRAMES
    rows = -(-len(images) // columns)
    canvas = Image.new("RGB", (columns * width, rows * height), "black")
    for index, image in enumerate(images):
        canvas.paste(image, ((index % columns) * width, (index // columns) * height))
    canvas.save(out)
    for frame in frames:
        frame.unlink()


def render_example(source: Path, out_root: Path, options: argparse.Namespace) -> Result:
    result = Result(name=source.name)
    project = out_root / source.name
    if options.fresh and project.exists():
        shutil.rmtree(project)
    shutil.copytree(
        source, project, dirs_exist_ok=True, ignore=shutil.ignore_patterns(*PROJECT_OUTPUTS)
    )
    env = process_environment()
    if options.images == "demo":
        env["SHORTGEN__PROVIDERS__IMAGE"] = "folder" if (project / "art").is_dir() else "fake_image"
    try:
        app.prepare_project(project)
        ctx = app.build_context(project, config_file=options.config, env=env)
        result.aspect = ctx.config.project.aspect.value
        result.providers = ctx.config.providers.model_dump()
        for stage in STAGE_ORDER:
            started = time.perf_counter()
            app.run_pipeline(ctx, [stage])
            result.stage_seconds[stage.value] = round(time.perf_counter() - started, 2)
        result.total_seconds = round(sum(result.stage_seconds.values()), 2)
        inspect(ctx, result, out_root)
    except ShortgenError as failure:
        result.error = f"{type(failure).__name__}: {failure}"
    return result


def inspect(ctx: PipelineContext, result: Result, out_root: Path) -> None:
    manifest = ctx.store.load()
    result.scenes = len(manifest.project.scenes)
    result.words = sum(len(scene.narration.split()) for scene in manifest.project.scenes)
    video = ctx.workspace.root / "output/final.mp4"
    result.checks.append(Check("final video exists", video.is_file(), str(video)))
    if not video.is_file():
        return
    result.size_mb = round(video.stat().st_size / 1e6, 1)
    cues_file = ctx.workspace.root / "subtitles/captions.json"
    sheet = (
        CueSheet.model_validate_json(cues_file.read_text(encoding="utf-8"))
        if cues_file.is_file()
        else None
    )
    check_streams(ctx, video, sheet, result)
    check_assets(ctx, result)
    if sheet is not None:
        check_captions(ctx, video, sheet, result)
    started = time.perf_counter()
    reports = app.run_pipeline(ctx)
    result.rerun_seconds = round(time.perf_counter() - started, 2)
    redone = sum(report.completed for report in reports)
    result.checks.append(
        Check(
            "running again does no work",
            redone == 0,
            f"{redone} unit(s) redone in {result.rerun_seconds}s",
        )
    )
    contact_sheet(video, sheet, result.video_seconds, out_root / f"{result.name}-contact.png")
    shutil.copyfile(video, out_root / f"{result.name}.mp4")


def peak_memory_gb() -> tuple[float, float]:
    """Peak resident memory of this process and of its children (ffmpeg), in GB."""
    unit = 1 if sys.platform == "darwin" else 1024  # macOS reports bytes, Linux kilobytes
    own = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * unit / 1e9
    children = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * unit / 1e9
    return round(own, 2), round(children, 2)


def write_report(results: list[Result], out_root: Path) -> Path:
    ffmpeg = run(["ffmpeg", "-version"]).stdout.splitlines()[0]
    own, children = peak_memory_gb()
    lines = [
        "# shortgen end-to-end report",
        "",
        f"- shortgen {__version__}, Python {platform.python_version()}, {platform.system()} {platform.machine()}",
        f"- {ffmpeg}",
        f"- Peak memory: {own} GB in Python (models), {children} GB in ffmpeg",
        f"- Result: **{sum(r.passed for r in results)} of {len(results)} videos passed every check**",
        "",
        "| Video | Format | Scenes | Words | Length | Render time | x real time | Size | Checks |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        ratio = f"{r.total_seconds / r.video_seconds:.1f}" if r.video_seconds else "-"
        verdict = (
            "FAILED: " + r.error
            if r.error
            else f"{sum(c.ok for c in r.checks)}/{len(r.checks)} passed"
        )
        lines.append(
            f"| {r.name} | {r.aspect} | {r.scenes} | {r.words} | {r.video_seconds:.1f} s | {r.total_seconds:.0f} s | {ratio} | {r.size_mb} MB | {verdict} |"
        )
    lines += [
        "",
        "## Seconds per stage",
        "",
        "| Video | " + " | ".join(s.value for s in STAGE_ORDER) + " |",
        "|---|" + "---|" * len(STAGE_ORDER),
    ]
    for r in results:
        lines.append(
            f"| {r.name} | "
            + " | ".join(f"{r.stage_seconds.get(s.value, 0):.1f}" for s in STAGE_ORDER)
            + " |"
        )
    for r in results:
        lines += [
            "",
            f"## {r.name}",
            "",
            f"Providers: {', '.join(f'{k}={v}' for k, v in r.providers.items())}",
            "",
        ]
        lines += [
            f"- {'PASS' if c.ok else 'FAIL'}: {c.name}" + (f" ({c.detail})" if c.detail else "")
            for c in r.checks
        ]
    report = out_root / "report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = [{**asdict(r), "passed": r.passed} for r in results]
    (out_root / "report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--examples", type=Path, default=Path("examples"), help="folder of example projects"
    )
    parser.add_argument("--only", default="", help="comma-separated example names")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("e2e-runs"),
        help="where projects, videos and the report go",
    )
    parser.add_argument(
        "--config", type=Path, default=None, help="extra YAML config applied to every example"
    )
    parser.add_argument(
        "--images",
        choices=("model", "demo"),
        default="model",
        help="demo: bundled art or labelled placeholders instead of the image model",
    )
    parser.add_argument(
        "--fresh", action="store_true", help="delete previous runs first instead of resuming them"
    )
    options = parser.parse_args()
    wanted = {name for name in options.only.split(",") if name}
    sources = sorted(
        p
        for p in options.examples.iterdir()
        if (p / "script.md").is_file() and (not wanted or p.name in wanted)
    )
    options.out.mkdir(parents=True, exist_ok=True)
    results = []
    for source in sources:
        sys.stderr.write(f"\n=== {source.name} ===\n")
        results.append(render_example(source, options.out, options))
    report = write_report(results, options.out)
    sys.stdout.write(report.read_text(encoding="utf-8"))
    return 0 if results and all(result.passed for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
