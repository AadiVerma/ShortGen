"""The rendering stages: compose one segment per scene, then the final video."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from shortgen.domain.models import AssetKind
from shortgen.domain.subtitles import CueSheet
from shortgen.pipeline.context import PipelineContext
from shortgen.pipeline.contracts import ComposeInputs, RenderInputs
from shortgen.pipeline.manifest import Manifest
from shortgen.pipeline.stage import OutputSpec, Stage, StageName, UnitRun, WorkUnit
from shortgen.pipeline.stages import planned_scenes, scene_seconds, upstream
from shortgen.pipeline.workspace import Workspace
from shortgen.render.captions import write_caption_track
from shortgen.render.ffmpeg import (
    ConcatSpec,
    SegmentSpec,
    concat_command,
    concat_list,
    probe_duration,
    segment_command,
)
from shortgen.render.layout import fit_clip

log = logging.getLogger(__name__)


class ComposeStage(Stage[ComposeInputs]):
    """Clip + narration -> one frame-exact segment per scene."""

    name = StageName.COMPOSE

    def units(self, ctx: PipelineContext, manifest: Manifest) -> list[WorkUnit[ComposeInputs]]:
        render = ctx.config.render
        units: list[WorkUnit[ComposeInputs]] = []
        for scene in planned_scenes(manifest):
            alignment = upstream(manifest, StageName.ALIGN, scene.id, AssetKind.ALIGNMENT)
            inputs = ComposeInputs(
                clip=upstream(manifest, StageName.VIDEO, scene.id, AssetKind.CLIP),
                audio=upstream(manifest, StageName.SPEECH, scene.id, AssetKind.AUDIO),
                duration_s=scene_seconds(ctx, alignment),
                audio_offset_s=ctx.config.timing.pad_start_s,
                target=ctx.config.resolution,
                fps=ctx.config.project.fps,
                max_slowdown=render.max_slowdown,
                encode=render.intermediate,
            )
            outputs = (OutputSpec(AssetKind.SEGMENT, Workspace.segment(scene.id)),)
            units.append(WorkUnit(self.name, scene.id, inputs, outputs))
        return units

    def execute(self, unit: WorkUnit[ComposeInputs], run: UnitRun) -> None:
        inputs = unit.inputs
        ctx = run.ctx
        clip = ctx.workspace.resolve(inputs.clip.path)
        clip_s = probe_duration(ctx.commands, ctx.config.render.ffprobe_bin, clip)
        fit = fit_clip(clip_s, inputs.duration_s, inputs.max_slowdown)
        if fit.hold_s > 0:
            log.info(
                "%s: clip is %.2fs for a %.2fs scene; slowing %.2fx and holding the last frame",
                unit.label,
                clip_s,
                inputs.duration_s,
                fit.slowdown,
            )
        spec = SegmentSpec(
            clip=clip,
            audio=ctx.workspace.resolve(inputs.audio.path),
            output=run.targets[AssetKind.SEGMENT],
            frames=round(inputs.duration_s * inputs.fps),
            fps=inputs.fps,
            audio_offset_s=inputs.audio_offset_s,
            size=inputs.target,
            fit=fit,
            encode=inputs.encode,
        )
        command = segment_command(ctx.config.render.ffmpeg_bin, spec)
        ctx.commands.run(command.args, cwd=command.cwd)


class RenderStage(Stage[RenderInputs]):
    """Segments (+ captions) -> the final MP4."""

    name = StageName.RENDER

    def units(self, ctx: PipelineContext, manifest: Manifest) -> list[WorkUnit[RenderInputs]]:
        settings = ctx.config.subtitles
        captions = None
        if settings.enabled:
            captions = upstream(manifest, StageName.SUBTITLES, None, AssetKind.SUBTITLES_CUES)
        inputs = RenderInputs(
            segments=[
                upstream(manifest, StageName.COMPOSE, scene.id, AssetKind.SEGMENT)
                for scene in planned_scenes(manifest)
            ],
            captions=captions,
            caption_style=settings.style if settings.enabled else None,
            encode=ctx.config.render.final,
        )
        outputs = (OutputSpec(AssetKind.VIDEO, Workspace.final_video),)
        return [WorkUnit(self.name, None, inputs, outputs)]

    def execute(self, unit: WorkUnit[RenderInputs], run: UnitRun) -> None:
        inputs = unit.inputs
        workspace = run.ctx.workspace
        output = run.targets[AssetKind.VIDEO]
        segments = [workspace.resolve(ref.path) for ref in inputs.segments]
        # Scratch files (segment list, caption frames) live in a temp directory
        # beside the output and vanish with it, whatever happens.
        with tempfile.TemporaryDirectory(dir=output.parent, prefix=".render-") as scratch:
            list_file = Path(scratch) / "segments.ffconcat"
            list_file.write_text(concat_list(segments), encoding="utf-8")
            track, frames = self._captions(inputs, workspace, Path(scratch))
            spec = ConcatSpec(
                list_file=list_file,
                output=output,
                caption_track=track,
                frames=frames,
                encode=inputs.encode,
            )
            command = concat_command(run.ctx.config.render.ffmpeg_bin, spec)
            run.ctx.commands.run(command.args, cwd=command.cwd)

    @staticmethod
    def _captions(
        inputs: RenderInputs, workspace: Workspace, scratch: Path
    ) -> tuple[Path | None, int | None]:
        """Rasterise the cue sheet; returns the caption timeline and the exact frame count."""
        if inputs.captions is None or inputs.caption_style is None:
            return None, None
        text = workspace.resolve(inputs.captions.path).read_text(encoding="utf-8")
        sheet = CueSheet.model_validate_json(text)
        frames = round(sheet.duration_s * sheet.fps) or None
        if not sheet.cues:
            return None, frames
        return write_caption_track(sheet, inputs.caption_style, scratch / "captions"), frames
