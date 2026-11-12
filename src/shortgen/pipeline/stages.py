"""The generation stages: plan, speech, align, image, video and subtitles.

Each stage only knows provider *interfaces*. ``units`` describes the work as
typed, hashable inputs; ``execute`` is a function of those inputs alone.
"""

from __future__ import annotations

from pathlib import Path

from shortgen.domain.models import AssetKind, Scene
from shortgen.domain.scene_planner import compose_image_prompt, reference_images
from shortgen.domain.subtitles import (
    Cue,
    CueSheet,
    build_cues,
    chars_per_line,
    clamp_cues,
    render_ass,
    render_srt,
)
from shortgen.domain.timing import align_to_script, build_timeline, scene_duration, shift_words
from shortgen.errors import StageError
from shortgen.media import wav_duration
from shortgen.pipeline.context import PipelineContext
from shortgen.pipeline.contracts import (
    AlignInputs,
    FileRef,
    ImageInputs,
    NativeWords,
    PlanInputs,
    SceneAlignment,
    SpeechInputs,
    SubtitleInputs,
    VideoInputs,
)
from shortgen.pipeline.hashing import file_sha256
from shortgen.pipeline.manifest import Manifest
from shortgen.pipeline.stage import OutputSpec, Stage, StageName, UnitRun, WorkUnit
from shortgen.pipeline.workspace import IMAGE_SUFFIXES, Workspace, validate_input_file
from shortgen.providers.interfaces import (
    ImageGenerator,
    ImageRequest,
    PlanRequest,
    ScriptPlanner,
    SpeechRequest,
    SpeechSynthesizer,
    Transcriber,
    VideoGenerator,
    VideoRequest,
)
from shortgen.providers.spec import ProviderKind

NATIVE_SOURCE = "native"


def upstream(
    manifest: Manifest, stage: StageName, scene_id: str | None, kind: AssetKind
) -> FileRef:
    """Reference to a finished upstream output; raises ``StageError`` if it is missing."""
    asset = manifest.asset(stage, scene_id, kind)
    return FileRef(path=asset.path, sha256=asset.sha256)


def read_alignment(workspace: Workspace, ref: FileRef) -> SceneAlignment:
    """Load a scene's alignment file."""
    text = workspace.resolve(ref.path).read_text(encoding="utf-8")
    return SceneAlignment.model_validate_json(text)


def scene_seconds(ctx: PipelineContext, alignment: FileRef) -> float:
    """Timeline length of a scene, derived from its measured narration length."""
    audio_s = read_alignment(ctx.workspace, alignment).duration_s
    return scene_duration(audio_s, ctx.config.timing, ctx.config.project.fps)


def reference_path(workspace: Workspace, raw: str) -> Path:
    """Resolve a configured reference image (project-relative or absolute) and vet it."""
    return validate_input_file(workspace.root / Path(raw).expanduser(), IMAGE_SUFFIXES)


def planned_scenes(manifest: Manifest) -> list[Scene]:
    """The project's scenes; raises ``StageError`` if planning has not happened."""
    if not manifest.project.scenes:
        raise StageError("the project has no scenes yet; run the plan stage first")
    return manifest.project.scenes


class PlanStage(Stage[PlanInputs]):
    """Script (or topic) -> scenes, stored in the manifest itself."""

    name = StageName.PLAN
    provider_kind = ProviderKind.PLANNER

    def units(self, ctx: PipelineContext, manifest: Manifest) -> list[WorkUnit[PlanInputs]]:
        project = manifest.project
        inputs = PlanInputs(
            # A script written from a topic must not change the hash that produced it.
            script="" if project.topic else project.script,
            topic=project.topic,
            language=ctx.config.project.language,
            seed=ctx.config.project.seed,
            rules=ctx.config.planning,
            characters=ctx.config.characters,
            provider=ctx.fingerprint(ProviderKind.PLANNER),
        )
        return [WorkUnit(self.name, None, inputs)]

    def execute(self, unit: WorkUnit[PlanInputs], run: UnitRun) -> None:
        planner = run.provider.get(ScriptPlanner)
        project = run.manifest.project
        inputs = unit.inputs
        if inputs.topic and not project.script.strip():
            target = inputs.rules.topic_target_words
            project.script = planner.write_script(inputs.topic, target, inputs.language)
            run.ctx.workspace.script_path.write_text(project.script + "\n", encoding="utf-8")
        request = PlanRequest(
            script=project.script,
            language=inputs.language,
            rules=inputs.rules,
            characters=inputs.characters,
            seed=inputs.seed,
        )
        project.scenes = planner.plan(request)


class SpeechStage(Stage[SpeechInputs]):
    """Narration text -> one WAV per scene, plus any native word timings."""

    name = StageName.SPEECH
    provider_kind = ProviderKind.SPEECH

    def units(self, ctx: PipelineContext, manifest: Manifest) -> list[WorkUnit[SpeechInputs]]:
        provider = ctx.fingerprint(ProviderKind.SPEECH)
        speech = ctx.config.speech
        return [
            WorkUnit(
                self.name,
                scene.id,
                SpeechInputs(
                    text=scene.narration,
                    voice=speech.voice,
                    speed=speech.speed,
                    language=ctx.config.project.language,
                    provider=provider,
                ),
                (
                    OutputSpec(AssetKind.AUDIO, Workspace.audio(scene.id)),
                    OutputSpec(AssetKind.WORDS, Workspace.words(scene.id)),
                ),
            )
            for scene in planned_scenes(manifest)
        ]

    def execute(self, unit: WorkUnit[SpeechInputs], run: UnitRun) -> None:
        inputs = unit.inputs
        audio_path = run.targets[AssetKind.AUDIO]
        result = run.provider.get(SpeechSynthesizer).synthesize(
            SpeechRequest(
                text=inputs.text,
                voice=inputs.voice,
                speed=inputs.speed,
                language=inputs.language,
                output_path=audio_path,
            )
        )
        wav_duration(audio_path)  # fail now, not three stages later, if the audio is unusable
        native = NativeWords(words=result.words)
        run.targets[AssetKind.WORDS].write_text(native.model_dump_json(), encoding="utf-8")


class AlignStage(Stage[AlignInputs]):
    """Audio -> script words with timings. Loads a transcriber only if it must."""

    name = StageName.ALIGN
    provider_kind = ProviderKind.TRANSCRIBER

    def units(self, ctx: PipelineContext, manifest: Manifest) -> list[WorkUnit[AlignInputs]]:
        provider = ctx.fingerprint(ProviderKind.TRANSCRIBER)
        return [
            WorkUnit(
                self.name,
                scene.id,
                AlignInputs(
                    narration=scene.narration,
                    audio=upstream(manifest, StageName.SPEECH, scene.id, AssetKind.AUDIO),
                    native_words=upstream(manifest, StageName.SPEECH, scene.id, AssetKind.WORDS),
                    language=ctx.config.project.language,
                    provider=provider,
                ),
                (OutputSpec(AssetKind.ALIGNMENT, Workspace.alignment(scene.id)),),
            )
            for scene in planned_scenes(manifest)
        ]

    def execute(self, unit: WorkUnit[AlignInputs], run: UnitRun) -> None:
        inputs = unit.inputs
        workspace = run.ctx.workspace
        audio_path = workspace.resolve(inputs.audio.path)
        duration = wav_duration(audio_path)
        native = NativeWords.model_validate_json(
            workspace.resolve(inputs.native_words.path).read_text(encoding="utf-8")
        )
        if native.words:
            heard, source = native.words, NATIVE_SOURCE
        else:
            transcriber = run.provider.get(Transcriber)
            heard, source = transcriber.transcribe(audio_path, inputs.language), transcriber.name
        alignment = SceneAlignment(
            duration_s=duration,
            source=source,
            words=align_to_script(inputs.narration, heard, duration),
        )
        run.targets[AssetKind.ALIGNMENT].write_text(alignment.model_dump_json(), encoding="utf-8")


class ImageStage(Stage[ImageInputs]):
    """Shot description -> one illustration per scene."""

    name = StageName.IMAGE
    provider_kind = ProviderKind.IMAGE

    def units(self, ctx: PipelineContext, manifest: Manifest) -> list[WorkUnit[ImageInputs]]:
        provider = ctx.fingerprint(ProviderKind.IMAGE)
        generator = ctx.create_provider(ProviderKind.IMAGE)  # cheap: nothing is loaded
        if not isinstance(generator, ImageGenerator):
            raise StageError(f"{generator.name} is not an image generator")
        project = ctx.config.project
        return [
            WorkUnit(
                self.name,
                scene.id,
                ImageInputs(
                    prompt=compose_image_prompt(scene, project.style, ctx.config.characters),
                    negative_prompt=scene.shot.negative_prompt or project.negative_prompt,
                    target=ctx.config.resolution,
                    seed=project.seed + scene.index,
                    references=self._references(ctx, scene),
                    source=generator.source_fingerprint(scene.id),
                    provider=provider,
                ),
                (OutputSpec(AssetKind.IMAGE, Workspace.image(scene.id)),),
            )
            for scene in planned_scenes(manifest)
        ]

    @staticmethod
    def _references(ctx: PipelineContext, scene: Scene) -> list[FileRef]:
        """Validate and hash the reference images of the characters in a scene."""
        return [
            FileRef(path=raw, sha256=file_sha256(reference_path(ctx.workspace, raw)))
            for raw in reference_images(scene, ctx.config.characters)
        ]

    def execute(self, unit: WorkUnit[ImageInputs], run: UnitRun) -> None:
        inputs = unit.inputs
        workspace = run.ctx.workspace
        run.provider.get(ImageGenerator).generate(
            ImageRequest(
                scene_id=unit.scene_id or "",
                prompt=inputs.prompt,
                negative_prompt=inputs.negative_prompt,
                target=inputs.target,
                seed=inputs.seed,
                reference_images=[reference_path(workspace, ref.path) for ref in inputs.references],
                output_path=run.targets[AssetKind.IMAGE],
            )
        )


class VideoStage(Stage[VideoInputs]):
    """Illustration -> a clip as long as the scene's narration needs."""

    name = StageName.VIDEO
    provider_kind = ProviderKind.VIDEO

    def units(self, ctx: PipelineContext, manifest: Manifest) -> list[WorkUnit[VideoInputs]]:
        provider = ctx.fingerprint(ProviderKind.VIDEO)
        project = ctx.config.project
        units: list[WorkUnit[VideoInputs]] = []
        for scene in planned_scenes(manifest):
            alignment = upstream(manifest, StageName.ALIGN, scene.id, AssetKind.ALIGNMENT)
            inputs = VideoInputs(
                image=upstream(manifest, StageName.IMAGE, scene.id, AssetKind.IMAGE),
                prompt=scene.shot.motion_prompt,
                camera=scene.shot.camera,
                duration_s=scene_seconds(ctx, alignment),
                fps=project.fps,
                target=ctx.config.resolution,
                seed=project.seed + scene.index,
                provider=provider,
            )
            outputs = (OutputSpec(AssetKind.CLIP, Workspace.clip(scene.id)),)
            units.append(WorkUnit(self.name, scene.id, inputs, outputs))
        return units

    def execute(self, unit: WorkUnit[VideoInputs], run: UnitRun) -> None:
        inputs = unit.inputs
        run.provider.get(VideoGenerator).generate(
            VideoRequest(
                image_path=run.ctx.workspace.resolve(inputs.image.path),
                prompt=inputs.prompt,
                camera=inputs.camera,
                duration_s=inputs.duration_s,
                fps=inputs.fps,
                target=inputs.target,
                seed=inputs.seed,
                output_path=run.targets[AssetKind.CLIP],
            )
        )


class SubtitleStage(Stage[SubtitleInputs]):
    """Per-scene alignments -> cues on the global timeline, as JSON, SRT and ASS.

    The JSON cue sheet is what the render stage burns in; SRT and ASS are
    sidecar files for upload forms and video players.
    """

    name = StageName.SUBTITLES

    def units(self, ctx: PipelineContext, manifest: Manifest) -> list[WorkUnit[SubtitleInputs]]:
        settings = ctx.config.subtitles
        if not settings.enabled:
            return []
        inputs = SubtitleInputs(
            alignments=[
                upstream(manifest, StageName.ALIGN, scene.id, AssetKind.ALIGNMENT)
                for scene in planned_scenes(manifest)
            ],
            timing=ctx.config.timing,
            rules=settings.rules,
            style=settings.style,
            target=ctx.config.resolution,
            fps=ctx.config.project.fps,
        )
        outputs = (
            OutputSpec(AssetKind.SUBTITLES_ASS, Workspace.subtitles_ass),
            OutputSpec(AssetKind.SUBTITLES_SRT, Workspace.subtitles_srt),
            OutputSpec(AssetKind.SUBTITLES_CUES, Workspace.subtitles_cues),
        )
        return [WorkUnit(self.name, None, inputs, outputs)]

    def execute(self, unit: WorkUnit[SubtitleInputs], run: UnitRun) -> None:
        inputs = unit.inputs
        size = inputs.target
        sheet = self._cue_sheet(inputs, run.ctx.workspace)
        ass = render_ass(sheet.cues, inputs.style, size.width, size.height)
        run.targets[AssetKind.SUBTITLES_ASS].write_text(ass, encoding="utf-8")
        run.targets[AssetKind.SUBTITLES_SRT].write_text(render_srt(sheet.cues), encoding="utf-8")
        run.targets[AssetKind.SUBTITLES_CUES].write_text(sheet.model_dump_json(), encoding="utf-8")

    @staticmethod
    def _cue_sheet(inputs: SubtitleInputs, workspace: Workspace) -> CueSheet:
        """Build cues scene by scene so a caption never straddles a cut."""
        alignments = [read_alignment(workspace, ref) for ref in inputs.alignments]
        durations = [scene_duration(a.duration_s, inputs.timing, inputs.fps) for a in alignments]
        size = inputs.target
        max_chars = chars_per_line(size.width, size.height, inputs.style)
        cues: list[Cue] = []
        for alignment, span in zip(alignments, build_timeline(durations), strict=True):
            words = shift_words(alignment.words, span.start + inputs.timing.pad_start_s)
            cues.extend(clamp_cues(build_cues(words, inputs.rules, max_chars), span))
        return CueSheet(
            width=size.width,
            height=size.height,
            fps=inputs.fps,
            duration_s=sum(durations),
            cues=cues,
        )
