"""The default set of stages, in one place."""

from __future__ import annotations

from typing import Any

from shortgen.pipeline.render_stages import ComposeStage, RenderStage
from shortgen.pipeline.stage import Stage, StageName
from shortgen.pipeline.stages import (
    AlignStage,
    ImageStage,
    PlanStage,
    SpeechStage,
    SubtitleStage,
    VideoStage,
)

GENERATE_STAGES: tuple[StageName, ...] = (
    StageName.SPEECH,
    StageName.ALIGN,
    StageName.IMAGE,
    StageName.VIDEO,
)
RENDER_STAGES: tuple[StageName, ...] = (
    StageName.SUBTITLES,
    StageName.COMPOSE,
    StageName.RENDER,
)


def default_stages() -> list[Stage[Any]]:
    """One instance of every stage the standard pipeline runs."""
    return [
        PlanStage(),
        SpeechStage(),
        AlignStage(),
        ImageStage(),
        VideoStage(),
        SubtitleStage(),
        ComposeStage(),
        RenderStage(),
    ]
