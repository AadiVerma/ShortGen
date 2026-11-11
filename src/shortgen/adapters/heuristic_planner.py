"""Rule-based planner: no model at all."""

from __future__ import annotations

from shortgen.domain.models import Scene
from shortgen.domain.scene_planner import draft_scenes
from shortgen.errors import UnsupportedFeatureError
from shortgen.providers.interfaces import PlanRequest, ScriptPlanner


class HeuristicPlanner(ScriptPlanner):
    """Splits on paragraphs and sentences; the narration doubles as the image prompt."""

    def write_script(self, topic: str, target_words: int, language: str) -> str:
        raise UnsupportedFeatureError(
            "the heuristic planner cannot write a script from a topic; "
            "pass a script file, or set providers.planner to an LLM planner such as 'ollama'"
        )

    def plan(self, request: PlanRequest) -> list[Scene]:
        return draft_scenes(request.script, request.rules, request.characters)
