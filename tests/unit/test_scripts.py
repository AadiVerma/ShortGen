"""Scripts shipped in ``examples/`` and ``stories/`` must plan and direct every picture."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from shortgen.config import load_config
from shortgen.domain.models import Scene
from shortgen.domain.scene_planner import compose_image_prompt, draft_scenes

ROOT = Path(__file__).resolve().parents[2]
PROJECTS = sorted(
    folder
    for parent in ("examples", "stories")
    for folder in (ROOT / parent).iterdir()
    if (folder / "script.md").is_file()
)
# SDXL's text encoders read 77 tokens. Words plus punctuation marks undercount CLIP
# tokens a little, so the budget leaves room.
PROMPT_BUDGET = 62


def label(folder: Path) -> str:
    return f"{folder.parent.name}/{folder.name}"


# Words that made the image model draw fake text in the first real run.
TEXT_BAIT = ("poster", "infographic", "diagram", "label", "sign")
# A genre name brings its own subjects ("retro space illustration" drew a man on an alien
# planet), and "no people" in a prompt is read as "people".
STYLE_TRAPS = {"retro", "vintage", "poster", "no", "without"}


def plan(folder: Path) -> tuple[list[Scene], list[str]]:
    config = load_config(project_dir=folder)
    script = (folder / "script.md").read_text(encoding="utf-8")
    scenes = draft_scenes(script, config.planning, config.characters)
    prompts = [
        compose_image_prompt(scene, config.project.style, config.characters) for scene in scenes
    ]
    return scenes, prompts


def test_there_are_scripts_to_check() -> None:
    assert {folder.parent.name for folder in PROJECTS} == {"examples", "stories"}


@pytest.mark.parametrize(
    "folder", PROJECTS, ids=lambda folder: f"{folder.parent.name}/{folder.name}"
)
def test_every_scene_has_its_own_picture_description(folder: Path) -> None:
    scenes, _ = plan(folder)
    assert len(scenes) >= 5
    for scene in scenes:
        assert scene.shot.image_prompt != scene.narration, f"{scene.id} has no image: line"


@pytest.mark.parametrize(
    "folder", PROJECTS, ids=lambda folder: f"{folder.parent.name}/{folder.name}"
)
def test_picture_prompts_fit_the_image_model(folder: Path) -> None:
    _, prompts = plan(folder)
    for prompt in prompts:
        size = len(re.findall(r"\w+|[^\w\s]", prompt))
        assert size <= PROMPT_BUDGET, f"{size} tokens, the style would be cut off: {prompt}"


@pytest.mark.parametrize(
    "folder",
    [folder for folder in PROJECTS if folder.parent.name == "stories"],
    ids=lambda folder: folder.name,
)
def test_stories_follow_what_the_first_real_run_taught(folder: Path) -> None:
    config = load_config(project_dir=folder)
    scenes, prompts = plan(folder)
    assert config.project.aspect == "9:16"
    assert "text" in config.project.negative_prompt
    style_words = set(re.findall(r"[a-z]+", config.project.style.lower()))
    assert not style_words & STYLE_TRAPS, config.project.style
    for prompt in prompts:
        assert not any(word in prompt.lower().split() for word in TEXT_BAIT), prompt
    words = sum(len(scene.narration.split()) for scene in scenes)
    assert 50 <= words <= 140, "a Short of 25 to 55 seconds"
    assert f"`{folder.name}`" in (ROOT / "stories" / "README.md").read_text(encoding="utf-8")
