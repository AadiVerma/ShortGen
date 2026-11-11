from __future__ import annotations

import pytest

from shortgen.config import AppConfig
from shortgen.domain.models import CameraMove, Character
from shortgen.domain.scene_planner import (
    PlanningRules,
    SceneEnrichment,
    apply_enrichment,
    compose_image_prompt,
    draft_scenes,
    keep_author_directions,
    reference_images,
    split_paragraphs,
    split_script,
    split_sentences,
)
from shortgen.errors import PlanningError

MIRA = Character(name="Mira", description="a keeper in a red scarf", reference_image="refs/m.png")


@pytest.fixture
def rules(config: AppConfig) -> PlanningRules:
    return config.planning.model_copy(update={"min_scene_words": 3, "max_scene_words": 8})


def test_markdown_furniture_is_not_narrated() -> None:
    script = "# Title\n\n<!-- note -->\n> A *quiet* night.\n\nSecond `para`."
    assert split_paragraphs(script) == ["A quiet night.", "Second para."]


def test_sentences_keep_abbreviations_and_closing_quotes() -> None:
    text = 'Dr. Lee said "Stop!" Then she left. It cost 3.5 coins'
    assert split_sentences(text) == ['Dr. Lee said "Stop!"', "Then she left.", "It cost 3.5 coins"]


def test_paragraph_breaks_always_start_a_new_scene(rules: PlanningRules) -> None:
    assert split_script("One two.\n\nThree four.", rules) == ["One two.", "Three four."]


def test_sentences_pack_up_to_the_word_limit(rules: PlanningRules) -> None:
    script = "One two three. Four five six. Seven eight nine ten."
    assert split_script(script, rules) == ["One two three. Four five six.", "Seven eight nine ten."]


def test_short_trailing_chunk_folds_into_previous(rules: PlanningRules) -> None:
    script = "One two three four five six seven. Eight nine."
    assert split_script(script, rules) == ["One two three four five six seven. Eight nine."]


def test_long_sentence_splits_at_clauses(rules: PlanningRules) -> None:
    script = "The lamp turned slowly over the water, and far below the tide went out again."
    scenes = split_script(script, rules)
    assert scenes[0] == "The lamp turned slowly over the water,"
    assert all(len(scene.split()) <= rules.max_scene_words for scene in scenes)
    assert " ".join(scenes) == script


def test_narration_is_a_verbatim_partition_of_the_script(config: AppConfig) -> None:
    script = "Mira climbed the stairs. The light was out! Who would warn the ships? She lit it."
    scenes = draft_scenes(script, config.planning, [MIRA])
    assert " ".join(scene.narration for scene in scenes) == script
    assert [scene.id for scene in scenes] == [f"s{n:03d}" for n in range(1, len(scenes) + 1)]
    assert scenes[0].characters == ["Mira"]


def test_camera_moves_cycle_deterministically(rules: PlanningRules) -> None:
    scenes = draft_scenes("A b c.\n\nD e f.\n\nG h i.\n\nJ k l.\n\nM n o.", rules)
    cycle = rules.camera_cycle
    assert [s.shot.camera for s in scenes] == [cycle[i % len(cycle)] for i in range(5)]


def test_empty_script_is_rejected(rules: PlanningRules) -> None:
    with pytest.raises(PlanningError, match="no narratable text"):
        draft_scenes("# Only a heading\n", rules)


def test_enrichment_overrides_only_what_it_provides(rules: PlanningRules) -> None:
    scenes = draft_scenes("One two three.\n\nFour five six.", rules)
    enriched = apply_enrichment(
        scenes,
        [
            SceneEnrichment(image_prompt="a lighthouse at dusk", camera=CameraMove.STATIC),
            SceneEnrichment(),
        ],
    )
    assert enriched[0].shot.image_prompt == "a lighthouse at dusk"
    assert enriched[0].shot.camera is CameraMove.STATIC
    assert enriched[0].narration == scenes[0].narration
    assert enriched[1] == scenes[1]


def test_enrichment_count_must_match(rules: PlanningRules) -> None:
    scenes = draft_scenes("One two three.", rules)
    with pytest.raises(PlanningError, match="0 scene descriptions for 1 scenes"):
        apply_enrichment(scenes, [])


def test_image_prompt_adds_cast_and_style(rules: PlanningRules) -> None:
    scene = draft_scenes("Mira lit the lamp.", rules, [MIRA])[0]
    prompt = compose_image_prompt(scene, "ink wash", [MIRA])
    assert prompt == "Mira lit the lamp, Mira, a keeper in a red scarf, ink wash"
    assert reference_images(scene, [MIRA]) == ["refs/m.png"]


def test_absent_characters_add_nothing(rules: PlanningRules) -> None:
    scene = draft_scenes("The sea was calm.", rules, [MIRA])[0]
    assert compose_image_prompt(scene, "", [MIRA]) == "The sea was calm"
    assert reference_images(scene, [MIRA]) == []


def test_directions_in_comments_set_the_shot_and_are_never_narrated(rules: PlanningRules) -> None:
    script = (
        "# Title\n\n<!-- Format note, ignored. -->\n\n"
        "Meet Odysseus.\n<!-- image: a bearded warrior\n   on a green island -->\n"
        "<!-- Camera: zoom_out -->\n<!-- motion: waves roll in -->\n\n"
        "He went home."
    )
    first, second = draft_scenes(script, rules)
    assert first.narration == "Meet Odysseus."
    assert first.shot.image_prompt == "a bearded warrior on a green island"
    assert first.shot.motion_prompt == "waves roll in"
    assert first.shot.camera is CameraMove.ZOOM_OUT
    assert second.shot.image_prompt == "He went home."  # no direction: narration is the prompt
    assert second.shot.motion_prompt == rules.default_motion_prompt


def test_every_scene_cut_from_a_paragraph_shares_its_direction(rules: PlanningRules) -> None:
    script = (
        "One two three four five six seven. Eight nine ten eleven twelve.\n<!-- image: a lamp -->"
    )
    scenes = draft_scenes(script, rules)
    assert len(scenes) == 2
    assert {scene.shot.image_prompt for scene in scenes} == {"a lamp"}


def test_unknown_camera_direction_is_rejected_with_the_choices(rules: PlanningRules) -> None:
    with pytest.raises(PlanningError, match=r"unknown camera move 'dolly'.*zoom_in"):
        draft_scenes("A b c.\n<!-- camera: dolly -->", rules)


def test_script_directions_win_over_planner_suggestions(rules: PlanningRules) -> None:
    drafts = draft_scenes("One two three.\n<!-- image: my picture -->\n\nFour five six.", rules)
    suggested = apply_enrichment(
        drafts, [SceneEnrichment(image_prompt="llm one"), SceneEnrichment(image_prompt="llm two")]
    )
    kept = keep_author_directions(drafts, suggested)
    assert [scene.shot.image_prompt for scene in kept] == ["my picture", "llm two"]
