"""Deterministic script splitting and prompt assembly.

The narration of a scene is always a verbatim slice of the script. An LLM
planner may *enrich* scenes (better image prompts, camera moves) but never
rewrites what is spoken, so the words on screen always match the audio.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from pydantic import Field

from shortgen.domain.models import CameraMove, Character, FrozenModel, Scene, Shot, scene_id
from shortgen.errors import PlanningError

_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_DIRECTIVE = re.compile(
    r"<!--\s*(image|motion|camera)\s*:\s*(.*?)\s*-->", re.DOTALL | re.IGNORECASE
)
_MARK = "\x00"  # brackets a direction's index while the text around it is cleaned
_MARKED = re.compile(r"\x00(\d+)\x00")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s.*$", re.MULTILINE)
_EMPHASIS = re.compile(r"[*`]+|(?<!\w)_+|_+(?!\w)")
_QUOTE_MARK = re.compile(r"^\s*>\s?", re.MULTILINE)
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_TERMINATOR = re.compile(r"([.!?\u2026]+[\"'\u201d\u2019)\]]*)(\s+|$)")
_CLAUSE_BREAK = re.compile(r"(?<=[,;:\u2014\u2013])\s+")
_ABBREVIATIONS = frozenset({"mr.", "mrs.", "ms.", "dr.", "st.", "vs.", "etc.", "e.g.", "i.e."})


class PlanningRules(FrozenModel):
    """How a script is cut into scenes. All values come from configuration."""

    min_scene_words: int = Field(ge=1)
    max_scene_words: int = Field(ge=1)
    words_per_minute: int = Field(ge=60)
    topic_target_words: int = Field(ge=20)
    default_motion_prompt: str
    camera_cycle: list[CameraMove] = Field(min_length=1)


class SceneEnrichment(FrozenModel):
    """Optional visual direction for one scene, typically written by an LLM."""

    image_prompt: str = ""
    motion_prompt: str = ""
    camera: CameraMove | None = None


def split_directed_paragraphs(script: str) -> list[tuple[str, SceneEnrichment]]:
    """Narratable paragraphs, each with the visual directions written beside it.

    A paragraph may carry directions in HTML comments, which are never narrated:

        Meet Odysseus, king of a small island called Ithaca.
        <!-- image: a bearded Greek warrior in a bronze helmet on a green island -->
        <!-- camera: zoom_in -->

    ``image`` replaces the narration as the picture prompt, ``motion`` and
    ``camera`` do the same for movement. Any other comment is simply dropped.
    """
    directions: list[tuple[str, str]] = []

    def stash(match: re.Match[str]) -> str:
        directions.append((match.group(1).lower(), " ".join(match.group(2).split())))
        return f"{_MARK}{len(directions) - 1}{_MARK}"

    text = _COMMENT.sub("", _DIRECTIVE.sub(stash, script))
    text = _EMPHASIS.sub("", _QUOTE_MARK.sub("", _HEADING.sub("", text)))
    paragraphs: list[tuple[str, SceneEnrichment]] = []
    for block in _PARAGRAPH_BREAK.split(text):
        found = [directions[int(index)] for index in _MARKED.findall(block)]
        narration = " ".join(_MARKED.sub(" ", block).split())
        if narration:
            paragraphs.append((narration, _direction(found)))
    return paragraphs


def split_paragraphs(script: str) -> list[str]:
    """Strip Markdown furniture and return the narratable paragraphs."""
    return [narration for narration, _ in split_directed_paragraphs(script)]


def _direction(found: Sequence[tuple[str, str]]) -> SceneEnrichment:
    values = dict(found)
    camera = values.get("camera")
    if camera is not None and camera not in {move.value for move in CameraMove}:
        allowed = ", ".join(move.value for move in CameraMove)
        raise PlanningError(f"unknown camera move '{camera}' in the script; use one of: {allowed}")
    return SceneEnrichment(
        image_prompt=values.get("image", ""),
        motion_prompt=values.get("motion", ""),
        camera=CameraMove(camera) if camera else None,
    )


def split_sentences(paragraph: str) -> list[str]:
    """Split a paragraph into sentences, keeping common abbreviations intact."""
    sentences: list[str] = []
    start = 0
    for match in _TERMINATOR.finditer(paragraph):
        candidate = paragraph[start : match.end(1)].strip()
        if not candidate or candidate.split()[-1].lower() in _ABBREVIATIONS:
            continue
        sentences.append(candidate)
        start = match.end()
    tail = paragraph[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def split_long_sentence(sentence: str, max_words: int) -> list[str]:
    """Break an over-long sentence at clause boundaries, then by word count."""
    if len(sentence.split()) <= max_words:
        return [sentence]
    pieces: list[str] = []
    for clause in _pack(_CLAUSE_BREAK.split(sentence), max_words):
        words = clause.split()
        pieces.extend(" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words))
    return pieces


def split_script(script: str, rules: PlanningRules) -> list[str]:
    """Return the narration of each scene, in order."""
    return [narration for narration, _ in split_directed_script(script, rules)]


def split_directed_script(script: str, rules: PlanningRules) -> list[tuple[str, SceneEnrichment]]:
    """Narration and visual directions of each scene, in order.

    Paragraph breaks are always scene breaks. Inside a paragraph, sentences are
    packed greedily up to ``max_scene_words``; a trailing chunk shorter than
    ``min_scene_words`` is folded into the previous one (so a chunk may exceed
    the maximum by at most ``min_scene_words - 1`` words). Every scene cut from
    a paragraph inherits that paragraph's directions.
    """
    scenes: list[tuple[str, SceneEnrichment]] = []
    for paragraph, direction in split_directed_paragraphs(script):
        pieces = [
            piece
            for sentence in split_sentences(paragraph)
            for piece in split_long_sentence(sentence, rules.max_scene_words)
        ]
        chunks = _pack(pieces, rules.max_scene_words)
        if len(chunks) > 1 and len(chunks[-1].split()) < rules.min_scene_words:
            chunks[-2:] = [f"{chunks[-2]} {chunks[-1]}"]
        scenes.extend((chunk, direction) for chunk in chunks)
    return scenes


def find_characters(narration: str, characters: Sequence[Character]) -> list[str]:
    """Names of the known characters mentioned in ``narration`` (whole words)."""
    return [
        character.name
        for character in characters
        if re.search(rf"\b{re.escape(character.name)}\b", narration, re.IGNORECASE)
    ]


def draft_scenes(
    script: str, rules: PlanningRules, characters: Sequence[Character] = ()
) -> list[Scene]:
    """Build a complete plan without any model.

    A scene uses the directions written in the script when there are any;
    otherwise its narration doubles as the image prompt.
    """
    directed = split_directed_script(script, rules)
    if not directed:
        raise PlanningError("the script contains no narratable text")
    return [
        Scene(
            id=scene_id(index),
            index=index,
            narration=narration,
            shot=Shot(
                image_prompt=direction.image_prompt or narration,
                motion_prompt=direction.motion_prompt or rules.default_motion_prompt,
                camera=direction.camera or rules.camera_cycle[index % len(rules.camera_cycle)],
            ),
            characters=find_characters(narration, characters),
        )
        for index, (narration, direction) in enumerate(directed)
    ]


def keep_author_directions(drafts: Sequence[Scene], enriched: Sequence[Scene]) -> list[Scene]:
    """Let image prompts written in the script win over a planner's suggestions."""
    return [
        planned.model_copy(update={"shot": draft.shot})
        if draft.shot.image_prompt != draft.narration
        else planned
        for draft, planned in zip(drafts, enriched, strict=True)
    ]


def apply_enrichment(
    scenes: Sequence[Scene], enrichments: Sequence[SceneEnrichment]
) -> list[Scene]:
    """Overlay visual direction onto drafted scenes; blank fields keep the draft."""
    if len(scenes) != len(enrichments):
        raise PlanningError(
            f"planner returned {len(enrichments)} scene descriptions for {len(scenes)} scenes"
        )
    enriched: list[Scene] = []
    for scene, extra in zip(scenes, enrichments, strict=True):
        shot = Shot(
            image_prompt=extra.image_prompt.strip() or scene.shot.image_prompt,
            motion_prompt=extra.motion_prompt.strip() or scene.shot.motion_prompt,
            camera=extra.camera or scene.shot.camera,
        )
        enriched.append(scene.model_copy(update={"shot": shot}))
    return enriched


def compose_image_prompt(scene: Scene, style: str, characters: Sequence[Character]) -> str:
    """Join the shot, the descriptions of the characters in it, and the global style."""
    parts = [scene.shot.image_prompt.strip().rstrip(".")]
    parts.extend(
        f"{character.name}, {character.description}"
        for character in _cast(scene, characters)
        if character.description
    )
    if style.strip():
        parts.append(style.strip())
    return ", ".join(parts)


def reference_images(scene: Scene, characters: Sequence[Character]) -> list[str]:
    """Reference image paths for the characters that appear in ``scene``."""
    return [c.reference_image for c in _cast(scene, characters) if c.reference_image]


def _cast(scene: Scene, characters: Sequence[Character]) -> list[Character]:
    return [character for character in characters if character.name in scene.characters]


def _pack(pieces: Sequence[str], max_words: int) -> list[str]:
    """Greedily join consecutive pieces while the total stays within ``max_words``."""
    chunks: list[str] = []
    current: list[str] = []
    count = 0
    for piece in pieces:
        words = len(piece.split())
        if current and count + words > max_words:
            chunks.append(" ".join(current))
            current, count = [], 0
        current.append(piece)
        count += words
    if current:
        chunks.append(" ".join(current))
    return chunks
