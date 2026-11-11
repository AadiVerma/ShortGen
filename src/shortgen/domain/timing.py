"""Timing arithmetic: scene lengths, the global timeline and word alignment."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from difflib import SequenceMatcher

from pydantic import Field

from shortgen.domain.models import FrozenModel, Timing, WordTiming

_NON_WORD = re.compile(r"[^\w']+")
_SECONDS_PER_MINUTE = 60.0
_MAX_FUSED = 3  # written words one pronounced unit may cover ("on the", "did not")
_MAX_EXPANDED = 4  # pronounced units one written word may need ("3.5", "2026")
_DIGIT_WEIGHT = 4
_SYMBOL_WEIGHT = 6
_SPOKEN_SYMBOLS = frozenset("%$&@+=")
_REGROUP_PENALTY = 0.5  # in phonemes; makes one-to-one the default reading


class SpokenSpan(FrozenModel):
    """When one pronounced unit was heard, and how long it is in phonemes."""

    start: float = Field(ge=0)
    end: float = Field(ge=0)
    size: int = Field(ge=1)


class TimingRules(FrozenModel):
    """Breathing room around each scene's narration, in seconds."""

    pad_start_s: float = Field(ge=0)
    pad_end_s: float = Field(ge=0)
    min_scene_s: float = Field(gt=0)


def count_words(text: str) -> int:
    """Number of whitespace-separated words."""
    return len(text.split())


def estimate_duration(text: str, words_per_minute: int) -> float:
    """Rough spoken length of ``text``; used only before real audio exists."""
    return count_words(text) / words_per_minute * _SECONDS_PER_MINUTE


def scene_duration(audio_s: float, rules: TimingRules, fps: int) -> float:
    """Length of a scene on the timeline, rounded up to a whole number of frames.

    Whole-frame durations keep every rendered segment exactly as long as the
    timeline says, so subtitle timing cannot drift across scenes.
    """
    padded = max(rules.min_scene_s, rules.pad_start_s + audio_s + rules.pad_end_s)
    return math.ceil(round(padded * fps, 6)) / fps


def build_timeline(durations: Sequence[float]) -> list[Timing]:
    """Lay scenes end to end and return each scene's interval."""
    timeline: list[Timing] = []
    cursor = 0.0
    for duration in durations:
        timeline.append(Timing(start=cursor, end=cursor + duration))
        cursor += duration
    return timeline


def shift_words(words: Sequence[WordTiming], offset: float) -> list[WordTiming]:
    """Move word timings from scene-local time onto the global timeline."""
    return [WordTiming(word=w.word, start=w.start + offset, end=w.end + offset) for w in words]


def script_tokens(text: str) -> list[str]:
    """Display words of ``text``; stray punctuation is glued to its neighbour."""
    tokens: list[str] = []
    pending = ""
    for raw in text.split():
        if _normalize(raw):
            tokens.append(f"{pending}{raw}")
            pending = ""
        elif tokens:
            tokens[-1] = f"{tokens[-1]} {raw}"
        else:
            pending = f"{pending}{raw} "
    return tokens


def proportional_word_timings(text: str, duration: float) -> list[WordTiming]:
    """Estimate word timings by spreading ``duration`` in proportion to word length."""
    return _spread(script_tokens(text), 0.0, max(duration, 0.0))


def align_to_script(text: str, heard: Sequence[WordTiming], duration: float) -> list[WordTiming]:
    """Put heard timings onto the script's own words.

    Recognisers mishear, merge and drop words, but subtitles must show the
    script. Words that match keep their measured timing; runs that do not are
    spread across the gap between their nearest matched neighbours. With
    nothing heard, the whole text is spread across ``duration``.
    """
    tokens = script_tokens(text)
    spoken = [word for word in heard if _normalize(word.word)]
    if not tokens:
        return []
    if not spoken:
        return _spread(tokens, 0.0, max(duration, 0.0))
    matcher = SequenceMatcher(
        a=[_normalize(token) for token in tokens],
        b=[_normalize(word.word) for word in spoken],
        autojunk=False,
    )
    anchors = {
        block.a + offset: spoken[block.b + offset]
        for block in matcher.get_matching_blocks()
        for offset in range(block.size)
    }
    return _fill_gaps(tokens, anchors, duration)


def assign_spans(tokens: Sequence[str], spans: Sequence[SpokenSpan]) -> list[WordTiming]:
    """Give each script word a timing from spans measured for *pronounced* words.

    A phonemiser does not pronounce exactly the words that were written: it
    fuses clitics ("on the" becomes one unit) and expands numbers ("3.5"
    becomes three). This finds the cheapest in-order pairing where one span
    covers up to ``_MAX_FUSED`` words or one word covers up to ``_MAX_EXPANDED``
    spans, comparing lengths. Fused words share their span in proportion to
    their length. Returns an empty list if no pairing exists.
    """
    if not tokens or not spans:
        return []
    letters = [_spoken_weight(token) for token in tokens]
    scale = sum(span.size for span in spans) / sum(letters)
    moves = [(count, 1) for count in range(1, _MAX_FUSED + 1)]
    moves += [(1, count) for count in range(2, _MAX_EXPANDED + 1)]
    unreachable = float("inf")
    best = [[unreachable] * (len(spans) + 1) for _ in range(len(tokens) + 1)]
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    best[0][0] = 0.0
    for i in range(len(tokens) + 1):
        for j in range(len(spans) + 1):
            if best[i][j] == unreachable:
                continue
            for words, units in moves:
                if i + words > len(tokens) or j + units > len(spans):
                    continue
                written = scale * sum(letters[i : i + words])
                spoken = sum(span.size for span in spans[j : j + units])
                cost = best[i][j] + abs(written - spoken) + _REGROUP_PENALTY * (words + units - 2)
                if cost < best[i + words][j + units]:
                    best[i + words][j + units] = cost
                    came_from[(i + words, j + units)] = (i, j)
    if best[len(tokens)][len(spans)] == unreachable:
        return []
    return _timings_along(tokens, letters, spans, came_from)


def _spoken_weight(token: str) -> int:
    """Rough length of a written word when read aloud, in letters.

    Digits and symbols are short to write and long to say ("7" is "seven",
    "%" is "percent"), so they count for more than one letter.
    """
    weight = 0
    for char in token:
        if char.isdigit():
            weight += _DIGIT_WEIGHT
        elif char in _SPOKEN_SYMBOLS:
            weight += _SYMBOL_WEIGHT
        elif char.isalpha():
            weight += 1
    return max(1, weight)


def _timings_along(
    tokens: Sequence[str],
    letters: Sequence[int],
    spans: Sequence[SpokenSpan],
    came_from: dict[tuple[int, int], tuple[int, int]],
) -> list[WordTiming]:
    """Walk the chosen pairing backwards and turn each step into word timings."""
    steps: list[tuple[int, int, int, int]] = []
    position = (len(tokens), len(spans))
    while position != (0, 0):
        previous = came_from[position]
        steps.append((*previous, *position))
        position = previous
    words: list[WordTiming] = []
    for i, j, next_i, next_j in reversed(steps):
        start, end = spans[j].start, max(spans[j].start, spans[next_j - 1].end)
        cursor = start
        total = sum(letters[i:next_i])
        for index in range(i, next_i):
            share = (end - start) * letters[index] / total
            words.append(WordTiming(word=tokens[index], start=cursor, end=cursor + share))
            cursor += share
    return words


def _fill_gaps(
    tokens: Sequence[str], anchors: dict[int, WordTiming], duration: float
) -> list[WordTiming]:
    result: list[WordTiming] = []
    index = 0
    while index < len(tokens):
        if index in anchors:
            anchor = anchors[index]
            result.append(WordTiming(word=tokens[index], start=anchor.start, end=anchor.end))
            index += 1
            continue
        run_end = index
        while run_end < len(tokens) and run_end not in anchors:
            run_end += 1
        left = result[-1].end if result else 0.0
        right = anchors[run_end].start if run_end < len(tokens) else duration
        result.extend(_spread(tokens[index:run_end], left, max(left, right)))
        index = run_end
    return result


def _spread(tokens: Sequence[str], start: float, end: float) -> list[WordTiming]:
    """Divide ``[start, end)`` among ``tokens`` in proportion to their length."""
    weights = [len(token) + 1 for token in tokens]
    total = sum(weights)
    words: list[WordTiming] = []
    cursor = start
    for token, weight in zip(tokens, weights, strict=True):
        span = (end - start) * weight / total
        words.append(WordTiming(word=token, start=cursor, end=cursor + span))
        cursor += span
    return words


def _normalize(word: str) -> str:
    return _NON_WORD.sub("", word.lower())
