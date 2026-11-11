from __future__ import annotations

import itertools

import pytest

from shortgen.domain.models import WordTiming
from shortgen.domain.timing import (
    TimingRules,
    align_to_script,
    build_timeline,
    estimate_duration,
    proportional_word_timings,
    scene_duration,
    script_tokens,
    shift_words,
)

RULES = TimingRules(pad_start_s=0.15, pad_end_s=0.35, min_scene_s=1.5)


def w(word: str, start: float, end: float) -> WordTiming:
    return WordTiming(word=word, start=start, end=end)


def test_estimate_duration_uses_speaking_rate() -> None:
    assert estimate_duration("one two three four five", 150) == pytest.approx(2.0)


def test_scene_duration_pads_and_snaps_up_to_whole_frames() -> None:
    duration = scene_duration(2.01, RULES, fps=30)
    assert duration == pytest.approx(76 / 30)  # 2.51 s -> 75.3 frames -> 76
    assert scene_duration(2.0, RULES, fps=30) == pytest.approx(2.5)  # already whole frames


def test_scene_duration_has_a_floor() -> None:
    assert scene_duration(0.2, RULES, fps=30) == pytest.approx(1.5)


def test_timeline_is_contiguous() -> None:
    timeline = build_timeline([2.0, 3.5, 1.5])
    assert [(t.start, t.end) for t in timeline] == [(0.0, 2.0), (2.0, 5.5), (5.5, 7.0)]


def test_shift_moves_words_onto_the_timeline() -> None:
    assert shift_words([w("hi", 0.1, 0.4)], 5.0) == [w("hi", 5.1, 5.4)]


def test_stray_punctuation_joins_its_neighbour() -> None:
    assert script_tokens("— Wait — she said") == ["— Wait —", "she", "said"]


def test_proportional_timings_cover_the_whole_duration() -> None:
    words = proportional_word_timings("a longer word", 3.0)
    assert [x.word for x in words] == ["a", "longer", "word"]
    assert words[0].start == 0.0
    assert words[-1].end == pytest.approx(3.0)
    assert words[1].duration > words[0].duration
    assert all(a.end == pytest.approx(b.start) for a, b in itertools.pairwise(words))


def test_alignment_keeps_script_spelling_with_heard_timing() -> None:
    heard = [w(" the", 0.0, 0.2), w(" light", 0.2, 0.6), w(" went", 0.6, 0.9), w(" out", 0.9, 1.2)]
    aligned = align_to_script("The light went out.", heard, 1.3)
    assert [x.word for x in aligned] == ["The", "light", "went", "out."]
    assert [(x.start, x.end) for x in aligned] == [(0.0, 0.2), (0.2, 0.6), (0.6, 0.9), (0.9, 1.2)]


def test_misheard_words_are_interpolated_between_anchors() -> None:
    heard = [w("mirror", 0.0, 0.5), w("lit", 1.0, 1.2), w("the", 1.2, 1.3), w("lamp", 1.3, 1.8)]
    aligned = align_to_script("Mira quickly lit the lamp", heard, 2.0)
    assert [x.word for x in aligned] == ["Mira", "quickly", "lit", "the", "lamp"]
    assert aligned[0].start == 0.0
    assert aligned[1].end == pytest.approx(1.0)  # the unmatched run ends where "lit" starts
    assert (aligned[2].start, aligned[2].end) == (1.0, 1.2)


def test_dropped_trailing_words_fill_up_to_the_audio_end() -> None:
    aligned = align_to_script("one two three", [w("one", 0.0, 0.4)], 1.0)
    assert aligned[0].end == 0.4
    assert aligned[-1].end == pytest.approx(1.0)


def test_nothing_heard_falls_back_to_proportional() -> None:
    assert align_to_script("one two", [], 2.0) == proportional_word_timings("one two", 2.0)


def test_empty_text_has_no_words() -> None:
    assert align_to_script("", [w("x", 0, 1)], 1.0) == []


def test_word_timing_rejects_reversed_interval() -> None:
    with pytest.raises(ValueError, match="before start"):
        w("x", 2.0, 1.0)
