from __future__ import annotations

import pytest

from shortgen.config import AppConfig
from shortgen.domain.models import Timing, WordTiming
from shortgen.domain.subtitles import (
    Cue,
    SubtitleRules,
    build_cues,
    chars_per_line,
    clamp_cues,
    font_size_px,
    format_ass_time,
    format_srt_time,
    render_ass,
    render_srt,
    wrap_text,
)

RULES = SubtitleRules(max_lines=2, max_gap_s=0.6, min_cue_s=0.7, max_cue_s=4.0)


def timed(text: str, step: float = 0.3, start: float = 0.0) -> list[WordTiming]:
    """Words spoken back to back, ``step`` seconds each."""
    return [
        WordTiming(word=word, start=start + i * step, end=start + (i + 1) * step)
        for i, word in enumerate(text.split())
    ]


def test_time_formats() -> None:
    assert format_srt_time(3661.0405) == "01:01:01,040"
    assert format_ass_time(3661.046) == "1:01:01.05"
    assert format_srt_time(0.9996) == "00:00:01,000"  # rounding carries into seconds


def test_wrap_never_splits_a_word() -> None:
    assert wrap_text("a supercalifragilistic day", 10) == ["a", "supercalifragilistic", "day"]


def test_wrap_balances_lines_instead_of_leaving_an_orphan() -> None:
    assert wrap_text("By morning the sea was calm.", 26) == ["By morning the", "sea was calm."]
    assert wrap_text("short", 26) == ["short"]


def test_layout_scales_with_the_frame(config: AppConfig) -> None:
    style = config.subtitles.style
    assert font_size_px(1080, 1920, style) == font_size_px(1920, 1080, style) == 63
    assert chars_per_line(1920, 1080, style) > chars_per_line(1080, 1920, style)


def test_cue_breaks_at_sentence_end() -> None:
    cues = build_cues(timed("It was late. She left."), RULES, max_chars=40)
    assert [cue.lines for cue in cues] == [["It was late."], ["She left."]]


def test_cue_breaks_at_clauses_but_keeps_one_word_clauses_attached() -> None:
    cues = build_cues(timed("Meet Odysseus, king of Ithaca."), RULES, max_chars=40)
    assert [cue.lines for cue in cues] == [["Meet Odysseus,"], ["king of Ithaca."]]
    cues = build_cues(timed("First, a giant trapped them."), RULES, max_chars=40)
    assert [cue.lines for cue in cues] == [["First, a giant trapped them."]]
    cues = build_cues(timed("Eight arms, three hearts, blue blood."), RULES, max_chars=40)
    assert [cue.lines for cue in cues] == [["Eight arms,"], ["three hearts,"], ["blue blood."]]


def test_a_caption_does_not_end_on_a_dangling_article() -> None:
    one_line = RULES.model_copy(update={"max_lines": 1})
    words = timed("He was king of a small island called Ithaca")
    texts = [" ".join(cue.lines) for cue in build_cues(words, one_line, max_chars=18)]
    assert texts == ["He was king", "of a small island", "called Ithaca"]  # not "...king of"
    # When the next caption has no room for the word, it stays where it was.
    tight = [" ".join(cue.lines) for cue in build_cues(words, one_line, max_chars=14)]
    assert tight == ["He was king of", "a small island", "called Ithaca"]


def test_cue_breaks_on_a_long_pause() -> None:
    words = [*timed("one two"), *timed("three four", start=2.0)]
    cues = build_cues(words, RULES, max_chars=40)
    assert [cue.lines for cue in cues] == [["one two"], ["three four"]]


def test_cue_never_exceeds_max_lines() -> None:
    cues = build_cues(timed("aaaa bbbb cccc dddd eeee ffff gggg"), RULES, max_chars=9)
    assert all(len(cue.lines) <= RULES.max_lines for cue in cues)
    assert next(cue.lines for cue in cues) == ["aaaa bbbb", "cccc dddd"]
    assert " ".join(line for cue in cues for line in cue.lines) == (
        "aaaa bbbb cccc dddd eeee ffff gggg"
    )


def test_cue_breaks_when_it_would_run_too_long() -> None:
    cues = build_cues(timed("a b c d e f", step=1.0), RULES, max_chars=40)
    assert all(cue.duration <= RULES.max_cue_s for cue in cues)
    assert len(cues) == 2


def test_short_cue_is_held_but_not_into_the_next() -> None:
    words = [
        WordTiming(word="Go.", start=0.0, end=0.2),
        WordTiming(word="Now.", start=0.5, end=0.7),
    ]
    first, second = build_cues(words, RULES, max_chars=40)
    assert first.end == 0.5  # wanted 0.7, capped by the next cue
    assert second.end == pytest.approx(1.2)


def test_clamp_keeps_cues_inside_their_scene() -> None:
    cues = [Cue(start=0.5, end=3.5, lines=["x"]), Cue(start=4.0, end=5.0, lines=["y"])]
    assert clamp_cues(cues, Timing(start=1.0, end=3.0)) == [Cue(start=1.0, end=3.0, lines=["x"])]


def test_srt_document() -> None:
    cues = [
        Cue(start=0.0, end=1.5, lines=["Hello", "there"]),
        Cue(start=2.0, end=3.0, lines=["Bye"]),
    ]
    assert render_srt(cues) == (
        "1\n00:00:00,000 --> 00:00:01,500\nHello\nthere\n\n2\n00:00:02,000 --> 00:00:03,000\nBye\n"
    )
    assert render_srt([]) == ""


def test_ass_document_carries_geometry_style_and_events(config: AppConfig) -> None:
    style = config.subtitles.style.model_copy(update={"primary_color": "#FFCC00"})
    ass = render_ass([Cue(start=1.0, end=2.5, lines=["Top", "Bottom"])], style, 1080, 1920)
    assert "PlayResX: 1080\nPlayResY: 1920" in ass
    assert "Style: Default,Arial,63,&H0000CCFF,&H0000CCFF,&H00000000," in ass
    assert ",2,86,86,384,1\n" in ass  # bottom-centre, side and bottom margins in px
    assert r"Dialogue: 0,0:00:01.00,0:00:02.50,Default,,0,0,0,,Top\NBottom" in ass


def test_ass_text_cannot_inject_override_tags(config: AppConfig) -> None:
    style = config.subtitles.style.model_copy(update={"uppercase": True})
    ass = render_ass([Cue(start=0, end=1, lines=[r"{\an8}hi\N"])], style, 1080, 1920)
    assert ass.rstrip().endswith(",,AN8HIN")
