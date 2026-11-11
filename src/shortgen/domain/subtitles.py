"""Subtitle layout: words -> cues -> SRT and ASS text.

Layout is resolution-aware but renderer-agnostic. The ASS output is what
ffmpeg burns in; the SRT output is a convenience for upload forms.
"""

from __future__ import annotations

import re
import textwrap
from collections.abc import Sequence

from pydantic import Field

from shortgen.domain.models import FrozenModel, Timing, WordTiming

_SENTENCE_END = re.compile(r"[.!?\u2026][\"'\u201d\u2019)\]]*$")
_CLAUSE_END = re.compile(r"[,;:\u2014\u2013][\"'\u201d\u2019)\]]*$")
_WORD_ONLY = re.compile(r"[^a-z']")
_MIN_CLAUSE_WORDS = 2
_WEAK_ENDINGS = frozenset(
    {"a", "an", "the", "of", "to", "in", "on", "at", "by", "for", "with", "and", "or", "but"}
    | {"his", "her", "its", "their", "my", "your", "our", "this", "that", "these", "those"}
    | {"is", "was", "are", "were", "be", "been", "as", "from", "into"}
)
_ASS_OVERRIDE = re.compile(r"[{}]")
_HEX_COLOR = r"^#[0-9A-Fa-f]{6}$"
_SECONDS_PER_HOUR = 3600
_SECONDS_PER_MINUTE = 60


class SubtitleRules(FrozenModel):
    """When to start a new cue."""

    max_lines: int = Field(ge=1)
    max_gap_s: float = Field(ge=0)
    min_cue_s: float = Field(ge=0)
    max_cue_s: float = Field(gt=0)


class SubtitleStyle(FrozenModel):
    """How burned-in captions look. Ratios are relative to the frame."""

    font_name: str = Field(min_length=1)
    font_fallbacks: list[str] = Field(description="tried in order when font_name is missing")
    font_path: str | None = Field(default=None, description="explicit font file; skips lookup")
    line_height_ratio: float = Field(gt=0, description="line spacing / font size")
    font_size_ratio: float = Field(gt=0, lt=1, description="of the frame's short side")
    char_width_ratio: float = Field(gt=0, description="average glyph width / font size")
    margin_side_ratio: float = Field(ge=0, lt=0.5, description="of the frame width")
    margin_bottom_ratio: float = Field(ge=0, lt=1, description="of the frame height")
    primary_color: str = Field(pattern=_HEX_COLOR)
    outline_color: str = Field(pattern=_HEX_COLOR)
    outline_px: float = Field(ge=0)
    shadow_px: float = Field(ge=0)
    bold: bool
    uppercase: bool


class Cue(Timing):
    """One caption: its lines and when it is on screen."""

    lines: list[str] = Field(min_length=1)


class CueSheet(FrozenModel):
    """``subtitles/captions.json``: every cue on the final timeline, ready to draw."""

    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: int = Field(ge=1)
    duration_s: float = Field(ge=0)
    cues: list[Cue]


def font_size_px(width: int, height: int, style: SubtitleStyle) -> int:
    """Caption font size for a frame, scaled from its short side."""
    return max(1, round(min(width, height) * style.font_size_ratio))


def chars_per_line(width: int, height: int, style: SubtitleStyle) -> int:
    """How many average-width characters fit between the side margins."""
    usable = width * (1 - 2 * style.margin_side_ratio)
    glyph = font_size_px(width, height, style) * style.char_width_ratio
    return max(1, int(usable // glyph))


def wrap_text(text: str, max_chars: int) -> list[str]:
    """Wrap on spaces only, into lines of similar length.

    The line count is the minimum that fits ``max_chars``; the width is then
    narrowed as far as that count allows, so "By morning the sea was / calm."
    becomes "By morning the / sea was calm.". A single over-long word keeps
    its own line rather than being split.
    """
    lines = _greedy_wrap(text, max_chars)
    for width in range(1, max_chars):
        narrower = _greedy_wrap(text, width)
        if len(narrower) == len(lines):
            return narrower
    return lines


def _greedy_wrap(text: str, width: int) -> list[str]:
    lines = textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False)
    return lines or [text]


def build_cues(words: Sequence[WordTiming], rules: SubtitleRules, max_chars: int) -> list[Cue]:
    """Group timed words into cues that fit the frame and follow speech rhythm.

    Words are first cut into phrases at sentence ends and at pauses longer
    than ``max_gap_s``, then into clauses at commas, colons and dashes. A
    clause that is too big for one cue (more than ``max_lines`` lines, or
    longer than ``max_cue_s``) is divided into the fewest evenly sized cues
    that fit, without leaving a dangling "a" or "the" at the end of a caption.
    """
    groups = [
        group
        for phrase in _phrases(words, rules.max_gap_s)
        for clause in _clauses(phrase)
        for group in _balanced_groups(clause, rules, max_chars)
    ]
    cues = [
        Cue(start=group[0].start, end=group[-1].end, lines=wrap_text(_text(group), max_chars))
        for group in groups
    ]
    return _enforce_min_duration(cues, rules.min_cue_s)


def clamp_cues(cues: Sequence[Cue], limit: Timing) -> list[Cue]:
    """Trim cues so none is shown outside ``limit`` (one scene's interval)."""
    clamped: list[Cue] = []
    for cue in cues:
        start = min(max(cue.start, limit.start), limit.end)
        end = min(max(cue.end, start), limit.end)
        if end > start:
            clamped.append(Cue(start=start, end=end, lines=cue.lines))
    return clamped


def format_srt_time(seconds: float) -> str:
    """``HH:MM:SS,mmm`` as SubRip expects."""
    total_ms = round(seconds * 1000)
    whole, millis = divmod(total_ms, 1000)
    hours, rest = divmod(whole, _SECONDS_PER_HOUR)
    minutes, secs = divmod(rest, _SECONDS_PER_MINUTE)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_ass_time(seconds: float) -> str:
    """``H:MM:SS.cc`` (centiseconds) as Advanced SubStation Alpha expects."""
    total_cs = round(seconds * 100)
    whole, centis = divmod(total_cs, 100)
    hours, rest = divmod(whole, _SECONDS_PER_HOUR)
    minutes, secs = divmod(rest, _SECONDS_PER_MINUTE)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def render_srt(cues: Sequence[Cue]) -> str:
    """Serialise cues as a SubRip document."""
    blocks = [
        f"{number}\n{format_srt_time(cue.start)} --> {format_srt_time(cue.end)}\n"
        + "\n".join(cue.lines)
        for number, cue in enumerate(cues, start=1)
    ]
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def render_ass(cues: Sequence[Cue], style: SubtitleStyle, width: int, height: int) -> str:
    """Serialise cues as an ASS document sized for a ``width`` x ``height`` frame."""
    margin_side = round(width * style.margin_side_ratio)
    margin_bottom = round(height * style.margin_bottom_ratio)
    primary = _ass_color(style.primary_color)
    style_line = (
        f"Style: Default,{style.font_name},{font_size_px(width, height, style)},"
        f"{primary},{primary},{_ass_color(style.outline_color)},&H80000000,"
        f"{-1 if style.bold else 0},0,0,0,100,100,0,0,1,{style.outline_px:g},{style.shadow_px:g},"
        f"2,{margin_side},{margin_side},{margin_bottom},1"
    )
    events = [
        f"Dialogue: 0,{format_ass_time(cue.start)},{format_ass_time(cue.end)},Default,,0,0,0,,"
        + r"\N".join(_ass_text(line, style.uppercase) for line in cue.lines)
        for cue in cues
    ]
    return "\n".join(
        [
            "[Script Info]",
            "ScriptType: v4.00+",
            f"PlayResX: {width}",
            f"PlayResY: {height}",
            "WrapStyle: 2",
            "ScaledBorderAndShadow: yes",
            "",
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
            "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
            "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
            style_line,
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
            *events,
            "",
        ]
    )


def _phrases(words: Sequence[WordTiming], max_gap_s: float) -> list[list[WordTiming]]:
    """Cut after sentence-ending punctuation and wherever the speaker pauses."""
    phrases: list[list[WordTiming]] = []
    for word in words:
        previous = phrases[-1][-1] if phrases else None
        starts_phrase = (
            previous is None
            or _SENTENCE_END.search(previous.word) is not None
            or word.start - previous.end > max_gap_s
        )
        if starts_phrase:
            phrases.append([word])
        else:
            phrases[-1].append(word)
    return phrases


def _clauses(phrase: Sequence[WordTiming]) -> list[list[WordTiming]]:
    """Cut a phrase after commas, colons and dashes, keeping very short clauses attached.

    "Meet Odysseus, king of a small island" reads better as two captions than
    as one balanced by length alone. A clause shorter than ``_MIN_CLAUSE_WORDS``
    ("First,") is joined to its neighbour so it does not flash by on its own.
    """
    clauses: list[list[WordTiming]] = [[]]
    for word in phrase:
        clauses[-1].append(word)
        if _CLAUSE_END.search(word.word):
            clauses.append([])
    clauses = [clause for clause in clauses if clause]
    merged: list[list[WordTiming]] = []
    carry: list[WordTiming] = []
    for clause in clauses:
        clause = [*carry, *clause]
        carry = []
        if len(clause) < _MIN_CLAUSE_WORDS:
            carry = clause
        else:
            merged.append(clause)
    if carry:
        if merged:
            merged[-1].extend(carry)
        else:
            merged.append(carry)
    return merged


def _balanced_groups(
    phrase: Sequence[WordTiming], rules: SubtitleRules, max_chars: int
) -> list[list[WordTiming]]:
    """Split a phrase into the fewest similar-sized groups that each fit one cue."""
    for count in range(1, len(phrase) + 1):
        groups = _partition(phrase, count)
        if all(_fits(group, rules, max_chars) for group in groups):
            return _tidy_endings(groups, rules, max_chars)
    return [[word] for word in phrase]


def _tidy_endings(
    groups: list[list[WordTiming]], rules: SubtitleRules, max_chars: int
) -> list[list[WordTiming]]:
    """Move a dangling "a", "the", "of"... to the start of the next caption when it fits."""
    for index in range(len(groups) - 1):
        current, following = groups[index], groups[index + 1]
        dangling = _WORD_ONLY.sub("", current[-1].word.lower()) in _WEAK_ENDINGS
        if dangling and len(current) > 1 and _fits([current[-1], *following], rules, max_chars):
            following.insert(0, current.pop())
    return groups


def _partition(phrase: Sequence[WordTiming], count: int) -> list[list[WordTiming]]:
    """Divide ``phrase`` into at most ``count`` runs of roughly equal text length."""
    sizes = [len(word.word) + 1 for word in phrase]
    total = sum(sizes)
    groups: list[list[WordTiming]] = [[]]
    consumed = 0
    for word, size in zip(phrase, sizes, strict=True):
        boundary = total * len(groups) / count
        if groups[-1] and len(groups) < count and consumed + size / 2 > boundary:
            groups.append([])
        groups[-1].append(word)
        consumed += size
    return groups


def _fits(group: Sequence[WordTiming], rules: SubtitleRules, max_chars: int) -> bool:
    if len(group) == 1:
        return True  # a single word cannot be divided further
    if group[-1].end - group[0].start > rules.max_cue_s:
        return False
    return len(wrap_text(_text(group), max_chars)) <= rules.max_lines


def _enforce_min_duration(cues: Sequence[Cue], min_cue_s: float) -> list[Cue]:
    """Hold short cues on screen a little longer, never into the next cue."""
    held: list[Cue] = []
    for index, cue in enumerate(cues):
        ceiling = cues[index + 1].start if index + 1 < len(cues) else float("inf")
        end = max(cue.end, min(cue.start + min_cue_s, ceiling))
        held.append(Cue(start=cue.start, end=end, lines=cue.lines))
    return held


def _text(group: Sequence[WordTiming]) -> str:
    return " ".join(word.word for word in group)


def _ass_color(hex_color: str) -> str:
    """``#RRGGBB`` -> ASS ``&H00BBGGRR`` (opaque, blue first)."""
    red, green, blue = hex_color[1:3], hex_color[3:5], hex_color[5:7]
    return f"&H00{blue}{green}{red}".upper()


def _ass_text(line: str, uppercase: bool) -> str:
    """Neutralise override braces so script text can never inject ASS tags."""
    cleaned = _ASS_OVERRIDE.sub("", line).replace("\\", "")
    return cleaned.upper() if uppercase else cleaned
