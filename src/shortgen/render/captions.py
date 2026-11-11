"""Caption rasterisation: cues -> transparent PNG frames -> an ffmpeg image timeline.

ffmpeg can only draw text when it was built with libass or freetype, and many
builds (Homebrew's default formula among them) have neither. Rendering the
caption images here with Pillow and letting ffmpeg merely *overlay* them works
with every ffmpeg build. ffmpeg still does all the compositing and encoding.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from shortgen.domain.subtitles import Cue, CueSheet, SubtitleStyle, font_size_px
from shortgen.render.layout import Size

log = logging.getLogger(__name__)

#: Where operating systems keep fonts. Searched in order; `font_path` in the
#: subtitle style bypasses the search entirely.
FONT_DIRECTORIES: tuple[Path, ...] = (
    Path.home() / "Library/Fonts",
    Path("/Library/Fonts"),
    Path("/System/Library/Fonts/Supplemental"),
    Path("/System/Library/Fonts"),
    Path.home() / ".local/share/fonts",
    Path.home() / ".fonts",
    Path("/usr/local/share/fonts"),
    Path("/usr/share/fonts"),
)
TRACK_NAME = "captions.ffconcat"
_BLANK_NAME = "blank.png"
_FONT_SUFFIXES = frozenset({".ttf", ".otf", ".ttc"})
_NON_ALNUM = re.compile(r"[^a-z0-9]")
_TRANSPARENT = (0, 0, 0, 0)
_MIN_GAP_S = 1e-6

Font = ImageFont.FreeTypeFont | ImageFont.ImageFont


def find_font_file(
    names: Sequence[str], bold: bool, directories: Sequence[Path] = FONT_DIRECTORIES
) -> Path | None:
    """First installed font file matching one of ``names``, preferring the bold cut.

    Matching ignores case, spaces and punctuation, so "DejaVu Sans" finds
    ``DejaVuSans-Bold.ttf`` and "Arial" finds ``Arial Bold.ttf``.
    """
    installed: dict[str, Path] = {}
    for directory in directories:
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*")):
            if path.suffix.lower() in _FONT_SUFFIXES:
                installed.setdefault(_font_key(path.stem), path)
    for name in names:
        wanted = [f"{name} Bold", name] if bold else [name, f"{name} Regular"]
        for candidate in wanted:
            found = installed.get(_font_key(candidate))
            if found is not None:
                return found
    return None


def resolve_font_file(
    style: SubtitleStyle, directories: Sequence[Path] = FONT_DIRECTORIES
) -> Path | None:
    """The font file captions will use, or ``None`` for Pillow's built-in font."""
    if style.font_path:
        return Path(style.font_path).expanduser()
    return find_font_file([style.font_name, *style.font_fallbacks], style.bold, directories)


def load_font(font_file: Path | None, size_px: int) -> Font:
    """Open a font at a pixel size; without a file, use the font bundled with Pillow."""
    if font_file is None:
        return ImageFont.load_default(size=size_px)
    return ImageFont.truetype(str(font_file), size=size_px)


def draw_caption(
    lines: Sequence[str], style: SubtitleStyle, size: Size, font_file: Path | None
) -> Image.Image:
    """One transparent frame with the caption drawn bottom-centre.

    Line breaks were chosen from an average glyph width, so a line of wide
    letters can still overflow; the font is then scaled down to fit.
    """
    text = [line.upper() if style.uppercase else line for line in lines]
    usable = size.width * (1 - 2 * style.margin_side_ratio)
    font_px = font_size_px(size.width, size.height, style)
    font = load_font(font_file, font_px)
    widest = max(font.getlength(line) for line in text)
    if widest > usable:
        font_px = max(1, int(font_px * usable / widest))
        font = load_font(font_file, font_px)
    frame = Image.new("RGBA", (size.width, size.height), _TRANSPARENT)
    draw = ImageDraw.Draw(frame)
    line_height = round(font_px * style.line_height_ratio)
    baseline = size.height - round(size.height * style.margin_bottom_ratio)
    for offset, line in enumerate(reversed(text)):
        draw.text(
            (size.width / 2, baseline - offset * line_height),
            line,
            font=font,
            fill=style.primary_color,
            stroke_width=round(style.outline_px),
            stroke_fill=style.outline_color,
            anchor="ms",
        )
    return frame


def write_caption_track(sheet: CueSheet, style: SubtitleStyle, directory: Path) -> Path:
    """Write one PNG per cue plus the ffconcat timeline that schedules them.

    Returns the timeline file. Everything it references sits beside it and is
    named by this function, so no user-controlled path enters the list.
    """
    size = Size(width=sheet.width, height=sheet.height)
    font_file = resolve_font_file(style)
    if font_file is None:
        log.warning(
            "no installed font matches %s; captions use Pillow's built-in font "
            "(set subtitles.style.font_path to choose one)",
            ", ".join([style.font_name, *style.font_fallbacks]),
        )
    directory.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (size.width, size.height), _TRANSPARENT).save(directory / _BLANK_NAME)
    entries: list[tuple[str, float]] = []
    for number, cue in enumerate(sheet.cues, start=1):
        name = f"c{number:04d}.png"
        draw_caption(cue.lines, style, size, font_file).save(directory / name)
        entries.append((name, cue.start))
    track = directory / TRACK_NAME
    track.write_text(caption_timeline(sheet.cues, entries, sheet.duration_s), encoding="utf-8")
    return track


def caption_timeline(
    cues: Sequence[Cue], entries: Sequence[tuple[str, float]], total_s: float
) -> str:
    """ffconcat text showing each cue image for its interval and nothing in between."""
    lines = ["ffconcat version 1.0"]
    cursor = 0.0
    for cue, (name, _start) in zip(cues, entries, strict=True):
        start = max(cue.start, cursor)
        end = min(max(cue.end, start), total_s)
        if start - cursor > _MIN_GAP_S:
            lines += _shown(_BLANK_NAME, start - cursor)
        if end - start > _MIN_GAP_S:
            lines += _shown(name, end - start)
            cursor = end
        else:
            cursor = start
    if total_s - cursor > _MIN_GAP_S:
        lines += _shown(_BLANK_NAME, total_s - cursor)
    # The concat demuxer ignores the duration of the final entry unless the
    # file is listed once more after it.
    lines.append(f"file '{_BLANK_NAME}'")
    return "\n".join(lines) + "\n"


def _shown(name: str, seconds: float) -> list[str]:
    return [f"file '{name}'", f"duration {seconds:.6f}"]


def _font_key(name: str) -> str:
    return _NON_ALNUM.sub("", name.lower())
