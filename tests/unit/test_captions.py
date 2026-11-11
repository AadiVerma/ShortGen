"""Caption rasterisation: fonts, frames and the ffconcat timeline."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from shortgen.config import AppConfig
from shortgen.domain.subtitles import Cue, CueSheet, SubtitleStyle
from shortgen.render.captions import (
    TRACK_NAME,
    caption_timeline,
    draw_caption,
    find_font_file,
    resolve_font_file,
    write_caption_track,
)
from shortgen.render.layout import Size

FRAME = Size(width=360, height=640)


@pytest.fixture
def style(config: AppConfig) -> SubtitleStyle:
    """The shipped style, pinned to Pillow's built-in font so tests need no system fonts."""
    return config.subtitles.style.model_copy(
        update={"font_name": "No Such Font", "font_fallbacks": []}
    )


def bounding_box(frame: Image.Image) -> tuple[int, int, int, int]:
    box = frame.getchannel("A").getbbox()
    assert box is not None, "the caption frame is completely transparent"
    return box


def fonts(tmp_path: Path, *names: str) -> Path:
    for name in names:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(b"")
    return tmp_path


def test_font_lookup_ignores_case_spacing_and_prefers_bold(tmp_path: Path) -> None:
    directory = fonts(
        tmp_path, "Supplemental/Arial.ttf", "Supplemental/Arial Bold.ttf", "notes.txt"
    )
    assert find_font_file(["arial"], True, [directory]) == directory / "Supplemental/Arial Bold.ttf"
    assert find_font_file(["Arial"], False, [directory]) == directory / "Supplemental/Arial.ttf"


def test_font_lookup_falls_through_names_and_missing_directories(tmp_path: Path) -> None:
    directory = fonts(tmp_path, "dejavu/DejaVuSans-Bold.ttf")
    wanted = ["Arial", "Helvetica", "DejaVu Sans"]
    assert find_font_file(wanted, True, [tmp_path / "absent", directory]) == (
        directory / "dejavu/DejaVuSans-Bold.ttf"
    )
    assert find_font_file(["Arial"], True, [directory]) is None


def test_explicit_font_path_wins_over_lookup(style: SubtitleStyle, tmp_path: Path) -> None:
    pinned = style.model_copy(update={"font_path": "~/fonts/Mine.ttf"})
    assert resolve_font_file(pinned, [tmp_path]) == Path("~/fonts/Mine.ttf").expanduser()
    assert resolve_font_file(style, [tmp_path]) is None  # falls back to the built-in font


def test_caption_is_drawn_bottom_centre_inside_the_margins(style: SubtitleStyle) -> None:
    frame = draw_caption(["Meet Odysseus,", "the war hero"], style, FRAME, None)
    assert frame.size == (360, 640)
    assert frame.mode == "RGBA"
    left, top, right, bottom = bounding_box(frame)
    assert abs((left + right) / 2 - FRAME.width / 2) <= 3  # centred
    assert bottom <= FRAME.height * (1 - style.margin_bottom_ratio) + style.outline_px + 8
    assert top > FRAME.height / 2  # captions live in the lower half
    assert frame.getpixel((0, 0)) == (0, 0, 0, 0)


def test_two_lines_take_more_height_than_one(style: SubtitleStyle) -> None:
    one = bounding_box(draw_caption(["going home"], style, FRAME, None))
    two = bounding_box(draw_caption(["going home", "wasn't easy"], style, FRAME, None))
    assert two[3] - two[1] > one[3] - one[1]
    assert abs(two[3] - one[3]) <= 4  # both sit on the same baseline


def test_overlong_line_is_scaled_down_to_fit(style: SubtitleStyle) -> None:
    left, _, right, _ = bounding_box(draw_caption(["W" * 40], style, FRAME, None))
    assert left >= 0 and right <= FRAME.width
    assert (
        right - left <= FRAME.width * (1 - 2 * style.margin_side_ratio) + 2 * style.outline_px + 2
    )


def test_uppercase_style_is_applied(style: SubtitleStyle) -> None:
    lower = draw_caption(["quiet"], style, FRAME, None)
    upper = draw_caption(["quiet"], style.model_copy(update={"uppercase": True}), FRAME, None)
    assert bounding_box(upper) != bounding_box(lower)


def test_timeline_shows_each_cue_for_its_interval_and_blank_between() -> None:
    cues = [Cue(start=0.5, end=1.5, lines=["a"]), Cue(start=1.5, end=2.0, lines=["b"])]
    entries = [("c0001.png", 0.5), ("c0002.png", 1.5)]
    assert caption_timeline(cues, entries, total_s=3.0).splitlines() == [
        "ffconcat version 1.0",
        "file 'blank.png'",
        "duration 0.500000",
        "file 'c0001.png'",
        "duration 1.000000",
        "file 'c0002.png'",  # back to back: no blank frame in between
        "duration 0.500000",
        "file 'blank.png'",
        "duration 1.000000",
        "file 'blank.png'",  # the demuxer needs the last file repeated
    ]


def test_timeline_durations_add_up_to_the_video_even_with_bad_cues() -> None:
    cues = [
        Cue(start=0.0, end=2.0, lines=["a"]),
        Cue(start=1.0, end=1.5, lines=["swallowed by the previous cue"]),
        Cue(start=2.5, end=9.0, lines=["runs past the end"]),
    ]
    entries = [(f"c{i}.png", cue.start) for i, cue in enumerate(cues)]
    text = caption_timeline(cues, entries, total_s=4.0)
    durations = [
        float(line.split()[1]) for line in text.splitlines() if line.startswith("duration")
    ]
    assert sum(durations) == pytest.approx(4.0)
    assert "c1.png" not in text


def test_track_files_are_written_side_by_side(style: SubtitleStyle, tmp_path: Path) -> None:
    sheet = CueSheet(
        width=180,
        height=320,
        fps=30,
        duration_s=2.0,
        cues=[Cue(start=0.2, end=0.9, lines=["one"]), Cue(start=1.0, end=1.8, lines=["two"])],
    )
    track = write_caption_track(sheet, style, tmp_path / "it's a [dir]" / "captions")
    assert track.name == TRACK_NAME
    assert sorted(p.name for p in track.parent.iterdir()) == [
        "blank.png",
        "c0001.png",
        "c0002.png",
        TRACK_NAME,
    ]
    assert Image.open(track.parent / "c0001.png").size == (180, 320)
    assert str(tmp_path) not in track.read_text()  # only bare file names, nothing to escape
