"""Draws the original stick-figure art for examples/odyssey-cartoon as SVG files.

The pictures are simple on purpose: they exist so the bundled example can be
rendered into a real, watchable video with no image model at all (the
``folder`` image provider). Re-run this to regenerate the SVG sources:

    python scripts/draw_odyssey_art.py examples/odyssey-cartoon/art-src

Turn them into the PNGs the pipeline reads with any SVG renderer, for example
``rsvg-convert -w 1920 s001.svg -o ../art/s001.png``.
"""

from __future__ import annotations

import sys
from pathlib import Path

W, H = 1920, 1080
INK = "#2b2118"
SKIN = "#f7d9b5"
RED = "#c8442f"
BRONZE = "#d9a441"
WOOD = "#a9713d"
WOOD_DARK = "#7c4f27"
SEA = "#2f7fa8"
SEA_DARK = "#1f5f86"
STROKE = f'stroke="{INK}" stroke-linecap="round" stroke-linejoin="round"'


def sky(top: str, bottom: str) -> str:
    return (
        f'<defs><linearGradient id="sky" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0" stop-color="{top}"/><stop offset="1" stop-color="{bottom}"/>'
        f'</linearGradient></defs><rect width="{W}" height="{H}" fill="url(#sky)"/>'
    )


def sun(x: int, y: int, r: int, color: str = "#ffd166") -> str:
    return f'<circle cx="{x}" cy="{y}" r="{r}" fill="{color}" opacity="0.95"/>'


def ground(y: int, color: str, bump: int = 30) -> str:
    return (
        f'<path d="M0,{y} Q{W // 4},{y - bump} {W // 2},{y} T{W},{y} L{W},{H} L0,{H} Z" '
        f'fill="{color}" {STROKE} stroke-width="8"/>'
    )


def sea(y: int, color: str = SEA, rows: int = 3) -> str:
    parts = [f'<rect x="0" y="{y}" width="{W}" height="{H - y}" fill="{color}"/>']
    for row in range(rows):
        wave_y = y + 40 + row * 90
        d = f"M-40,{wave_y} " + " ".join("q60,-36 120,0 t120,0" for _ in range(9))
        parts.append(
            f'<path d="{d}" fill="none" stroke="#bfe6f5" stroke-width="7" opacity="0.7" stroke-linecap="round"/>'
        )
    return "".join(parts)


def person(
    x: int,
    y: int,
    scale: float = 1.0,
    arms: tuple[tuple[int, int], tuple[int, int]] = ((-40, -92), (40, -92)),
    legs: tuple[tuple[int, int], tuple[int, int]] = ((-26, 0), (26, 0)),
    face: str = "smile",
    extra_back: str = "",
    extra_front: str = "",
    skin: str = SKIN,
) -> str:
    """A stick figure standing with its feet at (x, y). Local units: 200 tall."""
    mouths = {
        "smile": '<path d="M-10,-168 Q0,-158 10,-168" fill="none" stroke-width="4"/>',
        "scared": '<ellipse cx="0" cy="-165" rx="7" ry="9" fill="#2b2118" stroke="none"/>',
        "grim": '<path d="M-10,-164 L10,-164" fill="none" stroke-width="4"/>',
        "sing": '<ellipse cx="0" cy="-165" rx="6" ry="8" fill="#7a2a2a" stroke="none"/>',
        "none": "",
    }
    (lx, ly), (rx, ry) = arms
    (flx, fly), (frx, fry) = legs
    return (
        f'<g transform="translate({x},{y}) scale({scale})" {STROKE} stroke-width="9" fill="none">'
        f"{extra_back}"
        f'<path d="M0,-150 L0,-80 M0,-80 L{flx},{fly} M0,-80 L{frx},{fry} '
        f'M0,-138 L{lx},{ly} M0,-138 L{rx},{ry}"/>'
        f'<circle cx="0" cy="-178" r="28" fill="{skin}"/>'
        f'<circle cx="-10" cy="-182" r="3.5" fill="{INK}" stroke="none"/>'
        f'<circle cx="10" cy="-182" r="3.5" fill="{INK}" stroke="none"/>'
        f"{mouths[face]}{extra_front}</g>"
    )


HELMET = (
    f'<path d="M-31,-180 A31,31 0 0 1 31,-180 L31,-168 L-31,-168 Z" fill="{BRONZE}" stroke-width="7"/>'
    f'<path d="M-8,-208 Q18,-252 46,-196 Q18,-216 2,-206 Z" fill="{RED}" stroke-width="6"/>'
)
CAPE = f'<path d="M-6,-146 L-52,-30 L-10,-44 Z" fill="{RED}" stroke-width="6"/>'
BEARD = '<path d="M-20,-166 Q0,-128 20,-166 Q0,-150 -20,-166 Z" fill="#8a6a4a" stroke-width="5"/>'
SPEAR = (
    '<path d="M46,-10 L46,-250" stroke-width="7"/>'
    '<path d="M36,-250 L46,-290 L56,-250 Z" fill="#cfd4d8" stroke-width="5"/>'
)
CROWN = f'<path d="M-24,-200 L-24,-226 L-12,-212 L0,-232 L12,-212 L24,-226 L24,-200 Z" fill="{BRONZE}" stroke-width="5"/>'


def odysseus(
    x: int, y: int, scale: float = 1.0, face: str = "smile", spear: bool = True, **kw: object
) -> str:
    front = HELMET + BEARD + (SPEAR if spear else "")
    arms = kw.pop("arms", ((-40, -92), (46, -110) if spear else (40, -92)))
    return person(x, y, scale, arms=arms, face=face, extra_back=CAPE, extra_front=front, **kw)  # type: ignore[arg-type]


def sailor(x: int, y: int, scale: float = 0.7, face: str = "scared", **kw: object) -> str:
    return person(x, y, scale, face=face, **kw)  # type: ignore[arg-type]


def ship(x: int, y: int, scale: float = 1.0, sail: str = "#f3ead7", crew: int = 3) -> str:
    heads = "".join(
        f'<circle cx="{-110 + i * 70}" cy="-52" r="17" fill="{SKIN}" stroke-width="6"/>'
        for i in range(crew)
    )
    return (
        f'<g transform="translate({x},{y}) scale({scale})" {STROKE} stroke-width="8">'
        f'<path d="M0,-40 L0,-330" fill="none"/>'
        f'<path d="M12,-320 Q150,-230 12,-110 Z" fill="{sail}"/>'
        f'<path d="M12,-260 Q92,-225 12,-180" fill="none" stroke="{RED}" stroke-width="12"/>'
        f"{heads}"
        f'<path d="M-200,-40 L200,-40 Q170,40 110,48 L-110,48 Q-170,40 -200,-40 Z" fill="{WOOD}"/>'
        f'<path d="M-200,-40 Q-240,-80 -215,-110" fill="none"/>'
        f'<path d="M-150,0 L150,0" stroke="{WOOD_DARK}" stroke-width="6" fill="none"/></g>'
    )


def flames(x: int, y: int, scale: float = 1.0) -> str:
    return (
        f'<g transform="translate({x},{y}) scale({scale})" stroke="none">'
        '<path d="M0,0 Q-60,-70 -10,-150 Q0,-100 30,-120 Q70,-50 0,0 Z" fill="#f08a24"/>'
        '<path d="M0,0 Q-28,-45 -4,-95 Q8,-60 22,-70 Q36,-30 0,0 Z" fill="#ffd166"/></g>'
    )


def sheep(x: int, y: int, scale: float = 1.0) -> str:
    wool = "".join(
        f'<circle cx="{cx}" cy="{cy}" r="34"/>'
        for cx, cy in [
            (-60, -80),
            (-20, -98),
            (25, -98),
            (65, -80),
            (-40, -52),
            (10, -48),
            (50, -55),
        ]
    )
    return (
        f'<g transform="translate({x},{y}) scale({scale})" {STROKE} stroke-width="7">'
        '<path d="M-45,-30 L-45,0 M-10,-30 L-10,0 M30,-30 L30,0 M60,-30 L60,0" fill="none"/>'
        f'<g fill="#fbf7ee">{wool}</g>'
        '<ellipse cx="104" cy="-84" rx="26" ry="22" fill="#3b2f2a"/>'
        '<circle cx="112" cy="-90" r="4" fill="#fff" stroke="none"/></g>'
    )


def note(x: int, y: int, color: str = "#fff3c4") -> str:
    return (
        f'<g transform="translate({x},{y})" fill="{color}" stroke="{color}" stroke-width="7" stroke-linecap="round">'
        '<ellipse cx="0" cy="0" rx="14" ry="10"/><path d="M13,0 L13,-52 L40,-42" fill="none"/></g>'
    )


def scene_1() -> str:
    """Meet Odysseus, king of a small island called Ithaca."""
    island = (
        f'<path d="M260,820 Q560,520 960,560 Q1380,520 1660,820 Z" fill="#8fb65c" {STROKE} stroke-width="8"/>'
        f'<path d="M1180,600 L1180,500 L1330,500 L1330,600 Z M1160,500 L1255,430 L1350,500 Z" fill="#f3ead7" {STROKE} stroke-width="7"/>'
        f'<circle cx="640" cy="560" r="46" fill="#5f9a3e" {STROKE} stroke-width="7"/><path d="M640,606 L640,660" {STROKE} stroke-width="9"/>'
    )
    return (
        sky("#ffd9a0", "#ffeccc")
        + sun(1560, 210, 110)
        + sea(760)
        + island
        + odysseus(930, 770, 2.5)
    )


def scene_2() -> str:
    """He won a ten-year war with one clever trick: a giant wooden horse."""
    wall = (
        f'<path d="M0,760 L0,430 L60,430 L60,380 L130,380 L130,430 L200,430 L200,380 L270,380 L270,430 '
        f'L340,430 L340,380 L410,380 L410,430 L480,430 L480,760 Z" fill="#b9a58a" {STROKE} stroke-width="8"/>'
        f'<path d="M170,760 L170,600 Q240,520 310,600 L310,760 Z" fill="#5a4630" {STROKE} stroke-width="7"/>'
    )
    horse = (
        f'<g {STROKE} stroke-width="9">'
        f'<rect x="820" y="330" width="520" height="250" rx="60" fill="{WOOD}"/>'
        f'<path d="M1240,360 L1330,150 L1450,170 L1500,250 L1420,270 L1380,240 L1340,400 Z" fill="{WOOD}"/>'
        f'<path d="M1318,170 Q1250,230 1250,330" fill="none" stroke="{WOOD_DARK}" stroke-width="16"/>'
        f'<circle cx="1425" cy="205" r="9" fill="{INK}" stroke="none"/>'
        f'<path d="M880,580 L870,760 M1010,580 L1010,760 M1180,580 L1180,760 M1300,580 L1310,760" stroke-width="26" stroke="{WOOD_DARK}"/>'
        f'<rect x="800" y="760" width="580" height="34" rx="10" fill="{WOOD_DARK}"/>'
        f'<circle cx="880" cy="820" r="40" fill="{WOOD}"/><circle cx="1300" cy="820" r="40" fill="{WOOD}"/>'
        f'<path d="M900,400 L900,560 M1000,350 L1000,570 M1100,350 L1100,570 M1200,350 L1200,570" stroke="{WOOD_DARK}" stroke-width="5" fill="none"/>'
        f'<rect x="1030" y="410" width="110" height="80" rx="8" fill="#3b2a18"/>'
        f'<circle cx="1062" cy="452" r="7" fill="#fff" stroke="none"/><circle cx="1108" cy="452" r="7" fill="#fff" stroke="none"/>'
        f'<path d="M820,420 Q740,430 730,540" fill="none" stroke="{WOOD_DARK}" stroke-width="18"/></g>'
    )
    return (
        sky("#3a2a4d", "#f08a55")
        + sun(1700, 300, 70, "#ffe9b0")
        + ground(790, "#c79a5b")
        + wall
        + flames(110, 380, 1.3)
        + flames(330, 380, 1.0)
        + horse
        + odysseus(1640, 960, 1.9)
    )


def scene_3() -> str:
    """Winning was the easy part. Getting home took ten more years."""
    path = '<path d="M220,330 C520,180 700,520 1000,380 S1500,160 1740,360" fill="none" stroke="#fff" stroke-width="9" stroke-dasharray="6 30" stroke-linecap="round" opacity="0.9"/>'
    home = f'<path d="M1650,470 Q1740,360 1830,470 Z" fill="#8fb65c" {STROKE} stroke-width="7"/><path d="M1722,410 l18,-22 l18,22 v24 h-36 Z" fill="#f3ead7" {STROKE} stroke-width="5"/>'
    troy = (
        f'<path d="M110,470 L110,400 L150,400 L150,370 L190,370 L190,400 L230,400 L230,470 Z" fill="#b9a58a" {STROKE} stroke-width="6"/>'
        + flames(170, 372, 0.6)
    )
    return (
        sky("#f4a259", "#ffe2a8")
        + sun(960, 470, 150, "#ffd166")
        + path
        + troy
        + home
        + sea(480, SEA, 5)
        + ship(900, 760, 1.5)
    )


def scene_4() -> str:
    """First, a one-eyed giant trapped his crew in a cave and started eating them."""
    cave = (
        f'<rect width="{W}" height="{H}" fill="#2d2622"/>'
        f'<path d="M120,1080 L120,520 Q200,120 960,110 Q1720,120 1800,520 L1800,1080 Z" fill="#5b4a3f" {STROKE} stroke-width="10"/>'
        f'<path d="M1500,1080 L1500,640 Q1620,470 1740,640 L1740,1080 Z" fill="#ffd9a0" {STROKE} stroke-width="8"/>'
    )
    giant = (
        f'<g {STROKE} stroke-width="10">'
        f'<path d="M560,1080 Q520,640 800,600 Q1080,640 1040,1080 Z" fill="#e9a57b"/>'
        f'<path d="M620,760 Q420,700 400,560 M980,760 Q1160,720 1230,600" fill="none" stroke-width="44" stroke="#e9a57b"/>'
        f'<path d="M620,760 Q420,700 400,560 M980,760 Q1160,720 1230,600" fill="none"/>'
        f'<circle cx="800" cy="440" r="190" fill="#f0b48c"/>'
        f'<circle cx="800" cy="400" r="74" fill="#fff"/><circle cx="812" cy="408" r="30" fill="{INK}" stroke="none"/>'
        f'<path d="M700,318 L890,290" stroke-width="18"/>'
        f'<path d="M700,530 Q800,600 900,530 Z" fill="#7a2a2a"/>'
        f'<path d="M730,536 l18,26 l18,-22 l18,24 l18,-24 l18,22 l18,-26" fill="#fff" stroke-width="5"/></g>'
    )
    held = sailor(1230, 640, 0.75, arms=((-50, -170), (50, -170)))
    crew = (
        sailor(1330, 990, 0.9)
        + sailor(1440, 1010, 0.8, arms=((-46, -150), (46, -150)))
        + odysseus(1620, 1010, 0.95, face="grim")
    )
    return cave + giant + held + crew


def scene_5() -> str:
    """Odysseus blinded the giant and escaped by clinging under its sheep."""
    hanging = (
        f'<g transform="translate(930,742) scale(1.25)" {STROKE} stroke-width="9" fill="none">'
        f'<path d="M-110,60 L80,60 M-110,60 L-150,-10 M-110,60 L-160,10 M80,60 L130,-10 M80,60 L150,20"/>'
        f'<circle cx="-146" cy="64" r="28" fill="{SKIN}"/>'
        f'<path d="M-177,62 A31,31 0 0 1 -115,62 L-115,50 L-177,50 Z" fill="{BRONZE}" stroke-width="7" transform="rotate(-90 -146 64)"/>'
        f'<circle cx="-150" cy="56" r="3.5" fill="{INK}" stroke="none"/><circle cx="-150" cy="74" r="3.5" fill="{INK}" stroke="none"/></g>'
    )
    giant_hand = (
        f'<g {STROKE} stroke-width="10"><path d="M1920,160 Q1500,180 1380,420" fill="none" stroke-width="70" stroke="#e9a57b"/>'
        f'<path d="M1920,125 Q1500,146 1350,410 M1920,196 Q1540,214 1412,440" fill="none"/>'
        f'<path d="M1320,470 l-40,60 M1370,480 l-10,70 M1420,476 l24,64" fill="none" stroke="#e9a57b" stroke-width="36"/></g>'
    )
    return (
        sky("#bfe3f2", "#fff1cf")
        + sun(300, 220, 100)
        + ground(820, "#9cc56a")
        + giant_hand
        + sheep(430, 860, 1.5)
        + sheep(900, 800, 2.3)
        + hanging
        + sheep(1480, 880, 1.4)
    )


def scene_6() -> str:
    """Then came the Sirens, whose song pulled sailors onto the rocks."""
    rock = f'<path d="M1180,760 Q1260,420 1420,400 Q1560,300 1700,460 Q1860,520 1900,760 Z" fill="#6f6a72" {STROKE} stroke-width="9"/>'
    wing = '<path d="M-6,-140 Q-90,-190 -110,-90 Q-70,-110 -6,-110 Z" fill="#f3ead7" stroke-width="6"/><path d="M6,-140 Q90,-190 110,-90 Q70,-110 6,-110 Z" fill="#f3ead7" stroke-width="6"/>'
    hair = '<path d="M-28,-186 Q0,-226 28,-186 Q34,-150 22,-130 M-28,-186 Q-34,-150 -22,-130" fill="none" stroke="#c8442f" stroke-width="10"/>'
    sirens = person(
        1400,
        420,
        1.2,
        face="sing",
        extra_back=wing,
        extra_front=hair,
        arms=((-60, -150), (60, -150)),
    ) + person(
        1640,
        470,
        1.1,
        face="sing",
        extra_back=wing,
        extra_front=hair,
        arms=((-60, -110), (60, -150)),
    )
    notes = (
        note(1080, 300)
        + note(900, 380, "#ffd166")
        + note(1010, 480)
        + note(760, 300, "#ffd166")
        + note(620, 420)
    )
    wrecks = f'<path d="M1150,790 l70,-90 l30,100 Z M1010,800 l40,-60 l30,70 Z" fill="{WOOD_DARK}" {STROKE} stroke-width="6"/>'
    return (
        sky("#5b4b8a", "#f2a0a0")
        + sun(300, 250, 90, "#ffe9b0")
        + sea(700, SEA_DARK, 4)
        + rock
        + sirens
        + notes
        + wrecks
        + ship(430, 850, 1.3)
    )


def scene_7() -> str:
    """He plugged his crew's ears with wax and had himself tied to the mast."""
    mast = f'<rect x="930" y="60" width="60" height="820" fill="{WOOD}" {STROKE} stroke-width="9"/><path d="M990,120 Q1500,300 990,560 Z" fill="#f3ead7" {STROKE} stroke-width="9"/>'
    ropes = "".join(
        f'<path d="M880,{y} Q960,{y + 26} 1040,{y}" fill="none" stroke="#e8d9a8" stroke-width="16" stroke-linecap="round"/>'
        f'<path d="M880,{y} Q960,{y + 26} 1040,{y}" fill="none" {STROKE} stroke-width="3"/>'
        for y in (560, 610, 660, 710)
    )
    deck = f'<path d="M120,880 L1800,880 Q1700,1040 1500,1080 L420,1080 Q220,1040 120,880 Z" fill="{WOOD}" {STROKE} stroke-width="10"/>'
    wax = '<circle cx="-30" cy="-178" r="9" fill="#ffe08a" stroke-width="4"/><circle cx="30" cy="-178" r="9" fill="#ffe08a" stroke-width="4"/>'
    crew = sailor(
        430, 880, 1.25, face="smile", extra_front=wax, arms=((-70, -70), (60, -60))
    ) + sailor(1480, 880, 1.25, face="smile", extra_front=wax, arms=((-60, -60), (70, -70)))
    hero = odysseus(960, 880, 1.9, face="scared", spear=False, arms=((-22, -84), (22, -84)))
    return (
        sky("#5b4b8a", "#f2a0a0")
        + sea(820, SEA_DARK, 2)
        + mast
        + deck
        + hero
        + ropes
        + crew
        + note(1380, 300)
        + note(1560, 420, "#ffd166")
        + note(520, 330)
    )


def scene_8() -> str:
    """Storms, monsters and an angry sea god took every ship and every man he had."""
    bolt = '<path d="M1180,60 L1060,360 L1160,360 L1040,640 L1300,290 L1190,290 L1290,60 Z" fill="#ffe14d" stroke="#2b2118" stroke-width="8" stroke-linejoin="round"/>'
    rain = "".join(
        f'<path d="M{x},{y} l-30,90" stroke="#cfe6f2" stroke-width="6" stroke-linecap="round" opacity="0.6"/>'
        for x in range(80, 1900, 130)
        for y in (120, 380)
    )
    trident = f'<g {STROKE} stroke-width="12" fill="none"><path d="M330,760 L330,240 M250,330 L250,240 M410,330 L410,240 M250,330 L410,330"/></g>'
    god = f'<g {STROKE} stroke-width="10"><path d="M40,760 Q60,430 250,420 Q440,430 470,760 Z" fill="#3f8f8a"/><circle cx="250" cy="330" r="96" fill="#7fc7c0"/><path d="M190,300 L240,318 M310,300 L262,318" stroke-width="12"/><circle cx="216" cy="332" r="9" fill="{INK}" stroke="none"/><circle cx="286" cy="332" r="9" fill="{INK}" stroke="none"/><path d="M200,390 Q250,360 300,390" fill="none"/><path d="M170,380 Q250,520 330,380 Q250,440 170,380 Z" fill="#e8f3f1"/></g>'
    raft = f'<g {STROKE} stroke-width="8"><rect x="1240" y="800" width="360" height="40" rx="12" fill="{WOOD}"/><path d="M1300,800 L1300,840 M1380,800 L1380,840 M1460,800 L1460,840 M1540,800 L1540,840" fill="none" stroke="{WOOD_DARK}" stroke-width="5"/></g>'
    wreck = f'<path d="M640,800 l140,-150 l50,170 Z" fill="{WOOD_DARK}" {STROKE} stroke-width="7"/><path d="M700,650 l60,-160" {STROKE} stroke-width="9"/>'
    return (
        sky("#1f2440", "#3f4d6b")
        + rain
        + bolt
        + sea(760, "#22506e", 4)
        + god
        + trident
        + wreck
        + raft
        + odysseus(1420, 800, 1.25, face="scared", spear=False, arms=((-60, -150), (60, -150)))
    )


def scene_9() -> str:
    """Twenty years after he left, he washed ashore at home, disguised as a beggar."""
    palace = (
        f'<g {STROKE} stroke-width="9"><path d="M1180,360 L1450,220 L1720,360 Z" fill="#f3ead7"/>'
        f'<rect x="1200" y="360" width="500" height="40" fill="#e6d9bd"/>'
        + "".join(
            f'<rect x="{x}" y="400" width="44" height="300" fill="#f3ead7"/>'
            for x in (1220, 1330, 1440, 1550, 1640)
        )
        + '<rect x="1180" y="700" width="540" height="40" fill="#e6d9bd"/></g>'
    )
    cloak = (
        '<path d="M-44,-150 Q0,-250 44,-150 L60,-10 L-60,-10 Z" fill="#8a6f52" stroke-width="8"/>'
    )
    face = f'<circle cx="0" cy="-172" r="24" fill="{SKIN}" stroke-width="7"/><circle cx="-8" cy="-176" r="3.2" fill="{INK}" stroke="none"/><circle cx="9" cy="-176" r="3.2" fill="{INK}" stroke="none"/><path d="M-18,-160 Q0,-118 18,-160 Q0,-146 -18,-160 Z" fill="#d9d4cc" stroke-width="5"/><path d="M52,-120 L70,0" stroke-width="9"/>'
    beggar = f'<g transform="translate(700,880) scale(1.9)" {STROKE} fill="none"><path d="M-20,-10 L-26,0 M20,-10 L26,0" stroke-width="9"/>{cloak}{face}</g>'
    tree = f'<circle cx="300" cy="470" r="130" fill="#5f9a3e" {STROKE} stroke-width="9"/><path d="M300,600 L300,800" {STROKE} stroke-width="22"/>'
    return (
        sky("#ffd9a0", "#fff1cf")
        + sun(960, 230, 100)
        + sea(640, SEA, 1)
        + ground(760, "#d9c38a", 20)
        + palace
        + tree
        + beggar
    )


def scene_10() -> str:
    """His wife Penelope had waited the whole time, fending off a hall full of suitors."""
    hall = (
        f'<rect width="{W}" height="{H}" fill="#f0dfbf"/><rect y="820" width="{W}" height="260" fill="#c9a877" {STROKE} stroke-width="8"/>'
        + "".join(
            f'<rect x="{x}" y="0" width="70" height="820" fill="#e6d1a8" {STROKE} stroke-width="7"/>'
            for x in (120, 1730)
        )
    )
    loom = (
        f'<g {STROKE} stroke-width="10" fill="none"><path d="M240,860 L240,300 L640,300 L640,860"/>'
        + "".join(
            f'<path d="M{x},310 L{x},700" stroke="#c8442f" stroke-width="5"/>'
            for x in range(280, 620, 34)
        )
        + '<rect x="262" y="520" width="356" height="180" fill="#7fb7c9" stroke-width="6"/></g>'
    )
    dress = '<path d="M0,-150 L-58,0 L58,0 Z" fill="#7b5ea7" stroke-width="8"/>'
    bun = '<circle cx="0" cy="-214" r="16" fill="#5a3a22" stroke-width="6"/><path d="M-28,-186 Q0,-222 28,-186" fill="none" stroke="#5a3a22" stroke-width="12"/>'
    penelope = person(
        760, 860, 1.7, face="grim", extra_back=dress, extra_front=bun, arms=((-70, -110), (40, -80))
    )
    suitors = "".join(
        person(
            x,
            880 - (i % 2) * 30,
            1.15,
            face="smile",
            arms=((-50, -160 + i * 8), (56, -100)),
            extra_front=f'<path d="M-10,-152 l10,26 l10,-26" fill="{RED}" stroke-width="4"/>',
        )
        for i, x in enumerate((1120, 1260, 1400, 1540, 1660))
    )
    return hall + loom + penelope + suitors


def scene_11() -> str:
    """One impossible shot with his old bow proved who he was. The king was home."""
    axes = "".join(
        f'<g transform="translate({x},0)" {STROKE} stroke-width="8"><path d="M0,860 L0,520" fill="none"/>'
        f'<path d="M-46,520 Q0,450 46,520 Q0,500 -46,520 Z" fill="#cfd4d8"/><circle cx="0" cy="560" r="16" fill="#f0dfbf"/></g>'
        for x in range(860, 1660, 130)
    )
    arrow = f'<g {STROKE} stroke-width="8" fill="none"><path d="M640,560 L1740,560"/><path d="M1740,560 l-34,-18 M1740,560 l-34,18"/></g>'
    bow = '<path d="M70,-240 Q150,-130 70,-20" stroke-width="10"/><path d="M70,-240 L26,-130 L70,-20" stroke-width="4"/>'
    hero = person(
        560,
        880,
        1.9,
        face="grim",
        extra_back=CAPE,
        extra_front=CROWN + BEARD + bow,
        arms=((26, -130), (70, -130)),
    )
    hall = f'<rect width="{W}" height="{H}" fill="#f0dfbf"/><rect y="860" width="{W}" height="220" fill="#c9a877" {STROKE} stroke-width="8"/>'
    rays = "".join(
        f'<path d="M560,520 L{x},{y}" stroke="#ffd166" stroke-width="10" opacity="0.5" stroke-linecap="round"/>'
        for x, y in ((120, 160), (300, 60), (560, 30), (820, 60), (1000, 160))
    )
    return hall + rays + axes + arrow + hero


SCENES = [
    scene_1,
    scene_2,
    scene_3,
    scene_4,
    scene_5,
    scene_6,
    scene_7,
    scene_8,
    scene_9,
    scene_10,
    scene_11,
]


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for number, scene in enumerate(SCENES, start=1):
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">'
            f"<title>{scene.__doc__}</title>{scene()}</svg>\n"
        )
        (out_dir / f"s{number:03d}.svg").write_text(svg, encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("examples/odyssey-cartoon/art-src"))
