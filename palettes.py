import json
import os

_CUSTOM_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "palettes_custom.json")

PALETTES = {
    "fire": [
        (0, 0, 0),
        (80, 0, 0),
        (200, 60, 0),
        (255, 160, 0),
        (255, 230, 100),
        (255, 255, 255),
    ],
    "ice": [
        (0, 0, 0),
        (0, 30, 60),
        (0, 80, 120),
        (0, 180, 200),
        (100, 230, 240),
        (255, 255, 255),
    ],
    "neon": [
        (0, 0, 0),
        (60, 0, 80),
        (180, 0, 180),
        (255, 0, 120),
        (255, 80, 200),
        (255, 255, 255),
    ],
    "rainbow": [
        (255, 0, 0),
        (255, 165, 0),
        (255, 255, 0),
        (0, 200, 0),
        (0, 100, 255),
        (150, 0, 255),
        (255, 0, 0),
    ],
    "gold": [
        (0, 0, 0),
        (60, 30, 0),
        (160, 100, 0),
        (255, 200, 0),
        (255, 240, 160),
        (255, 255, 255),
    ],
    "violet": [
        (0, 0, 0),
        (20, 0, 60),
        (80, 0, 160),
        (160, 60, 255),
        (220, 160, 255),
        (255, 255, 255),
    ],
    "copper": [
        (0, 0, 0),
        (50, 20, 0),
        (150, 60, 20),
        (220, 120, 60),
        (255, 200, 150),
        (255, 255, 255),
    ],
    "cyber": [
        (0, 0, 0),
        (0, 50, 20),
        (0, 150, 100),
        (0, 255, 180),
        (150, 255, 230),
        (255, 255, 255),
    ],
    "rose_gold": [
        (0, 0, 0),
        (60, 30, 40),
        (180, 100, 110),
        (255, 180, 190),
        (255, 220, 210),
        (255, 255, 255),
    ],
    "forest": [
        (0, 0, 0),
        (5, 30, 10),
        (20, 80, 30),
        (60, 180, 80),
        (180, 240, 160),
        (255, 255, 255),
    ],
    "obsidian": [
        (0, 0, 0),
        (20, 20, 25),
        (50, 50, 65),
        (120, 120, 140),
        (200, 200, 215),
        (255, 255, 255),
    ],
    "solar": [
        (0, 0, 0),
        (80, 0, 0),
        (200, 50, 0),
        (255, 180, 0),
        (255, 255, 150),
        (255, 255, 255),
    ],
    "synthwave": [
        (0, 0, 0),
        (40, 0, 100),
        (100, 0, 255),
        (255, 0, 255),
        (0, 255, 255),
        (255, 255, 255),
    ],
    "midnight": [
        (0, 0, 0),
        (5, 10, 30),
        (20, 40, 80),
        (80, 120, 180),
        (180, 200, 220),
        (255, 255, 255),
    ],
    "toxic": [
        (0, 0, 0),
        (0, 40, 40),
        (0, 120, 100),
        (100, 255, 150),
        (200, 255, 200),
        (255, 255, 255),
    ],
    "crimson": [
        (0, 0, 0),
        (50, 0, 0),
        (150, 10, 10),
        (255, 50, 50),
        (255, 180, 180),
        (255, 255, 255),
    ],
}


def _load_custom() -> dict:
    if not os.path.exists(_CUSTOM_FILE):
        return {}
    try:
        with open(_CUSTOM_FILE, "r") as f:
            data = json.load(f)
        return {name: [tuple(c) for c in stops] for name, stops in data.items()}
    except Exception:
        return {}


def save_custom_palette(name: str, stops: list) -> None:
    existing = _load_custom()
    existing[name] = [list(c) for c in stops]
    with open(_CUSTOM_FILE, "w") as f:
        json.dump(existing, f, indent=2)


def delete_custom_palette(name: str) -> None:
    existing = _load_custom()
    if name in existing:
        del existing[name]
        with open(_CUSTOM_FILE, "w") as f:
            json.dump(existing, f, indent=2)


def get_all_palette_names() -> list:
    custom = _load_custom()
    names = list(PALETTES.keys())
    for k in custom:
        if k not in PALETTES:
            names.append(k)
    return names


def get_palette(name: str) -> list:
    custom = _load_custom()
    if name in custom:
        return custom[name]
    return PALETTES[name]


def sample_palette(stops: list[tuple[int, int, int]], t: float) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    n = len(stops) - 1
    scaled = t * n
    lo = int(scaled)
    hi = min(lo + 1, n)
    frac = scaled - lo
    r0, g0, b0 = stops[lo]
    r1, g1, b1 = stops[hi]
    return (
        round(r0 + (r1 - r0) * frac),
        round(g0 + (g1 - g0) * frac),
        round(b0 + (b1 - b0) * frac),
    )
