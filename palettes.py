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
}


def get_palette(name: str) -> list[tuple[int, int, int]]:
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
        int(r0 + (r1 - r0) * frac),
        int(g0 + (g1 - g0) * frac),
        int(b0 + (b1 - b0) * frac),
    )
