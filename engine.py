import math
import random
import numpy as np
from dataclasses import dataclass, field
from variations import VARIATION_REGISTRY

@dataclass
class Transform:
    a: float
    b: float
    c: float
    d: float
    e: float
    f: float
    weight: float
    color: float                          # [0, 1] palette coordinate
    variations: dict[str, float]          # {variation_name: weight}

    def apply(self, x: float, y: float) -> tuple[float, float]:
        # Affine step
        nx = self.a * x + self.b * y + self.c
        ny = self.d * x + self.e * y + self.f
        # Variation blending
        ox, oy = 0.0, 0.0
        total = sum(self.variations.values())
        for name, w in self.variations.items():
            fn = VARIATION_REGISTRY[name]
            vx, vy = fn(nx, ny)
            ox += (w / total) * vx
            oy += (w / total) * vy
        return ox, oy


def build_transforms(
    num: int,
    seed: int,
    variations: list[str],
) -> list[Transform]:
    rng = random.Random(seed)
    transforms = []
    raw_weights = [rng.uniform(0.3, 1.0) for _ in range(num)]
    total = sum(raw_weights)
    for i in range(num):
        # Random affine coefficients — bounded to keep orbits from exploding
        a = rng.uniform(-1.0, 1.0)
        b = rng.uniform(-0.5, 0.5)
        c = rng.uniform(-0.5, 0.5)
        d = rng.uniform(-0.5, 0.5)
        e = rng.uniform(-1.0, 1.0)
        f = rng.uniform(-0.5, 0.5)
        color = i / max(num - 1, 1)
        var_weights = {v: rng.uniform(0.1, 1.0) for v in variations}
        transforms.append(Transform(
            a=a, b=b, c=c, d=d, e=e, f=f,
            weight=raw_weights[i] / total,
            color=color,
            variations=var_weights,
        ))
    return transforms


def run_chaos_game(
    transforms: list[Transform],
    width: int,
    height: int,
    iterations: int,
    symmetry: int,
    zoom: float,
    rotation: float,
    center: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.zeros((height, width), dtype=np.float64)
    colors = np.zeros((height, width), dtype=np.float64)

    # Camera: scale and rotation matrix
    scale = min(width, height) * 0.35 * zoom
    rot_rad = math.radians(rotation)
    cos_r = math.cos(rot_rad)
    sin_r = math.sin(rot_rad)
    cx, cy = center

    # Cumulative weights for weighted random selection
    cum_weights = []
    running = 0.0
    for t in transforms:
        running += t.weight
        cum_weights.append(running)

    def pick_transform() -> Transform:
        r = random.random()
        for i, cw in enumerate(cum_weights):
            if r <= cw:
                return transforms[i]
        return transforms[-1]

    x, y = random.uniform(-1, 1), random.uniform(-1, 1)
    color_coord = 0.0
    BURN_IN = 20

    sym_angle = 2.0 * math.pi / symmetry

    for i in range(iterations + BURN_IN):
        t = pick_transform()
        x, y = t.apply(x, y)
        color_coord = (color_coord + t.color) * 0.5

        if i < BURN_IN:
            continue

        # Apply symmetry — plot point + rotated copies
        for s in range(symmetry):
            angle = s * sym_angle
            cos_s = math.cos(angle)
            sin_s = math.sin(angle)
            rx = x * cos_s - y * sin_s
            ry = x * sin_s + y * cos_s

            # Camera transform
            wx = (rx - cx) * cos_r - (ry - cy) * sin_r
            wy = (rx - cx) * sin_r + (ry - cy) * cos_r

            px = int(wx * scale + width / 2)
            py = int(wy * scale + height / 2)

            if 0 <= px < width and 0 <= py < height:
                counts[py, px] += 1
                colors[py, px] = (colors[py, px] + color_coord) * 0.5

    return counts, colors
