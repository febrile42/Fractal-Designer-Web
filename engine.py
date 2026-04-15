import math
import random
import numpy as np
from dataclasses import dataclass, field
from variations import VARIATION_REGISTRY
from palettes import get_palette
from PIL import Image

_CHAOS_BURN_IN = 20

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
    _var_total: float = field(init=False, repr=False)

    def __post_init__(self):
        self._var_total = sum(self.variations.values()) or 1.0

    def apply(self, x: float, y: float) -> tuple[float, float]:
        # Affine step
        nx = self.a * x + self.b * y + self.c
        ny = self.d * x + self.e * y + self.f
        # Variation blending
        ox, oy = 0.0, 0.0
        total = self._var_total
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

    if symmetry < 1:
        raise ValueError(f"symmetry must be >= 1, got {symmetry}")

    x, y = random.uniform(-1, 1), random.uniform(-1, 1)
    color_coord = 0.0

    sym_angle = 2.0 * math.pi / symmetry
    sym_table = [
        (math.cos(s * sym_angle), math.sin(s * sym_angle))
        for s in range(symmetry)
    ]

    for i in range(iterations + _CHAOS_BURN_IN):
        t = pick_transform()
        x, y = t.apply(x, y)
        color_coord = (color_coord + t.color) * 0.5

        # Reset if orbit escaped to infinity
        if not (math.isfinite(x) and math.isfinite(y)):
            x, y = random.uniform(-1, 1), random.uniform(-1, 1)
            color_coord = 0.0
            continue

        if i < _CHAOS_BURN_IN:
            continue

        # Apply symmetry — plot point + rotated copies
        for cos_s, sin_s in sym_table:
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


def tone_map(
    counts: np.ndarray,
    gamma: float,
    brightness: float,
) -> np.ndarray:
    if gamma <= 0:
        raise ValueError(f"gamma must be > 0, got {gamma}")
    counts = np.clip(counts, 0, None)   # guard against negative counts
    max_count = counts.max()
    if max_count == 0:
        return np.zeros_like(counts, dtype=np.float64)
    log_counts = np.log1p(counts)
    log_max = math.log1p(max_count)
    alpha = log_counts / log_max
    return np.clip(alpha ** (1.0 / gamma) * brightness, 0.0, 1.0)


def render(config: dict) -> Image.Image:
    ss = config.get("supersample", 1)
    w = config["width"] * ss
    h = config["height"] * ss
    quality = config["quality"]
    iterations = quality * w * h

    transforms = build_transforms(
        num=config["num_transforms"],
        seed=config["seed"],
        variations=config["variations"],
    )

    counts, color_coords = run_chaos_game(
        transforms=transforms,
        width=w,
        height=h,
        iterations=iterations,
        symmetry=config["symmetry"],
        zoom=config["zoom"],
        rotation=config["rotation"],
        center=tuple(config["center"]),
    )

    luminance = tone_map(counts, config["gamma"], config["brightness"])

    palette_name = config.get("palette", "fire")
    if isinstance(palette_name, list):
        stops = palette_name
    else:
        stops = get_palette(palette_name)

    vibrancy = float(np.clip(config.get("vibrancy", 1.0), 0.0, 1.0))
    bg = config.get("background", [0, 0, 0])

    # Vectorized palette sampling
    stops_arr = np.array(stops, dtype=np.float64)          # (N, 3)
    n = len(stops) - 1
    t_clamped = np.clip(color_coords, 0.0, 1.0)
    scaled = t_clamped * n
    lo = np.floor(scaled).astype(np.int32)
    hi = np.minimum(lo + 1, n)
    frac = (scaled - lo)[..., np.newaxis]                  # (h, w, 1) for broadcasting
    palette_rgb = stops_arr[lo] + (stops_arr[hi] - stops_arr[lo]) * frac

    lum3 = luminance[:, :, np.newaxis]                          # (h, w, 1)
    flat_pixels = palette_rgb * lum3                            # luminance-scaled

    safe_vib = max(vibrancy, 0.01)
    vib_pixels = palette_rgb * (lum3 ** (1.0 / safe_vib))      # vibrancy-scaled

    blended = flat_pixels * (1 - vibrancy) + vib_pixels * vibrancy
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    # Apply background where luminance is near zero
    mask = luminance < 1e-6
    blended[mask] = bg

    pixels = blended

    img = Image.fromarray(pixels, "RGB")
    if ss > 1:
        img = img.resize((config["width"], config["height"]), Image.LANCZOS)
    return img
