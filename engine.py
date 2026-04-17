from __future__ import annotations

import math
import random
import numpy as np
from collections.abc import Generator
from dataclasses import dataclass, field
from variations import VARIATION_REGISTRY
from palettes import get_palette
from PIL import Image

try:
    import cupy as cp
    from engine_gpu import prepare_transforms, launch, gpu_tone_map, gpu_make_image
    GPU_AVAILABLE = True
    BACKEND = "cuda"
    _to_numpy = cp.asnumpy
except (ImportError, ModuleNotFoundError):
    import numpy as cp  # type: ignore[assignment]
    _to_numpy = np.asarray
    try:
        from engine_metal import (  # type: ignore[assignment]
            prepare_transforms, launch,
            gpu_tone_map, gpu_make_image,
        )
        GPU_AVAILABLE = True
        BACKEND = "metal"
    except (ImportError, ModuleNotFoundError, Exception):
        from engine_cpu import (  # type: ignore[assignment]
            prepare_transforms, launch,
            gpu_tone_map, gpu_make_image,
        )
        GPU_AVAILABLE = False
        BACKEND = "cpu"

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
    if symmetry < 1:
        raise ValueError(f"symmetry must be >= 1, got {symmetry}")

    var_names = list(transforms[0].variations.keys())
    tdata = prepare_transforms(transforms, var_names)
    counts_gpu = cp.zeros((height, width), dtype=cp.float32)
    csum_gpu = cp.zeros((height, width), dtype=cp.float32)

    launch(tdata, width, height, iterations, symmetry, zoom, rotation,
           center, counts_gpu, csum_gpu)

    # Convert accumulated color_sum to color average for API compat
    safe = cp.where(counts_gpu > 0, counts_gpu, cp.float32(1.0))
    cavg = cp.where(counts_gpu > 0, csum_gpu / safe, cp.float32(0.0))

    return (_to_numpy(counts_gpu).astype(np.float64),
            _to_numpy(cavg).astype(np.float64))


def run_chaos_game_partial(
    transforms: list[Transform],
    width: int,
    height: int,
    iterations: int,
    symmetry: int,
    zoom: float,
    rotation: float,
    center: tuple[float, float],
    counts: np.ndarray,
    colors: np.ndarray,
    state: int | None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Run iterations on GPU, accumulating into existing arrays.

    state is an integer batch counter (or None for first call).
    Returns (counts, color_sum, next_state).
    """
    if symmetry < 1:
        raise ValueError(f"symmetry must be >= 1, got {symmetry}")

    var_names = list(transforms[0].variations.keys())
    batch = 0 if state is None else state
    tdata = prepare_transforms(transforms, var_names)

    counts_gpu = cp.asarray(counts, dtype=cp.float32)
    csum_gpu = cp.asarray(colors, dtype=cp.float32)

    launch(tdata, width, height, iterations, symmetry, zoom, rotation,
           tuple(center), counts_gpu, csum_gpu, seed_offset=batch)

    return (_to_numpy(counts_gpu).astype(np.float64),
            _to_numpy(csum_gpu).astype(np.float64),
            batch + 1)


def tone_map(
    counts: np.ndarray,
    gamma: float,
    brightness: float,
) -> np.ndarray:
    if gamma <= 0:
        raise ValueError(f"gamma must be > 0, got {gamma}")
    counts_gpu = cp.asarray(counts, dtype=cp.float32)
    result = gpu_tone_map(counts_gpu, gamma, brightness)
    return _to_numpy(result).astype(np.float64)


def _make_image(
    counts: np.ndarray,
    colors: np.ndarray,
    config: dict,
    ss: int,
) -> Image.Image:
    """Tone-map and colorize accumulated arrays into a PIL Image.

    `colors` is color averages in [0, 1]. GPU is used for all computation.
    """
    counts_gpu = cp.asarray(counts, dtype=cp.float32)
    colors_gpu = cp.asarray(colors, dtype=cp.float32)
    # Reconstruct color_sum from averages so gpu_make_image can work
    csum_gpu = colors_gpu * counts_gpu
    return gpu_make_image(counts_gpu, csum_gpu, config, ss)


# Logarithmic pass fractions for streaming preview
_STREAM_FRACTIONS = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.37]


def render_stream(
    config: dict,
    cancel_event=None,
) -> Generator[tuple[Image.Image, float], None, None]:
    """Yield (PIL.Image, progress) at logarithmic iteration checkpoints.

    All computation runs on GPU. Only the final PIL image transfers to CPU.
    If cancel_event (threading.Event) is set between yields, stops early.
    """
    ss = config.get("supersample", 1)
    w = config["width"] * ss
    h = config["height"] * ss
    total = config["quality"] * w * h
    seed = config.get("seed", 42)

    transforms = build_transforms(
        num=config["num_transforms"],
        seed=seed,
        variations=config["variations"],
    )

    var_names = config["variations"]
    tdata = prepare_transforms(transforms, var_names)

    # GPU arrays persist across chunks — no CPU round-trip between frames
    counts_gpu = cp.zeros((h, w), dtype=cp.float32)
    csum_gpu = cp.zeros((h, w), dtype=cp.float32)
    done = 0

    for i, frac in enumerate(_STREAM_FRACTIONS):
        if cancel_event is not None and cancel_event.is_set():
            return

        if i == len(_STREAM_FRACTIONS) - 1:
            chunk = max(1, total - done)
        else:
            chunk = max(1, int(total * frac))

        launch(tdata, w, h, chunk, config["symmetry"], config["zoom"],
               config["rotation"], tuple(config["center"]),
               counts_gpu, csum_gpu, seed_offset=i, base_seed=seed)
        done += chunk

        img = gpu_make_image(counts_gpu, csum_gpu, config, ss)
        yield img, min(done / total, 1.0)


def render(config: dict) -> Image.Image:
    ss = config.get("supersample", 1)
    w = config["width"] * ss
    h = config["height"] * ss
    iterations = config["quality"] * w * h
    seed = config.get("seed", 42)

    transforms = build_transforms(
        num=config["num_transforms"],
        seed=seed,
        variations=config["variations"],
    )

    var_names = config["variations"]
    tdata = prepare_transforms(transforms, var_names)

    counts_gpu = cp.zeros((h, w), dtype=cp.float32)
    csum_gpu = cp.zeros((h, w), dtype=cp.float32)

    launch(tdata, w, h, iterations, config["symmetry"], config["zoom"],
           config["rotation"], tuple(config["center"]),
           counts_gpu, csum_gpu, base_seed=seed)

    return gpu_make_image(counts_gpu, csum_gpu, config, ss)
