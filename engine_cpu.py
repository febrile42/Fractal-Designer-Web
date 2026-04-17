"""CPU (numpy) fallback for fractal rendering — no CUDA required."""

import math
import random
import numpy as np
from PIL import Image

_BURN = 20


def prepare_transforms(transforms, var_names):
    return transforms, list(var_names)


def launch(tdata, width, height, iterations, symmetry, zoom, rotation, center,
           counts, color_sum, seed_offset=0, base_seed=42):
    transforms, _ = tdata
    if iterations <= 0:
        return

    rng = random.Random(int(base_seed) ^ (int(seed_offset) * 1_000_003))

    cum = []
    run = 0.0
    for t in transforms:
        run += t.weight
        cum.append(run)

    scale = min(width, height) * 0.35 * zoom
    rr = math.radians(rotation)
    cosr, sinr = math.cos(rr), math.sin(rr)
    cx, cy = float(center[0]), float(center[1])

    sa = 2.0 * math.pi / symmetry
    sym_cos = [math.cos(s * sa) for s in range(symmetry)]
    sym_sin = [math.sin(s * sa) for s in range(symmetry)]

    x = rng.uniform(-1.0, 1.0)
    y = rng.uniform(-1.0, 1.0)
    c = 0.0

    for i in range(_BURN + iterations):
        r = rng.random()
        ti = len(transforms) - 1
        for j, cw in enumerate(cum):
            if r <= cw:
                ti = j
                break

        t = transforms[ti]
        x, y = t.apply(x, y)
        c = (c + t.color) * 0.5

        if not (math.isfinite(x) and math.isfinite(y)):
            x = rng.uniform(-1.0, 1.0)
            y = rng.uniform(-1.0, 1.0)
            c = 0.0
            continue

        if i < _BURN:
            continue

        for s in range(symmetry):
            rx = x * sym_cos[s] - y * sym_sin[s]
            ry = x * sym_sin[s] + y * sym_cos[s]
            wx = (rx - cx) * cosr - (ry - cy) * sinr
            wy = (rx - cx) * sinr + (ry - cy) * cosr
            px = int(wx * scale + width * 0.5)
            py = int(wy * scale + height * 0.5)
            if 0 <= px < width and 0 <= py < height:
                counts[py, px] += 1.0
                color_sum[py, px] += c


def gpu_tone_map(counts, gamma, brightness):
    c = np.clip(counts, 0, None).astype(np.float32)
    mx = float(c.max())
    if mx == 0:
        return np.zeros_like(c, dtype=np.float32)
    log_c = np.log1p(c)
    log_mx = math.log1p(mx)
    alpha = (log_c / np.float32(log_mx)).astype(np.float32)
    return np.clip(alpha ** np.float32(1.0 / gamma) * np.float32(brightness),
                   0.0, 1.0).astype(np.float32)


def gpu_make_image(counts, color_sum, config, ss):
    from palettes import get_palette

    safe = np.where(counts > 0, counts, np.float32(1.0))
    color_avg = np.where(counts > 0, color_sum / safe, np.float32(0.0))

    lum = gpu_tone_map(counts, config["gamma"], config["brightness"])

    palette_name = config.get("palette", "fire")
    stops = palette_name if isinstance(palette_name, list) else get_palette(palette_name)
    stops_arr = np.array(stops, dtype=np.float32)
    n = len(stops) - 1

    t_cl = np.clip(color_avg, 0.0, 1.0)
    scaled = t_cl * n
    lo = np.floor(scaled).astype(np.int32)
    hi = np.minimum(lo + 1, n)
    frac = (scaled - lo.astype(np.float32))[..., np.newaxis]
    pal_rgb = stops_arr[lo] + (stops_arr[hi] - stops_arr[lo]) * frac

    vibrancy = float(np.clip(config.get("vibrancy", 1.0), 0.0, 1.0))
    bg = config.get("background", [0, 0, 0])

    lum3 = lum[:, :, np.newaxis]
    flat = pal_rgb * lum3
    safe_v = max(vibrancy, 0.01)
    vib = pal_rgb * (lum3 ** np.float32(1.0 / safe_v))
    blended = flat * (1 - vibrancy) + vib * vibrancy
    blended = np.clip(np.round(blended), 0, 255).astype(np.uint8)

    mask = lum < 1e-6
    blended[mask] = np.array(bg, dtype=np.uint8)

    img = Image.fromarray(blended, "RGB")
    if ss > 1:
        img = img.resize((config["width"], config["height"]), Image.LANCZOS)
    return img
