# Flame Fractal Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python flame fractal generator with a Dear PyGui GUI for real-time parameter control and preview rendering.

**Architecture:** Four modules — `variations.py` (math functions), `palettes.py` (color data), `engine.py` (IFS chaos game + tone mapping), and `fractal.py` (Dear PyGui GUI + threading). The engine has no GUI dependency and can be tested independently.

**Tech Stack:** Python 3.10+, NumPy, Pillow, Dear PyGui

---

## File Map

| File | Responsibility |
|---|---|
| `requirements.txt` | Pin dependencies |
| `variations.py` | All variation functions (sinusoidal, swirl, spherical, etc.) |
| `palettes.py` | Built-in palettes as RGB stop lists + interpolation |
| `engine.py` | Transform dataclass, chaos game, tone mapping, render() |
| `fractal.py` | Dear PyGui GUI, threading, preview/save |
| `tests/test_variations.py` | Unit tests for variation functions |
| `tests/test_palettes.py` | Unit tests for palette interpolation |
| `tests/test_engine.py` | Unit tests for engine components |

---

## Task 1: Project Setup

**Files:**
- Create: `requirements.txt`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create requirements.txt**

```
numpy>=1.24
pillow>=10.0
dearpygui>=1.11
pytest>=7.0
```

- [ ] **Step 2: Create tests package**

```bash
mkdir -p tests
touch tests/__init__.py
```

- [ ] **Step 3: Install dependencies**

```bash
pip install -r requirements.txt
```

Expected: All packages install without error.

- [ ] **Step 4: Verify installs**

```bash
python -c "import numpy; import PIL; import dearpygui.dearpygui as dpg; print('OK')"
```

Expected output: `OK`

- [ ] **Step 5: Commit**

```bash
git init
git add requirements.txt tests/__init__.py
git commit -m "chore: project setup with dependencies"
```

---

## Task 2: Variation Functions

**Files:**
- Create: `variations.py`
- Create: `tests/test_variations.py`

Variation functions take `(x, y)` floats and return a transformed `(x, y)` tuple. They are the mathematical heart of flame fractals — each one creates a distinct visual character.

- [ ] **Step 1: Write failing tests**

Create `tests/test_variations.py`:

```python
import math
import pytest
from variations import linear, sinusoidal, spherical, swirl, horseshoe, polar, curl, pdj

def test_linear_identity():
    assert linear(1.0, 2.0) == (1.0, 2.0)

def test_sinusoidal():
    x, y = sinusoidal(math.pi / 2, 0.0)
    assert abs(x - 1.0) < 1e-9
    assert abs(y - 0.0) < 1e-9

def test_spherical_unit():
    # spherical maps (1,0) → (1,0) since r=1
    x, y = spherical(1.0, 0.0)
    assert abs(x - 1.0) < 1e-6
    assert abs(y - 0.0) < 1e-6

def test_spherical_origin_returns_zero():
    # near origin should not explode — clamped
    x, y = spherical(0.0, 0.0)
    assert x == 0.0 and y == 0.0

def test_swirl_preserves_radius():
    import math
    x, y = swirl(1.0, 0.0)
    r = math.sqrt(x**2 + y**2)
    assert abs(r - 1.0) < 1e-6

def test_horseshoe_origin():
    x, y = horseshoe(0.0, 0.0)
    assert x == 0.0 and y == 0.0

def test_polar():
    # polar(1, 0): theta=0, r=1 → (0/pi, 1-1) = (0, 0)
    x, y = polar(1.0, 0.0)
    assert abs(x - 0.0) < 1e-9
    assert abs(y - 0.0) < 1e-9

def test_curl_origin():
    x, y = curl(0.0, 0.0, c1=0.5, c2=0.1)
    assert x == 0.0 and y == 0.0

def test_pdj_returns_tuple():
    x, y = pdj(0.5, 0.3, a=1.5, b=1.8, c=1.2, d=2.0)
    assert isinstance(x, float)
    assert isinstance(y, float)
```

- [ ] **Step 2: Run tests — expect failures**

```bash
pytest tests/test_variations.py -v
```

Expected: `ModuleNotFoundError: No module named 'variations'`

- [ ] **Step 3: Implement variations.py**

```python
import math

EPS = 1e-10

def linear(x: float, y: float) -> tuple[float, float]:
    return x, y

def sinusoidal(x: float, y: float) -> tuple[float, float]:
    return math.sin(x), math.sin(y)

def spherical(x: float, y: float) -> tuple[float, float]:
    r2 = x * x + y * y
    if r2 < EPS:
        return 0.0, 0.0
    inv = 1.0 / r2
    return x * inv, y * inv

def swirl(x: float, y: float) -> tuple[float, float]:
    r2 = x * x + y * y
    sin_r2 = math.sin(r2)
    cos_r2 = math.cos(r2)
    return x * sin_r2 - y * cos_r2, x * cos_r2 + y * sin_r2

def horseshoe(x: float, y: float) -> tuple[float, float]:
    r = math.sqrt(x * x + y * y)
    if r < EPS:
        return 0.0, 0.0
    inv = 1.0 / r
    return inv * (x - y) * (x + y), inv * 2.0 * x * y

def polar(x: float, y: float) -> tuple[float, float]:
    r = math.sqrt(x * x + y * y)
    theta = math.atan2(x, y)
    return theta / math.pi, r - 1.0

def curl(x: float, y: float, c1: float = 0.5, c2: float = 0.1) -> tuple[float, float]:
    t1 = 1.0 + c1 * x + c2 * (x * x - y * y)
    t2 = c1 * y + 2.0 * c2 * x * y
    r2 = t1 * t1 + t2 * t2
    if r2 < EPS:
        return 0.0, 0.0
    inv = 1.0 / r2
    return (x * t1 + y * t2) * inv, (y * t1 - x * t2) * inv

def pdj(x: float, y: float, a: float = 1.5, b: float = 1.8,
        c: float = 1.2, d: float = 2.0) -> tuple[float, float]:
    return math.sin(a * y) - math.cos(b * x), math.sin(c * x) - math.cos(d * y)

# Registry: name → function
VARIATION_REGISTRY = {
    "linear": linear,
    "sinusoidal": sinusoidal,
    "spherical": spherical,
    "swirl": swirl,
    "horseshoe": horseshoe,
    "polar": polar,
    "curl": curl,
    "pdj": pdj,
}
```

- [ ] **Step 4: Run tests — expect all pass**

```bash
pytest tests/test_variations.py -v
```

Expected: 9 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add variations.py tests/test_variations.py
git commit -m "feat: add variation functions for flame fractal"
```

---

## Task 3: Color Palettes

**Files:**
- Create: `palettes.py`
- Create: `tests/test_palettes.py`

Palettes are lists of `(r, g, b)` stops in `[0, 255]`. `sample_palette(palette, t)` interpolates smoothly for any `t` in `[0, 1]`.

- [ ] **Step 1: Write failing tests**

Create `tests/test_palettes.py`:

```python
import pytest
from palettes import sample_palette, get_palette, PALETTES

def test_sample_at_zero_returns_first_stop():
    stops = [(0, 0, 0), (255, 255, 255)]
    r, g, b = sample_palette(stops, 0.0)
    assert (r, g, b) == (0, 0, 0)

def test_sample_at_one_returns_last_stop():
    stops = [(0, 0, 0), (255, 255, 255)]
    r, g, b = sample_palette(stops, 1.0)
    assert (r, g, b) == (255, 255, 255)

def test_sample_midpoint():
    stops = [(0, 0, 0), (100, 200, 50)]
    r, g, b = sample_palette(stops, 0.5)
    assert abs(r - 50) < 1
    assert abs(g - 100) < 1
    assert abs(b - 25) < 1

def test_sample_clamps_t():
    stops = [(0, 0, 0), (255, 0, 0)]
    r, g, b = sample_palette(stops, 1.5)
    assert (r, g, b) == (255, 0, 0)

def test_get_palette_fire():
    stops = get_palette("fire")
    assert len(stops) >= 2
    assert all(len(s) == 3 for s in stops)

def test_get_palette_unknown_raises():
    with pytest.raises(KeyError):
        get_palette("nonexistent_palette")

def test_all_builtin_palettes_valid():
    for name in PALETTES:
        stops = get_palette(name)
        assert len(stops) >= 2
        for r, g, b in stops:
            assert 0 <= r <= 255
            assert 0 <= g <= 255
            assert 0 <= b <= 255
```

- [ ] **Step 2: Run tests — expect failures**

```bash
pytest tests/test_palettes.py -v
```

Expected: `ModuleNotFoundError: No module named 'palettes'`

- [ ] **Step 3: Implement palettes.py**

```python
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
```

- [ ] **Step 4: Run tests — expect all pass**

```bash
pytest tests/test_palettes.py -v
```

Expected: 7 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add palettes.py tests/test_palettes.py
git commit -m "feat: add color palettes with smooth interpolation"
```

---

## Task 4: Transform Dataclass & Chaos Game

**Files:**
- Create: `engine.py`
- Create: `tests/test_engine.py`

The `Transform` dataclass holds the affine coefficients, variation weights, color coordinate, and probability. `build_transforms()` generates N random transforms from a seed. `run_chaos_game()` iterates the chaos game and returns a raw histogram.

- [ ] **Step 1: Write failing tests**

Create `tests/test_engine.py`:

```python
import math
import numpy as np
import pytest
from engine import Transform, build_transforms, run_chaos_game

def test_transform_has_required_fields():
    t = Transform(
        a=1.0, b=0.0, c=0.0, d=0.0, e=1.0, f=0.0,
        weight=1.0,
        color=0.5,
        variations={"linear": 1.0},
    )
    assert t.color == 0.5
    assert t.weight == 1.0

def test_build_transforms_count():
    transforms = build_transforms(num=4, seed=42, variations=["swirl", "sinusoidal"])
    assert len(transforms) == 4

def test_build_transforms_weights_sum_to_one():
    transforms = build_transforms(num=4, seed=42, variations=["linear"])
    total = sum(t.weight for t in transforms)
    assert abs(total - 1.0) < 1e-9

def test_build_transforms_same_seed_reproducible():
    t1 = build_transforms(num=3, seed=99, variations=["swirl"])
    t2 = build_transforms(num=3, seed=99, variations=["swirl"])
    assert t1[0].a == t2[0].a
    assert t1[0].b == t2[0].b

def test_build_transforms_different_seeds_differ():
    t1 = build_transforms(num=3, seed=1, variations=["linear"])
    t2 = build_transforms(num=3, seed=2, variations=["linear"])
    assert t1[0].a != t2[0].a

def test_chaos_game_returns_correct_shape():
    transforms = build_transforms(num=3, seed=42, variations=["linear"])
    counts, colors = run_chaos_game(
        transforms=transforms,
        width=64, height=64,
        iterations=10000,
        symmetry=1,
        zoom=1.0, rotation=0.0, center=(0.0, 0.0),
    )
    assert counts.shape == (64, 64)
    assert colors.shape == (64, 64)

def test_chaos_game_has_nonzero_hits():
    transforms = build_transforms(num=3, seed=42, variations=["linear"])
    counts, colors = run_chaos_game(
        transforms=transforms,
        width=64, height=64,
        iterations=10000,
        symmetry=1,
        zoom=1.0, rotation=0.0, center=(0.0, 0.0),
    )
    assert counts.max() > 0
```

- [ ] **Step 2: Run tests — expect failures**

```bash
pytest tests/test_engine.py -v
```

Expected: `ModuleNotFoundError: No module named 'engine'`

- [ ] **Step 3: Implement Transform dataclass and build_transforms in engine.py**

```python
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
```

- [ ] **Step 4: Implement run_chaos_game in engine.py**

Append to `engine.py`:

```python
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
```

- [ ] **Step 5: Run tests — expect all pass**

```bash
pytest tests/test_engine.py -v
```

Expected: 7 tests PASSED

- [ ] **Step 6: Commit**

```bash
git add engine.py tests/test_engine.py
git commit -m "feat: Transform dataclass, build_transforms, chaos game"
```

---

## Task 5: Tone Mapping & render()

**Files:**
- Modify: `engine.py` — add `tone_map()` and `render()`
- Modify: `tests/test_engine.py` — add tone mapping + render tests

`tone_map()` converts raw histogram counts into a luminance array via log-density. `render()` is the top-level function the GUI calls — it orchestrates everything and returns a PIL Image.

- [ ] **Step 1: Write failing tests — append to tests/test_engine.py**

```python
from engine import tone_map, render
import numpy as np
from PIL import Image

def test_tone_map_all_zero_returns_zero():
    counts = np.zeros((10, 10))
    result = tone_map(counts, gamma=2.5, brightness=3.0)
    assert result.max() == 0.0

def test_tone_map_shape_preserved():
    counts = np.ones((20, 30)) * 100
    result = tone_map(counts, gamma=2.5, brightness=3.0)
    assert result.shape == (20, 30)

def test_tone_map_values_in_range():
    rng = np.random.default_rng(0)
    counts = rng.integers(0, 1000, size=(50, 50)).astype(float)
    result = tone_map(counts, gamma=2.5, brightness=3.0)
    assert result.min() >= 0.0
    assert result.max() <= 1.0

def test_render_returns_pil_image():
    config = {
        "width": 64,
        "height": 64,
        "quality": 5,
        "supersample": 1,
        "symmetry": 4,
        "num_transforms": 3,
        "variations": ["linear", "swirl"],
        "seed": 1,
        "zoom": 1.0,
        "rotation": 0.0,
        "center": [0.0, 0.0],
        "palette": "fire",
        "background": [0, 0, 0],
        "gamma": 2.5,
        "brightness": 3.0,
        "vibrancy": 1.0,
    }
    img = render(config)
    assert isinstance(img, Image.Image)
    assert img.size == (64, 64)
    assert img.mode == "RGB"
```

- [ ] **Step 2: Run tests — expect new tests to fail**

```bash
pytest tests/test_engine.py::test_tone_map_all_zero_returns_zero -v
```

Expected: `ImportError: cannot import name 'tone_map'`

- [ ] **Step 3: Implement tone_map() — append to engine.py**

```python
def tone_map(
    counts: np.ndarray,
    gamma: float,
    brightness: float,
) -> np.ndarray:
    max_count = counts.max()
    if max_count == 0:
        return np.zeros_like(counts, dtype=np.float64)
    log_counts = np.log1p(counts)
    log_max = math.log1p(max_count)
    alpha = log_counts / log_max
    return np.clip(alpha ** (1.0 / gamma) * brightness, 0.0, 1.0)
```

- [ ] **Step 4: Implement render() — add imports at top of engine.py, then append render()**

Add these two imports at the very top of `engine.py` (after the existing imports):

```python
from palettes import get_palette, sample_palette
from PIL import Image
```

Then append `render()` at the bottom of `engine.py`:

```python
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

    vibrancy = config.get("vibrancy", 1.0)
    bg = config.get("background", [0, 0, 0])

    # Build RGB array
    pixels = np.zeros((h, w, 3), dtype=np.uint8)
    for py in range(h):
        for px in range(w):
            lum = luminance[py, px]
            if lum < 1e-6:
                pixels[py, px] = bg
            else:
                coord = float(color_coords[py, px])
                r, g, b = sample_palette(stops, coord)
                # vibrancy: blend between luminance-scaled and flat
                flat_r = int(r * lum)
                flat_g = int(g * lum)
                flat_b = int(b * lum)
                vib_r = int(r * (lum ** (1.0 / max(vibrancy, 0.01))))
                vib_g = int(g * (lum ** (1.0 / max(vibrancy, 0.01))))
                vib_b = int(b * (lum ** (1.0 / max(vibrancy, 0.01))))
                pixels[py, px] = [
                    int(flat_r * (1 - vibrancy) + vib_r * vibrancy),
                    int(flat_g * (1 - vibrancy) + vib_g * vibrancy),
                    int(flat_b * (1 - vibrancy) + vib_b * vibrancy),
                ]

    img = Image.fromarray(pixels, "RGB")
    if ss > 1:
        img = img.resize((config["width"], config["height"]), Image.LANCZOS)
    return img
```

- [ ] **Step 5: Run all engine tests**

```bash
pytest tests/test_engine.py -v
```

Expected: All 11 tests PASSED. Note: `test_render_returns_pil_image` may take ~5s at quality=5.

- [ ] **Step 6: Commit**

```bash
git add engine.py tests/test_engine.py
git commit -m "feat: tone mapping and top-level render() function"
```

---

## Task 6: Optimize render() with NumPy Vectorization

**Files:**
- Modify: `engine.py` — vectorize the pixel-building loop

The per-pixel Python loop in `render()` is too slow for real use. Replace it with NumPy array ops before wiring up the GUI.

- [ ] **Step 1: Run existing tests to confirm baseline**

```bash
pytest tests/test_engine.py -v
```

Expected: All 11 pass.

- [ ] **Step 2: Replace pixel loop in render() with vectorized version**

Replace the `# Build RGB array` section in `render()` with:

```python
    # Vectorized pixel building
    flat_coords = color_coords.ravel()
    palette_rgb = np.array([sample_palette(stops, float(t)) for t in flat_coords],
                           dtype=np.float64).reshape(h, w, 3)

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
```

- [ ] **Step 3: Run tests to confirm nothing broke**

```bash
pytest tests/test_engine.py -v
```

Expected: All 11 tests PASSED.

- [ ] **Step 4: Commit**

```bash
git add engine.py
git commit -m "perf: vectorize render() pixel loop with NumPy"
```

---

## Task 7: Dear PyGui — Window, Layout & Static Controls

**Files:**
- Create: `fractal.py`

Build the GUI shell: two-column layout with all control groups on the left and an image display area on the right. No rendering yet — just the UI structure with placeholder values.

- [ ] **Step 1: Implement fractal.py GUI skeleton**

```python
import dearpygui.dearpygui as dpg

# ── Default config ────────────────────────────────────────────────────────────
CONFIG = {
    "width": 1080,
    "height": 1920,
    "output": "fractal.png",
    "quality": 50,
    "supersample": 2,
    "symmetry": 6,
    "num_transforms": 4,
    "variations": ["swirl", "sinusoidal"],
    "seed": 42,
    "zoom": 1.0,
    "rotation": 0.0,
    "center": [0.0, 0.0],
    "palette": "fire",
    "background": [0, 0, 0],
    "gamma": 2.5,
    "brightness": 3.0,
    "vibrancy": 1.0,
}

PALETTES = ["fire", "ice", "neon", "rainbow", "gold", "violet"]
VARIATION_NAMES = ["linear", "sinusoidal", "spherical", "swirl",
                   "horseshoe", "polar", "curl", "pdj"]
ASPECT_PRESETS = {
    "9:16": (1080, 1920),
    "1:1":  (1080, 1080),
    "16:9": (1920, 1080),
    "4:3":  (1440, 1080),
    "3:2":  (1620, 1080),
}

PANEL_W = 340


def build_gui():
    dpg.create_context()
    dpg.create_viewport(title="Flame Fractal Generator", width=1400, height=900)
    dpg.setup_dearpygui()

    with dpg.window(tag="main_win", no_title_bar=True, no_move=True,
                    no_resize=True, no_scrollbar=True):

        with dpg.group(horizontal=True):

            # ── LEFT PANEL ────────────────────────────────────────────
            with dpg.child_window(tag="left_panel", width=PANEL_W,
                                  border=False, auto_resize_y=True):
                _build_output_group()
                _build_quality_group()
                _build_shape_group()
                _build_camera_group()
                _build_color_group()
                _build_tone_group()
                _build_buttons()

            # ── RIGHT PANEL ───────────────────────────────────────────
            with dpg.child_window(tag="right_panel", border=False,
                                  auto_resize_x=True, auto_resize_y=True):
                dpg.add_text("Press [Preview] to render", tag="status_text")
                dpg.add_image(texture_tag="preview_tex", tag="preview_img",
                              show=False)

    dpg.set_primary_window("main_win", True)


def _build_output_group():
    with dpg.collapsing_header(label="Output", default_open=True):
        dpg.add_input_int(label="Width", tag="cfg_width",
                          default_value=CONFIG["width"], min_value=64, max_value=8192, width=120)
        dpg.add_input_int(label="Height", tag="cfg_height",
                          default_value=CONFIG["height"], min_value=64, max_value=8192, width=120)
        dpg.add_input_text(label="Output file", tag="cfg_output",
                           default_value=CONFIG["output"], width=180)
        dpg.add_text("Aspect presets:")
        with dpg.group(horizontal=True):
            for label in ASPECT_PRESETS:
                dpg.add_button(label=label,
                               callback=lambda s, a, u=label: _apply_aspect(u),
                               width=55)


def _apply_aspect(label: str):
    w, h = ASPECT_PRESETS[label]
    dpg.set_value("cfg_width", w)
    dpg.set_value("cfg_height", h)


def _build_quality_group():
    with dpg.collapsing_header(label="Quality", default_open=True):
        dpg.add_slider_int(label="Samples/px", tag="cfg_quality",
                           default_value=CONFIG["quality"],
                           min_value=5, max_value=500, width=200)
        dpg.add_checkbox(label="Supersample (2x)", tag="cfg_supersample",
                         default_value=CONFIG["supersample"] == 2)


def _build_shape_group():
    with dpg.collapsing_header(label="Shape", default_open=True):
        dpg.add_slider_int(label="Symmetry", tag="cfg_symmetry",
                           default_value=CONFIG["symmetry"],
                           min_value=1, max_value=12, width=200)
        dpg.add_slider_int(label="Transforms", tag="cfg_num_transforms",
                           default_value=CONFIG["num_transforms"],
                           min_value=2, max_value=8, width=200)
        dpg.add_input_int(label="Seed", tag="cfg_seed",
                          default_value=CONFIG["seed"], width=120)
        dpg.add_text("Variations:")
        for name in VARIATION_NAMES:
            dpg.add_checkbox(label=name, tag=f"var_{name}",
                             default_value=name in CONFIG["variations"])


def _build_camera_group():
    with dpg.collapsing_header(label="Camera", default_open=False):
        dpg.add_slider_float(label="Zoom", tag="cfg_zoom",
                             default_value=CONFIG["zoom"],
                             min_value=0.1, max_value=10.0, width=200)
        dpg.add_slider_float(label="Rotation", tag="cfg_rotation",
                             default_value=CONFIG["rotation"],
                             min_value=0.0, max_value=360.0, width=200)
        dpg.add_slider_float(label="Center X", tag="cfg_center_x",
                             default_value=CONFIG["center"][0],
                             min_value=-2.0, max_value=2.0, width=200)
        dpg.add_slider_float(label="Center Y", tag="cfg_center_y",
                             default_value=CONFIG["center"][1],
                             min_value=-2.0, max_value=2.0, width=200)


def _build_color_group():
    with dpg.collapsing_header(label="Color", default_open=True):
        dpg.add_combo(label="Palette", tag="cfg_palette",
                      items=PALETTES, default_value=CONFIG["palette"], width=140)
        dpg.add_text("Background RGB:")
        with dpg.group(horizontal=True):
            for i, ch in enumerate(["R", "G", "B"]):
                dpg.add_input_int(label=ch, tag=f"cfg_bg_{ch.lower()}",
                                  default_value=CONFIG["background"][i],
                                  min_value=0, max_value=255, width=70)


def _build_tone_group():
    with dpg.collapsing_header(label="Tone Mapping", default_open=True):
        dpg.add_slider_float(label="Gamma", tag="cfg_gamma",
                             default_value=CONFIG["gamma"],
                             min_value=0.5, max_value=5.0, width=200)
        dpg.add_slider_float(label="Brightness", tag="cfg_brightness",
                             default_value=CONFIG["brightness"],
                             min_value=0.1, max_value=10.0, width=200)
        dpg.add_slider_float(label="Vibrancy", tag="cfg_vibrancy",
                             default_value=CONFIG["vibrancy"],
                             min_value=0.0, max_value=1.0, width=200)


def _build_buttons():
    dpg.add_spacer(height=8)
    with dpg.group(horizontal=True):
        dpg.add_button(label="Preview", tag="btn_preview",
                       callback=_on_preview, width=140, height=36)
        dpg.add_button(label="Render & Save", tag="btn_render",
                       callback=_on_render, width=150, height=36)


def _on_preview():
    pass  # wired in Task 8


def _on_render():
    pass  # wired in Task 8


if __name__ == "__main__":
    build_gui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()
```

- [ ] **Step 2: Verify GUI launches**

```bash
python fractal.py
```

Expected: Window opens with two-column layout. All control groups visible. Buttons do nothing yet.

- [ ] **Step 3: Commit**

```bash
git add fractal.py
git commit -m "feat: Dear PyGui shell with all control groups"
```

---

## Task 8: Wire Rendering — Preview & Save

**Files:**
- Modify: `fractal.py` — implement config collection, threading, image display, preview/save

- [ ] **Step 1: Add collect_config(), display_image(), and threading to fractal.py**

Add these functions before the `if __name__ == "__main__":` block:

```python
import threading
import time
import numpy as np
from engine import render as flame_render
from PIL import Image


def collect_config() -> dict:
    active_vars = [name for name in VARIATION_NAMES
                   if dpg.get_value(f"var_{name}")]
    if not active_vars:
        active_vars = ["linear"]
    return {
        "width":          dpg.get_value("cfg_width"),
        "height":         dpg.get_value("cfg_height"),
        "output":         dpg.get_value("cfg_output"),
        "quality":        dpg.get_value("cfg_quality"),
        "supersample":    2 if dpg.get_value("cfg_supersample") else 1,
        "symmetry":       dpg.get_value("cfg_symmetry"),
        "num_transforms": dpg.get_value("cfg_num_transforms"),
        "variations":     active_vars,
        "seed":           dpg.get_value("cfg_seed"),
        "zoom":           dpg.get_value("cfg_zoom"),
        "rotation":       dpg.get_value("cfg_rotation"),
        "center":         [dpg.get_value("cfg_center_x"),
                           dpg.get_value("cfg_center_y")],
        "palette":        dpg.get_value("cfg_palette"),
        "background":     [dpg.get_value("cfg_bg_r"),
                           dpg.get_value("cfg_bg_g"),
                           dpg.get_value("cfg_bg_b")],
        "gamma":          dpg.get_value("cfg_gamma"),
        "brightness":     dpg.get_value("cfg_brightness"),
        "vibrancy":       dpg.get_value("cfg_vibrancy"),
    }


def _set_buttons_enabled(enabled: bool):
    dpg.configure_item("btn_preview", enabled=enabled)
    dpg.configure_item("btn_render", enabled=enabled)


def _show_image(img: Image.Image):
    """Convert PIL Image to DPG texture and display it."""
    img_rgba = img.convert("RGBA")
    w, h = img_rgba.size
    data = np.array(img_rgba, dtype=np.float32) / 255.0
    flat = data.ravel().tolist()

    if dpg.does_item_exist("preview_tex"):
        dpg.delete_item("preview_tex")

    with dpg.texture_registry():
        dpg.add_static_texture(width=w, height=h, default_value=flat,
                               tag="preview_tex")

    # Scale image to fit right panel (max 1000px wide)
    max_w = dpg.get_item_width("right_panel") - 20
    scale = min(1.0, max_w / w)
    disp_w, disp_h = int(w * scale), int(h * scale)

    if dpg.does_item_exist("preview_img"):
        dpg.configure_item("preview_img", texture_tag="preview_tex",
                           width=disp_w, height=disp_h, show=True)
    else:
        dpg.add_image("preview_tex", tag="preview_img",
                      width=disp_w, height=disp_h, parent="right_panel")


def _run_render(config: dict, save: bool):
    _set_buttons_enabled(False)
    dpg.set_value("status_text", "Rendering…")
    t0 = time.perf_counter()
    try:
        img = flame_render(config)
        elapsed = time.perf_counter() - t0
        _show_image(img)
        if save:
            path = config["output"]
            img.save(path)
            dpg.set_value("status_text",
                          f"Saved to {path}  ({elapsed:.1f}s)")
        else:
            dpg.set_value("status_text",
                          f"Preview done  ({elapsed:.1f}s)  "
                          f"{config['width']}×{config['height']}")
    except Exception as e:
        dpg.set_value("status_text", f"Error: {e}")
    finally:
        _set_buttons_enabled(True)
```

- [ ] **Step 2: Replace _on_preview and _on_render stubs**

```python
def _on_preview():
    cfg = collect_config()
    cfg["quality"] = 15
    cfg["supersample"] = 1
    threading.Thread(target=_run_render, args=(cfg, False), daemon=True).start()


def _on_render():
    cfg = collect_config()
    threading.Thread(target=_run_render, args=(cfg, True), daemon=True).start()
```

- [ ] **Step 3: Run the app and test preview**

```bash
python fractal.py
```

Expected:
- Click **Preview** → "Rendering…" appears, image populates right panel in ~2–5s, status shows time
- UI stays responsive while rendering
- Click **Render & Save** → renders at full quality, saves PNG, status shows save path
- Aspect ratio preset buttons update width/height inputs correctly

- [ ] **Step 4: Commit**

```bash
git add fractal.py
git commit -m "feat: wire preview and render with threading and image display"
```

---

## Task 9: Run Full Test Suite & Final Verification

**Files:**
- No changes — verification only

- [ ] **Step 1: Run all tests**

```bash
pytest tests/ -v
```

Expected: All tests PASSED (no failures, no errors).

- [ ] **Step 2: Test a full high-quality render**

Change `CONFIG["quality"]` temporarily to `200` in `fractal.py` and render a 1080×1920 image. Verify the output PNG looks like the reference images — glowing core, wispy tendrils, dark background.

- [ ] **Step 3: Test each palette**

In the GUI, cycle through fire / ice / neon / rainbow / gold / violet and preview each. Verify distinct color differences.

- [ ] **Step 4: Test symmetry extremes**

Set symmetry to 1 (asymmetric), then 12 (dense mandala). Confirm both render without errors.

- [ ] **Step 5: Final commit**

```bash
git add .
git commit -m "feat: flame fractal generator complete"
```
