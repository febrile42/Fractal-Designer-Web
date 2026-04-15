# Flame Fractal Generator — Design Spec
**Date:** 2026-04-15
**Status:** Approved

---

## Overview

A Python script that generates glassy, luminous flame fractal images using Scott Draves' flame algorithm (as used in Apophysis and Electric Sheep). The output matches the aesthetic of the reference images: dark backgrounds, radiant glowing cores, wispy translucent tendrils, and additive color blending. A Dear PyGui side-panel GUI allows real-time parameter adjustment with preview rendering.

---

## Files

```
Fractal/
├── fractal.py          # Main entry point — launches GUI
├── engine.py           # Pure flame algorithm (no GUI dependency)
├── variations.py       # All variation functions (swirl, spherical, etc.)
├── palettes.py         # Built-in color palettes
└── requirements.txt    # dearpygui, pillow, numpy
```

---

## Algorithm (engine.py)

### 1. IFS Transforms
Define N affine transforms, each with:
- Coefficients `a, b, c, d, e, f` (randomly seeded, then bounded to keep orbits stable)
- A probability weight (how often this transform is selected)
- A color coordinate `[0, 1]` (maps to the palette)
- One or more variation functions with weights

### 2. Variation Functions (variations.py)
Each variation warps the point `(x, y)` mathematically. Supported variations:
- `linear` — identity passthrough
- `sinusoidal` — `(sin(x), sin(y))`
- `spherical` — `1/r² * (x, y)`
- `swirl` — rotate by `r²`
- `horseshoe` — `1/r * ((x-y)(x+y), 2xy)`
- `polar` — `(θ/π, r-1)`
- `curl` — complex curl distortion
- `blob` — radial blob deformation with high/low/waves params
- `pdj` — Peter de Jong attractors

### 3. Chaos Game (iterate)
```
pick random starting point (x, y)
repeat quality * width * height times:
    pick transform T weighted by probability
    apply affine: (x, y) = T.affine(x, y)
    apply variations: (x, y) = T.vary(x, y)
    apply symmetry: plot point + N-1 rotations
    map (x, y) → pixel (px, py) via camera
    histogram[px, py].count += 1
    histogram[px, py].color = lerp(histogram[px,py].color, T.color, 0.5)
```
First 20 iterations are discarded (burn-in to reach attractor).

### 4. Symmetry
After each transform application, the point is rotated `360/symmetry` degrees repeatedly and all rotated copies are plotted. This creates the mandala/snowflake structures.

### 5. Log-Density Tone Mapping
```
alpha[px, py] = log(1 + count[px, py]) / log(1 + max_count)
color[px, py] = palette[color_coord] * alpha^(1/gamma)
```
Produces the smooth glassy luminance gradient (bright core, dim wisps) instead of flat hard edges.

### 6. Additive Color Blending
Colors accumulate additively in the histogram. Final pixel values are clamped to `[0, 255]`. This gives the glowing oversaturated core and delicate translucent edges.

### 7. Supersampling
Render at `supersample * (width, height)`, then downsample with box filter for antialiasing.

---

## Parameters

```python
CONFIG = {
    # Output
    "width": 1080,
    "height": 1920,
    "output": "fractal.png",

    # Quality
    "quality": 50,        # samples per pixel (10=fast preview, 500=high quality)
    "supersample": 2,     # render multiplier for antialiasing (1 or 2)

    # Shape
    "symmetry": 6,        # rotational symmetry order (1–12)
    "num_transforms": 4,  # number of IFS transforms (2–8)
    "variations": ["swirl", "sinusoidal", "spherical"],
    "seed": 42,           # random seed for transform coefficients

    # Camera
    "zoom": 1.0,
    "rotation": 0.0,      # degrees
    "center": [0.0, 0.0],

    # Color
    "palette": "fire",    # "fire", "ice", "neon", "rainbow", or list of hex colors
    "background": [0, 0, 0],

    # Tone mapping
    "gamma": 2.5,
    "brightness": 3.0,
    "vibrancy": 1.0,      # 0 = flat color, 1 = luminance-modulated color
}
```

---

## Built-in Palettes (palettes.py)

| Name      | Description                                      |
|-----------|--------------------------------------------------|
| `fire`    | Black → deep orange → yellow → white             |
| `ice`     | Black → dark teal → cyan → white                 |
| `neon`    | Black → purple → magenta → hot pink → white      |
| `rainbow` | Full HSV cycle, high saturation                  |
| `custom`  | User-defined list of hex color stops             |

Palettes are stored as lists of RGB stops; the algorithm interpolates smoothly between them.

---

## GUI (fractal.py — Dear PyGui)

### Layout
Two-column window:
- **Left panel (320px):** All parameter controls, Preview button, Render & Save button
- **Right panel (fills remaining width):** Preview image display, render stats (time, resolution)

### Control Groups
1. **Output** — width/height inputs, aspect ratio preset buttons (9:16, 1:1, 16:9, 4:3, 3:2)
2. **Quality** — quality slider (10–500), supersample checkbox
3. **Shape** — symmetry slider (1–12), num_transforms slider (2–8), seed input, variation checkboxes
4. **Camera** — zoom slider (0.1–10), rotation slider (0–360), center X/Y sliders (-2 to 2)
5. **Color** — palette dropdown, custom hex color inputs (up to 5 stops), background color RGB inputs
6. **Tone Mapping** — gamma slider (0.5–5.0), brightness slider (0.1–10.0), vibrancy slider (0–1)

### Buttons
- **[Preview]** — renders at `quality=15, supersample=1` for speed (~1–2s), displays in right panel
- **[Render & Save]** — renders at current quality settings, saves PNG to `output` path, shows save confirmation

### Threading
Rendering runs in a background thread (`threading.Thread`). While rendering:
- Buttons are disabled
- A progress label shows "Rendering…"
- UI remains interactive
On completion, the image updates in the right panel and buttons re-enable.

---

## Dependencies

```
numpy>=1.24
pillow>=10.0
dearpygui>=1.11
```

---

## Success Criteria

- Script runs with `python fractal.py` and opens the GUI
- Preview renders in under 3 seconds at default quality
- All parameters update the fractal correctly when re-rendered
- Output PNG matches the glassy/flame aesthetic of the reference images
- Aspect ratio presets correctly resize width/height fields
- Render & Save produces a valid PNG at the specified resolution
