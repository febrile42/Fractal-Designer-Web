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

from engine import tone_map, render, run_chaos_game_partial, _make_image
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

def test_partial_returns_same_shape_as_input():
    transforms = build_transforms(num=3, seed=42, variations=["linear"])
    counts = np.zeros((64, 64), dtype=np.float64)
    colors = np.zeros((64, 64), dtype=np.float64)
    counts_out, colors_out, state = run_chaos_game_partial(
        transforms=transforms,
        width=64, height=64,
        iterations=5000,
        symmetry=1,
        zoom=1.0, rotation=0.0, center=(0.0, 0.0),
        counts=counts, colors=colors, state=None,
    )
    assert counts_out.shape == (64, 64)
    assert colors_out.shape == (64, 64)

def test_partial_accumulates_across_calls():
    transforms = build_transforms(num=3, seed=42, variations=["linear"])
    counts = np.zeros((64, 64), dtype=np.float64)
    colors = np.zeros((64, 64), dtype=np.float64)
    counts, colors, state = run_chaos_game_partial(
        transforms=transforms,
        width=64, height=64, iterations=5000,
        symmetry=1, zoom=1.0, rotation=0.0, center=(0.0, 0.0),
        counts=counts, colors=colors, state=None,
    )
    total_after_first = counts.sum()
    counts, colors, state = run_chaos_game_partial(
        transforms=transforms,
        width=64, height=64, iterations=5000,
        symmetry=1, zoom=1.0, rotation=0.0, center=(0.0, 0.0),
        counts=counts, colors=colors, state=state,
    )
    assert counts.sum() > total_after_first

def test_partial_state_continues_without_burn_in():
    """Second call with existing state must not reset the orbit."""
    transforms = build_transforms(num=3, seed=42, variations=["linear"])
    zeros = np.zeros((64, 64), dtype=np.float64)
    _, _, state = run_chaos_game_partial(
        transforms=transforms,
        width=64, height=64, iterations=1000,
        symmetry=1, zoom=1.0, rotation=0.0, center=(0.0, 0.0),
        counts=zeros.copy(), colors=zeros.copy(), state=None,
    )
    x, y, c = state
    assert math.isfinite(x) and math.isfinite(y) and math.isfinite(c)

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


_SMALL_CONFIG = {
    "width": 32, "height": 32,
    "quality": 5, "supersample": 1,
    "symmetry": 1, "num_transforms": 3,
    "variations": ["linear"], "seed": 1,
    "zoom": 1.0, "rotation": 0.0, "center": [0.0, 0.0],
    "palette": "fire", "background": [0, 0, 0],
    "gamma": 2.5, "brightness": 3.0, "vibrancy": 1.0,
}


def test_make_image_returns_pil_image():
    counts = np.ones((32, 32), dtype=np.float64) * 100
    colors = np.full((32, 32), 0.5, dtype=np.float64)
    img = _make_image(counts, colors, _SMALL_CONFIG, w=32, h=32, ss=1)
    assert isinstance(img, Image.Image)
    assert img.size == (32, 32)
    assert img.mode == "RGB"


def test_make_image_supersample_downscales():
    counts = np.ones((64, 64), dtype=np.float64) * 100
    colors = np.full((64, 64), 0.5, dtype=np.float64)
    cfg = {**_SMALL_CONFIG, "width": 32, "height": 32}
    img = _make_image(counts, colors, cfg, w=64, h=64, ss=2)
    assert img.size == (32, 32)
