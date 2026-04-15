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
