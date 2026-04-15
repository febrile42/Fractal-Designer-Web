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
