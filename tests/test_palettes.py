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
