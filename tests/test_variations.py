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
    # polar(1, 0) with atan2(x,y): theta=atan2(1,0)=pi/2; r=1 → (0.5, 0.0)
    x, y = polar(1.0, 0.0)
    assert abs(x - 0.5) < 1e-9
    assert abs(y - 0.0) < 1e-9

def test_polar_on_y_axis():
    # polar(0, 1) with atan2(x,y): theta=atan2(0,1)=0; r=1 → (0.0, 0.0)
    x, y = polar(0.0, 1.0)
    assert abs(x - 0.0) < 1e-9
    assert abs(y - 0.0) < 1e-9

def test_curl_origin():
    x, y = curl(0.0, 0.0, c1=0.5, c2=0.1)
    assert x == 0.0 and y == 0.0

def test_pdj_returns_tuple():
    x, y = pdj(0.5, 0.3, a=1.5, b=1.8, c=1.2, d=2.0)
    assert isinstance(x, float)
    assert isinstance(y, float)
