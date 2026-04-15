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
