"""CUDA kernels and GPU helpers for flame-fractal rendering."""

import math
import numpy as np
import cupy as cp
from PIL import Image

# Variation name → CUDA kernel switch-case ID
VARIATION_IDS = {
    "linear": 0, "sinusoidal": 1, "spherical": 2, "swirl": 3,
    "horseshoe": 4, "polar": 5, "curl": 6, "pdj": 7,
}

# ── CUDA kernel ───────────────────────────────────────────────────────────────
_CUDA_SRC = r"""
extern "C" {

__device__ __forceinline__
unsigned long long _xnext(unsigned long long *s0, unsigned long long *s1) {
    unsigned long long t = *s0;
    *s0 = *s1;
    t ^= t << 23;
    *s1 = t ^ *s0 ^ (t >> 17) ^ (*s0 >> 26);
    return *s1 + *s0;
}

__device__ __forceinline__
float _rf(unsigned long long *s0, unsigned long long *s1) {
    return (float)(_xnext(s0, s1) >> 11) * (1.0f / 9007199254740992.0f);
}

__global__ void chaos_game(
    const float * __restrict__ affine,
    const float * __restrict__ cum_weights,
    const float * __restrict__ t_colors,
    const float * __restrict__ var_weights,
    const int   * __restrict__ var_ids,
    const int    num_transforms,
    const int    num_vars,
    float       *counts,
    float       *color_sum,
    const int    W,
    const int    H,
    const int    iters,
    const int    burn,
    const int    sym,
    const float  scale,
    const float  cosr,
    const float  sinr,
    const float  cx,
    const float  cy,
    const float * __restrict__ sc,
    const float * __restrict__ ss,
    const unsigned long long seed
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    /* per-thread xorshift128+ RNG */
    unsigned long long s0 = seed ^ ((unsigned long long)(tid + 1) * 6364136223846793005ULL);
    unsigned long long s1 = s0 * 2862933555777941757ULL + 7ULL;
    for (int w = 0; w < 8; w++) _xnext(&s0, &s1);

    float x = _rf(&s0, &s1) * 2.0f - 1.0f;
    float y = _rf(&s0, &s1) * 2.0f - 1.0f;
    float cc = 0.0f;
    const int total = burn + iters;

    for (int it = 0; it < total; it++) {
        /* pick transform via cumulative weights */
        float r = _rf(&s0, &s1);
        int ti = num_transforms - 1;
        for (int j = 0; j < num_transforms; j++) {
            if (r <= cum_weights[j]) { ti = j; break; }
        }

        /* affine step */
        const int ai = ti * 6;
        float nx = affine[ai] * x + affine[ai + 1] * y + affine[ai + 2];
        float ny = affine[ai + 3] * x + affine[ai + 4] * y + affine[ai + 5];

        /* variation blend */
        float ox = 0.0f, oy = 0.0f;
        for (int v = 0; v < num_vars; v++) {
            float vx, vy;
            switch (var_ids[v]) {
            case 0: /* linear */
                vx = nx; vy = ny; break;
            case 1: /* sinusoidal */
                vx = sinf(nx); vy = sinf(ny); break;
            case 2: { /* spherical */
                float r2 = nx * nx + ny * ny;
                if (r2 < 1e-10f) { vx = 0; vy = 0; }
                else { float iv = 1.0f / r2; vx = nx * iv; vy = ny * iv; }
                break; }
            case 3: { /* swirl */
                float r2 = nx * nx + ny * ny;
                float sr = sinf(r2), cr = cosf(r2);
                vx = nx * sr - ny * cr; vy = nx * cr + ny * sr; break; }
            case 4: { /* horseshoe */
                float rr = sqrtf(nx * nx + ny * ny);
                if (rr < 1e-10f) { vx = 0; vy = 0; }
                else { float iv = 1.0f / rr;
                       vx = iv * (nx - ny) * (nx + ny);
                       vy = iv * 2.0f * nx * ny; }
                break; }
            case 5: { /* polar */
                float theta = atan2f(nx, ny);
                vx = theta * 0.31830988618f;
                vy = sqrtf(nx * nx + ny * ny) - 1.0f;
                break; }
            case 6: { /* curl */
                float c1 = 0.5f, c2 = 0.1f;
                float t1 = 1.0f + c1 * nx + c2 * (nx * nx - ny * ny);
                float t2 = c1 * ny + 2.0f * c2 * nx * ny;
                float r2 = t1 * t1 + t2 * t2;
                if (r2 < 1e-10f) { vx = 0; vy = 0; }
                else { float iv = 1.0f / r2;
                       vx = (nx * t1 + ny * t2) * iv;
                       vy = (ny * t1 - nx * t2) * iv; }
                break; }
            case 7: { /* pdj */
                float pa = 1.5f, pb = 1.8f, pc = 1.2f, pd = 2.0f;
                vx = sinf(pa * ny) - cosf(pb * nx);
                vy = sinf(pc * nx) - cosf(pd * ny);
                break; }
            default:
                vx = nx; vy = ny;
            }
            float wt = var_weights[ti * num_vars + v];
            ox += wt * vx;
            oy += wt * vy;
        }

        x = ox; y = oy;
        cc = (cc + t_colors[ti]) * 0.5f;

        if (!isfinite(x) || !isfinite(y)) {
            x = _rf(&s0, &s1) * 2.0f - 1.0f;
            y = _rf(&s0, &s1) * 2.0f - 1.0f;
            cc = 0.0f;
            continue;
        }

        if (it < burn) continue;

        /* symmetry copies + camera transform + histogram plot */
        for (int s = 0; s < sym; s++) {
            float rx = x * sc[s] - y * ss[s];
            float ry = x * ss[s] + y * sc[s];
            float wx = (rx - cx) * cosr - (ry - cy) * sinr;
            float wy = (rx - cx) * sinr + (ry - cy) * cosr;
            int px = (int)(wx * scale + W * 0.5f);
            int py = (int)(wy * scale + H * 0.5f);
            if (px >= 0 && px < W && py >= 0 && py < H) {
                atomicAdd(&counts[py * W + px], 1.0f);
                atomicAdd(&color_sum[py * W + px], cc);
            }
        }
    }
}

}  /* extern "C" */
"""

# ── Constants ─────────────────────────────────────────────────────────────────
_BLOCK = 256
_MAX_THREADS = 1 << 17   # 131072
_MIN_IPT = 64             # minimum iterations per thread for efficiency
_BURN = 20

_kernel = None


def _get_kernel():
    global _kernel
    if _kernel is None:
        _kernel = cp.RawKernel(_CUDA_SRC, "chaos_game")
    return _kernel


def prepare_transforms(transforms, var_names):
    """Pack Transform list into flat GPU arrays for the kernel."""
    nt = len(transforms)
    nv = len(var_names)

    affine = np.empty(nt * 6, np.float32)
    cum = np.empty(nt, np.float32)
    tcol = np.empty(nt, np.float32)
    vw = np.empty(nt * nv, np.float32)
    vid = np.array([VARIATION_IDS[n] for n in var_names], np.int32)

    run = 0.0
    for i, t in enumerate(transforms):
        affine[i * 6: i * 6 + 6] = (t.a, t.b, t.c, t.d, t.e, t.f)
        run += t.weight
        cum[i] = run
        tcol[i] = t.color
        tot = t._var_total
        for j, vn in enumerate(var_names):
            vw[i * nv + j] = t.variations.get(vn, 0.0) / tot

    return (cp.asarray(affine), cp.asarray(cum), cp.asarray(tcol),
            cp.asarray(vw), cp.asarray(vid), nv)


def launch(tdata, width, height, iterations, symmetry, zoom, rotation, center,
           counts_gpu, color_sum_gpu, seed_offset=0, base_seed=42):
    """Launch the chaos game CUDA kernel, accumulating into GPU arrays."""
    if iterations <= 0:
        return

    affine, cum, tcol, vw, vid, nv = tdata
    nt = len(cum)

    scale = np.float32(min(width, height) * 0.35 * zoom)
    rr = math.radians(rotation)
    cosr = np.float32(math.cos(rr))
    sinr = np.float32(math.sin(rr))
    cx_f = np.float32(center[0])
    cy_f = np.float32(center[1])

    sa = 2.0 * math.pi / symmetry
    sc = cp.array([math.cos(s * sa) for s in range(symmetry)], np.float32)
    ss = cp.array([math.sin(s * sa) for s in range(symmetry)], np.float32)

    n_thr = min(_MAX_THREADS, max(_BLOCK, iterations // _MIN_IPT))
    n_thr = ((n_thr + _BLOCK - 1) // _BLOCK) * _BLOCK
    ipt = max(1, iterations // n_thr)
    n_blk = n_thr // _BLOCK
    seed = np.uint64(base_seed + seed_offset * 1000003)

    _get_kernel()(
        (n_blk,), (_BLOCK,),
        (affine, cum, tcol, vw, vid,
         np.int32(nt), np.int32(nv),
         counts_gpu, color_sum_gpu,
         np.int32(width), np.int32(height),
         np.int32(ipt), np.int32(_BURN),
         np.int32(symmetry),
         scale, cosr, sinr, cx_f, cy_f,
         sc, ss, seed)
    )


def gpu_tone_map(counts_gpu, gamma, brightness):
    """Tone-map hit counts on GPU. Returns cupy float32 array in [0, 1]."""
    c = cp.clip(counts_gpu, 0, None)
    mx = float(c.max())
    if mx == 0:
        return cp.zeros_like(c, dtype=cp.float32)
    log_c = cp.log1p(c)
    log_mx = math.log1p(mx)
    alpha = log_c / np.float32(log_mx)
    return cp.clip(alpha ** np.float32(1.0 / gamma) * np.float32(brightness),
                   0.0, 1.0)


def gpu_make_image(counts_gpu, color_sum_gpu, config, ss):
    """Tone-map, colorize, and produce a PIL Image entirely on GPU.

    counts_gpu / color_sum_gpu are the raw accumulated arrays (float32).
    color_sum is divided by counts to get average color coordinate.
    """
    from palettes import get_palette

    # Color average (safe division)
    safe = cp.where(counts_gpu > 0, counts_gpu, cp.float32(1.0))
    color_avg = cp.where(counts_gpu > 0, color_sum_gpu / safe, cp.float32(0.0))

    lum = gpu_tone_map(counts_gpu, config["gamma"], config["brightness"])

    palette_name = config.get("palette", "fire")
    stops = palette_name if isinstance(palette_name, list) else get_palette(palette_name)
    stops_arr = cp.array(stops, dtype=cp.float32)
    n = len(stops) - 1

    t_cl = cp.clip(color_avg, 0.0, 1.0)
    scaled = t_cl * n
    lo = cp.floor(scaled).astype(cp.int32)
    hi = cp.minimum(lo + 1, n)
    frac = (scaled - lo.astype(cp.float32))[..., cp.newaxis]
    pal_rgb = stops_arr[lo] + (stops_arr[hi] - stops_arr[lo]) * frac

    vibrancy = float(np.clip(config.get("vibrancy", 1.0), 0.0, 1.0))
    bg = config.get("background", [0, 0, 0])

    lum3 = lum[:, :, cp.newaxis]
    flat = pal_rgb * lum3
    safe_v = max(vibrancy, 0.01)
    vib = pal_rgb * (lum3 ** np.float32(1.0 / safe_v))
    blended = flat * (1 - vibrancy) + vib * vibrancy
    blended = cp.clip(cp.round(blended), 0, 255).astype(cp.uint8)

    mask = lum < 1e-6
    bg_gpu = cp.array(bg, dtype=cp.uint8)
    blended[mask] = bg_gpu

    img = Image.fromarray(cp.asnumpy(blended), "RGB")
    if ss > 1:
        img = img.resize((config["width"], config["height"]), Image.LANCZOS)
    return img
