"""Apple Silicon GPU backend using Metal via metalcompute.

Requires macOS 14+ and Apple GPU family 7+ (M1 or later).
Atomic float support (needed for histogram accumulation) landed in MSL 2.4 / Apple GPU family 7.
"""
from __future__ import annotations

import array
import math
import struct

import metalcompute as mc
import numpy as np
from PIL import Image

from engine_cpu import gpu_make_image, gpu_tone_map  # tone-map/colorize on CPU

_MAX_THREADS = 1 << 17   # 131072
_MIN_IPT = 64
_BURN = 20

VARIATION_IDS = {
    "linear": 0, "sinusoidal": 1, "spherical": 2, "swirl": 3,
    "horseshoe": 4, "polar": 5, "curl": 6, "pdj": 7,
}

_MSL = r"""
#include <metal_stdlib>
using namespace metal;

// xorshift128+ RNG — identical to the CUDA version
inline ulong xnext(thread ulong& s0, thread ulong& s1) {
    ulong t = s0;
    s0 = s1;
    t ^= t << 23;
    s1 = t ^ s0 ^ (t >> 17) ^ (s0 >> 26);
    return s1 + s0;
}
inline float rf(thread ulong& s0, thread ulong& s1) {
    return float(xnext(s0, s1) >> 11) * (1.0f / 9007199254740992.0f);
}

// atomic<float> unavailable in this MSL version; emulate via CAS on uint bits.
inline void atomic_add_f32(volatile device atomic_uint* atom, float val) {
    uint old = atomic_load_explicit(atom, memory_order_relaxed);
    uint next;
    do {
        next = as_type<uint>(as_type<float>(old) + val);
    } while (!atomic_compare_exchange_weak_explicit(
        atom, &old, next, memory_order_relaxed, memory_order_relaxed));
}

kernel void chaos_game(
    const device float*          affine       [[ buffer(0) ]],
    const device float*          cum_weights  [[ buffer(1) ]],
    const device float*          t_colors     [[ buffer(2) ]],
    const device float*          var_weights  [[ buffer(3) ]],
    const device int*            var_ids      [[ buffer(4) ]],
    const device uint*           iparams      [[ buffer(5) ]],  // [nt,nv,W,H,ipt,burn,sym,seed_lo,seed_hi]
    const device float*          fparams      [[ buffer(6) ]],  // [scale,cosr,sinr,cx,cy]
    const device float*          sym_cos      [[ buffer(7) ]],
    const device float*          sym_sin      [[ buffer(8) ]],
    volatile device atomic_uint* counts       [[ buffer(9)  ]],  // uint, stores hit counts
    volatile device atomic_uint* color_sum    [[ buffer(10) ]],  // uint, stores float bits
    uint tid [[ thread_position_in_grid ]]
) {
    const int nt   = (int)iparams[0];
    const int nv   = (int)iparams[1];
    const int W    = (int)iparams[2];
    const int H    = (int)iparams[3];
    const int ipt  = (int)iparams[4];
    const int burn = (int)iparams[5];
    const int sym  = (int)iparams[6];
    const ulong seed = (ulong)iparams[7] | ((ulong)iparams[8] << 32u);

    const float scale = fparams[0];
    const float cosr  = fparams[1];
    const float sinr  = fparams[2];
    const float cx    = fparams[3];
    const float cy    = fparams[4];

    ulong s0 = seed ^ (ulong(tid + 1u) * 6364136223846793005UL);
    ulong s1 = s0   * 2862933555777941757UL + 7UL;
    for (int w = 0; w < 8; w++) xnext(s0, s1);

    float x  = rf(s0, s1) * 2.0f - 1.0f;
    float y  = rf(s0, s1) * 2.0f - 1.0f;
    float cc = 0.0f;

    for (int it = 0; it < burn + ipt; it++) {
        // Pick transform via cumulative weights
        float r = rf(s0, s1);
        int ti = nt - 1;
        for (int j = 0; j < nt; j++) {
            if (r <= cum_weights[j]) { ti = j; break; }
        }

        // Affine step
        const int ai = ti * 6;
        float nx = affine[ai]*x + affine[ai+1]*y + affine[ai+2];
        float ny = affine[ai+3]*x + affine[ai+4]*y + affine[ai+5];

        // Variation blend
        float ox = 0.0f, oy = 0.0f;
        for (int v = 0; v < nv; v++) {
            float vx, vy;
            switch (var_ids[v]) {
            case 0: vx = nx; vy = ny; break;
            case 1: vx = sin(nx); vy = sin(ny); break;
            case 2: {
                float r2 = nx*nx + ny*ny;
                if (r2 < 1e-10f) { vx = 0; vy = 0; }
                else { float iv = 1.0f/r2; vx = nx*iv; vy = ny*iv; }
                break; }
            case 3: {
                float r2 = nx*nx + ny*ny;
                vx = nx*sin(r2) - ny*cos(r2);
                vy = nx*cos(r2) + ny*sin(r2);
                break; }
            case 4: {
                float rr = sqrt(nx*nx + ny*ny);
                if (rr < 1e-10f) { vx = 0; vy = 0; }
                else { float iv = 1.0f/rr;
                       vx = iv*(nx-ny)*(nx+ny); vy = iv*2.0f*nx*ny; }
                break; }
            case 5: {
                vx = atan2(nx, ny) * 0.31830988618f;
                vy = sqrt(nx*nx + ny*ny) - 1.0f;
                break; }
            case 6: {
                float t1 = 1.0f + 0.5f*nx + 0.1f*(nx*nx - ny*ny);
                float t2 = 0.5f*ny + 0.2f*nx*ny;
                float r2 = t1*t1 + t2*t2;
                if (r2 < 1e-10f) { vx = 0; vy = 0; }
                else { float iv = 1.0f/r2;
                       vx = (nx*t1+ny*t2)*iv; vy = (ny*t1-nx*t2)*iv; }
                break; }
            case 7: {
                vx = sin(1.5f*ny) - cos(1.8f*nx);
                vy = sin(1.2f*nx) - cos(2.0f*ny);
                break; }
            default: vx = nx; vy = ny;
            }
            ox += var_weights[ti * nv + v] * vx;
            oy += var_weights[ti * nv + v] * vy;
        }

        x = ox; y = oy;
        cc = (cc + t_colors[ti]) * 0.5f;

        if (!isfinite(x) || !isfinite(y)) {
            x = rf(s0, s1) * 2.0f - 1.0f;
            y = rf(s0, s1) * 2.0f - 1.0f;
            cc = 0.0f;
            continue;
        }
        if (it < burn) continue;

        for (int s = 0; s < sym; s++) {
            float rx = x * sym_cos[s] - y * sym_sin[s];
            float ry = x * sym_sin[s] + y * sym_cos[s];
            float wx = (rx - cx) * cosr - (ry - cy) * sinr;
            float wy = (rx - cx) * sinr + (ry - cy) * cosr;
            int px = int(wx * scale + W * 0.5f);
            int py = int(wy * scale + H * 0.5f);
            if (px >= 0 && px < W && py >= 0 && py < H) {
                atomic_fetch_add_explicit(&counts[py * W + px], 1u, memory_order_relaxed);
                atomic_add_f32(&color_sum[py * W + px], cc);
            }
        }
    }
}
"""

_dev: mc.Device | None = None
_fn = None


def _get_fn():
    global _dev, _fn
    if _fn is None:
        _dev = mc.Device()
        _fn = _dev.kernel(_MSL).function("chaos_game")
    return _dev, _fn


def prepare_transforms(transforms, var_names):
    """Pack transform list into flat numpy arrays (same interface as engine_gpu)."""
    nt = len(transforms)
    nv = len(var_names)

    affine = np.empty(nt * 6, np.float32)
    cum    = np.empty(nt, np.float32)
    tcol   = np.empty(nt, np.float32)
    vw     = np.empty(nt * nv, np.float32)
    vid    = np.array([VARIATION_IDS[n] for n in var_names], np.int32)

    run = 0.0
    for i, t in enumerate(transforms):
        affine[i * 6: i * 6 + 6] = (t.a, t.b, t.c, t.d, t.e, t.f)
        run += t.weight
        cum[i] = run
        tcol[i] = t.color
        tot = t._var_total
        for j, vn in enumerate(var_names):
            vw[i * nv + j] = t.variations.get(vn, 0.0) / tot

    return (affine, cum, tcol, vw, vid, nv)


def launch(tdata, width, height, iterations, symmetry, zoom, rotation, center,
           counts, color_sum, seed_offset=0, base_seed=42):
    """Run the chaos game on Apple Silicon GPU via Metal, accumulating into numpy arrays."""
    if iterations <= 0:
        return

    dev, fn = _get_fn()
    affine, cum, tcol, vw, vid, nv = tdata
    nt = len(cum)

    scale = np.float32(min(width, height) * 0.35 * zoom)
    rr    = math.radians(rotation)
    cosr  = np.float32(math.cos(rr))
    sinr  = np.float32(math.sin(rr))
    cx    = np.float32(center[0])
    cy    = np.float32(center[1])

    sa      = 2.0 * math.pi / symmetry
    sym_cos = np.array([math.cos(s * sa) for s in range(symmetry)], np.float32)
    sym_sin = np.array([math.sin(s * sa) for s in range(symmetry)], np.float32)

    n_thr = min(_MAX_THREADS, max(64, iterations // _MIN_IPT))
    n_thr = ((n_thr + 63) // 64) * 64   # round up to multiple of 64 (Metal SIMD width = 32)
    ipt   = max(1, iterations // n_thr)

    seed     = int(base_seed) + int(seed_offset) * 1_000_003
    seed_lo  = seed & 0xFFFF_FFFF
    seed_hi  = (seed >> 32) & 0xFFFF_FFFF

    iparams = array.array("I", [nt, nv, width, height, ipt, _BURN, symmetry, seed_lo, seed_hi])
    fparams = array.array("f", [float(scale), float(cosr), float(sinr), float(cx), float(cy)])

    n_pixels = height * width
    counts_buf    = dev.buffer(n_pixels * 4)   # float32, zero-initialized by Metal
    color_sum_buf = dev.buffer(n_pixels * 4)

    fn(
        n_thr,
        affine, cum, tcol, vw, vid.astype(np.int32),
        iparams, fparams,
        sym_cos, sym_sin,
        counts_buf, color_sum_buf,
    )

    # counts buffer stores uint32 hit counts; reinterpret as float32 to accumulate
    counts += np.frombuffer(bytes(counts_buf), dtype=np.uint32).reshape(height, width).astype(np.float32)
    # color_sum buffer stores float32 bits packed in uint32; view as float32 directly
    color_sum += np.frombuffer(bytes(color_sum_buf), dtype=np.uint32).view(np.float32).reshape(height, width)
