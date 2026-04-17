# Fractal Designer

A browser-based flame fractal renderer — no server, no build step, no dependencies. Open `static/index.html` and hit **Render**.

All computation runs client-side via [WebGPU](https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API) compute shaders.

## Features

- 8 variation types: linear, sinusoidal, spherical, swirl, horseshoe, polar, curl, PDJ
- 16 color palettes with smooth interpolation and vibrancy control
- Logarithmic tone mapping with gamma and brightness controls
- Rotational symmetry (1–12)
- Supersampling up to 4×
- Progressive preview — image builds up across 7 passes
- Save output as PNG

## Requirements

A browser with WebGPU support (Chrome 113+, Edge 113+). Safari and Firefox require flags or are not yet supported.

## Usage

Open `static/index.html` directly in a WebGPU-capable browser. Tweak the controls and hit **Render**. Use **Randomize** (⚄) to explore random configurations, or **Save** to export a PNG.

## Credits

Forked from [r0k3tkutt3r/Fractal-Designer](https://github.com/r0k3tkutt3r/Fractal-Designer). Original project built a Python/Dear PyGui desktop app; this fork migrated the renderer to a fully client-side WebGPU implementation.
