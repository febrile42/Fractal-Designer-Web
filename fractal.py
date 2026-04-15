import dearpygui.dearpygui as dpg
import threading
import time
import numpy as np
from engine import render_stream
from PIL import Image

# ── Default config ────────────────────────────────────────────────────────────
CONFIG = {
    "width": 1080,
    "height": 1920,
    "output": "fractal.png",
    "quality": 50,
    "supersample": 2,
    "symmetry": 6,
    "num_transforms": 4,
    "variations": ["swirl", "sinusoidal"],
    "seed": 42,
    "zoom": 1.0,
    "rotation": 0.0,
    "center": [0.0, 0.0],
    "palette": "fire",
    "background": [0, 0, 0],
    "gamma": 2.5,
    "brightness": 3.0,
    "vibrancy": 1.0,
}

PALETTES = ["fire", "ice", "neon", "rainbow", "gold", "violet"]
VARIATION_NAMES = ["linear", "sinusoidal", "spherical", "swirl",
                   "horseshoe", "polar", "curl", "pdj"]
ASPECT_PRESETS = {
    "9:16": (1080, 1920),
    "1:1":  (1080, 1080),
    "16:9": (1920, 1080),
    "4:3":  (1440, 1080),
    "3:2":  (1620, 1080),
}

PANEL_W = 340


def build_gui():
    dpg.create_context()
    dpg.create_viewport(title="Flame Fractal Generator", width=1400, height=900)
    dpg.setup_dearpygui()

    # Pre-create a 1x1 dummy texture so add_image doesn't crash
    with dpg.texture_registry(tag="tex_registry"):
        dpg.add_static_texture(width=1, height=1,
                               default_value=[0.0, 0.0, 0.0, 0.0],
                               tag="preview_tex")

    with dpg.window(tag="main_win", no_title_bar=True, no_move=True,
                    no_resize=True, no_scrollbar=True):

        with dpg.group(horizontal=True):

            # ── LEFT PANEL ────────────────────────────────────────────
            with dpg.child_window(tag="left_panel", width=PANEL_W,
                                  border=False, auto_resize_y=True):
                _build_output_group()
                _build_quality_group()
                _build_shape_group()
                _build_camera_group()
                _build_color_group()
                _build_tone_group()
                _build_buttons()

            # ── RIGHT PANEL ───────────────────────────────────────────
            with dpg.child_window(tag="right_panel", border=False,
                                  auto_resize_y=True):
                dpg.add_text("Press [Preview] to render", tag="status_text")
                dpg.add_image(texture_tag="preview_tex", tag="preview_img",
                              show=False)

    dpg.set_primary_window("main_win", True)


def _build_output_group():
    with dpg.collapsing_header(label="Output", default_open=True):
        dpg.add_input_int(label="Width", tag="cfg_width",
                          default_value=CONFIG["width"], min_value=64, max_value=8192, width=120)
        dpg.add_input_int(label="Height", tag="cfg_height",
                          default_value=CONFIG["height"], min_value=64, max_value=8192, width=120)
        dpg.add_input_text(label="Output file", tag="cfg_output",
                           default_value=CONFIG["output"], width=180)
        dpg.add_text("Aspect presets:")
        with dpg.group(horizontal=True):
            for label in ASPECT_PRESETS:
                dpg.add_button(label=label,
                               callback=lambda s, a, u: _apply_aspect(u),
                               user_data=label,
                               width=55)
        dpg.add_text("", tag="aspect_selected")


def _apply_aspect(label: str):
    w, h = ASPECT_PRESETS[label]
    dpg.set_value("cfg_width", w)
    dpg.set_value("cfg_height", h)
    dpg.set_value("aspect_selected", f"Selected: {label}  ({w}×{h})")


def _build_quality_group():
    with dpg.collapsing_header(label="Quality", default_open=True):
        dpg.add_slider_int(label="Samples/px", tag="cfg_quality",
                           default_value=CONFIG["quality"],
                           min_value=5, max_value=500, width=200)
        dpg.add_checkbox(label="Supersample (2x)", tag="cfg_supersample",
                         default_value=CONFIG["supersample"] == 2)


def _build_shape_group():
    with dpg.collapsing_header(label="Shape", default_open=True):
        dpg.add_slider_int(label="Symmetry", tag="cfg_symmetry",
                           default_value=CONFIG["symmetry"],
                           min_value=1, max_value=12, width=200)
        dpg.add_slider_int(label="Transforms", tag="cfg_num_transforms",
                           default_value=CONFIG["num_transforms"],
                           min_value=2, max_value=8, width=200)
        dpg.add_input_int(label="Seed", tag="cfg_seed",
                          default_value=CONFIG["seed"], width=120)
        dpg.add_text("Variations:")
        for name in VARIATION_NAMES:
            dpg.add_checkbox(label=name, tag=f"var_{name}",
                             default_value=name in CONFIG["variations"])


def _build_camera_group():
    with dpg.collapsing_header(label="Camera", default_open=False):
        dpg.add_slider_float(label="Zoom", tag="cfg_zoom",
                             default_value=CONFIG["zoom"],
                             min_value=0.1, max_value=10.0, width=200)
        dpg.add_slider_float(label="Rotation", tag="cfg_rotation",
                             default_value=CONFIG["rotation"],
                             min_value=0.0, max_value=360.0, width=200)
        dpg.add_slider_float(label="Center X", tag="cfg_center_x",
                             default_value=CONFIG["center"][0],
                             min_value=-2.0, max_value=2.0, width=200)
        dpg.add_slider_float(label="Center Y", tag="cfg_center_y",
                             default_value=CONFIG["center"][1],
                             min_value=-2.0, max_value=2.0, width=200)


def _build_color_group():
    with dpg.collapsing_header(label="Color", default_open=True):
        dpg.add_combo(label="Palette", tag="cfg_palette",
                      items=PALETTES, default_value=CONFIG["palette"], width=140)
        dpg.add_text("Background RGB:")
        with dpg.group(horizontal=True):
            for i, ch in enumerate(["R", "G", "B"]):
                dpg.add_input_int(label=ch, tag=f"cfg_bg_{ch.lower()}",
                                  default_value=CONFIG["background"][i],
                                  min_value=0, max_value=255, width=70)


def _build_tone_group():
    with dpg.collapsing_header(label="Tone Mapping", default_open=True):
        dpg.add_slider_float(label="Gamma", tag="cfg_gamma",
                             default_value=CONFIG["gamma"],
                             min_value=0.5, max_value=5.0, width=200)
        dpg.add_slider_float(label="Brightness", tag="cfg_brightness",
                             default_value=CONFIG["brightness"],
                             min_value=0.1, max_value=10.0, width=200)
        dpg.add_slider_float(label="Vibrancy", tag="cfg_vibrancy",
                             default_value=CONFIG["vibrancy"],
                             min_value=0.0, max_value=1.0, width=200)


def _build_buttons():
    dpg.add_spacer(height=8)
    with dpg.group(horizontal=True):
        dpg.add_button(label="Preview", tag="btn_preview",
                       callback=_on_preview, width=140, height=36)
        dpg.add_button(label="Render & Save", tag="btn_render",
                       callback=_on_render, width=150, height=36)


def collect_config() -> dict:
    active_vars = [name for name in VARIATION_NAMES
                   if dpg.get_value(f"var_{name}")]
    if not active_vars:
        active_vars = ["linear"]
    return {
        "width":          dpg.get_value("cfg_width"),
        "height":         dpg.get_value("cfg_height"),
        "output":         dpg.get_value("cfg_output"),
        "quality":        dpg.get_value("cfg_quality"),
        "supersample":    2 if dpg.get_value("cfg_supersample") else 1,
        "symmetry":       dpg.get_value("cfg_symmetry"),
        "num_transforms": dpg.get_value("cfg_num_transforms"),
        "variations":     active_vars,
        "seed":           dpg.get_value("cfg_seed"),
        "zoom":           dpg.get_value("cfg_zoom"),
        "rotation":       dpg.get_value("cfg_rotation"),
        "center":         [dpg.get_value("cfg_center_x"),
                           dpg.get_value("cfg_center_y")],
        "palette":        dpg.get_value("cfg_palette"),
        "background":     [dpg.get_value("cfg_bg_r"),
                           dpg.get_value("cfg_bg_g"),
                           dpg.get_value("cfg_bg_b")],
        "gamma":          dpg.get_value("cfg_gamma"),
        "brightness":     dpg.get_value("cfg_brightness"),
        "vibrancy":       dpg.get_value("cfg_vibrancy"),
    }


def _set_buttons_enabled(enabled: bool):
    dpg.configure_item("btn_preview", enabled=enabled)
    dpg.configure_item("btn_render", enabled=enabled)


def _prepare_texture(img: Image.Image) -> tuple[list, int, int]:
    """Convert PIL Image to flat float RGBA list. Safe to call off the DPG mutex."""
    img_rgba = img.convert("RGBA")
    w, h = img_rgba.size
    flat = (np.array(img_rgba, dtype=np.float32) / 255.0).ravel().tolist()
    return flat, w, h


def _show_texture(flat: list, w: int, h: int) -> None:
    """Upload pre-converted texture data to DPG. Must be called under dpg.mutex()."""
    if dpg.does_item_exist("preview_tex"):
        dpg.delete_item("preview_tex")

    dpg.add_static_texture(width=w, height=h, default_value=flat,
                           tag="preview_tex", parent="tex_registry")

    panel_w = dpg.get_item_width("right_panel")
    max_w = max(panel_w - 20, 100)
    scale = min(1.0, max_w / w)
    disp_w, disp_h = int(w * scale), int(h * scale)

    dpg.configure_item("preview_img", texture_tag="preview_tex",
                       width=disp_w, height=disp_h, show=True)


def _run_render(config: dict, save: bool) -> None:
    with dpg.mutex():
        _set_buttons_enabled(False)
        dpg.set_value("status_text", "Rendering…")
    t0 = time.perf_counter()
    img = None
    try:
        for img, progress in render_stream(config):
            if not dpg.is_dearpygui_running():
                return
            pct = int(progress * 100)
            flat, w, h = _prepare_texture(img)   # CPU work outside mutex
            with dpg.mutex():
                _show_texture(flat, w, h)
                dpg.set_value("status_text", f"Rendering… {pct}%")

        if img is None or not dpg.is_dearpygui_running():
            return

        elapsed = time.perf_counter() - t0
        with dpg.mutex():
            if save:
                path = config["output"]
                img.save(path)
                dpg.set_value("status_text",
                              f"Saved to {path}  ({elapsed:.1f}s)")
            else:
                dpg.set_value("status_text",
                              f"Preview done  ({elapsed:.1f}s)  "
                              f"{config['width']}×{config['height']}")
    except Exception as e:
        if dpg.is_dearpygui_running():
            with dpg.mutex():
                dpg.set_value("status_text", f"Error: {e}")
    finally:
        if dpg.is_dearpygui_running():
            with dpg.mutex():
                _set_buttons_enabled(True)


def _on_preview():
    cfg = collect_config()
    cfg["quality"] = 15
    cfg["supersample"] = 1
    threading.Thread(target=_run_render, args=(cfg, False), daemon=True).start()


def _on_render():
    cfg = collect_config()
    threading.Thread(target=_run_render, args=(cfg, True), daemon=True).start()


if __name__ == "__main__":
    build_gui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()
