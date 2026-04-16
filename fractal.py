import dearpygui.dearpygui as dpg
import threading
import time
import queue
import json
import os
import math
import numpy as np
from engine import render_stream
from PIL import Image
import palettes as _pal

# All DPG item creation/modification must happen on the main thread.
# Background threads enqueue callables here; the render loop drains them.
_main_queue: queue.SimpleQueue = queue.SimpleQueue()

# Set by the Cancel button; checked between render frames.
_cancel_event = threading.Event()

SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")

# Gamma uses a log-scale slider so low values get more resolution.
# Slider stores log(gamma); actual gamma = exp(slider value).
_GAMMA_LOG_MIN = math.log(0.05)   # actual gamma ≈ 0.05
_GAMMA_LOG_MAX = math.log(8.0)    # actual gamma = 8.0

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

VARIATION_NAMES = ["linear", "sinusoidal", "spherical", "swirl",
                   "horseshoe", "polar", "curl", "pdj"]

ASPECT_PRESETS = {
    "9:16": (1080, 1920),
    "1:1":  (1080, 1080),
    "16:9": (1920, 1080),
    "4:3":  (1440, 1080),
    "3:2":  (1620, 1080),
}

# Creative presets — these set all fractal/color/tone settings but leave
# width/height/output/quality/supersample untouched.
PRESETS = {
    "Inferno": {
        "palette": "fire", "symmetry": 6, "num_transforms": 4,
        "variations": ["swirl", "sinusoidal"], "seed": 42,
        "zoom": 1.0, "rotation": 0.0, "center": [0.0, 0.0],
        "gamma": 2.5, "brightness": 4.0, "vibrancy": 1.0,
        "background": [0, 0, 0],
    },
    "Aurora": {
        "palette": "cyber", "symmetry": 3, "num_transforms": 3,
        "variations": ["sinusoidal", "polar"], "seed": 17,
        "zoom": 0.8, "rotation": 0.0, "center": [0.0, 0.0],
        "gamma": 2.0, "brightness": 2.5, "vibrancy": 0.8,
        "background": [0, 5, 15],
    },
    "Mandala": {
        "palette": "rainbow", "symmetry": 8, "num_transforms": 5,
        "variations": ["pdj", "swirl"], "seed": 100,
        "zoom": 1.2, "rotation": 0.0, "center": [0.0, 0.0],
        "gamma": 2.2, "brightness": 3.0, "vibrancy": 1.0,
        "background": [0, 0, 0],
    },
    "Nebula": {
        "palette": "synthwave", "symmetry": 5, "num_transforms": 6,
        "variations": ["spherical", "curl"], "seed": 77,
        "zoom": 0.9, "rotation": 45.0, "center": [0.0, 0.0],
        "gamma": 3.0, "brightness": 5.0, "vibrancy": 0.9,
        "background": [0, 0, 5],
    },
    "Smoke": {
        "palette": "obsidian", "symmetry": 4, "num_transforms": 4,
        "variations": ["linear", "polar"], "seed": 55,
        "zoom": 1.0, "rotation": 0.0, "center": [0.0, 0.0],
        "gamma": 1.8, "brightness": 2.5, "vibrancy": 0.3,
        "background": [0, 0, 0],
    },
    "Copper Spiral": {
        "palette": "copper", "symmetry": 7, "num_transforms": 4,
        "variations": ["swirl", "horseshoe"], "seed": 23,
        "zoom": 1.0, "rotation": 0.0, "center": [0.0, 0.0],
        "gamma": 2.3, "brightness": 3.0, "vibrancy": 0.7,
        "background": [0, 0, 0],
    },
    "Toxic Bloom": {
        "palette": "toxic", "symmetry": 6, "num_transforms": 5,
        "variations": ["horseshoe", "sinusoidal"], "seed": 88,
        "zoom": 1.1, "rotation": 0.0, "center": [0.0, 0.0],
        "gamma": 2.5, "brightness": 3.5, "vibrancy": 1.0,
        "background": [0, 5, 0],
    },
    "Midnight Crystal": {
        "palette": "midnight", "symmetry": 5, "num_transforms": 4,
        "variations": ["spherical", "pdj"], "seed": 34,
        "zoom": 1.0, "rotation": 0.0, "center": [0.0, 0.0],
        "gamma": 2.8, "brightness": 4.0, "vibrancy": 0.8,
        "background": [0, 0, 10],
    },
    "Solar Flare": {
        "palette": "solar", "symmetry": 4, "num_transforms": 4,
        "variations": ["swirl", "curl"], "seed": 61,
        "zoom": 0.7, "rotation": 0.0, "center": [0.0, 0.0],
        "gamma": 2.8, "brightness": 6.0, "vibrancy": 1.0,
        "background": [5, 0, 0],
    },
    "Rose Mandala": {
        "palette": "rose_gold", "symmetry": 10, "num_transforms": 4,
        "variations": ["pdj", "sinusoidal"], "seed": 200,
        "zoom": 1.3, "rotation": 18.0, "center": [0.0, 0.0],
        "gamma": 2.2, "brightness": 3.5, "vibrancy": 0.9,
        "background": [5, 0, 5],
    },
    "Forest Veil": {
        "palette": "forest", "symmetry": 3, "num_transforms": 5,
        "variations": ["polar", "horseshoe"], "seed": 147,
        "zoom": 1.0, "rotation": 0.0, "center": [0.0, 0.0],
        "gamma": 2.0, "brightness": 2.0, "vibrancy": 0.6,
        "background": [0, 3, 0],
    },
}

PANEL_W = 340


# ── Settings persistence ──────────────────────────────────────────────────────

def load_settings() -> dict:
    if not os.path.exists(SETTINGS_FILE):
        return CONFIG.copy()
    try:
        with open(SETTINGS_FILE) as f:
            saved = json.load(f)
        merged = CONFIG.copy()
        merged.update(saved)
        return merged
    except Exception:
        return CONFIG.copy()


def save_settings(cfg: dict) -> None:
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(cfg, f, indent=2)
    except Exception:
        pass


# Loaded once at startup; all _build_* functions read from this.
_settings = load_settings()


# ── GUI construction ──────────────────────────────────────────────────────────

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
                                  border=False):
                _build_presets_group()
                _build_output_group()
                _build_quality_group()
                _build_shape_group()
                _build_camera_group()
                _build_color_group()
                _build_tone_group()
                _build_buttons()

            # ── RIGHT PANEL ───────────────────────────────────────────
            with dpg.child_window(tag="right_panel", border=False):
                dpg.add_text("Press [Preview] to render", tag="status_text")
                dpg.add_image(texture_tag="preview_tex", tag="preview_img",
                              show=False)

    dpg.set_primary_window("main_win", True)

    def _on_resize():
        vw = dpg.get_viewport_client_width()
        vh = dpg.get_viewport_client_height()
        dpg.configure_item("right_panel", width=vw - PANEL_W, height=vh)
        dpg.configure_item("left_panel", height=vh)

    dpg.set_viewport_resize_callback(lambda s, a: _on_resize())


def _build_presets_group():
    with dpg.collapsing_header(label="Presets", default_open=False):
        dpg.add_text("Loads creative settings (not output/quality):")
        preset_names = list(PRESETS.keys())
        for i in range(0, len(preset_names), 2):
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label=preset_names[i], width=155,
                    callback=lambda s, a, u: _apply_preset(u),
                    user_data=preset_names[i],
                )
                if i + 1 < len(preset_names):
                    dpg.add_button(
                        label=preset_names[i + 1], width=155,
                        callback=lambda s, a, u: _apply_preset(u),
                        user_data=preset_names[i + 1],
                    )


def _apply_preset(name: str):
    p = PRESETS[name]
    if "symmetry" in p:
        dpg.set_value("cfg_symmetry", p["symmetry"])
    if "num_transforms" in p:
        dpg.set_value("cfg_num_transforms", p["num_transforms"])
    if "seed" in p:
        dpg.set_value("cfg_seed", p["seed"])
    if "variations" in p:
        for vname in VARIATION_NAMES:
            dpg.set_value(f"var_{vname}", vname in p["variations"])
    if "zoom" in p:
        dpg.set_value("cfg_zoom", p["zoom"])
    if "rotation" in p:
        dpg.set_value("cfg_rotation", p["rotation"])
    if "center" in p:
        dpg.set_value("cfg_center_x", p["center"][0])
        dpg.set_value("cfg_center_y", p["center"][1])
    if "palette" in p:
        dpg.set_value("cfg_palette", p["palette"])
    if "background" in p:
        dpg.set_value("cfg_bg_r", p["background"][0])
        dpg.set_value("cfg_bg_g", p["background"][1])
        dpg.set_value("cfg_bg_b", p["background"][2])
    if "gamma" in p:
        log_g = math.log(max(p["gamma"], 0.05))
        dpg.set_value("cfg_gamma", log_g)
        dpg.set_value("gamma_display", f"= {p['gamma']:.3f}")
    if "brightness" in p:
        dpg.set_value("cfg_brightness", p["brightness"])
    if "vibrancy" in p:
        dpg.set_value("cfg_vibrancy", p["vibrancy"])


def _build_output_group():
    with dpg.collapsing_header(label="Output", default_open=True):
        dpg.add_input_int(label="Width", tag="cfg_width",
                          default_value=_settings["width"],
                          min_value=64, max_value=8192, width=120)
        dpg.add_input_int(label="Height", tag="cfg_height",
                          default_value=_settings["height"],
                          min_value=64, max_value=8192, width=120)
        dpg.add_input_text(label="Output file", tag="cfg_output",
                           default_value=_settings["output"], width=180)
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
                           default_value=_settings["quality"],
                           min_value=5, max_value=500, width=200)
        dpg.add_checkbox(label="Supersample (2x)", tag="cfg_supersample",
                         default_value=_settings["supersample"] == 2)


def _build_shape_group():
    with dpg.collapsing_header(label="Shape", default_open=True):
        dpg.add_slider_int(label="Symmetry", tag="cfg_symmetry",
                           default_value=_settings["symmetry"],
                           min_value=1, max_value=12, width=200)
        dpg.add_slider_int(label="Transforms", tag="cfg_num_transforms",
                           default_value=_settings["num_transforms"],
                           min_value=2, max_value=8, width=200)
        dpg.add_input_int(label="Seed", tag="cfg_seed",
                          default_value=_settings["seed"], width=120)
        dpg.add_text("Variations:")
        for name in VARIATION_NAMES:
            dpg.add_checkbox(label=name, tag=f"var_{name}",
                             default_value=name in _settings["variations"])


def _build_camera_group():
    with dpg.collapsing_header(label="Camera", default_open=False):
        dpg.add_slider_float(label="Zoom", tag="cfg_zoom",
                             default_value=_settings["zoom"],
                             min_value=0.1, max_value=10.0, width=200)
        dpg.add_slider_float(label="Rotation", tag="cfg_rotation",
                             default_value=_settings["rotation"],
                             min_value=0.0, max_value=360.0, width=200)
        dpg.add_slider_float(label="Center X", tag="cfg_center_x",
                             default_value=_settings["center"][0],
                             min_value=-2.0, max_value=2.0, width=200)
        dpg.add_slider_float(label="Center Y", tag="cfg_center_y",
                             default_value=_settings["center"][1],
                             min_value=-2.0, max_value=2.0, width=200)


def _build_color_group():
    with dpg.collapsing_header(label="Color", default_open=True):
        all_palettes = _pal.get_all_palette_names()
        current_pal = _settings["palette"]
        if current_pal not in all_palettes:
            current_pal = all_palettes[0]
        dpg.add_combo(label="Palette", tag="cfg_palette",
                      items=all_palettes, default_value=current_pal, width=140)
        dpg.add_text("Background RGB:")
        with dpg.group(horizontal=True):
            for i, ch in enumerate(["R", "G", "B"]):
                dpg.add_input_int(label=ch, tag=f"cfg_bg_{ch.lower()}",
                                  default_value=_settings["background"][i],
                                  min_value=0, max_value=255, width=70)

        # ── Add custom palette ────────────────────────────────────────
        dpg.add_spacer(height=4)
        with dpg.collapsing_header(label="Add Custom Palette", default_open=False):
            dpg.add_input_text(label="Name", tag="new_pal_name", width=150)
            dpg.add_text("Color stops:")
            _DEFAULT_STOPS = [
                [0, 0, 0, 255],
                [80, 0, 0, 255],
                [200, 60, 0, 255],
                [255, 160, 0, 255],
                [255, 230, 100, 255],
                [255, 255, 255, 255],
            ]
            for i in range(6):
                dpg.add_color_edit(
                    label=f"Stop {i + 1}",
                    tag=f"new_pal_stop_{i}",
                    default_value=_DEFAULT_STOPS[i],
                    no_alpha=True,
                    width=200,
                )
            dpg.add_button(label="Save Palette", callback=_on_add_palette, width=130)
            dpg.add_text("", tag="add_pal_status")


def _on_add_palette():
    name = dpg.get_value("new_pal_name").strip()
    if not name:
        dpg.set_value("add_pal_status", "Enter a palette name.")
        return
    stops = []
    for i in range(6):
        rgba = dpg.get_value(f"new_pal_stop_{i}")
        stops.append((int(rgba[0]), int(rgba[1]), int(rgba[2])))
    _pal.save_custom_palette(name, stops)
    # Refresh the palette combo to include the new entry
    all_names = _pal.get_all_palette_names()
    dpg.configure_item("cfg_palette", items=all_names)
    dpg.set_value("cfg_palette", name)
    dpg.set_value("add_pal_status", f"Saved '{name}'!")


def _update_gamma_display(sender, app_data):
    dpg.set_value("gamma_display", f"= {math.exp(app_data):.3f}")


def _build_tone_group():
    with dpg.collapsing_header(label="Tone Mapping", default_open=True):
        log_default = math.log(max(_settings["gamma"], 0.05))
        with dpg.group(horizontal=True):
            dpg.add_slider_float(label="Gamma", tag="cfg_gamma",
                                 default_value=log_default,
                                 min_value=_GAMMA_LOG_MIN, max_value=_GAMMA_LOG_MAX,
                                 width=170, callback=_update_gamma_display)
            dpg.add_text(f"= {_settings['gamma']:.3f}", tag="gamma_display")
        dpg.add_slider_float(label="Brightness", tag="cfg_brightness",
                             default_value=_settings["brightness"],
                             min_value=0.1, max_value=10.0, width=200)
        dpg.add_slider_float(label="Vibrancy", tag="cfg_vibrancy",
                             default_value=_settings["vibrancy"],
                             min_value=0.0, max_value=1.0, width=200)


def _build_buttons():
    dpg.add_spacer(height=8)
    with dpg.group(horizontal=True):
        dpg.add_button(label="Preview", tag="btn_preview",
                       callback=_on_preview, width=140, height=36)
        dpg.add_button(label="Render & Save", tag="btn_render",
                       callback=_on_render, width=150, height=36)
        dpg.add_button(label="Cancel", tag="btn_cancel",
                       callback=_on_cancel, width=80, height=36, show=False)


# ── Config collection ─────────────────────────────────────────────────────────

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
        "gamma":          math.exp(dpg.get_value("cfg_gamma")),
        "brightness":     dpg.get_value("cfg_brightness"),
        "vibrancy":       dpg.get_value("cfg_vibrancy"),
    }


# ── Rendering ─────────────────────────────────────────────────────────────────

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
    if dpg.does_alias_exist("preview_tex"):
        dpg.remove_alias("preview_tex")

    dpg.add_static_texture(width=w, height=h, default_value=flat,
                           tag="preview_tex", parent="tex_registry")

    panel_w = dpg.get_item_width("right_panel")
    panel_h = dpg.get_item_height("right_panel")
    status_h = dpg.get_item_height("status_text")
    avail_w = max(panel_w - 20, 100)
    avail_h = max(panel_h - status_h - 20, 100)
    scale = min(avail_w / w, avail_h / h)
    disp_w, disp_h = int(w * scale), int(h * scale)

    dpg.configure_item("preview_img", texture_tag="preview_tex",
                       width=disp_w, height=disp_h, show=True)


def _run_render(config: dict, save: bool) -> None:
    _cancel_event.clear()
    _main_queue.put(lambda: (
        _set_buttons_enabled(False),
        dpg.configure_item("btn_cancel", show=True),
        dpg.set_value("status_text", "Rendering…"),
    ))
    t0 = time.perf_counter()
    img = None
    try:
        for img, progress in render_stream(config):
            if not dpg.is_dearpygui_running():
                return
            if _cancel_event.is_set():
                _main_queue.put(lambda: dpg.set_value("status_text", "Cancelled"))
                return
            pct = int(progress * 100)
            flat, w, h = _prepare_texture(img)
            _main_queue.put(lambda f=flat, W=w, H=h, p=pct: (
                _show_texture(f, W, H),
                dpg.set_value("status_text", f"Rendering… {p}%"),
            ))

        if img is None or not dpg.is_dearpygui_running():
            return

        elapsed = time.perf_counter() - t0
        if save:
            path = config["output"]
            img.save(path)
            _main_queue.put(lambda msg=f"Saved to {path}  ({elapsed:.1f}s)":
                            dpg.set_value("status_text", msg))
        else:
            msg = (f"Preview done  ({elapsed:.1f}s)  "
                   f"{config['width']}×{config['height']}")
            _main_queue.put(lambda m=msg: dpg.set_value("status_text", m))
    except Exception as e:
        _main_queue.put(lambda msg=f"Error: {e}":
                        dpg.set_value("status_text", msg))
    finally:
        _main_queue.put(lambda: (
            _set_buttons_enabled(True),
            dpg.configure_item("btn_cancel", show=False),
        ))


def _on_cancel():
    _cancel_event.set()


def _on_preview():
    cfg = collect_config()
    save_settings(cfg)
    cfg["quality"] = 15
    cfg["supersample"] = 1
    threading.Thread(target=_run_render, args=(cfg, False), daemon=True).start()


def _on_render():
    cfg = collect_config()
    save_settings(cfg)
    threading.Thread(target=_run_render, args=(cfg, True), daemon=True).start()


def _apply_initial_sizing():
    """Set panel sizes to match the viewport on startup."""
    vw = dpg.get_viewport_client_width()
    vh = dpg.get_viewport_client_height()
    if vw > 0 and vh > 0:
        dpg.configure_item("right_panel", width=vw - PANEL_W, height=vh)
        dpg.configure_item("left_panel", height=vh)


if __name__ == "__main__":
    build_gui()
    dpg.show_viewport()
    # Render one frame so viewport dimensions are available, then size panels
    dpg.render_dearpygui_frame()
    _apply_initial_sizing()
    while dpg.is_dearpygui_running():
        # Drain main-thread callbacks (texture uploads, status updates, etc.)
        try:
            while True:
                _main_queue.get_nowait()()
        except queue.Empty:
            pass
        dpg.render_dearpygui_frame()
    # Persist settings on clean close
    try:
        save_settings(collect_config())
    except Exception:
        pass
    dpg.destroy_context()
