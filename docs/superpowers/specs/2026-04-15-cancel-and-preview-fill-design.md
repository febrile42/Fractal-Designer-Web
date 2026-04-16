# Cancel Render + Preview Fill Design

## Goal

Two independent UI improvements to the flame fractal generator:
1. A Cancel button that stops an in-progress render after the current frame
2. The preview image fills as much of the right panel as possible (width and height)

## Architecture

Both changes are isolated to `fractal.py`. No engine changes required.

---

## Feature 1: Cancel Render

### State

A module-level `threading.Event` named `_cancel_event` acts as the stop signal. It is cleared at the start of each render and set when the user clicks Cancel.

### Button

A `Cancel` button (tag `btn_cancel`) is added to the button row alongside Preview and Render & Save. It starts hidden (`show=False`). When a render begins it is shown and enabled; when the render finishes (complete, cancelled, or error) it is hidden again.

The existing Preview and Render & Save buttons are disabled while rendering (existing behaviour). Cancel is the only active button during a render.

### Cancellation flow

In `_run_render`, after preparing each texture frame (`_prepare_texture`), check `_cancel_event.is_set()` before enqueuing the texture upload. If set, enqueue a status update `"Cancelled"` and return. The `finally` block re-enables the buttons and hides Cancel regardless of how the function exits.

### No partial-save on cancel

If the user cancels a Render & Save, the file is not saved. The last displayed frame stays visible.

---

## Feature 2: Preview Fill

### Current behaviour

`_show_texture` scales the image to fit only the panel width: `scale = min(1.0, (panel_w - 20) / w)`. This leaves large empty vertical space below the image.

### New behaviour

Scale to fit both dimensions:

```
available_w = panel_w - 20
available_h = panel_h - status_h - 20
scale = min(1.0, available_w / w, available_h / h)
```

Where:
- `panel_w` = `dpg.get_item_width("right_panel")`
- `panel_h` = `dpg.get_item_height("right_panel")`
- `status_h` = `dpg.get_item_height("status_text")`

The 20px padding is applied to both axes. Scale is still capped at 1.0 to avoid upscaling small preview images.

---

## Components changed

| File | Change |
|------|--------|
| `fractal.py` | Add `_cancel_event`, Cancel button, cancel check in `_run_render`, update `_show_texture` scale logic |

## Error handling

- If `available_h` comes back as 0 or negative (panel not yet laid out), fall back to width-only scaling.
- Cancel during the very first frame (before any image is shown) leaves the panel blank with status "Cancelled".

## Testing

No automated tests for UI layout. Manual verification:
- Cancel during preview stops after the next completed frame
- Cancel during render does not write the output file
- Preview image fills width on landscape fractals and height on portrait fractals
- Resize the window: image reflows to fill available space on next render
