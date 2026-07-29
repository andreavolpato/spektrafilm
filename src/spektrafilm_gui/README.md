# spektrafilm_gui — interactive shell

The user-facing interactive application: image preview, parameter widgets, profile
management, animation, exports.

## Philosophy

This package is the **interactive layer** on top of `spektrafilm`. It depends on the
runtime; it is not depended on by the runtime.

The GUI exists to make the science accessible.
Its job is to:

- Let users load and explore images.
- Surface the runtime’s parameters in widgets that respect the underlying model.
- Show the runtime’s output in a form a human can see (encoded for the display).
- Persist user state, profiles, and project files.

The science itself is **not the GUI’s responsibility**. If a piece of behavior is about
*what the simulation does*, it belongs in the runtime.
If it is about *how the user interacts with the simulation*, it belongs here.

## What the GUI owns

- **Display encoding**. The GUI presents linear-light pipeline output on encoded screens
  (sRGB, Display P3). It is the GUI’s job to do the final CCTF encode that turns linear
  RGB into screen-ready pixels.
  Today this is handled by setting `IOParams.output_cctf_encoding = True` and letting
  the runtime’s convenience flag do the work; that arrangement is fine for simple SDR
  cases.
- **Parameter widgets, layout, persistence**. The widget specs, controllers, state
  bridge, and persistence machinery all live here.
- **Color-space pickers**. If the GUI grows a color-space picker for inputs/outputs, it
  owns the UX of that picker — what names are shown, how unfamiliar names are explained.
  It may consult `spektrafilm_lut_creator`’s registry, or it may maintain its own
  smaller curated list, depending on what makes sense for users.
- **Image I/O and preview pipelines**. Loading raw / TIFF / EXR, preview rescaling,
  output formatting for napari layers.

## Layering rule

`spektrafilm_gui` imports from `spektrafilm` and may import from
`spektrafilm_lut_creator`. **Neither imports from the GUI.** If you find yourself
wanting the runtime to “know about” something GUI-specific (preview state, an option
that only the GUI cares about), the right answer is almost always to keep that knowledge
in the GUI.

## Practical guidance

- New widgets, new layouts, new controllers go here.
- New physics goes in `spektrafilm`. The GUI exposes it via widgets; it does not
  implement it.
- New LUT formats or registry entries go in `spektrafilm_lut_creator`. The GUI surfaces
  them via menus / dialogs.
- “Preview mode” tweaks to the pipeline live in the runtime (`settings.preview_mode`);
  the GUI flips the flag.
- Anything that needs to round-trip through a project file goes through
  `controller_persistence.py` and the state bridge.

## Where to look

- `app.py` — entry point
- `napari_layout.py` — napari integration
- `controller*.py` — per-domain controllers (runtime, layers, profile sync, persistence)
- `state.py`, `state_bridge.py` — central state + napari ↔ params plumbing
- `widget_sections.py`, `widget_editors.py` — widget definitions
- `params_mapper.py` — params ↔ widget value translation
- `persistence.py` — project I/O
- `polaroid_animation.py`, `virtual_photo_paper_back.py` — specialty UIs
- `theme*.py`, `icons.py` — visual styling
