# spektrafilm — runtime core

The physically-based film-simulation engine.

## Philosophy

This package is **the core**. It is where the science and the physical models live.
Everything else in the repo depends on it; it depends on nothing in the repo.

The runtime is deliberately kept **clutter-free**. The physics is already difficult on
its own; the runtime is the one place in the codebase where that complexity is allowed
to breathe without application-layer concerns pressing in.
The cost of letting clutter in here is exponential: it makes the science harder to read,
harder to reason about, harder to extend with new physical models.

## The contract

The pipeline is **linear RGB in named primaries → film + print + scan physics → linear
RGB in named primaries**. That is the contract.

- **Input**: linear-light RGB in a named RGB primaries set (a colour-science
  `RGB_COLOURSPACES` key, e.g. `"ITU-R BT.2020"`). The scene-referred light is what the
  physics actually models.
- **Output**: linear-light RGB in a named RGB primaries set (the output display’s
  primaries). Whatever the physics produces, in whatever range it produces.
- **Primaries are the only color information the runtime needs.** No display gamma, no
  log curves, no HDR transfer functions, no display white-point assumptions baked into
  the math.

A tap dispatcher (see `runtime/topology.py`) lets callers enter and exit the pipeline at
named internal points (`log_e_film`, `cmy_film`, `log_e_print`, `cmy_print`). The
contract above is the `rgb_in → rgb_out` happy path; intermediate taps have their own
domain conventions documented at the call site.

## What the runtime does NOT do

These belong to layers above:

- **Transport encodings**: CCTF curves (sRGB, BT.1886), camera log curves (LogC3,
  S-Log3, V-Log), HDR encodings (PQ, HLG). The runtime speaks linear.
- **Display referencing**: clipping output to `[0, 1]`, tone-mapping, HDR roll-off.
  The runtime returns physics; whoever consumes it decides what “display white” means.
- **LUT creation**: format support, sampling grids, bundle layouts, registries of
  color-space names. That lives in `spektrafilm_lut_creator`.
- **Interactive UX**: image preview, parameter widgets, color-space pickers, file
  dialogs. That lives in `spektrafilm_gui`.
- **Backward-compatibility shims**: deprecated parameter names, fallback paths, legacy
  callers. The runtime is a moving target by design; downstream packages adapt.

## The CCTF convenience

`IOParams` carries two boolean flags — `input_cctf_decoding` and `output_cctf_encoding`
— that let a caller opt into colour-science’s built-in CCTF for the named colourspace.
This is a **convenience**, not a principle.
It works only when the colourspace’s bundled CCTF is what the caller wants (i.e.,
sRGB-class SDR cases where primaries and curve are conventionally paired).

It exists because (a) it costs nothing — the colour-science call is already there — and
(b) the GUI’s preview pipeline benefits from treating display encoding as a single flag
flip. It does **not** work for log spaces paired with non-default gamuts (Apple Log +
BT.2020, BMG5 + BMD Wide Gamut), and it does **not** work for HDR where unclipped linear
is the right output.

For anything beyond the simple SDR case, callers pass `input_cctf_decoding=False` /
`output_cctf_encoding=False`, hand the runtime linear, and own the transport encoding
themselves. The LUT creator does this universally.

If the convenience flags ever stand in the way of the philosophy, they go.
The philosophy doesn’t.

## Layering

```text
spektrafilm_gui  ──┐
                   ├──depends on──► spektrafilm
spektrafilm_lut_creator ─┘
```

`spektrafilm` imports nothing from `spektrafilm_gui` or `spektrafilm_lut_creator`. Ever.
If the runtime needs something that currently lives in one of those packages, it gets
moved into the runtime — not imported up the stack.

## Practical guidance for changes

- New physics goes here.
  New formats, registries, display logic do not.
- A new color space “doesn’t work in the runtime” almost always means the user is
  passing a non-canonical (primaries, CCTF) pairing.
  Fix it in the caller, not by adding a registry to the runtime.
- If you find yourself adding a string-keyed lookup table here, stop and ask whether the
  lookup belongs in `spektrafilm_lut_creator` or in the GUI.
- The pipeline’s behavior on a synthetic gray ramp is the regression baseline.
  Math changes should be deliberate; bit-for-bit drift matters and is caught by
  `tests/test_regression_baselines.py`.

## Where to look

- `runtime/topology.py` — tap dispatcher, the Node/Tap model
- `runtime/pipeline.py` — orchestrates stages around the topology
- `runtime/stages/` — filming, printing, scanning (the physics)
- `runtime/services/` — shared services (spectral LUT cache, resize, etc.)
- `model/` — physical models (emulsion, density curves, halation, diffusion)
- `profiles/` — film + paper profile data + I/O
