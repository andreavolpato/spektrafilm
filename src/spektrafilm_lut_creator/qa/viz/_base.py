"""Shared palette + plot-style primitives used by every QA viz module.

Constants here define the suite's visual signature (dark background,
fixed palette, canonical typography sizing). Functions here are the
small primitives — identity reference lines, axis setup, OkLab /
gamut-triangle conversions — that every topic-specific viz module
reaches for.

Re-exported in :mod:`spektrafilm_lut_creator.qa.viz` so callers can
keep using ``viz.add_footer``, ``viz.BG`` etc. unchanged.
"""

from __future__ import annotations

import colour
import numpy as np

# ---------------------------------------------------------------------------
# Palette.
# ---------------------------------------------------------------------------
BG = "#0a0a0a"
FG = "#cccccc"
HI = "#ffffff"
DIM = "#666666"
RED, GREEN, BLUE = "#ff5050", "#50ff80", "#5080ff"
WARN = "#ffd060"

PANE_EDGE_RGBA = (1.0, 1.0, 1.0, 0.22)
GRID_RGBA = (1.0, 1.0, 1.0, 0.12)

# ---------------------------------------------------------------------------
# Canonical typography (n090 §3 trust signals — figures of-a-piece).
# Suptitle = figure-level title; PANEL = inner-panel title.
# ---------------------------------------------------------------------------
SUPTITLE_FS = 14
PANEL_TITLE_FS = 12
SUPTITLE_PAD = 10
PANEL_TITLE_PAD = 8
IDENTITY_COLOR = "#555555"
IDENTITY_ALPHA = 0.7
FOOTER_FS = 8
FOOTER_COLOR = DIM
FOOTER_BAND_FRAC = 0.04
"""Fraction of figure height reserved for the version footer.

Sized so x-axis tick labels of the bottom subplot row can't bleed into
the band: at the suite's typical 9–10 inch tall figures and 160 DPI,
0.04 ≈ 0.4 inch ≈ 60 px of clear space — enough for a centered 8pt
footer with breathing room on either side."""
HEADER_BAND_FRAC = 0.08
"""Fraction of figure height reserved at the top for the suptitle.

When :func:`add_footer` clamps the constrained-layout rect to leave a
band free at the bottom for the version footer, the same rect's *top*
needs to be set explicitly too — otherwise matplotlib treats the rect
as the working area and tries to fit the suptitle inside it, with
nowhere to go above. The suptitle ends up flush with the figure's top
edge and overlaps with panel titles.

0.08 is sized to accommodate the suite's largest suptitle (two lines
at 14pt) on figures down to ~5 inches tall. Single-line suptitles
waste a couple of percent of the figure to whitespace, which is the
acceptable price for not having to thread "is your suptitle one or
two lines" through every viz function."""


def _identity_line(
    ax, *, label: str | None = "identity", lo: float = 0.0, hi: float = 1.0
) -> None:
    """Standard y=x reference line for 2D transfer plots.

    Centralizing the styling so every figure uses the same gray /
    dash style — no more drift between ``#444444`` and ``#555555``.
    """
    ax.plot(
        [lo, hi],
        [lo, hi],
        color=IDENTITY_COLOR,
        lw=1.0,
        ls="--",
        alpha=IDENTITY_ALPHA,
        label=label,
        zorder=1,
    )


def add_footer(fig, version: str) -> None:
    """Stamp the canonical ``spektrafilm <version>`` footer on a figure.

    Reserves :data:`FOOTER_BAND_FRAC` of the figure height as a clear
    band at the bottom, then places the version text centered in it.
    The reserve is applied through whichever layout system the figure
    uses:

    - **Constrained-layout figures** (every 2D figure in this module)
      have their layout engine's ``rect`` clamped to leave the bottom
      band free — subplots reflow upward and tick labels stop at the
      band's top edge instead of bleeding into the footer.
    - **Manually-laid-out figures** (the 3D scatters, which call
      :func:`_fill_3d`) get a bumped ``subplots_adjust(bottom=...)``
      instead, preserving the other margins ``_fill_3d`` set.

    Called from :func:`spektrafilm_lut_creator.qa.tests._save` so every
    saved figure carries the producing version automatically — a
    trust signal that survives the figure leaving its bundle (n090 §3).
    """
    engine = fig.get_layout_engine()
    if engine is not None:
        # Constrained layout: clamp the subplot rect on *both* sides so
        # neither the footer band nor the suptitle gets eaten. The top
        # bound is the load-bearing one — without it constrained-layout
        # tries to fit the suptitle inside the working area and the
        # title ends up flush with the figure's top edge, colliding
        # with panel titles.
        engine.set(rect=(0.0, FOOTER_BAND_FRAC, 1.0, 1.0 - HEADER_BAND_FRAC))
    elif fig.subplotpars.bottom < FOOTER_BAND_FRAC:
        # Manual layout (the _fill_3d path): widen the bottom margin
        # only if it's currently tighter than the band. Other margins
        # set by _fill_3d are preserved.
        fig.subplots_adjust(bottom=FOOTER_BAND_FRAC)
    # Text centered vertically within the band.
    fig.text(
        0.5,
        FOOTER_BAND_FRAC / 2.0,
        f"spektrafilm {version}",
        color=FOOTER_COLOR,
        fontsize=FOOTER_FS,
        ha="center",
        va="center",
        alpha=0.85,
    )


def _setup_3d(ax) -> None:
    """Apply the project's consistent 3D-axis styling.

    The fix for the "axis lines barely visible" issue: pane edges at
    α=0.22 and a faint grid line set provide depth cues without
    overwhelming the data points.
    """
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor(PANE_EDGE_RGBA)
        axis._axinfo["grid"]["color"] = GRID_RGBA
        axis._axinfo["grid"]["linewidth"] = 0.5
        for lbl in axis.get_ticklabels():
            lbl.set_color(FG)
    ax.grid(True)


def _setup_2d(ax) -> None:
    """Consistent 2D axis styling."""
    ax.tick_params(colors=FG)
    for spine in ax.spines.values():
        spine.set_color("#555555")
    ax.grid(True, alpha=0.12, color=HI)


def _fill_3d(
    fig,
    *,
    has_cbar: bool = False,
    top: float | None = None,
    bottom: float | None = None,
    wspace: float = 0.0,
) -> None:
    """Push 3D axes to fill the figure canvas.

    matplotlib 3D axes default to a generous interior margin; for the
    plots we ship that leaves the data sitting in the middle 50–60% of
    the figure with a lot of dead black background around it. We pull
    the axes box outward with subplots_adjust; ``bbox_inches="tight"``
    at save time then trims off whatever tick decorations stick out.

    ``top`` / ``bottom`` default to :data:`HEADER_BAND_FRAC` /
    :data:`FOOTER_BAND_FRAC` complements so the suptitle and version
    footer have the same reserved bands as the constrained-layout
    figures. Callers can override for 2-line suptitles that need more
    top space.
    """
    if top is None:
        top = 1.0 - HEADER_BAND_FRAC
    if bottom is None:
        bottom = FOOTER_BAND_FRAC
    right = 0.92 if has_cbar else 1.0
    fig.subplots_adjust(left=0.0, right=right, bottom=bottom, top=top, wspace=wspace)


def _to_oklab(rgb: np.ndarray, cs_name: str) -> np.ndarray:
    """RGB encoded in ``cs_name`` → OkLab via reflectance-scale XYZ.

    Used by every plot that overlays OkLab structure on LUT output.
    Uses :func:`to_xyz_qa` so HDR spaces (where decode_cctf returns
    absolute nits) get normalized to the reflectance scale OkLab and
    every downstream chromaticity convention assumes.
    """
    from spektrafilm_lut_creator.color_spaces import to_xyz_qa

    xyz = to_xyz_qa(np.asarray(rgb, dtype=float), cs_name)
    return np.asarray(colour.XYZ_to_Oklab(xyz), dtype=float)


def _gamut_triangle_xy(cs_name: str) -> np.ndarray:
    """Closed (R, G, B, R) chromaticity triangle for the registry entry's primaries."""
    from spektrafilm_lut_creator.color_spaces import get as get_cs

    entry = get_cs(cs_name)
    primaries = np.asarray(
        colour.RGB_COLOURSPACES[entry.primaries].primaries, dtype=float
    )
    return np.vstack([primaries, primaries[:1]])


def _format_output_gamut_compress(spec) -> str:
    """One-line summary of the output gamut compression spec for plot
    suptitles: ``compress: <algorithm> · knee=(t, l, p)``.
    ``algorithm='off'`` short-circuits to ``compress: off``."""
    if spec.algorithm == "off":
        return "compress: off"
    t, l, p = spec.knee
    return f"compress: {spec.algorithm} · knee=(t={t:g}, l={l:g}, p={p:g})"


# ---------------------------------------------------------------------------
# Cube views (3D scatters of the LUT in input or output coordinates).
# ---------------------------------------------------------------------------
