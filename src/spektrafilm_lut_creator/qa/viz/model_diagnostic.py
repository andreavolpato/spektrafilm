"""Viz panels for the model-diagnostic QA tests."""
from __future__ import annotations

import colour
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from spektrafilm_lut_creator.qa.viz._base import (
    BG, FG, HI, DIM, RED, GREEN, BLUE, WARN,
    PANE_EDGE_RGBA, GRID_RGBA,
    SUPTITLE_FS, PANEL_TITLE_FS, SUPTITLE_PAD, PANEL_TITLE_PAD,
    IDENTITY_COLOR, IDENTITY_ALPHA,
    FOOTER_FS, FOOTER_COLOR, FOOTER_BAND_FRAC, HEADER_BAND_FRAC,
    _identity_line, add_footer,
    _setup_3d, _setup_2d, _fill_3d,
    _to_oklab, _gamut_triangle_xy,
)


def density_transfer_curves(
    sweep_x: np.ndarray,
    r_sweep: list[tuple[float, np.ndarray, float]],
    g_sweep: list[tuple[float, np.ndarray, float]],
    b_sweep: list[tuple[float, np.ndarray, float]],
    neutral_samples: np.ndarray,
    *,
    title: str = "Density-domain transfer curves — coupler diagnostic",
) -> Figure:
    """Density-domain transfer curves — coupler diagnostic.

    Four panels in a 2×2 grid:

    - **Top row** (R, G, B sweeps): each panel sweeps one input channel
      and shows D-R, D-G, D-B output as a *family* of curves taken at
      several constant values of the other channels (the "pins"). The
      spread between curves at different pins IS the coupler signature:
      DIR couplers in the film simulation make the output channels
      respond to inputs they otherwise shouldn't, and that cross-talk
      shows up as a visible vertical gap between same-color curves at
      different pins. The center pin (typically 0.5) is drawn at full
      alpha; outer pins fade so the eye reads the family as a textured
      band rather than a tangle of equally-weighted lines.
    - **Bottom right** (neutral sweep): vary R=G=B together. The
      canonical film characteristic curve from datasheets — toe,
      linear segment, shoulder all visible.

    Density convention: D = -log₁₀(output) with the Y axis inverted
    so D=0 (white) sits at top and high D (black) sits at the bottom,
    matching film datasheet conventions.

    Parameters
    ----------
    sweep_x :
        Shape ``(N,)`` — the input encoded values being swept (same
        x-axis for all panels). The caller computes this; typically
        ``np.linspace(0, 1, N)``.
    r_sweep, g_sweep, b_sweep :
        Lists of ``(pin_density, samples, alpha)`` triples. ``samples``
        is shape ``(N, 3)`` of LUT output at the sweep points with the
        non-swept channels pinned. ``pin_density`` is the *output*
        density the off-diagonal curves are expected to start at on the
        Y-axis — the caller picked the actual input-code pins by
        inverting the neutral characteristic curve so the off-diagonals
        enter each panel at ``pin_density``. Used only for the axis
        label. ``alpha`` controls the line transparency for that pin's
        curves.
    neutral_samples :
        Shape ``(N, 3)`` of LUT output along the neutral diagonal
        (R=G=B=sweep_x). Drawn at full alpha in the bottom-right panel.
    title :
        Figure suptitle.
    """
    floor = 1.0e-4

    # Shared Y range across all 4 panels (so D values are visually
    # comparable across the R / G / B / neutral sweeps). Top stays at
    # D=0 (white); bottom sits just past the deepest observed density
    # — no snapping to round numbers, the plot surface is for the data.
    all_densities = []
    for sweep_data in (r_sweep, g_sweep, b_sweep):
        for _, samples, _ in sweep_data:
            d = -np.log10(np.clip(samples, floor, 1.0))
            all_densities.append(d)
    neutral_d = -np.log10(np.clip(neutral_samples, floor, 1.0))
    all_densities.append(neutral_d)
    observed_max = float(max(d.max() for d in all_densities))
    y_max = observed_max * 1.05

    fig, axes_2d = plt.subplots(2, 2, figsize=(13, 10), facecolor=BG,
                                layout="constrained")
    # Materialize as a list — zip() on a flatiter advances it past the
    # zipped count when the other iterable exhausts first, leaving
    # nothing for the neutral panel below.
    axes = list(axes_2d.flat)

    setups = [
        ("R", r_sweep, RED),
        ("G", g_sweep, GREEN),
        ("B", b_sweep, BLUE),
    ]
    for ax, (label, sweep_data, axis_color) in zip(axes[:3], setups):
        ax.set_facecolor(BG)
        # Plot every (pin, samples) family in the panel. The matching
        # channel's center-pin curve sits on top of the stack so it's
        # visually dominant.
        for pin, samples, alpha in sweep_data:
            density = -np.log10(np.clip(samples, floor, 1.0))
            ax.plot(sweep_x, density[:, 0], color=RED, lw=2.0,
                    alpha=alpha, zorder=2 + alpha)
            ax.plot(sweep_x, density[:, 1], color=GREEN, lw=2.0,
                    alpha=alpha, zorder=2 + alpha)
            ax.plot(sweep_x, density[:, 2], color=BLUE, lw=2.0,
                    alpha=alpha, zorder=2 + alpha)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, y_max)
        ax.invert_yaxis()
        pin_densities = sorted({pin for pin, _, _ in sweep_data})
        pins_text = ", ".join(f"{p:g}" for p in pin_densities)
        ax.set_xlabel(
            f"{label} input code   (other channels start at D ≈ {{{pins_text}}})",
            color=FG, fontsize=9,
        )
        ax.set_ylabel("D = -log₁₀(output)", color=FG)
        ax.set_title(f"{label} sweep", color=axis_color, fontsize=PANEL_TITLE_FS, pad=PANEL_TITLE_PAD)
        _setup_2d(ax)
        # Legend uses three full-alpha proxies so the swatch is readable
        # against the multi-alpha lines actually drawn.
        from matplotlib.lines import Line2D
        proxies = [
            Line2D([0], [0], color=RED, lw=2.0, label="D-R"),
            Line2D([0], [0], color=GREEN, lw=2.0, label="D-G"),
            Line2D([0], [0], color=BLUE, lw=2.0, label="D-B"),
        ]
        ax.legend(handles=proxies, facecolor="#1a1a1a", labelcolor=FG,
                  framealpha=0.85, loc="upper right", fontsize=9)

    # Neutral diagonal panel — single full-alpha family, the canonical
    # film D-vs-input curve.
    ax_n = axes[3]
    ax_n.set_facecolor(BG)
    ax_n.plot(sweep_x, neutral_d[:, 0], color=RED, lw=2.2, label="D-R")
    ax_n.plot(sweep_x, neutral_d[:, 1], color=GREEN, lw=2.2, label="D-G")
    ax_n.plot(sweep_x, neutral_d[:, 2], color=BLUE, lw=2.2, label="D-B")
    ax_n.set_xlim(0, 1)
    ax_n.set_ylim(0, y_max)
    ax_n.invert_yaxis()
    ax_n.set_xlabel(
        "neutral (R=G=B) input code   (the canonical D-vs-input curve)",
        color=FG, fontsize=9,
    )
    ax_n.set_ylabel("D = -log₁₀(output)", color=FG)
    ax_n.set_title("neutral (R=G=B) sweep", color=HI, fontsize=PANEL_TITLE_FS, pad=PANEL_TITLE_PAD)
    _setup_2d(ax_n)
    ax_n.legend(facecolor="#1a1a1a", labelcolor=FG, framealpha=0.85,
                loc="upper right", fontsize=9)

    fig.suptitle(title, color=HI, fontsize=SUPTITLE_FS)
    return fig

def oklab_ab_slices(grid_output: np.ndarray, out_cs: str) -> Figure:
    """LUT output projected into OkLab, binned by L* (8 panels)."""
    lab = _to_oklab(grid_output, out_cs)
    L = lab[:, 0]
    colors = np.clip(grid_output, 0.0, 1.0)
    n_bins = 8
    L_edges = np.linspace(L.min(), L.max(), n_bins + 1)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), facecolor=BG, layout="constrained")
    for i, ax in enumerate(axes.flat):
        ax.set_facecolor(BG)
        lo, hi = L_edges[i], L_edges[i + 1]
        mask = (L >= lo) & (L < hi if i < n_bins - 1 else L <= hi)
        if np.any(mask):
            ax.scatter(lab[mask, 1], lab[mask, 2],
                       c=colors[mask], s=8, alpha=0.7, edgecolors="none")
        ax.axhline(0, color="#555555", lw=0.6)
        ax.axvline(0, color="#555555", lw=0.6)
        ax.set_xlim(-0.4, 0.4); ax.set_ylim(-0.4, 0.4); ax.set_aspect("equal")
        ax.set_title(f"L* ∈ [{lo:.2f}, {hi:.2f}]", color=FG, fontsize=10)
        ax.tick_params(colors=FG, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#555555")
    fig.suptitle("OkLab a*–b* slices of the LUT output",
                 color=HI, fontsize=SUPTITLE_FS)
    fig.supxlabel("a* (red–green)", color=FG)
    fig.supylabel("b* (yellow–blue)", color=FG)
    return fig

def hue_twist_oklab(
    grid_input: np.ndarray, grid_output: np.ndarray,
    in_cs: str, out_cs: str,
) -> Figure:
    """Input vs output OkLab hue diagram, binned by input chroma."""
    lab_in = _to_oklab(grid_input, in_cs)
    lab_out = _to_oklab(grid_output, out_cs)
    h_in = np.degrees(np.arctan2(lab_in[:, 2], lab_in[:, 1]))
    h_out = np.degrees(np.arctan2(lab_out[:, 2], lab_out[:, 1]))
    c_in = np.sqrt(lab_in[:, 1] ** 2 + lab_in[:, 2] ** 2)
    c_max = float(c_in.max()) if c_in.size else 0.0
    if c_max <= 0.0:
        c_max = 1.0
    bands = [(0.20, 0.35), (0.35, 0.50), (0.50, 0.70), (0.70, 1.001)]
    band_colors = plt.cm.plasma(np.linspace(0.2, 0.95, len(bands)))
    fig, ax = plt.subplots(figsize=(9, 9), facecolor=BG, layout="constrained")
    ax.set_facecolor(BG)
    _identity_line(ax, lo=-180, hi=180)
    ax.plot([-180, 180], [180, 540], color="#333333", lw=0.6, ls=":")
    ax.plot([-180, 180], [-540, -180], color="#333333", lw=0.6, ls=":")
    for (lo, hi), col in zip(bands, band_colors):
        mask = (c_in >= lo * c_max) & (c_in < hi * c_max)
        if not np.any(mask):
            continue
        ax.scatter(h_in[mask], h_out[mask], color=[col],
                   s=10, alpha=0.55, edgecolors="none",
                   label=f"chroma {lo * c_max:.02f}–{hi * c_max:.02f}")
    ax.set_xlim(-180, 180); ax.set_ylim(-180, 180)
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-180, 181, 60))
    ax.set_xlabel("input hue (OkLab)  [°]", color=FG)
    ax.set_ylabel("output hue (OkLab)  [°]", color=FG)
    ax.set_title("Hue-twist diagram (OkLab)", color=HI, fontsize=SUPTITLE_FS, pad=SUPTITLE_PAD)
    _setup_2d(ax)
    ax.legend(facecolor="#1a1a1a", labelcolor=FG, framealpha=0.9,
              loc="upper left", fontsize=9)
    ax.set_aspect("equal")
    return fig

def oklab_displacement(
    grid_input: np.ndarray, grid_output: np.ndarray,
    in_cs: str, out_cs: str,
) -> Figure:
    """Quiver of OkLab a*-b* displacement at three lightness bands."""
    lab_in = _to_oklab(grid_input, in_cs)
    lab_out = _to_oklab(grid_output, out_cs)
    L_mid = 0.5 * (lab_in[:, 0] + lab_out[:, 0])
    L_lo, L_hi = float(L_mid.min()), float(L_mid.max())
    if L_hi <= L_lo:
        L_hi = L_lo + 1e-6
    edges = np.linspace(L_lo, L_hi, 4)
    band_titles = (
        f"L* ∈ [{edges[0]:.2f}, {edges[1]:.2f}]   (shadows)",
        f"L* ∈ [{edges[1]:.2f}, {edges[2]:.2f}]   (midtones)",
        f"L* ∈ [{edges[2]:.2f}, {edges[3]:.2f}]   (highlights)",
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), facecolor=BG, layout="constrained")
    for i, (ax, title) in enumerate(zip(axes, band_titles)):
        ax.set_facecolor(BG)
        lo, hi = edges[i], edges[i + 1]
        mask = (L_mid >= lo) & (L_mid <= hi)
        if np.any(mask):
            u = lab_out[mask, 1] - lab_in[mask, 1]
            v = lab_out[mask, 2] - lab_in[mask, 2]
            colors = np.clip(grid_output[mask], 0.0, 1.0)
            ax.quiver(lab_in[mask, 1], lab_in[mask, 2], u, v,
                      color=colors, angles="xy", scale_units="xy", scale=1.0,
                      width=0.0035, alpha=0.78)
        ax.axhline(0, color="#555555", lw=0.6)
        ax.axvline(0, color="#555555", lw=0.6)
        ax.set_xlim(-0.4, 0.4); ax.set_ylim(-0.4, 0.4); ax.set_aspect("equal")
        ax.set_title(title, color=FG, fontsize=11)
        ax.set_xlabel("a*  (red–green)", color=FG)
        ax.set_ylabel("b*  (yellow–blue)", color=FG)
        _setup_2d(ax)
    fig.suptitle("OkLab displacement: where the LUT sends each input chromaticity",
                 color=HI, fontsize=SUPTITLE_FS)
    return fig

def chromaticity_1931(
    grid_output: np.ndarray, in_cs: str, out_cs: str,
    *, locus_xy: np.ndarray | None = None,
) -> Figure:
    """CIE 1931 chromaticity with spectral locus, gamut triangles, LUT footprint."""
    from spektrafilm_lut_creator.color_spaces import to_xyz

    xyz = to_xyz(grid_output, out_cs)
    xy = np.asarray(colour.XYZ_to_xy(xyz), dtype=float)
    if locus_xy is None:
        cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
        wavelengths = np.arange(380, 781, 1)
        locus_xyz = np.asarray(cmfs[wavelengths], dtype=float)
        locus_xy = np.asarray(colour.XYZ_to_xy(locus_xyz), dtype=float)
    in_tri = _gamut_triangle_xy(in_cs)
    out_tri = _gamut_triangle_xy(out_cs)

    fig, ax = plt.subplots(figsize=(9, 9), facecolor=BG, layout="constrained")
    ax.set_facecolor(BG)
    ax.plot(locus_xy[:, 0], locus_xy[:, 1], color="#aaaaaa", lw=1.4, zorder=2)
    ax.plot([locus_xy[-1, 0], locus_xy[0, 0]],
            [locus_xy[-1, 1], locus_xy[0, 1]],
            color="#777777", lw=1.0, ls="--", zorder=2)
    ax.plot(in_tri[:, 0], in_tri[:, 1], color="#66ffee", lw=1.4, ls="--",
            alpha=0.85, zorder=3, label=f"input gamut: {in_cs}")
    ax.plot(out_tri[:, 0], out_tri[:, 1], color="#ffee66", lw=1.6, alpha=0.95,
            zorder=3, label=f"output gamut: {out_cs}")
    ax.scatter(xy[:, 0], xy[:, 1], c=np.clip(grid_output, 0.0, 1.0),
               s=8, alpha=0.65, edgecolors="none", zorder=4)
    ax.set_xlim(-0.05, 0.85); ax.set_ylim(-0.05, 0.95)
    ax.set_xlabel("x", color=FG); ax.set_ylabel("y", color=FG)
    ax.set_title("CIE 1931 chromaticity — LUT output footprint",
                 color=HI, fontsize=SUPTITLE_FS, pad=SUPTITLE_PAD)
    _setup_2d(ax)
    ax.legend(facecolor="#1a1a1a", labelcolor=FG, framealpha=0.9,
              loc="upper right", fontsize=9)
    return fig


# ---------------------------------------------------------------------------
# Diagnostic plots specific to model-side tests.
# ---------------------------------------------------------------------------

def planckian_path(
    cct: np.ndarray, xy_out: np.ndarray, locus_xy: np.ndarray, out_cs: str,
) -> Figure:
    """The Planckian / daylight sweep traced on the 1931 chromaticity plot."""
    fig, ax = plt.subplots(figsize=(9, 9), facecolor=BG, layout="constrained")
    ax.set_facecolor(BG)
    ax.plot(locus_xy[:, 0], locus_xy[:, 1], color="#888888", lw=1.2)
    ax.plot([locus_xy[-1, 0], locus_xy[0, 0]],
            [locus_xy[-1, 1], locus_xy[0, 1]],
            color="#666666", lw=1.0, ls="--")
    out_tri = _gamut_triangle_xy(out_cs)
    ax.plot(out_tri[:, 0], out_tri[:, 1], color="#ffee66", lw=1.6, alpha=0.95,
            label=f"output gamut: {out_cs}")
    sc = ax.scatter(xy_out[:, 0], xy_out[:, 1], c=cct, cmap="plasma",
                    s=60, edgecolors="white", linewidths=0.5, zorder=4)
    ax.plot(xy_out[:, 0], xy_out[:, 1], color="#cccccc", lw=0.8, alpha=0.6, zorder=3)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.04)
    cbar.set_label("CCT [K]", color=FG)
    cbar.ax.tick_params(colors=FG)
    cbar.outline.set_edgecolor("#555555")
    ax.set_xlim(-0.05, 0.85); ax.set_ylim(-0.05, 0.95)
    ax.set_xlabel("x", color=FG); ax.set_ylabel("y", color=FG)
    ax.set_title("Planckian / daylight sweep — output chromaticity per CCT",
                 color=HI, fontsize=SUPTITLE_FS, pad=SUPTITLE_PAD)
    _setup_2d(ax)
    ax.legend(facecolor="#1a1a1a", labelcolor=FG, framealpha=0.9,
              loc="upper right", fontsize=9)
    return fig

def dynamic_range_curve(
    stops: np.ndarray,
    output_y: np.ndarray,
    encoded_clip_mask: np.ndarray,
    stats: dict,
    *,
    in_cs: str,
    out_cs: str,
    slope_threshold: float = 0.10,
) -> Figure:
    """Density vs scene-linear log E (stops), the canonical film
    characteristic plot.

    Layers:

    1. The main curve — output density ``D = -log10(Y)`` vs input
       stops above middle gray. Y axis inverted so D=0 (white) sits
       at the top, high D (black) at the bottom — film datasheet
       convention. The actual sampled stops are drawn as explicit
       data-point markers along the curve.
    2. **Midgray exposure readout** (the headline diagnostic): a
       vertical line at stop 0 = the camera's true 18% gray input, a
       horizontal dashed line at the *ideal* output density
       ``-log10(0.18) ≈ 0.745``, and a ringed marker where the LUT
       actually renders that gray. The vertical gap between marker and
       ideal line is the baked-in exposure shift — green when within
       ±0.25 stop, red when it isn't. A correctly-exposed log LUT lands
       the marker on the ideal line.
    3. **90% diffuse-white marker** at stop +2.32 — the other classic
       exposure reference.
    4. **Encoding-clip shading + dashed bars**: warm-tinted shaded
       regions covering the *full extent* of stops the input encoding
       can't represent (clipped to 0 or 1 at the input).
    5. **Toe / shoulder shading**: light gray bands where slope falls
       below the active threshold — the model's compression decisions.

    The x-axis spans the **full input ramp** (every probed stop, including
    the encoding-clipped flats at both ends) and the full output density
    range. The toe / shoulder / encoding-clip shading marks which part of
    that full span the LUT actually renders, so the effective range stays
    legible without cropping the curve.

    A small text box prints the summary numbers (midgray offset,
    encoded range, active range, collapsed toe/shoulder) so the figure
    is self-contained.
    """
    stops = np.asarray(stops, dtype=float).ravel()
    output_y = np.asarray(output_y, dtype=float).ravel()
    encoded_clip_mask = np.asarray(encoded_clip_mask, dtype=bool).ravel()
    density = -np.log10(np.clip(output_y, 1e-4, None))
    IDEAL_MID_D = -np.log10(0.18)  # ≈ 0.745, output density of a faithful 18% gray
    WHITE_STOPS = float(np.log2(0.90 / 0.18))  # 90% diffuse white ≈ +2.32 EV

    fig, ax = plt.subplots(figsize=(12, 7), facecolor=BG, layout="constrained")
    ax.set_facecolor(BG)

    # Toe / shoulder shading inside the encoded range, plus encoding-clip
    # shading outside it so the *full extent* of clipped stops is visible
    # at a glance (the "x" markers alone made the clip range easy to miss).
    encoded_ok = ~encoded_clip_mask
    enc_idx = np.where(encoded_ok)[0]
    if enc_idx.size >= 2:
        enc_lo_x, enc_hi_x = stops[enc_idx[0]], stops[enc_idx[-1]]
        toe_n = int(round(stats.get("toe_collapsed_stops", 0.0) /
                          max((stops[1] - stops[0]), 1e-9)))
        shoulder_n = int(round(stats.get("shoulder_collapsed_stops", 0.0) /
                               max((stops[1] - stops[0]), 1e-9)))
        if toe_n > 0:
            toe_hi = stops[min(enc_idx[0] + toe_n, len(stops) - 1)]
            ax.axvspan(enc_lo_x, toe_hi, color="#666666", alpha=0.18,
                       label=f"toe collapse: {stats['toe_collapsed_stops']:.1f} stops")
        if shoulder_n > 0:
            shoulder_lo = stops[max(enc_idx[-1] - shoulder_n, 0)]
            ax.axvspan(shoulder_lo, enc_hi_x, color="#888866", alpha=0.18,
                       label=f"shoulder collapse: {stats['shoulder_collapsed_stops']:.1f} stops")

        # Encoding-clip shading: full vertical extent of the clipped
        # stops, one span on each side that actually clips. Each
        # carries its own count so the legend describes the situation
        # asymmetric SDR inputs typically produce (no shadow clip in
        # the ramp range, large highlight clip).
        if enc_lo_x > stops[0]:
            lo_stops = stops[enc_idx[0]] - stops[0]
            ax.axvspan(stops[0], enc_lo_x, color=WARN, alpha=0.12,
                       label=f"encoding clip: {lo_stops:.1f} stops (shadows)")
        if enc_hi_x < stops[-1]:
            hi_stops = stops[-1] - stops[enc_idx[-1]]
            ax.axvspan(enc_hi_x, stops[-1], color=WARN, alpha=0.12,
                       label=f"encoding clip: {hi_stops:.1f} stops (highlights)")

        # Boundary markers — kept as a precise visual anchor at the
        # exact clip threshold (the shading shows extent; the dashed
        # line shows where the encoding starts to clip).
        ax.axvline(enc_lo_x, color=WARN, lw=1.0, ls="--", alpha=0.7)
        ax.axvline(enc_hi_x, color=WARN, lw=1.0, ls="--", alpha=0.7)

    # Main curve, with the actual sampled stops drawn as explicit data
    # points (subsampled so the markers stay legible at n≈257).
    markevery = max(1, stops.size // 48)
    ax.plot(stops, density, color=HI, lw=2.0, zorder=5,
            marker="o", ms=3.5, markevery=markevery,
            markerfacecolor=HI, markeredgecolor=BG, markeredgewidth=0.4,
            label="LUT samples")
    # Mark where input is clipped — render those segments dimmer.
    if np.any(encoded_clip_mask):
        ax.plot(stops[encoded_clip_mask], density[encoded_clip_mask],
                color=DIM, lw=1.6, marker="x", ms=4, ls="None", zorder=4,
                label="encoded-clipped stops")

    # ---- Midgray exposure readout (the headline diagnostic) ----------
    midgray_offset = float(stats.get("midgray_offset_stops", 0.0))
    midgray_y = float(stats.get("midgray_output_y", 0.18))
    midgray_D = -np.log10(max(midgray_y, 1e-4))
    on_target = abs(midgray_offset) <= 0.25
    mid_color = GREEN if on_target else RED
    # Ideal output density for a faithful 18% gray (where the marker
    # *should* sit).
    ax.axhline(IDEAL_MID_D, color=GREEN, lw=1.0, ls=":", alpha=0.7,
               label="ideal 18% out (D=0.74)")
    # The camera's true 18% gray input.
    ax.axvline(0, color="#777777", lw=0.9, alpha=0.7)
    ax.text(0.04, 0.02, "camera 18% gray (input)", color=DIM, fontsize=9,
            transform=ax.get_xaxis_transform())
    # Where it actually renders, ringed so it reads against the curve.
    ax.plot([0.0], [midgray_D], marker="o", ms=11, markerfacecolor="none",
            markeredgecolor=mid_color, markeredgewidth=2.2, zorder=7,
            label=f"rendered 18% → {midgray_offset:+.2f} stops")
    # A short connector showing the gap from ideal when it's off.
    if not on_target:
        ax.annotate("", xy=(0.0, midgray_D), xytext=(0.0, IDEAL_MID_D),
                    arrowprops=dict(arrowstyle="->", color=mid_color, lw=1.6),
                    zorder=6)

    # 90% diffuse-white reference.
    ax.axvline(WHITE_STOPS, color="#555555", lw=0.8, ls="--", alpha=0.5)
    ax.text(WHITE_STOPS + 0.05, 0.02, "90% white", color=DIM, fontsize=9,
            transform=ax.get_xaxis_transform())

    ax.set_xlabel("scene-linear stops above camera middle gray  (log₂(linear / 0.18))",
                  color=FG)
    ax.set_ylabel("D = -log₁₀(output Y)   (film density convention)", color=FG)
    ax.set_ylim(0, float(density.max()) * 1.05)
    ax.invert_yaxis()
    # Show the *full* input ramp and output range — every stop the ramp
    # probed, including the encoding-clipped flats at both ends. The
    # toe/shoulder/encoding-clip shading (above) marks which part of this
    # full span the LUT actually renders, so the effective range is still
    # legible without cropping the curve.
    ax.set_xlim(stops[0], stops[-1])
    _setup_2d(ax)

    # Headline summary in the upper-left of the plot — film-datasheet
    # vibe ("Effective: 5.3 stops"). Midgray offset leads, since it's the
    # exposure-correctness readout.
    mid_tag = "OK" if on_target else "OFF"
    summary_lines = [
        f"midgray offset:         {midgray_offset:+.2f} stops [{mid_tag}]",
        f"input encoding range:   {stats.get('encoded_range_stops', 0):.1f} stops",
        f"active rendering range: {stats.get('active_range_stops', 0):.1f} stops",
        f"  toe collapsed:        {stats.get('toe_collapsed_stops', 0):.1f} stops",
        f"  shoulder collapsed:   {stats.get('shoulder_collapsed_stops', 0):.1f} stops",
        f"slope threshold:        {slope_threshold} D/stop",
    ]
    ax.text(0.02, 0.98, "\n".join(summary_lines),
            transform=ax.transAxes, color=FG, fontsize=10, family="monospace",
            va="top", ha="left",
            bbox=dict(facecolor="#1a1a1a", edgecolor=mid_color,
                      alpha=0.92, boxstyle="round,pad=0.5"))

    ax.set_title(
        f"Dynamic range — {in_cs} → {out_cs}",
        color=HI, fontsize=SUPTITLE_FS, pad=SUPTITLE_PAD,
    )
    ax.legend(facecolor="#1a1a1a", labelcolor=FG, framealpha=0.85,
              loc="lower right", fontsize=9)
    return fig

def spectral_locus_envelope(
    out_cs: str,
    edges_output_xy: list[np.ndarray],
) -> Figure:
    """Output chromaticity of every input cube edge overlaid on the spectral locus.

    The 12 cube edges form the canonical hue circle in the input
    space; their output chromaticity envelope is the model's
    reachable gamut at maximum saturation. Compared against the
    spectral locus and the output primaries triangle, this reveals
    how much of the visible-color "rim" the model preserves.
    """
    cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    wavelengths = np.arange(380, 781, 1)
    locus_xyz = np.asarray(cmfs[wavelengths], dtype=float)
    locus_xy = np.asarray(colour.XYZ_to_xy(locus_xyz), dtype=float)
    out_tri = _gamut_triangle_xy(out_cs)

    fig, ax = plt.subplots(figsize=(9, 9), facecolor=BG, layout="constrained")
    ax.set_facecolor(BG)
    ax.plot(locus_xy[:, 0], locus_xy[:, 1], color="#aaaaaa", lw=1.4, label="spectral locus")
    ax.plot([locus_xy[-1, 0], locus_xy[0, 0]],
            [locus_xy[-1, 1], locus_xy[0, 1]],
            color="#777777", lw=1.0, ls="--")
    ax.plot(out_tri[:, 0], out_tri[:, 1], color="#ffee66", lw=1.6, alpha=0.95,
            label=f"output gamut: {out_cs}")
    cmap = plt.cm.hsv(np.linspace(0, 1, len(edges_output_xy)))
    for col, xy in zip(cmap, edges_output_xy):
        ax.plot(xy[:, 0], xy[:, 1], color=col, lw=1.2, alpha=0.85)
    ax.set_xlim(-0.05, 0.85); ax.set_ylim(-0.05, 0.95)
    ax.set_xlabel("x", color=FG); ax.set_ylabel("y", color=FG)
    ax.set_title("Spectral-locus envelope — cube edges through the LUT",
                 color=HI, fontsize=SUPTITLE_FS, pad=SUPTITLE_PAD)
    _setup_2d(ax)
    ax.legend(facecolor="#1a1a1a", labelcolor=FG, framealpha=0.9,
              loc="upper right", fontsize=9)
    return fig


# ---------------------------------------------------------------------------
# Noise propagation.
# ---------------------------------------------------------------------------
