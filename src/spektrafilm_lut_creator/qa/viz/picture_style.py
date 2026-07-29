"""Viz panels for the picture-style QA tests + a few shared helpers."""

from __future__ import annotations

import colour
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.ticker import FormatStrFormatter

from spektrafilm.utils.gamut_compression import OutputGamutCompressSpec
from spektrafilm_lut_creator.qa.viz._base import (
    BG,
    DIM,
    FG,
    HI,
    PANEL_TITLE_FS,
    PANEL_TITLE_PAD,
    SUPTITLE_FS,
    _format_output_gamut_compress,
    _setup_2d,
)


def oklab_gamut_slice_outline(
    cs_name: str,
    *,
    L: float,
    n_hues: int = 181,
    n_iter: int = 18,
) -> tuple[np.ndarray, np.ndarray]:
    """Closed OkLab gamut-slice outline at constant ``L``.

    The noise-sensitivity panels live on a fixed OkLab luminance slice,
    so the relevant gamut reference is the in-gamut chroma envelope on
    that slice, not the RGB-primary triangle projected from xy. The
    latter can explode the a*b* axis scale for wide or log-gamut input
    spaces and visually hide the actual field near the origin.
    """
    from spektrafilm_lut_creator.color_spaces import get as get_cs

    entry = get_cs(cs_name)
    primaries = colour.RGB_COLOURSPACES[entry.primaries]
    hue = np.linspace(0.0, 2.0 * np.pi, n_hues, endpoint=False)
    lo = np.zeros_like(hue)
    hi = np.full_like(hue, 0.45)

    def in_gamut(chroma: np.ndarray) -> np.ndarray:
        pts = np.stack(
            [
                np.full_like(chroma, L),
                chroma * np.cos(hue),
                chroma * np.sin(hue),
            ],
            axis=-1,
        )
        xyz = np.asarray(colour.Oklab_to_XYZ(pts), dtype=float)
        rgb_linear = np.asarray(
            colour.XYZ_to_RGB(
                xyz,
                colourspace=primaries.name,
                apply_cctf_encoding=False,
                illuminant=np.asarray(primaries.whitepoint, dtype=float),
            ),
            dtype=float,
        )
        return np.all((rgb_linear >= 0.0) & (rgb_linear <= 1.0), axis=-1)

    for _ in range(3):
        grow = in_gamut(hi)
        if not np.any(grow):
            break
        hi = np.where(grow, hi * 1.5, hi)

    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        ok = in_gamut(mid)
        lo = np.where(ok, mid, lo)
        hi = np.where(ok, hi, mid)

    outline = np.stack(
        [
            np.full_like(lo, L),
            lo * np.cos(hue),
            lo * np.sin(hue),
        ],
        axis=-1,
    )
    white = np.array([L, 0.0, 0.0], dtype=float)
    return np.vstack([outline, outline[:1]]), white


def _oklab_to_encoded_rgb(oklab: np.ndarray, cs_name: str) -> np.ndarray:
    """OkLab points to encoded RGB in ``cs_name`` for display."""
    from spektrafilm_lut_creator.color_spaces import from_xyz

    xyz = np.asarray(colour.Oklab_to_XYZ(np.asarray(oklab, dtype=float)), dtype=float)
    return np.clip(np.asarray(from_xyz(xyz, cs_name), dtype=float), 0.0, 1.0)


def _encoded_rgb_to_display_rgb(rgb: np.ndarray, cs_name: str) -> np.ndarray:
    """Encoded RGB in ``cs_name`` converted to sRGB for on-screen display."""
    from spektrafilm_lut_creator.color_spaces import from_xyz, to_xyz_qa

    xyz = np.asarray(to_xyz_qa(np.asarray(rgb, dtype=float), cs_name), dtype=float)
    return np.clip(np.asarray(from_xyz(xyz, "sRGB"), dtype=float), 0.0, 1.0)


def rg_plane_slices(
    table: np.ndarray,
    n: int,
    out_cs: str,
    *,
    n_slices: int = 9,
) -> Figure:
    """R-G plane slices through the cube at varying B-input values,
    rendered as **sRGB display images** (hard-clipped).

    The cube table is encoded in the bundle's output color space. A
    naive imshow would treat those values as if they were sRGB-encoded,
    which produces visibly wrong colors for any non-sRGB output
    (Rec.2020, DCI-P3, P3-D65 PQ, …). Each slice is decoded to linear
    in the output primaries, chromatically adapted to sRGB primaries,
    sRGB-encoded, and hard-clipped — so what's on screen is the LUT's
    R-G response at that B as it would appear on an sRGB display.
    """
    from spektrafilm_lut_creator.color_spaces import decode_cctf, output_midgray_gain
    from spektrafilm_lut_creator.color_spaces import get as get_cs

    # Fixed 3x3 grid; default 9 slices fills it exactly. If the cube
    # resolution is too small for 9 slices we use as many as fit and
    # leave the trailing axes blank.
    n_slices = min(n_slices, n)
    grid_cols = 3
    grid_rows = int(np.ceil(n_slices / grid_cols))
    indices = np.linspace(0, n - 1, n_slices, dtype=int)
    out_entry = get_cs(out_cs)
    # Bring the cube's linear values back to reflectance scale so HDR
    # outputs (where decode_cctf returns nits) don't blow past [0,1]
    # and clip uniformly to white after the sRGB display conversion.
    out_gain = output_midgray_gain(out_cs)

    fig, axes_2d = plt.subplots(
        grid_rows,
        grid_cols,
        figsize=(2.4 * grid_cols, 2.6 * grid_rows + 0.4),
        facecolor=BG,
        layout="constrained",
    )
    axes = np.atleast_1d(axes_2d).reshape(grid_rows, grid_cols).flatten()
    for i, ax in enumerate(axes):
        if i >= n_slices:
            ax.axis("off")
            continue
        idx = indices[i]
        slice_encoded = np.asarray(table[idx, :, :, :], dtype=float)
        slice_linear = decode_cctf(slice_encoded, out_cs) / out_gain
        srgb_linear = np.asarray(
            colour.RGB_to_RGB(
                slice_linear,
                out_entry.primaries,
                "sRGB",
                chromatic_adaptation_transform="CAT16",
            ),
            dtype=float,
        )
        srgb_encoded = np.asarray(
            colour.cctf_encoding(np.clip(srgb_linear, 0.0, 1.0), function="sRGB"),
            dtype=float,
        )
        ax.imshow(
            np.clip(srgb_encoded, 0.0, 1.0),
            origin="lower",
            extent=(0, 1, 0, 1),
            interpolation="bilinear",
        )
        b_val = idx / (n - 1)
        ax.set_title(f"B = {b_val:.2f}", color=FG, fontsize=10)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.tick_params(colors=FG, length=2)
        for spine in ax.spines.values():
            spine.set_color("#555555")
        # Only the leftmost column gets a G label; only the bottom
        # row gets an R label — keeps the grid uncluttered.
        row, col = i // grid_cols, i % grid_cols
        if col == 0:
            ax.set_ylabel("G in", color=FG, fontsize=8)
        if row == grid_rows - 1 or (i + grid_cols) >= n_slices:
            ax.set_xlabel("R in", color=FG, fontsize=8)
    fig.suptitle(
        f"R-G cube slices at varying B   (output {out_cs} → sRGB display, hard-clipped)",
        color=HI,
        fontsize=SUPTITLE_FS,
    )
    return fig


def gamut_edge_stress(
    panels: list[tuple[str, np.ndarray, dict]],
    *,
    in_cs: str,
    out_cs: str,
    gamut_compress: OutputGamutCompressSpec | None = None,
) -> Figure:
    """Granger-style RGB stress chart panels.

    Each panel is a vertical linear-RGB gradient (white at the top,
    saturated RGB-cube edge in the middle as the hue cycles across
    columns, black at the bottom) generated in one target RGB color
    space, CAT-adapted into the bundle's input encoding, pushed
    through the LUT, and displayed in sRGB (hard-clipped). Pixels
    whose target-space color does not fit the bundle's input encoding
    are left black.

    ``panels`` is ``[(target_cs_name, srgb_image (H,W,3), stats_dict)]``.
    """
    # Panel aspect is 3:1 (width:height) to match Mononodes-style charts —
    # the gradient image itself is built at 3:1 too, so aspect="equal"
    # honors that and avoids the very-long banner shape.
    n_panels = len(panels)
    panel_width = 9.0
    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(panel_width, (panel_width / 3.0) * n_panels + 0.8),
        facecolor=BG,
        layout="constrained",
    )
    axes = np.atleast_1d(axes)
    for ax, (cs_name, img, stats) in zip(axes, panels):
        ax.imshow(img, aspect="equal", origin="upper", interpolation="nearest")
        oog_sat = stats.get("oog_fraction_saturated_row", 0.0)
        ax.set_title(
            f"target: {cs_name}   ·   saturated-row OOG vs {in_cs}: {oog_sat:.1%}",
            color=FG,
            fontsize=11,
            pad=4,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#555555")
    # The per-column gradient construction (white → saturated edge →
    # black, OOG pixels black) is documented in the QA report; keeping
    # the suptitle to one line avoids stomping on the top panel.
    title = f"Gamut-edge stress test — {in_cs} → {out_cs}"
    if gamut_compress is not None:
        title += "\n" + _format_output_gamut_compress(gamut_compress)
    fig.suptitle(title, color=HI, fontsize=SUPTITLE_FS)
    return fig


def _noise_ellipse_panel_extent(
    field: dict,
    *,
    in_cs: str,
    out_cs: str,
    L_slice: float,
) -> float:
    """Common a*b* extent for one noise-ellipse slice."""
    out_tri_oklab, _ = oklab_gamut_slice_outline(out_cs, L=L_slice)
    in_tri_oklab, _ = oklab_gamut_slice_outline(in_cs, L=L_slice)
    input_lab = np.asarray(field["input_oklab"], dtype=float)
    output_lab = np.asarray(field["output_oklab"], dtype=float)
    return (
        max(
            float(np.abs(out_tri_oklab[:, 1:]).max(initial=0.0)),
            float(np.abs(in_tri_oklab[:, 1:]).max(initial=0.0)),
            float(np.abs(input_lab[:, 1:]).max(initial=0.0)),
            float(np.abs(output_lab[:, 1:]).max(initial=0.0)),
            0.30,
        )
        * 1.10
    )


def _draw_noise_ellipse_panel(
    ax,
    *,
    field: dict,
    in_cs: str,
    out_cs: str,
    L_slice: float,
    sigma_in_encoded: float,
    ellipse_display_scale: float,
    extent: float,
    title: str,
    show_legend: bool,
    show_ylabel: bool,
) -> None:
    """Render one fixed-L OkLab noise-ellipse panel onto ``ax``."""
    from matplotlib.patches import Ellipse

    out_tri_oklab, out_white_oklab = oklab_gamut_slice_outline(out_cs, L=L_slice)
    in_tri_oklab, _ = oklab_gamut_slice_outline(in_cs, L=L_slice)
    input_lab = np.asarray(field["input_oklab"], dtype=float)
    output_lab = np.asarray(field["output_oklab"], dtype=float)
    output_enc = np.clip(np.asarray(field["output_encoded"], dtype=float), 0.0, 1.0)
    cov_oklab = np.asarray(field["cov_oklab"], dtype=float)
    sigma1 = np.asarray(field["sigma1"], dtype=float)

    _setup_2d(ax)
    ax.set_aspect("equal")
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_xlabel("OkLab a*", color=FG)
    ax.set_ylabel("OkLab b*" if show_ylabel else "", color=FG)
    ax.axhline(0.0, color="#444444", lw=0.7, zorder=0)
    ax.axvline(0.0, color="#444444", lw=0.7, zorder=0)

    ax.plot(
        in_tri_oklab[:, 1],
        in_tri_oklab[:, 2],
        color="#5599ff",
        lw=1.0,
        alpha=0.55,
        ls="--",
        label=f"{in_cs} gamut (input)" if show_legend else None,
    )
    rim_segments = np.stack(
        [out_tri_oklab[:-1, 1:3], out_tri_oklab[1:, 1:3]],
        axis=1,
    )
    rim_midpoints = 0.5 * (out_tri_oklab[:-1] + out_tri_oklab[1:])
    rim_colors = _oklab_to_encoded_rgb(rim_midpoints, out_cs)
    ax.plot(
        out_tri_oklab[:, 1],
        out_tri_oklab[:, 2],
        color="#111111",
        lw=3.0,
        alpha=0.45,
        zorder=2.75,
    )
    ax.add_collection(
        LineCollection(
            rim_segments,
            colors=rim_colors,
            linewidths=2.0,
            alpha=0.95,
            zorder=2.8,
            capstyle="round",
            joinstyle="round",
        )
    )
    if show_legend:
        ax.plot(
            [], [], color="#ffee66", lw=2.0, alpha=0.9, label=f"{out_cs} gamut (output)"
        )
    ax.plot(
        out_white_oklab[1],
        out_white_oklab[2],
        "D",
        color=DIM,
        markersize=7,
        markeredgecolor=FG,
        markeredgewidth=0.8,
        label=f"{out_cs} white" if show_legend else None,
        zorder=4,
    )

    if sigma1.size > 0 and np.isfinite(sigma1).any():
        s1_max = float(np.nanpercentile(sigma1, 98))
    else:
        s1_max = 1.0
    s1_max = max(s1_max, 1e-6)
    cmap = plt.get_cmap("magma")

    for (a_in, b_in), (a_out, b_out) in zip(input_lab[:, 1:], output_lab[:, 1:]):
        ax.plot(
            [a_in, a_out], [b_in, b_out], color="#888888", lw=0.5, alpha=0.55, zorder=2
        )

    ax.scatter(
        input_lab[:, 1],
        input_lab[:, 2],
        c="#666666",
        s=6,
        alpha=0.6,
        edgecolors="none",
        zorder=2.5,
        label="pre-LUT" if show_legend else None,
    )

    cov_ab = cov_oklab[:, 1:, 1:]
    scale = float(ellipse_display_scale)
    for k in range(output_lab.shape[0]):
        c = cov_ab[k]
        try:
            evals, evecs = np.linalg.eigh(c)
        except np.linalg.LinAlgError:
            continue
        evals = np.clip(evals, 0.0, None)
        w = 2.0 * np.sqrt(evals[1]) * 2.0 * scale
        h = 2.0 * np.sqrt(evals[0]) * 2.0 * scale
        angle = np.degrees(np.arctan2(evecs[1, 1], evecs[0, 1]))
        face = tuple(output_enc[k])
        edge_c = cmap(min(sigma1[k] / s1_max, 1.0))
        ax.add_patch(
            Ellipse(
                xy=(output_lab[k, 1], output_lab[k, 2]),
                width=w,
                height=h,
                angle=angle,
                facecolor=face,
                alpha=0.90,
                edgecolor=edge_c,
                linewidth=1.0,
                zorder=3,
            )
        )

    ax.set_title(title, color=HI, fontsize=PANEL_TITLE_FS, pad=PANEL_TITLE_PAD)
    if show_legend:
        ax.legend(
            facecolor="#1a1a1a",
            edgecolor="#555555",
            labelcolor=FG,
            framealpha=0.92,
            loc="upper right",
            fontsize=8,
        )


def noise_sensitivity(
    *,
    field: dict,
    heatmap: dict,
    rosette: dict,
    in_cs: str,
    out_cs: str,
    L_slice: float,
    sigma_in_encoded: float,
    ellipse_display_scale: float,
) -> Figure:  # noqa: PLR0915
    """Three-panel "LUT noise sensitivity" figure.

    Left: noise ellipses on the OkLab a*b* plane at a fixed input
    luminance — each ellipse is the post-LUT 2σ noise distribution at
    that input chromaticity (scaled by ``ellipse_display_scale`` for
    visibility), centered on the post-LUT chromaticity, filled with the
    post-LUT color, with a hairline back to the pre-LUT position so hue
    rotation is visible too.

    Center: σ₁(J) heatmap over the same a*b* slice — the worst-case
    noise-gain scalar at every chromaticity. Hot spots are where the
    LUT amplifies the most noise; eccentric ellipses in those spots in
    the left panel give the *direction* of that amplification.

    Right: hue rosette — polar plot of σ_L (luminance noise gain) and
    σ_ab (chroma noise gain) at one chroma ring, vs input hue. One-look
    summary of "is the LUT noisier on warm or cool hues?"
    """
    fig = plt.figure(figsize=(18, 7), facecolor=BG, layout="constrained")
    gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 0.06, 0.9])
    ax_ell = fig.add_subplot(gs[0, 0], facecolor=BG)
    ax_heat = fig.add_subplot(gs[0, 1], facecolor=BG)
    cax_heat = fig.add_subplot(gs[0, 2], facecolor=BG)
    ax_rose = fig.add_subplot(gs[0, 3], projection="polar", facecolor=BG)

    extent = _noise_ellipse_panel_extent(
        field,
        in_cs=in_cs,
        out_cs=out_cs,
        L_slice=L_slice,
    )
    # output_lab = np.asarray(field["output_oklab"], dtype=float)
    sigma1 = np.asarray(field["sigma1"], dtype=float)

    if sigma1.size > 0 and np.isfinite(sigma1).any():
        s1_max = float(np.nanpercentile(sigma1, 98))
    else:
        s1_max = 1.0
    s1_max = max(s1_max, 1e-6)
    _draw_noise_ellipse_panel(
        ax_ell,
        field=field,
        in_cs=in_cs,
        out_cs=out_cs,
        L_slice=L_slice,
        sigma_in_encoded=sigma_in_encoded,
        ellipse_display_scale=ellipse_display_scale,
        extent=extent,
        title=(
            f"Noise ellipses on OkLab a*b*\n"
            f"L*={L_slice:.2f}, σ_in={sigma_in_encoded:g}, "
            f"drawn at {ellipse_display_scale:g}× actual size"
        ),
        show_legend=True,
        show_ylabel=True,
    )

    out_tri_oklab, _ = oklab_gamut_slice_outline(out_cs, L=L_slice)
    in_tri_oklab, _ = oklab_gamut_slice_outline(in_cs, L=L_slice)

    # ---- Panel 2: σ₁(J) heatmap ------------------------------------------
    _setup_2d(ax_heat)
    ax_heat.set_aspect("equal")
    ax_heat.set_xlim(-extent, extent)
    ax_heat.set_ylim(-extent, extent)
    ax_heat.set_xlabel("OkLab a*", color=FG)
    ax_heat.set_ylabel("OkLab b*", color=FG)

    aa = np.asarray(heatmap["aa"], dtype=float)
    bb = np.asarray(heatmap["bb"], dtype=float)
    s1g = np.asarray(heatmap["sigma1_grid"], dtype=float)
    pcm = ax_heat.pcolormesh(
        aa,
        bb,
        s1g,
        cmap="magma",
        vmin=0.0,
        vmax=s1_max,
        shading="auto",
        zorder=1,
    )
    cbar = fig.colorbar(pcm, cax=cax_heat)
    cbar.set_label("σ₁(J)  —  worst-case noise gain", color=FG)
    cbar.ax.set_facecolor(BG)
    cbar.ax.tick_params(colors=FG)
    cbar.outline.set_edgecolor("#555555")
    # Heatmap axes are *input* chromaticity, so the load-bearing
    # reference is the input gamut. Output gamut is added as a thin
    # dashed line so users see where the LUT compresses to.
    ax_heat.plot(
        in_tri_oklab[:, 1],
        in_tri_oklab[:, 2],
        color="#5599ff",
        lw=1.2,
        alpha=0.85,
        zorder=2.5,
        label=f"{in_cs} gamut (input)",
    )
    ax_heat.plot(
        out_tri_oklab[:, 1],
        out_tri_oklab[:, 2],
        color="#ffee66",
        lw=1.0,
        alpha=0.75,
        ls="--",
        zorder=2.4,
        label=f"{out_cs} gamut (output)",
    )
    ax_heat.legend(
        facecolor="#1a1a1a",
        edgecolor="#555555",
        labelcolor=FG,
        framealpha=0.92,
        loc="upper right",
        fontsize=8,
    )
    ax_heat.set_title(
        f"σ₁(J)  —  worst-case noise gain\nacross {in_cs} chromaticity",
        color=HI,
        fontsize=PANEL_TITLE_FS,
        pad=PANEL_TITLE_PAD,
    )

    # ---- Panel 3: hue rosette --------------------------------------------
    ax_rose.set_facecolor(BG)
    ax_rose.tick_params(colors=FG, pad=2, labelsize=8)
    ax_rose.grid(True, alpha=0.18, color=HI)
    for spine in ax_rose.spines.values():
        spine.set_color("#555555")
    hue_rad = np.radians(np.asarray(rosette["hue_deg"], dtype=float))
    sL = np.asarray(rosette["sigma_L"], dtype=float)
    sab = np.asarray(rosette["sigma_ab"], dtype=float)
    ring_colors = _encoded_rgb_to_display_rgb(
        np.asarray(rosette.get("output_encoded", [[0.0, 0.0, 0.0]]), dtype=float),
        out_cs,
    )
    hue_closed = np.append(hue_rad, hue_rad[:1])
    sL_closed = np.append(sL, sL[:1])
    sab_closed = np.append(sab, sab[:1])
    rose_peak = max(
        float(np.nanmax(sL)) if sL.size else 0.0,
        float(np.nanmax(sab)) if sab.size else 0.0,
        1e-6,
    )

    def _nice_tick_step(limit: float, n_target: int = 6) -> float:
        rough = max(limit / max(n_target, 1), 1e-6)
        scale = 10.0 ** np.floor(np.log10(rough))
        mantissa = rough / scale
        if mantissa <= 1.0:
            nice = 1.0
        elif mantissa <= 2.0:
            nice = 2.0
        elif mantissa <= 5.0:
            nice = 5.0
        else:
            nice = 10.0
        return nice * scale

    rose_tick_step = _nice_tick_step(rose_peak)
    rose_inner = np.ceil(rose_peak / rose_tick_step) * rose_tick_step
    rose_outer = rose_inner + 0.70 * rose_tick_step
    rose_ticks = np.arange(
        rose_tick_step, rose_inner + 0.5 * rose_tick_step, rose_tick_step
    )
    if hue_rad.size > 1:
        prev_hue = np.roll(hue_rad, 1)
        next_hue = np.roll(hue_rad, -1)
        prev_hue[0] -= 2.0 * np.pi
        next_hue[-1] += 2.0 * np.pi
        rim_theta = np.mod(0.5 * (prev_hue + hue_rad), 2.0 * np.pi)
        rim_width = 0.5 * (next_hue - prev_hue)
    else:
        rim_theta = np.array([0.0])
        rim_width = np.array([2.0 * np.pi])
    ax_rose.bar(
        rim_theta,
        np.full_like(rim_theta, rose_outer - rose_inner),
        width=rim_width,
        bottom=np.full_like(rim_theta, rose_inner),
        color=ring_colors,
        edgecolor="none",
        alpha=1.0,
        align="edge",
        zorder=0.2,
    )
    ax_rose.set_yticks(rose_ticks)
    ax_rose.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax_rose.set_ylim(0.0, rose_outer)
    ax_rose.plot(
        hue_closed,
        sL_closed,
        color="#88ccff",
        lw=1.6,
        alpha=0.95,
        label="σ_L  (luminance)",
    )
    ax_rose.plot(
        hue_closed,
        sab_closed,
        color="#ff88aa",
        lw=1.6,
        alpha=0.95,
        label="σ_ab (chroma)",
    )
    ax_rose.fill(hue_closed, sab_closed, color="#ff88aa", alpha=0.12)
    ax_rose.fill(hue_closed, sL_closed, color="#88ccff", alpha=0.12)
    ax_rose.set_title(
        "Per-hue noise gain  (at mid-chroma ring)",
        color=HI,
        fontsize=PANEL_TITLE_FS,
        pad=PANEL_TITLE_PAD,
    )
    ax_rose.legend(
        facecolor="#1a1a1a",
        edgecolor="#555555",
        labelcolor=FG,
        framealpha=0.92,
        loc="lower left",
        bbox_to_anchor=(-0.18, -0.10),
        fontsize=8,
    )

    fig.suptitle(
        f"LUT noise sensitivity  —  {in_cs} → {out_cs}",
        color=HI,
        fontsize=SUPTITLE_FS,
    )
    return fig


def noise_gradient(
    *,
    clean_srgb: np.ndarray,
    noisy_srgb: np.ndarray,
    in_cs: str,
    out_cs: str,
    sigma_in_encoded: float,
    mid_L: float,
    rng_seed: int,
) -> Figure:
    """Two-panel visual noise diagnostic on a continuous OkLab gradient."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.6), facecolor=BG)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.84, bottom=0.12, wspace=0.14)
    panels = (
        ("clean LUT output", clean_srgb),
        (f"same gradient with σ_in={sigma_in_encoded:g} noise", noisy_srgb),
    )
    for ax, (title, image) in zip(axes, panels):
        ax.set_facecolor(BG)
        ax.imshow(np.clip(image, 0.0, 1.0), origin="lower", interpolation="nearest")
        ax.set_title(title, color=HI, fontsize=PANEL_TITLE_FS, pad=PANEL_TITLE_PAD)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#555555")

    fig.suptitle(
        f"Noise gradient  —  {in_cs} → {out_cs}",
        color=HI,
        fontsize=SUPTITLE_FS,
    )
    fig.text(
        0.5,
        0.085,
        f"horizontal: hue cycle in OkLab   ·   vertical: black → saturated → white   ·   "
        f"peak saturation at L*={mid_L:.2f} (18% gray)   ·   RNG seed={rng_seed}",
        color=FG,
        fontsize=9,
        ha="center",
        va="center",
        alpha=0.88,
    )
    return fig
