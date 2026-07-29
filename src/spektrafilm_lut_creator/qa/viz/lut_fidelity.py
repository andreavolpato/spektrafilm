"""Viz panels for the LUT-fidelity QA tests."""

from __future__ import annotations

import colour
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from spektrafilm_lut_creator.qa.viz._base import (
    BG,
    BLUE,
    FG,
    FOOTER_BAND_FRAC,
    GREEN,
    HEADER_BAND_FRAC,
    HI,
    PANEL_TITLE_FS,
    PANEL_TITLE_PAD,
    RED,
    SUPTITLE_FS,
    SUPTITLE_PAD,
    WARN,
    _fill_3d,
    _identity_line,
    _setup_2d,
    _setup_3d,
    _to_oklab,
)


def cube_sculpture(
    grid_input: np.ndarray,
    grid_output: np.ndarray,
    *,
    color_by: np.ndarray | None = None,
    color_label: str = "",
    cmap: str = "magma",
    title: str = "LUT cube — cells colored by output RGB",
) -> Figure:
    """3D scatter of the cube in input coordinates.

    If ``color_by`` is None, points are colored by their output RGB —
    the LUT renders itself. If ``color_by`` is provided (e.g., ΔITP
    per cell), points are colored by that scalar via ``cmap``.
    """
    fig = plt.figure(figsize=(11, 9), facecolor=BG)
    ax = fig.add_subplot(111, projection="3d", facecolor=BG)
    if color_by is None:
        colors = np.clip(grid_output, 0.0, 1.0)
        sc = ax.scatter(
            grid_input[:, 0],
            grid_input[:, 1],
            grid_input[:, 2],
            c=colors,
            s=22,
            alpha=0.92,
            edgecolors="none",
            depthshade=False,
        )
    else:
        sc = ax.scatter(
            grid_input[:, 0],
            grid_input[:, 1],
            grid_input[:, 2],
            c=np.asarray(color_by).ravel(),
            cmap=cmap,
            s=22,
            alpha=0.92,
            edgecolors="none",
            depthshade=False,
        )
        cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.12)
        cbar.set_label(color_label or "scalar", color=FG)
        cbar.ax.tick_params(colors=FG)
        cbar.outline.set_edgecolor("#555555")
    ax.set_xlabel("R in", color=FG, labelpad=8)
    ax.set_ylabel("G in", color=FG, labelpad=8)
    ax.set_zlabel("B in", color=FG, labelpad=8)
    ax.set_title(title, color=HI, pad=SUPTITLE_PAD, fontsize=SUPTITLE_FS)
    _setup_3d(ax)
    ax.view_init(elev=22, azim=-58)
    _fill_3d(fig, has_cbar=color_by is not None)
    return fig


def cube_deformation(grid_input: np.ndarray, grid_output: np.ndarray) -> Figure:
    """Side-by-side: input-position cube vs output-position cube.

    The gap between the two is the LUT's deformation. Both panels use
    output RGB as the point color, so chromatic location is consistent
    across the comparison.
    """
    colors = np.clip(grid_output, 0.0, 1.0)
    fig = plt.figure(figsize=(15, 7), facecolor=BG)
    ax1 = fig.add_subplot(121, projection="3d", facecolor=BG)
    ax1.scatter(
        grid_input[:, 0],
        grid_input[:, 1],
        grid_input[:, 2],
        c=colors,
        s=18,
        alpha=0.9,
        edgecolors="none",
        depthshade=False,
    )
    ax1.set_title(
        "input positions", color=HI, pad=PANEL_TITLE_PAD, fontsize=PANEL_TITLE_FS
    )
    ax2 = fig.add_subplot(122, projection="3d", facecolor=BG)
    ax2.scatter(
        colors[:, 0],
        colors[:, 1],
        colors[:, 2],
        c=colors,
        s=18,
        alpha=0.9,
        edgecolors="none",
        depthshade=False,
    )
    ax2.set_title(
        "output positions", color=HI, pad=PANEL_TITLE_PAD, fontsize=PANEL_TITLE_FS
    )
    for ax in (ax1, ax2):
        ax.set_xlabel("R", color=FG, labelpad=6)
        ax.set_ylabel("G", color=FG, labelpad=6)
        ax.set_zlabel("B", color=FG, labelpad=6)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        _setup_3d(ax)
        ax.view_init(elev=22, azim=-58)
    fig.suptitle(
        "LUT deformation: where does each input land?", color=HI, fontsize=SUPTITLE_FS
    )
    _fill_3d(fig, has_cbar=False)
    return fig


def cube_edges(table: np.ndarray, grid_input: np.ndarray, n: int) -> Figure:
    """Trace the 12 saturated cube edges in input vs output coordinates."""
    sweep = np.arange(n)
    pin_pairs = [(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)]
    edges: list[np.ndarray] = []
    for b, g in pin_pairs:
        edges.append(np.stack([np.full(n, b), np.full(n, g), sweep], axis=-1))
    for b, r in pin_pairs:
        edges.append(np.stack([np.full(n, b), sweep, np.full(n, r)], axis=-1))
    for g, r in pin_pairs:
        edges.append(np.stack([sweep, np.full(n, g), np.full(n, r)], axis=-1))

    flat_table = table.reshape(n**3, 3)

    def flat_idx(bgr):
        return bgr[..., 0] * n * n + bgr[..., 1] * n + bgr[..., 2]

    fig = plt.figure(figsize=(15, 7), facecolor=BG)
    ax1 = fig.add_subplot(121, projection="3d", facecolor=BG)
    ax2 = fig.add_subplot(122, projection="3d", facecolor=BG)
    for edge in edges:
        idx = flat_idx(edge)
        in_pos = grid_input[idx]
        out_pos = np.clip(flat_table[idx], 0.0, 1.0)
        ax1.plot(
            in_pos[:, 0], in_pos[:, 1], in_pos[:, 2], color="#aaaaaa", lw=0.8, alpha=0.6
        )
        ax1.scatter(
            in_pos[:, 0],
            in_pos[:, 1],
            in_pos[:, 2],
            c=out_pos,
            s=22,
            alpha=0.95,
            edgecolors="none",
            depthshade=False,
        )
        ax2.plot(
            out_pos[:, 0],
            out_pos[:, 1],
            out_pos[:, 2],
            color="#aaaaaa",
            lw=0.8,
            alpha=0.6,
        )
        ax2.scatter(
            out_pos[:, 0],
            out_pos[:, 1],
            out_pos[:, 2],
            c=out_pos,
            s=22,
            alpha=0.95,
            edgecolors="none",
            depthshade=False,
        )
    ax1.set_title(
        "cube edges — input coordinates",
        color=HI,
        pad=PANEL_TITLE_PAD,
        fontsize=PANEL_TITLE_FS,
    )
    ax2.set_title(
        "cube edges — output coordinates",
        color=HI,
        pad=PANEL_TITLE_PAD,
        fontsize=PANEL_TITLE_FS,
    )
    for ax in (ax1, ax2):
        ax.set_xlabel("R", color=FG, labelpad=6)
        ax.set_ylabel("G", color=FG, labelpad=6)
        ax.set_zlabel("B", color=FG, labelpad=6)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        _setup_3d(ax)
        ax.view_init(elev=22, azim=-58)
    fig.suptitle(
        "Saturated cube edges — the canonical hue+saturation cycle",
        color=HI,
        fontsize=SUPTITLE_FS,
    )
    _fill_3d(fig, has_cbar=False)
    return fig


# ---------------------------------------------------------------------------
# Transfer-curve views (1D per-axis slices of the cube table).
# ---------------------------------------------------------------------------


def transfer_curves(
    sweep_x: np.ndarray,
    samples: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    pin_label: str,
    violation_marks: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    suptitle: str = "Per-axis transfer curves through middle-gray",
) -> Figure:
    """Per-axis transfer curves through a chosen centerline.

    Each panel sweeps one input channel from 0 to 1 with the other two
    held at a fixed encoded value, plotting R/G/B output. The caller
    computes the LUT samples via trilinear interpolation, so the
    centerline can be placed at any encoded value (not just the cube
    midpoint) — critical for log-encoded input spaces where the
    encoded midpoint isn't perceptually meaningful.

    Parameters
    ----------
    sweep_x :
        Shape ``(N,)`` — the input encoded values being swept (typically
        ``np.linspace(0, 1, N)``). Same x-axis for all three panels.
    samples :
        Three arrays of shape ``(N, 3)`` for the R-sweep, G-sweep, and
        B-sweep respectively. Each row is the LUT's encoded RGB output
        at that sweep point.
    pin_label :
        Human-readable label for what the "other channels" are pinned
        to (e.g. ``"0.46 (middle gray)"``). Shown on each panel's
        x-axis label so the figure is self-explanatory across input
        color spaces.
    violation_marks :
        Optional ``(R_mask, G_mask, B_mask)`` of monotonicity violations
        (each ``(N-1,)`` boolean), marked in red on the matching
        channel's curve.
    suptitle :
        Figure-level title. Default reflects the new
        middle-gray-centerline behavior.
    """
    setups = [
        ("R", samples[0], RED),
        ("G", samples[1], GREEN),
        ("B", samples[2], BLUE),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=BG, layout="constrained")
    for i, (ax, (label, axis_samples, axis_color)) in enumerate(zip(axes, setups)):
        ax.set_facecolor(BG)
        ax.plot(sweep_x, axis_samples[:, 0], color=RED, lw=2.2, label="R out")
        ax.plot(sweep_x, axis_samples[:, 1], color=GREEN, lw=2.2, label="G out")
        ax.plot(sweep_x, axis_samples[:, 2], color=BLUE, lw=2.2, label="B out")
        _identity_line(ax)
        if violation_marks is not None:
            mask = violation_marks[i]
            if mask.any():
                xs = sweep_x[1:][mask]
                ys = axis_samples[1:, i][mask]
                ax.scatter(
                    xs,
                    ys,
                    s=60,
                    marker="x",
                    color="#ff3366",
                    label="monotonicity violation",
                    zorder=5,
                )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(f"{label} in  (other channels = {pin_label})", color=FG)
        ax.set_ylabel("output", color=FG)
        ax.set_title(
            f"{label} sweep",
            color=axis_color,
            fontsize=PANEL_TITLE_FS,
            pad=PANEL_TITLE_PAD,
        )
        _setup_2d(ax)
        ax.legend(
            facecolor="#1a1a1a",
            labelcolor=FG,
            framealpha=0.85,
            loc="upper left",
            fontsize=9,
        )
    fig.suptitle(suptitle, color=HI, fontsize=SUPTITLE_FS)
    return fig


# ---------------------------------------------------------------------------
# Diagnostics.
# ---------------------------------------------------------------------------


def jacobian_condition_3d(log_cond_field: np.ndarray, n: int) -> Figure:
    """3D scatter of interior cube cells colored by log10(cond J)."""
    interior = np.linspace(0.0, 1.0, n)[1:-1]
    BB, GG, RR = np.meshgrid(interior, interior, interior, indexing="ij")
    fig = plt.figure(figsize=(11, 9), facecolor=BG)
    ax = fig.add_subplot(111, projection="3d", facecolor=BG)
    sc = ax.scatter(
        RR.ravel(),
        GG.ravel(),
        BB.ravel(),
        c=log_cond_field.ravel(),
        cmap="magma",
        s=14,
        alpha=0.85,
        edgecolors="none",
        depthshade=False,
    )
    cbar = fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.12)
    cbar.set_label("log₁₀(cond J)", color=FG)
    cbar.ax.tick_params(colors=FG)
    cbar.outline.set_edgecolor("#555555")
    ax.set_xlabel("R in", color=FG, labelpad=8)
    ax.set_ylabel("G in", color=FG, labelpad=8)
    ax.set_zlabel("B in", color=FG, labelpad=8)
    ax.set_title(
        "Local Jacobian condition number (cube smoothness)",
        color=HI,
        pad=SUPTITLE_PAD,
        fontsize=SUPTITLE_FS,
    )
    _setup_3d(ax)
    ax.view_init(elev=22, azim=-58)
    _fill_3d(fig, has_cbar=True)
    return fig


def output_histograms(grid_output: np.ndarray) -> Figure:
    """Per-channel output distributions with clipping markers and CDF overlay."""
    fig, axes = plt.subplots(
        1, 3, figsize=(16, 4.8), facecolor=BG, layout="constrained"
    )
    for ax, ch, col, name in zip(axes, range(3), (RED, GREEN, BLUE), ("R", "G", "B")):
        ax.set_facecolor(BG)
        values = grid_output[:, ch]
        ax.hist(
            values,
            bins=80,
            range=(-0.02, 1.02),
            color=col,
            alpha=0.85,
            edgecolor="none",
        )
        clipped_lo = int(np.sum(values <= 1e-6))
        clipped_hi = int(np.sum(values >= 1.0 - 1e-6))
        ax.axvline(0.0, color=WARN, lw=1.0, ls=":")
        ax.axvline(1.0, color=WARN, lw=1.0, ls=":")
        ax2 = ax.twinx()
        sorted_vals = np.sort(values)
        ax2.plot(
            sorted_vals,
            np.linspace(0, 1, sorted_vals.size),
            color="#ffffff",
            lw=1.0,
            alpha=0.8,
        )
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("CDF", color="#cccccc", fontsize=9)
        ax2.tick_params(colors="#cccccc", labelsize=8)
        for spine in ax2.spines.values():
            spine.set_color("#555555")
        ax.set_xlim(-0.02, 1.02)
        ax.set_xlabel(f"{name} output", color=FG)
        ax.set_ylabel("count", color=FG)
        ax.set_title(
            f"{name}    clipped lo: {clipped_lo}   hi: {clipped_hi}",
            color=col,
            fontsize=PANEL_TITLE_FS,
            pad=PANEL_TITLE_PAD,
        )
        _setup_2d(ax)
    fig.suptitle(
        "Per-channel output distributions (with CDF + clipping markers)",
        color=HI,
        fontsize=SUPTITLE_FS,
    )
    return fig


# ---------------------------------------------------------------------------
# OkLab / perceptual views.
# ---------------------------------------------------------------------------


def _linear_rgb_to_oklab(rgb_linear: np.ndarray, primaries_name: str) -> np.ndarray:
    """Output-primaries linear RGB → OkLab via XYZ.

    Parallel to :func:`_to_oklab` but for inputs that have already had
    their CCTF stripped (or never had one applied) — e.g. the unbounded
    pipeline's output.
    """
    primaries = colour.RGB_COLOURSPACES[primaries_name]
    xyz = colour.RGB_to_XYZ(
        np.asarray(rgb_linear, dtype=float),
        colourspace=primaries.name,
        apply_cctf_decoding=False,
        illuminant=primaries.whitepoint,
    )
    return np.asarray(colour.XYZ_to_Oklab(xyz), dtype=float)


def _linear_rgb_to_xy(rgb_linear: np.ndarray, primaries) -> np.ndarray:
    """Linear RGB → xy chromaticity in the given primaries' frame."""
    xyz = colour.RGB_to_XYZ(
        np.asarray(rgb_linear, dtype=float),
        colourspace=primaries.name,
        apply_cctf_decoding=False,
        illuminant=np.asarray(primaries.whitepoint, dtype=float),
    )
    b = xyz.sum(axis=-1, keepdims=True)
    return np.asarray(xyz[..., :2] / np.where(np.abs(b) > 1e-12, b, 1.0), dtype=float)


def _render_oklab_gamut_3d(
    ax,
    *,
    cube_lab: np.ndarray,
    cube_colors: np.ndarray,
    rim_unbounded_lab: np.ndarray,
    rim_compressed_lab: np.ndarray,
    rim_hues: np.ndarray,
    rim_n_per_segment: int,
    rim_n_segments: int,
    rim_bright_mask: np.ndarray,
    rim_oog_mask: np.ndarray,
    compression_active: bool,
    out_cs_name: str,
) -> None:
    """Draw the 3D OkLab cube cloud + before/after rim envelope onto ``ax``.

    Frame is keyed to the cube + unbounded rim's combined extent so the
    out-of-gamut excursion is visible.
    """
    ax.scatter(
        cube_lab[:, 1],
        cube_lab[:, 2],
        cube_lab[:, 0],
        c=cube_colors,
        s=4,
        alpha=0.18,
        edgecolors="none",
        depthshade=False,
        zorder=1.0,
    )

    # Skip the three "lit-white" edges per axis (same convention as
    # the xy panel) — they're saturation sweeps and clutter the rim
    # envelope with lines running into the achromatic axis.
    for k in range(rim_n_segments):
        if k % 4 == 3:
            continue
        s, e = k * rim_n_per_segment, (k + 1) * rim_n_per_segment
        if not rim_bright_mask[s:e].any():
            continue
        seg_hue = rim_hues[s:e]
        col = plt.cm.hsv(
            np.angle(np.exp(1j * 2 * np.pi * np.mean(seg_hue))) / (2 * np.pi) % 1.0
        )
        ax.plot(
            rim_unbounded_lab[s:e, 1],
            rim_unbounded_lab[s:e, 2],
            rim_unbounded_lab[s:e, 0],
            color=col,
            lw=2.0,
            alpha=0.95,
            zorder=3.0,
        )
        if compression_active:
            ax.plot(
                rim_compressed_lab[s:e, 1],
                rim_compressed_lab[s:e, 2],
                rim_compressed_lab[s:e, 0],
                color=col,
                lw=1.0,
                alpha=0.65,
                ls="--",
                zorder=2.8,
            )

    # Sparse displacement arrows on a brightest-OOG subset (3D quiver).
    if compression_active and rim_oog_mask.any():
        oog_idx = np.flatnonzero(rim_oog_mask)
        n_arrows = min(len(oog_idx), 80)
        if n_arrows > 0:
            rng = np.random.default_rng(0)
            pick = rng.choice(oog_idx, size=n_arrows, replace=False)
            u = rim_compressed_lab[pick, 1] - rim_unbounded_lab[pick, 1]
            v = rim_compressed_lab[pick, 2] - rim_unbounded_lab[pick, 2]
            w = rim_compressed_lab[pick, 0] - rim_unbounded_lab[pick, 0]
            ax.quiver(
                rim_unbounded_lab[pick, 1],
                rim_unbounded_lab[pick, 2],
                rim_unbounded_lab[pick, 0],
                u,
                v,
                w,
                color="#ffaa55",
                alpha=0.55,
                arrow_length_ratio=0.18,
                linewidth=0.9,
                zorder=3.5,
            )

    # Frame on the union of cube + unbounded rim. The unbounded rim
    # extends past the cube along OOG directions, which is the point
    # of the visualization — show it.
    pts = np.vstack([cube_lab, rim_unbounded_lab[rim_bright_mask]])
    a_lo, a_hi = float(pts[:, 1].min()), float(pts[:, 1].max())
    b_lo, b_hi = float(pts[:, 2].min()), float(pts[:, 2].max())
    L_lo, L_hi = float(pts[:, 0].min()), float(pts[:, 0].max())
    pad = 0.10
    a_pad = pad * max(a_hi - a_lo, 1e-3)
    b_pad = pad * max(b_hi - b_lo, 1e-3)
    L_pad = pad * max(L_hi - L_lo, 1e-3)
    ax.set_xlim(a_lo - a_pad, a_hi + a_pad)
    ax.set_ylim(b_lo - b_pad, b_hi + b_pad)
    ax.set_zlim(L_lo - L_pad, L_hi + L_pad)
    ax.set_xlabel("a*", color=FG, labelpad=6)
    ax.set_ylabel("b*", color=FG, labelpad=6)
    ax.set_zlabel("L*", color=FG, labelpad=6)
    _setup_3d(ax)
    ax.view_init(elev=18, azim=-58)
    ax.text2D(
        0.02,
        0.97,
        f"OkLab gamut  ({out_cs_name})\n"
        f"cloud = compressed LUT cube · rim solid = unbounded · "
        f"rim dashed = compressed",
        transform=ax.transAxes,
        color=HI,
        fontsize=10,
        ha="left",
        va="top",
    )


def _render_xy_compression_panel(
    ax,
    *,
    xy_unbounded: np.ndarray,
    xy_compressed: np.ndarray,
    rim_hues: np.ndarray,
    rim_n_per_segment: int,
    rim_n_segments: int,
    rim_bright_mask: np.ndarray,
    rim_oog_mask: np.ndarray,
    out_tri: np.ndarray,
    out_white: np.ndarray,
    out_cs_name: str,
    in_cs_name: str,
    compression_spec,
    compression_active: bool,
    oog_fraction: float,
    displacement: np.ndarray,
) -> None:
    """Draw the xy chromaticity compression preview onto ``ax``.

    Spectral locus + output primaries triangle + unbounded vs
    compressed rim envelope, with displacement arrows on the
    out-of-gamut subset and a compact stats panel.
    """
    from spektrafilm.utils.gamut_compression import spectral_locus_xy

    bg, fg, accent, dim = BG, FG, "#ffee66", "#888888"
    locus = spectral_locus_xy()

    for spine in ax.spines.values():
        spine.set_color("#555555")
    ax.tick_params(colors=fg)
    ax.grid(True, alpha=0.12, color=accent)

    ax.plot(
        locus[:, 0],
        locus[:, 1],
        color=dim,
        lw=1.0,
        alpha=0.45,
        label="visible spectral locus",
    )
    locus_path = plt.Polygon(
        locus, closed=True, facecolor="#cccccc", alpha=0.025, edgecolor="none"
    )
    ax.add_patch(locus_path)

    tri = np.vstack([out_tri, out_tri[:1]])
    ax.fill(tri[:, 0], tri[:, 1], color="#ffffff", alpha=0.04, zorder=1.5)
    ax.plot(
        tri[:, 0],
        tri[:, 1],
        color=fg,
        lw=2.0,
        alpha=0.95,
        label=f"{out_cs_name} gamut",
        zorder=2,
    )
    primary_colors = ["#ff5566", "#66ff88", "#5599ff"]
    primary_labels = ["R", "G", "B"]
    for (px, py), pcol, plab in zip(out_tri, primary_colors, primary_labels):
        ax.plot(
            px,
            py,
            "o",
            color=pcol,
            markersize=11,
            markeredgecolor=bg,
            markeredgewidth=1.5,
            zorder=4,
        )
        offset = np.array([px, py]) - out_white
        n = np.linalg.norm(offset) + 1e-9
        lx, ly = np.array([px, py]) + 0.035 * offset / n
        ax.text(
            lx,
            ly,
            plab,
            color=pcol,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            zorder=5,
        )
    ax.plot(
        out_white[0],
        out_white[1],
        "D",
        color=fg,
        markersize=10,
        markeredgecolor=bg,
        markeredgewidth=1.2,
        label=f"{out_cs_name} white",
        zorder=4,
    )

    bright_idx = np.flatnonzero(rim_bright_mask)
    for k in range(rim_n_segments):
        if k % 4 == 3:
            continue
        s, e = k * rim_n_per_segment, (k + 1) * rim_n_per_segment
        if not rim_bright_mask[s:e].any():
            continue
        seg_hue = rim_hues[s:e]
        col = plt.cm.hsv(
            np.angle(np.exp(1j * 2 * np.pi * np.mean(seg_hue))) / (2 * np.pi) % 1.0
        )
        ax.plot(
            xy_unbounded[s:e, 0],
            xy_unbounded[s:e, 1],
            color=col,
            lw=4.0,
            alpha=0.25,
            zorder=2.6,
        )
        ax.plot(
            xy_unbounded[s:e, 0],
            xy_unbounded[s:e, 1],
            color=col,
            lw=1.6,
            alpha=0.95,
            zorder=2.7,
        )
        if compression_active:
            ax.plot(
                xy_compressed[s:e, 0],
                xy_compressed[s:e, 1],
                color=col,
                lw=1.0,
                alpha=0.7,
                ls="--",
                zorder=2.65,
            )

    ax.scatter(
        xy_unbounded[bright_idx, 0],
        xy_unbounded[bright_idx, 1],
        c=rim_hues[bright_idx],
        cmap=plt.cm.hsv,
        s=20,
        alpha=0.95,
        edgecolors="none",
        zorder=3,
        label="unbounded rim",
    )
    if compression_active and rim_oog_mask.any():
        ax.scatter(
            xy_compressed[bright_idx, 0],
            xy_compressed[bright_idx, 1],
            c=rim_hues[bright_idx],
            cmap=plt.cm.hsv,
            s=10,
            alpha=0.7,
            edgecolors="none",
            zorder=3.5,
            label="compressed rim",
        )
        oog_bright_idx = np.flatnonzero(rim_oog_mask)
        n_arrows = min(len(oog_bright_idx), 120)
        if n_arrows > 0:
            rng = np.random.default_rng(0)
            pick = rng.choice(oog_bright_idx, size=n_arrows, replace=False)
            ax.quiver(
                xy_unbounded[pick, 0],
                xy_unbounded[pick, 1],
                xy_compressed[pick, 0] - xy_unbounded[pick, 0],
                xy_compressed[pick, 1] - xy_unbounded[pick, 1],
                color="#ffaa55",
                alpha=0.5,
                angles="xy",
                scale_units="xy",
                scale=1.0,
                width=0.0022,
                headwidth=4,
                headlength=5,
                zorder=3.7,
            )

    if compression_active and rim_oog_mask.any():
        text = (
            f"algorithm:    {compression_spec.algorithm}\n"
            f"threshold:    {compression_spec.knee[0]}\n"
            f"limit:        {compression_spec.knee[1]}\n"
            f"power:        {compression_spec.knee[2]}\n"
            f"\n"
            f"input:        {in_cs_name}\n"
            f"output:       {out_cs_name}\n"
            f"OOG fraction: {oog_fraction:.1%}\n"
            f"OOG samples:  {int(rim_oog_mask.sum())}\n"
            f"max disp:     {displacement[rim_oog_mask].max():.4f}\n"
            f"p99 disp:     "
            f"{np.percentile(displacement[rim_oog_mask], 99):.4f}\n"
            f"mean disp:    {displacement[rim_oog_mask].mean():.4f}"
        )
    elif not compression_active:
        text = (
            f"algorithm:    {compression_spec.algorithm}\n"
            f"mode:         OFF\n"
            f"\n"
            f"input:        {in_cs_name}\n"
            f"output:       {out_cs_name}\n"
            f"OOG fraction: {oog_fraction:.1%}\n"
            f"(compression disabled — rim shown as-is)"
        )
    else:
        text = (
            f"algorithm:    {compression_spec.algorithm}\n"
            f"\n(no bright OOG samples — the simulation's rim is\n"
            f" already inside the output gamut for this input)"
        )
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        color=fg,
        family="monospace",
        fontsize=9,
        bbox=dict(
            facecolor="#1a1a1a",
            edgecolor="#555555",
            alpha=0.92,
            boxstyle="round,pad=0.5",
        ),
        zorder=10,
    )

    leg = ax.legend(
        loc="upper right",
        fontsize=8,
        facecolor="#1a1a1a",
        edgecolor="#555555",
        labelcolor=fg,
    )
    leg.get_frame().set_alpha(0.9)

    ax.set_xlim(-0.05, 0.85)
    ax.set_ylim(-0.05, 0.95)
    ax.set_xlabel("x", color=fg)
    ax.set_ylabel("y", color=fg)
    ax.set_aspect("equal")


def gamut_compression_3d_xy(
    *,
    grid_output_compressed: np.ndarray,
    rim_unbounded_linear: np.ndarray,
    rim_compressed_linear: np.ndarray,
    rim_input_hues: np.ndarray,
    rim_n_per_segment: int,
    rim_n_segments: int,
    in_cs_name: str,
    out_cs_name: str,
    compression_spec,
) -> Figure:
    """1x2 figure: 3D OkLab gamut (cube cloud + rim before/after) and 2D xy
    chromaticity preview of the same compression event.

    Left panel — perceptual volume + rim envelope:
      - Faint scatter of the LUT's compressed output cube (= what the
        LUT actually delivers) for volume context, colored by the
        cube's input RGB.
      - Bright colored polylines tracing the **unbounded rim** — the
        chromaticities the simulation would reach with output gamut
        compression disabled — overlaid on top.
      - Dimmer dashed polylines for the **compressed rim** so the eye
        can follow each cube-edge's displacement under compression.
      - Sparse displacement arrows on the OOG subset (when there is
        out-of-gamut content to show).

    Right panel — the canonical xy chromaticity preview:
      - Spectral locus + output gamut triangle.
      - Unbounded rim (rainbow scatter) → compressed rim (faded
        inner ring) → optional displacement arrows.
      - Compact stats panel.

    All RGB inputs for the rim are in the output color space's
    **linear** primaries (no CCTF applied), matching what the runtime's
    unbounded pipeline produces. ``grid_output_compressed`` is the
    shipped LUT's encoded output (matches ``ctx.grid_output``).
    """
    from spektrafilm_lut_creator.color_spaces import get as get_cs

    out_entry = get_cs(out_cs_name)
    out_primaries_name = out_entry.primaries
    out_primaries = colour.RGB_COLOURSPACES[out_primaries_name]
    out_white = np.asarray(out_primaries.whitepoint, dtype=float)
    out_tri = np.asarray(out_primaries.primaries, dtype=float)

    cube_lab = _to_oklab(grid_output_compressed, out_cs_name)
    cube_colors = np.clip(grid_output_compressed, 0.0, 1.0)
    rim_unbounded_lab = _linear_rgb_to_oklab(
        rim_unbounded_linear,
        out_primaries_name,
    )
    rim_compressed_lab = _linear_rgb_to_oklab(
        rim_compressed_linear,
        out_primaries_name,
    )
    xy_unbounded = _linear_rgb_to_xy(rim_unbounded_linear, out_primaries)
    xy_compressed = _linear_rgb_to_xy(rim_compressed_linear, out_primaries)

    # ACES RGC "achromatic distance" — how far past the output gamut
    # each rim sample sits, in the same RGB metric the algorithm uses.
    # ach > 1e-2 skips near-black samples whose direction is dominated
    # by noise.
    ach = rim_unbounded_linear.max(axis=-1)
    safe_ach = np.where(ach > 1e-6, ach, 1.0)
    d_per_channel = (ach[..., None] - rim_unbounded_linear) / safe_ach[..., None]
    d_max = d_per_channel.max(axis=-1)
    bright_mask = ach > 1e-2
    oog_mask = (d_max > 1.0) & bright_mask
    oog_fraction = float(oog_mask.sum() / max(int(bright_mask.sum()), 1))
    displacement = np.linalg.norm(
        rim_unbounded_linear - rim_compressed_linear,
        axis=-1,
    )
    compression_active = getattr(compression_spec, "mode", "off") != "off"

    fig = plt.figure(figsize=(18, 9), facecolor=BG)
    ax3d = fig.add_subplot(121, projection="3d", facecolor=BG)
    ax2d = fig.add_subplot(122, facecolor=BG)
    _render_oklab_gamut_3d(
        ax3d,
        cube_lab=cube_lab,
        cube_colors=cube_colors,
        rim_unbounded_lab=rim_unbounded_lab,
        rim_compressed_lab=rim_compressed_lab,
        rim_hues=rim_input_hues,
        rim_n_per_segment=rim_n_per_segment,
        rim_n_segments=rim_n_segments,
        rim_bright_mask=bright_mask,
        rim_oog_mask=oog_mask,
        compression_active=compression_active,
        out_cs_name=out_cs_name,
    )
    _render_xy_compression_panel(
        ax2d,
        xy_unbounded=xy_unbounded,
        xy_compressed=xy_compressed,
        rim_hues=rim_input_hues,
        rim_n_per_segment=rim_n_per_segment,
        rim_n_segments=rim_n_segments,
        rim_bright_mask=bright_mask,
        rim_oog_mask=oog_mask,
        out_tri=out_tri,
        out_white=out_white,
        out_cs_name=out_cs_name,
        in_cs_name=in_cs_name,
        compression_spec=compression_spec,
        compression_active=compression_active,
        oog_fraction=oog_fraction,
        displacement=displacement,
    )

    algorithm_label = {
        "oklch": "OkLch chroma reduction",
        "aces_rgc": "ACES RGC v1.3",
    }.get(
        getattr(compression_spec, "algorithm", ""),
        getattr(compression_spec, "algorithm", ""),
    )
    fig.suptitle(
        f"output gamut compression — {in_cs_name} → {out_cs_name}   "
        f"via {algorithm_label}",
        color=HI,
        fontsize=SUPTITLE_FS,
    )
    fig.subplots_adjust(
        left=0.02,
        right=0.98,
        top=1.0 - HEADER_BAND_FRAC,
        bottom=FOOTER_BAND_FRAC,
        wspace=0.05,
    )
    return fig


def offgrid_error_scatter(
    samples_encoded: np.ndarray,
    delta_field: np.ndarray,
    *,
    title: str = "Off-grid ΔITP",
    cbar_label: str = "ΔITP",
) -> Figure:
    """3D scatter of off-grid samples colored by per-sample error.

    Highlights the cube regions where the LUT's interpolated value
    drifts most from the reference pipeline.
    """
    fig = plt.figure(figsize=(11, 9), facecolor=BG)
    ax = fig.add_subplot(111, projection="3d", facecolor=BG)
    sc = ax.scatter(
        samples_encoded[:, 0],
        samples_encoded[:, 1],
        samples_encoded[:, 2],
        c=delta_field,
        cmap="viridis",
        s=4,
        alpha=0.55,
        edgecolors="none",
        depthshade=False,
    )
    cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.12)
    # Scatter dots are semi-transparent so overlapping samples read; the
    # colorbar should NOT inherit that alpha — restore full saturation
    # so the legend's hue mapping is unambiguous.
    cbar.solids.set_alpha(1.0)
    cbar.solids.set_edgecolor("face")
    cbar.set_label(cbar_label, color=FG)
    cbar.ax.tick_params(colors=FG)
    cbar.outline.set_edgecolor("#555555")
    ax.set_xlabel("R in", color=FG, labelpad=8)
    ax.set_ylabel("G in", color=FG, labelpad=8)
    ax.set_zlabel("B in", color=FG, labelpad=8)
    ax.set_title(title, color=HI, pad=SUPTITLE_PAD, fontsize=SUPTITLE_FS)
    _setup_3d(ax)
    ax.view_init(elev=22, azim=-58)
    _fill_3d(fig, has_cbar=True)
    return fig
