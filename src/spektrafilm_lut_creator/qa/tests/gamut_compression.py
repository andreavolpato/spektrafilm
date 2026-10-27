"""Input-gamut-compression QA diagnostics — preview + smoothness probes."""
from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

import colour
import spektrafilm_lut_creator.color_spaces as color_spaces

from spektrafilm_lut_creator.color_spaces import to_xyz
from spektrafilm_lut_creator.qa import evaluators, metrics, patterns, reference, viz
from spektrafilm_lut_creator.qa.result import Result
from spektrafilm_lut_creator.qa.tests._helpers import _save, MIDGRAY_18_OKLAB_L

if TYPE_CHECKING:
    from spektrafilm_lut_creator.qa.suite import QAContext


def _cube_xy_in_film_frame(ctx: "QAContext", reference_illuminant: str):
    """Project the QA cube's input samples to CIE xy in the film's
    reference-illuminant frame — same path ``_rgb_to_tc_b`` runs at
    runtime, minus the LUT lookup.

    Returns ``(xy, b)`` where ``xy`` is shape ``(N, 2)`` and ``b`` is
    ``X+Y+Z`` (for the brightness threshold).
    """
    from spektrafilm_lut_creator.color_spaces import get as get_cs
    in_cs = get_cs(ctx.spec.input_color_space)
    rgb = ctx.grid_input  # encoded; ctx.reference uses encoded inputs too
    # Decode to linear if the input space has a CCTF.
    if in_cs.cctf is not None:
        rgb_linear = np.asarray(
            colour.cctf_decoding(rgb, function=in_cs.cctf), dtype=float,
        )
    else:
        rgb_linear = np.asarray(rgb, dtype=float)
    ref_xy = np.asarray(
        colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"][
            reference_illuminant
        ], dtype=float,
    )
    xyz = colour.RGB_to_XYZ(
        rgb_linear,
        colourspace=in_cs.primaries,
        apply_cctf_decoding=False,
        illuminant=ref_xy,
        chromatic_adaptation_transform="CAT16",
    )
    b = np.asarray(xyz, dtype=float).sum(axis=-1)
    safe_b = np.where(b > 1e-12, b, 1.0)
    xy = xyz[..., :2] / safe_b[..., None]
    return np.asarray(xy, dtype=float), b

def _film_reference_illuminant(ctx: "QAContext") -> str:
    """Resolve the film's reference illuminant by loading its profile.

    The film profile carries the illuminant the spectral sensitivities
    were measured under — and the spektrafilm runtime CAT16-adapts
    input chromaticities to this illuminant before feeding the Hanatos
    LUT. The compression in the baked LUT operates in this same frame,
    so the QA plot must too. Falls back to ``"D55"`` if anything goes
    wrong (most film profiles).
    """
    try:
        from spektrafilm.data.profiles_loader import load_profile
        profile = load_profile(ctx.spec.film_profile)
        ref = profile.info.reference_illuminant
        return str(ref) if ref else "D55"
    except Exception:
        return "D55"

def input_gamut_compression_preview(ctx: "QAContext") -> Result:
    """Visualize what the input gamut compression does for this bundle.

    For the bundle's input color space, we project the QA cube to CIE
    xy in the film's reference-illuminant frame (the same projection
    the runtime does just before the Hanatos LUT lookup), identify
    out-of-locus samples, and draw arrows from each OOG sample to its
    compressed destination. The figure is informational — the
    compression itself is correct by construction; this lets a colorist
    see at a glance how much of their input cube gets modified.

    Styling mirrors spektrafilm-research/studies/a40_lut_system/
    tune_input_gamut_compression.py ``plot_compression_preview`` so the
    QA artifact reads the same way as the study figures.

    References
    ----------
    - ACES Reference Gamut Compression v1.3 (AMPAS, 2020).
    - Hanatos et al., *Sigmoidal Compression for Reflectance Manifold* (2025).
    - spektrafilm-research n100 §5.
    """
    from spektrafilm.utils.gamut_compression import (
        compress_xy, spectral_locus_xy,
    )
    from matplotlib.path import Path as MplPath

    # Palette matches the tuning script (BG/FG/HI/DIM and the
    # OOG/compressed/arrow colors). Keeping these inline rather than
    # importing from the research tree keeps QA self-contained.
    # ``accent`` is the yellow-ish color used for the input-gamut
    # triangle overlay (visible against the dark BG); titles use the
    # shared viz.HI white so they match the rest of the report.
    bg, fg, accent, dim = "#0a0a0a", "#cccccc", "#ffee66", "#888888"
    ok_color = "#66cc99"
    oog_color = "#ff6666"
    moved_color = "#66ccff"
    arrow_color = "#ffaa55"

    spec = ctx.spec.input_gamut_compress
    ref_illuminant = _film_reference_illuminant(ctx)
    ref_xy_arr = np.asarray(
        colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"][
            ref_illuminant
        ], dtype=float,
    )

    xy, b = _cube_xy_in_film_frame(ctx, ref_illuminant)
    locus = spectral_locus_xy()

    # Match the tuning script's brightness gates so the numbers here
    # are directly comparable to its OOG figures.
    bright_mask = b > 1e-2
    degenerate_mask = b <= 1e-4
    in_locus = MplPath(locus).contains_points(xy)
    oog_mask = (~in_locus) & (~degenerate_mask)
    oog_bright_mask = oog_mask & bright_mask
    valid = ~degenerate_mask
    oog_fraction = float(oog_mask.sum() / max(int(valid.sum()), 1))

    # Compress the entire cube once; we'll only draw arrows on the
    # bright OOG subset (the population the knee was sized for).
    if spec.active:
        xy_out = compress_xy(xy, ref_xy_arr, spec)
    else:
        xy_out = xy.copy()

    fig, ax = plt.subplots(figsize=(9, 9), facecolor=bg, layout="constrained")
    ax.set_facecolor(bg)
    for spine in ax.spines.values():
        spine.set_color("#555555")
    ax.tick_params(colors=fg)
    ax.grid(True, alpha=0.12, color=accent)

    # Spectral locus.
    ax.plot(locus[:, 0], locus[:, 1], color=fg, lw=1.3, alpha=0.95,
            label="spectral locus")

    # Input gamut triangle (native primaries, native white).
    try:
        from spektrafilm_lut_creator.color_spaces import get as _get_cs
        in_entry = _get_cs(ctx.spec.input_color_space)
        in_cs_obj = colour.RGB_COLOURSPACES[in_entry.primaries]
        pri = np.asarray(in_cs_obj.primaries, dtype=float)
        tri = np.vstack([pri, pri[:1]])
        ax.plot(tri[:, 0], tri[:, 1], color=accent, lw=1.4, alpha=0.7,
                label=f"{ctx.spec.input_color_space} gamut")
        ax.fill(tri[:, 0], tri[:, 1], color=accent, alpha=0.04)
    except Exception:
        pass

    # Reference illuminant marker.
    ax.plot(ref_xy_arr[0], ref_xy_arr[1], "D", color=dim, markersize=8,
            markeredgecolor=fg, markeredgewidth=0.8,
            label=f"film ref illum ({ref_illuminant})")

    # Background: in-locus samples (faint), then OOG originals (red),
    # then compressed positions (cyan) for both OOG and the in-locus
    # samples the knee actually shifted, then arrows on top.
    in_locus_valid = in_locus & valid
    if in_locus_valid.any():
        ax.scatter(xy[in_locus_valid, 0], xy[in_locus_valid, 1],
                   c=ok_color, s=2, alpha=0.25, edgecolors="none",
                   zorder=2)
        # In-locus samples the compression actually moved. The 1e-3 xy
        # threshold filters samples well below the knee onset whose
        # nominal displacement is just floating-point noise, so the
        # bulk of stationary interior points don't clutter the figure.
        if spec.active:
            in_locus_bright = in_locus_valid & bright_mask
            if in_locus_bright.any():
                bright_in_idx = np.flatnonzero(in_locus_bright)
                disp_in = np.linalg.norm(
                    xy_out[bright_in_idx] - xy[bright_in_idx], axis=-1,
                )
                moved_idx = bright_in_idx[disp_in > 1e-3]
                if moved_idx.size > 0:
                    ax.scatter(
                        xy_out[moved_idx, 0], xy_out[moved_idx, 1],
                        c=moved_color, s=3, alpha=0.7,
                        edgecolors="none", zorder=3.8,
                    )
                    n_in = min(moved_idx.size, 400)
                    pick_in = np.random.default_rng(1).choice(
                        moved_idx, size=n_in, replace=False,
                    )
                    ax.quiver(
                        xy[pick_in, 0], xy[pick_in, 1],
                        xy_out[pick_in, 0] - xy[pick_in, 0],
                        xy_out[pick_in, 1] - xy[pick_in, 1],
                        color=arrow_color, alpha=0.4,
                        angles="xy", scale_units="xy", scale=1.0,
                        width=0.0018, headwidth=4, headlength=5,
                        zorder=3.3,
                    )
    if oog_mask.any():
        ax.scatter(xy[oog_mask, 0], xy[oog_mask, 1], c=oog_color,
                   s=4, alpha=0.45, edgecolors="none", zorder=3,
                   label="OOG (original)")
    if oog_bright_mask.any() and spec.active:
        ax.scatter(xy_out[oog_bright_mask, 0], xy_out[oog_bright_mask, 1],
                   c=moved_color, s=5, alpha=0.85, edgecolors="none",
                   zorder=4, label="compressed")
        # Displacement arrows on bright OOG only, capped at 400 for
        # legibility on dense V-Gamut-like inputs.
        bright_idx = np.flatnonzero(oog_bright_mask)
        n_arrows = min(len(bright_idx), 400)
        rng = np.random.default_rng(0)
        pick = rng.choice(bright_idx, size=n_arrows, replace=False)
        x0 = xy[pick, 0]; y0 = xy[pick, 1]
        x1 = xy_out[pick, 0]; y1 = xy_out[pick, 1]
        ax.quiver(
            x0, y0, x1 - x0, y1 - y0,
            color=arrow_color, alpha=0.55,
            angles="xy", scale_units="xy", scale=1.0,
            width=0.0025, headwidth=4, headlength=5, zorder=3.5,
        )

    # Stats panel in the upper left, monospace so the columns line up.
    if oog_bright_mask.any() and spec.active:
        disp = np.linalg.norm(
            xy_out[oog_bright_mask] - xy[oog_bright_mask], axis=-1,
        )
        text = (
            f"algorithm:    {spec.algorithm}\n"
            f"active:       {spec.active}\n"
            f"threshold:    {spec.knee[0]}\n"
            f"limit:        {spec.knee[1]}\n"
            f"power:        {spec.knee[2]}\n"
            f"\n"
            f"input:        {ctx.spec.input_color_space}\n"
            f"OOG fraction: {oog_fraction:.1%}\n"
            f"OOG (bright): {int(oog_bright_mask.sum())}\n"
            f"max disp:     {disp.max():.4f}\n"
            f"p99 disp:     {np.percentile(disp, 99):.4f}\n"
            f"mean disp:    {disp.mean():.4f}"
        )
    elif not spec.active:
        text = (
            f"algorithm:    {spec.algorithm}\n"
            f"active:       False\n"
            f"\n"
            f"input:        {ctx.spec.input_color_space}\n"
            f"OOG fraction: {oog_fraction:.1%}\n"
            f"(compression disabled — OOG samples passed through unchanged)"
        )
    else:
        text = (
            f"algorithm:    {spec.algorithm}\n"
            f"threshold:    {spec.knee[0]}\n"
            f"limit:        {spec.knee[1]}\n"
            f"power:        {spec.knee[2]}\n"
            f"\n(no bright OOG samples — nothing to compress)"
        )
    ax.text(
        0.02, 0.98, text,
        transform=ax.transAxes, va="top", ha="left",
        color=fg, family="monospace", fontsize=9,
        bbox=dict(facecolor="#1a1a1a", edgecolor="#555555",
                  alpha=0.92, boxstyle="round,pad=0.5"),
    )

    ax.set_xlim(-0.05, 0.85)
    ax.set_ylim(-0.05, 0.95)
    ax.set_xlabel("x", color=fg)
    ax.set_ylabel("y", color=fg)
    ax.set_aspect("equal")
    ax.set_title(
        f"compression preview — {ctx.spec.input_color_space} via "
        f"{spec.algorithm} (t={spec.knee[0]}, l={spec.knee[1]}, "
        f"p={spec.knee[2]})",
        color=viz.HI, fontsize=viz.SUPTITLE_FS, pad=viz.SUPTITLE_PAD,
    )
    ax.legend(facecolor="#1a1a1a", labelcolor=fg, framealpha=0.9,
              loc="upper right", fontsize=8)

    path = _save(ctx, fig, "input_gamut_compression_preview")

    return Result(
        name="input_gamut_compression_preview",
        summary={
            "active": spec.active,
            "algorithm": spec.algorithm,
            "knee_threshold": float(spec.knee[0]),
            "knee_limit": float(spec.knee[1]),
            "knee_power": float(spec.knee[2]),
            "oog_fraction": oog_fraction,
            "n_oog_samples": int(oog_mask.sum()),
            "n_oog_bright": int(oog_bright_mask.sum()),
            "reference_illuminant": ref_illuminant,
        },
        figure_path=path,
        units="",
        interpretation=(
            "Shows which cube samples fall outside the visible spectral "
            "locus (where Hanatos 2025 spectral upsampling is well "
            "defined) and where the compression maps them. Red = input "
            "OOG, cyan = compressed destination, orange arrows = "
            "displacement (bright OOG subset only, capped at 400 for "
            "legibility). Samples in the locus pass through unchanged. "
            "The compression is baked into the per-film tc_lut at build "
            "time (n100 §3.1); this plot is the build's audit trail. "
            "Informational only — no pass/fail."
        ),
        passed=None,
    )

def input_gamut_compression_smoothness(ctx: "QAContext") -> Result:
    """Probe the compression's smoothness on a circumferential ring.

    A circle around the film reference illuminant that crosses the
    spectral locus in several places. After compression the output
    should be a smooth closed curve, color-coded uniformly with input
    angle (HSV hue → input azimuth). Visible kinks, bunching, or color
    jumps reveal hue smoothness issues.

    Styling mirrors spektrafilm-research/studies/a40_lut_system/
    tune_input_gamut_compression.py ``plot_smoothness_circumferential``
    so the QA artifact reads identically to the study figure.

    References
    ----------
    - spektrafilm-research n100 §5.1 (smoothness probes).
    - ``tune_input_gamut_compression.py`` ``plot_smoothness_circumferential``.
    """
    from spektrafilm.utils.gamut_compression import (
        compress_xy, spectral_locus_xy,
    )

    # ``accent`` is the yellow-ish color used for the input-gamut
    # triangle overlay (visible against the dark BG); titles use the
    # shared viz.HI white so they match the rest of the report.
    bg, fg, accent, dim = "#0a0a0a", "#cccccc", "#ffee66", "#888888"

    spec = ctx.spec.input_gamut_compress
    ref_illuminant = _film_reference_illuminant(ctx)
    ref_xy_arr = np.asarray(
        colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"][
            ref_illuminant
        ], dtype=float,
    )

    # Radius matches the tuning script default (0.30 puts about half the
    # circle in OOG territory for D55-centred rings — comfortable for
    # showing the knee in action without sweeping into unphysical xy).
    radius = 0.30
    n_samples = 720
    angles = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    direction = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    probe_input = ref_xy_arr[None, :] + direction * radius
    angles_deg = np.degrees(angles)

    probe_output = compress_xy(probe_input, ref_xy_arr, spec)
    locus = spectral_locus_xy()

    # Try to draw the input gamut triangle in the background — useful
    # context for "which directions cross the gamut edge first".
    pri = None
    try:
        from spektrafilm_lut_creator.color_spaces import get as _get_cs
        in_entry = _get_cs(ctx.spec.input_color_space)
        in_cs_obj = colour.RGB_COLOURSPACES[in_entry.primaries]
        pri = np.asarray(in_cs_obj.primaries, dtype=float)
    except Exception:
        pass

    # Smoothness metric: ratio of worst inter-sample step to median.
    # Close to 1 = perfectly even hue spacing; >>1 = a discontinuity.
    diffs = np.diff(probe_output, axis=0)
    step_lengths = np.linalg.norm(diffs, axis=-1)
    worst_step = float(step_lengths.max())
    median_step = float(np.median(step_lengths))
    ratio = worst_step / max(median_step, 1e-9)

    fig, ax = plt.subplots(figsize=(7, 7), facecolor=bg, layout="constrained")
    ax.set_facecolor(bg)
    for spine in ax.spines.values():
        spine.set_color("#555555")
    ax.tick_params(colors=fg)
    ax.grid(True, alpha=0.12, color=accent)

    # Spectral locus.
    ax.plot(locus[:, 0], locus[:, 1], color=fg, lw=1.0, alpha=0.85,
            label="spectral locus")

    # Input gamut triangle.
    if pri is not None:
        tri = np.vstack([pri, pri[:1]])
        ax.plot(tri[:, 0], tri[:, 1], color=accent, lw=1.0, alpha=0.5,
                label=f"{ctx.spec.input_color_space} gamut")

    # Input circle as a dashed reference.
    ax.plot(probe_input[:, 0], probe_input[:, 1], color=dim, lw=0.6,
            ls="--", alpha=0.5)

    # Compressed output colored by input angle — the visual signature of
    # smoothness is a clean rainbow ring with no color discontinuities.
    ax.scatter(probe_output[:, 0], probe_output[:, 1],
               c=angles_deg, cmap=plt.cm.hsv, s=6, alpha=0.95,
               edgecolors="none")

    # Reference illuminant.
    ax.plot(ref_xy_arr[0], ref_xy_arr[1], "D", color=dim, markersize=7,
            markeredgecolor=fg, markeredgewidth=0.7,
            label=f"film ref ({ref_illuminant})")

    ax.set_xlim(-0.05, 0.85)
    ax.set_ylim(-0.05, 0.95)
    ax.set_xlabel("x", color=fg)
    ax.set_ylabel("y", color=fg)
    ax.set_aspect("equal")
    ax.set_title(
        f"circumferential smoothness probe — "
        f"{ctx.spec.input_color_space} via {spec.algorithm} "
        f"(t={spec.knee[0]}, l={spec.knee[1]}, p={spec.knee[2]})\n"
        f"r = {radius} from {ref_illuminant}    "
        f"worst/median step {ratio:.2f}",
        color=viz.HI, fontsize=viz.SUPTITLE_FS, pad=viz.SUPTITLE_PAD,
    )
    leg = ax.legend(loc="upper right", fontsize=8,
                    facecolor="#1a1a1a", edgecolor="#555555",
                    labelcolor=fg)
    leg.get_frame().set_alpha(0.9)

    path = _save(ctx, fig, "input_gamut_compression_smoothness")

    return Result(
        name="input_gamut_compression_smoothness",
        summary={
            "active": spec.active,
            "algorithm": spec.algorithm,
            "knee_threshold": float(spec.knee[0]),
            "knee_limit": float(spec.knee[1]),
            "knee_power": float(spec.knee[2]),
            "probe_radius": float(radius),
            "probe_samples": int(n_samples),
            "worst_step": worst_step,
            "median_step": median_step,
            "worst_over_median_step": ratio,
            "reference_illuminant": ref_illuminant,
        },
        figure_path=path,
        units="",
        interpretation=(
            "A ring of input chromaticities around the film reference "
            "illuminant runs through the compression and emerges as a "
            "rainbow ring of output points. A smooth, evenly-spaced "
            "ring means the compression is hue-uniform; bunching or "
            "color jumps reveal hue discontinuities that would translate "
            "into banding in the baked LUT. `worst_over_median_step` "
            "near 1 is ideal; >>1 indicates a kink. Informational only — "
            "no pass/fail."
        ),
        passed=None,
    )


# ---------------------------------------------------------------------------
# The ordered default test list.
# ---------------------------------------------------------------------------
