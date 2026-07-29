"""Model-diagnostic QA tests — does the spektrafilm pipeline itself produce sensible output."""

from __future__ import annotations

from typing import TYPE_CHECKING

import colour
import matplotlib.pyplot as plt
import numpy as np

import spektrafilm_lut_creator.color_spaces as color_spaces
from spektrafilm_lut_creator.color_spaces import to_xyz_qa
from spektrafilm_lut_creator.qa import evaluators, metrics, patterns, viz
from spektrafilm_lut_creator.qa.result import Result
from spektrafilm_lut_creator.qa.tests._helpers import _save

if TYPE_CHECKING:
    from spektrafilm_lut_creator.qa.suite import QAContext


def characteristic_curve(ctx: QAContext) -> Result:
    """Pipeline response in the density domain — coupler diagnostic.

    Reads complementary to ``monotonicity``: that test measures
    "is the cube invertible?" along the centerline; this one shows
    "how do the channels interact?" by sweeping each channel against
    several constant values of the other two ("pins") and overlaying
    the resulting density curves.

    Pins are picked so the **off-diagonal curves enter each panel at
    target output densities** ``[0.2, 0.4, 0.6, 0.8, 1.0]``. Concretely:
    we build a fine neutral (R=G=B=t) characteristic curve from the
    LUT, invert it once to find the input codes ``t`` whose neutral
    output density equals each target D, and pin the non-swept channels
    at those codes. So in the R panel, at sweep_x=0, the G and B curves
    start near D=0.2/0.4/0.6/0.8/1.0 — the legend value matches what
    you see on the Y-axis at the left edge of the panel. (Not exact —
    pulling the swept channel to 0 perturbs the off-diagonal output via
    chemistry coupling — but well within "about", which is the point of
    the visualization.)

    The vertical spread between same-color curves at different pins is
    the **DIR coupler signature**: if the developer-inhibitor couplers
    in the film simulation are active, pushing one channel's pin level
    shifts the other channels' density response visibly; an uncoupled
    simulation would show all same-color curves stacked.

    The bottom-right panel keeps the canonical neutral (R=G=B) curve
    — the classic film D-vs-input characteristic from datasheets.

    Uses trilinear interpolation on the LUT (the LUT already encodes
    the on-grid pipeline response; we sample at the pins which may not
    align with the cube grid).

    References
    ----------
    - Hunt, *The Reproduction of Colour* — characteristic curves.
    - Any film stock datasheet (Kodak, Fuji) — the canonical D vs
      log E shape this plot is patterned on.
    """
    table = ctx.lut.table
    n = ctx.lut.resolution

    # Build the neutral characteristic curve at fine resolution and
    # invert it to find the pin input codes whose neutral output density
    # equals each target. Use the mean of the three output channels as
    # the density reference — for a calibrated neutral the three channels
    # track closely, and the mean is robust to small per-channel
    # divergence.
    pin_densities = (0.2, 0.4, 0.6, 0.8, 1.0)
    alphas = (0.3, 0.6, 1.0, 0.6, 0.3)
    n_neutral_probe = 257
    t_probe = np.linspace(0.0, 1.0, n_neutral_probe).astype(np.float32)
    neutral_probe_in = np.stack([t_probe, t_probe, t_probe], axis=-1)
    neutral_probe_out = np.asarray(
        evaluators.apply_trilinear(table, neutral_probe_in),
        dtype=float,
    )
    neutral_d = -np.log10(
        np.clip(np.mean(neutral_probe_out, axis=-1), 1e-4, 1.0),
    )
    # np.interp wants x ascending; density is monotonically decreasing in
    # t (more input → less density on a print), so flip both vectors.
    pin_codes = tuple(
        float(np.interp(d, neutral_d[::-1], t_probe[::-1].astype(float)))
        for d in pin_densities
    )

    # Trilinear-sampled sweeps at finer-than-cube resolution; gives
    # smooth curves even at low cube resolutions and lets us pin the
    # non-swept channels at exactly the requested values rather than
    # snapping to cube cells.
    n_samples = 65
    sweep_x = np.linspace(0.0, 1.0, n_samples)
    r_sweep_data: list[tuple[float, np.ndarray, float]] = []
    g_sweep_data: list[tuple[float, np.ndarray, float]] = []
    b_sweep_data: list[tuple[float, np.ndarray, float]] = []
    for pin_d, pin_code, alpha in zip(pin_densities, pin_codes, alphas):
        pin_arr = np.full(n_samples, pin_code)
        r_in = np.stack([sweep_x, pin_arr, pin_arr], axis=-1).astype(np.float32)
        g_in = np.stack([pin_arr, sweep_x, pin_arr], axis=-1).astype(np.float32)
        b_in = np.stack([pin_arr, pin_arr, sweep_x], axis=-1).astype(np.float32)
        # The pin value in the data tuple is the *density* — used for
        # legend labels in viz.density_transfer_curves so the label
        # units match the panel's Y-axis.
        r_sweep_data.append((pin_d, evaluators.apply_trilinear(table, r_in), alpha))
        g_sweep_data.append((pin_d, evaluators.apply_trilinear(table, g_in), alpha))
        b_sweep_data.append((pin_d, evaluators.apply_trilinear(table, b_in), alpha))

    # Neutral R=G=B sweep — the canonical D-vs-input curve.
    neutral_in = np.stack([sweep_x, sweep_x, sweep_x], axis=-1).astype(np.float32)
    neutral_samples = evaluators.apply_trilinear(table, neutral_in)

    fig = viz.density_transfer_curves(
        sweep_x,
        r_sweep_data,
        g_sweep_data,
        b_sweep_data,
        neutral_samples,
    )
    path = _save(ctx, fig, "characteristic_curve")

    # Quantify the system gamma at the midpoint of the neutral ramp:
    # slope of log10(output) vs log10(input) at index n//2.
    mid = n // 2
    axis_codes = np.linspace(1e-6, 1.0, n)
    neutral_out = np.array([table[i, i, i, :] for i in range(n)])
    log_in = np.log10(axis_codes)
    log_out = np.log10(np.clip(np.mean(neutral_out, axis=-1), 1e-4, 1.0))
    # Local slope at mid via central difference.
    gamma_mid = float(
        (log_out[mid + 1] - log_out[mid - 1]) / (log_in[mid + 1] - log_in[mid - 1])
    )

    # Channel divergence at mid: how far the three CMY densities spread.
    densities = -np.log10(np.clip(neutral_out, 1e-4, 1.0))
    spread = float(np.max(np.ptp(densities, axis=-1)))

    return Result(
        name="characteristic_curve",
        summary={
            "system_gamma_at_mid": gamma_mid,
            "max_channel_density_spread": spread,
        },
        figure_path=path,
        units="density",
        interpretation=(
            "The system's response to neutral input should be smooth, "
            "with the three channels tracking each other (small spread). "
            "Big channel divergence on a neutral ramp is a calibration "
            "or chemistry-model bug, not a LUT bug; check the print "
            "chemistry's neutral handling."
        ),
        passed=None,
    )


def planckian_sweep(ctx: QAContext) -> Result:
    """Pipeline response to white surfaces under daylight illuminants.

    A spektrafilm bundle should send "white under D55", "white under
    D65", "white under D75", etc. to a smooth, monotonic curve in
    output chromaticity. Kinks or fold-backs reveal white-balance
    handling bugs.

    References
    ----------
    - CIE 15:2018 (daylight illuminants).
    - Poynton, *Color FAQ* — white-point handling.
    """
    spec = ctx.spec
    samples_encoded, cct = patterns.planckian_sweep(
        spec.input_color_space,
        n=16,
        exposure_ev=ctx.frame.exposure_ev,
    )

    # Apply the LUT (cheap) — that's what users will see.
    lut_out_encoded = evaluators.apply_trilinear(ctx.lut.table, samples_encoded)
    out_xyz = to_xyz_qa(lut_out_encoded, spec.output_color_space)
    out_xy = np.asarray(colour.XYZ_to_xy(out_xyz), dtype=float)

    # Smoothness: max angular deviation of consecutive sweep segments
    # from a straight line through the cloud (a sanity proxy for
    # monotone smoothness without imposing a specific curve shape).
    diffs = np.diff(out_xy, axis=0)
    norms = np.linalg.norm(diffs, axis=1) + 1e-12
    cos_theta = np.sum(diffs[:-1] * diffs[1:], axis=1) / (norms[:-1] * norms[1:])
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    bend_angle_deg = np.degrees(np.arccos(cos_theta))
    max_bend = float(bend_angle_deg.max()) if bend_angle_deg.size else 0.0

    locus_xy = patterns.spectral_locus_chromaticities()
    fig = viz.planckian_path(cct, out_xy, locus_xy, spec.output_color_space)
    path = _save(ctx, fig, "planckian_sweep")

    # > 30 deg between consecutive segments on a daylight sweep is
    # surprising; pure monotone smoothness would give ~0 deg.
    passed = max_bend <= 30.0

    return Result(
        name="planckian_sweep",
        summary={
            "max_bend_angle_deg": max_bend,
            "cct_range_k": f"{int(cct[0])}-{int(cct[-1])}",
        },
        figure_path=path,
        units="degrees",
        interpretation=(
            "White points across the daylight CCT range should map to a "
            "smooth curve in output chromaticity. Sharp bends suggest "
            "the model is doing something discontinuous to chromatic "
            "adaptation — worth investigating the scan illuminant "
            "handling and the print's spectral response curves."
        ),
        reference_values={
            "max_bend_angle_deg": "≤ 30° — daylight CCT sweep should map to a smooth curve; pure monotone smoothness gives ~0°",
        },
        passed=passed,
    )


def hue_twist_oklab(ctx: QAContext) -> Result:
    """Maximum hue rotation per saturation band, in OkLab.

    Reported as **informational** for v1: spektrafilm is a film
    simulation and film simulations legitimately rotate hue
    (yellow-green shift, red darkening, etc.) — a "pass/fail"
    threshold without per-stock baselines is just noise. The numbers
    are meant for tracking drift vs. previous bakes, and once the
    baselines work from n080 §10 ships, this becomes a gated test.

    Filtering: input samples with OkLab chroma > 0.6 are dropped
    (they lie outside the visible spectral locus — V-Gamut and
    ProPhoto primaries extend there — and don't have meaningful hue
    coordinates).

    References
    ----------
    - Ottosson, *OkLab*, https://bottosson.github.io/posts/oklab/.
    - Yedlin, *Display Prep Demo*, yedlin.net.
    - Sobotka, *AgX*, github.com/sobotka/AgX.
    """
    in_cs = ctx.spec.input_color_space
    out_cs = ctx.spec.output_color_space
    lab_in_all = viz._to_oklab(ctx.grid_input, in_cs)
    lab_out_all = viz._to_oklab(ctx.grid_output, out_cs)

    # Drop out-of-locus inputs (V-Gamut, ProPhoto extremes, etc.).
    # OkLab chroma > 0.6 corresponds to colors well outside any
    # physically realizable gamut; hue at those coordinates is an
    # extrapolation artifact rather than a meaningful measurement.
    c_in_all = np.sqrt(lab_in_all[:, 1] ** 2 + lab_in_all[:, 2] ** 2)
    in_locus = c_in_all <= 0.6
    n_filtered = int((~in_locus).sum())
    lab_in = lab_in_all[in_locus]
    lab_out = lab_out_all[in_locus]
    grid_in_filt = ctx.grid_input[in_locus]
    grid_out_filt = ctx.grid_output[in_locus]

    info = dict(metrics.hue_rotation_per_band(lab_in, lab_out))
    info["samples_in_locus"] = int(in_locus.sum())
    info["samples_filtered_out_of_locus"] = n_filtered

    fig = viz.hue_twist_oklab(grid_in_filt, grid_out_filt, in_cs, out_cs)
    path = _save(ctx, fig, "hue_twist_oklab")

    return Result(
        name="hue_twist_oklab",
        summary=info,
        figure_path=path,
        units="degrees",
        interpretation=(
            "Hue rotation as a function of input chroma is the single "
            "thing colorists notice without instrumentation. Some "
            "rotation is part of the look (a film simulation IS a hue "
            "rotation in places); the magnitude is stock-specific. "
            "Use this number to track drift between bakes of the same "
            "stock — large jumps signal a model-side regression. "
            "Per-stock pass/fail thresholds wait on the baselines work "
            "(n080 §10)."
        ),
        passed=None,  # informational; no absolute threshold pre-baselines
    )


def dynamic_range_usage(ctx: QAContext) -> Result:
    """How many input stops does the LUT actually render — in the
    colorist's unit (scene-linear stops above middle gray).

    Generates a neutral ramp uniform in **scene-linear log2 stops**
    across ``[-8, +8]`` EV (range that covers most practical
    cameras), CCTF-encodes it for the input space, clips to the LUT
    input domain ``[0, 1]``, applies the LUT, and decodes the
    output's CCTF to get scene-linear output luminance. The
    resulting D vs log E curve is the **canonical film characteristic
    plot** every film datasheet ships.

    Separates two sources of range loss:

    - **Encoding clip**: stops outside what the input encoding can
      represent (e.g., sRGB caps at ~+2.5 EV above middle gray; V-Log
      reaches +8 EV). Not the LUT's fault.
    - **Toe / shoulder collapse**: stops within the encoded range
      where output slope falls below 0.10 density per stop — the
      LUT's rendering decision to compress shadows or highlights.

    Reported informational for v1 — there's no universal "correct"
    answer for how many stops a film simulation should preserve, but
    knowing the number is a colorist staple.

    References
    ----------
    - Hunt, *The Reproduction of Colour* — characteristic curves.
    - ARRI K1S0-057 LogC whitepaper.
    - ANSI/SMPTE RP 180 (18% middle gray).
    """
    in_cs = ctx.spec.input_color_space
    out_cs = ctx.spec.output_color_space
    entry = color_spaces.get(in_cs)
    # Anchor the exposure ramp at the camera's *true* native middle gray —
    # the linear value the input encoding assigns to an 18% gray card
    # (``entry.midgray_linear``: 0.18 for SDR/log, the reference-white nits
    # for PQ/HLG). NOT ``ctx.frame.input_midgray_linear`` (= 0.18 / input
    # gain), which moves the anchor *with* the LUT's exposure gain and so
    # pre-cancels it: stop 0 then means "whatever input happens to land on
    # film midgray," which silently re-centers any exposure error and makes
    # the curve look correct even when a properly-exposed gray renders dark.
    # With the true anchor, stop 0 IS the camera's 18% gray, so where it
    # lands in output density is an honest exposure readout (see n200).
    native_midgray = float(entry.midgray_linear)
    stops, encoded_in, encoded_clip_mask = patterns.dynamic_range_neutral_ramp(
        in_cs,
        middle_gray_linear=native_midgray,
    )

    # Apply the LUT (already composed if 2-LUT) at the encoded inputs.
    lut_out_encoded = evaluators.apply_trilinear(ctx.lut.table, encoded_in)

    # Decode the output's CCTF to get scene-linear and take the Y
    # (luminance) component via XYZ. `to_xyz_qa` handles the CCTF decode
    # plus primaries-to-XYZ transform AND normalizes HDR outputs back to
    # reflectance scale so the SDR `-log10(Y)` density formula works.
    xyz_out = to_xyz_qa(lut_out_encoded, out_cs)
    y_out = np.asarray(xyz_out[:, 1], dtype=float)
    # Some output spaces' linear Y can dip very slightly negative on
    # extreme gamut edges (numerical) — clip the floor.
    y_out = np.clip(y_out, 1e-6, None)

    stats = metrics.dynamic_range_stats(stops, y_out, encoded_clip_mask)
    # Midgray exposure readout: where does the camera's 18% gray (stop 0)
    # actually render? Ideal is output Y ≈ 0.18 (offset 0). Midgray is
    # pinned by construction (n200), so any offset beyond the film's own
    # rendering of 18% gray is either the spec's deliberate exposure_ev
    # or a bug. Either way: report it loudly.
    midgray_y = float(np.interp(0.0, stops, y_out))
    midgray_offset_stops = float(np.log2(midgray_y / 0.18))
    stats["midgray_output_y"] = midgray_y
    stats["midgray_offset_stops"] = midgray_offset_stops

    fig = viz.dynamic_range_curve(
        stops,
        y_out,
        encoded_clip_mask,
        stats,
        in_cs=in_cs,
        out_cs=out_cs,
    )
    path = _save(ctx, fig, "dynamic_range_usage")

    drift = (
        "on target"
        if abs(midgray_offset_stops) <= 0.25
        else f"{midgray_offset_stops:+.2f} stops off"
    )
    return Result(
        name="dynamic_range_usage",
        summary=stats,
        figure_path=path,
        units="stops",
        interpretation=(
            f"Midgray exposure: the camera's 18% gray renders at "
            f"{midgray_offset_stops:+.2f} stops vs the ideal 0.18 output "
            f"({drift}). Midgray is pinned by construction, so this should "
            f"be ≈0 (plus the film's own rendering of 18% gray) unless the "
            f"spec bakes a deliberate exposure_ev — anything else means "
            f"the LUT is re-exposing midgray.\n"
            "The 'active rendering range' is how many input stops the "
            "LUT distinguishes — slope above 0.10 D/stop. Below that "
            "threshold, an input stop change barely moves the output, "
            "so the stop is effectively collapsed. The 'input encoding "
            "range' is a property of the input color space (sRGB ~2.5 "
            "EV above middle gray, V-Log ~8 EV), not the LUT. Toe and "
            "shoulder collapsed stops sit *within* the encoded range — "
            "they're rendering decisions, not encoding limits."
        ),
        passed=None,
    )


def spectral_locus_envelope(ctx: QAContext) -> Result:
    """Full chromaticity map of the LUT cube — every cube cell as a
    dot in xy, colored by its actual output RGB.

    The shipped LUT already contains the output color for every input
    cube cell. Projecting all of them to xy and coloring each dot by
    its own rendered RGB shows the simulation's complete chromaticity
    footprint at once: where in xy the LUT maps colors, how densely
    each region is sampled, and what color you'd actually see at each
    location.

    Density is conveyed by alpha blending: small markers at low alpha
    accumulate visually in regions where many cube cells project to
    similar chromaticities (the dye-gamut "shoulders" and the
    achromatic core), and fade to single dots in the sparse rim.

    Replaces the older rim-only envelope plot — the rim envelope's
    "where does the gamut reach" role is now covered by the right
    panel of ``output_gamut_compression``; this plot answers the
    complementary "what does the full LUT *look* like in xy" question.

    References
    ----------
    - Mansencal (@KelSolaar), colour-science visualizations.
    - ACES Reference Gamut Compression.
    """
    out_cs = ctx.spec.output_color_space

    # 1) Every cube cell as a sample, flattened to (N³, 3).
    cube = ctx.lut.table  # encoded output RGB, shape (N, N, N, 3) in [0, 1]
    n = cube.shape[0]
    rgb_encoded = np.asarray(cube, dtype=float).reshape(-1, 3)

    # 2) Project each cell to xy in the output color space's frame.
    #    to_xyz_qa takes encoded RGB and handles the CCTF + matrix,
    #    normalizing HDR output back to reflectance scale.
    xyz = to_xyz_qa(rgb_encoded, out_cs)
    xy = np.asarray(colour.XYZ_to_xy(xyz), dtype=float)

    # 3) Skip degenerate (near-black) samples whose xy is unreliable.
    Y = np.asarray(xyz[:, 1], dtype=float)
    valid = (Y > 1e-4) & np.all(np.isfinite(xy), axis=-1)

    # 4) Output primaries + white for the reference frame.
    out_primaries = colour.RGB_COLOURSPACES[
        __import__("spektrafilm_lut_creator.color_spaces", fromlist=["get"])
        .get(out_cs)
        .primaries
    ]
    out_white = np.asarray(out_primaries.whitepoint, dtype=float)
    out_tri = np.asarray(out_primaries.primaries, dtype=float)

    # 5) Spectral locus for the outer reference.
    from spektrafilm.utils.gamut_compression import spectral_locus_xy

    locus = spectral_locus_xy()

    # ``accent`` is the yellow-ish color used for the input-gamut
    # triangle overlay (visible against the dark BG); titles use the
    # shared viz.HI white so they match the rest of the report.
    bg, fg, accent, dim = "#0a0a0a", "#cccccc", "#ffee66", "#888888"

    fig, ax = plt.subplots(figsize=(10, 10), facecolor=bg, layout="constrained")
    ax.set_facecolor(bg)
    for spine in ax.spines.values():
        spine.set_color("#555555")
    ax.tick_params(colors=fg)
    ax.grid(True, alpha=0.08, color=accent)

    # Reference frame — drawn before scatter so dots sit on top.
    ax.plot(
        locus[:, 0],
        locus[:, 1],
        color=dim,
        lw=1.0,
        alpha=0.5,
        label="visible spectral locus",
    )
    locus_fill = plt.Polygon(
        locus,
        closed=True,
        facecolor="#cccccc",
        alpha=0.015,
        edgecolor="none",
    )
    ax.add_patch(locus_fill)

    tri = np.vstack([out_tri, out_tri[:1]])
    ax.plot(tri[:, 0], tri[:, 1], color=fg, lw=1.6, alpha=0.85, label=f"{out_cs} gamut")
    primary_colors = ["#ff5566", "#66ff88", "#5599ff"]
    primary_labels = ["R", "G", "B"]
    for (px, py), pcol, plab in zip(out_tri, primary_colors, primary_labels):
        ax.plot(
            px,
            py,
            "o",
            color=pcol,
            markersize=10,
            markeredgecolor=bg,
            markeredgewidth=1.5,
            zorder=4,
        )
        offset = np.array([px, py]) - out_white
        norm = np.linalg.norm(offset) + 1e-9
        lx, ly = np.array([px, py]) + 0.035 * offset / norm
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
        markersize=9,
        markeredgecolor=bg,
        markeredgewidth=1.2,
        label=f"{out_cs} white",
        zorder=4,
    )

    # The main event — every cube cell as a dot at its xy position,
    # colored by its own encoded output RGB. Two layers:
    #   * a fatter, very-low-alpha layer for a soft "glow" where many
    #     cells overlap
    #   * a tighter, slightly-stronger layer for individual dot legibility
    # Together they read as "rendered color in chromaticity space."
    rgb_color = np.clip(rgb_encoded[valid], 0.0, 1.0)
    xy_valid = xy[valid]

    # Marker sizes / alphas scale gently with cube size so 17³ and 33³
    # look comparable. Empirically chosen.
    n_pts = len(xy_valid)
    if n_pts > 0:
        # Glow layer: large markers, very low alpha → density bloom.
        s_glow = max(8.0, 80.0 / np.sqrt(n / 17.0))
        a_glow = 0.10
        # Dot layer: small markers, moderate alpha → individual cells.
        s_dot = max(2.0, 12.0 / np.sqrt(n / 17.0))
        a_dot = 0.55
        ax.scatter(
            xy_valid[:, 0],
            xy_valid[:, 1],
            c=rgb_color,
            s=s_glow,
            alpha=a_glow,
            edgecolors="none",
            zorder=2.5,
        )
        ax.scatter(
            xy_valid[:, 0],
            xy_valid[:, 1],
            c=rgb_color,
            s=s_dot,
            alpha=a_dot,
            edgecolors="none",
            zorder=3,
        )

    # Stats — quantify how much of the cube ends up where.
    inside_tri = _in_triangle(xy_valid, out_tri)
    n_total = int(n_pts)
    n_inside = int(inside_tri.sum())
    inside_fraction = n_inside / max(n_total, 1)

    # Rim fraction — fraction of valid samples within `on_locus_eps`
    # of the spectral locus polyline. Kept for backwards compatibility
    # with the prior summary dict.
    on_locus_eps = 0.02
    if n_pts > 0:
        dist_to_locus = np.min(
            np.linalg.norm(
                xy_valid[:, None, :] - locus[None, :, :],
                axis=-1,
            ),
            axis=1,
        )
        rim_fraction = float(np.mean(dist_to_locus < on_locus_eps))
    else:
        rim_fraction = 0.0

    text = (
        f"input:     {ctx.spec.input_color_space}\n"
        f"output:    {out_cs}\n"
        f"film:      {ctx.spec.film_profile}\n"
        f"print:     {ctx.print_name}\n"
        f"\n"
        f"cube res:  {n}³ = {n**3} cells\n"
        f"valid:     {n_total} ({n_total / max(n**3, 1):.0%})\n"
        f"inside gamut: {inside_fraction:.1%}\n"
        f"near locus:   {rim_fraction:.1%}"
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
    ax.set_title(
        f"LUT chromaticity map — {ctx.spec.input_color_space} → {out_cs}",
        color=viz.HI,
        fontsize=viz.SUPTITLE_FS,
        pad=viz.SUPTITLE_PAD,
    )

    path = _save(ctx, fig, "spectral_locus_envelope")

    return Result(
        name="spectral_locus_envelope",
        summary={
            "cube_resolution": int(n),
            "cube_cells": int(n**3),
            "valid_cells": int(n_total),
            "inside_output_gamut_fraction": float(inside_fraction),
            "near_locus_fraction": float(rim_fraction),
        },
        figure_path=path,
        units="",
        interpretation=(
            "Every cube cell projected to xy and rendered at its own "
            "output RGB color. Density variations show where the LUT "
            "concentrates color reproduction (achromatic core dense, "
            "saturated rim sparse). `inside_output_gamut_fraction` "
            "near 1.0 confirms output gamut compression is keeping "
            "the simulation inside the output primaries triangle as "
            "intended. The complementary `output_gamut_compression` "
            "figure shows the rim envelope and the compression's "
            "effect explicitly."
        ),
        passed=None,
    )


def _in_triangle(xy: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Vectorized point-in-triangle test for the output primaries triangle."""
    from matplotlib.path import Path as MplPath

    path = MplPath(np.vstack([tri, tri[:1]]))
    return path.contains_points(xy)


# ---------------------------------------------------------------------------
# The ordered default test list.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Input gamut compression diagnostics. Two informational plots shipped in
# every bundle's qa/ folder so colorists can see exactly what the input
# gamut compression is doing for their input space (compression preview)
# and confirm the compression is smooth (circumferential probe). Driven
# by ctx.spec.input_gamut_compress; with active=False both tests still
# produce a figure but it just says "compression disabled".
# ---------------------------------------------------------------------------
