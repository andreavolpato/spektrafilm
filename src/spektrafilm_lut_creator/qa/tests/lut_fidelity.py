"""LUT-fidelity QA tests — does the cube preserve the pipeline within industry tolerance."""

from __future__ import annotations

from typing import TYPE_CHECKING

import colour
import numpy as np

from spektrafilm_lut_creator.qa import evaluators, metrics, patterns, viz
from spektrafilm_lut_creator.qa.result import Result
from spektrafilm_lut_creator.qa.tests._helpers import _save

if TYPE_CHECKING:
    from spektrafilm_lut_creator.qa.suite import QAContext


def off_grid_identity(ctx: QAContext) -> Result:
    """Off-grid ΔE₀₀ between the LUT (trilinear + tetrahedral) and the
    live spektrafilm pipeline.

    The single most load-bearing test in the suite. The exact-grid test
    (already in M4) verifies the bake is self-consistent at corners;
    this test verifies what real users will see — the LUT applied at
    off-grid positions via the same interpolation methods Resolve,
    Nuke, FFmpeg, and OBS actually use.

    Industry tolerances (SDR, CIEDE2000): ``max ≤ 2.0`` and
    ``p99 ≤ 1.0`` for both trilinear and tetrahedral. ΔE₀₀ is the
    metric colorists reference daily; ΔITP (BT.2124) is also computed
    and reported but the pass criterion is ΔE₀₀.

    References
    ----------
    - CIE 142:2001 (CIEDE2000) — the workhorse perceptual metric.
    - ITU-R BT.2124 — HDR-aware perceptual color difference (ΔITP).
    - Kirk, *Tetrahedral Interpolation* (FilmLight Truelight whitepapers).
    - OCIO ``ociochecklut`` — the reference implementation we mirror.
    """
    ref = ctx.reference
    table = ctx.lut.table
    out_cs = ctx.spec.output_color_space

    lut_out_tri = evaluators.apply_trilinear(table, ref.rng_samples_encoded)
    lut_out_tet = evaluators.apply_tetrahedral(table, ref.rng_samples_encoded)

    de_tri = metrics.delta_e_2000(
        lut_out_tri, ref.pipeline_out_encoded, output_color_space=out_cs
    )
    de_tet = metrics.delta_e_2000(
        lut_out_tet, ref.pipeline_out_encoded, output_color_space=out_cs
    )
    stats_tri = metrics.summary_stats(de_tri)
    stats_tet = metrics.summary_stats(de_tet)

    # Secondary ΔITP for HDR-side comparability; not used for pass/fail
    # while bundles are SDR-dominant.
    itp_tri = metrics.delta_itp(
        lut_out_tri, ref.pipeline_out_encoded, output_color_space=out_cs
    )
    stats_itp = metrics.summary_stats(itp_tri)

    summary = {
        "trilinear_dE2000_max": stats_tri["max"],
        "trilinear_dE2000_p99": stats_tri["p99"],
        "trilinear_dE2000_p50": stats_tri["p50"],
        "tetrahedral_dE2000_max": stats_tet["max"],
        "tetrahedral_dE2000_p99": stats_tet["p99"],
        "tetrahedral_dE2000_p50": stats_tet["p50"],
        "trilinear_dITP_max": stats_itp["max"],
        "trilinear_dITP_p99": stats_itp["p99"],
    }
    passed = bool(
        stats_tri["max"] <= 2.0
        and stats_tri["p99"] <= 1.0
        and stats_tet["max"] <= 2.0
        and stats_tet["p99"] <= 1.0
    )

    fig = viz.offgrid_error_scatter(
        ref.rng_samples_encoded,
        de_tri,
        title=(
            f"Off-grid ΔE₀₀ (trilinear) — max={stats_tri['max']:.3f}, "
            f"p99={stats_tri['p99']:.3f}"
        ),
        cbar_label="ΔE₀₀",
    )
    path = _save(ctx, fig, "off_grid_identity")

    return Result(
        name="off_grid_identity",
        summary=summary,
        scalar_field=de_tri,
        figure_path=path,
        units="ΔE₀₀",
        interpretation=(
            "ΔE₀₀ measures perceptual error between the LUT's prediction "
            "(via interpolation in the host's mode — trilinear or "
            "tetrahedral) and the live pipeline at off-grid positions. "
            "Above the visibility threshold users will see interpolation "
            "artifacts the on-grid test cannot detect; remedies are "
            "higher LUT resolution or wire-shaping changes. ΔITP is "
            "reported as a secondary, HDR-aware companion metric."
        ),
        reference_values={
            "trilinear_dE2000_max": "≤ 2.0 — perceptual visibility threshold for graphics work",
            "trilinear_dE2000_p99": "≤ 1.0 — interpolation-quality target across the cube",
            "tetrahedral_dE2000_max": "≤ 2.0 — same threshold under tetrahedral interpolation",
            "tetrahedral_dE2000_p99": "≤ 1.0 — same target under tetrahedral interpolation",
        },
        passed=passed,
    )


def monotonicity(ctx: QAContext) -> Result:
    """Diagonal axes of the cube must be non-decreasing in their
    matching output channel.

    A negative finite-difference is a fold-back: increasing R input
    decreased R output, which produces non-invertible regions that
    confound grading. Off-diagonal non-monotonicity (e.g. green-in-red
    curve) can be physically legitimate from DIR couplers or crosstalk
    in print chemistry, so we don't count those.

    The cube-wide violation count
    (``metrics.monotonicity_violations``) is the master pass/fail
    statistic. The figure visualizes one informative centerline:
    each panel sweeps one channel with the other two held at the
    input color space's **middle-gray-encoded** value (linear 0.18
    encoded through the input CCTF). This is more honest than the
    cube-midpoint sweep for log-encoded inputs — for V-Log the
    encoded midpoint (0.5) corresponds to mid-bright, not gray;
    pinning at the true middle-gray-encoded position (≈0.42 for
    V-Log, ≈0.46 for sRGB, 0.18 for ACEScg) gives a centerline that
    actually represents the channel's behavior at a neutral gray.

    References
    ----------
    - OCIO v2 design notes on monotonic LUT structure.
    - FilmLight Truelight whitepapers.
    """
    from spektrafilm_lut_creator.color_spaces import encode_cctf

    table = ctx.lut.table
    info = metrics.monotonicity_violations(table)

    # Midgray-encoded for the LUT *as configured*: the BakeFrame's
    # input_midgray_linear is 0.18 / input_gain — for an HDR PQ bundle
    # it lands at 100 nits, not at 0.18 linear (which sits 9 stops
    # below midgray in PQ's container).
    mid_gray_linear = np.full((1, 3), ctx.frame.input_midgray_linear, dtype=float)
    mid_gray_encoded = encode_cctf(mid_gray_linear, ctx.frame.input_color_space)
    pin = float(np.asarray(mid_gray_encoded).flatten()[0])

    # Sweep each axis at the middle-gray centerline. Density-65 sampling
    # gives a finer curve than the cube's native resolution; trilinear
    # interpolation on the LUT smooths small cube-grid quantization
    # artifacts so the visible violations reflect real fold-backs,
    # not float-precision jitter between cube cells.
    n_samples = 65
    sweep = np.linspace(0.0, 1.0, n_samples)
    pin_arr = np.full(n_samples, pin)
    sweep_inputs = {
        "R": np.stack([sweep, pin_arr, pin_arr], axis=-1),
        "G": np.stack([pin_arr, sweep, pin_arr], axis=-1),
        "B": np.stack([pin_arr, pin_arr, sweep], axis=-1),
    }
    sweep_outputs = tuple(
        evaluators.apply_trilinear(table, np.asarray(samples, dtype=np.float32))
        for samples in sweep_inputs.values()
    )

    # Per-panel violation masks: where the matching output channel's
    # finite-diff is negative along the swept axis.
    masks = (
        np.diff(sweep_outputs[0][:, 0]) < 0.0,
        np.diff(sweep_outputs[1][:, 1]) < 0.0,
        np.diff(sweep_outputs[2][:, 2]) < 0.0,
    )
    centerline_violations = int(sum(int(m.sum()) for m in masks))

    pin_label = f"{pin:.3f} (mid-gray encoded)"
    fig = viz.transfer_curves(
        sweep,
        sweep_outputs,
        pin_label=pin_label,
        violation_marks=masks,
        suptitle=(
            f"Per-axis transfer curves through middle-gray "
            f"({ctx.spec.input_color_space} encoded {pin:.3f})"
        ),
    )
    path = _save(ctx, fig, "monotonicity")

    passed = info["violations"] == 0
    return Result(
        name="monotonicity",
        summary={
            "violations": int(info["violations"]),
            "worst_negative_diff": float(info["worst_negative_diff"]),
            "centerline_pin_encoded": pin,
            "centerline_violations": centerline_violations,
        },
        figure_path=path,
        units="cells",
        interpretation=(
            "Each diagonal axis-channel pair (R-in vs R-out, etc.) "
            "must be monotonic for the LUT to be invertible without "
            "fold-backs. The `violations` count is cube-wide; the "
            "figure visualizes the centerline sweep through "
            "middle-gray-encoded for honest comparison across input "
            "color spaces (without this, log inputs like V-Log would "
            "be evaluated through their encoded midpoint, which is "
            "mid-bright rather than gray and produces visually "
            "confusing curve shapes). Violations on the centerline "
            "indicate either a model regime that legitimately produces "
            "a fold (DIR couplers in shadows, gamut compression at "
            "the saturation knee) or a bake artifact at the cube "
            "boundary; investigate both before relaxing the test."
        ),
        reference_values={
            "violations": "== 0 — any cube-wide fold-back is a hard invertibility break",
            "worst_negative_diff": "== 0.0 when violations == 0; a tiny negative (≈ -1e-5) on the centerline is bake jitter rather than a real fold",
        },
        passed=passed,
    )


def jacobian_condition(ctx: QAContext) -> Result:
    """Local 3×3 Jacobian condition number — a smoothness diagnostic.

    Gamut compression and density shoulders produce regions where the
    local linear approximation of the transform is near-singular
    (long thin parallelepipeds in output space). Healthy cube cells
    have log-cond ~ O(1); pathological cells climb above 3 (cond ~
    1000), signaling visible artifacts.

    References
    ----------
    - Siragusano, *The Beauty of Per-Pixel Math* (FilmLight, Vimeo).
    - Hable, filmicworlds.com.
    """
    table = ctx.lut.table
    n = ctx.lut.resolution
    field = metrics.local_jacobian_log_cond(table)
    stats = metrics.summary_stats(field)
    fig = viz.jacobian_condition_3d(field, n)
    path = _save(ctx, fig, "jacobian_condition")

    return Result(
        name="jacobian_condition",
        summary={
            "max_log10_cond": stats["max"],
            "p99_log10_cond": stats["p99"],
            "p50_log10_cond": stats["p50"],
        },
        scalar_field=field,
        figure_path=path,
        units="log10(cond J)",
        interpretation=(
            "Where the cube colors locally compress onto a near-curve "
            "(e.g., the highlight shoulder collapsing chroma), log-cond "
            "rises sharply. Shape of the high-cond region matters more "
            "than its absolute value — a thin shell near the gamut "
            "boundary is expected; a fat interior region is suspicious."
        ),
        passed=None,  # informational — no hard threshold
    )


def total_variation(ctx: QAContext) -> Result:
    """Per-axis total variation + axial-FFT high-band energy.

    A noisy bake (NaN propagation, numerical instability, bad
    chemistry models) lifts these. Reported informational — typical
    spektrafilm bundles will need baselines before this gates CI.
    """
    table = ctx.lut.table
    tv = metrics.total_variation(table)
    fft = metrics.axial_fft_highband_ratio(table)
    summary = {**tv, **fft}
    fig = viz.output_histograms(ctx.grid_output)
    path = _save(ctx, fig, "total_variation")

    return Result(
        name="total_variation",
        summary=summary,
        figure_path=path,
        units="",
        interpretation=(
            "Total variation is the mean absolute finite-difference of "
            "the cube table — a smoothness scalar. The axial-FFT "
            "high-band ratio adds spectral-domain evidence: a bake with "
            "banding shows lifted energy in the upper half of the "
            "axial spectrum. The histogram plot is a sanity check on "
            "clipping incidence at 0 and 1."
        ),
        passed=None,
    )


def output_gamut_compression(ctx: QAContext) -> Result:
    """Detect cube-face folds, report gamut compression ratio, and
    visualize the output gamut before/after compression.

    Combines two diagnostics that share the same underlying data
    (LUT cube + an unbounded re-run of the simulation):

    1. **Fold-back metric** — `metrics.gamut_self_intersection_score`
       counts cube-face triangles that flip orientation. Any fold is a
       hard non-invertibility and fails the test.
    2. **Hull volume ratio** — output OkLab convex-hull volume divided
       by input OkLab convex-hull volume. < 1 expected (LUTs compress);
       > 1 means expansion (suspect).
    3. **Gamut compression preview** — a 1x2 figure with the LUT's
       compressed gamut volume (faint cube cloud) and the
       unbounded→compressed rim envelope, shown in OkLab (left) and
       xy chromaticity (right). The xy panel duplicates what the
       standalone preview test used to render; merging the two avoids
       two separate unbounded-pipeline runs.

    References
    ----------
    - ACES Reference Gamut Compression test imagery.
    - Morovic, gamut-mapping CIC papers.
    - spektrafilm-research n110 (output compression design).
    - ACES Reference Gamut Compression v1.3 (AMPAS, 2020).
    """
    from spektrafilm.utils.gamut_compression import compress_rgb
    from spektrafilm_lut_creator.color_spaces import get as _get_cs

    table = ctx.lut.table
    flips = metrics.gamut_self_intersection_score(table)
    hull = metrics.gamut_hull_volume_ratio(
        ctx.grid_input,
        ctx.grid_output,
        ctx.spec.output_color_space,
    )

    # Rim — saturated cube edges — and its unbounded pipeline output.
    rim_samples, rim_segments = patterns.saturated_cube_edges(n=96)
    out_cs_name = ctx.spec.output_color_space
    out_primaries_name = _get_cs(out_cs_name).primaries
    compression_spec = ctx.spec.output_gamut_compress

    rim_unbounded = _run_unbounded_pipeline_for_rim(ctx, rim_samples)
    rim_compressed = (
        compress_rgb(
            rim_unbounded, compression_spec, output_color_space=out_primaries_name
        )
        if compression_spec.active
        else rim_unbounded.copy()
    )

    hsv = np.asarray(colour.RGB_to_HSV(rim_samples), dtype=float)
    rim_hues = hsv[..., 0]
    n_per_seg = len(rim_segments[0])
    n_segments = len(rim_segments)

    # Stats for the merged summary (mirror the old preview's numbers).
    ach = rim_unbounded.max(axis=-1)
    bright_mask = ach > 1e-2
    safe_ach = np.where(ach > 1e-6, ach, 1.0)
    d_max = ((ach[..., None] - rim_unbounded) / safe_ach[..., None]).max(axis=-1)
    oog_mask = (d_max > 1.0) & bright_mask
    oog_fraction = float(oog_mask.sum() / max(int(bright_mask.sum()), 1))
    rim_disp = np.linalg.norm(rim_unbounded - rim_compressed, axis=-1)

    summary = {
        "fold_triangles": int(flips["flips"]),
        "fold_fraction": float(flips["fraction"]),
        "input_hull_volume": float(hull["input_hull_volume"]),
        "output_hull_volume": float(hull["output_hull_volume"]),
        "compression_ratio": float(hull["compression_ratio"]),
        "compression_algorithm": compression_spec.algorithm,
        "rim_oog_fraction": oog_fraction,
        "rim_oog_samples": int(oog_mask.sum()),
        "rim_max_displacement": float(rim_disp[oog_mask].max())
        if oog_mask.any()
        else 0.0,
        "rim_mean_displacement": float(rim_disp[oog_mask].mean())
        if oog_mask.any()
        else 0.0,
    }
    # Hard failure when face folds appear. Compression ratio > 1.05 is
    # suspicious (rare expansion); < 0.05 is suspicious (extreme
    # collapse). Rim displacement/OOG is informational only.
    passed = flips["flips"] == 0 and 0.05 <= hull["compression_ratio"] <= 1.05

    fig = viz.gamut_compression_3d_xy(
        grid_output_compressed=ctx.grid_output,
        rim_unbounded_linear=rim_unbounded,
        rim_compressed_linear=rim_compressed,
        rim_input_hues=rim_hues,
        rim_n_per_segment=n_per_seg,
        rim_n_segments=n_segments,
        in_cs_name=ctx.spec.input_color_space,
        out_cs_name=out_cs_name,
        compression_spec=compression_spec,
    )
    path = _save(ctx, fig, "output_gamut_compression")

    return Result(
        name="output_gamut_compression",
        summary=summary,
        figure_path=path,
        units="",
        interpretation=(
            "Face folds mean the cube surface maps onto itself — a "
            "non-invertible region that breaks grading. The compression "
            "ratio quantifies how much perceptual volume the LUT throws "
            "away; numbers in [0.05, 1.05] are normal, outside means "
            "either degenerate output (very small ratio) or unexpected "
            "expansion (ratio > 1). The figure's left panel shows the "
            "LUT's compressed gamut in OkLab (faint cube cloud) with "
            "the unbounded rim (solid colored lines) and the compressed "
            "rim (dashed) overlaid; the right panel is the canonical "
            "xy-chromaticity preview of the same compression event."
        ),
        reference_values={
            "fold_triangles": "== 0 — any cube-face fold is a hard non-invertibility",
            "compression_ratio": "in [0.05, 1.05] — < 0.05 is extreme collapse, > 1.05 is unexpected expansion",
        },
        passed=passed,
    )


# ---------------------------------------------------------------------------
# Model diagnostic.
# ---------------------------------------------------------------------------


def _run_unbounded_pipeline_for_rim(
    ctx: QAContext,
    samples_encoded: np.ndarray,
) -> np.ndarray:
    """Run a one-off pipeline with output gamut compression *off* to
    capture the simulation's unbounded reach in output-primaries linear
    RGB.

    The shipped LUT bake has compression baked in (via scanning.py);
    that's the "after" state. To visualize the "before" we need to
    re-run the same pipeline once with compression disabled. Per-call
    cost is small (cube-rim sample count, hundreds of samples).
    """
    from spektrafilm.runtime.params_builder import digest_params, init_params
    from spektrafilm.runtime.pipeline import SimulationPipeline
    from spektrafilm.utils.gamut_compression import OutputGamutCompressSpec
    from spektrafilm_lut_creator.color_spaces import (
        decode_cctf,
    )
    from spektrafilm_lut_creator.color_spaces import (
        get as get_color_space,
    )

    spec = ctx.spec
    in_entry = get_color_space(spec.input_color_space)
    out_entry = get_color_space(spec.output_color_space)
    print_profile = (
        ctx.bundle.meta.stocks.prints[ctx.print_index]
        if ctx.bundle.meta.stocks
        else spec.print_profiles[ctx.print_index]
    )

    params = init_params(film_profile=spec.film_profile, print_profile=print_profile)
    params.debug.lut_mode = True
    params.io.input_color_space = in_entry.primaries
    params.io.output_color_space = out_entry.primaries
    params.io.input_cctf_decoding = False
    params.io.output_cctf_encoding = False
    # Disable output gamut compression so we can see the simulation's
    # unbounded reach. The shipping bake always has output compression
    # engaged; this is QA-only.
    params.io.input_gamut_compress = spec.input_gamut_compress
    params.io.output_gamut_compress = OutputGamutCompressSpec(algorithm="off")
    params = digest_params(params)
    pipeline = SimulationPipeline(params)

    samples_linear = decode_cctf(samples_encoded, spec.input_color_space)
    image_in = samples_linear.reshape(1, -1, 3).astype(np.float32)
    image_out = np.asarray(pipeline.process(image_in), dtype=float)
    return image_out.reshape(-1, 3)


# ---------------------------------------------------------------------------
# Noise sensitivity + gamut edge stress + R-G plane slices.
# ---------------------------------------------------------------------------
