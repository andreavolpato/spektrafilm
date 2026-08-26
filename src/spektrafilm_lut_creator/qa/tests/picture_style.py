"""Picture-style QA diagnostics — noise, gamut edge stress, R-G slices."""

from __future__ import annotations

from typing import TYPE_CHECKING

import colour
import numpy as np

import spektrafilm_lut_creator.color_spaces as color_spaces
from spektrafilm_lut_creator.qa import evaluators, metrics, viz
from spektrafilm_lut_creator.qa.result import Result
from spektrafilm_lut_creator.qa.tests._helpers import MIDGRAY_18_OKLAB_L, _save

if TYPE_CHECKING:
    from spektrafilm_lut_creator.qa.suite import QAContext


def _polar_oklch_input_samples(
    frame,
    *,
    L: float,
    chroma_rings: tuple[float, ...],
    n_hues: int,
    include_neutral: bool = True,
) -> dict:
    """Build a polar OkLCh sample grid at constant L, then convert each
    point into encoded RGB in the frame's input color space.

    Returns only samples that lie inside the input color space's gamut
    (linear RGB in ``[0, 1]``) — out-of-gamut points get dropped rather
    than clipped so that the per-sample Jacobian reflects the requested
    chromaticity instead of a wall-clamped neighbor.
    """
    in_entry = color_spaces.get(frame.input_color_space)
    pts_oklab: list[list[float]] = []
    hue_deg: list[float] = []
    chroma_val: list[float] = []
    ring_idx: list[int] = []
    if include_neutral:
        pts_oklab.append([L, 0.0, 0.0])
        hue_deg.append(0.0)
        chroma_val.append(0.0)
        ring_idx.append(-1)
    hues = np.linspace(0.0, 2.0 * np.pi, n_hues, endpoint=False)
    for ri, c in enumerate(chroma_rings):
        for h in hues:
            pts_oklab.append([L, c * np.cos(h), c * np.sin(h)])
            hue_deg.append(float(np.degrees(h)))
            chroma_val.append(float(c))
            ring_idx.append(ri)
    pts_oklab_arr = np.asarray(pts_oklab, dtype=float)
    xyz = np.asarray(colour.Oklab_to_XYZ(pts_oklab_arr), dtype=float)
    rgb_linear = np.asarray(
        colour.XYZ_to_RGB(
            xyz, colourspace=in_entry.primaries, apply_cctf_encoding=False
        ),
        dtype=float,
    )
    in_gamut = np.all((rgb_linear >= 0.0) & (rgb_linear <= 1.0), axis=-1)
    rgb_linear = rgb_linear[in_gamut]
    rgb_encoded = frame.encode_input(rgb_linear)
    # Keep central differences off the [0,1] boundary so the Jacobian
    # is two-sided everywhere.
    pad = 6e-3
    boundary_ok = np.all(
        (rgb_encoded > pad) & (rgb_encoded < 1.0 - pad),
        axis=-1,
    )
    rgb_encoded = rgb_encoded[boundary_ok]
    keep = np.where(in_gamut)[0][boundary_ok]
    return {
        "input_encoded": rgb_encoded,
        "hue_deg": np.asarray(hue_deg, dtype=float)[keep],
        "chroma": np.asarray(chroma_val, dtype=float)[keep],
        "ring_idx": np.asarray(ring_idx, dtype=int)[keep],
    }


def _noise_gain_heatmap(
    table: np.ndarray,
    frame,
    *,
    L: float,
    extent: float,
    n_grid: int,
    sigma_in_encoded: float,
    eps: float,
) -> dict:
    """Compute σ₁(J) on a dense OkLab a*b* grid at constant L for the
    heatmap panel. Out-of-input-gamut cells return NaN (rendered as
    transparent in matplotlib pcolormesh)."""
    in_entry = color_spaces.get(frame.input_color_space)
    aa = np.linspace(-extent, extent, n_grid)
    bb = np.linspace(-extent, extent, n_grid)
    A, B = np.meshgrid(aa, bb, indexing="xy")
    oklab_grid = np.stack(
        [np.full_like(A, L), A, B],
        axis=-1,
    ).reshape(-1, 3)
    xyz = np.asarray(colour.Oklab_to_XYZ(oklab_grid), dtype=float)
    rgb_linear = np.asarray(
        colour.XYZ_to_RGB(
            xyz, colourspace=in_entry.primaries, apply_cctf_encoding=False
        ),
        dtype=float,
    )
    pad = 6e-3
    in_gamut = np.all((rgb_linear >= 0.0) & (rgb_linear <= 1.0), axis=-1)
    rgb_linear_safe = np.clip(rgb_linear, 0.0, 1.0)
    rgb_encoded = frame.encode_input(rgb_linear_safe)
    rgb_encoded = np.clip(rgb_encoded, pad, 1.0 - pad)
    field = metrics.noise_sensitivity_field(
        table,
        rgb_encoded,
        in_cs=frame.input_color_space,
        out_cs=frame.output_color_space,
        sigma_in_encoded=sigma_in_encoded,
        eps=eps,
    )
    sigma1 = field["sigma1"].copy()
    sigma1[~in_gamut] = np.nan
    return {
        "aa": aa,
        "bb": bb,
        "sigma1_grid": sigma1.reshape(n_grid, n_grid),
        "in_gamut_grid": in_gamut.reshape(n_grid, n_grid),
    }


def _encoded_image_to_srgb_display(
    image_encoded: np.ndarray, cs_name: str
) -> np.ndarray:
    """Convert an encoded RGB image in ``cs_name`` to display-ready sRGB.

    Uses ``to_xyz_qa`` so HDR outputs (PQ / HLG) get normalized back to
    reflectance scale before the XYZ→sRGB conversion — otherwise PQ
    Y values (in nits) blow past 1.0 and the entire preview clips white.
    """
    xyz = color_spaces.to_xyz_qa(
        np.asarray(image_encoded, dtype=float).reshape(-1, 3),
        cs_name,
    )
    srgb = color_spaces.from_xyz(xyz, "sRGB")
    return np.clip(np.asarray(srgb, dtype=float), 0.0, 1.0).reshape(image_encoded.shape)


def _build_oklab_noise_gradient_input(
    frame,
    *,
    width: int,
    height: int,
    rim_scale: float,
) -> tuple[np.ndarray, dict]:
    """Square OkLab hue-cycle gradient for visual noise assessment.

    Horizontal axis cycles hue. Vertical axis runs black → saturated
    hue row → white, with the peak row anchored at the Oklab lightness
    of 18% gray.
    """
    entry = color_spaces.get(frame.input_color_space)
    hue = np.linspace(0.0, 2.0 * np.pi, width, endpoint=False)
    v = np.linspace(0.0, 1.0, height)
    L_rows = np.where(
        v <= 0.5,
        (v / 0.5) * MIDGRAY_18_OKLAB_L,
        MIDGRAY_18_OKLAB_L + ((v - 0.5) / 0.5) * (1.0 - MIDGRAY_18_OKLAB_L),
    )
    sat_weight = 1.0 - np.abs(2.0 * v - 1.0)

    oklab = np.zeros((height, width, 3), dtype=float)
    for row, (L_row, sat_row) in enumerate(zip(L_rows, sat_weight, strict=True)):
        outline, _ = viz.oklab_gamut_slice_outline(
            frame.input_color_space,
            L=float(L_row),
            n_hues=width,
        )
        chroma_max = np.sqrt(outline[:-1, 1] ** 2 + outline[:-1, 2] ** 2)
        chroma = rim_scale * sat_row * chroma_max
        oklab[row, :, 0] = L_row
        oklab[row, :, 1] = chroma * np.cos(hue)
        oklab[row, :, 2] = chroma * np.sin(hue)

    xyz = np.asarray(colour.Oklab_to_XYZ(oklab.reshape(-1, 3)), dtype=float)
    rgb_linear = np.asarray(
        colour.XYZ_to_RGB(xyz, colourspace=entry.primaries, apply_cctf_encoding=False),
        dtype=float,
    ).reshape(height, width, 3)
    in_gamut = np.all((rgb_linear >= 0.0) & (rgb_linear <= 1.0), axis=-1)
    rgb_encoded = frame.encode_input(np.clip(rgb_linear, 0.0, 1.0))
    return np.asarray(rgb_encoded, dtype=float), {
        "in_gamut_fraction": float(in_gamut.mean()),
        "rim_scale": float(rim_scale),
        "mid_L": float(MIDGRAY_18_OKLAB_L),
    }


def noise_sensitivity(ctx: QAContext) -> Result:
    """Propagate isotropic input noise through the LUT and visualize
    how much noise comes out, and *in which direction*.

    Replaces the older ``shot_noise_grid`` test, which showed visible
    grain on swatches but couldn't quantify the result and tended to
    hide chromatic-amplification problems behind row-normalization.
    The classical first-order error-propagation identity

        Σ_out(x) ≈ J(x) · Σ_in · J(x)ᵀ

    with isotropic ``Σ_in = σ²·I`` in encoded input RGB and ``J`` the
    local Jacobian of the LUT-then-OkLab transform gives both the
    *magnitude* (largest singular value ``σ₁(J)``) and the *direction*
    (eigenvectors of the 2×2 a*b* sub-covariance) of the noise
    amplification at any input chromaticity. Orange surfaces with red
    speckle from the LUT show up here as an eccentric ellipse pointing
    toward the red axis at the orange sample — directly diagnostic.

    Three panels:

    - **left**: noise ellipses on OkLab a*b* (24 hues × 4 chroma rings
      + neutral, at a single input luminance), each ellipse drawn at
      ``ellipse_display_scale × 2σ`` and filled with its post-LUT
      color, with hairlines back to the pre-LUT positions.
    - **center**: σ₁(J) heatmap over a dense a*b* grid — the
      worst-case noise gain at every chromaticity.
    - **right**: hue rosette — polar plot of σ_L (luminance noise
      gain) and σ_ab (chroma noise gain) vs input hue at the
      middle-chroma ring.

    Reported informational for v1: per-stock pass/fail thresholds wait
    on baselines (n080 §10), but the headline scalars (``max_sigma1``,
    ``max_anisotropy``, ``max_hue_rotation_under_noise_deg``,
    ``worst_hue_deg``) track drift across bakes.

    References
    ----------
    - Garcia, Prasad, Foi (2020), *The Geometry of Noise in Color and
      Spectral Image Sensors*, Sensors.
      https://pmc.ncbi.nlm.nih.gov/articles/PMC7471994/
    - Wang, Aristova, Hardeberg (2010), *Evaluating the effect of
      noise on 3D LUT-based color transformations*, CGIV.
    - DXOMark color-sensitivity score (CCM noise propagation).
    - MacAdam, *Visual Sensitivities to Color Differences in Daylight*
      (1942) — the canonical "covariance ellipses on chromaticity"
      visual convention this figure adopts.
    """
    in_cs = ctx.spec.input_color_space
    out_cs = ctx.spec.output_color_space

    # Knobs — chosen so the figure works for both narrow-gamut (sRGB)
    # and wide-gamut (V-Gamut, ACEScg) inputs at the Oklab lightness of
    # 18% gray. sigma_in 0.005 ≈ ~7-stop SNR; 1.5× display gain keeps
    # ellipses readable without distorting orientations.
    L_slice = MIDGRAY_18_OKLAB_L
    chroma_rings = (0.07, 0.14, 0.21)
    n_hues = 16
    sigma_in_encoded = 0.005
    ellipse_display_scale = 1.5
    eps = 5e-3
    n_heatmap_grid = 96

    samples = _polar_oklch_input_samples(
        ctx.frame,
        L=L_slice,
        chroma_rings=chroma_rings,
        n_hues=n_hues,
    )
    field = metrics.noise_sensitivity_field(
        ctx.lut.table,
        samples["input_encoded"],
        in_cs=in_cs,
        out_cs=out_cs,
        sigma_in_encoded=sigma_in_encoded,
        eps=eps,
    )

    # Heatmap extent sized from the larger constant-L gamut slice so the
    # heatmap and ellipse panels share their axis limits without the old
    # projected-primary triangle blowing the view out.
    out_slice_oklab, _ = viz.oklab_gamut_slice_outline(out_cs, L=L_slice)
    in_slice_oklab, _ = viz.oklab_gamut_slice_outline(in_cs, L=L_slice)
    extent = (
        float(
            max(
                np.abs(out_slice_oklab[:, 1]).max(),
                np.abs(out_slice_oklab[:, 2]).max(),
                np.abs(in_slice_oklab[:, 1]).max(),
                np.abs(in_slice_oklab[:, 2]).max(),
            )
        )
        * 1.15
    )
    extent = max(extent, 0.30)
    heatmap = _noise_gain_heatmap(
        ctx.lut.table,
        ctx.frame,
        L=L_slice,
        extent=extent,
        n_grid=n_heatmap_grid,
        sigma_in_encoded=sigma_in_encoded,
        eps=eps,
    )

    # Rosette — pull the middle chroma ring out of the sample field.
    if len(chroma_rings) > 0:
        target_ring = len(chroma_rings) // 2
        ring_mask = samples["ring_idx"] == target_ring
        # Sort by hue so the polar line is monotone.
        order = np.argsort(samples["hue_deg"][ring_mask])
        rosette = {
            "hue_deg": samples["hue_deg"][ring_mask][order],
            "sigma_L": field["sigma_L"][ring_mask][order],
            "sigma_ab": field["sigma_ab"][ring_mask][order],
            "output_encoded": field["output_encoded"][ring_mask][order],
        }
    else:
        rosette = {
            "hue_deg": np.array([0.0]),
            "sigma_L": np.array([0.0]),
            "sigma_ab": np.array([0.0]),
            "output_encoded": np.array([[0.0, 0.0, 0.0]]),
        }

    # Summary stats — hue rotation *induced by* the LUT (not by noise;
    # the noise direction information is captured by anisotropy + the
    # ellipses themselves).
    in_lab = field["input_oklab"]
    out_lab = field["output_oklab"]
    h_in = np.degrees(np.arctan2(in_lab[:, 2], in_lab[:, 1]))
    h_out = np.degrees(np.arctan2(out_lab[:, 2], out_lab[:, 1]))
    hue_delta = np.abs(((h_out - h_in + 180.0) % 360.0) - 180.0)
    chroma_in = np.sqrt(in_lab[:, 1] ** 2 + in_lab[:, 2] ** 2)
    chromatic = chroma_in > 1e-3
    if chromatic.any():
        worst_hue_arg = int(np.argmax(hue_delta[chromatic]))
        worst_hue_in = float(h_in[chromatic][worst_hue_arg])
        worst_hue_rotation = float(hue_delta[chromatic][worst_hue_arg])
    else:
        worst_hue_in = 0.0
        worst_hue_rotation = 0.0

    sigma1 = field["sigma1"]
    anisotropy = field["anisotropy"]

    fig = viz.noise_sensitivity(
        field=field,
        heatmap=heatmap,
        rosette=rosette,
        in_cs=in_cs,
        out_cs=out_cs,
        L_slice=L_slice,
        sigma_in_encoded=sigma_in_encoded,
        ellipse_display_scale=ellipse_display_scale,
    )
    path = _save(ctx, fig, "noise_sensitivity")

    return Result(
        name="noise_sensitivity",
        summary={
            "L_slice": L_slice,
            "sigma_in_encoded": sigma_in_encoded,
            "n_input_samples": int(field["sigma1"].size),
            "max_sigma1": float(np.nanmax(sigma1)) if sigma1.size else 0.0,
            "p99_sigma1": float(np.nanpercentile(sigma1, 99)) if sigma1.size else 0.0,
            "p50_sigma1": float(np.nanpercentile(sigma1, 50)) if sigma1.size else 0.0,
            "max_anisotropy": float(np.nanmax(anisotropy)) if anisotropy.size else 0.0,
            "p99_anisotropy": float(np.nanpercentile(anisotropy, 99))
            if anisotropy.size
            else 0.0,
            "max_hue_rotation_deg": worst_hue_rotation,
            "worst_hue_deg": worst_hue_in,
            "ellipse_display_scale": ellipse_display_scale,
        },
        figure_path=path,
        units="OkLab per encoded-RGB",
        interpretation=(
            "First-order noise propagation Σ_out ≈ J·Σ_in·Jᵀ with "
            "isotropic encoded-input noise. σ₁(J) is the worst-case "
            "noise-gain scalar at each input chromaticity; ellipse "
            "orientation is the direction that gain points in OkLab "
            "a*b*. Eccentric ellipses point along the direction the "
            "LUT amplifies most — e.g., orange samples with an ellipse "
            "tilted toward the red axis exactly reproduces the "
            "user-reported 'red speckle on orange' phenomenon. The "
            "σ₁ heatmap shows where in chromaticity the LUT is "
            "noisiest; the hue rosette summarizes per-hue luminance "
            "vs chroma noise gain at one chroma ring. Per-stock "
            "pass/fail thresholds wait on baselines (n080 §10)."
        ),
        passed=None,
    )


def noise_gradient(ctx: QAContext) -> Result:
    """Continuous OkLab hue gradient with seeded encoded-input noise.

    Intended as the visual companion to :func:`noise_sensitivity`.
    Instead of statistical summaries or sparse ellipse samples, this
    renders a continuous square gradient that cycles hue left→right and
    runs black → saturated hue → white bottom→top, then shows what the
    LUT output looks like with a uniform ``σ=0.02`` encoded-input
    noise field applied.
    """
    in_cs = ctx.spec.input_color_space
    out_cs = ctx.spec.output_color_space

    width = 512
    height = 512
    rim_scale = 0.96
    sigma_in_encoded = 0.02
    rng_seed = 0

    input_clean, gradient_stats = _build_oklab_noise_gradient_input(
        ctx.frame,
        width=width,
        height=height,
        rim_scale=rim_scale,
    )
    rng = np.random.default_rng(rng_seed)
    input_noisy = np.clip(
        input_clean + rng.normal(0.0, sigma_in_encoded, size=input_clean.shape),
        0.0,
        1.0,
    )

    output_clean = evaluators.apply_trilinear(
        ctx.lut.table,
        input_clean.reshape(-1, 3),
    ).reshape(height, width, 3)
    output_noisy = evaluators.apply_trilinear(
        ctx.lut.table,
        input_noisy.reshape(-1, 3),
    ).reshape(height, width, 3)
    clean_display = _encoded_image_to_srgb_display(output_clean, out_cs)
    noisy_display = _encoded_image_to_srgb_display(output_noisy, out_cs)

    delta = np.abs(output_noisy - output_clean)
    fig = viz.noise_gradient(
        clean_srgb=clean_display,
        noisy_srgb=noisy_display,
        in_cs=in_cs,
        out_cs=out_cs,
        sigma_in_encoded=sigma_in_encoded,
        mid_L=MIDGRAY_18_OKLAB_L,
        rng_seed=rng_seed,
    )
    path = _save(ctx, fig, "noise_gradient")

    return Result(
        name="noise_gradient",
        summary={
            "image_side": width,
            "sigma_in_encoded": sigma_in_encoded,
            "mid_L": MIDGRAY_18_OKLAB_L,
            "rim_scale": gradient_stats["rim_scale"],
            "input_in_gamut_fraction": gradient_stats["in_gamut_fraction"],
            "mean_abs_output_delta": float(np.mean(delta)),
            "p99_abs_output_delta": float(np.percentile(delta, 99)),
        },
        figure_path=path,
        units="encoded RGB",
        interpretation=(
            "Continuous visual noise diagnostic. The clean panel is a "
            "square OkLab hue-cycle gradient with black → saturated → white "
            "vertically; the noisy panel applies a single seeded "
            "σ=0.02 encoded-input noise realization before the LUT. If the "
            "LUT amplifies certain hues or tones, the noisy panel reveals it "
            "as visible grain, hue wobble, or texture on what should be a "
            "smooth gradient."
        ),
        passed=None,
    )


def _build_gamut_edge_stress_panel(
    target_cs: str,
    in_cs: str,
    out_cs: str,
    pipeline,
    *,
    width: int = 768,
    height: int = 256,
) -> tuple[np.ndarray, dict]:
    """Build one Granger-style RGB stress panel.

    Each column is a continuous tent gradient built in the **target
    space's CCTF-encoded RGB** (Mononodes LUT-Inspector convention —
    ramps are perceptually uniform in encoded RGB, not linear):

    - top row:    white ``(1, 1, 1)``
    - mid row:    a saturated point on the RGB-cube edge in encoded
                  RGB (R → Y → G → C → B → M → R across columns)
    - bottom row: black ``(0, 0, 0)``

    Within a column the encoded value is linearly interpolated:
    ``pixel_enc = w_white·(1,1,1) + w_sat·C(hue) + w_black·(0,0,0)``,
    with tent weights ``w_white = max(0, 1-2v)``,
    ``w_sat = 1 - |2v-1|``, ``w_black = max(0, 2v-1)`` for
    ``v ∈ [0, 1]`` top→bottom.

    The encoded image is CCTF-decoded to target linear, CAT-adapted
    into the bundle's input primaries (no clipping — saturated rim
    pixels keep their negative components), and pushed through the
    actual runtime pipeline. This is what a real workflow does: the
    runtime's input gamut compression handles chromaticities inside
    the visible locus directly via spectral upsampling, so there's no
    need to force the input through the LUT's [0, 1]³ cube boundary
    via hard clipping. The LUT is *not* used for this test — running
    the runtime is the honest answer for stress-test inputs whose
    chromaticities sit outside the bundle's declared input primaries.

    Output goes through the runtime's output gamut compression (toward
    the bundle's output primaries, baked into the pipeline). The
    result is then CAT'd to sRGB linear and hard-clipped to ``[0, 1]``
    for display — the runtime's output gamut compression is expected
    to have already pulled values inside the cube, so the clip should
    be near-identity in well-behaved bundles. Any visible clip cliff
    here is a bake-time disclosure that the bundle's compression
    didn't fully contain the simulation's reach for this target.

    The ``oog_fraction_*`` stats report how many pixels of the
    target-space gradient lie outside the bundle's input-primaries
    cube — a diagnostic of "how much of this source the bundle can't
    natively represent in its declared input gamut," kept even though
    the pipeline handles those pixels without clipping.
    """
    from spektrafilm_lut_creator.color_spaces import decode_cctf
    from spektrafilm_lut_creator.color_spaces import get as get_cs

    W, H = width, height

    # Saturated-edge color per column. 6 segments around the RGB cube:
    # R → Y → G → C → B → M → R.
    t = np.linspace(0.0, 6.0, W, endpoint=False)
    seg = np.floor(t).astype(int) % 6
    f = (t - np.floor(t)).astype(float)
    sat = np.zeros((W, 3), dtype=float)
    builders = (
        lambda f: np.stack([np.ones_like(f), f, np.zeros_like(f)], axis=-1),
        lambda f: np.stack([1.0 - f, np.ones_like(f), np.zeros_like(f)], axis=-1),
        lambda f: np.stack([np.zeros_like(f), np.ones_like(f), f], axis=-1),
        lambda f: np.stack([np.zeros_like(f), 1.0 - f, np.ones_like(f)], axis=-1),
        lambda f: np.stack([f, np.zeros_like(f), np.ones_like(f)], axis=-1),
        lambda f: np.stack([np.ones_like(f), np.zeros_like(f), 1.0 - f], axis=-1),
    )
    for s, build in enumerate(builders):
        m = seg == s
        if m.any():
            sat[m] = build(f[m])

    # Tent weights down the column: white at v=0, saturated at v=0.5,
    # black at v=1. Applied to the target space's CCTF-encoded RGB so
    # the ramp is perceptually uniform — the LUT-Inspector convention.
    v = np.linspace(0.0, 1.0, H)
    w_white = np.maximum(0.0, 1.0 - 2.0 * v).reshape(H, 1, 1)
    w_sat = (1.0 - np.abs(2.0 * v - 1.0)).reshape(H, 1, 1)
    # w_black contributes zero so omitted.
    image_target_encoded = w_white + w_sat * sat[None, :, :]

    target_entry = get_cs(target_cs)
    in_entry = get_cs(in_cs)
    out_entry = get_cs(out_cs)

    # CCTF-decode to target linear, CAT to bundle-input linear. No
    # clipping at this boundary — the pipeline will handle any negative
    # components via input gamut compression toward the spectral locus.
    image_linear = decode_cctf(image_target_encoded, target_cs)
    input_linear = np.asarray(
        colour.RGB_to_RGB(
            image_linear,
            target_entry.primaries,
            in_entry.primaries,
            chromatic_adaptation_transform="CAT16",
        ),
        dtype=float,
    )

    # OOG-to-input diagnostic: how much of the target gradient sits
    # outside the bundle's declared input primaries cube. Computed
    # purely for the stat; the pipeline doesn't need it.
    oog_mask = np.any(
        (input_linear < 0.0) | (input_linear > 1.0),
        axis=-1,
    )

    # Run the gradient through the actual runtime pipeline. Pipeline
    # expects (H, W, 3); lut_mode disables every spatial effect so the
    # layout is purely a performance knob. Output is linear RGB in the
    # bundle's output primaries.
    image_in = input_linear.reshape(1, H * W, 3).astype(np.float32)
    image_out_linear = np.asarray(
        pipeline.process(image_in),
        dtype=float,
    ).reshape(H, W, 3)

    # Display conversion: CAT from bundle output primaries to sRGB
    # linear, then a hard clip to [0, 1] before sRGB-encoding. This
    # step used to be an OkLch chroma reduction toward sRGB, but OkLab's
    # well-known blue hue rotation produced a visible cyan↔magenta seam
    # at the deep-blue corner — visible in the saturated row of the
    # strip. A hard clip keeps the test "honest about" what the bundle
    # actually delivers: any value the runtime's output gamut
    # compression didn't already pull inside [0, 1]³ now clips at the
    # cube boundary, exactly as a downstream consumer would see it.
    srgb_linear = np.asarray(
        colour.RGB_to_RGB(
            image_out_linear,
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
    srgb_encoded = np.clip(srgb_encoded, 0.0, 1.0)

    mid_band = slice(max(0, H // 2 - H // 16), H // 2 + H // 16)
    stats = {
        "oog_fraction_total": float(oog_mask.mean()),
        "oog_fraction_saturated_row": float(oog_mask[mid_band].mean()),
    }
    return srgb_encoded, stats


def output_gamut_edge_stress(ctx: QAContext) -> Result:
    """Granger-style RGB stress chart at the edges of three target
    color spaces, rendered through the actual runtime pipeline.

    For each target space (Rec.709, DCI-P3, Rec.2020) we build a
    vertical linear-RGB gradient — white at the top, the saturated
    edge of the target's RGB cube in the middle (hue cycle across
    columns), black at the bottom — CAT-adapt it into the bundle's
    input primaries (no clipping), and process it through the runtime
    pipeline rather than the baked LUT. The runtime handles
    chromaticities outside the bundle's declared input cube directly:
    spectral upsampling works anywhere inside the visible locus, and
    the bundle's input gamut compression (toward the locus) handles
    anything beyond. Output goes through the bundle's output gamut
    compression and is rendered to sRGB for display via OkLch chroma
    reduction toward the sRGB primaries.

    Why the runtime and not the LUT: the LUT is sampled in
    ``[0, 1]^3`` of the bundle's input encoded cube; saturated rim
    pixels in a wider target space (P3, Rec.2020) lie outside that
    cube and the LUT physically can't evaluate them without first
    clipping them in — which is exactly the artifact the test is
    supposed to surface. Running the runtime is the honest answer
    for "what would the bundle produce if applied to this source."

    Visible bands, kinks, hue jumps, or posterization in the rendered
    gradient signal model-side pathology at saturated input — a
    regime the rest of the suite probes only statistically.

    References
    ----------
    - Mononodes LUT Inspector — Granger-style RGB stress chart,
      https://mononodes.com/lut-inspector/.
    - Mononodes Cube Slice DCTL — RGB cube face gradients.
    """
    from spektrafilm.runtime.params_builder import digest_params, init_params
    from spektrafilm.runtime.pipeline import SimulationPipeline
    from spektrafilm_lut_creator.color_spaces import get as get_color_space

    spec = ctx.spec
    in_cs = spec.input_color_space
    out_cs = spec.output_color_space
    in_entry = get_color_space(in_cs)
    out_entry = get_color_space(out_cs)
    print_profile = (
        ctx.bundle.meta.stocks.prints[ctx.print_index]
        if ctx.bundle.meta.stocks
        else spec.print_profiles[ctx.print_index]
    )

    # Build the runtime pipeline once and share it across the three
    # target gradients. lut_mode disables spatial effects; the
    # input_gamut_compress / output_gamut_compress settings mirror the
    # bundle's bake-time configuration so the stress test renders what
    # the bundle would actually produce.
    params = init_params(film_profile=spec.film_profile, print_profile=print_profile)
    params.debug.lut_mode = True
    params.io.input_color_space = in_entry.primaries
    params.io.output_color_space = out_entry.primaries
    params.io.input_cctf_decoding = False
    params.io.output_cctf_encoding = False
    params.io.input_gamut_compress = spec.input_gamut_compress
    params.io.output_gamut_compress = spec.output_gamut_compress
    params = digest_params(params)
    pipeline = SimulationPipeline(params)

    target_spaces = ["Rec.709", "DCI-P3", "Rec.2020"]
    panels: list[tuple[str, np.ndarray, dict]] = []
    summary: dict[str, float] = {}
    for cs in target_spaces:
        img, stats = _build_gamut_edge_stress_panel(cs, in_cs, out_cs, pipeline)
        panels.append((cs, img, stats))
        summary[f"{cs}_oog_fraction_saturated_row"] = stats[
            "oog_fraction_saturated_row"
        ]

    fig = viz.gamut_edge_stress(
        panels,
        in_cs=in_cs,
        out_cs=out_cs,
        gamut_compress=spec.output_gamut_compress,
    )
    path = _save(ctx, fig, "output_gamut_edge_stress")

    return Result(
        name="output_gamut_edge_stress",
        summary=summary,
        figure_path=path,
        units="",
        interpretation=(
            "Each panel is a Granger-style RGB stress chart at the "
            "edges of one target RGB space: each column is a tent "
            "white → saturated_edge(hue) → black in target-cs encoded "
            "RGB. The image is CAT-adapted into the bundle's input "
            "primaries (no clipping) and pushed through the runtime "
            "pipeline, which handles chromaticities outside the "
            "bundle's input cube via spectral upsampling + input "
            "gamut compression toward the visible locus. Output is "
            "rendered to sRGB for viewing via CAT + a hard clip to "
            "[0, 1] (the runtime's own output gamut compression is "
            "expected to have already done the cube containment). "
            "The gradient should be continuous and smooth from top to "
            "bottom; visible bands, hue jumps, or posterization "
            "reveal model-side pathology at saturated input. "
            "`oog_fraction_*` reports how many pixels lie outside the "
            "bundle's input primaries cube — a diagnostic of how much "
            "of each target's saturated rim the bundle has to handle "
            "as out-of-input-gamut input (the pipeline still renders "
            "them, but they're physically outside the bundle's "
            "declared input gamut)."
        ),
        passed=None,
    )


def rg_plane_slices(ctx: QAContext) -> Result:
    """R-G cube slices at evenly-spaced B-input values, displayed in sRGB.

    Each panel shows the LUT's R-G response at one B input level. The
    slice is in the bundle's output color space; we decode to linear,
    chromatically adapt to sRGB, sRGB-encode, and hard-clip — so the
    rendered colors are visually accurate on an sRGB display
    regardless of the bundle's output space.
    """
    fig = viz.rg_plane_slices(
        ctx.lut.table,
        ctx.lut.resolution,
        ctx.spec.output_color_space,
    )
    path = _save(ctx, fig, "rg_plane_slices")

    return Result(
        name="rg_plane_slices",
        summary={
            "n_slices": int(min(9, ctx.lut.resolution)),
            "cube_resolution": int(ctx.lut.resolution),
        },
        figure_path=path,
        units="",
        interpretation=(
            "Cube cross-sections at constant input B, displayed in "
            "sRGB (hard-clipped). Smooth color gradation across the "
            "panels indicates a well-behaved cube along B; abrupt "
            "changes between adjacent slices point at low-resolution "
            "or noisy regions. Within each panel the gradient runs R "
            "left→right, G bottom→top — corner colors are the LUT's "
            "renderings of input (R, G, B) corners at that B."
        ),
        passed=None,
    )
