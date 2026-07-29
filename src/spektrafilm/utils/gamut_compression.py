"""Input gamut compression for the Hanatos 2025 spectral upsampling step.

Wide-gamut input color spaces (V-Gamut, ACEScg, AP0, …) can encode
chromaticities outside the visible spectral locus — combinations of
camera primaries that do not correspond to any physically realizable
spectrum. The Hanatos 2025 spectral upsampling is only well-defined
inside the locus; chromaticities outside it produce extrapolation noise.

This module provides:

- :class:`InputGamutCompressSpec` — the per-bundle configuration object that
  selects the algorithm and parameters.
- :func:`compress_xy` — the algorithm dispatcher; takes CIE xy values
  and returns compressed CIE xy values.
- :func:`remap_tc_lut_for_compression` — bakes the compression into a
  ``compute_hanatos2025_tc_lut``-produced LUT via remap-resample, so the
  runtime lookup path does not need to know about the compression.

The default settings use a full-range Reinhard knee
(``knee=(0.0, 1.0, 6.0)``), which applies a gentle, always-on roll-off
toward the boundary instead of leaving a hard identity region before an
abrupt shoulder. See the spektrafilm-research note
``n100_soft_input_clipping`` §5 for the rationale and comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import colour
import numpy as np
from scipy.ndimage import map_coordinates

# ---------------------------------------------------------------------------
# Spec dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InputGamutCompressSpec:
    """Configuration for input gamut compression.

    Attributes
    ----------
    active :
        ``True`` applies the Reinhard-knee compression; ``False``
        disables compression and passes input chromaticities through
        unchanged (the LUT bake then behaves as it did before this
        feature landed).
    algorithm :
        ``"xy"`` — radial compression in CIE 1931 chromaticity from the
        scene white toward the visible spectral locus (ACES RGC family).
        Preserves *dominant wavelength* and is the straightest / smoothest
        path by construction (n100 §5.1 measured it as the smoothest input
        path; the former ``"jzazbz"`` / ``"cam16ucs"`` perceptual input
        options were dropped — they were measurably less smooth in-locus and
        only well-defined near the locus. Perceptual chroma reduction remains
        an *output*-side option, where the boundary is the RGB cube).
    boundary :
        Target boundary the radial knee compresses toward.
        ``"inscribed_hull"`` (default) — a smooth star-shaped hull inscribed
        inside the visible spectral locus (:func:`inscribed_locus_hull`),
        which removes the locus-polygon vertex / cusp kinks and keeps every
        compressed chromaticity strictly inside the locus (b40). ``"locus"``
        — the raw spectral-locus polygon (kept for A/B and regression).
    hull_detail :
        Number of FFT modes kept when building the inscribed hull (the
        mode-space Gaussian width). **Higher** keeps more modes → the hull hugs
        the locus tighter (more cusp chroma) but re-admits kinks; **lower** keeps
        fewer modes → smoother, rounding the spectral tips inward toward the
        inscribed circle. (Note the direction: this is modes-kept, not a spatial
        blur width — larger = *less* smoothing.) Only used when
        ``boundary == "inscribed_hull"``.
    knee :
        ``(threshold, limit, power)`` of the Reinhard knee:
        ``d' = t + s · n / (1 + n^p)^(1/p)`` where ``n = (d - t)/s``
        and ``s = limit - t``. Default ``(0.815, 1.0, 1.2)`` — the ACES RGC
        pair: the ``0.815`` threshold leaves in-gamut / midrange colors
        untouched, and the ``1.2`` power gives a slow asymptotic tail so
        deep-imaginary colors keep their gradation instead of clipping to the
        boundary. ``limit = 1.0`` puts the asymptote on the boundary (the
        inscribed hull, inside the locus), so compressed values stay inside.
    """

    active: bool = True
    algorithm: Literal["xy"] = "xy"
    boundary: Literal["locus", "inscribed_hull"] = "inscribed_hull"
    hull_detail: float = 5.0
    knee: tuple[float, float, float] = (0.815, 1.0, 1.2)

    def __post_init__(self) -> None:
        valid_algos = ("xy",)
        if self.algorithm not in valid_algos:
            raise ValueError(
                f"algorithm must be one of {valid_algos}, got {self.algorithm!r}"
            )
        valid_boundaries = ("locus", "inscribed_hull")
        if self.boundary not in valid_boundaries:
            raise ValueError(
                f"boundary must be one of {valid_boundaries}, got {self.boundary!r}"
            )
        if not (self.hull_detail > 0.0):
            raise ValueError(f"hull_detail must be > 0, got {self.hull_detail}")
        t, l, p = self.knee
        if not (0.0 <= t < 1.0):
            raise ValueError(f"knee threshold must be in [0, 1), got {t}")
        if not (l > 0.0):
            raise ValueError(f"knee limit must be > 0, got {l}")
        if not (p > 0.0):
            raise ValueError(f"knee power must be > 0, got {p}")


@dataclass(frozen=True)
class OutputGamutCompressSpec:
    """Configuration for output gamut compression.

    Independent of :class:`InputGamutCompressSpec` (which targets the visible
    spectral locus on the input side, in chromaticity). This spec targets
    the *output primaries cube* on the output side, in destination RGB.
    Both algorithms reuse the same Reinhard knee math.

    Attributes
    ----------
    algorithm :
        ``"off"`` disables output gamut compression and passes output
        RGB through unchanged. With nothing else clipping downstream
        in the runtime, the LUT cube can then leak outside [0, 1] —
        use ``"off"`` only for diagnostic builds.

        ``"cam16ucs"`` — CIECAM16 Uniform Color Space chroma reduction
        (Li & Luo 2017, CIE-endorsed). Same JpCphp polar shape as
        ``oklch`` / ``jzazbz``, but in CAM16-UCS, which models
        chromatic adaptation and viewing-condition response explicitly
        and has the cleanest constant-hue lines of the three across
        the full gamut — including the blue region where OkLab drifts.
        Viewing conditions are fixed at typical display review
        (``L_A = 64 cd/m²``, ``Y_b = 20``, Average surround). Heavier
        per-pixel than OkLab or JzAzBz (full CAM forward/inverse), so
        bake time grows ~4–6× on the compression stage; pick this when
        you want the smoothest principled chroma reduction and can
        afford the cost.

        ``"oklch"`` — perceptual-hue-preserving chroma
        reduction in OkLab. Convert output RGB → OkLab → OkLch
        (L, C, h), look up ``C_max(L, h)`` for the output color
        space's RGB cube, apply the Reinhard knee to ``C / C_max``,
        reconstruct. Preserves OkLab hue ``h`` and OkLab lightness
        ``L`` — the colorist sees the same hue at the same perceptual
        brightness, just less saturated. A smooth principled choice when
        the output gamut is the round-to-hexagonal envelope of a CMY dye
        set (see n110 §2), where perceptual-chroma reduction around a
        round envelope reads more smoothly than a per-channel knee
        against a triangle.

        ``"aces_rgc"`` (default) — ACES Reference Gamut Compression v1.3
        native form. Per-channel knee on the achromatic distance
        ``d = (max(R,G,B) - c) / max(R,G,B)``. Preserves the achromatic
        RGB mixture. This is the cinema-industry standard for output
        gamut mapping; same operation as OCIO's
        ``FixedFunctionTransform(style=ACES_GamutComp13)`` (the knee
        math is verified bit-identical in
        ``validate_compression_against_references.py``). Default because
        it matches downstream Resolve/Nuke gamut-compress tools and is
        far cheaper per pixel than the perceptual algorithms (no CAM /
        OkLab forward+inverse), which dominates the scanner stage cost.

        ``"jzazbz"`` — JzCzhz chroma reduction, structurally identical
        to ``oklch`` but in the JzAzBz perceptual space (Safdar 2017)
        with the reference white luminance fixed at 100 cd/m² (SDR
        diffuse white). JzAzBz handles the blue/cyan hue region more
        gracefully than OkLab, which has a documented hue twist there
        — cyan can drift toward blue under heavy oklch compression on
        sRGB / Rec.2020 outputs. Use this if you see that artifact.
        Default knee is currently a placeholder mirroring the OkLch
        defaults; the JzAzBz Cz envelope still needs to be measured
        per output space before final defaults can be picked.

        ``"oklrab"`` — OkLab with Ottosson's 2023 rebased lightness
        ``Lr`` (a monotonic 1D remap of ``L`` so the lightness scale
        is closer to CIELAB ``L*``). Same chromatic axes ``(a, b)`` as
        ``oklch``; only the ``C_max(L, h)`` table is indexed by ``Lr``
        instead of ``L``. The knee response is then perceptually more
        uniform across light and dark regions because equal increments
        in ``Lr`` correspond to closer-to-equal perceived lightness
        changes. Same per-pixel cost as ``oklch`` (one extra scalar
        operation per sample for the ``L → Lr`` remap).

    knee :
        ``(threshold, limit, power)`` of the Reinhard knee. Default
        ``(0.0, 1.0, 6.0)`` applies a soft full-range roll-off so
        chroma starts compressing smoothly from the center instead of
        waiting for a late knee near the gamut boundary. Limit stays at
        1.0 so the asymptote sits on the cube edge and the LUT cube
        remains in [0, 1] without needing a hard clip. The knee
        operates on the unitless normalized chroma ``C / C_max`` and is
        reused unchanged across ``oklch`` and ``jzazbz``.
    lightness_compression :
        ``(threshold, limit, power)`` of a *one-sided* soft compression
        on the perceptual lightness coordinate (L for oklch/oklrab, Jz
        for jzazbz, Jp for cam16ucs), or ``None`` to disable. It rolls
        super-bright highlights smoothly down into ``[0, white]`` —
        pixels above the output whitepoint, which the chroma step alone
        cannot pull in (their ``C_max`` collapses to zero). Parameters
        are in *normalized* units where ``1.0`` = the output color
        space's perceptual white; the default ``(0.7, 1.0, 2.2)`` keeps
        the asymptote exactly at white with identity below
        ``0.7·white``.

        Crucially this is one-sided and anchored at black: an input of
        0 stays exactly 0. The previous two-sided knee mirrored the
        roll-off around mid-grey and so lifted deep shadows toward a
        non-zero floor — invisible in OkLab's cube-root lightness, but a
        visibly raised black in the PQ-based (JzAzBz) and CIECAM
        (CAM16-UCS) lightness scales, which allocate far more of their
        range to the darks. Below-black float inputs (L < 0) are left
        untouched; the simulation output is expected to stay at Y >= 0
        by physical construction. Not applied to ``algorithm="aces_rgc"``
        (no perceptual lightness axis).
    """

    algorithm: Literal[
        "off",
        "aces_rgc",
        "oklch",
        "oklrab",
        "jzazbz",
        "cam16ucs",
    ] = "oklch"
    knee: tuple[float, float, float] = (0.95, 1.0, 1.6)
    # One-sided soft compression on the perceptual lightness coordinate
    # (L for oklch/oklrab, Jz for jzazbz, Jp for cam16ucs), rolling
    # super-bright highlights smoothly down into [0, white]. The chroma
    # knee above only touches chromaticity; pixels with lightness above
    # the output whitepoint are otherwise left out-of-cube because C_max
    # collapses to zero there.
    lightness_compression: tuple[float, float, float] | None = (0.95, 1.0, 1.6)

    def __post_init__(self) -> None:
        valid_algos = ("off", "aces_rgc", "oklch", "oklrab", "jzazbz", "cam16ucs")
        if self.algorithm not in valid_algos:
            raise ValueError(
                f"algorithm must be one of {valid_algos}, got {self.algorithm!r}"
            )
        t, l, p = self.knee
        if not (0.0 <= t < 1.0):
            raise ValueError(f"knee threshold must be in [0, 1), got {t}")
        if not (l > 0.0):
            raise ValueError(f"knee limit must be > 0, got {l}")
        if not (p > 0.0):
            raise ValueError(f"knee power must be > 0, got {p}")
        if self.lightness_compression is not None:
            lt, ll, lp = self.lightness_compression
            if not (0.0 <= lt < 1.0):
                raise ValueError(
                    f"lightness_compression threshold must be in [0, 1), got {lt}"
                )
            if not (ll > 0.0):
                raise ValueError(f"lightness_compression limit must be > 0, got {ll}")
            if not (lp > 0.0):
                raise ValueError(f"lightness_compression power must be > 0, got {lp}")

    @property
    def active(self) -> bool:
        return self.algorithm != "off"


# ---------------------------------------------------------------------------
# Spectral locus singleton
# ---------------------------------------------------------------------------

_SPECTRAL_LOCUS_XY_CACHE: np.ndarray | None = None


def spectral_locus_xy() -> np.ndarray:
    """Closed polygon of the CIE 1931 2° visible spectral locus in xy.

    Sampled at 5 nm steps from 380 to 700 nm; the returned array
    repeats the first vertex at the end so it is suitable for
    `matplotlib.path.Path` and ray-polygon intersection.
    """
    global _SPECTRAL_LOCUS_XY_CACHE
    if _SPECTRAL_LOCUS_XY_CACHE is None:
        cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
        # 5 nm sampling is fine; the locus polygon is dense enough for
        # robust ray-intersection and in-polygon tests.
        wavelengths = np.arange(380, 705, 5)
        xyz = np.asarray(cmfs[wavelengths])
        total = xyz.sum(axis=-1)
        xy = xyz[:, :2] / np.fmax(total[:, None], 1e-12)
        _SPECTRAL_LOCUS_XY_CACHE = np.concatenate([xy, xy[:1]], axis=0)
    return _SPECTRAL_LOCUS_XY_CACHE


# ---------------------------------------------------------------------------
# Reinhard knee + xy radial compressor
# ---------------------------------------------------------------------------


def reinhard_knee(
    d: np.ndarray,
    *,
    threshold: float,
    limit: float,
    power: float,
) -> np.ndarray:
    """Smooth knee on normalized distance.

    Identity below ``threshold``; smoothly asymptotic at ``limit`` above
    it. The exact formula matches the ACES RGC v1.3 reference (verified
    bit-identical by spektrafilm-research/studies/a40_lut_system/
    validate_compression_against_references.py).
    """
    out = np.asarray(d, dtype=float).copy()
    mask = out > threshold
    if np.any(mask):
        scale = limit - threshold
        x = (out[mask] - threshold) / scale
        y = x / np.power(1.0 + np.power(x, power), 1.0 / power)
        out[mask] = threshold + scale * y
    return out


def _ray_polygon_distance(
    origin: np.ndarray,
    direction: np.ndarray,
    polygon: np.ndarray,
) -> np.ndarray:
    """Distance from ``origin`` along unit ``direction`` to the first
    intersection with the closed polygon.

    Vectorized: ``direction`` has shape ``(..., 2)``; the polygon is
    a single closed loop of shape ``(N+1, 2)`` with the first vertex
    repeated at the end. Returns shape ``(...)``.

    Uses the parametric line-segment intersection
    (origin + t · direction = a + s · (b - a)) with the usual
    1e-12 epsilon guard for parallel rays. NaN for rays that miss the
    polygon (should not happen for the visible locus + interior origin).
    """
    direction = np.asarray(direction, dtype=float)
    flat = direction.reshape(-1, 2)
    n_rays = flat.shape[0]
    t_min = np.full(n_rays, np.inf)

    a = polygon[:-1]
    b = polygon[1:]
    edge = b - a

    for k in range(len(edge)):
        ex, ey = edge[k]
        ax, ay = a[k]
        dx = flat[:, 0]
        dy = flat[:, 1]
        denom = dx * ey - dy * ex
        valid = np.abs(denom) > 1e-12
        ox = origin[0] - ax
        oy = origin[1] - ay
        # Solve t * d - s * edge = -o  →  t = (o × edge) / (d × edge)
        t = np.where(
            valid,
            (-ox * ey + oy * ex) / np.where(valid, denom, 1.0),
            np.inf,
        )
        s = np.where(
            valid,
            (-ox * dy + oy * dx) / np.where(valid, denom, 1.0),
            np.inf,
        )
        # Small epsilon on the edge parameter so a ray passing exactly through a
        # vertex still registers on an adjacent edge (s just outside [0, 1] by
        # float error) instead of missing both — otherwise the ray returns inf,
        # and compress_xy_radial's `d_compressed * boundary` becomes 0 * inf =
        # NaN. Matters for dense polygons (the inscribed hull) whose vertices can
        # align exactly with query directions.
        good = valid & (t > 1e-9) & (s >= -1e-9) & (s <= 1.0 + 1e-9)
        t_min = np.where(good & (t < t_min), t, t_min)

    return t_min.reshape(direction.shape[:-1])


def compress_xy_radial(
    xy: np.ndarray,
    white_xy: np.ndarray,
    *,
    threshold: float,
    limit: float,
    power: float,
    locus: np.ndarray | None = None,
) -> np.ndarray:
    """ACES-RGC-style radial compression toward the spectral locus.

    For each xy, compute the normalized distance ``d`` from ``white_xy``
    along the ray to the locus boundary; pass ``d`` through the Reinhard
    knee; scale ``d`` back into a new xy along the same ray. Hue
    (= dominant wavelength) is preserved by construction.
    """
    if locus is None:
        locus = spectral_locus_xy()
    xy = np.asarray(xy, dtype=float)
    white_xy = np.asarray(white_xy, dtype=float)
    delta = xy - white_xy
    dist = np.linalg.norm(delta, axis=-1)
    # At-white samples have zero distance and undefined direction; they
    # produce NaN/inf in the intermediate ray-polygon math but are
    # substituted with the original xy by the final np.where. Silence
    # the spurious intermediate warnings.
    with np.errstate(invalid="ignore", divide="ignore"):
        safe_dist = np.fmax(dist, 1e-12)
        direction = delta / safe_dist[..., None]
        boundary = _ray_polygon_distance(white_xy, direction, locus)
        d_norm = dist / np.fmax(boundary, 1e-12)
        d_compressed = reinhard_knee(
            d_norm,
            threshold=threshold,
            limit=limit,
            power=power,
        )
        new_xy = white_xy + direction * (d_compressed * boundary)[..., None]
    # At-white points have undefined direction; pass through unchanged.
    return np.where((dist < 1e-9)[..., None], xy, new_xy)


# ---------------------------------------------------------------------------
# Inscribed smooth hull of the spectral locus (study b40)
# ---------------------------------------------------------------------------

# Cache key: (round(white, 6), detail, id(locus)). Built once per bundle
# white; the build is a few FFTs + a ray-cast, but the LUT bake calls compress
# repeatedly so caching keeps it a one-time cost.
_INSCRIBED_HULL_CACHE: dict[tuple, np.ndarray] = {}


def fourier_lowpass(y: np.ndarray, mode_sigma: float) -> np.ndarray:
    """Low-pass a periodic function by attenuating FFT mode ``k`` with a Gaussian
    ``exp(-0.5 (k/mode_sigma)^2)``.

    A cusp is the highest-frequency feature of a radial boundary function, so this
    dissolves cusps into a smooth hull. DC (mean radius) is preserved.
    """
    A = np.fft.rfft(y)
    k = np.arange(A.shape[0])
    return np.fft.irfft(A * np.exp(-0.5 * (k / mode_sigma) ** 2), len(y))


def inscribed_locus_hull(
    white_xy: np.ndarray,
    detail: float,
    locus: np.ndarray | None = None,
    *,
    n_theta: int = 1024,
) -> np.ndarray:
    """Smooth star-shaped hull inscribed *inside* the spectral locus, centered on
    ``white_xy``, returned as a closed polygon for :func:`compress_xy_radial`.

    Build: sample the locus's radial reach ``R(theta)`` from ``white_xy`` at
    ``n_theta`` angles, FFT-low-pass it to dissolve the cusps, then scale it
    *down* to the worst overshoot so ``R_eff <= R`` everywhere — the hull lies
    strictly inside the true locus. Compressing radially against this hull
    therefore lands every input inside the locus (no out-of-locus inpainting is
    assumed downstream), while removing the locus-polygon vertex / cusp kinks
    that a raw-polygon boundary injects into the compression field.

    ``detail`` is the number of FFT modes kept (the mode-space Gaussian width
    passed to :func:`fourier_lowpass`). It trades cusp chroma against smoothness:
    **higher detail** keeps more modes → the hull hugs the locus tighter (more
    cusp chroma) but re-admits kinks; **lower detail** keeps fewer modes → smoother
    but rounds the spectral tips inward, collapsing toward the inscribed circle as
    detail → 0. See spektrafilm-research b40 (n030/n040, s030).

    Cached per ``(white_xy, detail, locus)``.
    """
    if locus is None:
        locus = spectral_locus_xy()
    white_xy = np.asarray(white_xy, dtype=float)
    key = (
        round(float(white_xy[0]), 6),
        round(float(white_xy[1]), 6),
        float(detail),
        id(locus),
    )
    cached = _INSCRIBED_HULL_CACHE.get(key)
    if cached is not None:
        return cached

    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    dirs = np.stack([np.cos(theta), np.sin(theta)], axis=-1)
    reach = _ray_polygon_distance(white_xy, dirs, locus)
    # Every ray from an interior white should hit the closed locus; guard the
    # rare miss (numerical) with the median so the FFT sees a finite function.
    reach = np.nan_to_num(reach, nan=float(np.nanmedian(reach)))

    reach_lp = np.fmax(fourier_lowpass(reach, detail), 1e-12)
    reach_eff = reach_lp * float(np.min(reach / reach_lp))  # inscribe: R_eff <= R
    hull = white_xy + reach_eff[:, None] * dirs
    hull = np.concatenate([hull, hull[:1]], axis=0)  # close the polygon

    _INSCRIBED_HULL_CACHE[key] = hull
    return hull


# ---------------------------------------------------------------------------
# Perceptual chroma compressor + C_max(L, h) bisection cache
#
# Shared C_max(L, h) resolution for the OUTPUT-side polar chroma compressors
# (against the RGB cube). The input side no longer uses a perceptual C_max
# table — its jzazbz / cam16ucs paths were removed (see InputGamutCompressSpec);
# the input compressor is xy-radial against the spectral locus / inscribed hull.
# ---------------------------------------------------------------------------


_OKLCH_CMAX_TABLE_N_L = 64
_OKLCH_CMAX_TABLE_N_H = 720
_OKLCH_CMAX_TABLE_N_BISECT = 18


def _xy_to_xyz_unit_y(xy: np.ndarray) -> np.ndarray:
    """xyY → XYZ at Y = 1 (lightness-preserving reconstruction)."""
    x = xy[..., 0]
    y = xy[..., 1]
    safe_y = np.fmax(y, 1e-12)
    X = x / safe_y
    Y = np.ones_like(x)
    Z = (1.0 - x - y) / safe_y
    return np.stack([X, Y, Z], axis=-1)


def _c_max_lookup(
    L: np.ndarray,
    h: np.ndarray,
    L_grid: np.ndarray,
    h_grid: np.ndarray,
    C_max_table: np.ndarray,
) -> np.ndarray:
    """Bilinear lookup of C_max(L, h)."""
    L = np.clip(L, L_grid[0], L_grid[-1])
    h_step = h_grid[1] - h_grid[0]
    h_idx = (h - h_grid[0]) / h_step
    h_lo = np.floor(h_idx).astype(int) % len(h_grid)
    h_hi = (h_lo + 1) % len(h_grid)
    h_frac = h_idx - np.floor(h_idx)

    L_idx = (L - L_grid[0]) / (L_grid[-1] - L_grid[0]) * (len(L_grid) - 1)
    L_lo = np.clip(np.floor(L_idx).astype(int), 0, len(L_grid) - 2)
    L_hi = L_lo + 1
    L_frac = L_idx - L_lo

    v00 = C_max_table[L_lo, h_lo]
    v01 = C_max_table[L_lo, h_hi]
    v10 = C_max_table[L_hi, h_lo]
    v11 = C_max_table[L_hi, h_hi]
    return (
        v00 * (1 - L_frac) * (1 - h_frac)
        + v01 * (1 - L_frac) * h_frac
        + v10 * L_frac * (1 - h_frac)
        + v11 * L_frac * h_frac
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def compress_xy(
    xy: np.ndarray,
    white_xy: np.ndarray,
    spec: InputGamutCompressSpec,
    *,
    locus: np.ndarray | None = None,
) -> np.ndarray:
    """Apply the compression specified by ``spec`` to CIE xy values.

    Parameters
    ----------
    xy :
        Array of CIE xy values, shape ``(..., 2)``.
    white_xy :
        The achromatic axis (radial center) the compression operates
        around — the fixed D65 compression center in the spektrafilm
        runtime (see the bake callers; independent of the film/adaptation
        white).
    spec :
        Configuration. With ``spec.active`` False returns ``xy``
        unchanged. ``spec.boundary`` selects the compression target: the
        raw spectral-locus polygon (``"locus"``) or a smooth inscribed
        hull of it (``"inscribed_hull"``, at ``spec.hull_detail``).
    locus :
        Optional spectral locus polygon override. Defaults to
        :func:`spectral_locus_xy` (CIE 1931 2° at 5 nm sampling). The
        inscribed hull, when selected, is built from this locus.
    """
    if not spec.active:
        return np.asarray(xy, dtype=float)
    threshold, limit, power = spec.knee
    if spec.boundary == "inscribed_hull":
        boundary = inscribed_locus_hull(white_xy, spec.hull_detail, locus=locus)
    else:
        boundary = locus  # None -> compress_xy_radial falls back to the locus
    return compress_xy_radial(
        xy,
        white_xy,
        threshold=threshold,
        limit=limit,
        power=power,
        locus=boundary,
    )


# ---------------------------------------------------------------------------
# Output-side: per-channel ACES RGC in destination RGB
# ---------------------------------------------------------------------------


def compress_rgb_aces_rgc(
    rgb: np.ndarray,
    *,
    threshold: float,
    limit: float,
    power: float,
) -> np.ndarray:
    """ACES Reference Gamut Compression v1.3 in its native form.

    For each pixel, compute the achromatic value ``ach = max(R, G, B)``
    and per-channel distance ``d = (ach - c) / ach``. ``d >= 0`` for
    every channel; ``d > 1`` exactly when ``c < 0`` (the channel is
    "more saturated than the achromatic envelope"). Apply the Reinhard
    knee to each per-channel distance, then reconstruct:
    ``c' = ach * (1 - d')``.

    With the default ``limit = 1.0``, the knee asymptotes at ``d = 1``,
    so negative channels (which had ``d > 1``) are pulled to exactly
    ``c' = 0`` — the cube boundary. Positive channels inside the
    threshold ``t`` are unchanged. Hue is preserved in the colorist
    sense (the channel-mixture ratios stay coherent rather than the
    dominant-wavelength chromaticity).

    Notes on amplitude
    ------------------
    This operation does *not* touch the achromatic value itself, so
    pixels with ``ach > 1`` retain ``ach > 1`` after compression. The
    perceptual sibling algorithms (``oklch`` / ``oklrab`` / ``jzazbz``
    / ``cam16ucs``) have a companion ``lightness_compression`` that
    catches this case on the lightness axis; ``aces_rgc`` does not. Bundles
    that ship ``aces_rgc`` rely on the simulation staying in [0, 1]
    by physical construction.

    References
    ----------
    AMPAS Gamut Mapping Virtual Working Group, *ACES Reference Gamut
    Compression v1.3*, 2020. Same per-channel formulation as OCIO's
    ``FixedFunctionTransform(style=ACES_GamutComp13)``; knee math
    A/B-validated in
    ``spektrafilm-research/studies/a40_lut_system/
    validate_compression_against_references.py``.
    """
    rgb = np.asarray(rgb, dtype=float)
    # Achromatic envelope. For all-non-positive pixels (extreme shadows)
    # we fall back to identity — no meaningful chromaticity to compress.
    ach = np.max(rgb, axis=-1)
    safe_ach = np.where(ach > 1e-12, ach, 1.0)

    d = (ach[..., None] - rgb) / safe_ach[..., None]
    d_compressed = reinhard_knee(
        d,
        threshold=threshold,
        limit=limit,
        power=power,
    )
    rgb_compressed = ach[..., None] * (1.0 - d_compressed)
    # Pixels with ach <= 0 keep their original (near-black) values.
    return np.where(ach[..., None] > 1e-12, rgb_compressed, rgb)


# ---------------------------------------------------------------------------
# Output-side: OkLch chroma reduction to the output primaries cube.
# Constant-lightness perceptual chroma reduction, with the boundary being the
# destination RGB cube (the output primaries' [0, 1]³ in linear RGB) rather than
# the visible spectral locus. Perceptual chroma reduction is an output-side-only
# option; the input side is xy-radial against the locus / inscribed hull (see
# InputGamutCompressSpec).
# ---------------------------------------------------------------------------


_OUTPUT_CMAX_CACHE: dict[
    tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]
] = {}

# JzAzBz operates on absolute XYZ in cd/m². For SDR output we anchor
# diffuse white at 100 cd/m² — pixels at linear RGB=1.0 correspond to
# Y = 100 cd/m². Configurable in future via spec; constant for now.
_JZAZBZ_Y_W_CDM2 = 100.0

# CAM16-UCS viewing conditions: typical display review per CIE 159
# guidance — L_A = 64 cd/m² (a fifth of typical room max), Y_b = 20
# (mid-gray surround), Average surround (the colour-science default).
# Adapting whitepoint is the output color space's whitepoint at Y=1.
_CAM16UCS_L_A = 64.0
_CAM16UCS_Y_B = 20.0


def _output_cs_whitepoint_xyz(output_color_space: str) -> np.ndarray:
    """The output color space's whitepoint as XYZ at Y=1 — needed as
    ``XYZ_w`` for CAM16's forward/inverse, kept in lockstep with the
    normalized-XYZ convention used everywhere else in this module."""
    cs = colour.RGB_COLOURSPACES[output_color_space]
    xy = np.asarray(cs.whitepoint, dtype=float)
    return _xy_to_xyz_unit_y(xy)


# Ottosson 2023: OkLab "Lr" — a 1D nonlinear remap of OkLab's L so the
# lightness scale tracks CIELAB L* more closely.
# https://bottosson.github.io/posts/colorpicker/
_OKLRAB_K1 = 0.206
_OKLRAB_K2 = 0.03
_OKLRAB_K3 = (1.0 + _OKLRAB_K1) / (1.0 + _OKLRAB_K2)


def _oklab_L_to_oklrab_Lr(L: np.ndarray) -> np.ndarray:
    """Forward Lr from OkLab L (Ottosson 2023). Monotonic on [0, 1],
    fixes Lr(0)=0 and Lr(1)≈1; outside that range the formula stays
    real and continuous."""
    k1, k2, k3 = _OKLRAB_K1, _OKLRAB_K2, _OKLRAB_K3
    t = k3 * L - k1
    return 0.5 * (t + np.sqrt(t * t + 4.0 * k2 * k3 * L))


def _oklrab_Lr_to_oklab_L(Lr: np.ndarray) -> np.ndarray:
    """Inverse of :func:`_oklab_L_to_oklrab_Lr`."""
    k1, k2, k3 = _OKLRAB_K1, _OKLRAB_K2, _OKLRAB_K3
    return (Lr * (Lr + k1)) / (k3 * (Lr + k2))


def _compress_lightness(
    L: np.ndarray,
    *,
    params: tuple[float, float, float],
    L_white: float,
) -> np.ndarray:
    """One-sided soft compression on a perceptual lightness coordinate.

    A single Reinhard roll-off on the high side: identity below
    ``threshold * L_white``, smoothly asymptotic to ``L_white`` above
    it. This pulls super-bright highlights (``L > L_white``) back into
    ``[0, L_white]`` while leaving shadows and midtones untouched.

    ``(threshold, limit, power)`` are in *normalized* units where
    ``1.0`` corresponds to ``L_white`` — the perceptual lightness of
    the output color space's whitepoint. With the default
    ``(0.7, 1.0, 2.2)`` the roll-off starts at 70% of perceptual white
    and asymptotes exactly at white.

    Black is anchored: an input of 0 normalizes to 0, which sits below
    the threshold and so passes through the Reinhard knee unchanged. The
    earlier two-sided knee mirrored this roll-off around mid-grey and so
    pulled deep shadows toward a non-zero floor — negligible in OkLab's
    cube-root lightness but a visibly raised black in the PQ-based
    (JzAzBz) and CIECAM (CAM16-UCS) lightness scales. Below-black inputs
    (``L < 0``) are left untouched; the simulation is expected to stay at
    Y >= 0 by physical construction.
    """
    threshold, limit, power = params
    L_norm = L / L_white
    L_norm = reinhard_knee(
        L_norm,
        threshold=threshold,
        limit=limit,
        power=power,
    )
    return L_norm * L_white


def _build_polar_perceptual_c_max_table(
    output_color_space: str,
    *,
    polar_to_xyz_unit: callable,
    L_grid: np.ndarray,
    chroma_initial_upper: float,
    n_h: int,
    n_bisect: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bisect to find max chroma at each (L, h) such that the resulting
    linear RGB in ``output_color_space`` is inside [0, 1]^3.

    The perceptual space (OkLab, JzAzBz, …) is parameterized by
    ``polar_to_xyz_unit``: a callable ``(L, a, b) → XYZ`` returning
    XYZ normalized so diffuse white has Y=1 (i.e. the convention
    ``colour.XYZ_to_RGB`` expects with ``apply_cctf_decoding=False``).
    Any absolute-luminance scaling (e.g. JzAzBz's Y_w in cd/m²) is the
    callable's responsibility.
    """
    cs = colour.RGB_COLOURSPACES[output_color_space]
    white = np.asarray(cs.whitepoint, dtype=float)

    h_grid = np.linspace(-np.pi, np.pi, n_h, endpoint=False)
    L_mesh, h_mesh = np.meshgrid(L_grid, h_grid, indexing="ij")

    lo = np.zeros_like(L_mesh)
    hi = np.full_like(L_mesh, chroma_initial_upper)

    for _ in range(n_bisect):
        mid = (lo + hi) * 0.5
        a = mid * np.cos(h_mesh)
        b = mid * np.sin(h_mesh)
        lab = np.stack([L_mesh, a, b], axis=-1).reshape(-1, 3)
        xyz = polar_to_xyz_unit(lab)
        rgb = np.asarray(
            colour.XYZ_to_RGB(
                xyz,
                colourspace=output_color_space,
                illuminant=white,
                apply_cctf_encoding=False,
            )
        )
        # In-gamut iff every channel is in [0, 1]. Small slack absorbs
        # the matrix-inverse floating-point noise that would otherwise
        # leave the largest in-gamut chroma slightly conservative.
        in_gamut = np.all(
            (rgb >= -1e-6) & (rgb <= 1.0 + 1e-6),
            axis=-1,
        ).reshape(L_mesh.shape)
        lo = np.where(in_gamut, mid, lo)
        hi = np.where(in_gamut, hi, mid)
    return L_grid, h_grid, lo


def _get_output_c_max_table(
    space: str,
    output_color_space: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the cached ``C_max(L, h)`` table for ``space`` against
    ``output_color_space``. ``space`` is ``"oklch"`` or ``"jzazbz"``.
    """
    key = (space, output_color_space)
    if key not in _OUTPUT_CMAX_CACHE:
        if space == "oklch":
            _OUTPUT_CMAX_CACHE[key] = _build_polar_perceptual_c_max_table(
                output_color_space,
                polar_to_xyz_unit=lambda lab: np.asarray(
                    colour.Oklab_to_XYZ(lab),
                ),
                L_grid=np.linspace(0.02, 1.0, _OKLCH_CMAX_TABLE_N_L),
                chroma_initial_upper=0.5,
                n_h=_OKLCH_CMAX_TABLE_N_H,
                n_bisect=_OKLCH_CMAX_TABLE_N_BISECT,
            )
        elif space == "oklrab":
            # Polar coord is (Lr, a, b); reconstruct (L, a, b) by
            # inverting the Lr remap, then OkLab → XYZ as usual.
            _OUTPUT_CMAX_CACHE[key] = _build_polar_perceptual_c_max_table(
                output_color_space,
                polar_to_xyz_unit=lambda lab: np.asarray(
                    colour.Oklab_to_XYZ(
                        np.stack(
                            [
                                _oklrab_Lr_to_oklab_L(lab[..., 0]),
                                lab[..., 1],
                                lab[..., 2],
                            ],
                            axis=-1,
                        )
                    ),
                ),
                # Lr is monotonic on [0, 1] with Lr(0)=0 and Lr(1)≈1,
                # so the same grid bounds as oklch are appropriate.
                L_grid=np.linspace(0.02, 1.0, _OKLCH_CMAX_TABLE_N_L),
                chroma_initial_upper=0.5,
                n_h=_OKLCH_CMAX_TABLE_N_H,
                n_bisect=_OKLCH_CMAX_TABLE_N_BISECT,
            )
        elif space == "jzazbz":
            _OUTPUT_CMAX_CACHE[key] = _build_polar_perceptual_c_max_table(
                output_color_space,
                # colour.Jzazbz_to_XYZ returns absolute XYZ in cd/m² —
                # divide by Y_w so the result is normalized (Y=1 at
                # diffuse white) for XYZ_to_RGB.
                polar_to_xyz_unit=lambda jab: (
                    np.asarray(
                        colour.Jzazbz_to_XYZ(jab),
                    )
                    / _JZAZBZ_Y_W_CDM2
                ),
                # Diffuse white at Y_w=100 → Jz ≈ 0.167; grid spans
                # near-black through above-white with margin so the
                # rare super-white sample still hits a sane lookup.
                L_grid=np.linspace(0.002, 0.18, _OKLCH_CMAX_TABLE_N_L),
                # Max realistic Cz at Y_w=100 is ~0.14 (e.g. sRGB pure
                # red is 0.135); 0.3 leaves bisection headroom.
                chroma_initial_upper=0.3,
                n_h=_OKLCH_CMAX_TABLE_N_H,
                n_bisect=_OKLCH_CMAX_TABLE_N_BISECT,
            )
        elif space == "cam16ucs":
            xyz_w = _output_cs_whitepoint_xyz(output_color_space)
            _OUTPUT_CMAX_CACHE[key] = _build_polar_perceptual_c_max_table(
                output_color_space,
                polar_to_xyz_unit=lambda jab: np.asarray(
                    colour.CAM16UCS_to_XYZ(
                        jab,
                        XYZ_w=xyz_w,
                        L_A=_CAM16UCS_L_A,
                        Y_b=_CAM16UCS_Y_B,
                    ),
                ),
                # Jp ≈ 100 at diffuse white; grid spans near-black to
                # a hair above white so rare super-white samples still
                # hit a sane lookup (their max chroma will be ~0).
                L_grid=np.linspace(1.0, 110.0, _OKLCH_CMAX_TABLE_N_L),
                # Max Cp at primary corners is ~75 (sRGB red); 150
                # gives bisection headroom for wider output gamuts.
                chroma_initial_upper=150.0,
                n_h=_OKLCH_CMAX_TABLE_N_H,
                n_bisect=_OKLCH_CMAX_TABLE_N_BISECT,
            )
        else:
            raise ValueError(f"unknown perceptual space {space!r}")
    return _OUTPUT_CMAX_CACHE[key]


def compress_rgb_oklch_chroma(
    rgb: np.ndarray,
    output_color_space: str,
    *,
    threshold: float,
    limit: float,
    power: float,
    lightness_compression: tuple[float, float, float] | None = None,
) -> np.ndarray:
    """OkLch chroma reduction to the output RGB cube.

    For each pixel, convert linear ``output_color_space`` RGB → XYZ →
    OkLab → OkLch (L, C, h). Look up the maximum in-gamut chroma
    ``C_max(L, h)`` for the destination RGB cube and apply the Reinhard
    knee on ``C / C_max``. Reconstruct OkLch → OkLab → XYZ → RGB.

    L (perceptual lightness) and h (perceptual hue) are preserved by
    the chroma step; only the perceptual chroma shrinks. With
    ``limit = 1.0`` the chroma knee asymptotes at ``C / C_max = 1``,
    i.e. the cube boundary itself for the in-range lightness band.

    If ``lightness_compression`` is supplied, a one-sided soft
    compression is applied to ``L`` *before* the chroma step (so the
    ``C_max(L, h)`` lookup runs at the corrected lightness and the cube
    guarantee is exact). This closes the residual where the chroma step
    alone cannot reach: ``L > 1`` super-bright highlights, which collapse
    ``C_max`` to zero and would otherwise remain in the OOG region. Black
    is left at exactly 0.

    Heavier per-pixel than :func:`compress_rgb_aces_rgc`: each call
    runs RGB ↔ XYZ ↔ OkLab, a hypot + atan2, a bilinear table lookup,
    a knee, and cos/sin reconstruction. The OkLch C_max table is
    cached per output color space (one build per illuminant), so
    repeated bakes against the same output space are fast.
    """
    rgb = np.asarray(rgb, dtype=float)
    cs = colour.RGB_COLOURSPACES[output_color_space]
    white = np.asarray(cs.whitepoint, dtype=float)

    # RGB → XYZ → OkLab → OkLch (L, C, h).
    xyz = np.asarray(
        colour.RGB_to_XYZ(
            rgb,
            colourspace=output_color_space,
            illuminant=white,
            apply_cctf_decoding=False,
        )
    )
    lab = np.asarray(colour.XYZ_to_Oklab(xyz))
    L = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]

    # Lightness compression runs first so C_max is looked up at the
    # corrected L; OkLab's perceptual white sits at L = 1.0 so the
    # normalized parameters apply directly.
    if lightness_compression is not None:
        L = _compress_lightness(L, params=lightness_compression, L_white=1.0)

    C = np.hypot(a, b)
    h = np.arctan2(b, a)

    # Compress C against the per-output-space C_max(L, h).
    c_max_lookup_table = _get_output_c_max_table("oklch", output_color_space)
    C_max = _c_max_lookup(L, h, *c_max_lookup_table)
    safe_C_max = np.fmax(C_max, 1e-9)
    d_norm = C / safe_C_max
    d_compressed = reinhard_knee(
        d_norm,
        threshold=threshold,
        limit=limit,
        power=power,
    )
    C_new = d_compressed * safe_C_max

    # OkLch → OkLab → XYZ → RGB.
    a_new = C_new * np.cos(h)
    b_new = C_new * np.sin(h)
    lab_new = np.stack([L, a_new, b_new], axis=-1)
    xyz_new = np.asarray(colour.Oklab_to_XYZ(lab_new))
    rgb_new = np.asarray(
        colour.XYZ_to_RGB(
            xyz_new,
            colourspace=output_color_space,
            illuminant=white,
            apply_cctf_encoding=False,
        )
    )
    return rgb_new


def compress_rgb_oklrab_chroma(
    rgb: np.ndarray,
    output_color_space: str,
    *,
    threshold: float,
    limit: float,
    power: float,
    lightness_compression: tuple[float, float, float] | None = None,
) -> np.ndarray:
    """Oklrab chroma reduction to the output RGB cube.

    Identical to :func:`compress_rgb_oklch_chroma` except the per-pixel
    ``C_max`` lookup is indexed by Ottosson's rebased lightness ``Lr``
    (a 1D nonlinear remap of OkLab ``L``) instead of raw ``L``. Because
    the ``a, b`` axes are unchanged and the chroma compression doesn't
    touch ``L``, the round-trip reduces to: compute ``Lr`` once for the
    lookup, then run the same OkLab forward/inverse with ``L``
    preserved.

    If ``lightness_compression`` is supplied, a one-sided soft
    compression is applied to ``L`` *before* both the ``Lr`` remap and
    the chroma step. OkLab's perceptual white sits at L = 1.0 so the
    normalized parameters apply directly; the Lr remap then maps the
    corrected L to a corrected Lr. Black is left at exactly 0.
    """
    rgb = np.asarray(rgb, dtype=float)
    cs = colour.RGB_COLOURSPACES[output_color_space]
    white = np.asarray(cs.whitepoint, dtype=float)

    xyz = np.asarray(
        colour.RGB_to_XYZ(
            rgb,
            colourspace=output_color_space,
            illuminant=white,
            apply_cctf_decoding=False,
        )
    )
    lab = np.asarray(colour.XYZ_to_Oklab(xyz))
    L = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]

    if lightness_compression is not None:
        L = _compress_lightness(L, params=lightness_compression, L_white=1.0)

    Lr = _oklab_L_to_oklrab_Lr(L)
    C = np.hypot(a, b)
    h = np.arctan2(b, a)

    c_max_lookup_table = _get_output_c_max_table("oklrab", output_color_space)
    C_max = _c_max_lookup(Lr, h, *c_max_lookup_table)
    safe_C_max = np.fmax(C_max, 1e-9)
    d_norm = C / safe_C_max
    d_compressed = reinhard_knee(
        d_norm,
        threshold=threshold,
        limit=limit,
        power=power,
    )
    C_new = d_compressed * safe_C_max

    a_new = C_new * np.cos(h)
    b_new = C_new * np.sin(h)
    # L is preserved through the chroma step (the knee touches only
    # C); the optional lightness knee above is the only thing that
    # changes L, and its result already lives in `L` here.
    lab_new = np.stack([L, a_new, b_new], axis=-1)
    xyz_new = np.asarray(colour.Oklab_to_XYZ(lab_new))
    rgb_new = np.asarray(
        colour.XYZ_to_RGB(
            xyz_new,
            colourspace=output_color_space,
            illuminant=white,
            apply_cctf_encoding=False,
        )
    )
    return rgb_new


def _jzazbz_white_Jz(output_color_space: str) -> float:
    """Jz coordinate of the output color space's whitepoint at the
    absolute reference luminance ``_JZAZBZ_Y_W_CDM2``. Used as the
    perceptual-white normalizer for the lightness knee."""
    cs = colour.RGB_COLOURSPACES[output_color_space]
    xyz_w_rel = _xy_to_xyz_unit_y(np.asarray(cs.whitepoint, dtype=float))
    Jz_w = float(
        np.asarray(
            colour.XYZ_to_Jzazbz(xyz_w_rel * _JZAZBZ_Y_W_CDM2),
        )[..., 0]
    )
    return Jz_w


def compress_rgb_jzazbz_chroma(
    rgb: np.ndarray,
    output_color_space: str,
    *,
    threshold: float,
    limit: float,
    power: float,
    lightness_compression: tuple[float, float, float] | None = None,
) -> np.ndarray:
    """JzCzhz chroma reduction to the output RGB cube.

    Same algorithm shape as :func:`compress_rgb_oklch_chroma` but in
    JzAzBz (Safdar 2017) instead of OkLab. Reference white luminance
    is fixed at 100 cd/m² (``_JZAZBZ_Y_W_CDM2``): linear RGB=1 maps to
    Y=100 cd/m² before the JzAzBz forward, undone after the inverse.

    Use this when oklch's known blue-region hue twist produces visible
    cyan→blue drift on a target output gamut (commonly sRGB and
    Rec.2020). JzAzBz keeps perceived hue stable across the
    magenta↔cyan arc at the cost of a heavier per-pixel transform
    (PQ encoding + matrix vs OkLab's cube-root + matrix).

    If ``lightness_compression`` is supplied, a one-sided soft
    compression is applied to ``Jz`` before the chroma step, normalized
    by the output whitepoint's Jz (≈0.167 at Y_w = 100 cd/m² for typical
    SDR whites). Black (Jz = 0) is left untouched.
    """
    rgb = np.asarray(rgb, dtype=float)
    cs = colour.RGB_COLOURSPACES[output_color_space]
    white = np.asarray(cs.whitepoint, dtype=float)

    # RGB → XYZ (cd/m² at Y_w) → JzAzBz → polar.
    xyz = np.asarray(
        colour.RGB_to_XYZ(
            rgb,
            colourspace=output_color_space,
            illuminant=white,
            apply_cctf_decoding=False,
        )
    )
    jab = np.asarray(colour.XYZ_to_Jzazbz(xyz * _JZAZBZ_Y_W_CDM2))
    Jz = jab[..., 0]
    az = jab[..., 1]
    bz = jab[..., 2]

    if lightness_compression is not None:
        Jz = _compress_lightness(
            Jz,
            params=lightness_compression,
            L_white=_jzazbz_white_Jz(output_color_space),
        )

    Cz = np.hypot(az, bz)
    hz = np.arctan2(bz, az)

    # Compress Cz against the per-output-space Cz_max(Jz, hz).
    c_max_lookup_table = _get_output_c_max_table("jzazbz", output_color_space)
    Cz_max = _c_max_lookup(Jz, hz, *c_max_lookup_table)
    safe_Cz_max = np.fmax(Cz_max, 1e-9)
    d_norm = Cz / safe_Cz_max
    d_compressed = reinhard_knee(
        d_norm,
        threshold=threshold,
        limit=limit,
        power=power,
    )
    Cz_new = d_compressed * safe_Cz_max

    # JzCzhz → JzAzBz → XYZ → RGB.
    az_new = Cz_new * np.cos(hz)
    bz_new = Cz_new * np.sin(hz)
    jab_new = np.stack([Jz, az_new, bz_new], axis=-1)
    xyz_new = np.asarray(colour.Jzazbz_to_XYZ(jab_new)) / _JZAZBZ_Y_W_CDM2
    rgb_new = np.asarray(
        colour.XYZ_to_RGB(
            xyz_new,
            colourspace=output_color_space,
            illuminant=white,
            apply_cctf_encoding=False,
        )
    )
    return rgb_new


def _cam16ucs_white_Jp(output_color_space: str) -> float:
    """Jp coordinate of the output whitepoint under the module's fixed
    CAM16-UCS viewing conditions. Used as the perceptual-white
    normalizer for the lightness knee. ≈100 for typical displays.
    """
    xyz_w = _output_cs_whitepoint_xyz(output_color_space)
    Jp_w = float(
        np.asarray(
            colour.XYZ_to_CAM16UCS(
                xyz_w,
                XYZ_w=xyz_w,
                L_A=_CAM16UCS_L_A,
                Y_b=_CAM16UCS_Y_B,
            ),
        )[..., 0]
    )
    return Jp_w


def compress_rgb_cam16ucs_chroma(
    rgb: np.ndarray,
    output_color_space: str,
    *,
    threshold: float,
    limit: float,
    power: float,
    lightness_compression: tuple[float, float, float] | None = None,
) -> np.ndarray:
    """CAM16-UCS chroma reduction to the output RGB cube.

    Same algorithm shape as :func:`compress_rgb_oklch_chroma` but in
    CIECAM16-UCS (Li & Luo 2017). The adapting whitepoint ``XYZ_w`` is
    the output color space's whitepoint at Y=1; viewing conditions are
    fixed at ``L_A = 64 cd/m²``, ``Y_b = 20``, Average surround — the
    canonical "display review" setup.

    Heavier per-pixel than OkLab or JzAzBz (each call runs the CAM16
    forward and inverse), but produces the cleanest constant-hue
    constraint of the three perceptual options — useful when smoothness
    around the blue/cyan arc matters more than bake time.

    If ``lightness_compression`` is supplied, a one-sided soft
    compression is applied to ``Jp`` before the chroma step, normalized
    by the output whitepoint's Jp (≈100 under the configured viewing
    conditions). Black (Jp = 0) is left untouched.
    """
    rgb = np.asarray(rgb, dtype=float)
    cs = colour.RGB_COLOURSPACES[output_color_space]
    white = np.asarray(cs.whitepoint, dtype=float)
    xyz_w = _output_cs_whitepoint_xyz(output_color_space)

    # RGB → XYZ → CAM16-UCS → polar.
    xyz = np.asarray(
        colour.RGB_to_XYZ(
            rgb,
            colourspace=output_color_space,
            illuminant=white,
            apply_cctf_decoding=False,
        )
    )
    jab = np.asarray(
        colour.XYZ_to_CAM16UCS(
            xyz,
            XYZ_w=xyz_w,
            L_A=_CAM16UCS_L_A,
            Y_b=_CAM16UCS_Y_B,
        )
    )
    Jp = jab[..., 0]
    ap = jab[..., 1]
    bp = jab[..., 2]

    if lightness_compression is not None:
        Jp = _compress_lightness(
            Jp,
            params=lightness_compression,
            L_white=_cam16ucs_white_Jp(output_color_space),
        )

    Cp = np.hypot(ap, bp)
    hp = np.arctan2(bp, ap)

    # Compress Cp against the per-output-space Cp_max(Jp, hp).
    c_max_lookup_table = _get_output_c_max_table("cam16ucs", output_color_space)
    Cp_max = _c_max_lookup(Jp, hp, *c_max_lookup_table)
    safe_Cp_max = np.fmax(Cp_max, 1e-9)
    d_norm = Cp / safe_Cp_max
    d_compressed = reinhard_knee(
        d_norm,
        threshold=threshold,
        limit=limit,
        power=power,
    )
    Cp_new = d_compressed * safe_Cp_max

    # JpCphp → CAM16-UCS → XYZ → RGB.
    ap_new = Cp_new * np.cos(hp)
    bp_new = Cp_new * np.sin(hp)
    jab_new = np.stack([Jp, ap_new, bp_new], axis=-1)
    xyz_new = np.asarray(
        colour.CAM16UCS_to_XYZ(
            jab_new,
            XYZ_w=xyz_w,
            L_A=_CAM16UCS_L_A,
            Y_b=_CAM16UCS_Y_B,
        )
    )
    rgb_new = np.asarray(
        colour.XYZ_to_RGB(
            xyz_new,
            colourspace=output_color_space,
            illuminant=white,
            apply_cctf_encoding=False,
        )
    )
    return rgb_new


def compress_rgb(
    rgb: np.ndarray,
    spec: OutputGamutCompressSpec,
    *,
    output_color_space: str | None = None,
) -> np.ndarray:
    """Apply output gamut compression to RGB values per ``spec``.

    Parameters
    ----------
    rgb :
        RGB array in the destination output primaries (linear),
        shape ``(..., 3)``.
    spec :
        Output compression configuration. ``algorithm='off'`` returns
        ``rgb`` unchanged.
    output_color_space :
        Name of the destination color space (e.g. ``"sRGB"``).
        Required for ``"oklch"`` / ``"jzazbz"`` / ``"cam16ucs"`` because
        those algorithms build / look up the per-color-space
        ``C_max(L, h)`` table. Ignored for ``"aces_rgc"``, which
        operates purely in destination RGB.
    """
    if spec.algorithm == "off":
        return np.asarray(rgb, dtype=float)
    threshold, limit, power = spec.knee
    if spec.algorithm == "aces_rgc":
        return compress_rgb_aces_rgc(
            rgb,
            threshold=threshold,
            limit=limit,
            power=power,
        )
    perceptual_fns = {
        "oklch": compress_rgb_oklch_chroma,
        "oklrab": compress_rgb_oklrab_chroma,
        "jzazbz": compress_rgb_jzazbz_chroma,
        "cam16ucs": compress_rgb_cam16ucs_chroma,
    }
    if spec.algorithm in perceptual_fns:
        if output_color_space is None:
            raise ValueError(
                f"output_color_space is required when algorithm={spec.algorithm!r}"
            )
        return perceptual_fns[spec.algorithm](
            rgb,
            output_color_space=output_color_space,
            threshold=threshold,
            limit=limit,
            power=power,
            lightness_compression=spec.lightness_compression,
        )
    raise ValueError(f"unknown output algorithm {spec.algorithm!r}")


# ---------------------------------------------------------------------------
# TC LUT remap-resample (per n100 §3.1)
# ---------------------------------------------------------------------------


def remap_tc_lut_for_compression(
    tc_lut: np.ndarray,
    reference_illuminant_xy: np.ndarray,
    spec: InputGamutCompressSpec,
    *,
    locus: np.ndarray | None = None,
) -> np.ndarray:
    """Bake the compression into a 128×128×3 tc_lut.

    The runtime lookup path is RGB → CIE xy → tri2quad → tc → bilinear
    sample of ``tc_lut`` → raw. We do **not** want to inject a
    per-sample compression call into that path; instead we remap the
    LUT once at build time so the on-disk LUT internally absorbs the
    compression:

        new_lut[xy] = old_lut[compress(xy)]

    Concretely, for each grid cell of the new LUT we:
      1. Decode its tc index back to a CIE xy via ``_quad2tri``.
      2. Run that xy through the compressor.
      3. Re-encode the compressed xy to tc via ``_tri2quad``.
      4. Bilinearly sample the original LUT at that tc.

    Per-channel bilinear sampling via ``scipy.ndimage.map_coordinates``
    (order=1, ``mode="nearest"`` boundary so out-of-grid tc values get
    the closest-edge LUT value rather than wrapping or extrapolating
    past 0/1 in raw RGB).

    Parameters
    ----------
    tc_lut :
        The per-film ``compute_hanatos2025_tc_lut`` output, shape
        ``(H, W, n_ch)`` (n_ch=3 for color, 1 for BW).
    reference_illuminant_xy :
        The film's reference illuminant xy (the compression's
        achromatic axis). Must match the illuminant used inside
        ``_rgb_to_tc_b`` so the compression operates around the same
        white the runtime evaluates against.
    spec :
        Compression configuration. ``spec.active`` False returns
        ``tc_lut`` unchanged.
    locus :
        Optional spectral locus override (see :func:`compress_xy`).
    """
    if not spec.active:
        return tc_lut

    # Import here to avoid a circular module-load between
    # spectral_upsampling.py and gamut_compression.py.
    from spektrafilm.utils.spectral_upsampling import _quad2tri, _tri2quad

    # Channel-generic: each channel is sampled independently below, so this
    # works for BW (n_ch=1) as well as color (n_ch=3).
    H, W, C = tc_lut.shape

    # Grid of LUT cell positions in tc space (the LUT's own indexing).
    # Cell (i, j) corresponds to tc = (i / (H-1), j / (W-1)).
    i_idx, j_idx = np.meshgrid(
        np.arange(H, dtype=float),
        np.arange(W, dtype=float),
        indexing="ij",
    )
    tc_cells = np.stack([i_idx / (H - 1), j_idx / (W - 1)], axis=-1)

    # Step 1: tc → CIE xy via _quad2tri (LUT's inverse coordinate map).
    xy_cells = _quad2tri(tc_cells)

    # Step 2: apply compression.
    xy_compressed = compress_xy(
        xy_cells.reshape(-1, 2),
        np.asarray(reference_illuminant_xy, dtype=float),
        spec,
        locus=locus,
    ).reshape(H, W, 2)

    # Step 3: CIE xy → tc via _tri2quad (LUT's forward coordinate map).
    tc_compressed = _tri2quad(xy_compressed)

    # Step 4: bilinear sample old LUT at tc_compressed. map_coordinates
    # expects coordinates in *grid index* space (not [0, 1] normalized),
    # so convert tc ∈ [0, 1] to indices ∈ [0, H-1] / [0, W-1].
    coord_i = tc_compressed[..., 0] * (H - 1)
    coord_j = tc_compressed[..., 1] * (W - 1)
    coords = np.stack([coord_i.ravel(), coord_j.ravel()], axis=0)

    new_lut = np.empty_like(tc_lut)
    for ch in range(C):
        sampled = map_coordinates(
            tc_lut[..., ch],
            coords,
            order=1,
            mode="nearest",
        )
        new_lut[..., ch] = sampled.reshape(H, W)
    return new_lut
