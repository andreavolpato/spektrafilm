"""Stimulus pattern generators for the QA suite.

Each function returns a flat array of shape ``(M, 3)`` of *encoded*
RGB values in the bundle's input color space, suitable for direct
LUT application or for ``reference.run_pipeline_at``.

The patterns are deliberately small (a few hundred samples each) — they
target specific failure modes, not coverage. The bulk off-grid coverage
lives in ``reference.ReferenceSamples``.
"""
from __future__ import annotations

import numpy as np

import colour


def neutral_ramp(input_color_space: str, n: int = 64) -> np.ndarray:
    """Equal R = G = B sweeping the input domain ``[0, 1]``.

    Probes the achromatic axis of the system — for the characteristic
    curve test, this is the cleanest interpretation of "the film's
    response to a neutral target."
    """
    del input_color_space  # encoded inputs are uniform [0,1] regardless of space
    t = np.linspace(0.0, 1.0, n)
    return np.stack([t, t, t], axis=-1)


def highlight_ramps_per_channel(
    input_color_space: str,
    n: int = 64,
    lo: float | None = None,
    hi: float = 1.0,
) -> tuple[np.ndarray, float]:
    """Per-channel ramp from middle-gray to max encoded in input space.

    Returns ``(samples, lo_used)`` where ``samples`` has shape
    ``(3 * n, 3)``. The first ``n`` rows ramp R alone with G, B held
    at the input space's middle-gray-encoded value; the next ``n``
    ramp G; the last ``n`` ramp B.

    The ramp's low end (``lo``) defaults to the input color space's
    **middle-gray-encoded** value — i.e. linear 0.18 through the input
    CCTF. So:

    - sRGB: lo ≈ 0.46 → covers ~2.5 stops above middle gray (the
      sRGB highlight zone).
    - V-Log: lo ≈ 0.42 → covers ~8 stops above middle gray (the
      log highlight headroom).
    - ACEScg (linear): lo = 0.18 → covers ~2.5 stops above middle
      gray.

    Each input space gets a sweep honest to its own dynamic range.
    Without this anchoring, ``lo=0.4`` was below mid-gray for sRGB
    but mid-dark for V-Log, producing visually confusing curves.

    The non-swept channels are also pinned at the same middle-gray-
    encoded value so the ramp traces a *neutral-gray-anchored*
    highlight sweep across input spaces. ``lo_used`` is the actual
    encoded value used, returned so callers can label the figure.
    """
    if lo is None:
        from spektrafilm_lut_creator.color_spaces import encode_cctf
        mid_gray_linear = np.full((1, 3), 0.18, dtype=float)
        mid_gray_encoded = encode_cctf(mid_gray_linear, input_color_space)
        lo = float(np.asarray(mid_gray_encoded).flatten()[0])

    pin = lo
    t = np.linspace(lo, hi, n)
    ramps = []
    for axis in range(3):
        block = np.full((n, 3), pin)
        block[:, axis] = t
        ramps.append(block)
    return np.concatenate(ramps, axis=0), float(lo)


def planckian_sweep(
    input_color_space: str,
    cct_range_k: tuple[float, float] = (2700.0, 10000.0),
    n: int = 16,
    exposure_ev: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Daylight-locus white points across a CCT range.

    Returns ``(samples_encoded, cct_array)``:

    - ``samples_encoded``: shape ``(n, 3)``, "perfect white surface
      under illuminant CCT" expressed as encoded RGB in the input
      space. Built by computing each illuminant's chromaticity, setting
      Y=1, converting XYZ → input-space RGB → CCTF-encoded.
    - ``cct_array``: shape ``(n,)`` of the CCT values, for plotting.

    A spektrafilm pipeline that handles white-balance gracefully sends
    these samples to a smooth, monotone curve in the output's
    chromaticity. Kinks or fold-backs reveal handling bugs.

    References
    ----------
    - CIE 15:2018 daylight phase recommendation for CCT > 4000K.
    - Planckian locus for low CCT.
    """
    from spektrafilm_lut_creator.color_spaces import encode_qa_input, get as get_cs

    entry = get_cs(input_color_space)
    cct = np.linspace(cct_range_k[0], cct_range_k[1], n)
    # Use the daylight locus for CCT >= 4000K (the standard CIE
    # convention) and the Planckian locus for lower CCT.
    xy = np.zeros((n, 2))
    for i, t in enumerate(cct):
        if t >= 4000.0:
            xy[i] = colour.temperature.CCT_to_xy_CIE_D(t)
        else:
            xy[i] = colour.temperature.CCT_to_xy_CIE_D(4000.0)
            # Below 4000K the CIE D function is undefined; use the
            # Planckian locus via Robertson 1968 instead.
            xy[i] = colour.temperature.CCT_to_xy(t, method="Kang 2002")
    # Scale to Y = 1 (perfect diffuse white).
    xyz = np.zeros((n, 3))
    xyz[:, 0] = xy[:, 0] / np.clip(xy[:, 1], 1e-6, None)
    xyz[:, 1] = 1.0
    xyz[:, 2] = (1.0 - xy[:, 0] - xy[:, 1]) / np.clip(xy[:, 1], 1e-6, None)
    linear_rgb = np.asarray(
        colour.XYZ_to_RGB(xyz, colourspace=entry.primaries, apply_cctf_encoding=False),
        dtype=float,
    )
    # Normalize each row so the maximum channel = 1 — this is what a
    # camera would see if exposed for the brightest channel. Avoids
    # samples that fall outside [0,1] in the encoded space after CCTF.
    peak = np.clip(np.max(linear_rgb, axis=-1, keepdims=True), 1e-6, None)
    linear_rgb = linear_rgb / peak
    samples_encoded = encode_qa_input(
        np.clip(linear_rgb, 0.0, 1.0), input_color_space, exposure_ev,
    )
    return np.asarray(samples_encoded, dtype=np.float32), cct


def saturated_cube_edges(n: int = 33) -> tuple[np.ndarray, list[np.ndarray]]:
    """Trace the 12 edges of the encoded input ``[0, 1]^3``.

    Returns ``(samples, segments)``:

    - ``samples``: shape ``(12 * n, 3)`` of encoded RGB values along
      every edge, sampled at ``n`` points per edge.
    - ``segments``: list of 12 arrays of shape ``(n, 3)`` — the same
      samples but grouped by edge. Convenient for per-edge plotting.

    These are the most-saturated colors a LUT can be fed. The edges
    form the canonical hue + saturation cycle and are the natural
    stimulus for hue-twist and spectral-locus-envelope tests.
    """
    sweep = np.linspace(0.0, 1.0, n)
    pin = [0.0, 1.0]
    segments: list[np.ndarray] = []
    # Three groups of 4 edges, one per axis being swept.
    for a in pin:
        for b in pin:
            # R sweep, G=a, B=b
            seg = np.stack([sweep, np.full(n, a), np.full(n, b)], axis=-1)
            segments.append(seg)
    for a in pin:
        for b in pin:
            # G sweep, R=a, B=b
            seg = np.stack([np.full(n, a), sweep, np.full(n, b)], axis=-1)
            segments.append(seg)
    for a in pin:
        for b in pin:
            # B sweep, R=a, G=b
            seg = np.stack([np.full(n, a), np.full(n, b), sweep], axis=-1)
            segments.append(seg)
    return np.concatenate(segments, axis=0), segments


def dynamic_range_neutral_ramp(
    input_color_space: str,
    *,
    stop_lo: float = -8.0,
    stop_hi: float | None = None,
    n: int = 257,
    middle_gray_linear: float = 0.18,
    margin_stops: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Neutral ramp sampled uniformly in **scene-linear log2 stops**.

    Returns ``(stops, encoded_rgb, encoded_clip_mask)``:

    - ``stops``: shape ``(n,)`` linear stops above/below middle gray.
      When ``stop_hi`` is ``None`` (default), the upper bound is
      derived from the input encoding so the ramp spans the full
      headroom the encoding carries. For V-Log that is ~+8 EV; for
      ARRI LogC4 ~+11.35 EV; for sRGB / Rec.709 ~+2.47 EV. A
      ``margin_stops`` pad is added past that ceiling so the
      encoding-clip cliff sits inside the ramp and the figure marks
      it. The lower bound stays fixed (default ``-8`` EV): log
      encodings have soft footroom rather than a hard cliff on the
      low side, so adapting the bottom doesn't surface anything and
      a fixed floor keeps stop-resolution consistent across cameras.
    - ``encoded_rgb``: shape ``(n, 3)``, scene-linear values
      ``0.18 * 2^stop`` encoded for the input color space (CCTF
      applied) and clipped to ``[0, 1]``. This is what gets fed to
      the LUT.
    - ``encoded_clip_mask``: shape ``(n,)`` boolean, ``True`` where
      the pre-clip encoded value was outside ``[0, 1]`` and got
      clipped (i.e., these stops aren't representable in the input
      encoding — important context for the dynamic-range viz).

    The stop axis is the unit colorists think in; this is what
    distinguishes the "D vs log E" film characteristic plot from the
    encoded-code transfer plot.

    Adapting the upper bound to the input encoding matters most for
    log curves with asymmetric headroom. V-Log places encoded 1.0 at
    exactly +8 EV above middle gray, so a fixed +8 ceiling happens
    to fit. ARRI LogC4 places encoded 1.0 at +11.35 EV — a fixed +8
    ceiling misses ~3.4 stops of the camera's representable
    headroom, and ``encoded_range_stops`` then reports the window
    rather than the encoding. Probing ``encode_cctf`` to find the
    ceiling fixes that.

    References
    ----------
    - 0.18 middle-gray convention: ANSI/SMPTE RP 180.
    - Stops as the unit: any cinematography reference (Hunt, Holben).
    - V-Log spec: Panasonic V-Log Reference Manual.
    - LogC4 spec: ARRI LogC4 White Paper (2022).
    """
    from spektrafilm_lut_creator.color_spaces import encode_cctf

    if stop_hi is None:
        stop_hi = _encoding_stop_ceiling(
            input_color_space, middle_gray_linear,
        ) + margin_stops
    if stop_hi <= stop_lo:
        raise ValueError(f"stop_hi {stop_hi} must exceed stop_lo {stop_lo}")
    stops = np.linspace(stop_lo, stop_hi, n)
    linear_value = middle_gray_linear * (2.0 ** stops)
    linear_rgb = np.stack([linear_value] * 3, axis=-1)
    pre_clip = encode_cctf(linear_rgb, input_color_space)
    # Track which stops fall outside the input encoding's representable
    # range (encoded < 0 or > 1) so the viz can mark the cliff edges.
    encoded_clip_mask = np.any((pre_clip < 0.0) | (pre_clip > 1.0), axis=-1)
    encoded = np.clip(pre_clip, 0.0, 1.0).astype(np.float32)
    return stops, encoded, encoded_clip_mask


def _encoding_stop_ceiling(
    input_color_space: str, middle_gray_linear: float,
) -> float:
    """Highest stop above middle gray the encoding still fits in code
    ``[0, 1]``.

    Probes the encoding curve over a wide window and returns the
    largest stop whose pre-clip encoded value stays ≤ 1.0. For V-Log
    this is +8.00 EV; for ARRI LogC4 +11.35 EV; for sRGB / Rec.709
    +2.47 EV.
    """
    from spektrafilm_lut_creator.color_spaces import encode_cctf

    probe = np.linspace(-12.0, 20.0, 3201)
    lin = middle_gray_linear * (2.0 ** probe)
    enc = encode_cctf(np.stack([lin] * 3, axis=-1), input_color_space)
    ok = np.all(enc <= 1.0, axis=-1)
    if not ok.any():
        raise ValueError(
            f"input encoding {input_color_space!r} clips at every probed "
            f"stop — encoding may be invalid"
        )
    idx = np.where(ok)[0]
    return float(probe[idx[-1]])


def spectral_locus_chromaticities() -> np.ndarray:
    """xy chromaticities along the CIE 1931 2° spectral locus, 380–780 nm.

    Returns a closed-loop ``(N, 2)`` array (last sample = first
    sample). Used for the spectral-locus-envelope viz; doesn't directly
    feed the pipeline because the locus extends outside any RGB gamut.
    """
    cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    wavelengths = np.arange(380, 781, 1)
    xyz = np.asarray(cmfs[wavelengths], dtype=float)
    xy = np.asarray(colour.XYZ_to_xy(xyz), dtype=float)
    return np.vstack([xy, xy[:1]])
