"""The convert-film 3D LUT (``use_convert_lut``) must match the exact per-pixel
Gauss-Newton solver to within an imperceptible tolerance, and its cache must
rebuild only when the scan model (illuminant / base / dye) changes — not when the
device calibration or exposure gain (the analytic pre-transform) is tuned.
"""

from __future__ import annotations

import numpy as np
import pytest

from spektrafilm.runtime.pipeline import SimulationPipeline
from spektrafilm.runtime.services.spectral_lut_compute import SpectralLUTService

from .conftest import make_fast_test_params


def _realistic_negative(size: int = 64) -> np.ndarray:
    """A synthetic scanned negative: film a photo-like scene and scan it.

    The scene is mid-tone biased and moderately saturated (like a real photograph),
    not the fully-saturated random speckle that would push pixels into the
    ill-conditioned deep-density corner. That corner is stress-tested separately in
    the c40 study; here we assert the common case stays imperceptible.
    """
    params = make_fast_test_params()
    params.workflow.route = "input > film > scan"
    params.io.input_color_space = params.io.output_color_space = "sRGB"
    params.io.input_cctf_decoding = False
    params.io.output_cctf_encoding = False
    rng = np.random.default_rng(0)
    base = rng.uniform(0.12, 0.85, (size, size, 3))
    scene = 0.6 * base + 0.4 * base.mean(axis=-1, keepdims=True)  # pull toward neutral
    return SimulationPipeline(params).process(scene, collect="rgb_out")


def _convert_pipeline(route: str, *, use_convert_lut: bool) -> SimulationPipeline:
    params = make_fast_test_params()
    params.workflow.route = route
    params.io.input_color_space = params.io.output_color_space = "sRGB"
    params.io.input_cctf_decoding = False
    params.io.output_cctf_encoding = False
    params.settings.use_convert_lut = use_convert_lut
    return SimulationPipeline(params)


@pytest.mark.parametrize(
    "route",
    [
        "input > convert-film > scan",
        "input > convert-film > print > scan",
    ],
)
def test_convert_lut_matches_solver(route) -> None:
    neg = _realistic_negative()
    out_solver = _convert_pipeline(route, use_convert_lut=False).process(
        neg, collect="rgb_out"
    )
    out_lut = _convert_pipeline(route, use_convert_lut=True).process(
        neg, collect="rgb_out"
    )
    assert out_lut.shape == out_solver.shape
    assert np.all(np.isfinite(out_lut))
    # Imperceptible on a realistic negative (c40: dE mean ~0.03). The recovered-
    # density discrepancy lives in ill-conditioned directions that barely move RGB.
    # Bound sized for the near-boundary output knee (4260c7b): its curvature above
    # 0.95 of the gamut boundary raises the print route's mean to ~7e-3.
    assert np.mean(np.abs(out_lut - out_solver)) < 8e-3
    assert np.max(np.abs(out_lut - out_solver)) < 4e-2


def test_convert_lut_off_is_exact_solver() -> None:
    """use_convert_lut=False must route straight to the solver (bit-for-bit): the
    stage output equals the converter's own convert() on the decoded image."""
    neg = _realistic_negative()
    pipe = _convert_pipeline("input > convert-film > scan", use_convert_lut=False)
    stage = pipe._converting_stage
    out = pipe.process(neg, collect="cmy_film")
    expected = stage._converter.convert(stage._decode_to_linear(neg))
    np.testing.assert_allclose(out, expected, rtol=0, atol=0)


def test_convert_invert_pretransform_compose() -> None:
    """convert(rgb) == invert(pretransform(rgb))."""
    pipe = _convert_pipeline("input > convert-film > scan", use_convert_lut=False)
    conv = pipe._converting_stage._converter
    rgb = _realistic_negative(16)
    np.testing.assert_allclose(
        conv.convert(rgb), conv.invert(conv.pretransform(rgb)), rtol=0, atol=0
    )


def test_gamut_bounds_contain_negative_channels() -> None:
    """The gamut box must extend below zero (XYZ->RGB makes saturated dye mixes
    go slightly negative); otherwise the LUT would clamp those pixels."""
    pipe = _convert_pipeline("input > convert-film > scan", use_convert_lut=False)
    lo, hi = pipe._converting_stage._converter.gamut_bounds()
    assert lo.shape == (3,) and hi.shape == (3,)
    assert np.all(hi > lo)
    assert np.any(lo < 0.0)


def test_convert_lut_cache_invalidation() -> None:
    """The cached LUT is reused while the inverse is unchanged, and rebuilt when
    the scan model changes — detected by the fixed probe."""
    pipe = _convert_pipeline("input > convert-film > scan", use_convert_lut=True)
    neg = _realistic_negative(24)
    pipe.process(neg, collect="cmy_film")
    svc = pipe._lut_service
    first = svc.convert_lut_memory
    assert first is not None

    # Tuning the analytic pre-transform (gain) must NOT rebuild the LUT.
    pipe._converting_stage._converter.gain = np.float32(1.4)
    pipe.process(neg, collect="cmy_film")
    assert svc.convert_lut_memory is first

    # Changing the scan model (here the dye stack) changes the probe -> rebuild.
    conv = pipe._converting_stage._converter
    conv.dyeT = conv.dyeT * np.float32(1.05)
    conv._refresh_seed()
    pipe.process(neg, collect="cmy_film")
    assert svc.convert_lut_memory is not first


def test_spectral_compute_convert_passthrough_when_off() -> None:
    """With use_lut=False the service calls the function directly and never bakes."""
    svc = SpectralLUTService(lut_resolution=9)
    calls = {"n": 0}

    def invert_fn(rgb):
        calls["n"] += 1
        return rgb * 2.0

    data = np.linspace(0, 1, 2 * 2 * 3).reshape(2, 2, 3)
    out = svc.spectral_compute_convert(
        data,
        invert_fn=invert_fn,
        data_min=np.zeros(3),
        data_max=np.ones(3),
        use_lut=False,
    )
    np.testing.assert_allclose(out, data * 2.0)
    assert svc.convert_lut_memory is None
    assert calls["n"] == 1
