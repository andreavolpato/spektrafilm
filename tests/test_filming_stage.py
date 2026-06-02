from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import spektrafilm.runtime.stages.filming as filming_module


def test_rgb_to_film_raw_passes_linear_sensitivity_to_hanatos2025(monkeypatch) -> None:
    """`_rgb_to_film_raw` should pass `10**log_sensitivity` (linear sensitivity)
    to `rgb_to_raw_hanatos2025`. With color_filter='none' the sensitivity is
    passed through unmodified. Hanatos2025 adaptation is handled separately via
    the LUT service's `tc_lut`, not by mutating sensitivity in this method.
    """
    captured: dict[str, np.ndarray] = {}
    log_sensitivity = np.array([[-1.0, -2.0, -3.0], [-0.5, -1.5, -2.5]], dtype=float)

    def fake_rgb_to_raw_hanatos2025(
        rgb,
        sensitivity,
        *,
        color_space=None,
        apply_cctf_decoding=None,
        reference_illuminant=None,
        tc_lut=None,
    ):
        del color_space, apply_cctf_decoding, reference_illuminant, tc_lut
        captured['sensitivity'] = np.asarray(sensitivity, dtype=float)
        return np.ones(rgb.shape, dtype=float)

    monkeypatch.setattr(filming_module, 'rgb_to_raw_hanatos2025', fake_rgb_to_raw_hanatos2025)

    stage = object.__new__(filming_module.FilmingStage)
    setattr(stage, '_film', SimpleNamespace(
        info=SimpleNamespace(reference_illuminant='D55'),
        data=SimpleNamespace(log_sensitivity=log_sensitivity),
    ))
    setattr(stage, '_camera', SimpleNamespace(color_filter='none'))
    setattr(stage, '_settings', SimpleNamespace(rgb_to_raw_method='hanatos2025'))
    setattr(stage, '_lut_service', SimpleNamespace(get_filming_tc_lut=lambda sensitivity: None))

    rgb = np.ones((1, 1, 3), dtype=float)

    getattr(stage, '_rgb_to_film_raw')(rgb)

    np.testing.assert_allclose(captured['sensitivity'], 10.0 ** log_sensitivity)


def test_rgb_to_film_raw_applies_color_filter_to_sensitivity(monkeypatch) -> None:
    """A selected camera color filter multiplies its spectral transmittance into
    the sensitivity (per wavelength, broadcast across channels) before spectral
    upsampling -- no renormalization."""
    captured: dict[str, np.ndarray] = {}
    log_sensitivity = np.array([[-1.0, -2.0, -3.0], [-0.5, -1.5, -2.5]], dtype=float)
    transmittance = np.array([0.5, 0.25], dtype=float)

    def fake_rgb_to_raw_hanatos2025(rgb, sensitivity, **kwargs):
        captured['sensitivity'] = np.asarray(sensitivity, dtype=float)
        return np.ones(rgb.shape, dtype=float)

    monkeypatch.setattr(filming_module, 'rgb_to_raw_hanatos2025', fake_rgb_to_raw_hanatos2025)
    monkeypatch.setattr(filming_module, 'color_filter_transmittance', lambda key: transmittance)

    stage = object.__new__(filming_module.FilmingStage)
    setattr(stage, '_film', SimpleNamespace(
        info=SimpleNamespace(reference_illuminant='D55'),
        data=SimpleNamespace(log_sensitivity=log_sensitivity),
    ))
    setattr(stage, '_camera', SimpleNamespace(color_filter='hoya_y2'))
    setattr(stage, '_settings', SimpleNamespace(rgb_to_raw_method='hanatos2025'))
    setattr(stage, '_lut_service', SimpleNamespace(get_filming_tc_lut=lambda sensitivity: None))

    getattr(stage, '_rgb_to_film_raw')(np.ones((1, 1, 3), dtype=float))

    np.testing.assert_allclose(captured['sensitivity'], 10.0 ** log_sensitivity * transmittance[:, None])
