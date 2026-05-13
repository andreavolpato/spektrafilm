import numpy as np
import pytest

from spektrafilm.profiles.io import Hanatos2025SensitivityAdaptation
from spektrafilm.utils import spectral_upsampling as spectral_upsampling_module


pytestmark = pytest.mark.unit


def test_rgb_to_raw_hanatos2025_computes_tc_lut_when_missing(monkeypatch):
    sensitivity = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=np.float64,
    )
    rgb = np.zeros((2, 3, 3), dtype=np.float64)

    def fake_rgb_to_tc_b(data, **_kwargs):
        tc = np.zeros(data.shape[:-1] + (2,), dtype=np.float64)
        if data.shape == (1, 1, 3):
            scale = np.ones((1, 1), dtype=np.float64)
        else:
            scale = np.full(data.shape[:-1], 2.0, dtype=np.float64)
        return tc, scale

    lut_calls = []

    def fake_compute_hanatos2025_tc_lut(arg_sensitivity, _adaptation):
        lut_calls.append(arg_sensitivity.copy())
        return np.zeros((2, 2, 3), dtype=np.float64)

    def fake_apply_lut_cubic_2d(_tc_lut, tc):
        lut_raw = np.empty(tc.shape[:-1] + (3,), dtype=np.float64)
        lut_raw[..., 0] = 2.0
        lut_raw[..., 1] = 4.0
        lut_raw[..., 2] = 6.0
        return lut_raw

    monkeypatch.setattr(spectral_upsampling_module, '_rgb_to_tc_b', fake_rgb_to_tc_b)
    monkeypatch.setattr(spectral_upsampling_module, 'compute_hanatos2025_tc_lut', fake_compute_hanatos2025_tc_lut)
    monkeypatch.setattr(spectral_upsampling_module, 'apply_lut_cubic_2d', fake_apply_lut_cubic_2d)

    raw = spectral_upsampling_module.rgb_to_raw_hanatos2025(
        rgb,
        sensitivity,
        color_space='sRGB',
        apply_cctf_decoding=False,
        reference_illuminant='D65',
    )

    assert len(lut_calls) == 1
    np.testing.assert_allclose(lut_calls[0], sensitivity)
    expected = np.empty_like(raw)
    expected[..., 0] = 4.0
    expected[..., 1] = 8.0
    expected[..., 2] = 12.0
    assert raw.shape == (2, 3, 3)
    np.testing.assert_allclose(raw, expected)


def test_rgb_to_raw_hanatos2025_lut_path_supports_image_rgb(monkeypatch):
    sensitivity = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=np.float64,
    )
    rgb = np.zeros((2, 3, 3), dtype=np.float64)

    def fake_rgb_to_tc_b(data, **_kwargs):
        tc = np.zeros(data.shape[:-1] + (2,), dtype=np.float64)
        if data.shape == (1, 1, 3):
            scale = np.ones((1, 1), dtype=np.float64)
        else:
            scale = np.full(data.shape[:-1], 2.0, dtype=np.float64)
        return tc, scale

    def fake_apply_lut_cubic_2d(_tc_lut, tc):
        lut_raw = np.empty(tc.shape[:-1] + (3,), dtype=np.float64)
        lut_raw[..., 0] = 2.0
        lut_raw[..., 1] = 4.0
        lut_raw[..., 2] = 6.0
        return lut_raw

    monkeypatch.setattr(spectral_upsampling_module, '_rgb_to_tc_b', fake_rgb_to_tc_b)
    monkeypatch.setattr(spectral_upsampling_module, 'apply_lut_cubic_2d', fake_apply_lut_cubic_2d)

    raw = spectral_upsampling_module.rgb_to_raw_hanatos2025(
        rgb,
        sensitivity,
        color_space='sRGB',
        apply_cctf_decoding=False,
        reference_illuminant='D65',
        tc_lut=np.zeros((2, 2, 3), dtype=np.float64),
    )

    expected = np.empty_like(raw)
    expected[..., 0] = 4.0
    expected[..., 1] = 8.0
    expected[..., 2] = 12.0
    assert raw.shape == (2, 3, 3)
    np.testing.assert_allclose(raw, expected)


def test_spectral_bandpass_windows_return_wavelength_channel_arrays():
    erf4 = spectral_upsampling_module.eval_erf4_spectral_bandpass(
        np.array([415.0, 12.0, 667.0, 76.0], dtype=np.float64)
    )
    logiflex8 = spectral_upsampling_module.eval_logiflex8_spectral_bandpass(
        np.array([415.0, 12.0, 667.0, 76.0, 430.0, 650.0, 1.0, 1.0], dtype=np.float64)
    )

    assert erf4.shape == (81, 3)
    assert logiflex8.shape == (81, 3)
    np.testing.assert_allclose(erf4[:, 0], erf4[:, 1])
    np.testing.assert_allclose(erf4[:, 1], erf4[:, 2])


def test_compute_hanatos2025_tc_lut_normalizes_window_to_preserve_midgray(monkeypatch):
    lut = np.array(
        [
            [[1.0, 10.0], [2.0, 20.0]],
            [[3.0, 30.0], [4.0, 40.0]],
        ],
        dtype=np.float64,
    )
    sensitivity = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=np.float64,
    )
    window = np.array(
        [
            [0.5, 0.25, 0.75],
            [0.8, 0.6, 0.4],
        ],
        dtype=np.float64,
    )
    illuminant = np.array([2.0, 4.0], dtype=np.float64)

    adaptation = Hanatos2025SensitivityAdaptation(
        window_params=np.array([415.0, 12.0, 667.0, 76.0], dtype=np.float64),
        reference_illuminant='D55',
        apply_window=True,
        apply_surface=False,
    )

    monkeypatch.setattr(spectral_upsampling_module, 'HANATOS2025_SPECTRA_LUT', lut)
    monkeypatch.setattr(spectral_upsampling_module, 'eval_spectral_bandpass_window', lambda _params: window)
    monkeypatch.setattr(spectral_upsampling_module, 'standard_illuminant', lambda _label: illuminant)

    raw_lut = spectral_upsampling_module.compute_hanatos2025_tc_lut(sensitivity, adaptation)

    normalization = np.sum(sensitivity * illuminant[:, None] * window, axis=0) / np.sum(sensitivity * illuminant[:, None], axis=0)
    expected = np.einsum('ijl,lm->ijm', lut, sensitivity * (window / normalization))
    np.testing.assert_allclose(raw_lut, expected)