from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import spektrafilm.runtime.services.filter_enlarger_source as filter_source_module
from spektrafilm.config import SPECTRAL_SHAPE
from spektrafilm.model.color_filters import color_enlarger
from spektrafilm.runtime.services.filter_enlarger_source import EnlargerService


pytestmark = pytest.mark.unit


def _make_enlarger_params() -> SimpleNamespace:
    return SimpleNamespace(
        print_exposure_compensation=True,
        c_filter_neutral=11.0,
        m_filter_neutral=22.0,
        y_filter_neutral=33.0,
        m_filter_shift=4.0,
        y_filter_shift=5.0,
        preflash_m_filter_shift=6.0,
        preflash_y_filter_shift=7.0,
    )


def test_enlarger_service_passes_filters_in_cmy_order(monkeypatch) -> None:
    params = _make_enlarger_params()
    captured_calls: list[np.ndarray] = []

    def fake_color_enlarger(light_source, filter_cc_values):
        del light_source
        captured_calls.append(np.asarray(filter_cc_values, dtype=np.float64))
        return np.ones(3, dtype=np.float64)

    monkeypatch.setattr(filter_source_module, 'color_enlarger', fake_color_enlarger)
    service = EnlargerService(params)
    light_source = np.ones(3, dtype=np.float64)

    service.enlarger_filtered_illuminant(light_source)
    service.preflash_filtered_illuminant(light_source)

    np.testing.assert_allclose(captured_calls[0], np.array([11.0, 26.0, 38.0]))
    np.testing.assert_allclose(captured_calls[1], np.array([11.0, 28.0, 40.0]))


def test_color_enlarger_cc_filters_target_expected_spectral_bands() -> None:
    wavelengths = SPECTRAL_SHAPE.wavelengths
    light_source = np.ones_like(wavelengths, dtype=np.float64)
    bands = {
        'blue': wavelengths < 480,
        'green': (wavelengths >= 500) & (wavelengths < 600),
        'red': wavelengths >= 620,
    }
    cases = [
        ((100.0, 0.0, 0.0), 'red'),
        ((0.0, 100.0, 0.0), 'green'),
        ((0.0, 0.0, 100.0), 'blue'),
    ]

    for filter_values, attenuated_band in cases:
        filtered = color_enlarger(light_source, filter_values)
        band_means = {
            band: float(np.nanmean(filtered[mask]))
            for band, mask in bands.items()
        }

        assert band_means[attenuated_band] < 0.2
        assert band_means[attenuated_band] == min(band_means.values())
        for band, mean in band_means.items():
            if band != attenuated_band:
                assert mean > 0.7


@pytest.mark.parametrize('cc_value', [30.0, 60.0, 100.0])
def test_color_enlarger_cc_scale_matches_density_definition(cc_value: float) -> None:
    wavelengths = SPECTRAL_SHAPE.wavelengths
    light_source = np.ones_like(wavelengths, dtype=np.float64)

    filtered = color_enlarger(light_source, (0.0, 0.0, cc_value))
    expected_minimum_transmittance = 10 ** (-cc_value / 100.0)

    assert np.nanmin(filtered) == pytest.approx(expected_minimum_transmittance, abs=1e-3)