from __future__ import annotations

import pytest
import numpy as np

from spektrafilm.runtime.services.spectral_lut_compute import SpectralLUTService


pytestmark = pytest.mark.unit


def test_get_filming_tc_lut_reuses_cache_for_equal_arrays(monkeypatch) -> None:
    service = SpectralLUTService(lut_resolution=17)
    sensitivity = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    calls: list[np.ndarray] = []

    def fake_compute(arg: np.ndarray) -> np.ndarray:
        calls.append(np.array(arg, copy=True))
        return np.array(arg, copy=True)

    monkeypatch.setattr(
        "spektrafilm.runtime.services.spectral_lut_compute.compute_hanatos2025_tc_lut",
        fake_compute,
    )

    first = service.get_filming_tc_lut(sensitivity)
    second = service.get_filming_tc_lut(np.array(sensitivity, copy=True))

    assert len(calls) == 1
    np.testing.assert_array_equal(first, second)


def test_spectral_compute_reuses_cached_enlarger_lut(monkeypatch) -> None:
    service = SpectralLUTService(lut_resolution=17)
    image = np.full((2, 2, 3), 0.5, dtype=np.float64)
    lut_calls: list[object] = []

    def fake_compute_with_lut(data, _function, xmin, xmax, steps, lut=None):
        _ = (xmin, xmax, steps)
        lut_calls.append(lut)
        if lut is None:
            return np.asarray(data) + 0.1, 'cached-lut'
        return np.asarray(data) + 0.1, lut

    monkeypatch.setattr(
        'spektrafilm.runtime.services.spectral_lut_compute.compute_with_lut',
        fake_compute_with_lut,
    )

    service.spectral_compute(
        image,
        spectral_calculation=lambda data: np.asarray(data) * 2.0,
        data_min=0.0,
        data_max=1.0,
        use_lut=True,
        use_enlarger_lut_memory=True,
    )
    service.spectral_compute(
        image,
        spectral_calculation=lambda data: np.asarray(data) * 2.0,
        data_min=0.0,
        data_max=1.0,
        use_lut=True,
        use_enlarger_lut_memory=True,
    )

    assert lut_calls == [None, 'cached-lut']
