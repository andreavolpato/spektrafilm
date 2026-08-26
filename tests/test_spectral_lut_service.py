import numpy as np
import pytest

from spektrafilm.data.profiles_loader import Hanatos2025SensitivityAdaptation
from spektrafilm.runtime.services import (
    spectral_lut_compute as spectral_lut_compute_module,
)

pytestmark = pytest.mark.unit


def test_filming_tc_lut_recomputes_when_spectral_gaussian_blur_changes(
    monkeypatch,
) -> None:
    calls: list[float] = []

    # The service dispatches the irradiance method through the generic
    # compute_tc_lut, forwarding the adaptation via hanatos2025_adaptation=.
    def fake_compute_tc_lut(
        method,
        sensitivity,
        reference_illuminant=None,
        spectra_lut=None,
        gamut_compress=None,
        hanatos2025_adaptation=None,
    ):
        del method, sensitivity, reference_illuminant, spectra_lut, gamut_compress
        calls.append(float(hanatos2025_adaptation.spectral_gaussian_blur))
        return np.full(
            (2, 2, 3), hanatos2025_adaptation.spectral_gaussian_blur + 1.0, dtype=float
        )

    monkeypatch.setattr(
        spectral_lut_compute_module,
        "compute_tc_lut",
        fake_compute_tc_lut,
    )

    service = spectral_lut_compute_module.SpectralLUTService(lut_resolution=17)
    sensitivity = np.ones((4, 3), dtype=float)
    adaptation = Hanatos2025SensitivityAdaptation(
        window_params=np.empty((0,), dtype=float),
        surface_params=np.empty((0, 3), dtype=float),
        spectral_gaussian_blur=0.0,
        reference_illuminant="D55",
        apply_window=True,
        apply_surface=True,
    )

    service.set_hanatos2025_adaptation(adaptation)
    first = service.get_filming_tc_lut("hanatos2025", sensitivity, "D55")

    adaptation.spectral_gaussian_blur = 4.0
    service.set_hanatos2025_adaptation(adaptation)
    second = service.get_filming_tc_lut("hanatos2025", sensitivity, "D55")

    assert calls == [0.0, 4.0]
    assert np.array_equal(first, second) is False
