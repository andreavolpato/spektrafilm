import numpy as np
from spectral_film_lab.engine.parametric import parametric_density_curves_model


class TestParametricDensityCurvesModel:
    def test_monotonically_increasing(self):
        """Density curves for a negative film should be monotonically non-decreasing."""
        log_exposure = np.linspace(-3, 2, 200)
        gamma = [0.6, 0.6, 0.6]
        log_exposure_0 = [-1.5, -1.5, -1.5]
        density_max = [2.5, 2.5, 2.5]
        toe_size = [0.3, 0.3, 0.3]
        shoulder_size = [0.5, 0.5, 0.5]
        result = parametric_density_curves_model(
            log_exposure, gamma, log_exposure_0, density_max, toe_size, shoulder_size
        )
        for ch in range(3):
            diff = np.diff(result[:, ch])
            assert np.all(diff >= -1e-10), f"Channel {ch} is not monotonically increasing"

    def test_density_near_zero_at_low_exposure(self):
        """At very low exposures, density should be near zero."""
        log_exposure = np.linspace(-6, 2, 200)
        gamma = [0.6, 0.6, 0.6]
        log_exposure_0 = [-1.0, -1.0, -1.0]
        density_max = [2.5, 2.5, 2.5]
        toe_size = [0.3, 0.3, 0.3]
        shoulder_size = [0.5, 0.5, 0.5]
        result = parametric_density_curves_model(
            log_exposure, gamma, log_exposure_0, density_max, toe_size, shoulder_size
        )
        assert np.all(result[:5, :] < 0.01)
