import numpy as np
from spectral_film_lab.model.couplers import (
    compute_dir_couplers_matrix,
    compute_density_curves_before_dir_couplers,
    compute_exposure_correction_dir_couplers,
)


class TestDirCouplers:
    def test_no_diffusion_is_diagonal(self):
        """With zero diffusion, the matrix should be diagonal (no cross-layer effect)."""
        matrix = compute_dir_couplers_matrix([0.7, 0.7, 0.5], layer_diffusion=0)
        off_diagonal = matrix - np.diag(np.diag(matrix))
        np.testing.assert_allclose(off_diagonal, 0, atol=1e-10)

    def test_zero_couplers_returns_original_curves(self):
        """With zero coupler amounts, density curves should be unchanged."""
        log_exposure = np.linspace(-3, 1, 100)
        density_curves = np.column_stack([
            np.clip(log_exposure + 1.5, 0, 2.5),
            np.clip(log_exposure + 1.5, 0, 2.5),
            np.clip(log_exposure + 1.5, 0, 2.0),
        ])
        matrix = compute_dir_couplers_matrix([0, 0, 0])
        result = compute_density_curves_before_dir_couplers(
            density_curves, log_exposure, matrix
        )
        np.testing.assert_allclose(result, density_curves, atol=1e-10)

    def test_zero_density_no_exposure_correction(self):
        """When density is zero, couplers should not alter the exposure."""
        log_raw = np.ones((8, 8, 3)) * (-1.0)
        density_cmy = np.zeros((8, 8, 3))
        density_max = np.array([2.5, 2.5, 2.0])
        matrix = compute_dir_couplers_matrix([0.7, 0.7, 0.5])
        result = compute_exposure_correction_dir_couplers(
            log_raw, density_cmy, density_max, matrix, diffusion_size_pixel=0
        )
        np.testing.assert_allclose(result, log_raw, atol=1e-10)

