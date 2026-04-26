import numpy as np
import pytest
from spektrafilm.model.density_curves import interpolate_exposure_to_density
from spektrafilm.model.couplers import (
    apply_density_correction_dir_couplers,
    compute_dir_couplers_matrix,
    compute_density_curves_before_dir_couplers,
    compute_exposure_correction_dir_couplers,
)
from spektrafilm.runtime.params_schema import DirCouplersParams


pytestmark = pytest.mark.unit


class TestDirCouplers:
    def test_no_interlayer_inhibition_is_diagonal(self):
        """With zero interlayer inhibition, the matrix should be diagonal (no cross-layer effect)."""
        params = DirCouplersParams(inhibition_interlayer=0.0)
        matrix = compute_dir_couplers_matrix(params)
        off_diagonal = matrix - np.diag(np.diag(matrix))
        np.testing.assert_allclose(off_diagonal, 0, atol=1e-10)

    def test_zero_couplers_returns_original_curves(self):
        """With zero coupler inhibition, density curves should be unchanged."""
        log_exposure = np.linspace(-3, 1, 100)
        density_curves = np.column_stack([
            np.clip(log_exposure + 1.5, 0, 2.5),
            np.clip(log_exposure + 1.5, 0, 2.5),
            np.clip(log_exposure + 1.5, 0, 2.0),
        ])
        params = DirCouplersParams(
            inhibition_samelayer=0.0,
            inhibition_interlayer=0.0,
        )
        matrix = compute_dir_couplers_matrix(params)
        result = compute_density_curves_before_dir_couplers(
            density_curves, log_exposure, matrix
        )
        np.testing.assert_allclose(result, density_curves, atol=1e-10)

    def test_zero_density_no_exposure_correction(self):
        """When density is zero, couplers should not alter the exposure."""
        log_raw = np.ones((8, 8, 3)) * (-1.0)
        density_cmy = np.zeros((8, 8, 3))
        density_max = np.array([2.5, 2.5, 2.0])
        matrix = compute_dir_couplers_matrix(DirCouplersParams())
        result = compute_exposure_correction_dir_couplers(
            log_raw, density_cmy, density_max, matrix, diffusion_size_pixel=0
        )
        np.testing.assert_allclose(result, log_raw, atol=1e-10)

    @pytest.mark.parametrize("profile_type", ["negative", "positive"])
    def test_apply_density_correction_dir_couplers_matches_manual_pipeline(self, profile_type):
        log_exposure = np.linspace(-3.0, 1.0, 100)
        density_curves = np.column_stack([
            np.clip(log_exposure + 1.8, 0.0, 2.4),
            np.clip(log_exposure + 1.6, 0.0, 2.2),
            np.clip(log_exposure + 1.4, 0.0, 2.0),
        ])
        log_raw = np.full((4, 4, 3), -0.8)
        log_raw[:, :, 1] -= 0.2
        density_cmy = interpolate_exposure_to_density(log_raw, density_curves, log_exposure, 1.1)
        dir_couplers = DirCouplersParams(
            active=True,
            inhibition_strength=0.7,
            inhibition_samelayer=0.9,
            inhibition_interlayer=1.1,
            gamma_samelayer_rgb=(0.5, 0.4, 0.3),
            gamma_interlayer_r_to_gb=(0.3, 0.25),
            gamma_interlayer_g_to_rb=(0.2, 0.25),
            gamma_interlayer_b_to_rg=(0.15, 0.2),
            diffusion_size_um=6.0,
        )
        pixel_size_um = 2.0
        gamma_factor = 1.1
        positive = profile_type == "positive"

        result = apply_density_correction_dir_couplers(
            density_cmy,
            log_raw,
            pixel_size_um,
            log_exposure,
            density_curves,
            dir_couplers,
            profile_type,
            gamma_factor=gamma_factor,
        )

        matrix = compute_dir_couplers_matrix(dir_couplers) * dir_couplers.inhibition_strength
        density_curves_0 = compute_density_curves_before_dir_couplers(
            density_curves,
            log_exposure,
            matrix,
            positive=positive,
        )
        density_max = np.nanmax(density_curves, axis=0)
        log_raw_0 = compute_exposure_correction_dir_couplers(
            log_raw,
            density_cmy,
            density_max,
            matrix,
            dir_couplers.diffusion_size_um / pixel_size_um,
            positive=positive,
        )
        expected = interpolate_exposure_to_density(
            log_raw_0,
            density_curves_0,
            log_exposure,
            gamma_factor,
        )

        np.testing.assert_allclose(result, expected, atol=1e-10)
