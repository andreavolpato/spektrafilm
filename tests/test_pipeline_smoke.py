import numpy as np
from spectral_film_lab.runtime.process import photo_process, photo_params


class TestPipelineSmoke:
    """Smoke tests for the full photo simulation pipeline."""

    def test_print_output_valid(self, small_rgb_image, default_params):
        """Pipeline output should be finite and in [0, 1]."""
        result = photo_process(small_rgb_image, default_params)
        assert result.shape[2] == 3
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_negative_output_valid(self, small_rgb_image, default_params):
        """Computing negative scan should also produce valid output."""
        default_params.io.scan_film = True
        result = photo_process(small_rgb_image, default_params)
        assert result.shape[2] == 3
        assert np.all(np.isfinite(result))

    def test_compute_negative_changes_pipeline_branch(self, default_params):
        """Negative scan mode should produce a different valid result than print mode."""
        patch = np.ones((4, 4, 3)) * np.array([0.30, 0.10, 0.05])

        default_params.io.scan_film = False
        print_result = photo_process(patch, default_params)

        default_params.io.scan_film = True
        negative_result = photo_process(patch, default_params)

        assert print_result.shape == (4, 4, 3)
        assert negative_result.shape == (4, 4, 3)
        assert np.all(np.isfinite(print_result))
        assert np.all(np.isfinite(negative_result))
        assert np.all(print_result >= 0.0)
        assert np.all(print_result <= 1.0)
        assert np.all(negative_result >= 0.0)
        assert np.all(negative_result <= 1.0)
        assert not np.allclose(print_result, negative_result, atol=1e-3), (
            "scan_film should change the pipeline branch and produce a different result"
        )

    def test_uniform_gray_input(self, default_params):
        """A uniform midgray image should produce uniform output (no spatial artifacts)."""
        gray = np.ones((8, 8, 3)) * 0.184
        result = photo_process(gray, default_params)
        pixel_0 = result[2, 2, :]
        for i in range(1, 6):
            for j in range(1, 6):
                np.testing.assert_allclose(result[i, j, :], pixel_0, atol=1e-6)

    def test_upscale_factor_increases_output_size(self, default_params):
        """Upscaling with upscale_factor>1 should increase final output dimensions."""
        patch = np.ones((4, 4, 3)) * np.array([0.30, 0.10, 0.05])
        default_params.io.preview_resize_factor = 1.0
        default_params.io.upscale_factor = 2.0

        result = photo_process(patch, default_params)

        assert result.shape == (8, 8, 3)
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)
        
    def test_preview_resize_factor_keeps_output_size(self, default_params):
        """Downscaling with preview_resize_factor<1 should reduce final output dimensions."""
        patch = np.ones((8, 8, 3)) * np.array([0.30, 0.10, 0.05])
        default_params.io.preview_resize_factor = 0.5
        default_params.io.upscale_factor = 1.0

        result = photo_process(patch, default_params)

        assert result.shape == (8, 8, 3)
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_upscale_factor_reduces_output_size(self, default_params):
        """Downscaling with upscale_factor<1 should reduce final output dimensions."""
        patch = np.ones((8, 8, 3)) * np.array([0.30, 0.10, 0.05])
        default_params.io.preview_resize_factor = 1.0
        default_params.io.upscale_factor = 0.5

        result = photo_process(patch, default_params)

        assert result.shape == (4, 4, 3)
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_black_input_no_crash(self, default_params):
        """Black input should not crash (no log(0) or division-by-zero)."""
        black = np.zeros((8, 8, 3))
        result = photo_process(black, default_params)
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_bright_input_no_crash(self, default_params):
        """Very bright inputs should not crash (saturation edge case)."""
        white = np.ones((8, 8, 3)) * 10000
        result = photo_process(white, default_params)
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_monotonicity(self, default_params):
        """Brighter input should produce brighter output — the fundamental transfer curve property."""
        levels = [0.02, 0.05, 0.18, 0.5, 0.90]
        means = []
        for level in levels:
            patch = np.ones((4, 4, 3)) * level
            result = photo_process(patch, default_params)
            means.append(np.mean(result))
        for i in range(len(means) - 1):
            assert means[i] < means[i + 1], (
                f"Monotonicity broken: input {levels[i]} → {means[i]:.4f}, "
                f"input {levels[i+1]} → {means[i+1]:.4f}"
            )

    def test_exposure_compensation(self, default_params):
        """Positive EV should brighten, negative EV should darken, relative to 0 EV."""
        gray = np.ones((4, 4, 3)) * 0.18
        # Disable print exposure compensation so the enlarger doesn't
        # auto-correct for the negative's exposure change.
        default_params.enlarger.print_exposure_compensation = False

        default_params.camera.exposure_compensation_ev = 0.0
        base = np.mean(photo_process(gray, default_params))

        default_params.camera.exposure_compensation_ev = +2.0
        bright = np.mean(photo_process(gray, default_params))

        default_params.camera.exposure_compensation_ev = -2.0
        dark = np.mean(photo_process(gray, default_params))

        assert dark < base < bright

    def test_print_exposure_compensation_dampens_ev_change(self, default_params):
        """Print exposure compensation should reduce the effect of negative EV changes."""
        gray = np.ones((4, 4, 3)) * 0.18

        default_params.camera.exposure_compensation_ev = 0.0
        default_params.enlarger.print_exposure_compensation = True
        base_comp = photo_process(gray, default_params)

        default_params.camera.exposure_compensation_ev = +2.0
        bright_comp = photo_process(gray, default_params)

        default_params.camera.exposure_compensation_ev = 0.0
        default_params.enlarger.print_exposure_compensation = False
        base_no_comp = photo_process(gray, default_params)

        default_params.camera.exposure_compensation_ev = +2.0
        bright_no_comp = photo_process(gray, default_params)

        delta_comp = abs(np.mean(bright_comp) - np.mean(base_comp))
        delta_no_comp = abs(np.mean(bright_no_comp) - np.mean(base_no_comp))

        assert delta_comp < delta_no_comp, (
            f"Expected print exposure compensation to dampen EV changes, "
            f"but compensated delta {delta_comp:.4f} was not smaller than "
            f"uncompensated delta {delta_no_comp:.4f}"
        )
        assert delta_comp < 0.05, (
            f"Expected print exposure compensation to keep the brightness shift small, "
            f"but compensated delta was {delta_comp:.4f}"
        )

    def test_different_film_stocks(self, default_params):
        """Different negative profiles must produce visibly different results."""
        green_patch = np.ones((10, 10, 3)) * np.array([0.05, 0.4, 0.05])
        result_portra = photo_process(green_patch, default_params)  # default is portra 400

        params_fuji = photo_params(film_profile='fujifilm_c200_auc')
        params_fuji.debug.deactivate_spatial_effects = True
        params_fuji.debug.deactivate_stochastic_effects = True
        params_fuji.camera.auto_exposure = False
        result_fuji = photo_process(green_patch, params_fuji)

        # Different stocks should produce numerically different colors
        assert not np.allclose(result_portra, result_fuji, atol=1e-8), \
            "Two different film stocks produced identical output"

    def test_color_differentiation(self, default_params):
        """R, G, B inputs should produce distinct outputs"""
        colors = {
            'red':   np.array([[[0.5, 0.05, 0.05]]]),
            'green': np.array([[[0.05, 0.5, 0.05]]]),
            'blue':  np.array([[[0.05, 0.05, 0.5]]]),
        }
        results = {}
        for name, patch in colors.items():
            tile = np.tile(patch, (4, 4, 1))
            results[name] = photo_process(tile, default_params)[1, 1, :]

        # Each primary should produce a distinct color
        assert not np.allclose(results['red'], results['green'], atol=1e-2)
        assert not np.allclose(results['green'], results['blue'], atol=1e-2)
        assert not np.allclose(results['red'], results['blue'], atol=1e-2)

    def test_deterministic(self, default_params):
        """With stochastic effects off, same input must give identical output."""
        gray = np.ones((4, 4, 3)) * 0.18
        result1 = photo_process(gray, default_params)
        result2 = photo_process(gray, default_params)
        np.testing.assert_array_equal(result1, result2)
    
    def test_lut_vs_direct_consistency(self, default_params):
        """LUT-accelerated path should approximate direct spectral calculation."""
        gray = np.ones((4, 4, 3)) * 0.18

        # Direct spectral path (LUTs off — already the default in default_params)
        result_direct = photo_process(gray, default_params)

        # LUT-accelerated path
        default_params.settings.use_enlarger_lut = True
        default_params.settings.use_scanner_lut = True
        default_params.settings.lut_resolution = 17
        result_lut = photo_process(gray, default_params)

        assert np.all(np.isfinite(result_lut))
        assert np.all(result_lut >= 0.0)
        assert np.all(result_lut <= 1.0)
        np.testing.assert_allclose(result_lut, result_direct, atol=0.02,
            err_msg="LUT path deviates from direct spectral calculation by more than 2%")

    def test_auto_exposure_smoke(self, default_params):
        """Auto-exposure should run without error and normalise brightness toward midgray."""
        bright_patch = np.ones((8, 8, 3)) * 0.8
        default_params.camera.auto_exposure = True
        default_params.enlarger.print_exposure_compensation = False

        for method in ('center_weighted', 'median'):
            default_params.camera.auto_exposure_method = method
            result = photo_process(bright_patch, default_params)
            assert result.shape == (8, 8, 3), f"Shape mismatch with method={method}"
            assert np.all(np.isfinite(result)), f"Non-finite values with method={method}"
            assert np.all(result >= 0.0) and np.all(result <= 1.0)

        # Auto-exposed bright image should be darker than without auto-exposure
        default_params.camera.auto_exposure = False
        default_params.camera.exposure_compensation_ev = 0.0
        result_manual = photo_process(bright_patch, default_params)

        default_params.camera.auto_exposure = True
        default_params.camera.auto_exposure_method = 'center_weighted'
        result_auto = photo_process(bright_patch, default_params)

        assert np.mean(result_auto) < np.mean(result_manual), (
            "Auto-exposure on a bright image should reduce brightness"
        )
        assert np.mean(result_auto) > 0.45 and np.mean(result_auto) < 0.51, (
            f"Auto-exposure produced a flat gray of {np.mean(result_auto):.4f}, expected around 0.48"
        )

    def test_return_film_log_raw_early_return(self, default_params):
        """return_film_log_raw should return raw film exposure before development."""
        gray = np.ones((4, 4, 3)) * 0.18
        default_params.debug.return_film_log_raw = True
        result = photo_process(gray, default_params)
        assert result.shape == (4, 4, 3)
        assert np.all(np.isfinite(result))
