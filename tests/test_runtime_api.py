import numpy as np
import pytest

from spektrafilm import AgXPhoto, Simulator, init_params, photo_params, simulate
from spektrafilm.model.stocks import FilmStocks, PrintPapers


pytestmark = pytest.mark.integration


class TestRuntimeApi:
    def test_simulate_matches_simulator_process(self, small_rgb_image, default_params):
        new_result = simulate(small_rgb_image, default_params)
        direct_result = Simulator(default_params).process(small_rgb_image)

        np.testing.assert_allclose(new_result, direct_result, atol=1e-12)

    def test_public_runtime_symbols_remain_available(self):
        params = init_params()
        simulator = Simulator(params)

        assert params.film.info.stock == 'kodak_portra_400'
        assert isinstance(simulator, Simulator)

    def test_art_extlut_compatibility_path_runs(self):
        """reference this https://github.com/artraweditor/ART/blob/master/tools/extlut/spektrafilm_mklut.py"""
        def make_art_params():
            params = photo_params(
                FilmStocks.kodak_portra_400.value,
                PrintPapers.kodak_portra_endura.value,
            )
            params.camera.auto_exposure = False
            params.camera.auto_exposure_method = 'median'
            params.camera.exposure_compensation_ev = 0.0
            params.debug.deactivate_spatial_effects = True
            params.debug.deactivate_stochastic_effects = True
            params.enlarger.lens_blur = 0.0
            params.enlarger.m_filter_shift = 0.0
            params.enlarger.print_exposure = 1.0
            params.enlarger.print_exposure_compensation = True
            params.enlarger.y_filter_shift = 0.0
            params.io.compute_negative = False
            params.io.crop = False
            params.io.full_image = True
            params.io.input_cctf_decoding = False
            params.io.input_color_space = 'sRGB'
            params.io.output_cctf_encoding = False
            params.io.output_color_space = 'ACES2065-1'
            params.io.preview_resize_factor = 1.0
            params.io.upscale_factor = 1.0
            params.scanner.lens_blur = 0.0
            params.scanner.unsharp_mask = (0.0, 0.0)
            params.settings.use_camera_lut = False
            params.settings.use_enlarger_lut = False
            params.settings.use_scanner_lut = False
            params.settings.rgb_to_raw_method = 'mallett2019'
            params.film_render.grain.active = False
            params.film_render.halation.active = False
            params.film_render.density_curve_gamma = 1.0
            params.film_render.dir_couplers.active = True
            params.film_render.dir_couplers.amount = 1.0
            params.print_render.glare.active = False
            params.print_render.density_curve_gamma = 1.0
            return params

        image = np.array([[[0.184, 0.184, 0.184]]], dtype=np.float64)

        params = make_art_params()
        assert params.io.compute_negative is False
        assert params.io.full_image is True
        assert params.io.preview_resize_factor == 1.0
        assert params.settings.use_camera_lut is False

        output = AgXPhoto(params).process(image)
        assert output.shape == image.shape
        assert np.isfinite(output).all()

        shifted_params = make_art_params()
        shifted_params.enlarger.y_filter_shift = 0.5
        shifted_params.enlarger.m_filter_shift = -0.5
        shifted_output = AgXPhoto(shifted_params).process(image)
        assert shifted_output.shape == image.shape
        assert np.isfinite(shifted_output).all()