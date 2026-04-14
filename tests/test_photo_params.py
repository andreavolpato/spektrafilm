import numpy as np
import spektrafilm.runtime.params_builder as params_builder_module
from pytest import mark

from spektrafilm.runtime.params_builder import digest_params, init_params
from spektrafilm.runtime.process import Simulator


pytestmark = mark.unit


class TestInitParamsDefaults:
    def test_init_params_defaults_contract(self):
        params = init_params()

        for section in (
            'film',
            'print',
            'film_render',
            'print_render',
            'camera',
            'enlarger',
            'scanner',
            'io',
            'debug',
            'settings',
        ):
            assert hasattr(params, section)

        assert hasattr(params.film, 'info')
        assert hasattr(params.film, 'data')
        assert hasattr(params.print, 'info')
        assert hasattr(params.print, 'data')
        assert params.film.info.stock == 'kodak_portra_400'
        assert params.print.info.stock == 'kodak_portra_endura'

        assert params.camera.exposure_compensation_ev == 0.0
        assert params.camera.auto_exposure is True
        assert params.camera.auto_exposure_method == 'center_weighted'
        assert params.camera.lens_blur_um == 0.0
        assert params.camera.film_format_mm == 35.0

        assert params.enlarger.illuminant == 'TH-KG3'
        assert params.enlarger.print_exposure == 1.0
        assert params.enlarger.print_exposure_compensation is True
        assert params.enlarger.normalize_print_exposure is True
        assert params.enlarger.y_filter_shift == 0.0
        assert params.enlarger.m_filter_shift == 0.0
        assert np.isfinite(params.enlarger.y_filter_neutral)
        assert np.isfinite(params.enlarger.m_filter_neutral)
        assert np.isfinite(params.enlarger.c_filter_neutral)

        assert params.scanner.lens_blur == 0.0
        assert params.scanner.white_correction is False
        assert params.scanner.white_level == 0.98
        assert params.scanner.black_correction is False
        assert params.scanner.black_level == 0.01
        assert params.scanner.unsharp_mask == (0.7, 0.7)

        assert params.film_render.density_curve_gamma == 1.0
        assert params.film_render.grain.active is True
        assert params.film_render.halation.active is True
        assert params.film_render.dir_couplers.active is True
        assert params.film_render.dir_couplers.amount == 1.0
        assert params.film_render.dir_couplers.ratio_rgb is None

        assert params.print_render.density_curve_gamma == 1.0
        assert params.print_render.glare.active is True

        assert params.io.input_color_space == 'ProPhoto RGB'
        assert params.io.input_cctf_decoding is False
        assert params.io.output_color_space == 'sRGB'
        assert params.io.output_cctf_encoding is True
        assert params.io.crop is False
        assert params.io.upscale_factor == 1.0
        assert params.io.scan_film is False

        assert params.debug.deactivate_spatial_effects is False
        assert params.debug.deactivate_stochastic_effects is False
        assert params.debug.output_film_log_raw is False
        assert params.debug.output_film_density_cmy is False
        assert params.debug.output_print_density_cmy is False
        assert params.debug.print_timings is False

        assert params.settings.rgb_to_raw_method == 'hanatos2025'
        assert params.settings.use_enlarger_lut is False
        assert params.settings.use_scanner_lut is False
        assert params.settings.lut_resolution == 17
        assert params.settings.use_fast_stats is False
        assert params.settings.preview_max_size == 640

class TestSimulatorDebugSwitches:
    def test_deactivate_spatial_effects_params(self):
        params = init_params()
        params.debug.deactivate_spatial_effects = True

        photo = Simulator(digest_params(params))

        assert photo.film_render.halation.size_um == [0, 0, 0]
        assert photo.film_render.halation.scattering_size_um == [0, 0, 0]
        assert photo.film_render.dir_couplers.diffusion_size_um == 0
        assert photo.film_render.grain.blur == 0.0
        assert photo.film_render.grain.blur_dye_clouds_um == 0.0
        assert photo.print_render.glare.blur == 0
        assert photo.camera.lens_blur_um == 0.0
        assert photo.enlarger.lens_blur == 0.0
        assert photo.scanner.lens_blur == 0.0
        assert photo.scanner.unsharp_mask == (0.0, 0.0)

    def test_deactivate_stochastic_effects_params(self):
        params = init_params()
        params.debug.deactivate_stochastic_effects = True

        photo = Simulator(digest_params(params))

        assert photo.film_render.grain.active is False
        assert photo.print_render.glare.active is False


class TestDigestParamsFilmDefaults:
    def test_negative_profile_keeps_explicit_scan_film_choice(self):
        params = init_params()
        params.io.scan_film = True

        digest_params(params)

        assert params.io.scan_film is True

    def test_missing_neutral_filter_database_entry_keeps_current_filters(self, monkeypatch):
        params = init_params()
        params.enlarger.c_filter_neutral = 12.0
        params.enlarger.m_filter_neutral = 34.0
        params.enlarger.y_filter_neutral = 56.0

        monkeypatch.setattr(
            params_builder_module,
            '_get_neutral_print_filters',
            lambda: {},
        )

        digest_params(params)

        assert params.enlarger.c_filter_neutral == 12.0
        assert params.enlarger.m_filter_neutral == 34.0
        assert params.enlarger.y_filter_neutral == 56.0
