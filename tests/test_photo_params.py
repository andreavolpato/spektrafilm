import numpy as np
import spektrafilm.runtime.params_builder as params_builder_module
from pytest import mark

from spektrafilm.runtime.params_builder import digest_params, init_params


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

        digest_params(params)

        assert params.film_render.halation.scatter_core_um == (0.0, 0.0, 0.0)
        assert params.film_render.halation.scatter_tail_um == (0.0, 0.0, 0.0)
        assert params.film_render.halation.halation_first_sigma_um == (0.0, 0.0, 0.0)
        assert params.film_render.dir_couplers.diffusion_size_um == 0
        assert params.film_render.grain.blur == 0.0
        assert params.film_render.grain.blur_dye_clouds_um == 0.0
        assert params.print_render.glare.blur == 0
        assert params.camera.lens_blur_um == 0.0
        assert params.enlarger.lens_blur == 0.0
        assert params.scanner.lens_blur == 0.0
        assert params.scanner.unsharp_mask == (0.0, 0.0)

    def test_deactivate_stochastic_effects_params(self):
        params = init_params()
        params.debug.deactivate_stochastic_effects = True

        digest_params(params)

        assert params.film_render.grain.active is False
        assert params.print_render.glare.active is False


class TestDigestParamsFilmDefaults:
    def test_negative_profile_keeps_explicit_scan_film_choice(self):
        params = init_params()
        params.io.scan_film = True

        digest_params(params)

        assert params.io.scan_film is True

    def test_halation_preset_picks_still_weak_for_kodak_gold_200(self):
        params = digest_params(init_params(film_profile='kodak_gold_200'))
        assert params.film.info.use == 'still'
        assert params.film.info.antihalation == 'weak'
        assert params.film_render.halation.halation_first_sigma_um == (65.0, 65.0, 65.0)
        assert params.film_render.halation.halation_strength == (0.08, 0.02, 0.0)

    def test_halation_preset_picks_still_strong_for_kodak_portra_400(self):
        params = digest_params(init_params(film_profile='kodak_portra_400'))
        assert params.film.info.use == 'still'
        assert params.film.info.antihalation == 'strong'
        assert params.film_render.halation.halation_first_sigma_um == (65.0, 65.0, 65.0)
        assert params.film_render.halation.halation_strength == (0.015, 0.005, 0.0)

    def test_halation_preset_picks_cine_strong_for_kodak_vision3_250d(self):
        params = digest_params(init_params(film_profile='kodak_vision3_250d'))
        assert params.film.info.use == 'cine'
        assert params.film.info.antihalation == 'strong'
        assert params.film_render.halation.halation_first_sigma_um == (50.0, 50.0, 50.0)
        assert params.film_render.halation.halation_strength == (0.015, 0.005, 0.0)

    def test_halation_preset_covers_cine_no_for_cinestill_like_stocks(self):
        # Cinestill = Vision3 with rem-jet removed. Base is still PET (cine sigma_h),
        # but the antihalation layer is gone (no strength).
        params = init_params(film_profile='kodak_vision3_250d')
        params.film.info.antihalation = 'no'
        digest_params(params)
        assert params.film_render.halation.halation_first_sigma_um == (50.0, 50.0, 50.0)
        assert params.film_render.halation.halation_strength == (0.30, 0.10, 0.015)

    def test_halation_preset_covers_still_no_for_redscale_like_stocks(self):
        # Redscale reverses the film so the antihalation backing faces the lens,
        # effectively disabling it. Base is still triacetate (still sigma_h).
        params = init_params(film_profile='kodak_portra_400')
        params.film.info.antihalation = 'no'
        digest_params(params)
        assert params.film_render.halation.halation_first_sigma_um == (65.0, 65.0, 65.0)
        assert params.film_render.halation.halation_strength == (0.30, 0.10, 0.015)

    def test_halation_preset_covers_cine_weak_for_older_cine_stocks(self):
        # Older cine stocks (pre-Vision3 ECN negatives) kept a PET base but had
        # less effective antihalation than the modern line.
        params = init_params(film_profile='kodak_vision3_250d')
        params.film.info.antihalation = 'weak'
        digest_params(params)
        assert params.film_render.halation.halation_first_sigma_um == (50.0, 50.0, 50.0)
        assert params.film_render.halation.halation_strength == (0.08, 0.02, 0.0)

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
