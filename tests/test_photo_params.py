import numpy as np
import pytest

from spektrafilm.runtime.process import AgXPhoto, photo_params


class TestPhotoParamsDefaults:
    def test_photo_params_defaults_contract(self):
        params = photo_params()

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
        assert params.film.info.stock == 'kodak_portra_400_auc'
        assert params.print.info.stock == 'kodak_portra_endura_uc'

        assert params.camera.exposure_compensation_ev == 0.0
        assert params.camera.auto_exposure is True
        assert params.camera.auto_exposure_method == 'center_weighted'
        assert params.camera.lens_blur_um == 0.0
        assert params.camera.film_format_mm == 35.0

        assert params.enlarger.illuminant == 'TH-KG3-L'
        assert params.enlarger.print_exposure == 1.0
        assert params.enlarger.print_exposure_compensation is True
        assert params.enlarger.y_filter_shift == 0.0
        assert params.enlarger.m_filter_shift == 0.0
        assert params.enlarger.just_preflash is False
        assert np.isfinite(params.enlarger.y_filter_neutral)
        assert np.isfinite(params.enlarger.m_filter_neutral)
        assert np.isfinite(params.enlarger.c_filter_neutral)

        assert params.scanner.lens_blur == 0.0
        assert params.scanner.unsharp_mask == (0.7, 0.7)

        assert params.film_render.density_curve_gamma == 1.0
        assert params.film_render.base_density_scale == 1.0
        assert params.film_render.grain.active is True
        assert params.film_render.halation.active is True
        assert params.film_render.dir_couplers.active is True

        assert params.print_render.density_curve_gamma == 1.0
        assert params.print_render.base_density_scale == 0.4
        assert params.print_render.glare.active is True

        assert params.io.input_color_space == 'ProPhoto RGB'
        assert params.io.input_cctf_decoding is False
        assert params.io.output_color_space == 'sRGB'
        assert params.io.output_cctf_encoding is True
        assert params.io.crop is False
        assert params.io.preview_resize_factor == 0.3
        assert params.io.upscale_factor == 1.0
        assert params.io.full_image is False
        assert params.io.scan_film is False

        assert params.debug.deactivate_spatial_effects is False
        assert params.debug.deactivate_stochastic_effects is False
        assert params.debug.return_film_log_raw is False
        assert params.debug.return_film_density_cmy is False
        assert params.debug.return_print_density_cmy is False
        assert params.debug.print_timings is False

        assert params.settings.rgb_to_raw_method == 'hanatos2025'
        assert params.settings.use_enlarger_lut is False
        assert params.settings.use_scanner_lut is False
        assert params.settings.lut_resolution == 32
        assert params.settings.use_fast_stats is False

class TestAgXPhotoDebugSwitches:
    def test_deactivate_spatial_effects_params(self):
        params = photo_params()
        params.debug.deactivate_spatial_effects = True

        photo = AgXPhoto(params)

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
        params = photo_params()
        params.debug.deactivate_stochastic_effects = True

        photo = AgXPhoto(params)

        assert photo.film_render.grain.active is False
        assert photo.print_render.glare.active is False


class TestRuntimeParamsCompatibility:
    def test_lut_storage_path_is_initialized(self):
        params = photo_params()
        params.debug.deactivate_spatial_effects = True
        params.debug.deactivate_stochastic_effects = True
        params.settings.use_enlarger_lut = True
        params.settings.use_scanner_lut = True
        photo = AgXPhoto(params)

        image = np.ones((4, 4, 3), dtype=np.float64) * 0.18
        photo.process(image)
