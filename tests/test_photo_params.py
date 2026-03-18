import numpy as np

from spectral_film_lab.runtime.process import AgXPhoto, photo_params


class TestPhotoParamsDefaults:
    def test_photo_params_defaults_contract(self):
        params = photo_params()

        for section in (
            'negative',
            'print_paper',
            'negative_render',
            'print_render',
            'camera',
            'enlarger',
            'scanner',
            'io',
            'debug',
            'settings',
        ):
            assert hasattr(params, section)

        assert hasattr(params.negative, 'info')
        assert hasattr(params.negative, 'data')
        assert hasattr(params.print_paper, 'info')
        assert hasattr(params.print_paper, 'data')
        assert params.negative.info.stock == 'kodak_portra_400_auc'
        assert params.print_paper.info.stock == 'kodak_portra_endura_uc'

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

        assert params.negative_render.density_curve_gamma == 1.0
        assert params.negative_render.base_density_scale == 1.0
        assert params.negative_render.grain.active is True
        assert params.negative_render.halation.active is True
        assert params.negative_render.dir_couplers.active is True

        assert params.print_render.density_curve_gamma == 1.0
        assert params.print_render.base_density_scale == 0.4
        assert params.print_render.glare.active is True

        assert params.io.input_color_space == 'ProPhoto RGB'
        assert params.io.input_cctf_decoding is False
        assert params.io.output_color_space == 'sRGB'
        assert params.io.output_cctf_encoding is True
        assert params.io.crop is False
        assert params.io.preview_resize_factor == 1.0
        assert params.io.upscale_factor == 1.0
        assert params.io.full_image is False
        assert params.io.compute_negative is False
        assert params.io.compute_film_raw is False

        assert params.debug.deactivate_spatial_effects is False
        assert params.debug.deactivate_stochastic_effects is False
        assert params.debug.return_negative_density_cmy is False
        assert params.debug.return_print_density_cmy is False
        assert params.debug.print_timings is False
        assert hasattr(params.debug, 'luts')
        assert params.debug.luts.enlarger_lut is None
        assert params.debug.luts.scanner_lut is None

        assert params.settings.rgb_to_raw_method == 'hanatos2025'
        assert params.settings.use_enlarger_lut is False
        assert params.settings.use_scanner_lut is False
        assert params.settings.lut_resolution == 17
        assert params.settings.use_fast_stats is False

class TestAgXPhotoDebugSwitches:
    def test_deactivate_spatial_effects_params(self):
        params = photo_params()
        params.debug.deactivate_spatial_effects = True

        photo = AgXPhoto(params)

        assert photo.negative_render.halation.size_um == [0, 0, 0]
        assert photo.negative_render.halation.scattering_size_um == [0, 0, 0]
        assert photo.negative_render.dir_couplers.diffusion_size_um == 0
        assert photo.negative_render.grain.blur == 0.0
        assert photo.negative_render.grain.blur_dye_clouds_um == 0.0
        assert photo.print_render.glare.blur == 0
        assert photo.camera.lens_blur_um == 0.0
        assert photo.enlarger.lens_blur == 0.0
        assert photo.scanner.lens_blur == 0.0
        assert photo.scanner.unsharp_mask == (0.0, 0.0)

    def test_deactivate_stochastic_effects_params(self):
        params = photo_params()
        params.debug.deactivate_stochastic_effects = True

        photo = AgXPhoto(params)

        assert photo.negative_render.grain.active is False
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

        assert photo.debug.luts.enlarger_lut is not None
        assert photo.debug.luts.scanner_lut is not None

    def test_accepts_legacy_dict_like_params(self):
        params = photo_params()
        legacy_params = {
            'negative': params.negative,
            'print_paper': params.print_paper,
            'negative_render': vars(params.negative_render).copy(),
            'print_render': vars(params.print_render).copy(),
            'camera': vars(params.camera).copy(),
            'enlarger': vars(params.enlarger).copy(),
            'scanner': vars(params.scanner).copy(),
            'io': vars(params.io).copy(),
            'debug': {
                **vars(params.debug),
                'luts': vars(params.debug.luts).copy(),
            },
            'settings': vars(params.settings).copy(),
        }
        legacy_params['debug']['deactivate_spatial_effects'] = True

        photo = AgXPhoto(legacy_params)

        assert photo.debug.deactivate_spatial_effects is True
        assert photo.negative_render.halation.size_um == [0, 0, 0]
        assert photo.debug.luts.enlarger_lut is None