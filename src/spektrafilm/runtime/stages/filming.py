from __future__ import annotations

import numpy as np

from spektrafilm.model.color_filters import compute_band_pass_filter
from spektrafilm.model.diffusion import apply_gaussian_blur_um, apply_halation_um
from spektrafilm.model.emulsion import compute_density_spectral, develop, develop_simple
from spektrafilm.utils.autoexposure import measure_autoexposure_ev
from spektrafilm.utils.spectral_upsampling import rgb_to_raw_hanatos2025, rgb_to_raw_mallett2019
from spektrafilm.utils.timings import timeit


class FilmingStage:
    def __init__(self, film, film_render_params, camera_params, io_params, settings_params,
                 lut_service, resize_service, enlarger_service):
        self._film = film
        self._film_render = film_render_params
        self._camera = camera_params
        self._io = io_params
        self._settings = settings_params
        self._lut_service = lut_service
        self._resize_service = resize_service
        self._enlarger_service = enlarger_service
        self._enlarger_service.density_spectral_midgray = self._compute_density_spectral_midgray_to_balance_print()
        self._pixel_size_um = None

    # public methods

    @timeit("_auto_exposure")
    def auto_exposure(self, image: np.ndarray) -> float:
        if self._camera.auto_exposure:
            small_preview = self._resize_service.small_preview(image)
            autoexposure_ev = measure_autoexposure_ev(
                small_preview,
                self._io.input_color_space,
                self._io.input_cctf_decoding,
                method=self._camera.auto_exposure_method,
            )
            return image * 2 ** autoexposure_ev
        return image

    @timeit("_expose_film")
    def expose(self, image: np.ndarray) -> np.ndarray:
        raw = self._rgb_to_film_raw(
            image,
            color_space=self._io.input_color_space,
            apply_cctf_decoding=self._io.input_cctf_decoding,
        )
        self._pixel_size_um = self._camera.film_format_mm * 1000 / np.max(image.shape[0:2])
        raw *= 2 ** self._camera.exposure_compensation_ev
        raw = apply_gaussian_blur_um(raw, self._camera.lens_blur_um, self._pixel_size_um)
        raw = apply_halation_um(raw, self._film_render.halation, self._pixel_size_um)
        log_raw = np.log10(np.fmax(raw, 0.0) + 1e-10)
        return log_raw

    @timeit("_develop_film")
    def develop(self, log_raw: np.ndarray) -> np.ndarray:
        return develop(
            log_raw,
            self._pixel_size_um,
            self._film.data.log_exposure,
            self._film.data.density_curves,
            self._film.data.density_curves_layers,
            self._film_render.dir_couplers,
            self._film_render.grain,
            self._film.info.type,
            gamma_factor=self._film_render.density_curve_gamma,
            use_fast_stats=self._settings.use_fast_stats,
        )

    # private methods

    def _rgb_to_film_raw(
        self,
        rgb: np.ndarray,
        *,
        color_space: str = "sRGB",
        apply_cctf_decoding: bool = False,
    ) -> np.ndarray:
        sensitivity = 10 ** self._film.data.log_sensitivity
        sensitivity = np.nan_to_num(sensitivity)

        if self._camera.filter_uv[0] > 0 or self._camera.filter_ir[0] > 0:
            band_pass_filter = compute_band_pass_filter(self._camera.filter_uv, self._camera.filter_ir)
            sensitivity *= band_pass_filter[:, None]

        if self._settings.rgb_to_raw_method == "hanatos2025":
            raw = rgb_to_raw_hanatos2025(rgb, sensitivity,
                            color_space=color_space, 
                            apply_cctf_decoding=apply_cctf_decoding, 
                            reference_illuminant=self._film.info.reference_illuminant,
                            tc_lut=self._lut_service.get_filming_tc_lut(sensitivity))
        elif self._settings.rgb_to_raw_method == "mallett2019":
            raw = rgb_to_raw_mallett2019(rgb, sensitivity,
                            color_space=color_space,
                            apply_cctf_decoding=apply_cctf_decoding,
                            reference_illuminant=self._film.info.reference_illuminant)
        return raw
    
    def _compute_density_spectral_midgray_to_balance_print(self):
        if self._enlarger_service.print_exposure_compensation:
            neg_exp_comp_ev = self._camera.exposure_compensation_ev
        else:
            neg_exp_comp_ev = 0.0
        rgb_midgray = np.array([[[0.184] * 3]]) * 2 ** neg_exp_comp_ev
        raw_midgray = self._rgb_to_film_raw(rgb_midgray)
        log_raw_midgray = np.log10(raw_midgray + 1e-10)
        density_midgray = develop_simple(
            log_raw_midgray,
            self._film.data.log_exposure,
            self._film.data.density_curves,
            gamma_factor=self._film_render.density_curve_gamma,
        )
        density_spectral_midgray = compute_density_spectral(
            self._film.data.channel_density,
            density_midgray,
            base_density=self._film.data.base_density,
        )
        return density_spectral_midgray
