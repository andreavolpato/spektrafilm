from __future__ import annotations

import numpy as np
from opt_einsum import contract

from spektrafilm.model.emulsion import compute_density_spectral, develop_simple
from spektrafilm.model.density_curves import remove_viewing_glare_comp
from spektrafilm.model.illuminants import standard_illuminant
from spektrafilm.utils.timings import timeit
from spektrafilm.utils.conversions import density_to_light


class PrintingStage:
    def __init__(
        self,
        film,
        film_render_params,
        print,
        print_render_params,
        enlarger_params,
        settings_params,
        lut_service,
        enlarger_service,
    ):
        self._film = film
        self._film_render = film_render_params
        self._print = print
        self._print_render = print_render_params
        self._enlarger = enlarger_params
        self._settings = settings_params
        self._lut_service = lut_service
        self._enlarger_service = enlarger_service

    @timeit("_expose_print")
    def expose(self, film_density_channels: np.ndarray) -> np.ndarray:
        return self._lut_service.compute(
            film_density_channels,
            spectral_calculation=self.film_cmy_to_print_log_raw,
            data_min=-np.array(self._film_render.grain.density_min),
            data_max=np.nanmax(self._film.data.density_curves, axis=0),
            use_lut=self._settings.use_enlarger_lut,
            save_enlarger_lut=True,
        )

    @timeit("_develop_print")
    def develop(self, log_raw: np.ndarray) -> np.ndarray:
        return develop_simple(
            log_raw,
            self._print.data.log_exposure,
            self._print_corrected_density_curves(),
            gamma_factor=self._print_render.density_curve_gamma,
        )

    def film_cmy_to_print_log_raw(self, density_channels: np.ndarray) -> np.ndarray:
        sensitivity = 10 ** self._print.data.log_sensitivity
        sensitivity = np.nan_to_num(sensitivity)
        enlarger_light_source = standard_illuminant(self._enlarger.illuminant)

        raw = np.zeros_like(density_channels)
        if not self._enlarger.just_preflash:
            density_spectral = compute_density_spectral(
                self._film.data.channel_density,
                density_channels,
                base_density=self._film.data.base_density,
                base_density_scale=self._film_render.base_density_scale,
            )
            print_illuminant = self._enlarger_service.enlarger_filtered_illuminant(enlarger_light_source)
            light = density_to_light(density_spectral, print_illuminant)
            raw = contract("ijk, kl->ijl", light, sensitivity)
            raw *= self._enlarger.print_exposure
            raw *= self.compute_exposure_factor_midgray(sensitivity, print_illuminant)

        raw_preflash = self.compute_raw_preflash(enlarger_light_source, sensitivity)
        raw += raw_preflash
        return np.log10(np.fmax(raw, 0.0) + 1e-10)

    def compute_raw_preflash(self, light_source, sensitivity):
        if self._enlarger.preflash_exposure > 0:
            preflash_illuminant = self._enlarger_service.preflash_filtered_illuminant(light_source)
            density_base = np.asarray(self._film.data.base_density)[None, None, :]
            light_preflash = density_to_light(density_base, preflash_illuminant)
            raw_preflash = contract("ijk, kl->ijl", light_preflash, sensitivity)
            return raw_preflash * self._enlarger.preflash_exposure
        return np.zeros((3,))

    def compute_exposure_factor_midgray(self, sensitivity, print_illuminant):
        if not self._enlarger.normalize_print_exposure:
            return 1.0

        density_spectral_midgray = self._enlarger_service.density_spectral_midgray
        if density_spectral_midgray is None:
            return 1.0

        light_midgray = density_to_light(density_spectral_midgray, print_illuminant)
        raw_midgray = contract("ijk, kl->ijl", light_midgray, sensitivity)
        raw_midgray = np.fmax(raw_midgray, 1e-10)
        # use the geometric mean to normalize the exposure
        raw_midgray_geomean = np.exp(np.mean(np.log(raw_midgray), axis=2, keepdims=True))
        return 1 / raw_midgray_geomean

    def _print_corrected_density_curves(self):
        if self._print_render.glare.compensation_removal_factor > 0:
            log_exposure = self._print.data.log_exposure
            density_curves = self._print.data.density_curves
            density_curves = remove_viewing_glare_comp(
                log_exposure,
                density_curves,
                factor=self._print_render.glare.compensation_removal_factor,
                density=self._print_render.glare.compensation_removal_density,
                transition=self._print_render.glare.compensation_removal_transition,
            )
            return density_curves
        else:
            return self._print.data.density_curves