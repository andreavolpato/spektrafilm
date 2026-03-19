from __future__ import annotations

import numpy as np
from opt_einsum import contract

from spectral_film_lab.model.emulsion import compute_density_spectral, develop_simple
from spectral_film_lab.model.density_curves import remove_viewing_glare_comp
from spectral_film_lab.model.illuminants import standard_illuminant
from spectral_film_lab.utils.timings import timeit
from spectral_film_lab.utils.conversions import density_to_light


class PrintingStage:
    def __init__(
        self,
        source_profile,
        print_profile,
        source_render_params,
        print_render_params,
        enlarger_params,
        filming_stage,
        lut_cache,
        film_density_normalizer,
        illuminant_service,
        *,
        camera_exposure_compensation_ev: float,
        use_enlarger_lut: bool,
    ):
        self._source = source_profile
        self._print = print_profile
        self._source_render = source_render_params
        self._print_render = print_render_params
        self._camera_exposure_compensation_ev = camera_exposure_compensation_ev
        self._enlarger = enlarger_params
        self._filming_stage = filming_stage
        self._lut_cache = lut_cache
        self._film_density_normalizer = film_density_normalizer
        self._illuminant_service = illuminant_service
        self._use_enlarger_lut = use_enlarger_lut

    @timeit("_apply_profiles_changes")
    def apply_profiles_changes(self):
        if self._print_render.glare.compensation_removal_factor > 0:
            log_exposure = self._print.data.log_exposure
            density_curves = self._print.data.density_curves
            self._print.data.density_curves = remove_viewing_glare_comp(
                log_exposure,
                density_curves,
                factor=self._print_render.glare.compensation_removal_factor,
                density=self._print_render.glare.compensation_removal_density,
                transition=self._print_render.glare.compensation_removal_transition,
            )

    @timeit("_expose_print")
    def expose(self, film_density_channels: np.ndarray) -> np.ndarray:
        film_density_normalized = self._film_density_normalizer.normalize(film_density_channels)

        def spectral_calculation(density_channels_normalized):
            density_channels = self._film_density_normalizer.denormalize(density_channels_normalized)
            return self.film_density_to_print_log_raw(density_channels)

        return self._lut_cache.compute(
            film_density_normalized,
            spectral_calculation,
            use_lut=self._use_enlarger_lut,
            save_enlarger_lut=True,
        )

    @timeit("_develop_print")
    def develop(self, log_raw: np.ndarray) -> np.ndarray:
        return develop_simple(
            self._print,
            log_raw,
            gamma_factor=self._print_render.density_curve_gamma,
        )

    def film_density_to_print_log_raw(self, density_channels: np.ndarray) -> np.ndarray:
        sensitivity = 10 ** self._print.data.log_sensitivity
        sensitivity = np.nan_to_num(sensitivity)
        enlarger_light_source = standard_illuminant(self._enlarger.illuminant)

        raw = np.zeros_like(density_channels)
        if not self._enlarger.just_preflash:
            density_spectral = compute_density_spectral(
                self._source,
                density_channels,
                base_density_scale=self._source_render.base_density_scale,
            )
            print_illuminant = self._illuminant_service.print_illuminant(enlarger_light_source)
            light = density_to_light(density_spectral, print_illuminant)
            raw = contract("ijk, kl->ijl", light, sensitivity)
            raw *= self._enlarger.print_exposure
            raw *= self.compute_exposure_factor_midgray(sensitivity, print_illuminant)

        raw_preflash = self.compute_raw_preflash(enlarger_light_source, sensitivity)
        raw += raw_preflash
        return np.log10(raw + 1e-10)

    def compute_raw_preflash(self, light_source, sensitivity):
        if self._enlarger.preflash_exposure > 0:
            preflash_illuminant = self._illuminant_service.preflash_illuminant(light_source)
            density_base = np.asarray(self._source.data.base_density)[None, None, :]
            light_preflash = density_to_light(density_base, preflash_illuminant)
            raw_preflash = contract("ijk, kl->ijl", light_preflash, sensitivity)
            return raw_preflash * self._enlarger.preflash_exposure
        return np.zeros((3,))

    def compute_exposure_factor_midgray(self, sensitivity, print_illuminant):
        neg_exp_comp_ev = self._camera_exposure_compensation_ev if self._enlarger.print_exposure_compensation else 0.0
        rgb_midgray = np.array([[[0.184] * 3]]) * 2 ** neg_exp_comp_ev
        raw_midgray = self._filming_stage.rgb_to_film_raw(rgb_midgray, exposure_ev=0.0)
        log_raw_midgray = np.log10(raw_midgray + 1e-10)
        density_midgray = develop_simple(
            self._source,
            log_raw_midgray,
            gamma_factor=self._source_render.density_curve_gamma,
        )
        density_spectral_midgray = compute_density_spectral(
            self._source,
            density_midgray,
            base_density_scale=self._source_render.base_density_scale,
        )
        light_midgray = density_to_light(density_spectral_midgray, print_illuminant)
        raw_midgray = contract("ijk, kl->ijl", light_midgray, sensitivity)
        return 1 / raw_midgray[:, :, 1]
