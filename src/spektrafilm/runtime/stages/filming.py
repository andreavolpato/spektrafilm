from __future__ import annotations

import numpy as np

from spektrafilm.model.color_filters import color_filter_transmittance
from spektrafilm.model.diffusion import apply_diffusion_filter_um, apply_gaussian_blur_um, apply_halation_um, boost_highlights
from spektrafilm.model.develop import compute_density_spectral, develop, develop_simple
from spektrafilm.utils.autoexposure import measure_raw_autoexposure_ev
from spektrafilm.utils.spectral_upsampling import rgb_to_raw_hanatos2025, rgb_to_raw_mallett2019


class FilmingStage:
    def __init__(self, film, film_render_params, camera_params, io_params, settings_params,
                 lut_service, resize_service, enlarger_service, color_reference_service):
        self._film = film
        self._film_render = film_render_params
        self._camera = camera_params
        self._io = io_params
        self._settings = settings_params
        self._lut_service = lut_service
        self._resize_service = resize_service
        self._enlarger_service = enlarger_service
        
        # send info for hanatos2025 sensitivity adaptation to LUT service
        hanatos2025_adaptation = self._film.hanatos2025_adaptation()
        hanatos2025_adaptation.apply_window = self._settings.apply_hanatos2025_adaptation_window
        hanatos2025_adaptation.apply_surface = self._settings.apply_hanatos2025_adaptation_surface
        hanatos2025_adaptation.spectral_gaussian_blur = self._settings.spectral_gaussian_blur
        self._lut_service.set_hanatos2025_adaptation(hanatos2025_adaptation)
        # Input gamut compression is per-bundle config (lives on params.io
        # so the GUI and bundle bakes share the same code path). The
        # service caches the LUT and invalidates it when this spec changes.
        self._lut_service.set_input_gamut_compress(self._io.input_gamut_compress)
        self._enlarger_service.density_spectral_midgray, self._enlarger_service.density_spectral_midgray_comp = self._compute_density_spectral_midgray_to_balance_print()
        self._color_reference_service = color_reference_service

    # public methods

    def expose(self, image: np.ndarray) -> np.ndarray:
        raw = self._rgb_to_film_raw(
            image,
            color_space=self._io.input_color_space,
            apply_cctf_decoding=self._io.input_cctf_decoding,
        )
        raw = self._auto_exposure(raw)
        raw *= 2 ** self._camera.exposure_compensation_ev
        boost_highlights(raw, self._film_render.halation.boost_ev,
                         self._film_render.halation.boost_range,
                         self._film_render.halation.protect_ev, out=raw)
        raw = apply_diffusion_filter_um(
            raw,
            self._camera.diffusion_filter,
            pixel_size_um=self._resize_service.pixel_size_um,
        )
        raw = apply_gaussian_blur_um(raw, self._camera.lens_blur_um, self._resize_service.pixel_size_um)
        raw = apply_halation_um(raw, self._film_render.halation, self._resize_service.pixel_size_um)
        raw *= self._color_reference_service.black_white_filming_exposure_correction()
        log_raw = np.log10(np.fmax(raw, 0.0) + 1e-10)
        return log_raw

    def develop(self, log_raw: np.ndarray) -> np.ndarray:
        # The film chemistry morph (and BW development-time selection) is already
        # baked into the curves / sublayers upstream (pipeline init), so these are
        # the single chosen, morphed development — develop runs at unit gamma.
        return develop(
            log_raw,
            self._resize_service.pixel_size_um,
            self._film.data.log_exposure,
            self._film.data.density_curves,
            self._film.data.density_curves_layers,
            self._film_render.dir_couplers,
            self._film_render.grain,
            self._film.info.type,
            use_fast_stats=self._settings.use_fast_stats,
        )

    # private methods

    def _auto_exposure(self, raw: np.ndarray) -> np.ndarray:
        """Scale the film raw so the metered region lands at the raw midgray
        reference (raw == 1). Metering uses the full frame captured before
        cropping/rescaling, so the exposure does not shift with the crop; a
        no-op when auto-exposure is disabled."""
        if not self._camera.auto_exposure:
            return raw
        autoexposure_ev = measure_raw_autoexposure_ev(
            self._metering_raw(),
            method=self._camera.auto_exposure_method,
        )
        return raw * 2 ** autoexposure_ev

    def _metering_raw(self) -> np.ndarray:
        """Film raw used for autoexposure metering: the full-frame preview
        captured before the crop, run through the same film-sensitivity path as
        the exposed raw. Falls back to a downsampled preview only if no full
        frame was stashed (e.g. when injecting mid-pipeline)."""
        preview = self._resize_service.full_frame_preview
        if preview is None:
            raise RuntimeError('autoexposure metering preview is unavailable; '
                               'crop_and_rescale must run before filming')
        return self._rgb_to_film_raw(
            preview,
            color_space=self._io.input_color_space,
            apply_cctf_decoding=self._io.input_cctf_decoding,
        )

    def _rgb_to_film_raw(
        self,
        rgb: np.ndarray,
        *,
        color_space: str = "sRGB",
        apply_cctf_decoding: bool = False,
    ) -> np.ndarray:
        sensitivity = 10 ** self._film.data.log_sensitivity
        sensitivity = np.nan_to_num(sensitivity)

        # A camera taking filter (UV / haze / colored) sits in front of the lens
        # and multiplies the incoming light spectrum; fold its transmittance into
        # the spectral sensitivity. No renormalization -- the attenuation and
        # color cast are the intended effect.
        transmittance = color_filter_transmittance(self._camera.color_filter)
        if transmittance is not None:
            sensitivity = sensitivity * transmittance[:, None]

        if self._settings.rgb_to_raw_method == "hanatos2025":
            tc_lut = self._lut_service.get_filming_tc_lut(sensitivity)
            raw = rgb_to_raw_hanatos2025(
                rgb,
                sensitivity,
                color_space=color_space,
                apply_cctf_decoding=apply_cctf_decoding,
                reference_illuminant=self._film.info.reference_illuminant,
                tc_lut=tc_lut,
            )
        elif self._settings.rgb_to_raw_method == "mallett2019":
            raw = rgb_to_raw_mallett2019(rgb, sensitivity,
                            color_space=color_space,
                            apply_cctf_decoding=apply_cctf_decoding,
                            reference_illuminant=self._film.info.reference_illuminant)
        else:
            raise ValueError(f"Unsupported rgb_to_raw method: {self._settings.rgb_to_raw_method}")
        return raw
    
    def _compute_density_spectral_midgray_to_balance_print(self):
        rgb_midgray = np.array([[[0.184] * 3]])
        raw_midgray = self._rgb_to_film_raw(rgb_midgray)
        # Match the real exposure path: autoexposure normalizes midgray to raw==1,
        # so a camera color filter's attenuation is compensated here too. Without
        # this the print-balance reference keeps the filter-attenuated exposure
        # while the actual negative is autoexposed, shifting print exposure. The
        # same midgray-derived scale is applied to the exposure-compensated
        # reference so their relationship is preserved.
        autoexposure_scale = self._midgray_autoexposure_scale(raw_midgray)
        density_spectral_midgray = self._raw_to_density_spectral(raw_midgray * autoexposure_scale)
        if self._enlarger_service.print_exposure_compensation:
            neg_exp_comp_ev = self._camera.exposure_compensation_ev
            raw_midgray_comp = self._rgb_to_film_raw(rgb_midgray * 2 ** neg_exp_comp_ev)
            density_spectral_midgray_comp = self._raw_to_density_spectral(
                raw_midgray_comp * autoexposure_scale)
        else:
            density_spectral_midgray_comp = None
        return density_spectral_midgray, density_spectral_midgray_comp

    def _midgray_autoexposure_scale(self, raw_midgray: np.ndarray) -> float:
        """The exposure scale autoexposure would apply to land midgray at raw==1
        (1.0 when autoexposure is off). Mirrors `_auto_exposure` so the print-
        balance reference tracks the actual negative through a camera filter."""
        if not self._camera.auto_exposure:
            return 1.0
        ev = measure_raw_autoexposure_ev(raw_midgray, method=self._camera.auto_exposure_method)
        return 2 ** ev

    def _raw_to_density_spectral(self, raw: np.ndarray) -> np.ndarray:
        log_raw = np.log10(raw + 1e-10)
        density_cmy = develop_simple(
            log_raw,
            self._film.data.log_exposure,
            self._film.data.density_curves,
        )
        return compute_density_spectral(
            self._film.data.channel_density,
            density_cmy,
            base_density=self._film.data.base_density,
            base_density_params=self._film_render.base,
            is_film=True,
        )
    