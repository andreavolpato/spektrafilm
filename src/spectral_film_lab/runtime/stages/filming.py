from __future__ import annotations

import numpy as np

from spectral_film_lab.model.color_filters import compute_band_pass_filter
from spectral_film_lab.model.diffusion import apply_gaussian_blur_um, apply_halation_um
from spectral_film_lab.model.emulsion import Film
from spectral_film_lab.utils.autoexposure import measure_autoexposure_ev
from spectral_film_lab.utils.spectral_upsampling import rgb_to_raw_hanatos2025, rgb_to_raw_mallett2019
from spectral_film_lab.utils.timings import timeit


class FilmingStage:
    def __init__(self, source_profile, source_render_params, camera_params, io_params, rgb_to_raw_method):
        self._source = source_profile
        self._source_render = source_render_params
        self._camera = camera_params
        self._io = io_params
        self._rgb_to_raw_method = rgb_to_raw_method

    @timeit("_auto_exposure")
    def auto_exposure(self, image: np.ndarray) -> float:
        if self._camera.auto_exposure:
            autoexposure_ev = measure_autoexposure_ev(
                image,
                self._io.input_color_space,
                self._io.input_cctf_decoding,
                method=self._camera.auto_exposure_method,
            )
            return autoexposure_ev + self._camera.exposure_compensation_ev
        return self._camera.exposure_compensation_ev

    @timeit("_expose_film")
    def expose(self, image: np.ndarray, exposure_ev: float, pixel_size_um: float) -> np.ndarray:
        raw = self.rgb_to_film_raw(
            image,
            exposure_ev,
            color_space=self._io.input_color_space,
            apply_cctf_decoding=self._io.input_cctf_decoding,
        )
        raw = apply_gaussian_blur_um(raw, self._camera.lens_blur_um, pixel_size_um)
        raw = apply_halation_um(raw, self._source_render.halation, pixel_size_um)
        return raw

    @timeit("_develop_film")
    def develop(self, log_raw: np.ndarray, pixel_size_um: float, use_fast_stats: bool) -> np.ndarray:
        film = Film(self._source, self._source_render)
        return film.develop(log_raw, pixel_size_um, use_fast_stats=use_fast_stats)

    def rgb_to_film_raw(
        self,
        rgb: np.ndarray,
        exposure_ev: float,
        *,
        color_space: str = "sRGB",
        apply_cctf_decoding: bool = False,
    ) -> np.ndarray:
        sensitivity = 10 ** self._source.data.log_sensitivity
        sensitivity = np.nan_to_num(sensitivity)

        if self._camera.filter_uv[0] > 0 or self._camera.filter_ir[0] > 0:
            band_pass_filter = compute_band_pass_filter(self._camera.filter_uv, self._camera.filter_ir)
            sensitivity *= band_pass_filter[:, None]

        method = self._rgb_to_raw_method
        if method == "mallett2019":
            raw = rgb_to_raw_mallett2019(
                rgb,
                sensitivity,
                color_space=color_space,
                apply_cctf_decoding=apply_cctf_decoding,
                reference_illuminant=self._source.info.reference_illuminant,
            )
        elif method == "hanatos2025":
            raw = rgb_to_raw_hanatos2025(
                rgb,
                sensitivity,
                color_space=color_space,
                apply_cctf_decoding=apply_cctf_decoding,
                reference_illuminant=self._source.info.reference_illuminant,
            )
        else:
            raise ValueError(f"Unsupported rgb_to_raw_method: {method}")

        return raw * 2 ** exposure_ev
