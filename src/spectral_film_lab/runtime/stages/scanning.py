from __future__ import annotations

import colour
import numpy as np
from opt_einsum import contract

from spectral_film_lab.config import STANDARD_OBSERVER_CMFS
from spectral_film_lab.model.diffusion import apply_gaussian_blur, apply_unsharp_mask
from spectral_film_lab.model.emulsion import compute_density_spectral
from spectral_film_lab.model.glare import add_glare
from spectral_film_lab.model.illuminants import standard_illuminant
from spectral_film_lab.utils.timings import timeit
from spectral_film_lab.utils.conversions import density_to_light


class ScanningStage:
    def __init__(
        self,
        film,
        film_render_params,
        print,
        print_render_params,
        scanner_params,
        io_params,
        settings_params,
        lut_service,
    ):
        self._film = film
        self._film_render = film_render_params
        self._print = print
        self._print_render = print_render_params
        self._scanner = scanner_params
        self._io = io_params
        self._settings = settings_params
        self._lut_service = lut_service

    @timeit("_scan")
    def scan(self, density_channels: np.ndarray) -> np.ndarray:
        rgb = self.density_to_rgb(density_channels, use_lut=self._settings.use_scanner_lut)
        rgb = self.apply_blur_and_unsharp(rgb)
        return self.apply_cctf_encoding_and_clip(rgb)

    def density_to_rgb(self, density_channels: np.ndarray, *, use_lut: bool) -> np.ndarray:
        if self._io.scan_film:
            profile = self._film
            base_density_scale = self._film_render.base_density_scale
            glare = None
            density_min = -np.array(self._film_render.grain.density_min)
            density_max = np.nanmax(self._film.data.density_curves, axis=0)
        else:
            profile = self._print
            base_density_scale = self._print_render.base_density_scale
            glare = self._print_render.glare
            density_min = np.nanmin(self._print.data.density_curves, axis=0)
            density_max = np.nanmax(self._print.data.density_curves, axis=0)

        scan_illuminant = standard_illuminant(profile.info.viewing_illuminant)
        normalization = np.sum(scan_illuminant * STANDARD_OBSERVER_CMFS[:, 1], axis=0)

        def spectral_calculation(density_cmy: np.ndarray) -> np.ndarray:
            density_spectral = compute_density_spectral(profile, density_cmy, base_density_scale=base_density_scale)
            light = density_to_light(density_spectral, scan_illuminant)
            xyz = contract("ijk,kl->ijl", light, STANDARD_OBSERVER_CMFS[:]) / normalization
            return np.log10(np.fmax(xyz, 0.0) + 1e-10)

        log_xyz = self._lut_service.compute(
            density_channels,
            data_min=density_min,
            data_max=density_max,
            spectral_calculation=spectral_calculation,
            use_lut=use_lut,
            save_scanner_lut=True,
        )
        xyz = 10 ** log_xyz

        illuminant_xyz = contract("k,kl->l", scan_illuminant, STANDARD_OBSERVER_CMFS[:]) / normalization
        xyz = add_glare(xyz, illuminant_xyz, glare)
        illuminant_xy = colour.XYZ_to_xy(illuminant_xyz)
        return colour.XYZ_to_RGB(
            xyz,
            colourspace=self._io.output_color_space,
            apply_cctf_encoding=False,
            illuminant=illuminant_xy,
        )

    def apply_blur_and_unsharp(self, rgb: np.ndarray) -> np.ndarray:
        rgb = apply_gaussian_blur(rgb, self._scanner.lens_blur)
        sigma, amount = self._scanner.unsharp_mask
        if sigma > 0 and amount > 0:
            rgb = apply_unsharp_mask(rgb, sigma=sigma, amount=amount)
        return rgb

    def apply_cctf_encoding_and_clip(self, rgb: np.ndarray) -> np.ndarray:
        if self._io.output_cctf_encoding:
            rgb = colour.RGB_to_RGB(
                rgb,
                self._io.output_color_space,
                self._io.output_color_space,
                apply_cctf_decoding=False,
                apply_cctf_encoding=True,
            )
        return np.clip(rgb, a_min=0, a_max=1)
