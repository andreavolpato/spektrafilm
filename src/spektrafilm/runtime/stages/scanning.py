from __future__ import annotations

import colour
import numpy as np
from opt_einsum import contract

from spektrafilm.color_management import ColorEncoding, output_encoding_from_io
from spektrafilm.config import STANDARD_OBSERVER_CMFS
from spektrafilm.model.diffusion import apply_gaussian_blur, apply_unsharp_mask
from spektrafilm.model.emulsion import compute_density_spectral
from spektrafilm.model.glare import add_glare
from spektrafilm.model.illuminants import standard_illuminant
from spektrafilm.utils.timings import timeit
from spektrafilm.utils.conversions import density_to_light


class ScanningStage:
    def __init__(
        self,
        film,
        film_render_params,
        print_profile,
        print_render_params,
        scanner_params,
        io_params,
        settings_params,
        lut_service,
        color_reference_service,
    ):
        self._film = film
        self._film_render = film_render_params
        self._print = print_profile
        self._print_render = print_render_params
        self._scanner = scanner_params
        self._io = io_params
        self._settings = settings_params
        self._lut_service = lut_service
        self._color_reference_service = color_reference_service
        
        self.cmy_to_log_xyz = self._return_callable_cmy_to_log_xyz()
        
        # communicate to the color reference service the callable to convert cmy densities to log xyz
        self._color_reference_service.cmy_to_log_xyz = self.cmy_to_log_xyz
        
    # public methods

    @timeit("scan")
    def scan(self, density_channels: np.ndarray, output_encoding: ColorEncoding | None = None) -> np.ndarray:
        if output_encoding is None:
            output_encoding = output_encoding_from_io(self._io)
        rgb = self._density_to_rgb(density_channels, use_lut=self._settings.use_scanner_lut)
        rgb = self._apply_blur_and_unsharp(rgb)
        return self._apply_cctf_encoding_and_clip(rgb, output_encoding)

    # private methods

    def _density_to_rgb(self, density_channels: np.ndarray, *, use_lut: bool) -> np.ndarray:
        if self._io.scan_film:
            glare = None
            density_min = -np.array(self._film_render.grain.density_min)
            density_max = np.nanmax(self._film.data.density_curves, axis=0)
            scan_illuminant = standard_illuminant(self._film.info.viewing_illuminant)
        else:
            glare = self._print_render.glare
            density_min = np.nanmin(self._print.data.density_curves, axis=0)
            density_max = np.nanmax(self._print.data.density_curves, axis=0)
            scan_illuminant = standard_illuminant(self._print.info.viewing_illuminant)
            
        normalization = np.sum(scan_illuminant * STANDARD_OBSERVER_CMFS[:, 1], axis=0)

        log_xyz = self._lut_service.spectral_compute_scanner(
            density_channels,
            spectral_calculation=self.cmy_to_log_xyz,
            data_min=density_min,
            data_max=density_max,
            use_lut=use_lut,
        )
        xyz = 10 ** log_xyz
        xyz = self._color_reference_service.black_white_xyz_correction(xyz)
        illuminant_xyz = contract("k,kl->l", scan_illuminant, STANDARD_OBSERVER_CMFS[:]) / normalization
        illuminant_xy = colour.XYZ_to_xy(illuminant_xyz)
        xyz = add_glare(xyz, illuminant_xyz, glare)
        return colour.XYZ_to_RGB(
            xyz,
            colourspace=self._io.output_color_space,
            apply_cctf_encoding=False,
            illuminant=illuminant_xy,
        )

    def _return_callable_cmy_to_log_xyz(self):
        if self._io.scan_film:
            channel_density = self._film.data.channel_density
            base_density = self._film.data.base_density
            scan_illuminant = standard_illuminant(self._film.info.viewing_illuminant)
        else:
            channel_density = self._print.data.channel_density
            base_density = self._print.data.base_density
            scan_illuminant = standard_illuminant(self._print.info.viewing_illuminant)
            
        normalization = np.sum(scan_illuminant * STANDARD_OBSERVER_CMFS[:, 1], axis=0)

        def cmy_to_log_xyz(density_cmy: np.ndarray) -> np.ndarray:
            density_spectral = compute_density_spectral(
                channel_density,
                density_cmy,
                base_density,
            )
            light = density_to_light(density_spectral, scan_illuminant)
            xyz = contract("ijk,kl->ijl", light, STANDARD_OBSERVER_CMFS[:]) / normalization
            return np.log10(np.fmax(xyz, 0.0) + 1e-10)
        return cmy_to_log_xyz

    def _apply_blur_and_unsharp(self, rgb: np.ndarray) -> np.ndarray:
        rgb = apply_gaussian_blur(rgb, self._scanner.lens_blur)
        sigma, amount = self._scanner.unsharp_mask
        if sigma > 0 and amount > 0:
            rgb = apply_unsharp_mask(rgb, sigma=sigma, amount=amount)
        return rgb

    def _apply_cctf_encoding_and_clip(self, rgb: np.ndarray, encoding: ColorEncoding) -> np.ndarray:
        if encoding.is_cctf_encoded:
            rgb = colour.RGB_to_RGB(
                rgb,
                encoding.color_space,
                encoding.color_space,
                apply_cctf_decoding=False,
                apply_cctf_encoding=True,
            )
        if encoding.clip_negatives:
            rgb = np.maximum(rgb, 0.0)
        if encoding.clip_highlights:
            rgb = np.minimum(rgb, 1.0)
        return rgb


