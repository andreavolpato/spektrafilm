from __future__ import annotations

import numpy as np

from spektrafilm.model.emulsion import develop_simple

class ColorReferenceService:
    def __init__(self, film_data,  film_render,
                       print_data, print_render, scan_film):
        self._film_data = film_data
        self._film_render = film_render
        self._print_data = print_data
        self._print_render = print_render
        self.scan_film = scan_film
        
        self.density_spectral_black_film = self._film_data.base_density
        self.density_spectral_white_film = self._compute_film_white_spectral_density()
        
        self.log_raw_print_black = None
        self.log_raw_print_white = None

    def _compute_film_white_spectral_density(self):
        max_density = np.nanmax(self._film_data.density_curves, axis=0)
        return max_density*self._film_data.channel_density + self._film_data.base_density[:,None]
    
    def cmy_print_black(self):
        """CMY density of the black reference for print, for black correction in scanning"""
        return develop_simple(
            self.log_raw_print_black,
            self._print_data.log_exposure,
            self._print_data.density_curves,
            gamma_factor=self._print_render.density_curve_gamma,
        )

    def cmy_print_white(self):
        """CMY density of the white reference for print, for white correction in scanning"""
        return develop_simple(
            self.log_raw_print_white,
            self._print_data.log_exposure,
            self._print_data.density_curves,
            gamma_factor=self._print_render.density_curve_gamma,
        )