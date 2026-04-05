from __future__ import annotations

import numpy as np

class ColorReferenceService:
    def __init__(self, film_data, print_data, scan_film):
        self._film_data = film_data
        self._print_data = print_data
        self.scan_film = scan_film
        
        self.density_spectral_black_film = self._film_data.base_density
        self.density_spectral_white_film = self._compute_film_white_spectral_density()
        
        self.log_raw_print_black = None
        self.log_raw_print_white = None
        self.cmy_print_black = None
        self.cmy_print_white = None
        

    def _compute_film_white_spectral_density(self):
        max_density = np.nanmax(self._film_data.density_curves, axis=0)
        return max_density*self._film_data.channel_density + self._film_data.base_density[:,None]
        