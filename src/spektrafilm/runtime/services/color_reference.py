from __future__ import annotations

import numpy as np
from typing import Callable
from colour import RGB_to_RGB

from spektrafilm.model.emulsion import develop_simple

class ColorReferenceService:
    """Manages the color reference for black and white corrections."""
    def __init__(self, film_profile,  film_render,
                       print_profile, print_render,
                       black_correction, white_correction,
                       black_level, white_level,
                       io_params):
        self._film = film_profile
        self._film_render = film_render
        self._print = print_profile
        self._print_render = print_render
        self._scan_film = io_params.scan_film
        self._output_color_space = io_params.output_color_space
        self._output_cctf_encoding = io_params.output_cctf_encoding

        self._black_correction = black_correction
        self._white_correction = white_correction
        self._black_level = _remove_sRGB_cctf(black_level)
        self._white_level = _remove_sRGB_cctf(white_level)
        
        self.log_raw_print_black = None
        self.log_raw_print_white = None

    # public methods

    def black_white_xyz_correction(self, xyz,
                                   cmy_to_log_xyz: Callable):
        """Apply black and white correction to the XYZ values,
        based on the black and white reference densities
        cmy_to_log_xyz() must be defined in the scanning stage and passed to convert cmy densities
        """
        if self._black_correction or self._white_correction:
            if self._scan_film and self._film.info.type == 'positive':
                cmy_black = self._cmy_film_black()
                cmy_white = self._cmy_film_white()
            elif not self._scan_film and self._print.info.type == 'negative':
                cmy_black = self._cmy_print_black()
                cmy_white = self._cmy_print_white()
            # do not correct for scan_film and negative film
            else: raise ValueError("Unsupported film/print type for black and white correction.")
            log_xyz_black = cmy_to_log_xyz(cmy_black)
            log_xyz_white = cmy_to_log_xyz(cmy_white)
            y_black = (10 ** log_xyz_black)[:, :, 1]
            y_white = (10 ** log_xyz_white)[:, :, 1]
            return _black_and_white_linear_scaling(xyz, y_black=y_black, y_white=y_white,
                                                   black_correction=self._black_correction,
                                                   white_correction=self._white_correction,
                                                   black_level=self._black_level,
                                                   white_level=self._white_level)
        else:
            return xyz

    # private methods
    
    def _cmy_print_black(self):
        """CMY density of the black reference for print, for black correction in scanning"""
        return develop_simple(
            self.log_raw_print_black,
            self._print.data.log_exposure,
            self._print.data.density_curves,
            gamma_factor=self._print_render.density_curve_gamma,
        )

    def _cmy_print_white(self):
        """CMY density of the white reference for print, for white correction in scanning"""
        return develop_simple(
            self.log_raw_print_white,
            self._print.data.log_exposure,
            self._print.data.density_curves,
            gamma_factor=self._print_render.density_curve_gamma,
        )
    
    def _cmy_film_black(self):
        if self._film.is_positive:
            return  np.nanmax(self._film.data.density_curves, axis=0)[None, None, :]
        else:
            return np.zeros((1, 1, 3)) 
    
    def _cmy_film_white(self):
        if self._film.is_positive:
            return np.zeros((1, 1, 3))
        else:
            return np.nanmax(self._film.data.density_curves, axis=0)[None, None, :]
        

# private functions

def _black_and_white_linear_scaling(xyz, *,
                                    y_black, y_white,
                                    black_correction, white_correction, 
                                    black_level, white_level):        
    if black_correction and not white_correction:
        white_level = y_white
    if white_correction and not black_correction:
        black_level = y_black
    if black_correction or white_correction:                           
        m = (white_level - black_level) / (y_white - y_black + 1e-10)
        q = black_level - m * y_black
        def _correction_func(y):
            return np.clip(m * y + q, 0, 1)
        y = xyz[:, :, 1]
        y_corrected = _correction_func(y)
        scale = y_corrected / (y + 1e-10)
        return xyz * scale[:, :, None]
    if not black_correction and not white_correction:
        return xyz
    
def _remove_sRGB_cctf(y_input):
    return RGB_to_RGB(y_input*np.ones((1,1,3)),
                    'sRGB',
                    'sRGB',
                    apply_cctf_decoding=True,
                    apply_cctf_encoding=False,
                ).mean()