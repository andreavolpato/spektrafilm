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
        
        # local memory for black and white reference densities, to avoid redundant calculations during correction
        self._y_black = None # positive film or print cmy density for black
        self._y_white = None # positive film or print cmy density for white
        self._black_white_exposure_correction = None # exposure correction factor to apply to the print exposure based on the black and white reference densities
        
        # communication with the scanning stage
        self.cmy_to_log_xyz = None # callable to convert cmy densities to log xyz, to be defined in the scanning stage during pipeline init!
        self.log_raw_print_black = None
        self.log_raw_print_white = None

    # public methods

    def _update_cmy_black_white_references(self, in_print=False):
        if not self._black_correction and not self._white_correction:
            return
        elif self._scan_film and self._film.info.type == 'negative':
            return
        else:
            if self._scan_film and self._film.info.type == 'positive' and not in_print:
                cmy_black = np.nanmax(self._film.data.density_curves, axis=0)[None, None, :]
                cmy_white = np.zeros((1, 1, 3))
                log_xyz_black = self.cmy_to_log_xyz(cmy_black)
                log_xyz_white = self.cmy_to_log_xyz(cmy_white)
                self._y_black = (10 ** log_xyz_black)[:, :, 1]
                self._y_white = (10 ** log_xyz_white)[:, :, 1]
            elif not self._scan_film and self._print.info.type == 'negative' and in_print:
                cmy_black = develop_simple(
                    self.log_raw_print_black,
                    self._print.data.log_exposure,
                    self._print.data.density_curves,
                    gamma_factor=self._print_render.density_curve_gamma,
                )
                cmy_white = develop_simple(
                    self.log_raw_print_white,
                    self._print.data.log_exposure,
                    self._print.data.density_curves,
                    gamma_factor=self._print_render.density_curve_gamma,
                )
                log_xyz_black = self.cmy_to_log_xyz(cmy_black)
                log_xyz_white = self.cmy_to_log_xyz(cmy_white)
                self._y_black = (10 ** log_xyz_black)[:, :, 1]
                self._y_white = (10 ** log_xyz_white)[:, :, 1]
            return
    
    def black_white_filming_exposure_correction(self): # in filming and printing
        if not self._black_correction and not self._white_correction:
            return 1.0
        elif self._film.info.type == 'negative':
            return 1.0
        elif self._scan_film and self._film.info.type == 'positive':
            density_midgray = -np.log10(0.184)
            self._update_cmy_black_white_references(in_print=False)
            midgray_corrected = self._correction_fucntion()[1]
            density_midgray_corrected = -np.log10(midgray_corrected)
            density_curve_av = np.nanmean(self._film.data.density_curves, axis=1)
            density_min_av = np.nanmean(self._film.data.base_density)
            log_exposure = self._film.data.log_exposure
            log_exposure_midgray_corrected = -np.interp(-(density_midgray_corrected-density_min_av), 
                                                -density_curve_av, log_exposure)
            log_exposure_midgray = -np.interp(-(density_midgray-density_min_av),
                                                -density_curve_av, log_exposure)
            exposure_correction = 10 ** (log_exposure_midgray_corrected - log_exposure_midgray)
            return 1/exposure_correction
        else:
            return 1.0

    def black_white_printing_exposure_correction(self): # in printing
        if not self._black_correction and not self._white_correction:
            return 1.0
        elif self._print.info.type == 'negative':
            density_midgray = -np.log10(0.184)
            self._update_cmy_black_white_references(in_print=True)
            midgray_corrected = self._correction_fucntion()[1]
            density_midgray_corrected = -np.log10(midgray_corrected)
            density_curve_av = np.nanmean(self._print.data.density_curves, axis=1)
            density_min_av = np.nanmean(self._print.data.base_density)
            log_exposure = self._print.data.log_exposure
            log_exposure_midgray_corrected = np.interp(density_midgray_corrected-density_min_av, 
                                                density_curve_av, log_exposure)
            log_exposure_midgray = np.interp(density_midgray-density_min_av,
                                                density_curve_av, log_exposure)
            exposure_correction = 10 ** (log_exposure_midgray_corrected - log_exposure_midgray)
            return exposure_correction

    def black_white_xyz_correction(self, xyz): # in scanning
        """Apply black and white correction to the XYZ values,
        based on the black and white reference densities
        cmy_to_log_xyz() must be defined in the scanning stage and passed to convert cmy densities
        """
        if not self._black_correction and not self._white_correction:
            return xyz
        if self._scan_film and self._film.info.type == 'negative':
            return xyz # do not correct negative film scans
        else:
            correction_func, _ = self._correction_fucntion()
            y = xyz[:, :, 1]
            y_corrected = correction_func(y)
            scale = y_corrected / (y + 1e-10)
            return xyz * scale[:, :, None]


    def _correction_fucntion(self):
        white_level = self._white_level
        black_level = self._black_level
        if self._black_correction and not self._white_correction:
            white_level = self._y_white
        if self._white_correction and not self._black_correction:
            black_level = self._y_black
        if self._black_correction or self._white_correction:                           
            m = (white_level - black_level) / (self._y_white - self._y_black + 1e-10)
            q = black_level - m * self._y_black
            def correction_func(y):
                return np.clip(m * y + q, 0, 1)
        midgray_black_white_corrected = (0.184 - q)/m
        return correction_func, midgray_black_white_corrected

# private functions
    
def _remove_sRGB_cctf(y_input):
    return RGB_to_RGB(y_input*np.ones((1,1,3)),
                    'sRGB',
                    'sRGB',
                    apply_cctf_decoding=True,
                    apply_cctf_encoding=False,
                ).mean()
    
