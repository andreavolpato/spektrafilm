from __future__ import annotations

import copy

from PIL.ImageQt import rgb
import numpy as np
from skimage import exposure

from spektrafilm.runtime.services import (
    EnlargerService,
    ResizingService,
    SpectralLUTService,
    ColorReferenceService,
)
from spektrafilm.runtime.stages import FilmingStage, PrintingStage, ScanningStage



class SimulationPipeline:
    """Thin runtime orchestrator that composes stage objects."""

    def __init__(self, params, update_params=False):
        self._params = copy.deepcopy(params)

        self.camera = self._params.camera
        self.film = self._params.film
        self.film_render = self._params.film_render
        self.enlarger = self._params.enlarger
        self.print = self._params.print
        self.print_render = self._params.print_render
        self.scanner = self._params.scanner
        self.io = self._params.io
        self.debug = self._params.debug
        self.settings = self._params.settings

        self.timings = {}

        self._resize_service = ResizingService(self.io, self.camera.film_format_mm)
        if not update_params:
            self._lut_service = SpectralLUTService(self.settings.lut_resolution)
        self._enlarger_service = EnlargerService(self.enlarger)
        self._color_reference_service = ColorReferenceService(self.film, self.film_render,
                                                              self.print, self.print_render,
                                                              self.scanner.black_correction, self.scanner.white_correction,
                                                              self.scanner.black_level, self.scanner.white_level,
                                                              self.io)

        
        self._filming_stage = FilmingStage(
            self.film,
            self.film_render,
            self.camera,
            self.io,
            self.settings,
            self._lut_service,
            self._resize_service, # to get pixel size um for blurs
            self._enlarger_service, # to compute and save density spectral midgray to balance print
            self._color_reference_service,
        )
        self._printing_stage = PrintingStage(
            self.film,
            self.film_render,
            self.print,
            self.print_render,
            self.enlarger,
            self.settings,
            self._lut_service,
            self._enlarger_service,
            self._resize_service, # to get pixel size um for diffusion filter
            self._color_reference_service,
        )
        self._scanning_stage = ScanningStage(
            self.film,
            self.film_render,
            self.print,
            self.print_render,
            self.scanner,
            self.io,
            self.settings,
            self._lut_service,
            self._color_reference_service,
        )
        
        # timing communication
        self._filming_stage.timings = self.timings
        self._printing_stage.timings = self.timings
        self._scanning_stage.timings = self.timings
        self._lut_service.timings = self.timings

    def process(self, image):
        """Process an image through the simulation pipeline."""
        if self.debug.debug_mode == 'off':
            image = self._pipeline(image)
        else:
            image = self._pipeline_debug(image)
        return image
    
    def update(self, params):
        """Update params and re-initialize stages that depend on them."""
        self.__init__(params, update_params=True)
        
    def soft_update(self,
                    exposure_compensation_ev=None,
                    print_exposure=None,
                    c_filter_neutral=None,
                    m_filter_neutral=None,
                    y_filter_neutral=None,
                    film_density_curves=None,
                    print_density_curves=None,):
        if exposure_compensation_ev is not None:
            self.camera.exposure_compensation_ev = exposure_compensation_ev
        if print_exposure is not None:
            self.enlarger.print_exposure = print_exposure
        if c_filter_neutral is not None:
            self.enlarger.c_filter_neutral = c_filter_neutral
        if m_filter_neutral is not None:
            self.enlarger.m_filter_neutral = m_filter_neutral
        if y_filter_neutral is not None:
            self.enlarger.y_filter_neutral = y_filter_neutral
        if film_density_curves is not None:
            self.film.data.density_curves = film_density_curves
        if print_density_curves is not None:
            self.print.data.density_curves = print_density_curves
        # if argument is not in the defined list return an error
        for key in locals().keys():
            if key not in ['exposure_compensation_ev',
                           'print_exposure',
                           'c_filter_neutral',
                           'm_filter_neutral',
                           'y_filter_neutral',
                           'film_density_curves',
                           'print_density_curves']:
                raise ValueError(f"Unsupported parameter for soft update: {key}")    
        
    # private methods
    
    def _pipeline(self, image):
        image = self._preprocess(image)
        if self.io.scan_film: # replace with route switch
            rgb_scan = self._pipeline_scan_film(image)
        else:
            rgb_scan = self._pipeline_print(image)
        return rgb_scan
    
    def _preprocess(self, image):
        image = np.double(np.array(image)[:, :, 0:3])
        image = self._filming_stage.auto_exposure(image) # autoexposure service?
        image = self._resize_service.crop_and_rescale(image)
        return image
    
    def _pipeline_scan_film(self, rgb_image):
        log_raw_film = self._filming_stage.expose(rgb_image)
        cmy_film = self._filming_stage.develop(log_raw_film)
        rgb_scan = self._scanning_stage.scan(cmy_film)
        return rgb_scan
    
    def _pipeline_print(self, rgb_image):
        log_raw_film = self._filming_stage.expose(rgb_image)
        cmy_film = self._filming_stage.develop(log_raw_film)
        log_raw_print = self._printing_stage.expose(cmy_film)
        cmy_print = self._printing_stage.develop(log_raw_print)
        rgb_scan = self._scanning_stage.scan(cmy_print)
        return rgb_scan
    
################################################################################
    
    # debug_methods

    def _pipeline_debug(self, rgb_image):
        if self.debug.debug_mode == "output":
            return self._debug_output_pipeline(rgb_image)
        elif self.debug.debug_mode == "inject":
            return self._debug_inject_pipeline(rgb_image)
    
    def _debug_output_pipeline(self, rgb_image):
        """Run the pipeline with additional outputs for debugging."""
        rgb_image = self._preprocess(rgb_image)
        log_raw_film = self._filming_stage.expose(rgb_image)
        if self.debug.output_film_log_raw:
            return log_raw_film
        
        cmy_film = self._filming_stage.develop(log_raw_film)
        if self.debug.output_film_density_cmy:
            return cmy_film
        
        log_raw_print = self._printing_stage.expose(cmy_film)
        cmy_print = self._printing_stage.develop(log_raw_print)
        if self.debug.output_print_density_cmy:
            return cmy_print
        
        rgb_scan = self._scanning_stage.scan(cmy_print)
        return rgb_scan
    
    def _debug_inject_pipeline(self, cmy_film):
        """Run the pipeline with additional inputs for debugging."""
        if self.debug.inject_film_density_cmy:
            log_raw_print = self._printing_stage.expose(cmy_film)
            cmy_print = self._printing_stage.develop(log_raw_print)
            rgb_scan = self._scanning_stage.scan(cmy_print)
            return rgb_scan

