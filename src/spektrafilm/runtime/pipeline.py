from __future__ import annotations

import copy

import numpy as np

from spektrafilm.runtime.services import (
    EnlargerService,
    ResizingService,
    SpectralLUTService,
)
from spektrafilm.runtime.stages import FilmingStage, PrintingStage, ScanningStage


class SimulationPipeline:
    """Thin runtime orchestrator that composes stage objects."""

    def __init__(self, params):
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

        self._apply_debug_switches()

        self._lut_service = SpectralLUTService(self.settings.lut_resolution)
        self._enlarger_service = EnlargerService(self.enlarger)
        self._resizing_service = ResizingService(self.io, self.camera.film_format_mm)

        self._filming_stage = FilmingStage(
            self.film,
            self.film_render,
            self.camera,
            self.io,
            self.settings,
            self._resizing_service, # pixel size for grain, halation, and lens blur calculations
            self._enlarger_service, # to compute and save density spectral midgray to balance print
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
        )
        self._filming_stage.timings = self.timings
        self._printing_stage.timings = self.timings
        self._scanning_stage.timings = self.timings

    def process(self, image):
        image = self._preprocess(image)
        image = self._pipeline(image)
        image = self._postprocess(image)
        return image

    def _preprocess(self, image):
        image = np.double(np.array(image)[:, :, 0:3])
        image = self._filming_stage.auto_exposure(image) # autoexposure service?
        image = self._resizing_service.crop_and_rescale(image)
        return image

    def _pipeline(self, rgb_image):
        if self.io.scan_film: # replace with route switch
            log_raw_film = self._filming_stage.expose(rgb_image)
            cmy_film = self._filming_stage.develop(log_raw_film)
            rgb_scan = self._scanning_stage.scan(cmy_film)
        else:
            log_raw_film = self._filming_stage.expose(rgb_image)
            cmy_film = self._filming_stage.develop(log_raw_film)
            log_raw_print = self._printing_stage.expose(cmy_film)
            cmy_print = self._printing_stage.develop(log_raw_print)
            rgb_scan = self._scanning_stage.scan(cmy_print)
        
        # debugging outputs
        if self.debug.return_film_log_raw: return log_raw_film
        if self.debug.return_film_density_cmy: return cmy_film
        if self.debug.return_print_density_cmy: return cmy_print
        return rgb_scan
    
    def _postprocess(self, image):
        return self._resizing_service.rescale_to_original(image)

    def _apply_debug_switches(self):
        if self.debug.deactivate_spatial_effects:
            self.film_render.halation.size_um = [0, 0, 0]
            self.film_render.halation.scattering_size_um = [0, 0, 0]
            self.film_render.dir_couplers.diffusion_size_um = 0
            self.film_render.grain.blur = 0.0
            self.film_render.grain.blur_dye_clouds_um = 0.0
            self.print_render.glare.blur = 0
            self.camera.lens_blur_um = 0.0
            self.enlarger.lens_blur = 0.0
            self.scanner.lens_blur = 0.0
            self.scanner.unsharp_mask = (0.0, 0.0)

        if self.debug.deactivate_stochastic_effects:
            self.film_render.grain.active = False
            self.print_render.glare.active = False

