from __future__ import annotations

import copy

import numpy as np

from spectral_film_lab.runtime.params_schema import coerce_runtime_params
from spectral_film_lab.runtime.services import (
    EnlargerService,
    ResizingService,
    SpectralLUTCache,
)
from spectral_film_lab.runtime.stages import FilmingStage, PrintingStage, ScanningStage


class RuntimePipeline:
    """Thin runtime orchestrator that composes stage objects."""

    def __init__(self, params):
        params = coerce_runtime_params(params)
        self._params = copy.deepcopy(params)

        self.camera = self._params.camera
        self.source = self._params.source
        self.source_render = self._params.source_render
        self.enlarger = self._params.enlarger
        self.print = self._params.print
        self.print_render = self._params.print_render
        self.scanner = self._params.scanner
        self.io = self._params.io
        self.debug = self._params.debug
        self.settings = self._params.settings

        self.timings = {}

        self._apply_debug_switches()

        self._lut_service = SpectralLUTCache(self.settings.lut_resolution, self.debug.luts)
        self._enlarger_service = EnlargerService(self.enlarger)
        self._resizing_service = ResizingService(self.io, self.camera.film_format_mm)

        self._filming_stage = FilmingStage(
            self.source,
            self.source_render,
            self.camera,
            self.io,
            self.settings,
            self._resizing_service, # pixel size for grain, halation, and lens blur calculations
        )
        self._printing_stage = PrintingStage(
            self.source,
            self.print,
            self.source_render,
            self.print_render,
            self.enlarger,
            self.settings,
            self._filming_stage, # to be removed
            self._lut_service,
            self._enlarger_service,
            camera_exposure_compensation_ev=self.camera.exposure_compensation_ev,
        )
        self._scanning_stage = ScanningStage(
            self.source,
            self.print,
            self.source_render,
            self.print_render,
            self.scanner,
            self.io,
            self._lut_service,
            use_scanner_lut=self.settings.use_scanner_lut,
        )
        self._filming_stage.timings = self.timings
        self._printing_stage.timings = self.timings
        self._scanning_stage.timings = self.timings

    def process(self, rgb_image):
        if not self.io.full_image:
            self.source_render.grain.active = False
            self.source_render.halation.active = False
            
        rgb_image = np.double(np.array(rgb_image)[:, :, 0:3])
        rgb_image = self._filming_stage.auto_exposure(rgb_image)
        image = self._resizing_service.crop_and_rescale(rgb_image)
        
        log_raw_film = self._filming_stage.expose(image)
        
        if self.io.compute_film_raw:
            return 10**log_raw_film

        cmy_film = self._filming_stage.develop(log_raw_film)
        
        if self.debug.return_source_density_cmy:
            return cmy_film

        if self.io.compute_source:
            scan = self._scanning_stage.scan(cmy_film)
        else:
            lof_raw_print = self._printing_stage.expose(cmy_film)
            cmy_print = self._printing_stage.develop(lof_raw_print)
            if self.debug.return_print_density_cmy:
                return cmy_print
            scan = self._scanning_stage.scan(cmy_print)
        
        return self._resizing_service.rescale_to_original(scan)

    def _apply_debug_switches(self):
        if self.debug.deactivate_spatial_effects:
            self.source_render.halation.size_um = [0, 0, 0]
            self.source_render.halation.scattering_size_um = [0, 0, 0]
            self.source_render.dir_couplers.diffusion_size_um = 0
            self.source_render.grain.blur = 0.0
            self.source_render.grain.blur_dye_clouds_um = 0.0
            self.print_render.glare.blur = 0
            self.camera.lens_blur_um = 0.0
            self.enlarger.lens_blur = 0.0
            self.scanner.lens_blur = 0.0
            self.scanner.unsharp_mask = (0.0, 0.0)

        if self.debug.deactivate_stochastic_effects:
            self.source_render.grain.active = False
            self.print_render.glare.active = False

