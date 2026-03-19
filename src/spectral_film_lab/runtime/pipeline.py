from __future__ import annotations

import copy

import numpy as np

from spectral_film_lab.runtime.params_schema import coerce_runtime_params
from spectral_film_lab.runtime.services import (
    EnlargerIlluminant,
    FilmDensityNormalizer,
    PrintDensityNormalizer,
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

        self._lut_cache = SpectralLUTCache(self.settings.lut_resolution, self.debug.luts)
        self._film_density_normalizer = FilmDensityNormalizer(self.source, self.source_render.grain.density_min)
        self._print_density_normalizer = PrintDensityNormalizer(self.print)
        self._illuminant_service = EnlargerIlluminant(self.enlarger)
        self._resizing_service = ResizingService(self.io, self.camera.film_format_mm)

        self._filming_stage = FilmingStage(
            self.source,
            self.source_render,
            self.camera,
            self.io,
            rgb_to_raw_method=self.settings.rgb_to_raw_method,
        )
        self._printing_stage = PrintingStage(
            self.source,
            self.print,
            self.source_render,
            self.print_render,
            self.enlarger,
            self._filming_stage,
            self._lut_cache,
            self._film_density_normalizer,
            self._illuminant_service,
            camera_exposure_compensation_ev=self.camera.exposure_compensation_ev,
            use_enlarger_lut=self.settings.use_enlarger_lut,
        )
        self._scanning_stage = ScanningStage(
            self.source,
            self.print,
            self.source_render,
            self.print_render,
            self.scanner,
            self.io,
            self._lut_cache,
            self._film_density_normalizer,
            self._print_density_normalizer,
            use_scanner_lut=self.settings.use_scanner_lut,
        )
        self._filming_stage.timings = self.timings
        self._printing_stage.timings = self.timings
        self._scanning_stage.timings = self.timings

    def process(self, image):
        image = np.double(np.array(image)[:, :, 0:3])
        image, preview_resize_factor, pixel_size_um = self._resizing_service.crop_and_rescale(image)
        
        exposure_ev = self._filming_stage.auto_exposure(image)

        if not self.io.full_image:
            self.source_render.grain.active = False
            self.source_render.halation.active = False

        raw_film = self._filming_stage.expose(image, exposure_ev, pixel_size_um)
        if self.io.compute_film_raw:
            return raw_film

        log_raw_film = np.log10(np.fmax(raw_film, 0.0) + 1e-10)
        density_channels = self._filming_stage.develop(log_raw_film, pixel_size_um, self.settings.use_fast_stats)
        if self.debug.return_source_density_cmy:
            return density_channels

        self._printing_stage.apply_profiles_changes()
        
        if not self.io.compute_source:
            lof_raw_print = self._printing_stage.expose(density_channels)
            density_channels = self._printing_stage.develop(lof_raw_print)
            if self.debug.return_print_density_cmy:
                return density_channels

        scan = self._scanning_stage.scan(density_channels)
        return self._resizing_service.rescale_to_original(scan, preview_resize_factor)

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

