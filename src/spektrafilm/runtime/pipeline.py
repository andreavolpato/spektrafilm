from __future__ import annotations

import copy
from time import perf_counter

import numpy as np

from spektrafilm.model.density_curves import refresh_density_curves_from_model
from spektrafilm.runtime.services import (
    EnlargerService,
    ResizingService,
    SpectralLUTService,
    ColorReferenceService,
)
from spektrafilm.runtime.stages import FilmingStage, PrintingStage, ScanningStage
from spektrafilm.runtime.topology import Node, Tap, run_topology
from spektrafilm.utils.timings import format_timings



class SimulationPipeline:
    """Thin runtime orchestrator that composes stage objects around a
    tap-based topology dispatcher.

    The pipeline declares its stages as a list of :class:`Node` objects via
    :meth:`_build_topology`. Calling :meth:`process` walks the topology in
    declared order, firing each node whose input taps are present in state,
    and returns the value at the requested ``collect`` tap.
    """

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
        self.taps = self._params.taps

        # The parametric density_curves_model is the source of truth. Sample it
        # onto each profile's log_exposure grid so every downstream consumer
        # (develop, grain, couplers, Dmax probes) reads model-derived curves.
        # Done before services/stages capture the profiles below.
        for profile in (self.film, self.print):
            if profile is not None:
                refresh_density_curves_from_model(profile.data, profile.info.type)

        self.timings = {}
        self._last_elapsed_time = None

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
            self._resize_service,
            self._enlarger_service,
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
            self._resize_service,
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

        self._topology: list[Node] = self._build_topology()

    def process(self, image, *, inject: str | None = None, collect: str | None = None):
        """Run the pipeline.

        Defaults run end-to-end (``rgb_in`` → ``rgb_out``). Pass
        ``inject`` / ``collect`` to enter or exit at a named tap.
        Call-site kwargs override the persistent :class:`TapsParams`
        defaults.
        """
        inject = inject or self.taps.inject or Tap.RGB_IN
        collect = collect or self.taps.collect or Tap.RGB_OUT

        self.timings.clear()
        start = perf_counter()
        try:
            return run_topology(
                self._topology, inject, collect, image,
                on_fire=self._record_node_timing,
            )
        finally:
            self._last_elapsed_time = perf_counter() - start

    def get_timings(self):
        return self.timings

    def get_total_elapsed_time(self):
        return self._last_elapsed_time

    def format_timings(self):
        return format_timings(
            self.get_timings(),
            total_elapsed_time=self.get_total_elapsed_time(),
        )

    def print_timings(self):
        print(self.format_timings())

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
        invalidates_print_balance_reference = False
        if exposure_compensation_ev is not None:
            self.camera.exposure_compensation_ev = exposure_compensation_ev
            invalidates_print_balance_reference = True
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
            invalidates_print_balance_reference = True
        if print_density_curves is not None:
            self.print.data.density_curves = print_density_curves
        if invalidates_print_balance_reference:
            (
                self._enlarger_service.density_spectral_midgray,
                self._enlarger_service.density_spectral_midgray_comp,
            ) = self._filming_stage._compute_density_spectral_midgray_to_balance_print()

    # private methods

    def _build_topology(self) -> list[Node]:
        f, p, s = self._filming_stage, self._printing_stage, self._scanning_stage
        common = [
            Node((Tap.RGB_IN,),     (Tap.RGB_PRE,),    self._preprocess, "preprocess"),
            Node((Tap.RGB_PRE,),    (Tap.LOG_E_FILM,), f.expose,         "filming.expose"),
            Node((Tap.LOG_E_FILM,), (Tap.CMY_FILM,),   f.develop,        "filming.develop"),
        ]
        if self.io.scan_film:
            return common + [
                Node((Tap.CMY_FILM,), (Tap.RGB_OUT,), s.scan, "scanning.scan_film"),
            ]
        return common + [
            Node((Tap.CMY_FILM,),    (Tap.LOG_E_PRINT,), p.expose,  "printing.expose"),
            Node((Tap.LOG_E_PRINT,), (Tap.CMY_PRINT,),   p.develop, "printing.develop"),
            Node((Tap.CMY_PRINT,),   (Tap.RGB_OUT,),     s.scan,    "scanning.scan_print"),
        ]

    def _preprocess(self, image):
        image = np.double(np.array(image)[:, :, 0:3])
        image = self._filming_stage.auto_exposure(image)
        image = self._resize_service.crop_and_rescale(image)
        return image

    def _record_node_timing(self, node: Node, elapsed: float) -> None:
        self.timings[node.label] = self.timings.get(node.label, 0.0) + elapsed
