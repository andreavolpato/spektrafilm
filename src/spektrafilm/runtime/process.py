from __future__ import annotations

from spektrafilm.runtime.factory import build_runtime_params
from spektrafilm.runtime.pipeline import SimulationPipeline
from spektrafilm.runtime.diagnostics import DiagnosticsPipeline
from spektrafilm.utils.timings import plot_timings


def photo_params(
    film_profile: str = "kodak_portra_400_auc",
    print_profile: str = "kodak_portra_endura_uc",
    ymc_filters_from_database: bool = True,
):
    return build_runtime_params(
        film_profile=film_profile,
        print_profile=print_profile,
        ymc_filters_from_database=ymc_filters_from_database,
    )


class AgXPhoto:
    """Compatibility class over the modular runtime pipeline."""

    def __init__(self, params, debug=True):
        #if debug:
        #    self._pipeline = DiagnosticsPipeline(params)
        #else:
        self._pipeline = SimulationPipeline(params)
        self.camera = self._pipeline.camera
        self.film = self._pipeline.film
        self.film_render = self._pipeline.film_render
        self.enlarger = self._pipeline.enlarger
        self.print = self._pipeline.print
        self.print_render = self._pipeline.print_render
        self.scanner = self._pipeline.scanner
        self.io = self._pipeline.io
        self.debug = self._pipeline.debug
        self.settings = self._pipeline.settings
        self.timings = self._pipeline.timings

    def process(self, image):
        return self._pipeline.process(image)


def photo_process(image, params):
    photo = AgXPhoto(params)
    image_out = photo.process(image)
    if photo.debug.print_timings:
        print(photo.timings)
        plot_timings(photo.timings)
    return image_out
