from __future__ import annotations

from spektrafilm.runtime.factory import build_runtime_params
from spektrafilm.runtime.params_schema import RuntimePhotoParams
from spektrafilm.runtime.pipeline import SimulationPipeline


def create_params(
    film_profile: str = "kodak_portra_400_auc",
    print_profile: str = "kodak_portra_endura_uc",
    ymc_filters_from_database: bool = True,
) -> RuntimePhotoParams:
    """Build a ready-to-use runtime parameter object."""
    return build_runtime_params(
        film_profile=film_profile,
        print_profile=print_profile,
        ymc_filters_from_database=ymc_filters_from_database,
    )


class Simulator:
    """User-facing wrapper around the runtime simulation pipeline."""

    def __init__(self, params: RuntimePhotoParams):
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

    def simulate(self, image):
        return self._pipeline.process(image)

    def process(self, image):
        return self.simulate(image)


def simulate(image, params: RuntimePhotoParams):
    simulator = Simulator(params)
    return simulator.simulate(image)

################################################################################
# Legacy for compatibility with agx-emulsion
################################################################################

def photo_params(
    film_profile: str = "kodak_portra_400_auc",
    print_profile: str = "kodak_portra_endura_uc",
    ymc_filters_from_database: bool = True,
) -> RuntimePhotoParams:
    """Legacy alias for create_params()."""
    return create_params(
        film_profile=film_profile,
        print_profile=print_profile,
        ymc_filters_from_database=ymc_filters_from_database,
    )

class AgXPhoto(Simulator):
    """Legacy alias for Simulator."""

    def __init__(self, params: RuntimePhotoParams, debug: bool = False):
        del debug
        super().__init__(params)


def photo_process(image, params: RuntimePhotoParams):
    """Legacy alias for simulate()."""
    return simulate(image, params)


__all__ = [
    "AgXPhoto",
    "RuntimePhotoParams",
    "Simulator",
    "create_params",
    "photo_params",
    "photo_process",
    "simulate",
]