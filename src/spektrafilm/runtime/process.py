"""Runtime process entry points."""

from __future__ import annotations

from spektrafilm.runtime.params_schema import RuntimePhotoParams
from spektrafilm.runtime.pipeline import SimulationPipeline
from spektrafilm.utils.preview import resize_for_preview
from spektrafilm.runtime.params_builder import digest_params, init_params

class Simulator:
    """User-facing wrapper around the runtime simulation pipeline.
    The params passed to the constructor should be static and not be changed.
    """

    def __init__(self, params: RuntimePhotoParams):
        self._params = params # should stay static
        self._pipeline = SimulationPipeline(params)
        self._sync_public_state_from_pipeline()

    def _sync_public_state_from_pipeline(self) -> None:
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
        """Process the input image through the simulation pipeline and return the final result."""
        return self._pipeline.process(image)
    
    def update_params(self, params):
        """Update the parameters of the simulation pipeline."""
        self._params = params
        self._pipeline.update_params(params)
        self._sync_public_state_from_pipeline()

######################################################################################
# Convenience functions for single-call simulation without needing to instantiate the Simulator class.

def simulate(image, params: RuntimePhotoParams,
             digest_params_first: bool = True,
             print_timings: bool = False):
    """Convenience function to run the simulation pipeline with a single call.
    The simulator needs digested parameters to run. By default they are digested on the fly.
    If you already have digested parameters or want to digest them yourself, set digest_params_first=False.
    """
    if digest_params_first:
        params = digest_params(params)
    simulator = Simulator(params)
    result = simulator.process(image)
    if print_timings:
        print("Simulation timings:", simulator.timings)
    return result


def simulate_preview(image, params: RuntimePhotoParams,
                     digest_params_first: bool = True,
                     print_timings: bool = False):
    """Convenience function to run the simulation pipeline with a single call.
    The simulator needs digested parameters to run. By default they are digested on the fly.
    If you already have digested parameters or want to digest them yourself, set digest_params_first=False.
    """
    max_size = params.settings.preview_max_size
    result = simulate(resize_for_preview(image, max_size), params,
                      digest_params_first=digest_params_first,
                      print_timings=print_timings)
    return result


#######################################################################################################
# Legacy for ART, to be removed in the future when the old API is fully deprecated.

class AgXPhoto(Simulator):
    def __init__(self, params: RuntimePhotoParams):
        digested_params = digest_params(params)
        super().__init__(digested_params)

# photo_params is init_params
def photo_params(film_profile, print_profile) -> RuntimePhotoParams:
    """Legacy helper to build a RuntimePhotoParams with default film and print profiles.
    Build a runtime parameter object.
    It needs to be digested with digest_params before being used in the runtime pipeline.
    film_profile - label string for the film profile to use, e.g. "kodak_portra_400
    print_profile - label string for the print profile to use, e.g. "kodak_portra_endura"
    """
    params = init_params(film_profile=film_profile, print_profile=print_profile)
    params.io.full_image = True # legacy compatibility, has no effect
    params.io.preview_resize_factor = 1.0 # legacy compatibility, has no effect
    return params

__all__ = [
    "RuntimePhotoParams",
    "Simulator",
    "simulate",
    "simulate_preview",
    "AgXPhoto", # legacy for ART
    "photo_params", # legacy for ART
]