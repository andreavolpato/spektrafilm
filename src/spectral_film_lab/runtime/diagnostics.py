import numpy as np

from spectral_film_lab.runtime.pipeline import SimulationPipeline

class DiagnosticsPipeline(SimulationPipeline):
    """Runtime pipeline with additional diagnostics and debugging capabilities."""

    def __init__(self, params):
        super().__init__(params)
    
    