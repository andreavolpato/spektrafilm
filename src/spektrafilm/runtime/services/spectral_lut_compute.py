from __future__ import annotations

from typing import Any, Callable
import numpy as np

from spektrafilm.utils.lut import compute_with_lut


class SpectralLUTService:
    def __init__(self, lut_resolution: int):
        self._lut_resolution = lut_resolution
        self.enlarger_lut : np.ndarray | None = None
        self.scanner_lut : np.ndarray | None = None

    def compute(
        self,
        data,
        spectral_calculation: Callable,
        data_min,
        data_max,
        *,
        use_lut: bool = False,
        save_enlarger_lut: bool = False,
        save_scanner_lut: bool = False,
    ):
        if use_lut:
            data_out, lut = compute_with_lut(data,
                                             spectral_calculation,
                                             xmin=data_min,
                                             xmax=data_max,
                                             steps=self._lut_resolution)
            if save_enlarger_lut:
                self.enlarger_lut = lut
            if save_scanner_lut:
                self.scanner_lut = lut
            return data_out
        return spectral_calculation(data)
