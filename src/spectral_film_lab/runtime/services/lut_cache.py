from __future__ import annotations

from typing import Any, Callable

from spectral_film_lab.utils.lut import compute_with_lut


class SpectralLUTCache:
    def __init__(self, lut_resolution: int, debug_luts: Any):
        self._lut_resolution = lut_resolution
        self._debug_luts = debug_luts

    def compute(
        self,
        data,
        spectral_calculation: Callable,
        *,
        use_lut: bool = False,
        save_enlarger_lut: bool = False,
        save_scanner_lut: bool = False,
    ):
        if use_lut:
            data_out, lut = compute_with_lut(data, spectral_calculation, steps=self._lut_resolution)
            if save_enlarger_lut:
                self._debug_luts.enlarger_lut = lut
            if save_scanner_lut:
                self._debug_luts.scanner_lut = lut
            return data_out
        return spectral_calculation(data)
