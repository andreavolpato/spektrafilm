from __future__ import annotations

from typing import Callable
import numpy as np

from spektrafilm.utils.lut import compute_with_lut
from spektrafilm.utils.spectral_upsampling import compute_hanatos2025_tc_lut
from spektrafilm.utils.timings import timeit


class SpectralLUTService:
    def __init__(self, lut_resolution: int):
        self._lut_resolution = lut_resolution
        self.timings = {}
        self.filming_tc_lut_memory : np.ndarray | None = None # tc_lut memory
        self.enlarger_lut_memory : np.ndarray | None = None # enlarger lut memory
        self.scanner_lut_memory : np.ndarray | None = None # scanner lut memory
        self._film_sensitivity = None # to track if tc_lut needs to be recomputed when film sensitivity changes
        self._enlarger_test_results_memory = None # to test if enlarger LUTs are identical for same input
        self._scanner_test_results_memory = None # to test if scanner LUTs are identical for same input
        
        self._cmy_test_values = np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                                          [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]]) # to test if LUTs are identical

    @timeit("spectral_compute_enlarger")
    def spectral_compute_enlarger(self,
        cmy_data,
        spectral_calculation: Callable,
        data_min,
        data_max,
        *,
        use_lut: bool = False,
    ):
        if not use_lut:
            return spectral_calculation(cmy_data)

        test_results = spectral_calculation(np.array(self._cmy_test_values))

        if (
            self.enlarger_lut_memory is not None
            and self._enlarger_test_results_memory is not None
            and np.array_equal(test_results, self._enlarger_test_results_memory)
        ):
            data_out, _ = compute_with_lut(cmy_data,
                                           spectral_calculation,
                                           xmin=data_min,
                                           xmax=data_max,
                                           steps=self._lut_resolution,
                                           lut=self.enlarger_lut_memory)
        else:
            data_out, lut = compute_with_lut(cmy_data,
                                             spectral_calculation,
                                             xmin=data_min,
                                             xmax=data_max,
                                             steps=self._lut_resolution)
            self.enlarger_lut_memory = lut
            self._enlarger_test_results_memory = np.array(test_results, copy=True)

        if data_out is None:
            raise RuntimeError('LUT computation did not produce an output')
        return data_out

    @timeit("spectral_compute_scanner")
    def spectral_compute_scanner(self,
        cmy_data,
        spectral_calculation: Callable,
        data_min,
        data_max,
        *,
        use_lut: bool = False,
    ):
        if not use_lut:
            return spectral_calculation(cmy_data)

        test_results = spectral_calculation(np.array(self._cmy_test_values))

        if (
            self.scanner_lut_memory is not None
            and self._scanner_test_results_memory is not None
            and np.array_equal(test_results, self._scanner_test_results_memory)
        ):
            data_out, _ = compute_with_lut(cmy_data,
                                           spectral_calculation,
                                           xmin=data_min,
                                           xmax=data_max,
                                           steps=self._lut_resolution,
                                           lut=self.scanner_lut_memory)
        else:
            data_out, lut = compute_with_lut(cmy_data,
                                             spectral_calculation,
                                             xmin=data_min,
                                             xmax=data_max,
                                             steps=self._lut_resolution)
            self.scanner_lut_memory = lut
            self._scanner_test_results_memory = np.array(test_results, copy=True)

        if data_out is None:
            raise RuntimeError('LUT computation did not produce an output')
        return data_out

    @timeit("get_filming_tc_lut")
    def get_filming_tc_lut(self, sensitivity):
        sensitivity = np.asarray(sensitivity)
        if (
            self.filming_tc_lut_memory is not None
            and self._film_sensitivity is not None
            and np.array_equal(self._film_sensitivity, sensitivity)
        ):
            return self.filming_tc_lut_memory

        self._film_sensitivity = np.array(sensitivity, copy=True)
        self.filming_tc_lut_memory = compute_hanatos2025_tc_lut(sensitivity)
        return self.filming_tc_lut_memory
