from __future__ import annotations

from typing import Callable
import numpy as np

from spektrafilm.profiles.io import Hanatos2025SensitivityAdaptation
from spektrafilm.utils.lut import compute_with_lut
from spektrafilm.utils.spectral_upsampling import compute_hanatos2025_tc_lut
from spektrafilm.utils.timings import timeit


class SpectralLUTService:
    def __init__(self, lut_resolution: int):
        self._lut_resolution = lut_resolution
        
        self.timings = {}
        self.hanatos2025_adaptation = None # to be set by filming stage with info from film profile and settings
        
        # external memory
        self.filming_tc_lut_memory : np.ndarray | None = None # tc_lut memory
        self.enlarger_lut_memory : np.ndarray | None = None # enlarger lut memory
        self.scanner_lut_memory : np.ndarray | None = None # scanner lut memory
        
        # local memory
        self._film_sensitivity = None # to track if tc_lut needs to be recomputed when film sensitivity changes
        self._cached_filming_adaptation = None # full adaptation state for which the cached tc_lut was computed
        self._enlarger_test_results_memory = None # to test if enlarger LUTs are identical for same input
        self._scanner_test_results_memory = None # to test if scanner LUTs are identical for same input
        
        self._cmy_test_values = np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                                          [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]]) # to test if LUTs are identical

    def set_hanatos2025_adaptation(self, adaptation: Hanatos2025SensitivityAdaptation) -> None:
        adaptation_copy = self._copy_hanatos2025_adaptation(adaptation)
        self.hanatos2025_adaptation = adaptation_copy
        if not self._same_hanatos2025_adaptation(self._cached_filming_adaptation, adaptation_copy):
            self.filming_tc_lut_memory = None
            self._film_sensitivity = None
            self._cached_filming_adaptation = None

    @staticmethod
    def _copy_hanatos2025_adaptation(
        adaptation: Hanatos2025SensitivityAdaptation | None,
    ) -> Hanatos2025SensitivityAdaptation | None:
        if adaptation is None:
            return None
        return Hanatos2025SensitivityAdaptation(
            window_params=np.array(adaptation.window_params, copy=True),
            surface_params=np.array(adaptation.surface_params, copy=True),
            spectral_gaussian_blur=float(adaptation.spectral_gaussian_blur),
            reference_illuminant=adaptation.reference_illuminant,
            apply_window=bool(adaptation.apply_window),
            apply_surface=bool(adaptation.apply_surface),
            active=adaptation.active,
        )

    @staticmethod
    def _same_hanatos2025_adaptation(
        left: Hanatos2025SensitivityAdaptation | None,
        right: Hanatos2025SensitivityAdaptation | None,
    ) -> bool:
        if left is None or right is None:
            return left is right
        return (
            bool(left.apply_window) == bool(right.apply_window)
            and bool(left.apply_surface) == bool(right.apply_surface)
            and float(left.spectral_gaussian_blur) == float(right.spectral_gaussian_blur)
            and left.reference_illuminant == right.reference_illuminant
            and np.array_equal(left.window_params, right.window_params)
            and np.array_equal(left.surface_params, right.surface_params)
        )

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
            and self._same_hanatos2025_adaptation(self._cached_filming_adaptation, self.hanatos2025_adaptation)
            and np.array_equal(self._film_sensitivity, sensitivity)
        ):
            return self.filming_tc_lut_memory

        self._film_sensitivity = np.array(sensitivity, copy=True)
        self._cached_filming_adaptation = self._copy_hanatos2025_adaptation(self.hanatos2025_adaptation)
        self.filming_tc_lut_memory = compute_hanatos2025_tc_lut(sensitivity, self.hanatos2025_adaptation)
        return self.filming_tc_lut_memory
