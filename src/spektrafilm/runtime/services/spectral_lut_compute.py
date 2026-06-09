from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np

from spektrafilm.data.profiles_loader import Hanatos2025SensitivityAdaptation
from spektrafilm.utils.gamut_compression import InputGamutCompressSpec
from spektrafilm.utils.lut import compute_with_lut
from spektrafilm.utils.spectral_upsampling import compute_tc_lut, lut_descriptor
from spektrafilm.utils.timings import timeit


################################################################################
# Memoization primitive shared by every tc_lut
#
# Every tc_lut (irradiance hanatos2025, reflectance jakob2019 / otsu2018) is the
# same computation: expensive to build, recomputed only when its inputs change.
# They differ only in how the cache *key* is derived and which function builds
# the LUT. `_MemoLUT` captures the caching once so each method is just
# "build a key -> memo.get(key, compute)".


def _key_equal(a, b) -> bool:
    """Structural equality for cache keys: tuples that may hold ndarrays, nested
    tuples, strings, frozen specs, or None."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, tuple) and isinstance(b, tuple):
        return len(a) == len(b) and all(_key_equal(x, y) for x, y in zip(a, b))
    return a == b


def _key_snapshot(part):
    """Deep-copy the mutable (ndarray) parts of a key so a later in-place
    mutation of a caller's array can't silently match a stale cache."""
    if isinstance(part, np.ndarray):
        return np.array(part, copy=True)
    if isinstance(part, tuple):
        return tuple(_key_snapshot(p) for p in part)
    return part


@dataclass
class _MemoLUT:
    """A single LUT memoized on a structural key. Recomputes only when the key
    changes; the stored key snapshots its arrays so caller-side mutation is
    safe."""
    lut: np.ndarray | None = None
    _key: tuple | None = None

    def invalidate(self) -> None:
        self.lut = None
        self._key = None

    def get(self, key: tuple, compute: Callable[[], np.ndarray]) -> np.ndarray:
        if self.lut is not None and self._key is not None and _key_equal(self._key, key):
            return self.lut
        self.lut = compute()
        self._key = _key_snapshot(key)
        return self.lut


@dataclass
class _TestCacheLUT:
    """A CMY->spectral LUT validated by a small probe ('test results') rather
    than an input key: the cache is reused when re-running the caller's
    spectral_calculation on a fixed CMY probe reproduces the stored output."""
    lut: np.ndarray | None = None
    test_results: np.ndarray | None = None


def _adaptation_key(adaptation: Hanatos2025SensitivityAdaptation | None) -> tuple:
    """Comparable key for the (mutable, array-bearing) hanatos adaptation.
    Mirrors the fields that change the baked irradiance tc_lut."""
    if adaptation is None:
        return ()
    return (
        bool(adaptation.apply_window),
        bool(adaptation.apply_surface),
        float(adaptation.spectral_gaussian_blur),
        adaptation.reference_illuminant,
        np.asarray(adaptation.window_params),
        np.asarray(adaptation.surface_params),
    )


class SpectralLUTService:
    def __init__(self, lut_resolution: int):
        self._lut_resolution = lut_resolution

        self.timings = {}
        self.hanatos2025_adaptation: Hanatos2025SensitivityAdaptation | None = None
        # Input gamut compression spec (set by the pipeline from
        # params.io.input_gamut_compress). Part of every tc_lut key, so a change
        # drives a rebuild on the next call.
        self.input_gamut_compress: InputGamutCompressSpec = InputGamutCompressSpec()

<<<<<<< HEAD
        # external memory
        self.filming_tc_lut_memory : np.ndarray | None = None # tc_lut memory
        self.enlarger_lut_memory : np.ndarray | None = None # enlarger lut memory
        self.scanner_lut_memory : np.ndarray | None = None # scanner lut memory
        self.convert_lut_memory : np.ndarray | None = None # convert (scan-inverse) lut memory
=======
        # One memo per rgb_to_raw method (lazily created, keyed by identifier). Key
        # derivation differs by kind (reflectance vs irradiance); the caching does not.
        self._tc_lut_memos: dict[str, _MemoLUT] = {}

        # Probe-validated CMY->spectral LUTs.
        self._enlarger_cache = _TestCacheLUT()
        self._scanner_cache = _TestCacheLUT()
>>>>>>> d8ad561 (feat: arctic2026alpha spectral upsampling)

        # local memory
        self._film_sensitivity = None # to track if tc_lut needs to be recomputed when film sensitivity changes
        self._cached_filming_adaptation = None # full adaptation state for which the cached tc_lut was computed
        self._cached_input_gamut_compress: InputGamutCompressSpec | None = None
        self._enlarger_test_results_memory = None # to test if enlarger LUTs are identical for same input
        self._scanner_test_results_memory = None # to test if scanner LUTs are identical for same input
        self._convert_test_results_memory = None # to test if convert LUTs are identical for same input
        
        self._cmy_test_values = np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                                          [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]]) # CMY probe for the test caches

    @staticmethod
    def _supports_3d_lut(cmy_data) -> bool:
        data = np.asarray(cmy_data)
        return data.ndim >= 3 and data.shape[-1] == 3

    @staticmethod
    def _is_single_channel(cmy_data) -> bool:
        """True for a one-channel (black-and-white) emulsion, which uses the 1D
        LUT path instead of the 3D one."""
        data = np.asarray(cmy_data)
        return data.ndim >= 3 and data.shape[-1] == 1

    def set_hanatos2025_adaptation(self, adaptation: Hanatos2025SensitivityAdaptation) -> None:
        # Snapshot defensively: the active config must not change under us if the
        # caller later mutates the adaptation object without calling set again.
        # The filming memo rebuilds lazily when the derived key changes.
        self.hanatos2025_adaptation = self._copy_hanatos2025_adaptation(adaptation)

    def set_input_gamut_compress(self, spec: InputGamutCompressSpec) -> None:
        """Set the input-gamut-compression spec baked into the tc_luts. It is
        part of every tc_lut key, so the memos would rebuild lazily anyway;
        invalidate eagerly to free the stale arrays."""
        if spec != self.input_gamut_compress:
            self.input_gamut_compress = spec
            self._tc_lut_memos.clear()       # drop every method's stale tc_lut

    @staticmethod
    def _copy_hanatos2025_adaptation(
        adaptation: Hanatos2025SensitivityAdaptation | None,
    ) -> Hanatos2025SensitivityAdaptation | None:
        if adaptation is None:
            return None
        # Preserve None (absent adaptation, e.g. BW): np.array(None) would make a
        # 0-d object array that slips past the `is not None` guards downstream.
        copy_params = lambda p: None if p is None else np.array(p, copy=True)
        return Hanatos2025SensitivityAdaptation(
            window_params=copy_params(adaptation.window_params),
            surface_params=copy_params(adaptation.surface_params),
            spectral_gaussian_blur=float(adaptation.spectral_gaussian_blur),
            reference_illuminant=adaptation.reference_illuminant,
            apply_window=bool(adaptation.apply_window),
            apply_surface=bool(adaptation.apply_surface),
            active=adaptation.active,
        )

    # -- tc_luts ---------------------------------------------------------------

<<<<<<< HEAD
    @timeit("spectral_compute_convert")
    def spectral_compute_convert(self,
        rgb_data,
        invert_fn: Callable,
        data_min,
        data_max,
        *,
        use_lut: bool = False,
        steps: int | None = None,
    ):
        """Accelerate the scan-inverse (convert-film) with a 3D LUT.

        ``invert_fn`` maps gained/calibration-corrected RGB ``(..., 3)`` to film
        cmy density; ``data_min`` / ``data_max`` are the per-channel gamut bounds
        (``FilmConverter.gamut_bounds``). The LUT is baked over that RGB box and
        reused while the inverse is unchanged. As with the scanner / enlarger LUTs,
        a small fixed probe detects when the underlying transform changed (i.e. the
        scan model's illuminant / base / dye / Dmax moved) and forces a rebuild.

        The device calibration and exposure gain are applied *before* this call
        (``FilmConverter.pretransform``), so tuning them never invalidates the LUT.
        """
        if not use_lut:
            return invert_fn(rgb_data)
        # The convert input is always 3-channel RGB (a scanned negative). The
        # cmy output may be 1 (B&W) or 3 (colour) channels; _create_lut_3d takes
        # the output channel count from invert_fn, so both work on the 3D path.
        if not self._supports_3d_lut(rgb_data):
            return invert_fn(rgb_data)
        steps = self._lut_resolution if steps is None else int(steps)

        test_results = invert_fn(np.array(self._cmy_test_values))

        if (
            self.convert_lut_memory is not None
            and self._convert_test_results_memory is not None
            and self.convert_lut_memory.shape[0] == steps
            and np.array_equal(test_results, self._convert_test_results_memory)
        ):
            data_out, _ = compute_with_lut(rgb_data,
                                           invert_fn,
                                           xmin=data_min,
                                           xmax=data_max,
                                           steps=steps,
                                           lut=self.convert_lut_memory)
        else:
            data_out, lut = compute_with_lut(rgb_data,
                                             invert_fn,
                                             xmin=data_min,
                                             xmax=data_max,
                                             steps=steps)
            self.convert_lut_memory = lut
            self._convert_test_results_memory = np.array(test_results, copy=True)

        if data_out is None:
            raise RuntimeError('LUT computation did not produce an output')
        return data_out
=======
    def _tc_lut_memo(self, method: str) -> _MemoLUT:
        """The per-method memo, created on first use."""
        memo = self._tc_lut_memos.get(method)
        if memo is None:
            memo = self._tc_lut_memos[method] = _MemoLUT()
        return memo
>>>>>>> d8ad561 (feat: arctic2026alpha spectral upsampling)

    @timeit("get_filming_tc_lut")
    def get_filming_tc_lut(self, method, sensitivity, reference_illuminant):
        """tc_lut for any rgb_to_raw `method`, dispatched on its descriptor kind and
        memoized per method. REFLECTANCE methods (jakob2019 / otsu2018 / mallett2019 /
        arctic2026…) relight under the film reference illuminant and key on
        (sensitivity, reference + scene illuminants, gamut compress). The IRRADIANCE
        method (hanatos2025) integrates directly and keys on (sensitivity, the full
        hanatos adaptation, gamut compress) instead — its reference_illuminant rides in
        the adaptation, not the relight."""
        sensitivity = np.asarray(sensitivity)
        descriptor = lut_descriptor(method)
        if descriptor.get('kind') == 'irradiance':
            adaptation = self.hanatos2025_adaptation
            key = (method, sensitivity, _adaptation_key(adaptation), self.input_gamut_compress)
            compute = lambda: compute_tc_lut(
                method, sensitivity, hanatos2025_adaptation=adaptation,
                gamut_compress=self.input_gamut_compress)
        else:
            scene_illuminant = descriptor['reflectance']['scene_illuminant']
            key = (method, sensitivity, reference_illuminant, scene_illuminant,
                   self.input_gamut_compress)
            compute = lambda: compute_tc_lut(
                method, sensitivity, reference_illuminant,
                gamut_compress=self.input_gamut_compress)
        return self._tc_lut_memo(method).get(key, compute)

    # -- CMY->spectral LUTs ----------------------------------------------------

    @timeit("spectral_compute_enlarger")
    def spectral_compute_enlarger(self, cmy_data, spectral_calculation: Callable,
                                  data_min, data_max, *, use_lut: bool = False):
        return self._spectral_compute_with_test_cache(
            self._enlarger_cache, cmy_data, spectral_calculation,
            data_min, data_max, use_lut)

    @timeit("spectral_compute_scanner")
    def spectral_compute_scanner(self, cmy_data, spectral_calculation: Callable,
                                 data_min, data_max, *, use_lut: bool = False):
        return self._spectral_compute_with_test_cache(
            self._scanner_cache, cmy_data, spectral_calculation,
            data_min, data_max, use_lut)

    def _spectral_compute_with_test_cache(self, cache: _TestCacheLUT, cmy_data,
                                          spectral_calculation: Callable,
                                          data_min, data_max, use_lut: bool):
        """Shared CMY->spectral LUT path (enlarger, scanner). The cached LUT is
        reused when a fixed CMY probe reproduces the stored output, i.e. the
        spectral_calculation closure is unchanged."""
        if not use_lut:
            return spectral_calculation(cmy_data)

        test_results = spectral_calculation(np.array(self._cmy_test_values))

        if (
            cache.lut is not None
            and cache.test_results is not None
            and np.array_equal(test_results, cache.test_results)
        ):
            data_out, _ = compute_with_lut(cmy_data, spectral_calculation,
                                           xmin=data_min, xmax=data_max,
                                           steps=self._lut_resolution, lut=cache.lut)
        else:
            data_out, lut = compute_with_lut(cmy_data, spectral_calculation,
                                             xmin=data_min, xmax=data_max,
                                             steps=self._lut_resolution)
            cache.lut = lut
            cache.test_results = np.array(test_results, copy=True)

        if data_out is None:
            raise RuntimeError('LUT computation did not produce an output')
        return data_out
