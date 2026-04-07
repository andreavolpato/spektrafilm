from __future__ import annotations

import numpy as np
import pytest

from .conftest import make_fast_test_params

from spektrafilm.runtime.process import simulate


pytestmark = pytest.mark.integration


def _assert_valid_output(
    result: np.ndarray,
    *,
    shape: tuple[int, int, int] | None = None,
    bounded: bool = True,
) -> None:
    if shape is not None:
        assert result.shape == shape
    assert result.shape[2] == 3
    assert np.all(np.isfinite(result))
    if bounded:
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)


def _tile_rgb(rgb: tuple[float, float, float], size: int) -> np.ndarray:
    return np.ones((size, size, 3), dtype=np.float64) * np.asarray(rgb, dtype=np.float64)


def test_pipeline_returns_valid_outputs_for_edge_cases(default_params) -> None:
    cases = [
        {
            'image': _tile_rgb((0.30, 0.10, 0.05), 4),
            'configure': lambda params: setattr(params.io, 'upscale_factor', 2.0),
            'shape': (8, 8, 3),
        },
        {
            'image': _tile_rgb((0.30, 0.10, 0.05), 8),
            'configure': lambda params: setattr(params.io, 'upscale_factor', 0.5),
            'shape': (4, 4, 3),
        },
        {
            'image': np.zeros((8, 8, 3), dtype=np.float64),
            'configure': lambda params: None,
            'shape': (8, 8, 3),
        },
        {
            'image': np.ones((8, 8, 3), dtype=np.float64) * 10000.0,
            'configure': lambda params: None,
            'shape': (8, 8, 3),
        },
        {
            'image': _tile_rgb((0.18, 0.18, 0.18), 4),
            'configure': lambda params: setattr(params.debug, 'return_film_log_raw', True),
            'shape': (4, 4, 3),
            'bounded': False,
        },
    ]

    for case in cases:
        default_params.io.upscale_factor = 1.0
        default_params.io.scan_film = False
        default_params.debug.return_film_log_raw = False
        case['configure'](default_params)

        result = simulate(case['image'], default_params)

        _assert_valid_output(result, shape=case['shape'], bounded=case.get('bounded', True))


def test_scan_film_changes_pipeline_branch(default_params) -> None:
    patch = _tile_rgb((0.30, 0.10, 0.05), 4)

    default_params.io.scan_film = False
    print_result = simulate(patch, default_params)

    default_params.io.scan_film = True
    negative_result = simulate(patch, default_params)

    _assert_valid_output(print_result, shape=(4, 4, 3))
    _assert_valid_output(negative_result, shape=(4, 4, 3))
    assert not np.allclose(print_result, negative_result, atol=1e-3)


def test_uniform_gray_input_has_no_spatial_artifacts(default_params) -> None:
    gray = _tile_rgb((0.184, 0.184, 0.184), 8)
    result = simulate(gray, default_params)

    _assert_valid_output(result, shape=(8, 8, 3))
    center_pixel = result[2, 2, :]
    for row in range(1, 6):
        for col in range(1, 6):
            np.testing.assert_allclose(result[row, col, :], center_pixel, atol=1e-6)


def test_transfer_curve_controls_behave_consistently(default_params) -> None:
    gray = _tile_rgb((0.18, 0.18, 0.18), 4)
    levels = [0.02, 0.05, 0.18, 0.5, 0.90]
    means = [np.mean(simulate(_tile_rgb((level, level, level), 4), default_params)) for level in levels]

    for index in range(len(means) - 1):
        assert means[index] < means[index + 1]

    default_params.enlarger.print_exposure_compensation = False
    default_params.camera.exposure_compensation_ev = 0.0
    base = np.mean(simulate(gray, default_params))
    default_params.camera.exposure_compensation_ev = +2.0
    bright = np.mean(simulate(gray, default_params))
    default_params.camera.exposure_compensation_ev = -2.0
    dark = np.mean(simulate(gray, default_params))
    assert dark < base < bright

    default_params.camera.exposure_compensation_ev = 0.0
    default_params.enlarger.print_exposure_compensation = True
    base_comp = simulate(gray, default_params)
    default_params.camera.exposure_compensation_ev = +2.0
    bright_comp = simulate(gray, default_params)

    default_params.camera.exposure_compensation_ev = 0.0
    default_params.enlarger.print_exposure_compensation = False
    base_no_comp = simulate(gray, default_params)
    default_params.camera.exposure_compensation_ev = +2.0
    bright_no_comp = simulate(gray, default_params)

    delta_comp = abs(np.mean(bright_comp) - np.mean(base_comp))
    delta_no_comp = abs(np.mean(bright_no_comp) - np.mean(base_no_comp))
    assert delta_comp < delta_no_comp
    assert delta_comp < 0.05


def test_normalize_print_exposure_false_bypasses_compensation(default_params) -> None:
    gray = _tile_rgb((0.18, 0.18, 0.18), 4)

    default_params.enlarger.normalize_print_exposure = False
    default_params.enlarger.print_exposure_compensation = True
    default_params.camera.auto_exposure = False
    default_params.camera.exposure_compensation_ev = +1.0
    result_with_comp = simulate(gray, default_params)

    default_params.enlarger.print_exposure_compensation = False
    result_without_comp = simulate(gray, default_params)

    np.testing.assert_allclose(result_with_comp, result_without_comp, atol=1e-6)


def test_pipeline_distinguishes_stocks_and_input_chroma(default_params) -> None:
    green_patch = _tile_rgb((0.05, 0.4, 0.05), 10)
    result_portra = simulate(green_patch, default_params)

    params_fuji = make_fast_test_params(film_profile='fujifilm_c200')
    result_fuji = simulate(green_patch, params_fuji)

    assert not np.allclose(result_portra, result_fuji, atol=1e-8)

    colors = {
        'red': _tile_rgb((0.5, 0.05, 0.05), 4),
        'green': _tile_rgb((0.05, 0.5, 0.05), 4),
        'blue': _tile_rgb((0.05, 0.05, 0.5), 4),
    }
    results = {name: simulate(image, default_params)[1, 1, :] for name, image in colors.items()}

    assert not np.allclose(results['red'], results['green'], atol=1e-2)
    assert not np.allclose(results['green'], results['blue'], atol=1e-2)
    assert not np.allclose(results['red'], results['blue'], atol=1e-2)


def test_pipeline_is_deterministic_with_stochastic_effects_disabled(default_params) -> None:
    gray = _tile_rgb((0.18, 0.18, 0.18), 4)
    result_1 = simulate(gray, default_params)
    result_2 = simulate(gray, default_params)

    np.testing.assert_array_equal(result_1, result_2)


def test_lut_path_stays_close_to_direct_path(default_params) -> None:
    gray = _tile_rgb((0.18, 0.18, 0.18), 4)

    result_direct = simulate(gray, default_params)

    default_params.settings.use_enlarger_lut = True
    default_params.settings.use_scanner_lut = True
    default_params.settings.lut_resolution = 17
    result_lut = simulate(gray, default_params)

    _assert_valid_output(result_lut, shape=(4, 4, 3))
    np.testing.assert_allclose(result_lut, result_direct, atol=0.02)


def test_auto_exposure_normalizes_bright_inputs(default_params) -> None:
    bright_patch = _tile_rgb((0.8, 0.8, 0.8), 8)
    default_params.camera.auto_exposure = True
    default_params.enlarger.print_exposure_compensation = False

    for method in ('center_weighted', 'median'):
        default_params.camera.auto_exposure_method = method
        result = simulate(bright_patch, default_params)
        _assert_valid_output(result, shape=(8, 8, 3))

    default_params.camera.auto_exposure = False
    default_params.camera.exposure_compensation_ev = 0.0
    manual_result = simulate(bright_patch, default_params)

    default_params.camera.auto_exposure = True
    default_params.camera.auto_exposure_method = 'center_weighted'
    auto_result = simulate(bright_patch, default_params)

    assert np.mean(auto_result) < np.mean(manual_result)
    assert 0.45 < np.mean(auto_result) < 0.51
