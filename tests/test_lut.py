import numpy as np
import pytest

from spectral_film_lab.utils.fast_interp_lut import apply_lut_cubic_2d, apply_lut_cubic_3d
from spectral_film_lab.utils.lut import _create_lut_3d, compute_with_lut


def affine_transform(data):
    output = np.empty_like(data)
    output[..., 0] = 1.5 * data[..., 0] + 0.25 * data[..., 1]
    output[..., 1] = 0.5 * data[..., 1] + 0.75 * data[..., 2]
    output[..., 2] = 1.25 * data[..., 2] + 0.1 * data[..., 0]
    return output


def affine_transform_2d(data):
    output = np.empty((data.shape[0], data.shape[1], 3), dtype=np.float64)
    output[..., 0] = 2.0 * data[..., 0] + 0.5 * data[..., 1]
    output[..., 1] = -0.25 * data[..., 0] + 1.5 * data[..., 1]
    output[..., 2] = 0.75 * data[..., 0] + 0.25 * data[..., 1]
    return output


def test_compute_with_lut_basic_behavior():
    grid = np.linspace(0.0, 1.0, 9, dtype=np.float64)
    data = np.array(
        [
            [[grid[1], grid[2], grid[3]], [grid[4], grid[5], grid[6]]],
            [[grid[6], grid[3], grid[2]], [grid[7], grid[6], grid[1]]],
        ],
        dtype=np.float64,
    )

    output, lut = compute_with_lut(data, affine_transform, steps=9)

    assert output.shape == data.shape
    assert lut.shape == (9, 9, 9, 3)
    np.testing.assert_allclose(output, affine_transform(data), atol=1e-8, rtol=1e-8)


def test_compute_with_lut_normalizes_custom_input_range_before_sampling():
    xmin = 0.2
    xmax = 2.5
    steps = 17
    grid = np.linspace(xmin, xmax, 9, dtype=np.float64)
    data = np.array(
        [
            [[grid[1], grid[2], grid[3]], [grid[4], grid[5], grid[7]]],
            [[grid[7], grid[6], grid[2]], [grid[2], grid[4], grid[7]]],
        ],
        dtype=np.float64,
    )

    output, lut = compute_with_lut(data, affine_transform, xmin=xmin, xmax=xmax, steps=steps)

    assert lut.shape == (steps, steps, steps, 3)
    np.testing.assert_allclose(output, affine_transform(data), atol=1e-8, rtol=1e-8)


def test_compute_with_lut_supports_per_channel_input_ranges():
    xmin = np.array([0.2, -1.0, 10.0], dtype=np.float64)
    xmax = np.array([2.5, 3.0, 20.0], dtype=np.float64)
    steps = 17
    grid_r = np.linspace(xmin[0], xmax[0], 9, dtype=np.float64)
    grid_g = np.linspace(xmin[1], xmax[1], 9, dtype=np.float64)
    grid_b = np.linspace(xmin[2], xmax[2], 9, dtype=np.float64)
    data = np.array(
        [
            [[grid_r[1], grid_g[2], grid_b[3]], [grid_r[4], grid_g[5], grid_b[7]]],
            [[grid_r[7], grid_g[6], grid_b[2]], [grid_r[2], grid_g[4], grid_b[7]]],
        ],
        dtype=np.float64,
    )

    output, lut = compute_with_lut(data, affine_transform, xmin=xmin, xmax=xmax, steps=steps)

    assert lut.shape == (steps, steps, steps, 3)
    np.testing.assert_allclose(output, affine_transform(data), atol=1e-8, rtol=1e-8)


def test_compute_with_lut_matches_affine_transform_at_range_endpoints():
    xmin = np.array([0.2, -1.0, 10.0], dtype=np.float64)
    xmax = np.array([2.5, 3.0, 20.0], dtype=np.float64)
    steps = 17
    grid_r = np.linspace(xmin[0], xmax[0], 9, dtype=np.float64)
    grid_g = np.linspace(xmin[1], xmax[1], 9, dtype=np.float64)
    grid_b = np.linspace(xmin[2], xmax[2], 9, dtype=np.float64)
    data = np.array(
        [
            [[grid_r[0], grid_g[0], grid_b[0]], [grid_r[8], grid_g[4], grid_b[8]]],
            [[grid_r[4], grid_g[8], grid_b[1]], [grid_r[8], grid_g[8], grid_b[8]]],
        ],
        dtype=np.float64,
    )

    output, lut = compute_with_lut(data, affine_transform, xmin=xmin, xmax=xmax, steps=steps)

    assert lut.shape == (steps, steps, steps, 3)
    np.testing.assert_allclose(output, affine_transform(data), atol=1e-8, rtol=1e-8)


def test_apply_lut_cubic_3d_matches_affine_transform_in_outermost_cells():
    xmin = np.array([0.2, -1.0, 10.0], dtype=np.float64)
    xmax = np.array([2.5, 3.0, 20.0], dtype=np.float64)
    data = np.array(
        [
            [[0.22, -0.75, 10.5], [2.42, 2.75, 19.5]],
            [[0.48, 2.6, 10.2], [2.35, -0.8, 19.8]],
        ],
        dtype=np.float64,
    )

    lut = _create_lut_3d(affine_transform, xmin=xmin, xmax=xmax, steps=9)
    normalized = (data - xmin) / (xmax - xmin)
    output = apply_lut_cubic_3d(lut, normalized)

    np.testing.assert_allclose(output, affine_transform(data), atol=1e-8, rtol=1e-8)


def test_apply_lut_cubic_2d_matches_affine_transform_in_outermost_cells():
    lut_size = 9
    axis = np.linspace(0.0, 1.0, lut_size, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(axis, axis, indexing='ij')
    lut_2d = affine_transform_2d(np.stack((grid_x, grid_y), axis=-1))
    image = np.array(
        [
            [[0.05, 0.10], [0.95, 0.90]],
            [[0.08, 0.92], [0.93, 0.07]],
        ],
        dtype=np.float64,
    )

    output = apply_lut_cubic_2d(lut_2d, image)

    np.testing.assert_allclose(output, affine_transform_2d(image), atol=1e-8, rtol=1e-8)


def test_compute_with_lut_rejects_invalid_input_range():
    data = np.zeros((1, 1, 3), dtype=np.float64)

    with pytest.raises(ValueError, match='xmax must be greater than xmin'):
        compute_with_lut(data, affine_transform, xmin=1.0, xmax=1.0)