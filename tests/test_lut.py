import numpy as np
import pytest

from spectral_film_lab.utils.lut import compute_with_lut


def affine_transform(data):
    output = np.empty_like(data)
    output[..., 0] = 1.5 * data[..., 0] + 0.25 * data[..., 1]
    output[..., 1] = 0.5 * data[..., 1] + 0.75 * data[..., 2]
    output[..., 2] = 1.25 * data[..., 2] + 0.1 * data[..., 0]
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
    grid = np.linspace(xmin, xmax, 9, dtype=np.float64)
    data = np.array(
        [
            [[grid[1], grid[2], grid[3]], [grid[4], grid[5], grid[7]]],
            [[grid[7], grid[6], grid[2]], [grid[2], grid[4], grid[7]]],
        ],
        dtype=np.float64,
    )

    output, lut = compute_with_lut(data, affine_transform, xmin=xmin, xmax=xmax, steps=9)

    assert lut.shape == (9, 9, 9, 3)
    np.testing.assert_allclose(output, affine_transform(data), atol=1e-8, rtol=1e-8)


def test_compute_with_lut_supports_per_channel_input_ranges():
    xmin = np.array([0.2, -1.0, 10.0], dtype=np.float64)
    xmax = np.array([2.5, 3.0, 20.0], dtype=np.float64)
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

    output, lut = compute_with_lut(data, affine_transform, xmin=xmin, xmax=xmax, steps=9)

    assert lut.shape == (9, 9, 9, 3)
    np.testing.assert_allclose(output, affine_transform(data), atol=1e-8, rtol=1e-8)


def test_compute_with_lut_rejects_invalid_input_range():
    data = np.zeros((1, 1, 3), dtype=np.float64)

    with pytest.raises(ValueError, match='xmax must be greater than xmin'):
        compute_with_lut(data, affine_transform, xmin=1.0, xmax=1.0)