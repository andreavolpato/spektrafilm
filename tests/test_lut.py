import numpy as np
import pytest

from spektrafilm.utils.fast_interp_lut import (
    apply_lut_cubic_3d,
    apply_lut_pchip_3d,
)
from spektrafilm.utils.lut import compute_with_lut


pytestmark = pytest.mark.unit


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


def monotone_transform_3d(data):
    output = np.empty_like(data)
    output[..., 0] = 0.55 * data[..., 0] ** 2 + 0.25 * data[..., 1] + 0.15 * data[..., 2]
    output[..., 1] = 0.10 * data[..., 0] + 0.70 * data[..., 1] ** 2 + 0.20 * data[..., 2]
    output[..., 2] = 0.20 * data[..., 0] + 0.15 * data[..., 1] + 0.65 * data[..., 2] ** 2
    return output


def make_monotone_3d_lut_and_image():
    lut_size = 17
    lut_axis = np.linspace(0.0, 1.0, lut_size, dtype=np.float64)
    grid_r, grid_g, grid_b = np.meshgrid(lut_axis, lut_axis, lut_axis, indexing='ij')
    lut_input = np.stack((grid_r, grid_g, grid_b), axis=-1)
    lut = monotone_transform_3d(lut_input)

    image_height = 33
    image_width = 37
    image_r = np.linspace(0.0, 1.0, image_width, dtype=np.float64)[None, :].repeat(image_height, axis=0)
    image_g = np.linspace(0.0, 1.0, image_height, dtype=np.float64)[:, None].repeat(image_width, axis=1)
    image_b = 0.35 * image_r + 0.65 * image_g
    image = np.stack((image_r, image_g, image_b), axis=-1)
    ground_truth = monotone_transform_3d(image)
    return lut, image, ground_truth


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


def test_prepare_monotone_3d_lut_pchip_matches_ground_truth_with_small_error():
    lut, image, ground_truth = make_monotone_3d_lut_and_image()

    output_prepared = apply_lut_pchip_3d(lut, image)

    diff_prepared = output_prepared - ground_truth
    rmse_prepared = np.sqrt(np.mean(diff_prepared**2))
    max_error_prepared = np.max(np.abs(diff_prepared))

    assert rmse_prepared < 2e-4
    assert max_error_prepared < 2e-3


def test_apply_monotone_3d_lut_mitchell_matches_ground_truth_with_small_error():
    lut, image, ground_truth = make_monotone_3d_lut_and_image()

    output_mitchell = apply_lut_cubic_3d(lut, image)

    diff_mitchell = output_mitchell - ground_truth
    rmse_mitchell = np.sqrt(np.mean(diff_mitchell**2))
    max_error_mitchell = np.max(np.abs(diff_mitchell))

    assert rmse_mitchell < 2e-3
    assert max_error_mitchell < 1.2e-2

def test_compute_with_lut_rejects_invalid_input_range():
    data = np.zeros((1, 1, 3), dtype=np.float64)

    with pytest.raises(ValueError, match='xmax must be greater than xmin'):
        compute_with_lut(data, affine_transform, xmin=1.0, xmax=1.0)