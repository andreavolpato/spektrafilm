"""Tests for the sampling-grid helpers."""

from __future__ import annotations

import numpy as np
import pytest

from spektrafilm_lut_creator.grid import cube_grid, grid_as_image, image_as_grid


class TestCubeGrid:
    def test_shape(self):
        grid = cube_grid(33)
        assert grid.shape == (33**3, 3)

    def test_endpoints(self):
        """First sample is (0,0,0); last is (1,1,1)."""
        grid = cube_grid(5)
        np.testing.assert_allclose(grid[0], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(grid[-1], [1.0, 1.0, 1.0])

    def test_red_varies_fastest(self):
        """The Adobe .cube spec mandates R fastest, then G, then B."""
        N = 4
        grid = cube_grid(N)
        # First N rows: R increments from 0..1, G and B held at 0
        np.testing.assert_allclose(grid[:N, 0], np.linspace(0.0, 1.0, N))
        np.testing.assert_allclose(grid[:N, 1], 0.0)
        np.testing.assert_allclose(grid[:N, 2], 0.0)
        # After exactly N rows, R wraps to 0 and G ticks up by 1/(N-1)
        np.testing.assert_allclose(grid[N, 0], 0.0)
        np.testing.assert_allclose(grid[N, 1], 1.0 / (N - 1))
        np.testing.assert_allclose(grid[N, 2], 0.0)
        # After exactly N*N rows, R and G wrap to 0 and B ticks up
        np.testing.assert_allclose(grid[N * N], [0.0, 0.0, 1.0 / (N - 1)])

    def test_uniform_spacing(self):
        N = 17
        grid = cube_grid(N)
        # All unique values per channel should be the same N-step linspace
        for ch in range(3):
            uniq = np.unique(grid[:, ch])
            np.testing.assert_allclose(uniq, np.linspace(0.0, 1.0, N))

    def test_resolution_too_small_raises(self):
        with pytest.raises(ValueError, match=">= 2"):
            cube_grid(1)


class TestGridAsImage:
    def test_default_shape(self):
        N = 5
        grid = cube_grid(N)
        img = grid_as_image(grid, N)
        assert img.shape == (N, N**2, 3)

    def test_round_trip(self):
        N = 7
        grid = cube_grid(N)
        img = grid_as_image(grid, N)
        back = image_as_grid(img, N)
        np.testing.assert_array_equal(grid, back)

    def test_round_trip_after_arbitrary_image_shape(self):
        """If a builder reshapes to a different (H, W), image_as_grid still
        recovers the canonical flat grid as long as N**3 samples remain."""
        N = 5
        grid = cube_grid(N)
        img = grid_as_image(grid, N)
        img_reshaped = img.reshape(N * N, N, 3)
        back = image_as_grid(img_reshaped, N)
        np.testing.assert_array_equal(grid, back)

    def test_wrong_grid_shape_raises(self):
        with pytest.raises(ValueError, match="does not match"):
            grid_as_image(np.zeros((10, 3)), 5)

    def test_wrong_image_sample_count_raises(self):
        with pytest.raises(ValueError, match="resolution\\*\\*3"):
            image_as_grid(np.zeros((10, 10, 3)), 5)

    def test_wrong_image_channels_raises(self):
        with pytest.raises(ValueError, match="last axis must be 3"):
            image_as_grid(np.zeros((5, 5)), 5)
