"""Round-trip tests for the Hald-CLUT PNG format plugin."""

from __future__ import annotations

import numpy as np
import pytest

from spektrafilm_lut_creator.formats import LUT_FORMATS, Lut, get_format
from spektrafilm_lut_creator.grid import cube_grid


def _identity_lut(resolution: int) -> Lut:
    flat = cube_grid(resolution)
    return Lut(table=flat.reshape(resolution, resolution, resolution, 3))


class TestRegistry:
    def test_hald_png_plugin_registered(self):
        assert "hald_png" in LUT_FORMATS
        fmt = get_format("hald_png")
        assert fmt.name == "hald_png"
        assert ".png" in fmt.extensions


class TestHaldPNGWrite:
    @pytest.mark.parametrize(
        "resolution,expected_side",
        [
            (16, 64),  # level 4 → 4³ = 64
            (25, 125),  # level 5
            (36, 216),  # level 6
        ],
    )
    def test_image_side_matches_hald_level(self, tmp_path, resolution, expected_side):
        lut = _identity_lut(resolution)
        path = tmp_path / "h.png"
        get_format("hald_png").write(lut, path)
        from PIL import Image

        with Image.open(path) as im:
            assert im.size == (expected_side, expected_side)
            assert im.mode == "RGB"

    def test_rejects_non_perfect_square_resolution(self, tmp_path):
        """Hald is strict: N must be a perfect square (level²)."""
        lut = _identity_lut(33)
        path = tmp_path / "bad.png"
        with pytest.raises(ValueError, match="perfect-square cube resolution"):
            get_format("hald_png").write(lut, path)


class TestHaldPNGRoundTrip:
    def test_identity_round_trips_within_8bit_quantization(self, tmp_path):
        """8-bit ONGs hold values to ~1/255 per channel."""
        lut = _identity_lut(16)
        path = tmp_path / "rt.png"
        get_format("hald_png").write(lut, path)
        loaded = get_format("hald_png").read(path)
        assert loaded.table.shape == lut.table.shape
        np.testing.assert_allclose(loaded.table, lut.table, atol=1.0 / 255.0)

    def test_nontrivial_lut_round_trips(self, tmp_path):
        rng = np.random.default_rng(42)
        table = rng.random((16, 16, 16, 3))
        lut = Lut(table=table)
        path = tmp_path / "rand.png"
        get_format("hald_png").write(lut, path)
        loaded = get_format("hald_png").read(path)
        np.testing.assert_allclose(loaded.table, table, atol=1.5 / 255.0)
