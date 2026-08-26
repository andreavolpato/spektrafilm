"""Round-trip tests for the .3dl format plugin."""

from __future__ import annotations

import numpy as np

from spektrafilm_lut_creator.formats import LUT_FORMATS, Lut, get_format
from spektrafilm_lut_creator.grid import cube_grid


def _identity_lut(resolution: int) -> Lut:
    flat = cube_grid(resolution)
    return Lut(table=flat.reshape(resolution, resolution, resolution, 3))


class TestRegistry:
    def test_threedl_plugin_registered(self):
        assert "3dl" in LUT_FORMATS
        fmt = get_format("3dl")
        assert fmt.name == "3dl"
        assert ".3dl" in fmt.extensions


class TestThreeDLWrite:
    def test_writes_shape_line_followed_by_data(self, tmp_path):
        lut = _identity_lut(5)
        path = tmp_path / "id5.3dl"
        get_format("3dl").write(lut, path)
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        # Shape line is N evenly-spaced integers spanning [0, 1023].
        shape = [int(v) for v in lines[0].split()]
        assert len(shape) == 5
        assert shape[0] == 0
        assert shape[-1] == 1023
        # Data: N^3 triplets.
        data_lines = lines[1:]
        assert len(data_lines) == 5**3
        for line in data_lines:
            vals = line.split()
            assert len(vals) == 3
            for v in vals:
                assert 0 <= int(v) <= 1023

    def test_writes_optional_header_comments(self, tmp_path):
        lut = _identity_lut(3)
        path = tmp_path / "with_header.3dl"
        get_format("3dl").write(lut, path, header_lines=["spektrafilm test", "v0.1"])
        text = path.read_text(encoding="utf-8")
        assert text.startswith("# spektrafilm test\n# v0.1\n")


class TestThreeDLRoundTrip:
    def test_identity_round_trips_within_quantization(self, tmp_path):
        """10-bit quantization caps round-trip error at ~1/1023 per channel."""
        lut = _identity_lut(17)
        path = tmp_path / "rt17.3dl"
        get_format("3dl").write(lut, path)
        loaded = get_format("3dl").read(path)
        assert loaded.table.shape == lut.table.shape
        np.testing.assert_allclose(loaded.table, lut.table, atol=1.0 / 1023.0)

    def test_nontrivial_lut_round_trips(self, tmp_path):
        rng = np.random.default_rng(42)
        table = rng.random((5, 5, 5, 3))
        lut = Lut(table=table)
        path = tmp_path / "rand.3dl"
        get_format("3dl").write(lut, path)
        loaded = get_format("3dl").read(path)
        np.testing.assert_allclose(loaded.table, table, atol=1.0 / 1023.0)


class TestThreeDLValueClipping:
    def test_negative_and_supersaturated_values_clip_to_legal_range(self, tmp_path):
        table = np.full((3, 3, 3, 3), -0.1)
        table[1, 1, 1] = (1.5, 0.5, -0.5)
        lut = Lut(table=table)
        path = tmp_path / "clipped.3dl"
        get_format("3dl").write(lut, path)
        for line in path.read_text(encoding="utf-8").strip().splitlines()[1:]:
            for v in line.split():
                assert 0 <= int(v) <= 1023
