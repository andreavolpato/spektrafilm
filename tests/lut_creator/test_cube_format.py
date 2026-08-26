"""Round-trip tests for the .cube format plugin."""

from __future__ import annotations

import numpy as np
import pytest

from spektrafilm_lut_creator.formats import LUT_FORMATS, Lut, get_format
from spektrafilm_lut_creator.grid import cube_grid


def _identity_lut(resolution: int, title: str = "") -> Lut:
    """Build an identity LUT: code in → code out, in canonical .cube order."""
    flat = cube_grid(resolution)
    table = flat.reshape(resolution, resolution, resolution, 3)
    return Lut(table=table, title=title)


class TestRegistry:
    def test_cube_plugin_registered(self):
        assert "cube" in LUT_FORMATS
        fmt = get_format("cube")
        assert fmt.name == "cube"
        assert ".cube" in fmt.extensions

    def test_unknown_format_raises(self):
        with pytest.raises(KeyError, match="Unknown LUT format"):
            get_format("not_a_real_format")


class TestCubeRoundTrip:
    def test_identity_lut_round_trip(self, tmp_path):
        lut = _identity_lut(17, title="identity-17")
        path = tmp_path / "id17.cube"

        get_format("cube").write(lut, path)
        loaded = get_format("cube").read(path)

        assert loaded.title == "identity-17"
        assert loaded.resolution == 17
        np.testing.assert_allclose(loaded.table, lut.table, atol=1e-9)
        np.testing.assert_allclose(loaded.domain_min, lut.domain_min)
        np.testing.assert_allclose(loaded.domain_max, lut.domain_max)

    def test_arbitrary_lut_round_trip(self, tmp_path):
        rng = np.random.default_rng(0)
        N = 5
        table = rng.uniform(size=(N, N, N, 3))
        lut = Lut(
            table=table,
            title="random",
            domain_min=(0.1, 0.0, -0.1),
            domain_max=(1.1, 1.0, 0.9),
        )
        path = tmp_path / "random.cube"

        get_format("cube").write(lut, path)
        loaded = get_format("cube").read(path)

        np.testing.assert_allclose(loaded.table, table, atol=1e-9)
        np.testing.assert_allclose(loaded.domain_min, (0.1, 0.0, -0.1))
        np.testing.assert_allclose(loaded.domain_max, (1.1, 1.0, 0.9))

    def test_written_body_has_red_fastest_order(self, tmp_path):
        """Row N of the file body must correspond to (R=0, G=1/(N-1), B=0)."""
        N = 4
        lut = _identity_lut(N)
        path = tmp_path / "order.cube"
        get_format("cube").write(lut, path)

        body_lines = [
            ln
            for ln in path.read_text(encoding="utf-8").splitlines()
            if ln
            and not ln.startswith(("TITLE", "DOMAIN_MIN", "DOMAIN_MAX", "LUT_3D_SIZE"))
        ]
        # First N rows: R = linspace(0..1), G = B = 0
        first_n = [list(map(float, line.split())) for line in body_lines[:N]]
        np.testing.assert_allclose(
            first_n,
            np.column_stack([np.linspace(0, 1, N), np.zeros(N), np.zeros(N)]),
        )

    def test_comments_and_blank_lines_ignored_on_read(self, tmp_path):
        path = tmp_path / "commented.cube"
        path.write_text(
            "# header comment\n"
            'TITLE "with comments"\n'
            "\n"
            "DOMAIN_MIN 0 0 0\n"
            "DOMAIN_MAX 1 1 1\n"
            "LUT_3D_SIZE 2\n"
            "# body starts\n"
            "0 0 0\n1 0 0\n"
            "0 1 0\n1 1 0\n"
            "0 0 1\n1 0 1\n"
            "0 1 1\n1 1 1\n",
            encoding="utf-8",
        )
        loaded = get_format("cube").read(path)
        assert loaded.title == "with comments"
        assert loaded.resolution == 2

    def test_missing_size_raises(self, tmp_path):
        path = tmp_path / "no_size.cube"
        path.write_text("DOMAIN_MIN 0 0 0\n0 0 0\n", encoding="utf-8")
        with pytest.raises(ValueError, match="missing LUT_3D_SIZE"):
            get_format("cube").read(path)

    def test_body_size_mismatch_raises(self, tmp_path):
        path = tmp_path / "short.cube"
        path.write_text("LUT_3D_SIZE 3\n0 0 0\n1 1 1\n", encoding="utf-8")
        with pytest.raises(ValueError, match="body has .* entries"):
            get_format("cube").read(path)

    def test_lut_1d_size_rejected(self, tmp_path):
        path = tmp_path / "1d.cube"
        path.write_text("LUT_1D_SIZE 17\n", encoding="utf-8")
        with pytest.raises(ValueError, match="1D .cube"):
            get_format("cube").read(path)
