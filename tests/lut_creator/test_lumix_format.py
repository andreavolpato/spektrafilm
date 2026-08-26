"""Tests for the Panasonic Lumix ``.cube`` format plugin.

The exact header layout matters here — Lumix's parser is known to be
strict about the ``#LUMIXPHOTOSTYLE`` tag position and the ordering of
``LUT_3D_SIZE`` before ``DOMAIN_*``. These tests pin the layout against
the working reference script.
"""

from __future__ import annotations

import numpy as np

from spektrafilm_lut_creator.formats import Lut, get_format
from spektrafilm_lut_creator.formats.lumix import (
    LUMIX_PHOTOSTYLE_BY_INPUT,
    LumixCubeFormat,
)


def _make_lut(n: int = 3) -> Lut:
    """Tiny identity-ish LUT for layout tests."""
    table = np.empty((n, n, n, 3), dtype=float)
    axis = np.linspace(0.0, 1.0, n)
    for b in range(n):
        for g in range(n):
            for r in range(n):
                table[b, g, r, :] = (axis[r], axis[g], axis[b])
    return Lut(table=table, title="lumix_test")


class TestLumixRegistry:
    def test_registers_as_lumix(self):
        fmt = get_format("lumix")
        assert isinstance(fmt, LumixCubeFormat)
        assert fmt.name == "lumix"

    def test_photostyle_map_contains_v_log(self):
        # The verified mapping for V-Log; further entries grow only on
        # field-confirmed compatibility.
        assert LUMIX_PHOTOSTYLE_BY_INPUT["Panasonic V-Log"] == "VLOG"


class TestLumixLayout:
    def test_header_order_matches_working_reference(self, tmp_path):
        """The Lumix-verified layout is:
        TITLE -> #LUMIXPHOTOSTYLE -> LUT_3D_SIZE -> DOMAIN_MIN -> DOMAIN_MAX -> '' -> data
        """
        lut = _make_lut(n=3)
        path = tmp_path / "out.cube"
        get_format("lumix").write(lut, path, photo_style_tag="VLOG")
        lines = path.read_text(encoding="utf-8").splitlines()

        assert lines[0].startswith('TITLE "lumix_test"')
        assert lines[1] == "#LUMIXPHOTOSTYLE VLOG"
        assert lines[2] == "LUT_3D_SIZE 3"
        assert lines[3].startswith("DOMAIN_MIN ")
        assert lines[4].startswith("DOMAIN_MAX ")
        assert lines[5] == ""  # blank line before data
        # First data row immediately follows.
        first_data = lines[6].split()
        assert len(first_data) == 3
        for value in first_data:
            float(value)  # parseable

    def test_no_photostyle_tag_when_omitted(self, tmp_path):
        path = tmp_path / "no_tag.cube"
        get_format("lumix").write(_make_lut(n=3), path)
        text = path.read_text(encoding="utf-8")
        assert "#LUMIXPHOTOSTYLE" not in text

    def test_no_extra_comments_emitted(self, tmp_path):
        """Provenance comments are intentionally suppressed in Lumix mode.

        Even when ``header_lines`` is passed (protocol parity with the
        standard cube writer), it is ignored.
        """
        path = tmp_path / "stripped.cube"
        get_format("lumix").write(
            _make_lut(n=3),
            path,
            photo_style_tag="VLOG",
            header_lines=["spektrafilm v0", "Should not appear"],
        )
        text = path.read_text(encoding="utf-8")
        comment_lines = [line for line in text.splitlines() if line.startswith("#")]
        assert comment_lines == ["#LUMIXPHOTOSTYLE VLOG"]

    def test_fixed_decimal_precision_in_body(self, tmp_path):
        """Body uses .6f fixed precision (no scientific notation)."""
        path = tmp_path / "precision.cube"
        get_format("lumix").write(_make_lut(n=3), path, photo_style_tag="VLOG")
        # Find first data line — it follows the blank line after DOMAIN_MAX.
        lines = path.read_text(encoding="utf-8").splitlines()
        first_data = lines[6]
        for token in first_data.split():
            # Must be a plain decimal (digits + optional '.' + digits); no 'e'.
            assert "e" not in token.lower(), (
                f"unexpected scientific notation: {token!r}"
            )
            # Six decimals after the dot.
            if "." in token:
                _, frac = token.split(".")
                assert len(frac) == 6, f"expected 6 decimals, got {frac!r}"


class TestLumixRoundTrip:
    def test_write_then_read_returns_same_table(self, tmp_path):
        original = _make_lut(n=5)
        path = tmp_path / "rt.cube"
        get_format("lumix").write(original, path, photo_style_tag="VLOG")
        loaded = get_format("lumix").read(path)
        np.testing.assert_allclose(loaded.table, original.table, atol=1e-6)
        assert loaded.title == original.title

    def test_standard_cube_reader_handles_lumix_files(self, tmp_path):
        """A Lumix file is a valid Adobe-spec cube; the standard reader must accept it."""
        path = tmp_path / "compat.cube"
        get_format("lumix").write(_make_lut(n=5), path, photo_style_tag="VLOG")
        loaded = get_format("cube").read(path)
        assert loaded.resolution == 5
