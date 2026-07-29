"""Tests for the canonical LUT filename / title normalization.

The filename is the user-facing contract for what an exported bundle
looks like on disk. The normalization rules are:

- stock names: strip the brand prefix, fuse the first two remaining
  underscore-segments without delimiter
- version: ``0.3.2`` → ``v032``; PEP 440 dev / local suffixes stripped
- filename: ``lut_{version}_{film}_{print}.cube``
- title (inside the cube): ``{version}_{film}_{print}`` (compact)
"""

from __future__ import annotations

import pytest

from spektrafilm_lut_creator.naming import lut_filename, lut_title
from spektrafilm_lut_creator.naming import normalize_stock as _normalize_stock
from spektrafilm_lut_creator.naming import normalize_version as _normalize_version

from .factories import make_bundle_spec


def _canonical_lut_filename(spec, print_stock, version_tag):
    """Test helper — mirrors the old combined-1-LUT shape."""
    return lut_filename(
        film_profile=spec.film_profile,
        version_tag=version_tag,
        print_profile=print_stock,
        suffix=None,
    )


def _canonical_lut_title(spec, print_stock, version_tag):
    """Test helper — mirrors the old combined-1-LUT title shape."""
    return lut_title(
        film_profile=spec.film_profile,
        version_tag=version_tag,
        print_profile=print_stock,
        suffix=None,
    )


class TestNormalizeStock:
    @pytest.mark.parametrize(
        "stock,expected",
        [
            ("kodak_portra_400", "portra400"),
            ("kodak_supra_endura", "supraendura"),
            ("kodak_ultra_endura", "ultraendura"),
            ("kodak_portra_endura", "portraendura"),
            ("kodak_endura_premier", "endurapremier"),
            ("kodak_ektacolor_edge", "ektacoloredge"),
            ("kodak_2383", "2383"),
            ("kodak_vision3_500t", "vision3500t"),
            ("fujifilm_pro_400h", "pro400h"),
            ("fujifilm_c200", "c200"),
            ("fujifilm_crystal_archive_typeii", "crystalarchive"),
            ("fujifilm_velvia_100", "velvia100"),
            # Already short — fall-through.
            ("portra400", "portra400"),
        ],
    )
    def test_known_stocks(self, stock, expected):
        assert _normalize_stock(stock) == expected

    def test_unknown_brand_preserves_first_two_segments(self):
        # No brand prefix recognized → fuse first two segments anyway.
        assert _normalize_stock("acme_film_xyz") == "acmefilm"


class TestNormalizeVersion:
    @pytest.mark.parametrize(
        "version,expected",
        [
            ("0.3.2", "v032"),
            ("0.4.1", "v041"),
            ("1.0.0", "v100"),
            # PEP 440 dev suffix gets stripped before joining.
            ("0.3.2.dev0", "v032"),
            ("0.3.2+abc123", "v032"),
            ("0.3.2.dev0+abc123", "v032"),
        ],
    )
    def test_known_versions(self, version, expected):
        assert _normalize_version(version) == expected


class TestCanonicalFilename:
    def test_user_example(self):
        """Matches the user's stated reference example."""
        spec = make_bundle_spec(
            name="any",
            print_profiles=("kodak_supra_endura",),
            input_color_space="Panasonic V-Log",
            resolution=33,
        )
        filename = _canonical_lut_filename(spec, "kodak_supra_endura", "v032")
        assert filename == "lut_v032_portra400_supraendura.cube"

    def test_per_print_in_multi_bundle(self):
        spec = make_bundle_spec(
            name="any",
            print_profiles=("kodak_supra_endura", "fujifilm_crystal_archive_typeii"),
            input_color_space="Panasonic V-Log",
            resolution=33,
        )
        a = _canonical_lut_filename(spec, "kodak_supra_endura", "v032")
        b = _canonical_lut_filename(spec, "fujifilm_crystal_archive_typeii", "v032")
        assert a == "lut_v032_portra400_supraendura.cube"
        assert b == "lut_v032_portra400_crystalarchive.cube"
        assert a != b


class TestCanonicalTitle:
    def test_drops_color_spaces(self):
        spec = make_bundle_spec(
            name="any",
            print_profiles=("kodak_supra_endura",),
            input_color_space="Panasonic V-Log",
            resolution=33,
        )
        title = _canonical_lut_title(spec, "kodak_supra_endura", "v032")
        assert title == "v032_portra400_supraendura"
        assert "vlog" not in title
        assert "srgb" not in title
