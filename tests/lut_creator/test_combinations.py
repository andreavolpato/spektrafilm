"""Tests for the ``include_combinations`` feature (n130).

A combination cube is a contiguous sub-chain of the canonical LUTs
collapsed into a single cube. The builder enumerates the combinations
for each topology from ``_TOPOLOGY_COMBINATIONS`` and bakes them via
the unified ``_bake_sublut`` helper. These tests verify:

- The opt-in flag is False by default and emits nothing extra.
- Each multi-LUT topology produces the right *count* and naming.
- Combinations land in a ``combinations/`` subfolder.
- Each combination has a matching ``LutFileMeta`` with the right
  ``subchain_<ids>`` role, domain/range, and print field.
- Sub-chain cubes are physically equivalent to their canonical
  counterparts (4-LUT ``l1234`` == 1-LUT combined cube, byte-equal
  tables for the same spec).
- Canonical cubes are unchanged when ``include_combinations`` is on.
- The README and (when emitted) the OCIO config carry the combinations
  block / discoverability note.
"""

from __future__ import annotations

import numpy as np
import pytest

from spektrafilm_lut_creator.builders import (
    _COMBINATIONS_SUBDIR,
    _TOPOLOGY_COMBINATIONS,
    BundleBuilder,
)
from spektrafilm_lut_creator.bundles import BundleSpec

from .factories import make_bundle_spec

_RESOLUTION = 5
_INPUT_CS = "ACEScct"
_OUTPUT_CS = "sRGB"
_FILM = "kodak_portra_400"
_PRINT = "kodak_portra_endura"


def _spec(topology: str, *, include_combinations: bool, prints=(_PRINT,)) -> BundleSpec:
    return make_bundle_spec(
        name=f"combo_test_{topology}",
        print_profiles=prints,
        topology=topology,
        include_combinations=include_combinations,
    )


class TestSpecDefault:
    def test_include_combinations_defaults_to_false(self):
        spec = make_bundle_spec(name="x", topology="4lut")
        assert spec.include_combinations is False


class TestNoCombinationsByDefault:
    """With the flag off, no bundle gains extra cubes regardless of topology."""

    @pytest.mark.parametrize("topology", ["1lut", "2lut", "3lut", "4lut"])
    def test_no_subchain_entries_in_metadata(self, topology):
        bundle = BundleBuilder(_spec(topology, include_combinations=False)).build()
        for entry in bundle.meta.luts:
            assert not entry.role.startswith("subchain_"), (
                f"unexpected subchain entry in default {topology} bundle: {entry}"
            )

    @pytest.mark.parametrize("topology", ["1lut", "2lut", "3lut", "4lut"])
    def test_no_combinations_subdir_in_paths(self, topology):
        bundle = BundleBuilder(_spec(topology, include_combinations=False)).build()
        for rel_path, _ in bundle.luts:
            assert _COMBINATIONS_SUBDIR not in rel_path


class TestOneLutIsNoOp:
    """1-LUT bundles have nothing to collapse; the flag is documented
    as a no-op."""

    def test_one_lut_with_flag_produces_no_extra_cubes(self):
        bundle_off = BundleBuilder(_spec("1lut", include_combinations=False)).build()
        bundle_on = BundleBuilder(_spec("1lut", include_combinations=True)).build()
        assert len(bundle_on.luts) == len(bundle_off.luts)
        # Tables identical too (same spec, deterministic bake).
        for (path_off, lut_off), (path_on, lut_on) in zip(
            bundle_off.luts, bundle_on.luts
        ):
            assert path_off == path_on
            np.testing.assert_array_equal(lut_off.table, lut_on.table)


class TestTwoLutCombinations:
    """2-LUT topology adds exactly one combination: l12 = rgb_in → rgb_out
    (per-print, since the full chain depends on the print stock)."""

    @pytest.fixture(scope="class")
    def bundle(self):
        return BundleBuilder(_spec("2lut", include_combinations=True)).build()

    def test_one_extra_cube_per_print(self, bundle):
        # Canonical 2-LUT: 1 film + 1 print. Plus 1 combination (l12) per print.
        n_prints = 1
        n_canonical = 1 + n_prints
        n_combinations = 1 * n_prints
        assert len(bundle.luts) == n_canonical + n_combinations

    def test_subchain_role_is_l12(self, bundle):
        sub = [e for e in bundle.meta.luts if e.role.startswith("subchain_")]
        assert len(sub) == 1
        assert sub[0].role == "subchain_12"

    def test_subchain_path_under_combinations_folder(self, bundle):
        sub = [e for e in bundle.meta.luts if e.role.startswith("subchain_")][0]
        assert sub.path.startswith(_COMBINATIONS_SUBDIR + "/")
        assert sub.path.endswith("_l12.cube")

    def test_subchain_domain_range(self, bundle):
        sub = [e for e in bundle.meta.luts if e.role.startswith("subchain_")][0]
        assert sub.domain == "input_rgb"
        assert sub.range == "output_rgb"
        assert sub.print_profile == _PRINT


class TestThreeLutCombinations:
    """3-LUT adds l12 (shared), l23 (per-print), l123 (per-print)."""

    @pytest.fixture(scope="class")
    def bundle(self):
        return BundleBuilder(_spec("3lut", include_combinations=True)).build()

    def test_total_extra_cubes(self, bundle):
        # Canonical: 2 shared (L1, L2) + 1 per-print (L3) = 3 for 1 print.
        # Combinations: 1 shared (l12) + 2 per-print (l23, l123) = 3 for 1 print.
        n_prints = 1
        n_canonical = 2 + n_prints
        n_combinations = 1 + 2 * n_prints
        assert len(bundle.luts) == n_canonical + n_combinations

    def test_subchain_roles(self, bundle):
        roles = sorted(
            e.role for e in bundle.meta.luts if e.role.startswith("subchain_")
        )
        assert roles == ["subchain_12", "subchain_123", "subchain_23"]

    def test_l12_is_shared(self, bundle):
        l12 = next(e for e in bundle.meta.luts if e.role == "subchain_12")
        assert l12.print_profile is None
        # Shared cube path omits the print segment.
        assert "portraendura" not in l12.path

    def test_l23_and_l123_are_per_print(self, bundle):
        for role in ("subchain_23", "subchain_123"):
            entry = next(e for e in bundle.meta.luts if e.role == role)
            assert entry.print_profile == _PRINT
            assert "portraendura" in entry.path


class TestFourLutCombinations:
    """4-LUT adds 6 combinations: l12 (shared), l23/l34/l123/l234/l1234 (per-print)."""

    @pytest.fixture(scope="class")
    def bundle(self):
        return BundleBuilder(_spec("4lut", include_combinations=True)).build()

    def test_total_extra_cubes(self, bundle):
        # Canonical: 2 shared (L1, L2) + 2 per-print (L3, L4) = 4 for 1 print.
        # Combinations: 1 shared (l12) + 5 per-print (l23, l34, l123, l234, l1234) = 6 for 1 print.
        n_prints = 1
        n_canonical = 2 + 2 * n_prints
        n_combinations = 1 + 5 * n_prints
        assert len(bundle.luts) == n_canonical + n_combinations

    def test_subchain_roles(self, bundle):
        expected = {
            "subchain_12",
            "subchain_23",
            "subchain_34",
            "subchain_123",
            "subchain_234",
            "subchain_1234",
        }
        actual = {e.role for e in bundle.meta.luts if e.role.startswith("subchain_")}
        assert actual == expected

    def test_l12_is_shared_others_are_per_print(self, bundle):
        shared_roles = {"subchain_12"}
        for entry in bundle.meta.luts:
            if not entry.role.startswith("subchain_"):
                continue
            if entry.role in shared_roles:
                assert entry.print_profile is None, f"{entry.role} should be shared"
            else:
                assert entry.print_profile == _PRINT, (
                    f"{entry.role} should be per-print"
                )

    def test_all_subchain_cubes_clamped_to_unit(self, bundle):
        # Sub-chain LUTs must satisfy the same [0, 1] cube invariant as
        # the canonical ones.
        for rel_path, lut in bundle.luts:
            if _COMBINATIONS_SUBDIR not in rel_path:
                continue
            assert lut.table.shape == (_RESOLUTION, _RESOLUTION, _RESOLUTION, 3)
            assert lut.table.min() >= 0.0
            assert lut.table.max() <= 1.0


class TestCanonicalUnchangedWhenCombinationsOn:
    """Turning the flag on must not perturb the canonical cubes — only
    add new ones in ``combinations/``."""

    @pytest.mark.parametrize("topology", ["2lut", "3lut", "4lut"])
    def test_canonical_tables_byte_identical(self, topology):
        bundle_off = BundleBuilder(_spec(topology, include_combinations=False)).build()
        bundle_on = BundleBuilder(_spec(topology, include_combinations=True)).build()

        canonical_off = {p: lut for p, lut in bundle_off.luts}
        canonical_on = {
            p: lut for p, lut in bundle_on.luts if _COMBINATIONS_SUBDIR not in p
        }
        # Same canonical filenames.
        assert set(canonical_off.keys()) == set(canonical_on.keys())
        # Same canonical tables.
        for path in canonical_off:
            np.testing.assert_array_equal(
                canonical_off[path].table,
                canonical_on[path].table,
                err_msg=f"canonical cube {path} changed when combinations flag flipped",
            )

    @pytest.mark.parametrize("topology", ["1lut", "2lut", "3lut", "4lut"])
    def test_qa_effective_lut_unchanged_by_combinations(self, topology):
        """The QA suite composes the canonical L1..LN chain via
        ``_effective_lut``. With combinations on, the bundle list
        interleaves extra cubes; ``_effective_lut`` must still resolve
        to the same composed table by looking up cubes by role rather
        than by positional index.

        Regression test for the bug where 3-LUT + combinations caused
        QA to compose L1 ∘ L2 ∘ l12 (nonsense) instead of L1 ∘ L2 ∘ L3."""
        from spektrafilm_lut_creator.qa.suite import _effective_lut

        bundle_off = BundleBuilder(_spec(topology, include_combinations=False)).build()
        bundle_on = BundleBuilder(_spec(topology, include_combinations=True)).build()

        _, lut_off = _effective_lut(bundle_off, print_index=0)
        _, lut_on = _effective_lut(bundle_on, print_index=0)
        np.testing.assert_array_equal(
            lut_off.table,
            lut_on.table,
            err_msg=(
                f"{topology}: QA effective LUT differs when combinations "
                f"flag is flipped — _effective_lut is picking the wrong "
                f"cube from the interleaved bundle list"
            ),
        )


class TestL1234EqualsOneLutCombined:
    """The 4-LUT bundle's ``l1234`` sub-chain is mathematically the same
    transform as a 1-LUT bundle's combined cube for the same (film,
    print, resolution, color spaces). Both go through the same
    ``_bake_sublut(inject='rgb_in', collect='rgb_out')`` against a
    per-print pipeline, so the cube tables must be byte-identical."""

    def test_byte_identical_to_1lut_combined(self):
        spec_4lut = _spec("4lut", include_combinations=True)
        spec_1lut = _spec("1lut", include_combinations=False)
        bundle_4lut = BundleBuilder(spec_4lut).build()
        bundle_1lut = BundleBuilder(spec_1lut).build()

        l1234 = next(
            lut for path, lut in bundle_4lut.luts if path.endswith("_l1234.cube")
        )
        combined = bundle_1lut.luts[0][1]
        np.testing.assert_array_equal(l1234.table, combined.table)


class TestL12MatchesTwoLutFilmInThreeAndFourLut:
    """The shared l12 (rgb_in → cmy_film) is print-independent and the
    same math whether the bundle is 3-LUT or 4-LUT. Both should produce
    byte-identical tables to the 2-LUT bundle's film LUT for the same
    spec."""

    def test_4lut_l12_equals_2lut_film(self):
        bundle_4lut = BundleBuilder(_spec("4lut", include_combinations=True)).build()
        bundle_2lut = BundleBuilder(_spec("2lut", include_combinations=False)).build()

        l12 = next(lut for path, lut in bundle_4lut.luts if path.endswith("_l12.cube"))
        film = next(
            lut for path, lut in bundle_2lut.luts if path.endswith("_film.cube")
        )
        np.testing.assert_array_equal(l12.table, film.table)


class TestMultiPrintCombinations:
    """Per-print combinations multiply by print count; shared ones don't."""

    _PRINTS = ("kodak_portra_endura", "fujifilm_crystal_archive_typeii")

    def test_4lut_two_prints_combination_count(self):
        spec = _spec("4lut", include_combinations=True, prints=self._PRINTS)
        bundle = BundleBuilder(spec).build()
        # 2 shared canonical + 2*2 per-print canonical = 6 canonical.
        # 1 shared combination + 5 per-print * 2 prints = 11 combinations.
        sub_count = sum(1 for e in bundle.meta.luts if e.role.startswith("subchain_"))
        assert sub_count == 11
        # Total cubes = 6 + 11 = 17.
        assert len(bundle.luts) == 17

    def test_shared_l12_not_duplicated_across_prints(self):
        spec = _spec("4lut", include_combinations=True, prints=self._PRINTS)
        bundle = BundleBuilder(spec).build()
        l12_entries = [e for e in bundle.meta.luts if e.role == "subchain_12"]
        assert len(l12_entries) == 1
        assert l12_entries[0].print_profile is None


class TestReadmeContainsCombinationsSection:
    def test_readme_lists_subchain_cubes(self):
        from spektrafilm_lut_creator.builders import _bundle_readme_text

        bundle = BundleBuilder(_spec("4lut", include_combinations=True)).build()
        text = _bundle_readme_text(bundle.meta)
        assert "## Pre-collapsed sub-chains" in text
        # Each role's label should appear in the table.
        for role in (
            "subchain_12",
            "subchain_23",
            "subchain_34",
            "subchain_123",
            "subchain_234",
            "subchain_1234",
        ):
            label = "l" + role[len("subchain_") :]
            assert f"| {label} " in text, f"README missing row for {label}"

    def test_readme_omits_section_when_no_combinations(self):
        from spektrafilm_lut_creator.builders import _bundle_readme_text

        bundle = BundleBuilder(_spec("4lut", include_combinations=False)).build()
        text = _bundle_readme_text(bundle.meta)
        assert "## Pre-collapsed sub-chains" not in text


class TestOcioDescriptionMentionsCombinations:
    """When both ``ocio_config`` and ``include_combinations`` are on,
    the emitted OCIO config's top-level description nudges OCIO users
    toward the ``combinations/`` folder (discoverability only — the
    config doesn't reference those cubes)."""

    def test_description_includes_combinations_note(self):
        from spektrafilm_lut_creator import ocio_emit

        spec = BundleSpec(
            name="combo_ocio_test",
            film_profile=_FILM,
            print_profiles=(_PRINT,),
            input_color_space="Panasonic V-Log",
            output_color_space=_OUTPUT_CS,
            topology="1lut",
            resolution=_RESOLUTION,
            ocio_config=True,
            include_combinations=True,
        )
        bundle = BundleBuilder(spec).build()
        yaml_text = ocio_emit.emit_ocio_config(bundle, spec)
        assert "combinations/" in yaml_text

    def test_description_omits_note_when_combinations_off(self):
        from spektrafilm_lut_creator import ocio_emit

        spec = BundleSpec(
            name="no_combo_ocio_test",
            film_profile=_FILM,
            print_profiles=(_PRINT,),
            input_color_space="Panasonic V-Log",
            output_color_space=_OUTPUT_CS,
            topology="1lut",
            resolution=_RESOLUTION,
            ocio_config=True,
            include_combinations=False,
        )
        bundle = BundleBuilder(spec).build()
        yaml_text = ocio_emit.emit_ocio_config(bundle, spec)
        assert "combinations/" not in yaml_text


class TestTopologyCombinationTable:
    """Sanity checks on the static enumeration table."""

    def test_one_lut_has_no_combinations(self):
        assert _TOPOLOGY_COMBINATIONS["1lut"] == ()

    def test_two_lut_has_one_combination(self):
        assert len(_TOPOLOGY_COMBINATIONS["2lut"]) == 1

    def test_three_lut_has_three_combinations(self):
        assert len(_TOPOLOGY_COMBINATIONS["3lut"]) == 3

    def test_four_lut_has_six_combinations(self):
        assert len(_TOPOLOGY_COMBINATIONS["4lut"]) == 6

    def test_subchain_labels_are_contiguous(self):
        """Every sub-chain's stage_ids must be a contiguous run starting
        somewhere in {1, 2, 3, 4} and ending later in the same set.
        Non-contiguous combinations (e.g., l13) are physically
        meaningless and must not appear."""
        for topology, entries in _TOPOLOGY_COMBINATIONS.items():
            for entry in entries:
                ids = entry.stage_ids
                assert len(ids) >= 2, f"{topology}: {ids} should span at least 2 stages"
                # Contiguity.
                for a, b in zip(ids, ids[1:]):
                    assert b == a + 1, f"{topology}: {ids} is not contiguous"

    def test_shared_iff_collect_tap_is_filming_side(self):
        """A sub-chain is print-independent iff its collect tap is at or
        before ``cmy_film`` (i.e., it stays in the filming half of the
        pipeline)."""
        filming_taps = {"log_e_film", "cmy_film"}
        for topology, entries in _TOPOLOGY_COMBINATIONS.items():
            for entry in entries:
                expected_shared = entry.collect_tap in filming_taps
                assert entry.is_shared == expected_shared, (
                    f"{topology} {entry.label}: is_shared={entry.is_shared} "
                    f"but collect_tap={entry.collect_tap}"
                )
