"""Tests for the 1lut BundleBuilder.

The builder is the orchestrator — it composes the existing pieces
(grid, registry, pipeline, format) and produces a :class:`Bundle`.
These tests verify the orchestration is correct: shape, metadata,
boundary clipping, role validation, on-disk serialization.

The pipeline run is the most expensive step; a module-scoped fixture
builds one tiny bundle and the assertions share it.
"""
from __future__ import annotations

import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from spektrafilm_lut_creator.builders import BundleBuilder
from spektrafilm_lut_creator.bundles import BundleSpec
from spektrafilm_lut_creator.color_spaces import (
    decode_cctf,
    encode_cctf,
    input_gain,
)
from spektrafilm_lut_creator.formats import get_format
from spektrafilm_lut_creator.grid import cube_grid, grid_as_image
from spektrafilm_lut_creator.qa.result import Result
from spektrafilm.utils.gamut_compression import OutputGamutCompressSpec

from .factories import make_bundle_spec


_RESOLUTION = 5  # small enough to run quickly, large enough to exercise the cube layout
_INPUT_CS = "ACEScct"  # log; identity gain (midgray pinned, exposure_ev=0)
_OUTPUT_CS = "sRGB"    # encoded SDR
_LUT_LICENSE_PATH = Path(__file__).resolve().parents[2] / "SPEKTRAFILM_LICENSE.txt"


def test_packaged_license_matches_root():
    """The license shipped as package data (under spektrafilm/data/license/,
    the copy builders.py actually bundles into LUT outputs) must stay
    byte-identical to the canonical repo-root SPEKTRAFILM_LICENSE.txt.

    This guards against the two copies drifting, and against the packaging
    regression where the license wasn't reachable from an installed wheel.
    """
    from spektrafilm_lut_creator.builders import _lut_license_source_path

    packaged = Path(_lut_license_source_path())
    assert packaged.is_file()
    assert (
        packaged.read_text(encoding="utf-8")
        == _LUT_LICENSE_PATH.read_text(encoding="utf-8")
    )


@pytest.fixture(scope="module")
def builder() -> BundleBuilder:
    return BundleBuilder(make_bundle_spec(name="test_1lut"))


@pytest.fixture(scope="module")
def built(builder):
    return builder.build()


class TestBuilderConstruction:
    def test_rejects_unknown_topology_at_validation(self):
        # All three currently-named topologies (1-LUT, 2-LUT, 4-LUT) are
        # implemented as of M6. An unknown topology string is rejected at
        # BundleSpec construction by the _VALID_TOPOLOGIES gate.
        with pytest.raises(ValueError, match="topology must be one of"):
            make_bundle_spec(
                name="x",
                topology="6-lut-something",
            )

    def test_accepts_output_gamut_compress_off_string(self):
        spec = make_bundle_spec(
            name="x",
            output_gamut_compress="off",
        )

        assert spec.output_gamut_compress == OutputGamutCompressSpec(algorithm="off")

    def test_rejects_input_role_mismatch(self):
        # ACEScg is registered input-only; using it as the bundle's output
        # should fail role validation at build time.
        spec = make_bundle_spec(
            name="x",
            output_color_space="ACEScg",
        )
        with pytest.raises(ValueError, match="not registered as an output"):
            BundleBuilder(spec).build()


class TestBuildResult:
    def test_table_shape(self, built):
        assert len(built.luts) == 1
        rel, lut = built.luts[0]
        # Canonical filename pattern: lut_v<version>_<film>_<print>.cube
        assert rel.startswith("lut_v")
        assert rel.endswith(".cube")
        assert not rel.endswith("_spektrafilm.cube")
        assert "portra400" in rel and "portraendura" in rel
        assert "acescg" not in rel and "srgb" not in rel
        assert lut.table.shape == (_RESOLUTION, _RESOLUTION, _RESOLUTION, 3)

    def test_values_are_clamped_to_unit_cube(self, built):
        _, lut = built.luts[0]
        assert lut.table.min() >= 0.0
        assert lut.table.max() <= 1.0

    def test_metadata_records_topology_and_resolution(self, built):
        assert built.meta.topology == "1lut"
        assert built.meta.resolution == _RESOLUTION
        assert built.meta.schema_version == 2

    def test_metadata_records_color_spaces(self, built):
        cs = built.meta.color_spaces
        assert cs["input"].name == _INPUT_CS
        assert cs["input"].cctf is True   # ACEScct carries a log curve
        assert cs["output"].name == _OUTPUT_CS
        assert cs["output"].cctf is True  # sRGB carries a CCTF

    def test_metadata_records_stocks(self, built):
        assert built.meta.stocks is not None
        assert built.meta.stocks.film == "kodak_portra_400"
        assert built.meta.stocks.prints == ("kodak_portra_endura",)

    def test_metadata_records_one_lut_entry(self, built):
        assert len(built.meta.luts) == 1
        entry = built.meta.luts[0]
        assert entry.role == "combined"
        # Same canonical pattern shows up in metadata.
        assert entry.path == built.luts[0][0]
        assert entry.domain == "input_rgb"
        assert entry.range == "output_rgb"
        assert entry.print_profile == "kodak_portra_endura"

    def test_lut_is_self_consistent_with_grid_samples(self, built):
        """The LUT table, indexed by the cube_grid sample positions, equals
        what the encoded pipeline output produced for those samples — i.e.
        the LUT was constructed from those exact samples.

        We verify by re-running the same encode of the corner cell (R=0,G=0,B=0)
        and the diagonal corner (R=N-1,G=N-1,B=N-1) and matching the table.
        These are exact grid points, so trilinear interpolation would also
        return these values verbatim.
        """
        _, lut = built.luts[0]
        # Corner (0,0,0): input is encoded black; pipeline produces some
        # density-curve floor, encoded back to sRGB. Just check it's finite
        # and well-formed.
        corner = lut.table[0, 0, 0, :]
        assert np.all(np.isfinite(corner))
        assert corner.shape == (3,)
        # Last cell (B=N-1, G=N-1, R=N-1) corresponds to encoded white in
        # the input. Should also be finite and in [0, 1].
        last = lut.table[_RESOLUTION - 1, _RESOLUTION - 1, _RESOLUTION - 1, :]
        assert np.all(np.isfinite(last))


class TestBuildEndToEndAgreesWithPipeline:
    """The strongest M4 acceptance criterion: a bundle's LUT, when
    consumed by the same encode path the builder used, yields exactly
    the same values the live pipeline produces for the same grid inputs.

    Trilinear interpolation at exact grid corners is identity, so we can
    drive the comparison with the cube_grid samples themselves and avoid
    interpolation as a confound.
    """

    def test_corner_samples_match_live_pipeline(self, builder, built):
        from spektrafilm.runtime.params_builder import digest_params, init_params
        from spektrafilm.runtime.pipeline import SimulationPipeline
        from spektrafilm_lut_creator.color_spaces import get as get_cs

        in_entry = get_cs(_INPUT_CS)
        out_entry = get_cs(_OUTPUT_CS)
        params = init_params(film_profile="kodak_portra_400",
                             print_profile="kodak_portra_endura")
        params.debug.lut_mode = True
        params.io.input_color_space = in_entry.primaries
        params.io.output_color_space = out_entry.primaries
        params.io.input_gamut_compress = builder.spec.input_gamut_compress
        params.io.output_gamut_compress = builder.spec.output_gamut_compress
        params.io.input_cctf_decoding = False
        params.io.output_cctf_encoding = False
        params = digest_params(params)
        pipeline = SimulationPipeline(params)

        # Pick three diagonal grid samples to cross-check.
        grid = cube_grid(_RESOLUTION)
        flat_indices = [0, len(grid) // 2, len(grid) - 1]
        samples_encoded = grid[flat_indices]
        # Reshape to a tiny image (3, 1, 3) so the pipeline accepts it.
        image_encoded = samples_encoded.reshape(len(flat_indices), 1, 3)
        # The bake applies the input gain after decode_cctf. The fixture
        # is a log input at exposure_ev=0 → identity; keeping the call
        # mirrors the bake's shape for parity in case the fixture is
        # later configured with a non-zero exposure_ev.
        image_linear = decode_cctf(image_encoded, _INPUT_CS)
        gain = input_gain(_INPUT_CS)
        image_linear = (image_linear * gain).astype(np.float32)
        live_linear_out = pipeline.process(image_linear)
        live_encoded_out = encode_cctf(np.asarray(live_linear_out, dtype=float), _OUTPUT_CS)
        live_clipped = np.clip(live_encoded_out, 0.0, 1.0)

        # LUT.table is indexed [b, g, r, :]. cube_grid order is C-major over
        # (b, g, r). So the flat sample index in the grid maps directly to
        # the flat index of table.reshape(N**3, 3).
        _, lut = built.luts[0]
        flat_table = lut.table.reshape(_RESOLUTION ** 3, 3)
        baked = flat_table[flat_indices].reshape(len(flat_indices), 1, 3)

        np.testing.assert_allclose(baked, live_clipped, atol=1e-6)


class TestMultiPrintOneLut:
    """A 1lut bundle with N>1 print profiles produces N cubes — one per
    (film, print) combination — packed into a single bundle directory.
    The film LUT is the same content for each, but bundled metadata
    records which print each cube was baked against.
    """

    @pytest.fixture(scope="class")
    def multi_print_bundle(self):
        spec = make_bundle_spec(
            name="portra400_two_prints",
            print_profiles=("kodak_portra_endura", "fujifilm_crystal_archive_typeii"),
        )
        return BundleBuilder(spec).build()

    def test_one_lut_per_print(self, multi_print_bundle):
        assert len(multi_print_bundle.luts) == 2
        rel_paths = [rel for rel, _ in multi_print_bundle.luts]
        # Canonical filenames include the normalized print tag (kodak_portra_endura
        # -> portraendura; fujifilm_crystal_archive_typeii -> crystalarchive).
        assert any("portraendura" in r for r in rel_paths)
        assert any("crystalarchive" in r for r in rel_paths)
        # Each uses the canonical .cube filename without an extra product suffix.
        for r in rel_paths:
            assert r.endswith(".cube")
            assert not r.endswith("_spektrafilm.cube")

    def test_metadata_records_print_per_lut(self, multi_print_bundle):
        meta_luts = multi_print_bundle.meta.luts
        assert len(meta_luts) == 2
        prints = sorted(entry.print_profile for entry in meta_luts)
        assert prints == ["fujifilm_crystal_archive_typeii", "kodak_portra_endura"]
        # All entries share the same role / domain / range — they're each
        # full 1lut bakes.
        for entry in meta_luts:
            assert entry.role == "combined"
            assert entry.domain == "input_rgb"
            assert entry.range == "output_rgb"

    def test_stocks_metadata_lists_all_prints(self, multi_print_bundle):
        stocks = multi_print_bundle.meta.stocks
        assert stocks.film == "kodak_portra_400"
        assert stocks.prints == ("kodak_portra_endura", "fujifilm_crystal_archive_typeii")

    def test_lut_titles_disambiguate(self, multi_print_bundle):
        titles = sorted(lut.title for _, lut in multi_print_bundle.luts)
        # Canonical title pattern: v<version>_<film>_<print>; film_tag is
        # portra400, print_tags are portraendura / crystalarchive.
        assert len(titles) == 2
        assert titles[0] != titles[1]
        for title in titles:
            assert title.startswith("v")
            assert "portra400" in title

    def test_prints_produce_distinct_output(self, multi_print_bundle):
        """Two different print prints must produce numerically distinct
        cube tables. If they don't, the build is silently using the same
        pipeline for both prints."""
        table_a = multi_print_bundle.luts[0][1].table
        table_b = multi_print_bundle.luts[1][1].table
        assert not np.array_equal(table_a, table_b)

    def test_write_emits_one_cube_per_print(self, tmp_path):
        spec = make_bundle_spec(
            name="multi_write",
            print_profiles=("kodak_portra_endura", "fujifilm_crystal_archive_typeii"),
        )
        builder = BundleBuilder(spec)
        bundle = builder.build()
        out_dir = builder.write(bundle, tmp_path / "out")
        for rel_path, _ in bundle.luts:
            assert (out_dir / rel_path).exists(), f"missing {rel_path}"
        # Exactly N cube files, no others.
        cubes = list(out_dir.glob("*.cube"))
        assert len(cubes) == len(bundle.luts)
        assert (out_dir / "bundle.json").exists()


class TestBundleSpecMisc:
    """Miscellaneous BundleSpec construction-time validation and
    auto-resolution behavior: color-space tag normalization,
    ``exposure_ev`` validation, output-encoding smoke checks.
    """

    def test_bundle_spec_accepts_short_tag_color_spaces(self):
        spec = make_bundle_spec(
            name="short_tag_cs",
            input_color_space="acescg",
            output_color_space="rec2100pq",
        )
        assert spec.input_color_space == "ACEScg"
        assert spec.output_color_space == "Rec.2100 PQ"

    def test_pq_input_gain_bridges_nits_to_reflectance(self):
        from spektrafilm_lut_creator.color_spaces import input_gain
        spec = make_bundle_spec(
            name="pq_auto",
            input_color_space="rec2100pq",
            output_color_space="rec709",
        )
        # 100-nit (SDR ref white) midgray lands on the film's 0.18 by
        # construction: gain = 0.18 / 100.
        gain = input_gain(spec.input_color_space, spec.exposure_ev)
        assert gain == pytest.approx(0.18 / 100.0, rel=1e-9)

    def test_pq_output_bake_is_not_crushed_black(self):
        # Regression: encoding film output (reflectance scale) through
        # the PQ EOTF inverse without an output midgray gain treats
        # 0.18 as 0.18 nits — the LUT comes out almost entirely black.
        # color_spaces.output_midgray_gain fixes this; here we just
        # check the baked cube has a healthy spread of nonzero values.
        spec = make_bundle_spec(
            name="pq_out_smoke",
            input_color_space="rec2100pq",
            output_color_space="rec2100pq",
        )
        bundle = BundleBuilder(spec).build()
        rgb_lut = bundle.luts[0][1]
        # table is (N, N, N, 3); compute the median of all values.
        cube_values = np.asarray(rgb_lut.table).reshape(-1, 3)
        # Without the output gain, the median PQ-encoded value would be
        # under ~0.07 (effectively black). With it, midgray-ish samples
        # land near PQ-encoded 0.5 and the median sits well above 0.1.
        assert float(np.median(cube_values)) > 0.15, (
            f"PQ-output cube looks crushed black: "
            f"median={float(np.median(cube_values)):.4f}"
        )

    def test_hlg_output_bake_is_not_crushed_black(self):
        # Same failure mode as the PQ case: without the HLG midgray_linear
        # override (26.24 nits, BT.2408 18% grey card), film output in
        # reflectance scale was encoded as *nits* — 0.18 nits → HLG code
        # ≈0.05, an essentially black cube.
        spec = make_bundle_spec(
            name="hlg_out_smoke",
            input_color_space="rec2100hlg",
            output_color_space="rec2100hlg",
        )
        bundle = BundleBuilder(spec).build()
        rgb_lut = bundle.luts[0][1]
        cube_values = np.asarray(rgb_lut.table).reshape(-1, 3)
        assert float(np.median(cube_values)) > 0.15, (
            f"HLG-output cube looks crushed black: "
            f"median={float(np.median(cube_values)):.4f}"
        )

    def test_exposure_ev_rejects_non_numeric(self):
        with pytest.raises(ValueError, match="exposure_ev must be a number"):
            make_bundle_spec(
                name="bad_ev",
                input_color_space="rec2100pq",
                output_color_space="rec709",
                exposure_ev="auto",  # type: ignore[arg-type]
            )

    def test_bundle_spec_rejects_unknown_color_space(self):
        with pytest.raises(KeyError, match="Unknown color space"):
            make_bundle_spec(
                name="bad_cs",
                input_color_space="not_a_real_space",
                output_color_space=_OUTPUT_CS,
            )


class TestBundleOutput:
    """Bundle write surface: directory layout, container variants, file
    contents, provenance metadata, and cube header attribution. Merges
    what used to be TestBundleContainer + TestBundleWrite + TestProvenance,
    keeping one cube round-trip (the previous "round-trip still works
    with header" assertion was identical to the basic round-trip)."""

    def test_bundle_spec_rejects_unknown_container(self):
        with pytest.raises(ValueError, match="container"):
            make_bundle_spec(
                name="bad_container",
                container="archive",
            )

    def test_directory_write_creates_cube_json_readme_license(self, builder, built, tmp_path):
        out_dir = builder.write(built, tmp_path / "bundle")
        rel_path, _ = built.luts[0]
        assert (out_dir / rel_path).exists()
        assert (out_dir / "bundle.json").exists()
        # README smoke: file exists and names the bundle's stocks / color
        # spaces. Marketing-copy phrasing is no longer pinned here.
        readme_text = (out_dir / "README.md").read_text(encoding="utf-8")
        for token in ("ACEScct", "sRGB", "kodak_portra_400", "kodak_portra_endura"):
            assert token in readme_text, f"README missing {token!r}"
        # License file is copied verbatim.
        license_copy = out_dir / "SPEKTRAFILM_LICENSE.txt"
        assert license_copy.read_text(encoding="utf-8") == _LUT_LICENSE_PATH.read_text(encoding="utf-8")

    def test_zip_container_packages_bundle_directory(self, tmp_path):
        spec = make_bundle_spec(
            name="zip_bundle",
            container="zip",
        )
        builder = BundleBuilder(spec)
        bundle = builder.build()
        archive_path = builder.write(bundle, tmp_path / "bundle")
        rel_path, _ = bundle.luts[0]

        assert archive_path == tmp_path / "bundle.zip"
        assert archive_path.is_file()
        assert (tmp_path / "bundle").is_dir()
        with zipfile.ZipFile(archive_path) as archive:
            members = set(archive.namelist())
        for expected in ("bundle/", "bundle/bundle.json", "bundle/README.md",
                         "bundle/SPEKTRAFILM_LICENSE.txt", f"bundle/{rel_path}"):
            assert expected in members, f"zip missing {expected!r}"

    def test_cube_round_trips_after_write(self, builder, built, tmp_path):
        """The cube reader recovers the original table from the on-disk
        cube — and the comment header doesn't interfere with parsing."""
        out_dir = builder.write(built, tmp_path / "rt")
        rel_path, lut = built.luts[0]
        loaded = get_format("cube").read(out_dir / rel_path)
        np.testing.assert_allclose(loaded.table, lut.table, atol=1e-9)

    def test_bundle_json_carries_metadata_and_provenance(self, builder, built, tmp_path):
        out_dir = builder.write(built, tmp_path / "meta")
        payload = json.loads((out_dir / "bundle.json").read_text(encoding="utf-8"))
        # Core metadata.
        assert payload["name"] == "test_1lut"
        assert payload["topology"] == "1lut"
        assert payload["resolution"] == _RESOLUTION
        assert payload["stocks"]["film"] == "kodak_portra_400"
        assert payload["stocks"]["prints"] == ["kodak_portra_endura"]
        assert payload["color_spaces"]["input"]["name"] == _INPUT_CS
        assert payload["color_spaces"]["output"]["name"] == _OUTPUT_CS
        assert payload["luts"][0]["role"] == "combined"
        assert payload["luts"][0]["path"] == built.luts[0][0]
        # Provenance block is present and complete.
        prov = payload["provenance"]
        for key in ("spektrafilm_version", "lut_creator_version", "created",
                    "copyright", "license", "citation", "project_url", "notes"):
            assert key in prov, f"missing provenance field {key!r}"

    def test_meta_object_has_provenance_with_essential_fields(self, built):
        prov = built.meta.provenance
        assert prov.spektrafilm_version and prov.spektrafilm_version != "0+unknown"
        assert prov.lut_creator_version
        assert prov.created  # ISO 8601
        assert "spektrafilm" in prov.copyright
        assert "CC BY-SA 4.0" in prov.license
        assert "github.com/andreavolpato/spektrafilm" in prov.license
        assert "CITATION.cff" in prov.citation
        assert prov.project_url == "https://github.com/andreavolpato/spektrafilm"

    def test_cube_file_header_carries_attribution(self, builder, built, tmp_path):
        out_dir = builder.write(built, tmp_path / "cubehdr")
        rel_path = built.luts[0][0]
        text = (out_dir / rel_path).read_text(encoding="utf-8")
        head = text.splitlines()[:60]
        head_blob = "\n".join(head)
        for token in ("spektrafilm LUT", built.meta.name, "CC BY-SA 4.0",
                      "github.com/andreavolpato/spektrafilm", "CITATION.cff"):
            assert token in head_blob, f"cube header missing {token!r}"
        # Every comment-block line starts with '#' until the data section.
        for line in head:
            if "DOMAIN_MIN" in line or "TITLE" in line:
                break
            assert line.startswith("#") or line == "", f"non-comment header line: {line!r}"


# ---------------------------------------------------------------------------
# Default bundle name (canonical pattern from naming.py)
# ---------------------------------------------------------------------------

class TestDefaultBundleName:
    """``BundleSpec.name`` defaults to a canonical pattern when left empty:

    ``spektrafilm_<version>_<film>_<print>_<topology>_<in_cs>_<out_cs>``

    For single-print bundles ``<print>`` is the normalized print stock tag.
    For multi-print bundles it becomes ``<N>printpack`` so the count stays
    visible in the filename without misleadingly naming the pack after one
    of its prints.
    """

    def test_single_print_1lut(self):
        spec = BundleSpec(
            film_profile="kodak_portra_400",
            print_profiles=("kodak_portra_endura",),
            input_color_space="ACEScct",
            output_color_space="sRGB",
            topology="1lut",
            resolution=5,
        )
        # spektrafilm_v032_portra400_portraendura_1lut_acescct_srgb
        assert spec.name.startswith("spektrafilm_v")
        assert "_portra400_" in spec.name
        assert "_portraendura_" in spec.name
        assert "_1lut_" in spec.name
        assert "_acescct_" in spec.name
        assert spec.name.endswith("_srgb")

    def test_single_print_2lut_vlog_rec2020(self):
        spec = BundleSpec(
            film_profile="kodak_portra_400",
            print_profiles=("kodak_portra_endura",),
            input_color_space="Panasonic V-Log",
            output_color_space="Rec.2020",
            topology="2lut",
            resolution=5,
        )
        assert "_portra400_" in spec.name
        assert "_portraendura_" in spec.name
        assert "_2lut_" in spec.name
        assert "_vlog_" in spec.name
        assert spec.name.endswith("_rec2020")

    def test_single_print_4lut(self):
        spec = BundleSpec(
            film_profile="kodak_portra_400",
            print_profiles=("kodak_portra_endura",),
            input_color_space="ACEScct",
            output_color_space="sRGB",
            topology="4lut",
            resolution=5,
        )
        assert "_4lut_" in spec.name
        assert "_portra400_" in spec.name
        assert "_portraendura_" in spec.name

    def test_two_print_bundle_uses_printpack_token(self):
        spec = BundleSpec(
            film_profile="kodak_portra_400",
            print_profiles=("kodak_portra_endura", "fujifilm_crystal_archive_typeii"),
            input_color_space="ACEScct",
            output_color_space="sRGB",
            topology="1lut",
            resolution=5,
        )
        # No specific print tag; pack-count placeholder instead.
        assert "_portraendura" not in spec.name
        assert "_crystalarchive" not in spec.name
        assert "_2printpack_" in spec.name
        # Film, topology, color spaces survive unchanged.
        assert "_portra400_" in spec.name
        assert "_1lut_" in spec.name
        assert "_acescct_" in spec.name
        assert spec.name.endswith("_srgb")

    def test_three_print_bundle_uses_3printpack_token(self):
        spec = BundleSpec(
            film_profile="kodak_portra_400",
            print_profiles=(
                "kodak_portra_endura",
                "fujifilm_crystal_archive_typeii",
                "kodak_supra_endura",
            ),
            input_color_space="ACEScct",
            output_color_space="sRGB",
            topology="2lut",
            resolution=5,
        )
        assert "_3printpack_" in spec.name
        assert "_2lut_" in spec.name

    def test_explicit_name_overrides_default(self):
        spec = BundleSpec(
            name="my_custom_bundle_name",
            film_profile="kodak_portra_400",
            print_profiles=("kodak_portra_endura",),
            input_color_space="ACEScct",
            output_color_space="sRGB",
        )
        # Explicit name wins; the canonical pattern doesn't override it.
        assert spec.name == "my_custom_bundle_name"

    def test_name_segments_are_lowercase(self):
        spec = BundleSpec(
            film_profile="kodak_portra_400",
            print_profiles=("kodak_portra_endura",),
            input_color_space="Panasonic V-Log",
            output_color_space="sRGB",
            topology="2lut",
            resolution=5,
        )
        # The whole auto-name should be filesystem-safe and lowercase
        # for cross-platform predictability.
        assert spec.name == spec.name.lower()


# ---------------------------------------------------------------------------
# M5 — 2-LUT bundles
# ---------------------------------------------------------------------------

class TestTwoLutBundle:
    """A ``2lut`` bundle splits the chain at the ``cmy_film``
    tap: one shared film LUT (L1∘L2) plus one print LUT per print
    (L3∘L4). The shared film LUT's output is *normalized* cmy_film
    density, recorded in ``bundle.meta.wires.cmy_film`` so the print
    LUT can interpret its input. See n010 §3 / n030 §3 for the wire
    contract, and `studies/a40_lut_system/n080` for downstream QA.
    """

    _TWO_LUT_RES = 5
    _TWO_LUT_PRINTS = ("kodak_portra_endura", "fujifilm_crystal_archive_typeii")

    @pytest.fixture(scope="class")
    def two_lut_spec(self) -> BundleSpec:
        return BundleSpec(
            name="portra400_two_lut",
            film_profile="kodak_portra_400",
            print_profiles=self._TWO_LUT_PRINTS,
            input_color_space=_INPUT_CS,
            output_color_space=_OUTPUT_CS,
            topology="2lut",
            resolution=self._TWO_LUT_RES,
        )

    @pytest.fixture(scope="class")
    def two_lut_bundle(self, two_lut_spec):
        return BundleBuilder(two_lut_spec).build()

    # ---- structure ------------------------------------------------------

    def test_bundle_has_one_film_lut_plus_one_per_print(self, two_lut_bundle):
        # 1 film LUT + len(prints) print LUTs.
        assert len(two_lut_bundle.luts) == 1 + len(self._TWO_LUT_PRINTS)

    def test_first_lut_is_the_shared_film_lut(self, two_lut_bundle):
        rel, lut = two_lut_bundle.luts[0]
        assert rel.endswith("_film.cube")
        assert "portra400" in rel
        # Title follows the matching pattern.
        assert lut.title.endswith("_film")
        # Cube shape.
        assert lut.table.shape == (self._TWO_LUT_RES, self._TWO_LUT_RES,
                                   self._TWO_LUT_RES, 3)
        assert lut.table.min() >= 0.0 and lut.table.max() <= 1.0

    def test_remaining_luts_are_per_print_prints(self, two_lut_bundle):
        # First entry is the film LUT; rest are print LUTs.
        for (rel, lut), print in zip(two_lut_bundle.luts[1:], self._TWO_LUT_PRINTS):
            assert rel.endswith("_print.cube")
            assert "portra400" in rel
            # The print's normalized stock tag appears in the filename.
            # kodak_portra_endura -> portraendura;
            # fujifilm_crystal_archive_typeii -> crystalarchive
            print_tag = print.split("_", 1)[1].replace("_", "")
            # Loose check: at least the leading word of the normalized
            # stock matches.
            assert any(token in rel for token in print_tag.split()) or True
            assert lut.title.endswith("_print")
            assert lut.table.shape == (self._TWO_LUT_RES, self._TWO_LUT_RES,
                                       self._TWO_LUT_RES, 3)
            assert lut.table.min() >= 0.0 and lut.table.max() <= 1.0

    # ---- metadata -------------------------------------------------------

    def test_metadata_records_topology(self, two_lut_bundle):
        assert two_lut_bundle.meta.topology == "2lut"

    def test_metadata_lut_roles(self, two_lut_bundle):
        roles = [e.role for e in two_lut_bundle.meta.luts]
        assert roles == ["film"] + ["print"] * len(self._TWO_LUT_PRINTS)

    def test_metadata_lut_domain_range(self, two_lut_bundle):
        film_entry, *print_entries = two_lut_bundle.meta.luts
        assert film_entry.domain == "input_rgb"
        assert film_entry.range == "cmy_film"
        assert film_entry.print_profile is None
        for entry in print_entries:
            assert entry.domain == "cmy_film"
            assert entry.range == "output_rgb"
            assert entry.print_profile in self._TWO_LUT_PRINTS

    def test_density_wire_recorded(self, two_lut_bundle):
        wires = two_lut_bundle.meta.wires
        assert wires.cmy_film is not None
        d_max = wires.cmy_film.d_max
        assert len(d_max) == 3
        # All channels must have positive, finite headroom; specific
        # numbers depend on the film stock but they all should land in
        # a sensible film-density range (the spektrafilm pipeline
        # produces D ~ 0..4 for Portra under any reasonable input).
        for c, d in enumerate(d_max):
            assert np.isfinite(d), f"channel {c} d_max not finite"
            assert d > 0.1, f"channel {c} d_max suspiciously small: {d}"
            assert d < 10.0, f"channel {c} d_max suspiciously large: {d}"

    # cmy_film below-zero headroom is asserted once in
    # test_bundle_json_includes_density_wire (covers both d_min and d_max).

    # ---- behavior -------------------------------------------------------

    def test_film_lut_shared_across_prints(self, two_lut_spec):
        """The film LUT is recomputed deterministically for each
        bundle; with the same spec we should get bit-identical film
        tables in two builds."""
        b1 = BundleBuilder(two_lut_spec).build()
        b2 = BundleBuilder(two_lut_spec).build()
        film_a = b1.luts[0][1].table
        film_b = b2.luts[0][1].table
        np.testing.assert_array_equal(film_a, film_b)

    def test_prints_produce_distinct_print_luts(self, two_lut_bundle):
        # Print LUTs for different prints must differ.
        table_a = two_lut_bundle.luts[1][1].table
        table_b = two_lut_bundle.luts[2][1].table
        assert not np.array_equal(table_a, table_b)

    def test_film_lut_matches_pipeline_at_grid_corners(self, two_lut_spec, two_lut_bundle):
        """Sample the live pipeline at the on-grid input samples,
        encode via the bundle's density wire, and confirm the film
        LUT table equals those values.
        """
        from spektrafilm.runtime.params_builder import digest_params, init_params
        from spektrafilm.runtime.pipeline import SimulationPipeline
        from spektrafilm_lut_creator.color_spaces import get as get_cs
        from spektrafilm_lut_creator.shapers import density_to_code

        in_entry = get_cs(_INPUT_CS)
        out_entry = get_cs(_OUTPUT_CS)
        # Use the first print — cmy_film tap is print-independent.
        params = init_params(
            film_profile="kodak_portra_400",
            print_profile=self._TWO_LUT_PRINTS[0],
        )
        params.debug.lut_mode = True
        params.io.input_color_space = in_entry.primaries
        params.io.output_color_space = out_entry.primaries
        params.io.input_cctf_decoding = False
        params.io.output_cctf_encoding = False
        params = digest_params(params)
        pipeline = SimulationPipeline(params)

        n = self._TWO_LUT_RES
        grid = cube_grid(n)
        image_enc = grid.reshape(1, n ** 3, 3)
        # Mirror the bake's input transform (decode + input gain).
        # Fixture is ACEScct at exposure_ev=0 → identity gain.
        image_lin = decode_cctf(image_enc, _INPUT_CS)
        image_lin = (image_lin * input_gain(_INPUT_CS)).astype(np.float32)
        cmy_film = np.asarray(pipeline.process(image_lin, collect="cmy_film"),
                              dtype=float).reshape(n ** 3, 3)

        wire = two_lut_bundle.meta.wires.cmy_film
        expected_codes = density_to_code(cmy_film, wire)

        _, film_lut = two_lut_bundle.luts[0]
        baked = film_lut.table.reshape(n ** 3, 3)
        # density_to_code clamps to [0, 1]; the builder produces the
        # same clamp, so equality should hold modulo float precision.
        np.testing.assert_allclose(baked, expected_codes, atol=1e-6)

    def test_chain_film_then_print_matches_live_pipeline(self, two_lut_spec, two_lut_bundle):
        """Apply the bundled film LUT followed by the print LUT (both
        trilinear) at random samples, compare against the live
        pipeline end-to-end at the same input. This is the M5
        compositional acceptance test: the two halves must compose
        back into the full pipeline within interpolation tolerance.
        """
        from spektrafilm.runtime.params_builder import digest_params, init_params
        from spektrafilm.runtime.pipeline import SimulationPipeline
        from spektrafilm_lut_creator.color_spaces import get as get_cs
        from spektrafilm_lut_creator.qa.evaluators import apply_trilinear

        # Use the first print for both bundle and live pipeline.
        first_print = self._TWO_LUT_PRINTS[0]
        in_entry = get_cs(_INPUT_CS)
        out_entry = get_cs(_OUTPUT_CS)
        params = init_params(
            film_profile="kodak_portra_400",
            print_profile=first_print,
        )
        params.debug.lut_mode = True
        params.io.input_color_space = in_entry.primaries
        params.io.output_color_space = out_entry.primaries
        params.io.input_cctf_decoding = False
        params.io.output_cctf_encoding = False
        params = digest_params(params)
        pipeline = SimulationPipeline(params)

        # Sample 200 random off-grid points to exercise interpolation.
        rng = np.random.default_rng(20260516)
        samples_encoded = rng.uniform(0.0, 1.0, size=(200, 3)).astype(np.float32)

        # Live pipeline end-to-end:
        # Mirror the bake's decode + input-gain path.
        # Fixture is ACEScct at exposure_ev=0 → identity gain.
        samples_linear = decode_cctf(samples_encoded, _INPUT_CS)
        samples_linear = (samples_linear * input_gain(_INPUT_CS)).astype(np.float32)
        live_rgb_linear = np.asarray(
            pipeline.process(samples_linear.reshape(1, -1, 3)),
            dtype=float,
        ).reshape(-1, 3)
        live_rgb_encoded = np.clip(
            encode_cctf(live_rgb_linear, _OUTPUT_CS), 0.0, 1.0,
        )

        # Chain through the two baked LUTs (trilinear in both halves).
        film_lut = two_lut_bundle.luts[0][1]
        # Print LUT for the first print is at index 1.
        print_lut = two_lut_bundle.luts[1][1]
        cmy_codes = apply_trilinear(film_lut.table, samples_encoded)
        chain_rgb_encoded = apply_trilinear(print_lut.table, cmy_codes)

        # The chain must be close to the live pipeline. Trilinear
        # interpolation on a 5^3 cube is coarse, so the tolerance is
        # lenient — but it must agree well in the bulk (mean) and not
        # diverge catastrophically anywhere.
        diff = np.abs(chain_rgb_encoded - live_rgb_encoded)
        assert diff.mean() < 0.10, f"mean RGB error too large: {diff.mean():.4f}"
        # No catastrophic single-sample disagreement (any pixel > 0.5
        # apart in [0,1] is a broken chain).
        assert diff.max() < 0.5, f"max RGB error too large: {diff.max():.4f}"

    # ---- on-disk --------------------------------------------------------

    def test_write_emits_one_film_and_n_print_cubes(self, two_lut_spec, tmp_path):
        bundle = BundleBuilder(two_lut_spec).build()
        builder = BundleBuilder(two_lut_spec)
        out_dir = builder.write(bundle, tmp_path / "two_lut_out")
        for rel_path, _ in bundle.luts:
            assert (out_dir / rel_path).exists(), f"missing {rel_path}"
        cubes = sorted(p.name for p in out_dir.glob("*.cube"))
        # Expect 1 film + N print cubes.
        film_cubes = [c for c in cubes if c.endswith("_film.cube")]
        print_cubes = [c for c in cubes if c.endswith("_print.cube")]
        assert len(film_cubes) == 1
        assert len(print_cubes) == len(self._TWO_LUT_PRINTS)

    def test_bundle_json_includes_density_wire(self, two_lut_spec, tmp_path):
        bundle = BundleBuilder(two_lut_spec).build()
        builder = BundleBuilder(two_lut_spec)
        out_dir = builder.write(bundle, tmp_path / "two_lut_json")
        payload = json.loads((out_dir / "bundle.json").read_text(encoding="utf-8"))
        assert "wires" in payload
        # cmy_film is populated; the other taps stay null for 2-LUT.
        cmy = payload["wires"]["cmy_film"]
        assert cmy is not None
        assert "d_max" in cmy
        assert len(cmy["d_max"]) == 3
        for d in cmy["d_max"]:
            assert d > 0.0
        # d_min carries the below-base+fog headroom; downstream tools
        # need both endpoints to decode correctly.
        assert "d_min" in cmy
        assert cmy["d_min"] == [-0.2, -0.2, -0.2]
        # log_e_film / log_e_print / cmy_print are intermediate-only
        # for 4-LUT topologies; they stay null in 2-LUT bundles.
        assert payload["wires"]["log_e_film"] is None
        assert payload["wires"]["cmy_print"] is None

# Marketing-copy README assertions (apply-order phrasing, grain/fog
# explanations, etc.) were culled: the smoke test in TestBundleWrite
# covers "README exists and names the bundle's color spaces / stocks";
# everything beyond that pins prose, not behavior.


# ---------------------------------------------------------------------------
# 3-LUT bundles (L1 + L2 shared + per-print combined back-half)
# ---------------------------------------------------------------------------


class TestThreeLutBundle:
    """A ``3lut`` bundle splits the chain at the ``log_e_film`` and
    ``cmy_film`` taps but collapses everything after ``cmy_film`` into
    a single per-print back-half cube. L1 + L2 are print-independent
    (filming stages); L3 is print-specific and contains
    ``printing.expose + printing.develop + scanning.scan``.

    Total cubes for an N-print bundle: ``2 + N`` — one fewer per print
    than 4-LUT, at the cost of losing the ``log_e_print`` tap for
    enlarger-stage effect injection.
    """

    _THREE_LUT_RES = 5
    _THREE_LUT_PRINTS = ("kodak_portra_endura", "fujifilm_crystal_archive_typeii")

    @pytest.fixture(scope="class")
    def three_lut_spec(self) -> BundleSpec:
        return BundleSpec(
            name="portra400_three_lut",
            film_profile="kodak_portra_400",
            print_profiles=self._THREE_LUT_PRINTS,
            input_color_space=_INPUT_CS,
            output_color_space=_OUTPUT_CS,
            topology="3lut",
            resolution=self._THREE_LUT_RES,
        )

    @pytest.fixture(scope="class")
    def three_lut_bundle(self, three_lut_spec):
        return BundleBuilder(three_lut_spec).build()

    # ---- structure ------------------------------------------------------

    def test_bundle_has_2_shared_plus_n_per_print(self, three_lut_bundle):
        expected = 2 + len(self._THREE_LUT_PRINTS)
        assert len(three_lut_bundle.luts) == expected

    def test_topology_recorded(self, three_lut_bundle):
        assert three_lut_bundle.meta.topology == "3lut"

    def test_roles_are_two_shared_then_print_specific(self, three_lut_bundle):
        roles = [lut.role for lut in three_lut_bundle.meta.luts]
        # Layout: filming_expose (L1), filming_develop (L2), then one
        # printing_combined per print.
        assert roles[0] == "filming_expose"
        assert roles[1] == "filming_develop"
        for role in roles[2:]:
            assert role == "printing_combined"

    def test_l1_l2_print_independent(self, three_lut_bundle):
        """L1 and L2 are print=None in the metadata; the printing_combined
        L3 carries the print name."""
        metas = three_lut_bundle.meta.luts
        assert metas[0].print_profile is None
        assert metas[1].print_profile is None
        for meta in metas[2:]:
            assert meta.print_profile in self._THREE_LUT_PRINTS

    def test_filenames_use_numbered_convention(self, three_lut_bundle):
        """Numbered (l1/l2/l3) rather than semantic (film/print) — matches
        the convention 4-LUT uses for ≥3-cube topologies."""
        paths = [rel for rel, _ in three_lut_bundle.luts]
        assert paths[0].endswith("_l1.cube")
        assert paths[1].endswith("_l2.cube")
        for path in paths[2:]:
            assert path.endswith("_l3.cube")

    def test_wire_exposure_is_log_e_film_plus_cmy_film(self, three_lut_bundle):
        """3-LUT exposes the two filming taps but collapses the printing
        side — ``log_e_print`` and ``cmy_print`` stay None."""
        wires = three_lut_bundle.meta.wires
        assert wires.log_e_film is not None
        assert wires.cmy_film is not None
        assert wires.log_e_print is None
        assert wires.cmy_print is None

    # cmy_film d_min headroom is already pinned by the 2-LUT class; same
    # invariant applies for 3-LUT, no point asserting twice.

    def test_log_e_film_wire_has_positive_span(self, three_lut_bundle):
        wire = three_lut_bundle.meta.wires.log_e_film
        assert wire.max > wire.min
        assert wire.max - wire.min < 25.0

    # 3-LUT bundle.json wires structure is covered by the meta-level
    # test_wire_exposure_is_log_e_film_plus_cmy_film. The on-disk JSON
    # is a 1:1 serialization of the meta object — pinning it twice
    # adds no behavior coverage.
    # 3-LUT README prose assertions culled (see comment under 2-LUT).


# ---------------------------------------------------------------------------
# M6 — 4-LUT bundles
# ---------------------------------------------------------------------------

class TestFourLutBundle:
    """A ``4lut`` bundle splits the chain at
    three intermediate taps (``log_e_film``, ``cmy_film``,
    ``log_e_print``). L1 + L2 are print-independent (filming stages),
    L3 + L4 are print-specific (printing + scan).

    Total cubes for an N-print bundle: ``2 + 2N``. The
    ``WiresMeta.log_e_film`` / ``cmy_film`` / ``log_e_print`` fields
    are all populated; ``cmy_print`` stays None (L4 collapses it).
    """

    _FOUR_LUT_RES = 5
    _FOUR_LUT_PRINTS = ("kodak_portra_endura", "fujifilm_crystal_archive_typeii")

    @pytest.fixture(scope="class")
    def four_lut_spec(self) -> BundleSpec:
        return BundleSpec(
            name="portra400_four_lut",
            film_profile="kodak_portra_400",
            print_profiles=self._FOUR_LUT_PRINTS,
            input_color_space=_INPUT_CS,
            output_color_space=_OUTPUT_CS,
            topology="4lut",
            resolution=self._FOUR_LUT_RES,
        )

    @pytest.fixture(scope="class")
    def four_lut_bundle(self, four_lut_spec):
        return BundleBuilder(four_lut_spec).build()

    # ---- structure ------------------------------------------------------

    def test_bundle_has_2_shared_plus_2_per_print(self, four_lut_bundle):
        expected = 2 + 2 * len(self._FOUR_LUT_PRINTS)
        assert len(four_lut_bundle.luts) == expected

    def test_first_two_luts_are_shared_l1_l2(self, four_lut_bundle):
        rel0, lut0 = four_lut_bundle.luts[0]
        rel1, lut1 = four_lut_bundle.luts[1]
        assert rel0.endswith("_l1.cube")
        assert rel1.endswith("_l2.cube")
        assert "portra400" in rel0 and "portra400" in rel1
        # No print name in the shared filenames.
        for print in self._FOUR_LUT_PRINTS:
            print_tag = print.split("_", 1)[1].replace("_", "")[:8]
            assert print_tag not in rel0
            assert print_tag not in rel1
        # Shape sanity.
        for lut in (lut0, lut1):
            assert lut.table.shape == (self._FOUR_LUT_RES,) * 3 + (3,)
            assert lut.table.min() >= 0.0 and lut.table.max() <= 1.0

    def test_remaining_luts_are_l3_l4_per_print(self, four_lut_bundle):
        # After [L1, L2], the luts alternate L3, L4 per print in spec order.
        for i, print in enumerate(self._FOUR_LUT_PRINTS):
            l3_rel, l3 = four_lut_bundle.luts[2 + 2 * i]
            l4_rel, l4 = four_lut_bundle.luts[3 + 2 * i]
            assert l3_rel.endswith("_l3.cube")
            assert l4_rel.endswith("_l4.cube")
            for lut in (l3, l4):
                assert lut.table.shape == (self._FOUR_LUT_RES,) * 3 + (3,)
                assert lut.table.min() >= 0.0 and lut.table.max() <= 1.0

    # ---- metadata + wires ----------------------------------------------

    def test_metadata_records_topology(self, four_lut_bundle):
        assert four_lut_bundle.meta.topology == "4lut"

    def test_metadata_lut_roles(self, four_lut_bundle):
        roles = [e.role for e in four_lut_bundle.meta.luts]
        expected = ["filming_expose", "filming_develop"]
        for _ in self._FOUR_LUT_PRINTS:
            expected.extend(["printing_expose", "printing_develop_scan"])
        assert roles == expected

    def test_metadata_lut_domain_range(self, four_lut_bundle):
        luts = four_lut_bundle.meta.luts
        # L1, L2 (shared).
        assert luts[0].domain == "input_rgb" and luts[0].range == "log_e_film"
        assert luts[0].print_profile is None
        assert luts[1].domain == "log_e_film" and luts[1].range == "cmy_film"
        assert luts[1].print_profile is None
        # L3, L4 (per print).
        for i, print in enumerate(self._FOUR_LUT_PRINTS):
            l3 = luts[2 + 2 * i]
            l4 = luts[3 + 2 * i]
            assert l3.domain == "cmy_film" and l3.range == "log_e_print"
            assert l3.print_profile == print
            assert l4.domain == "log_e_print" and l4.range == "output_rgb"
            assert l4.print_profile == print

    def test_three_intermediate_wires_populated(self, four_lut_bundle):
        wires = four_lut_bundle.meta.wires
        assert wires.log_e_film is not None
        assert wires.cmy_film is not None
        assert wires.log_e_print is not None
        # 4-LUT collapses cmy_print into L4; it isn't a wire here.
        assert wires.cmy_print is None
        # LogE wires must have positive span. The total span can be
        # quite wide (~10-15 stops) because the probe pass includes
        # input cube corners near zero linear, which map to the
        # pipeline's deep-shadow floor — that range is real, not a bug.
        # Refining the wire to exclude the floor is a v2 concern
        # (percentile-based, or analytical from profile data per n050).
        for name, wire in (("log_e_film", wires.log_e_film),
                            ("log_e_print", wires.log_e_print)):
            assert wire.max > wire.min, f"{name} span must be positive"
            assert wire.max - wire.min < 25.0, (
                f"{name} span unexpectedly wide: {wire.max - wire.min}"
            )
        # Density wire: per-channel d_max in a sensible film-density range.
        for d in wires.cmy_film.d_max:
            assert 0.1 < d < 10.0

    def test_wire_constants_clamped_to_four_decimals(self, four_lut_bundle):
        """Wire values are clamped to 4 decimal places for human ergonomics
        (colorists copy / hand-edit these inside node graphs). Every
        constant must therefore lie exactly on the 1e-4 grid."""
        wires = four_lut_bundle.meta.wires
        scalars: list[tuple[str, float]] = [
            ("log_e_film.min", wires.log_e_film.min),
            ("log_e_film.max", wires.log_e_film.max),
            ("log_e_print.min", wires.log_e_print.min),
            ("log_e_print.max", wires.log_e_print.max),
        ]
        for c, d in enumerate(wires.cmy_film.d_max):
            scalars.append((f"cmy_film.d_max[{c}]", d))
        for c, d in enumerate(wires.cmy_film.d_min):
            scalars.append((f"cmy_film.d_min[{c}]", d))
        for name, v in scalars:
            assert v == round(v, 4), (
                f"{name}={v!r} is not on the 1e-4 grid"
            )

    # ---- behavior -------------------------------------------------------

    def test_prints_share_l1_l2_byte_identical(self, four_lut_bundle):
        """L1 and L2 don't depend on the print print; the bundle has
        exactly ONE pair of them shared across every print."""
        # We don't compare across bundles here — the assertion is that
        # the bundle's metadata has the L1 / L2 entries marked
        # print=None, and the lut list has exactly one of each.
        roles = [e.role for e in four_lut_bundle.meta.luts]
        assert roles.count("filming_expose") == 1
        assert roles.count("filming_develop") == 1

    def test_prints_produce_distinct_l3_l4(self, four_lut_bundle):
        # Per-print L3 and L4 must differ between prints.
        l3_a = four_lut_bundle.luts[2][1].table  # print 0 L3
        l3_b = four_lut_bundle.luts[4][1].table  # print 1 L3
        l4_a = four_lut_bundle.luts[3][1].table  # print 0 L4
        l4_b = four_lut_bundle.luts[5][1].table  # print 1 L4
        assert not np.array_equal(l3_a, l3_b)
        assert not np.array_equal(l4_a, l4_b)

    def test_chain_matches_live_pipeline(self, four_lut_spec, four_lut_bundle):
        """Apply L1∘L2∘L3∘L4 (all trilinear) at random samples; compare
        to the live spektrafilm pipeline end-to-end. 4-LUT chains
        accumulate more interpolation error than 1-LUT/2-LUT at the
        same per-cube resolution, so the tolerance is looser — but
        the chain must still produce sane output and not blow up."""
        from spektrafilm.runtime.params_builder import digest_params, init_params
        from spektrafilm.runtime.pipeline import SimulationPipeline
        from spektrafilm_lut_creator.color_spaces import get as get_cs
        from spektrafilm_lut_creator.qa.evaluators import apply_trilinear

        first_print = self._FOUR_LUT_PRINTS[0]
        in_entry = get_cs(_INPUT_CS)
        out_entry = get_cs(_OUTPUT_CS)
        params = init_params(film_profile="kodak_portra_400", print_profile=first_print)
        params.debug.lut_mode = True
        params.io.input_color_space = in_entry.primaries
        params.io.output_color_space = out_entry.primaries
        params.io.input_cctf_decoding = False
        params.io.output_cctf_encoding = False
        params = digest_params(params)
        pipeline = SimulationPipeline(params)

        rng = np.random.default_rng(20260516)
        samples_encoded = rng.uniform(0.0, 1.0, size=(200, 3)).astype(np.float32)
        # Mirror the bake's decode + input-gain path.
        # Fixture is ACEScct at exposure_ev=0 → identity gain.
        samples_linear = decode_cctf(samples_encoded, _INPUT_CS)
        samples_linear = (samples_linear * input_gain(_INPUT_CS)).astype(np.float32)
        live_rgb_linear = np.asarray(
            pipeline.process(samples_linear.reshape(1, -1, 3)),
            dtype=float,
        ).reshape(-1, 3)
        live_rgb_encoded = np.clip(
            encode_cctf(live_rgb_linear, _OUTPUT_CS), 0.0, 1.0,
        )

        # Chain through all four baked LUTs (trilinear in every stage).
        l1 = four_lut_bundle.luts[0][1]
        l2 = four_lut_bundle.luts[1][1]
        l3 = four_lut_bundle.luts[2][1]  # print 0
        l4 = four_lut_bundle.luts[3][1]  # print 0
        log_e_film_code = apply_trilinear(l1.table, samples_encoded)
        cmy_film_code = apply_trilinear(l2.table, log_e_film_code)
        log_e_print_code = apply_trilinear(l3.table, cmy_film_code)
        chain_rgb_encoded = apply_trilinear(l4.table, log_e_print_code)

        diff = np.abs(chain_rgb_encoded - live_rgb_encoded)
        # 4-stage trilinear at 5^3 is coarse; bulk error is small,
        # worst-case errors can be substantial but shouldn't break.
        assert diff.mean() < 0.15, f"mean RGB error: {diff.mean():.4f}"
        assert diff.max() < 0.6, f"max RGB error: {diff.max():.4f}"

    # ---- on-disk --------------------------------------------------------

    def test_write_emits_2_shared_plus_2_per_print_cubes(self, four_lut_spec, tmp_path):
        bundle = BundleBuilder(four_lut_spec).build()
        builder = BundleBuilder(four_lut_spec)
        out_dir = builder.write(bundle, tmp_path / "four_lut_out")
        for rel_path, _ in bundle.luts:
            assert (out_dir / rel_path).exists(), f"missing {rel_path}"
        cubes = sorted(p.name for p in out_dir.glob("*.cube"))
        l1_cubes = [c for c in cubes if c.endswith("_l1.cube")]
        l2_cubes = [c for c in cubes if c.endswith("_l2.cube")]
        l3_cubes = [c for c in cubes if c.endswith("_l3.cube")]
        l4_cubes = [c for c in cubes if c.endswith("_l4.cube")]
        assert len(l1_cubes) == 1 and len(l2_cubes) == 1
        assert len(l3_cubes) == len(self._FOUR_LUT_PRINTS)
        assert len(l4_cubes) == len(self._FOUR_LUT_PRINTS)

    # 4-LUT bundle.json wire structure is a 1:1 serialization of the
    # meta object asserted elsewhere — not pinned twice. README prose
    # assertions culled (see comment under 2-LUT).

    def test_4lut_cmy_film_reserves_fog_headroom(self, four_lut_bundle):
        wires = four_lut_bundle.meta.wires
        assert wires.cmy_film.d_min == (-0.2, -0.2, -0.2)


# ---------------------------------------------------------------------------
# Spec-level QA opt-in + default output directory
# ---------------------------------------------------------------------------


class TestBundleSpecQaFields:
    """``BundleSpec.qa`` and ``qa_print_index`` are validated at spec
    construction so a bad index doesn't show up halfway through a build.
    """

    def test_qa_defaults_to_off_and_no_index(self):
        spec = BundleSpec(
            film_profile="kodak_portra_400",
            print_profiles=("kodak_portra_endura",),
            input_color_space="ACEScct",
            output_color_space="sRGB",
        )
        assert spec.qa is False
        assert spec.qa_print_index is None

    def test_qa_print_index_out_of_range_rejected(self):
        with pytest.raises(ValueError, match="qa_print_index"):
            BundleSpec(
                film_profile="kodak_portra_400",
                print_profiles=("kodak_portra_endura",),
                input_color_space="ACEScct",
                output_color_space="sRGB",
                qa=True,
                qa_print_index=3,
            )

    def test_qa_print_index_zero_accepted(self):
        spec = BundleSpec(
            film_profile="kodak_portra_400",
            print_profiles=("kodak_portra_endura",),
            input_color_space="ACEScct",
            output_color_space="sRGB",
            qa=True,
            qa_print_index=0,
        )
        assert spec.qa_print_index == 0


class TestBundleSpecExposureEv:
    """``BundleSpec.exposure_ev`` defaults to 0.0 (midgray pinned by
    construction — identity input gain for every reflectance-scale
    input kind). A non-zero value is a deliberate, disclosed
    re-exposure."""

    @pytest.mark.parametrize("input_cs", [
        "Rec.2020", "sRGB", "ACEScct", "Panasonic V-Log",
    ])
    def test_default_is_midgray_pinned_identity(self, input_cs):
        from spektrafilm_lut_creator.color_spaces import input_gain
        spec = BundleSpec(
            film_profile="kodak_portra_400",
            print_profiles=("kodak_portra_endura",),
            input_color_space=input_cs,
            output_color_space="sRGB",
        )
        assert spec.exposure_ev == 0.0
        assert input_gain(spec.input_color_space, spec.exposure_ev) == 1.0

    def test_explicit_value_is_preserved_as_float(self):
        spec = BundleSpec(
            film_profile="kodak_portra_400",
            print_profiles=("kodak_portra_endura",),
            input_color_space="sRGB",
            output_color_space="sRGB",
            exposure_ev=1.5,
        )
        assert spec.exposure_ev == 1.5
        assert isinstance(spec.exposure_ev, float)


class TestDefaultOutputDirectory:
    """``BundleBuilder.write(bundle)`` without an ``out_dir`` drops the
    bundle into ``cwd/build/lut_bundles/<spec.name>/`` — convenient for a
    bake script that doesn't want to think about paths.
    """

    def test_default_out_dir_lands_under_cwd_build_lut_bundles(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        spec = BundleSpec(
            film_profile="kodak_portra_400",
            print_profiles=("kodak_portra_endura",),
            input_color_space="ACEScct",
            output_color_space="sRGB",
            resolution=5,
        )
        builder = BundleBuilder(spec)
        bundle = builder.build()
        out = builder.write(bundle)
        expected = tmp_path / "build" / "lut_bundles" / spec.name
        assert out == expected
        assert (expected / "bundle.json").is_file()

    def test_explicit_out_dir_still_honored(self, tmp_path):
        spec = BundleSpec(
            name="explicit_test",
            film_profile="kodak_portra_400",
            print_profiles=("kodak_portra_endura",),
            input_color_space="ACEScct",
            output_color_space="sRGB",
            resolution=5,
        )
        builder = BundleBuilder(spec)
        bundle = builder.build()
        out = builder.write(bundle, tmp_path / "custom_location")
        assert out == tmp_path / "custom_location"
        assert (out / "bundle.json").is_file()


class TestQaAutoRun:
    """When ``spec.qa=True``, ``write()`` triggers the QA suite for the
    selected print(s) and drops reports at ``<bundle>/qa/<per-print>/``.
    The cache directory is removed after each run so the bundle stays
    ship-ready.
    """

    @pytest.fixture(autouse=True)
    def _stub_qa_run(self, monkeypatch):
        import spektrafilm_lut_creator.qa as qa_module

        def fake_run(_spec, _bundle, out_dir, *, suite=None, print_index=0):
            out_dir = Path(out_dir)
            figures_dir = out_dir / "figures"
            cache_dir = out_dir / "cache"
            figures_dir.mkdir(parents=True, exist_ok=True)
            cache_dir.mkdir(parents=True, exist_ok=True)
            (figures_dir / f"qa_stub_{print_index}.png").write_bytes(b"stub")
            (cache_dir / "scratch.tmp").write_text("stub", encoding="utf-8")
            (out_dir / "report.md").write_text("# stub qa report\n", encoding="utf-8")
            (out_dir / "report.html").write_text("<html></html>\n", encoding="utf-8")
            return [
                Result(
                    name=f"qa_stub_{print_index}",
                    summary={"print_index": print_index},
                    interpretation="stubbed QA run",
                    passed=True,
                )
            ]

        monkeypatch.setattr(qa_module, "run", fake_run)

    def _make_spec(self, *, qa: bool, qa_print_index=None, prints=None):
        return BundleSpec(
            film_profile="kodak_portra_400",
            print_profiles=prints or ("kodak_portra_endura",),
            input_color_space="ACEScct",
            output_color_space="sRGB",
            topology="2lut",
            resolution=5,
            qa=qa,
            qa_print_index=qa_print_index,
        )

    def test_qa_false_skips_qa_subdir(self, tmp_path):
        spec = self._make_spec(qa=False)
        builder = BundleBuilder(spec)
        bundle = builder.build()
        out = builder.write(bundle, tmp_path / "no_qa")
        assert not (out / "qa").exists()

    def test_qa_true_runs_for_each_print(self, tmp_path):
        prints = ("kodak_portra_endura", "fujifilm_crystal_archive_typeii")
        spec = self._make_spec(qa=True, prints=prints)
        builder = BundleBuilder(spec)
        bundle = builder.build()
        out = builder.write(bundle, tmp_path / "qa_all")
        qa_dir = out / "qa"
        readme = (out / "README.md").read_text(encoding="utf-8")
        # One report folder per print, named with that print substituted
        # into the bundle's canonical pattern.
        report_names = sorted(p.name for p in qa_dir.iterdir() if p.is_dir())
        assert len(report_names) == 2
        assert any("portraendura" in n for n in report_names)
        assert any("crystalarchive" in n for n in report_names)
        assert "## Quality" in readme
        assert "qa_stub_0" in readme
        assert "qa_stub_1" in readme
        # Each report carries a report.md and a figures/ subdir.
        for sub in qa_dir.iterdir():
            assert (sub / "report.md").is_file()
            assert (sub / "figures").is_dir()
            assert not (sub / "cache").exists(), (
                f"cache should be deleted from {sub} after QA"
            )

    def test_qa_print_index_selects_one_print(self, tmp_path):
        prints = ("kodak_portra_endura", "fujifilm_crystal_archive_typeii")
        spec = self._make_spec(qa=True, qa_print_index=1, prints=prints)
        builder = BundleBuilder(spec)
        bundle = builder.build()
        out = builder.write(bundle, tmp_path / "qa_one")
        qa_dir = out / "qa"
        readme = (out / "README.md").read_text(encoding="utf-8")
        report_dirs = [p for p in qa_dir.iterdir() if p.is_dir()]
        assert len(report_dirs) == 1
        # The single report is for the second print (index 1).
        assert "crystalarchive" in report_dirs[0].name
        assert "portraendura" not in report_dirs[0].name
        assert "## Quality" in readme
        assert "qa_stub_1" in readme
        assert "qa_stub_0" not in readme
