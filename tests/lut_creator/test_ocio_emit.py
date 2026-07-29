"""Tests for the OCIO 2 config emission (M8a).

Three layers of validation, in order of importance:

1. **PyOpenColorIO load** — the emitted ``config.ocio`` must parse and
   instantiate without errors. Catches syntax bugs immediately.
2. **Colorspace path resolution** — OCIO can build a processor from
   ACES2065-1 to the spektrafilm colorspace, and the processor produces
   finite output on a sampled grid.
3. **Cube-application consistency** — the named colorspace path
   (ACES2065-1 -> spektrafilm_<film>_<print>) produces the same numbers
   as the equivalent explicit transform chain (AP0 -> input encoded ->
   apply .cube directly), confirming OCIO is composing the steps the
   way the emitter intends.

The bundle build itself is the most expensive step; a module-scoped
fixture builds one tiny ACEScg -> sRGB bundle and writes it to disk
under ``tmp_path_factory``, so every assertion in this file shares it.

See ``studies/a40_lut_system/n120_ocio_config_emission.md``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from spektrafilm_lut_creator import ocio_emit
from spektrafilm_lut_creator.builders import BundleBuilder
from spektrafilm_lut_creator.bundles import BundleSpec

from .factories import make_bundle_spec

pytest.importorskip(
    "PyOpenColorIO",
    reason="PyOpenColorIO required for OCIO config validation tests; "
    "install with `pip install opencolorio`.",
)


_RESOLUTION = 5
_FILM = "kodak_portra_400"
_PRINT = "kodak_portra_endura"
_INPUT_CS = "ACEScct"
_OUTPUT_CS = "sRGB"


# ---------------------------------------------------------------------------
# Pure-function tests (no bundle build).
# ---------------------------------------------------------------------------


class TestSupportedPredicate:
    def _spec(self, **overrides) -> BundleSpec:
        return make_bundle_spec(
            name="t",
            ocio_config=True,  # tests below assume the OCIO path is opted-in
            **overrides,
        )

    def test_supported_for_1lut_acescg_to_srgb(self):
        assert ocio_emit.is_supported(self._spec())
        assert ocio_emit.unsupported_reason(self._spec()) == ""

    @pytest.mark.parametrize("topo", ["1lut", "2lut", "3lut", "4lut"])
    def test_supported_for_all_topologies(self, topo):
        spec = self._spec(topology=topo)
        assert ocio_emit.is_supported(spec)
        assert ocio_emit.unsupported_reason(spec) == ""

    def test_unsupported_for_unknown_output(self):
        spec = self._spec(output_color_space="Adobe RGB")  # not in _COLORSPACE_BUILTIN
        assert not ocio_emit.is_supported(spec)
        msg = ocio_emit.unsupported_reason(spec)
        assert "Adobe RGB" in msg
        assert "supported outputs" in msg.lower()

    def test_emit_raises_for_unsupported(self):
        # Pick a registered-but-unsupported output color space (no
        # BuiltinTransform mapping in _COLORSPACE_BUILTIN yet) — the
        # NotImplementedError raises before bundle.meta is touched.
        spec = self._spec(output_color_space="Adobe RGB")
        with pytest.raises(NotImplementedError, match="Adobe RGB"):
            ocio_emit.emit_ocio_config(bundle=None, spec=spec)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Bundle-based fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def written_bundle(tmp_path_factory) -> tuple[Path, BundleSpec, Bundle]:
    """Build and write one tiny 1-LUT bundle. Used by every test below."""
    from spektrafilm_lut_creator.bundles import Bundle  # noqa: F401 (typing)

    spec = make_bundle_spec(name="ocio_emit_fixture", ocio_config=True)
    builder = BundleBuilder(spec)
    bundle = builder.build()
    out_dir = tmp_path_factory.mktemp("ocio_bundle")
    builder.write(bundle, out_dir / spec.name)
    return (out_dir / spec.name), spec, bundle


# ---------------------------------------------------------------------------
# Validation tests.
# ---------------------------------------------------------------------------


class TestConfigOnDisk:
    def test_config_ocio_written(self, written_bundle):
        bundle_dir, _spec, _bundle = written_bundle
        config_path = bundle_dir / "config.ocio"
        assert config_path.is_file()
        text = config_path.read_text(encoding="utf-8")
        assert "ocio_profile_version: 2.4" in text
        assert "spektrafilm_portra400_portraendura" in text
        assert "search_path: ." in text

    def test_default_skips_emission(self, tmp_path):
        """OCIO emission is opt-in; the default BundleSpec produces no config.ocio."""
        spec = make_bundle_spec(name="default_no_ocio")
        assert spec.ocio_config is False
        builder = BundleBuilder(spec)
        out_dir = builder.write(builder.build(), tmp_path / spec.name)
        assert not (out_dir / "config.ocio").exists()

    def test_explicit_false_skips_emission(self, tmp_path):
        spec = make_bundle_spec(name="explicit_no_ocio", ocio_config=False)
        builder = BundleBuilder(spec)
        out_dir = builder.write(builder.build(), tmp_path / spec.name)
        assert not (out_dir / "config.ocio").exists()


class TestConfigLoad:
    """The headline test: OCIO must accept the file."""

    def test_pyopencolorio_loads_config(self, written_bundle):
        import PyOpenColorIO as ocio

        bundle_dir, _spec, _bundle = written_bundle
        config = ocio.Config.CreateFromFile(str(bundle_dir / "config.ocio"))
        # Calling validate() raises if the config is malformed.
        config.validate()

    def test_config_declares_expected_colorspaces(self, written_bundle):
        import PyOpenColorIO as ocio

        bundle_dir, _spec, _bundle = written_bundle
        config = ocio.Config.CreateFromFile(str(bundle_dir / "config.ocio"))
        names = {cs.getName() for cs in config.getColorSpaces()}
        assert "ACES2065-1" in names
        assert _INPUT_CS in names
        assert _OUTPUT_CS in names
        assert "spektrafilm_portra400_portraendura" in names

    def test_aces_interchange_role_set(self, written_bundle):
        import PyOpenColorIO as ocio

        bundle_dir, _spec, _bundle = written_bundle
        config = ocio.Config.CreateFromFile(str(bundle_dir / "config.ocio"))
        assert config.getRoleColorSpace("aces_interchange") == "ACES2065-1"
        assert config.getRoleColorSpace("scene_linear") == "ACES2065-1"


class TestProcessorEvaluation:
    """The spektrafilm colorspace path resolves to a finite-output processor."""

    def test_processor_from_ap0_to_spektrafilm_produces_finite_output(
        self, written_bundle
    ):
        import PyOpenColorIO as ocio

        bundle_dir, _spec, _bundle = written_bundle
        config = ocio.Config.CreateFromFile(str(bundle_dir / "config.ocio"))
        proc = config.getProcessor("ACES2065-1", "spektrafilm_portra400_portraendura")
        cpu = proc.getDefaultCPUProcessor()
        rng = np.random.default_rng(seed=0)
        samples = rng.uniform(0.0, 1.0, size=(64, 3)).astype(np.float32)
        out = samples.copy()
        cpu.applyRGB(out)
        assert np.all(np.isfinite(out)), "processor produced non-finite values"


class TestCubeApplicationConsistency:
    """Compositional check: the named colorspace path (AP0 -> spektrafilm)
    equals the explicit transform chain it represents (AP0 -> input -> .cube).

    This catches emitter bugs that would cause OCIO to silently chain the
    transforms in an unexpected order, or omit a stage."""

    def test_named_path_matches_explicit_chain(self, written_bundle):
        import PyOpenColorIO as ocio

        bundle_dir, _spec, bundle = written_bundle
        config = ocio.Config.CreateFromFile(str(bundle_dir / "config.ocio"))

        # Path A: the named colorspace path via the config.
        proc_named = config.getProcessor(
            "ACES2065-1", "spektrafilm_portra400_portraendura"
        ).getDefaultCPUProcessor()

        # Path B: the explicit GroupTransform that the spektrafilm
        # colorspace's from_scene_reference encodes inline.
        lut_relpath = bundle.luts[0][0]
        group = ocio.GroupTransform()
        group.appendTransform(
            ocio.ColorSpaceTransform(
                src="ACES2065-1",
                dst=_INPUT_CS,
            )
        )
        group.appendTransform(
            ocio.FileTransform(
                src=str(bundle_dir / lut_relpath),
                interpolation=ocio.INTERP_TETRAHEDRAL,
            )
        )
        proc_explicit = config.getProcessor(group).getDefaultCPUProcessor()

        # Apply both processors to a moderately-sized random grid in AP0.
        # AP0 values can extend outside [0, 1] in principle, but for this
        # consistency test any common input is fine — we're checking
        # numerical agreement between two paths through the same config.
        rng = np.random.default_rng(seed=42)
        samples = rng.uniform(0.0, 1.0, size=(256, 3)).astype(np.float32)

        out_named = samples.copy()
        proc_named.applyRGB(out_named)
        out_explicit = samples.copy()
        proc_explicit.applyRGB(out_explicit)

        # Both processors are evaluating the same underlying chain; the
        # only freedom OCIO has is in the order of internal precision
        # tricks, which should agree to float32 precision.
        np.testing.assert_allclose(out_named, out_explicit, atol=1e-6, rtol=1e-5)


# ---------------------------------------------------------------------------
# Multi-LUT topology coverage (M8b).
# ---------------------------------------------------------------------------

# Expected intermediate colorspace names for each multi-LUT topology, given
# the (film, print) fixture pair below. Driven by the naming convention in
# ocio_emit._intermediate_specs.
_EXPECTED_INTERMEDIATES: dict[str, list[str]] = {
    "1lut": [],
    "2lut": ["cmy_film_portra400"],
    "3lut": ["cmy_film_portra400", "log_e_film_portra400"],
    "4lut": [
        "cmy_film_portra400",
        "log_e_film_portra400",
        "log_e_print_portra400_portraendura",
    ],
}


@pytest.fixture(scope="module", params=["1lut", "2lut", "3lut", "4lut"])
def topology_bundle(request, tmp_path_factory) -> tuple[Path, BundleSpec, Bundle]:
    """One built+written bundle per topology, shared across the tests below."""
    from spektrafilm_lut_creator.bundles import Bundle  # noqa: F401

    topo = request.param
    spec = make_bundle_spec(
        name=f"topo_{topo}_fixture",
        topology=topo,
        ocio_config=True,
    )
    builder = BundleBuilder(spec)
    bundle = builder.build()
    out_dir = tmp_path_factory.mktemp(f"topo_{topo}")
    builder.write(bundle, out_dir / spec.name)
    return (out_dir / spec.name), spec, bundle


class TestAllTopologies:
    """Every topology produces a validated config with the right structure."""

    def test_config_validates(self, topology_bundle):
        import PyOpenColorIO as ocio

        bundle_dir, _spec, _bundle = topology_bundle
        config = ocio.Config.CreateFromFile(str(bundle_dir / "config.ocio"))
        config.validate()

    def test_intermediates_emitted_per_topology(self, topology_bundle):
        import PyOpenColorIO as ocio

        bundle_dir, spec, _bundle = topology_bundle
        config = ocio.Config.CreateFromFile(str(bundle_dir / "config.ocio"))
        names = {cs.getName() for cs in config.getColorSpaces()}
        for expected in _EXPECTED_INTERMEDIATES[spec.topology]:
            assert expected in names, (
                f"{spec.topology}: expected intermediate {expected!r} "
                f"in config, got {sorted(names)}"
            )

    def test_spektrafilm_colorspace_evaluates_to_finite_output(self, topology_bundle):
        import PyOpenColorIO as ocio

        bundle_dir, _spec, _bundle = topology_bundle
        config = ocio.Config.CreateFromFile(str(bundle_dir / "config.ocio"))
        proc = config.getProcessor(
            "ACES2065-1", "spektrafilm_portra400_portraendura"
        ).getDefaultCPUProcessor()
        rng = np.random.default_rng(seed=0)
        samples = rng.uniform(0.05, 0.95, size=(32, 3)).astype(np.float32)
        out = samples.copy()
        proc.applyRGB(out)
        assert np.all(np.isfinite(out))

    def test_intermediate_colorspaces_evaluate(self, topology_bundle):
        """Every intermediate's from_scene_reference path must produce
        finite output. Catches chain-construction bugs where an intermediate's
        prefix chain is malformed."""
        import PyOpenColorIO as ocio

        bundle_dir, spec, _bundle = topology_bundle
        config = ocio.Config.CreateFromFile(str(bundle_dir / "config.ocio"))
        rng = np.random.default_rng(seed=1)
        samples = rng.uniform(0.05, 0.95, size=(32, 3)).astype(np.float32)
        for intermediate_name in _EXPECTED_INTERMEDIATES[spec.topology]:
            proc = config.getProcessor(
                "ACES2065-1", intermediate_name
            ).getDefaultCPUProcessor()
            out = samples.copy()
            proc.applyRGB(out)
            assert np.all(np.isfinite(out)), (
                f"intermediate {intermediate_name!r} produced non-finite output"
            )

    def test_wire_constants_in_description(self, topology_bundle):
        """Multi-LUT intermediates must embed their wire constants in
        the description so a downstream consumer can decode normalized
        code values to physical units."""
        bundle_dir, spec, bundle = topology_bundle
        text = (bundle_dir / "config.ocio").read_text(encoding="utf-8")
        wires = bundle.meta.wires
        if spec.topology == "1lut":
            assert wires.cmy_film is None
            assert wires.log_e_film is None
            assert wires.log_e_print is None
            return
        if wires.cmy_film is not None:
            # Each per-channel d_max value should appear as a 4-decimal float.
            for c in wires.cmy_film.d_max:
                assert f"{c:.4f}" in text, (
                    f"cmy_film d_max component {c:.4f} missing from config description"
                )
        if wires.log_e_film is not None:
            assert f"{wires.log_e_film.min:.4f}" in text
            assert f"{wires.log_e_film.max:.4f}" in text


class TestMultiPrintFourLut:
    """A 2-print 4-LUT bundle exercises the dedup logic: shared
    intermediates (cmy_film, log_e_film) appear once; per-print
    intermediates (log_e_print) appear N times."""

    @pytest.fixture(scope="class")
    def multi_print_4lut(self, tmp_path_factory):
        spec = make_bundle_spec(
            name="multi_print_4lut",
            print_profiles=(_PRINT, "fujifilm_crystal_archive_typeii"),
            topology="4lut",
            ocio_config=True,
        )
        builder = BundleBuilder(spec)
        bundle = builder.build()
        out_dir = tmp_path_factory.mktemp("multi_print_4lut")
        builder.write(bundle, out_dir / spec.name)
        return (out_dir / spec.name), spec, bundle

    def test_shared_intermediates_deduped(self, multi_print_4lut):
        import PyOpenColorIO as ocio

        bundle_dir, _spec, _bundle = multi_print_4lut
        config = ocio.Config.CreateFromFile(str(bundle_dir / "config.ocio"))
        names = [cs.getName() for cs in config.getColorSpaces()]
        assert names.count("cmy_film_portra400") == 1
        assert names.count("log_e_film_portra400") == 1

    def test_per_print_intermediates_distinct(self, multi_print_4lut):
        import PyOpenColorIO as ocio

        bundle_dir, _spec, _bundle = multi_print_4lut
        config = ocio.Config.CreateFromFile(str(bundle_dir / "config.ocio"))
        names = {cs.getName() for cs in config.getColorSpaces()}
        assert "log_e_print_portra400_portraendura" in names
        assert "log_e_print_portra400_crystalarchive" in names

    def test_prints_produce_distinct_output(self, multi_print_4lut):
        import PyOpenColorIO as ocio

        bundle_dir, _spec, _bundle = multi_print_4lut
        config = ocio.Config.CreateFromFile(str(bundle_dir / "config.ocio"))
        rng = np.random.default_rng(seed=2)
        samples = rng.uniform(0.1, 0.9, size=(32, 3)).astype(np.float32)
        out_a = samples.copy()
        out_b = samples.copy()
        config.getProcessor(
            "ACES2065-1", "spektrafilm_portra400_portraendura"
        ).getDefaultCPUProcessor().applyRGB(out_a)
        config.getProcessor(
            "ACES2065-1", "spektrafilm_portra400_crystalarchive"
        ).getDefaultCPUProcessor().applyRGB(out_b)
        # Different prints must produce numerically different output —
        # otherwise the OCIO chain isn't actually routing through the
        # per-print L3/L4 cubes.
        assert not np.allclose(out_a, out_b, atol=1e-3), (
            "prints produced identical output; per-print chain may be broken"
        )


# ---------------------------------------------------------------------------
# Display + View structure (M8c).
# ---------------------------------------------------------------------------


class TestDisplayAndViews:
    """1-LUT bundles expose each print as a View on the output Display.
    Multi-LUT bundles keep the minimal stub (Raw view only) so the value
    proposition stays "expose intermediates via colorspaces"."""

    def test_one_lut_single_print_emits_spektrafilm_view(self, written_bundle):
        import PyOpenColorIO as ocio

        bundle_dir, _spec, _bundle = written_bundle
        config = ocio.Config.CreateFromFile(str(bundle_dir / "config.ocio"))
        views = list(config.getViews(_OUTPUT_CS))
        # One per-print spektrafilm view + Raw fallback.
        assert "Raw" in views
        spektrafilm_views = [v for v in views if v.startswith("Spektrafilm ")]
        assert len(spektrafilm_views) == 1
        assert "Kodak Portra 400" in spektrafilm_views[0]
        assert "Kodak Portra Endura" in spektrafilm_views[0]

    def test_one_lut_multi_print_emits_view_per_print(self, tmp_path):
        spec = make_bundle_spec(
            name="multi_print_views",
            print_profiles=(_PRINT, "fujifilm_crystal_archive_typeii"),
            ocio_config=True,
        )
        builder = BundleBuilder(spec)
        builder.write(builder.build(), tmp_path / spec.name)
        import PyOpenColorIO as ocio

        config = ocio.Config.CreateFromFile(str(tmp_path / spec.name / "config.ocio"))
        views = list(config.getViews(_OUTPUT_CS))
        spektrafilm_views = [v for v in views if v.startswith("Spektrafilm ")]
        # N prints -> N spektrafilm views, plus Raw.
        assert len(spektrafilm_views) == 2
        assert "Raw" in views

    def test_view_resolves_to_spektrafilm_colorspace(self, written_bundle):
        """The View's colorspace pointer must resolve to the spektrafilm
        colorspace, not the bare output color space (which is the Raw view)."""
        import PyOpenColorIO as ocio

        bundle_dir, _spec, _bundle = written_bundle
        config = ocio.Config.CreateFromFile(str(bundle_dir / "config.ocio"))
        spektrafilm_view = next(
            v for v in config.getViews(_OUTPUT_CS) if v.startswith("Spektrafilm ")
        )
        cs_name = config.getDisplayViewColorSpaceName(_OUTPUT_CS, spektrafilm_view)
        assert cs_name == "spektrafilm_portra400_portraendura"

    @pytest.mark.parametrize("topo", ["2lut", "3lut", "4lut"])
    def test_multilut_keeps_minimal_stub_view(self, topo, tmp_path):
        """Per n120 §1: multi-LUT bundles get colorspace-only emission;
        Views beyond the Raw fallback would hide the very intermediates
        the user asked for."""
        spec = make_bundle_spec(
            name=f"gating_{topo}",
            topology=topo,
            ocio_config=True,
        )
        builder = BundleBuilder(spec)
        builder.write(builder.build(), tmp_path / spec.name)
        import PyOpenColorIO as ocio

        config = ocio.Config.CreateFromFile(str(tmp_path / spec.name / "config.ocio"))
        views = list(config.getViews(_OUTPUT_CS))
        assert views == ["Raw"], f"{topo} should emit only Raw view; got {views}"

    def test_active_views_lists_all_emitted(self, written_bundle):
        import PyOpenColorIO as ocio

        bundle_dir, _spec, _bundle = written_bundle
        config = ocio.Config.CreateFromFile(str(bundle_dir / "config.ocio"))
        active = list(config.getActiveViews())
        emitted = list(config.getViews(_OUTPUT_CS))
        assert set(active) == set(emitted), (
            f"active_views mismatch: active={active}, emitted={emitted}"
        )


# ---------------------------------------------------------------------------
# Extended color-space coverage. Sweep representative input/output pairs
# through the full build+write+validate pipeline. One combination per row;
# the resolution stays small so the sweep finishes in a few seconds.
# ---------------------------------------------------------------------------

_COVERAGE_PAIRS = [
    # (input_color_space, output_color_space)
    ("Panasonic V-Log", "Rec.709"),
    ("Panasonic V-Log", "Rec.2020"),
    ("Panasonic V-Log", "Rec.2100 PQ"),
    ("Panasonic V-Log", "Rec.2100 HLG"),
    ("Sony S-Log3", "P3-D65 PQ"),
    ("Canon Log 3", "sRGB"),
    ("RED Log3G10", "sRGB"),
    ("ARRI LogC4", "Display P3"),
    ("ACEScct", "Rec.709"),
    ("ACEScc", "sRGB"),
]


class TestExtendedColorSpaceCoverage:
    """Bundle bakes through OCIO's BuiltinTransform catalog should succeed
    for every (input, output) pair where both endpoints have a chain
    declared in ``_COLORSPACE_BUILTIN``. This is the test that catches
    regressions when a future OCIO update renames or removes a builtin."""

    @pytest.mark.parametrize("input_cs,output_cs", _COVERAGE_PAIRS)
    def test_validates_and_evaluates(self, input_cs, output_cs, tmp_path):
        import PyOpenColorIO as ocio

        spec = make_bundle_spec(
            name=f"coverage_{input_cs.split()[0]}_{output_cs.split()[0]}".lower(),
            input_color_space=input_cs,
            output_color_space=output_cs,
            ocio_config=True,
        )
        builder = BundleBuilder(spec)
        out_dir = builder.write(builder.build(), tmp_path / spec.name)
        config = ocio.Config.CreateFromFile(str(out_dir / "config.ocio"))
        config.validate()

        proc = config.getProcessor(
            "ACES2065-1", "spektrafilm_portra400_portraendura"
        ).getDefaultCPUProcessor()
        samples = np.array([[0.18, 0.18, 0.18]], dtype=np.float32)
        out_rgb = samples.copy()
        proc.applyRGB(out_rgb)
        assert np.all(np.isfinite(out_rgb)), (
            f"{input_cs} -> {output_cs} produced non-finite output"
        )
        # All emitted output spaces target [0, 1] code-value ranges.
        # Cubes can clip mildly outside that for extreme scene values, but
        # 18% gray should land well inside the cube.
        assert (out_rgb >= 0.0).all() and (out_rgb <= 1.0).all(), (
            f"{input_cs} -> {output_cs}: 18% gray output {out_rgb[0]} escapes [0, 1]"
        )
