"""Lightweight tests for the QA runner and topology helpers.

The full QA suite is intentionally not exercised here: it is expensive,
generates a large figure set, and duplicates builder-level integration
coverage for ``spec.qa=True``. This file keeps only cheap checks for the
public runner shell and the topology-aware LUT selection logic.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from spektrafilm_lut_creator.builders import BundleBuilder
from spektrafilm_lut_creator.bundles import BundleSpec
from spektrafilm_lut_creator.qa import DEFAULT_SUITE, list_tests, run
from spektrafilm_lut_creator.qa import suite as qa_suite
from spektrafilm_lut_creator.qa.result import Result

from ..factories import make_bundle_spec

_RESOLUTION = 3
_FIRST_PRINT = "kodak_portra_endura"
_SECOND_PRINT = "fujifilm_crystal_archive_typeii"


def _qa_spec(
    *, name: str, topology: str = "1lut", print_profiles=(_FIRST_PRINT,), **overrides
) -> BundleSpec:
    return make_bundle_spec(
        name=name,
        topology=topology,
        print_profiles=print_profiles,
        resolution=_RESOLUTION,
        **overrides,
    )


@pytest.fixture(scope="module")
def spec() -> BundleSpec:
    return _qa_spec(name="qa_smoke")


@pytest.fixture(scope="module")
def bundle(spec: BundleSpec):
    return BundleBuilder(spec).build()


@pytest.fixture(params=["1lut", "2lut", "3lut", "4lut"])
def multi_print_bundle(request):
    spec = _qa_spec(
        name=f"qa_{request.param}_selection",
        topology=request.param,
        print_profiles=(_FIRST_PRINT, _SECOND_PRINT),
    )
    return spec, BundleBuilder(spec).build()


def _stub_reference(*_args, **_kwargs):
    return SimpleNamespace(
        cache_key="stub",
        rng_samples_encoded=np.zeros((1, 3), dtype=np.float32),
    )


def test_default_suite_has_expected_tests():
    # 5 LUT-fidelity + 5 model-diagnostic + 2 input gamut compression
    # diagnostics + 4 picture-style diagnostics (noise sensitivity,
    # noise gradient, gamut edge stress, R-G plane slices) = 16. The
    # output-gamut compression preview is folded into
    # `output_gamut_compression`'s right panel rather than shipping as
    # its own test. (black_toe dropped per n090 §6 — flat line on log
    # inputs; highlight_rolloff dropped likewise.)
    assert len(DEFAULT_SUITE) == 16
    names = list_tests()
    assert "off_grid_identity" in names
    assert "monotonicity" in names
    assert "jacobian_condition" in names
    assert "total_variation" in names
    assert "output_gamut_compression" in names
    assert "characteristic_curve" in names
    assert "dynamic_range_usage" in names
    assert "planckian_sweep" in names
    assert "hue_twist_oklab" in names
    assert "spectral_locus_envelope" in names
    assert "input_gamut_compression_preview" in names
    assert "input_gamut_compression_smoothness" in names
    assert "noise_sensitivity" in names
    assert "noise_gradient" in names
    assert "output_gamut_edge_stress" in names
    assert "rg_plane_slices" in names


def test_run_accepts_custom_smoke_suite(monkeypatch, spec, bundle, tmp_path):
    monkeypatch.setattr(qa_suite.reference, "compute_or_load", _stub_reference)

    def smoke(ctx) -> Result:
        assert ctx.print_name == _FIRST_PRINT
        assert ctx.lut.table.shape == (_RESOLUTION, _RESOLUTION, _RESOLUTION, 3)
        return Result(
            name="smoke",
            summary={"samples": int(ctx.grid_input.shape[0])},
            interpretation="smoke run",
            passed=None,
        )

    results = run(spec, bundle, tmp_path, suite=(smoke,))

    assert [result.name for result in results] == ["smoke"]
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "smoke" in report
    assert "kodak_portra_endura" in report
    assert (tmp_path / "report.html").is_file()


def test_effective_lut_tracks_requested_print(multi_print_bundle):
    _spec, bundle = multi_print_bundle

    assert qa_suite._print_name(bundle, 1) == _SECOND_PRINT
    rel_path, lut = qa_suite._effective_lut(bundle, 1)

    assert "crystalarchive" in rel_path
    assert lut.table.shape == (_RESOLUTION, _RESOLUTION, _RESOLUTION, 3)
    assert np.all(np.isfinite(lut.table))


def test_print_index_out_of_range_raises(multi_print_bundle):
    _spec, bundle = multi_print_bundle

    with pytest.raises(IndexError, match="out of range"):
        qa_suite._print_name(bundle, 2)

    with pytest.raises(IndexError, match="out of range"):
        qa_suite._effective_lut(bundle, 2)
