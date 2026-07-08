"""Contract tests for the params-first single-pair API (b60/n010 §10.6).

These guard the *policy* half of the design — was the simulator
configured right — which QA deliberately cannot see (its reference
pipeline shares the bake's params, so a policy mistake is identical on
both sides). Fidelity (cube vs model) stays QA's job.
"""
from __future__ import annotations

import numpy as np
import pytest

from spektrafilm.runtime.params_builder import init_params
from spektrafilm_lut_creator import bake
from spektrafilm_lut_creator.single import LutSpec, build_luts

pytestmark = pytest.mark.integration

_SPEC = LutSpec(
    input_color_space="ACEScct",
    output_color_space="sRGB",
    topology="1lut",
    resolution=5,
    name="test_single",
)
_PRINT = "kodak_portra_endura"


def _params():
    return init_params("kodak_portra_400", _PRINT)


def _graded_params():
    """Params carrying per-image trims, un-bakeable state, and a foreign
    route — everything the neutral-bake contract must strip."""
    p = _params()
    p.camera.exposure_compensation_ev = 1.5
    p.camera.auto_exposure = True
    p.enlarger.print_exposure = 2.0
    p.enlarger.y_filter_shift = 12.0
    p.enlarger.m_filter_shift = -8.0
    p.enlarger.preflash_exposure = 0.2
    p.io.crop = True
    p.io.upscale_factor = 2.0
    p.settings.preview_mode = True
    p.settings.use_enlarger_lut = True
    p.workflow.route = "input > convert-film > scan"
    return p


@pytest.fixture(scope="module")
def default_bundle():
    return build_luts(_params(), _SPEC)


def _table(bundle):
    return bundle.luts[0][1].table


def test_caller_params_never_mutated():
    params = _params()
    before = bake._flatten(params)
    build_luts(params, _SPEC)
    after = bake._flatten(params)
    assert before == after
    assert params.debug.lut_mode is False


def test_trims_and_route_bake_neutral(default_bundle):
    """Neutral-bake contract: a graded session bakes the same cube as a
    neutral one — per-image trims, un-bakeable effects, preview mode and
    a convert-film route must all be stripped, not baked."""
    graded = build_luts(_graded_params(), _SPEC)
    np.testing.assert_array_equal(_table(graded), _table(default_bundle))


def test_look_flows_into_bake(default_bundle):
    """The point of the params-first API: a look knob (film base scale)
    changes the baked cube."""
    params = _params()
    params.film_render.base.scale = 1.5
    bundle = build_luts(params, _SPEC)
    assert np.max(np.abs(_table(bundle) - _table(default_bundle))) > 1e-4


def test_bundle_carries_baked_params(default_bundle):
    baked = default_bundle.baked_params[_PRINT]
    assert baked.debug.lut_mode is True
    assert baked.workflow.route == bake.BAKE_ROUTE


def test_digest_changes_disclosed():
    """The snapshot discloses what the lut_mode digest neutralized."""
    bundle = build_luts(_graded_params(), _SPEC)
    changes = bundle.meta.params_snapshot[_PRINT]["digest_changes"]
    assert changes["enlarger.y_filter_shift"] == {"from": 12.0, "to": 0.0}
    assert changes["camera.exposure_compensation_ev"] == {"from": 1.5, "to": 0.0}
    assert changes["io.crop"] == {"from": True, "to": False}


def test_qa_reference_uses_baked_params():
    """One source of truth (n010 §8): the QA reference pipeline is built
    from bundle.baked_params, so caller-tuned params flow into QA. A
    look tweak that the old spec-reconstruction path could never see
    must show up in the reference pipeline's params."""
    from spektrafilm_lut_creator.qa import reference
    from spektrafilm_lut_creator.single import _bundle_spec

    params = _params()
    params.film_render.base.scale = 1.5
    bundle = build_luts(params, _SPEC)
    spec = _bundle_spec(params, _SPEC)
    pipeline = reference._reference_pipeline(spec, bundle, 0)
    assert pipeline._params.film_render.base.scale == 1.5
    assert pipeline._params.debug.lut_mode is True
