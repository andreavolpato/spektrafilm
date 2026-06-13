from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from spektrafilm.model.illuminants import Illuminants
from spektrafilm.model.stocks import FilmStocks, PrintPapers
from spektrafilm.utils.gamut_compression import InputGamutCompressSpec
from spektrafilm_gui.params_mapper import build_params_from_state
import spektrafilm_gui.state as state_module
from spektrafilm_gui.state import (
    DEFAULT_FILM_STOCK,
    DEFAULT_PRINT_PAPER,
    PROJECT_DEFAULT_GUI_STATE,
    build_default_gui_state,
    clone_gui_state,
    gui_state_from_params,
)
from spektrafilm.runtime.api import init_params


def make_state():
    state = clone_gui_state(PROJECT_DEFAULT_GUI_STATE)
    state.simulation.enlarger.illuminant = Illuminants.lamp.value
    return state


def test_build_params_maps_grain_fields() -> None:
    state = make_state()
    state.grain.rms_granularity = (6.0, 7.0, 12.0)
    state.grain.particle_scale_sublayers = (1.0, 0.5, 0.25)

    params = build_params_from_state(state)

    assert params.film_render.grain.rms_granularity == (6.0, 7.0, 12.0)
    assert params.film_render.grain.particle_scale_sublayers == (1.0, 0.5, 0.25)


def test_build_params_copies_scanner_group() -> None:
    state = make_state()
    state.scanner = replace(
        state.scanner,
        lens_blur=0.4,
        white_correction=True,
        white_level=0.72,
        black_correction=False,
        black_level=0.14,
        unsharp_mask=(0.5, 1.2),
    )

    params = build_params_from_state(state)

    assert params.scanner.lens_blur == 0.4
    assert params.scanner.white_correction is True
    assert params.scanner.white_level == 0.72
    assert params.scanner.black_correction is False
    assert params.scanner.black_level == 0.14
    assert params.scanner.unsharp_mask == (0.5, 1.2)
    # mapper copies, does not alias the widget's group
    assert params.scanner is not state.scanner


def test_build_params_copies_halation_group() -> None:
    # state.halation is the runtime HalationParams group, in runtime units
    # (0-1 fractions, surfaced verbatim in the GUI). The mapper copies it
    # through unchanged — no GUI-only transforms.
    state = make_state()
    state.halation.boost_ev = 1.25
    state.halation.protect_ev = 2.5
    state.halation.boost_range = 0.35
    state.halation.halation_strength = (0.12, 0.06, 0.03)
    state.halation.scatter_tail_weight = (0.30, 0.25, 0.20)

    params = build_params_from_state(state)

    assert params.film_render.halation.boost_ev == 1.25
    assert params.film_render.halation.protect_ev == 2.5
    assert params.film_render.halation.boost_range == 0.35
    np.testing.assert_allclose(params.film_render.halation.halation_strength, np.array([0.12, 0.06, 0.03]))
    np.testing.assert_allclose(params.film_render.halation.scatter_tail_weight, np.array([0.30, 0.25, 0.20]))
    # mapper copies, does not alias the widget's group
    assert params.film_render.halation is not state.halation


def test_build_params_propagates_halation_high_level_knobs() -> None:
    state = make_state()
    state.halation.scatter_amount = 0.5
    state.halation.scatter_spatial_scale = 1.5
    state.halation.halation_amount = 2.0
    state.halation.halation_spatial_scale = 0.75
    state.halation.halation_n_bounces = 2
    state.halation.halation_bounce_decay = 0.4
    state.halation.halation_renormalize = False

    params = build_params_from_state(state)

    assert params.film_render.halation.scatter_amount == 0.5
    assert params.film_render.halation.scatter_spatial_scale == 1.5
    assert params.film_render.halation.halation_amount == 2.0
    assert params.film_render.halation.halation_spatial_scale == 0.75
    assert params.film_render.halation.halation_n_bounces == 2
    assert params.film_render.halation.halation_bounce_decay == 0.4
    assert params.film_render.halation.halation_renormalize is False


def test_build_params_maps_runtime_strings() -> None:
    state = make_state()
    state.camera.auto_exposure_method = 'median'
    state.input_image.io.input_color_space = 'Display P3'
    state.input_image.settings.rgb_to_raw_method = 'mallett2019'
    state.simulation.io.output_color_space = 'ACES2065-1'
    state.simulation.workflow.saving_cctf_encoding = False
    state.gui_only.display.settings.preview_max_size = 1024

    params = build_params_from_state(state)

    assert params.camera.auto_exposure_method == 'median'
    assert params.io.input_color_space == 'Display P3'
    assert params.settings.rgb_to_raw_method == 'mallett2019'
    assert params.settings.preview_max_size == 1024
    assert params.io.output_color_space == 'ACES2065-1'
    assert params.io.output_cctf_encoding is True


def test_build_params_maps_enlarger_diffusion_filter() -> None:
    state = make_state()
    state.enlarger_diffusion.active = True
    state.enlarger_diffusion.filter_family = 'pro_mist'
    state.enlarger_diffusion.strength = 0.5
    state.enlarger_diffusion.spatial_scale = 1.6
    state.enlarger_diffusion.halo_warmth = 0.3

    params = build_params_from_state(state)

    assert params.enlarger.diffusion_filter.active is True
    assert params.enlarger.diffusion_filter.filter_family == 'pro_mist'
    assert params.enlarger.diffusion_filter.strength == 0.5
    assert params.enlarger.diffusion_filter.spatial_scale == 1.6
    assert params.enlarger.diffusion_filter.halo_warmth == 0.3


def test_build_params_copies_camera_group_and_standalone_camera_diffusion() -> None:
    state = make_state()
    state.camera.lens_blur_um = 4.0
    state.camera.exposure_compensation_ev = 0.75
    # The camera diffusion filter is now its own section (FILM tab), owned by
    # state.camera_diffusion -- NOT the passthrough state.camera.diffusion_filter.
    state.camera_diffusion = replace(
        state.camera_diffusion,
        active=True,
        filter_family='glimmerglass',
        strength=0.25,
        spatial_scale=1.2,
        halo_warmth=-0.15,
    )

    params = build_params_from_state(state)

    assert params.camera.lens_blur_um == 4.0
    assert params.camera.exposure_compensation_ev == 0.75
    assert params.camera.diffusion_filter.active is True
    assert params.camera.diffusion_filter.filter_family == 'glimmerglass'
    assert params.camera.diffusion_filter.strength == 0.25
    assert params.camera.diffusion_filter.spatial_scale == 1.2
    assert params.camera.diffusion_filter.halo_warmth == -0.15
    assert params.camera is not state.camera


def test_build_params_camera_diffusion_owner_wins_over_stale_camera_passthrough() -> None:
    # Regression guard: the camera section passes its diffusion_filter through
    # untouched, so a stale value there must NOT win over the standalone owner.
    state = make_state()
    state.camera.diffusion_filter = replace(state.camera.diffusion_filter, strength=99.0)
    state.camera_diffusion = replace(state.camera_diffusion, strength=0.5)

    params = build_params_from_state(state)

    assert params.camera.diffusion_filter.strength == 0.5


def test_build_params_uses_preview_tuned_lut_settings() -> None:
    params = build_params_from_state(make_state())

    assert params.settings.use_enlarger_lut is True
    assert params.settings.use_scanner_lut is True
    assert params.settings.lut_resolution == 17
    assert params.settings.use_fast_stats is True


def test_build_params_copies_input_gamut_compress_group() -> None:
    state = make_state()
    state.input_gamut_compress = replace(
        state.input_gamut_compress,
        active=False,
        algorithm='oklch',
        knee=(0.1, 1.0, 4.0),
    )

    params = build_params_from_state(state)

    assert params.io.input_gamut_compress.active is False
    assert params.io.input_gamut_compress.algorithm == 'oklch'
    assert params.io.input_gamut_compress.knee == (0.1, 1.0, 4.0)
    # mapper copies, does not alias the widget's group
    assert params.io.input_gamut_compress is not state.input_gamut_compress


def test_build_params_copies_output_gamut_compress_group() -> None:
    state = make_state()
    state.output_gamut_compress = replace(
        state.output_gamut_compress,
        algorithm='aces_rgc',
        knee=(0.2, 1.0, 3.0),
    )

    params = build_params_from_state(state)

    assert params.io.output_gamut_compress.algorithm == 'aces_rgc'
    assert params.io.output_gamut_compress.knee == (0.2, 1.0, 3.0)
    assert params.io.output_gamut_compress is not state.output_gamut_compress


def test_gui_state_from_params_mirrors_input_gamut_compress_group() -> None:
    params = init_params(
        film_profile=FilmStocks.kodak_gold_200.value,
        print_profile=PrintPapers.kodak_supra_endura.value,
    )
    params.io.input_gamut_compress = InputGamutCompressSpec(active=False)

    state = gui_state_from_params(
        params,
        film_stock=FilmStocks.kodak_gold_200.value,
        print_paper=PrintPapers.kodak_supra_endura.value,
    )

    assert state.input_gamut_compress.active is False


def test_gui_state_from_params_fills_bw_grain_channel_padding() -> None:
    # A single-channel (B&W) stock's grain preset pads its per-channel tuples to
    # three with None (e.g. uniformity=(0.6, None, None)); the runtime reads only
    # channel 0, but the GUI editors are typed tuple[float, float, float] and
    # crash on None. gui_state_from_params must mirror the first real channel
    # across the padding so no None reaches the widgets (regression: the profile
    # sync crashed mid-way, leaving the development-time dropdown un-repopulated).
    from spektrafilm.runtime.api import digest_params

    params = digest_params(init_params(
        film_profile=FilmStocks.kodak_doublex.value,
        print_profile=PrintPapers.kodak_2302.value,
    ))
    # precondition: the runtime grain genuinely carries None padding for B&W
    assert any(c is None for c in params.film_render.grain.uniformity)

    state = gui_state_from_params(
        params,
        film_stock=FilmStocks.kodak_doublex.value,
        print_paper=PrintPapers.kodak_2302.value,
    )

    for field_name in ('rms_granularity', 'uniformity', 'density_min'):
        values = getattr(state.grain, field_name)
        assert all(c is not None for c in values), field_name
        # padding mirrors channel 0, not a misleading zero
        assert values == (values[0],) * len(values), field_name


def test_build_default_gui_state_uses_runtime_defaults() -> None:
    state = build_default_gui_state(
        film_stock=FilmStocks.kodak_gold_200.value,
        print_paper=PrintPapers.kodak_supra_endura.value,
    )

    assert state.grain.blur == 0.89
    assert state.grain.micro_structure == (0.2, 30)
    assert state.halation.boost_ev == 0.0
    assert state.halation.protect_ev == 4.0
    assert state.halation.boost_range == 0.3
    assert state.halation.scatter_amount == 1.0
    assert state.halation.scatter_spatial_scale == 1.0
    assert state.halation.halation_amount == 1.0
    assert state.halation.halation_spatial_scale == 1.0
    # kodak_gold_200 is (use=still, antihalation=weak), so _apply_halation_preset
    # seeds halation_strength from the weak-AH row of §5: (0.08, 0.02, 0.0).
    # state.halation holds the runtime group verbatim (0-1 fractions, shown
    # as-is in the GUI — no percentage transform).
    assert state.halation.halation_strength == (0.08, 0.02, 0.0)
    assert state.halation.halation_n_bounces == 3
    assert state.halation.halation_bounce_decay == 0.5
    assert state.halation.halation_renormalize is True
    assert state.input_image.io.crop_size == (0.1, 0.1)
    assert state.simulation.io.output_color_space == 'sRGB'
    assert state.simulation.workflow.saving_color_space == 'sRGB'
    assert state.simulation.workflow.saving_cctf_encoding is True
    assert state.camera_diffusion.active is False
    assert state.camera_diffusion.filter_family == 'black_pro_mist'
    assert state.camera_diffusion.strength == 0.5
    assert state.camera_diffusion.spatial_scale == 1.0
    assert state.camera_diffusion.halo_warmth == 0.0
    assert state.enlarger_diffusion.active is False
    assert state.enlarger_diffusion.filter_family == 'black_pro_mist'
    assert state.enlarger_diffusion.strength == 0.5
    assert state.enlarger_diffusion.spatial_scale == 1.0
    assert state.enlarger_diffusion.halo_warmth == 0.0
    assert state.scanner.white_correction is False
    assert state.scanner.white_level == 0.98
    assert state.scanner.black_correction is False
    assert state.scanner.black_level == 0.01
    assert state.gui_only.display.use_display_transform is True
    assert state.gui_only.display.gray_18_canvas is True
    assert state.camera.auto_exposure_method == 'center_weighted'
    assert state.gui_only.display.white_padding == 0.03
    assert state.gui_only.display.settings.preview_max_size == 640
    assert state.gui_only.display.output_interpolation == 'spline36'


def test_build_default_gui_state_applies_selection_defaults(monkeypatch) -> None:
    raw_params = object()
    digested_params = object()
    captured: dict[str, object] = {}

    def fake_init_params(*, film_profile, print_profile):
        captured['init_args'] = (film_profile, print_profile)
        return raw_params

    def fake_digest_after_selection(params):
        captured['digest_input'] = params
        return digested_params

    def fake_gui_state_from_params(params, *, film_stock, print_paper):
        captured['gui_args'] = (params, film_stock, print_paper)
        return 'gui-state'

    monkeypatch.setattr(state_module, 'init_params', fake_init_params)
    monkeypatch.setattr(state_module, 'digest_after_selection', fake_digest_after_selection)
    monkeypatch.setattr(state_module, 'gui_state_from_params', fake_gui_state_from_params)

    state = state_module.build_default_gui_state(
        film_stock='film-stock',
        print_paper='print-paper',
    )

    assert state == 'gui-state'
    assert captured['init_args'] == ('film-stock', 'print-paper')
    assert captured['digest_input'] is raw_params
    assert captured['gui_args'] == (digested_params, 'film-stock', 'print-paper')


def test_digest_after_selection_sets_route_from_film_type(monkeypatch) -> None:
    positive_params = SimpleNamespace(film=SimpleNamespace(is_positive=True), workflow=SimpleNamespace(route='input > film > print > scan'))
    negative_params = SimpleNamespace(film=SimpleNamespace(is_positive=False), workflow=SimpleNamespace(route='input > film > print > scan'))
    digested_params = [positive_params, negative_params]

    def fake_digest_params(_params):
        return digested_params.pop(0)

    monkeypatch.setattr(state_module, 'digest_params', fake_digest_params)

    positive_result = state_module.digest_after_selection(object())
    negative_result = state_module.digest_after_selection(object())

    assert positive_result.workflow.route == 'input > film > scan'
    assert negative_result.workflow.route == 'input > film > print > scan'


def test_project_default_gui_state_matches_builder() -> None:
    built_state = build_default_gui_state(
        film_stock=DEFAULT_FILM_STOCK,
        print_paper=DEFAULT_PRINT_PAPER,
    )

    assert PROJECT_DEFAULT_GUI_STATE == built_state
