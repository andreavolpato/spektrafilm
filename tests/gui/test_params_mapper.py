from __future__ import annotations

import numpy as np

from spektrafilm.model.illuminants import Illuminants
from spektrafilm.model.stocks import FilmStocks, PrintPapers
from spektrafilm_gui.params_mapper import build_params_from_state
from spektrafilm_gui.state import (
    DEFAULT_FILM_STOCK,
    DEFAULT_PRINT_PAPER,
    PROJECT_DEFAULT_GUI_STATE,
    build_default_gui_state,
    clone_gui_state,
)


def make_state():
    state = clone_gui_state(PROJECT_DEFAULT_GUI_STATE)
    state.simulation.print_illuminant = Illuminants.lamp.value
    return state


def test_build_params_maps_grain_fields() -> None:
    state = make_state()
    state.grain.particle_area_um2 = 0.42
    state.grain.particle_scale = (1.1, 1.2, 1.3)
    state.grain.particle_scale_layers = (2.2, 1.2, 0.6)

    params = build_params_from_state(state)

    assert params.film_render.grain.agx_particle_area_um2 == 0.42
    assert params.film_render.grain.agx_particle_scale == (1.1, 1.2, 1.3)
    assert params.film_render.grain.agx_particle_scale_layers == (2.2, 1.2, 0.6)


def test_build_params_maps_scanner_corrections() -> None:
    state = make_state()
    state.simulation.scan_white_correction = 0.72
    state.simulation.scan_black_correction = 0.14

    params = build_params_from_state(state)

    assert params.scanner.white_correction == 0.72
    assert params.scanner.black_correction == 0.14


def test_build_params_converts_halation_percentages_to_fractions() -> None:
    state = make_state()
    state.halation.halation_strength = (12.0, 6.0, 3.0)
    state.halation.scattering_strength = (8.0, 4.0, 2.0)

    params = build_params_from_state(state)

    np.testing.assert_allclose(params.film_render.halation.strength, np.array([0.12, 0.06, 0.03]))
    np.testing.assert_allclose(params.film_render.halation.scattering_strength, np.array([0.08, 0.04, 0.02]))


def test_build_params_maps_runtime_strings() -> None:
    state = make_state()
    state.simulation.auto_exposure_method = 'median'
    state.input_image.input_color_space = 'Display P3'
    state.input_image.spectral_upsampling_method = 'mallett2019'
    state.simulation.output_color_space = 'ACES2065-1'
    state.simulation.saving_cctf_encoding = False

    params = build_params_from_state(state)

    assert params.camera.auto_exposure_method == 'median'
    assert params.io.input_color_space == 'Display P3'
    assert params.settings.rgb_to_raw_method == 'mallett2019'
    assert params.io.output_color_space == 'ACES2065-1'
    assert params.io.output_cctf_encoding is True


def test_build_default_gui_state_uses_runtime_defaults() -> None:
    state = build_default_gui_state(
        film_stock=FilmStocks.kodak_gold_200.value,
        print_paper=PrintPapers.kodak_supra_endura.value,
    )

    assert state.grain.blur == 0.65
    assert state.grain.micro_structure == (0.2, 30)
    assert state.halation.halation_strength == (3.0, 0.3, 0.1)
    assert state.input_image.preview_resize_factor == 0.3
    assert state.input_image.crop_size == (0.1, 0.1)
    assert state.simulation.output_color_space == 'sRGB'
    assert state.simulation.saving_color_space == 'sRGB'
    assert state.simulation.saving_cctf_encoding is True
    assert state.simulation.scan_white_correction == 0.0
    assert state.simulation.scan_black_correction == 0.0
    assert state.display.use_display_transform is True
    assert state.display.gray_18_canvas is True
    assert state.simulation.auto_exposure_method == 'center_weighted'
    assert state.display.white_padding == 0.03


def test_project_default_gui_state_matches_builder() -> None:
    built_state = build_default_gui_state(
        film_stock=DEFAULT_FILM_STOCK,
        print_paper=DEFAULT_PRINT_PAPER,
    )

    assert PROJECT_DEFAULT_GUI_STATE == built_state