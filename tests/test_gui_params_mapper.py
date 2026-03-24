from __future__ import annotations

import numpy as np

from spektrafilm_gui.params_mapper import build_params_from_state
from spektrafilm_gui.state import (
    DEFAULT_FILM_STOCK,
    DEFAULT_PRINT_PAPER,
    DisplayState,
    PROJECT_DEFAULT_GUI_STATE,
    build_default_gui_state,
    CouplersState,
    GlareState,
    GrainState,
    GuiState,
    HalationState,
    InputImageState,
    PreflashingState,
    SimulationState,
    SpecialState,
)
from spektrafilm.model.illuminants import Illuminants
from spektrafilm.model.stocks import FilmStocks, PrintPapers


def make_state() -> GuiState:
    return GuiState(
        input_image=InputImageState(
            preview_resize_factor=0.3,
            upscale_factor=1.0,
            crop=False,
            crop_center=(0.5, 0.5),
            crop_size=(0.1, 0.1),
            input_color_space="ProPhoto RGB",
            apply_cctf_decoding=False,
            spectral_upsampling_method="hanatos2025",
            filter_uv=(1.0, 410.0, 8.0),
            filter_ir=(1.0, 675.0, 15.0),
        ),
        grain=GrainState(
            active=True,
            sublayers_active=True,
            particle_area_um2=0.2,
            particle_scale=(0.8, 1.0, 2.0),
            particle_scale_layers=(2.5, 1.0, 0.5),
            density_min=(0.07, 0.08, 0.12),
            uniformity=(0.97, 0.97, 0.99),
            blur=0.65,
            blur_dye_clouds_um=1.0,
            micro_structure=(0.1, 30.0),
        ),
        preflashing=PreflashingState(
            exposure=0.0,
            y_filter_shift=0.0,
            m_filter_shift=0.0,
            just_preflash=False,
        ),
        halation=HalationState(
            active=True,
            scattering_strength=(1.0, 2.0, 4.0),
            scattering_size_um=(30.0, 20.0, 15.0),
            halation_strength=(3.0, 0.3, 0.1),
            halation_size_um=(200.0, 200.0, 200.0),
        ),
        couplers=CouplersState(
            active=True,
            dir_couplers_amount=1.0,
            dir_couplers_ratio=(1.0, 1.0, 1.0),
            dir_couplers_diffusion_um=10.0,
            diffusion_interlayer=2.0,
            high_exposure_shift=0.0,
        ),
        glare=GlareState(
            active=True,
            percent=0.1,
            roughness=0.4,
            blur=0.5,
            compensation_removal_factor=0.0,
            compensation_removal_density=1.2,
            compensation_removal_transition=0.3,
        ),
        special=SpecialState(
            film_channel_swap=(0, 1, 2),
            film_gamma_factor=1.0,
            print_channel_swap=(0, 1, 2),
            print_gamma_factor=1.0,
            print_density_min_factor=0.4,
        ),
        simulation=SimulationState(
            film_stock=FilmStocks.kodak_gold_200.value,
            film_format_mm=35.0,
            camera_lens_blur_um=0.0,
            exposure_compensation_ev=0.0,
            auto_exposure=True,
            auto_exposure_method="center_weighted",
            print_paper=PrintPapers.kodak_supra_endura.value,
            print_illuminant=Illuminants.lamp.value,
            print_exposure=1.0,
            print_exposure_compensation=True,
            print_y_filter_shift=0.0,
            print_m_filter_shift=0.0,
            scan_lens_blur=0.0,
            scan_unsharp_mask=(0.7, 0.7),
            output_color_space="ProPhoto RGB",
            saving_color_space="sRGB",
            saving_cctf_encoding=True,
            scan_film=False,
            compute_full_image=False,
        ),
        display=DisplayState(
            use_display_transform=True,
            gray_18_canvas=False,
            white_padding=0.0,
        ),
    )


def test_build_params_maps_grain_fields() -> None:
    state = make_state()
    state.grain.particle_area_um2 = 0.42
    state.grain.particle_scale = (1.1, 1.2, 1.3)
    state.grain.particle_scale_layers = (2.2, 1.2, 0.6)

    params = build_params_from_state(state)

    assert params.film_render.grain.agx_particle_area_um2 == 0.42
    assert params.film_render.grain.agx_particle_scale == (1.1, 1.2, 1.3)
    assert params.film_render.grain.agx_particle_scale_layers == (2.2, 1.2, 0.6)


def test_build_params_converts_halation_percentages_to_fractions() -> None:
    state = make_state()
    state.halation.halation_strength = (12.0, 6.0, 3.0)
    state.halation.scattering_strength = (8.0, 4.0, 2.0)

    params = build_params_from_state(state)

    np.testing.assert_allclose(params.film_render.halation.strength, np.array([0.12, 0.06, 0.03]))
    np.testing.assert_allclose(params.film_render.halation.scattering_strength, np.array([0.08, 0.04, 0.02]))


def test_build_params_maps_runtime_strings() -> None:
    state = make_state()
    state.simulation.auto_exposure_method = "median"
    state.input_image.input_color_space = "Display P3"
    state.input_image.spectral_upsampling_method = "mallett2019"
    state.simulation.output_color_space = "ACES2065-1"
    state.simulation.saving_cctf_encoding = False

    params = build_params_from_state(state)

    assert params.camera.auto_exposure_method == "median"
    assert params.io.input_color_space == "Display P3"
    assert params.settings.rgb_to_raw_method == "mallett2019"
    assert params.io.output_color_space == "ACES2065-1"
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
    assert state.simulation.output_color_space == "sRGB"
    assert state.simulation.saving_color_space == "sRGB"
    assert state.simulation.saving_cctf_encoding is True
    assert state.display.use_display_transform is True
    assert state.display.gray_18_canvas is True
    assert state.simulation.auto_exposure_method == "center_weighted"
    assert state.display.white_padding == 0.03


def test_project_default_gui_state_matches_builder() -> None:
    built_state = build_default_gui_state(
        film_stock=DEFAULT_FILM_STOCK,
        print_paper=DEFAULT_PRINT_PAPER,
    )

    assert PROJECT_DEFAULT_GUI_STATE == built_state