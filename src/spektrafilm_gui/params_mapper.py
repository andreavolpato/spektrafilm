from __future__ import annotations

from dataclasses import replace

from spektrafilm_gui.state import GuiState
from spektrafilm.runtime.api import init_params
from spektrafilm.runtime.params_schema import RuntimePhotoParams


def build_params_from_state(state: GuiState) -> RuntimePhotoParams:
    params = init_params(
        film_profile=state.selection.film_stock,
        print_profile=state.selection.print_paper,
    )

    _apply_special(params, state)
    _apply_glare(params, state)
    _apply_camera(params, state)
    _apply_io(params, state)
    _apply_halation(params, state)
    _apply_grain(params, state)
    _apply_couplers(params, state)
    _apply_chemistry(params, state)
    _apply_film_chemistry(params, state)
    _apply_enlarger(params, state)
    _apply_scanner(params, state)
    _apply_settings(params, state)
    return params


def _apply_special(params: RuntimePhotoParams, state: GuiState) -> None:
    def swap_channels(profile, new_cmy_order=(0,2,1)):
        profile.data.channel_density = profile.data.channel_density[:,new_cmy_order]
        return profile
    if state.special.film_channel_swap != (0, 1, 2):
        params.film = swap_channels(params.film, state.special.film_channel_swap)
    if state.special.print_channel_swap != (0, 1, 2):
        params.print = swap_channels(params.print, state.special.print_channel_swap)


def _apply_glare(params: RuntimePhotoParams, state: GuiState) -> None:
    params.print_render.glare = replace(state.glare)


def _apply_camera(params: RuntimePhotoParams, state: GuiState) -> None:
    params.camera = replace(state.camera)
    # state.camera still carries a passthrough diffusion_filter (CameraParams
    # field, unchanged in the runtime schema). The standalone camera-diffusion
    # widget owns the real value via state.camera_diffusion, so overwrite it
    # AFTER the wholesale replace -- same fan-out reconciliation pattern as
    # _apply_enlarger uses for enlarger.diffusion_filter.
    params.camera.diffusion_filter = replace(state.camera_diffusion)


def _apply_io(params: RuntimePhotoParams, state: GuiState) -> None:
    params.io = replace(state.input_image.io)
    params.io.output_color_space = state.simulation.io.output_color_space
    params.io.output_cctf_encoding = True
    params.io.scan_film = state.simulation.io.scan_film
    params.io.input_gamut_compress = replace(state.input_gamut_compress)
    params.io.output_gamut_compress = replace(state.output_gamut_compress)


def _apply_halation(params: RuntimePhotoParams, state: GuiState) -> None:
    # state.halation is the runtime HalationParams group, edited in place by
    # the GUI in runtime units; copy so digest-time mutations don't alias it.
    params.film_render.halation = replace(state.halation)


def _apply_grain(params: RuntimePhotoParams, state: GuiState) -> None:
    params.film_render.grain = replace(state.grain)


def _apply_couplers(params: RuntimePhotoParams, state: GuiState) -> None:
    # state.couplers is the runtime DirCouplersParams group, edited in place
    # by the GUI; copy it so later digest-time mutations don't alias the
    # widget's stored group.
    params.film_render.dir_couplers = replace(state.couplers)


def _apply_chemistry(params: RuntimePhotoParams, state: GuiState) -> None:
    params.print_render.chemistry = replace(state.chemistry)


def _apply_film_chemistry(params: RuntimePhotoParams, state: GuiState) -> None:
    params.film_render.chemistry = replace(state.film_chemistry)


def _apply_enlarger(params: RuntimePhotoParams, state: GuiState) -> None:
    params.enlarger.illuminant = state.simulation.enlarger.illuminant
    params.enlarger.print_exposure = state.simulation.enlarger.print_exposure
    params.enlarger.print_exposure_compensation = state.simulation.enlarger.print_exposure_compensation
    params.enlarger.y_filter_shift = state.simulation.enlarger.y_filter_shift
    params.enlarger.m_filter_shift = state.simulation.enlarger.m_filter_shift
    params.enlarger.diffusion_filter = replace(state.enlarger_diffusion)
    params.enlarger.preflash_exposure = state.preflashing.preflash_exposure
    params.enlarger.preflash_y_filter_shift = state.preflashing.preflash_y_filter_shift
    params.enlarger.preflash_m_filter_shift = state.preflashing.preflash_m_filter_shift


def _apply_scanner(params: RuntimePhotoParams, state: GuiState) -> None:
    params.scanner = replace(state.scanner)


def _apply_settings(params: RuntimePhotoParams, state: GuiState) -> None:
    params.settings = replace(state.input_image.settings)
    params.settings.preview_max_size = state.gui_only.display.settings.preview_max_size
    params.settings.use_enlarger_lut = True
    params.settings.use_scanner_lut = True
    params.settings.lut_resolution = 17
    params.settings.use_fast_stats = True
