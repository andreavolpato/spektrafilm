from __future__ import annotations

import numpy as np

from profiles_creator.factory import swap_channels
from spectral_film_lab.gui.state import GuiState
from spectral_film_lab.runtime.params_schema import RuntimePhotoParams
from spectral_film_lab.runtime.process import photo_params


def build_params_from_state(state: GuiState) -> RuntimePhotoParams:
    params = photo_params(
        film_profile=state.simulation.film_stock,
        print_profile=state.simulation.print_paper,
    )

    _apply_special(params, state)
    _apply_glare(params, state)
    _apply_camera(params, state)
    _apply_io(params, state)
    _apply_halation(params, state)
    _apply_grain(params, state)
    _apply_couplers(params, state)
    _apply_enlarger(params, state)
    _apply_scanner(params, state)
    _apply_settings(params, state)
    return params


def _apply_special(params: RuntimePhotoParams, state: GuiState) -> None:
    if state.special.film_channel_swap != (0, 1, 2):
        params.film = swap_channels(params.film, state.special.film_channel_swap)
    if state.special.print_channel_swap != (0, 1, 2):
        params.print = swap_channels(params.print, state.special.print_channel_swap)

    params.film_render.density_curve_gamma = state.special.film_gamma_factor
    params.print_render.density_curve_gamma = state.special.print_gamma_factor
    params.print_render.base_density_scale = state.special.print_density_min_factor


def _apply_glare(params: RuntimePhotoParams, state: GuiState) -> None:
    params.print_render.glare.active = state.glare.active
    params.print_render.glare.percent = state.glare.percent
    params.print_render.glare.roughness = state.glare.roughness
    params.print_render.glare.blur = state.glare.blur
    params.print_render.glare.compensation_removal_factor = state.glare.compensation_removal_factor
    params.print_render.glare.compensation_removal_density = state.glare.compensation_removal_density
    params.print_render.glare.compensation_removal_transition = state.glare.compensation_removal_transition


def _apply_camera(params: RuntimePhotoParams, state: GuiState) -> None:
    params.camera.lens_blur_um = state.simulation.camera_lens_blur_um
    params.camera.exposure_compensation_ev = state.simulation.exposure_compensation_ev
    params.camera.auto_exposure = state.simulation.auto_exposure
    params.camera.auto_exposure_method = state.simulation.auto_exposure_method
    params.camera.film_format_mm = state.simulation.film_format_mm
    params.camera.filter_uv = state.input_image.filter_uv
    params.camera.filter_ir = state.input_image.filter_ir


def _apply_io(params: RuntimePhotoParams, state: GuiState) -> None:
    params.io.preview_resize_factor = state.input_image.preview_resize_factor
    params.io.upscale_factor = state.input_image.upscale_factor
    params.io.crop = state.input_image.crop
    params.io.crop_center = state.input_image.crop_center
    params.io.crop_size = state.input_image.crop_size
    params.io.input_color_space = state.input_image.input_color_space
    params.io.input_cctf_decoding = state.input_image.apply_cctf_decoding
    params.io.output_color_space = state.simulation.output_color_space
    params.io.output_cctf_encoding = state.simulation.output_cctf_encoding
    params.io.full_image = state.simulation.compute_full_image
    params.io.scan_film = state.simulation.scan_film


def _apply_halation(params: RuntimePhotoParams, state: GuiState) -> None:
    params.film_render.halation.active = state.halation.active
    params.film_render.halation.strength = np.array(state.halation.halation_strength) / 100.0
    params.film_render.halation.size_um = np.array(state.halation.halation_size_um)
    params.film_render.halation.scattering_strength = np.array(state.halation.scattering_strength) / 100.0
    params.film_render.halation.scattering_size_um = np.array(state.halation.scattering_size_um)


def _apply_grain(params: RuntimePhotoParams, state: GuiState) -> None:
    params.film_render.grain.active = state.grain.active
    params.film_render.grain.sublayers_active = state.grain.sublayers_active
    params.film_render.grain.agx_particle_area_um2 = state.grain.particle_area_um2
    params.film_render.grain.agx_particle_scale = state.grain.particle_scale
    params.film_render.grain.agx_particle_scale_layers = state.grain.particle_scale_layers
    params.film_render.grain.density_min = state.grain.density_min
    params.film_render.grain.uniformity = state.grain.uniformity
    params.film_render.grain.blur = state.grain.blur
    params.film_render.grain.blur_dye_clouds_um = state.grain.blur_dye_clouds_um
    params.film_render.grain.micro_structure = state.grain.micro_structure


def _apply_couplers(params: RuntimePhotoParams, state: GuiState) -> None:
    params.film_render.dir_couplers.active = state.couplers.active
    params.film_render.dir_couplers.amount = state.couplers.dir_couplers_amount
    params.film_render.dir_couplers.ratio_rgb = state.couplers.dir_couplers_ratio
    params.film_render.dir_couplers.diffusion_size_um = state.couplers.dir_couplers_diffusion_um
    params.film_render.dir_couplers.diffusion_interlayer = state.couplers.diffusion_interlayer
    params.film_render.dir_couplers.high_exposure_shift = state.couplers.high_exposure_shift


def _apply_enlarger(params: RuntimePhotoParams, state: GuiState) -> None:
    params.enlarger.illuminant = state.simulation.print_illuminant
    params.enlarger.print_exposure = state.simulation.print_exposure
    params.enlarger.print_exposure_compensation = state.simulation.print_exposure_compensation
    params.enlarger.y_filter_shift = state.simulation.print_y_filter_shift
    params.enlarger.m_filter_shift = state.simulation.print_m_filter_shift
    params.enlarger.preflash_exposure = state.preflashing.exposure
    params.enlarger.preflash_y_filter_shift = state.preflashing.y_filter_shift
    params.enlarger.preflash_m_filter_shift = state.preflashing.m_filter_shift
    params.enlarger.just_preflash = state.preflashing.just_preflash


def _apply_scanner(params: RuntimePhotoParams, state: GuiState) -> None:
    params.scanner.lens_blur = state.simulation.scan_lens_blur
    params.scanner.unsharp_mask = state.simulation.scan_unsharp_mask


def _apply_settings(params: RuntimePhotoParams, state: GuiState) -> None:
    params.settings.rgb_to_raw_method = state.input_image.spectral_upsampling_method
    params.settings.use_enlarger_lut = True
    params.settings.use_scanner_lut = True
    params.settings.lut_resolution = 32
    params.settings.use_fast_stats = True