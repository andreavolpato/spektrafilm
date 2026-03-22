from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from spectral_film_lab.gui.options import AutoExposureMethods, RGBColorSpaces, RGBtoRAWMethod
from spectral_film_lab.gui.state import (
    CouplersState,
    DEFAULT_FILM_STOCK,
    DEFAULT_PRINT_PAPER,
    GlareState,
    GrainState,
    GuiState,
    HalationState,
    InputImageState,
    PreflashingState,
    PROJECT_DEFAULT_GUI_STATE,
    SimulationState,
    SpecialState,
)
from spectral_film_lab.model.illuminants import Illuminants
from spectral_film_lab.model.stocks import FilmStocks, PrintPapers


def _enum_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    return value


def _enum_member_by_value(enum_cls: type[Enum], value: str) -> Enum:
    for member in enum_cls:
        if member.value == value:
            return member
    raise ValueError(f"{value!r} is not a valid {enum_cls.__name__} value")


DEFAULT_GUI_STATE = PROJECT_DEFAULT_GUI_STATE
DEFAULT_FILM_STOCK_MEMBER = _enum_member_by_value(FilmStocks, DEFAULT_FILM_STOCK)
DEFAULT_PRINT_PAPER_MEMBER = _enum_member_by_value(PrintPapers, DEFAULT_PRINT_PAPER)
DEFAULT_INPUT_COLOR_SPACE = _enum_member_by_value(RGBColorSpaces, DEFAULT_GUI_STATE.input_image.input_color_space)
DEFAULT_SPECTRAL_UPSAMPLING_METHOD = _enum_member_by_value(RGBtoRAWMethod, DEFAULT_GUI_STATE.input_image.spectral_upsampling_method)
DEFAULT_AUTO_EXPOSURE_METHOD = _enum_member_by_value(AutoExposureMethods, DEFAULT_GUI_STATE.simulation.auto_exposure_method)
DEFAULT_PRINT_ILLUMINANT = _enum_member_by_value(Illuminants, DEFAULT_GUI_STATE.simulation.print_illuminant)
DEFAULT_OUTPUT_COLOR_SPACE = _enum_member_by_value(RGBColorSpaces, DEFAULT_GUI_STATE.simulation.output_color_space)


@dataclass(slots=True)
class MagicGuiWidgets:
    input_image: Any
    grain: Any
    preflashing: Any
    halation: Any
    couplers: Any
    glare: Any
    special: Any
    simulation: Any


def _apply_input_image_state(widgets: MagicGuiWidgets, state: InputImageState) -> None:
    widgets.input_image.preview_resize_factor.value = state.preview_resize_factor
    widgets.input_image.upscale_factor.value = state.upscale_factor
    widgets.input_image.crop.value = state.crop
    widgets.input_image.crop_center.value = state.crop_center
    widgets.input_image.crop_size.value = state.crop_size
    widgets.input_image.input_color_space.value = _enum_member_by_value(RGBColorSpaces, state.input_color_space)
    widgets.input_image.apply_cctf_decoding.value = state.apply_cctf_decoding
    widgets.input_image.spectral_upsampling_method.value = _enum_member_by_value(RGBtoRAWMethod, state.spectral_upsampling_method)
    widgets.input_image.filter_uv.value = state.filter_uv
    widgets.input_image.filter_ir.value = state.filter_ir


def _apply_grain_state(widgets: MagicGuiWidgets, state: GrainState) -> None:
    widgets.grain.active.value = state.active
    widgets.grain.sublayers_active.value = state.sublayers_active
    widgets.grain.particle_area_um2.value = state.particle_area_um2
    widgets.grain.particle_scale.value = state.particle_scale
    widgets.grain.particle_scale_layers.value = state.particle_scale_layers
    widgets.grain.density_min.value = state.density_min
    widgets.grain.uniformity.value = state.uniformity
    widgets.grain.blur.value = state.blur
    widgets.grain.blur_dye_clouds_um.value = state.blur_dye_clouds_um
    widgets.grain.micro_structure.value = state.micro_structure


def _apply_preflashing_state(widgets: MagicGuiWidgets, state: PreflashingState) -> None:
    widgets.preflashing.exposure.value = state.exposure
    widgets.preflashing.y_filter_shift.value = state.y_filter_shift
    widgets.preflashing.m_filter_shift.value = state.m_filter_shift
    widgets.preflashing.just_preflash.value = state.just_preflash


def _apply_halation_state(widgets: MagicGuiWidgets, state: HalationState) -> None:
    widgets.halation.active.value = state.active
    widgets.halation.scattering_strength.value = state.scattering_strength
    widgets.halation.scattering_size_um.value = state.scattering_size_um
    widgets.halation.halation_strength.value = state.halation_strength
    widgets.halation.halation_size_um.value = state.halation_size_um


def _apply_couplers_state(widgets: MagicGuiWidgets, state: CouplersState) -> None:
    widgets.couplers.active.value = state.active
    widgets.couplers.dir_couplers_amount.value = state.dir_couplers_amount
    widgets.couplers.dir_couplers_ratio.value = state.dir_couplers_ratio
    widgets.couplers.dir_couplers_diffusion_um.value = state.dir_couplers_diffusion_um
    widgets.couplers.diffusion_interlayer.value = state.diffusion_interlayer
    widgets.couplers.high_exposure_shift.value = state.high_exposure_shift


def _apply_glare_state(widgets: MagicGuiWidgets, state: GlareState) -> None:
    widgets.glare.active.value = state.active
    widgets.glare.percent.value = state.percent
    widgets.glare.roughness.value = state.roughness
    widgets.glare.blur.value = state.blur
    widgets.glare.compensation_removal_factor.value = state.compensation_removal_factor
    widgets.glare.compensation_removal_density.value = state.compensation_removal_density
    widgets.glare.compensation_removal_transition.value = state.compensation_removal_transition


def _apply_special_state(widgets: MagicGuiWidgets, state: SpecialState) -> None:
    widgets.special.film_channel_swap.value = state.film_channel_swap
    widgets.special.film_gamma_factor.value = state.film_gamma_factor
    widgets.special.print_channel_swap.value = state.print_channel_swap
    widgets.special.print_gamma_factor.value = state.print_gamma_factor
    widgets.special.print_density_min_factor.value = state.print_density_min_factor


def _apply_simulation_state(widgets: MagicGuiWidgets, state: SimulationState) -> None:
    widgets.simulation.film_stock.value = _enum_member_by_value(FilmStocks, state.film_stock)
    widgets.simulation.film_format_mm.value = state.film_format_mm
    widgets.simulation.camera_lens_blur_um.value = state.camera_lens_blur_um
    widgets.simulation.exposure_compensation_ev.value = state.exposure_compensation_ev
    widgets.simulation.auto_exposure.value = state.auto_exposure
    widgets.simulation.auto_exposure_method.value = _enum_member_by_value(AutoExposureMethods, state.auto_exposure_method)
    widgets.simulation.print.value = _enum_member_by_value(PrintPapers, state.print_paper)
    widgets.simulation.print_illuminant.value = _enum_member_by_value(Illuminants, state.print_illuminant)
    widgets.simulation.print_exposure.value = state.print_exposure
    widgets.simulation.print_exposure_compensation.value = state.print_exposure_compensation
    widgets.simulation.print_y_filter_shift.value = state.print_y_filter_shift
    widgets.simulation.print_m_filter_shift.value = state.print_m_filter_shift
    widgets.simulation.scan_lens_blur.value = state.scan_lens_blur
    widgets.simulation.scan_unsharp_mask.value = state.scan_unsharp_mask
    widgets.simulation.output_color_space.value = _enum_member_by_value(RGBColorSpaces, state.output_color_space)
    widgets.simulation.output_cctf_encoding.value = state.output_cctf_encoding
    widgets.simulation.scan_film.value = state.scan_film
    widgets.simulation.compute_full_image.value = state.compute_full_image


def apply_gui_state(state: GuiState, *, widgets: MagicGuiWidgets) -> None:
    _apply_input_image_state(widgets, state.input_image)
    _apply_grain_state(widgets, state.grain)
    _apply_preflashing_state(widgets, state.preflashing)
    _apply_halation_state(widgets, state.halation)
    _apply_couplers_state(widgets, state.couplers)
    _apply_glare_state(widgets, state.glare)
    _apply_special_state(widgets, state.special)
    _apply_simulation_state(widgets, state.simulation)


def _collect_input_image_state(widgets: MagicGuiWidgets) -> InputImageState:
    return InputImageState(
        preview_resize_factor=widgets.input_image.preview_resize_factor.value,
        upscale_factor=widgets.input_image.upscale_factor.value,
        crop=widgets.input_image.crop.value,
        crop_center=tuple(widgets.input_image.crop_center.value),
        crop_size=tuple(widgets.input_image.crop_size.value),
        input_color_space=_enum_value(widgets.input_image.input_color_space.value),
        apply_cctf_decoding=widgets.input_image.apply_cctf_decoding.value,
        spectral_upsampling_method=_enum_value(widgets.input_image.spectral_upsampling_method.value),
        filter_uv=tuple(widgets.input_image.filter_uv.value),
        filter_ir=tuple(widgets.input_image.filter_ir.value),
    )


def _collect_grain_state(widgets: MagicGuiWidgets) -> GrainState:
    return GrainState(
        active=widgets.grain.active.value,
        sublayers_active=widgets.grain.sublayers_active.value,
        particle_area_um2=widgets.grain.particle_area_um2.value,
        particle_scale=tuple(widgets.grain.particle_scale.value),
        particle_scale_layers=tuple(widgets.grain.particle_scale_layers.value),
        density_min=tuple(widgets.grain.density_min.value),
        uniformity=tuple(widgets.grain.uniformity.value),
        blur=widgets.grain.blur.value,
        blur_dye_clouds_um=widgets.grain.blur_dye_clouds_um.value,
        micro_structure=tuple(widgets.grain.micro_structure.value),
    )


def _collect_preflashing_state(widgets: MagicGuiWidgets) -> PreflashingState:
    return PreflashingState(
        exposure=widgets.preflashing.exposure.value,
        y_filter_shift=widgets.preflashing.y_filter_shift.value,
        m_filter_shift=widgets.preflashing.m_filter_shift.value,
        just_preflash=widgets.preflashing.just_preflash.value,
    )


def _collect_halation_state(widgets: MagicGuiWidgets) -> HalationState:
    return HalationState(
        active=widgets.halation.active.value,
        scattering_strength=tuple(widgets.halation.scattering_strength.value),
        scattering_size_um=tuple(widgets.halation.scattering_size_um.value),
        halation_strength=tuple(widgets.halation.halation_strength.value),
        halation_size_um=tuple(widgets.halation.halation_size_um.value),
    )


def _collect_couplers_state(widgets: MagicGuiWidgets) -> CouplersState:
    return CouplersState(
        active=widgets.couplers.active.value,
        dir_couplers_amount=widgets.couplers.dir_couplers_amount.value,
        dir_couplers_ratio=tuple(widgets.couplers.dir_couplers_ratio.value),
        dir_couplers_diffusion_um=widgets.couplers.dir_couplers_diffusion_um.value,
        diffusion_interlayer=widgets.couplers.diffusion_interlayer.value,
        high_exposure_shift=widgets.couplers.high_exposure_shift.value,
    )


def _collect_glare_state(widgets: MagicGuiWidgets) -> GlareState:
    return GlareState(
        active=widgets.glare.active.value,
        percent=widgets.glare.percent.value,
        roughness=widgets.glare.roughness.value,
        blur=widgets.glare.blur.value,
        compensation_removal_factor=widgets.glare.compensation_removal_factor.value,
        compensation_removal_density=widgets.glare.compensation_removal_density.value,
        compensation_removal_transition=widgets.glare.compensation_removal_transition.value,
    )


def _collect_special_state(widgets: MagicGuiWidgets) -> SpecialState:
    return SpecialState(
        film_channel_swap=tuple(widgets.special.film_channel_swap.value),
        film_gamma_factor=widgets.special.film_gamma_factor.value,
        print_channel_swap=tuple(widgets.special.print_channel_swap.value),
        print_gamma_factor=widgets.special.print_gamma_factor.value,
        print_density_min_factor=widgets.special.print_density_min_factor.value,
    )


def _collect_simulation_state(
    film_stock: Any,
    film_format_mm: float,
    camera_lens_blur_um: float,
    exposure_compensation_ev: float,
    auto_exposure: bool,
    auto_exposure_method: Any,
    print_paper: Any,
    print_illuminant: Any,
    print_exposure: float,
    print_exposure_compensation: bool,
    print_y_filter_shift: float,
    print_m_filter_shift: float,
    scan_lens_blur: float,
    scan_unsharp_mask: tuple[float, float],
    output_color_space: Any,
    output_cctf_encoding: bool,
    scan_film: bool,
    compute_full_image: bool,
) -> SimulationState:
    return SimulationState(
        film_stock=_enum_value(film_stock),
        film_format_mm=film_format_mm,
        camera_lens_blur_um=camera_lens_blur_um,
        exposure_compensation_ev=exposure_compensation_ev,
        auto_exposure=auto_exposure,
        auto_exposure_method=_enum_value(auto_exposure_method),
        print_paper=_enum_value(print_paper),
        print_illuminant=_enum_value(print_illuminant),
        print_exposure=print_exposure,
        print_exposure_compensation=print_exposure_compensation,
        print_y_filter_shift=print_y_filter_shift,
        print_m_filter_shift=print_m_filter_shift,
        scan_lens_blur=scan_lens_blur,
        scan_unsharp_mask=tuple(scan_unsharp_mask),
        output_color_space=_enum_value(output_color_space),
        output_cctf_encoding=output_cctf_encoding,
        scan_film=scan_film,
        compute_full_image=compute_full_image,
    )


def collect_gui_state(
    *,
    widgets: MagicGuiWidgets,
    film_stock: Any,
    film_format_mm: float,
    camera_lens_blur_um: float,
    exposure_compensation_ev: float,
    auto_exposure: bool,
    auto_exposure_method: Any,
    print_paper: Any,
    print_illuminant: Any,
    print_exposure: float,
    print_exposure_compensation: bool,
    print_y_filter_shift: float,
    print_m_filter_shift: float,
    scan_lens_blur: float,
    scan_unsharp_mask: tuple[float, float],
    output_color_space: Any,
    output_cctf_encoding: bool,
    scan_film: bool,
    compute_full_image: bool,
) -> GuiState:
    return GuiState(
        input_image=_collect_input_image_state(widgets),
        grain=_collect_grain_state(widgets),
        preflashing=_collect_preflashing_state(widgets),
        halation=_collect_halation_state(widgets),
        couplers=_collect_couplers_state(widgets),
        glare=_collect_glare_state(widgets),
        special=_collect_special_state(widgets),
        simulation=_collect_simulation_state(
            film_stock=film_stock,
            film_format_mm=film_format_mm,
            camera_lens_blur_um=camera_lens_blur_um,
            exposure_compensation_ev=exposure_compensation_ev,
            auto_exposure=auto_exposure,
            auto_exposure_method=auto_exposure_method,
            print_paper=print_paper,
            print_illuminant=print_illuminant,
            print_exposure=print_exposure,
            print_exposure_compensation=print_exposure_compensation,
            print_y_filter_shift=print_y_filter_shift,
            print_m_filter_shift=print_m_filter_shift,
            scan_lens_blur=scan_lens_blur,
            scan_unsharp_mask=scan_unsharp_mask,
            output_color_space=output_color_space,
            output_cctf_encoding=output_cctf_encoding,
            scan_film=scan_film,
            compute_full_image=compute_full_image,
        ),
    )