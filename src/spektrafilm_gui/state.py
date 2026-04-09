from __future__ import annotations

from dataclasses import dataclass, is_dataclass, replace
from typing import TypeVar

from spektrafilm.model.stocks import FilmStocks, PrintPapers
from spektrafilm.runtime.api import digest_params, init_params
from spektrafilm.runtime.params_schema import RuntimePhotoParams


StateSection = TypeVar('StateSection')


@dataclass(slots=True)
class InputImageState:
    upscale_factor: float
    crop: bool
    crop_center: tuple[float, float]
    crop_size: tuple[float, float]
    input_color_space: str
    apply_cctf_decoding: bool
    spectral_upsampling_method: str
    filter_uv: tuple[float, float, float]
    filter_ir: tuple[float, float, float]


@dataclass(slots=True)
class LoadRawState:
    white_balance: str
    temperature: float
    tint: float
    lens_correction: bool


@dataclass(slots=True)
class GrainState:
    active: bool
    sublayers_active: bool
    particle_area_um2: float
    particle_scale: tuple[float, float, float]
    particle_scale_layers: tuple[float, float, float]
    density_min: tuple[float, float, float]
    uniformity: tuple[float, float, float]
    blur: float
    blur_dye_clouds_um: float
    micro_structure: tuple[float, float]


@dataclass(slots=True)
class PreflashingState:
    exposure: float
    y_filter_shift: float
    m_filter_shift: float


@dataclass(slots=True)
class HalationState:
    active: bool
    scattering_strength: tuple[float, float, float]
    scattering_size_um: tuple[float, float, float]
    halation_strength: tuple[float, float, float]
    halation_size_um: tuple[float, float, float]


@dataclass(slots=True)
class CouplersState:
    active: bool
    dir_couplers_amount: float
    dir_couplers_ratio: tuple[float, float, float]
    dir_couplers_diffusion_um: float
    diffusion_interlayer: float
    high_exposure_shift: float


@dataclass(slots=True)
class GlareState:
    active: bool
    percent: float
    roughness: float
    blur: float


@dataclass(slots=True)
class SpecialState:
    film_channel_swap: tuple[int, int, int]
    film_gamma_factor: float
    print_channel_swap: tuple[int, int, int]
    print_gamma_factor: float


@dataclass(slots=True)
class SimulationState:
    film_stock: str
    film_format_mm: float
    camera_lens_blur_um: float
    exposure_compensation_ev: float
    auto_exposure: bool
    auto_exposure_method: str
    print_paper: str
    print_illuminant: str
    print_exposure: float
    print_exposure_compensation: bool
    print_y_filter_shift: float
    print_m_filter_shift: float
    diffusion_strength: float
    diffusion_spatial_scale: float
    diffusion_intensity: float
    scan_lens_blur: float
    scan_white_correction: float
    scan_black_correction: float
    scan_unsharp_mask: tuple[float, float]
    output_color_space: str
    saving_color_space: str
    saving_cctf_encoding: bool
    auto_preview: bool
    scan_film: bool


@dataclass(slots=True)
class DisplayState:
    use_display_transform: bool
    gray_18_canvas: bool
    white_padding: float
    preview_max_size: int


@dataclass(slots=True)
class GuiState:
    input_image: InputImageState
    load_raw: LoadRawState
    grain: GrainState
    preflashing: PreflashingState
    halation: HalationState
    couplers: CouplersState
    glare: GlareState
    special: SpecialState
    simulation: SimulationState
    display: DisplayState


def clone_state_section(section: StateSection) -> StateSection:
    if not is_dataclass(section):
        raise TypeError('Expected a dataclass instance to clone.')
    return replace(section)


def clone_gui_state(state: GuiState) -> GuiState:
    return GuiState(
        input_image=clone_state_section(state.input_image),
        load_raw=clone_state_section(state.load_raw),
        grain=clone_state_section(state.grain),
        preflashing=clone_state_section(state.preflashing),
        halation=clone_state_section(state.halation),
        couplers=clone_state_section(state.couplers),
        glare=clone_state_section(state.glare),
        special=clone_state_section(state.special),
        simulation=clone_state_section(state.simulation),
        display=clone_state_section(state.display),
    )


def gui_state_from_params(
    params: RuntimePhotoParams,
    *,
    film_stock: str,
    print_paper: str,
) -> GuiState:
    return GuiState(
        input_image=InputImageState(
            upscale_factor=params.io.upscale_factor,
            crop=params.io.crop,
            crop_center=tuple(params.io.crop_center),
            crop_size=tuple(params.io.crop_size),
            input_color_space=params.io.input_color_space,
            apply_cctf_decoding=params.io.input_cctf_decoding,
            spectral_upsampling_method=params.settings.rgb_to_raw_method,
            filter_uv=tuple(params.camera.filter_uv),
            filter_ir=tuple(params.camera.filter_ir),
        ),
        load_raw=LoadRawState(
            white_balance='as_shot',
            temperature=5500.0,
            tint=1.0,
            lens_correction=False,
        ),
        grain=GrainState(
            active=params.film_render.grain.active,
            sublayers_active=params.film_render.grain.sublayers_active,
            particle_area_um2=params.film_render.grain.agx_particle_area_um2,
            particle_scale=tuple(params.film_render.grain.agx_particle_scale),
            particle_scale_layers=tuple(params.film_render.grain.agx_particle_scale_layers),
            density_min=tuple(params.film_render.grain.density_min),
            uniformity=tuple(params.film_render.grain.uniformity),
            blur=params.film_render.grain.blur,
            blur_dye_clouds_um=params.film_render.grain.blur_dye_clouds_um,
            micro_structure=tuple(params.film_render.grain.micro_structure),
        ),
        preflashing=PreflashingState(
            exposure=params.enlarger.preflash_exposure,
            y_filter_shift=params.enlarger.preflash_y_filter_shift,
            m_filter_shift=params.enlarger.preflash_m_filter_shift,
        ),
        halation=HalationState(
            active=params.film_render.halation.active,
            scattering_strength=tuple(value * 100.0 for value in params.film_render.halation.scattering_strength),
            scattering_size_um=tuple(params.film_render.halation.scattering_size_um),
            halation_strength=tuple(value * 100.0 for value in params.film_render.halation.strength),
            halation_size_um=tuple(params.film_render.halation.size_um),
        ),
        couplers=CouplersState(
            active=params.film_render.dir_couplers.active,
            dir_couplers_amount=params.film_render.dir_couplers.amount,
            dir_couplers_ratio=tuple(params.film_render.dir_couplers.ratio_rgb),
            dir_couplers_diffusion_um=params.film_render.dir_couplers.diffusion_size_um,
            diffusion_interlayer=params.film_render.dir_couplers.diffusion_interlayer,
            high_exposure_shift=params.film_render.dir_couplers.high_exposure_shift,
        ),
        glare=GlareState(
            active=params.print_render.glare.active,
            percent=params.print_render.glare.percent,
            roughness=params.print_render.glare.roughness,
            blur=params.print_render.glare.blur,
        ),
        special=SpecialState(
            film_channel_swap=(0, 1, 2),
            film_gamma_factor=params.film_render.density_curve_gamma,
            print_channel_swap=(0, 1, 2),
            print_gamma_factor=params.print_render.density_curve_gamma,
        ),
        simulation=SimulationState(
            film_stock=film_stock,
            film_format_mm=params.camera.film_format_mm,
            camera_lens_blur_um=params.camera.lens_blur_um,
            exposure_compensation_ev=params.camera.exposure_compensation_ev,
            auto_exposure=params.camera.auto_exposure,
            auto_exposure_method=params.camera.auto_exposure_method,
            print_paper=print_paper,
            print_illuminant=params.enlarger.illuminant,
            print_exposure=params.enlarger.print_exposure,
            print_exposure_compensation=params.enlarger.print_exposure_compensation,
            print_y_filter_shift=params.enlarger.y_filter_shift,
            print_m_filter_shift=params.enlarger.m_filter_shift,
            diffusion_strength=float(params.enlarger.diffusion_filter[0]),
            diffusion_spatial_scale=float(params.enlarger.diffusion_filter[1]),
            diffusion_intensity=float(params.enlarger.diffusion_filter[2]),
            scan_lens_blur=params.scanner.lens_blur,
            scan_white_correction=params.scanner.white_correction,
            scan_black_correction=params.scanner.black_correction,
            scan_unsharp_mask=tuple(params.scanner.unsharp_mask),
            output_color_space="sRGB",
            saving_color_space="sRGB",
            saving_cctf_encoding=params.io.output_cctf_encoding,
            auto_preview=True,
            scan_film=params.io.scan_film,
        ),
        display=DisplayState(
            use_display_transform=True,
            gray_18_canvas=True,
            white_padding=0.03,
            preview_max_size=params.settings.preview_max_size,
        ),
    )


def digest_after_selection(params: RuntimePhotoParams) -> RuntimePhotoParams:
    params = digest_params(params)
    params.io.scan_film = bool(params.film.is_positive)
    return params


def build_default_gui_state(*, film_stock: str, print_paper: str) -> GuiState:
    params = digest_after_selection(init_params(film_profile=film_stock, print_profile=print_paper))
    return gui_state_from_params(params, film_stock=film_stock, print_paper=print_paper)


DEFAULT_FILM_STOCK = FilmStocks.kodak_gold_200.value
DEFAULT_PRINT_PAPER = PrintPapers.kodak_supra_endura.value
PROJECT_DEFAULT_GUI_STATE = build_default_gui_state(
    film_stock=DEFAULT_FILM_STOCK,
    print_paper=DEFAULT_PRINT_PAPER,
)