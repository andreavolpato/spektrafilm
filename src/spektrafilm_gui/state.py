from __future__ import annotations

from dataclasses import dataclass, field, is_dataclass, replace
from typing import Any, TypeVar

from spektrafilm.model.stocks import FilmStocks, PrintPapers
from spektrafilm.runtime.api import digest_params, init_params
from spektrafilm.runtime.params_schema import (
    CameraParams,
    DiffusionFilterParams,
    DirCouplersParams,
    EnlargerParams,
    GlareParams,
    GrainParams,
    HalationParams,
    IOParams,
    RuntimePhotoParams,
    ScannerParams,
    SettingsParams,
)
from spektrafilm.utils.gamut_compression import InputGamutCompressSpec, OutputGamutCompressSpec
from spektrafilm.utils.morph_curves import FilmChemistryParams, PrintChemistryParams
from spektrafilm_gui.params_manifest import DISPLAY_PANEL_FIELDS, INPUT_IMAGE_FIELDS, SIMULATION_FIELDS, SPECIAL_FIELDS


StateSection = TypeVar('StateSection')


@dataclass(slots=True)
class InputImageState:
    """Thin container for the two runtime groups the input_image panels edit.

    No GUI-side facades: the Input / Crop / Spectral panels bind their editors
    directly to dotted paths on this object (``io.upscale_factor``,
    ``settings.rgb_to_raw_method``) via the INPUT_IMAGE_FIELDS manifest. ``io``
    and ``settings`` mirror the runtime groups verbatim.
    """

    io: IOParams = field(default_factory=IOParams)
    settings: SettingsParams = field(default_factory=SettingsParams)


@dataclass(slots=True)
class LoadRawState:
    white_balance: str
    temperature: float
    tint: float
    lens_correction: bool


@dataclass(slots=True)
class SpecialState:
    film_channel_swap: tuple[int, int, int]
    print_channel_swap: tuple[int, int, int]


@dataclass(slots=True)
class SelectionState:
    film_stock: str
    print_paper: str


@dataclass(slots=True)
class SimulationWorkflowState:
    saving_color_space: str
    saving_cctf_encoding: bool
    auto_preview: bool


@dataclass(slots=True)
class SimulationState:
    selection: SelectionState = field(default_factory=lambda: SelectionState(film_stock='', print_paper=''))
    enlarger: EnlargerParams = field(default_factory=EnlargerParams)
    io: IOParams = field(default_factory=IOParams)
    workflow: SimulationWorkflowState = field(default_factory=lambda: SimulationWorkflowState(
        saving_color_space='sRGB',
        saving_cctf_encoding=True,
        auto_preview=True,
    ))


def _read_attr_path(root: Any, path: str) -> object:
    obj = root
    for part in path.split('.'):
        obj = getattr(obj, part)
    return obj


def input_image_to_dict(state: InputImageState) -> dict[str, object]:
    # Flat dict keyed by the runtime leaf (e.g. "input_cctf_decoding"). The
    # field-to-group mapping is derived from the INPUT_IMAGE_FIELDS manifest,
    # so the binding lives in exactly one place.
    return {spec.leaf: _read_attr_path(state, spec.path) for spec in INPUT_IMAGE_FIELDS}


def special_to_dict(state: SpecialState) -> dict[str, object]:
    return {spec.leaf: _read_attr_path(state, spec.path) for spec in SPECIAL_FIELDS}


def simulation_to_dict(state: SimulationState) -> dict[str, object]:
    return {spec.leaf: _read_attr_path(state, spec.path) for spec in SIMULATION_FIELDS}


def display_to_dict(state: DisplayState) -> dict[str, object]:
    return {spec.leaf: _read_attr_path(state, spec.path) for spec in DISPLAY_PANEL_FIELDS}


_INPUT_IMAGE_LEGACY_ALIASES = {
    'apply_cctf_decoding': 'input_cctf_decoding',
    'spectral_upsampling_method': 'rgb_to_raw_method',
}


def normalize_input_image_dict(data: dict[str, Any]) -> dict[str, Any]:
    if 'io' in data or 'settings' in data:
        return dict(data)
    # Legacy / flat dict (keyed by runtime leaf, or a pre-refactor GUI alias)
    # -> nested {io: {...}, settings: {...}}, derived from the manifest paths.
    nested: dict[str, dict[str, Any]] = {'io': {}, 'settings': {}}
    for spec in INPUT_IMAGE_FIELDS:
        group, _, leaf = spec.path.partition('.')
        legacy = [old for old, new in _INPUT_IMAGE_LEGACY_ALIASES.items() if new == leaf]
        for key in (leaf, *legacy):
            if key in data:
                nested[group][leaf] = data[key]
                break
    return nested


def normalize_special_dict(data: dict[str, Any]) -> dict[str, Any]:
    if 'film_render' in data:
        return dict(data)
    return {
        'film_channel_swap': data.get('film_channel_swap', (0, 1, 2)),
        'print_channel_swap': data.get('print_channel_swap', (0, 1, 2)),
    }


def normalize_simulation_dict(data: dict[str, Any]) -> dict[str, Any]:
    if 'enlarger' in data or 'io' in data or 'workflow' in data:
        return dict(data)
    return {
        'selection': {
            'film_stock': data.get('film_stock', PROJECT_DEFAULT_GUI_STATE.selection.film_stock),
            'print_paper': data.get('print_paper', PROJECT_DEFAULT_GUI_STATE.selection.print_paper),
        },
        'enlarger': {
            'illuminant': data.get('print_illuminant', PROJECT_DEFAULT_GUI_STATE.simulation.enlarger.illuminant),
            'print_exposure': data.get('print_exposure', PROJECT_DEFAULT_GUI_STATE.simulation.enlarger.print_exposure),
            'print_exposure_compensation': data.get('print_exposure_compensation', PROJECT_DEFAULT_GUI_STATE.simulation.enlarger.print_exposure_compensation),
            'y_filter_shift': data.get('print_y_filter_shift', PROJECT_DEFAULT_GUI_STATE.simulation.enlarger.y_filter_shift),
            'm_filter_shift': data.get('print_m_filter_shift', PROJECT_DEFAULT_GUI_STATE.simulation.enlarger.m_filter_shift),
        },
        'io': {
            'output_color_space': data.get('output_color_space', PROJECT_DEFAULT_GUI_STATE.simulation.io.output_color_space),
            'scan_film': data.get('scan_film', PROJECT_DEFAULT_GUI_STATE.simulation.io.scan_film),
        },
        'workflow': {
            'saving_color_space': data.get('saving_color_space', PROJECT_DEFAULT_GUI_STATE.simulation.workflow.saving_color_space),
            'saving_cctf_encoding': data.get('saving_cctf_encoding', PROJECT_DEFAULT_GUI_STATE.simulation.workflow.saving_cctf_encoding),
            'auto_preview': data.get('auto_preview', PROJECT_DEFAULT_GUI_STATE.simulation.workflow.auto_preview),
        },
    }


def normalize_display_dict(data: dict[str, Any]) -> dict[str, Any]:
    if 'settings' in data:
        return dict(data)
    return {
        'use_display_transform': data.get('use_display_transform', PROJECT_DEFAULT_GUI_STATE.gui_only.display.use_display_transform),
        'gray_18_canvas': data.get('gray_18_canvas', PROJECT_DEFAULT_GUI_STATE.gui_only.display.gray_18_canvas),
        'white_padding': data.get('white_padding', PROJECT_DEFAULT_GUI_STATE.gui_only.display.white_padding),
        'output_interpolation': data.get('output_interpolation', PROJECT_DEFAULT_GUI_STATE.gui_only.display.output_interpolation),
        'settings': {
            'preview_max_size': data.get('preview_max_size', PROJECT_DEFAULT_GUI_STATE.gui_only.display.settings.preview_max_size),
        },
    }


@dataclass(slots=True)
class DisplayState:
    use_display_transform: bool
    gray_18_canvas: bool
    white_padding: float
    output_interpolation: str = 'spline36'
    settings: SettingsParams = field(default_factory=SettingsParams)


@dataclass(slots=True)
class GuiOnlyState:
    load_raw: LoadRawState
    display: DisplayState


@dataclass(slots=True)
class GuiState:
    input_image: InputImageState
    grain: GrainParams
    preflashing: EnlargerParams
    halation: HalationParams
    couplers: DirCouplersParams
    chemistry: PrintChemistryParams
    film_chemistry: FilmChemistryParams
    camera: CameraParams
    enlarger_diffusion: DiffusionFilterParams
    camera_diffusion: DiffusionFilterParams
    glare: GlareParams
    scanner: ScannerParams
    input_gamut_compress: InputGamutCompressSpec
    output_gamut_compress: OutputGamutCompressSpec
    special: SpecialState
    simulation: SimulationState
    gui_only: GuiOnlyState

    @property
    def selection(self) -> SelectionState:
        return self.simulation.selection


def clone_state_section(section: StateSection) -> StateSection:
    if not is_dataclass(section):
        raise TypeError('Expected a dataclass instance to clone.')
    if isinstance(section, InputImageState):
        return replace(section, io=replace(section.io), settings=replace(section.settings))
    if isinstance(section, SimulationState):
        return replace(
            section,
            selection=replace(section.selection),
            enlarger=replace(section.enlarger),
            io=replace(section.io),
            workflow=replace(section.workflow),
        )
    if isinstance(section, DisplayState):
        return replace(section, settings=replace(section.settings))
    if isinstance(section, GuiOnlyState):
        return replace(
            section,
            load_raw=replace(section.load_raw),
            display=clone_state_section(section.display),
        )
    return replace(section)


def clone_gui_state(state: GuiState) -> GuiState:
    return GuiState(
        input_image=clone_state_section(state.input_image),
        grain=clone_state_section(state.grain),
        preflashing=clone_state_section(state.preflashing),
        halation=clone_state_section(state.halation),
        couplers=clone_state_section(state.couplers),
        chemistry=clone_state_section(state.chemistry),
        film_chemistry=clone_state_section(state.film_chemistry),
        camera=clone_state_section(state.camera),
        enlarger_diffusion=clone_state_section(state.enlarger_diffusion),
        camera_diffusion=clone_state_section(state.camera_diffusion),
        glare=clone_state_section(state.glare),
        scanner=clone_state_section(state.scanner),
        input_gamut_compress=clone_state_section(state.input_gamut_compress),
        output_gamut_compress=clone_state_section(state.output_gamut_compress),
        special=clone_state_section(state.special),
        simulation=clone_state_section(state.simulation),
        gui_only=clone_state_section(state.gui_only),
    )


def gui_state_from_params(
    params: RuntimePhotoParams,
    *,
    film_stock: str,
    print_paper: str,
) -> GuiState:
    return GuiState(
        input_image=InputImageState(
            io=replace(params.io),
            settings=replace(params.settings),
        ),
        grain=replace(params.film_render.grain),
        preflashing=replace(params.enlarger),
        halation=replace(params.film_render.halation),
        couplers=replace(params.film_render.dir_couplers),
        chemistry=replace(params.print_render.chemistry),
        film_chemistry=replace(params.film_render.chemistry),
        camera=replace(params.camera),
        enlarger_diffusion=replace(params.enlarger.diffusion_filter),
        camera_diffusion=replace(params.camera.diffusion_filter),
        glare=replace(params.print_render.glare),
        scanner=replace(params.scanner),
        input_gamut_compress=replace(params.io.input_gamut_compress),
        output_gamut_compress=replace(params.io.output_gamut_compress),
        special=SpecialState(
            film_channel_swap=(0, 1, 2),
            print_channel_swap=(0, 1, 2),
        ),
        simulation=SimulationState(
            selection=SelectionState(
                film_stock=film_stock,
                print_paper=print_paper,
            ),
            enlarger=EnlargerParams(
                illuminant=params.enlarger.illuminant,
                print_exposure=params.enlarger.print_exposure,
                print_exposure_compensation=params.enlarger.print_exposure_compensation,
                y_filter_shift=params.enlarger.y_filter_shift,
                m_filter_shift=params.enlarger.m_filter_shift,
            ),
            io=IOParams(
                output_color_space=params.io.output_color_space,
                output_cctf_encoding=params.io.output_cctf_encoding,
                scan_film=params.io.scan_film,
            ),
            workflow=SimulationWorkflowState(
                saving_color_space="sRGB",
                saving_cctf_encoding=params.io.output_cctf_encoding,
                auto_preview=True,
            ),
        ),
        gui_only=GuiOnlyState(
            load_raw=LoadRawState(
                white_balance='as_shot',
                temperature=5500.0,
                tint=1.0,
                lens_correction=False,
            ),
            display=DisplayState(
                use_display_transform=True,
                gray_18_canvas=True,
                output_interpolation='spline36',
                white_padding=0.03,
                settings=SettingsParams(preview_max_size=params.settings.preview_max_size),
            ),
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
