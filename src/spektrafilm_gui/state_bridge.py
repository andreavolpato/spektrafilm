from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, cast

from spektrafilm_gui.state import (
    GuiState,
    PROJECT_DEFAULT_GUI_STATE,
    clone_gui_state,
)
from spektrafilm_gui.widgets import WidgetBundle


class SupportsSectionState(Protocol):
    def set_state(self, state: object) -> None:
        ...

    def get_state(self) -> object:
        ...


DEFAULT_GUI_STATE = PROJECT_DEFAULT_GUI_STATE
GUI_STATE_SECTION_NAMES = (
    'display',
    'input_image',
    'load_raw',
    'grain',
    'preflashing',
    'halation',
    'couplers',
    'chemistry',
    'print_base',
    'film_chemistry',
    'film_base',
    'camera',
    'enlarger_diffusion',
    'camera_diffusion',
    'glare',
    'scanner',
    'input_gamut_compress',
    'output_gamut_compress',
    'special',
    'simulation',
)


@dataclass(frozen=True)
class SectionStateAccessor:
    get: Callable[[GuiState], object]
    set: Callable[[GuiState, object], None]


def _top_level_section_accessor(section_name: str) -> SectionStateAccessor:
    return SectionStateAccessor(
        get=lambda state: getattr(state, section_name),
        set=lambda state, value: setattr(state, section_name, value),
    )


SECTION_STATE_ACCESSORS = {
    'display': SectionStateAccessor(
        get=lambda state: state.gui_only.display,
        set=lambda state, value: setattr(state.gui_only, 'display', value),
    ),
    'input_image': _top_level_section_accessor('input_image'),
    'load_raw': SectionStateAccessor(
        get=lambda state: state.gui_only.load_raw,
        set=lambda state, value: setattr(state.gui_only, 'load_raw', value),
    ),
    'grain': _top_level_section_accessor('grain'),
    'preflashing': _top_level_section_accessor('preflashing'),
    'halation': _top_level_section_accessor('halation'),
    'couplers': _top_level_section_accessor('couplers'),
    'chemistry': _top_level_section_accessor('chemistry'),
    'print_base': _top_level_section_accessor('print_base'),
    'film_chemistry': _top_level_section_accessor('film_chemistry'),
    'film_base': _top_level_section_accessor('film_base'),
    'camera': _top_level_section_accessor('camera'),
    'enlarger_diffusion': _top_level_section_accessor('enlarger_diffusion'),
    'camera_diffusion': _top_level_section_accessor('camera_diffusion'),
    'glare': _top_level_section_accessor('glare'),
    'scanner': _top_level_section_accessor('scanner'),
    'input_gamut_compress': _top_level_section_accessor('input_gamut_compress'),
    'output_gamut_compress': _top_level_section_accessor('output_gamut_compress'),
    'special': _top_level_section_accessor('special'),
    'simulation': _top_level_section_accessor('simulation'),
}


def _get_stateful_widget(widgets: WidgetBundle, section_name: str) -> SupportsSectionState:
    return cast(SupportsSectionState, getattr(widgets, section_name))


def _get_section_state(state: GuiState, section_name: str) -> object:
    return SECTION_STATE_ACCESSORS[section_name].get(state)


def _set_section_state(state: GuiState, section_name: str, value: object) -> None:
    SECTION_STATE_ACCESSORS[section_name].set(state, value)


def apply_gui_state(state: GuiState, *, widgets: WidgetBundle) -> None:
    apply_gui_state_sections(state, widgets=widgets, section_names=GUI_STATE_SECTION_NAMES)


def apply_gui_state_sections(
    state: GuiState,
    *,
    widgets: WidgetBundle,
    section_names: tuple[str, ...],
) -> None:
    for section_name in section_names:
        _get_stateful_widget(widgets, section_name).set_state(_get_section_state(state, section_name))
    if 'simulation' in section_names:
        widgets.simulation.set_auto_preview_value(state.simulation.workflow.auto_preview)
        widgets.simulation.reset_scan_for_print_value()


def collect_gui_state(
    *,
    widgets: WidgetBundle,
) -> GuiState:
    gui_state = clone_gui_state(DEFAULT_GUI_STATE)
    for section_name in GUI_STATE_SECTION_NAMES:
        _set_section_state(gui_state, section_name, _get_stateful_widget(widgets, section_name).get_state())
    gui_state.simulation.workflow.auto_preview = widgets.simulation.auto_preview_value()
    return gui_state