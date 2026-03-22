from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
from typing import TYPE_CHECKING, Any

from spectral_film_lab.gui.state import (
    CouplersState,
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
from spectral_film_lab.gui.widget_specs import GUI_SECTION_ENUMS

if TYPE_CHECKING:
    from spectral_film_lab.gui.widgets import (
        CouplersSection,
        FilePickerSection,
        GlareSection,
        GrainSection,
        HalationSection,
        InputImageSection,
        PreflashingSection,
        SimulationSection,
        SimulationInputSection,
        SpecialSection,
    )


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


@dataclass(slots=True)
class GuiWidgets:
    filepicker: FilePickerSection
    simulation_input: SimulationInputSection
    input_image: InputImageSection
    grain: GrainSection
    preflashing: PreflashingSection
    halation: HalationSection
    couplers: CouplersSection
    glare: GlareSection
    special: SpecialSection
    simulation: SimulationSection


def _normalize_widget_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return value


def _apply_section_state(section_name: str, widget_group: Any, state: Any) -> None:
    if hasattr(widget_group, "set_state"):
        widget_group.set_state(state)
        return
    enum_fields = GUI_SECTION_ENUMS.get(section_name, {})
    for field_info in fields(type(state)):
        field_name = field_info.name
        widget = getattr(widget_group, field_name)
        value = getattr(state, field_name)
        enum_cls = enum_fields.get(field_name)
        if enum_cls is not None:
            value = _enum_member_by_value(enum_cls, value)
        widget.value = value


def _collect_section_state(section_name: str, widget_group: Any, state_cls: type[Any]) -> Any:
    if hasattr(widget_group, "get_state"):
        return widget_group.get_state()
    enum_fields = GUI_SECTION_ENUMS.get(section_name, {})
    values = {}
    for field_info in fields(state_cls):
        field_name = field_info.name
        widget = getattr(widget_group, field_name)
        value = widget.value
        enum_cls = enum_fields.get(field_name)
        if enum_cls is not None:
            value = _enum_value(value)
        else:
            value = _normalize_widget_value(value)
        values[field_name] = value
    return state_cls(**values)


def apply_gui_state(state: GuiState, *, widgets: GuiWidgets) -> None:
    _apply_section_state("input_image", widgets.input_image, state.input_image)
    _apply_section_state("grain", widgets.grain, state.grain)
    _apply_section_state("preflashing", widgets.preflashing, state.preflashing)
    _apply_section_state("halation", widgets.halation, state.halation)
    _apply_section_state("couplers", widgets.couplers, state.couplers)
    _apply_section_state("glare", widgets.glare, state.glare)
    _apply_section_state("special", widgets.special, state.special)
    _apply_section_state("simulation", widgets.simulation, state.simulation)
    widgets.simulation.set_scan_film_value(state.simulation.scan_film)


def collect_gui_state(
    *,
    widgets: GuiWidgets,
) -> GuiState:
    gui_state = GuiState(
        input_image=_collect_section_state("input_image", widgets.input_image, InputImageState),
        grain=_collect_section_state("grain", widgets.grain, GrainState),
        preflashing=_collect_section_state("preflashing", widgets.preflashing, PreflashingState),
        halation=_collect_section_state("halation", widgets.halation, HalationState),
        couplers=_collect_section_state("couplers", widgets.couplers, CouplersState),
        glare=_collect_section_state("glare", widgets.glare, GlareState),
        special=_collect_section_state("special", widgets.special, SpecialState),
        simulation=_collect_section_state("simulation", widgets.simulation, SimulationState),
    )
    gui_state.simulation.scan_film = widgets.simulation.scan_film_value()
    return gui_state