from __future__ import annotations

from dataclasses import fields
from typing import Protocol, cast

from spektrafilm_gui.state import (
    GuiState,
    PROJECT_DEFAULT_GUI_STATE,
)
from spektrafilm_gui.widgets import WidgetBundle


class SupportsSectionState(Protocol):
    def set_state(self, state: object) -> None:
        ...

    def get_state(self) -> object:
        ...


DEFAULT_GUI_STATE = PROJECT_DEFAULT_GUI_STATE
GUI_STATE_SECTION_NAMES = tuple(field_info.name for field_info in fields(GuiState))


def _get_stateful_widget(widgets: WidgetBundle, section_name: str) -> SupportsSectionState:
    return cast(SupportsSectionState, getattr(widgets, section_name))


def apply_gui_state(state: GuiState, *, widgets: WidgetBundle) -> None:
    apply_gui_state_sections(state, widgets=widgets, section_names=GUI_STATE_SECTION_NAMES)


def apply_gui_state_sections(
    state: GuiState,
    *,
    widgets: WidgetBundle,
    section_names: tuple[str, ...],
) -> None:
    for section_name in section_names:
        _get_stateful_widget(widgets, section_name).set_state(getattr(state, section_name))
    if 'simulation' in section_names:
        widgets.simulation.set_auto_preview_value(state.simulation.auto_preview)
        widgets.simulation.set_scan_film_value(state.simulation.scan_film)
        widgets.simulation.reset_scan_for_print_value()


def collect_gui_state(
    *,
    widgets: WidgetBundle,
) -> GuiState:
    gui_state = GuiState(**{section_name: _get_stateful_widget(widgets, section_name).get_state() for section_name in GUI_STATE_SECTION_NAMES})
    gui_state.simulation.auto_preview = widgets.simulation.auto_preview_value()
    gui_state.simulation.scan_film = widgets.simulation.scan_film_value()
    return gui_state