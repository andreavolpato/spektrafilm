from __future__ import annotations

from dataclasses import fields

from spektrafilm_gui.state import GuiState, clone_state_section
from spektrafilm_gui.state_bridge import GUI_STATE_SECTION_NAMES, GuiWidgets, apply_gui_state, collect_gui_state
from tests.gui_test_utils import make_gui_state


class StubSection:
    def __init__(self, state: object):
        self._state = state

    def set_state(self, state: object) -> None:
        self._state = state

    def get_state(self) -> object:
        return self._state


class StubSimulationSection(StubSection):
    def __init__(self, state: object, *, scan_film: bool = False):
        super().__init__(state)
        self._scan_film = scan_film

    def set_scan_film_value(self, value: bool) -> None:
        self._scan_film = value

    def scan_film_value(self) -> bool:
        return self._scan_film


def _make_state() -> GuiState:
    state = make_gui_state()
    state.input_image.preview_resize_factor = 0.45
    state.input_image.upscale_factor = 1.5
    state.grain.active = False
    state.preflashing.just_preflash = True
    state.halation.halation_strength = (7.0, 5.0, 3.0)
    state.couplers.diffusion_interlayer = 1.75
    state.glare.blur = 0.8
    state.special.print_gamma_factor = 1.15
    state.simulation.print_exposure = 1.3
    state.simulation.saving_cctf_encoding = False
    state.simulation.scan_film = True
    state.display.use_display_transform = False
    state.display.white_padding = 0.24
    return state


def _make_widgets(state: GuiState) -> GuiWidgets:
    return GuiWidgets(
        filepicker=object(),
        gui_config=object(),
        display=StubSection(clone_state_section(state.display)),
        input_image=StubSection(clone_state_section(state.input_image)),
        grain=StubSection(clone_state_section(state.grain)),
        preflashing=StubSection(clone_state_section(state.preflashing)),
        halation=StubSection(clone_state_section(state.halation)),
        couplers=StubSection(clone_state_section(state.couplers)),
        glare=StubSection(clone_state_section(state.glare)),
        special=StubSection(clone_state_section(state.special)),
        simulation=StubSimulationSection(clone_state_section(state.simulation), scan_film=state.simulation.scan_film),
    )


def test_gui_state_section_names_match_gui_state_fields() -> None:
    assert GUI_STATE_SECTION_NAMES == tuple(field.name for field in fields(GuiState))


def test_apply_gui_state_updates_all_sections_and_scan_film() -> None:
    source_state = _make_state()
    widgets = _make_widgets(make_gui_state())

    apply_gui_state(source_state, widgets=widgets)

    for section_name in GUI_STATE_SECTION_NAMES:
        assert widgets.__getattribute__(section_name).get_state() == getattr(source_state, section_name)
    assert widgets.simulation.scan_film_value() is True


def test_collect_gui_state_reads_all_sections_and_bottom_bar_scan_flag() -> None:
    source_state = _make_state()
    source_state.simulation.scan_film = False
    widgets = _make_widgets(source_state)
    widgets.simulation.set_scan_film_value(True)

    collected_state = collect_gui_state(widgets=widgets)

    assert collected_state == _make_state()