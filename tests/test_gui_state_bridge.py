from __future__ import annotations

from copy import deepcopy

from spectral_film_lab.gui.state import PROJECT_DEFAULT_GUI_STATE, GuiState
from spectral_film_lab.gui.state_bridge import GUI_STATE_SECTION_NAMES, GuiWidgets, apply_gui_state, collect_gui_state


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
    state = deepcopy(PROJECT_DEFAULT_GUI_STATE)
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
    state.simulation.use_display_transform = False
    state.simulation.scan_film = True
    return state


def _make_widgets(state: GuiState) -> GuiWidgets:
    return GuiWidgets(
        filepicker=object(),
        simulation_input=object(),
        input_image=StubSection(deepcopy(state.input_image)),
        grain=StubSection(deepcopy(state.grain)),
        preflashing=StubSection(deepcopy(state.preflashing)),
        halation=StubSection(deepcopy(state.halation)),
        couplers=StubSection(deepcopy(state.couplers)),
        glare=StubSection(deepcopy(state.glare)),
        special=StubSection(deepcopy(state.special)),
        simulation=StubSimulationSection(deepcopy(state.simulation), scan_film=state.simulation.scan_film),
    )


def test_gui_state_section_names_match_gui_state_fields() -> None:
    assert GUI_STATE_SECTION_NAMES == tuple(GuiState.__dataclass_fields__)


def test_apply_gui_state_updates_all_sections_and_scan_film() -> None:
    source_state = _make_state()
    widgets = _make_widgets(deepcopy(PROJECT_DEFAULT_GUI_STATE))

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