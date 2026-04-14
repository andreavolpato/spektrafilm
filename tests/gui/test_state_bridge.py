from __future__ import annotations

from dataclasses import fields

from spektrafilm_gui.state import GuiState, clone_state_section
from spektrafilm_gui.state_bridge import GUI_STATE_SECTION_NAMES, apply_gui_state, collect_gui_state
from spektrafilm_gui.widgets import WidgetBundle

from .helpers import make_test_gui_state


class StubSection:
    def __init__(self, state: object):
        self._state = state

    def set_state(self, state: object) -> None:
        self._state = state

    def get_state(self) -> object:
        return self._state


class StubSimulationSection(StubSection):
    def __init__(self, state: object, *, auto_preview: bool = True, scan_film: bool = False):
        super().__init__(state)
        self._auto_preview = auto_preview
        self._scan_film = scan_film
        self.reset_scan_for_print_calls = 0

    def set_auto_preview_value(self, value: bool) -> None:
        self._auto_preview = value

    def auto_preview_value(self) -> bool:
        return self._auto_preview

    def set_scan_film_value(self, value: bool) -> None:
        self._scan_film = value

    def scan_film_value(self) -> bool:
        return self._scan_film

    def reset_scan_for_print_value(self) -> None:
        self.reset_scan_for_print_calls += 1


def _make_state() -> GuiState:
    state = make_test_gui_state()
    state.input_image.upscale_factor = 1.5
    state.load_raw.white_balance = 'custom'
    state.load_raw.temperature = 3200.0
    state.load_raw.tint = 0.85
    state.grain.active = False
    state.halation.halation_strength = (7.0, 5.0, 3.0)
    state.couplers.diffusion_interlayer = 1.75
    state.glare.blur = 0.8
    state.special.print_gamma_factor = 1.15
    state.simulation.print_exposure = 1.3
    state.simulation.diffusion_strength = 0.5
    state.simulation.diffusion_spatial_scale = 1.6
    state.simulation.diffusion_intensity = 0.7
    state.simulation.saving_cctf_encoding = False
    state.simulation.scan_film = True
    state.display.use_display_transform = False
    state.display.gray_18_canvas = True
    state.display.white_padding = 0.24
    state.display.preview_max_size = 896
    return state


def _make_widgets(state: GuiState) -> WidgetBundle:
    return WidgetBundle(
        filepicker=object(),
        gui_config=object(),
        display=StubSection(clone_state_section(state.display)),
        input_image=StubSection(clone_state_section(state.input_image)),
        load_raw=StubSection(clone_state_section(state.load_raw)),
        grain=StubSection(clone_state_section(state.grain)),
        preflashing=StubSection(clone_state_section(state.preflashing)),
        diffusion=object(),
        halation=StubSection(clone_state_section(state.halation)),
        couplers=StubSection(clone_state_section(state.couplers)),
        glare=StubSection(clone_state_section(state.glare)),
        special=StubSection(clone_state_section(state.special)),
        simulation=StubSimulationSection(
            clone_state_section(state.simulation),
            auto_preview=state.simulation.auto_preview,
            scan_film=state.simulation.scan_film,
        ),
        preview_crop=object(),
        camera=object(),
        exposure_control=object(),
        enlarger=object(),
        scanner=object(),
        spectral_upsampling=object(),
        tune=object(),
        output=object(),
    )


def test_gui_state_section_names_match_gui_state_fields() -> None:
    assert GUI_STATE_SECTION_NAMES == tuple(field.name for field in fields(GuiState))


def test_apply_gui_state_updates_all_sections_and_scan_film() -> None:
    source_state = _make_state()
    widgets = _make_widgets(make_test_gui_state())

    apply_gui_state(source_state, widgets=widgets)

    for section_name in GUI_STATE_SECTION_NAMES:
        assert widgets.__getattribute__(section_name).get_state() == getattr(source_state, section_name)
    assert widgets.simulation.auto_preview_value() is source_state.simulation.auto_preview
    assert widgets.simulation.scan_film_value() is True
    assert widgets.simulation.reset_scan_for_print_calls == 1


def test_collect_gui_state_reads_all_sections_and_bottom_bar_scan_flag() -> None:
    source_state = _make_state()
    source_state.simulation.auto_preview = False
    source_state.simulation.scan_film = False
    widgets = _make_widgets(source_state)
    widgets.simulation.set_auto_preview_value(True)
    widgets.simulation.set_scan_film_value(True)

    collected_state = collect_gui_state(widgets=widgets)

    expected_state = _make_state()
    expected_state.simulation.auto_preview = True
    assert collected_state == expected_state