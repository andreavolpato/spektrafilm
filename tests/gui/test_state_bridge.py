from __future__ import annotations

from dataclasses import replace

from spektrafilm_gui.state import GuiState, clone_state_section
from spektrafilm_gui.state_bridge import GUI_STATE_SECTION_NAMES, apply_gui_state, collect_gui_state
from spektrafilm_gui.widgets import WidgetBundle

from .helpers import make_test_gui_state


def _section_state(state: GuiState, section_name: str) -> object:
    if section_name == 'display':
        return state.gui_only.display
    if section_name == 'load_raw':
        return state.gui_only.load_raw
    return getattr(state, section_name)


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
    state.input_image.io.upscale_factor = 1.5
    state.gui_only.load_raw.white_balance = 'custom'
    state.gui_only.load_raw.temperature = 3200.0
    state.gui_only.load_raw.tint = 0.85
    state.grain.active = False
    state.halation.halation_strength = (7.0, 5.0, 3.0)
    state.couplers.inhibition_interlayer = 1.75
    # PrintChemistryParams is a frozen runtime dataclass; reassign via replace
    state.chemistry = replace(state.chemistry, gamma_factor=1.15)
    state.glare.blur = 0.8
    state.scanner = replace(state.scanner, white_correction=True, white_level=0.9, lens_blur=0.3)
    state.input_gamut_compress = replace(state.input_gamut_compress, active=False, algorithm='oklch')
    state.output_gamut_compress = replace(state.output_gamut_compress, algorithm='aces_rgc')
    state.simulation.enlarger.print_exposure = 1.3
    state.camera.lens_blur_um = 3.5
    state.camera.exposure_compensation_ev = 0.5
    state.camera_diffusion = replace(
        state.camera_diffusion,
        active=True,
        filter_family='glimmerglass',
        strength=0.25,
        spatial_scale=1.2,
        halo_warmth=-0.15,
    )
    state.enlarger_diffusion.active = True
    state.enlarger_diffusion.filter_family = 'pro_mist'
    state.enlarger_diffusion.strength = 0.5
    state.enlarger_diffusion.spatial_scale = 1.6
    state.enlarger_diffusion.halo_warmth = 0.2
    state.simulation.workflow.saving_cctf_encoding = False
    state.simulation.io.scan_film = True
    state.gui_only.display.use_display_transform = False
    state.gui_only.display.gray_18_canvas = True
    state.gui_only.display.white_padding = 0.24
    state.gui_only.display.settings.preview_max_size = 896
    return state


def _make_widgets(state: GuiState) -> WidgetBundle:
    return WidgetBundle(
        filepicker=object(),
        gui_config=object(),
        display=StubSection(clone_state_section(state.gui_only.display)),
        input_image=StubSection(clone_state_section(state.input_image)),
        load_raw=StubSection(clone_state_section(state.gui_only.load_raw)),
        grain=StubSection(clone_state_section(state.grain)),
        preflashing=StubSection(clone_state_section(state.preflashing)),
        enlarger_diffusion=StubSection(clone_state_section(state.enlarger_diffusion)),
        camera=StubSection(clone_state_section(state.camera)),
        camera_diffusion=StubSection(clone_state_section(state.camera_diffusion)),
        halation=StubSection(clone_state_section(state.halation)),
        couplers=StubSection(clone_state_section(state.couplers)),
        chemistry=StubSection(clone_state_section(state.chemistry)),
        base=StubSection(clone_state_section(state.base)),
        film_chemistry=StubSection(clone_state_section(state.film_chemistry)),
        glare=StubSection(clone_state_section(state.glare)),
        scanner=StubSection(clone_state_section(state.scanner)),
        input_gamut_compress=StubSection(clone_state_section(state.input_gamut_compress)),
        output_gamut_compress=StubSection(clone_state_section(state.output_gamut_compress)),
        special=StubSection(clone_state_section(state.special)),
        simulation=StubSimulationSection(
            clone_state_section(state.simulation),
            auto_preview=state.simulation.workflow.auto_preview,
            scan_film=state.simulation.io.scan_film,
        ),
        preview_crop=object(),
        enlarger=object(),
        spectral_upsampling=object(),
        output=object(),
    )


def test_gui_state_section_names_match_gui_state_fields() -> None:
    assert GUI_STATE_SECTION_NAMES == (
        'display',
        'input_image',
        'load_raw',
        'grain',
        'preflashing',
        'halation',
        'couplers',
        'chemistry',
        'base',
        'film_chemistry',
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


def test_apply_gui_state_updates_all_sections_and_scan_film() -> None:
    source_state = _make_state()
    widgets = _make_widgets(make_test_gui_state())

    apply_gui_state(source_state, widgets=widgets)

    for section_name in GUI_STATE_SECTION_NAMES:
        assert widgets.__getattribute__(section_name).get_state() == _section_state(source_state, section_name)
    assert widgets.simulation.auto_preview_value() is source_state.simulation.workflow.auto_preview
    assert widgets.simulation.scan_film_value() is True
    assert widgets.simulation.reset_scan_for_print_calls == 1


def test_collect_gui_state_reads_all_sections_and_bottom_bar_scan_flag() -> None:
    source_state = _make_state()
    source_state.simulation.workflow.auto_preview = False
    source_state.simulation.io.scan_film = False
    widgets = _make_widgets(source_state)
    widgets.simulation.set_auto_preview_value(True)
    widgets.simulation.set_scan_film_value(True)

    collected_state = collect_gui_state(widgets=widgets)

    expected_state = _make_state()
    expected_state.simulation.workflow.auto_preview = True
    assert collected_state == expected_state