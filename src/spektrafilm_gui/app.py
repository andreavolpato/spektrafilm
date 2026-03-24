from dataclasses import dataclass
from importlib import import_module
from typing import Any, cast

from qtpy import QtCore, QtGui, QtWidgets

from spektrafilm_gui.controller import GuiController
from spektrafilm_gui.persistence import load_default_gui_state
from spektrafilm_gui.state_bridge import (
    apply_gui_state,
    GuiWidgets,
)
from spektrafilm_gui.napari_layout import (
    ControlsPanelWidgets,
    build_main_window,
    build_controls_panel,
    configure_napari_chrome,
    show_viewer_window,
)
from spektrafilm_gui.widgets import (
    CouplersSection,
    DisplaySection,
    EnlargerSection,
    ExposureControlSection,
    FilePickerSection,
    GlareSection,
    GrainSection,
    GuiConfigSection,
    HalationSection,
    InputImageSection,
    OutputSection,
    PreflashingSection,
    SimulationSection,
    SpectralUpsamplingSection,
    ScannerSection,
    SpecialSection,
    TuneSection,
    PreviewCropSection,
    CameraSection,
)
from spektrafilm.utils.numba_warmup import warmup

@dataclass(slots=True)
class GuiApp:
    viewer: Any
    widgets: GuiWidgets
    panel_widgets: ControlsPanelWidgets
    controller: GuiController
    main_window: QtWidgets.QMainWindow


def _build_fallback_dark_palette() -> QtGui.QPalette:
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor('#323232'))
    palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor('#ffffff'))
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor('#424242'))
    palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor('#585858'))
    palette.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor('#424242'))
    palette.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor('#ffffff'))
    palette.setColor(QtGui.QPalette.Text, QtGui.QColor('#ffffff'))
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor('#424242'))
    palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor('#ffffff'))
    palette.setColor(QtGui.QPalette.BrightText, QtGui.QColor('#ffffff'))
    palette.setColor(QtGui.QPalette.Mid, QtGui.QColor('#707070'))
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor('#70c6ee'))
    palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor('#ffffff'))
    placeholder_role = getattr(QtGui.QPalette, 'PlaceholderText', None)
    if placeholder_role is not None:
        palette.setColor(placeholder_role, QtGui.QColor('#B1B1B1'))

    palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, QtGui.QColor('#B1B1B1'))
    palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.Text, QtGui.QColor('#B1B1B1'))
    palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, QtGui.QColor('#B1B1B1'))
    palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.HighlightedText, QtGui.QColor('#B1B1B1'))
    return palette


def _palette_is_dark(palette: QtGui.QPalette) -> bool:
    return palette.color(QtGui.QPalette.Window).lightness() < palette.color(QtGui.QPalette.WindowText).lightness()


def _apply_system_palette() -> None:
    app = QtWidgets.QApplication.instance()
    if app is None:
        return

    style_hints_factory = getattr(app, 'styleHints', None)
    if not callable(style_hints_factory):
        return
    style_hints = style_hints_factory()

    qt_color_scheme = getattr(QtCore.Qt, 'ColorScheme', None)
    set_color_scheme = getattr(style_hints, 'setColorScheme', None)
    if qt_color_scheme is None or not callable(set_color_scheme):
        return

    dark_scheme = getattr(qt_color_scheme, 'Dark', None)
    if dark_scheme is not None:
        set_color_scheme(dark_scheme)

    if _palette_is_dark(app.palette()):
        return

    app.setPalette(_build_fallback_dark_palette())


def _create_viewer() -> Any:
    napari = import_module('napari')
    get_settings = import_module('napari.settings').get_settings
    viewer_cls = cast(Any, getattr(napari, 'Viewer'))
    viewer = viewer_cls(show=False)
    settings = get_settings()
    appearance = getattr(settings, 'appearance', None)
    if appearance is not None:
        setattr(cast(Any, appearance), 'theme', 'dark')
    return viewer


def _create_widgets() -> tuple[GuiWidgets, ControlsPanelWidgets]:
    grain = GrainSection()
    input_image = InputImageSection()
    preflashing = PreflashingSection()
    halation = HalationSection()
    couplers = CouplersSection()
    glare = GlareSection()
    filepicker = FilePickerSection()
    gui_config = GuiConfigSection()
    display = DisplaySection()
    simulation = SimulationSection()
    special = SpecialSection(simulation)
    spectral_upsampling = SpectralUpsamplingSection(input_image)
    tune = TuneSection(special)
    preview_crop = PreviewCropSection(input_image)
    camera = CameraSection(simulation)
    exposure_control = ExposureControlSection(simulation)
    enlarger = EnlargerSection(simulation)
    scanner = ScannerSection(simulation)
    output = OutputSection(simulation)

    gui_widgets = GuiWidgets(
        filepicker=filepicker,
        gui_config=gui_config,
        display=display,
        input_image=input_image,
        grain=grain,
        preflashing=preflashing,
        halation=halation,
        couplers=couplers,
        glare=glare,
        special=special,
        simulation=simulation,
    )
    panel_widgets = ControlsPanelWidgets(
        preview_crop=preview_crop,
        camera=camera,
        exposure_control=exposure_control,
        enlarger=enlarger,
        scanner=scanner,
        spectral_upsampling=spectral_upsampling,
        tune=tune,
        input_image=input_image,
        output=output,
        grain=grain,
        preflashing=preflashing,
        halation=halation,
        couplers=couplers,
        glare=glare,
        filepicker=filepicker,
        gui_config=gui_config,
        display=display,
        special=special,
        simulation=simulation,
    )
    return gui_widgets, panel_widgets


def _connect_controller_signals(controller: GuiController, widgets: GuiWidgets) -> None:
    widgets.filepicker.load_requested.connect(controller.load_input_image)
    widgets.filepicker.input_layer.currentTextChanged.connect(controller.select_input_layer)
    widgets.gui_config.save_current_as_default_requested.connect(controller.save_current_as_default)
    widgets.gui_config.save_current_to_file_requested.connect(controller.save_current_state_to_file)
    widgets.gui_config.load_from_file_requested.connect(controller.load_state_from_file)
    widgets.gui_config.restore_factory_default_requested.connect(controller.restore_factory_default)
    widgets.simulation.preview_requested.connect(controller.run_preview)
    widgets.simulation.scan_requested.connect(controller.run_scan)
    widgets.simulation.save_requested.connect(controller.save_output_layer)
    widgets.display.use_display_transform.toggled.connect(controller.report_display_transform_status)
    widgets.display.gray_18_canvas.toggled.connect(controller.set_gray_18_canvas_enabled)


def _gray_18_canvas_enabled(widgets: GuiWidgets) -> bool:
    toggle = getattr(widgets.display, 'gray_18_canvas', None)
    is_checked = getattr(toggle, 'isChecked', None)
    return bool(is_checked()) if callable(is_checked) else False


def create_app() -> GuiApp:
    warmup()
    viewer = _create_viewer()
    _apply_system_palette()
    widgets, panel_widgets = _create_widgets()
    apply_gui_state(load_default_gui_state(), widgets=widgets)
    controller = GuiController(viewer=viewer, widgets=widgets)
    controller.sync_display_transform_availability(report_status=False)
    _connect_controller_signals(controller, widgets)
    controller.refresh_input_layers()
    configure_napari_chrome(viewer, gray_18_canvas=_gray_18_canvas_enabled(widgets))
    controls_panel = build_controls_panel(viewer, panel_widgets)
    main_window = build_main_window(viewer, controls_panel)
    return GuiApp(
        viewer=viewer,
        widgets=widgets,
        panel_widgets=panel_widgets,
        controller=controller,
        main_window=main_window,
    )

def main():
    napari = import_module('napari')
    app = create_app()
    show_viewer_window(app.viewer)
    napari.run()


if __name__ == "__main__":
    main()
