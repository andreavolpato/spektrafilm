from dataclasses import dataclass
from typing import Any, cast

import napari
from napari.settings import get_settings
from qtpy import QtWidgets

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
    SimulationInputSection,
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


def _create_viewer() -> Any:
    viewer = cast(Any, getattr(napari, 'Viewer'))(show=False)
    settings = get_settings()
    appearance = getattr(settings, 'appearance', None)
    if appearance is not None:
        setattr(cast(Any, appearance), 'theme', 'light')
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
    simulation_input = SimulationInputSection()
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
        simulation_input=simulation_input,
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
        special=special,
        simulation=simulation,
        simulation_input=simulation_input,
    )
    return gui_widgets, panel_widgets


def _connect_controller_signals(controller: GuiController, widgets: GuiWidgets) -> None:
    widgets.filepicker.load_requested.connect(controller.load_input_image)
    widgets.gui_config.save_current_as_default_requested.connect(controller.save_current_as_default)
    widgets.gui_config.save_current_to_file_requested.connect(controller.save_current_state_to_file)
    widgets.gui_config.load_from_file_requested.connect(controller.load_state_from_file)
    widgets.gui_config.restore_factory_default_requested.connect(controller.restore_factory_default)
    widgets.simulation.preview_requested.connect(controller.run_preview)
    widgets.simulation.scan_requested.connect(controller.run_scan)
    widgets.simulation.save_requested.connect(controller.save_output_layer)
    widgets.simulation.use_display_transform.toggled.connect(controller.report_display_transform_status)


def create_app() -> GuiApp:
    warmup()
    viewer = _create_viewer()
    widgets, panel_widgets = _create_widgets()
    apply_gui_state(load_default_gui_state(), widgets=widgets)
    controller = GuiController(viewer=viewer, widgets=widgets)
    _connect_controller_signals(controller, widgets)
    controller.refresh_input_layers()
    configure_napari_chrome(viewer)
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
    app = create_app()
    show_viewer_window(app.viewer)
    napari.run()


if __name__ == "__main__":
    main()
