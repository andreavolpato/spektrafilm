from __future__ import annotations

from dataclasses import dataclass

import napari
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication, QFrame, QLabel, QScrollArea, QTabWidget, QVBoxLayout, QWidget

from spectral_film_lab.gui.widgets import (
    CollapsibleSection,
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
    ScannerSection,
    SimulationSection,
    SimulationInputSection,
    SpectralUpsamplingSection,
    SpecialSection,
    TuneSection,
    PreviewCropSection,
    CameraSection,
)


DEFAULT_CONTROLS_PANEL_WIDTH = 420


@dataclass(slots=True)
class ControlsPanelWidgets:
    enlarger: EnlargerSection
    exposure_control: ExposureControlSection
    input_image: InputImageSection
    output: OutputSection
    scanner: ScannerSection
    grain: GrainSection
    preflashing: PreflashingSection
    halation: HalationSection
    couplers: CouplersSection
    glare: GlareSection
    filepicker: FilePickerSection
    special: SpecialSection
    simulation: SimulationSection
    gui_config: GuiConfigSection
    simulation_input: SimulationInputSection
    spectral_upsampling: SpectralUpsamplingSection
    tune: TuneSection
    preview_crop: PreviewCropSection
    camera: CameraSection


def configure_napari_chrome(viewer: napari.Viewer) -> None:
    qt_window = getattr(viewer.window, '_qt_window', None)
    if qt_window is not None:
        menu_bar = qt_window.menuBar()
        if menu_bar is not None:
            menu_bar.hide()

    qt_viewer = getattr(viewer.window, '_qt_viewer', None)
    if qt_viewer is None:
        return

    set_welcome_visible = getattr(qt_viewer, 'set_welcome_visible', None)
    if callable(set_welcome_visible):
        set_welcome_visible(False)

    layer_controls = getattr(qt_viewer, 'dockLayerControls', None)
    if layer_controls is not None:
        layer_controls.hide()

    layer_list = getattr(qt_viewer, 'dockLayerList', None)
    if layer_list is not None:
        layer_list.hide()


def _wrap_scrollable(widget: QWidget) -> QScrollArea:
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    scroll.setFrameShape(QFrame.NoFrame)
    scroll.setWidget(widget)
    return scroll


def _wrap_framed_panel(title: str, widget: QWidget) -> QFrame:
    frame = QFrame()
    frame.setFrameShape(QFrame.StyledPanel)
    frame.setFrameShadow(QFrame.Plain)
    frame.setLineWidth(1)
    frame.setMidLineWidth(0)

    layout = QVBoxLayout(frame)
    layout.setContentsMargins(8, 8, 8, 8)
    layout.setSpacing(6)
    layout.addWidget(QLabel(title))
    layout.addWidget(widget)
    return frame


def build_controls_panel(viewer: napari.Viewer, widgets: ControlsPanelWidgets) -> QWidget:
    main_tab = QWidget()
    main_layout = QVBoxLayout(main_tab)
    main_layout.setContentsMargins(0, 0, 0, 0)
    main_layout.addWidget(widgets.filepicker)
    main_layout.addWidget(widgets.preview_crop)
    main_layout.addWidget(widgets.input_image)
    main_layout.addWidget(widgets.camera)
    main_layout.addWidget(widgets.simulation)
    main_layout.addWidget(widgets.exposure_control)
    main_layout.addWidget(widgets.enlarger)
    main_layout.addWidget(widgets.scanner)
    main_layout.addWidget(widgets.output)
    main_layout.addStretch(1)

    film_tab = QWidget()
    film_layout = QVBoxLayout(film_tab)
    film_layout.setContentsMargins(0, 0, 0, 0)
    film_layout.addWidget(widgets.halation)
    film_layout.addWidget(widgets.couplers)
    film_layout.addWidget(widgets.grain)
    film_layout.addStretch(1)

    paper_tab = QWidget()
    paper_layout = QVBoxLayout(paper_tab)
    paper_layout.setContentsMargins(0, 0, 0, 0)
    paper_layout.addWidget(widgets.glare)
    paper_layout.addWidget(widgets.preflashing)
    paper_layout.addStretch(1)

    advanced_tab = QWidget()
    advanced_layout = QVBoxLayout(advanced_tab)
    advanced_layout.setContentsMargins(0, 0, 0, 0)
    advanced_layout.addWidget(widgets.spectral_upsampling)
    advanced_layout.addWidget(widgets.tune)
    advanced_layout.addWidget(widgets.special)
    advanced_layout.addStretch(1)

    panel = QTabWidget()
    panel.addTab(_wrap_scrollable(main_tab), 'Main')
    panel.addTab(_wrap_scrollable(film_tab), 'Film')
    panel.addTab(_wrap_scrollable(paper_tab), 'Print')
    panel.addTab(_wrap_scrollable(advanced_tab), 'Advanced')

    config_tab = QWidget()
    config_layout = QVBoxLayout(config_tab)
    config_layout.setContentsMargins(0, 0, 0, 0)
    config_layout.addWidget(widgets.gui_config)

    napari_layers_content = QWidget()
    napari_layers_content_layout = QVBoxLayout(napari_layers_content)
    napari_layers_content_layout.setContentsMargins(0, 0, 0, 0)
    napari_layers_content_layout.setSpacing(6)
    napari_layers_content_layout.addWidget(widgets.simulation_input)

    qt_viewer = getattr(viewer.window, '_qt_viewer', None)
    layer_list = getattr(qt_viewer, 'dockLayerList', None) if qt_viewer is not None else None
    if layer_list is not None and hasattr(layer_list, 'widget'):
        layer_list_widget = layer_list.widget()
        if layer_list_widget is not None:
            napari_layers_content_layout.addWidget(layer_list_widget)

    config_layout.addWidget(CollapsibleSection('Napari Layers', napari_layers_content, expanded=False))
    config_layout.addStretch(1)
    panel.addTab(_wrap_scrollable(config_tab), 'Config')

    container = QWidget()
    container_layout = QVBoxLayout(container)
    container_layout.setContentsMargins(0, 0, 0, 0)
    container_layout.setSpacing(8)
    container_layout.addWidget(panel, 1)
    container_layout.addWidget(widgets.simulation.action_bar())

    return container


def dialog_parent(viewer: napari.Viewer) -> QWidget | None:
    return getattr(viewer.window, '_qt_window', None)


def set_status(viewer: napari.Viewer, message: str, *, timeout_ms: int = 5000) -> None:
    status_bar = getattr(dialog_parent(viewer), 'statusBar', None)
    if callable(status_bar):
        status_bar().showMessage(message, timeout_ms)


def add_dock_widget(viewer: napari.Viewer, widget: QWidget, *, area: str, name: str, tabify: bool) -> None:
    widget.setMinimumWidth(DEFAULT_CONTROLS_PANEL_WIDTH)
    dock_widget = viewer.window.add_dock_widget(widget, area=area, name=name, tabify=tabify)
    set_title_bar_widget = getattr(dock_widget, 'setTitleBarWidget', None)
    if callable(set_title_bar_widget):
        set_title_bar_widget(QWidget(dock_widget))
    dock_widget.setStyleSheet('QDockWidget { border: none; }')
    dock_widget.resize(DEFAULT_CONTROLS_PANEL_WIDTH, dock_widget.height())


def show_viewer_window(viewer: napari.Viewer) -> None:
    viewer.window.show()

    app = QApplication.instance()
    if app is None:
        return

    active_window = app.activeWindow()
    if active_window is not None:
        active_window.showMaximized()
        return

    for window in reversed(app.topLevelWidgets()):
        if window.isVisible():
            window.showMaximized()
            return