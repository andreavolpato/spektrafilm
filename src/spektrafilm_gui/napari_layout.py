from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Callable
from typing import Any, cast

from qtpy import QtGui, QtWidgets
from qtpy.QtCore import Qt

if TYPE_CHECKING:
    import napari

QFrame = QtWidgets.QFrame
QIcon = QtGui.QIcon
QMainWindow = QtWidgets.QMainWindow
QPushButton = QtWidgets.QPushButton
QStatusBar = QtWidgets.QStatusBar
QScrollArea = QtWidgets.QScrollArea
QWidget = QtWidgets.QWidget

from spektrafilm_gui.theme import APP_STYLE_SHEET
from spektrafilm_gui.theme_palette import (
    GRAY_0,
    SIZE_APP_MARGIN,
    SIZE_FOOTER_BOTTOM_INSET,
    SIZE_FOOTER_ITEM_SPACING,
    SIZE_PANEL_MARGIN,
    SIZE_TAB_CONTENT_TOP_MARGIN,
)
from spektrafilm_gui.widgets import (
    CollapsibleSection,
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
    ScannerSection,
    SimulationSection,
    SpectralUpsamplingSection,
    SpecialSection,
    TuneSection,
    PreviewCropSection,
    CameraSection,
)


DEFAULT_CONTROLS_PANEL_WIDTH = 420
DEFAULT_VIEWER_SPLITTER_WIDTH = 1040


class AppMainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self._viewer_status_bar: QStatusBar | None = None

    def set_viewer_status_bar(self, status_bar: QStatusBar) -> None:
        self._viewer_status_bar = status_bar

    def statusBar(self) -> QStatusBar:  # noqa: N802 - Qt API name
        if self._viewer_status_bar is not None:
            return self._viewer_status_bar
        return super().statusBar()


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
    display: DisplaySection
    spectral_upsampling: SpectralUpsamplingSection
    tune: TuneSection
    preview_crop: PreviewCropSection
    camera: CameraSection


def _get_current_stylesheet() -> str:
    try:
        return str(import_module('napari.qt').get_current_stylesheet())
    except (ImportError, AttributeError):
        return ''


def configure_napari_chrome(viewer: napari.Viewer) -> None:
    qt_window = getattr(viewer.window, '_qt_window', None)
    if qt_window is not None:
        menu_bar = qt_window.menuBar()
        if menu_bar is not None:
            menu_bar.hide()

    qt_viewer = getattr(viewer.window, '_qt_viewer', None)
    if qt_viewer is None:
        return

    if hasattr(qt_viewer, 'setStyleSheet'):
        qt_viewer.setStyleSheet(f'background: {GRAY_0};')

    canvas = getattr(qt_viewer, 'canvas', None)
    if canvas is not None:
        if hasattr(canvas, 'bgcolor'):
            setattr(canvas, 'bgcolor', GRAY_0)
        native = getattr(canvas, 'native', None)
        if native is not None and hasattr(native, 'setStyleSheet'):
            native.setStyleSheet(f'background: {GRAY_0};')

    set_welcome_visible = getattr(qt_viewer, 'set_welcome_visible', None)
    if callable(set_welcome_visible):
        set_welcome_visible(False)

    layer_controls = getattr(qt_viewer, 'dockLayerControls', None)
    if layer_controls is not None:
        layer_controls.hide()

    layer_list = getattr(qt_viewer, 'dockLayerList', None)
    if layer_list is not None:
        layer_list.hide()


def set_host_window(viewer: napari.Viewer, host_window: QWidget) -> None:
    viewer_window = getattr(viewer, 'window', None)
    if viewer_window is not None:
        setattr(viewer_window, '_agx_host_window', host_window)


def _host_window(viewer: napari.Viewer) -> QWidget | None:
    viewer_window = getattr(viewer, 'window', None)
    if viewer_window is None:
        return None
    host_window = getattr(viewer_window, '_agx_host_window', None)
    if host_window is not None:
        return host_window
    qt_window = getattr(viewer_window, '_qt_window', None)
    if qt_window is not None:
        return qt_window
    return None


def take_viewer_widget(viewer: napari.Viewer) -> QWidget:
    viewer_window = getattr(viewer, 'window', None)
    if viewer_window is None:
        raise RuntimeError('Napari viewer window is not available')

    qt_window = getattr(viewer_window, '_qt_window', None)
    if qt_window is not None:
        take_central_widget = getattr(qt_window, 'takeCentralWidget', None)
        if callable(take_central_widget):
            central_widget = take_central_widget()
            if central_widget is not None:
                return central_widget

    qt_viewer = getattr(viewer_window, '_qt_viewer', None)
    if qt_viewer is not None:
        return qt_viewer
    raise RuntimeError('Napari Qt viewer widget is not available')


def _wrap_scrollable(widget: QWidget) -> QScrollArea:
    scroll = QtWidgets.QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setSizeAdjustPolicy(QScrollArea.AdjustIgnored)
    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    scroll.setFrameShape(QFrame.NoFrame)
    scroll.setWidget(widget)
    return scroll


def _build_sidebar(controls_panel: QWidget) -> QFrame:
    sidebar = QtWidgets.QFrame()
    sidebar.setObjectName('sidebarPanel')

    layout = QtWidgets.QVBoxLayout(sidebar)
    layout.setContentsMargins(SIZE_PANEL_MARGIN, SIZE_PANEL_MARGIN, SIZE_PANEL_MARGIN, SIZE_FOOTER_BOTTOM_INSET)
    layout.setSpacing(0)
    layout.addWidget(controls_panel, 1)
    return sidebar


def _build_controls_tab(*widgets: QWidget) -> QWidget:
    tab = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(tab)
    layout.setContentsMargins(0, SIZE_TAB_CONTENT_TOP_MARGIN, 0, 0)
    layout.setAlignment(Qt.AlignTop)
    for widget in widgets:
        layout.addWidget(widget)
    return tab


def _borrow_layer_list_widget(viewer: napari.Viewer) -> QWidget | None:
    qt_viewer = getattr(viewer.window, '_qt_viewer', None)
    layer_list = getattr(qt_viewer, 'dockLayerList', None) if qt_viewer is not None else None
    if layer_list is None or not hasattr(layer_list, 'widget'):
        return None
    widget = layer_list.widget()
    if isinstance(widget, QWidget):
        widget.setStyleSheet(_get_current_stylesheet())
    return widget


def _home_view_target_layer(viewer: napari.Viewer) -> object | None:
    layers = getattr(viewer, 'layers', None)
    if layers is None:
        return None

    selection = getattr(layers, 'selection', None)
    active_layer = getattr(selection, 'active', None)
    if active_layer is not None and hasattr(active_layer, 'visible'):
        return active_layer

    for layer in reversed(list(layers)):
        if getattr(layer, 'visible', False):
            return layer
    return None


def reset_viewer_camera(viewer: napari.Viewer) -> None:
    reset_view = getattr(viewer, 'reset_view', None)
    if callable(reset_view):
        target_layer = _home_view_target_layer(viewer)
        if target_layer is None:
            reset_view()
            return

        layers = list(getattr(viewer, 'layers', []))
        visibility_by_layer = [(layer, getattr(layer, 'visible', True)) for layer in layers]
        try:
            for layer, _ in visibility_by_layer:
                if hasattr(layer, 'visible'):
                    layer.visible = layer is target_layer
            reset_view()
        finally:
            for layer, was_visible in visibility_by_layer:
                if hasattr(layer, 'visible'):
                    layer.visible = was_visible


def _build_viewer_panel(
    viewer_widget: QWidget,
    status_bar: QStatusBar,
    *,
    on_home_view: Callable[[], None] | None = None,
) -> QFrame:
    panel = QtWidgets.QFrame()
    panel.setObjectName('viewerPanel')

    status_container = QtWidgets.QWidget()
    status_layout = QtWidgets.QHBoxLayout(status_container)
    status_layout.setContentsMargins(0, 0, 0, 0)
    status_layout.setSpacing(SIZE_FOOTER_ITEM_SPACING)
    status_bar.setContentsMargins(0, 0, 0, 0)
    status_bar.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
    status_layout.addWidget(status_bar, 1)

    home_button = QPushButton('reset view')
    home_button.setObjectName('homeViewButton')
    if on_home_view is not None:
        home_button.clicked.connect(on_home_view)
    row_height = home_button.sizeHint().height()
    status_bar.setFixedHeight(row_height)
    status_container.setFixedHeight(row_height)
    status_layout.addWidget(home_button)

    layout = QtWidgets.QVBoxLayout(panel)
    layout.setContentsMargins(SIZE_PANEL_MARGIN, SIZE_PANEL_MARGIN, SIZE_PANEL_MARGIN, SIZE_FOOTER_BOTTOM_INSET)
    layout.setSpacing(0)
    layout.addWidget(viewer_widget, 1)
    layout.addWidget(status_container)
    return panel


def build_controls_panel(viewer: napari.Viewer, widgets: ControlsPanelWidgets) -> QWidget:
    panel = QtWidgets.QTabWidget()
    panel.setObjectName('controlsTabWidget')
    panel.setDocumentMode(True)
    panel.addTab(
        _wrap_scrollable(
            _build_controls_tab(
                widgets.filepicker,
                widgets.preview_crop,
                widgets.input_image,
                widgets.camera,
                widgets.simulation,
                widgets.exposure_control,
                widgets.enlarger,
                widgets.scanner,
                widgets.output,
            ),
        ),
        'MAIN',
    )
    panel.addTab(_wrap_scrollable(_build_controls_tab(widgets.halation, widgets.couplers, widgets.grain)), 'FILM')
    panel.addTab(_wrap_scrollable(_build_controls_tab(widgets.glare, widgets.preflashing)), 'PRINT')
    panel.addTab(
        _wrap_scrollable(_build_controls_tab(widgets.spectral_upsampling, widgets.tune, widgets.special)),
        'ADVANCED',
    )

    napari_layers_content = QtWidgets.QWidget()
    napari_layers_content_layout = QtWidgets.QVBoxLayout(napari_layers_content)
    napari_layers_content_layout.setContentsMargins(0, 0, 0, 0)
    napari_layers_content_layout.setSpacing(6)

    layer_list_widget = _borrow_layer_list_widget(viewer)
    if layer_list_widget is not None:
        napari_layers_content_layout.addWidget(layer_list_widget)

    panel.addTab(
        _wrap_scrollable(
            _build_controls_tab(
                widgets.gui_config,
                widgets.display,
                CollapsibleSection('napari layers', napari_layers_content, expanded=False),
            ),
        ),
        'CONFIG',
    )

    container = QtWidgets.QWidget()
    container_layout = QtWidgets.QVBoxLayout(container)
    container_layout.setContentsMargins(0, 0, 0, 0)
    container_layout.setSpacing(4)
    container_layout.addWidget(panel, 1)
    container_layout.addWidget(widgets.simulation.action_bar())

    return container


def build_main_window(viewer: napari.Viewer, controls_panel: QWidget) -> QMainWindow:
    viewer_widget = take_viewer_widget(viewer)
    status_bar = QtWidgets.QStatusBar()
    status_bar.setSizeGripEnabled(False)

    main_window = AppMainWindow()
    main_window.setWindowTitle('spektrafilm')
    main_window.setWindowIcon(QIcon())
    main_window.resize(DEFAULT_CONTROLS_PANEL_WIDTH + DEFAULT_VIEWER_SPLITTER_WIDTH, 980)
    main_window.setStyleSheet(APP_STYLE_SHEET)
    main_window.set_viewer_status_bar(status_bar)
    set_host_window(viewer, main_window)

    splitter = QtWidgets.QSplitter(Qt.Horizontal)
    splitter.setChildrenCollapsible(False)
    splitter.addWidget(_build_viewer_panel(viewer_widget, status_bar, on_home_view=lambda: reset_viewer_camera(viewer)))
    splitter.addWidget(_build_sidebar(controls_panel))
    splitter.setStretchFactor(0, 1)
    splitter.setStretchFactor(1, 0)
    splitter.setSizes([DEFAULT_VIEWER_SPLITTER_WIDTH, DEFAULT_CONTROLS_PANEL_WIDTH])

    central = QtWidgets.QWidget()
    central.setObjectName('appCentral')
    central_layout = QtWidgets.QHBoxLayout(central)
    central_layout.setContentsMargins(SIZE_APP_MARGIN, SIZE_APP_MARGIN, SIZE_APP_MARGIN, SIZE_APP_MARGIN)
    central_layout.addWidget(splitter, 1)

    main_window.setCentralWidget(central)
    main_window.statusBar().showMessage('ready', 3000)
    return main_window


def dialog_parent(viewer: napari.Viewer) -> QWidget | None:
    return _host_window(viewer)


def set_status(viewer: napari.Viewer, message: str, *, timeout_ms: int = 5000) -> None:
    host_window = cast(Any, _host_window(viewer))
    if host_window is None:
        return
    status_bar = host_window.statusBar() if hasattr(host_window, 'statusBar') else None
    if status_bar is not None:
        status_bar.showMessage(message, timeout_ms)


def show_viewer_window(viewer: napari.Viewer) -> None:
    host_window = _host_window(viewer)
    if host_window is not None and hasattr(host_window, 'show'):
        host_window.show()

    app = QtWidgets.QApplication.instance()
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