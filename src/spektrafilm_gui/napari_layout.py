from __future__ import annotations

from importlib import import_module
import sys
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
    GRAY_18,
    SIZE_APP_MARGIN,
    SIZE_FOOTER_BOTTOM_INSET,
    SIZE_FOOTER_MIN_HEIGHT,
    SIZE_FOOTER_ITEM_SPACING,
    SIZE_FOOTER_TOP_SPACING,
    SIZE_PANEL_MARGIN,
    SIZE_SPLITTER_HANDLE_MARGIN_LEFT,
    SIZE_TAB_CONTENT_TOP_MARGIN,
)
from spektrafilm_gui.widgets import WidgetBundle
from spektrafilm_gui.widgets import CollapsibleSection, platform_default_font


DEFAULT_CONTROLS_PANEL_WIDTH = 420
DEFAULT_VIEWER_SPLITTER_WIDTH = 1040
_DWMWA_USE_IMMERSIVE_DARK_MODE = 20
_DWMWA_USE_IMMERSIVE_DARK_MODE_BEFORE_20H1 = 19
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

def _get_current_stylesheet() -> str:
    try:
        return str(import_module('napari.qt').get_current_stylesheet())
    except (ImportError, AttributeError):
        return ''


def _request_dark_title_bar(window: QWidget) -> bool:
    if sys.platform != 'win32':
        return False

    try:
        hwnd = int(window.winId())
    except (AttributeError, TypeError, ValueError):
        return False

    if hwnd == 0:
        return False

    try:
        ctypes = import_module('ctypes')
        dwmapi = ctypes.windll.dwmapi
        set_window_attribute = dwmapi.DwmSetWindowAttribute
    except (ImportError, AttributeError):
        return False

    value = ctypes.c_int(1)
    value_size = ctypes.sizeof(value)
    for attribute in (_DWMWA_USE_IMMERSIVE_DARK_MODE, _DWMWA_USE_IMMERSIVE_DARK_MODE_BEFORE_20H1):
        try:
            result = set_window_attribute(hwnd, attribute, ctypes.byref(value), value_size)
        except (OSError, AttributeError, TypeError, ValueError):
            continue
        if result == 0:
            return True
    return False
def _canvas_background_color(*, gray_18_canvas: bool) -> str:
    return GRAY_18 if gray_18_canvas else GRAY_0


def set_canvas_background(viewer: napari.Viewer, *, gray_18_canvas: bool) -> None:
    background = _canvas_background_color(gray_18_canvas=gray_18_canvas)

    viewer_window = getattr(viewer, 'window', None)
    if viewer_window is None:
        return

    qt_viewer = getattr(viewer_window, '_qt_viewer', None)
    if qt_viewer is None:
        return

    if hasattr(qt_viewer, 'setStyleSheet'):
        qt_viewer.setStyleSheet(f'background: {background};')

    canvas = getattr(qt_viewer, 'canvas', None)
    if canvas is not None:
        if hasattr(canvas, 'bgcolor'):
            setattr(canvas, 'bgcolor', background)
        native = getattr(canvas, 'native', None)
        if native is not None and hasattr(native, 'setStyleSheet'):
            native.setStyleSheet(f'background: {background};')


def configure_napari_chrome(viewer: napari.Viewer, *, gray_18_canvas: bool = True) -> None:
    qt_window = getattr(viewer.window, '_qt_window', None)
    if qt_window is not None:
        menu_bar = qt_window.menuBar()
        if menu_bar is not None:
            menu_bar.hide()

    qt_viewer = getattr(viewer.window, '_qt_viewer', None)
    if qt_viewer is None:
        return

    set_canvas_background(viewer, gray_18_canvas=gray_18_canvas)

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


def reset_viewer_camera(viewer: napari.Viewer) -> None:
    reset_view = getattr(viewer, 'reset_view', None)
    if callable(reset_view):
        reset_view()


def _viewer_device_pixel_ratio(viewer: napari.Viewer) -> float:
    viewer_window = getattr(viewer, 'window', None)
    qt_viewer = getattr(viewer_window, '_qt_viewer', None)
    if qt_viewer is None:
        return 1.0

    device_pixel_ratio = getattr(qt_viewer, 'devicePixelRatioF', None)
    if callable(device_pixel_ratio):
        try:
            return max(float(device_pixel_ratio()), 0.01)
        except (TypeError, ValueError):
            return 1.0

    if hasattr(qt_viewer, 'devicePixelRatio'):
        try:
            return max(float(qt_viewer.devicePixelRatio()), 0.01)
        except (TypeError, ValueError):
            return 1.0

    return 1.0


def _layer_pixel_world_size(layer: object | None) -> float:
    if layer is None:
        return 1.0

    scale = getattr(layer, 'scale', None)
    if scale is None:
        return 1.0

    if isinstance(scale, (int, float)):
        try:
            return max(abs(float(scale)), 1e-12)
        except (TypeError, ValueError):
            return 1.0

    try:
        scale_values = tuple(scale)
    except TypeError:
        return 1.0

    axis_scales: list[float] = []
    for value in scale_values[:2]:
        try:
            numeric_value = abs(float(value))
        except (TypeError, ValueError):
            continue
        if numeric_value > 0.0:
            axis_scales.append(numeric_value)

    if not axis_scales:
        return 1.0
    return sum(axis_scales) / len(axis_scales)


def _home_view_target_layer(viewer: napari.Viewer) -> object | None:
    layers = getattr(viewer, 'layers', None)
    if layers is None:
        return None

    selection = getattr(layers, 'selection', None)
    active_layer = getattr(selection, 'active', None)
    if active_layer is not None and getattr(active_layer, 'visible', True):
        return active_layer

    for layer in reversed(list(layers)):
        if getattr(layer, 'visible', True):
            return layer
    return None


def set_viewer_zoom_percent(viewer: napari.Viewer, percent: float) -> None:
    viewer_window = getattr(viewer, 'window', None)
    camera = getattr(viewer, 'camera', None)
    if camera is None:
        camera = getattr(viewer_window, 'camera', None)
    if camera is None or not hasattr(camera, 'zoom'):
        return

    pixel_ratio = _viewer_device_pixel_ratio(viewer)
    target_layer = _home_view_target_layer(viewer)
    layer_pixel_world_size = _layer_pixel_world_size(target_layer)
    zoom_scale = max(0.0, float(percent)) / 100.0
    camera.zoom = max((pixel_ratio * zoom_scale) / layer_pixel_world_size, pixel_ratio * 0.01)


def _build_viewer_panel(
    viewer_widget: QWidget,
    status_bar: QStatusBar,
    *,
    on_rotate_ccw: Callable[[], None] | None = None,
    on_rotate_cw: Callable[[], None] | None = None,
    on_zoom_100: Callable[[], None] | None = None,
    on_zoom_200: Callable[[], None] | None = None,
    on_zoom_400: Callable[[], None] | None = None,
    on_home_view: Callable[[], None] | None = None,
) -> QFrame:
    panel = QtWidgets.QFrame()
    panel.setObjectName('viewerPanel')
    divider_gap = int(SIZE_SPLITTER_HANDLE_MARGIN_LEFT.removesuffix('px'))

    status_container = QtWidgets.QWidget()
    status_layout = QtWidgets.QHBoxLayout(status_container)
    status_layout.setContentsMargins(0, 0, 0, 0)
    status_layout.setSpacing(SIZE_FOOTER_ITEM_SPACING)
    status_bar.setContentsMargins(0, 0, 0, 0)
    status_bar.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
    status_layout.addWidget(status_bar, 1)

    rotate_ccw_button = QPushButton('ccw rotate')
    rotate_ccw_button.setObjectName('rotateCcwButton')
    if on_rotate_ccw is not None:
        rotate_ccw_button.clicked.connect(on_rotate_ccw)

    rotate_cw_button = QPushButton('cw rotate')
    rotate_cw_button.setObjectName('rotateCwButton')
    if on_rotate_cw is not None:
        rotate_cw_button.clicked.connect(on_rotate_cw)

    zoom_100_button = QPushButton('100%')
    zoom_100_button.setObjectName('zoom100Button')
    zoom_100_button.setToolTip('Pixel of the screen mapped 1 to 1 to the image pixel')
    if on_zoom_100 is not None:
        zoom_100_button.clicked.connect(on_zoom_100)

    zoom_200_button = QPushButton('200%')
    zoom_200_button.setObjectName('zoom200Button')
    zoom_200_button.setToolTip('2 screen pixels mapped to 1 image pixel')
    if on_zoom_200 is not None:
        zoom_200_button.clicked.connect(on_zoom_200)

    zoom_400_button = QPushButton('400%')
    zoom_400_button.setObjectName('zoom400Button')
    zoom_400_button.setToolTip('4 screen pixels mapped to 1 image pixel')
    if on_zoom_400 is not None:
        zoom_400_button.clicked.connect(on_zoom_400)

    home_button = QPushButton('reset view')
    home_button.setObjectName('homeViewButton')
    if on_home_view is not None:
        home_button.clicked.connect(on_home_view)
    row_height = int(SIZE_FOOTER_MIN_HEIGHT.removesuffix('px'))
    rotate_ccw_button.setFixedHeight(row_height)
    rotate_cw_button.setFixedHeight(row_height)
    zoom_100_button.setFixedHeight(row_height)
    zoom_200_button.setFixedHeight(row_height)
    zoom_400_button.setFixedHeight(row_height)
    home_button.setFixedHeight(row_height)
    status_bar.setFixedHeight(row_height)
    status_container.setFixedHeight(row_height)
    status_layout.addWidget(rotate_ccw_button)
    status_layout.addWidget(rotate_cw_button)
    status_layout.addWidget(zoom_100_button)
    status_layout.addWidget(zoom_200_button)
    status_layout.addWidget(zoom_400_button)
    status_layout.addWidget(home_button)

    layout = QtWidgets.QVBoxLayout(panel)
    layout.setContentsMargins(SIZE_PANEL_MARGIN, SIZE_PANEL_MARGIN, divider_gap, SIZE_FOOTER_BOTTOM_INSET)
    layout.setSpacing(0)
    layout.addWidget(viewer_widget, 1)
    layout.addSpacing(SIZE_FOOTER_TOP_SPACING)
    layout.addWidget(status_container)
    return panel


def build_controls_panel(viewer: napari.Viewer, widgets: WidgetBundle) -> QWidget:
    panel = QtWidgets.QTabWidget()
    panel.setObjectName('controlsTabWidget')
    panel.setDocumentMode(True)
    panel.setUsesScrollButtons(False)
    panel.tabBar().setDrawBase(False)
    panel.addTab(
        _wrap_scrollable(
            _build_controls_tab(
                widgets.filepicker,
                widgets.load_raw,
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
    panel.addTab(_wrap_scrollable(_build_controls_tab(widgets.glare, widgets.preflashing, widgets.diffusion)), 'PRINT')
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


def build_main_window(
    viewer: napari.Viewer,
    controls_panel: QWidget,
    *,
    on_rotate_ccw: Callable[[], None] | None = None,
    on_rotate_cw: Callable[[], None] | None = None,
) -> QMainWindow:
    viewer_widget = take_viewer_widget(viewer)
    status_bar = QtWidgets.QStatusBar()
    status_bar.setSizeGripEnabled(False)

    main_window = AppMainWindow()
    main_window.setWindowTitle('spektrafilm')
    main_window.setWindowIcon(QIcon())
    main_window.resize(DEFAULT_CONTROLS_PANEL_WIDTH + DEFAULT_VIEWER_SPLITTER_WIDTH, 980)
    main_window.setFont(platform_default_font())
    main_window.setStyleSheet(APP_STYLE_SHEET)
    main_window.set_viewer_status_bar(status_bar)
    set_host_window(viewer, main_window)

    splitter = QtWidgets.QSplitter(Qt.Horizontal)
    splitter.setChildrenCollapsible(False)
    splitter.addWidget(
        _build_viewer_panel(
            viewer_widget,
            status_bar,
            on_rotate_ccw=on_rotate_ccw,
            on_rotate_cw=on_rotate_cw,
            on_zoom_100=lambda: set_viewer_zoom_percent(viewer, 100.0),
            on_zoom_200=lambda: set_viewer_zoom_percent(viewer, 200.0),
            on_zoom_400=lambda: set_viewer_zoom_percent(viewer, 400.0),
            on_home_view=lambda: reset_viewer_camera(viewer),
        )
    )
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
    _request_dark_title_bar(main_window)
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