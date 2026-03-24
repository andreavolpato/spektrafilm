from __future__ import annotations

from types import SimpleNamespace

from spektrafilm_gui import napari_layout as napari_layout_module
from spektrafilm_gui.theme_palette import CONTROL_BG, TEXT_ACCENT
from spektrafilm_gui.theme_styles import CONTROL_STYLE
from spektrafilm_gui.napari_layout import (
    configure_napari_chrome,
    dialog_parent,
    reset_viewer_camera,
    set_host_window,
    set_status,
    set_viewer_zoom_percent,
    take_viewer_widget,
)
from tests.gui_test_utils import FakeLayer, FakeLayerList, make_viewer_namespace


class FakeStatusBar:
    def __init__(self) -> None:
        self.messages: list[tuple[str, int]] = []

    def showMessage(self, message: str, timeout_ms: int) -> None:  # noqa: N802 - Qt API name
        self.messages.append((message, timeout_ms))


class FakeWindowWithStatusBar:
    def __init__(self) -> None:
        self.status_bar = FakeStatusBar()

    def statusBar(self) -> FakeStatusBar:  # noqa: N802 - Qt API name
        return self.status_bar


def test_dialog_parent_prefers_custom_host_window() -> None:
    embedded_window = object()
    host_window = FakeWindowWithStatusBar()
    viewer = make_viewer_namespace(_qt_window=embedded_window)

    set_host_window(viewer, host_window)

    assert dialog_parent(viewer) is host_window


def test_set_status_targets_custom_host_window_status_bar() -> None:
    host_window = FakeWindowWithStatusBar()
    viewer = make_viewer_namespace(_qt_window=object())
    set_host_window(viewer, host_window)

    set_status(viewer, 'Simulation complete', timeout_ms=1500)

    assert host_window.status_bar.messages == [('Simulation complete', 1500)]


def test_take_viewer_widget_uses_taken_central_widget_when_available() -> None:
    central_widget = object()

    class FakeQtWindow:
        def takeCentralWidget(self):  # noqa: N802 - Qt API name
            return central_widget

    viewer = make_viewer_namespace(_qt_window=FakeQtWindow(), _qt_viewer=object())

    assert take_viewer_widget(viewer) is central_widget


def test_take_viewer_widget_falls_back_to_qt_viewer() -> None:
    qt_viewer = object()

    class FakeQtWindow:
        def takeCentralWidget(self):  # noqa: N802 - Qt API name
            return None

    viewer = make_viewer_namespace(_qt_window=FakeQtWindow(), _qt_viewer=qt_viewer)

    assert take_viewer_widget(viewer) is qt_viewer


def test_configure_napari_chrome_hides_default_panels_and_sets_background() -> None:
    class FakeNative:
        def __init__(self) -> None:
            self.stylesheet = None

        def setStyleSheet(self, stylesheet: str) -> None:  # noqa: N802 - Qt API name
            self.stylesheet = stylesheet

    class FakeCanvas:
        def __init__(self) -> None:
            self.bgcolor = None
            self.native = FakeNative()

    class FakeDock:
        def __init__(self) -> None:
            self.hidden = False

        def hide(self) -> None:
            self.hidden = True

    class FakeMenuBar:
        def __init__(self) -> None:
            self.hidden = False

        def hide(self) -> None:
            self.hidden = True

    class FakeQtWindow:
        def __init__(self) -> None:
            self.menu_bar = FakeMenuBar()

        def menuBar(self):  # noqa: N802 - Qt API name
            return self.menu_bar

    class FakeQtViewer:
        def __init__(self) -> None:
            self.stylesheet = None
            self.welcome_visible = None
            self.canvas = FakeCanvas()
            self.dockLayerControls = FakeDock()
            self.dockLayerList = FakeDock()

        def setStyleSheet(self, stylesheet: str) -> None:  # noqa: N802 - Qt API name
            self.stylesheet = stylesheet

        def set_welcome_visible(self, visible: bool) -> None:
            self.welcome_visible = visible

    qt_window = FakeQtWindow()
    qt_viewer = FakeQtViewer()
    viewer = make_viewer_namespace(_qt_window=qt_window, _qt_viewer=qt_viewer)

    configure_napari_chrome(viewer)

    assert qt_window.menu_bar.hidden is True
    assert qt_viewer.stylesheet == 'background: #767676;'
    assert qt_viewer.canvas.bgcolor == '#767676'
    assert qt_viewer.canvas.native.stylesheet == 'background: #767676;'
    assert qt_viewer.welcome_visible is False
    assert qt_viewer.dockLayerControls.hidden is True
    assert qt_viewer.dockLayerList.hidden is True


def test_configure_napari_chrome_uses_gray_18_canvas_when_enabled() -> None:
    class FakeNative:
        def __init__(self) -> None:
            self.stylesheet = None

        def setStyleSheet(self, stylesheet: str) -> None:  # noqa: N802 - Qt API name
            self.stylesheet = stylesheet

    class FakeCanvas:
        def __init__(self) -> None:
            self.bgcolor = None
            self.native = FakeNative()

    class FakeQtWindow:
        def menuBar(self):  # noqa: N802 - Qt API name
            return None

    class FakeQtViewer:
        def __init__(self) -> None:
            self.stylesheet = None
            self.canvas = FakeCanvas()
            self.dockLayerControls = SimpleNamespace(hide=lambda: None)
            self.dockLayerList = SimpleNamespace(hide=lambda: None)

        def setStyleSheet(self, stylesheet: str) -> None:  # noqa: N802 - Qt API name
            self.stylesheet = stylesheet

        def set_welcome_visible(self, visible: bool) -> None:
            return None

    viewer = make_viewer_namespace(_qt_window=FakeQtWindow(), _qt_viewer=FakeQtViewer())

    configure_napari_chrome(viewer, gray_18_canvas=True)

    assert viewer.window._qt_viewer.stylesheet == 'background: #767676;'
    assert viewer.window._qt_viewer.canvas.bgcolor == '#767676'
    assert viewer.window._qt_viewer.canvas.native.stylesheet == 'background: #767676;'


def test_reset_viewer_camera_calls_viewer_reset() -> None:
    layer = FakeLayer(name='active')
    viewer = SimpleNamespace(reset_calls=0, layers=FakeLayerList([layer], active=layer))

    def reset_view() -> None:
        viewer.reset_calls += 1

    viewer.reset_view = reset_view

    reset_viewer_camera(viewer)

    assert viewer.reset_calls == 1


def test_reset_viewer_camera_fits_active_layer_only_during_reset() -> None:
    active_layer = FakeLayer(name='active', visible=True)
    other_layer = FakeLayer(name='other', visible=True)
    viewer = SimpleNamespace(
        reset_calls=0,
        layers=FakeLayerList([other_layer, active_layer], active=active_layer),
        visibility_during_reset=None,
    )

    def reset_view() -> None:
        viewer.reset_calls += 1
        viewer.visibility_during_reset = [layer.visible for layer in viewer.layers]

    viewer.reset_view = reset_view

    reset_viewer_camera(viewer)

    assert viewer.reset_calls == 1
    assert viewer.visibility_during_reset == [False, True]
    assert [layer.visible for layer in viewer.layers] == [True, True]


def test_reset_viewer_camera_falls_back_to_top_visible_layer_when_no_active_layer() -> None:
    hidden_layer = FakeLayer(name='hidden', visible=False)
    visible_layer = FakeLayer(name='visible', visible=True)
    viewer = SimpleNamespace(
        reset_calls=0,
        layers=FakeLayerList([hidden_layer, visible_layer], active=None),
        visibility_during_reset=None,
    )

    def reset_view() -> None:
        viewer.reset_calls += 1
        viewer.visibility_during_reset = [layer.visible for layer in viewer.layers]

    viewer.reset_view = reset_view

    reset_viewer_camera(viewer)

    assert viewer.reset_calls == 1
    assert viewer.visibility_during_reset == [False, True]
    assert [layer.visible for layer in viewer.layers] == [False, True]


def test_reset_viewer_camera_ignores_missing_reset_method() -> None:
    viewer = SimpleNamespace()

    reset_viewer_camera(viewer)


def test_set_viewer_zoom_percent_defaults_to_zoom_one_hundred() -> None:
    camera = SimpleNamespace(zoom=4.0)
    viewer = make_viewer_namespace(camera=camera, _qt_viewer=None)

    set_viewer_zoom_percent(viewer, 100.0)

    assert camera.zoom == 1.0


def test_set_viewer_zoom_percent_scales_with_device_pixel_ratio() -> None:
    camera = SimpleNamespace(zoom=4.0)
    qt_viewer = SimpleNamespace(devicePixelRatioF=lambda: 1.5)
    viewer = make_viewer_namespace(camera=camera, _qt_viewer=qt_viewer)

    set_viewer_zoom_percent(viewer, 200.0)

    assert camera.zoom == 3.0


def test_request_dark_title_bar_returns_false_off_windows(monkeypatch) -> None:
    class FakeWindow:
        def winId(self):  # noqa: N802 - Qt API name
            raise AssertionError('winId should not be requested on non-Windows platforms')

    monkeypatch.setattr(napari_layout_module.sys, 'platform', 'linux')

    assert napari_layout_module._request_dark_title_bar(FakeWindow()) is False


def test_request_dark_title_bar_uses_windows_dwm_api(monkeypatch) -> None:
    class FakeWindow:
        def winId(self):  # noqa: N802 - Qt API name
            return 1234

    class FakeCtypes:
        def __init__(self) -> None:
            self.calls: list[tuple[int, int, int]] = []
            self.windll = SimpleNamespace(dwmapi=SimpleNamespace(DwmSetWindowAttribute=self._set_window_attribute))

        @staticmethod
        def c_int(value: int) -> int:
            return value

        @staticmethod
        def sizeof(value: object) -> int:
            return 4

        @staticmethod
        def byref(value: object) -> object:
            return value

        def _set_window_attribute(self, hwnd: int, attribute: int, value: object, value_size: int) -> int:
            self.calls.append((hwnd, attribute, value_size))
            return 0 if attribute == napari_layout_module._DWMWA_USE_IMMERSIVE_DARK_MODE else 1

    fake_ctypes = FakeCtypes()
    monkeypatch.setattr(napari_layout_module.sys, 'platform', 'win32')
    monkeypatch.setattr(
        napari_layout_module,
        'import_module',
        lambda name: fake_ctypes if name == 'ctypes' else __import__(name),
    )

    assert napari_layout_module._request_dark_title_bar(FakeWindow()) is True
    assert fake_ctypes.calls == [(1234, napari_layout_module._DWMWA_USE_IMMERSIVE_DARK_MODE, 4)]


def test_selected_menu_items_use_accent_border() -> None:
    assert 'QComboBox QAbstractItemView::item:selected {' in CONTROL_STYLE
    assert 'QMenu::item:selected {' in CONTROL_STYLE
    assert 'QAbstractItemView::item:selected {' in CONTROL_STYLE
    assert f'    selection-background-color: {CONTROL_BG};' in CONTROL_STYLE
    assert f'    background: {CONTROL_BG};' in CONTROL_STYLE
    assert f'    border-left: 2px solid {TEXT_ACCENT};' in CONTROL_STYLE


