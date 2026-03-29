from __future__ import annotations

from types import SimpleNamespace

import pytest

from spektrafilm_gui import napari_layout as napari_layout_module
from spektrafilm_gui.napari_layout import (
    configure_napari_chrome,
    dialog_parent,
    reset_viewer_camera,
    set_host_window,
    set_status,
    set_viewer_zoom_percent,
    take_viewer_widget,
)
from spektrafilm_gui.theme_palette import CONTROL_BG, TEXT_ACCENT, TEXT_BRIGHT, TEXT_SELECTION_BG
from spektrafilm_gui.theme_styles import CHROME_STYLE, CONTROL_STYLE

from .helpers import FakeLayer, FakeLayerList, make_test_viewer_namespace


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


class FakeStyledWidget:
    def __init__(self) -> None:
        self.stylesheet = None

    def setStyleSheet(self, stylesheet: str) -> None:  # noqa: N802 - Qt API name
        self.stylesheet = stylesheet


class FakeCanvas:
    def __init__(self) -> None:
        self.bgcolor = None
        self.native = FakeStyledWidget()


class FakeHideable:
    def __init__(self) -> None:
        self.hidden = False

    def hide(self) -> None:
        self.hidden = True


class FakeQtWindow:
    def __init__(self, *, central_widget=None, menu_bar=None) -> None:
        self._central_widget = central_widget
        self.menu_bar = menu_bar

    def takeCentralWidget(self):  # noqa: N802 - Qt API name
        return self._central_widget

    def menuBar(self):  # noqa: N802 - Qt API name
        return self.menu_bar


class FakeQtViewer:
    def __init__(self) -> None:
        self.stylesheet = None
        self.welcome_visible = None
        self.canvas = FakeCanvas()
        self.dockLayerControls = FakeHideable()
        self.dockLayerList = FakeHideable()

    def setStyleSheet(self, stylesheet: str) -> None:  # noqa: N802 - Qt API name
        self.stylesheet = stylesheet

    def set_welcome_visible(self, visible: bool) -> None:
        self.welcome_visible = visible


def _make_resettable_viewer(layers: list[FakeLayer], *, active: FakeLayer | None):
    viewer = SimpleNamespace(
        reset_calls=0,
        layers=FakeLayerList(layers, active=active),
        visibility_during_reset=None,
    )

    def reset_view() -> None:
        viewer.reset_calls += 1
        viewer.visibility_during_reset = [layer.visible for layer in viewer.layers]

    viewer.reset_view = reset_view
    return viewer


def _assert_chrome_background(qt_viewer: FakeQtViewer) -> None:
    assert qt_viewer.stylesheet == 'background: #767676;'
    assert qt_viewer.canvas.bgcolor == '#767676'
    assert qt_viewer.canvas.native.stylesheet == 'background: #767676;'


def _assert_camera_reset(viewer, *, during: list[bool], after: list[bool]) -> None:
    assert viewer.reset_calls == 1
    assert viewer.visibility_during_reset == during
    assert [layer.visible for layer in viewer.layers] == after


def _assert_selected_rule(selector: str, background_rule: str) -> None:
    assert f'{selector} {{' in CONTROL_STYLE
    assert background_rule in CONTROL_STYLE


def test_splitter_handle_style_is_hairline() -> None:
    assert 'QSplitter::handle {' in CHROME_STYLE
    assert '    width: 1px;' in CHROME_STYLE
    assert 'QSplitter::handle:horizontal {' in CHROME_STYLE
    assert '    margin: 0 0 0 8px;' in CHROME_STYLE


def test_dialog_parent_prefers_custom_host_window() -> None:
    embedded_window = object()
    host_window = FakeWindowWithStatusBar()
    viewer = make_test_viewer_namespace(_qt_window=embedded_window)

    set_host_window(viewer, host_window)

    assert dialog_parent(viewer) is host_window


def test_set_status_targets_custom_host_window_status_bar() -> None:
    host_window = FakeWindowWithStatusBar()
    viewer = make_test_viewer_namespace(_qt_window=object())
    set_host_window(viewer, host_window)

    set_status(viewer, 'Simulation complete', timeout_ms=1500)

    assert host_window.status_bar.messages == [('Simulation complete', 1500)]


@pytest.mark.parametrize('central_widget', [object(), None], ids=['taken-central-widget', 'qt-viewer-fallback'])
def test_take_viewer_widget_prefers_central_widget_then_qt_viewer(central_widget) -> None:
    qt_viewer = object()
    viewer = make_test_viewer_namespace(_qt_window=FakeQtWindow(central_widget=central_widget), _qt_viewer=qt_viewer)

    expected = central_widget if central_widget is not None else qt_viewer
    assert take_viewer_widget(viewer) is expected


def test_configure_napari_chrome_hides_default_panels_and_sets_background() -> None:
    qt_window = FakeQtWindow(menu_bar=FakeHideable())
    qt_viewer = FakeQtViewer()
    viewer = make_test_viewer_namespace(_qt_window=qt_window, _qt_viewer=qt_viewer)

    configure_napari_chrome(viewer)

    assert qt_window.menu_bar.hidden is True
    _assert_chrome_background(qt_viewer)
    assert qt_viewer.welcome_visible is False
    assert qt_viewer.dockLayerControls.hidden is True
    assert qt_viewer.dockLayerList.hidden is True


def test_configure_napari_chrome_uses_gray_18_canvas_when_enabled() -> None:
    qt_viewer = FakeQtViewer()
    viewer = make_test_viewer_namespace(_qt_window=FakeQtWindow(), _qt_viewer=qt_viewer)

    configure_napari_chrome(viewer, gray_18_canvas=True)

    _assert_chrome_background(qt_viewer)


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
    viewer = _make_resettable_viewer([other_layer, active_layer], active=active_layer)

    reset_viewer_camera(viewer)

    _assert_camera_reset(viewer, during=[False, True], after=[True, True])


def test_reset_viewer_camera_falls_back_to_top_visible_layer_when_no_active_layer() -> None:
    hidden_layer = FakeLayer(name='hidden', visible=False)
    visible_layer = FakeLayer(name='visible', visible=True)
    viewer = _make_resettable_viewer([hidden_layer, visible_layer], active=None)

    reset_viewer_camera(viewer)

    _assert_camera_reset(viewer, during=[False, True], after=[False, True])


def test_reset_viewer_camera_ignores_missing_reset_method() -> None:
    viewer = SimpleNamespace()

    reset_viewer_camera(viewer)


@pytest.mark.parametrize(
    ('zoom_percent', 'device_pixel_ratio', 'expected_zoom'),
    [(100.0, None, 1.0), (200.0, 1.5, 3.0)],
    ids=['default-100-percent', 'scaled-by-device-ratio'],
)
def test_set_viewer_zoom_percent_updates_camera(zoom_percent: float, device_pixel_ratio: float | None, expected_zoom: float) -> None:
    camera = SimpleNamespace(zoom=4.0)
    qt_viewer = None
    if device_pixel_ratio is not None:
        qt_viewer = SimpleNamespace(devicePixelRatioF=lambda: device_pixel_ratio)
    viewer = make_test_viewer_namespace(camera=camera, _qt_viewer=qt_viewer)

    set_viewer_zoom_percent(viewer, zoom_percent)

    assert camera.zoom == expected_zoom


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
        def sizeof(_value: object) -> int:
            return 4

        @staticmethod
        def byref(value: object) -> object:
            return value

        def _set_window_attribute(self, hwnd: int, attribute: int, _value: object, value_size: int) -> int:
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
    _assert_selected_rule('QComboBox QAbstractItemView::item:selected', f'    selection-background-color: {CONTROL_BG};')
    _assert_selected_rule('QMenu::item:selected', f'    background: {CONTROL_BG};')
    _assert_selected_rule('QAbstractItemView::item:selected', f'    background: {CONTROL_BG};')
    assert f'    border-left: 2px solid {TEXT_ACCENT};' in CONTROL_STYLE


def test_text_fields_use_gray_selection_background() -> None:
    assert 'QLineEdit,' in CONTROL_STYLE
    assert 'QAbstractSpinBox,' in CONTROL_STYLE
    assert 'QTextEdit,' in CONTROL_STYLE
    assert 'QPlainTextEdit {' in CONTROL_STYLE
    assert f'    selection-background-color: {TEXT_SELECTION_BG};' in CONTROL_STYLE
    assert f'    selection-color: {TEXT_BRIGHT};' in CONTROL_STYLE