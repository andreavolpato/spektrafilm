from __future__ import annotations

from types import SimpleNamespace

from spektrafilm_gui.napari_layout import configure_napari_chrome, dialog_parent, reset_viewer_camera, set_host_window, set_status, take_viewer_widget
from spektrafilm_gui.theme_palette import GRAY_0
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
    assert qt_viewer.welcome_visible is False
    assert qt_viewer.dockLayerControls.hidden is True
    assert qt_viewer.dockLayerList.hidden is True
    assert qt_viewer.stylesheet == f'background: {GRAY_0};'
    assert qt_viewer.canvas.bgcolor == GRAY_0
    assert qt_viewer.canvas.native.stylesheet == f'background: {GRAY_0};'


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