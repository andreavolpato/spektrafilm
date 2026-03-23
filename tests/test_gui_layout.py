from __future__ import annotations

from types import SimpleNamespace

from spektrafilm_gui.napari_layout import dialog_parent, set_host_window, set_status, take_viewer_widget


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
    viewer = SimpleNamespace(window=SimpleNamespace(_qt_window=embedded_window))

    set_host_window(viewer, host_window)

    assert dialog_parent(viewer) is host_window


def test_set_status_targets_custom_host_window_status_bar() -> None:
    host_window = FakeWindowWithStatusBar()
    viewer = SimpleNamespace(window=SimpleNamespace(_qt_window=object()))
    set_host_window(viewer, host_window)

    set_status(viewer, 'Simulation complete', timeout_ms=1500)

    assert host_window.status_bar.messages == [('Simulation complete', 1500)]


def test_take_viewer_widget_uses_taken_central_widget_when_available() -> None:
    central_widget = object()

    class FakeQtWindow:
        def takeCentralWidget(self):  # noqa: N802 - Qt API name
            return central_widget

    viewer = SimpleNamespace(window=SimpleNamespace(_qt_window=FakeQtWindow(), _qt_viewer=object()))

    assert take_viewer_widget(viewer) is central_widget


def test_take_viewer_widget_falls_back_to_qt_viewer() -> None:
    qt_viewer = object()

    class FakeQtWindow:
        def takeCentralWidget(self):  # noqa: N802 - Qt API name
            return None

    viewer = SimpleNamespace(window=SimpleNamespace(_qt_window=FakeQtWindow(), _qt_viewer=qt_viewer))

    assert take_viewer_widget(viewer) is qt_viewer