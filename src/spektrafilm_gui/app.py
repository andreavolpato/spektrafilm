from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable, cast

from qtpy import QtCore, QtGui, QtWidgets

from spektrafilm_gui.controller import GuiController
from spektrafilm_gui.napari_layout import (
    build_controls_panel,
    build_main_window,
    configure_napari_chrome,
    show_viewer_window,
)
from spektrafilm_gui.persistence import load_default_gui_state
from spektrafilm_gui.state_bridge import apply_gui_state
from spektrafilm_gui.theme_palette import ACCENT_COLOR_TEXT, GRAY_0, GRAY_1, GRAY_2, GRAY_3, TEXT_DIM, TEXT_MAIN, TEXT_SELECTION_BG
from spektrafilm_gui.widgets import WidgetBundle, create_widget_bundle
from spektrafilm.utils.numba_warmup import warmup

QThreadPool = getattr(QtCore, 'QThreadPool')
QRunnable = getattr(QtCore, 'QRunnable')
QTimer = getattr(QtCore, 'QTimer')

_background_warmup_started = False
_background_warmup_scheduled = False

@dataclass(slots=True)
class GuiApp:
    viewer: Any
    widgets: WidgetBundle
    controller: GuiController
    main_window: QtWidgets.QMainWindow


class _WarmupTask(QRunnable):
    def __init__(self, *, warmup_fn: Callable[[], None] = warmup) -> None:
        super().__init__()
        self._warmup_fn = warmup_fn

    def run(self) -> None:
        self._warmup_fn()


def _build_app_palette() -> QtGui.QPalette:
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor(GRAY_0))
    palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor(TEXT_MAIN))
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor(GRAY_1))
    palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(GRAY_2))
    palette.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor(GRAY_1))
    palette.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor(TEXT_MAIN))
    palette.setColor(QtGui.QPalette.Text, QtGui.QColor(TEXT_MAIN))
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor(GRAY_1))
    palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(TEXT_MAIN))
    palette.setColor(QtGui.QPalette.BrightText, QtGui.QColor(TEXT_MAIN))
    palette.setColor(QtGui.QPalette.Mid, QtGui.QColor(GRAY_3))
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(TEXT_SELECTION_BG))
    palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(TEXT_MAIN))
    placeholder_role = getattr(QtGui.QPalette, 'PlaceholderText', None)
    if placeholder_role is not None:
        palette.setColor(placeholder_role, QtGui.QColor(TEXT_DIM))

    palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, QtGui.QColor(TEXT_DIM))
    palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.Text, QtGui.QColor(TEXT_DIM))
    palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, QtGui.QColor(TEXT_DIM))
    palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.HighlightedText, QtGui.QColor(TEXT_DIM))
    return palette


def _apply_app_palette() -> None:
    app = QtWidgets.QApplication.instance()
    if app is None:
        return
    app.setPalette(_build_app_palette())


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


def _create_widgets() -> WidgetBundle:
    return create_widget_bundle()


def _start_background_warmup(
    *,
    thread_pool: Any | None = None,
    task_factory: Callable[[], QRunnable] = _WarmupTask,
) -> None:
    global _background_warmup_started, _background_warmup_scheduled
    if _background_warmup_started:
        return
    _background_warmup_started = True
    _background_warmup_scheduled = False
    pool = QThreadPool.globalInstance() if thread_pool is None else thread_pool
    pool.start(task_factory())


def _schedule_background_warmup(
    *,
    single_shot_fn: Callable[[int, Callable[[], None]], None] | None = None,
) -> None:
    global _background_warmup_scheduled
    if _background_warmup_started or _background_warmup_scheduled:
        return
    _background_warmup_scheduled = True
    scheduler = QTimer.singleShot if single_shot_fn is None else single_shot_fn
    scheduler(0, _start_background_warmup)


def connect_controller_signals(controller: GuiController, widgets: WidgetBundle) -> None:
    widgets.filepicker.load_requested.connect(controller.load_input_image)
    widgets.load_raw.load_requested.connect(controller.load_raw_image)
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


def gray_18_canvas_enabled(widgets: WidgetBundle) -> bool:
    toggle = getattr(widgets.display, 'gray_18_canvas', None)
    is_checked = getattr(toggle, 'isChecked', None)
    return bool(is_checked()) if callable(is_checked) else False


def initialize_controller(
    *,
    viewer: Any,
    widgets: WidgetBundle,
    controller_cls: type[GuiController] = GuiController,
    connect_signals_fn: Callable[[GuiController, WidgetBundle], None] = connect_controller_signals,
) -> GuiController:
    controller = controller_cls(viewer=viewer, widgets=widgets)
    controller.sync_display_transform_availability(report_status=False)
    connect_signals_fn(controller, widgets)
    controller.refresh_input_layers()
    return controller


def build_main_window_for_app(
    *,
    viewer: Any,
    widgets: WidgetBundle,
    configure_napari_chrome_fn: Callable[..., None] = configure_napari_chrome,
    build_controls_panel_fn: Callable[[Any, WidgetBundle], Any] = build_controls_panel,
    build_main_window_fn: Callable[[Any, Any], Any] = build_main_window,
) -> Any:
    configure_napari_chrome_fn(viewer, gray_18_canvas=gray_18_canvas_enabled(widgets))
    controls_panel = build_controls_panel_fn(viewer, widgets)
    return build_main_window_fn(viewer, controls_panel)


def create_app() -> GuiApp:
    viewer = _create_viewer()
    _apply_app_palette()
    widgets = _create_widgets()
    apply_gui_state(load_default_gui_state(), widgets=widgets)
    controller = initialize_controller(viewer=viewer, widgets=widgets)
    main_window = build_main_window_for_app(viewer=viewer, widgets=widgets)
    _schedule_background_warmup()
    return GuiApp(
        viewer=viewer,
        widgets=widgets,
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
