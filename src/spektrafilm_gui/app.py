from dataclasses import dataclass, fields
from importlib import import_module
from typing import Any, Callable, cast

import numpy as np
from qtpy import QtCore, QtGui, QtWidgets

from spektrafilm_gui.controller import GuiController
from spektrafilm_gui.napari_layout import (
    build_controls_panel,
    build_main_window,
    configure_napari_chrome,
    show_viewer_window,
)
from spektrafilm_gui.persistence import load_default_gui_state
from spektrafilm_gui.state_bridge import GUI_STATE_SECTION_NAMES, apply_gui_state
from spektrafilm_gui.theme_palette import GRAY_0, GRAY_1, GRAY_2, GRAY_3, TEXT_DIM, TEXT_MAIN, TEXT_SELECTION_BG
from spektrafilm_gui.widgets import WidgetBundle, create_widget_bundle
from spektrafilm.utils.numba_warmup import warmup

QThreadPool = getattr(QtCore, 'QThreadPool')
QRunnable = getattr(QtCore, 'QRunnable')
QTimer = getattr(QtCore, 'QTimer')

_background_warmup_started = False
_background_warmup_scheduled = False
_background_warmup_pool: Any | None = None
WARMUP_IMAGE_SHAPE = (16, 16, 3)

@dataclass(slots=True)
class GuiApp:
    viewer: Any
    widgets: WidgetBundle
    controller: GuiController
    main_window: QtWidgets.QMainWindow


def _warmup_launch_input_path(gui_state: object | None = None) -> None:
    state = load_default_gui_state() if gui_state is None else gui_state
    try:
        colour_module = import_module('colour')
        controller_runtime = import_module('spektrafilm_gui.controller_runtime')
        import_module('spektrafilm.utils.io')
        import_module('spektrafilm.utils.preview')
    except (AttributeError, ImportError, LookupError, OSError, RuntimeError, TypeError, ValueError):
        return

    warmup_image = np.full(WARMUP_IMAGE_SHAPE, 0.18, dtype=np.float64)
    try:
        controller_runtime.prepare_input_color_preview_image(
            warmup_image,
            input_color_space=state.input_image.input_color_space,
            apply_cctf_decoding=state.input_image.apply_cctf_decoding,
            colour_module=colour_module,
        )
    except (AttributeError, LookupError, RuntimeError, TypeError, ValueError):
        return


def _warmup_full_gui() -> None:
    gui_state = load_default_gui_state()
    warmup()

    colour_module = import_module('colour')
    pil_image_module = import_module('PIL.Image')
    imagecms_module = import_module('PIL.ImageCms')
    controller_runtime = import_module('spektrafilm_gui.controller_runtime')
    params_mapper = import_module('spektrafilm_gui.params_mapper')
    runtime_api = import_module('spektrafilm.runtime.api')
    import_module('spektrafilm.utils.io')
    import_module('spektrafilm.utils.preview')
    import_module('spektrafilm.utils.raw_file_processor')

    warmup_image = np.full(WARMUP_IMAGE_SHAPE, 0.18, dtype=np.float64)
    controller_runtime.prepare_input_color_preview_image(
        warmup_image,
        input_color_space=gui_state.input_image.input_color_space,
        apply_cctf_decoding=gui_state.input_image.apply_cctf_decoding,
        colour_module=colour_module,
    )

    params = params_mapper.build_params_from_state(gui_state)
    simulator = runtime_api.Simulator(runtime_api.digest_params(params))
    scan = np.asarray(simulator.process(warmup_image), dtype=np.float32)

    # Force the display path once as part of startup so the first preview avoids lazy import/setup cost.
    controller_runtime.prepare_output_display_image(
        scan,
        output_color_space=gui_state.simulation.output_color_space,
        use_display_transform=True,
        imagecms_module=imagecms_module,
        colour_module=colour_module,
        pil_image_module=pil_image_module,
    )


class _WarmupTask(QRunnable):
    def __init__(self, *, warmup_fn: Callable[[], None] | None = None) -> None:
        super().__init__()
        self._warmup_fn = _warmup_full_gui if warmup_fn is None else warmup_fn

    def run(self) -> None:
        try:
            self._warmup_fn()
        except (AttributeError, ImportError, LookupError, OSError, RuntimeError, TypeError, ValueError):
            return


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
    thread_pool_factory: Callable[[], Any] | None = None,
    task_factory: Callable[[], QRunnable] = _WarmupTask,
) -> None:
    global _background_warmup_started, _background_warmup_scheduled, _background_warmup_pool
    if _background_warmup_started:
        return
    _background_warmup_started = True
    _background_warmup_scheduled = False
    if thread_pool is None:
        if _background_warmup_pool is None:
            pool_factory = QThreadPool if thread_pool_factory is None else thread_pool_factory
            _background_warmup_pool = pool_factory()
            set_max_thread_count = getattr(_background_warmup_pool, 'setMaxThreadCount', None)
            if callable(set_max_thread_count):
                set_max_thread_count(1)
        pool = _background_warmup_pool
    else:
        pool = thread_pool
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


def _connect_auto_preview_signal(widget: Any, callback: Callable[..., None]) -> None:
    editors = getattr(widget, '_editors', None)
    if editors is not None:
        for editor in editors:
            _connect_auto_preview_signal(editor, callback)
        return

    for signal_name in ('toggled', 'currentTextChanged', 'valueChanged'):
        signal = getattr(widget, signal_name, None)
        if signal is not None and hasattr(signal, 'connect'):
            signal.connect(callback)
            return


def connect_auto_preview_signals(controller: GuiController, widgets: WidgetBundle) -> None:
    for section_name in GUI_STATE_SECTION_NAMES:
        section = getattr(widgets, section_name, None)
        if section is None:
            continue

        state_cls = getattr(section, '_state_cls', None)
        if state_cls is None:
            continue

        for field_info in fields(state_cls):
            if section_name == 'display' and field_info.name in {'preview_max_size', 'output_interpolation'}:
                continue
            _connect_auto_preview_signal(getattr(section, field_info.name), controller.request_auto_preview)

    widgets.simulation.bottom_auto_preview.toggled.connect(controller.request_auto_preview)
    widgets.simulation.bottom_scan_film.toggled.connect(controller.request_auto_preview)
    widgets.simulation.bottom_scan_for_print.toggled.connect(controller.request_auto_preview)


def connect_controller_signals(controller: GuiController, widgets: WidgetBundle) -> None:
    widgets.filepicker.load_requested.connect(controller.load_input_image)
    widgets.load_raw.load_requested.connect(controller.load_raw_image)
    widgets.simulation.film_stock.textActivated.connect(controller.apply_profile_defaults)
    widgets.simulation.print_paper.textActivated.connect(controller.apply_profile_defaults)
    widgets.gui_config.save_current_as_default_requested.connect(controller.save_current_as_default)
    widgets.gui_config.save_current_to_file_requested.connect(controller.save_current_state_to_file)
    widgets.gui_config.load_from_file_requested.connect(controller.load_state_from_file)
    widgets.gui_config.restore_factory_default_requested.connect(controller.restore_factory_default)
    widgets.simulation.preview_requested.connect(controller.run_preview)
    widgets.simulation.scan_requested.connect(controller.run_scan)
    widgets.simulation.save_requested.connect(controller.save_output_layer)
    widgets.display.use_display_transform.toggled.connect(controller.report_display_transform_status)
    widgets.display.gray_18_canvas.toggled.connect(controller.set_gray_18_canvas_enabled)
    widgets.display.output_interpolation.currentTextChanged.connect(controller.set_output_interpolation_mode)
    widgets.display.update_preview_requested.connect(controller.refresh_preview_cache)
    widgets.display.update_preview_requested.connect(controller.request_auto_preview)
    connect_auto_preview_signals(controller, widgets)


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
    show_startup_placeholder = getattr(controller, 'show_startup_placeholder', None)
    if callable(show_startup_placeholder):
        show_startup_placeholder()
    connect_signals_fn(controller, widgets)
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
    gui_state = load_default_gui_state()
    apply_gui_state(gui_state, widgets=widgets)
    _warmup_launch_input_path(gui_state)
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
