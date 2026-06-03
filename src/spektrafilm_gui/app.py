from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Any, Callable, cast

from qtpy import QtCore, QtGui, QtWidgets

from spektrafilm_gui.background import BackgroundRunner
from spektrafilm_gui.napari_layout import (
    build_controls_panel,
    build_main_window,
    configure_napari_chrome,
    set_status,
    show_viewer_window,
)
from spektrafilm_gui.theme_palette import GRAY_0, GRAY_1, GRAY_2, GRAY_3, TEXT_DIM, TEXT_MAIN, TEXT_SELECTION_BG

if TYPE_CHECKING:
    from spektrafilm_gui.controller import GuiController
    from spektrafilm_gui.widgets import WidgetBundle

QTimer = getattr(QtCore, 'QTimer')

WARMUP_IMAGE_SHAPE = (16, 16, 3)
# Canvas background before the saved state is loaded (loading state would pull
# the runtime). Re-applied from the real setting once controls are built.
DEFAULT_GRAY_18_CANVAS = True


@dataclass
class GuiApp:
    """Startup handle. The window shell exists immediately; ``widgets`` and
    ``controller`` are filled in once the deferred build completes."""

    viewer: Any
    runner: BackgroundRunner
    main_window: Any = None
    widgets: Any = None
    controller: Any = None


# ---------------------------------------------------------------------------
# Two-phase startup
#
# Phase 1 (create_app): build and show a light window shell — viewer + an empty
#   sidebar with a "Warming up…" placeholder. No runtime/colour/numba imports.
# Phase 2 (_warmup_full_gui, worker thread): import the heavy stack and prime
#   the pipeline while the window is already on screen and responsive.
# Phase 3 (_build_controls, GUI thread): now that the imports are warm, build
#   the widgets / controller / controls panel and mount them into the sidebar.
# ---------------------------------------------------------------------------


def create_app() -> GuiApp:
    viewer = _create_viewer()
    _apply_app_palette()
    configure_napari_chrome(viewer, gray_18_canvas=DEFAULT_GRAY_18_CANVAS)

    app = GuiApp(viewer=viewer, runner=BackgroundRunner())
    app.main_window = build_main_window(
        viewer,
        on_rotate_ccw=lambda: _rotate_input(app, clockwise=False),
        on_rotate_cw=lambda: _rotate_input(app, clockwise=True),
    )
    _schedule_startup(app)
    return app


def _schedule_startup(
    app: GuiApp,
    *,
    single_shot_fn: Callable[[int, Callable[[], None]], None] | None = None,
) -> None:
    """Warm the heavy imports off-thread once the event loop is running, then
    build the controls. Deferred with a 0 ms timer so the window paints first."""
    scheduler = QTimer.singleShot if single_shot_fn is None else single_shot_fn
    scheduler(0, lambda: _start_warmup(app))


def _start_warmup(app: GuiApp) -> None:
    set_status(app.viewer, 'Warming up...', timeout_ms=0)
    app.runner.submit(
        'startup',
        _warmup_full_gui,
        on_done=lambda _result: _build_controls(app),
        # Build the controls even if warmup failed — they just import cold.
        on_error=lambda _message: _build_controls(app),
    )


def _build_controls(app: GuiApp) -> None:
    """Phase 3 (GUI thread): build the runtime-bound GUI and mount it. The heavy
    modules are imported here, not at module scope, so importing this module —
    and therefore showing the window — stays cheap."""
    from spektrafilm_gui.controller import GuiController
    from spektrafilm_gui.persistence import load_default_gui_state
    from spektrafilm_gui.state_bridge import apply_gui_state
    from spektrafilm_gui.widgets import create_widget_bundle

    widgets = create_widget_bundle()
    gui_state = load_default_gui_state()
    apply_gui_state(gui_state, widgets=widgets)
    configure_napari_chrome(app.viewer, gray_18_canvas=gray_18_canvas_enabled(widgets))

    controller = GuiController(viewer=app.viewer, widgets=widgets)
    controller.sync_display_transform_availability(report_status=False)
    controller.show_startup_placeholder()
    connect_controller_signals(controller, widgets)

    app.main_window.mount_controls(build_controls_panel(app.viewer, widgets))
    app.widgets = widgets
    app.controller = controller
    set_status(app.viewer, 'Ready')


def _rotate_input(app: GuiApp, *, clockwise: bool) -> None:
    controller = app.controller
    if controller is None:
        return  # controls not built yet
    if clockwise:
        controller.rotate_input_image_clockwise()
    else:
        controller.rotate_input_image_counterclockwise()


def _warmup_full_gui() -> None:
    import numpy as np

    # Imported here (worker thread), not at module scope: this is the heavy
    # stack — numba, colour, the runtime pipeline, and the GUI modules that bind
    # to it — warmed so the controls build on the GUI thread is instant.
    import_module('spektrafilm.utils.numba_warmup').warmup()

    colour_module = import_module('colour')
    pil_image_module = import_module('PIL.Image')
    imagecms_module = import_module('PIL.ImageCms')
    controller_runtime = import_module('spektrafilm_gui.controller_runtime')
    params_mapper = import_module('spektrafilm_gui.params_mapper')
    runtime_api = import_module('spektrafilm.runtime.api')
    load_default_gui_state = import_module('spektrafilm_gui.persistence').load_default_gui_state
    import_module('spektrafilm.utils.io')
    import_module('spektrafilm.utils.preview')
    import_module('spektrafilm.utils.raw_file_processor')
    import_module('spektrafilm_gui.widgets')
    import_module('spektrafilm_gui.controller')

    gui_state = load_default_gui_state()
    warmup_image = np.full(WARMUP_IMAGE_SHAPE, 0.18, dtype=np.float64)
    controller_runtime.prepare_input_color_preview_image(
        warmup_image,
        input_color_space=gui_state.input_image.io.input_color_space,
        apply_cctf_decoding=gui_state.input_image.io.input_cctf_decoding,
        colour_module=colour_module,
    )

    params = params_mapper.build_params_from_state(gui_state)
    simulator = runtime_api.Simulator(runtime_api.digest_params(params))
    scan = np.asarray(simulator.process(warmup_image), dtype=np.float32)

    # Force the display path once so the first preview avoids lazy import/setup cost.
    controller_runtime.prepare_output_display_image(
        scan,
        output_color_space=gui_state.simulation.io.output_color_space,
        use_display_transform=True,
        imagecms_module=imagecms_module,
        colour_module=colour_module,
        pil_image_module=pil_image_module,
    )


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
    from spektrafilm_gui.state_bridge import GUI_STATE_SECTION_NAMES

    for section_name in GUI_STATE_SECTION_NAMES:
        section = getattr(widgets, section_name, None)
        if section is None or not getattr(section, '_is_params_group', False):
            continue

        skip_leaves = set(getattr(section, '_skip_auto_preview_leaves', ()))
        for leaf, editor in getattr(section, '_editors', {}).items():
            if leaf in skip_leaves:
                continue
            _connect_auto_preview_signal(editor, controller.request_auto_preview)

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


def main() -> None:
    napari = import_module('napari')
    app = create_app()
    show_viewer_window(app.viewer)
    napari.run()


if __name__ == "__main__":
    main()
