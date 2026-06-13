from __future__ import annotations

from types import SimpleNamespace

from qtpy import QtGui

from spektrafilm_gui import app as app_module
from spektrafilm_gui import params_manifest as param_manifest_module
from spektrafilm_gui import state as state_module
from spektrafilm_gui.state_bridge import GUI_STATE_SECTION_NAMES

from .helpers import StubToggle, make_test_gui_state


class FakeSignal:
    def __init__(self) -> None:
        self.connected: list[object] = []

    def connect(self, callback) -> None:
        self.connected.append(callback)


def _make_auto_preview_editor(value):
    if isinstance(value, bool):
        return SimpleNamespace(toggled=FakeSignal())
    if isinstance(value, str):
        return SimpleNamespace(currentTextChanged=FakeSignal())
    if isinstance(value, tuple):
        return SimpleNamespace(_editors=[SimpleNamespace(valueChanged=FakeSignal()) for _ in value])
    return SimpleNamespace(valueChanged=FakeSignal())


def _section_state(state, section_name: str):
    if section_name == 'display':
        return state.gui_only.display
    if section_name == 'load_raw':
        return state.gui_only.load_raw
    return getattr(state, section_name)


def test_create_viewer_uses_system_dark_theme(monkeypatch) -> None:
    fake_viewer = object()
    fake_appearance = SimpleNamespace(theme=None)
    fake_settings = SimpleNamespace(appearance=fake_appearance)

    monkeypatch.setattr(
        app_module,
        'import_module',
        lambda name: SimpleNamespace(Viewer=lambda show=False: fake_viewer)
        if name == 'napari'
        else SimpleNamespace(get_settings=lambda: fake_settings),
    )

    viewer = app_module._create_viewer()

    assert viewer is fake_viewer
    assert fake_appearance.theme == 'dark'


def test_apply_app_palette_uses_fixed_dark_palette(monkeypatch) -> None:
    captured: dict[str, object] = {}
    fake_app = SimpleNamespace(setPalette=lambda palette: captured.setdefault('palette', palette))

    monkeypatch.setattr(app_module.QtWidgets.QApplication, 'instance', staticmethod(lambda: fake_app))

    app_module._apply_app_palette()

    palette = captured['palette']
    assert isinstance(palette, QtGui.QPalette)
    assert palette.color(QtGui.QPalette.Window).name() == app_module.GRAY_0
    assert palette.color(QtGui.QPalette.Base).name() == app_module.GRAY_1
    assert palette.color(QtGui.QPalette.AlternateBase).name() == app_module.GRAY_2
    assert palette.color(QtGui.QPalette.WindowText).name() == app_module.TEXT_MAIN
    assert palette.color(QtGui.QPalette.Highlight).name() == app_module.TEXT_SELECTION_BG


def test_create_app_builds_window_shell_and_defers_controls(monkeypatch) -> None:
    # Phase 1: a light window shell is built and the controls build is deferred;
    # no widgets/controller exist yet.
    captured: dict[str, object] = {}
    fake_viewer = object()
    fake_window = SimpleNamespace(mount_controls=lambda panel: None)

    monkeypatch.setattr(app_module, '_create_viewer', lambda: fake_viewer)
    monkeypatch.setattr(app_module, '_apply_app_palette', lambda: captured.setdefault('palette', True))
    monkeypatch.setattr(
        app_module,
        'configure_napari_chrome',
        lambda viewer, *, gray_18_canvas=True: captured.setdefault('chrome', (viewer, gray_18_canvas)),
    )

    def fake_build_main_window(viewer, *, on_rotate_ccw=None, on_rotate_cw=None):
        captured['window_args'] = (viewer, on_rotate_ccw, on_rotate_cw)
        return fake_window

    monkeypatch.setattr(app_module, 'build_main_window', fake_build_main_window)
    monkeypatch.setattr(app_module, '_schedule_startup', lambda app: captured.setdefault('scheduled_app', app))

    app = app_module.create_app()

    assert captured['palette'] is True
    assert captured['chrome'] == (fake_viewer, app_module.DEFAULT_GRAY_18_CANVAS)
    assert app.viewer is fake_viewer
    assert app.main_window is fake_window
    assert app.widgets is None and app.controller is None
    assert captured['scheduled_app'] is app
    # Rotate callbacks are no-ops until the controller exists (does not raise).
    captured['window_args'][1]()
    captured['window_args'][2]()


def test_schedule_startup_warms_off_thread_then_builds_controls(monkeypatch) -> None:
    captured: dict[str, object] = {}
    submitted: dict[str, object] = {}

    class FakeRunner:
        def submit(self, channel, work, *, on_done, on_error=None):
            submitted.update(channel=channel, work=work, on_done=on_done, on_error=on_error)

    app = app_module.GuiApp(viewer=object(), runner=FakeRunner())

    monkeypatch.setattr(app_module, 'set_status', lambda *args, **kwargs: None)
    monkeypatch.setattr(app_module, '_build_controls', lambda built: captured.setdefault('built', built))

    def fake_single_shot(delay_ms, callback):
        captured['scheduled'] = (delay_ms, callback)

    app_module._schedule_startup(app, single_shot_fn=fake_single_shot)

    delay, callback = captured['scheduled']
    assert delay == 0
    assert 'channel' not in submitted  # nothing submitted until the timer fires

    callback()
    assert submitted['channel'] == 'startup'
    assert submitted['work'] is app_module._warmup_full_gui

    # The controls build runs whether warmup succeeds or fails.
    submitted['on_done'](None)
    assert captured['built'] is app
    submitted['on_error']('boom')
    assert captured['built'] is app


def test_build_controls_builds_widgets_controller_and_mounts(monkeypatch) -> None:
    captured: dict[str, object] = {}
    fake_widgets = SimpleNamespace(display=SimpleNamespace(gray_18_canvas=StubToggle(True)))
    fake_controls = object()
    fake_window = SimpleNamespace(mount_controls=lambda panel: captured.setdefault('mounted', panel))
    app = app_module.GuiApp(viewer=object(), runner=SimpleNamespace(), main_window=fake_window)

    class FakeController:
        def __init__(self, *, viewer, widgets):
            captured['ctor'] = (viewer, widgets)

        def sync_display_transform_availability(self, *, report_status):
            captured['sync'] = report_status

        def show_startup_placeholder(self):
            captured['placeholder'] = True

    monkeypatch.setattr('spektrafilm_gui.widgets.create_widget_bundle', lambda: fake_widgets)
    monkeypatch.setattr('spektrafilm_gui.persistence.load_default_gui_state', lambda: 'gui-state')
    monkeypatch.setattr(
        'spektrafilm_gui.state_bridge.apply_gui_state',
        lambda state, *, widgets: captured.setdefault('applied', (state, widgets)),
    )
    monkeypatch.setattr('spektrafilm_gui.controller.GuiController', FakeController)
    monkeypatch.setattr(app_module, 'configure_napari_chrome', lambda *args, **kwargs: None)
    monkeypatch.setattr(app_module, 'build_controls_panel', lambda viewer, widgets: fake_controls)
    monkeypatch.setattr(app_module, 'connect_controller_signals', lambda c, w: captured.setdefault('connected', (c, w)))
    monkeypatch.setattr(app_module, 'set_status', lambda *args, **kwargs: None)

    app_module._build_controls(app)

    assert captured['applied'] == ('gui-state', fake_widgets)
    assert captured['sync'] is False
    assert captured['placeholder'] is True
    assert captured['mounted'] is fake_controls
    assert app.widgets is fake_widgets
    assert isinstance(app.controller, FakeController)
    assert captured['connected'][0] is app.controller


def test_connect_controller_signals_wires_all_widget_events() -> None:
    captured: dict[str, object] = {}
    controller = SimpleNamespace(
        load_input_image=object(),
        load_raw_image=object(),
        apply_profile_defaults=object(),
        save_current_as_default=object(),
        save_current_state_to_file=object(),
        load_state_from_file=object(),
        restore_factory_default=object(),
        run_preview=object(),
        run_scan=object(),
        save_output_layer=object(),
        report_display_transform_status=object(),
        set_gray_18_canvas_enabled=object(),
        set_output_interpolation_mode=object(),
        refresh_preview_cache=object(),
        request_auto_preview=object(),
    )
    original_connect_auto_preview_signals = app_module.connect_auto_preview_signals
    widgets = SimpleNamespace(
        filepicker=SimpleNamespace(load_requested=FakeSignal()),
        load_raw=SimpleNamespace(load_requested=FakeSignal()),
        gui_config=SimpleNamespace(
            save_current_as_default_requested=FakeSignal(),
            save_current_to_file_requested=FakeSignal(),
            load_from_file_requested=FakeSignal(),
            restore_factory_default_requested=FakeSignal(),
        ),
        simulation=SimpleNamespace(
            film_stock=SimpleNamespace(textActivated=FakeSignal()),
            print_paper=SimpleNamespace(textActivated=FakeSignal()),
            preview_requested=FakeSignal(),
            scan_requested=FakeSignal(),
            save_requested=FakeSignal(),
        ),
        display=SimpleNamespace(
            use_display_transform=SimpleNamespace(toggled=FakeSignal()),
            gray_18_canvas=SimpleNamespace(toggled=FakeSignal()),
            output_interpolation=SimpleNamespace(currentTextChanged=FakeSignal()),
            preview_max_size=SimpleNamespace(valueChanged=FakeSignal()),
            update_preview_requested=FakeSignal(),
        ),
    )

    try:
        app_module.connect_auto_preview_signals = lambda ctl, wdg: captured.setdefault('auto_preview_args', (ctl, wdg))
        app_module.connect_controller_signals(controller, widgets)
    finally:
        app_module.connect_auto_preview_signals = original_connect_auto_preview_signals

    assert widgets.filepicker.load_requested.connected == [controller.load_input_image]
    assert widgets.load_raw.load_requested.connected == [controller.load_raw_image]
    assert widgets.simulation.film_stock.textActivated.connected == [controller.apply_profile_defaults]
    assert widgets.simulation.print_paper.textActivated.connected == [controller.apply_profile_defaults]
    assert widgets.gui_config.save_current_as_default_requested.connected == [controller.save_current_as_default]
    assert widgets.gui_config.save_current_to_file_requested.connected == [controller.save_current_state_to_file]
    assert widgets.gui_config.load_from_file_requested.connected == [controller.load_state_from_file]
    assert widgets.gui_config.restore_factory_default_requested.connected == [controller.restore_factory_default]
    assert widgets.simulation.preview_requested.connected == [controller.run_preview]
    assert widgets.simulation.scan_requested.connected == [controller.run_scan]
    assert widgets.simulation.save_requested.connected == [controller.save_output_layer]
    assert widgets.display.use_display_transform.toggled.connected == [controller.report_display_transform_status]
    assert widgets.display.gray_18_canvas.toggled.connected == [controller.set_gray_18_canvas_enabled]
    assert widgets.display.output_interpolation.currentTextChanged.connected == [controller.set_output_interpolation_mode]
    assert widgets.display.update_preview_requested.connected == [
        controller.refresh_preview_cache,
        controller.request_auto_preview,
    ]
    assert captured['auto_preview_args'] == (controller, widgets)


def test_connect_auto_preview_signals_covers_hidden_linked_controls_and_footer_toggles() -> None:
    gui_state = make_test_gui_state()
    controller = SimpleNamespace(request_auto_preview=lambda *args: None)
    widgets = SimpleNamespace()

    for section_name in GUI_STATE_SECTION_NAMES:
        if section_name == 'load_raw':
            setattr(widgets, section_name, None)
            continue
        state_section = _section_state(gui_state, section_name)
        section = SimpleNamespace(_is_params_group=True)
        if section_name == 'input_image':
            # input_image is now a path-bound section (_is_params_group): editors
            # keyed by leaf, bound to dotted paths on InputImageState (io.* / settings.*).
            ii_editors: dict = {}
            for spec in param_manifest_module.INPUT_IMAGE_FIELDS:
                editor = _make_auto_preview_editor(state_module._read_attr_path(state_section, spec.path))
                ii_editors[spec.leaf] = editor
                setattr(section, spec.leaf, editor)
            section._editors = ii_editors
            setattr(widgets, section_name, section)
            continue
        elif section_name == 'display':
            section._skip_auto_preview_leaves = {'preview_max_size', 'output_interpolation'}
            display_editors: dict = {}
            for spec in param_manifest_module.DISPLAY_PANEL_FIELDS:
                editor = _make_auto_preview_editor(state_module._read_attr_path(state_section, spec.path))
                display_editors[spec.leaf] = editor
                setattr(section, spec.leaf, editor)
            section._editors = display_editors
        elif section_name == 'special':
            special_editors: dict = {}
            for spec in param_manifest_module.SPECIAL_FIELDS:
                editor = _make_auto_preview_editor(state_module._read_attr_path(state_section, spec.path))
                special_editors[spec.leaf] = editor
                setattr(section, spec.leaf, editor)
            section._editors = special_editors
        elif section_name == 'simulation':
            section._skip_auto_preview_leaves = {'auto_preview'}
            simulation_editors: dict = {}
            for spec in param_manifest_module.SIMULATION_FIELDS:
                editor = _make_auto_preview_editor(state_module._read_attr_path(state_section, spec.path))
                simulation_editors[spec.leaf] = editor
                setattr(section, spec.leaf, editor)
            section._editors = simulation_editors
        else:
            group_editors: dict = {}
            for field_name, value in vars(state_section).items():
                editor = _make_auto_preview_editor(value)
                group_editors[field_name] = editor
                setattr(section, field_name, editor)
            section._editors = group_editors
        setattr(widgets, section_name, section)

    widgets.simulation.bottom_auto_preview = SimpleNamespace(toggled=FakeSignal())
    widgets.simulation.bottom_scan_for_print = SimpleNamespace(toggled=FakeSignal())

    app_module.connect_auto_preview_signals(controller, widgets)

    assert widgets.input_image.upscale_factor.valueChanged.connected == [controller.request_auto_preview]
    assert widgets.input_image.crop_size._editors[0].valueChanged.connected == [controller.request_auto_preview]
    assert widgets.simulation.print_y_filter_shift.valueChanged.connected == [controller.request_auto_preview]
    assert widgets.camera.film_format_mm.valueChanged.connected == [controller.request_auto_preview]
    assert widgets.enlarger_diffusion.strength.valueChanged.connected == [controller.request_auto_preview]
    assert widgets.camera.exposure_compensation_ev.valueChanged.connected == [controller.request_auto_preview]
    assert widgets.scanner.lens_blur.valueChanged.connected == [controller.request_auto_preview]
    assert widgets.scanner.white_correction.toggled.connected == [controller.request_auto_preview]
    assert widgets.scanner.white_level.valueChanged.connected == [controller.request_auto_preview]
    assert widgets.scanner.unsharp_mask._editors[0].valueChanged.connected == [controller.request_auto_preview]
    assert widgets.display.output_interpolation.currentTextChanged.connected == []
    assert widgets.display.preview_max_size.valueChanged.connected == []
    assert widgets.simulation.output_color_space.currentTextChanged.connected == [controller.request_auto_preview]
    assert widgets.simulation.bottom_auto_preview.toggled.connected == [controller.request_auto_preview]
    assert widgets.simulation.bottom_scan_for_print.toggled.connected == [controller.request_auto_preview]
    assert widgets.simulation.route.currentTextChanged.connected == [controller.request_auto_preview]


def test_connect_auto_preview_signals_wires_params_group_section_editors() -> None:
    # Path-bound group sections (ParamsGroupSection) expose their editors via
    # the _editors mapping; the wiring must reach them through the
    # _is_params_group branch (regression guard).
    controller = SimpleNamespace(request_auto_preview=lambda *args: None)
    scanner_editors = {
        'lens_blur': _make_auto_preview_editor(0.0),
        'white_correction': _make_auto_preview_editor(True),
        'unsharp_mask': _make_auto_preview_editor((0.7, 0.7)),
    }
    scanner_section = SimpleNamespace(_is_params_group=True, _editors=scanner_editors)
    camera_editors = {'film_format_mm': _make_auto_preview_editor(35.0)}
    camera_section = SimpleNamespace(_is_params_group=True, _editors=camera_editors)
    widgets = SimpleNamespace()
    for section_name in GUI_STATE_SECTION_NAMES:
        setattr(widgets, section_name, None)
    widgets.scanner = scanner_section
    widgets.camera = camera_section
    widgets.simulation = SimpleNamespace(
        bottom_auto_preview=SimpleNamespace(toggled=FakeSignal()),
        bottom_scan_for_print=SimpleNamespace(toggled=FakeSignal()),
        _is_params_group=False,
    )

    app_module.connect_auto_preview_signals(controller, widgets)

    assert scanner_editors['lens_blur'].valueChanged.connected == [controller.request_auto_preview]
    assert scanner_editors['white_correction'].toggled.connected == [controller.request_auto_preview]
    assert scanner_editors['unsharp_mask']._editors[0].valueChanged.connected == [controller.request_auto_preview]
    assert camera_editors['film_format_mm'].valueChanged.connected == [controller.request_auto_preview]


