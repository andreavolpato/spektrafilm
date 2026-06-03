from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from qtpy import QtGui

from spektrafilm_gui import app as app_module
from spektrafilm_gui import params_manifest as param_manifest_module
from spektrafilm_gui import state as state_module

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


def test_schedule_background_warmup_queues_only_once(monkeypatch) -> None:
    captured: list[tuple[int, object]] = []
    monkeypatch.setattr(app_module, '_background_warmup_started', False)
    monkeypatch.setattr(app_module, '_background_warmup_scheduled', False)
    monkeypatch.setattr(app_module, '_background_warmup_pool', None)

    def fake_single_shot(delay_ms: int, callback) -> None:
        captured.append((delay_ms, callback))

    app_module._schedule_background_warmup(single_shot_fn=fake_single_shot)
    app_module._schedule_background_warmup(single_shot_fn=fake_single_shot)

    assert captured == [(0, app_module._start_background_warmup)]


def test_start_background_warmup_starts_task_once(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(app_module, '_background_warmup_started', False)
    monkeypatch.setattr(app_module, '_background_warmup_scheduled', True)
    monkeypatch.setattr(app_module, '_background_warmup_pool', None)

    class FakeThreadPool:
        def __init__(self) -> None:
            self.started: list[object] = []

        def start(self, task) -> None:
            self.started.append(task)

    fake_task = object()
    fake_pool = FakeThreadPool()

    app_module._start_background_warmup(
        thread_pool=fake_pool,
        task_factory=lambda: captured.setdefault('task', fake_task),
    )
    app_module._start_background_warmup(
        thread_pool=fake_pool,
        task_factory=lambda: object(),
    )

    assert fake_pool.started == [fake_task]
    assert app_module._background_warmup_started is True
    assert app_module._background_warmup_scheduled is False


def test_start_background_warmup_uses_dedicated_pool_when_not_injected(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(app_module, '_background_warmup_started', False)
    monkeypatch.setattr(app_module, '_background_warmup_scheduled', True)
    monkeypatch.setattr(app_module, '_background_warmup_pool', None)

    class FakeThreadPool:
        def __init__(self) -> None:
            self.started: list[object] = []
            self.max_thread_count: int | None = None

        def setMaxThreadCount(self, value: int) -> None:  # noqa: N802 - Qt API name
            self.max_thread_count = value

        def start(self, task) -> None:
            self.started.append(task)

    app_module._start_background_warmup(
        thread_pool_factory=lambda: captured.setdefault('pool', FakeThreadPool()),
        task_factory=lambda: captured.setdefault('task', object()),
    )

    fake_pool = captured['pool']
    assert fake_pool.started == [captured['task']]
    assert fake_pool.max_thread_count == 1
    assert app_module._background_warmup_pool is fake_pool


def test_warmup_task_defaults_to_full_gui_warmup(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(app_module, '_warmup_full_gui', lambda: captured.setdefault('ran', True))

    app_module._WarmupTask().run()

    assert captured['ran'] is True


def test_warmup_task_swallows_background_failures() -> None:
    app_module._WarmupTask(warmup_fn=lambda: (_ for _ in ()).throw(RuntimeError('boom'))).run()


def test_warmup_launch_input_path_primes_first_image_load(monkeypatch) -> None:
    captured: dict[str, object] = {}
    fake_state = SimpleNamespace(
        input_image=SimpleNamespace(io=SimpleNamespace(input_color_space='ACES2065-1', input_cctf_decoding=False)),
    )
    fake_colour_module = object()
    fake_io_module = object()
    fake_preview_module = object()

    def fake_prepare_input_color_preview_image(image, **kwargs):
        captured['input_preview'] = (np.asarray(image), kwargs)
        return np.asarray(image, dtype=np.float32)

    fake_controller_runtime = SimpleNamespace(
        prepare_input_color_preview_image=fake_prepare_input_color_preview_image,
    )
    module_map = {
        'colour': fake_colour_module,
        'spektrafilm_gui.controller_runtime': fake_controller_runtime,
        'spektrafilm.utils.io': fake_io_module,
        'spektrafilm.utils.preview': fake_preview_module,
    }

    monkeypatch.setattr(app_module, 'import_module', lambda name: module_map[name])

    app_module._warmup_launch_input_path(fake_state)

    input_preview_image, input_preview_kwargs = captured['input_preview']
    assert input_preview_image.shape == app_module.WARMUP_IMAGE_SHAPE
    assert input_preview_kwargs['input_color_space'] == 'ACES2065-1'
    assert input_preview_kwargs['apply_cctf_decoding'] is False
    assert input_preview_kwargs['colour_module'] is fake_colour_module


def test_warmup_launch_input_path_swallows_launch_failures(monkeypatch) -> None:
    monkeypatch.setattr(app_module, 'import_module', lambda name: (_ for _ in ()).throw(RuntimeError(name)))

    app_module._warmup_launch_input_path(SimpleNamespace(input_image=SimpleNamespace(io=SimpleNamespace(input_color_space='sRGB', input_cctf_decoding=False))))


def test_create_app_syncs_display_transform_availability_before_connecting(monkeypatch) -> None:
    captured: dict[str, object] = {}

    fake_viewer = object()
    fake_widgets = SimpleNamespace(display=SimpleNamespace(use_display_transform=object(), gray_18_canvas=StubToggle(True)))
    fake_main_window = object()

    monkeypatch.setattr(app_module, '_background_warmup_started', False)
    monkeypatch.setattr(app_module, '_background_warmup_scheduled', False)
    monkeypatch.setattr(app_module, '_background_warmup_pool', None)
    monkeypatch.setattr(app_module, '_apply_app_palette', lambda: captured.setdefault('palette', True))
    monkeypatch.setattr(app_module, '_create_viewer', lambda: fake_viewer)
    monkeypatch.setattr(app_module, '_create_widgets', lambda: fake_widgets)
    fake_gui_state = object()
    monkeypatch.setattr(app_module, 'load_default_gui_state', lambda: fake_gui_state)
    monkeypatch.setattr(app_module, 'apply_gui_state', lambda state, *, widgets: captured.setdefault('applied', (state, widgets)))
    fake_controller = object()

    def fake_initialize_controller(*, viewer, widgets):
        captured['controller_args'] = (viewer, widgets)
        return fake_controller

    def fake_build_main_window_for_app(*, viewer, widgets, controller=None):
        captured['window_args'] = (viewer, widgets, controller)
        return fake_main_window

    monkeypatch.setattr(app_module, 'initialize_controller', fake_initialize_controller)
    monkeypatch.setattr(app_module, 'build_main_window_for_app', fake_build_main_window_for_app)
    monkeypatch.setattr(app_module, '_warmup_launch_input_path', lambda state: captured.setdefault('launch_warmup_state', state))
    monkeypatch.setattr(app_module, '_schedule_background_warmup', lambda: captured.setdefault('warmup_scheduled', True))

    app = app_module.create_app()

    assert captured['palette'] is True
    assert captured['applied'] == (fake_gui_state, fake_widgets)
    assert captured['launch_warmup_state'] is fake_gui_state
    assert captured['controller_args'] == (fake_viewer, fake_widgets)
    assert captured['window_args'] == (fake_viewer, fake_widgets, fake_controller)
    assert captured['warmup_scheduled'] is True
    assert app.viewer is fake_viewer
    assert app.widgets is fake_widgets
    assert app.controller is fake_controller
    assert app.main_window is fake_main_window


def test_build_main_window_for_app_passes_rotate_callbacks_when_controller_is_available() -> None:
    captured: dict[str, object] = {}
    viewer = object()
    widgets = SimpleNamespace(display=SimpleNamespace(gray_18_canvas=StubToggle(True)))
    controller = SimpleNamespace(
        rotate_input_image_counterclockwise=object(),
        rotate_input_image_clockwise=object(),
    )
    fake_controls_panel = object()
    fake_main_window = object()

    def fake_build_controls_panel(viewer, widgets):
        captured['panel_args'] = (viewer, widgets)
        return fake_controls_panel

    def fake_build_main_window(viewer, controls_panel, **kwargs):
        captured['window_args'] = (viewer, controls_panel, kwargs)
        return fake_main_window

    main_window = app_module.build_main_window_for_app(
        viewer=viewer,
        widgets=widgets,
        controller=controller,
        configure_napari_chrome_fn=lambda viewer, *, gray_18_canvas=False: captured.setdefault('chrome', (viewer, gray_18_canvas)),
        build_controls_panel_fn=fake_build_controls_panel,
        build_main_window_fn=fake_build_main_window,
    )

    assert main_window is fake_main_window
    assert captured['chrome'] == (viewer, True)
    assert captured['panel_args'] == (viewer, widgets)
    assert captured['window_args'][0] is viewer
    assert captured['window_args'][1] is fake_controls_panel
    assert captured['window_args'][2] == {
        'on_rotate_ccw': controller.rotate_input_image_counterclockwise,
        'on_rotate_cw': controller.rotate_input_image_clockwise,
    }


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

    for section_name in app_module.GUI_STATE_SECTION_NAMES:
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
            section._skip_auto_preview_leaves = {'auto_preview', 'scan_film'}
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
    widgets.simulation.bottom_scan_film = SimpleNamespace(toggled=FakeSignal())
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
    assert widgets.simulation.bottom_scan_film.toggled.connected == [controller.request_auto_preview]
    assert widgets.simulation.bottom_scan_for_print.toggled.connected == [controller.request_auto_preview]


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
    for section_name in app_module.GUI_STATE_SECTION_NAMES:
        setattr(widgets, section_name, None)
    widgets.scanner = scanner_section
    widgets.camera = camera_section
    widgets.simulation = SimpleNamespace(
        bottom_auto_preview=SimpleNamespace(toggled=FakeSignal()),
        bottom_scan_film=SimpleNamespace(toggled=FakeSignal()),
        bottom_scan_for_print=SimpleNamespace(toggled=FakeSignal()),
        _is_params_group=False,
    )

    app_module.connect_auto_preview_signals(controller, widgets)

    assert scanner_editors['lens_blur'].valueChanged.connected == [controller.request_auto_preview]
    assert scanner_editors['white_correction'].toggled.connected == [controller.request_auto_preview]
    assert scanner_editors['unsharp_mask']._editors[0].valueChanged.connected == [controller.request_auto_preview]
    assert camera_editors['film_format_mm'].valueChanged.connected == [controller.request_auto_preview]


def test_initialize_controller_syncs_connects_and_refreshes() -> None:
    captured: dict[str, object] = {}

    class FakeController:
        def __init__(self, *, viewer, widgets) -> None:
            captured['init'] = (viewer, widgets)

        def sync_display_transform_availability(self, *, report_status: bool) -> None:
            captured['sync'] = report_status

    widgets = object()
    viewer = object()

    controller = app_module.initialize_controller(
        viewer=viewer,
        widgets=widgets,
        controller_cls=FakeController,
        connect_signals_fn=lambda controller, widgets: captured.setdefault('connected', (controller, widgets)),
    )

    assert captured['init'] == (viewer, widgets)
    assert captured['sync'] is False
    assert captured['connected'][1] is widgets
    assert controller is captured['connected'][0]


def test_initialize_controller_shows_startup_placeholder_when_available() -> None:
    captured: dict[str, object] = {}

    class FakeController:
        def __init__(self, *, viewer, widgets) -> None:
            captured['init'] = (viewer, widgets)

        def sync_display_transform_availability(self, *, report_status: bool) -> None:
            captured['sync'] = report_status

        def show_startup_placeholder(self) -> None:
            captured['startup_placeholder'] = True

    widgets = object()
    viewer = object()

    controller = app_module.initialize_controller(
        viewer=viewer,
        widgets=widgets,
        controller_cls=FakeController,
        connect_signals_fn=lambda controller, widgets: captured.setdefault('connected', (controller, widgets)),
    )

    assert captured['init'] == (viewer, widgets)
    assert captured['sync'] is False
    assert captured['startup_placeholder'] is True
    assert captured['connected'][1] is widgets
    assert controller is captured['connected'][0]


def test_build_main_window_for_app_uses_gray_18_canvas_state() -> None:
    captured: dict[str, object] = {}
    viewer = object()
    widgets = SimpleNamespace(display=SimpleNamespace(gray_18_canvas=StubToggle(True)))
    fake_controls_panel = object()
    fake_main_window = object()

    def fake_build_controls_panel(viewer, widgets):
        captured['panel_args'] = (viewer, widgets)
        return fake_controls_panel

    def fake_build_main_window(viewer, controls_panel):
        captured['window_args'] = (viewer, controls_panel)
        return fake_main_window

    main_window = app_module.build_main_window_for_app(
        viewer=viewer,
        widgets=widgets,
        configure_napari_chrome_fn=lambda viewer, *, gray_18_canvas=False: captured.setdefault('chrome', (viewer, gray_18_canvas)),
        build_controls_panel_fn=fake_build_controls_panel,
        build_main_window_fn=fake_build_main_window,
    )

    assert captured['chrome'] == (viewer, True)
    assert captured['panel_args'] == (viewer, widgets)
    assert captured['window_args'][0] is viewer
    assert main_window is fake_main_window