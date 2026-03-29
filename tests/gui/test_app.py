from __future__ import annotations

from types import SimpleNamespace

from qtpy import QtGui

from spektrafilm_gui import app as app_module

from .helpers import StubToggle


class FakeSignal:
    def __init__(self) -> None:
        self.connected: list[object] = []

    def connect(self, callback) -> None:
        self.connected.append(callback)


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

    def fake_single_shot(delay_ms: int, callback) -> None:
        captured.append((delay_ms, callback))

    app_module._schedule_background_warmup(single_shot_fn=fake_single_shot)
    app_module._schedule_background_warmup(single_shot_fn=fake_single_shot)

    assert captured == [(0, app_module._start_background_warmup)]


def test_start_background_warmup_starts_task_once(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(app_module, '_background_warmup_started', False)
    monkeypatch.setattr(app_module, '_background_warmup_scheduled', True)

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


def test_create_app_syncs_display_transform_availability_before_connecting(monkeypatch) -> None:
    captured: dict[str, object] = {}

    fake_viewer = object()
    fake_widgets = SimpleNamespace(display=SimpleNamespace(use_display_transform=object(), gray_18_canvas=StubToggle(True)))
    fake_main_window = object()

    monkeypatch.setattr(app_module, '_background_warmup_started', False)
    monkeypatch.setattr(app_module, '_background_warmup_scheduled', False)
    monkeypatch.setattr(app_module, '_apply_app_palette', lambda: captured.setdefault('palette', True))
    monkeypatch.setattr(app_module, '_create_viewer', lambda: fake_viewer)
    monkeypatch.setattr(app_module, '_create_widgets', lambda: fake_widgets)
    monkeypatch.setattr(app_module, 'load_default_gui_state', lambda: object())
    monkeypatch.setattr(app_module, 'apply_gui_state', lambda state, *, widgets: captured.setdefault('applied', (state, widgets)))
    fake_controller = object()

    def fake_initialize_controller(*, viewer, widgets):
        captured['controller_args'] = (viewer, widgets)
        return fake_controller

    def fake_build_main_window_for_app(*, viewer, widgets):
        captured['window_args'] = (viewer, widgets)
        return fake_main_window

    monkeypatch.setattr(app_module, 'initialize_controller', fake_initialize_controller)
    monkeypatch.setattr(app_module, 'build_main_window_for_app', fake_build_main_window_for_app)
    monkeypatch.setattr(app_module, '_schedule_background_warmup', lambda: captured.setdefault('warmup_scheduled', True))

    app = app_module.create_app()

    assert captured['palette'] is True
    assert captured['controller_args'] == (fake_viewer, fake_widgets)
    assert captured['window_args'] == (fake_viewer, fake_widgets)
    assert captured['warmup_scheduled'] is True
    assert app.viewer is fake_viewer
    assert app.widgets is fake_widgets
    assert app.controller is fake_controller
    assert app.main_window is fake_main_window


def test_connect_controller_signals_wires_all_widget_events() -> None:
    controller = SimpleNamespace(
        load_input_image=object(),
        load_raw_image=object(),
        select_input_layer=object(),
        save_current_as_default=object(),
        save_current_state_to_file=object(),
        load_state_from_file=object(),
        restore_factory_default=object(),
        run_preview=object(),
        run_scan=object(),
        save_output_layer=object(),
        report_display_transform_status=object(),
        set_gray_18_canvas_enabled=object(),
    )
    widgets = SimpleNamespace(
        filepicker=SimpleNamespace(load_requested=FakeSignal(), input_layer=SimpleNamespace(currentTextChanged=FakeSignal())),
        load_raw=SimpleNamespace(load_requested=FakeSignal()),
        gui_config=SimpleNamespace(
            save_current_as_default_requested=FakeSignal(),
            save_current_to_file_requested=FakeSignal(),
            load_from_file_requested=FakeSignal(),
            restore_factory_default_requested=FakeSignal(),
        ),
        simulation=SimpleNamespace(preview_requested=FakeSignal(), scan_requested=FakeSignal(), save_requested=FakeSignal()),
        display=SimpleNamespace(use_display_transform=SimpleNamespace(toggled=FakeSignal()), gray_18_canvas=SimpleNamespace(toggled=FakeSignal())),
    )

    app_module.connect_controller_signals(controller, widgets)

    assert widgets.filepicker.load_requested.connected == [controller.load_input_image]
    assert widgets.load_raw.load_requested.connected == [controller.load_raw_image]
    assert widgets.filepicker.input_layer.currentTextChanged.connected == [controller.select_input_layer]
    assert widgets.gui_config.save_current_as_default_requested.connected == [controller.save_current_as_default]
    assert widgets.gui_config.save_current_to_file_requested.connected == [controller.save_current_state_to_file]
    assert widgets.gui_config.load_from_file_requested.connected == [controller.load_state_from_file]
    assert widgets.gui_config.restore_factory_default_requested.connected == [controller.restore_factory_default]
    assert widgets.simulation.preview_requested.connected == [controller.run_preview]
    assert widgets.simulation.scan_requested.connected == [controller.run_scan]
    assert widgets.simulation.save_requested.connected == [controller.save_output_layer]
    assert widgets.display.use_display_transform.toggled.connected == [controller.report_display_transform_status]
    assert widgets.display.gray_18_canvas.toggled.connected == [controller.set_gray_18_canvas_enabled]


def test_initialize_controller_syncs_connects_and_refreshes() -> None:
    captured: dict[str, object] = {}

    class FakeController:
        def __init__(self, *, viewer, widgets) -> None:
            captured['init'] = (viewer, widgets)

        def sync_display_transform_availability(self, *, report_status: bool) -> None:
            captured['sync'] = report_status

        def refresh_input_layers(self) -> None:
            captured['refresh'] = True

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
    assert captured['refresh'] is True
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