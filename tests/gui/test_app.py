from __future__ import annotations

from dataclasses import fields
from types import SimpleNamespace

import numpy as np
from qtpy import QtGui

from spektrafilm_gui import app as app_module

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
        input_image=SimpleNamespace(input_color_space='ACES2065-1', apply_cctf_decoding=False),
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

    app_module._warmup_launch_input_path(SimpleNamespace(input_image=SimpleNamespace(input_color_space='sRGB', apply_cctf_decoding=False)))


def test_warmup_full_gui_runs_preview_pipeline(monkeypatch) -> None:
    captured: dict[str, object] = {}
    fake_state = SimpleNamespace(
        input_image=SimpleNamespace(input_color_space='ACES2065-1', apply_cctf_decoding=False),
        simulation=SimpleNamespace(output_color_space='Display P3'),
    )
    fake_colour_module = object()
    fake_pil_image_module = object()
    fake_imagecms_module = object()
    fake_io_module = object()
    fake_preview_module = object()
    fake_raw_module = object()

    def fake_prepare_input_color_preview_image(image, **kwargs):
        captured['input_preview'] = (np.asarray(image), kwargs)
        return np.asarray(image, dtype=np.float32)

    def fake_prepare_output_display_image(image, **kwargs):
        captured['output_preview'] = (np.asarray(image), kwargs)
        return np.asarray(image, dtype=np.uint8), 'ok'

    def fake_build_params_from_state(state):
        captured['params_state'] = state
        return 'raw-params'

    class FakeSimulator:
        def __init__(self, params) -> None:
            captured['simulator_params'] = params

        def process(self, image):
            captured['process_image'] = np.asarray(image)
            return np.full_like(image, 0.5, dtype=np.float32)

    def fake_digest_params(params):
        captured['digested_params'] = params
        return 'digested-params'

    fake_runtime_api = SimpleNamespace(
        digest_params=fake_digest_params,
        Simulator=FakeSimulator,
    )
    fake_controller_runtime = SimpleNamespace(
        prepare_input_color_preview_image=fake_prepare_input_color_preview_image,
        prepare_output_display_image=fake_prepare_output_display_image,
    )
    fake_params_mapper = SimpleNamespace(build_params_from_state=fake_build_params_from_state)
    module_map = {
        'colour': fake_colour_module,
        'PIL.Image': fake_pil_image_module,
        'PIL.ImageCms': fake_imagecms_module,
        'spektrafilm_gui.controller_runtime': fake_controller_runtime,
        'spektrafilm_gui.params_mapper': fake_params_mapper,
        'spektrafilm.runtime.api': fake_runtime_api,
        'spektrafilm.utils.io': fake_io_module,
        'spektrafilm.utils.preview': fake_preview_module,
        'spektrafilm.utils.raw_file_processor': fake_raw_module,
    }

    monkeypatch.setattr(app_module, 'load_default_gui_state', lambda: fake_state)
    monkeypatch.setattr(app_module, 'warmup', lambda: captured.setdefault('numba_warmup', True))
    monkeypatch.setattr(app_module, 'import_module', lambda name: module_map[name])

    app_module._warmup_full_gui()

    assert captured['numba_warmup'] is True
    assert captured['params_state'] is fake_state
    assert captured['digested_params'] == 'raw-params'
    assert captured['simulator_params'] == 'digested-params'
    process_image = captured['process_image']
    assert process_image.shape == app_module.WARMUP_IMAGE_SHAPE
    assert process_image.dtype == np.float64
    input_preview_image, input_preview_kwargs = captured['input_preview']
    assert input_preview_image.shape == app_module.WARMUP_IMAGE_SHAPE
    assert input_preview_kwargs['input_color_space'] == 'ACES2065-1'
    assert input_preview_kwargs['apply_cctf_decoding'] is False
    assert input_preview_kwargs['colour_module'] is fake_colour_module
    output_preview_image, output_preview_kwargs = captured['output_preview']
    assert output_preview_image.shape == app_module.WARMUP_IMAGE_SHAPE
    assert output_preview_kwargs['output_color_space'] == 'Display P3'
    assert output_preview_kwargs['use_display_transform'] is True
    assert output_preview_kwargs['imagecms_module'] is fake_imagecms_module
    assert output_preview_kwargs['colour_module'] is fake_colour_module
    assert output_preview_kwargs['pil_image_module'] is fake_pil_image_module


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
        state_section = getattr(gui_state, section_name)
        section = SimpleNamespace(
            _state_cls=type(state_section),
            _hidden_fields={
                'upscale_factor',
                'crop',
                'crop_center',
                'crop_size',
                'spectral_upsampling_method',
                'filter_uv',
                'filter_ir',
                'film_gamma_factor',
                'print_gamma_factor',
                'film_format_mm',
                'camera_lens_blur_um',
                'camera_diffusion_filter_active',
                'camera_diffusion_filter_family',
                'camera_diffusion_filter_strength',
                'camera_diffusion_filter_spatial_scale',
                'camera_diffusion_filter_halo_warmth',
                'camera_diffusion_filter_core_intensity',
                'camera_diffusion_filter_core_size',
                'camera_diffusion_filter_halo_intensity',
                'camera_diffusion_filter_halo_size',
                'camera_diffusion_filter_bloom_intensity',
                'camera_diffusion_filter_bloom_size',
                'exposure_compensation_ev',
                'auto_exposure',
                'auto_exposure_method',
                'print_exposure',
                'print_exposure_compensation',
                'print_y_filter_shift',
                'print_m_filter_shift',
                'diffusion_filter_active',
                'diffusion_filter_family',
                'diffusion_filter_strength',
                'diffusion_filter_spatial_scale',
                'diffusion_filter_halo_warmth',
                'diffusion_filter_core_intensity',
                'diffusion_filter_core_size',
                'diffusion_filter_halo_intensity',
                'diffusion_filter_halo_size',
                'diffusion_filter_bloom_intensity',
                'diffusion_filter_bloom_size',
                'print_illuminant',
                'scan_lens_blur',
                'scan_white_correction',
                'scan_white_level',
                'scan_black_correction',
                'scan_black_level',
                'scan_unsharp_mask',
                'auto_preview',
                'scan_film',
                'output_color_space',
                'saving_color_space',
                'saving_cctf_encoding',
            },
        )
        for field_info in fields(type(state_section)):
            setattr(section, field_info.name, _make_auto_preview_editor(getattr(state_section, field_info.name)))
        setattr(widgets, section_name, section)

    widgets.simulation.bottom_auto_preview = SimpleNamespace(toggled=FakeSignal())
    widgets.simulation.bottom_scan_film = SimpleNamespace(toggled=FakeSignal())
    widgets.simulation.bottom_scan_for_print = SimpleNamespace(toggled=FakeSignal())

    app_module.connect_auto_preview_signals(controller, widgets)

    assert widgets.input_image.upscale_factor.valueChanged.connected == [controller.request_auto_preview]
    assert widgets.input_image.crop_size._editors[0].valueChanged.connected == [controller.request_auto_preview]
    assert widgets.special.film_gamma_factor.valueChanged.connected == [controller.request_auto_preview]
    assert widgets.simulation.print_y_filter_shift.valueChanged.connected == [controller.request_auto_preview]
    assert widgets.simulation.camera_diffusion_filter_strength.valueChanged.connected == [controller.request_auto_preview]
    assert widgets.simulation.diffusion_filter_strength.valueChanged.connected == [controller.request_auto_preview]
    assert widgets.simulation.exposure_compensation_ev.valueChanged.connected == [controller.request_auto_preview]
    assert widgets.simulation.scan_lens_blur.valueChanged.connected == [controller.request_auto_preview]
    assert widgets.simulation.scan_white_correction.toggled.connected == [controller.request_auto_preview]
    assert widgets.simulation.scan_white_level.valueChanged.connected == [controller.request_auto_preview]
    assert widgets.simulation.scan_unsharp_mask._editors[0].valueChanged.connected == [controller.request_auto_preview]
    assert widgets.display.output_interpolation.currentTextChanged.connected == []
    assert widgets.display.preview_max_size.valueChanged.connected == []
    assert widgets.simulation.output_color_space.currentTextChanged.connected == [controller.request_auto_preview]
    assert widgets.simulation.bottom_auto_preview.toggled.connected == [controller.request_auto_preview]
    assert widgets.simulation.bottom_scan_film.toggled.connected == [controller.request_auto_preview]
    assert widgets.simulation.bottom_scan_for_print.toggled.connected == [controller.request_auto_preview]


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