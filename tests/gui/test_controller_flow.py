from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from spektrafilm_gui import controller as controller_module
from spektrafilm_gui.controller import GuiController, PROFILE_SYNC_FIELDS
from spektrafilm_gui.controller_layers import (
    INPUT_COLOR_PREVIEW_LAYER_NAME,
    INPUT_LAYER_NAME,
    INPUT_PREVIEW_LAYER_NAME,
    WHITE_BORDER_LAYER_NAME,
)

from .helpers import FakeLayer, FakeViewer, make_test_controller_gui_state


pytestmark = pytest.mark.integration


def _run_simulation_case(
    monkeypatch,
    *,
    input_layer,
    source_layer_name: str = INPUT_PREVIEW_LAYER_NAME,
    gui_state=None,
    simulated_image=None,
    preview_image=None,
    preview_status: str = 'Display transform: disabled',
) -> dict[str, object]:
    controller = GuiController(viewer=object(), widgets=object())
    gui_state = make_test_controller_gui_state() if gui_state is None else gui_state
    simulated_image = np.full((4, 4, 3), 0.5, dtype=np.float32) if simulated_image is None else simulated_image
    preview_image = np.full((4, 4, 3), 99, dtype=np.uint8) if preview_image is None else preview_image
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        type(controller._layers),
        'selected_input_layer',
        lambda _self, name: input_layer if name == source_layer_name else None,
    )
    monkeypatch.setattr(controller, '_sync_white_border', lambda *, white_padding: captured.setdefault('white_padding', white_padding))
    monkeypatch.setattr(controller_module, 'collect_gui_state', lambda *, widgets: gui_state)
    monkeypatch.setattr(controller_module, 'build_params_from_state', lambda state: object())

    def fake_process_image_with_runtime(image, _params):
        captured['processing_input'] = image.copy()
        return simulated_image.copy()

    def fake_prepare_output_display_image(
        image_data,
        *,
        output_color_space,
        use_display_transform,
        padding_pixels=0.0,
    ):
        captured['display_args'] = {
            'image_data': image_data.copy(),
            'output_color_space': output_color_space,
            'use_display_transform': use_display_transform,
            'padding_pixels': padding_pixels,
        }
        return preview_image.copy(), preview_status

    def fake_set_or_add_output_layer(image, **kwargs):
        captured['output_layer'] = {'image': image.copy(), **kwargs}

    monkeypatch.setattr(controller, '_process_image_with_runtime', fake_process_image_with_runtime)
    monkeypatch.setattr(controller, '_prepare_output_display_image', fake_prepare_output_display_image)
    monkeypatch.setattr(controller, '_set_or_add_output_layer', fake_set_or_add_output_layer)
    monkeypatch.setattr(controller_module, 'set_status', lambda *args, **kwargs: None)

    controller._run_simulation(source_layer_name=source_layer_name)
    return captured


def test_load_input_image_builds_preview_stack_and_homes_view(monkeypatch) -> None:
    viewer = FakeViewer([
        FakeLayer(np.zeros((2, 2, 3), dtype=np.float32), name='older-1'),
        FakeLayer(np.zeros((2, 2, 3), dtype=np.float32), name='older-2'),
    ])
    widgets = SimpleNamespace(filepicker=SimpleNamespace(set_available_layers=lambda *args, **kwargs: None))
    controller = GuiController(viewer=viewer, widgets=widgets)
    gui_state = make_test_controller_gui_state()
    raw_image = np.full((4, 2, 3), 0.25, dtype=np.float32)
    preview_image = np.full((2, 1, 3), 0.5, dtype=np.float32)
    captured: dict[str, object] = {}

    monkeypatch.setattr(controller_module, 'load_image_oiio', lambda path: raw_image)
    monkeypatch.setattr(controller_module, 'collect_gui_state', lambda *, widgets: gui_state)
    monkeypatch.setattr(controller_module, 'build_params_from_state', lambda state: SimpleNamespace(settings=SimpleNamespace(preview_max_size=300)))
    monkeypatch.setattr(controller, '_resize_for_preview', lambda image, *, max_size: preview_image)
    monkeypatch.setattr(controller, '_prepare_input_color_preview_image', lambda *args, **kwargs: np.full((2, 1, 3), 0.75, dtype=np.float32))
    monkeypatch.setattr(controller_module, 'reset_viewer_camera', lambda viewer: captured.setdefault('reset_view', True))

    controller.load_input_image('C:/tmp/example.png')

    assert len(viewer.layers) == 6
    assert [layer.name for layer in viewer.layers[-4:]] == [
        WHITE_BORDER_LAYER_NAME,
        INPUT_LAYER_NAME,
        INPUT_PREVIEW_LAYER_NAME,
        INPUT_COLOR_PREVIEW_LAYER_NAME,
    ]
    np.testing.assert_allclose(viewer.layers[-3].metadata[controller_module.INPUT_RAW_DATA_KEY], raw_image)
    np.testing.assert_allclose(viewer.layers[-2].metadata[controller_module.INPUT_RAW_DATA_KEY], preview_image)
    np.testing.assert_allclose(viewer.layers[-1].metadata[controller_module.INPUT_RAW_DATA_KEY], preview_image)
    assert viewer.layers.selection.active is viewer.layers[-1]
    assert captured['reset_view'] is True


def test_load_raw_image_uses_pipeline_input_settings_and_builds_preview_stack(monkeypatch) -> None:
    viewer = FakeViewer([FakeLayer(np.zeros((2, 2, 3), dtype=np.float32), name='older')])
    widgets = SimpleNamespace(filepicker=SimpleNamespace(set_available_layers=lambda *args, **kwargs: None))
    controller = GuiController(viewer=viewer, widgets=widgets)
    gui_state = make_test_controller_gui_state()
    gui_state.input_image.input_color_space = 'Display P3'
    gui_state.input_image.apply_cctf_decoding = True
    gui_state.load_raw.white_balance = 'custom'
    gui_state.load_raw.temperature = 3200.0
    gui_state.load_raw.tint = 0.85
    raw_image = np.full((4, 2, 3), 0.4, dtype=np.float32)
    preview_image = np.full((2, 1, 3), 0.6, dtype=np.float32)
    captured: dict[str, object] = {}

    def fake_load_and_process_raw_file(path, **kwargs):
        captured['path'] = path
        captured['kwargs'] = kwargs
        return raw_image

    monkeypatch.setattr(controller_module, 'load_and_process_raw_file', fake_load_and_process_raw_file)
    monkeypatch.setattr(controller_module, 'collect_gui_state', lambda *, widgets: gui_state)
    monkeypatch.setattr(controller_module, 'build_params_from_state', lambda state: SimpleNamespace(settings=SimpleNamespace(preview_max_size=300)))
    monkeypatch.setattr(controller, '_resize_for_preview', lambda image, *, max_size: preview_image)
    monkeypatch.setattr(controller, '_prepare_input_color_preview_image', lambda *args, **kwargs: np.full((2, 1, 3), 0.8, dtype=np.float32))
    monkeypatch.setattr(controller_module, 'reset_viewer_camera', lambda viewer: captured.setdefault('reset_view', True))
    monkeypatch.setattr(
        controller_module,
        'set_status',
        lambda viewer, message, timeout_ms=5000: captured.setdefault('status', (message, timeout_ms)),
    )

    controller.load_raw_image('C:/tmp/example.nef')

    assert captured['status'] == ('Loading raw...', 0)
    assert captured['path'] == 'C:/tmp/example.nef'
    assert captured['kwargs'] == {
        'white_balance': 'custom',
        'temperature': 3200.0,
        'tint': 0.85,
        'lens_correction': False,
        'output_colorspace': 'Display P3',
        'output_cctf_encoding': True,
        'lens_info_out': {},
    }
    assert len(viewer.layers) == 5
    assert [layer.name for layer in viewer.layers[-4:]] == [
        WHITE_BORDER_LAYER_NAME,
        INPUT_LAYER_NAME,
        INPUT_PREVIEW_LAYER_NAME,
        INPUT_COLOR_PREVIEW_LAYER_NAME,
    ]
    assert captured['reset_view'] is True


def test_load_raw_image_reports_invalid_custom_white_balance_without_mutating_layers(monkeypatch) -> None:
    viewer = FakeViewer([
        FakeLayer(np.zeros((2, 2, 3), dtype=np.float32), name='older'),
    ])
    widgets = SimpleNamespace(filepicker=SimpleNamespace(set_available_layers=lambda *args, **kwargs: None))
    controller = GuiController(viewer=viewer, widgets=widgets)
    gui_state = make_test_controller_gui_state()
    gui_state.load_raw.white_balance = 'custom'
    gui_state.load_raw.temperature = 3200.0
    gui_state.load_raw.tint = 0.85
    statuses: list[tuple[str, int]] = []
    captured_dialog: dict[str, object] = {}

    def fake_load_and_process_raw_file(path, **kwargs):
        raise ValueError('RAW file does not expose a usable camera XYZ matrix for custom white balance.')

    def fake_critical(parent, title, message):
        captured_dialog['parent'] = parent
        captured_dialog['title'] = title
        captured_dialog['message'] = message

    monkeypatch.setattr(controller_module, 'load_and_process_raw_file', fake_load_and_process_raw_file)
    monkeypatch.setattr(controller_module, 'collect_gui_state', lambda *, widgets: gui_state)
    monkeypatch.setattr(controller_module, 'dialog_parent', lambda viewer: 'dialog-parent')
    monkeypatch.setattr(controller_module.QMessageBox, 'critical', fake_critical)
    monkeypatch.setattr(
        controller_module,
        'set_status',
        lambda viewer, message, timeout_ms=5000: statuses.append((message, timeout_ms)),
    )

    controller.load_raw_image('C:/tmp/example.nef')

    assert statuses == [('Loading raw...', 0), ('Load raw failed', 5000)]
    assert captured_dialog == {
        'parent': 'dialog-parent',
        'title': 'Load raw',
        'message': (
            'Failed to load RAW image.\n\n'
            'RAW file does not expose a usable camera XYZ matrix for custom white balance.'
        ),
    }
    assert len(viewer.layers) == 1


def test_load_raw_image_reports_when_lens_correction_is_not_applied(monkeypatch) -> None:
    viewer = FakeViewer([FakeLayer(np.zeros((2, 2, 3), dtype=np.float32), name='older')])
    widgets = SimpleNamespace(filepicker=SimpleNamespace(set_available_layers=lambda *args, **kwargs: None))
    controller = GuiController(viewer=viewer, widgets=widgets)
    gui_state = make_test_controller_gui_state()
    gui_state.load_raw.lens_correction = True
    raw_image = np.full((2, 2, 3), 0.4, dtype=np.float32)
    statuses: list[tuple[str, int]] = []
    captured: dict[str, object] = {}

    def fake_load_and_process_raw_file(path, **kwargs):
        captured['path'] = path
        captured['kwargs'] = kwargs
        return raw_image

    monkeypatch.setattr(controller_module, 'load_and_process_raw_file', fake_load_and_process_raw_file)
    monkeypatch.setattr(controller_module, 'collect_gui_state', lambda *, widgets: gui_state)
    monkeypatch.setattr(controller_module, 'build_params_from_state', lambda state: SimpleNamespace(settings=SimpleNamespace(preview_max_size=300)))
    monkeypatch.setattr(controller, '_resize_for_preview', lambda image, *, max_size: image)
    monkeypatch.setattr(controller, '_prepare_input_color_preview_image', lambda image, **kwargs: image)
    monkeypatch.setattr(controller_module, 'reset_viewer_camera', lambda viewer: None)
    monkeypatch.setattr(
        controller_module,
        'set_status',
        lambda viewer, message, timeout_ms=5000: statuses.append((message, timeout_ms)),
    )

    controller.load_raw_image('C:/tmp/example.nef')

    assert captured['path'] == 'C:/tmp/example.nef'
    assert captured['kwargs']['lens_correction'] is True
    assert captured['kwargs']['lens_info_out'] == {}
    assert statuses == [
        ('Loading raw...', 0),
        ('Loaded raw, lens correction not applied', 5000),
    ]


def test_select_input_layer_sets_active_layer_without_reordering() -> None:
    selected_layer = FakeLayer(np.zeros((2, 2, 3), dtype=np.float32), name=INPUT_PREVIEW_LAYER_NAME)
    viewer = FakeViewer([
        FakeLayer(np.zeros((2, 2, 3), dtype=np.float32), name=WHITE_BORDER_LAYER_NAME),
        FakeLayer(np.zeros((2, 2, 3), dtype=np.float32), name=INPUT_LAYER_NAME),
        selected_layer,
        FakeLayer(np.zeros((2, 2, 3), dtype=np.float32), name=INPUT_COLOR_PREVIEW_LAYER_NAME),
    ])
    controller = GuiController(viewer=viewer, widgets=object())

    controller.select_input_layer(INPUT_PREVIEW_LAYER_NAME)

    assert viewer.layers[2] is selected_layer
    assert viewer.layers.selection.active is selected_layer
    assert all(layer.visible is True for layer in viewer.layers)


def test_apply_profile_defaults_routes_through_selection_digest(monkeypatch) -> None:
    controller = GuiController(viewer=object(), widgets=object())
    gui_state = make_test_controller_gui_state()
    captured: dict[str, object] = {}
    built_params = object()
    digested_params = object()
    synced_state = object()

    def fake_build_params_from_state(state):
        captured['build_state'] = state
        return built_params

    def fake_digest_after_selection(params):
        captured['digested_input'] = params
        return digested_params

    def fake_gui_state_from_params(params, *, film_stock, print_paper):
        captured['synced_args'] = (params, film_stock, print_paper)
        return synced_state

    monkeypatch.setattr(controller_module, 'collect_gui_state', lambda *, widgets: gui_state)
    monkeypatch.setattr(controller_module, 'build_params_from_state', fake_build_params_from_state)
    monkeypatch.setattr(controller_module, 'digest_after_selection', fake_digest_after_selection)
    monkeypatch.setattr(controller_module, 'gui_state_from_params', fake_gui_state_from_params)
    monkeypatch.setattr(controller, '_apply_profile_sync_state', lambda state: captured.setdefault('applied_state', state))

    controller.apply_profile_defaults('ignored-by-handler')

    assert captured['digested_input'] is built_params
    assert captured['synced_args'] == (digested_params, gui_state.simulation.film_stock, gui_state.simulation.print_paper)
    assert captured['applied_state'] is synced_state


def test_apply_profile_sync_state_updates_runtime_owned_widget_fields() -> None:
    original_fields = dict(PROFILE_SYNC_FIELDS)
    try:
        controller_module.PROFILE_SYNC_FIELDS = {
            'couplers': ('dir_couplers_ratio',),
            'simulation': ('scan_film',),
        }
        captured: dict[str, object] = {}
        controller = GuiController(
            viewer=object(),
            widgets=SimpleNamespace(
                couplers=SimpleNamespace(dir_couplers_ratio=SimpleNamespace(value=(0.0, 0.0, 0.0))),
                simulation=SimpleNamespace(set_scan_film_value=lambda value: captured.setdefault('scan_film', []).append(value)),
            ),
        )
        synced_state = SimpleNamespace(
            couplers=SimpleNamespace(dir_couplers_ratio=(0.35, 0.2275, 0.1225)),
            simulation=SimpleNamespace(scan_film=True),
        )

        controller._apply_profile_sync_state(synced_state)

        assert controller._widgets.couplers.dir_couplers_ratio.value == (0.35, 0.2275, 0.1225)
        assert captured['scan_film'] == [True]
    finally:
        controller_module.PROFILE_SYNC_FIELDS = original_fields


def test_run_simulation_uses_processing_input_metadata(monkeypatch) -> None:
    raw_image = np.full((2, 2, 3), 0.25, dtype=np.float32)
    display_image = np.full((2, 2, 3), 0.5, dtype=np.float32)
    input_layer = FakeLayer(
        display_image,
        metadata={controller_module.INPUT_RAW_DATA_KEY: raw_image},
        name=INPUT_PREVIEW_LAYER_NAME,
    )
    captured = _run_simulation_case(
        monkeypatch,
        input_layer=input_layer,
        simulated_image=np.full((2, 2, 3), 0.5, dtype=np.float32),
        preview_image=np.full((2, 2, 3), 99, dtype=np.uint8),
    )

    np.testing.assert_allclose(captured['processing_input'], raw_image)
    assert captured['white_padding'] == make_test_controller_gui_state().display.white_padding


def test_run_simulation_passes_display_transform_settings(monkeypatch) -> None:
    input_layer = SimpleNamespace(data=np.full((2, 2, 3), 0.25, dtype=np.float32), metadata={})
    gui_state = make_test_controller_gui_state()
    gui_state.display.use_display_transform = True
    gui_state.display.white_padding = 0.5
    captured = _run_simulation_case(
        monkeypatch,
        input_layer=input_layer,
        gui_state=gui_state,
        simulated_image=np.full((4, 4, 3), 0.5, dtype=np.float32),
        preview_image=np.full((6, 6, 3), 99, dtype=np.uint8),
        preview_status='Display transform: active',
    )

    np.testing.assert_allclose(captured['display_args']['image_data'], np.full((4, 4, 3), 0.5, dtype=np.float32))
    assert captured['display_args']['output_color_space'] == gui_state.simulation.output_color_space
    assert captured['display_args']['use_display_transform'] is True
    assert captured['display_args']['padding_pixels'] == 0.0
    np.testing.assert_array_equal(captured['output_layer']['image'], np.full((6, 6, 3), 99, dtype=np.uint8))
    np.testing.assert_allclose(captured['output_layer']['float_image'], np.full((4, 4, 3), 0.5, dtype=np.float32))


@pytest.mark.parametrize(
    ('method_name', 'expected_call'),
    [
        ('run_preview', {'source_layer_name': INPUT_PREVIEW_LAYER_NAME, 'mode_label': 'Preview'}),
        ('run_scan', {'source_layer_name': INPUT_LAYER_NAME, 'mode_label': 'Scan'}),
    ],
    ids=['preview', 'scan'],
)
def test_run_preview_and_scan_start_async_simulation(monkeypatch, method_name: str, expected_call: dict[str, object]) -> None:
    controller = GuiController(viewer=object(), widgets=object())
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        controller,
        '_start_simulation',
        lambda *, source_layer_name, mode_label: captured.setdefault(
            'call',
            {'source_layer_name': source_layer_name, 'mode_label': mode_label},
        ),
    )

    getattr(controller, method_name)()

    assert captured['call'] == expected_call


def test_start_simulation_reports_persistent_computing_status(monkeypatch) -> None:
    input_layer = SimpleNamespace(data=np.full((2, 2, 3), 0.25, dtype=np.float32), metadata={})
    simulation_section = SimpleNamespace(preview_button=None, scan_button=None, save_button=None)
    widgets = SimpleNamespace(simulation=simulation_section)
    controller = GuiController(viewer=object(), widgets=widgets)
    gui_state = make_test_controller_gui_state()
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        type(controller._layers),
        'selected_input_layer',
        lambda _self, name: input_layer if name == INPUT_PREVIEW_LAYER_NAME else None,
    )
    monkeypatch.setattr(controller, '_sync_white_border', lambda *, white_padding: captured.setdefault('white_padding', white_padding))
    monkeypatch.setattr(controller_module, 'collect_gui_state', lambda *, widgets: gui_state)
    monkeypatch.setattr(controller_module, 'build_params_from_state', lambda state: object())
    monkeypatch.setattr(controller_module, 'set_status', lambda viewer, message, timeout_ms=5000: captured.setdefault('status', (message, timeout_ms)))
    monkeypatch.setattr(controller._thread_pool, 'start', lambda worker: captured.setdefault('worker', worker))

    controller._start_simulation(source_layer_name=INPUT_PREVIEW_LAYER_NAME, mode_label='Preview')

    assert captured['status'] == ('Computing preview...', 0)
    assert captured['white_padding'] == gui_state.display.white_padding
    assert controller._active_simulation_label == 'Preview'


@pytest.mark.parametrize(
    ('source_layer_name', 'mode_label', 'expected_grain_active', 'expected_halation_active'),
    [
        (INPUT_PREVIEW_LAYER_NAME, 'Preview', False, False),
        (INPUT_LAYER_NAME, 'Scan', True, True),
    ],
    ids=['preview-disables-grain-and-halation', 'scan-preserves-grain-and-halation'],
)
def test_start_simulation_tunes_preview_effects_only(
    monkeypatch,
    source_layer_name: str,
    mode_label: str,
    expected_grain_active: bool,
    expected_halation_active: bool,
) -> None:
    input_layer = SimpleNamespace(data=np.full((2, 2, 3), 0.25, dtype=np.float32), metadata={})
    simulation_section = SimpleNamespace(preview_button=None, scan_button=None, save_button=None)
    widgets = SimpleNamespace(simulation=simulation_section)
    controller = GuiController(viewer=object(), widgets=widgets)
    gui_state = make_test_controller_gui_state()
    params = SimpleNamespace(
        film_render=SimpleNamespace(
            grain=SimpleNamespace(active=True),
            halation=SimpleNamespace(active=True),
        )
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        type(controller._layers),
        'selected_input_layer',
        lambda _self, name: input_layer if name == source_layer_name else None,
    )
    monkeypatch.setattr(controller, '_sync_white_border', lambda *, white_padding: None)
    monkeypatch.setattr(controller_module, 'collect_gui_state', lambda *, widgets: gui_state)
    monkeypatch.setattr(controller_module, 'build_params_from_state', lambda state: params)
    monkeypatch.setattr(controller_module, 'set_status', lambda *args, **kwargs: None)
    monkeypatch.setattr(controller._thread_pool, 'start', lambda worker: captured.setdefault('request', worker._request))

    controller._start_simulation(source_layer_name=source_layer_name, mode_label=mode_label)

    assert params.film_render.grain.active is expected_grain_active
    assert params.film_render.halation.active is expected_halation_active
    assert captured['request'].params is params


def test_execute_simulation_request_routes_through_runtime_simulator_path(monkeypatch) -> None:
    controller = GuiController(viewer=object(), widgets=object())
    request = controller_module.SimulationRequest(
        mode_label='Preview',
        image=np.full((2, 2, 3), 0.25, dtype=np.float32),
        params=object(),
        output_color_space='sRGB',
        use_display_transform=False,
    )
    captured: dict[str, object] = {}

    def fake_process_image_with_runtime(image, params):
        captured['runtime_call'] = (image.copy(), params)
        return np.full((2, 2, 3), 0.5, dtype=np.float32)

    monkeypatch.setattr(
        controller,
        '_process_image_with_runtime',
        fake_process_image_with_runtime,
    )
    monkeypatch.setattr(
        controller,
        '_prepare_output_display_image',
        lambda image, **kwargs: (np.uint8(np.clip(image, 0.0, 1.0) * 255), 'Display transform: disabled'),
    )

    result = controller._execute_simulation_request(request)

    np.testing.assert_allclose(captured['runtime_call'][0], request.image)
    assert captured['runtime_call'][1] is request.params
    assert result.mode_label == 'Preview'


def test_on_simulation_finished_reports_completed_status(monkeypatch) -> None:
    controller = GuiController(viewer=object(), widgets=SimpleNamespace(simulation=SimpleNamespace(preview_button=None, scan_button=None, save_button=None)))
    controller._active_simulation_label = 'Preview'
    captured: dict[str, object] = {}

    monkeypatch.setattr(controller, '_set_or_add_output_layer', lambda image, **kwargs: captured.setdefault('output', (image, kwargs)))
    monkeypatch.setattr(controller_module, 'set_status', lambda viewer, message, timeout_ms=5000: captured.setdefault('status', (message, timeout_ms)))

    controller._on_simulation_finished(
        controller_module.SimulationResult(
            mode_label='Preview',
            display_image=np.full((2, 2, 3), 9, dtype=np.uint8),
            float_image=np.full((2, 2, 3), 0.5, dtype=np.float32),
            output_color_space='sRGB',
            use_display_transform=False,
            status_message='Display transform: disabled',
        )
    )

    assert captured['status'] == ('Preview completed. Display transform: disabled', 5000)
