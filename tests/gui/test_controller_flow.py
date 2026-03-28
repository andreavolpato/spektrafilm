from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from spektrafilm_gui import controller as controller_module
from spektrafilm_gui.controller import GuiController, INPUT_PADDING_PIXELS_KEY, INPUT_RAW_DATA_KEY

from .helpers import FakeLayer, FakeViewer, make_test_controller_gui_state


pytestmark = pytest.mark.integration


def _run_simulation_case(
    monkeypatch,
    *,
    input_layer,
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

    monkeypatch.setattr(controller, '_selected_input_layer', lambda: input_layer)
    monkeypatch.setattr(controller_module, 'collect_gui_state', lambda *, widgets: gui_state)
    monkeypatch.setattr(controller_module, 'build_params_from_state', lambda state: object())

    def fake_simulate(image, _params):
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

    monkeypatch.setattr(controller_module, 'simulate', fake_simulate)
    monkeypatch.setattr(controller, '_prepare_output_display_image', fake_prepare_output_display_image)
    monkeypatch.setattr(controller, '_set_or_add_output_layer', fake_set_or_add_output_layer)
    monkeypatch.setattr(controller_module, 'set_status', lambda *args, **kwargs: None)

    controller._run_simulation(compute_full_image=False)
    return captured


def test_load_input_image_pads_display_but_preserves_raw_metadata(monkeypatch) -> None:
    viewer = FakeViewer([
        FakeLayer(np.zeros((2, 2, 3), dtype=np.float32), name='older-1'),
        FakeLayer(np.zeros((2, 2, 3), dtype=np.float32), name='older-2'),
    ])
    widgets = SimpleNamespace(filepicker=SimpleNamespace(set_available_layers=lambda *args, **kwargs: None))
    controller = GuiController(viewer=viewer, widgets=widgets)
    gui_state = make_test_controller_gui_state()
    gui_state.display.white_padding = 0.5
    raw_image = np.full((2, 2, 3), 0.25, dtype=np.float32)

    monkeypatch.setattr(controller_module, 'load_image_oiio', lambda path: raw_image)
    monkeypatch.setattr(controller_module, 'collect_gui_state', lambda *, widgets: gui_state)

    controller.load_input_image('C:/tmp/example.png')

    assert len(viewer.layers) == 3
    layer = viewer.layers[-1]
    assert layer.name == 'example'
    assert layer.data.shape == (4, 4, 3)
    np.testing.assert_allclose(layer.data[1:3, 1:3], raw_image)
    np.testing.assert_allclose(layer.data[[0, -1], :, :], 1.0)
    np.testing.assert_allclose(layer.metadata[INPUT_RAW_DATA_KEY], raw_image)
    assert layer.metadata[INPUT_PADDING_PIXELS_KEY] == 1.0
    assert layer.visible is True
    assert all(other_layer.visible is False for other_layer in viewer.layers[:-1])
    assert viewer.reset_view_calls == 0


def test_select_input_layer_hides_other_layers_and_moves_target_to_top() -> None:
    selected_layer = FakeLayer(np.zeros((2, 2, 3), dtype=np.float32), name='selected')
    viewer = FakeViewer([
        FakeLayer(np.zeros((2, 2, 3), dtype=np.float32), name='older-1'),
        selected_layer,
        FakeLayer(np.zeros((2, 2, 3), dtype=np.float32), name='older-2'),
    ])
    controller = GuiController(viewer=viewer, widgets=object())
    controller._available_input_layers = lambda: list(viewer.layers)  # type: ignore[method-assign]

    controller.select_input_layer('selected')

    assert viewer.layers[-1] is selected_layer
    assert selected_layer.visible is True
    assert all(layer.visible is False for layer in viewer.layers[:-1])


def test_run_simulation_uses_unpadded_raw_input_metadata(monkeypatch) -> None:
    raw_image = np.full((2, 2, 3), 0.25, dtype=np.float32)
    padded_image = np.pad(raw_image, ((1, 1), (1, 1), (0, 0)), mode='constant', constant_values=1.0)
    input_layer = FakeLayer(
        padded_image,
        metadata={
            INPUT_RAW_DATA_KEY: raw_image,
            INPUT_PADDING_PIXELS_KEY: 1.0,
        },
        name='input',
    )
    captured = _run_simulation_case(
        monkeypatch,
        input_layer=input_layer,
        simulated_image=np.full((2, 2, 3), 0.5, dtype=np.float32),
        preview_image=np.full((2, 2, 3), 99, dtype=np.uint8),
    )

    np.testing.assert_allclose(captured['processing_input'], raw_image)


def test_run_simulation_uses_display_transform_preview_when_enabled(monkeypatch) -> None:
    input_layer = SimpleNamespace(data=np.full((2, 2, 3), 0.25, dtype=np.float32))
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
    assert captured['display_args']['padding_pixels'] == 2.0
    np.testing.assert_array_equal(captured['output_layer']['image'], np.full((6, 6, 3), 99, dtype=np.uint8))
    np.testing.assert_allclose(captured['output_layer']['float_image'], np.full((4, 4, 3), 0.5, dtype=np.float32))


def test_run_simulation_applies_white_padding_only_to_preview(monkeypatch) -> None:
    input_layer = SimpleNamespace(data=np.full((2, 2, 3), 0.25, dtype=np.float32))
    gui_state = make_test_controller_gui_state()
    gui_state.display.white_padding = 0.5
    captured = _run_simulation_case(
        monkeypatch,
        input_layer=input_layer,
        gui_state=gui_state,
        simulated_image=np.full((4, 4, 3), 0.5, dtype=np.float32),
        preview_image=np.full((8, 8, 3), 77, dtype=np.uint8),
    )

    display_input = captured['display_args']['image_data']
    assert display_input.shape == (4, 4, 3)
    np.testing.assert_allclose(display_input, np.full((4, 4, 3), 0.5, dtype=np.float32))
    assert captured['display_args']['padding_pixels'] == 2.0
    np.testing.assert_array_equal(captured['output_layer']['image'], np.full((8, 8, 3), 77, dtype=np.uint8))
    np.testing.assert_allclose(captured['output_layer']['float_image'], np.full((4, 4, 3), 0.5, dtype=np.float32))


def test_padding_pixels_uses_fraction_of_long_edge() -> None:
    controller = GuiController(viewer=object(), widgets=object())

    assert controller._padding_pixels_for_image(np.zeros((40, 100, 3), dtype=np.float32), 0.05) == 5
    assert controller._padding_pixels_for_image(np.zeros((40, 100, 3), dtype=np.float32), 0.019) == 1
    assert controller._padding_pixels_for_image(np.zeros((40, 100, 3), dtype=np.float32), 0.009) == 0


@pytest.mark.parametrize(
    ('method_name', 'expected_call'),
    [
        ('run_preview', {'compute_full_image': False, 'mode_label': 'Preview'}),
        ('run_scan', {'compute_full_image': True, 'mode_label': 'Scan'}),
    ],
    ids=['preview', 'scan'],
)
def test_run_preview_and_scan_start_async_simulation(monkeypatch, method_name: str, expected_call: dict[str, object]) -> None:
    controller = GuiController(viewer=object(), widgets=object())
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        controller,
        '_start_simulation',
        lambda *, compute_full_image, mode_label: captured.setdefault(
            'call',
            {'compute_full_image': compute_full_image, 'mode_label': mode_label},
        ),
    )

    getattr(controller, method_name)()

    assert captured['call'] == expected_call


def test_start_simulation_reports_persistent_computing_status(monkeypatch) -> None:
    input_layer = SimpleNamespace(data=np.full((2, 2, 3), 0.25, dtype=np.float32))
    simulation_section = SimpleNamespace(preview_button=None, scan_button=None, save_button=None)
    widgets = SimpleNamespace(simulation=simulation_section)
    controller = GuiController(viewer=object(), widgets=widgets)
    gui_state = make_test_controller_gui_state()
    captured: dict[str, object] = {}

    monkeypatch.setattr(controller, '_selected_input_layer', lambda: input_layer)
    monkeypatch.setattr(controller_module, 'collect_gui_state', lambda *, widgets: gui_state)
    monkeypatch.setattr(controller_module, 'build_params_from_state', lambda state: object())
    monkeypatch.setattr(controller_module, 'set_status', lambda viewer, message, timeout_ms=5000: captured.setdefault('status', (message, timeout_ms)))
    monkeypatch.setattr(controller._thread_pool, 'start', lambda worker: captured.setdefault('worker', worker))

    controller._start_simulation(compute_full_image=False, mode_label='Preview')

    assert captured['status'] == ('Computing preview...', 0)
    assert controller._active_simulation_label == 'Preview'


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