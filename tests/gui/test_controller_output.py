from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from spektrafilm_gui import controller as controller_module
from spektrafilm_gui.controller import (
    GuiController,
    OUTPUT_CCTF_ENCODING_KEY,
    OUTPUT_COLOR_SPACE_KEY,
    OUTPUT_DISPLAY_TRANSFORM_KEY,
    OUTPUT_FLOAT_DATA_KEY,
)

from .helpers import FakeLayer, StubToggle, make_test_controller_gui_state


pytestmark = pytest.mark.integration


def _make_output_layer(
    float_image: np.ndarray,
    *,
    output_color_space: str,
    output_cctf_encoding: bool,
    output_display_transform: bool = False,
) -> FakeLayer:
    return FakeLayer(
        np.uint8(float_image * 255),
        metadata={
            OUTPUT_FLOAT_DATA_KEY: float_image,
            OUTPUT_COLOR_SPACE_KEY: output_color_space,
            OUTPUT_CCTF_ENCODING_KEY: output_cctf_encoding,
            OUTPUT_DISPLAY_TRANSFORM_KEY: output_display_transform,
        },
    )


def _configure_save_output(monkeypatch, controller: GuiController, output_layer: FakeLayer, gui_state, captured: dict[str, object]) -> None:
    monkeypatch.setattr(controller, '_output_layer', lambda: output_layer)
    monkeypatch.setattr(controller_module, 'dialog_parent', lambda viewer: None)
    monkeypatch.setattr(controller_module, 'set_status', lambda viewer, message: captured.setdefault('status', message))
    monkeypatch.setattr(controller_module, 'load_dialog_dir', lambda key: '')
    monkeypatch.setattr(controller_module, 'save_dialog_dir', lambda key, directory: None)
    monkeypatch.setattr(
        controller_module.QFileDialog,
        'getSaveFileName',
        staticmethod(lambda *args, **kwargs: ('output.png', 'Images (*.png)')),
    )
    monkeypatch.setattr(controller_module, 'collect_gui_state', lambda *, widgets: gui_state)


def _capture_saved_output(monkeypatch, captured: dict[str, object]) -> None:
    def fake_save_image_oiio(filepath, image_data) -> None:
        captured.setdefault('saved', (filepath, image_data.copy()))

    monkeypatch.setattr(
        controller_module,
        'save_image_oiio',
        fake_save_image_oiio,
    )


def _run_save_output_case(
    monkeypatch,
    *,
    float_value: float,
    output_color_space: str,
    output_cctf_encoding: bool,
    saving_color_space: str,
    saving_cctf_encoding: bool,
    converted_delta: float | None,
) -> dict[str, object]:
    float_image = np.full((2, 2, 3), float_value, dtype=np.float32)
    output_layer = _make_output_layer(
        float_image,
        output_color_space=output_color_space,
        output_cctf_encoding=output_cctf_encoding,
    )
    controller = GuiController(viewer=object(), widgets=object())
    captured: dict[str, object] = {}
    gui_state = make_test_controller_gui_state()
    gui_state.simulation.saving_color_space = saving_color_space
    gui_state.simulation.saving_cctf_encoding = saving_cctf_encoding

    _configure_save_output(monkeypatch, controller, output_layer, gui_state, captured)

    if converted_delta is None:
        def fail_rgb_to_rgb(*args, **kwargs):
            raise AssertionError('RGB_to_RGB should not be called when color spaces and encoding flags match')

        monkeypatch.setattr(
            controller_module.colour,
            'RGB_to_RGB',
            fail_rgb_to_rgb,
        )
    else:
        def fake_rgb_to_rgb(image_data, input_color_space, output_color_space, apply_cctf_decoding, apply_cctf_encoding):
            captured['rgb_to_rgb'] = {
                'image_data': image_data.copy(),
                'input_color_space': input_color_space,
                'output_color_space': output_color_space,
                'apply_cctf_decoding': apply_cctf_decoding,
                'apply_cctf_encoding': apply_cctf_encoding,
            }
            return image_data + converted_delta

        monkeypatch.setattr(controller_module.colour, 'RGB_to_RGB', fake_rgb_to_rgb)

    _capture_saved_output(monkeypatch, captured)
    controller.save_output_layer()
    captured['float_image'] = float_image
    return captured


def _capture_status(monkeypatch) -> dict[str, object]:
    captured: dict[str, object] = {}

    def fake_set_status(_viewer, message) -> None:
        captured.setdefault('status', message)

    monkeypatch.setattr(controller_module, 'set_status', fake_set_status)
    return captured


def _assert_fallback_preview(preview: np.ndarray, status: str, *, expected_status: str) -> None:
    assert preview.dtype == np.uint8
    assert status == expected_status


def _assert_preview_conversion(
    preview: np.ndarray,
    status: str,
    *,
    expected_shape: tuple[int, int, int],
    expected_status: str,
    expected_center: np.ndarray,
    expected_corner: np.ndarray | None = None,
) -> None:
    assert preview.shape == expected_shape
    assert status == expected_status
    center_row = expected_shape[0] // 2
    center_col = expected_shape[1] // 2
    np.testing.assert_array_equal(preview[center_row, center_col], expected_center)
    if expected_corner is not None:
        np.testing.assert_array_equal(preview[0, 0], expected_corner)


def _make_display_transform_controller(*, with_toggle: bool) -> tuple[GuiController, StubToggle | None]:
    if not with_toggle:
        return GuiController(viewer=object(), widgets=object()), None

    toggle = StubToggle(True)
    controller = GuiController(
        viewer=object(),
        widgets=SimpleNamespace(display=SimpleNamespace(use_display_transform=toggle)),
    )
    return controller, toggle


@pytest.mark.parametrize(
    (
        'float_value',
        'output_color_space',
        'output_cctf_encoding',
        'saving_color_space',
        'saving_cctf_encoding',
        'converted_delta',
        'expected_input_space',
        'expected_output_space',
        'expected_saved_delta',
    ),
    [
        (0.25, 'sRGB', True, 'Display P3', False, 0.1, 'sRGB', 'Display P3', 0.1),
        (0.5, 'Display P3', True, 'Display P3', True, None, None, None, 0.0),
        (0.5, 'Display P3', True, 'Display P3', False, -0.1, 'Display P3', 'Display P3', -0.1),
    ],
    ids=['convert-color-space', 'skip-matching-render-metadata', 'reencode-cctf-only'],
)
def test_save_output_layer_respects_recorded_render_metadata(
    monkeypatch,
    float_value: float,
    output_color_space: str,
    output_cctf_encoding: bool,
    saving_color_space: str,
    saving_cctf_encoding: bool,
    converted_delta: float | None,
    expected_input_space: str | None,
    expected_output_space: str | None,
    expected_saved_delta: float,
) -> None:
    captured = _run_save_output_case(
        monkeypatch,
        float_value=float_value,
        output_color_space=output_color_space,
        output_cctf_encoding=output_cctf_encoding,
        saving_color_space=saving_color_space,
        saving_cctf_encoding=saving_cctf_encoding,
        converted_delta=converted_delta,
    )

    if converted_delta is None:
        assert 'rgb_to_rgb' not in captured
    else:
        rgb_to_rgb_call = captured['rgb_to_rgb']
        np.testing.assert_allclose(rgb_to_rgb_call['image_data'], captured['float_image'])
        assert rgb_to_rgb_call['input_color_space'] == expected_input_space
        assert rgb_to_rgb_call['output_color_space'] == expected_output_space
        assert rgb_to_rgb_call['apply_cctf_decoding'] is True
        assert rgb_to_rgb_call['apply_cctf_encoding'] is False

    saved_path, saved_image = captured['saved']
    assert saved_path == 'output.png'
    np.testing.assert_allclose(saved_image, captured['float_image'] + expected_saved_delta)


@pytest.mark.parametrize(
    ('image_data', 'padding_pixels', 'expected_shape', 'expected_center', 'expected_corner'),
    [
        (
            np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32),
            0.0,
            (1, 1, 3),
            np.array([0, 127, 255], dtype=np.uint8),
            None,
        ),
        (
            np.array([[[0.25, 0.5, 0.75]]], dtype=np.float32),
            1.0,
            (1, 1, 3),
            np.array([63, 127, 191], dtype=np.uint8),
            None,
        ),
    ],
    ids=['simple-preview', 'padding-ignored'],
)
def test_prepare_output_display_image_without_transform(
    image_data: np.ndarray,
    padding_pixels: float,
    expected_shape: tuple[int, int, int],
    expected_center: np.ndarray,
    expected_corner: np.ndarray | None,
) -> None:
    controller = GuiController(viewer=object(), widgets=object())

    preview, status = controller._prepare_output_display_image(
        image_data,
        output_color_space='sRGB',
        use_display_transform=False,
        padding_pixels=padding_pixels,
    )

    _assert_preview_conversion(
        preview,
        status,
        expected_shape=expected_shape,
        expected_status='Display transform: disabled',
        expected_center=expected_center,
        expected_corner=expected_corner,
    )


def test_prepare_output_display_image_uses_imagecms_transform(monkeypatch) -> None:
    controller = GuiController(viewer=object(), widgets=object())
    image_data = np.array([[[0.2, 0.4, 0.6]]], dtype=np.float32)
    captured: dict[str, object] = {}

    class FakePILImage:
        def __init__(self, array: np.ndarray):
            self.array = array

    monkeypatch.setattr(controller_module.ImageCms, 'get_display_profile', lambda: object())
    monkeypatch.setattr(controller_module.ImageCms, 'getProfileName', lambda profile: 'Studio Display ICC\x00')
    monkeypatch.setattr(controller_module.colour, 'RGB_to_RGB', lambda *args, **kwargs: np.full((1, 1, 3), 0.5, dtype=np.float32))
    monkeypatch.setattr(controller_module.ImageCms, 'createProfile', lambda name: f'profile:{name}')
    monkeypatch.setattr(
        controller_module.PILImage,
        'fromarray',
        lambda array, mode='RGB': captured.setdefault('source_image', FakePILImage(array.copy())),
    )

    def fake_profile_to_profile(source_image, source_profile, display_profile, outputMode='RGB'):
        captured['profile_to_profile'] = {
            'source_profile': source_profile,
            'display_profile': display_profile,
            'output_mode': outputMode,
            'image_data': source_image.array.copy(),
        }
        return np.full((1, 1, 3), 64, dtype=np.uint8)

    monkeypatch.setattr(controller_module.ImageCms, 'profileToProfile', fake_profile_to_profile)

    preview, status = controller._prepare_output_display_image(
        image_data,
        output_color_space='Display P3',
        use_display_transform=True,
    )

    np.testing.assert_array_equal(preview, np.full((1, 1, 3), 64, dtype=np.uint8))
    assert status == 'Display transform: active (Studio Display ICC)'
    assert captured['profile_to_profile']['source_profile'] == 'profile:sRGB'
    assert captured['profile_to_profile']['output_mode'] == 'RGB'
    np.testing.assert_array_equal(captured['profile_to_profile']['image_data'], np.full((1, 1, 3), 127, dtype=np.uint8))


def test_prepare_output_display_image_reports_missing_display_profile(monkeypatch) -> None:
    controller = GuiController(viewer=object(), widgets=object())
    image_data = np.array([[[0.2, 0.4, 0.6]]], dtype=np.float32)

    monkeypatch.setattr(controller_module.ImageCms, 'get_display_profile', lambda: None)

    preview, status = controller._prepare_output_display_image(
        image_data,
        output_color_space='Display P3',
        use_display_transform=True,
    )

    _assert_fallback_preview(preview, status, expected_status='Display transform: no display profile, using raw preview')


def test_prepare_output_display_image_reports_transform_failure(monkeypatch) -> None:
    controller = GuiController(viewer=object(), widgets=object())
    image_data = np.array([[[0.2, 0.4, 0.6]]], dtype=np.float32)

    monkeypatch.setattr(controller_module.ImageCms, 'get_display_profile', lambda: object())
    monkeypatch.setattr(controller_module.colour, 'RGB_to_RGB', lambda *args, **kwargs: np.full((1, 1, 3), 0.5, dtype=np.float32))
    monkeypatch.setattr(controller_module.ImageCms, 'createProfile', lambda name: f'profile:{name}')
    monkeypatch.setattr(controller_module.PILImage, 'fromarray', lambda array, mode='RGB': object())

    def raise_transform_error(*args, **kwargs):
        raise controller_module.ImageCms.PyCMSError('bad transform')

    monkeypatch.setattr(controller_module.ImageCms, 'profileToProfile', raise_transform_error)

    preview, status = controller._prepare_output_display_image(
        image_data,
        output_color_space='Display P3',
        use_display_transform=True,
    )

    _assert_fallback_preview(preview, status, expected_status='Display transform: transform failed, using raw preview')


@pytest.mark.parametrize(
    ('enabled', 'display_profile', 'profile_name', 'expected_status'),
    [
        (False, None, None, 'Display transform: disabled'),
        (True, object(), 'Adobe RGB Monitor\x00', 'Display transform: display profile found (Adobe RGB Monitor)'),
    ],
    ids=['disabled', 'profile-found'],
)
def test_report_display_transform_status_messages(
    monkeypatch,
    enabled: bool,
    display_profile: object | None,
    profile_name: str | None,
    expected_status: str,
) -> None:
    controller, _ = _make_display_transform_controller(with_toggle=False)
    captured = _capture_status(monkeypatch)

    monkeypatch.setattr(controller_module.ImageCms, 'get_display_profile', lambda: display_profile)
    if profile_name is not None:
        monkeypatch.setattr(controller_module.ImageCms, 'getProfileName', lambda profile: profile_name)

    controller.report_display_transform_status(enabled)

    assert captured['status'] == expected_status


def test_set_gray_18_canvas_enabled_updates_napari_background(monkeypatch) -> None:
    controller = GuiController(viewer=object(), widgets=object())
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        controller_module,
        'set_canvas_background',
        lambda viewer, *, gray_18_canvas: captured.setdefault('canvas', (viewer, gray_18_canvas)),
    )

    controller.set_gray_18_canvas_enabled(True)

    assert captured['canvas'] == (controller._viewer, True)


def test_report_display_transform_status_missing_profile(monkeypatch) -> None:
    controller, toggle = _make_display_transform_controller(with_toggle=True)
    captured = _capture_status(monkeypatch)

    monkeypatch.setattr(controller_module.ImageCms, 'get_display_profile', lambda: None)

    controller.report_display_transform_status(True)

    assert captured['status'] == 'Display transform unavailable: no display profile detected, disabled'
    assert toggle.checked is False


def test_sync_display_transform_availability_unchecks_when_profile_missing(monkeypatch) -> None:
    controller, toggle = _make_display_transform_controller(with_toggle=True)

    monkeypatch.setattr(controller_module.ImageCms, 'get_display_profile', lambda: None)

    available = controller.sync_display_transform_availability(report_status=False)

    assert available is False
    assert toggle.checked is False