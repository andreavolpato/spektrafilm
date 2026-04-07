from __future__ import annotations

import numpy as np

from spektrafilm_gui.controller import (
    INPUT_PADDING_PIXELS_KEY,
    INPUT_RAW_DATA_KEY,
    OUTPUT_CCTF_ENCODING_KEY,
    OUTPUT_COLOR_SPACE_KEY,
    OUTPUT_DISPLAY_TRANSFORM_KEY,
    OUTPUT_FLOAT_DATA_KEY,
)
from spektrafilm_gui.controller_layers import (
    INPUT_COLOR_PREVIEW_LAYER_NAME,
    INPUT_LAYER_NAME,
    INPUT_PREVIEW_LAYER_NAME,
    ViewerLayerService,
    WHITE_BORDER_LAYER_NAME,
)

from .helpers import FakeLayer, FakeViewer


def _make_service(viewer: FakeViewer) -> ViewerLayerService:
    return ViewerLayerService(
        viewer=viewer,
        input_raw_data_key=INPUT_RAW_DATA_KEY,
        input_padding_pixels_key=INPUT_PADDING_PIXELS_KEY,
        output_float_data_key=OUTPUT_FLOAT_DATA_KEY,
        output_color_space_key=OUTPUT_COLOR_SPACE_KEY,
        output_cctf_encoding_key=OUTPUT_CCTF_ENCODING_KEY,
        output_display_transform_key=OUTPUT_DISPLAY_TRANSFORM_KEY,
    )


def test_set_or_add_input_stack_creates_fixed_layers_with_shared_world_frame() -> None:
    viewer = FakeViewer([FakeLayer(name='older-1'), FakeLayer(name='older-2')])
    service = _make_service(viewer)
    captured: dict[str, object] = {}
    full_image = np.full((4, 2, 3), 0.25, dtype=np.float32)
    preview_image = np.full((2, 1, 3), 0.5, dtype=np.float32)
    color_preview_image = np.full((2, 1, 3), 0.75, dtype=np.float32)

    service.set_or_add_input_stack(
        full_image,
        preview_image=preview_image,
        color_preview_image=color_preview_image,
        white_padding=0.25,
        refresh_input_layers_fn=lambda *, selected_name: captured.setdefault('selected_name', selected_name),
    )

    assert [layer.name for layer in viewer.layers[-4:]] == [
        WHITE_BORDER_LAYER_NAME,
        INPUT_LAYER_NAME,
        INPUT_PREVIEW_LAYER_NAME,
        INPUT_COLOR_PREVIEW_LAYER_NAME,
    ]
    assert [layer.name for layer in service.available_input_layers()] == [
        INPUT_COLOR_PREVIEW_LAYER_NAME,
        INPUT_PREVIEW_LAYER_NAME,
        INPUT_LAYER_NAME,
    ]

    white_border = service.white_border_layer()
    input_layer = service.input_layer()
    preview_layer = service.preview_input_layer()
    color_preview_layer = service.color_preview_layer()
    assert white_border is not None
    assert input_layer is not None
    assert preview_layer is not None
    assert color_preview_layer is not None

    np.testing.assert_allclose(input_layer.metadata[INPUT_RAW_DATA_KEY], full_image)
    np.testing.assert_allclose(preview_layer.metadata[INPUT_RAW_DATA_KEY], preview_image)
    np.testing.assert_allclose(color_preview_layer.metadata[INPUT_RAW_DATA_KEY], preview_image)
    assert input_layer.metadata[INPUT_PADDING_PIXELS_KEY] == 0.0
    assert captured['selected_name'] == INPUT_COLOR_PREVIEW_LAYER_NAME
    assert viewer.layers.selection.active is color_preview_layer

    assert white_border.visible is True
    assert input_layer.visible is True
    assert preview_layer.visible is True
    assert color_preview_layer.visible is True
    assert input_layer.scale == (0.25, 0.25)
    assert preview_layer.scale == (0.5, 0.5)
    assert color_preview_layer.scale == (0.5, 0.5)
    assert white_border.scale == (0.75, 1.0)


def test_set_or_add_output_layer_matches_existing_input_world_geometry() -> None:
    viewer = FakeViewer()
    service = _make_service(viewer)
    service.set_or_add_input_stack(
        np.full((4, 2, 3), 0.25, dtype=np.float32),
        preview_image=np.full((2, 1, 3), 0.5, dtype=np.float32),
        color_preview_image=np.full((2, 1, 3), 0.75, dtype=np.float32),
        white_padding=0.1,
        refresh_input_layers_fn=lambda **_: None,
    )

    image = np.full((8, 4, 3), 77, dtype=np.uint8)
    float_image = np.full((8, 4, 3), 0.5, dtype=np.float32)
    service.set_or_add_output_layer(
        image,
        float_image=float_image,
        output_color_space='ACES2065-1',
        output_cctf_encoding=True,
        use_display_transform=False,
    )

    output_layer = service.output_layer()
    preview_layer = service.preview_input_layer()
    assert output_layer is not None
    assert preview_layer is not None
    assert viewer.layers[-1] is output_layer
    np.testing.assert_array_equal(output_layer.data, image)
    np.testing.assert_allclose(output_layer.metadata[OUTPUT_FLOAT_DATA_KEY], float_image)
    assert output_layer.metadata[OUTPUT_COLOR_SPACE_KEY] == 'ACES2065-1'
    assert output_layer.metadata[OUTPUT_CCTF_ENCODING_KEY] is True
    assert output_layer.metadata[OUTPUT_DISPLAY_TRANSFORM_KEY] is False
    assert output_layer.visible is True
    assert preview_layer.visible is True
    assert viewer.layers.selection.active is output_layer
    assert output_layer.scale == (0.125, 0.125)


def test_output_layer_render_settings_fall_back_without_output_layer() -> None:
    service = _make_service(FakeViewer([FakeLayer(name='input')]))

    assert service.output_layer_render_settings(default_color_space='sRGB', default_cctf_encoding=True) == ('sRGB', True)


def test_output_layer_float_data_returns_none_without_metadata() -> None:
    service = _make_service(FakeViewer([FakeLayer(name='output')]))

    assert service.output_layer_float_data() is None
