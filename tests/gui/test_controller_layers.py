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
from spektrafilm_gui.controller_layers import ViewerLayerService

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


def test_set_or_add_input_layer_preserves_raw_metadata_and_refreshes_selection() -> None:
    viewer = FakeViewer([FakeLayer(name='older-1'), FakeLayer(name='older-2')])
    service = _make_service(viewer)
    captured: dict[str, object] = {}
    image = np.full((2, 2, 3), 0.25, dtype=np.float32)

    service.set_or_add_input_layer(
        image,
        layer_name='example',
        white_padding=0.5,
        padding_pixels_for_image_fn=lambda data, fraction: 1,
        apply_white_padding_fn=lambda data, padding: np.pad(data, ((1, 1), (1, 1), (0, 0)), mode='constant', constant_values=1.0),
        refresh_input_layers_fn=lambda *, selected_name: captured.setdefault('selected_name', selected_name),
    )

    layer = viewer.layers[-1]
    assert layer.name == 'example'
    assert layer.data.shape == (4, 4, 3)
    np.testing.assert_allclose(layer.data[1:3, 1:3], image)
    np.testing.assert_allclose(layer.metadata[INPUT_RAW_DATA_KEY], image)
    assert layer.metadata[INPUT_PADDING_PIXELS_KEY] == 1.0
    assert captured['selected_name'] == 'example'
    assert layer.visible is True
    assert all(other_layer.visible is False for other_layer in viewer.layers[:-1])


def test_set_or_add_output_layer_updates_existing_output_metadata_and_visibility() -> None:
    output_layer = FakeLayer(name='output', visible=False)
    other_layer = FakeLayer(name='other', visible=True)
    viewer = FakeViewer([output_layer, other_layer])
    service = _make_service(viewer)
    image = np.full((2, 2, 3), 77, dtype=np.uint8)
    float_image = np.full((2, 2, 3), 0.5, dtype=np.float32)

    service.set_or_add_output_layer(
        image,
        float_image=float_image,
        output_color_space='ACES2065-1',
        output_cctf_encoding=True,
        use_display_transform=False,
    )

    assert viewer.layers[-1] is output_layer
    np.testing.assert_array_equal(output_layer.data, image)
    np.testing.assert_allclose(output_layer.metadata[OUTPUT_FLOAT_DATA_KEY], float_image)
    assert output_layer.metadata[OUTPUT_COLOR_SPACE_KEY] == 'ACES2065-1'
    assert output_layer.metadata[OUTPUT_CCTF_ENCODING_KEY] is True
    assert output_layer.metadata[OUTPUT_DISPLAY_TRANSFORM_KEY] is False
    assert output_layer.visible is True
    assert other_layer.visible is False


def test_output_layer_render_settings_fall_back_without_output_layer() -> None:
    service = _make_service(FakeViewer([FakeLayer(name='input')]))

    assert service.output_layer_render_settings(default_color_space='sRGB', default_cctf_encoding=True) == ('sRGB', True)


def test_output_layer_float_data_returns_none_without_metadata() -> None:
    service = _make_service(FakeViewer([FakeLayer(name='output')]))

    assert service.output_layer_float_data() is None