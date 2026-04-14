from __future__ import annotations

import numpy as np
import pytest

from spektrafilm_gui import controller_layers as controller_layers_module
from spektrafilm_gui.controller import (
    OUTPUT_CCTF_ENCODING_KEY,
    OUTPUT_COLOR_SPACE_KEY,
    OUTPUT_DISPLAY_TRANSFORM_KEY,
    OUTPUT_FLOAT_DATA_KEY,
)
from spektrafilm_gui.controller_layers import (
    INPUT_PREVIEW_LAYER_NAME,
    OUTPUT_LAYER_NAME,
    ViewerLayerService,
    WATERMARK_LAYER_NAME,
    WHITE_BORDER_LAYER_NAME,
)

from .helpers import FakeLayer, FakeLayerList, FakeViewer


def _make_service(viewer: FakeViewer) -> ViewerLayerService:
    return ViewerLayerService(
        viewer=viewer,
        output_float_data_key=OUTPUT_FLOAT_DATA_KEY,
        output_color_space_key=OUTPUT_COLOR_SPACE_KEY,
        output_cctf_encoding_key=OUTPUT_CCTF_ENCODING_KEY,
        output_display_transform_key=OUTPUT_DISPLAY_TRANSFORM_KEY,
    )


class VisibilityTrackingLayer:
    def __init__(self, data: np.ndarray | None = None, *, name: str = 'layer', visible: bool = True) -> None:
        self.name = name
        self.data = np.zeros((1, 1, 3), dtype=np.float32) if data is None else data
        self.metadata: dict[str, object] = {}
        self.scale = (1.0, 1.0)
        self.translate = (0.0, 0.0)
        self._type_string = 'image'
        self.visible_set_calls = 0
        self._visible = bool(visible)

    @property
    def visible(self) -> bool:
        return self._visible

    @visible.setter
    def visible(self, value: bool) -> None:
        self.visible_set_calls += 1
        self._visible = bool(value)


class VisibilityTrackingViewer(FakeViewer):
    def add_image(self, image: np.ndarray, name: str) -> VisibilityTrackingLayer:
        layer = VisibilityTrackingLayer(image, name=name)
        self.layers.append(layer)
        return layer


class MoveTrackingLayerList(FakeLayerList):
    def __init__(self, layers: list[FakeLayer], *, active=None) -> None:
        super().__init__(layers, active=active)
        self.move_calls = 0

    def move(self, src: int, dst: int) -> None:
        self.move_calls += 1
        super().move(src, dst)


class TrackingViewer(FakeViewer):
    def __init__(self, layers: list[FakeLayer] | None = None):
        super().__init__([])
        self.layers = MoveTrackingLayerList(layers or [])
        self.reset_view_calls = 0

    def add_image(self, image: np.ndarray, name: str) -> VisibilityTrackingLayer:
        layer = VisibilityTrackingLayer(image, name=name)
        self.layers.append(layer)
        return layer


class _ImmediateSignal:
    def __init__(self) -> None:
        self._callbacks: list[object] = []

    def connect(self, callback) -> None:
        self._callbacks.append(callback)

    def emit(self) -> None:
        for callback in tuple(self._callbacks):
            callback()


class _ImmediateTimer:
    def __init__(self) -> None:
        self.timeout = _ImmediateSignal()
        self.interval = None
        self.single_shot = None
        self.stopped = False

    def setInterval(self, interval: int) -> None:  # noqa: N802 - Qt API name
        self.interval = interval

    def setSingleShot(self, value: bool) -> None:  # noqa: N802 - Qt API name
        self.single_shot = value

    def start(self) -> None:
        for _ in range(512):
            if self.stopped:
                return
            self.timeout.emit()
        raise AssertionError('output animation did not stop during test')

    def stop(self) -> None:
        self.stopped = True

    def deleteLater(self) -> None:  # noqa: N802 - Qt API name
        return


@pytest.fixture(autouse=True)
def _complete_output_animation(monkeypatch) -> None:
    monkeypatch.setattr(controller_layers_module, 'QTimer', _ImmediateTimer)


@pytest.fixture(autouse=True)
def _stub_virtual_paper_back(monkeypatch) -> None:
    def fake_virtual_photo_paper_back(*, canvas_size, **_kwargs):
        width, height = (canvas_size, canvas_size) if isinstance(canvas_size, int) else canvas_size
        return np.full((int(height), int(width), 3), 0.2, dtype=np.float32)

    controller_layers_module.clear_watermark_image_cache()
    monkeypatch.setattr(controller_layers_module, 'virtual_photo_paper_back', fake_virtual_photo_paper_back)
    yield
    controller_layers_module.clear_watermark_image_cache()


def test_set_or_add_input_preview_layer_creates_fixed_layers_with_shared_world_frame() -> None:
    viewer = FakeViewer([FakeLayer(name='older-1'), FakeLayer(name='older-2')])
    service = _make_service(viewer)
    preview_image = np.full((2, 1, 3), 0.75, dtype=np.float32)

    service.set_or_add_input_preview_layer(
        preview_image,
        white_padding=0.25,
    )

    assert [layer.name for layer in viewer.layers[-3:]] == [
        WHITE_BORDER_LAYER_NAME,
        WATERMARK_LAYER_NAME,
        INPUT_PREVIEW_LAYER_NAME,
    ]

    white_border = service.white_border_layer()
    watermark_layer = service.watermark_layer()
    preview_layer = service.preview_input_layer()
    assert white_border is not None
    assert watermark_layer is not None
    assert preview_layer is not None
    assert viewer.layers.selection.active is white_border
    assert white_border.visible is True
    assert watermark_layer.visible is True
    assert preview_layer.visible is False
    assert watermark_layer.data.shape == (1024, 512, 3)
    assert watermark_layer.interpolation2d == 'spline36'
    assert preview_layer.scale == (0.5, 0.5)
    assert white_border.scale == (0.75, 1.0)


def test_repeated_input_preview_updates_skip_stack_reorder() -> None:
    viewer = TrackingViewer()
    service = _make_service(viewer)
    preview_image = np.full((2, 1, 3), 0.75, dtype=np.float32)

    service.set_or_add_input_preview_layer(
        preview_image,
        white_padding=0.25,
    )
    move_calls = viewer.layers.move_calls

    service.set_or_add_input_preview_layer(
        preview_image,
        white_padding=0.25,
    )

    assert viewer.layers.move_calls == move_calls
def test_set_or_add_output_layer_matches_existing_input_world_geometry() -> None:
    viewer = FakeViewer()
    service = _make_service(viewer)
    service.set_or_add_input_preview_layer(
        np.full((2, 1, 3), 0.75, dtype=np.float32),
        white_padding=0.1,
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
    assert output_layer.interpolation2d == 'spline36'
    assert output_layer.visible is True
    assert preview_layer.visible is False
    assert viewer.layers.selection.active is output_layer
    assert output_layer.scale == (0.125, 0.125)


def test_set_or_add_output_layer_applies_requested_interpolation_mode() -> None:
    viewer = FakeViewer()
    service = _make_service(viewer)
    service.set_or_add_input_preview_layer(
        np.full((2, 1, 3), 0.75, dtype=np.float32),
        white_padding=0.1,
    )

    service.set_or_add_output_layer(
        np.full((8, 4, 3), 77, dtype=np.uint8),
        float_image=np.full((8, 4, 3), 0.5, dtype=np.float32),
        output_color_space='ACES2065-1',
        output_cctf_encoding=True,
        use_display_transform=False,
        output_interpolation_mode='nearest',
    )

    output_layer = service.output_layer()
    assert output_layer is not None
    assert output_layer.interpolation2d == 'nearest'


def test_set_or_add_output_layer_preserves_square_pixels_for_cropped_aspect_changes() -> None:
    viewer = FakeViewer()
    service = _make_service(viewer)
    service.set_or_add_input_preview_layer(
        np.full((2, 1, 3), 0.75, dtype=np.float32),
        white_padding=0.1,
    )

    image = np.full((4, 4, 3), 77, dtype=np.uint8)
    float_image = np.full((4, 4, 3), 0.5, dtype=np.float32)
    service.set_or_add_output_layer(
        image,
        float_image=float_image,
        output_color_space='ACES2065-1',
        output_cctf_encoding=True,
        use_display_transform=False,
    )

    output_layer = service.output_layer()
    assert output_layer is not None
    assert output_layer.scale == (0.125, 0.125)
    assert output_layer.translate == (-0.25, -0.25)


def test_repeated_output_updates_skip_redundant_visibility_write_but_restore_hidden_layer() -> None:
    viewer = VisibilityTrackingViewer()
    service = _make_service(viewer)
    service.set_or_add_input_preview_layer(
        np.full((2, 1, 3), 0.75, dtype=np.float32),
        white_padding=0.1,
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
    assert isinstance(output_layer, VisibilityTrackingLayer)
    visible_set_calls = output_layer.visible_set_calls

    service.set_or_add_output_layer(
        image,
        float_image=float_image,
        output_color_space='ACES2065-1',
        output_cctf_encoding=True,
        use_display_transform=False,
    )

    assert output_layer.visible is True
    assert output_layer.visible_set_calls == visible_set_calls

    output_layer.visible = False
    service.set_or_add_output_layer(
        image,
        float_image=float_image,
        output_color_space='ACES2065-1',
        output_cctf_encoding=True,
        use_display_transform=False,
    )

    assert output_layer.visible is True
    assert output_layer.visible_set_calls == visible_set_calls + 2


def test_input_preview_hides_existing_output_layer_but_reuses_it_for_next_output() -> None:
    viewer = TrackingViewer()
    service = _make_service(viewer)
    preview_image = np.full((2, 1, 3), 0.75, dtype=np.float32)

    service.set_or_add_input_preview_layer(
        preview_image,
        white_padding=0.1,
    )

    output_image = np.full((8, 4, 3), 77, dtype=np.uint8)
    float_image = np.full((8, 4, 3), 0.5, dtype=np.float32)
    service.set_or_add_output_layer(
        output_image,
        float_image=float_image,
        output_color_space='ACES2065-1',
        output_cctf_encoding=True,
        use_display_transform=False,
    )

    hidden_output = service.image_layer(OUTPUT_LAYER_NAME)
    assert isinstance(hidden_output, VisibilityTrackingLayer)
    visible_set_calls = hidden_output.visible_set_calls

    service.set_or_add_input_preview_layer(
        preview_image,
        white_padding=0.1,
    )

    assert service.output_layer() is None
    assert service.image_layer(OUTPUT_LAYER_NAME) is hidden_output
    assert hidden_output.visible is False
    assert hidden_output.visible_set_calls == visible_set_calls + 1

    service.set_or_add_output_layer(
        output_image,
        float_image=float_image,
        output_color_space='ACES2065-1',
        output_cctf_encoding=True,
        use_display_transform=False,
    )

    assert service.output_layer() is hidden_output
    assert hidden_output.visible is True


def test_hidden_output_with_changed_shape_is_recreated_for_next_output() -> None:
    viewer = TrackingViewer()
    service = _make_service(viewer)
    preview_image = np.full((2, 1, 3), 0.75, dtype=np.float32)

    service.set_or_add_input_preview_layer(
        preview_image,
        white_padding=0.1,
    )

    first_output = np.full((4, 4, 3), 77, dtype=np.uint8)
    second_output = np.full((8, 8, 3), 88, dtype=np.uint8)
    service.set_or_add_output_layer(
        first_output,
        float_image=np.full((4, 4, 3), 0.5, dtype=np.float32),
        output_color_space='ACES2065-1',
        output_cctf_encoding=True,
        use_display_transform=False,
    )

    hidden_output = service.image_layer(OUTPUT_LAYER_NAME)
    assert isinstance(hidden_output, VisibilityTrackingLayer)

    service.set_or_add_input_preview_layer(
        preview_image,
        white_padding=0.1,
    )

    service.set_or_add_output_layer(
        second_output,
        float_image=np.full((8, 8, 3), 0.6, dtype=np.float32),
        output_color_space='ACES2065-1',
        output_cctf_encoding=True,
        use_display_transform=False,
    )

    recreated_output = service.output_layer()
    assert recreated_output is not None
    assert recreated_output is not hidden_output
    np.testing.assert_array_equal(recreated_output.data, second_output)


def test_input_preview_update_can_preserve_visible_output_and_active_layer() -> None:
    viewer = TrackingViewer()
    service = _make_service(viewer)
    preview_image = np.full((2, 1, 3), 0.75, dtype=np.float32)

    service.set_or_add_input_preview_layer(
        preview_image,
        white_padding=0.1,
    )

    output_image = np.full((8, 4, 3), 77, dtype=np.uint8)
    float_image = np.full((8, 4, 3), 0.5, dtype=np.float32)
    service.set_or_add_output_layer(
        output_image,
        float_image=float_image,
        output_color_space='ACES2065-1',
        output_cctf_encoding=True,
        use_display_transform=False,
    )

    output_layer = service.output_layer()
    assert isinstance(output_layer, VisibilityTrackingLayer)
    visible_set_calls = output_layer.visible_set_calls
    viewer.layers.selection.active = output_layer

    updated_preview = np.full((4, 2, 3), 0.5, dtype=np.float32)
    service.set_or_add_input_preview_layer(
        updated_preview,
        white_padding=0.1,
        hide_output=False,
        set_active=False,
    )

    np.testing.assert_allclose(service.preview_input_layer().data, updated_preview)
    assert service.preview_input_layer().visible is False
    assert output_layer.visible is True
    assert output_layer.visible_set_calls == visible_set_calls
    assert viewer.layers.selection.active is output_layer


def test_output_layer_render_settings_fall_back_without_output_layer() -> None:
    service = _make_service(FakeViewer([FakeLayer(name='input')]))

    assert service.output_layer_render_settings(default_color_space='sRGB', default_cctf_encoding=True) == ('sRGB', True)


def test_output_layer_float_data_returns_none_without_metadata() -> None:
    service = _make_service(FakeViewer([FakeLayer(name='output')]))

    assert service.output_layer_float_data() is None
