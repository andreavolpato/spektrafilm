from __future__ import annotations

import numpy as np

from spektrafilm_gui import controller_layers as controller_layers_module
from spektrafilm_gui.controller_layers import ViewerLayerService


class _FakeSignal:
    def __init__(self) -> None:
        self._callbacks: list[object] = []

    def connect(self, callback) -> None:
        self._callbacks.append(callback)

    def emit(self) -> None:
        for callback in tuple(self._callbacks):
            callback()


class _FakeTimer:
    created: list['_FakeTimer'] = []

    def __init__(self) -> None:
        self.timeout = _FakeSignal()
        self.interval = None
        self.single_shot = None
        self.started = False
        self.stopped = False
        self.deleted = False
        _FakeTimer.created.append(self)

    def setInterval(self, interval: int) -> None:  # noqa: N802 - Qt API name
        self.interval = interval

    def setSingleShot(self, value: bool) -> None:  # noqa: N802 - Qt API name
        self.single_shot = value

    def start(self) -> None:
        self.started = True
        for _ in range(512):
            if self.stopped:
                return
            self.timeout.emit()
        raise AssertionError('output animation did not stop during test')

    def stop(self) -> None:
        self.stopped = True

    def deleteLater(self) -> None:  # noqa: N802 - Qt API name
        self.deleted = True


class _FakeLayer:
    def __init__(self, data: np.ndarray | None = None, *, name: str = 'layer', visible: bool = True) -> None:
        self.name = name
        self._data = np.zeros((1, 1, 3), dtype=np.float32) if data is None else np.array(data, copy=True)
        self.metadata: dict[str, object] = {}
        self.visible = visible
        self.scale = (1.0, 1.0)
        self.translate = (0.0, 0.0)
        self._type_string = 'image'
        self.data_history: list[np.ndarray] = []
        self.refresh_calls = 0

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, value: np.ndarray) -> None:
        self._data = np.array(value, copy=True)
        self.data_history.append(np.array(self._data, copy=True))

    def refresh(self) -> None:
        self.refresh_calls += 1


class _FakeLayerSelection:
    def __init__(self) -> None:
        self.active = None


class _FakeLayerList(list):
    def __init__(self, layers: list[_FakeLayer] | None = None) -> None:
        super().__init__(layers or [])
        self.selection = _FakeLayerSelection()

    def move(self, src: int, dst: int) -> None:
        layer = self.pop(src)
        insert_at = dst - 1 if dst > src else dst
        self.insert(insert_at, layer)


class _FakeViewer:
    def __init__(self) -> None:
        self.layers = _FakeLayerList()

    def add_image(self, image: np.ndarray, name: str) -> _FakeLayer:
        layer = _FakeLayer(image, name=name)
        self.layers.append(layer)
        return layer


def _make_service() -> ViewerLayerService:
    return ViewerLayerService(
        viewer=_FakeViewer(),
        output_float_data_key='float',
        output_color_space_key='color',
        output_cctf_encoding_key='cctf',
        output_display_transform_key='display',
    )


def test_first_output_preview_runs_polaroid_frame_sequence(monkeypatch) -> None:
    _FakeTimer.created.clear()
    monkeypatch.setattr(controller_layers_module, 'QTimer', _FakeTimer)
    service = _make_service()
    service.set_or_add_input_preview_layer(np.full((2, 1, 3), 0.75, dtype=np.float32), white_padding=0.1)
    white_border = service.white_border_layer()
    assert white_border is not None
    output_image = np.full((4, 4, 3), 77, dtype=np.uint8)

    service.set_or_add_output_layer(
        output_image,
        float_image=np.full((4, 4, 3), 0.5, dtype=np.float32),
        output_color_space='ACES2065-1',
        output_cctf_encoding=True,
        use_display_transform=False,
    )

    output_layer = service.output_layer()
    assert output_layer is not None
    assert white_border.data_history == []
    assert len(output_layer.data_history) >= 2
    assert not np.array_equal(output_layer.data_history[0], output_image)
    np.testing.assert_array_equal(output_layer.data, output_image)
    assert output_layer.refresh_calls >= len(output_layer.data_history)
    assert len(_FakeTimer.created) == 1
    assert _FakeTimer.created[0].started is True
    assert _FakeTimer.created[0].stopped is True


def test_reused_input_preview_layer_refreshes_after_data_swap(monkeypatch) -> None:
    _FakeTimer.created.clear()
    monkeypatch.setattr(controller_layers_module, 'QTimer', _FakeTimer)
    service = _make_service()
    first_preview = np.full((2, 1, 3), 0.25, dtype=np.float32)
    second_preview = np.full((2, 1, 3), 0.75, dtype=np.float32)

    service.set_or_add_input_preview_layer(first_preview, white_padding=0.1)

    preview_layer = service.preview_input_layer()
    assert preview_layer is not None
    initial_refresh_calls = preview_layer.refresh_calls

    service.set_or_add_input_preview_layer(second_preview, white_padding=0.1)

    assert service.preview_input_layer() is preview_layer
    np.testing.assert_array_equal(preview_layer.data, second_preview)
    assert preview_layer.refresh_calls > initial_refresh_calls


def test_visible_output_updates_crossfade_without_restarting_polaroid_animation(monkeypatch) -> None:
    _FakeTimer.created.clear()
    monkeypatch.setattr(controller_layers_module, 'QTimer', _FakeTimer)
    service = _make_service()
    service.set_or_add_input_preview_layer(np.full((2, 1, 3), 0.75, dtype=np.float32), white_padding=0.1)
    first_image = np.full((4, 4, 3), 77, dtype=np.uint8)
    second_image = np.full((4, 4, 3), 88, dtype=np.uint8)

    service.set_or_add_output_layer(
        first_image,
        float_image=np.full((4, 4, 3), 0.5, dtype=np.float32),
        output_color_space='ACES2065-1',
        output_cctf_encoding=True,
        use_display_transform=False,
    )
    output_layer = service.output_layer()
    assert output_layer is not None
    initial_history_length = len(output_layer.data_history)

    service.set_or_add_output_layer(
        second_image,
        float_image=np.full((4, 4, 3), 0.6, dtype=np.float32),
        output_color_space='ACES2065-1',
        output_cctf_encoding=True,
        use_display_transform=False,
    )

    assert len(_FakeTimer.created) == 2
    assert _FakeTimer.created[1].started is True
    assert _FakeTimer.created[1].stopped is True
    np.testing.assert_array_equal(output_layer.data, second_image)
    new_frames = output_layer.data_history[initial_history_length:]
    assert len(new_frames) >= controller_layers_module.OUTPUT_LAYER_CROSSFADE_FRAMES
    assert not np.array_equal(new_frames[0], second_image)
    np.testing.assert_array_equal(new_frames[-1], second_image)


def test_preview_to_scan_shape_change_recreates_output_layer_without_restarting_animation(monkeypatch) -> None:
    _FakeTimer.created.clear()
    monkeypatch.setattr(controller_layers_module, 'QTimer', _FakeTimer)
    service = _make_service()
    service.set_or_add_input_preview_layer(np.full((2, 1, 3), 0.75, dtype=np.float32), white_padding=0.1)
    preview_image = np.full((4, 4, 3), 77, dtype=np.uint8)
    scan_image = np.full((8, 8, 3), 88, dtype=np.uint8)

    service.set_or_add_output_layer(
        preview_image,
        float_image=np.full((4, 4, 3), 0.5, dtype=np.float32),
        output_color_space='ACES2065-1',
        output_cctf_encoding=True,
        use_display_transform=False,
    )

    preview_layer = service.output_layer()
    assert preview_layer is not None

    service.set_or_add_output_layer(
        scan_image,
        float_image=np.full((8, 8, 3), 0.6, dtype=np.float32),
        output_color_space='ACES2065-1',
        output_cctf_encoding=True,
        use_display_transform=False,
    )

    scan_layer = service.output_layer()
    assert scan_layer is not None
    assert scan_layer is not preview_layer
    assert len(_FakeTimer.created) == 1
    np.testing.assert_array_equal(scan_layer.data, scan_image)
    assert scan_layer.refresh_calls >= 1


def test_large_output_skips_polaroid_animation(monkeypatch) -> None:
    _FakeTimer.created.clear()
    monkeypatch.setattr(controller_layers_module, 'QTimer', _FakeTimer)
    monkeypatch.setattr(controller_layers_module, 'OUTPUT_LAYER_ANIMATION_MAX_PIXELS', 4)
    service = _make_service()
    service.set_or_add_input_preview_layer(np.full((2, 1, 3), 0.75, dtype=np.float32), white_padding=0.1)
    output_image = np.full((3, 2, 3), 77, dtype=np.uint8)

    service.set_or_add_output_layer(
        output_image,
        float_image=np.full((3, 2, 3), 0.5, dtype=np.float32),
        output_color_space='ACES2065-1',
        output_cctf_encoding=True,
        use_display_transform=False,
    )

    output_layer = service.output_layer()
    assert output_layer is not None
    assert len(_FakeTimer.created) == 0
    assert output_layer.data_history == []
    np.testing.assert_array_equal(output_layer.data, output_image)