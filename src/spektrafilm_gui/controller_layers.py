from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from qtpy import QtCore

from spektrafilm_gui.polaroid_animation import prepare_polaroid_state, render_polaroid_frame

if TYPE_CHECKING:
    from napari.layers import Image as NapariImageLayer


QTimer = getattr(QtCore, 'QTimer')


@dataclass(slots=True)
class _LayerAnimationHandle:
    timer: Any
    final_image: np.ndarray


INPUT_LAYER_NAME = 'input'
INPUT_PREVIEW_LAYER_NAME = 'input_preview'
WHITE_BORDER_LAYER_NAME = 'white_border'
OUTPUT_LAYER_NAME = 'output'
STACK_LAYER_ORDER = (
    WHITE_BORDER_LAYER_NAME,
    INPUT_PREVIEW_LAYER_NAME,
    OUTPUT_LAYER_NAME,
)
OUTPUT_LAYER_ANIMATION_DURATION_MS = 1000
OUTPUT_LAYER_ANIMATION_INTERVAL_MS = 32
OUTPUT_LAYER_CROSSFADE_FRAMES = 8
OUTPUT_LAYER_ANIMATION_MAX_PIXELS = 1_500_000


def is_napari_image_layer(layer: object) -> bool:
    if getattr(layer, '_type_string', None) == 'image':
        return True

    layer_type = type(layer)
    if layer_type.__name__ == 'Image' and layer_type.__module__.startswith('napari.layers.image'):
        return True

    try:
        from napari.layers import Image as NapariImageLayer
    except ImportError:
        return False
    return isinstance(layer, NapariImageLayer)


def _normalized_world_size(image: np.ndarray) -> tuple[float, float]:
    data = np.asarray(image)
    if data.ndim < 2:
        return 1.0, 1.0

    height, width = data.shape[:2]
    long_edge = max(int(height), int(width), 1)
    return float(height) / float(long_edge), float(width) / float(long_edge)


def _padded_world_size(image_world_size: tuple[float, float], padding_fraction: float) -> tuple[float, float]:
    padding = max(0.0, float(padding_fraction))
    return image_world_size[0] + 2.0 * padding, image_world_size[1] + 2.0 * padding


def _set_layer_geometry(layer: NapariImageLayer, *, world_size: tuple[float, float]) -> None:
    data = np.asarray(layer.data)
    if data.ndim < 2:
        return

    height, width = data.shape[:2]
    scale = (
        world_size[0] / max(int(height), 1),
        world_size[1] / max(int(width), 1),
    )
    translate = (-0.5 * world_size[0], -0.5 * world_size[1])
    setattr(layer, 'scale', scale)
    setattr(layer, 'translate', translate)


def _fit_image_world_size(
    image: np.ndarray,
    *,
    bounding_world_size: tuple[float, float],
) -> tuple[float, float]:
    data = np.asarray(image)
    if data.ndim < 2:
        return bounding_world_size

    height, width = data.shape[:2]
    if height <= 0 or width <= 0:
        return bounding_world_size

    fit_scale = min(
        float(bounding_world_size[0]) / float(height),
        float(bounding_world_size[1]) / float(width),
    )
    return float(height) * fit_scale, float(width) * fit_scale


def _supports_output_layer_animation(image: np.ndarray) -> bool:
    data = np.asarray(image)
    if data.ndim != 3 or data.shape[2] < 3:
        return False
    height, width = data.shape[:2]
    if height <= 0 or width <= 0:
        return False
    return int(height) * int(width) <= OUTPUT_LAYER_ANIMATION_MAX_PIXELS


def _supports_output_layer_crossfade(source_image: np.ndarray, target_image: np.ndarray) -> bool:
    source = np.asarray(source_image)
    target = np.asarray(target_image)
    if source.shape != target.shape:
        return False
    return _supports_output_layer_animation(source) and _supports_output_layer_animation(target)


def _layer_data_shape(layer: NapariImageLayer) -> tuple[int, ...]:
    return tuple(int(dimension) for dimension in np.asarray(layer.data).shape)


def _image_to_float_frame(image: np.ndarray) -> np.ndarray:
    data = np.asarray(image)
    if np.issubdtype(data.dtype, np.integer):
        return data.astype(np.float32) / float(np.iinfo(data.dtype).max)
    return np.clip(data.astype(np.float32), 0.0, 1.0)


def _coerce_output_animation_frame(
    frame: np.ndarray,
    *,
    reference_image: np.ndarray,
) -> np.ndarray:
    reference = np.asarray(reference_image)
    if np.issubdtype(reference.dtype, np.integer):
        info = np.iinfo(reference.dtype)
        scaled = np.rint(np.clip(frame, 0.0, 1.0) * float(info.max))
        return np.asarray(np.clip(scaled, info.min, info.max), dtype=reference.dtype)
    if np.issubdtype(reference.dtype, np.floating):
        return np.asarray(frame, dtype=reference.dtype)
    return np.asarray(frame, dtype=np.float32)


def _blend_output_animation_frame(
    source_image: np.ndarray,
    target_image: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    source = _image_to_float_frame(source_image)
    target = _image_to_float_frame(target_image)
    return _coerce_output_animation_frame(
        (1.0 - float(alpha)) * source + float(alpha) * target,
        reference_image=target_image,
    )


def _layer_world_size(layer: NapariImageLayer) -> tuple[float, float]:
    data = np.asarray(layer.data)
    if data.ndim < 2:
        return 1.0, 1.0

    scale = getattr(layer, 'scale', (1.0, 1.0))
    if isinstance(scale, (int, float)):
        scale_y = scale_x = float(scale)
    else:
        scale_values = tuple(scale)
        scale_y = float(scale_values[0]) if len(scale_values) > 0 else 1.0
        scale_x = float(scale_values[1]) if len(scale_values) > 1 else scale_y
    height, width = data.shape[:2]
    return float(height) * scale_y, float(width) * scale_x


def set_output_layer_metadata(
    layer: NapariImageLayer,
    *,
    float_image: np.ndarray,
    output_color_space: str,
    output_cctf_encoding: bool,
    use_display_transform: bool,
    output_float_data_key: str,
    output_color_space_key: str,
    output_cctf_encoding_key: str,
    output_display_transform_key: str,
) -> None:
    layer.metadata[output_float_data_key] = np.asarray(float_image, dtype=np.float32)
    layer.metadata[output_color_space_key] = output_color_space
    layer.metadata[output_cctf_encoding_key] = output_cctf_encoding
    layer.metadata[output_display_transform_key] = use_display_transform


def set_output_layer_interpolation(layer: NapariImageLayer, mode: str) -> None:
    try:
        setattr(layer, 'interpolation2d', mode)
        _refresh_layer(layer)
        return
    except (AttributeError, TypeError, ValueError):
        pass

    try:
        setattr(layer, 'interpolation', mode)
        _refresh_layer(layer)
    except (AttributeError, TypeError, ValueError):
        return


def _refresh_layer(layer: NapariImageLayer) -> None:
    refresh = getattr(layer, 'refresh', None)
    if callable(refresh):
        refresh()


@dataclass(slots=True)
class ViewerLayerService:
    viewer: Any
    output_float_data_key: str
    output_color_space_key: str
    output_cctf_encoding_key: str
    output_display_transform_key: str
    _output_animations: dict[int, _LayerAnimationHandle] = field(default_factory=dict, init=False, repr=False)

    def image_layer(self, layer_name: str) -> NapariImageLayer | None:
        return next(
            (
                layer
                for layer in self.viewer.layers
                if is_napari_image_layer(layer) and getattr(layer, 'name', None) == layer_name
            ),
            None,
        )

    def preview_input_layer(self) -> NapariImageLayer | None:
        return self.image_layer(INPUT_PREVIEW_LAYER_NAME)

    def white_border_layer(self) -> NapariImageLayer | None:
        return self.image_layer(WHITE_BORDER_LAYER_NAME)

    def set_or_add_input_preview_layer(
        self,
        image: np.ndarray,
        *,
        white_padding: float,
        hide_output: bool = True,
        set_active: bool = True,
    ) -> None:
        image_world_size = _normalized_world_size(image)
        border_world_size = _padded_world_size(image_world_size, white_padding)

        if hide_output:
            self.hide_layer(OUTPUT_LAYER_NAME)

        white_border = self._set_or_add_image_layer(
            np.ones((*np.asarray(image).shape[:2], 3), dtype=np.float32),
            layer_name=WHITE_BORDER_LAYER_NAME,
        )
        _set_layer_geometry(white_border, world_size=border_world_size)

        preview_layer = self._set_or_add_image_layer(np.asarray(image), layer_name=INPUT_PREVIEW_LAYER_NAME)
        _set_layer_geometry(preview_layer, world_size=image_world_size)
        if preview_layer.visible:
            preview_layer.visible = False

        self._ensure_stack_order()
        if set_active:
            self.set_active_layer(white_border)

    def sync_white_border(self, *, white_padding: float) -> None:
        white_border = self.white_border_layer()
        if white_border is None:
            return

        preview_layer = self.preview_input_layer()
        if preview_layer is None:
            return

        border_world_size = _padded_world_size(_layer_world_size(preview_layer), white_padding)
        _set_layer_geometry(white_border, world_size=border_world_size)

    def current_image_world_size(self) -> tuple[float, float] | None:
        layer = self.preview_input_layer()
        if layer is None:
            return None
        return _layer_world_size(layer)

    def set_or_add_output_layer(
        self,
        image: np.ndarray,
        *,
        float_image: np.ndarray,
        output_color_space: str,
        output_cctf_encoding: bool,
        use_display_transform: bool,
        output_interpolation_mode: str = 'spline36',
    ) -> None:
        existing_layer = self.image_layer(OUTPUT_LAYER_NAME)
        animate_on_show = existing_layer is None or not bool(getattr(existing_layer, 'visible', True))
        output_image = np.asarray(image)
        crossfade_source_image: np.ndarray | None = None
        use_crossfade = False
        replace_layer = False
        if existing_layer is not None and not animate_on_show:
            crossfade_source_image = np.array(existing_layer.data, copy=True)
            self._stop_output_layer_animation(existing_layer, restore_final=False)
            use_crossfade = _supports_output_layer_crossfade(crossfade_source_image, output_image)
            replace_layer = not use_crossfade and _layer_data_shape(existing_layer) != tuple(int(dimension) for dimension in output_image.shape)

        if replace_layer and existing_layer is not None:
            self.remove_layer(OUTPUT_LAYER_NAME)
            existing_layer = None

        if existing_layer is None:
            layer = self._set_or_add_image_layer(output_image, layer_name=OUTPUT_LAYER_NAME)
        elif use_crossfade:
            layer = existing_layer
        else:
            layer = self._set_or_add_image_layer(output_image, layer_name=OUTPUT_LAYER_NAME)

        set_output_layer_metadata(
            layer,
            float_image=float_image,
            output_color_space=output_color_space,
            output_cctf_encoding=output_cctf_encoding,
            use_display_transform=use_display_transform,
            output_float_data_key=self.output_float_data_key,
            output_color_space_key=self.output_color_space_key,
            output_cctf_encoding_key=self.output_cctf_encoding_key,
            output_display_transform_key=self.output_display_transform_key,
        )
        set_output_layer_interpolation(layer, output_interpolation_mode)
        image_world_size = self.current_image_world_size()
        if image_world_size is not None:
            _set_layer_geometry(layer, world_size=_fit_image_world_size(np.asarray(image), bounding_world_size=image_world_size))
        self.hide_layer(INPUT_PREVIEW_LAYER_NAME)
        if not layer.visible:
            layer.visible = True
        self.move_layer_to_top(layer)
        self.set_active_layer(layer)
        if animate_on_show:
            self._start_output_layer_animation(layer, image=output_image)
        elif use_crossfade and crossfade_source_image is not None:
            self._start_output_layer_crossfade(
                layer,
                source_image=crossfade_source_image,
                target_image=output_image,
            )
        else:
            _refresh_layer(layer)

    def output_layer(self) -> NapariImageLayer | None:
        layer = self.image_layer(OUTPUT_LAYER_NAME)
        if layer is None or not layer.visible:
            return None
        return layer

    @staticmethod
    def set_output_layer_interpolation(layer: NapariImageLayer, mode: str) -> None:
        set_output_layer_interpolation(layer, mode)

    def hide_layer(self, layer_name: str) -> None:
        layer = self.image_layer(layer_name)
        if layer is None or not layer.visible:
            return
        self._stop_output_layer_animation(layer)
        layer.visible = False

    def remove_layer(self, layer_name: str) -> None:
        layer = self.image_layer(layer_name)
        if layer is None:
            return
        self._stop_output_layer_animation(layer)
        try:
            self.viewer.layers.remove(layer)
        except ValueError:
            return

    def move_layer_to_top(self, layer: NapariImageLayer) -> None:
        current_index = self.viewer.layers.index(layer)
        top_index = len(self.viewer.layers)
        if current_index != top_index - 1:
            self.viewer.layers.move(current_index, top_index)

    def set_active_layer(self, layer: NapariImageLayer | None) -> None:
        if layer is None:
            return
        selection = getattr(self.viewer.layers, 'selection', None)
        if selection is not None and hasattr(selection, 'active'):
            selection.active = layer

    def _set_or_add_image_layer(self, image: np.ndarray, *, layer_name: str) -> NapariImageLayer:
        existing_layer = self.image_layer(layer_name)
        if existing_layer is None:
            return self.viewer.add_image(image, name=layer_name)
        existing_layer.data = image
        return existing_layer

    def _ensure_stack_order(self) -> None:
        stack_layers = [
            layer
            for layer_name in STACK_LAYER_ORDER
            if (layer := self.image_layer(layer_name)) is not None
        ]
        if not stack_layers:
            return
        if list(self.viewer.layers[-len(stack_layers) :]) == stack_layers:
            return
        for layer in stack_layers:
            self.move_layer_to_top(layer)

    def _start_output_layer_animation(
        self,
        layer: NapariImageLayer,
        *,
        image: np.ndarray,
        duration_ms: int = OUTPUT_LAYER_ANIMATION_DURATION_MS,
        interval_ms: int = OUTPUT_LAYER_ANIMATION_INTERVAL_MS,
    ) -> None:
        output_image = np.array(image, copy=True)
        if not _supports_output_layer_animation(output_image):
            return

        timer_cls = QTimer
        if timer_cls is None:
            return

        self._stop_output_layer_animation(layer, restore_final=False)
        safe_interval_ms = max(int(interval_ms), 1)
        frame_count = max(int(np.ceil(float(duration_ms) / float(safe_interval_ms))), 2)
        frame_times = np.linspace(0.0, 1.0, num=frame_count, dtype=np.float32)
        state = prepare_polaroid_state(output_image)
        timer = timer_cls()
        set_interval = getattr(timer, 'setInterval', None)
        if callable(set_interval):
            set_interval(safe_interval_ms)
        set_single_shot = getattr(timer, 'setSingleShot', None)
        if callable(set_single_shot):
            set_single_shot(False)
        timeout_signal = getattr(timer, 'timeout', None)
        connect = getattr(timeout_signal, 'connect', None)
        start = getattr(timer, 'start', None)
        if not callable(connect) or not callable(start):
            return

        current_index = 0

        def _tick() -> None:
            nonlocal current_index
            current_index += 1
            if current_index >= len(frame_times):
                self._stop_output_layer_animation(layer)
                return
            layer.data = _coerce_output_animation_frame(
                render_polaroid_frame(state, float(frame_times[current_index])),
                reference_image=output_image,
            )
            if current_index >= len(frame_times) - 1:
                self._stop_output_layer_animation(layer)

        connect(_tick)
        self._output_animations[id(layer)] = _LayerAnimationHandle(
            timer=timer,
            final_image=output_image,
        )
        layer.data = _coerce_output_animation_frame(
            render_polaroid_frame(state, float(frame_times[0])),
            reference_image=output_image,
        )
        start()

    def _start_output_layer_crossfade(
        self,
        layer: NapariImageLayer,
        *,
        source_image: np.ndarray,
        target_image: np.ndarray,
        frame_count: int = OUTPUT_LAYER_CROSSFADE_FRAMES,
        interval_ms: int = OUTPUT_LAYER_ANIMATION_INTERVAL_MS,
    ) -> None:
        safe_frame_count = max(int(frame_count), 1)
        safe_interval_ms = max(int(interval_ms), 1)
        timer_cls = QTimer
        if timer_cls is None:
            layer.data = np.array(target_image, copy=True)
            return

        self._stop_output_layer_animation(layer, restore_final=False)
        timer = timer_cls()
        set_interval = getattr(timer, 'setInterval', None)
        if callable(set_interval):
            set_interval(safe_interval_ms)
        set_single_shot = getattr(timer, 'setSingleShot', None)
        if callable(set_single_shot):
            set_single_shot(False)
        timeout_signal = getattr(timer, 'timeout', None)
        connect = getattr(timeout_signal, 'connect', None)
        start = getattr(timer, 'start', None)
        if not callable(connect) or not callable(start):
            layer.data = np.array(target_image, copy=True)
            return

        current_step = 0
        source_frame = np.array(source_image, copy=True)
        final_image = np.array(target_image, copy=True)

        def _tick() -> None:
            nonlocal current_step
            current_step += 1
            if current_step >= safe_frame_count:
                self._stop_output_layer_animation(layer)
                return
            layer.data = _blend_output_animation_frame(
                source_frame,
                final_image,
                alpha=float(current_step) / float(safe_frame_count),
            )

        connect(_tick)
        self._output_animations[id(layer)] = _LayerAnimationHandle(
            timer=timer,
            final_image=final_image,
        )
        start()

    def _stop_output_layer_animation(
        self,
        layer: NapariImageLayer,
        *,
        restore_final: bool = True,
    ) -> None:
        handle = self._output_animations.pop(id(layer), None)
        if handle is None:
            return
        timer = handle.timer
        stop = getattr(timer, 'stop', None)
        if callable(stop):
            stop()
        delete_later = getattr(timer, 'deleteLater', None)
        if callable(delete_later):
            delete_later()
        if restore_final:
            layer.data = np.array(handle.final_image, copy=True)
            _refresh_layer(layer)

    def output_layer_float_data(self) -> np.ndarray | None:
        output_layer = self.output_layer()
        if output_layer is None:
            return None
        float_data = output_layer.metadata.get(self.output_float_data_key)
        if float_data is None:
            return None
        return np.asarray(float_data)

    def output_layer_render_settings(
        self,
        *,
        default_color_space: str,
        default_cctf_encoding: bool,
    ) -> tuple[str, bool]:
        output_layer = self.output_layer()
        if output_layer is None:
            return default_color_space, default_cctf_encoding
        color_space = output_layer.metadata.get(self.output_color_space_key, default_color_space)
        cctf_encoding = output_layer.metadata.get(self.output_cctf_encoding_key, default_cctf_encoding)
        return str(color_space), bool(cctf_encoding)