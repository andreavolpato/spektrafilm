from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from napari.layers import Image as NapariImageLayer


INPUT_LAYER_NAME = 'input'
INPUT_PREVIEW_LAYER_NAME = 'input_preview'
WHITE_BORDER_LAYER_NAME = 'white_border'
OUTPUT_LAYER_NAME = 'output'
STACK_LAYER_ORDER = (
    WHITE_BORDER_LAYER_NAME,
    INPUT_PREVIEW_LAYER_NAME,
    OUTPUT_LAYER_NAME,
)


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


@dataclass(slots=True)
class ViewerLayerService:
    viewer: Any
    output_float_data_key: str
    output_color_space_key: str
    output_cctf_encoding_key: str
    output_display_transform_key: str

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
    ) -> None:
        image_world_size = _normalized_world_size(image)
        border_world_size = _padded_world_size(image_world_size, white_padding)

        self.hide_layer(OUTPUT_LAYER_NAME)

        white_border = self._set_or_add_image_layer(
            np.ones((*np.asarray(image).shape[:2], 3), dtype=np.float32),
            layer_name=WHITE_BORDER_LAYER_NAME,
        )
        _set_layer_geometry(white_border, world_size=border_world_size)

        preview_layer = self._set_or_add_image_layer(np.asarray(image), layer_name=INPUT_PREVIEW_LAYER_NAME)
        _set_layer_geometry(preview_layer, world_size=image_world_size)

        self._ensure_stack_order()
        self.set_active_layer(preview_layer)

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
    ) -> None:
        layer = self._set_or_add_image_layer(np.asarray(image), layer_name=OUTPUT_LAYER_NAME)

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
        image_world_size = self.current_image_world_size()
        if image_world_size is not None:
            _set_layer_geometry(layer, world_size=image_world_size)
        if not layer.visible:
            layer.visible = True
        self.move_layer_to_top(layer)
        self.set_active_layer(layer)

    def output_layer(self) -> NapariImageLayer | None:
        layer = self.image_layer(OUTPUT_LAYER_NAME)
        if layer is None or not layer.visible:
            return None
        return layer

    def hide_layer(self, layer_name: str) -> None:
        layer = self.image_layer(layer_name)
        if layer is None or not layer.visible:
            return
        layer.visible = False

    def remove_layer(self, layer_name: str) -> None:
        layer = self.image_layer(layer_name)
        if layer is None:
            return
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