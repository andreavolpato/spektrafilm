from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

if TYPE_CHECKING:
    from napari.layers import Image as NapariImageLayer


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


def set_input_layer_metadata(
    layer: NapariImageLayer,
    *,
    raw_image: np.ndarray,
    padding_pixels: float,
    input_raw_data_key: str,
    input_padding_pixels_key: str,
) -> None:
    layer.metadata[input_raw_data_key] = np.asarray(raw_image)
    layer.metadata[input_padding_pixels_key] = float(padding_pixels)


def processing_input_image(layer: NapariImageLayer, *, input_raw_data_key: str) -> np.ndarray:
    metadata = getattr(layer, 'metadata', None)
    if not isinstance(metadata, dict):
        return np.asarray(layer.data)[..., :3]
    raw_image = metadata.get(input_raw_data_key)
    if raw_image is None:
        return np.asarray(layer.data)[..., :3]
    return np.asarray(raw_image)[..., :3]


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
    input_raw_data_key: str
    input_padding_pixels_key: str
    output_float_data_key: str
    output_color_space_key: str
    output_cctf_encoding_key: str
    output_display_transform_key: str

    def available_input_layers(self) -> list[NapariImageLayer]:
        return [layer for layer in self.viewer.layers if is_napari_image_layer(layer)]

    def selected_input_layer(self, layer_name: str | None) -> NapariImageLayer | None:
        if not layer_name:
            return None
        for layer in self.available_input_layers():
            if layer.name == layer_name:
                return layer
        return None

    def set_or_add_output_layer(
        self,
        image: np.ndarray,
        *,
        float_image: np.ndarray,
        output_color_space: str,
        output_cctf_encoding: bool,
        use_display_transform: bool,
    ) -> None:
        output_name = 'output'
        existing_layer = next((layer for layer in self.available_input_layers() if layer.name == output_name), None)
        if existing_layer is None:
            layer = self.viewer.add_image(image, name=output_name)
        else:
            existing_layer.data = image
            layer = existing_layer

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
        self.move_layer_to_top(layer)
        self.show_only_layer(layer)

    def set_or_add_input_layer(
        self,
        image: np.ndarray,
        *,
        layer_name: str,
        white_padding: float,
        padding_pixels_for_image_fn: Callable[[np.ndarray, float], int],
        apply_white_padding_fn: Callable[[np.ndarray, float], np.ndarray],
        refresh_input_layers_fn: Callable[..., None],
    ) -> None:
        padding_pixels = padding_pixels_for_image_fn(image, white_padding)
        display_image = apply_white_padding_fn(image, padding_pixels)
        existing_layer = next((layer for layer in self.available_input_layers() if layer.name == layer_name), None)
        if existing_layer is None:
            layer = self.viewer.add_image(display_image, name=layer_name)
        else:
            existing_layer.data = display_image
            layer = existing_layer
        set_input_layer_metadata(
            layer,
            raw_image=image,
            padding_pixels=padding_pixels,
            input_raw_data_key=self.input_raw_data_key,
            input_padding_pixels_key=self.input_padding_pixels_key,
        )
        self.move_layer_to_top(layer)
        self.show_only_layer(layer)
        refresh_input_layers_fn(selected_name=layer_name)

    def output_layer(self) -> NapariImageLayer | None:
        return next((layer for layer in self.available_input_layers() if layer.name == 'output'), None)

    def move_layer_to_top(self, layer: NapariImageLayer) -> None:
        current_index = self.viewer.layers.index(layer)
        top_index = len(self.viewer.layers)
        if current_index != top_index - 1:
            self.viewer.layers.move(current_index, top_index)

    def show_only_layer(self, target_layer: NapariImageLayer) -> None:
        for layer in self.viewer.layers:
            layer.visible = layer is target_layer

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