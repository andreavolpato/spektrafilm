from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import colour
import numpy as np
from PIL import Image as PILImage
from PIL import ImageCms
from qtpy.QtWidgets import QFileDialog, QMessageBox

from spektrafilm_gui.persistence import (
    clear_saved_default_gui_state,
    load_gui_state_from_path,
    save_default_gui_state,
    save_gui_state_to_path,
)
from spektrafilm_gui.state import PROJECT_DEFAULT_GUI_STATE
from spektrafilm_gui.state_bridge import GuiWidgets, apply_gui_state, collect_gui_state
from spektrafilm_gui.napari_layout import dialog_parent, set_status
from spektrafilm_gui.params_mapper import build_params_from_state
from spektrafilm.runtime.process import photo_process
from spektrafilm.utils.io import load_image_oiio, save_image_oiio

OUTPUT_FLOAT_DATA_KEY = 'pipeline_float_output'
OUTPUT_COLOR_SPACE_KEY = 'pipeline_output_color_space'
OUTPUT_CCTF_ENCODING_KEY = 'pipeline_output_cctf_encoding'
OUTPUT_DISPLAY_TRANSFORM_KEY = 'pipeline_use_display_transform'
DISPLAY_PREVIEW_COLOR_SPACE = 'sRGB'

if TYPE_CHECKING:
    import napari
    from napari.layers import Image as NapariImageLayer


def _is_napari_image_layer(layer: object) -> bool:
    try:
        from napari.layers import Image as NapariImageLayer
    except ImportError:
        return False
    return isinstance(layer, NapariImageLayer)


class GuiController:
    def __init__(self, *, viewer: napari.Viewer, widgets: GuiWidgets):
        self._viewer = viewer
        self._widgets = widgets

    def refresh_input_layers(self, *, selected_name: str | None = None) -> None:
        self._widgets.simulation_input.set_available_layers(
            [layer.name for layer in self._available_input_layers()],
            selected_name=selected_name,
        )

    def load_input_image(self, path: str) -> None:
        image = load_image_oiio(path)[..., :3]
        layer_name = Path(path).stem
        existing_layer = next((layer for layer in self._available_input_layers() if layer.name == layer_name), None)
        if existing_layer is None:
            self._viewer.add_image(image, name=layer_name)
        else:
            existing_layer.data = image
        self.refresh_input_layers(selected_name=layer_name)

    def run_preview(self) -> None:
        self._run_simulation(compute_full_image=False)

    def run_scan(self) -> None:
        self._run_simulation(compute_full_image=True)

    def report_display_transform_status(self, enabled: bool) -> None:
        set_status(self._viewer, self._display_transform_status_message(enabled))

    def save_output_layer(self) -> None:
        output_layer = self._output_layer()
        if output_layer is None:
            QMessageBox.warning(dialog_parent(self._viewer), 'Save output', 'Run a simulation before saving the output layer.')
            return

        filepath, _ = QFileDialog.getSaveFileName(
            dialog_parent(self._viewer),
            'Save output image',
            'output.png',
            'Images (*.png *.jpg *.jpeg *.exr)',
        )
        if not filepath:
            return

        gui_state = collect_gui_state(widgets=self._widgets)
        float_image_data = self._output_layer_float_data()
        if float_image_data is None:
            image_data = self._normalized_image_data(np.asarray(output_layer.data)[..., :3])
        else:
            image_data = np.asarray(float_image_data)[..., :3]

        source_color_space, source_cctf_encoding = self._output_layer_render_settings(
            default_color_space=gui_state.simulation.output_color_space,
            default_cctf_encoding=True,
        )
        saving_color_space = gui_state.simulation.saving_color_space
        saving_cctf_encoding = gui_state.simulation.saving_cctf_encoding
        if source_color_space != saving_color_space:
            image_data = colour.RGB_to_RGB(
                image_data,
                source_color_space,
                saving_color_space,
                apply_cctf_decoding=source_cctf_encoding,
                apply_cctf_encoding=saving_cctf_encoding,
            )
        elif source_cctf_encoding != saving_cctf_encoding:
            image_data = colour.RGB_to_RGB(
                image_data,
                source_color_space,
                saving_color_space,
                apply_cctf_decoding=source_cctf_encoding,
                apply_cctf_encoding=saving_cctf_encoding,
            )
        try:
            save_image_oiio(filepath, image_data)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(dialog_parent(self._viewer), 'Save output', f'Failed to save output image.\n\n{exc}')
            return

        set_status(self._viewer, f'Saved output image to {filepath}')

    def save_current_as_default(self) -> None:
        gui_state = collect_gui_state(widgets=self._widgets)
        try:
            save_default_gui_state(gui_state)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(dialog_parent(self._viewer), 'Save current as default', f'Failed to save default GUI state.\n\n{exc}')
            return

        set_status(self._viewer, 'Saved current GUI state as the startup default')

    def save_current_state_to_file(self) -> None:
        filepath, _ = QFileDialog.getSaveFileName(
            dialog_parent(self._viewer),
            'Save GUI state',
            'gui_state.json',
            'JSON (*.json)',
        )
        if not filepath:
            return

        gui_state = collect_gui_state(widgets=self._widgets)
        try:
            save_gui_state_to_path(gui_state, filepath)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(dialog_parent(self._viewer), 'Save GUI state', f'Failed to save GUI state.\n\n{exc}')
            return

        set_status(self._viewer, f'Saved GUI state to {filepath}')

    def load_state_from_file(self) -> None:
        filepath, _ = QFileDialog.getOpenFileName(
            dialog_parent(self._viewer),
            'Load GUI state',
            '',
            'JSON (*.json)',
        )
        if not filepath:
            return

        try:
            gui_state = load_gui_state_from_path(filepath)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            QMessageBox.critical(dialog_parent(self._viewer), 'Load GUI state', f'Failed to load GUI state.\n\n{exc}')
            return

        apply_gui_state(gui_state, widgets=self._widgets)
        set_status(self._viewer, f'Loaded GUI state from {filepath}')

    def restore_factory_default(self) -> None:
        try:
            clear_saved_default_gui_state()
        except OSError as exc:
            QMessageBox.critical(dialog_parent(self._viewer), 'Restore factory default', f'Failed to clear the saved startup default.\n\n{exc}')
            return

        apply_gui_state(PROJECT_DEFAULT_GUI_STATE, widgets=self._widgets)
        set_status(self._viewer, 'Restored factory default GUI state')

    def _available_input_layers(self) -> list[NapariImageLayer]:
        return [layer for layer in self._viewer.layers if _is_napari_image_layer(layer)]

    def _selected_input_layer(self) -> NapariImageLayer | None:
        layer_name = self._widgets.simulation_input.selected_input_layer_name()
        if not layer_name:
            return None
        for layer in self._available_input_layers():
            if layer.name == layer_name:
                return layer
        return None

    def _set_or_add_output_layer(
        self,
        image: np.ndarray,
        *,
        float_image: np.ndarray,
        output_color_space: str,
        output_cctf_encoding: bool,
        use_display_transform: bool,
    ) -> None:
        output_name = 'output'
        existing_layer = next((layer for layer in self._available_input_layers() if layer.name == output_name), None)
        if existing_layer is None:
            layer = self._viewer.add_image(image, name=output_name)
            self._set_output_layer_metadata(
                layer,
                float_image=float_image,
                output_color_space=output_color_space,
                output_cctf_encoding=output_cctf_encoding,
                use_display_transform=use_display_transform,
            )
            self._move_layer_to_top(layer)
            self._show_only_layer(layer)
            return
        existing_layer.data = image
        self._set_output_layer_metadata(
            existing_layer,
            float_image=float_image,
            output_color_space=output_color_space,
            output_cctf_encoding=output_cctf_encoding,
            use_display_transform=use_display_transform,
        )
        self._move_layer_to_top(existing_layer)
        self._show_only_layer(existing_layer)

    @staticmethod
    def _set_output_layer_metadata(
        layer: NapariImageLayer,
        *,
        float_image: np.ndarray,
        output_color_space: str,
        output_cctf_encoding: bool,
        use_display_transform: bool,
    ) -> None:
        layer.metadata[OUTPUT_FLOAT_DATA_KEY] = np.asarray(float_image, dtype=np.float32)
        layer.metadata[OUTPUT_COLOR_SPACE_KEY] = output_color_space
        layer.metadata[OUTPUT_CCTF_ENCODING_KEY] = output_cctf_encoding
        layer.metadata[OUTPUT_DISPLAY_TRANSFORM_KEY] = use_display_transform

    def _output_layer(self) -> NapariImageLayer | None:
        return next((layer for layer in self._available_input_layers() if layer.name == 'output'), None)

    def _move_layer_to_top(self, layer: NapariImageLayer) -> None:
        current_index = self._viewer.layers.index(layer)
        top_index = len(self._viewer.layers)
        if current_index != top_index - 1:
            self._viewer.layers.move(current_index, top_index)

    def _show_only_layer(self, target_layer: NapariImageLayer) -> None:
        for layer in self._viewer.layers:
            layer.visible = layer is target_layer

    def _output_layer_float_data(self) -> np.ndarray | None:
        output_layer = self._output_layer()
        if output_layer is None:
            return None
        float_data = output_layer.metadata.get(OUTPUT_FLOAT_DATA_KEY)
        if float_data is None:
            return None
        return np.asarray(float_data)

    def _output_layer_render_settings(
        self,
        *,
        default_color_space: str,
        default_cctf_encoding: bool,
    ) -> tuple[str, bool]:
        output_layer = self._output_layer()
        if output_layer is None:
            return default_color_space, default_cctf_encoding
        color_space = output_layer.metadata.get(OUTPUT_COLOR_SPACE_KEY, default_color_space)
        cctf_encoding = output_layer.metadata.get(OUTPUT_CCTF_ENCODING_KEY, default_cctf_encoding)
        return str(color_space), bool(cctf_encoding)

    @staticmethod
    def _normalized_image_data(image: np.ndarray) -> np.ndarray:
        if np.issubdtype(image.dtype, np.floating):
            return np.clip(image, 0.0, 1.0)
        if np.issubdtype(image.dtype, np.integer):
            max_value = np.iinfo(image.dtype).max
            if max_value == 0:
                return image.astype(np.float32)
            return image.astype(np.float32) / max_value
        return image.astype(np.float32)

    def _prepare_output_display_image(
        self,
        image_data: np.ndarray,
        *,
        output_color_space: str,
        use_display_transform: bool,
    ) -> tuple[np.ndarray, str]:
        normalized_image = self._normalized_image_data(np.asarray(image_data)[..., :3])
        preview_image = np.uint8(np.clip(normalized_image, 0.0, 1.0) * 255)
        if not use_display_transform:
            return preview_image, self._display_transform_status_message(False)
        try:
            return self._apply_display_transform(normalized_image, output_color_space=output_color_space)
        except (OSError, ValueError, TypeError, ImageCms.PyCMSError):
            return preview_image, 'Display transform: transform failed, using raw preview'

    @staticmethod
    def _display_transform_status_message(enabled: bool) -> str:
        if not enabled:
            return 'Display transform: disabled'
        display_profile = ImageCms.get_display_profile()
        if display_profile is None:
            return 'Display transform: no display profile, using raw preview'
        return 'Display transform: display profile found'

    @staticmethod
    def _apply_display_transform(image_data: np.ndarray, *, output_color_space: str) -> tuple[np.ndarray, str]:
        display_profile = ImageCms.get_display_profile()
        if display_profile is None:
            return np.uint8(np.clip(image_data, 0.0, 1.0) * 255), 'Display transform: no display profile, using raw preview'

        srgb_preview = colour.RGB_to_RGB(
            image_data,
            output_color_space,
            DISPLAY_PREVIEW_COLOR_SPACE,
            apply_cctf_decoding=True,
            apply_cctf_encoding=True,
        )
        srgb_preview_uint8 = np.uint8(np.clip(srgb_preview, 0.0, 1.0) * 255)
        source_profile = ImageCms.createProfile(DISPLAY_PREVIEW_COLOR_SPACE)
        source_image = PILImage.fromarray(srgb_preview_uint8, mode='RGB')
        transformed_image = ImageCms.profileToProfile(source_image, source_profile, display_profile, outputMode='RGB')
        return np.asarray(transformed_image, dtype=np.uint8), 'Display transform: active'

    def _run_simulation(self, *, compute_full_image: bool) -> None:
        input_layer = self._selected_input_layer()
        if input_layer is None:
            QMessageBox.warning(dialog_parent(self._viewer), 'Run simulation', 'Select an input image layer before running the simulation.')
            return

        state = collect_gui_state(widgets=self._widgets)
        state.simulation.compute_full_image = compute_full_image
        params = build_params_from_state(state)

        image = np.double(input_layer.data[:, :, :3])
        scan = photo_process(image, params)
        scan_display, display_status = self._prepare_output_display_image(
            scan,
            output_color_space=state.simulation.output_color_space,
            use_display_transform=state.simulation.use_display_transform,
        )
        self._set_or_add_output_layer(
            scan_display,
            float_image=scan,
            output_color_space=state.simulation.output_color_space,
            output_cctf_encoding=True,
            use_display_transform=state.simulation.use_display_transform,
        )