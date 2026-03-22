from __future__ import annotations

from pathlib import Path

import napari
import numpy as np
from napari.layers import Image
from qtpy.QtWidgets import QFileDialog, QMessageBox

from spectral_film_lab.gui.state_bridge import GuiWidgets, collect_gui_state
from spectral_film_lab.gui.napari_layout import dialog_parent, set_status
from spectral_film_lab.gui.params_mapper import build_params_from_state
from spectral_film_lab.runtime.process import photo_process
from spectral_film_lab.utils.io import load_image_oiio, save_image_oiio

OUTPUT_FLOAT_DATA_KEY = 'pipeline_float_output'


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

        float_image_data = self._output_layer_float_data()
        if float_image_data is None:
            image_data = self._normalized_image_data(np.asarray(output_layer.data)[..., :3])
        else:
            image_data = np.asarray(float_image_data)[..., :3]
        try:
            save_image_oiio(filepath, image_data)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(dialog_parent(self._viewer), 'Save output', f'Failed to save output image.\n\n{exc}')
            return

        set_status(self._viewer, f'Saved output image to {filepath}')

    def _available_input_layers(self) -> list[Image]:
        return [layer for layer in self._viewer.layers if isinstance(layer, Image)]

    def _selected_input_layer(self) -> Image | None:
        layer_name = self._widgets.simulation_input.selected_input_layer_name()
        if not layer_name:
            return None
        for layer in self._available_input_layers():
            if layer.name == layer_name:
                return layer
        return None

    def _set_or_add_output_layer(self, image: np.ndarray, *, float_image: np.ndarray) -> None:
        output_name = 'output'
        existing_layer = next((layer for layer in self._available_input_layers() if layer.name == output_name), None)
        if existing_layer is None:
            layer = self._viewer.add_image(image, name=output_name)
            layer.metadata[OUTPUT_FLOAT_DATA_KEY] = np.asarray(float_image, dtype=np.float32)
            self._move_layer_to_top(layer)
            self._show_only_layer(layer)
            return
        existing_layer.data = image
        existing_layer.metadata[OUTPUT_FLOAT_DATA_KEY] = np.asarray(float_image, dtype=np.float32)
        self._move_layer_to_top(existing_layer)
        self._show_only_layer(existing_layer)

    def _output_layer(self) -> Image | None:
        return next((layer for layer in self._available_input_layers() if layer.name == 'output'), None)

    def _move_layer_to_top(self, layer: Image) -> None:
        current_index = self._viewer.layers.index(layer)
        top_index = len(self._viewer.layers)
        if current_index != top_index - 1:
            self._viewer.layers.move(current_index, top_index)

    def _show_only_layer(self, target_layer: Image) -> None:
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
        scan_display = np.uint8(np.clip(scan, 0.0, 1.0) * 255)
        self._set_or_add_output_layer(scan_display, float_image=scan)