from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import TYPE_CHECKING

import colour
import numpy as np
from PIL import Image as PILImage
from PIL import ImageCms
from qtpy import QtCore
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
INPUT_RAW_DATA_KEY = 'input_raw_data'
INPUT_PADDING_PIXELS_KEY = 'input_display_padding_pixels'
DISPLAY_PREVIEW_COLOR_SPACE = 'sRGB'

if TYPE_CHECKING:
    import napari
    from napari.layers import Image as NapariImageLayer


QObject = QtCore.QObject
QRunnable = QtCore.QRunnable
QThreadPool = QtCore.QThreadPool
Signal = QtCore.Signal


@dataclass(slots=True)
class SimulationRequest:
    mode_label: str
    image: np.ndarray
    params: object
    output_color_space: str
    use_display_transform: bool
    padding_pixels: int


@dataclass(slots=True)
class SimulationResult:
    mode_label: str
    display_image: np.ndarray
    float_image: np.ndarray
    output_color_space: str
    use_display_transform: bool
    status_message: str


class _SimulationWorkerSignals(QObject):
    finished = Signal(object)
    failed = Signal(str)


class _SimulationWorker(QRunnable):
    def __init__(self, request: SimulationRequest):
        super().__init__()
        self._request = request
        self.signals = _SimulationWorkerSignals()

    def run(self) -> None:
        try:
            result = GuiController._execute_simulation_request(self._request)
        except Exception as exc:  # noqa: BLE001 - surface unexpected pipeline failures to the UI
            self.signals.failed.emit(f'{type(exc).__name__}: {exc}')
            return
        self.signals.finished.emit(result)


def _is_napari_image_layer(layer: object) -> bool:
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


class GuiController:
    def __init__(self, *, viewer: napari.Viewer, widgets: GuiWidgets):
        self._viewer = viewer
        self._widgets = widgets
        self._thread_pool = QThreadPool.globalInstance()
        self._active_simulation_worker: _SimulationWorker | None = None
        self._active_simulation_label: str | None = None

    def refresh_input_layers(self, *, selected_name: str | None = None) -> None:
        self._widgets.filepicker.set_available_layers(
            [layer.name for layer in self._available_input_layers()],
            selected_name=selected_name,
        )

    def load_input_image(self, path: str) -> None:
        image = load_image_oiio(path)[..., :3]
        gui_state = collect_gui_state(widgets=self._widgets)
        padding_pixels = self._padding_pixels_for_image(image, gui_state.display.white_padding)
        display_image = self._apply_white_padding(image, padding_pixels)
        layer_name = Path(path).stem
        existing_layer = next((layer for layer in self._available_input_layers() if layer.name == layer_name), None)
        if existing_layer is None:
            layer = self._viewer.add_image(display_image, name=layer_name)
            self._set_input_layer_metadata(layer, raw_image=image, padding_pixels=padding_pixels)
        else:
            existing_layer.data = display_image
            self._set_input_layer_metadata(existing_layer, raw_image=image, padding_pixels=padding_pixels)
            layer = existing_layer
        self._move_layer_to_top(layer)
        self._show_only_layer(layer)
        self.refresh_input_layers(selected_name=layer_name)

    def select_input_layer(self, layer_name: str) -> None:
        if not layer_name:
            return
        layer = next((item for item in self._available_input_layers() if item.name == layer_name), None)
        if layer is None:
            return
        self._move_layer_to_top(layer)
        self._show_only_layer(layer)

    def run_preview(self) -> None:
        self._start_simulation(compute_full_image=False, mode_label='Preview')

    def run_scan(self) -> None:
        self._start_simulation(compute_full_image=True, mode_label='Scan')

    def report_display_transform_status(self, enabled: bool) -> None:
        if enabled and not self.sync_display_transform_availability(report_status=True):
            return
        set_status(self._viewer, self._display_transform_status_message(enabled))

    def sync_display_transform_availability(self, *, report_status: bool) -> bool:
        if self._display_profile_available():
            return True

        self._set_display_transform_checked(False)
        if report_status:
            set_status(self._viewer, 'Display transform unavailable: no display profile detected, disabled')
        return False

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
        layer_name = self._widgets.filepicker.selected_input_layer_name()
        if not layer_name:
            return None
        for layer in self._available_input_layers():
            if layer.name == layer_name:
                return layer
        return None

    @staticmethod
    def _set_input_layer_metadata(
        layer: NapariImageLayer,
        *,
        raw_image: np.ndarray,
        padding_pixels: float,
    ) -> None:
        layer.metadata[INPUT_RAW_DATA_KEY] = np.asarray(raw_image)
        layer.metadata[INPUT_PADDING_PIXELS_KEY] = float(padding_pixels)

    @staticmethod
    def _processing_input_image(layer: NapariImageLayer) -> np.ndarray:
        metadata = getattr(layer, 'metadata', None)
        if not isinstance(metadata, dict):
            return np.asarray(layer.data)[..., :3]
        raw_image = metadata.get(INPUT_RAW_DATA_KEY)
        if raw_image is None:
            return np.asarray(layer.data)[..., :3]
        return np.asarray(raw_image)[..., :3]

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

    @staticmethod
    def _apply_white_padding(image_data: np.ndarray, padding_pixels: float) -> np.ndarray:
        padding = max(0, int(round(padding_pixels)))
        if padding == 0:
            return np.asarray(image_data)

        image = np.asarray(image_data)
        if image.ndim < 2:
            return image

        if np.issubdtype(image.dtype, np.integer):
            fill_value = np.iinfo(image.dtype).max
        else:
            fill_value = 1.0

        pad_width = [(padding, padding), (padding, padding)]
        pad_width.extend((0, 0) for _ in range(image.ndim - 2))
        return np.pad(image, pad_width, mode='constant', constant_values=fill_value)

    @staticmethod
    def _padding_pixels_for_image(image_data: np.ndarray, padding_fraction: float) -> int:
        image = np.asarray(image_data)
        if image.ndim < 2:
            return 0

        padding_fraction = max(0.0, float(padding_fraction))
        long_edge = max(int(image.shape[0]), int(image.shape[1]))
        return int(np.floor(long_edge * padding_fraction))

    @staticmethod
    def _prepare_output_display_image(
        image_data: np.ndarray,
        *,
        output_color_space: str,
        use_display_transform: bool,
        padding_pixels: float = 0.0,
    ) -> tuple[np.ndarray, str]:
        normalized_image = GuiController._normalized_image_data(np.asarray(image_data)[..., :3])
        preview_image = np.uint8(np.clip(normalized_image, 0.0, 1.0) * 255)
        if not use_display_transform:
            return GuiController._apply_white_padding(preview_image, padding_pixels), GuiController._display_transform_status_message(False)
        try:
            transformed_image, status = GuiController._apply_display_transform(normalized_image, output_color_space=output_color_space)
            return GuiController._apply_white_padding(transformed_image, padding_pixels), status
        except (OSError, ValueError, TypeError, ImageCms.PyCMSError):
            return GuiController._apply_white_padding(preview_image, padding_pixels), 'Display transform: transform failed, using raw preview'

    @staticmethod
    def _display_transform_status_message(enabled: bool) -> str:
        if not enabled:
            return 'Display transform: disabled'
        display_profile = ImageCms.get_display_profile()
        if display_profile is None:
            return 'Display transform: no display profile, using raw preview'
        return 'Display transform: display profile found'

    @staticmethod
    def _display_profile_available() -> bool:
        try:
            return ImageCms.get_display_profile() is not None
        except (OSError, ValueError, TypeError, ImageCms.PyCMSError):
            return False

    def _set_display_transform_checked(self, enabled: bool) -> None:
        display_section = getattr(self._widgets, 'display', None)
        toggle = getattr(display_section, 'use_display_transform', None)
        if toggle is None:
            return

        block_signals = getattr(toggle, 'blockSignals', None)
        set_checked = getattr(toggle, 'setChecked', None)
        if not callable(set_checked):
            return

        previous_block_state = None
        if callable(block_signals):
            previous_block_state = block_signals(True)
        try:
            set_checked(enabled)
        finally:
            if callable(block_signals):
                block_signals(bool(previous_block_state))

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

    @staticmethod
    def _execute_simulation_request(request: SimulationRequest) -> SimulationResult:
        scan = photo_process(request.image, request.params)
        scan_display, display_status = GuiController._prepare_output_display_image(
            scan,
            output_color_space=request.output_color_space,
            use_display_transform=request.use_display_transform,
            padding_pixels=request.padding_pixels,
        )
        return SimulationResult(
            mode_label=request.mode_label,
            display_image=scan_display,
            float_image=np.asarray(scan),
            output_color_space=request.output_color_space,
            use_display_transform=request.use_display_transform,
            status_message=display_status,
        )

    def _start_simulation(self, *, compute_full_image: bool, mode_label: str) -> None:
        if self._active_simulation_worker is not None:
            set_status(self._viewer, 'Simulation already running')
            return

        input_layer = self._selected_input_layer()
        if input_layer is None:
            QMessageBox.warning(dialog_parent(self._viewer), 'Run simulation', 'Select an input image layer before running the simulation.')
            return

        state = collect_gui_state(widgets=self._widgets)
        state.simulation.compute_full_image = compute_full_image
        params = build_params_from_state(state)

        image = np.double(self._processing_input_image(input_layer))
        request = SimulationRequest(
            mode_label=mode_label,
            image=image,
            params=params,
            output_color_space=state.simulation.output_color_space,
            use_display_transform=state.display.use_display_transform,
            padding_pixels=self._padding_pixels_for_image(image, state.display.white_padding),
        )

        worker = _SimulationWorker(request)
        worker.signals.finished.connect(self._on_simulation_finished)
        worker.signals.failed.connect(self._on_simulation_failed)
        self._active_simulation_worker = worker
        self._active_simulation_label = mode_label
        self._set_simulation_controls_enabled(False)
        set_status(self._viewer, f'Computing {mode_label.lower()}...', timeout_ms=0)
        self._thread_pool.start(worker)

    def _on_simulation_finished(self, result: SimulationResult) -> None:
        self._active_simulation_worker = None
        self._active_simulation_label = None
        self._set_simulation_controls_enabled(True)
        self._set_or_add_output_layer(
            result.display_image,
            float_image=result.float_image,
            output_color_space=result.output_color_space,
            output_cctf_encoding=True,
            use_display_transform=result.use_display_transform,
        )
        set_status(self._viewer, f'{result.mode_label} completed. {result.status_message}')

    def _on_simulation_failed(self, message: str) -> None:
        self._active_simulation_worker = None
        mode_label = self._active_simulation_label or 'Simulation'
        self._active_simulation_label = None
        self._set_simulation_controls_enabled(True)
        QMessageBox.critical(dialog_parent(self._viewer), 'Run simulation', f'Simulation failed.\n\n{message}')
        set_status(self._viewer, f'{mode_label} failed')

    def _set_simulation_controls_enabled(self, enabled: bool) -> None:
        simulation_section = getattr(self._widgets, 'simulation', None)
        if simulation_section is None:
            return
        for button_name in ('preview_button', 'scan_button', 'save_button'):
            button = getattr(simulation_section, button_name, None)
            set_enabled = getattr(button, 'setEnabled', None)
            if callable(set_enabled):
                set_enabled(enabled)

    def _run_simulation(self, *, compute_full_image: bool) -> None:
        input_layer = self._selected_input_layer()
        if input_layer is None:
            QMessageBox.warning(dialog_parent(self._viewer), 'Run simulation', 'Select an input image layer before running the simulation.')
            return

        state = collect_gui_state(widgets=self._widgets)
        state.simulation.compute_full_image = compute_full_image
        params = build_params_from_state(state)

        image = np.double(self._processing_input_image(input_layer))
        padding_pixels = self._padding_pixels_for_image(image, state.display.white_padding)
        scan = photo_process(image, params)
        scan_display, display_status = self._prepare_output_display_image(
            scan,
            output_color_space=state.simulation.output_color_space,
            use_display_transform=state.display.use_display_transform,
            padding_pixels=padding_pixels,
        )
        self._set_or_add_output_layer(
            scan_display,
            float_image=scan,
            output_color_space=state.simulation.output_color_space,
            output_cctf_encoding=True,
            use_display_transform=state.display.use_display_transform,
        )
        set_status(self._viewer, display_status)