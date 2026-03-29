from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from qtpy import QtCore, QtWidgets

from spektrafilm_gui import controller_persistence as persistence_actions
from spektrafilm_gui import controller_runtime as runtime
from spektrafilm_gui.controller_layers import (
    ViewerLayerService,
    processing_input_image,
    set_input_layer_metadata,
    set_output_layer_metadata,
)
from spektrafilm_gui.persistence import (
    clear_saved_default_gui_state,
    load_gui_state_from_path,
    save_default_gui_state,
    save_gui_state_to_path,
)
from spektrafilm_gui.state import PROJECT_DEFAULT_GUI_STATE
from spektrafilm_gui.napari_layout import dialog_parent, set_canvas_background, set_status
from spektrafilm_gui.params_mapper import build_params_from_state
from spektrafilm_gui.state_bridge import apply_gui_state, collect_gui_state
from spektrafilm_gui.widgets import WidgetBundle

OUTPUT_FLOAT_DATA_KEY = 'pipeline_float_output'
OUTPUT_COLOR_SPACE_KEY = 'pipeline_output_color_space'
OUTPUT_CCTF_ENCODING_KEY = 'pipeline_output_cctf_encoding'
OUTPUT_DISPLAY_TRANSFORM_KEY = 'pipeline_use_display_transform'
INPUT_RAW_DATA_KEY = 'input_raw_data'
INPUT_PADDING_PIXELS_KEY = 'input_display_padding_pixels'
if TYPE_CHECKING:
    import napari
    from napari.layers import Image as NapariImageLayer


QThreadPool = getattr(QtCore, 'QThreadPool')
QFileDialog = QtWidgets.QFileDialog
QMessageBox = QtWidgets.QMessageBox
SimulationRequest = runtime.SimulationRequest
SimulationResult = runtime.SimulationResult


class _LazyModuleProxy:
    def __init__(self, loader):
        self._loader = loader
        self._module = None

    def _load(self):
        if self._module is None:
            self._module = self._loader()
        return self._module

    def __getattr__(self, name: str):
        return getattr(self._load(), name)


def _import_colour_module():
    return import_module('colour')


def _import_pil_image_module():
    return import_module('PIL.Image')


def _import_imagecms_module():
    return import_module('PIL.ImageCms')


def simulate(*args, **kwargs):
    return import_module('spektrafilm.runtime.api').simulate(*args, **kwargs)


def load_image_oiio(*args, **kwargs):
    return import_module('spektrafilm.utils.io').load_image_oiio(*args, **kwargs)


def save_image_oiio(*args, **kwargs):
    return import_module('spektrafilm.utils.io').save_image_oiio(*args, **kwargs)


def load_and_process_raw_file(*args, **kwargs):
    return import_module('spektrafilm.utils.raw_file_processor').load_and_process_raw_file(*args, **kwargs)


colour = _LazyModuleProxy(_import_colour_module)
PILImage = _LazyModuleProxy(_import_pil_image_module)
ImageCms = _LazyModuleProxy(_import_imagecms_module)


class GuiController:
    def __init__(self, *, viewer: napari.Viewer, widgets: WidgetBundle):
        self._viewer = viewer
        self._widgets = widgets
        self._layers = ViewerLayerService(
            viewer=viewer,
            input_raw_data_key=INPUT_RAW_DATA_KEY,
            input_padding_pixels_key=INPUT_PADDING_PIXELS_KEY,
            output_float_data_key=OUTPUT_FLOAT_DATA_KEY,
            output_color_space_key=OUTPUT_COLOR_SPACE_KEY,
            output_cctf_encoding_key=OUTPUT_CCTF_ENCODING_KEY,
            output_display_transform_key=OUTPUT_DISPLAY_TRANSFORM_KEY,
        )
        self._thread_pool = QThreadPool.globalInstance()
        self._active_simulation_worker: runtime.SimulationWorker | None = None
        self._active_simulation_label: str | None = None

    def refresh_input_layers(self, *, selected_name: str | None = None) -> None:
        self._widgets.filepicker.set_available_layers(
            [layer.name for layer in self._available_input_layers()],
            selected_name=selected_name,
        )

    def load_input_image(self, path: str) -> None:
        image = load_image_oiio(path)[..., :3]
        gui_state = collect_gui_state(widgets=self._widgets)
        self._set_or_add_input_layer(image, layer_name=Path(path).stem, white_padding=gui_state.display.white_padding)

    def load_raw_image(self, path: str) -> None:
        gui_state = collect_gui_state(widgets=self._widgets)
        set_status(self._viewer, 'Loading raw...', timeout_ms=0)
        try:
            image = load_and_process_raw_file(
                path,
                white_balance=gui_state.load_raw.white_balance,
                temperature=gui_state.load_raw.temperature,
                tint=gui_state.load_raw.tint,
                output_colorspace=gui_state.input_image.input_color_space,
                output_cctf_encoding=gui_state.input_image.apply_cctf_decoding,
            )
        except (OSError, ValueError) as exc:
            QMessageBox.critical(dialog_parent(self._viewer), 'Load raw', f'Failed to load RAW image.\n\n{exc}')
            set_status(self._viewer, 'Load raw failed')
            return
        self._set_or_add_input_layer(image, layer_name=Path(path).stem, white_padding=gui_state.display.white_padding)

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

    def set_gray_18_canvas_enabled(self, enabled: bool) -> None:
        set_canvas_background(self._viewer, gray_18_canvas=enabled)

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
        persistence_actions.save_current_as_default(
            viewer=self._viewer,
            widgets=self._widgets,
            collect_gui_state_fn=collect_gui_state,
            save_default_gui_state_fn=save_default_gui_state,
            set_status_fn=set_status,
            dialog_parent_fn=dialog_parent,
            message_box=QMessageBox,
        )

    def save_current_state_to_file(self) -> None:
        persistence_actions.save_current_state_to_file(
            viewer=self._viewer,
            widgets=self._widgets,
            file_dialog=QFileDialog,
            collect_gui_state_fn=collect_gui_state,
            save_gui_state_to_path_fn=save_gui_state_to_path,
            set_status_fn=set_status,
            dialog_parent_fn=dialog_parent,
            message_box=QMessageBox,
        )

    def load_state_from_file(self) -> None:
        persistence_actions.load_state_from_file(
            viewer=self._viewer,
            widgets=self._widgets,
            file_dialog=QFileDialog,
            load_gui_state_from_path_fn=load_gui_state_from_path,
            apply_gui_state_fn=apply_gui_state,
            sync_canvas_background_fn=self._sync_canvas_background,
            set_status_fn=set_status,
            dialog_parent_fn=dialog_parent,
            message_box=QMessageBox,
        )

    def restore_factory_default(self) -> None:
        persistence_actions.restore_factory_default(
            viewer=self._viewer,
            widgets=self._widgets,
            project_default_gui_state=PROJECT_DEFAULT_GUI_STATE,
            clear_saved_default_gui_state_fn=clear_saved_default_gui_state,
            apply_gui_state_fn=apply_gui_state,
            sync_canvas_background_fn=self._sync_canvas_background,
            set_status_fn=set_status,
            dialog_parent_fn=dialog_parent,
            message_box=QMessageBox,
        )

    def _available_input_layers(self) -> list[NapariImageLayer]:
        return self._layers.available_input_layers()

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
        set_input_layer_metadata(
            layer,
            raw_image=raw_image,
            padding_pixels=padding_pixels,
            input_raw_data_key=INPUT_RAW_DATA_KEY,
            input_padding_pixels_key=INPUT_PADDING_PIXELS_KEY,
        )

    @staticmethod
    def _processing_input_image(layer: NapariImageLayer) -> np.ndarray:
        return processing_input_image(layer, input_raw_data_key=INPUT_RAW_DATA_KEY)

    def _set_or_add_output_layer(
        self,
        image: np.ndarray,
        *,
        float_image: np.ndarray,
        output_color_space: str,
        output_cctf_encoding: bool,
        use_display_transform: bool,
    ) -> None:
        self._layers.set_or_add_output_layer(
            image,
            float_image=float_image,
            output_color_space=output_color_space,
            output_cctf_encoding=output_cctf_encoding,
            use_display_transform=use_display_transform,
        )

    def _set_or_add_input_layer(
        self,
        image: np.ndarray,
        *,
        layer_name: str,
        white_padding: float,
    ) -> None:
        self._layers.set_or_add_input_layer(
            image,
            layer_name=layer_name,
            white_padding=white_padding,
            padding_pixels_for_image_fn=self._padding_pixels_for_image,
            apply_white_padding_fn=self._apply_white_padding,
            refresh_input_layers_fn=self.refresh_input_layers,
        )

    @staticmethod
    def _set_output_layer_metadata(
        layer: NapariImageLayer,
        *,
        float_image: np.ndarray,
        output_color_space: str,
        output_cctf_encoding: bool,
        use_display_transform: bool,
    ) -> None:
        set_output_layer_metadata(
            layer,
            float_image=float_image,
            output_color_space=output_color_space,
            output_cctf_encoding=output_cctf_encoding,
            use_display_transform=use_display_transform,
            output_float_data_key=OUTPUT_FLOAT_DATA_KEY,
            output_color_space_key=OUTPUT_COLOR_SPACE_KEY,
            output_cctf_encoding_key=OUTPUT_CCTF_ENCODING_KEY,
            output_display_transform_key=OUTPUT_DISPLAY_TRANSFORM_KEY,
        )

    def _output_layer(self) -> NapariImageLayer | None:
        return self._layers.output_layer()

    def _move_layer_to_top(self, layer: NapariImageLayer) -> None:
        self._layers.move_layer_to_top(layer)

    def _show_only_layer(self, target_layer: NapariImageLayer) -> None:
        self._layers.show_only_layer(target_layer)

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
        return runtime.normalized_image_data(image)

    @staticmethod
    def _apply_white_padding(image_data: np.ndarray, padding_pixels: float) -> np.ndarray:
        return runtime.apply_white_padding(image_data, padding_pixels)

    @staticmethod
    def _padding_pixels_for_image(image_data: np.ndarray, padding_fraction: float) -> int:
        return runtime.padding_pixels_for_image(image_data, padding_fraction)

    @staticmethod
    def _prepare_output_display_image(
        image_data: np.ndarray,
        *,
        output_color_space: str,
        use_display_transform: bool,
        padding_pixels: float = 0.0,
    ) -> tuple[np.ndarray, str]:
        return runtime.prepare_output_display_image(
            image_data,
            output_color_space=output_color_space,
            use_display_transform=use_display_transform,
            padding_pixels=padding_pixels,
            imagecms_module=ImageCms,
            colour_module=colour,
            pil_image_module=PILImage,
        )

    @staticmethod
    def _display_transform_status_message(enabled: bool) -> str:
        return runtime.display_transform_status_message(enabled, imagecms_module=ImageCms)

    @staticmethod
    def _display_profile_available() -> bool:
        return runtime.display_profile_available(imagecms_module=ImageCms)

    @staticmethod
    def _display_profile_details() -> tuple[object | None, str | None]:
        return runtime.display_profile_details(imagecms_module=ImageCms)

    @staticmethod
    def _display_profile_name(display_profile: object) -> str:
        return runtime.display_profile_name(display_profile, imagecms_module=ImageCms)

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

    def _sync_canvas_background(self) -> None:
        display_section = getattr(self._widgets, 'display', None)
        toggle = getattr(display_section, 'gray_18_canvas', None)
        is_checked = getattr(toggle, 'isChecked', None)
        self.set_gray_18_canvas_enabled(bool(is_checked()) if callable(is_checked) else False)

    @staticmethod
    def _apply_display_transform(image_data: np.ndarray, *, output_color_space: str) -> tuple[np.ndarray, str]:
        return runtime.apply_display_transform(
            image_data,
            output_color_space=output_color_space,
            colour_module=colour,
            imagecms_module=ImageCms,
            pil_image_module=PILImage,
        )

    def _execute_simulation_request(self, request: SimulationRequest) -> SimulationResult:
        return runtime.execute_simulation_request(
            request,
            simulate_fn=simulate,
            prepare_output_display_image_fn=self._prepare_output_display_image,
            padding_pixels_for_image_fn=self._padding_pixels_for_image,
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
            white_padding_fraction=float(state.display.white_padding),
        )

        worker = runtime.SimulationWorker(request, execute_request=self._execute_simulation_request)
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
        scan = simulate(image, params)
        padding_pixels = self._padding_pixels_for_image(scan, state.display.white_padding)
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