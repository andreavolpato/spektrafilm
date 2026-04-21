from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from qtpy import QtCore, QtWidgets

from spektrafilm_gui import controller_persistence as persistence_actions
from spektrafilm_gui import controller_profile_sync as profile_sync
from spektrafilm_gui import controller_runtime as runtime
from spektrafilm_gui.controller_layers import (
    INPUT_LAYER_NAME,
    INPUT_PREVIEW_LAYER_NAME,
    ViewerLayerService,
)
from spektrafilm_gui.persistence import (
    clear_saved_default_gui_state,
    load_dialog_dir,
    load_gui_state_from_path,
    save_default_gui_state,
    save_dialog_dir,
    save_gui_state_to_path,
)
from spektrafilm_gui.state import PROJECT_DEFAULT_GUI_STATE, digest_after_selection, gui_state_from_params
from spektrafilm_gui.napari_layout import dialog_parent, reset_viewer_camera, set_canvas_background, set_status
from spektrafilm_gui.params_mapper import build_params_from_state
from spektrafilm_gui.state_bridge import apply_gui_state, collect_gui_state
from spektrafilm_gui.widgets import WidgetBundle

OUTPUT_FLOAT_DATA_KEY = 'pipeline_float_output'
OUTPUT_COLOR_SPACE_KEY = 'pipeline_output_color_space'
OUTPUT_CCTF_ENCODING_KEY = 'pipeline_output_cctf_encoding'
OUTPUT_DISPLAY_TRANSFORM_KEY = 'pipeline_use_display_transform'
PROFILE_SYNC_FIELDS = profile_sync.PROFILE_SYNC_FIELDS
if TYPE_CHECKING:
    import napari
    from napari.layers import Image as NapariImageLayer


QThreadPool = getattr(QtCore, 'QThreadPool')
QTimer = getattr(QtCore, 'QTimer')
QFileDialog = QtWidgets.QFileDialog
QMessageBox = QtWidgets.QMessageBox
SimulationRequest = runtime.SimulationRequest
SimulationResult = runtime.SimulationResult
STARTUP_PREVIEW_ASPECT_RATIO = (3, 2)


class _DirMemoryDialog:
    """Wraps QFileDialog to open in the last-used directory via QSettings."""

    def __init__(self, key: str) -> None:
        self._key = key

    def get_save_file_name(self, parent, title, filename, file_filter):
        last_dir = load_dialog_dir(self._key)
        initial = str(Path(last_dir) / Path(filename).name) if last_dir else filename
        path, fmt = QFileDialog.getSaveFileName(parent, title, initial, file_filter)
        if path:
            save_dialog_dir(self._key, str(Path(path).parent))
        return path, fmt

    def get_open_file_name(self, parent, title, _initial, file_filter):
        path, fmt = QFileDialog.getOpenFileName(
            parent, title, load_dialog_dir(self._key), file_filter
        )
        if path:
            save_dialog_dir(self._key, str(Path(path).parent))
        return path, fmt


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


def runtime_simulator(*args, **kwargs):
    return import_module('spektrafilm.runtime.api').Simulator(*args, **kwargs)


def digest_params(*args, **kwargs):
    return import_module('spektrafilm.runtime.api').digest_params(*args, **kwargs)


def load_image_oiio(*args, **kwargs):
    return import_module('spektrafilm.utils.io').load_image_oiio(*args, **kwargs)


def save_image_oiio(*args, **kwargs):
    return import_module("spektrafilm.utils.io").save_image_oiio(*args, **kwargs)


def read_image_metadata(*args, **kwargs):
    return import_module("spektrafilm.utils.io").read_image_metadata(*args, **kwargs)


def write_image_metadata(*args, **kwargs):
    return import_module("spektrafilm.utils.io").write_image_metadata(*args, **kwargs)


def load_and_process_raw_file(*args, **kwargs):
    return import_module('spektrafilm.utils.raw_file_processor').load_and_process_raw_file(*args, **kwargs)


def resize_for_preview(*args, **kwargs):
    return import_module('spektrafilm.utils.preview').resize_for_preview(*args, **kwargs)


colour = _LazyModuleProxy(_import_colour_module)
PILImage = _LazyModuleProxy(_import_pil_image_module)
ImageCms = _LazyModuleProxy(_import_imagecms_module)


class GuiController:
    def __init__(self, *, viewer: napari.Viewer, widgets: WidgetBundle):
        self._viewer = viewer
        self._widgets = widgets
        self._layers = ViewerLayerService(
            viewer=viewer,
            output_float_data_key=OUTPUT_FLOAT_DATA_KEY,
            output_color_space_key=OUTPUT_COLOR_SPACE_KEY,
            output_cctf_encoding_key=OUTPUT_CCTF_ENCODING_KEY,
            output_display_transform_key=OUTPUT_DISPLAY_TRANSFORM_KEY,
        )
        self._thread_pool = QThreadPool.globalInstance()
        self._active_simulation_worker: runtime.SimulationWorker | None = None
        self._active_simulation_label: str | None = None
        self._runtime_simulator = None
        self._next_runtime_digest_applies_stock_specifics = True
        self._current_input_image: np.ndarray | None = None
        self._current_input_path: str | None = None
        self._current_preview_image: np.ndarray | None = None
        self._auto_preview_scheduled = False
        self._pending_auto_preview = False
        self._active_simulation_reports_status = True

    def show_startup_placeholder(self) -> None:
        if self._white_border_layer() is not None:
            return

        state = collect_gui_state(widgets=self._widgets)
        preview_height = max(int(state.display.preview_max_size), 1)
        preview_width = max(
            int(round(preview_height * STARTUP_PREVIEW_ASPECT_RATIO[1] / STARTUP_PREVIEW_ASPECT_RATIO[0])),
            1,
        )
        placeholder_preview = np.zeros((preview_height, preview_width, 3), dtype=np.uint8)
        self._layers.set_or_add_input_preview_layer(
            placeholder_preview,
            watermark_source_size=(preview_height, preview_width),
            white_padding=state.display.white_padding,
            hide_output=True,
            set_active=True,
        )
        self._home_input_stack()

    def load_input_image(self, path: str) -> None:
        image = load_image_oiio(path)[..., :3]
        self._current_input_path = path
        self._set_or_add_input_stack(image)
        self._request_auto_preview_if_enabled()

    def load_raw_image(self, path: str) -> None:
        gui_state = collect_gui_state(widgets=self._widgets)
        set_status(self._viewer, "Loading raw...", timeout_ms=0)
        lens_info: dict[str, str] = {}
        try:
            image = load_and_process_raw_file(
                path,
                white_balance=gui_state.load_raw.white_balance,
                temperature=gui_state.load_raw.temperature,
                tint=gui_state.load_raw.tint,
                lens_correction=gui_state.load_raw.lens_correction,
                output_colorspace=gui_state.input_image.input_color_space,
                output_cctf_encoding=gui_state.input_image.apply_cctf_decoding,
                lens_info_out=lens_info,
            )
        except (OSError, ValueError) as exc:
            QMessageBox.critical(dialog_parent(self._viewer), 'Load raw', f'Failed to load RAW image.\n\n{exc}')
            set_status(self._viewer, 'Load raw failed')
            return

        self._current_input_path = path
        self._set_or_add_input_stack(image)

        lens_summary = lens_info.get('summary')
        if lens_summary:
            set_status(
                self._viewer,
                f"Loaded raw and applied lens correction: {lens_summary}",
            )
        elif gui_state.load_raw.lens_correction:
            set_status(self._viewer, "Loaded raw, lens correction not applied")
        else:
            set_status(self._viewer, "Loaded raw")
        self._request_auto_preview_if_enabled()

    def refresh_preview_cache(self, *_args) -> None:
        input_image = self._current_input_image
        if input_image is None:
            return
        self._update_preview_cache(
            input_image,
            home_input_stack=False,
            hide_output=False,
        )

    def apply_profile_defaults(self, _selected_value: str) -> None:
        state = collect_gui_state(widgets=self._widgets)
        if not state.simulation.film_stock or not state.simulation.print_paper:
            return

        params = build_params_from_state(state)
        synced_state = gui_state_from_params(
            digest_after_selection(params),
            film_stock=state.simulation.film_stock,
            print_paper=state.simulation.print_paper,
        )
        self._apply_profile_sync_state(synced_state)
        self._next_runtime_digest_applies_stock_specifics = True

    def apply_film_profile_defaults(self, film_stock: str) -> None:
        self.apply_profile_defaults(film_stock)

    def _apply_profile_sync_state(self, synced_state) -> None:
        profile_sync.apply_profile_sync_state(
            widgets=self._widgets,
            synced_state=synced_state,
            profile_sync_fields=PROFILE_SYNC_FIELDS,
        )

    def run_preview(self) -> None:
        self._run_preview(report_status=True)

    def _run_preview(self, *, report_status: bool) -> None:
        self._start_simulation(
            source_layer_name=INPUT_PREVIEW_LAYER_NAME,
            mode_label='Preview',
            report_status=report_status,
        )

    def run_scan(self) -> None:
        self._start_simulation(source_layer_name=INPUT_LAYER_NAME, mode_label='Scan')

    def request_auto_preview(self, *_args) -> None:
        if self._auto_preview_scheduled:
            return
        self._auto_preview_scheduled = True
        QTimer.singleShot(0, self._run_scheduled_auto_preview)

    def _request_auto_preview_if_enabled(self) -> None:
        if not self._auto_preview_enabled() or self._current_preview_image is None:
            return
        self.request_auto_preview()

    def report_display_transform_status(self, enabled: bool) -> None:
        if enabled and not self.sync_display_transform_availability(report_status=True):
            return
        set_status(self._viewer, runtime.display_transform_status_message(enabled, imagecms_module=ImageCms))

    def set_gray_18_canvas_enabled(self, enabled: bool) -> None:
        set_canvas_background(self._viewer, gray_18_canvas=enabled)

    def set_output_interpolation_mode(self, mode: str) -> None:
        output_layer = self._output_layer()
        if output_layer is None:
            return
        self._layers.set_output_layer_interpolation(output_layer, mode)

    def sync_display_transform_availability(self, *, report_status: bool) -> bool:
        if runtime.display_profile_available(imagecms_module=ImageCms):
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

        filepath, _ = _DirMemoryDialog('save_output').get_save_file_name(
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
            image_data = runtime.normalized_image_data(np.asarray(output_layer.data)[..., :3])
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

        source_metadata = None

        if self._current_input_path is not None:
            source_metadata = read_image_metadata(self._current_input_path)

        try:
            save_image_oiio(filepath, image_data)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(dialog_parent(self._viewer), 'Save output', f'Failed to save output image.\n\n{exc}')
            return

        if source_metadata is not None:
            try:
                write_image_metadata(filepath, source_metadata)
            except Exception:
                pass

        set_status(self._viewer, f"Saved output image to {filepath}")

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
            file_dialog=_DirMemoryDialog('gui_state'),
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
            file_dialog=_DirMemoryDialog('gui_state'),
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

    def _preview_input_layer(self) -> NapariImageLayer | None:
        return self._layers.preview_input_layer()

    def _white_border_layer(self) -> NapariImageLayer | None:
        return self._layers.white_border_layer()

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
            output_interpolation_mode=self._output_interpolation_mode(),
        )

    def _set_or_add_input_stack(
        self,
        image: np.ndarray,
    ) -> None:
        self._update_preview_cache(
            image,
            home_input_stack=True,
            hide_output=True,
        )

    def _update_preview_cache(
        self,
        image: np.ndarray,
        *,
        home_input_stack: bool,
        hide_output: bool,
    ) -> None:
        state = collect_gui_state(widgets=self._widgets)
        preview_image = self._resize_for_preview(image, max_size=state.display.preview_max_size)
        preview_display_image = self._prepare_input_color_preview_image(
            preview_image,
            input_color_space=state.input_image.input_color_space,
            apply_cctf_decoding=state.input_image.apply_cctf_decoding,
        )
        self._current_input_image = image
        self._current_preview_image = preview_image
        self._layers.set_or_add_input_preview_layer(
            preview_display_image,
            watermark_source_size=tuple(int(dimension) for dimension in image.shape[:2]),
            white_padding=state.display.white_padding,
            hide_output=hide_output,
            set_active=home_input_stack or self._output_layer() is None,
        )
        if home_input_stack:
            self._home_input_stack()

    def _sync_white_border(self, *, white_padding: float) -> None:
        self._layers.sync_white_border(white_padding=white_padding)

    def _home_input_stack(self) -> None:
        if self._white_border_layer() is None:
            return
        reset_viewer_camera(self._viewer)
        self._set_active_layer(self._white_border_layer())

    def _simulation_input_image(self, *, source_layer_name: str) -> np.ndarray | None:
        if source_layer_name == INPUT_PREVIEW_LAYER_NAME:
            return self._current_preview_image
        if source_layer_name == INPUT_LAYER_NAME:
            return self._current_input_image
        return None

    def _auto_preview_enabled(self) -> bool:
        simulation_section = getattr(self._widgets, 'simulation', None)
        auto_preview_value = getattr(simulation_section, 'auto_preview_value', None)
        return bool(auto_preview_value()) if callable(auto_preview_value) else False

    def _run_scheduled_auto_preview(self) -> None:
        self._auto_preview_scheduled = False
        if not self._auto_preview_enabled() or self._current_preview_image is None:
            self._pending_auto_preview = False
            return
        if self._active_simulation_worker is not None:
            self._pending_auto_preview = True
            return
        self._run_preview(report_status=False)

    def _replay_pending_auto_preview(self) -> None:
        if not self._pending_auto_preview:
            return
        self._pending_auto_preview = False
        self.request_auto_preview()

    def _output_layer(self) -> NapariImageLayer | None:
        return self._layers.output_layer()

    def _set_active_layer(self, layer: NapariImageLayer | None) -> None:
        self._layers.set_active_layer(layer)

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

    def _output_interpolation_mode(self) -> str:
        display_section = getattr(self._widgets, 'display', None)
        editor = getattr(display_section, 'output_interpolation', None)
        value = getattr(editor, 'value', None)
        if isinstance(value, str) and value:
            return value
        current_text = getattr(editor, 'currentText', None)
        if callable(current_text):
            text = current_text()
            if isinstance(text, str) and text:
                return text
        return 'spline36'

    @staticmethod
    def _resize_for_preview(image_data: np.ndarray, *, max_size: int) -> np.ndarray:
        return resize_for_preview(image_data, max_size)

    @staticmethod
    def _prepare_input_color_preview_image(
        image_data: np.ndarray,
        *,
        input_color_space: str,
        apply_cctf_decoding: bool,
    ) -> np.ndarray:
        return runtime.prepare_input_color_preview_image(
            image_data,
            input_color_space=input_color_space,
            apply_cctf_decoding=apply_cctf_decoding,
            colour_module=colour,
        )

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

    def _process_image_with_runtime(self, image_data: np.ndarray, params) -> np.ndarray:
        apply_stocks_specifics = (
            self._runtime_simulator is None
            or self._next_runtime_digest_applies_stock_specifics
        )
        digested_params = digest_params(
            params,
            apply_stocks_specifics=apply_stocks_specifics,
        )
        try:
            if self._runtime_simulator is None:
                self._runtime_simulator = runtime_simulator(digested_params)
            else:
                self._runtime_simulator.update_params(digested_params)
            self._next_runtime_digest_applies_stock_specifics = False
            return self._runtime_simulator.process(image_data)
        except Exception:
            self._runtime_simulator = None
            raise

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

    def _execute_simulation_request(self, request: SimulationRequest) -> SimulationResult:
        return runtime.execute_simulation_request(
            request,
            run_simulation_fn=self._process_image_with_runtime,
            prepare_output_display_image_fn=self._prepare_output_display_image,
        )

    @staticmethod
    def _configure_simulation_params(params, *, source_layer_name: str):
        settings = getattr(params, 'settings', None)
        if settings is not None and hasattr(settings, 'preview_mode'):
            settings.preview_mode = source_layer_name == INPUT_PREVIEW_LAYER_NAME
        return params

    def _start_simulation(self, *, source_layer_name: str, mode_label: str, report_status: bool = True) -> None:
        if self._active_simulation_worker is not None:
            set_status(self._viewer, 'Simulation already running')
            return

        image_data = self._simulation_input_image(source_layer_name=source_layer_name)
        if image_data is None:
            QMessageBox.warning(dialog_parent(self._viewer), 'Run simulation', 'Load an input image before running the simulation.')
            return

        state = collect_gui_state(widgets=self._widgets)
        self._sync_white_border(white_padding=state.display.white_padding)
        params = self._configure_simulation_params(
            build_params_from_state(state),
            source_layer_name=source_layer_name,
        )

        image = np.double(image_data)
        request = SimulationRequest(
            mode_label=mode_label,
            image=image,
            params=params,
            output_color_space=state.simulation.output_color_space,
            use_display_transform=state.display.use_display_transform,
        )

        worker = runtime.SimulationWorker(request, execute_request=self._execute_simulation_request)
        worker.signals.finished.connect(self._on_simulation_finished)
        worker.signals.failed.connect(self._on_simulation_failed)
        self._active_simulation_worker = worker
        self._active_simulation_label = mode_label
        self._active_simulation_reports_status = report_status
        self._set_simulation_controls_enabled(False)
        if report_status:
            set_status(self._viewer, f'Computing {mode_label.lower()}...', timeout_ms=0)
        self._thread_pool.start(worker)

    def _on_simulation_finished(self, result: SimulationResult) -> None:
        report_status = self._active_simulation_reports_status
        self._active_simulation_worker = None
        self._active_simulation_label = None
        self._active_simulation_reports_status = True
        self._set_simulation_controls_enabled(True)
        self._set_or_add_output_layer(
            result.display_image,
            float_image=result.float_image,
            output_color_space=result.output_color_space,
            output_cctf_encoding=True,
            use_display_transform=result.use_display_transform,
        )
        if report_status:
            set_status(self._viewer, f'{result.mode_label} completed. {result.status_message}')
        self._replay_pending_auto_preview()

    def _on_simulation_failed(self, message: str) -> None:
        self._active_simulation_worker = None
        mode_label = self._active_simulation_label or 'Simulation'
        self._active_simulation_label = None
        self._active_simulation_reports_status = True
        self._set_simulation_controls_enabled(True)
        QMessageBox.critical(dialog_parent(self._viewer), 'Run simulation', f'Simulation failed.\n\n{message}')
        set_status(self._viewer, f'{mode_label} failed')
        self._replay_pending_auto_preview()

    def _set_simulation_controls_enabled(self, enabled: bool) -> None:
        simulation_section = getattr(self._widgets, 'simulation', None)
        if simulation_section is None:
            return
        for button_name in ('preview_button', 'scan_button', 'save_button'):
            button = getattr(simulation_section, button_name, None)
            set_enabled = getattr(button, 'setEnabled', None)
            if callable(set_enabled):
                set_enabled(enabled)

    def _run_simulation(self, *, source_layer_name: str) -> None:
        image_data = self._simulation_input_image(source_layer_name=source_layer_name)
        if image_data is None:
            QMessageBox.warning(dialog_parent(self._viewer), 'Run simulation', 'Load an input image before running the simulation.')
            return

        state = collect_gui_state(widgets=self._widgets)
        self._sync_white_border(white_padding=state.display.white_padding)
        params = self._configure_simulation_params(
            build_params_from_state(state),
            source_layer_name=source_layer_name,
        )

        image = np.double(image_data)
        scan = self._process_image_with_runtime(image, params)
        scan_display, display_status = self._prepare_output_display_image(
            scan,
            output_color_space=state.simulation.output_color_space,
            use_display_transform=state.display.use_display_transform,
        )
        self._set_or_add_output_layer(
            scan_display,
            float_image=scan,
            output_color_space=state.simulation.output_color_space,
            output_cctf_encoding=True,
            use_display_transform=state.display.use_display_transform,
        )
        set_status(self._viewer, display_status)