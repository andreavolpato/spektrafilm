from __future__ import annotations

from dataclasses import fields
from typing import Any, get_args, get_origin, get_type_hints

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from spectral_film_lab.gui.state import (
    CouplersState,
    GlareState,
    GrainState,
    HalationState,
    InputImageState,
    PreflashingState,
    SimulationState,
    SpecialState,
)
from spectral_film_lab.gui.widget_specs import GUI_SECTION_ENUMS, GUI_WIDGET_SPECS


def _enum_values(enum_cls):
    return [member.value for member in enum_cls]


def _build_collapsible_form_section(
    title: str,
    form: QFormLayout,
    *,
    expanded: bool,
) -> QVBoxLayout:
    content = QWidget()
    content.setLayout(form)

    root = QVBoxLayout()
    root.setContentsMargins(0, 0, 0, 0)
    root.addWidget(CollapsibleSection(title, content, expanded=expanded))
    return root


class CollapsibleSection(QWidget):
    def __init__(self, title: str, content: QWidget, *, expanded: bool = True):
        super().__init__()
        self._content = content

        self._toggle = QToolButton()
        self._toggle.setText(title)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(expanded)
        self._toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self._toggle.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._toggle.toggled.connect(self._set_expanded)

        self._frame = QFrame()
        self._frame.setFrameShape(QFrame.StyledPanel)
        self._frame.setFrameShadow(QFrame.Plain)
        frame_layout = QVBoxLayout(self._frame)
        frame_layout.setContentsMargins(8, 8, 8, 8)
        frame_layout.addWidget(self._content)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self._toggle)
        layout.addWidget(self._frame)

        self._set_expanded(expanded)

    def _set_expanded(self, expanded: bool) -> None:
        self._toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self._frame.setVisible(expanded)


class FloatEditor(QDoubleSpinBox):
    def __init__(self, *, decimals: int = 2, minimum: float = -1_000_000.0, maximum: float = 1_000_000.0):
        super().__init__()
        self.setDecimals(decimals)
        self.setMinimum(minimum)
        self.setMaximum(maximum)
        self.setKeyboardTracking(False)

    @property
    def value(self) -> float:
        return super().value()

    @value.setter
    def value(self, value: float) -> None:
        self.setValue(value)


class IntEditor(QSpinBox):
    def __init__(self, *, minimum: int = -1_000_000, maximum: int = 1_000_000):
        super().__init__()
        self.setMinimum(minimum)
        self.setMaximum(maximum)
        self.setKeyboardTracking(False)

    @property
    def value(self) -> int:
        return super().value()

    @value.setter
    def value(self, value: int) -> None:
        self.setValue(value)


class BoolEditor(QCheckBox):
    @property
    def value(self) -> bool:
        return self.isChecked()

    @value.setter
    def value(self, value: bool) -> None:
        self.setChecked(value)


class EnumEditor(QComboBox):
    def __init__(self, values: list[str]):
        super().__init__()
        self.addItems(values)

    @property
    def value(self) -> str:
        return self.currentText()

    @value.setter
    def value(self, value: str) -> None:
        index = self.findText(value)
        if index >= 0:
            self.setCurrentIndex(index)
        else:
            raise ValueError(f"{value!r} is not a valid option")


class TupleEditor(QWidget):
    def __init__(self, editors: list[QWidget]):
        super().__init__()
        self._editors = editors
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        for editor in self._editors:
            editor.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            layout.addWidget(editor)
        self.setLayout(layout)

    @property
    def value(self) -> tuple[Any, ...]:
        return tuple(editor.value for editor in self._editors)

    @value.setter
    def value(self, value: tuple[Any, ...]) -> None:
        for editor, component in zip(self._editors, value, strict=True):
            editor.value = component

    def setToolTip(self, text: str) -> None:  # noqa: N802 - Qt API name
        super().setToolTip(text)
        for editor in self._editors:
            editor.setToolTip(text)


class FloatTupleEditor(TupleEditor):
    def __init__(self, length: int, *, decimals: int = 2, minimum: float = -1_000_000.0, maximum: float = 1_000_000.0):
        super().__init__(
            [FloatEditor(decimals=decimals, minimum=minimum, maximum=maximum) for _ in range(length)],
        )


class IntTupleEditor(TupleEditor):
    def __init__(self, length: int, *, minimum: int = -1_000_000, maximum: int = 1_000_000):
        super().__init__([IntEditor(minimum=minimum, maximum=maximum) for _ in range(length)])


def _format_label(field_name: str) -> str:
    label = field_name.replace("_", " ")
    label = label.replace("uv", "UV").replace("ir", "IR").replace("cctf", "CCTF")
    return label.capitalize()


class DataclassSection(QWidget):
    def __init__(
        self,
        *,
        state_cls: type[Any],
        section_name: str,
        title: str,
        enum_fields: dict[str, type[Any]] | None = None,
        field_labels: dict[str, str] | None = None,
        hidden_fields: set[str] | None = None,
        collapsed_by_default: bool = False,
    ):
        super().__init__()
        self._state_cls = state_cls
        self._section_name = section_name
        self._title = title
        self._enum_fields = enum_fields or {}
        self._field_labels = field_labels or {}
        self._hidden_fields = hidden_fields or set()
        self._collapsed_by_default = collapsed_by_default
        self._type_hints = get_type_hints(state_cls)
        self._init_extra_widgets()
        self._build_ui()
        self._apply_specs()

    def _init_extra_widgets(self) -> None:
        return

    def _build_ui(self) -> None:
        form = QFormLayout()
        self._add_extra_rows_before(form)
        for field_info in fields(self._state_cls):
            field_name = field_info.name
            annotation = self._type_hints[field_name]
            widget = self._build_editor(field_name, annotation)
            setattr(self, field_name, widget)
            if field_name not in self._hidden_fields:
                form.addRow(self._field_labels.get(field_name, _format_label(field_name)), widget)
        self._add_extra_rows_after(form)

        content = QWidget()
        content.setLayout(form)

        root = QVBoxLayout()
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(CollapsibleSection(self._title, content, expanded=not self._collapsed_by_default))
        self.setLayout(root)

    def _add_extra_rows_before(self, form: QFormLayout) -> None:
        return

    def _add_extra_rows_after(self, form: QFormLayout) -> None:
        return

    def _build_editor(self, field_name: str, annotation: Any) -> QWidget:
        enum_cls = self._enum_fields.get(field_name)
        if enum_cls is not None:
            return EnumEditor(_enum_values(enum_cls))
        if annotation is bool:
            return BoolEditor()
        if annotation is int:
            return IntEditor()
        if annotation is float:
            return FloatEditor()
        if get_origin(annotation) is tuple:
            element_types = get_args(annotation)
            if element_types and all(element_type is int for element_type in element_types):
                return IntTupleEditor(len(element_types))
            return FloatTupleEditor(len(element_types))
        raise TypeError(f"Unsupported field type for {self._state_cls.__name__}.{field_name}: {annotation!r}")

    def _apply_specs(self) -> None:
        specs = GUI_WIDGET_SPECS.get(self._section_name, {})
        for field_name, spec in specs.items():
            widget = getattr(self, field_name)
            if spec.tooltip:
                widget.setToolTip(spec.tooltip)
            if spec.min_value is not None:
                self._apply_numeric_attr(widget, "setMinimum", spec.min_value)
            if spec.max_value is not None:
                self._apply_numeric_attr(widget, "setMaximum", spec.max_value)
            if spec.step is not None:
                self._apply_numeric_attr(widget, "setSingleStep", spec.step)

    @staticmethod
    def _apply_numeric_attr(widget: QWidget, method_name: str, value: float | int) -> None:
        method = getattr(widget, method_name, None)
        if callable(method):
            method(value)
            return
        editors = getattr(widget, "_editors", None)
        if editors is not None:
            for editor in editors:
                getattr(editor, method_name)(value)

    def set_state(self, state: Any) -> None:
        for field_info in fields(self._state_cls):
            field_name = field_info.name
            getattr(self, field_name).value = getattr(state, field_name)

    def get_state(self) -> Any:
        values = {field_info.name: getattr(self, field_info.name).value for field_info in fields(self._state_cls)}
        return self._state_cls(**values)


class InputImageSection(DataclassSection):
    def __init__(self):
        super().__init__(
            state_cls=InputImageState,
            section_name="input_image",
            title="Input",
            enum_fields=GUI_SECTION_ENUMS["input_image"],
            collapsed_by_default=True,
            field_labels={
                "preview_resize_factor": "Preview resize",
                "upscale_factor": "Upscale factor",
                "crop": "Crop",
                "crop_center": "Crop center",
                "crop_size": "Crop size",
                "input_color_space": "Input color space",
                "apply_cctf_decoding": "Apply CCTF decoding",
                "spectral_upsampling_method": "Spectral upsampling",
                "filter_uv": "UV filter",
                "filter_ir": "IR filter",
            },
            hidden_fields={
                "preview_resize_factor",
                "upscale_factor",
                "crop",
                "crop_center",
                "crop_size",
                "spectral_upsampling_method",
                "filter_uv",
                "filter_ir",
            },
        )


class PreviewCropSection(QWidget):
    def __init__(self, input_image_section: InputImageSection):
        super().__init__()
        form = QFormLayout()
        form.addRow("Preview resize", input_image_section.preview_resize_factor)
        form.addRow("Upscale factor", input_image_section.upscale_factor)
        form.addRow("Crop", input_image_section.crop)
        form.addRow("Crop center", input_image_section.crop_center)
        form.addRow("Crop size", input_image_section.crop_size)

        self.setLayout(_build_collapsible_form_section("Preview and crop", form, expanded=False))


class GrainSection(DataclassSection):
    def __init__(self):
        super().__init__(state_cls=GrainState, section_name="grain", title="Grain", collapsed_by_default=True)


class PreflashingSection(DataclassSection):
    def __init__(self):
        super().__init__(state_cls=PreflashingState, section_name="preflashing", title="Preflash", collapsed_by_default=True)


class HalationSection(DataclassSection):
    def __init__(self):
        super().__init__(state_cls=HalationState, section_name="halation", title="Halation", collapsed_by_default=True)


class CouplersSection(DataclassSection):
    def __init__(self):
        super().__init__(state_cls=CouplersState, section_name="couplers", title="Couplers", collapsed_by_default=True)


class GlareSection(DataclassSection):
    def __init__(self):
        super().__init__(state_cls=GlareState, section_name="glare", title="Glare", collapsed_by_default=True)


class SpecialSection(DataclassSection):
    def __init__(self, simulation_section: "SimulationSection"):
        self._simulation_section = simulation_section
        super().__init__(
            state_cls=SpecialState,
            section_name="special",
            title="Experimental",
            collapsed_by_default=True,
            hidden_fields={
                "film_gamma_factor",
                "print_gamma_factor",
                "print_density_min_factor",
            },
        )

    def _add_extra_rows_before(self, form: QFormLayout) -> None:
        form.addRow("Print illuminant", self._simulation_section.print_illuminant)


class SpectralUpsamplingSection(QWidget):
    def __init__(self, input_image_section: InputImageSection):
        super().__init__()
        form = QFormLayout()
        form.addRow("Spectral upsampling", input_image_section.spectral_upsampling_method)
        form.addRow("UV filter", input_image_section.filter_uv)
        form.addRow("IR filter", input_image_section.filter_ir)

        self.setLayout(_build_collapsible_form_section("Spectral upsampling", form, expanded=False))


class TuneSection(QWidget):
    def __init__(self, special_section: SpecialSection):
        super().__init__()
        form = QFormLayout()
        form.addRow("Film gamma factor", special_section.film_gamma_factor)
        form.addRow("Print gamma factor", special_section.print_gamma_factor)
        form.addRow("Print density min factor", special_section.print_density_min_factor)

        self.setLayout(_build_collapsible_form_section("Tune", form, expanded=True))


class FilePickerSection(QWidget):
    load_requested = Signal(str)

    def __init__(self):
        super().__init__()
        self._build_ui()

    def _build_ui(self) -> None:
        self.file_path = QLineEdit()
        self.file_path.setReadOnly(True)
        self.file_path.setPlaceholderText("No image selected")

        browse_button = QPushButton("Select file")
        browse_button.clicked.connect(self._choose_file)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.file_path)
        row.addWidget(browse_button)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.addLayout(row)

        root = QVBoxLayout()
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(CollapsibleSection("Image loader", content))
        self.setLayout(root)

    def _choose_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select input image")
        if not path:
            return
        self.file_path.setText(path)
        self.load_requested.emit(path)

    def set_path(self, path: str) -> None:
        self.file_path.setText(path)


class SimulationInputSection(QWidget):
    def __init__(self):
        super().__init__()
        self.input_layer = QComboBox()

        form = QFormLayout()
        form.addRow("Input layer", self.input_layer)

        self.setLayout(_build_collapsible_form_section("Simulation input", form, expanded=True))

    def set_available_layers(self, layer_names: list[str], *, selected_name: str | None = None) -> None:
        current_name = selected_name or self.selected_input_layer_name()
        self.input_layer.blockSignals(True)
        self.input_layer.clear()
        self.input_layer.addItems(layer_names)
        if current_name:
            index = self.input_layer.findText(current_name)
            if index >= 0:
                self.input_layer.setCurrentIndex(index)
        self.input_layer.blockSignals(False)

    def selected_input_layer_name(self) -> str | None:
        if self.input_layer.count() == 0:
            return None
        return self.input_layer.currentText()


class SimulationSection(DataclassSection):
    preview_requested = Signal()
    scan_requested = Signal()
    save_requested = Signal()

    def __init__(self):
        super().__init__(
            state_cls=SimulationState,
            section_name="simulation",
            title="Profiles",
            enum_fields=GUI_SECTION_ENUMS["simulation"],
            field_labels={
                "film_stock": "Film profile",
                "film_format_mm": "Film format mm",
                "camera_lens_blur_um": "Camera lens blur um",
                "exposure_compensation_ev": "Exposure compensation ev",
                "auto_exposure": "Auto exposure",
                "auto_exposure_method": "Auto exposure method",
                "print_paper": "Print profile",
                "print_illuminant": "Print illuminant",
                "print_exposure": "Print exposure",
                "print_exposure_compensation": "Print auto compensation",
                "print_y_filter_shift": "Print Y filter shift",
                "print_m_filter_shift": "Print M filter shift",
                "scan_lens_blur": "Scan lens blur",
                "scan_unsharp_mask": "Scan unsharp mask",
                "output_color_space": "Output color space",
                "output_cctf_encoding": "Output CCTF encoding",
                "scan_film": "Scan film",
                "compute_full_image": "Compute full image",
            },
            hidden_fields={
                "film_format_mm",
                "camera_lens_blur_um",
                "exposure_compensation_ev",
                "auto_exposure",
                "auto_exposure_method",
                "print_exposure",
                "print_exposure_compensation",
                "print_y_filter_shift",
                "print_m_filter_shift",
                "print_illuminant",
                "scan_lens_blur",
                "scan_unsharp_mask",
                "scan_film",
                "compute_full_image",
                "output_color_space",
                "output_cctf_encoding",
            },
        )

    def _init_extra_widgets(self) -> None:
        self.bottom_scan_film = QCheckBox()
        self.preview_button = QPushButton("Preview")
        self.preview_button.setToolTip("Run the simulation in preview mode, grain and halation are deactivated for speed")
        self.preview_button.clicked.connect(self.preview_requested.emit)

        self.scan_button = QPushButton("Scan")
        self.scan_button.setToolTip("Run the simulation at full resolution and with grain and halation")
        self.scan_button.clicked.connect(self.scan_requested.emit)

        self.save_button = QPushButton("Save")
        self.save_button.setToolTip("Save the current output layer to an image file")
        self.save_button.clicked.connect(self.save_requested.emit)

        self.action_buttons = QWidget()
        buttons_layout = QHBoxLayout(self.action_buttons)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.setSpacing(6)
        buttons_layout.addWidget(self.preview_button, 1)
        buttons_layout.addWidget(self.scan_button, 1)
        buttons_layout.addWidget(self.save_button, 1)

        self.bottom_bar = QWidget()
        bottom_layout = QVBoxLayout(self.bottom_bar)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(6)

        scan_film_row = QFormLayout()
        scan_film_row.setContentsMargins(0, 0, 0, 0)
        scan_film_row.addRow("Scan film", self.bottom_scan_film)

        bottom_layout.addLayout(scan_film_row)
        bottom_layout.addWidget(self.action_buttons)

    def _add_extra_rows_after(self, form: QFormLayout) -> None:
        return

    def action_bar(self) -> QWidget:
        return self.bottom_bar

    def set_scan_film_value(self, value: bool) -> None:
        self.bottom_scan_film.setChecked(value)

    def scan_film_value(self) -> bool:
        return self.bottom_scan_film.isChecked()


class OutputSection(QWidget):
    def __init__(self, simulation_section: SimulationSection):
        super().__init__()
        form = QFormLayout()
        form.addRow("Output color space", simulation_section.output_color_space)
        form.addRow("Output CCTF encoding", simulation_section.output_cctf_encoding)

        self.setLayout(_build_collapsible_form_section("Output", form, expanded=False))


class ExposureControlSection(QWidget):
    def __init__(self, simulation_section: SimulationSection):
        super().__init__()
        form = QFormLayout()
        self._add_tooltip_row(
            form,
            "Camera auto exposure",
            simulation_section.auto_exposure,
            "Use the auto-exposure feature of the virtual camera",
        )
        self._add_tooltip_row(
            form,
            "Camera compensation ev",
            simulation_section.exposure_compensation_ev,
            "Add a bias to the auto-exposure of the camera",
        )
        self._add_tooltip_row(
            form,
            "Print auto compensation",
            simulation_section.print_exposure_compensation,
            "Auto adjust the print exposure for the camera exposure compesation ev",
        )
        self._add_tooltip_row(
            form,
            "Print exposure",
            simulation_section.print_exposure,
            "Changes the exposure time set in the virtual enlarger",
        )

        self.setLayout(_build_collapsible_form_section("Exposure control", form, expanded=True))

    def _add_tooltip_row(self, form: QFormLayout, label_text: str, widget: QWidget, tooltip: str) -> None:
        label = QLabel(label_text)
        label.setToolTip(tooltip)
        widget.setToolTip(tooltip)
        form.addRow(label, widget)


class EnlargerSection(QWidget):
    def __init__(self, simulation_section: SimulationSection):
        super().__init__()
        form = QFormLayout()
        form.addRow("Print Y filter shift", simulation_section.print_y_filter_shift)
        form.addRow("Print M filter shift", simulation_section.print_m_filter_shift)

        self.setLayout(_build_collapsible_form_section("Enlarger", form, expanded=True))


class ScannerSection(QWidget):
    def __init__(self, simulation_section: SimulationSection):
        super().__init__()
        form = QFormLayout()
        form.addRow("Scan lens blur", simulation_section.scan_lens_blur)
        form.addRow("Scan unsharp mask", simulation_section.scan_unsharp_mask)

        self.setLayout(_build_collapsible_form_section("Scanner", form, expanded=False))


class CameraSection(QWidget):
    def __init__(self, simulation_section: SimulationSection):
        super().__init__()
        form = QFormLayout()
        form.addRow("Film format mm", simulation_section.film_format_mm)
        form.addRow("Auto exposure method", simulation_section.auto_exposure_method)
        form.addRow("Camera lens blur um", simulation_section.camera_lens_blur_um)
        self.setLayout(_build_collapsible_form_section("Camera", form, expanded=False))