from __future__ import annotations

from dataclasses import fields
from typing import Any, get_args, get_origin, get_type_hints

from qtpy import QtCore, QtGui, QtWidgets

QCheckBox = QtWidgets.QCheckBox
QComboBox = QtWidgets.QComboBox
QDoubleSpinBox = QtWidgets.QDoubleSpinBox
QFileDialog = QtWidgets.QFileDialog
QFormLayout = QtWidgets.QFormLayout
QFrame = QtWidgets.QFrame
QHBoxLayout = QtWidgets.QHBoxLayout
QLabel = QtWidgets.QLabel
QLineEdit = QtWidgets.QLineEdit
QPointF = getattr(QtCore, 'QPointF')
QPushButton = QtWidgets.QPushButton
QRect = getattr(QtCore, 'QRect')
QSize = getattr(QtCore, 'QSize')
QSizePolicy = QtWidgets.QSizePolicy
QSpinBox = QtWidgets.QSpinBox
QToolButton = QtWidgets.QToolButton
QVBoxLayout = QtWidgets.QVBoxLayout
QWidget = QtWidgets.QWidget
QColor = QtGui.QColor
QPainter = QtGui.QPainter
QPen = QtGui.QPen
Qt = QtCore.Qt
Signal = QtCore.Signal

from spektrafilm_gui.state import (
    CouplersState,
    GlareState,
    GrainState,
    HalationState,
    InputImageState,
    PreflashingState,
    SimulationState,
    SpecialState,
)
from spektrafilm_gui.widget_specs import GUI_SECTION_ENUMS, GUI_WIDGET_SPECS


def _enum_values(enum_cls):
    return [member.value for member in enum_cls]


def _normalize_ui_text(text: str) -> str:
    return text.lower()


def _build_collapsible_form_section(
    title: str,
    form: QFormLayout,
    *,
    expanded: bool,
) -> QVBoxLayout:
    content = QWidget()
    content.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
    content.setLayout(form)

    root = QVBoxLayout()
    root.setContentsMargins(0, 0, 0, 0)
    root.setSpacing(0)
    root.addWidget(CollapsibleSection(title, content, expanded=expanded))
    return root


def _new_form_layout() -> QFormLayout:
    form = QFormLayout()
    form.setContentsMargins(0, 0, 0, 0)
    form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    form.setFormAlignment(Qt.AlignTop | Qt.AlignLeft)
    return form


def _add_form_rows(form: QFormLayout, rows: list[tuple[str | QLabel, QWidget]]) -> None:
    for label, widget in rows:
        if isinstance(label, str):
            label = _normalize_ui_text(label)
        form.addRow(label, widget)


def _build_linked_form_section(
    title: str,
    rows: list[tuple[str | QLabel, QWidget]],
    *,
    expanded: bool,
) -> QVBoxLayout:
    form = _new_form_layout()
    _add_form_rows(form, rows)
    return _build_collapsible_form_section(title, form, expanded=expanded)


def _build_button(
    text: str,
    callback: Any,
    *,
    tooltip: str | None = None,
    preserve_case: bool = False,
) -> QPushButton:
    button = QPushButton(text if preserve_case else _normalize_ui_text(text))
    if tooltip:
        button.setToolTip(tooltip)
    button.clicked.connect(callback)
    return button


def _build_button_row(*widgets: QWidget, stretch: int | None = None, spacing: int = 6) -> QHBoxLayout:
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(spacing)
    for widget in widgets:
        if stretch is None:
            row.addWidget(widget)
        else:
            row.addWidget(widget, stretch)
    return row


def _build_vertical_container(*items: QHBoxLayout | QFormLayout | QWidget, spacing: int = 6) -> QWidget:
    container = QWidget()
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(spacing)
    layout.setAlignment(Qt.AlignTop)
    for item in items:
        if isinstance(item, QWidget):
            layout.addWidget(item)
        else:
            layout.addLayout(item)
    return container


def _set_single_collapsible_layout(widget: QWidget, title: str, content: QWidget, *, expanded: bool = True) -> None:
    root = QVBoxLayout()
    root.setContentsMargins(0, 0, 0, 0)
    root.setSpacing(0)
    root.addWidget(CollapsibleSection(_normalize_ui_text(title), content, expanded=expanded))
    widget.setLayout(root)


class CollapsibleSection(QWidget):
    def __init__(self, title: str, content: QWidget, *, expanded: bool = True):
        super().__init__()
        self._content = content

        self._toggle = QToolButton()
        self._toggle.setText(_normalize_ui_text(title))
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
        frame_layout.setContentsMargins(24, 8, 8, 8)
        frame_layout.setSpacing(0)
        frame_layout.setAlignment(Qt.AlignTop)
        frame_layout.addWidget(self._content)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.setAlignment(Qt.AlignTop)
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
        self.setFixedHeight(24)

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
        self.setFixedHeight(24)

    @property
    def value(self) -> int:
        return super().value()

    @value.setter
    def value(self, value: int) -> None:
        self.setValue(value)


class BoolEditor(QCheckBox):
    def __init__(self):
        super().__init__()
        self.setText('')
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setFixedHeight(24)

    @property
    def value(self) -> bool:
        return self.isChecked()

    @value.setter
    def value(self, value: bool) -> None:
        self.setChecked(value)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt API name
        return QSize(24, 24)

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt API name
        return self.sizeHint()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API name
        del event
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.Antialiasing, True)

            indicator_rect = self._indicator_rect()
            is_enabled = self.isEnabled()
            fill_color = QColor('#8a8a8a' if is_enabled else '#858585')
            checked_color = QColor('#969696' if is_enabled else '#8c8c8c')
            border_color = QColor('#a7a7a7' if self.isChecked() else '#8f8f8f')

            painter.setPen(QPen(border_color, 1))
            painter.setBrush(checked_color if self.isChecked() else fill_color)
            painter.drawRect(indicator_rect)

            if self.isChecked():
                self._draw_check_mark(painter, indicator_rect)
        finally:
            painter.end()

    def _indicator_rect(self) -> QRect:
        return QRect(1, max(1, (self.height() - 14) // 2), 14, 14)

    @staticmethod
    def _draw_check_mark(painter: QPainter, indicator_rect: QRect) -> None:
        painter.setPen(QPen(QColor('#f7f7f7'), 1.6, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.drawLine(
            QPointF(indicator_rect.left() + 3.0, indicator_rect.center().y() + 0.5),
            QPointF(indicator_rect.left() + 6.0, indicator_rect.bottom() - 3.5),
        )
        painter.drawLine(
            QPointF(indicator_rect.left() + 6.0, indicator_rect.bottom() - 3.5),
            QPointF(indicator_rect.right() - 2.5, indicator_rect.top() + 3.5),
        )


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
            editor.setMinimumWidth(56)
            editor.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
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
    return _normalize_ui_text(field_name.replace("_", " "))


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
        form = _new_form_layout()
        self._add_extra_rows_before(form)
        for field_info in fields(self._state_cls):
            field_name = field_info.name
            annotation = self._type_hints[field_name]
            widget = self._build_editor(field_name, annotation)
            setattr(self, field_name, widget)
            if field_name not in self._hidden_fields:
                form.addRow(_normalize_ui_text(self._field_labels.get(field_name, _format_label(field_name))), widget)
        self._add_extra_rows_after(form)

        content = QWidget()
        content.setLayout(form)

        root = QVBoxLayout()
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(CollapsibleSection(self._title, content, expanded=not self._collapsed_by_default))
        self.setLayout(root)

    def _add_extra_rows_before(self, form: QFormLayout) -> None:
        del form
        return

    def _add_extra_rows_after(self, form: QFormLayout) -> None:
        del form
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


class SimpleDataclassSection(DataclassSection):
    STATE_CLS: type[Any]
    SECTION_NAME: str
    TITLE: str
    COLLAPSED_BY_DEFAULT = True
    ENUM_FIELDS_KEY: str | None = None
    FIELD_LABELS: dict[str, str] = {}
    HIDDEN_FIELDS: set[str] = set()

    def __init__(self):
        super().__init__(
            state_cls=self.STATE_CLS,
            section_name=self.SECTION_NAME,
            title=self.TITLE,
            enum_fields=GUI_SECTION_ENUMS[self.ENUM_FIELDS_KEY] if self.ENUM_FIELDS_KEY is not None else None,
            field_labels=self.FIELD_LABELS,
            hidden_fields=self.HIDDEN_FIELDS,
            collapsed_by_default=self.COLLAPSED_BY_DEFAULT,
        )


class InputImageSection(SimpleDataclassSection):
    STATE_CLS = InputImageState
    SECTION_NAME = "input_image"
    TITLE = "Input"
    ENUM_FIELDS_KEY = "input_image"
    FIELD_LABELS = {
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
    }
    HIDDEN_FIELDS = {
        "preview_resize_factor",
        "upscale_factor",
        "crop",
        "crop_center",
        "crop_size",
        "spectral_upsampling_method",
        "filter_uv",
        "filter_ir",
    }


class PreviewCropSection(QWidget):
    def __init__(self, input_image_section: InputImageSection):
        super().__init__()
        self.setLayout(
            _build_linked_form_section(
                "Preview and crop",
                [
                    ("Preview resize", input_image_section.preview_resize_factor),
                    ("Upscale factor", input_image_section.upscale_factor),
                    ("Crop", input_image_section.crop),
                    ("Crop center", input_image_section.crop_center),
                    ("Crop size", input_image_section.crop_size),
                ],
                expanded=False,
            ),
        )


class GrainSection(SimpleDataclassSection):
    STATE_CLS = GrainState
    SECTION_NAME = "grain"
    TITLE = "Grain"


class PreflashingSection(SimpleDataclassSection):
    STATE_CLS = PreflashingState
    SECTION_NAME = "preflashing"
    TITLE = "Preflash"


class HalationSection(SimpleDataclassSection):
    STATE_CLS = HalationState
    SECTION_NAME = "halation"
    TITLE = "Halation"


class CouplersSection(SimpleDataclassSection):
    STATE_CLS = CouplersState
    SECTION_NAME = "couplers"
    TITLE = "Couplers"


class GlareSection(SimpleDataclassSection):
    STATE_CLS = GlareState
    SECTION_NAME = "glare"
    TITLE = "Glare"


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
        form.addRow(_normalize_ui_text("Print illuminant"), self._simulation_section.print_illuminant)


class SpectralUpsamplingSection(QWidget):
    def __init__(self, input_image_section: InputImageSection):
        super().__init__()
        self.setLayout(
            _build_linked_form_section(
                "Spectral upsampling",
                [
                    ("Spectral upsampling", input_image_section.spectral_upsampling_method),
                    ("UV filter", input_image_section.filter_uv),
                    ("IR filter", input_image_section.filter_ir),
                ],
                expanded=False,
            ),
        )


class TuneSection(QWidget):
    def __init__(self, special_section: SpecialSection):
        super().__init__()
        self.setLayout(
            _build_linked_form_section(
                "Tune",
                [
                    ("Film gamma factor", special_section.film_gamma_factor),
                    ("Print gamma factor", special_section.print_gamma_factor),
                    ("Print density min factor", special_section.print_density_min_factor),
                ],
                expanded=True,
            ),
        )


class FilePickerSection(QWidget):
    load_requested = Signal(str)

    def __init__(self):
        super().__init__()
        self._build_ui()

    def _build_ui(self) -> None:
        self.file_path = QLineEdit()
        self.file_path.setReadOnly(True)
        self.file_path.setPlaceholderText(_normalize_ui_text("No image selected"))

        browse_button = _build_button("Select file", self._choose_file)
        content = _build_vertical_container(_build_button_row(self.file_path, browse_button, spacing=4), spacing=0)
        _set_single_collapsible_layout(self, "Image loader", content)

    def _choose_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, _normalize_ui_text("Select input image"))
        if not path:
            return
        self.file_path.setText(path)
        self.load_requested.emit(path)

    def set_path(self, path: str) -> None:
        self.file_path.setText(path)


class GuiConfigSection(QWidget):
    save_current_as_default_requested = Signal()
    save_current_to_file_requested = Signal()
    load_from_file_requested = Signal()
    restore_factory_default_requested = Signal()

    def __init__(self):
        super().__init__()
        self._build_ui()

    def _build_ui(self) -> None:
        self.save_current_as_default_button = _build_button(
            "Save current as default",
            self.save_current_as_default_requested.emit,
        )
        self.save_current_to_file_button = _build_button(
            "Save current to file",
            self.save_current_to_file_requested.emit,
        )
        self.load_from_file_button = _build_button("Load from file", self.load_from_file_requested.emit)
        self.restore_factory_default_button = _build_button(
            "Restore factory default",
            self.restore_factory_default_requested.emit,
        )

        content = _build_vertical_container(
            _build_button_row(self.save_current_as_default_button, self.save_current_to_file_button),
            _build_button_row(self.load_from_file_button, self.restore_factory_default_button),
        )
        _set_single_collapsible_layout(self, "GUI parameters", content, expanded=True)


class SimulationInputSection(QWidget):
    def __init__(self):
        super().__init__()
        self.input_layer = QComboBox()

        form = _new_form_layout()
        form.addRow(_normalize_ui_text("Simulation input"), self.input_layer)

        self.setLayout(form)

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
                "saving_color_space": "Saving color space",
                "saving_cctf_encoding": "Saving CCTF encoding",
                "use_display_transform": "Use display transform",
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
                "saving_color_space",
                "saving_cctf_encoding",
                "use_display_transform",
            },
        )

    def _init_extra_widgets(self) -> None:
        self.bottom_scan_film = BoolEditor()
        self.preview_button = _build_button(
            "PREVIEW",
            self.preview_requested.emit,
            tooltip="Run the simulation in preview mode, grain and halation are deactivated for speed",
            preserve_case=True,
        )
        self.scan_button = _build_button(
            "SCAN",
            self.scan_requested.emit,
            tooltip="Run the simulation at full resolution and with grain and halation",
            preserve_case=True,
        )
        self.save_button = _build_button(
            "SAVE",
            self.save_requested.emit,
            tooltip="Save the current output layer to an image file",
            preserve_case=True,
        )

        scan_film_row = _new_form_layout()
        scan_film_row.addRow(_normalize_ui_text("Scan film"), self.bottom_scan_film)

        action_buttons = QWidget()
        action_buttons.setLayout(_build_button_row(self.preview_button, self.scan_button, self.save_button, stretch=1))

        self.bottom_bar = _build_vertical_container(scan_film_row, action_buttons)

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
        self.setLayout(
            _build_linked_form_section(
                "Output",
                [
                    ("Output color space", simulation_section.output_color_space),
                    ("Saving color space", simulation_section.saving_color_space),
                    ("Saving CCTF encoding", simulation_section.saving_cctf_encoding),
                    ("Use display transform", simulation_section.use_display_transform),
                ],
                expanded=False,
            ),
        )


class ExposureControlSection(QWidget):
    def __init__(self, simulation_section: SimulationSection):
        super().__init__()
        form = _new_form_layout()
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
        label = QLabel(_normalize_ui_text(label_text))
        label.setToolTip(tooltip)
        widget.setToolTip(tooltip)
        form.addRow(label, widget)


class EnlargerSection(QWidget):
    def __init__(self, simulation_section: SimulationSection):
        super().__init__()
        self.setLayout(
            _build_linked_form_section(
                "Enlarger",
                [
                    ("Print Y filter shift", simulation_section.print_y_filter_shift),
                    ("Print M filter shift", simulation_section.print_m_filter_shift),
                ],
                expanded=True,
            ),
        )


class ScannerSection(QWidget):
    def __init__(self, simulation_section: SimulationSection):
        super().__init__()
        self.setLayout(
            _build_linked_form_section(
                "Scanner",
                [
                    ("Scan lens blur", simulation_section.scan_lens_blur),
                    ("Scan unsharp mask", simulation_section.scan_unsharp_mask),
                ],
                expanded=False,
            ),
        )


class CameraSection(QWidget):
    def __init__(self, simulation_section: SimulationSection):
        super().__init__()
        self.setLayout(
            _build_linked_form_section(
                "Camera",
                [
                    ("Film format mm", simulation_section.film_format_mm),
                    ("Auto exposure method", simulation_section.auto_exposure_method),
                    ("Camera lens blur um", simulation_section.camera_lens_blur_um),
                ],
                expanded=False,
            ),
        )