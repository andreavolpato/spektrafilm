from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

from qtpy import QtCore, QtWidgets

QComboBox = QtWidgets.QComboBox
QFileDialog = QtWidgets.QFileDialog
QFormLayout = QtWidgets.QFormLayout
QHBoxLayout = QtWidgets.QHBoxLayout
QLabel = QtWidgets.QLabel
QLineEdit = QtWidgets.QLineEdit
QPushButton = QtWidgets.QPushButton
QSizePolicy = QtWidgets.QSizePolicy
QVBoxLayout = QtWidgets.QVBoxLayout
QWidget = QtWidgets.QWidget
Qt = QtCore.Qt
Signal = QtCore.Signal

from spektrafilm_gui.state import (
    CouplersState,
    DisplayState,
    GlareState,
    GrainState,
    HalationState,
    InputImageState,
    LoadRawState,
    PreflashingState,
    SimulationState,
    SpecialState,
)
from spektrafilm_gui.persistence import load_dialog_dir, save_dialog_dir
from spektrafilm_gui.theme_palette import SIZE_FOOTER_ITEM_SPACING
from spektrafilm_gui.widget_editors import BoolEditor, EnumEditor, FloatEditor, FloatTupleEditor, IntEditor, IntTupleEditor, ProfileEnumEditor
from spektrafilm_gui.widget_primitives import CollapsibleSection, normalize_ui_text as _normalize_ui_text
from spektrafilm_gui.widget_specs import GUI_SECTION_ENUMS, get_auxiliary_spec, get_button_spec, get_widget_spec


def _enum_values(enum_cls):
    return [member.value for member in enum_cls]


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
    role: str | None = None,
) -> QPushButton:
    button = QPushButton(text if preserve_case else _normalize_ui_text(text))
    if role is not None:
        button.setProperty('role', role)
    if tooltip:
        button.setToolTip(tooltip)
    button.clicked.connect(callback)
    return button


def _build_widget_label(section_name: str, field_name: str) -> QLabel:
    spec = get_widget_spec(section_name, field_name)
    label_text = spec.label or _format_label(field_name)
    label = QLabel(_normalize_ui_text(label_text))
    if spec.tooltip:
        label.setToolTip(spec.tooltip)
    return label


def _build_auxiliary_label(name: str) -> QLabel:
    spec = get_auxiliary_spec(name)
    label_text = spec.label or name.replace("_", " ")
    label = QLabel(_normalize_ui_text(label_text))
    if spec.tooltip:
        label.setToolTip(spec.tooltip)
    return label


def _spec_row(section_name: str, field_name: str, widget: QWidget) -> tuple[QLabel, QWidget]:
    return _build_widget_label(section_name, field_name), widget


def _compound_spec_row(section_name: str, label_field_name: str, *widgets: QWidget) -> tuple[QLabel, QWidget]:
    return _build_widget_label(section_name, label_field_name), _build_inline_container(*widgets, stretch_last=True)


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


def _build_inline_container(
    *widgets: QWidget,
    spacing: int = 6,
    add_stretch: bool = False,
    stretch_last: bool = False,
) -> QWidget:
    container = QWidget()
    container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    layout = QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(spacing)
    last_widget_index = len(widgets) - 1
    for index, widget in enumerate(widgets):
        if stretch_last and index == last_widget_index:
            size_policy = widget.sizePolicy()
            size_policy.setHorizontalPolicy(QSizePolicy.Expanding)
            widget.setSizePolicy(size_policy)
            layout.addWidget(widget, 1)
        else:
            layout.addWidget(widget)
    if add_stretch and not stretch_last:
        layout.addStretch(1)
    return container


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
        hidden_fields: set[str] | None = None,
        collapsed_by_default: bool = False,
    ):
        super().__init__()
        self._state_cls = state_cls
        self._section_name = section_name
        self._title = title
        self._enum_fields = enum_fields or {}
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
                form.addRow(_build_widget_label(self._section_name, field_name), self._row_widget(field_name, widget))
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

    def _row_widget(self, field_name: str, widget: QWidget) -> QWidget:
        del field_name
        return widget

    def _build_editor(self, field_name: str, annotation: Any) -> QWidget:
        spec = get_widget_spec(self._section_name, field_name)
        enum_cls = self._enum_fields.get(field_name)
        if enum_cls is not None:
            if self._section_name == 'simulation' and field_name in {'film_stock', 'print_paper'}:
                return ProfileEnumEditor(_enum_values(enum_cls))
            return EnumEditor(_enum_values(enum_cls))
        if annotation is bool:
            return BoolEditor()
        if annotation is int:
            return IntEditor()
        if annotation is float:
            return FloatEditor(decimals=2 if spec.decimals is None else spec.decimals)
        if get_origin(annotation) is tuple:
            element_types = get_args(annotation)
            if element_types and all(element_type is int for element_type in element_types):
                return IntTupleEditor(len(element_types))
            return FloatTupleEditor(len(element_types), decimals=2 if spec.decimals is None else spec.decimals)
        raise TypeError(f"Unsupported field type for {self._state_cls.__name__}.{field_name}: {annotation!r}")

    def _apply_specs(self) -> None:
        for field_info in fields(self._state_cls):
            field_name = field_info.name
            spec = get_widget_spec(self._section_name, field_name)
            widget = getattr(self, field_name)
            if spec.tooltip:
                widget.setToolTip(spec.tooltip)
            if spec.min_value is not None:
                self._apply_numeric_attr(widget, 'setMinimum', spec.min_value)
            if spec.max_value is not None:
                self._apply_numeric_attr(widget, 'setMaximum', spec.max_value)
            if spec.step is not None:
                self._apply_numeric_attr(widget, 'setSingleStep', spec.step)

    @staticmethod
    def _apply_numeric_attr(widget: QWidget, method_name: str, value: float | int) -> None:
        method = getattr(widget, method_name, None)
        if callable(method):
            method(value)
            return
        editors = getattr(widget, '_editors', None)
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
    HIDDEN_FIELDS: set[str] = set()

    def __init__(self):
        super().__init__(
            state_cls=self.STATE_CLS,
            section_name=self.SECTION_NAME,
            title=self.TITLE,
            enum_fields=GUI_SECTION_ENUMS[self.ENUM_FIELDS_KEY] if self.ENUM_FIELDS_KEY is not None else None,
            hidden_fields=self.HIDDEN_FIELDS,
            collapsed_by_default=self.COLLAPSED_BY_DEFAULT,
        )


class InputImageSection(SimpleDataclassSection):
    STATE_CLS = InputImageState
    SECTION_NAME = 'input_image'
    TITLE = 'Input'
    ENUM_FIELDS_KEY = 'input_image'
    HIDDEN_FIELDS = {
        'upscale_factor',
        'crop',
        'crop_center',
        'crop_size',
        'spectral_upsampling_method',
        'filter_uv',
        'filter_ir',
    }

    def __init__(self, filepicker_section: 'FilePickerSection'):
        self._filepicker_section = filepicker_section
        super().__init__()


class LoadRawSection(DataclassSection):
    load_requested = Signal(str)

    def __init__(self):
        super().__init__(
            state_cls=LoadRawState,
            section_name='load_raw',
            title='Import Raw',
            enum_fields=GUI_SECTION_ENUMS['load_raw'],
            collapsed_by_default=True,
        )

    def _init_extra_widgets(self) -> None:
        self.file_path = QLineEdit()
        self.file_path.setReadOnly(True)
        self.file_path.setPlaceholderText(_normalize_ui_text('No raw selected'))
        self.reprocess_button = _build_button('reprocess raw', self._reprocess_raw, role='compactAction')
        self.reprocess_button.setEnabled(False)

    def _add_extra_rows_before(self, form: QFormLayout) -> None:
        browse_button = _build_button(
            'Select file',
            self._choose_file,
            tooltip='Load and process a raw file using rawpy, output colorspace and cctf as defined in current input widget state',
            role='compactAction',
        )
        form.addRow(_build_vertical_container(_build_button_row(self.file_path, browse_button, spacing=4), spacing=0))

    def _add_extra_rows_after(self, form: QFormLayout) -> None:
        form.addRow(self.reprocess_button)

    def _choose_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, _normalize_ui_text('Select input raw'), load_dialog_dir('raw_input'))
        if not path:
            return
        save_dialog_dir('raw_input', str(Path(path).parent))
        self.set_path(path)
        self.load_requested.emit(path)

    def _reprocess_raw(self) -> None:
        path = self.file_path.text().strip()
        if not path:
            return
        self.load_requested.emit(path)

    def set_path(self, path: str) -> None:
        self.file_path.setText(path)
        self.reprocess_button.setEnabled(bool(path.strip()))


class PreviewCropSection(QWidget):
    def __init__(self, input_image_section: InputImageSection):
        super().__init__()
        self.setLayout(
            _build_linked_form_section(
                'Crop and upscale',
                [
                    _spec_row('input_image', 'upscale_factor', input_image_section.upscale_factor),
                    _spec_row('input_image', 'crop', input_image_section.crop),
                    _spec_row('input_image', 'crop_center', input_image_section.crop_center),
                    _spec_row('input_image', 'crop_size', input_image_section.crop_size),
                ],
                expanded=False,
            ),
        )


class GrainSection(SimpleDataclassSection):
    STATE_CLS = GrainState
    SECTION_NAME = 'grain'
    TITLE = 'Grain'


class PreflashingSection(SimpleDataclassSection):
    STATE_CLS = PreflashingState
    SECTION_NAME = 'preflashing'
    TITLE = 'Preflash'


class HalationSection(SimpleDataclassSection):
    STATE_CLS = HalationState
    SECTION_NAME = 'halation'
    TITLE = 'Halation'


class CouplersSection(SimpleDataclassSection):
    STATE_CLS = CouplersState
    SECTION_NAME = 'couplers'
    TITLE = 'Couplers'


class GlareSection(SimpleDataclassSection):
    STATE_CLS = GlareState
    SECTION_NAME = 'glare'
    TITLE = 'Glare'


class SpecialSection(DataclassSection):
    def __init__(self, simulation_section: 'SimulationSection'):
        self._simulation_section = simulation_section
        super().__init__(
            state_cls=SpecialState,
            section_name='special',
            title='Experimental',
            collapsed_by_default=True,
            hidden_fields={
                'film_gamma_factor',
                'print_gamma_factor',
            },
        )

    def _add_extra_rows_before(self, form: QFormLayout) -> None:
        form.addRow(_build_widget_label('simulation', 'print_illuminant'), self._simulation_section.print_illuminant)


class SpectralUpsamplingSection(QWidget):
    def __init__(self, input_image_section: InputImageSection):
        super().__init__()
        self.setLayout(
            _build_linked_form_section(
                'Spectral upsampling',
                [
                    _spec_row('input_image', 'spectral_upsampling_method', input_image_section.spectral_upsampling_method),
                    _spec_row('input_image', 'filter_uv', input_image_section.filter_uv),
                    _spec_row('input_image', 'filter_ir', input_image_section.filter_ir),
                ],
                expanded=False,
            ),
        )


class TuneSection(QWidget):
    def __init__(self, special_section: SpecialSection):
        super().__init__()
        self.setLayout(
            _build_linked_form_section(
                'Tune',
                [
                    _spec_row('special', 'film_gamma_factor', special_section.film_gamma_factor),
                    _spec_row('special', 'print_gamma_factor', special_section.print_gamma_factor),
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
        self.file_path.setPlaceholderText(_normalize_ui_text('No image selected'))

        browse_button = _build_button('Select file', self._choose_file, role='compactAction')
        content = _build_vertical_container(_build_button_row(self.file_path, browse_button, spacing=4), spacing=6)
        _set_single_collapsible_layout(self, 'Import RGB', content)

    def _choose_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, _normalize_ui_text('Select input image'), load_dialog_dir('rgb_input'))
        if not path:
            return
        save_dialog_dir('rgb_input', str(Path(path).parent))
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
            'Save current as default',
            self.save_current_as_default_requested.emit,
        )
        self.save_current_to_file_button = _build_button(
            'Save current to file',
            self.save_current_to_file_requested.emit,
        )
        self.load_from_file_button = _build_button('Load from file', self.load_from_file_requested.emit)
        self.restore_factory_default_button = _build_button(
            'Restore factory default',
            self.restore_factory_default_requested.emit,
        )

        content = _build_vertical_container(
            _build_button_row(self.save_current_as_default_button, self.save_current_to_file_button),
            _build_button_row(self.load_from_file_button, self.restore_factory_default_button),
        )
        _set_single_collapsible_layout(self, 'GUI parameters', content, expanded=True)


class DisplaySection(SimpleDataclassSection):
    STATE_CLS = DisplayState
    SECTION_NAME = 'display'
    TITLE = 'Display'
    COLLAPSED_BY_DEFAULT = False
    ENUM_FIELDS_KEY = 'display'
    HIDDEN_FIELDS = {'preview_max_size'}

    update_preview_requested = Signal()

    def _init_extra_widgets(self) -> None:
        self.update_preview_button = _build_button(
            'update',
            self.update_preview_requested.emit,
            preserve_case=True,
            role='compactAction',
        )
        self.update_preview_button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)

    def _add_extra_rows_after(self, form: QFormLayout) -> None:
        form.addRow(
            _build_widget_label('display', 'preview_max_size'),
            _build_vertical_container(
                _build_button_row(self.preview_max_size, self.update_preview_button, spacing=4),
                spacing=0,
            ),
        )


class SimulationSection(DataclassSection):
    preview_requested = Signal()
    scan_requested = Signal()
    save_requested = Signal()
    _glare_section: 'GlareSection | None'
    _scan_for_print_restore_state: dict[str, object] | None

    def __init__(self):
        super().__init__(
            state_cls=SimulationState,
            section_name='simulation',
            title='Profiles',
            enum_fields=GUI_SECTION_ENUMS['simulation'],
            hidden_fields={
                'film_format_mm',
                'camera_lens_blur_um',
                'camera_diffusion_filter_active',
                'camera_diffusion_filter_family',
                'camera_diffusion_filter_strength',
                'camera_diffusion_filter_spatial_scale',
                'camera_diffusion_filter_halo_warmth',
                'camera_diffusion_filter_core_intensity',
                'camera_diffusion_filter_core_size',
                'camera_diffusion_filter_halo_intensity',
                'camera_diffusion_filter_halo_size',
                'camera_diffusion_filter_bloom_intensity',
                'camera_diffusion_filter_bloom_size',
                'exposure_compensation_ev',
                'auto_exposure',
                'auto_exposure_method',
                'print_exposure',
                'print_exposure_compensation',
                'print_y_filter_shift',
                'print_m_filter_shift',
                'diffusion_filter_active',
                'diffusion_filter_family',
                'diffusion_filter_strength',
                'diffusion_filter_spatial_scale',
                'diffusion_filter_halo_warmth',
                'diffusion_filter_core_intensity',
                'diffusion_filter_core_size',
                'diffusion_filter_halo_intensity',
                'diffusion_filter_halo_size',
                'diffusion_filter_bloom_intensity',
                'diffusion_filter_bloom_size',
                'print_illuminant',
                'scan_lens_blur',
                'scan_white_correction',
                'scan_white_level',
                'scan_black_correction',
                'scan_black_level',
                'scan_unsharp_mask',
                'auto_preview',
                'scan_film',
                'output_color_space',
                'saving_color_space',
                'saving_cctf_encoding',
            },
        )

    def _init_extra_widgets(self) -> None:
        self._glare_section = None
        self._scan_for_print_restore_state = None
        self.bottom_auto_preview = BoolEditor()
        self.bottom_scan_film = BoolEditor()
        self.bottom_scan_for_print = BoolEditor()
        scan_for_print_spec = get_auxiliary_spec('scan_for_print')
        if scan_for_print_spec.tooltip:
            self.bottom_scan_for_print.setToolTip(scan_for_print_spec.tooltip)
        self.bottom_scan_for_print.toggled.connect(self._apply_scan_for_print_mode)
        preview_button_spec = get_button_spec('preview')
        self.preview_button = _build_button(
            preview_button_spec.text,
            self.preview_requested.emit,
            tooltip=preview_button_spec.tooltip,
            preserve_case=preview_button_spec.preserve_case,
            role='accentAction',
        )
        scan_button_spec = get_button_spec('scan')
        self.scan_button = _build_button(
            scan_button_spec.text,
            self.scan_requested.emit,
            tooltip=scan_button_spec.tooltip,
            preserve_case=scan_button_spec.preserve_case,
            role='accentAction',
        )
        save_button_spec = get_button_spec('save')
        self.save_button = _build_button(
            save_button_spec.text,
            self.save_requested.emit,
            tooltip=save_button_spec.tooltip,
            preserve_case=save_button_spec.preserve_case,
            role='accentAction',
        )

        scan_film_row = QHBoxLayout()
        scan_film_row.setContentsMargins(0, 0, 0, 0)
        scan_film_row.setSpacing(SIZE_FOOTER_ITEM_SPACING)
        scan_film_row.addWidget(_build_widget_label('simulation', 'auto_preview'))
        scan_film_row.addWidget(self.bottom_auto_preview)
        scan_film_row.addSpacing(SIZE_FOOTER_ITEM_SPACING)
        scan_film_row.addWidget(_build_widget_label('simulation', 'scan_film'))
        scan_film_row.addWidget(self.bottom_scan_film)
        scan_film_row.addSpacing(SIZE_FOOTER_ITEM_SPACING)
        scan_film_row.addWidget(_build_auxiliary_label('scan_for_print'))
        scan_film_row.addWidget(self.bottom_scan_for_print)
        scan_film_row.addStretch(1)

        action_buttons = QWidget()
        action_buttons.setLayout(
            _build_button_row(
                self.preview_button,
                self.scan_button,
                self.save_button,
                stretch=1,
                spacing=SIZE_FOOTER_ITEM_SPACING,
            ),
        )

        self.bottom_bar = QWidget()
        bottom_bar_layout = QVBoxLayout(self.bottom_bar)
        bottom_bar_layout.setContentsMargins(0, 0, 0, 0)
        bottom_bar_layout.setSpacing(SIZE_FOOTER_ITEM_SPACING)
        bottom_bar_layout.addLayout(scan_film_row)
        bottom_bar_layout.addWidget(action_buttons)

    def action_bar(self) -> QWidget:
        return self.bottom_bar

    def set_auto_preview_value(self, value: bool) -> None:
        self.bottom_auto_preview.setChecked(value)

    def auto_preview_value(self) -> bool:
        return self.bottom_auto_preview.isChecked()

    def set_scan_film_value(self, value: bool) -> None:
        self.bottom_scan_film.setChecked(value)

    def scan_film_value(self) -> bool:
        return self.bottom_scan_film.isChecked()

    def bind_scan_for_print_glare_section(self, glare_section: 'GlareSection') -> None:
        self._glare_section = glare_section

    def reset_scan_for_print_value(self) -> None:
        was_blocked = self.bottom_scan_for_print.blockSignals(True)
        self.bottom_scan_for_print.setChecked(False)
        self.bottom_scan_for_print.blockSignals(was_blocked)
        self._scan_for_print_restore_state = None

    def _apply_scan_for_print_mode(self, active: bool) -> None:
        if active:
            if self._scan_for_print_restore_state is None:
                self._scan_for_print_restore_state = {
                    'scan_white_correction': self.scan_white_correction.value,
                    'scan_black_correction': self.scan_black_correction.value,
                    'glare_active': None if self._glare_section is None else self._glare_section.active.value,
                }
            self.scan_white_correction.value = True
            self.scan_black_correction.value = True
            if self._glare_section is not None:
                self._glare_section.active.value = False
            return

        restore_state = self._scan_for_print_restore_state
        if restore_state is None:
            return
        self.scan_white_correction.value = restore_state['scan_white_correction']
        self.scan_black_correction.value = restore_state['scan_black_correction']
        glare_active = restore_state['glare_active']
        if self._glare_section is not None and glare_active is not None:
            self._glare_section.active.value = glare_active
        self._scan_for_print_restore_state = None


class OutputSection(QWidget):
    def __init__(self, simulation_section: SimulationSection):
        super().__init__()
        self.setLayout(
            _build_linked_form_section(
                'Output',
                [
                    _spec_row('simulation', 'output_color_space', simulation_section.output_color_space),
                    _spec_row('simulation', 'saving_color_space', simulation_section.saving_color_space),
                    _spec_row('simulation', 'saving_cctf_encoding', simulation_section.saving_cctf_encoding),
                ],
                expanded=False,
            ),
        )


class ExposureControlSection(QWidget):
    def __init__(self, simulation_section: SimulationSection):
        super().__init__()
        form = _new_form_layout()
        self._add_spec_row(form, 'simulation', 'auto_exposure', simulation_section.auto_exposure)
        self._add_spec_row(form, 'simulation', 'exposure_compensation_ev', simulation_section.exposure_compensation_ev)
        self._add_spec_row(form, 'simulation', 'print_exposure_compensation', simulation_section.print_exposure_compensation)
        self._add_spec_row(form, 'simulation', 'print_exposure', simulation_section.print_exposure)

        self.setLayout(_build_collapsible_form_section('Exposure control', form, expanded=True))

    def _add_spec_row(self, form: QFormLayout, section_name: str, field_name: str, widget: QWidget) -> None:
        spec = get_widget_spec(section_name, field_name)
        if spec.tooltip:
            widget.setToolTip(spec.tooltip)
        form.addRow(_build_widget_label(section_name, field_name), widget)


class EnlargerSection(QWidget):
    def __init__(self, simulation_section: SimulationSection):
        super().__init__()
        self.setLayout(
            _build_linked_form_section(
                'Enlarger',
                [
                    _spec_row('simulation', 'print_y_filter_shift', simulation_section.print_y_filter_shift),
                    _spec_row('simulation', 'print_m_filter_shift', simulation_section.print_m_filter_shift),
                ],
                expanded=True,
            ),
        )


class DiffusionSection(QWidget):
    def __init__(self, simulation_section: SimulationSection):
        super().__init__()
        self.setLayout(
            _build_linked_form_section(
                'Diffusion',
                [
                    _spec_row('simulation', 'diffusion_filter_active', simulation_section.diffusion_filter_active),
                    _spec_row('simulation', 'diffusion_filter_family', simulation_section.diffusion_filter_family),
                    _spec_row('simulation', 'diffusion_filter_strength', simulation_section.diffusion_filter_strength),
                    _spec_row('simulation', 'diffusion_filter_spatial_scale', simulation_section.diffusion_filter_spatial_scale),
                    _spec_row('simulation', 'diffusion_filter_halo_warmth', simulation_section.diffusion_filter_halo_warmth),
                    _spec_row('simulation', 'diffusion_filter_core_intensity', simulation_section.diffusion_filter_core_intensity),
                    _spec_row('simulation', 'diffusion_filter_core_size', simulation_section.diffusion_filter_core_size),
                    _spec_row('simulation', 'diffusion_filter_halo_intensity', simulation_section.diffusion_filter_halo_intensity),
                    _spec_row('simulation', 'diffusion_filter_halo_size', simulation_section.diffusion_filter_halo_size),
                    _spec_row('simulation', 'diffusion_filter_bloom_intensity', simulation_section.diffusion_filter_bloom_intensity),
                    _spec_row('simulation', 'diffusion_filter_bloom_size', simulation_section.diffusion_filter_bloom_size),
                ],
                expanded=False,
            ),
        )


class CameraDiffusionSection(QWidget):
    def __init__(self, simulation_section: SimulationSection):
        super().__init__()
        self.setLayout(
            _build_linked_form_section(
                'Diffusion',
                [
                    _spec_row('simulation', 'camera_diffusion_filter_active', simulation_section.camera_diffusion_filter_active),
                    _spec_row('simulation', 'camera_diffusion_filter_family', simulation_section.camera_diffusion_filter_family),
                    _spec_row('simulation', 'camera_diffusion_filter_strength', simulation_section.camera_diffusion_filter_strength),
                    _spec_row('simulation', 'camera_diffusion_filter_spatial_scale', simulation_section.camera_diffusion_filter_spatial_scale),
                    _spec_row('simulation', 'camera_diffusion_filter_halo_warmth', simulation_section.camera_diffusion_filter_halo_warmth),
                    _spec_row('simulation', 'camera_diffusion_filter_core_intensity', simulation_section.camera_diffusion_filter_core_intensity),
                    _spec_row('simulation', 'camera_diffusion_filter_core_size', simulation_section.camera_diffusion_filter_core_size),
                    _spec_row('simulation', 'camera_diffusion_filter_halo_intensity', simulation_section.camera_diffusion_filter_halo_intensity),
                    _spec_row('simulation', 'camera_diffusion_filter_halo_size', simulation_section.camera_diffusion_filter_halo_size),
                    _spec_row('simulation', 'camera_diffusion_filter_bloom_intensity', simulation_section.camera_diffusion_filter_bloom_intensity),
                    _spec_row('simulation', 'camera_diffusion_filter_bloom_size', simulation_section.camera_diffusion_filter_bloom_size),
                ],
                expanded=False,
            ),
        )


class ScannerSection(QWidget):
    def __init__(self, simulation_section: SimulationSection):
        super().__init__()
        self.setLayout(
            _build_linked_form_section(
                'Scanner',
                [
                    _spec_row('simulation', 'scan_lens_blur', simulation_section.scan_lens_blur),
                    _compound_spec_row(
                        'simulation',
                        'scan_white_correction',
                        simulation_section.scan_white_correction,
                        simulation_section.scan_white_level,
                    ),
                    _compound_spec_row(
                        'simulation',
                        'scan_black_correction',
                        simulation_section.scan_black_correction,
                        simulation_section.scan_black_level,
                    ),
                    _spec_row('simulation', 'scan_unsharp_mask', simulation_section.scan_unsharp_mask),
                ],
                expanded=False,
            ),
        )


class CameraSection(QWidget):
    def __init__(self, simulation_section: SimulationSection):
        super().__init__()
        self.setLayout(
            _build_linked_form_section(
                'Camera',
                [
                    _spec_row('simulation', 'film_format_mm', simulation_section.film_format_mm),
                    _spec_row('simulation', 'auto_exposure_method', simulation_section.auto_exposure_method),
                    _spec_row('simulation', 'camera_lens_blur_um', simulation_section.camera_lens_blur_um),
                ],
                expanded=False,
            ),
        )