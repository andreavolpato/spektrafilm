from __future__ import annotations

from dataclasses import replace
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

from spektrafilm_gui.options import RawWhiteBalance
from spektrafilm_gui.params_manifest import (
    CROP_PANEL_FIELDS,
    DISPLAY_PANEL_FIELDS,
    INPUT_IMAGE_FIELDS,
    INPUT_PANEL_FIELDS,
    SIMULATION_ENLARGER_PANEL_FIELDS,
    SIMULATION_EXPOSURE_PANEL_FIELDS,
    SIMULATION_FIELDS,
    SIMULATION_OUTPUT_PANEL_FIELDS,
    SIMULATION_PROFILE_PANEL_FIELDS,
    SIMULATION_SPECIAL_BORROWED_FIELDS,
    SPECIAL_FIELDS,
    SPECTRAL_PANEL_FIELDS,
    GroupManifest,
    ParamSpec,
)
from spektrafilm_gui.persistence import load_dialog_dir, save_dialog_dir
from spektrafilm_gui.state import (
    PROJECT_DEFAULT_GUI_STATE,
    DisplayState,
    InputImageState,
    LoadRawState,
    SelectionState,
    SimulationState,
    SpecialState,
    clone_state_section,
)
from spektrafilm_gui.theme_palette import SIZE_FOOTER_ITEM_SPACING
from spektrafilm_gui.widget_editors import (
    BoolEditor,
    DevelopmentTimeEditor,
    EnumEditor,
    FloatEditor,
    FloatTupleEditor,
    IntEditor,
    IntTupleEditor,
    ProfileEnumEditor,
    StrEditor,
)
from spektrafilm_gui.widget_primitives import CollapsibleSection
from spektrafilm_gui.widget_primitives import normalize_ui_text as _normalize_ui_text

LOAD_RAW_FIELDS = (
    ParamSpec(
        "white_balance",
        label="White balance",
        tooltip=(
            "Leave at daylight (D65): it is the colorimetric reference the rest of "
            "the pipeline assumes, and should not be changed. Do not use it to "
            "neutralize a colour cast — fix white balance downstream with the "
            "enlarger filters instead. (custom exposes temperature/tint for special cases.)"
        ),
        enum=RawWhiteBalance,
    ),
    ParamSpec(
        "temperature",
        label="Temperature",
        tooltip="Temperature in Kelvin for the custom whitebalance, not used for the other white balance settings",
        min=1000,
        step=100,
    ),
    ParamSpec(
        "tint",
        label="Tint",
        tooltip="Tint value for the custom white balance, not used for the other white balance settings",
        min=0,
        step=0.01,
    ),
    ParamSpec(
        "lens_correction", label="Lens correction", tooltip="Apply lens corrections"
    ),
)

_LOAD_RAW_FIELD_SPECS = {spec.leaf: spec for spec in LOAD_RAW_FIELDS}
_SECTION_FIELD_SPECS = {
    "load_raw": _LOAD_RAW_FIELD_SPECS,
    "simulation": {spec.leaf: spec for spec in SIMULATION_FIELDS},
}
_AUXILIARY_FIELD_SPECS = {
    "scan_for_print": ParamSpec(
        "scan_for_print",
        label="Black and white correction",
        tooltip="White and black correction of the scanner are active, and glare is deactivated.",
    ),
}
_SIMULATION_ACTION_BUTTON_SPECS = {
    "preview": {
        "text": "PREVIEW",
        "tooltip": "run the simulation on a small preview and deactivates grain, halation, blurs, unsharp mask (diffusion filters are active)",
        "preserve_case": True,
    },
    "scan": {
        "text": "SCAN",
        "tooltip": "Run the full simulation on the full-resolution input",
        "preserve_case": True,
    },
    "save": {
        "text": "SAVE",
        "tooltip": "Save the current output layer to an image file",
        "preserve_case": True,
    },
}


def _enum_values(enum_cls):
    return [member.value for member in enum_cls]


def _field_spec(section_name: str, field_name: str) -> ParamSpec | None:
    return _SECTION_FIELD_SPECS.get(section_name, {}).get(field_name)


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
        button.setProperty("role", role)
    if tooltip:
        button.setToolTip(tooltip)
    button.clicked.connect(callback)
    return button


def _build_widget_label(section_name: str, field_name: str) -> QLabel:
    spec = _field_spec(section_name, field_name)
    label_text = (spec.label if spec is not None else None) or _format_label(field_name)
    label = QLabel(_normalize_ui_text(label_text))
    if spec is not None and spec.tooltip:
        label.setToolTip(spec.tooltip)
    return label


def _build_auxiliary_label(name: str) -> QLabel:
    spec = _AUXILIARY_FIELD_SPECS.get(name)
    label_text = (spec.label if spec is not None else None) or name.replace("_", " ")
    label = QLabel(_normalize_ui_text(label_text))
    if spec is not None and spec.tooltip:
        label.setToolTip(spec.tooltip)
    return label


def _build_button_row(
    *widgets: QWidget, stretch: int | None = None, spacing: int = 6
) -> QHBoxLayout:
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(spacing)
    for widget in widgets:
        if stretch is None:
            row.addWidget(widget)
        else:
            row.addWidget(widget, stretch)
    return row


def _build_vertical_container(
    *items: QHBoxLayout | QFormLayout | QWidget, spacing: int = 6
) -> QWidget:
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


def _set_single_collapsible_layout(
    widget: QWidget, title: str, content: QWidget, *, expanded: bool = True
) -> None:
    root = QVBoxLayout()
    root.setContentsMargins(0, 0, 0, 0)
    root.setSpacing(0)
    root.addWidget(
        CollapsibleSection(_normalize_ui_text(title), content, expanded=expanded)
    )
    widget.setLayout(root)


def _format_label(field_name: str) -> str:
    return _normalize_ui_text(field_name.replace("_", " "))


def _apply_numeric_attr(widget: QWidget, method_name: str, value: float | int) -> None:
    method = getattr(widget, method_name, None)
    if callable(method):
        method(value)
        return
    editors = getattr(widget, "_editors", None)
    if editors is not None:
        for editor in editors:
            getattr(editor, method_name)(value)


def _editor_from_param_spec(annotation: Any, spec: ParamSpec, *, label: str) -> QWidget:
    """Build an editor for a ParamSpec given the field's runtime annotation.

    Shared by the group-bound (ParamsGroupSection) and path-bound (input_image)
    sections so the type -> editor mapping lives in one place.
    """
    decimals = 2 if spec.decimals is None else spec.decimals
    if spec.leaf == "development_time":
        # Stock-dependent choices, so the dropdown starts empty (None only) and
        # the controller repopulates it on profile change.
        return DevelopmentTimeEditor()
    if spec.enum is not None:
        if spec.leaf in {"film_stock", "print_paper"}:
            editor = ProfileEnumEditor([member.value for member in spec.enum])
        else:
            editor = EnumEditor([member.value for member in spec.enum])
    elif annotation is bool:
        editor = BoolEditor()
    elif annotation is str:
        editor = StrEditor()
    elif annotation is int:
        editor = IntEditor()
    elif annotation is float:
        editor = FloatEditor(decimals=decimals)
    elif get_origin(annotation) is tuple:
        element_types = get_args(annotation)
        if element_types and all(element_type is int for element_type in element_types):
            editor = IntTupleEditor(len(element_types))
        else:
            editor = FloatTupleEditor(len(element_types), decimals=decimals)
    else:
        raise TypeError(f"Unsupported field type for {label}: {annotation!r}")
    if spec.min is not None:
        _apply_numeric_attr(editor, "setMinimum", spec.min)
    if spec.max is not None:
        _apply_numeric_attr(editor, "setMaximum", spec.max)
    if spec.step is not None:
        _apply_numeric_attr(editor, "setSingleStep", spec.step)
    return editor


def _path_annotation(root_cls: type, path: str) -> Any:
    """Resolve the type annotation of a dotted path on a dataclass tree."""
    cls = root_cls
    annotation: Any = root_cls
    for part in path.split("."):
        annotation = get_type_hints(cls)[part]
        cls = annotation
    return annotation


def _read_path(root: Any, path: str) -> Any:
    obj = root
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def _write_path(root: Any, path: str, value: Any) -> None:
    parts = path.split(".")
    obj = root
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def _build_path_panel(
    title: str,
    specs: tuple[ParamSpec, ...],
    editors: dict[str, QWidget],
    *,
    expanded: bool,
) -> QVBoxLayout:
    """Lay out a collapsible panel from shared, path-bound editors (by leaf)."""
    form = _new_form_layout()
    for spec in specs:
        editor = editors[spec.leaf]
        label = QLabel(_normalize_ui_text(spec.label or _format_label(spec.leaf)))
        if spec.tooltip:
            label.setToolTip(spec.tooltip)
            editor.setToolTip(spec.tooltip)
        form.addRow(label, editor)
    return _build_collapsible_form_section(title, form, expanded=expanded)


class InputImageSection(QWidget):
    """Owner of the input_image cluster (io.* + settings.*), rendered across
    three panels on two tabs (Input here; Crop and Spectral embed this section's
    editors). Path-bound: each editor binds to a dotted path on InputImageState.
    Marked _is_params_group so auto-preview wiring and profile-sync drive it via
    _editors / set_state, exactly like ParamsGroupSection.
    """

    _is_params_group = True

    def __init__(self, filepicker_section: FilePickerSection):
        super().__init__()
        self._filepicker_section = filepicker_section
        self._source: InputImageState | None = None
        self._specs = {spec.leaf: spec for spec in INPUT_IMAGE_FIELDS}
        self._editors: dict[str, QWidget] = {}
        for spec in INPUT_IMAGE_FIELDS:
            editor = _editor_from_param_spec(
                _path_annotation(InputImageState, spec.path),
                spec,
                label=spec.path,
            )
            self._editors[spec.leaf] = editor
            setattr(self, spec.leaf, editor)
        self.setLayout(
            _build_path_panel(
                "Input", INPUT_PANEL_FIELDS, self._editors, expanded=False
            )
        )

    def set_state(self, state: InputImageState) -> None:
        self._source = state
        for spec in self._specs.values():
            self._editors[spec.leaf].value = _read_path(state, spec.path)

    def get_state(self) -> InputImageState:
        state = (
            clone_state_section(self._source)
            if self._source is not None
            else InputImageState()
        )
        for spec in self._specs.values():
            _write_path(state, spec.path, self._editors[spec.leaf].value)
        return state


class LoadRawSection(QWidget):
    load_requested = Signal(str)
    SECTION_NAME = "load_raw"
    TITLE = "Import Raw"
    _TYPE_HINTS = get_type_hints(LoadRawState)
    _STATE_FIELD_NAMES = tuple(_TYPE_HINTS)

    def __init__(self):
        super().__init__()
        self._source: LoadRawState | None = None
        self.file_path = QLineEdit()
        self.file_path.setReadOnly(True)
        self.file_path.setPlaceholderText(_normalize_ui_text("No raw selected"))
        self.reprocess_button = _build_button(
            "reprocess raw", self._reprocess_raw, role="compactAction"
        )
        self.reprocess_button.setEnabled(False)
        self._build_ui()

    def _build_ui(self) -> None:
        for field_name in self._STATE_FIELD_NAMES:
            spec = _LOAD_RAW_FIELD_SPECS[field_name]
            editor = _editor_from_param_spec(
                self._TYPE_HINTS[field_name],
                spec,
                label=f"{self.SECTION_NAME}.{field_name}",
            )
            setattr(self, field_name, editor)

        form = _new_form_layout()
        browse_button = _build_button(
            "Select file",
            self._choose_file,
            tooltip="Load and process a raw file using rawpy, output colorspace and cctf as defined in current input widget state",
            role="compactAction",
        )
        form.addRow(
            _build_vertical_container(
                _build_button_row(self.file_path, browse_button, spacing=4), spacing=0
            )
        )
        for field_name in self._STATE_FIELD_NAMES:
            form.addRow(
                _build_widget_label(self.SECTION_NAME, field_name),
                getattr(self, field_name),
            )
        form.addRow(self.reprocess_button)

        self.setLayout(
            _build_collapsible_form_section(self.TITLE, form, expanded=False)
        )

    def _choose_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, _normalize_ui_text("Select input raw"), load_dialog_dir("raw_input")
        )
        if not path:
            return
        save_dialog_dir("raw_input", str(Path(path).parent))
        self.set_path(path)
        self.load_requested.emit(path)

    def _reprocess_raw(self) -> None:
        path = self.file_path.text().strip()
        if not path:
            return
        self.load_requested.emit(path)

    def set_state(self, state: LoadRawState) -> None:
        self._source = state
        for field_name in self._STATE_FIELD_NAMES:
            getattr(self, field_name).value = getattr(state, field_name)

    def get_state(self) -> LoadRawState:
        state = (
            clone_state_section(self._source)
            if self._source is not None
            else clone_state_section(PROJECT_DEFAULT_GUI_STATE.gui_only.load_raw)
        )
        for field_name in self._STATE_FIELD_NAMES:
            setattr(state, field_name, getattr(self, field_name).value)
        return state

    def set_path(self, path: str) -> None:
        self.file_path.setText(path)
        self.reprocess_button.setEnabled(bool(path.strip()))


class PreviewCropSection(QWidget):
    def __init__(self, input_image_section: InputImageSection):
        super().__init__()
        self.setLayout(
            _build_path_panel(
                "Crop and upscale",
                CROP_PANEL_FIELDS,
                input_image_section._editors,
                expanded=False,
            ),
        )


class ParamsGroupSection(QWidget):
    """A GUI panel bound directly to a runtime parameter group.

    Driven by a :class:`GroupManifest` keyed on ``RuntimePhotoParams``
    paths. Builds one editor per declared field (type inferred from the
    group dataclass), reads and writes the runtime group object directly
    via ``get_state`` / ``set_state``, and passes through any group
    fields the manifest does not declare. This replaces the per-panel
    mirror dataclass + dedicated presentation table + mapper block trio
    with a single declaration.
    """

    # Marks this as a path-bound group section so profile-sync drives it
    # through set_state (applying unit transforms) rather than assigning
    # raw runtime values field-by-field.
    _is_params_group = True

    # Emitted with an ActionSpec.action_id when one of this panel's buttons is
    # clicked; the app wires it to a controller handler.
    action_triggered = Signal(str)

    def __init__(self, manifest: GroupManifest):
        super().__init__()
        self._manifest = manifest
        self._group_cls = manifest.group_cls
        self._type_hints = get_type_hints(manifest.group_cls)
        self._source: Any = None
        self._editors: dict[str, QWidget] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        # Build editors for every field (so the whole group stays editable,
        # persisted, and auto-preview wired) regardless of how it is laid out.
        for spec in self._manifest.fields:
            editor = self._build_editor(spec)
            self._editors[spec.leaf] = editor
            setattr(self, spec.leaf, editor)

        # Only lay out panel_fields; any remaining fields are displayed by a
        # section that borrows them.
        panel_specs = self._manifest.panel_fields or self._manifest.fields
        if self._manifest.subsections:
            content = self._build_subsectioned_content(panel_specs)
        else:
            content = QWidget()
            content.setLayout(self._build_form(panel_specs))

        if self._manifest.actions:
            wrapper = QWidget()
            vbox = QVBoxLayout(wrapper)
            vbox.setContentsMargins(0, 0, 0, 0)
            vbox.addWidget(content)
            vbox.addWidget(self._build_actions())
            content = wrapper

        root = QVBoxLayout()
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(
            CollapsibleSection(
                self._manifest.title,
                content,
                expanded=not self._manifest.collapsed_by_default,
            ),
        )
        self.setLayout(root)

    def _build_form(self, specs) -> QFormLayout:
        """A form laying out the editors (already built) for ``specs``, in order."""
        form = _new_form_layout()
        for spec in specs:
            editor = self._editors[spec.leaf]
            label = QLabel(_normalize_ui_text(spec.label or _format_label(spec.leaf)))
            if spec.tooltip:
                label.setToolTip(spec.tooltip)
                editor.setToolTip(spec.tooltip)
            form.addRow(label, editor)
        return form

    def _build_subsectioned_content(self, panel_specs) -> QWidget:
        """Lay the panel out as loose fields (top) followed by one nested
        collapsible per :class:`SubSection`. Subsections reference fields by leaf
        name; anything not grouped renders loose at the top (e.g. ``active``)."""
        by_leaf = {spec.leaf: spec for spec in panel_specs}
        grouped = {
            name for sub in self._manifest.subsections for name in sub.field_names
        }

        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setAlignment(Qt.AlignTop)

        loose = [spec for spec in panel_specs if spec.leaf not in grouped]
        if loose:
            loose_widget = QWidget()
            loose_widget.setLayout(self._build_form(loose))
            vbox.addWidget(loose_widget)

        for sub in self._manifest.subsections:
            specs = [by_leaf[name] for name in sub.field_names if name in by_leaf]
            sub_content = QWidget()
            sub_content.setLayout(self._build_form(specs))
            vbox.addWidget(
                CollapsibleSection(
                    sub.title, sub_content, expanded=sub.expanded, variant="subsection"
                )
            )
        return container

    def _build_editor(self, spec: ParamSpec) -> QWidget:
        annotation = self._type_hints[spec.leaf]
        return _editor_from_param_spec(
            annotation, spec, label=f"{self._group_cls.__name__}.{spec.leaf}"
        )

    def _build_actions(self) -> QWidget:
        bar = QWidget()
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(0, 4, 0, 0)
        for action in self._manifest.actions:
            button = QPushButton(_normalize_ui_text(action.label))
            if action.tooltip:
                button.setToolTip(action.tooltip)
            button.clicked.connect(
                lambda _checked=False, aid=action.action_id: self.action_triggered.emit(
                    aid
                )
            )
            layout.addWidget(button)
        layout.addStretch(1)
        return bar

    def set_field(self, leaf: str, value: Any) -> None:
        """Set a single field's editor value (e.g. write a fitted calibration
        string back from a controller action)."""
        self._editors[leaf].value = value

    def set_state(self, group: Any) -> None:
        self._source = group
        for leaf, editor in self._editors.items():
            editor.value = getattr(group, leaf)

    def get_state(self) -> Any:
        base = self._source if self._source is not None else self._group_cls()
        overrides = {leaf: editor.value for leaf, editor in self._editors.items()}
        return replace(base, **overrides)

    def set_development_time_choices(self, times: Any) -> None:
        """Repopulate the development-time dropdown (if this group has one) with
        the loaded stock's available development times. No-op otherwise."""
        editor = self._editors.get("development_time")
        if isinstance(editor, DevelopmentTimeEditor):
            editor.set_choices(None if times is None else [float(t) for t in times])


class SpecialSection(QWidget):
    _is_params_group = True

    def __init__(self, simulation_section: SimulationSection):
        super().__init__()
        self._simulation_section = simulation_section
        self._source: SpecialState | None = None
        self._specs = {spec.leaf: spec for spec in SPECIAL_FIELDS}
        self._editors: dict[str, QWidget] = {}
        form = _new_form_layout()
        borrowed = SIMULATION_SPECIAL_BORROWED_FIELDS[0]
        borrowed_label = QLabel(
            _normalize_ui_text(borrowed.label or _format_label(borrowed.leaf))
        )
        if borrowed.tooltip:
            borrowed_label.setToolTip(borrowed.tooltip)
        form.addRow(borrowed_label, simulation_section.print_illuminant)
        for spec in SPECIAL_FIELDS:
            editor = _editor_from_param_spec(
                _path_annotation(SpecialState, spec.path), spec, label=spec.path
            )
            self._editors[spec.leaf] = editor
            setattr(self, spec.leaf, editor)
            label = QLabel(_normalize_ui_text(spec.label or _format_label(spec.leaf)))
            if spec.tooltip:
                label.setToolTip(spec.tooltip)
                editor.setToolTip(spec.tooltip)
            form.addRow(label, editor)
        self.setLayout(
            _build_collapsible_form_section("Experimental", form, expanded=False)
        )

    def set_state(self, state: SpecialState) -> None:
        self._source = state
        for spec in self._specs.values():
            self._editors[spec.leaf].value = _read_path(state, spec.path)

    def get_state(self) -> SpecialState:
        state = (
            clone_state_section(self._source)
            if self._source is not None
            else SpecialState(
                film_channel_swap=(0, 1, 2),
                print_channel_swap=(0, 1, 2),
            )
        )
        for spec in self._specs.values():
            _write_path(state, spec.path, self._editors[spec.leaf].value)
        return state


class SpectralUpsamplingSection(QWidget):
    def __init__(self, input_image_section: InputImageSection):
        super().__init__()
        self.setLayout(
            _build_path_panel(
                "Spectral upsampling",
                SPECTRAL_PANEL_FIELDS,
                input_image_section._editors,
                expanded=False,
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

        browse_button = _build_button(
            "Select file", self._choose_file, role="compactAction"
        )
        content = _build_vertical_container(
            _build_button_row(self.file_path, browse_button, spacing=4), spacing=6
        )
        _set_single_collapsible_layout(self, "Import RGB", content, expanded=False)

    def _choose_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, _normalize_ui_text("Select input image"), load_dialog_dir("rgb_input")
        )
        if not path:
            return
        save_dialog_dir("rgb_input", str(Path(path).parent))
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
        self.load_from_file_button = _build_button(
            "Load from file", self.load_from_file_requested.emit
        )
        self.restore_factory_default_button = _build_button(
            "Restore factory default",
            self.restore_factory_default_requested.emit,
        )

        content = _build_vertical_container(
            _build_button_row(
                self.save_current_as_default_button, self.save_current_to_file_button
            ),
            _build_button_row(
                self.load_from_file_button, self.restore_factory_default_button
            ),
        )
        _set_single_collapsible_layout(self, "GUI parameters", content, expanded=True)


class DisplaySection(QWidget):
    _is_params_group = True
    _skip_auto_preview_leaves = {"preview_max_size", "output_interpolation"}
    update_preview_requested = Signal()

    def __init__(self):
        super().__init__()
        self._source: DisplayState | None = None
        self._specs = {spec.leaf: spec for spec in DISPLAY_PANEL_FIELDS}
        self._editors: dict[str, QWidget] = {}
        self.update_preview_button = _build_button(
            "update",
            self.update_preview_requested.emit,
            preserve_case=True,
            role="compactAction",
        )
        self.update_preview_button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        form = _new_form_layout()
        for spec in DISPLAY_PANEL_FIELDS:
            editor = _editor_from_param_spec(
                _path_annotation(DisplayState, spec.path), spec, label=spec.path
            )
            self._editors[spec.leaf] = editor
            setattr(self, spec.leaf, editor)
            label = QLabel(_normalize_ui_text(spec.label or _format_label(spec.leaf)))
            if spec.tooltip:
                label.setToolTip(spec.tooltip)
                editor.setToolTip(spec.tooltip)
            widget: QWidget = editor
            if spec.leaf == "preview_max_size":
                widget = _build_vertical_container(
                    _build_button_row(editor, self.update_preview_button, spacing=4),
                    spacing=0,
                )
            form.addRow(label, widget)
        self.setLayout(_build_collapsible_form_section("Display", form, expanded=True))

    def set_state(self, state: DisplayState) -> None:
        self._source = state
        for spec in self._specs.values():
            self._editors[spec.leaf].value = _read_path(state, spec.path)

    def get_state(self) -> DisplayState:
        state = (
            clone_state_section(self._source)
            if self._source is not None
            else DisplayState(
                use_display_transform=True,
                gray_18_canvas=True,
                white_padding=0.03,
                output_interpolation="spline36",
            )
        )
        for spec in self._specs.values():
            _write_path(state, spec.path, self._editors[spec.leaf].value)
        return state


class SimulationSection(QWidget):
    _is_params_group = True
    _skip_auto_preview_leaves = {"auto_preview"}
    preview_requested = Signal()
    scan_requested = Signal()
    save_requested = Signal()
    _glare_section: ParamsGroupSection | None
    _scanner_section: ParamsGroupSection | None
    _scan_for_print_restore_state: dict[str, object] | None

    def __init__(self):
        super().__init__()
        self._source: SimulationState | None = None
        self._specs = {spec.leaf: spec for spec in SIMULATION_FIELDS}
        self._editors: dict[str, QWidget] = {}
        self._glare_section = None
        self._scanner_section = None
        self._scan_for_print_restore_state = None
        for spec in SIMULATION_FIELDS:
            editor = _editor_from_param_spec(
                _path_annotation(SimulationState, spec.path), spec, label=spec.path
            )
            self._editors[spec.leaf] = editor
            setattr(self, spec.leaf, editor)
        self.bottom_auto_preview = self.auto_preview
        self.bottom_workflow = self.route
        self.bottom_scan_for_print = BoolEditor()
        scan_for_print_spec = _AUXILIARY_FIELD_SPECS["scan_for_print"]
        if scan_for_print_spec.tooltip:
            self.bottom_scan_for_print.setToolTip(scan_for_print_spec.tooltip)
        self.bottom_scan_for_print.toggled.connect(self._apply_scan_for_print_mode)
        preview_button_spec = _SIMULATION_ACTION_BUTTON_SPECS["preview"]
        self.preview_button = _build_button(
            preview_button_spec["text"],
            self.preview_requested.emit,
            tooltip=preview_button_spec["tooltip"],
            preserve_case=preview_button_spec["preserve_case"],
            role="accentAction",
        )
        scan_button_spec = _SIMULATION_ACTION_BUTTON_SPECS["scan"]
        self.scan_button = _build_button(
            scan_button_spec["text"],
            self.scan_requested.emit,
            tooltip=scan_button_spec["tooltip"],
            preserve_case=scan_button_spec["preserve_case"],
            role="accentAction",
        )
        save_button_spec = _SIMULATION_ACTION_BUTTON_SPECS["save"]
        self.save_button = _build_button(
            save_button_spec["text"],
            self.save_requested.emit,
            tooltip=save_button_spec["tooltip"],
            preserve_case=save_button_spec["preserve_case"],
            role="accentAction",
        )

        toggles_row = QHBoxLayout()
        toggles_row.setContentsMargins(0, 0, 0, 0)
        toggles_row.setSpacing(SIZE_FOOTER_ITEM_SPACING)
        toggles_row.addWidget(_build_widget_label("simulation", "auto_preview"))
        toggles_row.addWidget(self.bottom_auto_preview)
        toggles_row.addSpacing(SIZE_FOOTER_ITEM_SPACING)
        toggles_row.addWidget(_build_auxiliary_label("scan_for_print"))
        toggles_row.addWidget(self.bottom_scan_for_print)
        toggles_row.addStretch(1)

        workflow_row = QHBoxLayout()
        workflow_row.setContentsMargins(0, 0, 0, 0)
        workflow_row.setSpacing(SIZE_FOOTER_ITEM_SPACING)
        workflow_row.addWidget(_build_widget_label("simulation", "route"))
        workflow_row.addWidget(self.bottom_workflow, 1)

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
        bottom_bar_layout.addLayout(toggles_row)
        bottom_bar_layout.addLayout(workflow_row)
        bottom_bar_layout.addWidget(action_buttons)
        self.setLayout(
            _build_path_panel(
                "Profiles",
                SIMULATION_PROFILE_PANEL_FIELDS,
                self._editors,
                expanded=True,
            )
        )

    def set_state(self, state: SimulationState) -> None:
        self._source = state
        for spec in self._specs.values():
            self._editors[spec.leaf].value = _read_path(state, spec.path)

    def get_state(self) -> SimulationState:
        state = (
            clone_state_section(self._source)
            if self._source is not None
            else SimulationState(
                selection=SelectionState(film_stock="", print_paper=""),
            )
        )
        for spec in self._specs.values():
            _write_path(state, spec.path, self._editors[spec.leaf].value)
        return state

    def action_bar(self) -> QWidget:
        return self.bottom_bar

    def set_auto_preview_value(self, value: bool) -> None:
        self.bottom_auto_preview.setChecked(value)

    def auto_preview_value(self) -> bool:
        return self.bottom_auto_preview.isChecked()

    def bind_scan_for_print_sections(
        self,
        *,
        glare: ParamsGroupSection,
        scanner: ParamsGroupSection,
    ) -> None:
        self._glare_section = glare
        self._scanner_section = scanner

    def reset_scan_for_print_value(self) -> None:
        was_blocked = self.bottom_scan_for_print.blockSignals(True)
        self.bottom_scan_for_print.setChecked(False)
        self.bottom_scan_for_print.blockSignals(was_blocked)
        self._scan_for_print_restore_state = None

    def _apply_scan_for_print_mode(self, active: bool) -> None:
        scanner = self._scanner_section
        if scanner is None:
            return
        if active:
            if self._scan_for_print_restore_state is None:
                self._scan_for_print_restore_state = {
                    "white_correction": scanner.white_correction.value,
                    "black_correction": scanner.black_correction.value,
                    "glare_active": None
                    if self._glare_section is None
                    else self._glare_section.active.value,
                }
            scanner.white_correction.value = True
            scanner.black_correction.value = True
            if self._glare_section is not None:
                self._glare_section.active.value = False
            return

        restore_state = self._scan_for_print_restore_state
        if restore_state is None:
            return
        scanner.white_correction.value = restore_state["white_correction"]
        scanner.black_correction.value = restore_state["black_correction"]
        glare_active = restore_state["glare_active"]
        if self._glare_section is not None and glare_active is not None:
            self._glare_section.active.value = glare_active
        self._scan_for_print_restore_state = None


class OutputSection(QWidget):
    def __init__(self, simulation_section: SimulationSection):
        super().__init__()
        self.setLayout(
            _build_path_panel(
                "Output",
                SIMULATION_OUTPUT_PANEL_FIELDS,
                simulation_section._editors,
                expanded=False,
            ),
        )


class EnlargerSection(QWidget):
    def __init__(self, simulation_section: SimulationSection):
        super().__init__()
        # Print exposure controls and filter shifts are both owned by
        # simulation_section; render the exposure pair on top of the filter
        # shifts.
        fields = SIMULATION_EXPOSURE_PANEL_FIELDS + SIMULATION_ENLARGER_PANEL_FIELDS
        self.setLayout(
            _build_path_panel(
                "Enlarger", fields, simulation_section._editors, expanded=True
            ),
        )
