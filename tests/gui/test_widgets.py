from __future__ import annotations

from dataclasses import fields
import os
from types import SimpleNamespace
from typing import get_origin, get_type_hints

import numpy as np

from spektrafilm_gui import icons as icons_module
from spektrafilm_gui import params_manifest as param_manifest_module
from spektrafilm_gui import widget_primitives as primitives_module
from spektrafilm_gui import widget_sections as widgets_module
from spektrafilm_gui import widget_editors as widget_editors_module
from spektrafilm_gui import state as state_module


class FakeLineEdit:
    def __init__(self) -> None:
        self._text = ''
        self.read_only = False
        self.placeholder_text = None

    def setReadOnly(self, value: bool) -> None:  # noqa: N802 - Qt API name
        self.read_only = value

    def setPlaceholderText(self, text: str) -> None:  # noqa: N802 - Qt API name
        self.placeholder_text = text

    def setText(self, text: str) -> None:  # noqa: N802 - Qt API name
        self._text = text

    def text(self) -> str:  # noqa: N802 - Qt API name
        return self._text


class FakeButton:
    def __init__(self, text: str, callback, *, tooltip: str | None = None) -> None:
        self.text = text
        self.callback = callback
        self.tooltip = tooltip
        self.enabled = True

    def setEnabled(self, value: bool) -> None:  # noqa: N802 - Qt API name
        self.enabled = value

    def click(self) -> None:
        self.callback()


class FakeSignal:
    def __init__(self) -> None:
        self.emitted: list[tuple[str]] = []

    def emit(self, path: str) -> None:
        self.emitted.append((path,))


class FakeNoArgSignal:
    def __init__(self) -> None:
        self.emit_count = 0

    def emit(self) -> None:
        self.emit_count += 1


class FakeComboBox:
    def __init__(self) -> None:
        self.items: list[str] = []
        self.current_index = -1
        self.blocked_calls: list[bool] = []

    def blockSignals(self, blocked: bool) -> None:  # noqa: N802 - Qt API name
        self.blocked_calls.append(blocked)

    def clear(self) -> None:  # noqa: N802 - Qt API name
        self.items.clear()
        self.current_index = -1

    def addItems(self, items: list[str]) -> None:  # noqa: N802 - Qt API name
        self.items.extend(items)
        if self.items and self.current_index < 0:
            self.current_index = 0

    def findText(self, text: str) -> int:  # noqa: N802 - Qt API name
        try:
            return self.items.index(text)
        except ValueError:
            return -1

    def setCurrentIndex(self, index: int) -> None:  # noqa: N802 - Qt API name
        self.current_index = index

    def count(self) -> int:  # noqa: N802 - Qt API name
        return len(self.items)

    def currentText(self) -> str:  # noqa: N802 - Qt API name
        if self.current_index < 0:
            return ''
        return self.items[self.current_index]


class FakeForm:
    def __init__(self) -> None:
        self.rows: list[tuple[object, ...]] = []

    def addRow(self, *args) -> None:  # noqa: N802 - Qt API name
        self.rows.append(args)


class FakeValueEditor:
    def __init__(self, value) -> None:
        self.value = value


class FakeCheckbox:
    def __init__(self, checked: bool = False) -> None:
        self._checked = checked
        self.enabled = True

    def setChecked(self, value: bool) -> None:  # noqa: N802 - Qt API name
        self._checked = value

    def isChecked(self) -> bool:  # noqa: N802 - Qt API name
        return self._checked

    def setEnabled(self, value: bool) -> None:  # noqa: N802 - Qt API name
        self.enabled = value


def _make_load_raw_section():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from qtpy import QtWidgets

    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    emitted: list[str] = []
    section = widgets_module.LoadRawSection()
    section.load_requested.connect(emitted.append)
    return section, emitted


def _make_filepicker_section(monkeypatch):
    build_ui = getattr(widgets_module.FilePickerSection, '_build_ui')
    choose_file = getattr(widgets_module.FilePickerSection, '_choose_file')
    created_buttons: list[FakeButton] = []
    monkeypatch.setattr(widgets_module, 'QLineEdit', FakeLineEdit)
    monkeypatch.setattr(
        widgets_module,
        '_build_button',
        lambda text, callback, **kwargs: created_buttons.append(FakeButton(text, callback, tooltip=kwargs.get('tooltip'))) or created_buttons[-1],
    )
    monkeypatch.setattr(widgets_module, '_build_button_row', lambda *widgets, **kwargs: ('button-row', widgets, kwargs))
    monkeypatch.setattr(widgets_module, '_build_vertical_container', lambda *items, **kwargs: ('vertical-container', items, kwargs))
    monkeypatch.setattr(widgets_module, '_set_single_collapsible_layout', lambda *args, **kwargs: None)
    section = SimpleNamespace(load_requested=FakeSignal())
    setattr(section, '_choose_file', lambda: choose_file(section))
    build_ui(section)
    return section, created_buttons


def _make_gui_config_section(monkeypatch):
    build_ui = getattr(widgets_module.GuiConfigSection, '_build_ui')
    created_buttons: list[FakeButton] = []
    monkeypatch.setattr(
        widgets_module,
        '_build_button',
        lambda text, callback, **kwargs: created_buttons.append(FakeButton(text, callback, tooltip=kwargs.get('tooltip'))) or created_buttons[-1],
    )
    monkeypatch.setattr(widgets_module, '_build_button_row', lambda *widgets, **kwargs: ('button-row', widgets, kwargs))
    monkeypatch.setattr(widgets_module, '_build_vertical_container', lambda *items, **kwargs: ('vertical-container', items, kwargs))
    monkeypatch.setattr(widgets_module, '_set_single_collapsible_layout', lambda *args, **kwargs: None)
    section = SimpleNamespace(
        save_current_as_default_requested=FakeNoArgSignal(),
        save_current_to_file_requested=FakeNoArgSignal(),
        load_from_file_requested=FakeNoArgSignal(),
        restore_factory_default_requested=FakeNoArgSignal(),
    )
    build_ui(section)
    return section, created_buttons


def test_load_raw_section_adds_reprocess_button_after_raw_settings(monkeypatch) -> None:
    del monkeypatch
    section, _emitted = _make_load_raw_section()

    assert section.file_path.isReadOnly() is True
    assert section.file_path.placeholderText() == 'no raw selected'
    assert section.reprocess_button.text() == 'reprocess raw'
    assert section.reprocess_button.isEnabled() is False
    assert hasattr(section, 'white_balance')
    assert hasattr(section, 'temperature')
    assert hasattr(section, 'tint')
    assert hasattr(section, 'lens_correction')


def test_load_raw_section_reprocess_button_uses_selected_path(monkeypatch) -> None:
    del monkeypatch
    section, emitted = _make_load_raw_section()

    widgets_module.LoadRawSection.set_path(section, 'C:/tmp/example.nef')
    section.reprocess_button.click()
    widgets_module.LoadRawSection.set_path(section, '')
    section.reprocess_button.click()

    assert section.file_path.text() == ''
    assert emitted == ['C:/tmp/example.nef']
    assert section.reprocess_button.isEnabled() is False


def test_file_picker_choose_file_updates_path_and_emits_selected_file(monkeypatch) -> None:
    section, created_buttons = _make_filepicker_section(monkeypatch)
    monkeypatch.setattr(
        widgets_module.QFileDialog,
        'getOpenFileName',
        staticmethod(lambda *_args, **_kwargs: ('C:/tmp/example.tif', 'Images (*.tif)')),
    )

    created_buttons[0].click()

    assert created_buttons[0].text == 'Select file'
    assert section.file_path.text() == 'C:/tmp/example.tif'
    assert section.load_requested.emitted == [('C:/tmp/example.tif',)]


def test_file_picker_choose_file_ignores_cancelled_dialog(monkeypatch) -> None:
    section, created_buttons = _make_filepicker_section(monkeypatch)
    section.file_path.setText('C:/tmp/previous.tif')
    monkeypatch.setattr(
        widgets_module.QFileDialog,
        'getOpenFileName',
        staticmethod(lambda *_args, **_kwargs: ('', '')),
    )

    created_buttons[0].click()

    assert section.file_path.text() == 'C:/tmp/previous.tif'
    assert section.load_requested.emitted == []


def test_gui_config_buttons_emit_expected_actions(monkeypatch) -> None:
    section, created_buttons = _make_gui_config_section(monkeypatch)

    section.save_current_as_default_button.click()
    section.save_current_to_file_button.click()
    section.load_from_file_button.click()
    section.restore_factory_default_button.click()

    assert [button.text for button in created_buttons] == [
        'Save current as default',
        'Save current to file',
        'Load from file',
        'Restore factory default',
    ]
    assert section.save_current_as_default_requested.emit_count == 1
    assert section.save_current_to_file_requested.emit_count == 1
    assert section.load_from_file_requested.emit_count == 1
    assert section.restore_factory_default_requested.emit_count == 1


def test_input_image_section_does_not_add_auxiliary_rows() -> None:
    assert not hasattr(widgets_module.InputImageSection, '_add_extra_rows_before')


def test_scan_for_print_toggle_applies_and_restores_scanner_and_glare_state() -> None:
    toggle_scan_for_print = getattr(widgets_module.SimulationSection, '_apply_scan_for_print_mode')
    glare_section = SimpleNamespace(active=FakeValueEditor(True))
    scanner_section = SimpleNamespace(
        white_correction=FakeValueEditor(False),
        black_correction=FakeValueEditor(True),
    )

    section = SimpleNamespace(
        _glare_section=glare_section,
        _scanner_section=scanner_section,
        _scan_for_print_restore_state=None,
    )

    toggle_scan_for_print(section, True)

    assert scanner_section.white_correction.value is True
    assert scanner_section.black_correction.value is True
    assert glare_section.active.value is False

    toggle_scan_for_print(section, False)

    assert scanner_section.white_correction.value is False
    assert scanner_section.black_correction.value is True
    assert glare_section.active.value is True
    assert getattr(section, '_scan_for_print_restore_state') is None


def test_numeric_field_specs_define_minimum_and_step() -> None:
    sections = {
        'load_raw': state_module.LoadRawState,
    }

    missing: list[str] = []
    for section_name, state_cls in sections.items():
        section_specs = {spec.leaf: spec for spec in widgets_module.LOAD_RAW_FIELDS}
        for field_info in fields(state_cls):
            annotation = field_info.type
            is_numeric = annotation in (int, float) or get_origin(annotation) is tuple
            if not is_numeric:
                continue
            spec = section_specs.get(field_info.name)
            if spec is None:
                missing.append(f'{section_name}.{field_info.name}: missing spec')
                continue
            if spec.min is None:
                missing.append(f'{section_name}.{field_info.name}: missing min')
            if spec.step is None:
                missing.append(f'{section_name}.{field_info.name}: missing step')

    # input_image is now path-bound: its numeric ranges live on the
    # INPUT_IMAGE_FIELDS manifest.
    from spektrafilm.runtime.params_schema import IOParams, SettingsParams
    from spektrafilm_gui.params_manifest import INPUT_IMAGE_FIELDS

    group_hints = {'io': get_type_hints(IOParams), 'settings': get_type_hints(SettingsParams)}
    for spec in INPUT_IMAGE_FIELDS:
        group, _, leaf = spec.path.partition('.')
        annotation = group_hints[group][leaf]
        is_numeric = annotation in (int, float) or get_origin(annotation) is tuple
        if not is_numeric:
            continue
        if spec.min is None:
            missing.append(f'input_image.{spec.leaf}: missing min')
        if spec.step is None:
            missing.append(f'input_image.{spec.leaf}: missing step')

    from spektrafilm_gui.params_manifest import DISPLAY_PANEL_FIELDS, SIMULATION_FIELDS, SPECIAL_FIELDS

    display_specs = {spec.leaf: spec for spec in DISPLAY_PANEL_FIELDS}
    display_field_annotations = {
        'use_display_transform': bool,
        'gray_18_canvas': bool,
        'white_padding': float,
        'preview_max_size': int,
        'output_interpolation': str,
    }
    for field_name in tuple(spec.leaf for spec in DISPLAY_PANEL_FIELDS):
        annotation = display_field_annotations[field_name]
        is_numeric = annotation in (int, float) or get_origin(annotation) is tuple
        if not is_numeric:
            continue
        spec = display_specs.get(field_name)
        if spec is None:
            missing.append(f'display.{field_name}: missing spec')
            continue
        if spec.min is None:
            missing.append(f'display.{field_name}: missing min')
        if spec.step is None:
            missing.append(f'display.{field_name}: missing step')

    simulation_specs = {spec.leaf: spec for spec in SIMULATION_FIELDS}
    simulation_field_annotations = {
        'film_stock': str,
        'print_paper': str,
        'print_illuminant': str,
        'print_exposure': float,
        'print_exposure_compensation': bool,
        'print_y_filter_shift': float,
        'print_m_filter_shift': float,
        'output_color_space': str,
        'saving_color_space': str,
        'saving_cctf_encoding': bool,
        'auto_preview': bool,
        'scan_film': bool,
    }
    for field_name in tuple(spec.leaf for spec in SIMULATION_FIELDS):
        annotation = simulation_field_annotations[field_name]
        is_numeric = annotation in (int, float) or get_origin(annotation) is tuple
        if not is_numeric:
            continue
        spec = simulation_specs.get(field_name)
        if spec is None:
            missing.append(f'simulation.{field_name}: missing spec')
            continue
        if spec.min is None:
            missing.append(f'simulation.{field_name}: missing min')
        if spec.step is None:
            missing.append(f'simulation.{field_name}: missing step')

    special_specs = {spec.leaf: spec for spec in SPECIAL_FIELDS}
    for field_name in tuple(spec.leaf for spec in SPECIAL_FIELDS):
        annotation = tuple[int, int, int]
        is_numeric = annotation in (int, float) or get_origin(annotation) is tuple
        if not is_numeric:
            continue
        spec = special_specs.get(field_name)
        if spec is None:
            missing.append(f'special.{field_name}: missing spec')
            continue
        if spec.min is None:
            missing.append(f'special.{field_name}: missing min')
        if spec.step is None:
            missing.append(f'special.{field_name}: missing step')

    from spektrafilm_gui.params_manifest import PREFLASHING_MANIFEST

    for spec in PREFLASHING_MANIFEST.fields:
        if spec.leaf == 'preflash_exposure' and spec.min is None:
            missing.append(f'{spec.path}: missing min')
        if spec.step is None:
            missing.append(f'{spec.path}: missing step')

    assert missing == []


def test_params_group_section_mirrors_runtime_values_verbatim() -> None:
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from qtpy import QtWidgets

    from spektrafilm.runtime.params_schema import HalationParams
    from spektrafilm_gui.params_manifest import HALATION_MANIFEST

    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    section = widgets_module.ParamsGroupSection(HALATION_MANIFEST)

    # values surface verbatim — no GUI-only transform (the schema stores
    # 0-1 fractions and that is exactly what the widgets show / return)
    section.set_state(HalationParams(
        halation_strength=(0.08, 0.02, 0.0),
        scatter_tail_weight=(0.78, 0.65, 0.67),
    ))
    assert tuple(section.scatter_tail_weight.value) == (0.78, 0.65, 0.67)

    out = section.get_state()
    assert isinstance(out, HalationParams)
    np.testing.assert_allclose(out.halation_strength, (0.08, 0.02, 0.0))
    np.testing.assert_allclose(out.scatter_tail_weight, (0.78, 0.65, 0.67))


def test_couplers_section_surfaces_diffusion_tail_values() -> None:
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from qtpy import QtWidgets

    from spektrafilm.runtime.params_schema import DirCouplersParams
    from spektrafilm_gui.params_manifest import DIR_COUPLERS_MANIFEST

    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    section = widgets_module.ParamsGroupSection(DIR_COUPLERS_MANIFEST)

    section.set_state(DirCouplersParams(
        diffusion_size_um=18.0,
        diffusion_tail_um=240.0,
        diffusion_tail_weight=0.07,
    ))
    assert float(section.diffusion_tail_um.value) == 240.0
    assert float(section.diffusion_tail_weight.value) == 0.07

    out = section.get_state()
    assert isinstance(out, DirCouplersParams)
    assert out.diffusion_size_um == 18.0
    assert out.diffusion_tail_um == 240.0
    assert out.diffusion_tail_weight == 0.07


def test_param_manifests_are_complete_and_well_formed() -> None:
    from spektrafilm_gui.params_manifest import ALL_MANIFESTS

    for manifest in ALL_MANIFESTS:
        group_field_names = {field_info.name for field_info in fields(manifest.group_cls)}
        type_hints = get_type_hints(manifest.group_cls)
        seen: set[str] = set()
        for spec in manifest.fields:
            # every manifest leaf is a real field of the runtime group,
            # the path is consistent with the group path, and no field is
            # declared twice
            assert spec.leaf in group_field_names, f'{spec.path}: not a field of {manifest.group_cls.__name__}'
            assert spec.path == f'{manifest.group_path}.{spec.leaf}'
            assert spec.leaf not in seen, f'{spec.path}: declared twice'
            seen.add(spec.leaf)
            # the editor type must be inferable from the group's hints
            assert spec.leaf in type_hints, f'{spec.path}: missing type hint'


def test_section_header_icon_returns_empty_icon_without_pyconify(monkeypatch) -> None:
    monkeypatch.setattr(icons_module, 'pyconify', None)
    icons_module.section_header_icon.cache_clear()

    icon = icons_module.section_header_icon('Import RGB')

    assert icon.isNull() is True


def test_collapsible_section_shows_icon_for_mapped_main_tab_title(monkeypatch) -> None:
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from qtpy import QtGui, QtWidgets

    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    icon = QtGui.QIcon()
    pixmap = QtGui.QPixmap(icons_module.HEADER_ICON_SIZE, icons_module.HEADER_ICON_SIZE)
    pixmap.fill(QtGui.QColor('#ee9470'))
    icon.addPixmap(pixmap)
    monkeypatch.setattr(primitives_module, 'section_header_icon', lambda _title: icon)

    section = primitives_module.CollapsibleSection('Import RGB', QtWidgets.QWidget(), expanded=True)

    assert section.has_header_icon() is True


def test_widget_spec_decimals_configure_float_editors(monkeypatch) -> None:
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from qtpy import QtWidgets

    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    del monkeypatch
    build_editor = getattr(widgets_module, '_editor_from_param_spec')

    float_editor = build_editor(
        float,
        param_manifest_module.ParamSpec('test.float_value', decimals=4),
        label='test.float_value',
    )
    float_pair_editor = build_editor(
        tuple[float, float],
        param_manifest_module.ParamSpec('test.float_pair', decimals=3),
        label='test.float_pair',
    )

    assert float_editor.decimals() == 4
    assert [editor.decimals() for editor in float_pair_editor.editors] == [3, 3]


def test_simulation_section_profile_use_badges_follow_selected_profiles() -> None:
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from qtpy import QtWidgets

    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    section = widgets_module.SimulationSection()

    assert isinstance(section.film_stock, widget_editors_module.ProfileEnumEditor)
    assert isinstance(section.print_paper, widget_editors_module.ProfileEnumEditor)
    assert section.film_stock.currentText() == list(state_module.FilmStocks)[0].value
    assert section.print_paper.currentText() == list(state_module.PrintPapers)[0].value
    assert widget_editors_module.ProfileEnumEditor.display_text_for_value('kodak_portra_400') == 'still / neg / color / kodak_portra_400'
    assert widget_editors_module.ProfileEnumEditor.display_text_for_value('kodak_vision3_50d') == 'cine / neg / color / kodak_vision3_50d'
    assert widget_editors_module.ProfileEnumEditor.display_text_for_value('kodak_2393') == 'cine / neg / color / kodak_2393'
    assert widget_editors_module.ProfileEnumEditor.display_text_for_value('kodak_trix') == 'cine / pos / bw / kodak_trix'
