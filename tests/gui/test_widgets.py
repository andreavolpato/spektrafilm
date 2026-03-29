from __future__ import annotations

import os
from types import SimpleNamespace

from spektrafilm_gui import widget_specs as widget_specs_module
from spektrafilm_gui import icons as icons_module
from spektrafilm_gui import widget_primitives as primitives_module
from spektrafilm_gui import widget_sections as widgets_module


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


def _make_load_raw_section(monkeypatch):
    init_extra_widgets = getattr(widgets_module.LoadRawSection, '_init_extra_widgets')
    reprocess_raw = getattr(widgets_module.LoadRawSection, '_reprocess_raw')
    created_buttons: list[FakeButton] = []
    monkeypatch.setattr(widgets_module, 'QLineEdit', FakeLineEdit)
    monkeypatch.setattr(
        widgets_module,
        '_build_button',
        lambda text, callback, **kwargs: created_buttons.append(FakeButton(text, callback, tooltip=kwargs.get('tooltip'))) or created_buttons[-1],
    )
    monkeypatch.setattr(widgets_module, '_build_button_row', lambda *widgets, **kwargs: ('button-row', widgets, kwargs))
    monkeypatch.setattr(widgets_module, '_build_vertical_container', lambda *items, **kwargs: ('vertical-container', items, kwargs))
    section = SimpleNamespace(load_requested=FakeSignal())
    setattr(section, '_choose_file', lambda: None)
    setattr(section, '_reprocess_raw', lambda: reprocess_raw(section))
    init_extra_widgets(section)
    return section, created_buttons


def _make_filepicker_section(monkeypatch):
    build_ui = getattr(widgets_module.FilePickerSection, '_build_ui')
    choose_file = getattr(widgets_module.FilePickerSection, '_choose_file')
    selected_input_layer_name = getattr(widgets_module.FilePickerSection, 'selected_input_layer_name')
    created_buttons: list[FakeButton] = []
    monkeypatch.setattr(widgets_module, 'QLineEdit', FakeLineEdit)
    monkeypatch.setattr(widgets_module, 'QComboBox', FakeComboBox)
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
    setattr(section, 'selected_input_layer_name', lambda: selected_input_layer_name(section))
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
    section, created_buttons = _make_load_raw_section(monkeypatch)
    form = FakeForm()
    add_extra_rows_before = getattr(widgets_module.LoadRawSection, '_add_extra_rows_before')
    add_extra_rows_after = getattr(widgets_module.LoadRawSection, '_add_extra_rows_after')

    add_extra_rows_before(section, form)
    add_extra_rows_after(section, form)

    assert section.file_path.read_only is True
    assert section.file_path.placeholder_text == 'no raw selected'
    assert created_buttons[0].text == 'reprocess raw'
    assert section.reprocess_button.text == 'reprocess raw'
    assert section.reprocess_button.enabled is False
    assert created_buttons[1].text == 'Select file'
    assert created_buttons[1].tooltip == 'Load and process a raw file using rawpy, output colorspace and cctf as defined in current input widget state'
    assert form.rows[1] == (section.reprocess_button,)


def test_load_raw_section_reprocess_button_uses_selected_path(monkeypatch) -> None:
    section, _created_buttons = _make_load_raw_section(monkeypatch)

    widgets_module.LoadRawSection.set_path(section, 'C:/tmp/example.nef')
    section.reprocess_button.click()
    widgets_module.LoadRawSection.set_path(section, '')
    section.reprocess_button.click()

    assert section.file_path.text() == ''
    assert section.load_requested.emitted == [('C:/tmp/example.nef',)]
    assert section.reprocess_button.enabled is False


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


def test_file_picker_set_available_layers_preserves_current_selection(monkeypatch) -> None:
    section, _created_buttons = _make_filepicker_section(monkeypatch)

    widgets_module.FilePickerSection.set_available_layers(section, ['input', 'mask'], selected_name='mask')
    widgets_module.FilePickerSection.set_available_layers(section, ['input', 'mask', 'preview'])

    assert section.input_layer.items == ['input', 'mask', 'preview']
    assert section.input_layer.currentText() == 'mask'
    assert section.input_layer.blocked_calls == [True, False, True, False]


def test_file_picker_set_available_layers_uses_selected_name_override(monkeypatch) -> None:
    section, _created_buttons = _make_filepicker_section(monkeypatch)

    widgets_module.FilePickerSection.set_available_layers(section, ['input', 'mask', 'preview'], selected_name='preview')

    assert section.input_layer.currentText() == 'preview'


def test_file_picker_selected_input_layer_name_returns_none_when_empty(monkeypatch) -> None:
    section, _created_buttons = _make_filepicker_section(monkeypatch)

    assert widgets_module.FilePickerSection.selected_input_layer_name(section) is None


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


def test_input_image_section_adds_input_layer_before_input_color_space(monkeypatch) -> None:
    form = FakeForm()
    add_extra_rows_before = getattr(widgets_module.InputImageSection, '_add_extra_rows_before')
    monkeypatch.setattr(widgets_module, '_build_auxiliary_label', lambda name: f'label:{name}')
    section = SimpleNamespace(_filepicker_section=SimpleNamespace(input_layer='input-layer-combo'))

    add_extra_rows_before(section, form)

    assert form.rows == [('label:input_layer', 'input-layer-combo')]


def test_load_raw_widget_specs_include_requested_tooltips() -> None:
    assert widget_specs_module.get_widget_spec('load_raw', 'white_balance').tooltip == (
        'Select white balance settings, if custom you can tune temperature and tint'
    )
    assert widget_specs_module.get_widget_spec('load_raw', 'temperature').tooltip == (
        'Temperature in Kelvin for the custom whitebalance, not used for the other white balance settings'
    )
    assert widget_specs_module.get_widget_spec('load_raw', 'tint').tooltip == (
        'Tint value for the custom white balance, not used for the other white balance settings'
    )


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