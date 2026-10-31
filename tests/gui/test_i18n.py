import json
import re
import string
from pathlib import Path

import pytest
from qtpy import QtWidgets

from spektrafilm_gui import i18n
from spektrafilm_gui import param_manifest, widget_sections
from spektrafilm_gui.options import NapariInterpolationModes
from spektrafilm_gui.param_manifest import GroupManifest, ParamSpec
from spektrafilm_gui.widget_editors import EnumEditor
from spektrafilm_gui.widget_primitives import normalize_ui_text

CATALOG_PATH = Path(i18n.__file__).with_name('locales') / 'zh_CN.json'
GUI_SOURCES = sorted(Path(i18n.__file__).parent.glob('*.py'))


@pytest.fixture
def language(monkeypatch):
    def _set(name: str) -> None:
        monkeypatch.setenv(i18n.LANGUAGE_ENV_VAR, name)
        i18n._catalog.cache_clear()

    yield _set
    monkeypatch.undo()
    i18n._catalog.cache_clear()


def _catalog_keys() -> set[str]:
    return {key.strip().lower() for key in json.loads(CATALOG_PATH.read_text(encoding='utf-8'))}


def _specs():
    for module in (param_manifest, widget_sections):
        for value in vars(module).values():
            if isinstance(value, GroupManifest):
                yield from value.fields
            elif isinstance(value, ParamSpec):
                yield value
            elif isinstance(value, (tuple, list)) and value and isinstance(value[0], ParamSpec):
                yield from value
            elif isinstance(value, dict):
                yield from (item for item in value.values() if isinstance(item, ParamSpec))


@pytest.mark.parametrize(
    ('locale_name', 'expected'),
    [('zh_CN', 'zh_CN'), ('zh-Hans', 'zh_CN'), ('en', 'en'), ('en_US', 'en'), ('fr_FR', 'en')],
)
def test_language_resolution(language, locale_name, expected):
    language(locale_name)
    assert i18n.current_language() == expected


def test_english_returns_source_text(language):
    language('en')
    assert i18n.tr('Halation') == 'Halation'
    assert normalize_ui_text('Halation') == 'halation'
    assert i18n.tr('Saved GUI state to {filepath}', filepath='a.json') == 'Saved GUI state to a.json'


def test_chinese_translates_and_falls_back(language):
    language('zh_CN')
    assert i18n.tr('Halation') == '光晕'
    assert normalize_ui_text('Halation') == '光晕'
    assert i18n.tr('not in the catalog') == 'not in the catalog'
    assert i18n.tr('Saved GUI state to {filepath}', filepath='a.json') == '界面状态已保存到 a.json'


def test_every_parameter_label_and_tooltip_is_translated():
    keys = _catalog_keys()
    missing = set()
    for spec in _specs():
        for text in (spec.label or spec.leaf.replace('_', ' '), spec.tooltip):
            if text and text.strip().lower() not in keys:
                missing.add(text)
    assert not missing


def test_every_tr_literal_in_the_gui_is_translated():
    keys = _catalog_keys()
    pattern = re.compile(r"""\btr\(\s*(['"])(.+?)\1\s*[,)]""")
    missing = set()
    for source in GUI_SOURCES:
        for _, literal in pattern.findall(source.read_text(encoding='utf-8')):
            text = literal.encode().decode('unicode_escape').encode('latin-1', 'ignore').decode()
            if text.strip().lower() not in keys:
                missing.add(text)
    assert not missing


def test_translations_keep_the_format_fields():
    catalog = json.loads(CATALOG_PATH.read_text(encoding='utf-8'))
    for source, translation in catalog.items():
        fields = {field for _, field, _, _ in string.Formatter().parse(source) if field}
        translated_fields = {field for _, field, _, _ in string.Formatter().parse(translation) if field}
        assert fields == translated_fields, source


def test_interpolation_modes_stay_untranslated(language):
    # The display section forwards the combo box text to napari unchanged.
    language('zh_CN')
    for mode in NapariInterpolationModes:
        assert i18n.tr(mode.value) == mode.value


def test_enum_editor_shows_translation_and_stores_source_value(language):
    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    language('zh_CN')
    editor = EnumEditor(['as_shot', 'custom'])
    editor.value = 'custom'
    assert editor.currentText() == '自定义'
    assert editor.value == 'custom'
    with pytest.raises(ValueError):
        editor.value = '自定义'
