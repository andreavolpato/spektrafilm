from __future__ import annotations

from pathlib import Path

from spektrafilm_gui import controller as controller_module
from spektrafilm_gui.controller import _DirMemoryDialog


def test_get_save_file_name_combines_remembered_dir_with_filename(monkeypatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(controller_module, 'load_dialog_dir', lambda key: '/remembered/dir')
    monkeypatch.setattr(controller_module, 'save_dialog_dir', lambda key, directory: None)
    monkeypatch.setattr(
        controller_module.QFileDialog,
        'getSaveFileName',
        staticmethod(
            lambda parent, title, initial, fmt:
            captured.update({'initial': initial}) or ('/remembered/dir/output.png', fmt)
        ),
    )

    _DirMemoryDialog('k').get_save_file_name(None, '', 'output.png', '')

    assert Path(captured['initial']) == Path('/remembered/dir/output.png')


def test_get_save_file_name_uses_filename_when_no_dir(monkeypatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(controller_module, 'load_dialog_dir', lambda key: '')
    monkeypatch.setattr(controller_module, 'save_dialog_dir', lambda key, directory: None)
    monkeypatch.setattr(
        controller_module.QFileDialog,
        'getSaveFileName',
        staticmethod(
            lambda parent, title, initial, fmt:
            captured.update({'initial': initial}) or ('output.png', fmt)
        ),
    )

    _DirMemoryDialog('k').get_save_file_name(None, '', 'output.png', '')

    assert captured['initial'] == 'output.png'


def test_get_save_file_name_stores_parent_dir(monkeypatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(controller_module, 'load_dialog_dir', lambda key: '')
    monkeypatch.setattr(
        controller_module, 'save_dialog_dir',
        lambda key, directory: captured.update({'dir': directory}),
    )
    monkeypatch.setattr(
        controller_module.QFileDialog,
        'getSaveFileName',
        staticmethod(lambda *a, **kw: ('/some/dir/out.png', '')),
    )

    _DirMemoryDialog('k').get_save_file_name(None, '', 'out.png', '')

    assert Path(captured['dir']) == Path('/some/dir')


def test_get_save_file_name_does_not_store_on_cancel(monkeypatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(controller_module, 'load_dialog_dir', lambda key: '')
    monkeypatch.setattr(
        controller_module, 'save_dialog_dir',
        lambda key, directory: captured.update({'called': True}),
    )
    monkeypatch.setattr(
        controller_module.QFileDialog,
        'getSaveFileName',
        staticmethod(lambda *a, **kw: ('', '')),
    )

    _DirMemoryDialog('k').get_save_file_name(None, '', 'out.png', '')

    assert 'called' not in captured


def test_get_open_file_name_passes_remembered_dir(monkeypatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(controller_module, 'load_dialog_dir', lambda key: '/last/dir')
    monkeypatch.setattr(controller_module, 'save_dialog_dir', lambda key, directory: None)
    monkeypatch.setattr(
        controller_module.QFileDialog,
        'getOpenFileName',
        staticmethod(
            lambda parent, title, initial, fmt:
            captured.update({'initial': initial}) or ('/last/dir/f.raw', fmt)
        ),
    )

    _DirMemoryDialog('k').get_open_file_name(None, '', '', '')

    assert captured['initial'] == '/last/dir'


def test_get_open_file_name_does_not_store_on_cancel(monkeypatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(controller_module, 'load_dialog_dir', lambda key: '')
    monkeypatch.setattr(
        controller_module, 'save_dialog_dir',
        lambda key, directory: captured.update({'called': True}),
    )
    monkeypatch.setattr(
        controller_module.QFileDialog,
        'getOpenFileName',
        staticmethod(lambda *a, **kw: ('', '')),
    )

    _DirMemoryDialog('k').get_open_file_name(None, '', '', '')

    assert 'called' not in captured
