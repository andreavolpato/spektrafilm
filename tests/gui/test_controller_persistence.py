from __future__ import annotations

from spektrafilm_gui import controller as controller_module
from spektrafilm_gui.controller import GuiController
from spektrafilm_gui.state import PROJECT_DEFAULT_GUI_STATE

from .helpers import make_test_controller_gui_state


def test_load_state_from_file_syncs_canvas_background(monkeypatch) -> None:
    controller = GuiController(viewer=object(), widgets=object())
    captured: dict[str, object] = {}

    monkeypatch.setattr(controller_module.QFileDialog, 'getOpenFileName', staticmethod(lambda *args, **kwargs: ('state.json', 'JSON (*.json)')))
    monkeypatch.setattr(controller_module, 'load_dialog_dir', lambda key: '')
    monkeypatch.setattr(controller_module, 'save_dialog_dir', lambda key, directory: None)
    monkeypatch.setattr(controller_module, 'load_gui_state_from_path', lambda path: PROJECT_DEFAULT_GUI_STATE)
    monkeypatch.setattr(controller_module, 'apply_gui_state', lambda state, *, widgets: captured.setdefault('applied', (state, widgets)))
    monkeypatch.setattr(controller, '_sync_canvas_background', lambda: captured.setdefault('canvas_synced', True))
    monkeypatch.setattr(controller_module, 'dialog_parent', lambda viewer: None)
    monkeypatch.setattr(controller_module, 'set_status', lambda viewer, message: captured.setdefault('status', message))

    controller.load_state_from_file()

    assert captured['canvas_synced'] is True
    assert captured['status'] == 'Loaded GUI state from state.json'


def test_restore_factory_default_syncs_canvas_background(monkeypatch) -> None:
    controller = GuiController(viewer=object(), widgets=object())
    captured: dict[str, object] = {}

    monkeypatch.setattr(controller_module, 'clear_saved_default_gui_state', lambda: None)
    monkeypatch.setattr(controller_module, 'apply_gui_state', lambda state, *, widgets: captured.setdefault('applied', (state, widgets)))
    monkeypatch.setattr(controller, '_sync_canvas_background', lambda: captured.setdefault('canvas_synced', True))
    monkeypatch.setattr(controller_module, 'set_status', lambda viewer, message: captured.setdefault('status', message))

    controller.restore_factory_default()

    assert captured['canvas_synced'] is True
    assert captured['status'] == 'Restored factory default GUI state'


def test_save_current_as_default_persists_collected_state(monkeypatch) -> None:
    controller = GuiController(viewer=object(), widgets=object())
    gui_state = make_test_controller_gui_state()
    captured: dict[str, object] = {}

    monkeypatch.setattr(controller_module, 'collect_gui_state', lambda *, widgets: gui_state)
    monkeypatch.setattr(controller_module, 'save_default_gui_state', lambda state: captured.setdefault('state', state))
    monkeypatch.setattr(controller_module, 'set_status', lambda viewer, message: captured.setdefault('status', message))

    controller.save_current_as_default()

    assert captured['state'] is gui_state
    assert captured['status'] == 'Saved current GUI state as the startup default'


def test_save_current_state_to_file_persists_json(monkeypatch) -> None:
    controller = GuiController(viewer=object(), widgets=object())
    gui_state = make_test_controller_gui_state()
    captured: dict[str, object] = {}

    monkeypatch.setattr(controller_module, 'dialog_parent', lambda viewer: None)
    monkeypatch.setattr(controller_module, 'load_dialog_dir', lambda key: '')
    monkeypatch.setattr(controller_module, 'save_dialog_dir', lambda key, directory: None)
    monkeypatch.setattr(
        controller_module.QFileDialog,
        'getSaveFileName',
        staticmethod(lambda *args, **kwargs: ('gui_state.json', 'JSON (*.json)')),
    )
    monkeypatch.setattr(controller_module, 'collect_gui_state', lambda *, widgets: gui_state)
    monkeypatch.setattr(controller_module, 'save_gui_state_to_path', lambda state, path: captured.setdefault('saved', (state, path)))
    monkeypatch.setattr(controller_module, 'set_status', lambda viewer, message: captured.setdefault('status', message))

    controller.save_current_state_to_file()

    assert captured['saved'] == (gui_state, 'gui_state.json')
    assert captured['status'] == 'Saved GUI state to gui_state.json'


def test_load_state_from_file_applies_loaded_state(monkeypatch) -> None:
    controller = GuiController(viewer=object(), widgets=object())
    gui_state = make_test_controller_gui_state()
    captured: dict[str, object] = {}

    monkeypatch.setattr(controller_module, 'dialog_parent', lambda viewer: None)
    monkeypatch.setattr(controller_module, 'load_dialog_dir', lambda key: '')
    monkeypatch.setattr(controller_module, 'save_dialog_dir', lambda key, directory: None)
    monkeypatch.setattr(
        controller_module.QFileDialog,
        'getOpenFileName',
        staticmethod(lambda *args, **kwargs: ('gui_state.json', 'JSON (*.json)')),
    )
    monkeypatch.setattr(controller_module, 'load_gui_state_from_path', lambda path: gui_state)
    monkeypatch.setattr(controller_module, 'apply_gui_state', lambda state, *, widgets: captured.setdefault('applied', (state, widgets)))
    monkeypatch.setattr(controller_module, 'set_status', lambda viewer, message: captured.setdefault('status', message))

    controller.load_state_from_file()

    assert captured['applied'] == (gui_state, controller._widgets)
    assert captured['status'] == 'Loaded GUI state from gui_state.json'


def test_restore_factory_default_clears_saved_default_and_applies_factory_state(monkeypatch) -> None:
    controller = GuiController(viewer=object(), widgets=object())
    captured: dict[str, object] = {}

    monkeypatch.setattr(controller_module, 'clear_saved_default_gui_state', lambda: captured.setdefault('cleared', True))
    monkeypatch.setattr(controller_module, 'apply_gui_state', lambda state, *, widgets: captured.setdefault('applied', (state, widgets)))
    monkeypatch.setattr(controller_module, 'set_status', lambda viewer, message: captured.setdefault('status', message))

    controller.restore_factory_default()

    assert captured['cleared'] is True
    assert captured['applied'] == (PROJECT_DEFAULT_GUI_STATE, controller._widgets)
    assert captured['status'] == 'Restored factory default GUI state'