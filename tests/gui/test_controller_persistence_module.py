from __future__ import annotations

import json

from spektrafilm_gui import controller_persistence as persistence_module


class FakeMessageBox:
    def __init__(self) -> None:
        self.critical_calls: list[tuple[object, str, str]] = []

    def critical(self, parent, title: str, message: str) -> None:
        self.critical_calls.append((parent, title, message))


class FakeFileDialog:
    def __init__(self, *, save_result=('', ''), open_result=('', '')) -> None:
        self.save_result = save_result
        self.open_result = open_result

    def getSaveFileName(self, *args, **kwargs):  # noqa: N802 - Qt API name
        return self.save_result

    def getOpenFileName(self, *args, **kwargs):  # noqa: N802 - Qt API name
        return self.open_result


def test_save_current_state_to_file_saves_and_reports_status() -> None:
    captured: dict[str, object] = {}
    gui_state = object()

    persistence_module.save_current_state_to_file(
        viewer='viewer',
        widgets='widgets',
        file_dialog=FakeFileDialog(save_result=('gui_state.json', 'JSON (*.json)')),
        collect_gui_state_fn=lambda *, widgets: gui_state,
        save_gui_state_to_path_fn=lambda state, path: captured.setdefault('saved', (state, path)),
        set_status_fn=lambda viewer, message: captured.setdefault('status', (viewer, message)),
        dialog_parent_fn=lambda viewer: 'dialog-parent',
        message_box=FakeMessageBox(),
    )

    assert captured['saved'] == (gui_state, 'gui_state.json')
    assert captured['status'] == ('viewer', 'Saved GUI state to gui_state.json')


def test_load_state_from_file_reports_json_decode_error() -> None:
    message_box = FakeMessageBox()

    persistence_module.load_state_from_file(
        viewer='viewer',
        widgets='widgets',
        file_dialog=FakeFileDialog(open_result=('gui_state.json', 'JSON (*.json)')),
        load_gui_state_from_path_fn=lambda path: (_ for _ in ()).throw(json.JSONDecodeError('bad json', '{}', 0)),
        apply_gui_state_fn=lambda state, *, widgets: None,
        sync_canvas_background_fn=lambda: None,
        set_status_fn=lambda viewer, message: None,
        dialog_parent_fn=lambda viewer: 'dialog-parent',
        message_box=message_box,
    )

    assert message_box.critical_calls == [
        ('dialog-parent', 'Load GUI state', 'Failed to load GUI state.\n\nbad json: line 1 column 1 (char 0)'),
    ]


def test_restore_factory_default_reports_clear_failure() -> None:
    message_box = FakeMessageBox()

    persistence_module.restore_factory_default(
        viewer='viewer',
        widgets='widgets',
        project_default_gui_state='factory-state',
        clear_saved_default_gui_state_fn=lambda: (_ for _ in ()).throw(OSError('permission denied')),
        apply_gui_state_fn=lambda state, *, widgets: None,
        sync_canvas_background_fn=lambda: None,
        set_status_fn=lambda viewer, message: None,
        dialog_parent_fn=lambda viewer: 'dialog-parent',
        message_box=message_box,
    )

    assert message_box.critical_calls == [
        ('dialog-parent', 'Restore factory default', 'Failed to clear the saved startup default.\n\npermission denied'),
    ]