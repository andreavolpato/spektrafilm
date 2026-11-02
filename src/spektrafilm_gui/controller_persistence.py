from __future__ import annotations

import json
from typing import Any, Callable


def save_current_as_default(
    *,
    viewer: Any,
    widgets: Any,
    collect_gui_state_fn: Callable[..., Any],
    save_default_gui_state_fn: Callable[[Any], None],
    set_status_fn: Callable[..., None],
    dialog_parent_fn: Callable[[Any], Any],
    message_box: Any,
) -> None:
    gui_state = collect_gui_state_fn(widgets=widgets)
    try:
        save_default_gui_state_fn(gui_state)
    except (OSError, ValueError) as exc:
        message_box.critical(dialog_parent_fn(viewer), '保存默认设置', f'无法保存默认界面设置。\n\n{exc}')
        return

    set_status_fn(viewer, '当前界面设置已保存为启动默认值')


def save_current_state_to_file(
    *,
    viewer: Any,
    widgets: Any,
    file_dialog: Any,
    collect_gui_state_fn: Callable[..., Any],
    save_gui_state_to_path_fn: Callable[[Any, str], None],
    set_status_fn: Callable[..., None],
    dialog_parent_fn: Callable[[Any], Any],
    message_box: Any,
) -> None:
    filepath, _ = file_dialog.get_save_file_name(
        dialog_parent_fn(viewer),
        '保存界面设置',
        'gui_state.json',
        'JSON 文件 (*.json)',
    )
    if not filepath:
        return

    gui_state = collect_gui_state_fn(widgets=widgets)
    try:
        save_gui_state_to_path_fn(gui_state, filepath)
    except (OSError, ValueError) as exc:
        message_box.critical(dialog_parent_fn(viewer), '保存界面设置失败', f'无法保存界面设置。\n\n{exc}')
        return

    set_status_fn(viewer, f'界面设置已保存到 {filepath}')


def load_state_from_file(
    *,
    viewer: Any,
    widgets: Any,
    file_dialog: Any,
    load_gui_state_from_path_fn: Callable[[str], Any],
    apply_gui_state_fn: Callable[..., None],
    sync_canvas_background_fn: Callable[[], None],
    set_status_fn: Callable[..., None],
    dialog_parent_fn: Callable[[Any], Any],
    message_box: Any,
) -> None:
    filepath, _ = file_dialog.get_open_file_name(
        dialog_parent_fn(viewer),
        '加载界面设置',
        '',
        'JSON 文件 (*.json)',
    )
    if not filepath:
        return

    try:
        gui_state = load_gui_state_from_path_fn(filepath)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        message_box.critical(dialog_parent_fn(viewer), '加载界面设置失败', f'无法加载界面设置。\n\n{exc}')
        return

    apply_gui_state_fn(gui_state, widgets=widgets)
    sync_canvas_background_fn()
    set_status_fn(viewer, f'已从 {filepath} 加载界面设置')


def restore_factory_default(
    *,
    viewer: Any,
    widgets: Any,
    project_default_gui_state: Any,
    clear_saved_default_gui_state_fn: Callable[[], None],
    apply_gui_state_fn: Callable[..., None],
    sync_canvas_background_fn: Callable[[], None],
    set_status_fn: Callable[..., None],
    dialog_parent_fn: Callable[[Any], Any],
    message_box: Any,
) -> None:
    try:
        clear_saved_default_gui_state_fn()
    except OSError as exc:
        message_box.critical(
            dialog_parent_fn(viewer),
            '恢复出厂默认设置',
            f'无法清除已保存的启动默认值。\n\n{exc}',
        )
        return

    apply_gui_state_fn(project_default_gui_state, widgets=widgets)
    sync_canvas_background_fn()
    set_status_fn(viewer, '已恢复出厂默认界面设置')