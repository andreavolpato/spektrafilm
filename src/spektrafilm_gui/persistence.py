from __future__ import annotations

import json
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_origin, get_type_hints

from qtpy.QtCore import QSettings, QStandardPaths

from spektrafilm_gui.state import GuiState, PROJECT_DEFAULT_GUI_STATE, clone_gui_state


DEFAULT_GUI_STATE_FILENAME = "gui_default_state.json"

GuiStateType = TypeVar("GuiStateType")


def gui_state_to_dict(state: GuiState) -> dict[str, Any]:
    return asdict(state)


def gui_state_from_dict(data: dict[str, Any]) -> GuiState:
    if not isinstance(data, dict):
        raise ValueError("GUI state data must be a JSON object.")
    return _deserialize_dataclass(GuiState, data)


def load_default_gui_state() -> GuiState:
    default_path = default_gui_state_path()
    if not default_path.exists():
        return clone_gui_state(PROJECT_DEFAULT_GUI_STATE)
    return load_gui_state_from_path(default_path)


def save_default_gui_state(state: GuiState) -> Path:
    default_path = default_gui_state_path()
    save_gui_state_to_path(state, default_path)
    return default_path


def clear_saved_default_gui_state() -> None:
    default_path = default_gui_state_path()
    if default_path.exists():
        default_path.unlink()


def save_gui_state_to_path(state: GuiState, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as file:
        json.dump(gui_state_to_dict(state), file, indent=2)


def load_gui_state_from_path(path: str | Path) -> GuiState:
    source = Path(path)
    with source.open("r", encoding="utf-8") as file:
        return gui_state_from_dict(json.load(file))


def default_gui_state_path() -> Path:
    app_config_location = QStandardPaths.writableLocation(QStandardPaths.AppConfigLocation)
    if app_config_location:
        return Path(app_config_location) / DEFAULT_GUI_STATE_FILENAME
    return Path.home() / ".spektrafilm" / DEFAULT_GUI_STATE_FILENAME


def _deserialize_dataclass(cls: type[GuiStateType], data: dict[str, Any]) -> GuiStateType:
    if not isinstance(data, dict):
        raise ValueError(f"Expected an object for {cls.__name__}.")

    type_hints = get_type_hints(cls)
    values: dict[str, Any] = {}
    for field_info in fields(cls):
        field_name = field_info.name
        if field_name not in data:
            raise ValueError(f"Missing field {field_name!r} in {cls.__name__}.")
        values[field_name] = _deserialize_value(type_hints[field_name], data[field_name])
    return cls(**values)


def load_dialog_dir(key: str) -> str:
    return QSettings('spektrafilm', 'spektrafilm').value(f'dialog_dirs/{key}', '')


def save_dialog_dir(key: str, directory: str) -> None:
    QSettings('spektrafilm', 'spektrafilm').setValue(f'dialog_dirs/{key}', directory)


def _deserialize_value(annotation: Any, value: Any) -> Any:
    if is_dataclass(annotation):
        return _deserialize_dataclass(annotation, value)

    if get_origin(annotation) is tuple:
        if not isinstance(value, (list, tuple)):
            raise ValueError("Tuple fields must be encoded as arrays.")
        return tuple(value)

    return value