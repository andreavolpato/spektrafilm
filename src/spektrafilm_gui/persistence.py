from __future__ import annotations

import json
from dataclasses import asdict, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, cast

from qtpy import QtCore

from spektrafilm_gui.state import (
    PROJECT_DEFAULT_GUI_STATE,
    GuiState,
    clone_gui_state,
    display_to_dict,
    input_image_to_dict,
    normalize_display_dict,
    normalize_input_image_dict,
    normalize_simulation_dict,
    normalize_special_dict,
    simulation_to_dict,
    special_to_dict,
)

DEFAULT_GUI_STATE_FILENAME = "gui_default_state.json"
QSettings = cast(Any, getattr(QtCore, "QSettings"))
QStandardPaths = cast(Any, getattr(QtCore, "QStandardPaths"))


def _copy_section_dict(section: dict[str, Any]) -> dict[str, Any]:
    return dict(section)


SECTION_NORMALIZERS = {
    "input_image": normalize_input_image_dict,
    "special": normalize_special_dict,
    "simulation": normalize_simulation_dict,
}
GUI_ONLY_SECTION_NORMALIZERS = {
    "load_raw": _copy_section_dict,
    "display": normalize_display_dict,
}


def gui_state_to_dict(state: GuiState) -> dict[str, Any]:
    data = asdict(state)
    data.pop("gui_only", None)
    data.pop("selection", None)
    data["input_image"] = input_image_to_dict(state.input_image)
    data["load_raw"] = asdict(state.gui_only.load_raw)
    data["special"] = special_to_dict(state.special)
    data["simulation"] = simulation_to_dict(state.simulation)
    data["display"] = display_to_dict(state.gui_only.display)
    return data


def gui_state_from_dict(data: dict[str, Any]) -> GuiState:
    if not isinstance(data, dict):
        raise ValueError("GUI state data must be a JSON object.")
    normalized = dict(data)
    for section_name, normalizer in SECTION_NORMALIZERS.items():
        section = normalized.get(section_name)
        if isinstance(section, dict):
            normalized[section_name] = normalizer(section)
    gui_only = (
        dict(normalized.get("gui_only"))
        if isinstance(normalized.get("gui_only"), dict)
        else {}
    )
    for section_name, normalizer in GUI_ONLY_SECTION_NORMALIZERS.items():
        section = normalized.pop(section_name, None)
        if isinstance(section, dict):
            gui_only[section_name] = normalizer(section)
    if gui_only:
        normalized["gui_only"] = gui_only
    return _merge_into_dataclass(clone_gui_state(PROJECT_DEFAULT_GUI_STATE), normalized)


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
    app_config_location = QStandardPaths.writableLocation(
        QStandardPaths.AppConfigLocation
    )
    if app_config_location:
        return Path(app_config_location) / DEFAULT_GUI_STATE_FILENAME
    return Path.home() / ".spektrafilm" / DEFAULT_GUI_STATE_FILENAME


def _merge_into_dataclass(target: Any, data: Any) -> Any:
    if not isinstance(data, dict):
        return target
    params = getattr(target, "__dataclass_params__", None)
    frozen = bool(params and params.frozen)
    updates: dict[str, Any] = {}
    for field_info in fields(target):
        name = field_info.name
        if name not in data:
            continue
        current = getattr(target, name)
        value = data[name]
        if is_dataclass(current):
            merged = _merge_into_dataclass(current, value)
            if frozen:
                updates[name] = merged
            elif merged is not current:
                setattr(target, name, merged)
        elif isinstance(current, tuple) and isinstance(value, (list, tuple)):
            if frozen:
                updates[name] = tuple(value)
            else:
                setattr(target, name, tuple(value))
        elif not is_dataclass(current) and not isinstance(current, tuple):
            if frozen:
                updates[name] = value
            else:
                setattr(target, name, value)
    if frozen and updates:
        return replace(target, **updates)
    return target


def load_dialog_dir(key: str) -> str:
    return QSettings("spektrafilm", "spektrafilm").value(f"dialog_dirs/{key}", "")


def save_dialog_dir(key: str, directory: str) -> None:
    QSettings("spektrafilm", "spektrafilm").setValue(f"dialog_dirs/{key}", directory)
