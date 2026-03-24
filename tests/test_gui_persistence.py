from __future__ import annotations

from pathlib import Path

import pytest

from spektrafilm_gui.persistence import (
    clear_saved_default_gui_state,
    gui_state_from_dict,
    gui_state_to_dict,
    load_default_gui_state,
    load_gui_state_from_path,
    save_default_gui_state,
    save_gui_state_to_path,
)
from spektrafilm_gui.state import PROJECT_DEFAULT_GUI_STATE
from tests.gui_test_utils import make_gui_state


def test_gui_state_round_trip_preserves_tuple_fields() -> None:
    state = make_gui_state()
    state.input_image.crop_size = (0.25, 0.4)
    state.grain.particle_scale = (1.1, 1.2, 1.3)
    state.display.gray_18_canvas = True
    state.display.white_padding = 0.18

    restored = gui_state_from_dict(gui_state_to_dict(state))

    assert restored == state
    assert isinstance(restored.input_image.crop_size, tuple)
    assert isinstance(restored.grain.particle_scale, tuple)


def test_save_and_load_gui_state_file(tmp_path: Path) -> None:
    state = make_gui_state()
    state.simulation.print_exposure = 1.4
    state.special.print_gamma_factor = 1.2
    state.display.gray_18_canvas = True
    state.display.white_padding = 0.12
    destination = tmp_path / "gui_state.json"

    save_gui_state_to_path(state, destination)
    restored = load_gui_state_from_path(destination)

    assert restored == state


def test_load_default_gui_state_uses_factory_when_missing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "spektrafilm_gui.persistence.default_gui_state_path",
        lambda: tmp_path / "missing.json",
    )

    restored = load_default_gui_state()

    assert restored == PROJECT_DEFAULT_GUI_STATE
    assert restored is not PROJECT_DEFAULT_GUI_STATE


def test_save_default_and_clear_saved_default(monkeypatch, tmp_path: Path) -> None:
    default_path = tmp_path / "gui_default_state.json"
    monkeypatch.setattr(
        "spektrafilm_gui.persistence.default_gui_state_path",
        lambda: default_path,
    )
    state = make_gui_state()
    state.simulation.output_color_space = "ACES2065-1"

    saved_path = save_default_gui_state(state)
    loaded_state = load_default_gui_state()

    assert saved_path == default_path
    assert loaded_state == state

    clear_saved_default_gui_state()

    assert not default_path.exists()


def test_gui_state_from_dict_rejects_missing_fields() -> None:
    data = gui_state_to_dict(PROJECT_DEFAULT_GUI_STATE)
    del data["simulation"]

    with pytest.raises(ValueError, match="simulation"):
        gui_state_from_dict(data)