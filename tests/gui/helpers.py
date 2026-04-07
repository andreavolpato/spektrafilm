from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from spektrafilm_gui.state import GuiState, PROJECT_DEFAULT_GUI_STATE, clone_gui_state


class FakeLayer:
    def __init__(
        self,
        data: np.ndarray | None = None,
        metadata: dict[str, object] | None = None,
        *,
        name: str = 'layer',
        visible: bool = True,
    ):
        self.name = name
        self.data = np.zeros((1, 1, 3), dtype=np.float32) if data is None else data
        self.metadata = metadata or {}
        self.visible = visible
        self.scale = (1.0, 1.0)
        self.translate = (0.0, 0.0)
        self._type_string = 'image'


class FakeLayerSelection:
    def __init__(self, active=None) -> None:
        self.active = active


class FakeLayerList(list):
    def __init__(self, layers: list[FakeLayer], *, active=None) -> None:
        super().__init__(layers)
        self.selection = FakeLayerSelection(active=active)

    def move(self, src: int, dst: int) -> None:
        layer = self.pop(src)
        insert_at = dst - 1 if dst > src else dst
        self.insert(insert_at, layer)


class FakeViewer:
    def __init__(self, layers: list[FakeLayer] | None = None):
        self.layers = FakeLayerList(layers or [])
        self.reset_view_calls = 0

    def add_image(self, image: np.ndarray, name: str) -> FakeLayer:
        layer = FakeLayer(image, name=name)
        self.layers.append(layer)
        return layer

    def reset_view(self) -> None:
        self.reset_view_calls += 1


class StubToggle:
    def __init__(self, checked: bool = True) -> None:
        self.checked = checked
        self._signals_blocked = False

    def isChecked(self) -> bool:  # noqa: N802 - Qt API name
        return self.checked

    def blockSignals(self, blocked: bool) -> bool:  # noqa: N802 - Qt API name
        previous = self._signals_blocked
        self._signals_blocked = blocked
        return previous

    def setChecked(self, checked: bool) -> None:  # noqa: N802 - Qt API name
        self.checked = checked


def make_test_gui_state() -> GuiState:
    return clone_gui_state(PROJECT_DEFAULT_GUI_STATE)


def make_test_controller_gui_state() -> GuiState:
    state = make_test_gui_state()
    state.simulation.output_color_space = 'ACES2065-1'
    state.simulation.saving_color_space = 'Display P3'
    state.simulation.saving_cctf_encoding = False
    state.display.use_display_transform = False
    state.display.gray_18_canvas = False
    state.display.white_padding = 0.0
    return state


def make_test_viewer_namespace(**kwargs):
    return SimpleNamespace(window=SimpleNamespace(**kwargs))