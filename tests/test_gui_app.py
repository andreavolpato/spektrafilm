from __future__ import annotations

from types import SimpleNamespace

from spektrafilm_gui import app as app_module


def test_create_app_syncs_display_transform_availability_before_connecting(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeController:
        def __init__(self, *, viewer, widgets) -> None:
            captured["controller_init"] = (viewer, widgets)

        def sync_display_transform_availability(self, *, report_status: bool) -> None:
            captured["sync"] = report_status

        def refresh_input_layers(self) -> None:
            captured["refreshed"] = True

    fake_viewer = object()
    fake_widgets = SimpleNamespace(display=SimpleNamespace(use_display_transform=object()))
    fake_panel_widgets = object()
    fake_controls_panel = object()
    fake_main_window = object()

    monkeypatch.setattr(app_module, "warmup", lambda: None)
    monkeypatch.setattr(app_module, "_create_viewer", lambda: fake_viewer)
    monkeypatch.setattr(app_module, "_create_widgets", lambda: (fake_widgets, fake_panel_widgets))
    monkeypatch.setattr(app_module, "load_default_gui_state", lambda: object())
    monkeypatch.setattr(app_module, "apply_gui_state", lambda state, *, widgets: captured.setdefault("applied", (state, widgets)))
    monkeypatch.setattr(app_module, "GuiController", FakeController)
    monkeypatch.setattr(app_module, "_connect_controller_signals", lambda controller, widgets: captured.setdefault("connected", (controller, widgets)))
    monkeypatch.setattr(app_module, "configure_napari_chrome", lambda viewer: captured.setdefault("chrome", viewer))
    monkeypatch.setattr(app_module, "build_controls_panel", lambda viewer, panel_widgets: fake_controls_panel)
    monkeypatch.setattr(app_module, "build_main_window", lambda viewer, controls_panel: fake_main_window)

    app = app_module.create_app()

    assert captured["sync"] is False
    assert captured["connected"][1] is fake_widgets
    assert app.viewer is fake_viewer
    assert app.main_window is fake_main_window