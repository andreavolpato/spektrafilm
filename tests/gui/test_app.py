from __future__ import annotations

from types import SimpleNamespace

from qtpy import QtGui

from spektrafilm_gui import app as app_module

from .helpers import StubToggle


def test_create_viewer_uses_system_dark_theme(monkeypatch) -> None:
    fake_viewer = object()
    fake_appearance = SimpleNamespace(theme=None)
    fake_settings = SimpleNamespace(appearance=fake_appearance)

    monkeypatch.setattr(
        app_module,
        'import_module',
        lambda name: SimpleNamespace(Viewer=lambda show=False: fake_viewer)
        if name == 'napari'
        else SimpleNamespace(get_settings=lambda: fake_settings),
    )

    viewer = app_module._create_viewer()

    assert viewer is fake_viewer
    assert fake_appearance.theme == 'dark'


def test_apply_system_palette_uses_dark_palette_when_system_prefers_dark(monkeypatch) -> None:
    captured: dict[str, object] = {}
    fake_style_hints = SimpleNamespace(setColorScheme=lambda scheme: captured.setdefault('scheme', scheme))
    dark_palette = QtGui.QPalette()
    dark_palette.setColor(QtGui.QPalette.Window, QtGui.QColor('#1e1e1e'))
    dark_palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor('#ffffff'))
    fake_app = SimpleNamespace(styleHints=lambda: fake_style_hints, palette=lambda: dark_palette)

    monkeypatch.setattr(app_module.QtWidgets.QApplication, 'instance', staticmethod(lambda: fake_app))

    app_module._apply_system_palette()

    assert captured['scheme'] == app_module.QtCore.Qt.ColorScheme.Dark


def test_apply_system_palette_falls_back_to_dark_palette_when_qt_palette_stays_light(monkeypatch) -> None:
    captured: dict[str, object] = {}
    fake_style_hints = SimpleNamespace(setColorScheme=lambda scheme: captured.setdefault('scheme', scheme))
    light_palette = QtGui.QPalette()
    light_palette.setColor(QtGui.QPalette.Window, QtGui.QColor('#f3f3f3'))
    light_palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor('#000000'))
    fake_app = SimpleNamespace(
        styleHints=lambda: fake_style_hints,
        palette=lambda: light_palette,
        setPalette=lambda palette: captured.setdefault('palette', palette),
    )

    monkeypatch.setattr(app_module.QtWidgets.QApplication, 'instance', staticmethod(lambda: fake_app))

    app_module._apply_system_palette()

    assert captured['scheme'] == app_module.QtCore.Qt.ColorScheme.Dark
    palette = captured['palette']
    assert isinstance(palette, QtGui.QPalette)
    assert palette.color(QtGui.QPalette.Window).name() == '#323232'
    assert palette.color(QtGui.QPalette.Base).name() == '#424242'
    assert palette.color(QtGui.QPalette.AlternateBase).name() == '#585858'


def test_create_app_syncs_display_transform_availability_before_connecting(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeController:
        def __init__(self, *, viewer, widgets) -> None:
            captured['controller_init'] = (viewer, widgets)

        def sync_display_transform_availability(self, *, report_status: bool) -> None:
            captured['sync'] = report_status

        def refresh_input_layers(self) -> None:
            captured['refreshed'] = True

    fake_viewer = object()
    fake_widgets = SimpleNamespace(display=SimpleNamespace(use_display_transform=object(), gray_18_canvas=StubToggle(True)))
    fake_panel_widgets = object()
    fake_controls_panel = object()
    fake_main_window = object()

    monkeypatch.setattr(app_module, 'warmup', lambda: None)
    monkeypatch.setattr(app_module, '_apply_system_palette', lambda: captured.setdefault('palette', True))
    monkeypatch.setattr(app_module, '_create_viewer', lambda: fake_viewer)
    monkeypatch.setattr(app_module, '_create_widgets', lambda: (fake_widgets, fake_panel_widgets))
    monkeypatch.setattr(app_module, 'load_default_gui_state', lambda: object())
    monkeypatch.setattr(app_module, 'apply_gui_state', lambda state, *, widgets: captured.setdefault('applied', (state, widgets)))
    monkeypatch.setattr(app_module, 'GuiController', FakeController)
    monkeypatch.setattr(app_module, '_connect_controller_signals', lambda controller, widgets: captured.setdefault('connected', (controller, widgets)))
    monkeypatch.setattr(
        app_module,
        'configure_napari_chrome',
        lambda viewer, *, gray_18_canvas=False: captured.setdefault('chrome', (viewer, gray_18_canvas)),
    )
    monkeypatch.setattr(app_module, 'build_controls_panel', lambda viewer, panel_widgets: fake_controls_panel)
    monkeypatch.setattr(app_module, 'build_main_window', lambda viewer, controls_panel: fake_main_window)

    app = app_module.create_app()

    assert captured['palette'] is True
    assert captured['sync'] is False
    assert captured['connected'][1] is fake_widgets
    assert captured['chrome'] == (fake_viewer, True)
    assert app.viewer is fake_viewer
    assert app.main_window is fake_main_window