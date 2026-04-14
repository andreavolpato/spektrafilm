from types import SimpleNamespace

from spektrafilm_profile_creator.diagnostics.snapshot_viewer import (
    create_qt_diagnostic_profile_snapshot_viewer,
    launch_process_snapshot_viewer,
    list_diagnostic_profile_snapshots,
)
from spektrafilm_profile_creator.diagnostics.messages import log_event
import spektrafilm_profile_creator.diagnostics.snapshot_viewer as snapshot_viewer_module

from tests.profiles_creator.helpers import make_test_profile


def test_list_diagnostic_profile_snapshots_returns_sequence_order() -> None:
    log_event('later_stage', make_test_profile('stock_a'))
    log_event('earlier_stage', make_test_profile('stock_b'))
    log_event('later_stage', make_test_profile('stock_c'))

    entries = list_diagnostic_profile_snapshots()

    assert [entry['label'] for entry in entries] == [
        'later_stage #1 - stock_a',
        'earlier_stage #2 - stock_b',
        'later_stage #3 - stock_c',
    ]


def test_create_qt_diagnostic_profile_snapshot_viewer_builds_selector_and_redraws(monkeypatch, snapshot_viewer_harness) -> None:
    profile_a = make_test_profile('stock_a')
    profile_b = make_test_profile('stock_b')
    log_event('second_stage', profile_a)
    log_event('first_stage', profile_b)
    plotted = []

    monkeypatch.setattr(snapshot_viewer_module.importlib, 'import_module', snapshot_viewer_harness.import_module)
    monkeypatch.setattr(
        snapshot_viewer_module,
        'plot_profile',
        lambda profile, figure=None, **kwargs: plotted.append((profile.info.stock, figure)),
    )

    viewer = create_qt_diagnostic_profile_snapshot_viewer()

    controls = snapshot_viewer_harness.unpack(viewer)

    assert viewer.window_title == 'Diagnostic Profile Snapshots'
    assert controls.dropdown.items == ['second_stage #1 - stock_a', 'first_stage #2 - stock_b']
    assert controls.dropdown.currentIndex() == 1
    assert plotted[0][0] == 'stock_b'
    assert controls.summary.text == 'first_stage | sequence: 2 | stock: stock_b'
    assert 'first_stage' in controls.output_box.text
    assert controls.canvas.draw_calls == 1

    controls.dropdown.setCurrentIndex(0)

    assert plotted[1][0] == 'stock_a'
    assert controls.summary.text == 'second_stage | sequence: 1 | stock: stock_a'
    assert controls.canvas.draw_calls == 2

    controls.refresh_button.clicked.emit()

    assert controls.dropdown.currentIndex() == 1
    assert plotted[-1][0] == 'stock_b'


def test_launch_qt_snapshot_viewer_builds_and_shows_viewer(monkeypatch) -> None:
    events = {}

    class FakeApplication:
        @staticmethod
        def instance():
            return None

        def __init__(self, args):
            events['args'] = args

        def exec(self):
            events['exec_called'] = True

    fake_qt_widgets = SimpleNamespace(QApplication=FakeApplication)

    class FakeViewer:
        def resize(self, width, height):
            events['size'] = (width, height)

        def show(self):
            events['shown'] = True

    def fake_build_qt_snapshot_viewer(stock='kodak_portra_400', **kwargs):
        events['build'] = (stock, kwargs)
        return FakeViewer()

    monkeypatch.setattr(snapshot_viewer_module.importlib, 'import_module', lambda name: fake_qt_widgets)
    monkeypatch.setattr(
        snapshot_viewer_module,
        'build_qt_snapshot_viewer',
        fake_build_qt_snapshot_viewer,
    )

    viewer = launch_process_snapshot_viewer('kodak_gold_200', size=(900, 480), run_event_loop=True)

    assert events['build'][0] == 'kodak_gold_200'
    assert events['size'] == (900, 480)
    assert events['shown'] is True
    assert events['exec_called'] is True
    assert viewer is not None