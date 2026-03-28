from __future__ import annotations

from types import SimpleNamespace

import pytest

from spektrafilm_profile_creator.diagnostics.messages import clear_diagnostic_profile_snapshots

from tests.profiles_creator.create_profile_regression_baselines import compute_processed_profile, find_case


@pytest.fixture(autouse=True)
def clear_diagnostic_profile_snapshots_fixture():
    clear_diagnostic_profile_snapshots()
    yield
    clear_diagnostic_profile_snapshots()


@pytest.fixture(scope='session', name='portra_400_processed_profile')
def portra_400_processed_profile_fixture():
    case = find_case('create_profile_kodak_portra_400')
    return case, compute_processed_profile(case)


@pytest.fixture(scope='session', name='portra_endura_paper_processed_profile')
def portra_endura_paper_processed_profile_fixture():
    case = find_case('create_profile_kodak_portra_endura_paper')
    return case, compute_processed_profile(case)


class FakeSignal:
    def __init__(self) -> None:
        self.callbacks = []

    def connect(self, callback) -> None:
        self.callbacks.append(callback)

    def emit(self, value=None) -> None:
        for callback in self.callbacks:
            if value is None:
                callback()
            else:
                callback(value)


class FakeWidget:
    def __init__(self, text: str = '') -> None:
        self.text = text
        self.window_title = None
        self.layout = None
        self.flags = None
        self.read_only = False
        self.max_block_count = None

    def setWindowTitle(self, title):
        self.window_title = title

    def setText(self, text):
        self.text = text

    def setTextInteractionFlags(self, flags):
        self.flags = flags

    def setReadOnly(self, read_only):
        self.read_only = read_only

    def setMaximumBlockCount(self, count):
        self.max_block_count = count

    def setPlainText(self, text):
        self.text = text

    def clear(self):
        self.text = ''


class FakeLayout:
    def __init__(self, widget=None) -> None:
        self.items = []
        if widget is not None:
            widget.layout = self

    def addWidget(self, widget, stretch=0):
        self.items.append(('widget', widget, stretch))

    def addLayout(self, layout):
        self.items.append(('layout', layout))


class FakeComboBox:
    def __init__(self) -> None:
        self.items = []
        self.index = -1
        self.signals_blocked = False
        self.currentIndexChanged = FakeSignal()

    def clear(self):
        self.items = []
        self.index = -1

    def addItem(self, label):
        self.items.append(label)

    def setCurrentIndex(self, index):
        self.index = index
        if not self.signals_blocked:
            self.currentIndexChanged.emit(index)

    def currentIndex(self):
        return self.index

    def blockSignals(self, blocked):
        self.signals_blocked = blocked


class FakePushButton(FakeWidget):
    def __init__(self, text):
        super().__init__(text)
        self.clicked = FakeSignal()


class FakeFigure:
    def __init__(self, figsize=None) -> None:
        self.figsize = figsize
        self.cleared = 0

    def clear(self):
        self.cleared += 1


class FakeCanvas:
    def __init__(self, figure) -> None:
        self.figure = figure
        self.draw_calls = 0

    def draw_idle(self):
        self.draw_calls += 1


class FakeToolbar:
    def __init__(self, canvas, parent) -> None:
        self.canvas = canvas
        self.parent = parent


class SnapshotViewerHarness:
    def __init__(self) -> None:
        self.qt_widgets = SimpleNamespace(
            QWidget=FakeWidget,
            QVBoxLayout=FakeLayout,
            QHBoxLayout=FakeLayout,
            QComboBox=FakeComboBox,
            QPushButton=FakePushButton,
            QLabel=FakeWidget,
            QPlainTextEdit=FakeWidget,
        )
        self.qt_core = SimpleNamespace(Qt=SimpleNamespace(TextSelectableByMouse='selectable'))
        self.backend = SimpleNamespace(FigureCanvasQTAgg=FakeCanvas, NavigationToolbar2QT=FakeToolbar)
        self.figure_module = SimpleNamespace(Figure=FakeFigure)

    def import_module(self, name):
        mapping = {
            'qtpy.QtWidgets': self.qt_widgets,
            'qtpy.QtCore': self.qt_core,
            'matplotlib.backends.backend_qtagg': self.backend,
            'matplotlib.figure': self.figure_module,
        }
        return mapping[name]

    def unpack(self, viewer):
        root_layout = viewer.layout
        controls_layout = root_layout.items[0][1]
        return SimpleNamespace(
            dropdown=controls_layout.items[0][1],
            refresh_button=controls_layout.items[1][1],
            summary=root_layout.items[1][1],
            output_box=root_layout.items[2][1],
            canvas=root_layout.items[4][1],
        )


@pytest.fixture
def snapshot_viewer_harness():
    return SnapshotViewerHarness()