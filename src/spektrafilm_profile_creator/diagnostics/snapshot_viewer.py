from __future__ import annotations

import importlib

from spektrafilm_profile_creator.diagnostics.messages import (
    clear_diagnostic_profile_snapshots,
    get_diagnostic_profile_snapshots,
)
from spektrafilm_profile_creator.plotting import plot_profile
from spektrafilm_profile_creator.workflows import process_profile


def list_diagnostic_profile_snapshots(snapshot_store=None):
    if snapshot_store is None:
        snapshot_store = get_diagnostic_profile_snapshots()

    entries = []
    for title, snapshots in snapshot_store.items():
        for entry in snapshots:
            label = f'{title} #{entry["sequence"]}'
            stock = entry.get('stock')
            if stock:
                label = f'{label} - {stock}'
            entries.append(
                {
                    'label': label,
                    'title': title,
                    'sequence': entry['sequence'],
                    'stock': stock,
                    'output': entry['output'],
                    'profile': entry['profile'],
                }
            )

    entries.sort(key=lambda entry: entry['sequence'])
    return entries


def _resolve_snapshot(selection=-1, snapshot_store=None, entries=None):
    if entries is None:
        entries = list_diagnostic_profile_snapshots(snapshot_store)
    if not entries:
        raise ValueError('No diagnostic profile snapshots available.')

    if isinstance(selection, str):
        for entry in entries:
            if entry['label'] == selection:
                return entry
        raise KeyError(f'Unknown diagnostic profile snapshot: {selection}')

    return entries[selection]


def plot_diagnostic_profile_snapshot(selection=-1, snapshot_store=None, **plot_kwargs):
    entry = _resolve_snapshot(selection, snapshot_store=snapshot_store)
    plot_profile(entry['profile'], **plot_kwargs)
    return entry


def create_qt_diagnostic_profile_snapshot_viewer(snapshot_store=None, **plot_kwargs):
    entries = list_diagnostic_profile_snapshots(snapshot_store)
    if not entries:
        raise ValueError('No diagnostic profile snapshots available.')

    qt_widgets = importlib.import_module('qtpy.QtWidgets')
    qt_core = importlib.import_module('qtpy.QtCore')
    backend = importlib.import_module('matplotlib.backends.backend_qtagg')
    figure_module = importlib.import_module('matplotlib.figure')

    widget = qt_widgets.QWidget()
    widget.setWindowTitle('Diagnostic Profile Snapshots')

    root = qt_widgets.QVBoxLayout(widget)
    controls = qt_widgets.QHBoxLayout()
    dropdown = qt_widgets.QComboBox()
    refresh_button = qt_widgets.QPushButton('Refresh')
    summary = qt_widgets.QLabel()
    output_box = qt_widgets.QPlainTextEdit()
    summary.setTextInteractionFlags(getattr(qt_core.Qt, 'TextSelectableByMouse'))
    output_box.setReadOnly(True)
    output_box.setMaximumBlockCount(1000)

    controls.addWidget(dropdown, 1)
    controls.addWidget(refresh_button)
    root.addLayout(controls)
    root.addWidget(summary)
    root.addWidget(output_box)

    figure = figure_module.Figure(figsize=(12, 4))
    canvas = backend.FigureCanvasQTAgg(figure)
    toolbar_class = getattr(backend, 'NavigationToolbar2QT', None)
    if toolbar_class is not None:
        root.addWidget(toolbar_class(canvas, widget))
    root.addWidget(canvas, 1)

    def render(selection):
        entry = _resolve_snapshot(selection, entries=entries)
        stock_suffix = f' | stock: {entry["stock"]}' if entry['stock'] else ''
        summary.setText(f'{entry["title"]} | sequence: {entry["sequence"]}{stock_suffix}')
        output_box.setPlainText(entry['output'])
        plot_profile(entry['profile'], figure=figure, **plot_kwargs)
        canvas.draw_idle()

    def refresh():
        nonlocal entries
        entries = list_diagnostic_profile_snapshots(snapshot_store)
        dropdown.blockSignals(True)
        dropdown.clear()
        for entry in entries:
            dropdown.addItem(entry['label'])

        if not entries:
            summary.setText('No diagnostic profile snapshots available.')
            output_box.clear()
            dropdown.setCurrentIndex(-1)
            dropdown.blockSignals(False)
            figure.clear()
            canvas.draw_idle()
            return

        dropdown.setCurrentIndex(len(entries) - 1)
        dropdown.blockSignals(False)
        render(dropdown.currentIndex())

    def on_change(index):
        if index >= 0:
            render(index)

    dropdown.currentIndexChanged.connect(on_change)
    refresh_button.clicked.connect(refresh)
    refresh()
    return widget


def _capture_snapshots_for_stock(stock='kodak_portra_400'):
    clear_diagnostic_profile_snapshots()
    process_profile(stock)
    entries = list_diagnostic_profile_snapshots()
    print(f'Captured {len(entries)} profile snapshots for {stock}:')
    for entry in entries:
        print(f'  - {entry["label"]}')
    return entries


def build_qt_snapshot_viewer(stock='kodak_portra_400', **plot_kwargs):
    _capture_snapshots_for_stock(stock)
    return create_qt_diagnostic_profile_snapshot_viewer(**plot_kwargs)


def launch_process_snapshot_viewer(
    stock='kodak_portra_400',
    *,
    size=(1280, 720),
    run_event_loop=True,
    **plot_kwargs,
):
    qt_widgets = importlib.import_module('qtpy.QtWidgets')

    app = qt_widgets.QApplication.instance()
    owns_app = app is None
    if owns_app:
        app = qt_widgets.QApplication([])

    viewer = build_qt_snapshot_viewer(stock=stock, **plot_kwargs)
    if size is not None:
        viewer.resize(*size)
    viewer.show()

    if owns_app and run_event_loop:
        app.exec()
    return viewer


__all__ = [
    'build_qt_snapshot_viewer',
    'create_qt_diagnostic_profile_snapshot_viewer',
    'launch_process_snapshot_viewer',
    'list_diagnostic_profile_snapshots',
    'plot_diagnostic_profile_snapshot',
]