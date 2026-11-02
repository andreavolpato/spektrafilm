from __future__ import annotations

import os

import numpy as np
import pytest

from spektrafilm_gui.film_frame_dialog import FilmFrameDialog


def _qt_app():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from qtpy import QtWidgets

    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_film_frame_dialog_previews_landscape_photo_and_exposes_adjustments() -> None:
    app = _qt_app()
    dialog = FilmFrameDialog(
        np.full((48, 72, 3), 0.4, dtype=np.float32),
        film_stock_id='kodak_portra_400',
        date_label='25 7 24',
    )

    assert '72 × 48 px（1.500:1），横向' in dialog.orientation_label.text()
    preview = dialog.preview_label.pixmap()
    assert preview is not None
    assert preview.width() / preview.height() == pytest.approx(1.05, abs=0.03)
    assert dialog.template_id == 'kodak_portra_400'
    assert '部分核实' in dialog.template_status_label.text()
    assert dialog.options.border_width == 0.15
    assert dialog.options.show_date is False
    assert dialog.frame_number_checkbox.isEnabled()

    dialog.border_width_slider.setValue(18)
    dialog.fit_mode_combo.setCurrentIndex(dialog.fit_mode_combo.findData('cover'))
    dialog.date_checkbox.setChecked(True)
    dialog.frame_number_edit.setText('75')

    assert dialog.border_width_value.text() == '18%'
    assert dialog.options.border_width == 0.18
    assert dialog.options.fit_mode == 'cover'
    assert dialog.options.show_date is True
    assert dialog.frame_number_edit.text() == '75'

    dialog.close()
    app.processEvents()


def test_film_frame_dialog_disables_unverified_edge_numbering() -> None:
    app = _qt_app()
    dialog = FilmFrameDialog(
        np.full((72, 48, 3), 0.4, dtype=np.float32),
        film_stock_id='fujifilm_c200',
    )

    assert '竖向' in dialog.orientation_label.text()
    assert '待核实' in dialog.template_status_label.text()
    assert dialog.template_id == 'fujifilm_c200'
    assert not dialog.frame_number_checkbox.isEnabled()
    assert not dialog.frame_number_edit.isEnabled()
    assert dialog.options.show_date is False

    dialog.close()
    app.processEvents()
