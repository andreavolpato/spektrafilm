from __future__ import annotations

import os

import numpy as np

from spektrafilm_gui.film_frame_dialog import FilmFrameDialog


def _qt_app():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from qtpy import QtWidgets

    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_dialog_selects_template_from_stock_and_exposes_only_allowed_options() -> None:
    app = _qt_app()
    dialog = FilmFrameDialog(np.full((48, 72, 3), 0.4, dtype=np.float32), film_stock_id='kodak_gold_200')

    assert '72 × 48 px（1.500:1），横向' in dialog.orientation_label.text()
    assert dialog.template_id == 'kodak_gold_200'
    assert 'GB 200-7' in dialog.template_status_label.text()
    assert '不代表该型号所有批次' in dialog.template_status_label.text()
    assert dialog.preview_label.pixmap().width() == 560
    assert dialog.options.show_date is False
    assert dialog.options.output_long_edge == 3800
    assert dialog.frame_number == '12'

    dialog.frame_number_checkbox.setChecked(False)
    dialog.date_checkbox.setChecked(True)
    dialog.fit_mode_combo.setCurrentIndex(dialog.fit_mode_combo.findData('cover'))
    dialog.pan_x_slider.setValue(20)
    dialog.resolution_combo.setCurrentIndex(0)
    options = dialog.options
    assert options.show_frame_number is False
    assert options.show_date is True
    assert options.fit_mode == 'cover'
    assert options.pan_x == 0.2
    assert options.output_long_edge == 1900
    assert '1900 × 1750 px' in dialog.resolution_label.text()

    dialog.close()
    app.processEvents()


def test_dialog_portrait_photo_uses_portrait_preview_and_generic_template_has_no_number() -> None:
    app = _qt_app()
    dialog = FilmFrameDialog(np.full((72, 48, 3), 0.4, dtype=np.float32), film_stock_id='unknown_stock')

    assert '竖向' in dialog.orientation_label.text()
    assert dialog.template_id == 'generic_135'
    assert dialog.frame_number is None
    assert not dialog.frame_number_edit.isEnabled()
    preview = dialog.preview_label.pixmap()
    assert preview.height() == 560 and preview.width() < preview.height()

    dialog.close()
    app.processEvents()
