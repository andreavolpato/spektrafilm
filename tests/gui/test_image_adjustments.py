from __future__ import annotations

import os

import numpy as np
import pytest

from spektrafilm_gui.image_adjustments import (
    ADJUSTMENT_LABELS,
    BasicAdjustmentsState,
    apply_basic_adjustments,
)


def _srgb_encode(linear: float) -> float:
    return 12.92 * linear if linear <= 0.0031308 else 1.055 * linear ** (1.0 / 2.4) - 0.055


def test_default_adjustments_leave_image_unchanged() -> None:
    image = np.array([[[0.0, 0.25, 1.0]]], dtype=np.float32)

    result = apply_basic_adjustments(image, BasicAdjustmentsState())

    np.testing.assert_array_equal(result, image)


def test_exposure_adjustment_uses_stops_in_linear_light() -> None:
    image = np.full((2, 2, 3), 0.5, dtype=np.float32)

    result = apply_basic_adjustments(image, BasicAdjustmentsState(exposure=1.0))

    np.testing.assert_allclose(result, _srgb_encode(0.21404114 * 2.0), atol=1e-5)
    assert np.all(result > image)


def test_detail_controls_change_local_contrast() -> None:
    image = np.zeros((24, 24, 3), dtype=np.float32)
    image[:, 12:] = 0.6

    result = apply_basic_adjustments(image, BasicAdjustmentsState(clarity=75.0, texture=50.0))

    assert not np.array_equal(result, image)
    assert np.isfinite(result).all()


def test_adjustment_widget_uses_horizontal_sliders() -> None:
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from qtpy import QtCore, QtWidgets
    from spektrafilm_gui.widget_sections import BasicAdjustmentsSection

    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    section = BasicAdjustmentsSection()

    assert set(section._editors) == set(ADJUSTMENT_LABELS)
    for editor in section._editors.values():
        assert editor._slider.orientation() == QtCore.Qt.Horizontal
        assert editor.value == 0.0
    assert section._editors['texture']._slider.orientation() == QtCore.Qt.Horizontal


def test_camera_exposure_compensation_is_a_horizontal_ev_slider() -> None:
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from qtpy import QtCore, QtWidgets
    from spektrafilm_gui.param_manifest import CAMERA_MANIFEST
    from spektrafilm_gui.widget_sections import ParamsGroupSection

    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    section = ParamsGroupSection(CAMERA_MANIFEST)
    editor = section.exposure_compensation_ev

    assert editor._slider.orientation() == QtCore.Qt.Horizontal
    assert editor.value == 0.0
    assert editor._label.text() == '0.0 EV'
