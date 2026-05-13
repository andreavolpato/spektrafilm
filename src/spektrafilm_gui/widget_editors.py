from __future__ import annotations

from functools import lru_cache
from typing import Any

from qtpy import QtCore, QtGui, QtWidgets

QPointF = getattr(QtCore, 'QPointF')
QRect = getattr(QtCore, 'QRect')
QSize = getattr(QtCore, 'QSize')
QStyle = QtWidgets.QStyle
QStyleOptionComboBox = QtWidgets.QStyleOptionComboBox
QStyledItemDelegate = QtWidgets.QStyledItemDelegate
QStylePainter = QtWidgets.QStylePainter

from spektrafilm.profiles.io import load_profile
from spektrafilm_gui.theme_palette import (
    ACCENT_COLOR_TEXT,
    ACCENT_COLOR_TEXT_SECONDARY,
    BOOL_EDITOR_BORDER_CHECKED,
    BOOL_EDITOR_BORDER_UNCHECKED,
    BOOL_EDITOR_CHECKED_DISABLED,
    BOOL_EDITOR_CHECKED_ENABLED,
    BOOL_EDITOR_CHECKMARK,
    BOOL_EDITOR_FILL_DISABLED,
    BOOL_EDITOR_FILL_ENABLED,
    BOOL_EDITOR_HOVER_BG,
)
from spektrafilm_gui.theme import resolve_theme_qcolor


@lru_cache(maxsize=None)
def _profile_use_text_and_color(profile_name: str) -> tuple[str, str]:
    if not profile_name:
        return '', ACCENT_COLOR_TEXT
    try:
        profile = load_profile(profile_name)
    except (FileNotFoundError, TypeError, ValueError):
        return '', ACCENT_COLOR_TEXT
    if profile.is_cine:
        return 'cine', ACCENT_COLOR_TEXT_SECONDARY
    if profile.is_still:
        return 'still', ACCENT_COLOR_TEXT
    return '', ACCENT_COLOR_TEXT


def _draw_profile_display_text(
    painter: QtGui.QPainter,
    rect: QtCore.QRect,
    *,
    profile_name: str,
    base_color: QtGui.QColor,
    font: QtGui.QFont,
) -> None:
    profile_use, use_color = _profile_use_text_and_color(profile_name)
    if not profile_use:
        painter.setPen(base_color)
        painter.setFont(font)
        painter.drawText(rect, QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft, profile_name)
        return

    font_metrics = QtGui.QFontMetrics(font)
    prefix_text = f'{profile_use} '
    slash_text = '/ '
    prefix_width = font_metrics.horizontalAdvance(prefix_text)
    slash_width = font_metrics.horizontalAdvance(slash_text)

    painter.setFont(font)
    prefix_rect = QRect(rect.x(), rect.y(), prefix_width, rect.height())
    slash_rect = QRect(rect.x() + prefix_width, rect.y(), slash_width, rect.height())
    value_rect = QRect(rect.x() + prefix_width + slash_width, rect.y(), max(0, rect.width() - prefix_width - slash_width), rect.height())

    painter.setPen(QtGui.QColor(use_color))
    painter.drawText(prefix_rect, QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft | QtCore.Qt.TextSingleLine, prefix_text)
    painter.setPen(base_color)
    painter.drawText(slash_rect, QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft | QtCore.Qt.TextSingleLine, slash_text)
    painter.drawText(value_rect, QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft | QtCore.Qt.TextSingleLine, profile_name)


class ProfileEnumItemDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index) -> None:  # noqa: N802 - Qt API name
        display_value = index.data(QtCore.Qt.DisplayRole) or ''
        style_option = QtWidgets.QStyleOptionViewItem(option)
        self.initStyleOption(style_option, index)
        style_option.text = ''

        style = style_option.widget.style() if style_option.widget is not None else QtWidgets.QApplication.style()
        style.drawControl(QStyle.CE_ItemViewItem, style_option, painter, style_option.widget)

        text_color = style_option.palette.color(
            QtGui.QPalette.HighlightedText if style_option.state & QStyle.State_Selected else QtGui.QPalette.Text,
        )
        text_rect = style.subElementRect(QStyle.SE_ItemViewItemText, style_option, style_option.widget)
        _draw_profile_display_text(
            painter,
            text_rect.adjusted(0, 0, -4, 0),
            profile_name=display_value,
            base_color=text_color,
            font=style_option.font,
        )


class ProfileEnumEditor(QtWidgets.QComboBox):
    def __init__(self, values: list[str]):
        super().__init__()
        self.addItems(values)
        self.setItemDelegate(ProfileEnumItemDelegate(self))
        if self.count() > 0 and self.currentIndex() < 0:
            self.setCurrentIndex(0)

    @staticmethod
    def display_text_for_value(value: str) -> str:
        profile_use, _color = _profile_use_text_and_color(value)
        if not profile_use:
            return value
        return f'{profile_use} / {value}'

    @property
    def value(self) -> str:
        return self.currentText()

    @value.setter
    def value(self, value: str) -> None:
        index = self.findText(value)
        if index >= 0:
            self.setCurrentIndex(index)
        else:
            raise ValueError(f"{value!r} is not a valid option")

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API name
        del event
        painter = QStylePainter(self)
        option = QStyleOptionComboBox()
        self.initStyleOption(option)
        current_value = self.currentText() or (self.itemText(0) if self.count() > 0 else '')
        option.currentText = ''
        painter.drawComplexControl(QStyle.CC_ComboBox, option)
        painter.drawControl(QStyle.CE_ComboBoxLabel, option)
        text_rect = self.style().subControlRect(QStyle.CC_ComboBox, option, QStyle.SC_ComboBoxEditField, self)
        _draw_profile_display_text(
            painter,
            text_rect.adjusted(1, 0, -1, 0),
            profile_name=current_value,
            base_color=option.palette.color(QtGui.QPalette.Text),
            font=self.font(),
        )


class FloatEditor(QtWidgets.QDoubleSpinBox):
    def __init__(self, *, decimals: int = 2, minimum: float = -1_000_000.0, maximum: float = 1_000_000.0):
        super().__init__()
        self.setDecimals(decimals)
        self.setMinimum(minimum)
        self.setMaximum(maximum)
        self.setKeyboardTracking(False)
        self.setFixedHeight(24)

    @property
    def value(self) -> float:
        return super().value()

    @value.setter
    def value(self, value: float) -> None:
        self.setValue(value)


class IntEditor(QtWidgets.QSpinBox):
    def __init__(self, *, minimum: int = -1_000_000, maximum: int = 1_000_000):
        super().__init__()
        self.setMinimum(minimum)
        self.setMaximum(maximum)
        self.setKeyboardTracking(False)
        self.setFixedHeight(24)

    @property
    def value(self) -> int:
        return super().value()

    @value.setter
    def value(self, value: int) -> None:
        self.setValue(value)


class BoolEditor(QtWidgets.QCheckBox):
    def __init__(self):
        super().__init__()
        self.setText('')
        self.setMouseTracking(True)
        self.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.setFixedHeight(24)

    @property
    def value(self) -> bool:
        return self.isChecked()

    @value.setter
    def value(self, value: bool) -> None:
        self.setChecked(value)

    def sizeHint(self):  # noqa: N802 - Qt API name
        return QSize(24, 24)

    def minimumSizeHint(self):  # noqa: N802 - Qt API name
        return self.sizeHint()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API name
        del event
        painter = QtGui.QPainter(self)
        try:
            painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

            indicator_rect = self._indicator_rect()
            is_enabled = self.isEnabled()
            is_hovered = is_enabled and self.underMouse()
            fill_color = resolve_theme_qcolor(
                BOOL_EDITOR_HOVER_BG if is_hovered else BOOL_EDITOR_FILL_ENABLED if is_enabled else BOOL_EDITOR_FILL_DISABLED,
            )
            checked_color = resolve_theme_qcolor(
                BOOL_EDITOR_HOVER_BG if is_hovered else BOOL_EDITOR_CHECKED_ENABLED if is_enabled else BOOL_EDITOR_CHECKED_DISABLED,
            )
            border_color = resolve_theme_qcolor(BOOL_EDITOR_BORDER_CHECKED if self.isChecked() else BOOL_EDITOR_BORDER_UNCHECKED)

            painter.setPen(QtGui.QPen(border_color, 1))
            painter.setBrush(checked_color if self.isChecked() else fill_color)
            painter.drawRect(indicator_rect)

            if self.isChecked():
                self._draw_check_mark(painter, indicator_rect)
        finally:
            painter.end()

    def enterEvent(self, event) -> None:  # noqa: N802 - Qt API name
        super().enterEvent(event)
        self.update()

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt API name
        super().leaveEvent(event)
        self.update()

    def _indicator_rect(self):
        return QRect(1, max(1, (self.height() - 14) // 2), 14, 14)

    @staticmethod
    def _draw_check_mark(painter: QtGui.QPainter, indicator_rect) -> None:
        painter.setPen(
            QtGui.QPen(resolve_theme_qcolor(BOOL_EDITOR_CHECKMARK), 1.6, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin),
        )
        painter.drawLine(
            QPointF(indicator_rect.left() + 3.0, indicator_rect.center().y() + 0.5),
            QPointF(indicator_rect.left() + 6.0, indicator_rect.bottom() - 3.5),
        )
        painter.drawLine(
            QPointF(indicator_rect.left() + 6.0, indicator_rect.bottom() - 3.5),
            QPointF(indicator_rect.right() - 2.5, indicator_rect.top() + 3.5),
        )


class EnumEditor(QtWidgets.QComboBox):
    def __init__(self, values: list[str]):
        super().__init__()
        self.addItems(values)

    @property
    def value(self) -> str:
        return self.currentText()

    @value.setter
    def value(self, value: str) -> None:
        index = self.findText(value)
        if index >= 0:
            self.setCurrentIndex(index)
        else:
            raise ValueError(f"{value!r} is not a valid option")


class TupleEditor(QtWidgets.QWidget):
    def __init__(self, editors: list[QtWidgets.QWidget]):
        super().__init__()
        self._editors = editors
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        for editor in self._editors:
            editor.setMinimumWidth(56)
            editor.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
            layout.addWidget(editor)
        self.setLayout(layout)

    @property
    def editors(self) -> tuple[QtWidgets.QWidget, ...]:
        return tuple(self._editors)

    @property
    def value(self) -> tuple[Any, ...]:
        return tuple(editor.value for editor in self._editors)

    @value.setter
    def value(self, value: tuple[Any, ...]) -> None:
        for editor, component in zip(self._editors, value, strict=True):
            editor.value = component

    def setToolTip(self, text: str) -> None:  # noqa: N802 - Qt API name
        super().setToolTip(text)
        for editor in self._editors:
            editor.setToolTip(text)


class FloatTupleEditor(TupleEditor):
    def __init__(self, length: int, *, decimals: int = 2, minimum: float = -1_000_000.0, maximum: float = 1_000_000.0):
        super().__init__([FloatEditor(decimals=decimals, minimum=minimum, maximum=maximum) for _ in range(length)])


class IntTupleEditor(TupleEditor):
    def __init__(self, length: int, *, minimum: int = -1_000_000, maximum: int = 1_000_000):
        super().__init__([IntEditor(minimum=minimum, maximum=maximum) for _ in range(length)])