from __future__ import annotations

from functools import cache
from typing import Any

from qtpy import QtCore, QtGui, QtWidgets

QPointF = getattr(QtCore, "QPointF")
QRect = getattr(QtCore, "QRect")
QSize = getattr(QtCore, "QSize")
QStyle = QtWidgets.QStyle
QStyleOptionComboBox = QtWidgets.QStyleOptionComboBox
QStyledItemDelegate = QtWidgets.QStyledItemDelegate
QStylePainter = QtWidgets.QStylePainter

from spektrafilm.data.profiles_loader import load_profile
from spektrafilm_gui.theme import resolve_theme_qcolor
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
    TEXT_DIM,
)


@cache
def _profile_meta_tokens(profile_name: str) -> tuple[str, ...]:
    """Metadata badge tokens for a profile, e.g. ('still', 'neg', 'color').

    Reads use (still/cine), type (neg/pos) and channel model (color/bw) from
    the profile's info. Returns an empty tuple when the profile can't be
    loaded, so the editor falls back to showing the bare stock name.
    """
    if not profile_name:
        return ()
    try:
        profile = load_profile(profile_name)
    except (FileNotFoundError, TypeError, ValueError):
        return ()
    use = "cine" if profile.is_cine else "still" if profile.is_still else ""
    profile_type = (
        "pos" if profile.is_positive else "neg" if profile.is_negative else ""
    )
    channel = "bw" if profile.is_bw else "color" if profile.is_color else ""
    return tuple(token for token in (use, profile_type, channel) if token)


def _profile_token_color(token: str) -> str:
    """Accent color for the use token (still/cine); gray for everything else
    (type/channel badges)."""
    if token == "still":
        return ACCENT_COLOR_TEXT
    if token == "cine":
        return ACCENT_COLOR_TEXT_SECONDARY
    return TEXT_DIM


def _draw_profile_display_text(
    painter: QtGui.QPainter,
    rect: QtCore.QRect,
    *,
    profile_name: str,
    base_color: QtGui.QColor,
    font: QtGui.QFont,
) -> None:
    painter.setFont(font)
    tokens = _profile_meta_tokens(profile_name)
    if not tokens:
        painter.setPen(base_color)
        painter.drawText(
            rect, QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft, profile_name
        )
        return

    # Per-segment coloring: still/cine in their accent colors, the type/channel
    # badges and the "/" bars in gray, then the stock name in the base color.
    bar_color = QtGui.QColor(TEXT_DIM)
    segments: list[tuple[str, QtGui.QColor]] = []
    for token in tokens:
        segments.append((token, QtGui.QColor(_profile_token_color(token))))
        segments.append((" / ", bar_color))
    segments.append((profile_name, base_color))

    font_metrics = QtGui.QFontMetrics(font)
    x = rect.x()
    right = rect.x() + rect.width()
    for text, color in segments:
        if x >= right:
            break
        width = font_metrics.horizontalAdvance(text)
        seg_rect = QRect(x, rect.y(), min(width, right - x), rect.height())
        painter.setPen(color)
        painter.drawText(
            seg_rect,
            QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft | QtCore.Qt.TextSingleLine,
            text,
        )
        x += width


class ProfileEnumItemDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index) -> None:  # noqa: N802 - Qt API name
        display_value = index.data(QtCore.Qt.DisplayRole) or ""
        style_option = QtWidgets.QStyleOptionViewItem(option)
        self.initStyleOption(style_option, index)
        style_option.text = ""

        style = (
            style_option.widget.style()
            if style_option.widget is not None
            else QtWidgets.QApplication.style()
        )
        style.drawControl(
            QStyle.CE_ItemViewItem, style_option, painter, style_option.widget
        )

        text_color = style_option.palette.color(
            QtGui.QPalette.HighlightedText
            if style_option.state & QStyle.State_Selected
            else QtGui.QPalette.Text,
        )
        text_rect = style.subElementRect(
            QStyle.SE_ItemViewItemText, style_option, style_option.widget
        )
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
        tokens = _profile_meta_tokens(value)
        if not tokens:
            return value
        return f"{' / '.join(tokens)} / {value}"

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
        current_value = self.currentText() or (
            self.itemText(0) if self.count() > 0 else ""
        )
        option.currentText = ""
        painter.drawComplexControl(QStyle.CC_ComboBox, option)
        painter.drawControl(QStyle.CE_ComboBoxLabel, option)
        text_rect = self.style().subControlRect(
            QStyle.CC_ComboBox, option, QStyle.SC_ComboBoxEditField, self
        )
        _draw_profile_display_text(
            painter,
            text_rect.adjusted(1, 0, -1, 0),
            profile_name=current_value,
            base_color=option.palette.color(QtGui.QPalette.Text),
            font=self.font(),
        )


class FloatEditor(QtWidgets.QDoubleSpinBox):
    def __init__(
        self,
        *,
        decimals: int = 2,
        minimum: float = -1_000_000.0,
        maximum: float = 1_000_000.0,
    ):
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
        self.setText("")
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
                BOOL_EDITOR_HOVER_BG
                if is_hovered
                else BOOL_EDITOR_FILL_ENABLED
                if is_enabled
                else BOOL_EDITOR_FILL_DISABLED,
            )
            checked_color = resolve_theme_qcolor(
                BOOL_EDITOR_HOVER_BG
                if is_hovered
                else BOOL_EDITOR_CHECKED_ENABLED
                if is_enabled
                else BOOL_EDITOR_CHECKED_DISABLED,
            )
            border_color = resolve_theme_qcolor(
                BOOL_EDITOR_BORDER_CHECKED
                if self.isChecked()
                else BOOL_EDITOR_BORDER_UNCHECKED
            )

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
            QtGui.QPen(
                resolve_theme_qcolor(BOOL_EDITOR_CHECKMARK),
                1.6,
                QtCore.Qt.SolidLine,
                QtCore.Qt.RoundCap,
                QtCore.Qt.RoundJoin,
            ),
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


class StrEditor(QtWidgets.QLineEdit):
    """Free-text editor for a plain ``str`` field (e.g. the calibration matrix
    string). Emits ``editingFinished`` on commit, which the auto-preview wiring
    listens for (so it does not re-render on every keystroke)."""

    @property
    def value(self) -> str:
        return self.text()

    @value.setter
    def value(self, value: str) -> None:
        self.setText("" if value is None else str(value))


class DevelopmentTimeEditor(QtWidgets.QComboBox):
    """Dropdown of a BW stock's available development times.

    The dropdown always lists at least one concrete development time and never a
    'none' entry: a real family lists all of its times (defaulting to the
    floor-middle one), and a stock with no family (color, or single-time BW)
    shows its own single value or a ``1.0`` fallback. The value is a ``float``.
    Choices are stock-dependent, so the controller repopulates them via
    :meth:`set_choices` when the profile changes.
    """

    def __init__(self, times: list[float] | None = None):
        super().__init__()
        self._values: list[float] = []
        self.set_choices(times)

    def set_choices(self, times: list[float] | None) -> None:
        previous = self.value if self.count() else None
        # Always offer at least one concrete time; never an empty/'none' entry.
        times = [float(time) for time in (times or ())] or [1.0]
        self.blockSignals(True)
        self.clear()
        self._values = list(times)
        for time in times:
            self.addItem(f"{time:g} min")
        self.blockSignals(False)
        if previous is not None and previous in self._values:
            # Keep an explicit prior selection that still exists in the new stock.
            self.value = previous
        else:
            # No (still-valid) prior choice: default to the floor-middle time.
            self.value = None

    @property
    def value(self) -> float:
        index = self.currentIndex()
        if 0 <= index < len(self._values):
            return self._values[index]
        return self._values[(len(self._values) - 1) // 2]

    @value.setter
    def value(self, value: float | None) -> None:
        if value is not None and float(value) in self._values:
            self.setCurrentIndex(self._values.index(float(value)))
        else:
            # None ("no explicit choice") or an unavailable time: floor-middle.
            self.setCurrentIndex((len(self._values) - 1) // 2)


class TupleEditor(QtWidgets.QWidget):
    def __init__(self, editors: list[QtWidgets.QWidget]):
        super().__init__()
        self._editors = editors
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        for editor in self._editors:
            editor.setMinimumWidth(56)
            editor.setSizePolicy(
                QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
            )
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
    def __init__(
        self,
        length: int,
        *,
        decimals: int = 2,
        minimum: float = -1_000_000.0,
        maximum: float = 1_000_000.0,
    ):
        super().__init__(
            [
                FloatEditor(decimals=decimals, minimum=minimum, maximum=maximum)
                for _ in range(length)
            ]
        )


class IntTupleEditor(TupleEditor):
    def __init__(
        self, length: int, *, minimum: int = -1_000_000, maximum: int = 1_000_000
    ):
        super().__init__(
            [IntEditor(minimum=minimum, maximum=maximum) for _ in range(length)]
        )


# ruff: noqa: N802, N815
class SliderFloatEditor(QtWidgets.QWidget):
    valueChanged = QtCore.Signal(float)

    def __init__(
        self,
        *,
        minimum: float = 0.0,
        maximum: float = 100.0,
        step: float = 1.0,
        decimals: int = 0,
        suffix: str = "",
    ):
        super().__init__()
        self._minimum = minimum
        self._maximum = maximum
        self._step = step
        self._decimals = decimals
        self._suffix = suffix

        self._slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._slider.setRange(0, self._n_steps())
        self._slider.setFixedHeight(24)

        self._label = QtWidgets.QLabel()
        self._label.setFixedWidth(52)
        self._label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self._slider, 1)
        layout.addWidget(self._label)

        self._slider.valueChanged.connect(self._on_slider_changed)
        self._update_label()

    def _n_steps(self) -> int:
        if self._step <= 0:
            return 0
        return int(round((self._maximum - self._minimum) / self._step))

    def _on_slider_changed(self, _tick: int) -> None:
        self._update_label()
        self.valueChanged.emit(self.value)

    def _update_label(self) -> None:
        self._label.setText(f"{self.value:.{self._decimals}f}{self._suffix}")

    @property
    def value(self) -> float:
        return self._minimum + self._slider.value() * self._step

    @value.setter
    def value(self, v: float) -> None:
        tick = int(round((v - self._minimum) / self._step))
        tick = max(0, min(self._slider.maximum(), tick))
        was_blocked = self._slider.blockSignals(True)
        self._slider.setValue(tick)
        self._slider.blockSignals(was_blocked)
        self._update_label()

    def setMinimum(self, v: float) -> None:
        self._minimum = v
        self._slider.setRange(0, self._n_steps())

    def setMaximum(self, v: float) -> None:
        self._maximum = v
        self._slider.setRange(0, self._n_steps())

    def setSingleStep(self, v: float) -> None:
        self._step = v
        self._slider.setRange(0, self._n_steps())

    def setToolTip(self, text: str) -> None:
        super().setToolTip(text)
        self._slider.setToolTip(text)
        self._label.setToolTip(text)
