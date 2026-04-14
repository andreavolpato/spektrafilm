from __future__ import annotations

from qtpy import QtCore, QtGui, QtWidgets

QPointF = getattr(QtCore, 'QPointF')
QSize = getattr(QtCore, 'QSize')

from spektrafilm_gui.theme_palette import (
    HEADER_DIVIDER_LINE,
    SIZE_FORM_SPACING,
    SIZE_SECTION_FRAME_INDENT,
    SIZE_SECTION_FRAME_MARGIN,
    SIZE_SECTION_STACK_SPACING,
)
from spektrafilm_gui.icons import HEADER_ICON_SIZE, section_header_icon
from spektrafilm_gui.theme import resolve_theme_qcolor


def normalize_ui_text(text: str) -> str:
    return text.lower()


def platform_default_font() -> QtGui.QFont:
    return QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.GeneralFont)


class CollapsibleSection(QtWidgets.QWidget):
    def __init__(self, title: str, content: QtWidgets.QWidget, *, expanded: bool = True):
        super().__init__()
        self._content = content

        self._toggle = QtWidgets.QToolButton()
        self._toggle.setProperty('role', 'sectionToggle')
        self._toggle.setCheckable(True)
        self._toggle.setChecked(expanded)
        self._toggle.setAutoRaise(True)
        self._toggle.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        self._toggle.setArrowType(QtCore.Qt.DownArrow if expanded else QtCore.Qt.RightArrow)
        self._toggle.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        self._toggle.setCursor(QtCore.Qt.PointingHandCursor)
        self._toggle.toggled.connect(self._set_expanded)

        self._icon_label = QtWidgets.QLabel()
        self._icon_label.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        self._icon_label.setAlignment(QtCore.Qt.AlignCenter)
        self._icon_label.setFixedSize(HEADER_ICON_SIZE, HEADER_ICON_SIZE)
        self._apply_header_icon(title)

        self._title_button = QtWidgets.QToolButton()
        self._title_button.setProperty('role', 'sectionToggle')
        self._title_button.setText(normalize_ui_text(title))
        self._title_button.setAutoRaise(True)
        self._title_button.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        self._title_button.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        self._title_button.setCursor(QtCore.Qt.PointingHandCursor)
        self._title_button.setFocusPolicy(QtCore.Qt.NoFocus)
        self._title_button.clicked.connect(self._toggle.toggle)

        self._header_line = HeaderDivider()
        self._header_line.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self._header_line.setFixedHeight(max(self._toggle.sizeHint().height(), self._title_button.sizeHint().height()))

        header = QtWidgets.QWidget()
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(int(SIZE_FORM_SPACING.removesuffix('px')))
        header_layout.setAlignment(QtCore.Qt.AlignVCenter)
        header_layout.addWidget(self._toggle, 0, QtCore.Qt.AlignVCenter)
        header_layout.addWidget(self._icon_label, 0, QtCore.Qt.AlignVCenter)
        header_layout.addWidget(self._title_button, 0, QtCore.Qt.AlignVCenter)
        header_layout.addWidget(self._header_line, 1, QtCore.Qt.AlignVCenter)

        self._frame = QtWidgets.QFrame()
        self._frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self._frame.setFrameShadow(QtWidgets.QFrame.Plain)
        frame_layout = QtWidgets.QVBoxLayout(self._frame)
        frame_layout.setContentsMargins(
            SIZE_SECTION_FRAME_INDENT,
            SIZE_SECTION_FRAME_MARGIN,
            SIZE_SECTION_FRAME_MARGIN,
            SIZE_SECTION_FRAME_MARGIN,
        )
        frame_layout.setSpacing(0)
        frame_layout.setAlignment(QtCore.Qt.AlignTop)
        frame_layout.addWidget(self._content)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(int(SIZE_SECTION_STACK_SPACING.removesuffix('px')))
        layout.setAlignment(QtCore.Qt.AlignTop)
        layout.addWidget(header)
        layout.addWidget(self._frame)

        self._set_expanded(expanded)

    def _apply_header_icon(self, title: str) -> None:
        icon = section_header_icon(title)
        if icon.isNull():
            self._icon_label.hide()
            return

        pixmap = icon.pixmap(HEADER_ICON_SIZE, HEADER_ICON_SIZE)
        if pixmap.isNull():
            self._icon_label.hide()
            return

        self._icon_label.setPixmap(pixmap)
        self._icon_label.show()

    def _set_expanded(self, expanded: bool) -> None:
        self._toggle.setArrowType(QtCore.Qt.DownArrow if expanded else QtCore.Qt.RightArrow)
        self._frame.setVisible(expanded)

    def has_header_icon(self) -> bool:
        pixmap = self._icon_label.pixmap()
        return not self._icon_label.isHidden() and pixmap is not None and not pixmap.isNull()


class HeaderDivider(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)

    def sizeHint(self):  # noqa: N802 - Qt API name
        return QSize(48, max(12, self.fontMetrics().height()))

    def minimumSizeHint(self):  # noqa: N802 - Qt API name
        return QSize(12, max(12, self.fontMetrics().height()))

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API name
        del event
        painter = QtGui.QPainter(self)
        try:
            painter.setRenderHint(QtGui.QPainter.Antialiasing, False)
            pen = QtGui.QPen(resolve_theme_qcolor(HEADER_DIVIDER_LINE))
            pen.setCosmetic(True)
            painter.setPen(pen)
            y = (self.height() / 2) + 1
            painter.drawLine(QPointF(0, y), QPointF(self.width(), y))
        finally:
            painter.end()