from __future__ import annotations

from spectral_film_lab.gui.theme_palette import (
    BASE_BG,
    CHECKED_BG,
    CONTROL_BG,
    CONTROL_BG_HOVER,
    CONTROL_BG_PRESSED,
    CONTROL_MENU_BG,
    CONTROL_SELECTION_BG,
    DISABLED_BG,
    FONT_SIZE_BASE,
    FONT_SIZE_TAB,
    FONT_WEIGHT_BOLD,
    FONT_WEIGHT_SEMIBOLD,
    HEADER_BG,
    PANEL_BG,
    PANEL_BG_ACTIVE,
    SCROLLBAR_HANDLE_BG,
    SIZE_BUTTON_MIN_HEIGHT,
    SIZE_BUTTON_PADDING,
    SIZE_CHECKBOX_INDICATOR,
    SIZE_CHECKBOX_SPACING,
    SIZE_COMBO_DROPDOWN_WIDTH,
    SIZE_CONTROL_MIN_HEIGHT,
    SIZE_CONTROL_PADDING,
    SIZE_FORM_SPACING,
    SIZE_HEADER_PADDING,
    SIZE_SCROLLBAR_HANDLE_MIN,
    SIZE_SPLITTER_HANDLE_WIDTH,
    SIZE_SPIN_BUTTON_WIDTH,
    SIZE_TAB_MARGIN_RIGHT,
    SIZE_TAB_PADDING,
    SIZE_TAB_STRIP_OFFSET,
    SIZE_TOOLBUTTON_PADDING,
    SIZE_TOOLBUTTON_POPUP_PADDING_RIGHT,
    SPLITTER_BG,
    STATUS_BG,
    TAB_SELECTED_BG,
    TEXT_BRIGHT,
    TEXT_CONTROL,
    TEXT_HEADER,
    TEXT_HEADER_SECTION,
    TEXT_PRIMARY,
    TEXT_SOFT,
    TEXT_STATUS,
)


def join_style_sections(*sections: str) -> str:
    return '\n\n'.join(section.strip() for section in sections if section.strip())


WINDOW_STYLE = f"""
QMainWindow {{
    background: {BASE_BG};
    color: {TEXT_PRIMARY};
}}

QWidget#appCentral {{
    background: {BASE_BG};
}}

QWidget {{
    background: {BASE_BG};
    color: {TEXT_PRIMARY};
    font-size: {FONT_SIZE_BASE};
}}

QFrame#sidebarPanel,
QFrame#viewerPanel {{
    background: {PANEL_BG};
    border: none;
    border-radius: 0;
}}

QFrame {{
    background: {PANEL_BG};
}}

QLabel#sidebarEyebrow {{
    color: {TEXT_HEADER};
    font-size: {FONT_SIZE_BASE};
    font-weight: {FONT_WEIGHT_SEMIBOLD};
    letter-spacing: 0.08em;
    text-transform: lowercase;
}}
"""

TAB_STYLE = f"""
QTabWidget,
QTabWidget#controlsTabWidget {{
    background: {BASE_BG};
}}

QTabWidget::pane,
QTabWidget#controlsTabWidget::pane {{
    border: none;
    border-radius: 0;
    background: {BASE_BG};
    top: {SIZE_TAB_STRIP_OFFSET};
}}

QTabWidget::tab-bar {{
    alignment: left;
}}

QTabBar {{
    background: {BASE_BG};
}}

QTabWidget#controlsTabWidget > QWidget {{
    background: {BASE_BG};
}}

QTabWidget#controlsTabWidget QStackedWidget > QWidget {{
    background: {CONTROL_BG};
}}

QTabBar::tab {{
    background: {BASE_BG};
    color: {TEXT_SOFT};
    border: none;
    border-top-left-radius: 0;
    border-top-right-radius: 0;
    font-size: {FONT_SIZE_TAB};
    font-weight: {FONT_WEIGHT_SEMIBOLD};
    padding: {SIZE_TAB_PADDING};
    margin-right: {SIZE_TAB_MARGIN_RIGHT};
    min-height: {SIZE_CONTROL_MIN_HEIGHT};
}}

QTabBar::tab:selected {{
    background: {TAB_SELECTED_BG};
    color: {TEXT_BRIGHT};
    font-weight: {FONT_WEIGHT_BOLD};
}}

QTabBar::tab:hover {{
    background: {PANEL_BG_ACTIVE};
}}
"""

CONTROL_STYLE = f"""
QPushButton,
QToolButton,
QComboBox,
QLineEdit,
QAbstractSpinBox,
QTextEdit,
QPlainTextEdit,
QListView,
QTreeView,
QScrollArea,
QGroupBox,
QMenu {{
    background: {CONTROL_BG};
    color: {TEXT_CONTROL};
    border: none;
    outline: none;
}}

QPushButton,
QToolButton,
QComboBox,
QLineEdit,
QAbstractSpinBox {{
    min-height: {SIZE_CONTROL_MIN_HEIGHT};
    padding: {SIZE_CONTROL_PADDING};
}}

QPushButton {{
    min-height: {SIZE_BUTTON_MIN_HEIGHT};
    padding: {SIZE_BUTTON_PADDING};
    text-align: center;
}}

QPushButton:hover,
QToolButton:hover,
QComboBox:hover,
QLineEdit:hover,
QAbstractSpinBox:hover {{
    background: {CONTROL_BG_HOVER};
}}

QPushButton:pressed,
QToolButton:pressed,
QComboBox:editable,
QComboBox:on {{
    background: {CONTROL_BG_PRESSED};
}}

QPushButton:focus,
QToolButton:focus,
QComboBox:focus,
QLineEdit:focus,
QAbstractSpinBox:focus,
QCheckBox:focus,
QTabBar::tab:focus {{
    border: none;
    outline: none;
}}

QComboBox::drop-down {{
    border: none;
    width: {SIZE_COMBO_DROPDOWN_WIDTH};
    background: transparent;
}}

QAbstractSpinBox::up-button,
QAbstractSpinBox::down-button {{
    background: transparent;
    border: none;
    width: {SIZE_SPIN_BUTTON_WIDTH};
    subcontrol-origin: border;
}}

QAbstractSpinBox::up-button:hover,
QAbstractSpinBox::down-button:hover,
QAbstractSpinBox::up-button:pressed,
QAbstractSpinBox::down-button:pressed {{
    background: transparent;
    border: none;
}}

QComboBox QAbstractItemView {{
    background: {CONTROL_MENU_BG};
    color: {TEXT_PRIMARY};
    border: none;
    selection-background-color: {CONTROL_SELECTION_BG};
    selection-color: {TEXT_BRIGHT};
    outline: none;
}}

QAbstractItemView {{
    alternate-background-color: {CONTROL_BG_HOVER};
    selection-background-color: {CONTROL_SELECTION_BG};
    selection-color: {TEXT_BRIGHT};
    border: none;
    outline: none;
}}

QHeaderView::section {{
    background: {HEADER_BG};
    color: {TEXT_HEADER_SECTION};
    border: none;
    padding: {SIZE_HEADER_PADDING};
}}

QCheckBox {{
    spacing: {SIZE_CHECKBOX_SPACING};
}}

QCheckBox::indicator {{
    width: {SIZE_CHECKBOX_INDICATOR};
    height: {SIZE_CHECKBOX_INDICATOR};
    border: none;
    background: {CONTROL_BG};
}}

QCheckBox::indicator:checked {{
    background: {CHECKED_BG};
}}

QCheckBox::indicator:checked:disabled,
QCheckBox::indicator:unchecked:disabled {{
    background: {DISABLED_BG};
}}

QToolButton {{
    background: {PANEL_BG};
    padding: {SIZE_TOOLBUTTON_PADDING};
}}

QToolButton:hover,
QToolButton:checked,
QToolButton:pressed {{
    background: {PANEL_BG};
}}

QToolButton[popupMode="1"] {{
    padding-right: {SIZE_TOOLBUTTON_POPUP_PADDING_RIGHT};
}}

QFormLayout {{
    spacing: {SIZE_FORM_SPACING};
}}
"""

CHROME_STYLE = f"""
QSplitter::handle {{
    background: {SPLITTER_BG};
    width: {SIZE_SPLITTER_HANDLE_WIDTH};
}}

QSplitter::handle:horizontal {{
    margin: 0;
}}

QScrollBar:vertical,
QScrollBar:horizontal {{
    background: {PANEL_BG};
    border: none;
    margin: 0;
}}

QScrollBar::handle:vertical,
QScrollBar::handle:horizontal {{
    background: {SCROLLBAR_HANDLE_BG};
    border: none;
    min-height: {SIZE_SCROLLBAR_HANDLE_MIN};
    min-width: {SIZE_SCROLLBAR_HANDLE_MIN};
}}

QScrollBar::add-line,
QScrollBar::sub-line,
QScrollBar::add-page,
QScrollBar::sub-page {{
    background: transparent;
    border: none;
}}

QFrame,
QGroupBox,
QDockWidget,
QToolBox,
QLabel,
QWidget[class="qt-scrollarea-viewport"] {{
    border: none;
}}

QStatusBar {{
    background: {STATUS_BG};
    color: {TEXT_STATUS};
    border: none;
}}

QStatusBar::item {{
    border: none;
}}
"""