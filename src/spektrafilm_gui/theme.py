from __future__ import annotations

from qtpy import QtGui

from spektrafilm_gui import theme_styles

APP_STYLE_SHEET = theme_styles.join_style_sections(
    theme_styles.WINDOW_STYLE,
    theme_styles.TAB_STYLE,
    theme_styles.CONTROL_STYLE,
    theme_styles.CHROME_STYLE,
)


def resolve_theme_qcolor(color_spec: str) -> QtGui.QColor:
    return QtGui.QColor(color_spec)


def resolve_theme_color_name(color_spec: str) -> str:
    return resolve_theme_qcolor(color_spec).name()
