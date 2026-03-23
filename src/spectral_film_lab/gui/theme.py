from __future__ import annotations

from spectral_film_lab.gui import theme_styles

APP_STYLE_SHEET = theme_styles.join_style_sections(
    theme_styles.WINDOW_STYLE,
    theme_styles.TAB_STYLE,
    theme_styles.CONTROL_STYLE,
    theme_styles.CHROME_STYLE,
)
