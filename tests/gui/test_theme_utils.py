from __future__ import annotations

from spektrafilm_gui import theme as theme_utils_module


def test_resolve_theme_color_name_preserves_literal_colors() -> None:
    assert theme_utils_module.resolve_theme_color_name('#123456') == '#123456'


def test_resolve_theme_qcolor_preserves_literal_colors() -> None:
    color = theme_utils_module.resolve_theme_qcolor('#abcdef')

    assert color.name() == '#abcdef'