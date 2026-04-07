from __future__ import annotations

from functools import lru_cache

from qtpy import QtGui

from spektrafilm_gui.theme_palette import ACCENT_COLOR_TEXT

try:
    import pyconify
except ImportError:  # pragma: no cover - exercised in runtime fallback only
    pyconify = None


HEADER_ICON_SIZE = 16

_SECTION_HEADER_ICONS = {
    'import rgb': 'tabler:photo-plus',
    'import raw': 'tabler:photo-cog',
    'input': 'tabler:arrow-big-down-lines',
    'camera': 'tabler:camera',
    'profiles': 'tabler:toilet-paper',
    'exposure control': 'tabler:exposure',
    'enlarger': 'tabler:building-lighthouse',
    'scanner': 'tabler:scan',
    'preview and crop': 'tabler:crop',
    'crop and upscale': 'tabler:crop',
    'output': 'tabler:arrow-big-down-lines',
    'grain': 'tabler:grain',
    'halation': 'tabler:time-duration-0',
    'couplers': 'tabler:chart-sankey',
    'glare': 'tabler:background',
    'preflash': 'tabler:sparkles',
    'spectral upsampling': 'tabler:prism-light',
    'tune': 'tabler:stroke-curved',
    'experimental': 'tabler:flask',
    'gui parameters': 'tabler:adjustments-horizontal',
    'display': 'tabler:skew-x',
    'napari layers': 'tabler:stack-2',
}


def section_header_icon_name(title: str) -> str | None:
    return _SECTION_HEADER_ICONS.get(title.strip().lower())


@lru_cache(maxsize=None)
def section_header_icon(title: str, size: int = HEADER_ICON_SIZE) -> QtGui.QIcon:
    icon_name = section_header_icon_name(title)
    if icon_name is None or pyconify is None:
        return QtGui.QIcon()

    try:
        path = pyconify.svg_path(icon_name, color=ACCENT_COLOR_TEXT, width=size, height=size)
    except (OSError, TypeError, ValueError):
        return QtGui.QIcon()

    return QtGui.QIcon(str(path))