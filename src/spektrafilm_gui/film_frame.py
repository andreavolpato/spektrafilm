"""Film-frame compositing with the fixed 135 border template package.

The border artwork (sprocket holes, edge printing, colours, positions) is never
drawn here. It is loaded unchanged from ``assets/film_frames`` and laid over the
photo. The only optional modification is the sample frame number, which can be
hidden or replaced on the user's request.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FitMode = Literal['contain', 'cover']
FrameOrientation = Literal['landscape', 'portrait']

ASSET_DIR = Path(__file__).resolve().parent / 'assets' / 'film_frames'
MANIFEST_NAME = 'template_manifest.json'
DEFAULT_OUTPUT_LONG_EDGE = 3800

# Colour seen through the transparent perforations (matches the package previews).
_PERFORATION_BACKGROUND = np.array([233, 234, 228], dtype=np.float32) / 255.0
_DATE_INK = np.array([1.0, 0.33, 0.18], dtype=np.float32)

# Sample frame-number positions in landscape template pixels, taken from the
# template SVG text elements: (anchor, baseline x, baseline y, font size,
# erase rectangle x0, y0, x1, y1). The arrow next to "12A" is never touched.
_NUMBER_SLOTS = {
    'top_left': ('l', 180, 139, 100, (165, 55, 345, 146)),
    'top_right': ('r', 3570, 143, 99, (3390, 55, 3600, 146)),
    'bottom_left': ('l', 170, 3455, 100, (155, 3365, 335, 3475)),
    'bottom_right': ('r', 3565, 3455, 88, (3365, 3365, 3600, 3475)),
}
_LETTER_SPACING = 2.5

_STOCK_KEYWORDS = (
    ('portra_400', '01_Kodak_Portra_400'),
    ('gold_200', '02_Kodak_Gold_200'),
    ('ektar_100', '03_Kodak_Ektar_100'),
    ('colorplus', '04_Kodak_ColorPlus_200'),
    ('superia', '05_Fuji_Superia_XTRA_400'),
    ('xtra', '05_Fuji_Superia_XTRA_400'),
)
_GENERIC_TEMPLATE_ID = '06_通用无品牌_135'


@dataclass(frozen=True)
class FilmFrameTemplate:
    template_id: str
    display_name: str
    ui_name: str
    edge_label: str
    default_frame_number: str
    number_slots: tuple[str, ...]
    source: str

    @property
    def has_frame_number(self) -> bool:
        return bool(self.number_slots)


@dataclass(frozen=True)
class ImageAspect:
    width: int
    height: int
    ratio: float
    orientation: FrameOrientation


@dataclass(frozen=True)
class FilmFrameOptions:
    fit_mode: FitMode = 'contain'
    pan_x: float = 0.5
    pan_y: float = 0.5
    show_date: bool = False
    date_position: str = 'bottom_right'
    show_frame_number: bool = True
    output_long_edge: int = DEFAULT_OUTPUT_LONG_EDGE


@dataclass(frozen=True)
class _Geometry:
    canvas: tuple[int, int]
    window: tuple[int, int, int, int]


@lru_cache(maxsize=1)
def _manifest() -> dict:
    with (ASSET_DIR / MANIFEST_NAME).open(encoding='utf-8') as handle:
        return json.load(handle)


def _geometry(orientation: FrameOrientation) -> _Geometry:
    manifest = _manifest()
    return _Geometry(
        canvas=tuple(manifest[f'canvas_{orientation}_px']),
        window=tuple(manifest[f'photo_{orientation}_rect_px']),
    )


def _build_templates() -> tuple[FilmFrameTemplate, ...]:
    templates = []
    for variant in _manifest()['variants']:
        slots = tuple(slot for slot in _NUMBER_SLOTS if variant.get(slot))
        number = next((variant[slot] for slot in slots if slot != 'bottom_right'), '')
        templates.append(
            FilmFrameTemplate(
                template_id=variant['id'],
                display_name=variant['name'],
                ui_name=variant['ui'],
                edge_label=variant['label'],
                default_frame_number=number,
                number_slots=slots,
                source=variant['source'],
            )
        )
    return tuple(templates)


FILM_FRAME_TEMPLATES = _build_templates()
_TEMPLATES_BY_ID = {template.template_id: template for template in FILM_FRAME_TEMPLATES}


def get_film_frame_template(template_or_stock_id: str) -> FilmFrameTemplate:
    """Return a template by id, or the best matching one for a film stock id."""
    template = _TEMPLATES_BY_ID.get(template_or_stock_id)
    if template is not None:
        return template
    lowered = template_or_stock_id.lower()
    for keyword, template_id in _STOCK_KEYWORDS:
        if keyword in lowered:
            return _TEMPLATES_BY_ID[template_id]
    return _TEMPLATES_BY_ID[_GENERIC_TEMPLATE_ID]


def detect_image_aspect(image: np.ndarray) -> ImageAspect:
    if image.ndim < 2:
        raise ValueError('image must have height and width dimensions')
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError('image dimensions must be non-zero')
    return ImageAspect(
        width=width,
        height=height,
        ratio=width / height,
        orientation='landscape' if width >= height else 'portrait',
    )


def detect_frame_orientation(image: np.ndarray) -> FrameOrientation:
    return detect_image_aspect(image).orientation


def output_size(orientation: FrameOrientation, long_edge: int) -> tuple[int, int]:
    """Return (width, height) of the composite for a given long-edge length."""
    canvas_width, canvas_height = _geometry(orientation).canvas
    scale = long_edge / max(canvas_width, canvas_height)
    return max(1, round(canvas_width * scale)), max(1, round(canvas_height * scale))


def resize_rgb_image(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError('resize dimensions must be positive')
    channels = [
        np.asarray(
            Image.fromarray(np.asarray(image[..., index], dtype=np.float32), mode='F').resize(
                (width, height),
                Image.Resampling.LANCZOS,
            ),
            dtype=np.float32,
        )
        for index in range(3)
    ]
    return np.stack(channels, axis=2)


def _font(size: int) -> ImageFont.ImageFont:
    candidates = ['DejaVuSans-Bold.ttf']
    try:
        import matplotlib

        candidates.append(str(Path(matplotlib.get_data_path()) / 'fonts' / 'ttf' / 'DejaVuSans-Bold.ttf'))
    except ImportError:
        pass
    candidates += [
        '/System/Library/Fonts/Supplemental/Arial Bold.ttf',
        'arialbd.ttf',
        'Arial Bold.ttf',
    ]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size=max(1, size))
        except OSError:
            continue
    return ImageFont.load_default(size=max(1, size))


def _draw_spaced_text(draw: ImageDraw.ImageDraw, text: str, anchor: str, x: float, y: float, size: int, fill) -> None:
    font = _font(size)
    advances = [font.getlength(char) + _LETTER_SPACING for char in text]
    total = sum(advances) - _LETTER_SPACING
    cursor = x if anchor == 'l' else x - total
    for char, advance in zip(text, advances):
        draw.text((cursor, y), char, font=font, fill=fill, anchor='ls')
        cursor += advance


def _slot_text(slot: str, frame_number: str) -> str:
    return f'{frame_number}A' if slot == 'bottom_right' else frame_number


def _load_png(template_id: str, orientation: FrameOrientation) -> np.ndarray:
    suffix = '横向' if orientation == 'landscape' else '竖向'
    with Image.open(ASSET_DIR / f'{template_id}_{suffix}.png') as handle:
        return np.array(handle.convert('RGBA'), dtype=np.uint8)


@lru_cache(maxsize=None)
def _native_template(template_id: str, orientation: FrameOrientation, frame_number: str | None) -> np.ndarray:
    """Shipped template pixels; only the sample frame numbers are replaced on request."""
    template = _TEMPLATES_BY_ID[template_id]
    shipped = _load_png(template_id, orientation)
    if frame_number is None or not template.has_frame_number:
        return shipped

    pixels = _load_png(template_id, 'landscape')
    edited = Image.fromarray(pixels.copy(), mode='RGBA')
    draw = ImageDraw.Draw(edited)
    changed = np.zeros(pixels.shape[:2], dtype=bool)
    for slot in template.number_slots:
        anchor, x, y, size, (x0, y0, x1, y1) = _NUMBER_SLOTS[slot]
        region = pixels[y0:y1, x0:x1].astype(int)
        brightest = np.unravel_index(np.argmax(region[..., :3].sum(axis=2)), region.shape[:2])
        ink = tuple(int(v) for v in pixels[y0 + brightest[0], x0 + brightest[1], :3]) + (255,)
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), fill=tuple(int(v) for v in pixels[y0, x0]))
        changed[y0:y1, x0:x1] = True
        if frame_number:
            _draw_spaced_text(draw, _slot_text(slot, frame_number), anchor, x, y, size, ink)

    edited_pixels = np.asarray(edited, dtype=np.uint8)
    if orientation == 'landscape':
        return edited_pixels
    rotated = np.rot90(edited_pixels, k=3)
    rotated_mask = np.rot90(changed, k=3)[..., None]
    return np.where(rotated_mask, rotated, shipped).astype(np.uint8)


@lru_cache(maxsize=32)
def _scaled_template(
    template_id: str,
    orientation: FrameOrientation,
    frame_number: str | None,
    size: tuple[int, int],
) -> np.ndarray:
    native = _native_template(template_id, orientation, frame_number)
    rgba = Image.fromarray(np.ascontiguousarray(native), mode='RGBA')
    if rgba.size != size:
        rgba = rgba.resize(size, Image.Resampling.LANCZOS)
    return np.asarray(rgba, dtype=np.float32) / 255.0


def _fit_photo(
    source: np.ndarray,
    gate_width: int,
    gate_height: int,
    fit_mode: FitMode,
    pan_x: float,
    pan_y: float,
    fill: np.ndarray,
) -> np.ndarray:
    source_height, source_width = source.shape[:2]
    if fit_mode == 'cover':
        scale = max(gate_width / source_width, gate_height / source_height)
        scaled_width = max(gate_width, round(source_width * scale))
        scaled_height = max(gate_height, round(source_height * scale))
        resized = resize_rgb_image(source, (scaled_width, scaled_height))
        left = round((scaled_width - gate_width) * min(max(pan_x, 0.0), 1.0))
        top = round((scaled_height - gate_height) * min(max(pan_y, 0.0), 1.0))
        return resized[top:top + gate_height, left:left + gate_width]
    if fit_mode != 'contain':
        raise ValueError(f'Unsupported photo fit mode: {fit_mode}')

    scale = min(gate_width / source_width, gate_height / source_height)
    resized_width = min(gate_width, max(1, round(source_width * scale)))
    resized_height = min(gate_height, max(1, round(source_height * scale)))
    resized = resize_rgb_image(source, (resized_width, resized_height))
    gate = np.broadcast_to(fill, (gate_height, gate_width, 3)).copy()
    left = (gate_width - resized_width) // 2
    top = (gate_height - resized_height) // 2
    gate[top:top + resized_height, left:left + resized_width] = resized
    return gate


def _draw_date(window: np.ndarray, text: str, position: str) -> None:
    height, width = window.shape[:2]
    margin = max(4, round(min(width, height) * 0.04))
    font_size = max(8, round(min(width, height) * 0.06))
    anchors = {
        'bottom_right': ((width - margin, height - margin), 'rs'),
        'bottom_left': ((margin, height - margin), 'ls'),
        'top_right': ((width - margin, margin), 'ra'),
        'top_left': ((margin, margin), 'la'),
    }
    if position not in anchors:
        raise ValueError(f'Unsupported date position: {position}')
    xy, anchor = anchors[position]
    mask = Image.new('L', (width, height), 0)
    ImageDraw.Draw(mask).text(xy, text, fill=255, font=_font(font_size), anchor=anchor)
    alpha = np.asarray(mask, dtype=np.float32)[..., None] / 255.0
    window[:] = window * (1.0 - alpha) + _DATE_INK * alpha


def compose_film_frame(
    source: np.ndarray,
    *,
    template_id: str,
    date_label: str = '',
    frame_number: str | None = None,
    options: FilmFrameOptions | None = None,
) -> np.ndarray:
    """Composite ``source`` into the fixed template and return float32 RGB.

    Layers, bottom to top: perforation background, photo window fill, fitted
    photo (+ optional date stamp), then the unchanged template PNG on top.
    The photo is only ever placed inside the manifest's photo window.
    """
    options = options or FilmFrameOptions()
    source = np.asarray(source, dtype=np.float32)
    if source.ndim != 3 or source.shape[2] < 3:
        raise ValueError('source must be an RGB image')
    source = source[..., :3]
    template = get_film_frame_template(template_id)
    orientation = detect_frame_orientation(source)
    geometry = _geometry(orientation)
    canvas_width, canvas_height = output_size(orientation, options.output_long_edge)
    scale_x = canvas_width / geometry.canvas[0]
    scale_y = canvas_height / geometry.canvas[1]

    wx, wy, ww, wh = geometry.window
    x0, y0 = round(wx * scale_x), round(wy * scale_y)
    x1, y1 = round((wx + ww) * scale_x), round((wy + wh) * scale_y)

    custom_number = None
    if template.has_frame_number:
        shown = (frame_number if frame_number is not None else template.default_frame_number).strip()
        if not options.show_frame_number:
            custom_number = ''
        elif shown != template.default_frame_number:
            custom_number = shown
    overlay = _scaled_template(template.template_id, orientation, custom_number, (canvas_width, canvas_height))

    base = overlay[5, 5, :3]
    canvas = np.broadcast_to(_PERFORATION_BACKGROUND, (canvas_height, canvas_width, 3)).copy()
    window = _fit_photo(
        source, x1 - x0, y1 - y0, options.fit_mode, options.pan_x, options.pan_y, base,
    )
    if options.show_date and date_label.strip():
        _draw_date(window, date_label.strip(), options.date_position)
    canvas[y0:y1, x0:x1] = window

    alpha = overlay[..., 3:4]
    return np.clip(canvas * (1.0 - alpha) + overlay[..., :3] * alpha, 0.0, None).astype(np.float32)
