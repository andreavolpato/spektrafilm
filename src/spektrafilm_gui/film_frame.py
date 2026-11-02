"""Film-frame compositing with the fixed 135 border template package.

The border artwork (sprocket holes, edge printing, colours, positions) is never
drawn here. It is loaded unchanged from ``assets/film_frames`` and laid over the
photo. The only optional modification is the sample frame number, which can be
hidden or replaced on the user's request.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FitMode = Literal['contain', 'cover']
FrameOrientation = Literal['landscape', 'portrait']

ASSET_DIR = Path(__file__).resolve().parent / 'assets' / 'film_frames'
MANIFEST_NAME = 'template_manifest.json'
GEOMETRY_NAME = 'film135_geometry.json'
DEFAULT_OUTPUT_LONG_EDGE = 3800

STOCKS_NAME = 'film_stocks.json'
_PERFORATION_BACKGROUND = np.array([233, 234, 228], dtype=np.float32) / 255.0
_DATE_INK = np.array([1.0, 0.33, 0.18], dtype=np.float32)

_STOCK_KEYWORDS = (
    ('superia', 'fujifilm_xtra_400'),
    ('xtra', 'fujifilm_xtra_400'),
)
_GENERIC_TEMPLATE_ID = 'generic_135'
_ERASE_PAD = 8
_ARROW_CLEARANCE = 0
_BARCODE_ROWS = ((3310, 70), (3398, 70))
_BARCODE_X = 270
_BARCODE_MODULES = 31
_BARCODE_WIDTH = {'FujiNeg': 1410}
_BARCODE_DEFAULT_WIDTH = 1270
_BARCODE_MAX_FRAME = 63

_TEXT_RE = re.compile(
    r'<text x="([\d.]+)" y="([\d.]+)" text-anchor="(start|end)"[^>]*?font-size="([\d.]+)"'
    r'[^>]*?letter-spacing="([\d.]+)"[^>]*>([^<]*)</text>'
)
_ARROW_RE = re.compile(r'<path d="M ([\d.]+) ([\d.]+) H')


@dataclass(frozen=True)
class NumberSlot:
    """A frame-number text node found in the landscape SVG (landscape pixels)."""
    suffix: str
    anchor: str
    x: float
    y: float
    size: float
    spacing: float
    erase_rect: tuple[int, int, int, int]


@dataclass(frozen=True)
class FilmFrameTemplate:
    template_id: str
    display_name: str
    ui_name: str
    edge_label: str
    default_frame_number: str
    number_slots: tuple[NumberSlot, ...]
    source: str
    ink: tuple[int, int, int]
    layout_family: str
    dx_extract: int | None
    dx_drawn: bool

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
    letterbox_color: tuple[int, int, int] | None = None
    realistic: bool = False


@dataclass(frozen=True)
class _Geometry:
    canvas: tuple[int, int]
    window: tuple[int, int, int, int]


def _read_json(name: str) -> dict:
    with (ASSET_DIR / name).open(encoding='utf-8') as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def _manifest() -> dict:
    return _read_json(MANIFEST_NAME)


@lru_cache(maxsize=1)
def _geometry_data() -> dict:
    return _read_json(GEOMETRY_NAME)['layout']


def _geometry(orientation: FrameOrientation) -> _Geometry:
    layout = _geometry_data()[f'{orientation}_px']
    window = layout['window']
    return _Geometry(
        canvas=(layout['width'], layout['height']),
        window=(window['x'], window['y'], window['width'], window['height']),
    )


def _template_path(template_id: str, orientation: FrameOrientation, suffix: str = 'png') -> Path:
    return ASSET_DIR / f'{template_id}_{orientation}.{suffix}'


def _parse_number_slots(template_id: str, example: str) -> tuple[NumberSlot, ...]:
    """Find the sample frame-number text in the landscape SVG (the data source)."""
    svg = _template_path(template_id, 'landscape', 'svg').read_text(encoding='utf-8')
    arrows = [float(match.group(1)) for match in _ARROW_RE.finditer(svg)]
    slots = []
    for match in _TEXT_RE.finditer(svg):
        x, y, anchor, size, spacing, text = match.groups()
        if text not in (example, f'{example}A'):
            continue
        x, y, size = float(x), float(y), float(size)
        width = (len(text) * 0.7 * size) + _ERASE_PAD
        x0, x1 = (x - _ERASE_PAD, x + width) if anchor == 'start' else (x - width, x + _ERASE_PAD)
        right_arrows = [a for a in arrows if a > x0]
        if anchor == 'start' and right_arrows:
            x1 = min(x1, min(right_arrows) - _ARROW_CLEARANCE)
        slots.append(NumberSlot(
            suffix=text[len(example):],
            anchor=anchor,
            x=x,
            y=y,
            size=size,
            spacing=float(spacing),
            erase_rect=(round(x0), round(y - size - _ERASE_PAD), round(x1), round(y + 0.25 * size + _ERASE_PAD)),
        ))
    return tuple(slots)


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip('#')
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def _build_templates() -> tuple[FilmFrameTemplate, ...]:
    available = {entry['stock_id'] for entry in _manifest()['template_index']}
    templates = []
    for stock in _read_json(STOCKS_NAME)['stocks']:
        if stock['id'] not in available:
            continue
        marking = stock['marking']
        dx = stock.get('dx_135')
        example = str(marking['frame_number_example'])
        slots = _parse_number_slots(stock['id'], example)
        label = marking['edge_text_135']
        dx_drawn = bool(dx and dx['confidence'] == 'read_from_strip' and dx['render_by_default'])
        confidence = marking['edge_text_confidence']
        source = (
            f"数据来源：SpektraLab 公开资料整理的参考模板（印字可信度：{confidence or '无印字'}；"
            f"证据：{stock['evidence']}）。"
        )
        templates.append(
            FilmFrameTemplate(
                template_id=stock['id'],
                display_name=stock['name'],
                ui_name=stock['name'],
                edge_label=label,
                default_frame_number=example if slots else '',
                number_slots=slots,
                source=source,
                ink=_hex_to_rgb(stock['ink_hex']),
                layout_family=stock['edge_layout_family'],
                dx_extract=dx['extract'] if dx else None,
                dx_drawn=dx_drawn,
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
    # Arial first: the template SVG text is Arial/Helvetica bold.
    candidates = [
        '/System/Library/Fonts/Supplemental/Arial Bold.ttf',
        'arialbd.ttf',
        'Arial Bold.ttf',
        'LiberationSans-Bold.ttf',
        'DejaVuSans-Bold.ttf',
    ]
    try:
        import matplotlib

        candidates.append(str(Path(matplotlib.get_data_path()) / 'fonts' / 'ttf' / 'DejaVuSans-Bold.ttf'))
    except ImportError:
        pass
    for name in candidates:
        try:
            return ImageFont.truetype(name, size=max(1, size))
        except OSError:
            continue
    return ImageFont.load_default(size=max(1, size))


def _draw_spaced_text(draw: ImageDraw.ImageDraw, text: str, slot: NumberSlot, size: float, fill) -> None:
    def layout(em: float):
        font = _font(round(em))
        advances = [font.getlength(char) + slot.spacing for char in text]
        return font, advances, sum(advances) - slot.spacing

    font, advances, total = layout(size)
    if slot.anchor == 'start':
        room = slot.erase_rect[2] - slot.x - _ERASE_PAD
        if total > room:
            font, advances, total = layout(size * room / total)
    cursor = slot.x if slot.anchor == 'start' else slot.x - total
    for char, advance in zip(text, advances):
        draw.text((cursor, slot.y), char, font=font, fill=fill, anchor='ls')
        cursor += advance


def dx_bits(extract: int, frame: int, half: bool = False) -> tuple[list[int], list[int]]:
    """Clock and data tracks (31 modules) per the pack's edge_encoding_schema."""
    part1 = (extract >> 4) & 127
    part2 = extract & 15
    bits = [0] * 23
    for k in range(7):
        bits[k + 1] = (part1 >> (6 - k)) & 1
    for k in range(4):
        bits[k + 9] = (part2 >> (3 - k)) & 1
    for k in range(6):
        bits[k + 13] = (max(0, min(_BARCODE_MAX_FRAME, frame)) >> (5 - k)) & 1
    bits[19] = int(half)
    bits[21] = sum(bits[:21]) % 2
    clock = [1 if i < 5 or i >= 28 else ((i - 5) % 2) for i in range(_BARCODE_MODULES)]
    data = [
        int(i % 2 == 0) if i < 5 else int((i - 28) % 2 == 0) if i >= 28 else bits[i - 5]
        for i in range(_BARCODE_MODULES)
    ]
    return clock, data


def _barcode_geometry(template: FilmFrameTemplate) -> tuple[float, tuple[int, int, int, int]]:
    width = _BARCODE_WIDTH.get(template.layout_family, _BARCODE_DEFAULT_WIDTH)
    module = width / _BARCODE_MODULES
    top = _BARCODE_ROWS[0][0]
    bottom = _BARCODE_ROWS[-1][0] + _BARCODE_ROWS[-1][1]
    right = _BARCODE_X + _BARCODE_MODULES * module + 1
    return module, (_BARCODE_X - 4, top - 4, round(right) + 4, bottom + 4)


def _draw_barcode(draw: ImageDraw.ImageDraw, template: FilmFrameTemplate, frame: int) -> None:
    module, _ = _barcode_geometry(template)
    clock, data = dx_bits(template.dx_extract, frame)
    for bits, (top, height) in zip((clock, data), _BARCODE_ROWS):
        for index, bit in enumerate(bits):
            if bit:
                x = _BARCODE_X + index * module
                draw.rectangle((x, top, x + module + 0.4 - 1e-3, top + height - 1), fill=template.ink + (255,))


def _load_png(template_id: str, orientation: FrameOrientation) -> np.ndarray:
    with Image.open(_template_path(template_id, orientation)) as handle:
        return np.array(handle.convert('RGBA'), dtype=np.uint8)


def _parse_frame_number(text: str) -> int | None:
    return int(text) if text.isdigit() and int(text) <= _BARCODE_MAX_FRAME else None


@lru_cache(maxsize=None)
def _native_template(template_id: str, orientation: FrameOrientation, frame_number: str | None) -> np.ndarray:
    """Shipped template pixels; the sample frame number (and DX frame bits) are
    redrawn from the stock data only when the user changes or hides the number."""
    template = _TEMPLATES_BY_ID[template_id]
    shipped = _load_png(template_id, orientation)
    if frame_number is None or not template.has_frame_number:
        return shipped

    pixels = _load_png(template_id, 'landscape')
    edited = Image.fromarray(pixels.copy(), mode='RGBA')
    draw = ImageDraw.Draw(edited)
    changed = np.zeros(pixels.shape[:2], dtype=bool)
    ink = template.ink + (255,)

    def erase(rect: tuple[int, int, int, int]) -> None:
        x0, y0, x1, y1 = rect
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), fill=tuple(int(v) for v in pixels[y0, x0]))
        changed[y0:y1, x0:x1] = True

    for slot in template.number_slots:
        erase(slot.erase_rect)
        if frame_number:
            _draw_spaced_text(draw, frame_number + slot.suffix, slot, slot.size, ink)

    if template.dx_drawn:
        _, rect = _barcode_geometry(template)
        erase(rect)
        frame = _parse_frame_number(frame_number)
        if frame is not None:
            _draw_barcode(draw, template, frame)

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

    base = (
        np.array(options.letterbox_color, dtype=np.float32) / 255.0
        if options.letterbox_color is not None
        else overlay[5, 5, :3]
    )
    canvas = np.broadcast_to(_PERFORATION_BACKGROUND, (canvas_height, canvas_width, 3)).copy()
    window = _fit_photo(
        source, x1 - x0, y1 - y0, options.fit_mode, options.pan_x, options.pan_y, base,
    )
    if options.show_date and date_label.strip():
        _draw_date(window, date_label.strip(), options.date_position)
    canvas[y0:y1, x0:x1] = window

    if options.realistic:
        return _realistic_composite(canvas, overlay)
    alpha = overlay[..., 3:4]
    return np.clip(canvas * (1.0 - alpha) + overlay[..., :3] * alpha, 0.0, None).astype(np.float32)


def _blur(plane: np.ndarray, sigma: float) -> np.ndarray:
    image = Image.fromarray(np.clip(plane * 255.0, 0, 255).astype(np.uint8), mode='L')
    return np.asarray(image.filter(ImageFilter.GaussianBlur(sigma)), dtype=np.float32) / 255.0


def _realistic_composite(canvas: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    """Scan-like finish: soft hole/window edges, hole glow into the rebate and rebate grain.

    Purely a post-effect on the fixed layout; it never moves or changes any template element.
    """
    height, width = overlay.shape[:2]
    unit = width / 3800.0
    alpha = _blur(overlay[..., 3], max(0.6, 2.2 * unit))[..., None]
    rgb = overlay[..., :3]
    holes = ((overlay[..., 3] > 0.5) & (rgb.min(axis=2) > 0.8)).astype(np.float32)
    glow = _blur(holes, 14.0 * unit)[..., None]
    rebate = np.clip(overlay[..., 3:4], 0.0, 1.0)
    rng = np.random.default_rng(135)
    grain = rng.normal(0.0, 0.012, (height, width, 1)).astype(np.float32)
    film = np.clip(rgb + glow * 0.22 * (1.0 - holes[..., None]) + grain * rebate, 0.0, 1.0)
    return np.clip(canvas * (1.0 - alpha) + film * alpha, 0.0, None).astype(np.float32)
