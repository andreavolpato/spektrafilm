from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FitMode = Literal['contain', 'cover']
FrameOrientation = Literal['landscape', 'portrait']

_FILM_BLACK = np.array([0.008, 0.01, 0.012], dtype=np.float32)
_PERFORATION = np.array([0.88, 0.88, 0.84], dtype=np.float32)
_EDGE_INK = np.array([1.0, 0.39, 0.12], dtype=np.float32)
_USER_NUMBER_INK = np.array([0.88, 0.9, 0.86], dtype=np.float32)
_DATE_INK = np.array([1.0, 0.33, 0.18], dtype=np.float32)


@dataclass(frozen=True)
class FilmFrameTemplate:
    template_id: str
    display_name: str
    stock_ids: tuple[str, ...]
    reference_status: str
    reference_description: str
    top_label: str | None = None
    bottom_edge_code: str | None = None
    has_frame_number: bool = False


FILM_FRAME_TEMPLATES = (
    FilmFrameTemplate(
        template_id='kodak_portra_400',
        display_name='KODAK PORTRA 400',
        stock_ids=('kodak_portra_400',),
        reference_status='部分核实',
        reference_description='用户提供的 Portra 400 底片扫描参考；仅复现图中可辨认的边缘标记。',
        top_label='KODAK PORTRA 400',
        bottom_edge_code='73A →',
        has_frame_number=True,
    ),
    FilmFrameTemplate(
        template_id='kodak_gold_200',
        display_name='KODAK GOLD 200',
        stock_ids=('kodak_gold_200',),
        reference_status='待核实',
        reference_description='尚无可靠的该型号底片边缘参考；仅显示通用物理齿孔。',
    ),
    FilmFrameTemplate(
        template_id='kodak_colorplus_200',
        display_name='KODAK COLORPLUS 200',
        stock_ids=('kodak_colorplus_200',),
        reference_status='待核实',
        reference_description='尚无可靠的该型号底片边缘参考；仅显示通用物理齿孔。',
    ),
    FilmFrameTemplate(
        template_id='kodak_ultramax_400',
        display_name='KODAK ULTRAMAX 400',
        stock_ids=('kodak_ultramax_400',),
        reference_status='待核实',
        reference_description='尚无可靠的该型号底片边缘参考；仅显示通用物理齿孔。',
    ),
    FilmFrameTemplate(
        template_id='fujicolor_200',
        display_name='FUJICOLOR 200',
        stock_ids=('fujicolor_200',),
        reference_status='待核实',
        reference_description='尚无可靠的该型号底片边缘参考；仅显示通用物理齿孔。',
    ),
    FilmFrameTemplate(
        template_id='fujicolor_400',
        display_name='FUJICOLOR 400',
        stock_ids=('fujicolor_400',),
        reference_status='待核实',
        reference_description='尚无可靠的该型号底片边缘参考；仅显示通用物理齿孔。',
    ),
)

_TEMPLATES_BY_ID = {template.template_id: template for template in FILM_FRAME_TEMPLATES}
_TEMPLATES_BY_STOCK = {
    stock_id: template
    for template in FILM_FRAME_TEMPLATES
    for stock_id in template.stock_ids
}


@dataclass(frozen=True)
class ImageAspect:
    width: int
    height: int
    ratio: float
    orientation: FrameOrientation


@dataclass(frozen=True)
class FilmFrameOptions:
    border_width: float = 0.15
    fit_mode: FitMode = 'contain'
    show_date: bool = False
    date_position: str = 'bottom_right'
    show_frame_number: bool = True


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


def get_film_frame_template(template_or_stock_id: str) -> FilmFrameTemplate:
    template = _TEMPLATES_BY_ID.get(template_or_stock_id) or _TEMPLATES_BY_STOCK.get(template_or_stock_id)
    if template is not None:
        return template
    return FilmFrameTemplate(
        template_id=template_or_stock_id,
        display_name=template_or_stock_id,
        stock_ids=(template_or_stock_id,),
        reference_status='待核实',
        reference_description='尚无可靠的该型号底片边缘参考；仅显示通用物理齿孔。',
    )


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


def _fit_photo(source: np.ndarray, gate_width: int, gate_height: int, fit_mode: FitMode) -> np.ndarray:
    source_height, source_width = source.shape[:2]
    if fit_mode == 'cover':
        scale = max(gate_width / source_width, gate_height / source_height)
        scaled_width = max(gate_width, round(source_width * scale))
        scaled_height = max(gate_height, round(source_height * scale))
        resized = resize_rgb_image(source, (scaled_width, scaled_height))
        left = (scaled_width - gate_width) // 2
        top = (scaled_height - gate_height) // 2
        return resized[top:top + gate_height, left:left + gate_width]
    if fit_mode != 'contain':
        raise ValueError(f'Unsupported photo fit mode: {fit_mode}')

    scale = min(gate_width / source_width, gate_height / source_height)
    resized_width = max(1, round(source_width * scale))
    resized_height = max(1, round(source_height * scale))
    resized = resize_rgb_image(source, (resized_width, resized_height))
    gate = np.broadcast_to(_FILM_BLACK, (gate_height, gate_width, 3)).copy()
    left = (gate_width - resized_width) // 2
    top = (gate_height - resized_height) // 2
    gate[top:top + resized_height, left:left + resized_width] = resized
    return gate


def _font(size: int) -> ImageFont.ImageFont:
    for name in ('DejaVuSans-Bold.ttf', 'Arial.ttf'):
        try:
            return ImageFont.truetype(name, size=max(1, size))
        except OSError:
            pass
    return ImageFont.load_default()


def _draw_text(
    image: np.ndarray,
    text: str,
    xy: tuple[int, int],
    font_size: int,
    color: np.ndarray,
    *,
    anchor: str = 'la',
) -> None:
    if not text:
        return
    height, width = image.shape[:2]
    mask = Image.new('L', (width, height), 0)
    ImageDraw.Draw(mask).text(xy, text, fill=255, font=_font(font_size), anchor=anchor)
    alpha = np.asarray(mask, dtype=np.float32)[..., None] / 255.0
    image[:] = image * (1.0 - alpha) + color * alpha


def _draw_sprocket_row(
    image: np.ndarray,
    y_center: int,
    hole_width: int,
    hole_height: int,
) -> None:
    width, height = image.shape[1], image.shape[0]
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    radius = max(1, hole_height // 3)
    for index in range(8):
        x_center = round((index + 0.5) * width / 8)
        draw.rounded_rectangle(
            (
                x_center - hole_width // 2,
                y_center - hole_height // 2,
                x_center + hole_width // 2,
                y_center + hole_height // 2,
            ),
            radius=radius,
            fill=255,
        )
    alpha = np.asarray(mask, dtype=np.float32)[..., None] / 255.0
    image[:] = image * (1.0 - alpha) + _PERFORATION * alpha


def _draw_date(image: np.ndarray, text: str, position: str) -> None:
    height, width = image.shape[:2]
    margin = max(4, round(min(width, height) * 0.025))
    font_size = max(8, round(min(width, height) * 0.055))
    anchors = {
        'bottom_right': (width - margin, height - margin),
        'bottom_left': (margin, height - margin),
        'top_right': (width - margin, margin),
        'top_left': (margin, margin),
    }
    if position not in anchors:
        raise ValueError(f'Unsupported date position: {position}')
    anchor = 'ra' if position.endswith('right') else 'la'
    if position.startswith('bottom'):
        anchor = 'rd' if position.endswith('right') else 'ld'
    _draw_text(image, text, anchors[position], font_size, _DATE_INK, anchor=anchor)


def compose_film_frame(
    source: np.ndarray,
    *,
    template_id: str = 'kodak_portra_400',
    date_label: str = '',
    frame_number: str = '74',
    options: FilmFrameOptions | None = None,
) -> np.ndarray:
    source = np.asarray(source, dtype=np.float32)
    if (
        source.ndim != 3
        or source.shape[2] != 3
        or source.shape[0] == 0
        or source.shape[1] == 0
        or not np.isfinite(source).all()
    ):
        raise ValueError('source must be a non-empty finite RGB image')

    options = options or FilmFrameOptions()
    if not 0.05 <= options.border_width <= 0.35:
        raise ValueError('border_width must be between 0.05 and 0.35')
    template = get_film_frame_template(template_id)
    source_aspect = detect_image_aspect(source)
    photo = source
    if source_aspect.orientation == 'portrait':
        photo = np.rot90(source, k=1)

    gate_width = photo.shape[1]
    gate_height = round(gate_width / 1.5)
    band_height = max(1, round(gate_height * options.border_width / (1 - 2 * options.border_width)))
    canvas = np.broadcast_to(
        _FILM_BLACK,
        (gate_height + 2 * band_height, gate_width, 3),
    ).copy()
    fitted_photo = _fit_photo(photo, gate_width, gate_height, options.fit_mode)
    canvas[band_height:band_height + gate_height] = fitted_photo

    hole_width = max(3, round(gate_width * 0.045))
    hole_height = max(2, round(band_height * 0.15))
    _draw_sprocket_row(canvas, round(band_height * 0.70), hole_width, hole_height)
    _draw_sprocket_row(canvas, band_height + gate_height + round(band_height * 0.30), hole_width, hole_height)

    mark_size = max(6, round(band_height * 0.19))
    margin = max(2, round(gate_width * 0.025))
    if template.top_label:
        _draw_text(canvas, template.top_label, (margin, round(band_height * 0.10)), mark_size, _EDGE_INK)
    if template.has_frame_number and options.show_frame_number and frame_number:
        _draw_text(
            canvas,
            frame_number,
            (gate_width - margin, round(band_height * 0.10)),
            mark_size,
            _USER_NUMBER_INK,
            anchor='ra',
        )
    if template.bottom_edge_code:
        bottom_y = band_height + gate_height + round(band_height * 0.58)
        _draw_text(canvas, template.bottom_edge_code, (margin, bottom_y), mark_size, _EDGE_INK)
    if template.has_frame_number and options.show_frame_number and frame_number:
        _draw_text(
            canvas,
            frame_number,
            (gate_width - margin, band_height + gate_height + round(band_height * 0.58)),
            mark_size,
            _USER_NUMBER_INK,
            anchor='ra',
        )

    if source_aspect.orientation == 'portrait':
        canvas = np.rot90(canvas, k=3)
        date_gate = canvas[:, band_height:band_height + gate_height]
    else:
        date_gate = canvas[band_height:band_height + gate_height]
    if options.show_date and date_label:
        _draw_date(date_gate, date_label, options.date_position)
    return np.ascontiguousarray(canvas, dtype=np.float32)
