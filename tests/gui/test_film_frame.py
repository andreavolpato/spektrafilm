from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from spektrafilm_gui import film_frame
from spektrafilm_gui.film_frame import (
    ASSET_DIR,
    FILM_FRAME_TEMPLATES,
    FilmFrameOptions,
    compose_film_frame,
    detect_frame_orientation,
    get_film_frame_template,
    output_size,
)

TEMPLATE_IDS = [template.template_id for template in FILM_FRAME_TEMPLATES]


def _photo(height: int, width: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (0.2 + 0.6 * rng.random((height, width, 3))).astype(np.float32)


def _window(orientation: str, scale: float) -> tuple[int, int, int, int]:
    x, y, w, h = film_frame._geometry(orientation).window
    return round(x * scale), round(y * scale), round((x + w) * scale), round((y + h) * scale)


def _template_png(template_id: str, orientation: str) -> Path:
    return ASSET_DIR / f'{template_id}_{orientation}.png'


def test_eleven_templates_with_both_orientations_ship_with_data() -> None:
    assert len(FILM_FRAME_TEMPLATES) == 11
    for template_id in TEMPLATE_IDS:
        for orientation in ('landscape', 'portrait'):
            for extension in ('png', 'svg'):
                assert (ASSET_DIR / f'{template_id}_{orientation}.{extension}').is_file()
    for name in ('template_manifest.json', 'film_stocks.json', 'film135_geometry.json', 'edge_encoding_schema.json'):
        assert (ASSET_DIR / name).is_file()


@pytest.mark.parametrize('template_id', TEMPLATE_IDS)
def test_pack_png_has_transparent_window_and_opaque_rebate(template_id: str) -> None:
    for orientation, centre in (('landscape', (1900, 1750)), ('portrait', (1750, 1900))):
        with Image.open(_template_png(template_id, orientation)) as handle:
            rgba = handle.convert('RGBA')
        assert rgba.getpixel(centre)[3] == 0
        assert rgba.getpixel((0, 0))[3] == 255


def test_unverified_dx_is_never_drawn_and_marks_come_from_data() -> None:
    stocks = {stock['id']: stock for stock in film_frame._read_json(film_frame.STOCKS_NAME)['stocks']}
    for template in FILM_FRAME_TEMPLATES:
        dx = stocks[template.template_id]['dx_135']
        expected = bool(dx and dx['confidence'] == 'read_from_strip' and dx['render_by_default'])
        assert template.dx_drawn is expected
        assert template.edge_label == stocks[template.template_id]['marking']['edge_text_135']
    assert get_film_frame_template('kodak_portra_400').dx_drawn is False
    assert get_film_frame_template('kodak_gold_200').dx_drawn is True
    assert get_film_frame_template('generic_135').edge_label == ''
    assert get_film_frame_template('generic_135').has_frame_number is False


def test_manifest_geometry_matches_png_sizes() -> None:
    assert film_frame._geometry('landscape').canvas == (3800, 3500)
    assert film_frame._geometry('landscape').window == (100, 550, 3600, 2400)
    assert film_frame._geometry('portrait').canvas == (3500, 3800)
    assert film_frame._geometry('portrait').window == (550, 100, 2400, 3600)
    index = {(e['stock_id'], e['direction']): e['photo_window'] for e in film_frame._manifest()['template_index']}
    assert index[('kodak_gold_200', 'landscape')] == {'x': 100, 'y': 550, 'width': 3600, 'height': 2400}
    assert index[('kodak_gold_200', 'portrait')] == {'x': 550, 'y': 100, 'width': 2400, 'height': 3600}
    for template_id in TEMPLATE_IDS:
        with Image.open(_template_png(template_id, 'landscape')) as landscape:
            assert landscape.size == (3800, 3500)
        with Image.open(_template_png(template_id, 'portrait')) as portrait:
            assert portrait.size == (3500, 3800)


def test_orientation_follows_photo_and_square_uses_landscape() -> None:
    assert detect_frame_orientation(_photo(24, 36)) == 'landscape'
    assert detect_frame_orientation(_photo(36, 24)) == 'portrait'
    assert detect_frame_orientation(_photo(30, 30)) == 'landscape'


def test_film_stock_ids_resolve_to_templates() -> None:
    for stock_id in ('kodak_portra_400', 'kodak_gold_200', 'kodak_ektar_100', 'cinestill_800t'):
        assert get_film_frame_template(stock_id).template_id == stock_id
    assert get_film_frame_template('fujifilm_superia_xtra_400').template_id == 'fujifilm_xtra_400'
    assert get_film_frame_template('kodak_5207').template_id == 'generic_135'


@pytest.mark.parametrize('template_id', TEMPLATE_IDS)
@pytest.mark.parametrize('shape', [(240, 360), (360, 240)])
def test_frame_is_identical_for_any_photo_and_native_png_is_used(template_id: str, shape: tuple[int, int]) -> None:
    options = FilmFrameOptions(output_long_edge=3800, fit_mode='cover')
    first = compose_film_frame(_photo(*shape, seed=1), template_id=template_id, options=options)
    second = compose_film_frame(_photo(*shape, seed=2), template_id=template_id, options=options)

    orientation = 'landscape' if shape[1] > shape[0] else 'portrait'
    assert first.shape[:2] == film_frame._geometry(orientation).canvas[::-1]
    x0, y0, x1, y1 = _window(orientation, 1.0)
    outside = np.ones(first.shape[:2], dtype=bool)
    outside[y0:y1, x0:x1] = False
    np.testing.assert_array_equal(first[outside], second[outside])
    assert not np.array_equal(first[y0:y1, x0:x1], second[y0:y1, x0:x1])

    with Image.open(_template_png(template_id, orientation)) as handle:
        shipped = np.asarray(handle.convert('RGBA'), dtype=np.float32) / 255.0
    opaque = shipped[..., 3] == 1.0
    np.testing.assert_allclose(first[opaque], shipped[..., :3][opaque], atol=1e-6)


def test_photo_never_reaches_perforations() -> None:
    photo = np.ones((240, 360, 3), dtype=np.float32) * np.array([1.0, 0.0, 0.0], dtype=np.float32)
    result = compose_film_frame(photo, template_id='kodak_portra_400', options=FilmFrameOptions(output_long_edge=3800))
    assert result[340, 237].min() > 0.9  # white perforation, not red photo
    assert result[1500, 1900, 0] > 0.9 and result[1500, 1900, 1] < 0.1


def test_contain_does_not_stretch_and_cover_fills_window() -> None:
    wide = np.zeros((100, 400, 3), dtype=np.float32)
    wide[...] = [0.0, 1.0, 0.0]
    contain = compose_film_frame(wide, template_id='generic_135', options=FilmFrameOptions(output_long_edge=3800))
    x0, y0, x1, y1 = _window('landscape', 1.0)
    green = (contain[y0:y1, x0:x1, 1] > 0.9) & (contain[y0:y1, x0:x1, 0] < 0.1)
    rows = np.where(green.any(axis=1))[0]
    cols = np.where(green.any(axis=0))[0]
    assert (cols.max() - cols.min() + 1) / (rows.max() - rows.min() + 1) == pytest.approx(4.0, rel=0.02)

    cover = compose_film_frame(
        wide, template_id='generic_135', options=FilmFrameOptions(output_long_edge=3800, fit_mode='cover'),
    )
    assert (cover[y0 + 40:y1 - 40, x0 + 40:x1 - 40, 1] > 0.9).all()


def test_cover_pan_moves_crop() -> None:
    photo = np.zeros((400, 200, 3), dtype=np.float32)
    photo[:200] = [1.0, 0.0, 0.0]
    photo[200:] = [0.0, 0.0, 1.0]
    scale = 950 / 3800
    x0, y0, x1, y1 = _window('portrait', scale)
    # a 2:1 portrait photo cropped into a 2:3 window: pan_y shifts the visible band
    top = compose_film_frame(
        photo, template_id='generic_135',
        options=FilmFrameOptions(output_long_edge=950, fit_mode='cover', pan_y=0.0),
    )
    bottom = compose_film_frame(
        photo, template_id='generic_135',
        options=FilmFrameOptions(output_long_edge=950, fit_mode='cover', pan_y=1.0),
    )
    centre = ((y0 + y1) // 2, (x0 + x1) // 2)
    assert top[centre][0] > 0.8 or top[centre][2] > 0.8
    assert not np.array_equal(top[y0:y1, x0:x1], bottom[y0:y1, x0:x1])


def test_output_scales_whole_frame_proportionally() -> None:
    assert output_size('landscape', 1900) == (1900, 1750)
    assert output_size('portrait', 1900) == (1750, 1900)
    small = compose_film_frame(_photo(240, 360), template_id='kodak_gold_200', options=FilmFrameOptions(output_long_edge=950))
    assert small.shape[:2] == (875, 950)


def test_frame_number_hide_and_replace_only_touch_number_slots() -> None:
    options = FilmFrameOptions(output_long_edge=3800)
    base = compose_film_frame(_photo(240, 360), template_id='kodak_portra_400', options=options)
    hidden = compose_film_frame(
        _photo(240, 360), template_id='kodak_portra_400',
        options=FilmFrameOptions(output_long_edge=3800, show_frame_number=False),
    )
    changed = np.any(base != hidden, axis=2)
    assert changed.any()
    slots = [slot.erase_rect for slot in get_film_frame_template('kodak_portra_400').number_slots]
    ys, xs = np.where(changed)
    for y, x in zip(ys[::50], xs[::50]):
        assert any(x0 <= x < x1 and y0 <= y < y1 for x0, y0, x1, y1 in slots)
    # edge label and the arrow next to 12A are untouched
    assert np.array_equal(base[40:150, 150:1500], hidden[40:150, 150:1500])
    assert np.array_equal(base[3380:3420, 2650:2910], hidden[3380:3420, 2650:2910])

    replaced = compose_film_frame(
        _photo(240, 360), template_id='kodak_portra_400', frame_number='7', options=options,
    )
    assert np.any(replaced != base)
    assert np.any(replaced != hidden)


def test_frame_number_edit_in_portrait_never_touches_photo_window() -> None:
    base = compose_film_frame(
        _photo(360, 240), template_id='fujifilm_xtra_400', options=FilmFrameOptions(output_long_edge=3800),
    )
    hidden = compose_film_frame(
        _photo(360, 240), template_id='fujifilm_xtra_400',
        options=FilmFrameOptions(output_long_edge=3800, show_frame_number=False),
    )
    changed = np.any(base != hidden, axis=2)
    assert changed.any()
    x0, y0, x1, y1 = _window('portrait', 1.0)
    assert not changed[y0:y1, x0:x1].any()


def test_generic_template_has_no_frame_number() -> None:
    assert get_film_frame_template('generic_135').has_frame_number is False
    base = compose_film_frame(_photo(240, 360), template_id='generic_135')
    hidden = compose_film_frame(
        _photo(240, 360), template_id='generic_135', options=FilmFrameOptions(show_frame_number=False),
    )
    np.testing.assert_array_equal(base, hidden)


def test_date_stamp_is_off_by_default_and_stays_inside_photo_window() -> None:
    photo = _photo(240, 360)
    without = compose_film_frame(
        photo, template_id='kodak_portra_400', date_label='25 11 2', options=FilmFrameOptions(output_long_edge=1900),
    )
    with_date = compose_film_frame(
        photo, template_id='kodak_portra_400', date_label='25 11 2',
        options=FilmFrameOptions(output_long_edge=1900, show_date=True),
    )
    changed = np.any(without != with_date, axis=2)
    assert changed.any()
    x0, y0, x1, y1 = _window('landscape', 0.5)
    assert not changed[:y0].any() and not changed[y1:].any()
    assert not changed[:, :x0].any() and not changed[:, x1:].any()


def test_dx_barcode_only_redrawn_for_verified_stocks_and_hidden_with_number() -> None:
    options = FilmFrameOptions(output_long_edge=3800)
    for stock_id, drawn in (('kodak_gold_200', True), ('kodak_portra_400', False)):
        base = compose_film_frame(_photo(240, 360), template_id=stock_id, options=options)
        edited = compose_film_frame(_photo(240, 360), template_id=stock_id, frame_number='27', options=options)
        assert bool(np.any(base[3300:3390, 270:1540] != edited[3300:3390, 270:1540])) is drawn


def test_letterbox_colour_fills_contain_bars() -> None:
    photo = np.ones((240, 240, 3), dtype=np.float32)
    result = compose_film_frame(
        photo, template_id='kodak_gold_200',
        options=FilmFrameOptions(output_long_edge=3800, letterbox_color=(255, 255, 255)),
    )
    assert result[1750, 150].min() > 0.99


def test_realistic_finish_keeps_photo_and_layout() -> None:
    photo = _photo(240, 360)
    plain = compose_film_frame(photo, template_id='kodak_gold_200', options=FilmFrameOptions(output_long_edge=1900))
    real = compose_film_frame(photo, template_id='kodak_gold_200', options=FilmFrameOptions(output_long_edge=1900, realistic=True))
    assert plain.shape == real.shape
    assert np.any(plain != real)
    np.testing.assert_allclose(plain[875, 950], real[875, 950], atol=0.02)
