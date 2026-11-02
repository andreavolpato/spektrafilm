from __future__ import annotations

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


def test_six_templates_with_both_orientations_ship_with_manifest() -> None:
    assert len(FILM_FRAME_TEMPLATES) == 6
    for template_id in TEMPLATE_IDS:
        for suffix in ('横向', '竖向'):
            for extension in ('png', 'svg'):
                assert (ASSET_DIR / f'{template_id}_{suffix}.{extension}').is_file()


def test_manifest_geometry_matches_png_sizes() -> None:
    assert film_frame._geometry('landscape').canvas == (3800, 3500)
    assert film_frame._geometry('landscape').window == (100, 550, 3600, 2400)
    assert film_frame._geometry('portrait').canvas == (3500, 3800)
    assert film_frame._geometry('portrait').window == (550, 100, 2400, 3600)
    for template_id in TEMPLATE_IDS:
        with Image.open(ASSET_DIR / f'{template_id}_横向.png') as landscape:
            assert landscape.size == (3800, 3500)
        with Image.open(ASSET_DIR / f'{template_id}_竖向.png') as portrait:
            assert portrait.size == (3500, 3800)


def test_orientation_follows_photo_and_square_uses_landscape() -> None:
    assert detect_frame_orientation(_photo(24, 36)) == 'landscape'
    assert detect_frame_orientation(_photo(36, 24)) == 'portrait'
    assert detect_frame_orientation(_photo(30, 30)) == 'landscape'


def test_film_stock_ids_resolve_to_templates() -> None:
    assert get_film_frame_template('kodak_portra_400').template_id == '01_Kodak_Portra_400'
    assert get_film_frame_template('kodak_gold_200').template_id == '02_Kodak_Gold_200'
    assert get_film_frame_template('kodak_ektar_100').template_id == '03_Kodak_Ektar_100'
    assert get_film_frame_template('kodak_5207').template_id == '06_通用无品牌_135'


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

    suffix = '横向' if orientation == 'landscape' else '竖向'
    with Image.open(ASSET_DIR / f'{template_id}_{suffix}.png') as handle:
        shipped = np.asarray(handle.convert('RGBA'), dtype=np.float32) / 255.0
    opaque = shipped[..., 3] == 1.0
    np.testing.assert_allclose(first[opaque], shipped[..., :3][opaque], atol=1e-6)


def test_perforations_show_background_not_photo() -> None:
    photo = np.ones((240, 360, 3), dtype=np.float32) * np.array([1.0, 0.0, 0.0], dtype=np.float32)
    result = compose_film_frame(photo, template_id='01_Kodak_Portra_400', options=FilmFrameOptions(output_long_edge=3800))
    np.testing.assert_allclose(result[330, 237], film_frame._PERFORATION_BACKGROUND, atol=1e-3)
    assert result[1500, 1900, 0] > 0.9 and result[1500, 1900, 1] < 0.1


def test_contain_does_not_stretch_and_cover_fills_window() -> None:
    wide = np.zeros((100, 400, 3), dtype=np.float32)
    wide[...] = [0.0, 1.0, 0.0]
    contain = compose_film_frame(wide, template_id='06_通用无品牌_135', options=FilmFrameOptions(output_long_edge=3800))
    x0, y0, x1, y1 = _window('landscape', 1.0)
    green = (contain[y0:y1, x0:x1, 1] > 0.9) & (contain[y0:y1, x0:x1, 0] < 0.1)
    rows = np.where(green.any(axis=1))[0]
    cols = np.where(green.any(axis=0))[0]
    assert (cols.max() - cols.min() + 1) / (rows.max() - rows.min() + 1) == pytest.approx(4.0, rel=0.02)

    cover = compose_film_frame(
        wide, template_id='06_通用无品牌_135', options=FilmFrameOptions(output_long_edge=3800, fit_mode='cover'),
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
        photo, template_id='06_通用无品牌_135',
        options=FilmFrameOptions(output_long_edge=950, fit_mode='cover', pan_y=0.0),
    )
    bottom = compose_film_frame(
        photo, template_id='06_通用无品牌_135',
        options=FilmFrameOptions(output_long_edge=950, fit_mode='cover', pan_y=1.0),
    )
    centre = ((y0 + y1) // 2, (x0 + x1) // 2)
    assert top[centre][0] > 0.8 or top[centre][2] > 0.8
    assert not np.array_equal(top[y0:y1, x0:x1], bottom[y0:y1, x0:x1])


def test_output_scales_whole_frame_proportionally() -> None:
    assert output_size('landscape', 1900) == (1900, 1750)
    assert output_size('portrait', 1900) == (1750, 1900)
    small = compose_film_frame(_photo(240, 360), template_id='02_Kodak_Gold_200', options=FilmFrameOptions(output_long_edge=950))
    assert small.shape[:2] == (875, 950)


def test_frame_number_hide_and_replace_only_touch_number_slots() -> None:
    options = FilmFrameOptions(output_long_edge=3800)
    base = compose_film_frame(_photo(240, 360), template_id='01_Kodak_Portra_400', options=options)
    hidden = compose_film_frame(
        _photo(240, 360), template_id='01_Kodak_Portra_400',
        options=FilmFrameOptions(output_long_edge=3800, show_frame_number=False),
    )
    changed = np.any(base != hidden, axis=2)
    assert changed.any()
    slots = [film_frame._NUMBER_SLOTS[name][4] for name in ('top_left', 'bottom_left', 'bottom_right')]
    ys, xs = np.where(changed)
    for y, x in zip(ys[::50], xs[::50]):
        assert any(x0 <= x < x1 and y0 <= y < y1 for x0, y0, x1, y1 in slots)
    # edge label and arrow are untouched
    assert np.array_equal(base[40:150, 1500:2700], hidden[40:150, 1500:2700])
    assert np.array_equal(base[3400:3460, 3210:3355], hidden[3400:3460, 3210:3355])

    replaced = compose_film_frame(
        _photo(240, 360), template_id='01_Kodak_Portra_400', frame_number='7', options=options,
    )
    assert np.any(replaced != base)
    assert np.any(replaced != hidden)


def test_frame_number_edit_in_portrait_never_touches_photo_window() -> None:
    base = compose_film_frame(
        _photo(360, 240), template_id='05_Fuji_Superia_XTRA_400', options=FilmFrameOptions(output_long_edge=3800),
    )
    hidden = compose_film_frame(
        _photo(360, 240), template_id='05_Fuji_Superia_XTRA_400',
        options=FilmFrameOptions(output_long_edge=3800, show_frame_number=False),
    )
    changed = np.any(base != hidden, axis=2)
    assert changed.any()
    x0, y0, x1, y1 = _window('portrait', 1.0)
    assert not changed[y0:y1, x0:x1].any()


def test_generic_template_has_no_frame_number() -> None:
    assert get_film_frame_template('06_通用无品牌_135').has_frame_number is False
    base = compose_film_frame(_photo(240, 360), template_id='06_通用无品牌_135')
    hidden = compose_film_frame(
        _photo(240, 360), template_id='06_通用无品牌_135', options=FilmFrameOptions(show_frame_number=False),
    )
    np.testing.assert_array_equal(base, hidden)


def test_date_stamp_is_off_by_default_and_stays_inside_photo_window() -> None:
    photo = _photo(240, 360)
    without = compose_film_frame(
        photo, template_id='01_Kodak_Portra_400', date_label='25 11 2', options=FilmFrameOptions(output_long_edge=1900),
    )
    with_date = compose_film_frame(
        photo, template_id='01_Kodak_Portra_400', date_label='25 11 2',
        options=FilmFrameOptions(output_long_edge=1900, show_date=True),
    )
    changed = np.any(without != with_date, axis=2)
    assert changed.any()
    x0, y0, x1, y1 = _window('landscape', 0.5)
    assert not changed[:y0].any() and not changed[y1:].any()
    assert not changed[:, :x0].any() and not changed[:, x1:].any()
