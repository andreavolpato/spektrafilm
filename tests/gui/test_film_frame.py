from __future__ import annotations

import numpy as np
import pytest

from spektrafilm_gui.film_frame import (
    FILM_FRAME_TEMPLATES,
    FilmFrameOptions,
    compose_film_frame,
    detect_frame_orientation,
    detect_image_aspect,
)


@pytest.mark.parametrize(
    ('shape', 'expected_shape', 'expected_orientation'),
    [
        ((60, 90, 3), (86, 90, 3), 'landscape'),
        ((90, 60, 3), (90, 86, 3), 'portrait'),
    ],
)
def test_compose_film_frame_rotates_the_whole_frame_for_portrait(
    shape: tuple[int, int, int],
    expected_shape: tuple[int, int, int],
    expected_orientation: str,
) -> None:
    source = np.full(shape, 0.37, dtype=np.float32)

    framed = compose_film_frame(source, template_id='kodak_gold_200')

    assert detect_frame_orientation(source) == expected_orientation
    assert framed.shape == expected_shape
    assert framed.dtype == np.float32
    assert 0.95 <= framed.shape[1] / framed.shape[0] <= 1.06
    np.testing.assert_allclose(framed[framed.shape[0] // 2, framed.shape[1] // 2], 0.37, atol=0.01)
    np.testing.assert_allclose(framed[0, 0], [0.008, 0.01, 0.012], atol=0.01)


@pytest.mark.parametrize(
    ('shape', 'ratio', 'orientation'),
    [
        ((48, 72, 3), 1.5, 'landscape'),
        ((72, 48, 3), 2 / 3, 'portrait'),
        ((60, 60, 3), 1.0, 'landscape'),
        ((48, 64, 3), 4 / 3, 'landscape'),
        ((36, 64, 3), 16 / 9, 'landscape'),
    ],
)
def test_detects_dimensions_ratio_and_orientation(
    shape: tuple[int, int, int],
    ratio: float,
    orientation: str,
) -> None:
    info = detect_image_aspect(np.zeros(shape, dtype=np.float32))
    assert info.width == shape[1]
    assert info.height == shape[0]
    assert info.ratio == pytest.approx(ratio)
    assert info.orientation == orientation


def test_contain_preserves_nonstandard_photo_without_distortion() -> None:
    source = np.zeros((60, 80, 3), dtype=np.float32)
    source[:] = [0.8, 0.2, 0.1]
    framed = compose_film_frame(
        source,
        template_id='kodak_gold_200',
        options=FilmFrameOptions(fit_mode='contain'),
    )

    band_height = (framed.shape[0] - 53) // 2
    gate = framed[band_height:band_height + 53]
    content = np.all(np.isclose(gate, [0.8, 0.2, 0.1], atol=0.02), axis=2)
    rows, columns = np.where(content)
    assert (columns.max() - columns.min() + 1) / (rows.max() - rows.min() + 1) == pytest.approx(
        4 / 3,
        rel=0.04,
    )
    assert not content[:, 0].any()


def test_cover_fills_photo_gate_by_cropping_instead_of_stretching() -> None:
    source = np.full((60, 80, 3), [0.7, 0.3, 0.1], dtype=np.float32)
    framed = compose_film_frame(
        source,
        template_id='kodak_gold_200',
        options=FilmFrameOptions(fit_mode='cover'),
    )

    gate_height = round(framed.shape[1] / 1.5)
    band_height = (framed.shape[0] - gate_height) // 2
    gate = framed[band_height:band_height + gate_height]
    assert np.allclose(gate, [0.7, 0.3, 0.1], atol=0.02)


def test_unverified_templates_do_not_invent_manufacturer_markings() -> None:
    source = np.full((60, 90, 3), 0.37, dtype=np.float32)
    unverified = compose_film_frame(source, template_id='kodak_gold_200')
    portra = compose_film_frame(source, template_id='kodak_portra_400')

    assert all(template.reference_status == '待核实' for template in FILM_FRAME_TEMPLATES[1:])
    assert np.count_nonzero(unverified != portra) > 0
    # The unverified model has no text, code, or frame number; only the
    # perforations and black rails remain.
    top_rail = unverified[:12]
    assert np.count_nonzero(top_rail != np.array([0.008, 0.01, 0.012])) > 0
    assert np.allclose(unverified[:3], [0.008, 0.01, 0.012])


def test_portra_frame_number_is_user_editable_and_date_defaults_off() -> None:
    source = np.full((60, 90, 3), 0.37, dtype=np.float32)
    default = compose_film_frame(source, template_id='kodak_portra_400')
    changed = compose_film_frame(source, template_id='kodak_portra_400', frame_number='75')

    assert FilmFrameOptions().show_date is False
    assert not np.array_equal(default, changed)


@pytest.mark.parametrize(
    'source',
    [
        np.zeros((0, 4, 3), dtype=np.float32),
        np.zeros((4, 4), dtype=np.float32),
        np.zeros((4, 4, 4), dtype=np.float32),
        np.full((4, 4, 3), np.nan, dtype=np.float32),
    ],
)
def test_compose_film_frame_rejects_invalid_images(source: np.ndarray) -> None:
    with pytest.raises(ValueError):
        compose_film_frame(source)


def test_compose_film_frame_rejects_invalid_fit_mode() -> None:
    source = np.zeros((60, 90, 3), dtype=np.float32)
    with pytest.raises(ValueError, match='fit mode'):
        compose_film_frame(source, options=FilmFrameOptions(fit_mode='stretch'))
