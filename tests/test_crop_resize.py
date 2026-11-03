import numpy as np
import pytest

from spektrafilm.utils.crop_resize import crop_image


@pytest.mark.unit
@pytest.mark.parametrize("shape", [(100, 150, 3), (150, 100, 3)])
def test_crop_size_one_returns_whole_image(shape):
    image = np.random.default_rng(0).random(shape)

    crop = crop_image(image, center=(0.5, 0.5), size=(1.0, 1.0))

    assert crop.shape == shape
    np.testing.assert_array_equal(crop, image)


@pytest.mark.unit
def test_crop_wider_than_short_side_keeps_full_height():
    image = np.random.default_rng(1).random((100, 150, 3))

    crop = crop_image(image, center=(0.5, 0.5), size=(0.5, 0.9))

    # 0.9 of the long side (135 px) is more than the 100 px height.
    assert crop.shape == (100, 75, 3)
    np.testing.assert_array_equal(crop, image[:, 38:113])


@pytest.mark.unit
def test_small_centered_crop_is_unchanged():
    image = np.random.default_rng(2).random((100, 150, 3))

    crop = crop_image(image, center=(0.5, 0.5), size=(0.1, 0.1))

    assert crop.shape == (15, 15, 3)
    np.testing.assert_array_equal(crop, image[42:57, 68:83])
