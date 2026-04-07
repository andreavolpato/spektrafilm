from __future__ import annotations

import numpy as np
import skimage.transform

from spektrafilm.utils.crop_resize import crop_image


class ResizingService:
    """Shared resize/crop operations used by runtime stages."""

    def __init__(self, io_params, film_format_mm: float):
        self._io = io_params
        self.film_format_mm = film_format_mm
        self.pixel_size_um = None

    def crop_and_rescale(self, image: np.ndarray) -> np.ndarray:

        if self._io.crop:
            image = crop_image(image, center=self._io.crop_center, size=self._io.crop_size)

        if self._io.upscale_factor != 1.0:
            image = skimage.transform.rescale(
                image,
                self._io.upscale_factor,
                channel_axis=2,
                order=3,
            )

        pixel_size_um = self.film_format_mm * 1000 / np.max(image.shape[0:2])
        self.pixel_size_um = pixel_size_um

        return image
