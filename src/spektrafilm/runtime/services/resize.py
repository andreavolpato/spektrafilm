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
        self._preview_resize_factor = None
        self._interpolation_order = 1 if self._io.full_image else 0

    def crop_and_rescale(self, image: np.ndarray) -> tuple[np.ndarray, float, float]:
        preview_resize_factor = self._io.preview_resize_factor
        upscale_factor = self._io.upscale_factor

        if self._io.crop:
            image = crop_image(image, center=self._io.crop_center, size=self._io.crop_size)

        if self._io.full_image:
            preview_resize_factor = 1.0

        scale = preview_resize_factor * upscale_factor
        if scale != 1.0:
            image = skimage.transform.rescale(image, scale, channel_axis=2, order=self._interpolation_order)

        pixel_size_um = self.film_format_mm * 1000 / np.max(image.shape[0:2])
        pixel_size_um /= scale
        self.pixel_size_um = pixel_size_um
        self._preview_resize_factor = preview_resize_factor
        
        return image

    def rescale_to_original(self, scan: np.ndarray) -> np.ndarray:
        if self._preview_resize_factor != 1.0:
            scan = skimage.transform.rescale(scan, 1 / self._preview_resize_factor, channel_axis=2, order=self._interpolation_order)
        return scan
