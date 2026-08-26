from __future__ import annotations

import colour
import numpy as np

from spektrafilm.model.convert import FilmConverter, parse_calibration_matrix
from spektrafilm.model.illuminants import standard_illuminant


class ConvertingStage:
    """The "convert-film" front-end: scanned-negative RGB -> film cmy density.

    Inverts the scan model (see :mod:`spektrafilm.model.convert`) so a scanned
    negative can be injected at the cmy_film tap and printed. Replaces filming
    (expose+develop) for the convert-film workflow routes. Uses the user-selected
    scan illuminant and exposure (``film_render.convert``) and the film base
    tuning (``film_render.base``).
    """

    def __init__(
        self, film, film_render_params, io_params, settings_params, lut_service
    ):
        self._film = film
        self._film_render = film_render_params
        self._io = io_params
        self._settings = settings_params
        self._lut_service = lut_service
        self._converter = self._build_converter()

    # public methods

    def convert(self, image: np.ndarray) -> np.ndarray:
        """Scanned-negative RGB (H, W, 3) -> film cmy density (H, W, 3).

        The cheap analytic device calibration + exposure gain runs inline; the
        scan-model inversion goes through the LUT service so it can be accelerated
        by a 3D LUT (``use_convert_lut``) — orders of magnitude faster than the
        per-pixel Gauss-Newton solve, with imperceptible output error (see c40)."""
        image = self._decode_to_linear(image)
        y = self._converter.pretransform(image)
        data_min, data_max = self._converter.gamut_bounds()
        return self._lut_service.spectral_compute_convert(
            y,
            invert_fn=self._converter.invert,
            data_min=data_min,
            data_max=data_max,
            use_lut=self._settings.use_convert_lut,
            steps=self._settings.convert_lut_resolution,
        )

    # private methods

    def _build_converter(self) -> FilmConverter:
        data = self._film.data
        # cmy domain = per-channel Dmax of the normalized density curves (the
        # range filming.develop emits onto cmy_film).
        dc = np.asarray(data.density_curves, float)
        cmy_max = np.nanmax(dc - np.nanmin(dc, axis=0), axis=0)
        convert = self._film_render.convert
        return FilmConverter(
            channel_density=data.channel_density,
            base_density=data.base_density,
            base_params=self._film_render.base,
            scan_illuminant=standard_illuminant(convert.scan_illuminant),
            output_colourspace=self._io.input_color_space,
            cmy_max=cmy_max,
            exposure_ev=convert.exposure_compensation_ev,
            calibration=parse_calibration_matrix(convert.calibration),
        )

    def _decode_to_linear(self, image: np.ndarray) -> np.ndarray:
        """The converter works in linear input-space RGB; decode the CCTF if the
        caller opted into colour-science's built-in decoding (mirrors filming)."""
        if self._io.input_cctf_decoding:
            cs = colour.RGB_COLOURSPACES[self._io.input_color_space]
            image = cs.cctf_decoding(image)
        return image
