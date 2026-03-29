from __future__ import annotations

from os import PathLike

import colour
import numpy as np
import rawpy


_TUNGSTEN_TEMPERATURE = 2850.0


def _normalise_user_wb(user_wb: np.ndarray) -> np.ndarray:
    """Normalise white balance multipliers so the primary green channel is 1.0."""

    green_index = 1
    green_value = user_wb[green_index]
    if green_value <= 0:
        raise ValueError('Could not derive a valid white balance multiplier for the green channel.')
    return user_wb / green_value


def _camera_response_to_xyz_white(raw: rawpy.RawPy, xyz: np.ndarray) -> np.ndarray:
    """Project an XYZ whitepoint into the camera response domain.

    The mapping uses ``raw.rgb_xyz_matrix`` exposed by LibRaw. For Bayer sensors
    the fourth channel is synthesised from the green response so the result can
    be passed directly to ``rawpy`` as a four-element ``user_wb`` vector.
    """

    matrix = np.asarray(raw.rgb_xyz_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != 3 or matrix.shape[0] < 3:
        raise ValueError('RAW file does not expose a usable camera XYZ matrix for custom white balance.')

    response = matrix @ xyz
    if response.shape[0] < 4:
        response = np.pad(response, (0, 4 - response.shape[0]), constant_values=response[1])
    elif response[3] <= 0:
        response[3] = response[1]

    if np.any(response[:3] <= 0):
        raise ValueError('Derived camera response is invalid for the requested white balance.')

    return response[:4]


def _user_wb_from_temperature_and_tint(
    raw: rawpy.RawPy,
    temperature: float,
    tint: float | None,
) -> list[float]:
    """Derive ``rawpy`` ``user_wb`` multipliers from temperature and tint.

    ``rawpy`` 0.26 no longer exposes direct temperature or preset white-balance
    controls, so this function approximates them by converting the requested
    correlated colour temperature to an XYZ whitepoint and projecting that
    whitepoint through the camera's XYZ matrix.

    The ``tint`` control scales the two green channels together after the base
    temperature-derived white balance has been computed.
    """

    xy = colour.CCT_to_xy(np.float64(temperature), method='Kang 2002')
    xyz = np.asarray(colour.xy_to_XYZ(xy), dtype=np.float64)
    response = _camera_response_to_xyz_white(raw, xyz)
    user_wb = _normalise_user_wb(1.0 / response)

    if tint is not None:
        user_wb[1] *= tint
        user_wb[3] *= tint

    return user_wb.tolist()


def _postprocess_params(
    raw: rawpy.RawPy,
    white_balance: str | tuple[float, float],
    temperature: float | None,
    tint: float | None,
) -> dict[str, object]:
    """Build the ``rawpy.postprocess`` parameters for the requested settings.

    The output is always configured as linear 16-bit ACES RGB so any later
    colourspace conversion is performed on linear data by ``colour`` rather than
    on display-referred output from LibRaw.
    """

    params: dict[str, object] = {
        'output_color': getattr(rawpy, 'ColorSpace').ACES,
        'output_bps': 16,
        'no_auto_bright': True,
        'gamma': (1, 1),
    }

    if white_balance == 'as_shot':
        params['use_camera_wb'] = True
    elif white_balance == 'daylight':
        params['user_wb'] = list(raw.daylight_whitebalance)
    elif white_balance == 'tungsten':
        params['user_wb'] = _user_wb_from_temperature_and_tint(raw, _TUNGSTEN_TEMPERATURE, 1.0)
    elif white_balance == 'custom':
        if temperature is None:
            raise ValueError('A custom raw white balance requires a temperature value.')
        params['user_wb'] = _user_wb_from_temperature_and_tint(raw, temperature, tint)
    else:
        custom_temperature, custom_tint = white_balance
        params['user_wb'] = _user_wb_from_temperature_and_tint(raw, custom_temperature, custom_tint)

    return params


def load_and_process_raw_file(
    raw_path: str | PathLike[str],
    white_balance='as_shot',
    temperature: float | None = None,
    tint: float | None = None,
    output_colorspace: str = 'ACES2065-1',
    output_cctf_encoding: bool = True,
) -> np.ndarray:
    """Load a RAW file into linear RGB and optionally convert its colourspace.

    The RAW is demosaiced by ``rawpy`` into linear 16-bit ACES RGB with auto
    brightening disabled. White balance can come from the camera metadata, the
    LibRaw daylight multipliers, or a temperature/tint approximation derived
    from the camera XYZ matrix.

    Parameters
    ----------
    raw_path
        Path to the RAW image.
    white_balance
        White-balance mode. Supported string values are ``'as_shot'``,
        ``'daylight'``, ``'tungsten'``, and ``'custom'``. A
        ``(temperature, tint)`` tuple is also accepted.
    temperature
        Correlated colour temperature in kelvin for ``'custom'`` mode.
    tint
        Multiplicative adjustment applied to both green channels for
        temperature-derived white balance.
    output_colorspace
        Output RGB colourspace name understood by ``colour.RGB_COLOURSPACES``.
    output_cctf_encoding
        Whether to apply the output colourspace transfer function when a
        colourspace conversion is requested.

    Returns
    -------
    numpy.ndarray
        RGB image as ``float32`` in the requested output colourspace.
    """

    with rawpy.imread(str(raw_path)) as raw:
        params = _postprocess_params(raw, white_balance, temperature, tint)
        rgb = raw.postprocess(**params).astype(np.float32) / np.float32(65535.0)

    if output_colorspace != 'ACES2065-1':
        rgb = colour.RGB_to_RGB(
            rgb,
            input_colourspace=colour.RGB_COLOURSPACES['ACES2065-1'],
            output_colourspace=colour.RGB_COLOURSPACES[output_colorspace],
            apply_cctf_decoding=False,
            apply_cctf_encoding=output_cctf_encoding,
        )

    return rgb


__all__ = ['load_and_process_raw_file']