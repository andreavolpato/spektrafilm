import math

import numpy as np
import scipy.interpolate


_ERF = np.vectorize(math.erf)


def low_pass_filter(wavelengths, wavelength_max, width, amplitude=1.0):
    return 1 - amplitude * (_ERF((wavelengths - wavelength_max) / width) + 1) / 2


def high_pass_filter(wavelengths, wavelength_min, width, amplitude=1.0):
    return 1 - amplitude + amplitude * (_ERF((wavelengths - wavelength_min) / width) + 1) / 2


def high_pass_gaussian(wavelengths, wavelength_max, width, amount):
    return amount * np.exp(-(wavelengths - wavelength_max + width) ** 2 / (2 * width ** 2))


def low_pass_gaussian(wavelengths, wavelength_max, width, amount):
    return amount * np.exp(-(wavelengths - wavelength_max - width) ** 2 / (2 * width ** 2))


def shift_stretch(wavelengths, spectrum, amplitude=1.0, width=1.0, shift=0.0):
    center = wavelengths[np.nanargmax(spectrum)]
    selection = ~np.isnan(spectrum)
    smoothing = 100
    spline = scipy.interpolate.make_smoothing_spline(
        wavelengths[selection],
        spectrum[selection],
        lam=smoothing,
    )
    spectrum_out = spline((wavelengths - center) / width + center + shift)
    spline = scipy.interpolate.make_smoothing_spline(wavelengths, spectrum_out, lam=smoothing)
    spectrum_out = spline(wavelengths)
    spectrum_out[spectrum_out < 0] = 0
    spectrum_out[~selection] = np.nan
    return amplitude * spectrum_out


def shift_stretch_cmy(
    wavelengths,
    cmy,
    dye_amplitude0,
    dye_width0,
    dye_shift0,
    dye_amplitude1,
    dye_width1,
    dye_shift1,
    dye_amplitude2,
    dye_width2,
    dye_shift2,
):
    cyan = shift_stretch(wavelengths, cmy[:, 0], dye_amplitude0, dye_width0, dye_shift0)
    magenta = shift_stretch(wavelengths, cmy[:, 1], dye_amplitude1, dye_width1, dye_shift1)
    yellow = shift_stretch(wavelengths, cmy[:, 2], dye_amplitude2, dye_width2, dye_shift2)
    return np.vstack([cyan, magenta, yellow]).T


def gaussian_profiles(wavelengths, coupler_parameters):
    density = np.zeros((np.size(wavelengths), np.size(coupler_parameters, axis=0)))
    for index, parameters in enumerate(coupler_parameters):
        density[:, index] += parameters[0] * np.exp(
            -(wavelengths - parameters[2]) ** 2 / (2 * parameters[1] ** 2)
        )
    return density
