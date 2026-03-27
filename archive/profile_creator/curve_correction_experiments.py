import copy

import numpy as np
import scipy
import scipy.interpolate

from spektrafilm_profile_creator.data.loader import load_densitometer_data
from spektrafilm_profile_creator.core.profile_transforms import (
    align_midscale_neutral_exposures,
    measure_log_exposure_midscale_neutral,
)


def compute_densitometer_correction(dye_density, densitometer_type='status_A'):
    densitometer_responsivities = load_densitometer_data(densitometer_type=densitometer_type)
    dye_density = dye_density[:, 0:3]
    dye_density[np.isnan(dye_density)] = 0
    return 1 / np.sum(densitometer_responsivities[:] * dye_density, axis=0)


def heavy_lifting_density_curves(
    profile,
    density_max=False,
    log_exposure=False,
    gamma=False,
    gamma_correction_range_ev=2,
):
    corrected = copy.copy(profile)
    if density_max:
        corrected = normalize_density_max(corrected)
    if log_exposure:
        corrected = align_midscale_neutral_exposures(corrected)
    if gamma:
        gamma_values, log_exposure_reference = measure_slopes(
            corrected,
            log_exposure_range=np.log10(2 ** gamma_correction_range_ev),
        )
        mean_gamma = np.mean(gamma_values)
        slope_correction = gamma_values / mean_gamma
        corrected = apply_gamma_correction(corrected, 1 / slope_correction, log_exposure_reference)
    return corrected


def measure_slopes(profile, log_exposure_range=np.log10(2 ** 2)):
    log_exposure_reference = measure_log_exposure_midscale_neutral(profile)
    log_exposure_0 = log_exposure_reference - log_exposure_range / 2
    log_exposure_1 = log_exposure_reference + log_exposure_range / 2
    density_curves = profile.data.density_curves
    log_exposure = profile.data.log_exposure
    gamma = np.zeros((3,))
    for index in range(3):
        selection = ~np.isnan(density_curves[:, index])
        density_1 = scipy.interpolate.CubicSpline(log_exposure[selection], density_curves[selection, index])(log_exposure_1[index])
        density_0 = scipy.interpolate.CubicSpline(log_exposure[selection], density_curves[selection, index])(log_exposure_0[index])
        gamma[index] = (density_1 - density_0) / (log_exposure_1[index] - log_exposure_0[index])
    print('Gamma:', gamma)
    return gamma, log_exposure_reference


def apply_gamma_correction(profile, gamma_correction, log_exposure_reference=(0.0, 0.0, 0.0)):
    density_curves = profile.data.density_curves
    log_exposure = profile.data.log_exposure
    gamma_correction = np.asarray(gamma_correction)
    density_curves_out = np.zeros_like(density_curves)
    for index in np.arange(3):
        density_curves_out[:, index] = np.interp(
            log_exposure,
            (log_exposure - log_exposure_reference[index]) / gamma_correction[index] + log_exposure_reference[index],
            density_curves[:, index],
        )
    profile.data.density_curves = density_curves_out
    return profile


def normalize_density_max(profile):
    density_max = np.nanmax(profile.data.density_curves, axis=0)
    density_max_mean = np.mean(density_max)
    print('Density max:', density_max)
    profile.data.density_curves *= density_max_mean / density_max
    print('Setting density max of all channels to:', density_max_mean)
    return profile
