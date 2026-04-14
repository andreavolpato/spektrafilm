import copy
import math

import matplotlib.pyplot as plt
import numpy as np
import scipy

from spektrafilm_profile_creator.data.loader import load_densitometer_data
from spektrafilm_profile_creator.core.densitometer import compute_densitometer_crosstalk_matrix


ERF = np.vectorize(math.erf)


def find_midscale_neutral_coefficients(channel_density, base_density, midscale_neutral_density, fit=True):
    mid_over_base = midscale_neutral_density - base_density

    if fit:
        selection = np.all(~np.isnan(channel_density), axis=1)

        def residues_midscale_neutral_dye_density(coefficients):
            residues = mid_over_base - np.nansum(channel_density * coefficients, axis=1)
            return residues[selection]

        fit_result = scipy.optimize.least_squares(residues_midscale_neutral_dye_density, [1, 1, 1])
        coefficients = fit_result.x
    else:
        density_max = np.nanmax(channel_density, axis=0)
        index_max = np.nanargmax(channel_density, axis=0)
        density_mid = np.zeros(3)
        for index, max_index in enumerate(index_max):
            density_mid[index] = np.interp(max_index, np.arange(np.size(mid_over_base)), mid_over_base)
        coefficients = density_mid / density_max
    return coefficients


def fit_log_scaled_absortion_coefficients(
    sensitivity,
    crosstalk_matrix,
    density_curves,
    log_exposure,
    density_level=0.2,
    log_exposure_reference_sensitivity=0,
):
    sensitivity[sensitivity == 0] = np.nan
    sensitivity_log_exposures = np.log10(1 / sensitivity)
    sensitivity_log_exposures[np.isnan(sensitivity_log_exposures)] = np.nanmax(sensitivity_log_exposures) + 6

    def unmixed_densities_at_a_sensitivity_log_exposure(log_absorption_coefficients_rgb, sensitivity_log_exposure_i):
        unmixed_densities = np.zeros(3)
        for index in np.arange(3):
            log_exposure_with_new_reference = log_exposure_reference_sensitivity + sensitivity_log_exposure_i
            unmixed_densities[index] = np.interp(
                log_exposure_with_new_reference + log_absorption_coefficients_rgb[index],
                log_exposure,
                density_curves[:, index],
            )
        if np.isnan(sensitivity_log_exposure_i):
            unmixed_densities = np.nan * np.ones(3)
        return unmixed_densities

    def densitometer_densities_at_sensitivity_log_exposures(log_absorption_coefficients_rgb, sensitivity_log_exposures_row):
        densitometer_density = np.zeros(3)
        for index in np.arange(3):
            unmixed_densities = unmixed_densities_at_a_sensitivity_log_exposure(
                log_absorption_coefficients_rgb,
                sensitivity_log_exposures_row[index],
            )
            densitometer_density[index] = np.sum(crosstalk_matrix[index, :] * unmixed_densities)
        return densitometer_density

    target = np.array([density_level, density_level, density_level])
    log_absorption_coefficients = np.zeros(sensitivity_log_exposures.shape)
    log_absorption_coefficients_0 = np.log10(sensitivity)
    log_absorption_coefficients_0[np.isnan(log_absorption_coefficients_0)] = np.nanmin(log_absorption_coefficients_0) - 2

    def residues(log_absorption_coefficients_rgb, sensitivity_row):
        return target - densitometer_densities_at_sensitivity_log_exposures(log_absorption_coefficients_rgb, sensitivity_row)

    for wavelength_index in np.arange(sensitivity_log_exposures.shape[0]):
        sensitivity_log_exposures_row = sensitivity_log_exposures[wavelength_index, :]
        fit_result = scipy.optimize.least_squares(
            residues,
            log_absorption_coefficients_0[wavelength_index, :],
            args=(sensitivity_log_exposures_row,),
            method='lm',
        )
        log_absorption_coefficients[wavelength_index, :] = fit_result.x

    log_absorption_coefficients[np.isnan(sensitivity)] = np.nan
    return log_absorption_coefficients


def find_log_exposure_reference(log_exposure, density_curves, density_reference=1.0, decreasing_density=False):
    selection = np.all(~np.isnan(density_curves), axis=1)
    sign = -1 if decreasing_density else 1
    log_exposure_reference = 0
    for index in np.arange(3):
        log_exposure_reference += np.interp(
            sign * density_reference,
            sign * density_curves[selection, index],
            log_exposure[selection],
        )
    return log_exposure_reference / 3


def unmix_sensitivity(profile, control_plot=False):
    print('----------------------------------------')
    print('# Unmixing Sensitivity - assumes unmixed densities')
    print(profile.info.stock, ' - ', profile.info.support, profile.info.type)

    log_sensitivity = profile.data.log_sensitivity
    density_curves = profile.data.density_curves
    channel_density = np.asarray(profile.data.channel_density)
    log_exposure = profile.data.log_exposure
    wavelengths = profile.data.wavelengths
    sensitivity_density_level = profile.info.log_sensitivity_density_over_min
    sensitivity = 10 ** log_sensitivity

    densitometer_responsivity = load_densitometer_data(densitometer_type=profile.info.densitometer)
    densitometer_crosstalk_matrix = compute_densitometer_crosstalk_matrix(densitometer_responsivity, channel_density)
    density_curves_densitometer_minus_dmin = np.einsum('ij,kj->ki', densitometer_crosstalk_matrix, density_curves)

    log_sensitivity_prefit = np.copy(log_sensitivity)
    log_exposure_reference_sensitivity = find_log_exposure_reference(
        log_exposure,
        density_curves_densitometer_minus_dmin,
        sensitivity_density_level,
        decreasing_density=profile.info.is_positive,
    )
    print('Log-exposure reference for sensitivity: ', log_exposure_reference_sensitivity)
    log_absorption_coefficients = fit_log_scaled_absortion_coefficients(
        sensitivity,
        densitometer_crosstalk_matrix,
        density_curves,
        log_exposure,
        sensitivity_density_level,
        log_exposure_reference_sensitivity,
    )

    profile.data.log_sensitivity = log_sensitivity

    if control_plot:
        _, axis = plt.subplots()
        axis.plot(wavelengths, log_absorption_coefficients, color='k')
        axis.plot(wavelengths, log_sensitivity_prefit, color='gray', linestyle='--')
        axis.legend(('r', 'g', 'b'))
        axis.set_title('Sensitivity unmix')
        axis.set_xlabel('Wavelength (nm)')
        axis.set_ylabel('Log Sensitivity')
        axis.set_title(profile.info.stock)

    return profile


def apply_masking_couplers(
    profile,
    control_plot=True,
    effectiveness=1.0,
    model='erf',
    cross_over_points=(585, 510, 200),
    transition_widths=(15, 15, 1),
    gaussian_model=(((435, 20, 0.09), (560, 20, 0.09)), ((470, 20, 0.09),), ((520, 20, 0.09),)),
):
    channel_density = np.asarray(profile.data.channel_density)
    base_density = np.asarray(profile.data.base_density)
    midscale_neutral_density = np.asarray(profile.data.midscale_neutral_density)
    wavelengths = profile.data.wavelengths

    if model == 'erf':
        wavelengths_scaled = (wavelengths[:, None] - cross_over_points) / transition_widths
        coupler_mask_spectral = (ERF(wavelengths_scaled) + 1 + effectiveness) / (2 + effectiveness)
        dye_density = copy.copy(channel_density)
        dye_density_with_couplers = dye_density * coupler_mask_spectral
        profile.data.channel_density = np.asarray(dye_density_with_couplers)
    elif model == 'gaussians':
        def spectral_profiles(wavelength_axis, parameters):
            density = np.zeros((np.size(wavelength_axis), 3))
            for index in np.arange(3):
                for peak_wavelength, width, amount in parameters[index]:
                    density[:, index] += amount * np.exp(-((wavelength_axis - peak_wavelength) ** 2) / (2 * width ** 2))
            return density

        coupler_mask_spectral_subtractive = spectral_profiles(wavelengths, gaussian_model)
        dye_density = copy.copy(channel_density)
        dye_density_with_couplers = dye_density - coupler_mask_spectral_subtractive * effectiveness
        profile.data.channel_density = np.asarray(dye_density_with_couplers)
    else:
        raise ValueError(f'Unsupported masking coupler model: {model}')

    if control_plot:
        dye_density_midscale_coefficients = find_midscale_neutral_coefficients(
            channel_density,
            base_density,
            midscale_neutral_density,
        )
        print('midscale_coefficients: ', dye_density_midscale_coefficients)
        mid_sim = dye_density_with_couplers * dye_density_midscale_coefficients
        mid_sim = np.sum(mid_sim, axis=1) + base_density

        _, axis = plt.subplots()
        axis.plot(wavelengths, dye_density[:, 0], color='tab:cyan')
        axis.plot(wavelengths, dye_density[:, 1], color='tab:pink')
        axis.plot(wavelengths, dye_density[:, 2], color='gold')
        axis.plot(wavelengths, dye_density_with_couplers[:, 0], color='tab:cyan', linestyle='--', label='_nolegend_')
        axis.plot(wavelengths, dye_density_with_couplers[:, 1], color='tab:pink', linestyle='--', label='_nolegend_')
        axis.plot(wavelengths, dye_density_with_couplers[:, 2], color='gold', linestyle='--', label='_nolegend_')
        axis.plot(wavelengths, base_density, color='gray', linewidth=1)
        axis.plot(wavelengths, midscale_neutral_density, color='lightgray', linewidth=1)
        axis.plot(wavelengths, mid_sim, color='gray', linestyle='--', linewidth=1)
        axis.legend(('C', 'M', 'Y', 'Min', 'Mid', 'Sim'))
        axis.set_xlabel('Wavelength (nm)')
        axis.set_ylabel('Diffuse Density')
        axis.set_xlim((350, 750))
        axis.set_title('Masking Couplers Effects to Dye Density')
    return profile


def rescale_dye_density_using_neutral(profile):
    channel_density = np.asarray(profile.data.channel_density)
    base_density = np.asarray(profile.data.base_density)
    midscale_neutral_density = np.asarray(profile.data.midscale_neutral_density)
    dye_density_midscale_coefficients = find_midscale_neutral_coefficients(
        channel_density,
        base_density,
        midscale_neutral_density,
    )
    dye_density_midscale_coefficients /= dye_density_midscale_coefficients[1]
    profile.data.channel_density = np.asarray(channel_density * dye_density_midscale_coefficients)
    return profile
