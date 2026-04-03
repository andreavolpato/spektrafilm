import copy

import numpy as np
import scipy

from spektrafilm_profile_creator.diagnostics.messages import log_event
from spektrafilm.utils.measure import measure_density_min


def remove_density_min(profile, reconstruct_base_density=False):
    data = profile.data
    info = profile.info
    log_exposure = data.log_exposure
    density_curves = data.density_curves
    base_density = np.asarray(data.base_density)
    wavelengths = data.wavelengths
    profile_type = info.type

    density_curve_min = measure_density_min(log_exposure, density_curves, profile_type)
    density_curves = density_curves - density_curve_min

    if reconstruct_base_density:
        # TODO make this more clear, maybe split in a function remove_density_and_reconstruct_base_density
        status_a_max_peak = [445, 530, 610]
        spectral_min = np.interp(wavelengths, status_a_max_peak, np.flip(density_curve_min))
        sigma_nm = 20
        sigma_points = sigma_nm / np.mean(np.diff(wavelengths))
        spectral_min = scipy.ndimage.gaussian_filter1d(spectral_min, sigma_points)
        base_density = spectral_min

    updated_profile = profile.update_data(
        base_density=np.asarray(base_density),
        density_curves=density_curves,
    )
    log_event(
        'remove_density_min',
        updated_profile,
        density_curve_min=density_curve_min,
    )
    return updated_profile


def adjust_log_exposure(
    profile,
    speed_point_density=0.2,
    stops_over_speed_point=3,
    midgray_transmittance=0.184,
):
    info = profile.info
    density_curves_green = profile.data.density_curves[:, 1]
    if info.is_paper or info.is_positive:
        speed_point_density = np.log10(1 / midgray_transmittance)
        stops_over_speed_point = 0

    log_event(
        'adjust_log_exposure',
        reference_density=speed_point_density,
        stops_over_reference_density_ev=stops_over_speed_point,
    )
    density_curves_green = density_curves_green - np.nanmin(density_curves_green)
    log_exposure = profile.data.log_exposure
    selection = ~np.isnan(density_curves_green)
    if info.is_positive:
        log_exposure_speed_point = np.interp(
            -speed_point_density,
            -density_curves_green[selection],
            log_exposure[selection],
        )
    else:
        log_exposure_speed_point = np.interp(
            speed_point_density,
            density_curves_green[selection],
            log_exposure[selection],
        )
    log_exposure_offset = np.log10(2 ** stops_over_speed_point)
    log_exposure_midgray = log_exposure_speed_point + log_exposure_offset
    updated_profile = profile.update_data(log_exposure=log_exposure - log_exposure_midgray)
    log_event(
        'adjust_log_exposure_result',
        updated_profile,
        log_exposure_reference=log_exposure_speed_point,
    )
    return updated_profile

def adjust_log_exposure_midgray_to_metameric_neutral(profile):
    info = profile.info
    data = profile.data
    density_curves = data.density_curves
    log_exposure = data.log_exposure
    fitted_cmy_midscale_neutral_density = info.fitted_cmy_midscale_neutral_density
    
    log_exposure_shifts = np.zeros(3)
    for index in np.arange(3):
        density_curve = density_curves[:, index]
        not_nan = ~np.isnan(density_curve)
        if info.type == 'positive':
            log_exposure_shift = np.interp(
                -fitted_cmy_midscale_neutral_density[index],
                -density_curve[not_nan],
                log_exposure[not_nan],
            )
        else:
            log_exposure_shift = np.interp(
                fitted_cmy_midscale_neutral_density[index],
                density_curve[not_nan],
                log_exposure[not_nan],
            )
        log_exposure_shifts[index] = log_exposure_shift
        density_curve = np.interp(
            log_exposure + log_exposure_shift,
            log_exposure,
            density_curve
        )
        density_curves[:, index] = density_curve
    updated_profile = profile.update_data(density_curves=density_curves)
    log_event(
        'adjust_log_exposure_to_metameric_neutral',
        updated_profile,
        log_exposure_shifts=log_exposure_shifts,
        
    )
    return updated_profile
    

def measure_log_exposure_midscale_neutral(profile, reference_channel=None):
    data = profile.data
    info = profile.info
    log_exposure_midscale_neutral = np.zeros((3,))
    fitted_midscale_density = info.fitted_cmy_midscale_neutral_density
    if np.size(fitted_midscale_density) == 1:
        fitted_midscale_density = np.ones(3) * fitted_midscale_density
    if reference_channel == 'green':
        fitted_midscale_density = np.ones(3) * fitted_midscale_density[1]
    for index in range(3):
        if info.is_positive:
            log_exposure_midscale_neutral[index] = np.interp(
                -fitted_midscale_density[index],
                -data.density_curves[:, index],
                data.log_exposure,
            )
        else:
            log_exposure_midscale_neutral[index] = np.interp(
                fitted_midscale_density[index],
                data.density_curves[:, index],
                data.log_exposure,
            )
    log_event('measure_midscale_neutral', log_exposure_midscale_neutral=log_exposure_midscale_neutral)
    return log_exposure_midscale_neutral


def align_midscale_neutral_exposures(profile, reference_channel=None):
    log_exposure_midscale_neutral = measure_log_exposure_midscale_neutral(
        profile,
        reference_channel,
    )
    density_curves = np.array(profile.data.density_curves, copy=True)
    log_exposure = profile.data.log_exposure
    for index in np.arange(3):
        density_curves[:, index] = np.interp(
            log_exposure,
            log_exposure - log_exposure_midscale_neutral[index],
            density_curves[:, index],
        )
    return profile.update(
        data={
            'density_curves': density_curves,
        },
        info={
            'log_exposure_midscale_neutral': (np.ones(3) * log_exposure_midscale_neutral[1]).tolist(),
        },
    )


def apply_scale_shift_stretch_density_curves(
    profile,
    density_scale=(1, 1, 1),
    log_exposure_shift=(0, 0, 0),
    log_exposure_stretch=(1, 1, 1),
):
    density_curves = copy.copy(profile.data.density_curves)
    log_exposure = copy.copy(profile.data.log_exposure)
    density_curves = density_curves * np.asarray(density_scale)
    for index in np.arange(3):
        density_curves[:, index] = np.interp(
            log_exposure,
            log_exposure / log_exposure_stretch[index] + log_exposure_shift[index],
            density_curves[:, index],
        )
    return profile.update_data(density_curves=density_curves)


def preprocess_profile(profile):
    profile = remove_density_min(profile)
    profile = adjust_log_exposure(profile)
    return profile
