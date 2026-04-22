import colour
import numpy as np
import scipy
import scipy.optimize

from colour.models import RGB_COLOURSPACE_sRGB
from spektrafilm.config import STANDARD_OBSERVER_CMFS
from spektrafilm.model.color_filters import compute_band_pass_filter, color_enlarger
from spektrafilm.model.illuminants import standard_illuminant
from spektrafilm_profile_creator.diagnostics.messages import log_event
from spektrafilm_profile_creator.data.loader import load_densitometer_data, load_raw_profile
from spektrafilm.utils.spectral_upsampling import rgb_to_smooth_spectrum
from spektrafilm_profile_creator.neutral_print_filters import DEFAULT_NEUTRAL_PRINT_FILTERS


def balance_film_sensitivity(profile, band_pass_filter=False):
    data = profile.data
    info = profile.info
    log_sensitivity = data.log_sensitivity
    midgray = np.array([[[0.184, 0.184, 0.184]]])
    illuminant = rgb_to_smooth_spectrum(midgray, color_space='ProPhoto RGB',
                                        apply_cctf_decoding=False,
                                        reference_illuminant=info.reference_illuminant)
    # illuminant = standard_illuminant(type=info.reference_illuminant)
    sensitivity = 10 ** log_sensitivity

    if band_pass_filter:
        filter_uv = (1, 410, 8)
        filter_ir = (1, 675, 15)
        band_pass = compute_band_pass_filter(filter_uv, filter_ir)
        illuminant *= band_pass

    neutral_exposures = np.nansum(illuminant[:, None] * sensitivity, axis=0)
    correction = 1 / neutral_exposures
    log_exposure_correction = np.log10(correction)

    sensitivity *= correction
    updated_log_sensitivity = np.log10(sensitivity)

    updated_profile = profile.update_data(log_sensitivity=updated_log_sensitivity)
    log_event(
        'balance_film_sensitivity',
        updated_profile,
        sensitivity_correction=correction,
        log_exposure_correction=log_exposure_correction,
    )
    return updated_profile

def balance_print_sensitivity(profile,
                              target_film,
                              reference_cc_filter_values = DEFAULT_NEUTRAL_PRINT_FILTERS, # cmy in cc units
                              ):
    data = profile.data
    info = profile.info
    log_sensitivity = data.log_sensitivity
    
    sensitivity = 10 ** log_sensitivity
    
    film_raw_profile = load_raw_profile(target_film)
    film_midscale_neutral_density = film_raw_profile.data.midscale_neutral_density
    transmittance_midscale_neutral = 10 ** (-film_midscale_neutral_density)
    
    illuminant = standard_illuminant(type=info.reference_illuminant)
    filtered_illuminant = color_enlarger(illuminant, filter_cc_values=reference_cc_filter_values)
    filtered_illuminant *= transmittance_midscale_neutral
    
    neutral_exposures = np.nansum(filtered_illuminant[:, None] * sensitivity, axis=0)
    
    # under the filtered illuminant the log exposure will be [0,0,0] by design
    correction = 1 / neutral_exposures
    log_exposure_correction = np.log10(correction)

    sensitivity *= correction
    updated_log_sensitivity = np.log10(sensitivity)

    updated_profile = profile.update_data(log_sensitivity=updated_log_sensitivity)
    log_event(
        'balance_print_sensitivity',
        updated_profile,
        sensitivity_correction=correction,
        log_exposure_correction=log_exposure_correction,
    )
    return updated_profile


def reconstruct_metameric_neutral(profile, midgray_value=0.184):
    info = profile.info
    data = profile.data
    illuminant = standard_illuminant(info.viewing_illuminant)
    channel_density = np.asarray(data.channel_density)
    base_density = np.asarray(data.base_density)

    def rgb_mid(mid_density):
        light = 10 ** (-mid_density) * illuminant[:]
        light[np.isnan(light)] = 0

        normalization = np.sum(illuminant * STANDARD_OBSERVER_CMFS[:, 1], axis=0)
        xyz = np.einsum('k,kl->l', light, STANDARD_OBSERVER_CMFS[:]) / normalization
        illuminant_xyz = np.einsum('k,kl->l', illuminant, STANDARD_OBSERVER_CMFS[:]) / normalization
        illuminant_xy = colour.XYZ_to_xy(illuminant_xyz)
        return colour.XYZ_to_RGB(xyz, RGB_COLOURSPACE_sRGB, apply_cctf_encoding=False, illuminant=illuminant_xy)

    def midscale_neutral(density_cmy):
        return np.sum(channel_density * density_cmy, axis=1) + base_density

    rgb_reference = np.ones(3) * midgray_value

    def residues(parameters):
        return rgb_reference - rgb_mid(midscale_neutral(density_cmy=parameters))

    fit = scipy.optimize.least_squares(residues, [1.0, 1.0, 1.0])
    fitted_density = fit.x
    mid = midscale_neutral(fitted_density)
    updated_profile = profile.update_info(fitted_cmy_midscale_neutral_density=np.array(fitted_density))
    updated_profile = updated_profile.update_data(midscale_neutral_density=mid)
    log_event(
        'reconstruct_metameric_neutral',
        updated_profile,
        fitted_density_cmy=fitted_density,
    )
    return updated_profile


def _valid_density_curve(log_exposure, source_density_curves, index):
    valid = np.isfinite(source_density_curves[:, index])
    valid_curve = source_density_curves[valid, index]
    valid_log_exposure = log_exposure[valid]
    return valid_log_exposure, valid_curve

def prelminary_neutral_shift(profile, per_channel_shift=False):
    data = profile.data
    info = profile.info
    source_density_curves = np.asarray(data.density_curves)
    density_curves = np.array(source_density_curves, copy=True)
    log_exposure = data.log_exposure
    midscale_neutral_density_minus_base = np.asarray(data.midscale_neutral_density) - np.asarray(data.base_density)
    midscale_neutral_transmittance = 10 ** (-midscale_neutral_density_minus_base)
    densitometer_sensitivity = load_densitometer_data(info.densitometer)

    status_density_midscale_neutral = -np.log10(
        np.nansum(densitometer_sensitivity * midscale_neutral_transmittance[:, None], axis=0)
        / np.nansum(densitometer_sensitivity, axis=0)
    )
    log_exposure_correction = np.zeros(3)
    interp_sign = -1 if info.is_positive else 1
    for index in range(3):
        valid_log_exposure, valid_curve = _valid_density_curve(log_exposure, source_density_curves, index)
        log_exposure_correction[index] = np.interp(
            interp_sign * status_density_midscale_neutral[index],
            interp_sign * valid_curve,
            valid_log_exposure,
        )
    
    if per_channel_shift:
        pass
    else: # correct log_exposure globally
        log_exposure_correction = np.nanmean(log_exposure_correction)
        log_exposure_correction = np.full(3, log_exposure_correction)
    
    for index in range(3):
        valid_log_exposure, valid_curve = _valid_density_curve(log_exposure, source_density_curves, index)
        density_curves[:, index] = np.interp(
            log_exposure + log_exposure_correction[index],
            valid_log_exposure,
            valid_curve,
            left=np.nan,
            right=np.nan,
        )

    updated_profile = profile.update_data(density_curves=density_curves)
    log_event(
        'prelminary_neutral_shift',
        updated_profile,
        status_density_midscale_neutral=status_density_midscale_neutral,
        log_exposure_correction=log_exposure_correction,
    )
    
    return updated_profile

__all__ = ['balance_film_sensitivity', 'reconstruct_metameric_neutral', 'prelminary_neutral_shift']
