import colour
import numpy as np
import scipy

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
    correction = neutral_exposures[1] / neutral_exposures
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
                              reference_cc_filter_values = DEFAULT_NEUTRAL_PRINT_FILTERS, # cmy in cc units
                              reference_film='kodak_portra_400',
                              ):
    data = profile.data
    info = profile.info
    log_sensitivity = data.log_sensitivity
    
    sensitivity = 10 ** log_sensitivity
    
    film_raw_profile = load_raw_profile(reference_film)
    film_midscale_neutral_density = film_raw_profile.data.midscale_neutral_density
    transmittance_midscale_neutral = 10 ** (-film_midscale_neutral_density)
    
    illuminant = standard_illuminant(type=info.reference_illuminant)
    filtered_illuminant = color_enlarger(illuminant, filter_cc_values=reference_cc_filter_values)
    filtered_illuminant *= transmittance_midscale_neutral
    
    neutral_exposures = np.nansum(filtered_illuminant[:, None] * sensitivity, axis=0)
    correction = neutral_exposures[1] / neutral_exposures
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


def balance_channel_density_with_densitometer(profile):
    data = profile.data
    info = profile.info
    channel_density = data.channel_density
    densitometer_type = info.densitometer
    densitometer_data = load_densitometer_data(densitometer_type)
    normalization = np.nansum(densitometer_data*channel_density, axis=0)/np.nansum(densitometer_data, axis=0)
    updated_profile = profile.update_data(channel_density=channel_density/normalization)
    log_event('balance_channel_density_with_densitometer',
        updated_profile,
        normalization=normalization,
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
    updated_profile = profile.update(
        info={
            'fitted_cmy_midscale_neutral_density': fitted_density,
        },
        data={
            # 'channel_density': channel_density * density_scale,
            'midscale_neutral_density': mid,
            # 'density_curves': data.density_curves / density_scale,
        },
    )
    log_event(
        'reconstruct_metameric_neutral',
        updated_profile,
        fitted_density_cmy=fitted_density,
    )
    return updated_profile


__all__ = ['balance_film_sensitivity', 'reconstruct_metameric_neutral']
