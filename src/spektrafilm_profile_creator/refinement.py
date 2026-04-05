import copy

import colour
from colour.models import RGB_COLOURSPACE_sRGB
import numpy as np
import scipy

from spektrafilm.config import STANDARD_OBSERVER_CMFS
from spektrafilm.model.color_filters import color_enlarger
from spektrafilm.model.illuminants import standard_illuminant
from spektrafilm.profiles.io import load_profile
from spektrafilm.runtime.api import simulate
from spektrafilm.runtime.params_schema import RuntimePhotoParams
from spektrafilm_profile_creator.core.density_curves import replace_fitted_density_curves
from spektrafilm_profile_creator.core.profile_transforms import apply_scale_shift_stretch_density_curves
from spektrafilm_profile_creator.diagnostics.messages import log_event
from spektrafilm_profile_creator.neutral_print_filters import fit_neutral_print_filters, DEFAULT_NEUTRAL_PRINT_FILTERS
from spektrafilm_profile_creator.data.loader import load_raw_profile



def _build_runtime_params(film_profile, print_profile):
    params = RuntimePhotoParams(
        film=film_profile.clone(),
        print=load_profile(print_profile),
    )
    params.enlarger.normalize_print_exposure = False
    return params


def refine_negative_curves_with_gray_ramp(
    source_profile,
    target_print,
    data_trustability=0.5,
    stretch_curves=False,
    ev_ramp=(-1, 0, 1, 2, 3, 4),
):
    
    params = _build_runtime_params(source_profile, target_print)
    params.film = replace_fitted_density_curves(params.film) # temporary replace with fitted to make couplers work in the extapolated range
    params.io.full_image = True
    params.camera.auto_exposure = False
    params.enlarger.print_exposure_compensation = True
    params.settings.rgb_to_raw_method = 'hanatos2025'
    fitted_y, fitted_m, _ = fit_neutral_print_filters(params, stock=source_profile.info.stock)
    params.enlarger.y_filter_neutral = fitted_y
    params.enlarger.m_filter_neutral = fitted_m

    density_scale, shift_correction, stretch_correction = fit_corrections_from_grey_ramp_negative(
        params,
        ev_ramp,
        data_trustability,
        stretch_curves,
    )
    corrected_profile = apply_scale_shift_stretch_density_curves(
        source_profile,
        density_scale,
        shift_correction,
        stretch_correction,
    )
    log_event(
        'correct_negative_curves_with_gray_ramp',
        corrected_profile,
        density_scale_correction=density_scale,
        shift_correction=shift_correction,
        stretch_correction=stretch_correction,
    )
    return corrected_profile


def fit_corrections_from_grey_ramp_negative(
    params,
    ev_ramp,
    data_trustability=1.0,
    stretch_curves=False,
):
    def residues(values):
        if stretch_curves:
            gray, reference = gray_ramp(
                params,
                ev_ramp,
                density_scale=values[0:3],
                shift_correction=(values[3], 0, values[4]),
                stretch_correction=(values[5], 1.0, values[6]),
            )
        else:
            gray, reference = gray_ramp(
                params,
                ev_ramp,
                density_scale=values[0:3],
                shift_correction=(values[3], 0, values[4]),
            )
            
        # gray_mean = np.mean(gray, axis=1).flatten()
        # gray_reference = gray_mean[:, None] * np.ones((1, 3))
        # zero_index = np.where(np.asarray(ev_ramp) == 0)[0]
        # if zero_index.size:
        #     gray_reference[zero_index] = reference.flatten()
        # log_event('fit_corrections_from_grey_ramp_reference', gray_reference=gray_reference)
        # residual = gray - gray_reference
        # residual = residual / gray_reference * 0.184

        residual = gray - reference

        residual = residual.flatten()

        bias_scale = 0.25 * (np.array(values[0:3]) - 1) * len(ev_ramp)
        if stretch_curves:
            bias_stretch = 1.0 * (np.array(values[5:8]) - 1) * len(ev_ramp)
            bias = np.concatenate((bias_scale, bias_stretch))
        else:
            bias = bias_scale

        return np.concatenate((residual, bias * data_trustability))

    x0 = [1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0] if stretch_curves else [1.0, 1.0, 1.0, 0.0, 0.0]
    fit = scipy.optimize.least_squares(residues, x0)
    density_scale = fit.x[0:3]
    shift_correction = (fit.x[3], 0, fit.x[4])
    stretch_correction = (fit.x[5], 1.0, fit.x[6]) if stretch_curves else (1.0, 1.0, 1.0)
    return density_scale, shift_correction, stretch_correction

###########################################################################################################


def refine_positive_curves_with_gray_ramp(
    positive_film_profile,
    data_trustability,
    stretch_curves=False,
    ev_ramp=(-2, -1, 0, 1, 2),
):
    params = _build_runtime_params(positive_film_profile, 'kodak_portra_endura')
    params.film = replace_fitted_density_curves(params.film) # temporary replace with fitted to make couplers work in the extapolated range
    params.io.scan_film = True
    params.io.full_image = True
    params.settings.rgb_to_raw_method = 'hanatos2025'

    density_scale, shift_correction, stretch_correction = fit_corrections_from_grey_ramp_positive(
        params,
        ev_ramp,
        data_trustability,
        stretch_curves,
        positive_film=True,
    )
    corrected_profile = apply_scale_shift_stretch_density_curves(
        positive_film_profile,
        density_scale,
        shift_correction,
        stretch_correction,
    )
    log_event(
        'correct_positive_curves_with_gray_ramp',
        corrected_profile,
        density_scale_correction=density_scale,
        shift_correction=shift_correction,
        stretch_correction=stretch_correction,
    )
    return corrected_profile

def fit_corrections_from_grey_ramp_positive(
    params,
    ev_ramp,
    data_trustability=1.0,
    stretch_curves=False,
    positive_film=False,
):
    def residues(values):
        if stretch_curves:
            gray, reference = gray_ramp(
                params,
                ev_ramp,
                density_scale=values[0:3],
                shift_correction=values[3:6],
                stretch_correction=values[6:9],
            )
        else:
            gray, reference = gray_ramp(
                params,
                ev_ramp,
                density_scale=values[0:3],
                shift_correction=values[3:6],
            )
            
        gray_mean = np.mean(gray, axis=1).flatten()
        gray_reference = gray_mean[:, None] * np.ones((1, 3))
        zero_index = np.where(np.asarray(ev_ramp) == 0)[0]
        if zero_index.size:
            gray_reference[zero_index] = reference.flatten()
        log_event('fit_corrections_from_grey_ramp_reference', gray_reference=gray_reference)
        residual = gray - gray_reference
        residual = residual / gray_reference * 0.184

        residual = residual.flatten()

        bias_scale = 0.25 * (np.array(values[0:3]) - 1) * len(ev_ramp)
        if stretch_curves:
            bias_stretch = 100.0 * (np.array(values[6:9]) - 1)
            bias = np.concatenate((bias_scale, bias_stretch))
        else:
            bias = bias_scale

        return np.concatenate((residual, bias * data_trustability))

    x0 = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0] if stretch_curves else [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    fit = scipy.optimize.least_squares(residues, x0)
    density_scale = fit.x[0:3]
    shift_correction = fit.x[3:6]
    stretch_correction = fit.x[6:9] if stretch_curves else [1, 1, 1]
    return density_scale, shift_correction, stretch_correction


##################################################################################################


def gray_ramp(
    params,
    ev_ramp,
    density_scale=(1, 1, 1),
    shift_correction=(0, 0, 0),
    stretch_correction=(1, 1, 1),
):
    working_params = copy.deepcopy(params)
    working_params.io.input_cctf_decoding = False
    working_params.io.input_color_space = 'sRGB'
    working_params.debug.deactivate_spatial_effects = True
    working_params.debug.deactivate_stochastic_effects = True
    working_params.print_render.glare.active = False
    working_params.io.output_cctf_encoding = False
    working_params.io.full_image = True
    working_params.film = apply_scale_shift_stretch_density_curves(
        working_params.film,
        density_scale,
        shift_correction,
        stretch_correction,
    )
    midgray_rgb = np.array([[[0.184, 0.184, 0.184]]])
    gray = np.zeros((np.size(ev_ramp), 3))
    for index in np.arange(np.size(ev_ramp)):
        working_params.camera.exposure_compensation_ev = ev_ramp[index]
        gray[index] = simulate(midgray_rgb, working_params).flatten()
    log_event('gray_ramp', gray=gray)
    return gray, midgray_rgb


######################################################################################
# Printing media

def refine_negative_print_profile_with_neutral_ramp(profile,
                                                    target_film,
                                                    data_trustability=1.0,
                                                    exposure_ev_ramp=(-0.8, -0.4, -0.2, -0.1, 0, 0.05, 0.15, 0.3),
                                                    reference_cc_filter_values=DEFAULT_NEUTRAL_PRINT_FILTERS):
    # TODO exposure_ev_ramp should be evenly spaced in the final rgb colorspace
    
    # get profile data and info
    data = profile.data
    info = profile.info
    log_sensitivity = data.log_sensitivity
    density_curves = data.density_curves
    log_exposure = data.log_exposure
    viewing_illuminant = info.viewing_illuminant
    reference_illuminant = info.reference_illuminant
    channel_density = data.channel_density
    base_density = data.base_density
    
    # preliminary computations
    print_exposures = 2 ** np.array(exposure_ev_ramp, dtype=np.float64)
    sensitivity = 10 ** log_sensitivity
    
    # get trasmittance of reference film midscale neutral  
    film_raw_profile = load_raw_profile(target_film)
    film_midscale_neutral_density = film_raw_profile.data.midscale_neutral_density
    transmittance_midscale_neutral = 10 ** (-film_midscale_neutral_density)
    
    # get the filtered illuminant
    reference_illuminant = standard_illuminant(type=reference_illuminant)
    filtered_illuminant = color_enlarger(reference_illuminant, filter_cc_values=reference_cc_filter_values)
    filtered_illuminant *= transmittance_midscale_neutral
    viewing_illuminant = standard_illuminant(type=viewing_illuminant)
    
    
    def rgb_print(print_exposure, density_curves):
        light_from_film = print_exposure * filtered_illuminant
        light_from_film[np.isnan(light_from_film)] = 0
        
        neutral_exposures = np.nansum(light_from_film[:, None] * sensitivity, axis=0)
        log_raw = np.log10(neutral_exposures)
        
        density_cmy = np.zeros((3,))
        for i in range(3):
            density_cmy[i] = np.interp(log_raw[i], log_exposure, density_curves[:,i])
        
        spectral_density = np.nansum(channel_density * density_cmy, axis=1) + base_density
        light_from_print = viewing_illuminant * 10 ** (-spectral_density)
        
        normalization = np.sum(viewing_illuminant * STANDARD_OBSERVER_CMFS[:, 1], axis=0)
        xyz = np.einsum('k,kl->l', light_from_print, STANDARD_OBSERVER_CMFS[:]) / normalization
        illuminant_xyz = np.einsum('k,kl->l', viewing_illuminant, STANDARD_OBSERVER_CMFS[:]) / normalization
        illuminant_xy = colour.XYZ_to_xy(illuminant_xyz)
        rgb = colour.XYZ_to_RGB(xyz, RGB_COLOURSPACE_sRGB, apply_cctf_encoding=False, illuminant=illuminant_xy)
        return rgb

    def compute_neutral_ramp(k, print_exposures):
        local_profile = profile.clone()
        local_density_curves = apply_scale_shift_stretch_density_curves(local_profile,
                                                                        density_scale=k[0:3],
                                                                        log_exposure_shift=k[3:6],
                                                                        ).data.density_curves
        
        neutral_rgb_print = np.zeros((len(print_exposures), 3))
        for i, print_exposure in enumerate(print_exposures):  
            neutral_rgb_print[i] = rgb_print(print_exposure, local_density_curves)
        return neutral_rgb_print

    def residues(k):
        gray = compute_neutral_ramp(k, print_exposures)
        log_event('fit_corrections_from_grey_ramp_reference', gray_ramp=gray)
        
        gray_mean = np.mean(gray, axis=1).flatten()
        gray_reference = gray_mean[:, None] * np.ones((1, 3))
        zero_index = np.where(np.asarray(print_exposures) == 1)[0]
        gray_reference[zero_index] = [0.184, 0.184, 0.184]
        residual = gray - gray_reference
        residual = residual / gray_reference * 0.184
        
        residual = residual.flatten()
        bias = 0.05 * (np.array(k[0:3]) - 1) * len(print_exposures)
        residual = np.concatenate((residual, bias * data_trustability))
        
        return residual.flatten()
    
    k0 = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    fit = scipy.optimize.least_squares(residues, k0)
    
    density_curves = apply_scale_shift_stretch_density_curves(profile.clone(),
                                                              density_scale=fit.x[0:3],
                                                              log_exposure_shift=fit.x[3:6],
                                                              ).data.density_curves
    
    updated_profile = profile.update_data(density_curves=density_curves)
    log_event(
        'refine_negative_print_profile_with_neutral_ramp',
        updated_profile,
        log_exposure_shift_correction=(fit.x[0], 0, fit.x[1]),
        log_exposure_stretch_correction=(fit.x[2], 0, fit.x[3]),
    )
    return updated_profile

__all__ = [
    'fit_neutral_print_filters',
    'refine_negative_curves_with_gray_ramp',
    'refine_positive_curves_with_gray_ramp',
    'refine_negative_print_profile_with_neutral_ramp',
]
