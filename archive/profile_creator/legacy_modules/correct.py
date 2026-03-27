import numpy as np
import scipy
import copy
from spektrafilm.runtime.api import create_params, simulate
from spektrafilm_profile_creator.fitting import fit_print_filters


def measure_log_exposure_midscale_neutral(profile, reference_channel=None):
    log_exposure_midscale_neutral = np.zeros((3,))
    d_mid = profile.info.fitted_cmy_midscale_neutral_density
    if np.size(d_mid)==1: 
        d_mid = np.ones(3) * d_mid
    if reference_channel=='green':
        d_mid = np.ones(3) * d_mid[1]
    for i in range(3):
        if profile.info.is_positive:
            log_exposure_midscale_neutral[i] = np.interp(-d_mid[i], -profile.data.density_curves[:,i], profile.data.log_exposure)
        else:
            log_exposure_midscale_neutral[i] = np.interp(d_mid[i], profile.data.density_curves[:,i], profile.data.log_exposure)
    print('Log exposure midscale neutral:', log_exposure_midscale_neutral)
    return log_exposure_midscale_neutral

def align_midscale_neutral_exposures(profile, reference_channel=None):
    log_exposure_midscale_neutral = measure_log_exposure_midscale_neutral(profile, reference_channel)
    dc = profile.data.density_curves
    le = profile.data.log_exposure
    for i in np.arange(3):
        dc[:, i] = np.interp(le, le - log_exposure_midscale_neutral[i], dc[:, i])
    profile.data.density_curves = dc
    profile.info.log_exposure_midscale_neutral = (np.ones(3) * log_exposure_midscale_neutral[1]).tolist()
    return profile

############################################################################################################
# fit gray strip

def correct_negative_curves_with_gray_ramp(source_profile,
                                           target_paper='kodak_portra_endura_uc',
                                           data_trustability=0.5,
                                           stretch_curves=False,
                                           ev_ramp=(-2, -1, 0, 1, 2, 3, 4, 5, 6)
                                           ):
    pl = create_params(print_profile=target_paper, ymc_filters_from_database=False)
    pl.film = copy.deepcopy(source_profile)
    pl.io.full_image = True
    pl.settings.rgb_to_raw_method = 'mallett2019'
    fit_print_filters(pl)
    
    density_scale, shift_corr, stretch_corr = fit_corrections_from_grey_ramp(pl, ev_ramp, data_trustability, stretch_curves)
    print('Density scale corr:', density_scale)
    print('Shift corr:', shift_corr)
    print('Stretch corr:', stretch_corr)
    profile_corrected = apply_scale_shift_stretch_density_curves(pl.film, density_scale, shift_corr, stretch_corr)
    return profile_corrected

def correct_positive_curves_with_gray_ramp(positive_film_profile,
                                           data_trustability=0.5,
                                           stretch_curves=False,
                                           ev_ramp=(-2, -1, 0, 1),
                                           ):
    pl = create_params(ymc_filters_from_database=False)
    pl.film = copy.deepcopy(positive_film_profile)
    pl.io.scan_film = True
    pl.io.full_image = True
    pl.settings.rgb_to_raw_method = 'hanatos2025'
    
    density_scale, shift_corr, stretch_corr = fit_corrections_from_grey_ramp(pl, ev_ramp, data_trustability, stretch_curves, positive_film=True)
    print('--- Fit of density curves corrections')
    print('Density scale corr:', density_scale)
    print('Shift corr:', shift_corr)
    print('Stretch corr:', stretch_corr)
    profile_corrected = apply_scale_shift_stretch_density_curves(pl.film, density_scale, shift_corr, stretch_corr)
    return profile_corrected

def fit_corrections_from_grey_ramp(p0, ev_ramp, data_trustability=1.0, stretch_curves=False, positive_film=False):
    def residues(x):
        if stretch_curves:  gray, reference = gray_ramp(p0, ev_ramp, density_scale=x[0:3], shift_corr=[x[3],0,x[4]], stretch_corr=x[5:8])
        else:               gray, reference = gray_ramp(p0, ev_ramp, density_scale=x[0:3], shift_corr=[x[3],0,x[4]])
        if positive_film:
            gray_mean = np.mean(gray, axis=1).flatten()
            gray_ref = gray_mean[:, None] * np.ones((1, 3))
            zero_idx = np.where(np.asarray(ev_ramp) == 0)[0]
            if zero_idx.size:
                gray_ref[zero_idx] = reference.flatten()
            print(f'gray_ref = {gray_ref}')
            res = gray - gray_ref
            res = res/gray_ref*0.184
        else:
            res = np.array(gray) - reference
        res = res.flatten()
        
        bias_scale = 2.0*(np.array(x[0:3])-1)
        if stretch_curves:
            bias_stretch = 100.0*(np.array(x[6:9])-1)
            bias = np.concatenate((bias_scale, bias_stretch))
        else: bias = bias_scale
        
        res = np.concatenate((res, bias*data_trustability))
        return res
    if stretch_curves:  x0 = [1., 1., 1.,  0., 0.,  1., 1., 1.]
    else:               x0 = [1., 1., 1.,  0., 0.]
    fit = scipy.optimize.least_squares(residues, x0)
    density_scale = fit.x[0:3]
    shift_corr = [fit.x[3], 0, fit.x[4]]
    if stretch_curves: stretch_corr = fit.x[5:8]
    else:              stretch_corr = [1,1,1]
    return density_scale, shift_corr, stretch_corr

def gray_ramp(p0, ev_ramp, density_scale=(1, 1, 1), shift_corr=(0, 0, 0), stretch_corr=(1, 1, 1)):
    pl = copy.deepcopy(p0)
    pl.io.input_cctf_decoding = False
    pl.io.input_color_space = 'sRGB'
    pl.debug.deactivate_spatial_effects = True
    pl.debug.deactivate_stochastic_effects = True
    pl.print_render.glare.active = False
    pl.io.output_cctf_encoding = False
    pl.io.full_image = True
    pl.film = apply_scale_shift_stretch_density_curves(pl.film, density_scale, shift_corr, stretch_corr)
    midgray_rgb = np.array([[[0.184,0.184,0.184]]])
    gray = np.zeros((np.size(ev_ramp),3))
    for i in np.arange(np.size(ev_ramp)):
        pl.camera.exposure_compensation_ev = ev_ramp[i]
        gray[i] = simulate(midgray_rgb, pl).flatten()
    print(f'gray = {gray}')
    return gray, midgray_rgb

def apply_scale_shift_stretch_density_curves(profile, density_scale=(1, 1, 1), log_exposure_shift=(0, 0, 0), log_exposure_strech=(1, 1, 1)):
    dc = copy.copy(profile.data.density_curves)
    le = copy.copy(profile.data.log_exposure)
    dc = dc * density_scale
    for i in np.arange(3):
        dc[:,i] = np.interp(le, le/log_exposure_strech[i]+log_exposure_shift[i], dc[:,i])
    profile.data.density_curves = dc
    return profile
