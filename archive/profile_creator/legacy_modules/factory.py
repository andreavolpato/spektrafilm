import numpy as np
import scipy
import copy
import matplotlib.pyplot as plt

from spektrafilm_profile_creator.reconstruct import reconstruct_dye_density
from spektrafilm_profile_creator.balance import balance_sensitivity, balance_metameric_neutral
from spektrafilm_profile_creator.correct import align_midscale_neutral_exposures
from spektrafilm_profile_creator.data.loader import load_agx_emulsion_data, load_densitometer_data
from spektrafilm_profile_creator.fitting import compute_density_curves, compute_density_curves_layers, fit_density_curves
from spektrafilm_profile_creator.plotting import plot_profile

from spektrafilm.profiles.io import Profile, ProfileData, ProfileInfo
from spektrafilm.utils.measure import measure_density_min

def compute_densitometer_crosstalk_matrix(densitometer_intensity, dye_density):
    crosstalk_matrix = np.zeros((3,3))
    dye_transmittance = 10**(-dye_density[:,0:3])
    for i in np.arange(3):
        for j in np.arange(3):
            crosstalk_matrix[i,j] = -np.log10(
                np.nansum(densitometer_intensity[:,i]*dye_transmittance[:,j])
                / np.nansum(densitometer_intensity[:,i])
                )
    return crosstalk_matrix

def unmix_density_curves(curves, crosstalk_matrix):
    inverse_cm = np.linalg.inv(crosstalk_matrix)
    density_curves_raw = np.einsum('ij, kj-> ki',inverse_cm, curves)
    density_curves_raw = np.clip(density_curves_raw,0,None)
    return density_curves_raw

################################################################################
# Create profile
################################################################################

def create_profile(stock='kodak_portra_400',
                   profile_type='negative',
                   support='film',
                   channel_model='color',
                   name=None,
                   densitometer='status_M',
                   log_sensitivity_density_over_min=0.2,
                   log_sensitivity_donor=None,
                   denisty_curves_donor=None,
                   dye_density_cmy_donor=None,
                   dye_density_min_mid_donor=None,
                   reference_illuminant='D55-KG3',
                   viewing_illuminant='D50',
                   ):
    ls, d, wl, c, le = load_agx_emulsion_data(stock=stock,
                                              log_sensitivity_donor=log_sensitivity_donor,
                                              denisty_curves_donor=denisty_curves_donor,
                                              dye_density_cmy_donor=dye_density_cmy_donor,
                                              dye_density_min_mid_donor=dye_density_min_mid_donor,
                                              profile_type=profile_type,
                                              support=support,
                                              channel_model=channel_model,
                                              )
    print(stock,' - ',support, profile_type)

    channel_density = np.asarray(d[:, :3])
    base_density = np.asarray(d[:, 3])
    midscale_neutral_density = np.asarray(d[:, 4])
    return Profile(
        info=ProfileInfo(
            stock=stock,
            name=stock if name is None else name,
            type=profile_type,
            support=support,
            channel_model=channel_model,
            densitometer=densitometer,
            log_sensitivity_density_over_min=log_sensitivity_density_over_min,
            reference_illuminant=reference_illuminant,
            viewing_illuminant=viewing_illuminant,
        ),
        data=ProfileData(
            log_sensitivity=ls,
            wavelengths=wl,
            density_curves=c,
            log_exposure=le,
            channel_density=channel_density,
            base_density=base_density,
            midscale_neutral_density=midscale_neutral_density,
            density_curves_layers=np.array((0,3,3)),
        ),
    )

def remove_density_min(profile):
    le = profile.data.log_exposure
    dc = profile.data.density_curves
    base_density = np.asarray(profile.data.base_density)
    wl = profile.data.wavelengths
    profile_type = profile.info.type
    dc_min = measure_density_min(le, dc, profile_type)
    dc = dc - dc_min
    print('Density curve min values:', dc_min)
    if profile.info.is_paper or profile.info.is_positive:
        status_a_max_peak = [445, 530, 610]
        smin = np.interp(wl, status_a_max_peak, np.flip(dc_min))
        sigma = 20
        sigma_points = sigma / np.mean(np.diff(wl))
        smin = scipy.ndimage.gaussian_filter1d(smin, sigma_points)
        base_density = smin

    profile.data.base_density = np.asarray(base_density)
    profile.data.density_curves = dc
    return profile

def adjust_log_exposure(profile,
                        speed_point_density=0.2,
                        stops_over_speed_point=3,
                        midgray_transmittance=0.184
                        ):
    if profile.info.is_paper or profile.info.is_positive:
        speed_point_density = np.log10(1/midgray_transmittance)
        stops_over_speed_point = 0
    print('Reference density:', speed_point_density)
    print('Stops over reference density:', stops_over_speed_point, 'EV')
    dcg = profile.data.density_curves[:,1]
    dcg = dcg - np.nanmin(dcg)
    le = profile.data.log_exposure
    sel = ~np.isnan(dcg)
    if profile.info.is_positive:
        le_speed_point = np.interp(-speed_point_density, -dcg[sel], le[sel])
    else:
        le_speed_point = np.interp(speed_point_density, dcg[sel], le[sel])
    print('Log exposure refenrece:', le_speed_point)
    le_over_speed_point = np.log10(2**stops_over_speed_point)
    le_midgray = le_speed_point + le_over_speed_point
    profile.data.log_exposure = le - le_midgray
    return profile

def unmix_density(profile):
    dc = profile.data.density_curves
    channel_density = np.asarray(profile.data.channel_density)
    ds = load_densitometer_data(densitometer_type=profile.info.densitometer)
    densitometer_crosstalk_matrix = compute_densitometer_crosstalk_matrix(ds, channel_density)
    print('densitometer C: ')
    print(densitometer_crosstalk_matrix)
    dc = unmix_density_curves(dc, densitometer_crosstalk_matrix)
    profile.data.density_curves = dc
    return profile

def densitometer_normalization(profile):
    dc = profile.data.density_curves
    channel_density = np.asarray(profile.data.channel_density)
    dstm = load_densitometer_data(densitometer_type=profile.info.densitometer)
    M = compute_densitometer_crosstalk_matrix(dstm, channel_density)
    norm_coeffs = np.diag(M)
    print('Dye density densitometer normalization coefficients:', norm_coeffs)
    channel_density = channel_density / norm_coeffs
    dc = dc * norm_coeffs
    profile.data.channel_density = np.asarray(channel_density)
    profile.data.density_curves = dc
    return profile

def replace_fitted_density_curves(profile, control_plot=False):
    dc = profile.data.density_curves
    le = profile.data.log_exposure
    profile_type = profile.info.type
    support = profile.info.support
    density_curves_fitting_parameters = fit_density_curves(le, dc, profile_type=profile_type, support=support)
    print('density_curves_fitting_parameters: ', density_curves_fitting_parameters)
    density_curves_prefit = np.copy(dc)
    dc = compute_density_curves(le, density_curves_fitting_parameters, profile_type=profile_type, support=support)
    profile.data.density_curves = dc
    profile.data.density_curves_layers = compute_density_curves_layers(le, density_curves_fitting_parameters, profile_type=profile_type, support=support)
    if control_plot:
        plt.figure()
        plt.plot(le, dc)
        plt.plot(le, density_curves_prefit, color='gray', linestyle='--')
        plt.legend(('r','g','b'))
        plt.xlabel('Log Exposure')
        plt.ylabel('Density (over B+F)')
    return profile

def preprocess_profile(profile):
    profile = remove_density_min(profile)
    profile = adjust_log_exposure(profile)
    return profile

def process_negative_profile(raw_profile,
                    dye_density_reconstruct_model='dmid_dmin',
                    ):
    profile = copy.copy(raw_profile)
    profile = remove_density_min(profile)
    profile = adjust_log_exposure(profile)
    profile = reconstruct_dye_density(profile,
                                        control_plot=True,
                                        model=dye_density_reconstruct_model)
    profile = unmix_density(profile)
    profile = replace_fitted_density_curves(profile)
    profile = balance_sensitivity(profile)
    profile = replace_fitted_density_curves(profile)
    plot_profile(profile, unmixed=True, original=raw_profile)
    return profile

def process_paper_profile(raw_profile, align_midscale_exposures=False):
    profile = copy.copy(raw_profile)
    profile = remove_density_min(profile)
    profile = adjust_log_exposure(profile)
    profile = balance_metameric_neutral(profile)
    profile = unmix_density(profile)
    if align_midscale_exposures:
        profile = align_midscale_neutral_exposures(profile)
    profile = replace_fitted_density_curves(profile)
    plot_profile(profile, unmixed=True, original=raw_profile)
    return profile
