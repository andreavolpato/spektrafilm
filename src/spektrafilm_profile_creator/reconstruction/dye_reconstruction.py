import matplotlib.pyplot as plt
import lmfit
import numpy as np

from spektrafilm.model.color_filters import dichroic_filters
from spektrafilm.model.illuminants import standard_illuminant
from spektrafilm.utils.measure import measure_slopes_at_exposure
from spektrafilm_profile_creator.core.densitometer import compute_densitometer_crosstalk_matrix
from spektrafilm_profile_creator.data.loader import load_densitometer_data
from spektrafilm_profile_creator.messages import log_event, log_parameters
from spektrafilm_profile_creator.reconstruction.spectral_primitives import (
    gaussian_profiles,
    high_pass_filter,
    high_pass_gaussian,
    low_pass_filter,
    low_pass_gaussian,
    shift_stretch_cmy,
)


def make_reconstruct_dye_density_params(model='model_a'):
    params = lmfit.Parameters()
    params.add('dye_amp0', value=1.0, min=0.5, max=1.5)
    params.add('dye_amp1', value=1.0, min=0.5, max=1.5)
    params.add('dye_amp2', value=1.0, min=0.5, max=1.5)
    params.add('dye_width0', value=1.0, min=0.9, max=1.1)
    params.add('dye_width1', value=1.0, min=0.9, max=1.1)
    params.add('dye_width2', value=1.0, min=0.9, max=1.1)
    params.add('dye_shift0', value=0.0, min=-30, max=30)
    params.add('dye_shift1', value=0.0, min=-30, max=30)
    params.add('dye_shift2', value=0.0, min=-30, max=30)

    if model[:6] == 'filter':
        params.add('lp_c_width', value=20, min=12, max=30)
        params.add('hp_m_width', value=20, min=12, max=30)
        params.add('lp_y_width', value=20, min=12, max=30)
        params.add('lp_m_width', value=20, min=12, max=30)
        params.add('hp_c_width', value=20, min=12, max=30)
        params.add('lp_c_wl', value=420, min=400, max=440)
        params.add('hp_m_wl', value=500, min=460, max=500)
        params.add('lp_y_wl', value=500, min=490, max=530)
        params.add('lp_m_wl', value=600, min=590, max=650)
        params.add('hp_c_wl', value=600, min=570, max=600)

    if model == 'filters_neg':
        params.add('high_y', value=-0.02, min=-0.1, max=0.0)
        params.add('low_m', value=-0.02, min=-0.1, max=0.0)
        params.add('high_m', value=-0.02, min=-0.1, max=0.0)
        params.add('low_c', value=-0.02, min=-0.1, max=0.0)
        params.add('low_c_y', value=-0.02, min=-0.1, max=0.0)

    if model[:10] == 'filteramps':
        params.add('lp_c_amp', value=0.8, min=0.0, max=1.0)
        params.add('hp_m_amp', value=0.8, min=0.0, max=1.0)
        params.add('lp_y_amp', value=0.8, min=0.0, max=1.0)
        params.add('lp_m_amp', value=0.8, min=0.0, max=1.0)
        params.add('hp_c_amp', value=0.8, min=0.0, max=1.0)

    if model == 'dmid_dmin':
        params.add('cpl_amp0', value=0.1, min=0.05, max=0.5)
        params.add('cpl_amp1', value=0.1, min=0.05, max=0.5)
        params.add('cpl_amp2', value=0.1, min=0.05, max=0.5)
        params.add('cpl_amp3', value=0.03, min=0.0, max=0.5)
        params.add('cpl_amp4', value=0.1, min=0.05, max=0.5)
        params.add('cpl_width0', value=20, min=10, max=40)
        params.add('cpl_width1', value=20, min=10, max=40)
        params.add('cpl_width2', value=20, min=10, max=40)
        params.add('cpl_width3', value=40, min=10, max=50)
        params.add('cpl_width4', value=20, min=10, max=40)
        params.add('cpl_max0', value=435, min=420, max=450)
        params.add('cpl_max1', value=560, min=540, max=600)
        params.add('cpl_max2', value=475, min=455, max=490)
        params.add('cpl_max3', value=700, min=650, max=720)
        params.add('cpl_max4', value=510, min=495, max=535)
        params.add('dmax1', value=2.3, min=2.0, max=5)
        params.add('dmax0', value=2.3, min=2.0, max=5)
        params.add('dmax2', value=2.3, min=2.0, max=5)
        params.add('fog0', value=0.07, min=0.05, max=0.10, vary=True)
        params.add('fog1', value=0.07, min=0.05, max=0.10, vary=True)
        params.add('fog2', value=0.07, min=0.05, max=0.10, vary=True)
        params.add('scat400', value=0.65, min=0.62, max=1.0)
        params.add('base', value=0.05, min=0, max=0.15)
    return params


def density_mid_min_model(params, wavelengths, cmy_model, model):
    density_min = np.zeros_like(wavelengths)
    filters = np.zeros_like(cmy_model)
    dye = shift_stretch_cmy(
        wavelengths,
        cmy_model,
        params['dye_amp0'],
        params['dye_width0'],
        params['dye_shift0'],
        params['dye_amp1'],
        params['dye_width1'],
        params['dye_shift1'],
        params['dye_amp2'],
        params['dye_width2'],
        params['dye_shift2'],
    )
    if model[:7] == 'filters':
        high_pass_magenta = high_pass_filter(wavelengths, params['hp_m_wl'], params['hp_m_width'])
        low_pass_yellow = low_pass_filter(wavelengths, params['lp_y_wl'], params['lp_y_width'])
        low_pass_magenta = low_pass_filter(wavelengths, params['lp_m_wl'], params['lp_m_width'])
        high_pass_cyan = high_pass_filter(wavelengths, params['hp_c_wl'], params['hp_c_width'])
        low_pass_cyan = low_pass_filter(wavelengths, params['lp_c_wl'], params['lp_c_width'])
        filters = np.stack((1 - (1 - high_pass_cyan) * (1 - low_pass_cyan), high_pass_magenta * low_pass_magenta, low_pass_yellow), axis=1)
    if model == 'filters':
        cmy = dye * filters
    if model == 'filters_neg':
        cmy = dye * filters
        cmy[:, 2] += ((1 - low_pass_yellow) * (1 - high_pass_cyan)) * 6 * params['high_y']
        cmy[:, 1] += (1 - low_pass_magenta) * params['low_m']
        cmy[:, 1] += (1 - high_pass_magenta) * params['high_m']
        cmy[:, 0] += ((1 - low_pass_yellow) * (1 - high_pass_cyan)) ** 6 * params['low_c']
        cmy[:, 0] += ((1 - high_pass_magenta) * (1 - low_pass_cyan)) ** 6 * params['low_c_y']

    if model[:10] == 'filteramps':
        high_pass_magenta = high_pass_filter(wavelengths, params['hp_m_wl'], params['hp_m_width'], params['hp_m_amp'])
        low_pass_yellow = low_pass_filter(wavelengths, params['lp_y_wl'], params['lp_y_width'], params['lp_y_amp'])
        low_pass_magenta = low_pass_filter(wavelengths, params['lp_m_wl'], params['lp_m_width'], params['lp_m_amp'])
        high_pass_cyan = high_pass_filter(wavelengths, params['hp_c_wl'], params['hp_c_width'], params['hp_c_amp'])
        low_pass_cyan = low_pass_filter(wavelengths, params['lp_c_wl'], params['lp_c_width'], params['lp_c_amp'])
        filters = np.stack((1 - (1 - high_pass_cyan) * (1 - low_pass_cyan), high_pass_magenta * low_pass_magenta, low_pass_yellow), axis=1)
    if model == 'filteramps':
        cmy = dye * filters
    if model == 'filteramps_gauss':
        high_pass_cyan_gauss = high_pass_gaussian(wavelengths, params['hp_c_wl'], params['hp_c_width'], params['hp_c_amp'])
        low_pass_cyan_gauss = low_pass_gaussian(wavelengths, params['lp_c_wl'], params['lp_c_width'], params['lp_c_amp'])
        high_pass_magenta_gauss = high_pass_gaussian(wavelengths, params['hp_m_wl'], params['hp_m_width'], params['hp_m_amp'])
        low_pass_magenta_gauss = low_pass_gaussian(wavelengths, params['lp_m_wl'], params['lp_m_width'], params['lp_m_amp'])
        low_pass_yellow_gauss = low_pass_gaussian(wavelengths, params['lp_y_wl'], params['lp_y_width'], params['lp_y_amp'])
        gaussians = np.stack(
            (
                high_pass_cyan_gauss + low_pass_cyan_gauss,
                high_pass_magenta_gauss + low_pass_magenta_gauss,
                low_pass_yellow_gauss,
            ),
            axis=1,
        )
        cmy = dye * filters - gaussians

    if model == 'dmid_dmin':
        channels_couplers_gaussians = [0, 0, 1, 1, 2]
        coupler_parameters = [
            [params['cpl_amp0'], params['cpl_width0'], params['cpl_max0']],
            [params['cpl_amp1'], params['cpl_width1'], params['cpl_max1']],
            [params['cpl_amp2'], params['cpl_width2'], params['cpl_max2']],
            [params['cpl_amp3'], params['cpl_width3'], params['cpl_max3']],
            [params['cpl_amp4'], params['cpl_width4'], params['cpl_max4']],
        ]
        couplers = gaussian_profiles(wavelengths, coupler_parameters)
        couplers_cmy = np.zeros((np.size(wavelengths), 3))
        for index in range(5):
            couplers_cmy[:, channels_couplers_gaussians[index]] += couplers[:, index]
        cmy = dye - couplers_cmy
        density_min, _, _, _, _, _ = density_min_components(
            params,
            wavelengths,
            cmy,
            couplers,
            channels_couplers_gaussians,
        )
        filters = couplers

    return cmy, dye, filters, density_min


def density_min_components(params, wavelengths, cmy, couplers, channels_couplers_gaussians):
    dye_amplitudes = [params['dye_amp0'], params['dye_amp1'], params['dye_amp2']]
    density_max = np.array([params['dmax0'], params['dmax1'], params['dmax2']])
    fog = np.array([params['fog0'], params['fog1'], params['fog2']])
    base = np.ones_like(wavelengths) * params['base']
    scattering = -np.log10(1 - params['scat400'] * 400 ** 4 / wavelengths ** 4)
    couplers_cmy = np.zeros_like(cmy)
    for index in range(5):
        couplers_cmy[:, channels_couplers_gaussians[index]] += (
            couplers[:, index]
            / dye_amplitudes[channels_couplers_gaussians[index]]
            * density_max[channels_couplers_gaussians[index]]
        )
    density_min = np.sum(couplers_cmy + fog * cmy, axis=1) + scattering + base
    return density_min, couplers_cmy, scattering, fog, base, density_max


def slopes_of_concentrations(log_exposure, density_curves, densitometer_crosstalk_matrix):
    concentrations = np.zeros_like(density_curves)
    inverse_crosstalk = np.linalg.inv(densitometer_crosstalk_matrix)
    for density_index in range(3):
        for channel_index in range(3):
            concentrations[:, density_index] += inverse_crosstalk[density_index, channel_index] * density_curves[:, channel_index]
    return measure_slopes_at_exposure(log_exposure, concentrations)


def residual_simple(
    params,
    wavelengths,
    cmy_model,
    data,
    densitometer_responsivity,
    paper_sensitivity,
    log_exposure,
    density_curves,
    model='model_a',
    biases=(1, 2, 2),
):
    cmy, _, _, density_min_sim = density_mid_min_model(params, wavelengths, cmy_model, model)

    paper_crosstalk = compute_densitometer_crosstalk_matrix(paper_sensitivity, cmy)
    out_of_diagonal_crosstalk = paper_crosstalk.flatten()[[1, 2, 3, 5, 6, 7]]
    densitometer_crosstalk = compute_densitometer_crosstalk_matrix(densitometer_responsivity, cmy)
    gammas = slopes_of_concentrations(log_exposure, density_curves, densitometer_crosstalk)
    gamma_residual = gammas - np.mean(gammas)

    mid_minus_min_sim = np.sum(cmy, axis=1)
    simulated = np.concatenate((mid_minus_min_sim, biases[0] * density_min_sim))
    residual = data - simulated
    return np.concatenate((residual, biases[1] * out_of_diagonal_crosstalk, biases[2] * gamma_residual))


def reconstruct_dye_density(
    profile,
    params=None,
    control_plot=False,
    print_params=False,
    target_print_paper=None,
    ymc_filter_values=(0.8, 0.6, 0.2),
    max_nfev=500,
    tol=5e-5,
    model='dmid_dmin',
    biases=(1, 2, 0),
):
    data = profile.data
    info = profile.info
    cmy_model = np.asarray(data.channel_density)
    wavelengths = data.wavelengths
    log_exposure = data.log_exposure
    density_curves = data.density_curves
    midscale_density = np.asarray(data.midscale_neutral_density)
    base_density = np.asarray(data.base_density)
    mid_minus_min = midscale_density - base_density
    experimental_data = np.concatenate((mid_minus_min, base_density))

    densitometer_responsivity = load_densitometer_data(info.densitometer)
    if target_print_paper is not None:
        paper_sensitivity = 10 ** target_print_paper.data.log_sensitivity
        illuminant = standard_illuminant('BB3200')
        filtered_illuminant = dichroic_filters.apply(illuminant, values=ymc_filter_values)
        paper_sensitivity = paper_sensitivity * filtered_illuminant[:, None]

        plt.figure()
        plt.plot(wavelengths, paper_sensitivity)
    else:
        paper_sensitivity = densitometer_responsivity

    if params is None:
        params = make_reconstruct_dye_density_params(model)

    fit = lmfit.minimize(
        residual_simple,
        params,
        args=(
            wavelengths,
            cmy_model,
            experimental_data,
            densitometer_responsivity,
            paper_sensitivity,
            log_exposure,
            density_curves,
            model,
            biases,
        ),
        nan_policy='omit',
        method='least_squares',
        ftol=tol,
        xtol=tol,
        gtol=tol,
        max_nfev=max_nfev,
    )

    cmy, cmy_without_filters, filters, density_min_sim = density_mid_min_model(
        fit.params,
        wavelengths,
        cmy_model,
        model,
    )

    densitometer_crosstalk = compute_densitometer_crosstalk_matrix(densitometer_responsivity, cmy)
    slopes = slopes_of_concentrations(log_exposure, density_curves, densitometer_crosstalk)

    if print_params:
        log_parameters('reconstruct_dye_density parameters', fit.params)
        log_event(
            'reconstruct_dye_density summary',
            slopes_at_reference_exposure=slopes,
            densitometer_crosstalk_matrix=densitometer_crosstalk,
        )

    if control_plot:
        color = ['tab:cyan', 'tab:pink', 'gold']

        fig, axes = plt.subplots(1, 3)
        fig.set_tight_layout(tight='rect')
        fig.set_figheight(4)
        fig.set_figwidth(12)
        fig.suptitle(info.name)

        for index in range(3):
            axes[0].plot(wavelengths, cmy[:, index], color=color[index], label='CMY'[index])
        axes[0].plot(wavelengths, np.sum(cmy, axis=1), 'k--', label='Sim', alpha=0.5)
        axes[0].plot(wavelengths, mid_minus_min, 'k', label='Exp')
        axes[0].legend()
        axes[0].set_xlabel('Wavelength (nm)')
        axes[0].set_ylabel('Diffuse Density')
        axes[0].set_title('Midscale neutral minus minimum')

        for index in range(3):
            axes[1].plot(wavelengths, cmy_without_filters[:, index], '--', color=color[index], label='_nolegend_')
            axes[1].plot(wavelengths, cmy[:, index], color=color[index], alpha=1.0, label='CMY'[index])
        axes[1].text(
            0.5,
            0.02,
            'Dashed lines are without masking couplers.',
            transform=axes[1].transAxes,
            ha='center',
            va='bottom',
            fontsize=9,
            color='k',
        )
        axes[1].set_xlabel('Wavelength (nm)')
        axes[1].set_ylabel('Diffuse Density')
        axes[1].set_title('CMY and masking couplers')
        axes[1].legend()

        if model == 'dmid_dmin':
            density_min_sim, couplers, scattering, fog, base, _ = density_min_components(
                fit.params,
                wavelengths,
                cmy,
                filters,
                [0, 0, 1, 1, 2],
            )
            axes[2].plot(wavelengths, np.ones_like(wavelengths) * base, 'tab:green', label='Base')
            axes[2].plot(wavelengths, base + scattering, 'tab:blue', label='Scattering')
            axes[2].plot(wavelengths, base + scattering + np.sum(fog * cmy, axis=1) + couplers[:, 0], color='tab:cyan', label='Mask C')
            axes[2].plot(wavelengths, base + scattering + np.sum(fog * cmy, axis=1) + couplers[:, 1], color='tab:pink', label='Mask M')
            axes[2].plot(wavelengths, base + scattering + np.sum(fog * cmy, axis=1) + couplers[:, 2], color='gold', label='Mask Y')
            axes[2].plot(wavelengths, base + scattering + np.sum(fog * cmy, axis=1), color='tab:orange', label='Fog')
            axes[2].plot(wavelengths, density_min_sim, 'k--', label='Sim', alpha=0.5)
        else:
            for index in range(3):
                axes[2].plot(wavelengths, filters[:, index], color=color[index], label='CMY'[index])
        axes[2].plot(wavelengths, base_density, 'k', label='Exp')
        axes[2].set_xlabel('Wavelength (nm)')
        axes[2].set_ylabel('Diffuse Density')
        axes[2].set_title('Minimum density')
        axes[2].set_ylim([0, np.nanmax(base_density) * 1.05])
        axes[2].legend()

    return profile.update(
        info={
            'fitted_cmy_midscale_neutral_density': np.nanmax(cmy, axis=0).tolist(),
        },
        data={
            'channel_density': cmy / np.nanmax(cmy, axis=0),
        },
    )


__all__ = [
    'make_reconstruct_dye_density_params',
    'density_mid_min_model',
    'density_min_components',
    'reconstruct_dye_density',
]
