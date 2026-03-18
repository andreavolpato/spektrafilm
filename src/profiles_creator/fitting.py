import copy

import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt

from spectral_film_lab.runtime.process import photo_params, photo_process


################################################################################
# Density curve fitting and synthesis
################################################################################

def density_curve_model_norm_cdfs(log_exposure,
                                  parameters=(
                                      0, 1, 2,
                                      0.5, 0.5, 0.5,
                                      0.3, 0.5, 0.7,
                                  ),
                                  profile_type='negative',
                                  support='film',
                                  number_of_layers=3):
    del support
    centers = parameters[0:3]
    amplitudes = parameters[3:6]
    sigmas = parameters[6:9]

    density_curve = np.zeros(log_exposure.shape)
    for i, (center, amplitude, sigma) in enumerate(zip(centers, amplitudes, sigmas)):
        if i <= number_of_layers - 1:
            if profile_type == 'positive':
                density_curve += scipy.stats.norm.cdf(-(log_exposure - center) / sigma) * amplitude
            else:
                density_curve += scipy.stats.norm.cdf((log_exposure - center) / sigma) * amplitude
    return density_curve


def distribution_model_norm_cdfs(log_exposure, parameters, number_of_layers=3):
    centers = parameters[0:3]
    amplitudes = parameters[3:6]
    sigmas = parameters[6:9]

    distribution = np.zeros((log_exposure.shape[0], 3))
    for i, (center, amplitude, sigma) in enumerate(zip(centers, amplitudes, sigmas)):
        if i <= number_of_layers - 1:
            distribution[:, i] += scipy.stats.norm.pdf((log_exposure - center) / sigma) * amplitude
    return distribution


def density_curve_layers(log_exposure, parameters, profile_type='negative', number_of_layers=3):
    n_layers = number_of_layers
    centers = parameters[0:n_layers]
    amplitudes = parameters[n_layers:2 * n_layers]
    sigmas = parameters[2 * n_layers:3 * n_layers]

    density_curve = np.zeros((log_exposure.shape[0], 3))
    for i, (center, amplitude, sigma) in enumerate(zip(centers, amplitudes, sigmas)):
        if i <= number_of_layers - 1:
            if profile_type == 'positive':
                density_curve[:, i] += scipy.stats.norm.cdf(-(log_exposure - center) / sigma) * amplitude
            else:
                density_curve[:, i] += scipy.stats.norm.cdf((log_exposure - center) / sigma) * amplitude
    return density_curve


def guess_start_and_bounds_norm_cdfs(log_exposure, data, profile_type, support='film'):
    del data
    if np.logical_or(profile_type == 'positive', support == 'paper'):
        x0 = [
            np.mean(log_exposure) - 0.25, np.mean(log_exposure), np.mean(log_exposure) + 0.25,
            0.5, 1, 0.5,
            0.2, 0.3, 0.5,
        ]
        x_lb = [
            np.min(log_exposure), np.min(log_exposure), np.min(log_exposure),
            0.25, 0.25, 0.25,
            0.05, 0.05, 0.05,
        ]
        x_ub = [
            np.max(log_exposure), np.max(log_exposure), np.max(log_exposure),
            5.0, 5.0, 5.0,
            1, 1, 1,
        ]
    else:
        density_max_layer = 1.35
        x0 = [
            np.mean(log_exposure) - 0.5, np.mean(log_exposure), np.mean(log_exposure) + 0.5,
            0.5, 0.5, 0.5,
            0.3, 0.6, 0.9,
        ]
        x_lb = [
            np.min(log_exposure), np.min(log_exposure), np.min(log_exposure),
            0.2, 0.2, 0.2,
            0.05, 0.05, 0.05,
        ]
        x_ub = [
            np.max(log_exposure), np.max(log_exposure), np.max(log_exposure),
            density_max_layer, density_max_layer, density_max_layer,
            2, 2, 2,
        ]
    return x0, (x_lb, x_ub)


def density_curve_model_log_line(log_exposure,
                                 parameters=(0.1, 1.5, -2.5, 2, 2, 2, 0.2, 0, 0.2, 0),
                                 profile_type='negative',
                                 support='film'):
    del support
    x = np.zeros(10)
    x[0:10] = parameters[0:10]
    density_min = x[0]
    gamma = x[1]
    log_exposure_reference = x[2]
    density_range = x[3]
    curvature_toe = x[4]
    curvature_shoulder = x[5]
    curvature_toe_slope = x[6]
    curvature_toe_max = x[7]
    curvature_shoulder_slope = x[8]
    curvature_shoulder_max = x[9]

    if profile_type == 'negative':
        log_exposure_zero = log_exposure_reference - 1.0 / gamma
    else:
        log_exposure_zero = log_exposure_reference - 0.735 / gamma

    if profile_type == 'positive':
        gamma = -gamma
        curvature_toe_slope = -curvature_toe_slope
        curvature_shoulder_slope = -curvature_shoulder_slope

    def sigmoid(x_value):
        return 1 / (1 + np.exp(-4 * gamma * x_value))

    shoulder_shape = gamma * curvature_shoulder * (
        1 + curvature_shoulder_max * sigmoid(-curvature_shoulder_slope * (log_exposure - log_exposure_zero - density_range / gamma))
    )
    toe_shape = gamma * curvature_toe * (
        1 + curvature_toe_max * sigmoid(curvature_toe_slope * (log_exposure - log_exposure_zero))
    )

    rise = gamma / toe_shape * np.log10(1 + 10 ** (toe_shape * (log_exposure - log_exposure_zero)))
    stop = gamma / shoulder_shape * np.log10(1 + 10 ** (shoulder_shape * (log_exposure - density_range / np.abs(gamma) - log_exposure_zero)))
    if profile_type == 'positive':
        return density_min - rise + stop
    return density_min + rise - stop


def guess_start_and_bounds_log_line(log_exposure, data, profile_type, support='film'):
    slope_scale = 2 if np.logical_or(profile_type == 'positive', support == 'paper') else 1
    x0 = [
        np.min(data),
        (np.max(data) - np.min(data)) / (np.max(log_exposure) - np.min(log_exposure)) * 2,
        np.mean(log_exposure),
        2.2,
        3,
        2,
        0, 0,
        0, 0,
    ]
    x_lb = [0, 0, np.min(log_exposure), 0, 0.5, 0.5, -4, 0, -4, 0]
    x_ub = [np.min(data) + 1, 5, np.max(log_exposure), 3.5, 16, 4, 8, 4 * slope_scale, 8, 4 * slope_scale]
    return x0, (x_lb, x_ub)


def compute_density_curves(log_exposure, parameters, profile_type, support='film', model='norm_cdfs'):
    density = np.zeros((np.size(log_exposure), 3))
    if model == 'norm_cdfs':
        model_function = density_curve_model_norm_cdfs
    elif model == 'log_line':
        model_function = density_curve_model_log_line
    else:
        raise ValueError(f'Unsupported density curve model: {model}')

    for i in np.arange(3):
        density[:, i] = model_function(log_exposure, parameters[i], profile_type=profile_type, support=support)
    return density


def compute_density_curves_layers(log_exposure, parameters, profile_type, support='film'):
    del support
    density = np.zeros((np.size(log_exposure), 3, 3))
    for i in np.arange(3):
        density[:, :, i] = density_curve_layers(log_exposure, parameters[i], profile_type=profile_type)
    return density


def fit_density_curve(log_exposure, data, profile_type='negative', support='film', model='norm_cdfs'):
    if model == 'norm_cdfs':
        model_function = density_curve_model_norm_cdfs
        guess_function = guess_start_and_bounds_norm_cdfs
    elif model == 'log_line':
        model_function = density_curve_model_log_line
        guess_function = guess_start_and_bounds_log_line
    else:
        raise ValueError(f'Unsupported density curve model: {model}')

    selection = ~np.isnan(data)
    log_exposure = log_exposure[selection]
    data = data[selection]
    x0, bounds = guess_function(log_exposure, data, profile_type, support=support)
    residues = lambda x: data - model_function(log_exposure, x, profile_type=profile_type, support=support)
    fit = scipy.optimize.least_squares(residues, x0, bounds=bounds)
    return fit.x


def fit_density_curves(log_exposure, density, profile_type='negative', support='film', model='norm_cdfs', plotting=False, stock='film_stock'):
    fitted_parameters = np.zeros((3, 9))
    for i in np.arange(3):
        fitted_parameters[i] = fit_density_curve(log_exposure, density[:, i], profile_type=profile_type, support=support, model=model)

    if plotting:
        if model == 'norm_cdfs':
            model_function = density_curve_model_norm_cdfs
        elif model == 'log_line':
            model_function = density_curve_model_log_line
        else:
            raise ValueError(f'Unsupported density curve model: {model}')
        _, ax = plt.subplots()
        colors = ['tab:red', 'tab:green', 'tab:blue']
        for i in np.arange(3):
            ax.plot(log_exposure, density[:, i], '.', color='k', label='_nolegend_')
            ax.plot(log_exposure, model_function(log_exposure, fitted_parameters[i], profile_type=profile_type, support=support), color=colors[i])
            if model == 'norm_cdfs':
                ax.plot(log_exposure, distribution_model_norm_cdfs(log_exposure, fitted_parameters[i]),
                        label='_nolegend_', color=colors[i], linewidth=1, linestyle='dashed')
        ax.set_xlabel('Log Exposure')
        ax.set_ylabel('Density')
        ax.set_title(stock + ' - ' + support + ' - ' + profile_type)
        ax.legend(('r', 'g', 'b'))
    return fitted_parameters


################################################################################
# Print filter fitting
################################################################################

def fit_print_filters_iter(profile):
    p = copy.copy(profile)
    p.debug.deactivate_spatial_effects = True
    p.debug.deactivate_stochastic_effects = True
    p.print_render.glare.compensation_removal_factor = 0.0
    p.io.input_cctf_decoding = False
    p.io.input_color_space = "sRGB"
    p.io.resize_factor = 1.0
    p.camera.auto_exposure = False
    p.enlarger.print_exposure_compensation = False
    midgray_rgb = np.array([[[0.184, 0.184, 0.184]]])
    c_filter = p.enlarger.c_filter_neutral

    def midgray_print(ymc_values, print_exposure):
        p.enlarger.y_filter_neutral = ymc_values[0]
        p.enlarger.m_filter_neutral = ymc_values[1]
        p.enlarger.print_exposure = print_exposure
        rgb = photo_process(midgray_rgb, p)
        return rgb

    def evaluate_residues(x):
        res = midgray_print([x[0], x[1], c_filter], x[2])
        res = res - midgray_rgb
        res = res.flatten()
        return res

    y_filter = p.enlarger.y_filter_neutral
    m_filter = p.enlarger.m_filter_neutral
    x0 = [y_filter, m_filter, 1.0]
    x = scipy.optimize.least_squares(
        evaluate_residues,
        x0,
        bounds=([0, 0, 0], [1, 1, 10]),
        ftol=1e-6,
        xtol=1e-6,
        gtol=1e-6,
        method="trf",
    )
    print("Total residues:", np.sum(np.abs(evaluate_residues(x.x))), "<-", evaluate_residues(x0))
    profile.enlarger.y_filter_neutral = x.x[0]
    profile.enlarger.m_filter_neutral = x.x[1]
    profile.enlarger.c_filter_neutral = c_filter
    return x.x[0], x.x[1], evaluate_residues(x.x)


def fit_print_filters(profile, iterations=10):
    print(profile.negative.info.stock)
    for i in range(iterations):
        filter_y, filter_m, residues = fit_print_filters_iter(profile)
        if np.sum(np.abs(residues)) < 1e-4 or i == iterations - 1:
            c_filter = profile.enlarger.c_filter_neutral
            print("Fitted Filters :" + f"[ {filter_y:.2f}, {filter_m:.2f}, {c_filter:.2f} ]")
            break

        profile.enlarger.y_filter_neutral = 0.5 * filter_y + np.random.uniform(0, 1) * 0.5
        profile.enlarger.m_filter_neutral = 0.5 * filter_m + np.random.uniform(0, 1) * 0.5
    return filter_y, filter_m, residues


def fit_all_stocks(iterations=5, randomess_starting_points=0.5):
    """Script helper retained for batch fitting from stock enums.

    This utility keeps the historical behavior but lives in profiles layer,
    where fitting logic now belongs.
    """
    from spectral_film_lab.model.stocks import FilmStocks, PrintPapers
    from spectral_film_lab.model.illuminants import Illuminants

    ymc_filters_0 = {}
    residues = {}
    for paper in PrintPapers:
        ymc_filters_0[paper.value] = {}
        residues[paper.value] = {}
        for light in Illuminants:
            ymc_filters_0[paper.value][light.value] = {}
            residues[paper.value][light.value] = {}
            for film in FilmStocks:
                ymc_filters_0[paper.value][light.value][film.value] = [0.90, 0.70, 0.35]
                residues[paper.value][light.value][film.value] = 0.184

    ymc_filters_out = copy.deepcopy(ymc_filters_0)
    r = randomess_starting_points

    for paper in PrintPapers:
        print(" " * 20)
        print("#" * 20)
        print(paper.value)
        for light in Illuminants:
            print("-" * 20)
            print(light.value)
            for stock in FilmStocks:
                if residues[paper.value][light.value][stock.value] > 5e-4:
                    y0 = ymc_filters_0[paper.value][light.value][stock.value][0]
                    m0 = ymc_filters_0[paper.value][light.value][stock.value][1]
                    c0 = ymc_filters_0[paper.value][light.value][stock.value][2]
                    y0 = np.clip(y0, 0, 1) * (1 - r) + np.random.uniform(0, 1) * r
                    m0 = np.clip(m0, 0, 1) * (1 - r) + np.random.uniform(0, 1) * r

                    p = photo_params(
                        negative=stock.value,
                        print_paper=paper.value,
                        ymc_filters_from_database=False,
                    )
                    p.enlarger.illuminant = light.value
                    p.enlarger.y_filter_neutral = y0
                    p.enlarger.m_filter_neutral = m0
                    p.enlarger.c_filter_neutral = c0

                    yf, mf, res = fit_print_filters(p, iterations=iterations)
                    ymc_filters_out[paper.value][light.value][stock.value] = [yf, mf, c0]
                    residues[paper.value][light.value][stock.value] = np.sum(np.abs(res))

    return ymc_filters_out, residues

