import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.stats
import scipy.optimize

from spektrafilm_profile_creator.diagnostics.messages import log_event


def density_curve_model_norm_cdfs(
    log_exposure,
    parameters=(0, 1, 2, 0.5, 0.5, 0.5, 0.3, 0.5, 0.7),
    profile_type='negative',
    support='film',
    number_of_layers=3,
):
    del support
    centers = parameters[0:3]
    amplitudes = parameters[3:6]
    sigmas = parameters[6:9]

    density_curve = np.zeros(log_exposure.shape)
    for index, (center, amplitude, sigma) in enumerate(zip(centers, amplitudes, sigmas)):
        if index <= number_of_layers - 1:
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
    for index, (center, amplitude, sigma) in enumerate(zip(centers, amplitudes, sigmas)):
        if index <= number_of_layers - 1:
            distribution[:, index] += scipy.stats.norm.pdf((log_exposure - center) / sigma) * amplitude / sigma
    return distribution


def density_curve_layers(log_exposure, parameters, profile_type='negative', number_of_layers=3):
    centers = parameters[0:number_of_layers]
    amplitudes = parameters[number_of_layers:2 * number_of_layers]
    sigmas = parameters[2 * number_of_layers:3 * number_of_layers]

    density_curve = np.zeros((log_exposure.shape[0], 3))
    for index, (center, amplitude, sigma) in enumerate(zip(centers, amplitudes, sigmas)):
        if index <= number_of_layers - 1:
            if profile_type == 'positive':
                density_curve[:, index] += scipy.stats.norm.cdf(-(log_exposure - center) / sigma) * amplitude
            else:
                density_curve[:, index] += scipy.stats.norm.cdf((log_exposure - center) / sigma) * amplitude
    return density_curve


def guess_start_and_bounds_norm_cdfs(log_exposure, data, profile_type, support='film'):
    del data
    if np.logical_or(profile_type == 'positive', support == 'paper'):
        x0 = [
            np.mean(log_exposure) - 0.25,
            np.mean(log_exposure),
            np.mean(log_exposure) + 0.25,
            0.5,
            1,
            0.5,
            0.2,
            0.3,
            0.5,
        ]
        x_lb = [
            np.min(log_exposure),
            np.min(log_exposure),
            np.min(log_exposure),
            0.25,
            0.25,
            0.25,
            0.05,
            0.05,
            0.05,
        ]
        x_ub = [
            np.max(log_exposure),
            np.max(log_exposure),
            np.max(log_exposure),
            5.0,
            5.0,
            5.0,
            1,
            1,
            1,
        ]
    else:
        density_max_layer = 1.35
        x0 = [
            np.mean(log_exposure) - 0.5,
            np.mean(log_exposure),
            np.mean(log_exposure) + 0.5,
            0.5,
            0.5,
            0.5,
            0.3,
            0.6,
            0.9,
        ]
        x_lb = [
            np.min(log_exposure),
            np.min(log_exposure),
            np.min(log_exposure),
            0.2,
            0.2,
            0.2,
            0.05,
            0.05,
            0.05,
        ]
        x_ub = [
            np.max(log_exposure),
            np.max(log_exposure),
            np.max(log_exposure),
            density_max_layer,
            density_max_layer,
            density_max_layer,
            2,
            2,
            2,
        ]
    x0 = np.clip(x0, x_lb, x_ub).tolist()
    return x0, (x_lb, x_ub)


def density_curve_model_log_line(
    log_exposure,
    parameters=(0.1, 1.5, -2.5, 2, 2, 2, 0.2, 0, 0.2, 0),
    profile_type='negative',
    support='film',
):
    del support
    values = np.zeros(10)
    values[0:10] = parameters[0:10]
    density_min = values[0]
    gamma = values[1]
    log_exposure_reference = values[2]
    density_range = values[3]
    curvature_toe = values[4]
    curvature_shoulder = values[5]
    curvature_toe_slope = values[6]
    curvature_toe_max = values[7]
    curvature_shoulder_slope = values[8]
    curvature_shoulder_max = values[9]

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
    stop = gamma / shoulder_shape * np.log10(
        1 + 10 ** (shoulder_shape * (log_exposure - density_range / np.abs(gamma) - log_exposure_zero))
    )
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
        0,
        0,
        0,
        0,
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

    for index in np.arange(3):
        density[:, index] = model_function(
            log_exposure,
            parameters[index],
            profile_type=profile_type,
            support=support,
        )
    return density


def compute_density_curves_layers(log_exposure, parameters, profile_type, support='film'):
    del support
    density = np.zeros((np.size(log_exposure), 3, 3))
    for index in np.arange(3):
        density[:, :, index] = density_curve_layers(
            log_exposure,
            parameters[index],
            profile_type=profile_type,
        )
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


def fit_density_curves(
    log_exposure,
    density,
    profile_type='negative',
    support='film',
    model='norm_cdfs',
    plotting=False,
    stock='film_stock',
):
    fitted_parameters = [
        fit_density_curve(
            log_exposure,
            density[:, index],
            profile_type=profile_type,
            support=support,
            model=model,
        )
        for index in np.arange(3)
    ]
    fitted_parameters = np.asarray(fitted_parameters, dtype=float)

    if plotting:
        if model == 'norm_cdfs':
            model_function = density_curve_model_norm_cdfs
        elif model == 'log_line':
            model_function = density_curve_model_log_line
        else:
            raise ValueError(f'Unsupported density curve model: {model}')
        _, axis = plt.subplots()
        colors = ['tab:red', 'tab:green', 'tab:blue']
        for index in np.arange(3):
            axis.plot(log_exposure, density[:, index], '.', color='k', label='_nolegend_')
            axis.plot(
                log_exposure,
                model_function(log_exposure, fitted_parameters[index], profile_type=profile_type, support=support),
                color=colors[index],
            )
            if model == 'norm_cdfs':
                axis.plot(
                    log_exposure,
                    distribution_model_norm_cdfs(log_exposure, fitted_parameters[index]),
                    label='_nolegend_',
                    color=colors[index],
                    linewidth=1,
                    linestyle='dashed',
                )
        axis.set_xlabel('Log Exposure')
        axis.set_ylabel('Density')
        axis.set_title(stock + ' - ' + support + ' - ' + profile_type)
        axis.legend(('r', 'g', 'b'))
    return fitted_parameters


def replace_fitted_density_curves(profile, control_plot=False, model='norm_cdfs'):
    data = profile.data
    info = profile.info
    density_curves = data.density_curves
    log_exposure = data.log_exposure
    profile_type = info.type
    support = info.support

    fitting_parameters = fit_density_curves(
        log_exposure,
        density_curves,
        profile_type=profile_type,
        support=support,
        model=model,
    )
    density_curves_prefit = np.copy(density_curves)
    fitted_density_curves = compute_density_curves(
        log_exposure,
        fitting_parameters,
        profile_type=profile_type,
        support=support,
        model=model,
    )
    fitted_layer_tensor = compute_density_curves_layers(
        log_exposure,
        fitting_parameters,
        profile_type=profile_type,
        support=support,
    )
    updated_profile = profile.update_data(
        density_curves=fitted_density_curves,
        density_curves_layers=fitted_layer_tensor,
    )
    log_event(
        'replace_fitted_density_curves',
        updated_profile,
        fitting_parameters=fitting_parameters,
    )

    if control_plot:
        plt.figure()
        plt.plot(log_exposure, fitted_density_curves)
        plt.plot(log_exposure, density_curves_prefit, color='gray', linestyle='--')
        plt.legend(('r', 'g', 'b'))
        plt.xlabel('Log Exposure')
        plt.ylabel('Density (over B+F)')
    return updated_profile
