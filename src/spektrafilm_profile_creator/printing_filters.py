import copy

import numpy as np
import scipy

from spektrafilm.runtime.api import simulate
from spektrafilm_profile_creator.messages import log_event


def fit_print_filters_iter(profile, start_filters):
    working_profile = copy.deepcopy(profile)
    working_profile.debug.deactivate_spatial_effects = True
    working_profile.debug.deactivate_stochastic_effects = True
    working_profile.print_render.glare.compensation_removal_factor = 0.0
    working_profile.io.input_cctf_decoding = False
    working_profile.io.input_color_space = 'sRGB'
    working_profile.io.resize_factor = 1.0
    working_profile.io.full_image = True
    working_profile.camera.auto_exposure = False
    working_profile.enlarger.print_exposure_compensation = False
    midgray_rgb = np.array([[[0.184, 0.184, 0.184]]])
    c_filter = working_profile.enlarger.c_filter_neutral

    def midgray_print(ymc_values, print_exposure):
        working_profile.enlarger.y_filter_neutral = ymc_values[0]
        working_profile.enlarger.m_filter_neutral = ymc_values[1]
        working_profile.enlarger.print_exposure = print_exposure
        return simulate(midgray_rgb, working_profile)

    def evaluate_residues(values):
        residual = midgray_print([values[0], values[1], c_filter], values[2])
        return (residual - midgray_rgb).flatten()

    y0, m0 = start_filters
    working_profile.enlarger.y_filter_neutral = y0
    working_profile.enlarger.m_filter_neutral = m0
    x0 = [y0, m0, 1.0]
    fit = scipy.optimize.least_squares(
        evaluate_residues,
        x0,
        bounds=([0, 0, 0], [1, 1, 10]),
        ftol=1e-6,
        xtol=1e-6,
        gtol=1e-6,
        method='trf',
    )
    log_event(
        'fit_print_filters_iter',
        total_residual=np.sum(np.abs(evaluate_residues(fit.x))),
        initial_residual=evaluate_residues(x0),
    )
    return fit.x[0], fit.x[1], evaluate_residues(fit.x)


def fit_print_filters(profile, iterations=10, stock=None):
    if stock is None:
        log_event('fit_print_filters')
    else:
        log_event('fit_print_filters', stock=stock)
    c_filter = profile.enlarger.c_filter_neutral
    current_y = profile.enlarger.y_filter_neutral
    current_m = profile.enlarger.m_filter_neutral
    for index in range(iterations):
        filter_y, filter_m, residues = fit_print_filters_iter(profile, start_filters=(current_y, current_m))
        if np.sum(np.abs(residues)) < 1e-4 or index == iterations - 1:
            log_event(
                'fit_print_filters_result',
                fitted_filters=(filter_y, filter_m, c_filter),
                residual=residues,
            )
            break

        current_y = 0.5 * filter_y + np.random.uniform(0, 1) * 0.5
        current_m = 0.5 * filter_m + np.random.uniform(0, 1) * 0.5
    return filter_y, filter_m, residues


__all__ = ['fit_print_filters']