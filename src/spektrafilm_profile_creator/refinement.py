import copy

import numpy as np
import scipy

from spektrafilm.profiles.io import load_profile
from spektrafilm.runtime.api import simulate
from spektrafilm.runtime.params_schema import RuntimePhotoParams
from spektrafilm_profile_creator.core.profile_transforms import apply_scale_shift_stretch_density_curves
from spektrafilm_profile_creator.diagnostics.messages import log_event
from spektrafilm_profile_creator.printing_filters import fit_print_filters


def _build_runtime_params(film_profile, print_profile):
    return RuntimePhotoParams(
        film=film_profile.clone(),
        print=load_profile(print_profile),
    )


def correct_negative_curves_with_gray_ramp(
    source_profile,
    target_paper='kodak_portra_endura',
    data_trustability=0.5,
    stretch_curves=False,
    ev_ramp=(-2, -1, 0, 1, 2, 3, 4, 5, 6),
):
    params = _build_runtime_params(source_profile, target_paper)
    params.io.full_image = True
    params.settings.rgb_to_raw_method = 'mallett2019'
    fitted_y, fitted_m, _ = fit_print_filters(params, stock=source_profile.info.stock)
    params.enlarger.y_filter_neutral = fitted_y
    params.enlarger.m_filter_neutral = fitted_m

    density_scale, shift_correction, stretch_correction = fit_corrections_from_grey_ramp(
        params,
        ev_ramp,
        data_trustability,
        stretch_curves,
    )
    corrected_profile = apply_scale_shift_stretch_density_curves(
        params.film,
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


def correct_positive_curves_with_gray_ramp(
    positive_film_profile,
    data_trustability=0.5,
    stretch_curves=False,
    ev_ramp=(-2, -1, 0, 1),
):
    params = _build_runtime_params(positive_film_profile, 'kodak_portra_endura')
    params.io.scan_film = True
    params.io.full_image = True
    params.settings.rgb_to_raw_method = 'hanatos2025'

    density_scale, shift_correction, stretch_correction = fit_corrections_from_grey_ramp(
        params,
        ev_ramp,
        data_trustability,
        stretch_curves,
        positive_film=True,
    )
    corrected_profile = apply_scale_shift_stretch_density_curves(
        params.film,
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


def fit_corrections_from_grey_ramp(
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
                shift_correction=[values[3], 0, values[4]],
                stretch_correction=values[5:8],
            )
        else:
            gray, reference = gray_ramp(
                params,
                ev_ramp,
                density_scale=values[0:3],
                shift_correction=[values[3], 0, values[4]],
            )
        if positive_film:
            gray_mean = np.mean(gray, axis=1).flatten()
            gray_reference = gray_mean[:, None] * np.ones((1, 3))
            zero_index = np.where(np.asarray(ev_ramp) == 0)[0]
            if zero_index.size:
                gray_reference[zero_index] = reference.flatten()
            log_event('fit_corrections_from_grey_ramp_reference', gray_reference=gray_reference)
            residual = gray - gray_reference
            residual = residual / gray_reference * 0.184
        else:
            residual = np.array(gray) - reference
        residual = residual.flatten()

        bias_scale = 2.0 * (np.array(values[0:3]) - 1)
        if stretch_curves:
            bias_stretch = 100.0 * (np.array(values[5:8]) - 1)
            bias = np.concatenate((bias_scale, bias_stretch))
        else:
            bias = bias_scale

        return np.concatenate((residual, bias * data_trustability))

    x0 = [1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0] if stretch_curves else [1.0, 1.0, 1.0, 0.0, 0.0]
    fit = scipy.optimize.least_squares(residues, x0)
    density_scale = fit.x[0:3]
    shift_correction = [fit.x[3], 0, fit.x[4]]
    stretch_correction = fit.x[5:8] if stretch_curves else [1, 1, 1]
    return density_scale, shift_correction, stretch_correction


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


__all__ = [
    'fit_print_filters',
    'correct_negative_curves_with_gray_ramp',
    'correct_positive_curves_with_gray_ramp',
    'fit_corrections_from_grey_ramp',
]
