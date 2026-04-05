import copy
from dataclasses import dataclass

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
from spektrafilm_profile_creator.data.loader import load_raw_profile
from spektrafilm_profile_creator.diagnostics.messages import log_event
from spektrafilm_profile_creator.neutral_print_filters import DEFAULT_NEUTRAL_PRINT_FILTERS, fit_neutral_print_filters


NEGATIVE_STAGE2_REGULARIZATION = {
    'scale': 0.35,
    'shift': 0.15,
    'stretch': 1.5,
}

POSITIVE_STAGE2_REGULARIZATION = {
    'scale': 0.45,
    'shift': 0.20,
    'stretch': 6.0,
}

PRINT_STAGE2_REGULARIZATION = {
    'scale': 0.60,
    'shift': 0.20,
}


@dataclass(frozen=True)
class DensityCurvesCorrection:
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    shift: tuple[float, float, float] = (0.0, 0.0, 0.0)
    stretch: tuple[float, float, float] = (1.0, 1.0, 1.0)


def _build_runtime_params(film_profile, print_profile):
    params = RuntimePhotoParams(
        film=film_profile.clone(),
        print=load_profile(print_profile),
    )
    params.enlarger.normalize_print_exposure = False
    return params


def _normalized_midgray_residual(midgray_rgb, reference_rgb):
    midgray_rgb = np.asarray(midgray_rgb, dtype=np.float64).flatten()
    reference_rgb = np.asarray(reference_rgb, dtype=np.float64).flatten()
    residual = midgray_rgb - reference_rgb
    return residual / reference_rgb * 0.184


def _gray_reference_for_ramp(gray, reference, anchor_axis_values, anchor_axis_value):
    gray_mean = np.mean(gray, axis=1).flatten()
    gray_reference = gray_mean[:, None] * np.ones((1, 3))
    anchor_index = np.where(np.asarray(anchor_axis_values) == anchor_axis_value)[0]
    if anchor_index.size:
        gray_reference[anchor_index] = np.asarray(reference, dtype=np.float64).flatten()
    return gray_reference


def _normalized_gray_ramp_residual(gray, reference, anchor_axis_values, anchor_axis_value):
    gray_reference = _gray_reference_for_ramp(gray, reference, anchor_axis_values, anchor_axis_value)
    log_event('fit_neutral_ramp_reference', gray_reference=gray_reference)
    residual = gray - gray_reference
    return (residual / gray_reference * 0.184).flatten()


def _build_red_blue_stretch(stretch_values):
    if len(stretch_values) == 0:
        return (1.0, 1.0, 1.0)
    return (stretch_values[0], 1.0, stretch_values[1])


def _build_stage_two_correction(values, anchor_shift, fit_stretch):
    return DensityCurvesCorrection(
        scale=tuple(values[0:3]),
        shift=tuple(np.asarray(anchor_shift, dtype=np.float64) + np.asarray(values[3:6], dtype=np.float64)),
        stretch=_build_red_blue_stretch(values[6:8] if fit_stretch else ()),
    )


def _stage_two_regularization(values, ramp_length, weights, fit_stretch):
    ramp_scale = np.sqrt(ramp_length)
    bias_scale = weights['scale'] * (np.asarray(values[0:3], dtype=np.float64) - 1.0) * ramp_scale
    bias_shift = weights['shift'] * np.asarray(values[3:6], dtype=np.float64) * ramp_scale
    if fit_stretch and 'stretch' in weights:
        bias_stretch = weights['stretch'] * (np.asarray(values[6:8], dtype=np.float64) - 1.0) * ramp_scale
        return np.concatenate((bias_scale, bias_shift, bias_stretch))
    return np.concatenate((bias_scale, bias_shift))


def fit_gray_anchor(
    evaluate_midgray,
    data_trustability: float,
    *,
    shift_weight: float,
    log_label: str,
) -> DensityCurvesCorrection:
    def residues(shift_values):
        midgray_rgb, reference_rgb = evaluate_midgray(tuple(shift_values))
        log_event(f'{log_label}_reference', midgray_rgb=midgray_rgb, midgray_reference=reference_rgb)
        anchor_residual = _normalized_midgray_residual(midgray_rgb, reference_rgb)
        bias_shift = shift_weight * np.asarray(shift_values, dtype=np.float64)
        return np.concatenate((anchor_residual, bias_shift * data_trustability))

    fit = scipy.optimize.least_squares(residues, [0.0, 0.0, 0.0])
    correction = DensityCurvesCorrection(shift=tuple(fit.x))
    log_event(log_label, shift_correction=correction.shift)
    return correction


def fit_neutral_ramp(
    evaluate_neutral_ramp,
    anchor_correction: DensityCurvesCorrection,
    data_trustability: float,
    *,
    regularization,
    anchor_axis_values,
    anchor_axis_value,
    fit_stretch: bool = False,
    neutral_ramp_refinement: bool,
) -> DensityCurvesCorrection:
    if not neutral_ramp_refinement:
        return anchor_correction

    anchor_axis_values = tuple(anchor_axis_values)

    def residues(values):
        correction = _build_stage_two_correction(values, anchor_correction.shift, fit_stretch)
        gray, reference = evaluate_neutral_ramp(correction)
        residual = _normalized_gray_ramp_residual(
            gray,
            reference,
            anchor_axis_values,
            anchor_axis_value,
        )
        bias = _stage_two_regularization(values, len(anchor_axis_values), regularization, fit_stretch)
        return np.concatenate((residual, bias * data_trustability))

    x0 = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0] + ([1.0, 1.0] if fit_stretch else [])
    fit = scipy.optimize.least_squares(residues, x0)
    return _build_stage_two_correction(fit.x, anchor_correction.shift, fit_stretch)


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


def _refine_film(
    profile,
    params,
    ev_ramp,
    data_trustability,
    *,
    regularization,
    fit_stretch,
    anchor_log_label,
    event_name,
    neutral_ramp_refinement,
):
    ev_ramp = tuple(ev_ramp)

    def evaluate_midgray(shift):
        gray, reference = gray_ramp(
            params,
            (0,),
            density_scale=(1.0, 1.0, 1.0),
            shift_correction=shift,
            stretch_correction=(1.0, 1.0, 1.0),
        )
        return gray[0], reference.flatten()

    def evaluate_neutral_ramp(correction: DensityCurvesCorrection):
        return gray_ramp(
            params,
            ev_ramp,
            density_scale=correction.scale,
            shift_correction=correction.shift,
            stretch_correction=correction.stretch,
        )

    anchor_correction = fit_gray_anchor(
        evaluate_midgray,
        data_trustability,
        shift_weight=0.1,
        log_label=anchor_log_label,
    )
    correction = fit_neutral_ramp(
        evaluate_neutral_ramp,
        anchor_correction,
        data_trustability,
        regularization=regularization,
        anchor_axis_values=ev_ramp,
        anchor_axis_value=0,
        fit_stretch=fit_stretch,
        neutral_ramp_refinement=neutral_ramp_refinement,
    )
    corrected_profile = apply_scale_shift_stretch_density_curves(
        profile,
        correction.scale,
        correction.shift,
        correction.stretch,
    )
    log_event(
        event_name,
        corrected_profile,
        scale_correction=correction.scale,
        shift_correction=correction.shift,
        stretch_correction=correction.stretch,
        neutral_ramp_refinement=neutral_ramp_refinement,
    )
    return corrected_profile


def _build_print_rgb_evaluator(profile, target_film, reference_cc_filter_values):
    data = profile.data
    info = profile.info
    log_sensitivity = np.asarray(data.log_sensitivity)
    log_exposure = np.asarray(data.log_exposure)
    channel_density = np.asarray(data.channel_density)
    base_density = np.asarray(data.base_density)
    sensitivity = 10 ** log_sensitivity

    film_raw_profile = load_raw_profile(target_film)
    film_midscale_neutral_density = film_raw_profile.data.midscale_neutral_density
    transmittance_midscale_neutral = 10 ** (-film_midscale_neutral_density)

    reference_illuminant = standard_illuminant(type=info.reference_illuminant)
    filtered_illuminant = color_enlarger(reference_illuminant, filter_cc_values=reference_cc_filter_values)
    filtered_illuminant *= transmittance_midscale_neutral
    viewing_illuminant = standard_illuminant(type=info.viewing_illuminant)

    normalization = np.sum(viewing_illuminant * STANDARD_OBSERVER_CMFS[:, 1], axis=0)
    illuminant_xyz = np.einsum('k,kl->l', viewing_illuminant, STANDARD_OBSERVER_CMFS[:]) / normalization
    illuminant_xy = colour.XYZ_to_xy(illuminant_xyz)

    def rgb_print(print_exposure, density_curves):
        light_from_film = print_exposure * filtered_illuminant
        light_from_film[np.isnan(light_from_film)] = 0

        neutral_exposures = np.nansum(light_from_film[:, None] * sensitivity, axis=0)
        log_raw = np.log10(neutral_exposures)

        density_cmy = np.zeros((3,))
        for index in range(3):
            density_cmy[index] = np.interp(log_raw[index], log_exposure, density_curves[:, index])

        spectral_density = np.nansum(channel_density * density_cmy, axis=1) + base_density
        light_from_print = viewing_illuminant * 10 ** (-spectral_density)
        xyz = np.einsum('k,kl->l', light_from_print, STANDARD_OBSERVER_CMFS[:]) / normalization
        return colour.XYZ_to_RGB(
            xyz,
            RGB_COLOURSPACE_sRGB,
            apply_cctf_encoding=False,
            illuminant=illuminant_xy,
        )

    return rgb_print


def _apply_print_correction(profile, correction: DensityCurvesCorrection):
    return apply_scale_shift_stretch_density_curves(
        profile.clone(),
        density_scale=correction.scale,
        log_exposure_shift=correction.shift,
    ).data.density_curves


def _refine_print(
    profile,
    target_film,
    exposure_ev_ramp,
    reference_cc_filter_values,
    data_trustability,
    *,
    neutral_ramp_refinement,
):
    print_exposures = 2 ** np.array(exposure_ev_ramp, dtype=np.float64)
    rgb_print = _build_print_rgb_evaluator(profile, target_film, reference_cc_filter_values)
    target_midgray_rgb = np.array([0.184, 0.184, 0.184], dtype=np.float64)

    def evaluate_midgray(shift):
        correction = DensityCurvesCorrection(shift=tuple(shift))
        density_curves = _apply_print_correction(profile, correction)
        return rgb_print(1.0, density_curves), target_midgray_rgb

    def evaluate_neutral_ramp(correction: DensityCurvesCorrection):
        density_curves = _apply_print_correction(profile, correction)
        gray = np.zeros((len(print_exposures), 3), dtype=np.float64)
        for index, print_exposure in enumerate(print_exposures):
            gray[index] = rgb_print(print_exposure, density_curves)
        log_event('fit_neutral_ramp_reference', gray_ramp=gray)
        return gray, target_midgray_rgb

    anchor_correction = fit_gray_anchor(
        evaluate_midgray,
        data_trustability,
        shift_weight=0.05,
        log_label='fit_gray_anchor_print',
    )
    correction = fit_neutral_ramp(
        evaluate_neutral_ramp,
        anchor_correction,
        data_trustability,
        regularization=PRINT_STAGE2_REGULARIZATION,
        anchor_axis_values=print_exposures,
        anchor_axis_value=1.0,
        neutral_ramp_refinement=neutral_ramp_refinement,
    )
    density_curves = _apply_print_correction(profile, correction)
    updated_profile = profile.update_data(density_curves=density_curves)
    log_event(
        'refine_negative_print',
        updated_profile,
        scale_correction=correction.scale,
        shift_correction=correction.shift,
        neutral_ramp_refinement=neutral_ramp_refinement,
    )
    return updated_profile


def refine_negative_film(
    source_profile,
    target_print,
    data_trustability=0.5,
    stretch_curves=False,
    ev_ramp=(-1, 0, 1, 2, 3, 4),
    neutral_ramp_refinement=True,
):
    params = _build_runtime_params(source_profile, target_print)
    params.film = replace_fitted_density_curves(params.film)
    params.io.full_image = True
    params.camera.auto_exposure = False
    params.enlarger.print_exposure_compensation = True
    params.settings.rgb_to_raw_method = 'hanatos2025'
    fitted_y, fitted_m, _ = fit_neutral_print_filters(params, stock=source_profile.info.stock)
    params.enlarger.y_filter_neutral = fitted_y
    params.enlarger.m_filter_neutral = fitted_m

    return _refine_film(
        source_profile,
        params,
        ev_ramp,
        data_trustability,
        regularization=NEGATIVE_STAGE2_REGULARIZATION,
        fit_stretch=stretch_curves,
        anchor_log_label='fit_gray_anchor_negative',
        event_name='refine_negative_film',
        neutral_ramp_refinement=neutral_ramp_refinement,
    )


def refine_positive_film(
    positive_film_profile,
    data_trustability,
    stretch_curves=False,
    ev_ramp=(-2, -1, 0, 1, 2),
    neutral_ramp_refinement=True,
):
    params = _build_runtime_params(positive_film_profile, 'kodak_portra_endura')
    params.film = replace_fitted_density_curves(params.film)
    params.io.scan_film = True
    params.io.full_image = True
    params.settings.rgb_to_raw_method = 'hanatos2025'

    return _refine_film(
        positive_film_profile,
        params,
        ev_ramp,
        data_trustability,
        regularization=POSITIVE_STAGE2_REGULARIZATION,
        fit_stretch=stretch_curves,
        anchor_log_label='fit_gray_anchor_positive',
        event_name='refine_positive_film',
        neutral_ramp_refinement=neutral_ramp_refinement,
    )


def refine_negative_print(
    profile,
    target_film,
    data_trustability=1.0,
    exposure_ev_ramp=(-0.8, -0.4, -0.2, -0.1, 0, 0.05, 0.15, 0.3),
    reference_cc_filter_values=DEFAULT_NEUTRAL_PRINT_FILTERS,
    neutral_ramp_refinement=False,
):
    return _refine_print(
        profile,
        target_film,
        exposure_ev_ramp,
        reference_cc_filter_values,
        data_trustability,
        neutral_ramp_refinement=neutral_ramp_refinement,
    )


__all__ = [
    'DensityCurvesCorrection',
    'fit_gray_anchor',
    'fit_neutral_ramp',
    'fit_neutral_print_filters',
    'refine_negative_film',
    'refine_positive_film',
    'refine_negative_print',
]
