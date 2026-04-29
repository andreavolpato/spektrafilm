import colour
from colour.models import RGB_COLOURSPACE_sRGB
import numpy as np

from spektrafilm.config import STANDARD_OBSERVER_CMFS
from spektrafilm.model.color_filters import color_enlarger
from spektrafilm.model.illuminants import standard_illuminant
from spektrafilm.profiles.io import load_profile
from spektrafilm.runtime.params_schema import RuntimePhotoParams
from spektrafilm_profile_creator.core.profile_transforms import apply_scale_shift_stretch_density_curves
from spektrafilm_profile_creator.data.loader import load_raw_profile
from spektrafilm_profile_creator.diagnostics.messages import log_event
from spektrafilm_profile_creator.neutral_print_filters import DEFAULT_NEUTRAL_PRINT_FILTERS

from spektrafilm_profile_creator.refinement.common import (
    DensityCurvesCorrection,
    MIDGRAY_VALUE,
    MIDGRAY_RGB_VECTOR,
    RuntimeSimulationSession,
    ShiftCorrection,
    fit_gray_anchor,
    fit_neutral_ramp,
    make_stage_two_regularization,
)




PRINT_STAGE2_REGULARIZATION = make_stage_two_regularization(
    target_ramp_mae=0.003,
    tolerated_scale_rms_centered_delta=0.05,
    tolerated_shift_log_exposure_rms_centered_delta=0.1,
    target_ramp_mae_softness=0.0003,
    in_band_shape_weight=0.05,
    excess_mae_weight=1.0,
    midgray_anchor_weight=0.2,
)


def _negative_print_metrics(
    profile,
    target_film: str,
    exposure_ev_ramp=(-1.6, -0.8, 0, 0.8, 1.6),
    reference_cc_filter_values=DEFAULT_NEUTRAL_PRINT_FILTERS,
):
    print_exposures = 2 ** np.asarray(exposure_ev_ramp, dtype=np.float64)
    evaluate_print_rgb = _build_print_rgb_evaluator(
        profile,
        target_film,
        reference_cc_filter_values,
    )
    density_curves = np.asarray(profile.data.density_curves, dtype=np.float64)
    gray = np.zeros((len(print_exposures), 3), dtype=np.float64)
    for index, print_exposure in enumerate(print_exposures):
        gray[index] = evaluate_print_rgb(print_exposure, density_curves)

    anchor_index = int(np.argmin(np.abs(np.asarray(exposure_ev_ramp, dtype=np.float64))))
    midgray_rgb = gray[anchor_index]
    return {
        'midgray_rgb': midgray_rgb,
        'midgray_mae': float(np.mean(np.abs(midgray_rgb - MIDGRAY_RGB_VECTOR))),
        'ramp_neutrality_mae': float(
            np.mean(np.abs(gray - np.mean(gray, axis=1, keepdims=True)))
        ),
    }


def _convert_exposure_ev_ramp_out_to_print_exposure_ramp_in(
    exposure_ev_ramp_out,
    anchored_profile,
) -> np.ndarray:
    exposure_ev_ramp_out = np.asarray(exposure_ev_ramp_out, dtype=np.float64)
    transmittance_out = MIDGRAY_VALUE * 2 ** exposure_ev_ramp_out
    density_out = -np.log10(transmittance_out)
    average_density_curve = np.mean(anchored_profile.data.density_curves, axis=1)
    average_density_min = np.nanmean(anchored_profile.data.base_density)
    average_density_curve += average_density_min
    valid = np.isfinite(average_density_curve)
    log_exposure_ramp = np.interp(
        density_out,
        average_density_curve[valid],
        anchored_profile.data.log_exposure[valid],
    )
    log_exposure_ramp -= log_exposure_ramp[2]
    return 10 ** log_exposure_ramp


def _build_negative_print_runtime_params(profile, target_film, reference_cc_filter_values):
    target_film_profile = load_profile(target_film)
    params = RuntimePhotoParams(
        film=target_film_profile,
        print=profile.clone(),
    )
    params.camera.auto_exposure = False
    params.enlarger.print_exposure_compensation = False
    params.enlarger.normalize_print_exposure = True
    params.enlarger.c_filter_neutral = float(reference_cc_filter_values[0])
    params.enlarger.m_filter_neutral = float(reference_cc_filter_values[1])
    params.enlarger.y_filter_neutral = float(reference_cc_filter_values[2])
    params.settings.rgb_to_raw_method = 'hanatos2025'
    params.settings.neutral_print_filters_from_database = False
    return params, target_film_profile


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

    def evaluate_print_rgb(print_exposure: float, density_curves: np.ndarray) -> np.ndarray:
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

    return evaluate_print_rgb


def refine_negative_print(
    profile,
    target_film,
    exposure_ev_ramp=(-1.6, -0.8, 0, 0.8, 1.6),
    reference_cc_filter_values=DEFAULT_NEUTRAL_PRINT_FILTERS,
    neutral_ramp_refinement=False,
):
    exposure_ev_ramp = tuple(exposure_ev_ramp)
    print_exposures_in = _convert_exposure_ev_ramp_out_to_print_exposure_ramp_in(
        exposure_ev_ramp,
        profile.clone(),
    )
    before_metrics = _negative_print_metrics(
        profile,
        target_film,
        exposure_ev_ramp=exposure_ev_ramp,
        reference_cc_filter_values=reference_cc_filter_values,
    )
    params, target_film_profile = _build_negative_print_runtime_params(
        profile,
        target_film,
        reference_cc_filter_values,
    )
    
    # ---------------------------------------------------------------------------
    # stage 1: fit the gray anchor point
    session = RuntimeSimulationSession.create(
        params,
        density_curves_target='print',
        inject_film_density_cmy=True,
    )
    target_film_density_cmy = np.asarray(
        target_film_profile.info.fitted_cmy_midscale_neutral_density,
        dtype=np.float64,
    ).reshape((1, 1, 3))
    target_midgray_rgb = MIDGRAY_RGB_VECTOR

    def evaluate_midgray_rgb(shift_correction: ShiftCorrection):
        return session.render(
            target_film_density_cmy,
            profile,
            correction=DensityCurvesCorrection(shift=tuple(shift_correction)),
            print_exposure=1.0,
        ).reshape((3,)), target_midgray_rgb

    anchor_correction = fit_gray_anchor(
        evaluate_midgray_rgb,
        shift_weight=0.0,
        log_label='fit_gray_anchor_print',
    )

    anchored_density_curves = apply_scale_shift_stretch_density_curves(
        profile.clone(),
        density_scale=anchor_correction.scale,
        log_exposure_shift=anchor_correction.shift,
    ).data.density_curves
    anchored_profile = profile.clone().update_data(density_curves=anchored_density_curves)
    anchored_params, _ = _build_negative_print_runtime_params(
        anchored_profile,
        target_film,
        reference_cc_filter_values,
    )
    session.refresh(anchored_params)
    anchored_midgray_rgb = session.render(
        target_film_density_cmy,
        anchored_profile,
        print_exposure=1.0,
    ).reshape((3,))
    anchored_metrics = _negative_print_metrics(
        anchored_profile,
        target_film,
        exposure_ev_ramp=exposure_ev_ramp,
        reference_cc_filter_values=reference_cc_filter_values,
    )
    log_event(
        'fit_negative_print_gray_anchor',
        anchored_profile,
        anchor_shift_correction=anchor_correction.shift,
        anchored_midgray_rgb=anchored_midgray_rgb,
        anchor_reference_rgb=MIDGRAY_RGB_VECTOR,
        before_midgray_mae=before_metrics['midgray_mae'],
        anchored_midgray_mae=anchored_metrics['midgray_mae'],
        before_ramp_neutrality_mae=before_metrics['ramp_neutrality_mae'],
        anchored_ramp_neutrality_mae=anchored_metrics['ramp_neutrality_mae'],
    )
    
    # ---------------------------------------------------------------------------
    # stage 2: fit the neutral ramp shape


    def evaluate_neutral_ramp_rgb(correction: DensityCurvesCorrection):
        gray = np.zeros((len(print_exposures_in), 3), dtype=np.float64)
        for index, print_exposure in enumerate(print_exposures_in):
            gray[index] = session.render(
                target_film_density_cmy,
                anchored_profile,
                correction=correction,
                print_exposure=float(print_exposure),
            ).reshape((3,))
        return gray, target_midgray_rgb

    correction = fit_neutral_ramp(
        evaluate_neutral_ramp_rgb,
        DensityCurvesCorrection(),
        regularization=PRINT_STAGE2_REGULARIZATION,
        anchor_axis_values=print_exposures_in,
        anchor_axis_value=1.0,
        neutral_ramp_refinement=neutral_ramp_refinement,
    )

    density_curves = apply_scale_shift_stretch_density_curves(
        anchored_profile.clone(),
        density_scale=correction.scale,
        log_exposure_shift=correction.shift,
    ).data.density_curves
    updated_profile = anchored_profile.update_data(density_curves=density_curves)
    ending_metrics = _negative_print_metrics(
        updated_profile,
        target_film,
        exposure_ev_ramp=exposure_ev_ramp,
        reference_cc_filter_values=reference_cc_filter_values,
    )
    log_event(
        'refine_negative_print',
        updated_profile,
        gray_anchor_shift=anchor_correction.shift,
        scale_correction=correction.scale,
        shift_correction=correction.shift,
        before_midgray_mae=before_metrics['midgray_mae'],
        anchored_midgray_mae=anchored_metrics['midgray_mae'],
        ending_midgray_mae=ending_metrics['midgray_mae'],
        before_ramp_neutrality_mae=before_metrics['ramp_neutrality_mae'],
        anchored_ramp_neutrality_mae=anchored_metrics['ramp_neutrality_mae'],
        ending_ramp_neutrality_mae=ending_metrics['ramp_neutrality_mae'],
        neutral_ramp_refinement=neutral_ramp_refinement,
    )
    return updated_profile


if __name__ == '__main__':
    from spektrafilm_profile_creator.core.balancing import (
        balance_print_sensitivity,
        prelminary_neutral_shift,
        reconstruct_metameric_neutral,
    )
    from spektrafilm_profile_creator.core.density_curves import replace_fitted_density_curves
    from spektrafilm_profile_creator.core.densitometer import densitometer_normalization, unmix_density
    from spektrafilm_profile_creator.core.profile_transforms import remove_density_min

    def _prepare_negative_print_profile(stock: str):
        raw_profile = load_raw_profile(stock)
        profile = raw_profile.as_profile()
        profile = remove_density_min(profile, reconstruct_base_density=True)
        profile = reconstruct_metameric_neutral(profile)
        profile = densitometer_normalization(profile)
        profile = balance_print_sensitivity(profile, target_film=raw_profile.recipe.target_film)
        profile = prelminary_neutral_shift(
            profile,
            per_channel_shift=raw_profile.recipe.neutral_log_exposure_correction,
        )
        profile = unmix_density(profile)
        return raw_profile, profile

    def _negative_print_gray_anchor_metrics(
        profile,
        target_film: str,
        reference_cc_filter_values=DEFAULT_NEUTRAL_PRINT_FILTERS,
    ):
        evaluate_print_rgb = _build_print_rgb_evaluator(profile, target_film, reference_cc_filter_values)

        def evaluate_midgray_rgb(shift_correction: ShiftCorrection):
            correction = DensityCurvesCorrection(shift=tuple(shift_correction))
            density_curves = apply_scale_shift_stretch_density_curves(
                profile.clone(),
                density_scale=correction.scale,
                log_exposure_shift=correction.shift,
            ).data.density_curves
            return evaluate_print_rgb(1.0, density_curves), MIDGRAY_RGB_VECTOR

        anchor_correction = fit_gray_anchor(
            evaluate_midgray_rgb,
            shift_weight=0.05,
            log_label='fit_gray_anchor_print_script',
        )
        anchored_density_curves = apply_scale_shift_stretch_density_curves(
            profile.clone(),
            density_scale=anchor_correction.scale,
            log_exposure_shift=anchor_correction.shift,
        ).data.density_curves
        anchored_profile = profile.clone().update_data(density_curves=anchored_density_curves)
        metrics = _negative_print_metrics(
            anchored_profile,
            target_film,
            reference_cc_filter_values=reference_cc_filter_values,
        )
        metrics['shift_correction'] = anchor_correction.shift
        return metrics

    def main() -> None:
        stock = 'kodak_portra_endura'
        raw_profile, profile = _prepare_negative_print_profile(stock)
        target_film = raw_profile.recipe.target_film
        if target_film is None:
            raise ValueError(f'{stock} does not define a target film in the raw profile recipe')

        before = _negative_print_metrics(profile, target_film)
        after_gray_anchor = _negative_print_gray_anchor_metrics(profile, target_film)
        refined_profile = refine_negative_print(
            profile,
            target_film=target_film,
            neutral_ramp_refinement=raw_profile.recipe.neutral_ramp_refinement,
        )
        refined_profile = replace_fitted_density_curves(refined_profile)
        after = _negative_print_metrics(refined_profile, target_film)

        source_density_curves = np.asarray(profile.data.density_curves, dtype=np.float64)
        refined_density_curves = np.asarray(refined_profile.data.density_curves, dtype=np.float64)
        assert refined_density_curves.shape == source_density_curves.shape
        assert np.isfinite(after['midgray_rgb']).all()
        assert np.isfinite(after['midgray_mae'])
        assert np.isfinite(after['ramp_neutrality_mae'])

        print('negative_print refinement diagnostic')
        print(f'print_stock={raw_profile.info.stock}')
        print(f'target_film={target_film}')
        print(f'neutral_ramp_refinement={raw_profile.recipe.neutral_ramp_refinement}')
        print(
            'midgray_mae '
            f'before={before["midgray_mae"]:.6f} '
            f'after_gray_anchor={after_gray_anchor["midgray_mae"]:.6f} '
            f'after={after["midgray_mae"]:.6f}'
        )
        print(
            'ramp_neutrality_mae '
            f'before={before["ramp_neutrality_mae"]:.6f} '
            f'after={after["ramp_neutrality_mae"]:.6f}'
        )
        print(
            'midgray_rgb '
            f'before={np.round(before["midgray_rgb"], 6).tolist()} '
            f'after_gray_anchor={np.round(after_gray_anchor["midgray_rgb"], 6).tolist()} '
            f'after={np.round(after["midgray_rgb"], 6).tolist()}'
        )
        print(
            'gray_anchor_shift '
            f'{np.round(np.asarray(after_gray_anchor["shift_correction"], dtype=np.float64), 6).tolist()}'
        )

    main()
