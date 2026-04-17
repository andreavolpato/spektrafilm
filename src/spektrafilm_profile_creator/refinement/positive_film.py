import numpy as np
import scipy

from spektrafilm_profile_creator.core.density_curves import replace_fitted_density_curves
from spektrafilm_profile_creator.core.profile_transforms import apply_scale_shift_stretch_density_curves
from spektrafilm_profile_creator.diagnostics.messages import log_event

from spektrafilm_profile_creator.refinement.common import (
    DensityCurvesCorrection,
    RuntimeSimulationSession,
    _build_runtime_params,
    _normalized_midgray_residual,
    ShiftCorrection,
    fit_gray_anchor,
    fit_neutral_ramp,
    gray_ramp,
    make_stage_two_regularization,
)


POSITIVE_STAGE2_REGULARIZATION = make_stage_two_regularization(
    target_ramp_mae=0.001,
    tolerated_scale_rms_centered_delta=0.05,
    tolerated_shift_log_exposure_rms_centered_delta=0.1,
    tolerated_stretch_rms_centered_delta=0.005,
    target_ramp_mae_softness=0.0001,
    in_band_shape_weight=0.05,
    excess_mae_weight=1.0,
    midgray_anchor_weight=0.0,
)


def _build_positive_film_runtime_params(profile):
    params = _build_runtime_params(profile, 'kodak_portra_endura')
    params.film = replace_fitted_density_curves(params.film)
    params.io.scan_film = True
    params.settings.rgb_to_raw_method = 'hanatos2025'
    return params


def _positive_film_metrics(profile, ev_ramp=(-2, -1, 0, 1, 2)):
    params = _build_positive_film_runtime_params(profile)
    gray, reference = gray_ramp(params, ev_ramp)
    anchor_index = int(np.argmin(np.abs(np.asarray(ev_ramp, dtype=np.float64))))
    midgray_rgb = gray[anchor_index]
    reference_vector = reference.reshape((3,))
    return {
        'midgray_rgb': midgray_rgb,
        'midgray_mae': float(np.mean(np.abs(midgray_rgb - reference_vector))),
        'ramp_neutrality_mae': float(
            np.mean(np.abs(gray - np.mean(gray, axis=1, keepdims=True)))
        ),
    }


def _convert_ev_ramp_out_to_ev_ramp_in(ev_ramp_out, anchored_profile) -> np.ndarray:
    ev_ramp_out = np.array(ev_ramp_out, dtype=np.float64)
    transmittance_out = 0.184 * 2**ev_ramp_out
    density_out = -np.log10(transmittance_out)
    average_density_curve = np.mean(anchored_profile.data.density_curves, axis=1)
    average_density_min = np.nanmean(anchored_profile.data.base_density)
    average_density_curve += average_density_min
    valid = np.isfinite(average_density_curve)
    log_exposure_ramp = np.interp(
        -density_out,
        -average_density_curve[valid],
        anchored_profile.data.log_exposure[valid],
    )
    log_exposure_ramp -= log_exposure_ramp[2]
    return log_exposure_ramp * np.log2(10)


def refine_positive_film(
    positive_film_profile,
    stretch_curves=False,
    ev_ramp=(-1.6, -0.8, 0, 0.8, 1.6),
    neutral_ramp_refinement=True,
):
    ev_ramp = tuple(ev_ramp)
    before_metrics = _positive_film_metrics(positive_film_profile, ev_ramp=ev_ramp)
    ev_ramp_in = _convert_ev_ramp_out_to_ev_ramp_in(ev_ramp, positive_film_profile)
    log_event(
        'start_refine_positive_film',
        ev_ramp_in=ev_ramp_in,)

    # ---------------------------------------------------------------------------
    # stage 0: build the positive-film runtime session from the fitted working curves
    params = _build_positive_film_runtime_params(positive_film_profile)
    session = RuntimeSimulationSession.create(params)

    # ---------------------------------------------------------------------------
    # stage 1: fit density-curve shift to anchor the scanned positive midgray
    def anchor_residual(shift_values: np.ndarray) -> np.ndarray:
        correction = DensityCurvesCorrection(shift=tuple(shift_values))
        gray, reference = session.gray_ramp(
            params.film,
            (0,),
            correction=correction,
        )
        return np.concatenate((
            _normalized_midgray_residual(gray[0], reference.flatten()),
            0.1 * np.asarray(shift_values, dtype=np.float64),
        ))

    fit = scipy.optimize.least_squares(anchor_residual, np.zeros(3, dtype=np.float64))
    anchor_correction = DensityCurvesCorrection(shift=tuple(fit.x))
    params.film = apply_scale_shift_stretch_density_curves(
        params.film,
        density_scale=(1.0, 1.0, 1.0),
        log_exposure_shift=anchor_correction.shift,
        log_exposure_stretch=(1.0, 1.0, 1.0),
    )
    anchored_profile = apply_scale_shift_stretch_density_curves(
        positive_film_profile.clone(),
        density_scale=(1.0, 1.0, 1.0),
        log_exposure_shift=anchor_correction.shift,
        log_exposure_stretch=(1.0, 1.0, 1.0),
    )
    session.refresh(params)
    anchored_midgray_rgb, anchored_reference = session.gray_ramp(params.film, (0,))
    anchored_metrics = _positive_film_metrics(anchored_profile, ev_ramp=ev_ramp)
    log_event(
        'fit_positive_film_gray_anchor',
        anchored_profile,
        anchor_shift_correction=anchor_correction.shift,
        anchored_midgray_rgb=anchored_midgray_rgb[0],
        anchor_reference_rgb=anchored_reference.flatten(),
        before_midgray_mae=before_metrics['midgray_mae'],
        anchored_midgray_mae=anchored_metrics['midgray_mae'],
        before_ramp_neutrality_mae=before_metrics['ramp_neutrality_mae'],
        anchored_ramp_neutrality_mae=anchored_metrics['ramp_neutrality_mae'],
    )

    # ---------------------------------------------------------------------------
    # stage 2: fit density curve scale and shift to neutralize a gray ramp
    def evaluate_neutral_ramp_rgb(correction: DensityCurvesCorrection):
        return session.gray_ramp(
            params.film,
            ev_ramp_in,
            correction=correction,
        )

    correction = fit_neutral_ramp(
        evaluate_neutral_ramp_rgb,
        DensityCurvesCorrection(),
        regularization=POSITIVE_STAGE2_REGULARIZATION,
        anchor_axis_values=ev_ramp_in,
        anchor_axis_value=0,
        fit_stretch=stretch_curves,
        neutral_ramp_refinement=neutral_ramp_refinement,
    )

    params.film = apply_scale_shift_stretch_density_curves(
        params.film,
        correction.scale,
        correction.shift,
        correction.stretch,
    )
    session.refresh(params)
    corrected_profile = apply_scale_shift_stretch_density_curves(
        anchored_profile,
        correction.scale,
        correction.shift,
        correction.stretch,
    )
    ending_metrics = _positive_film_metrics(corrected_profile, ev_ramp=ev_ramp)
    log_event(
        'refine_positive_film',
        corrected_profile,
        gray_anchor_shift=anchor_correction.shift,
        scale_correction=correction.scale,
        shift_correction=correction.shift,
        stretch_correction=correction.stretch,
        before_midgray_mae=before_metrics['midgray_mae'],
        anchored_midgray_mae=anchored_metrics['midgray_mae'],
        ending_midgray_mae=ending_metrics['midgray_mae'],
        before_ramp_neutrality_mae=before_metrics['ramp_neutrality_mae'],
        anchored_ramp_neutrality_mae=anchored_metrics['ramp_neutrality_mae'],
        ending_ramp_neutrality_mae=ending_metrics['ramp_neutrality_mae'],
        neutral_ramp_refinement=neutral_ramp_refinement,
    )
    return corrected_profile


if __name__ == '__main__':
    from spektrafilm_profile_creator.core.balancing import (
        balance_film_sensitivity,
        prelminary_neutral_shift,
        reconstruct_metameric_neutral,
    )
    from spektrafilm_profile_creator.core.densitometer import densitometer_normalization, unmix_density
    from spektrafilm_profile_creator.core.profile_transforms import remove_density_min
    from spektrafilm_profile_creator.data.loader import load_raw_profile

    def _prepare_positive_film_profile(stock: str):
        raw_profile = load_raw_profile(stock)
        profile = raw_profile.as_profile()
        profile = remove_density_min(profile, reconstruct_base_density=True)
        profile = reconstruct_metameric_neutral(profile)
        profile = densitometer_normalization(profile)
        profile = balance_film_sensitivity(profile)
        profile = prelminary_neutral_shift(profile, per_channel_shift=False)
        profile = unmix_density(profile)
        return raw_profile, profile

    def _positive_film_gray_anchor_metrics(profile):
        params = _build_positive_film_runtime_params(profile)

        def evaluate_midgray_rgb(shift_correction: ShiftCorrection):
            gray, reference = gray_ramp(
                params,
                (0,),
                density_scale=(1.0, 1.0, 1.0),
                shift_correction=shift_correction,
                stretch_correction=(1.0, 1.0, 1.0),
            )
            return gray[0], reference.flatten()

        anchor_correction = fit_gray_anchor(
            evaluate_midgray_rgb,
            shift_weight=0.1,
            log_label='fit_gray_anchor_positive_script',
        )
        anchored_profile = apply_scale_shift_stretch_density_curves(
            profile,
            density_scale=(1.0, 1.0, 1.0),
            log_exposure_shift=anchor_correction.shift,
            log_exposure_stretch=(1.0, 1.0, 1.0),
        )
        metrics = _positive_film_metrics(anchored_profile)
        metrics['shift_correction'] = anchor_correction.shift
        return metrics

    def main() -> None:
        stock = 'kodak_ektachrome_100'
        raw_profile, profile_for_refine = _prepare_positive_film_profile(stock)
        refined_profile = refine_positive_film(
            profile_for_refine,
            stretch_curves=raw_profile.recipe.stretch_curves,
            neutral_ramp_refinement=raw_profile.recipe.neutral_ramp_refinement,
        )
        refined_profile = replace_fitted_density_curves(refined_profile)

        _, profile_for_before = _prepare_positive_film_profile(stock)
        before = _positive_film_metrics(profile_for_before)

        _, profile_for_gray_anchor = _prepare_positive_film_profile(stock)
        after_gray_anchor = _positive_film_gray_anchor_metrics(profile_for_gray_anchor)

        after = _positive_film_metrics(refined_profile)

        source_density_curves = np.asarray(profile_for_refine.data.density_curves, dtype=np.float64)
        refined_density_curves = np.asarray(refined_profile.data.density_curves, dtype=np.float64)
        assert refined_density_curves.shape == source_density_curves.shape
        assert np.isfinite(after['midgray_rgb']).all()
        assert np.isfinite(after['midgray_mae'])
        assert np.isfinite(after['ramp_neutrality_mae'])

        print('positive_film refinement diagnostic')
        print(f'film_stock={raw_profile.info.stock}')
        print(f'stretch_curves={raw_profile.recipe.stretch_curves}')
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
