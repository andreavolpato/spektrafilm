from typing import Any

import numpy as np
import scipy

from spektrafilm_profile_creator.core.density_curves import replace_fitted_density_curves
from spektrafilm_profile_creator.core.profile_transforms import apply_scale_shift_stretch_density_curves
from spektrafilm_profile_creator.diagnostics.messages import log_event
from spektrafilm_profile_creator.neutral_print_filters import fit_neutral_filters

from spektrafilm_profile_creator.refinement.common import (
    DensityCurvesCorrection,
    MIDGRAY_RGB,
    MIDGRAY_RGB_VECTOR,
    MIDGRAY_VALUE,
    RuntimeSimulationSession,
    _build_runtime_params,
    _normalized_gray_ramp_residual,
    _normalized_midgray_residual,
    fit_neutral_ramp,
    make_stage_two_regularization,
)


NEGATIVE_STAGE2_REGULARIZATION = make_stage_two_regularization(
    target_ramp_mae=0.001,
    tolerated_scale_rms_centered_delta=0.05,
    tolerated_shift_log_exposure_rms_centered_delta=0.1,
    tolerated_stretch_rms_centered_delta=0.005,
    target_ramp_mae_softness=0.0001,
    in_band_shape_weight=0.05,
    excess_mae_weight=1.0,
)

NEGATIVE_FILM_MIDGRAY_DENSITY_ANCHOR = 0.75


def _apply_shift_only_density_curve_correction(profile, shift_correction):
    return apply_scale_shift_stretch_density_curves(
        profile.clone(),
        density_scale=(1.0, 1.0, 1.0),
        log_exposure_shift=shift_correction,
        log_exposure_stretch=(1.0, 1.0, 1.0),
    )


def _build_negative_film_runtime_params(profile, target_print):
    params = _build_runtime_params(profile, target_print)
    params.film = replace_fitted_density_curves(params.film)
    params.camera.auto_exposure = False
    params.enlarger.normalize_print_exposure = False
    params.enlarger.print_exposure_compensation = True
    params.settings.rgb_to_raw_method = 'hanatos2025'
    params.settings.neutral_print_filters_from_database = False

    film_density_cmy = np.asarray(
        profile.info.fitted_cmy_midscale_neutral_density,
        dtype=np.float64,
    ).reshape((3,))
    neutral_filter_fit = fit_neutral_filters(
        params,
        film_density_cmy=film_density_cmy,
        normalize_print_exposure=False,
    )
    params.enlarger.c_filter_neutral = neutral_filter_fit.c_filter
    params.enlarger.m_filter_neutral = neutral_filter_fit.m_filter
    params.enlarger.y_filter_neutral = neutral_filter_fit.y_filter
    return params, film_density_cmy, neutral_filter_fit


def _create_negative_film_sessions(params):
    output_session = RuntimeSimulationSession.create(params)
    film_density_session = RuntimeSimulationSession.create(
        params,
        output_film_density_cmy=True,
    )
    return output_session, film_density_session


def _render_midgray_rgb_and_film_density(
    output_session,
    film_density_session,
    profile,
    *,
    correction: DensityCurvesCorrection = DensityCurvesCorrection(),
    print_exposure: float | None = None,
):
    midgray_rgb = output_session.render(
        MIDGRAY_RGB,
        profile,
        correction=correction,
        exposure_compensation_ev=0.0,
        print_exposure=print_exposure,
    ).reshape((3,))
    film_density_cmy = film_density_session.render(
        MIDGRAY_RGB,
        profile,
        correction=correction,
        exposure_compensation_ev=0.0,
        print_exposure=print_exposure,
    ).reshape((3,))
    return midgray_rgb, film_density_cmy


def refine_negative_film(
    source_profile,
    target_print,
    stretch_curves=False,
    ev_ramp=(-2, -1, 0, 1, 2),
    neutral_ramp_refinement=True,
):
    ev_ramp = tuple(ev_ramp)
    
    # ----------------------------------------------------------------------------
    # stage 0: fit neutral print filters to midscale neutral density, without modifying the density curves
    params, film_density_cmy, neutral_filter_fit = _build_negative_film_runtime_params(
        source_profile,
        target_print,
    )
    log_event(
        'fit_negative_film_neutral_setup',
        fitted_filters=neutral_filter_fit.filters,
        print_exposure=neutral_filter_fit.print_exposure,
        injected_film_density_cmy=film_density_cmy,
        residual=neutral_filter_fit.residual,
    )

    # ---------------------------------------------------------------------------
    # stage 1: fit density curve shift to anchor midgray density, with print density mean 0.75 as the anchor point
    output_session, film_density_session = _create_negative_film_sessions(params)
    def anchor_residues(values: np.ndarray) -> np.ndarray:
        correction = DensityCurvesCorrection(shift=tuple(values[0:3]))
        print_exposure = float(values[3])
        midgray_rgb, film_density_cmy_local = _render_midgray_rgb_and_film_density(
            output_session,
            film_density_session,
            params.film,
            correction=correction,
            print_exposure=print_exposure,
        )
        film_density_mean = float(np.mean(film_density_cmy_local))
        return np.concatenate((
            _normalized_midgray_residual(midgray_rgb, MIDGRAY_RGB_VECTOR),
            np.array([
                (film_density_mean - NEGATIVE_FILM_MIDGRAY_DENSITY_ANCHOR)
                / NEGATIVE_FILM_MIDGRAY_DENSITY_ANCHOR
                * MIDGRAY_VALUE
            ], dtype=np.float64),
        ))

    fit = scipy.optimize.least_squares(
        anchor_residues,
        np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        bounds=([-np.inf, -np.inf, -np.inf, 0.0], [np.inf, np.inf, np.inf, 10.0]),
        ftol=1e-6,
        xtol=1e-6,
        gtol=1e-6,
    )

    anchor_correction = DensityCurvesCorrection(shift=tuple(fit.x[0:3]))
    anchor_print_exposure = float(fit.x[3])
    params.enlarger.print_exposure = anchor_print_exposure
    params.film = _apply_shift_only_density_curve_correction(params.film, anchor_correction.shift)
    anchored_source_profile = _apply_shift_only_density_curve_correction(
        source_profile,
        anchor_correction.shift,
    )
    output_session.refresh(params)
    film_density_session.refresh(params)
    midgray_rgb, anchored_film_density_cmy = _render_midgray_rgb_and_film_density(
        output_session,
        film_density_session,
        params.film,
        print_exposure=anchor_print_exposure,
    )
    log_event(
        'fit_negative_film_midgray_density_anchor',
        anchored_source_profile,
        anchor_shift_correction=anchor_correction.shift,
        anchor_print_exposure=anchor_print_exposure,
        anchored_midgray_rgb=midgray_rgb,
        anchored_film_density_cmy=anchored_film_density_cmy,
        anchored_film_density_mean=float(np.mean(anchored_film_density_cmy)),
        density_anchor=NEGATIVE_FILM_MIDGRAY_DENSITY_ANCHOR,
    )

    # ---------------------------------------------------------------------------
    # stage 2: fit density curve scale and shift to neutralize a gray ramp
    def evaluate_neutral_ramp_rgb(correction: DensityCurvesCorrection):
        return output_session.gray_ramp(
            params.film,
            ev_ramp,
            correction=correction,
            print_exposure=anchor_print_exposure,
        )

    def evaluate_neutral_ramp_mae() -> float:
        gray, reference = evaluate_neutral_ramp_rgb(DensityCurvesCorrection())
        ramp_residual = _normalized_gray_ramp_residual(
            gray,
            reference,
            ev_ramp,
            0,
        )
        return float(np.mean(np.abs(ramp_residual)))

    starting_mae = evaluate_neutral_ramp_mae()
    stage_two_correction = fit_neutral_ramp(
        evaluate_neutral_ramp_rgb,
        DensityCurvesCorrection(),
        regularization=NEGATIVE_STAGE2_REGULARIZATION,
        anchor_axis_values=ev_ramp,
        anchor_axis_value=0,
        fit_stretch=stretch_curves,
        neutral_ramp_refinement=neutral_ramp_refinement,
    )

    params.film = apply_scale_shift_stretch_density_curves(
        params.film,
        stage_two_correction.scale,
        stage_two_correction.shift,
        stage_two_correction.stretch,
    )
    output_session.refresh(params)
    film_density_session.refresh(params)
    ending_mae = evaluate_neutral_ramp_mae()
    _, final_film_density_cmy = _render_midgray_rgb_and_film_density(
        output_session,
        film_density_session,
        params.film,
        print_exposure=anchor_print_exposure,
    )
    refined_profile = params.film.clone().update_info(
        fitted_cmy_midscale_neutral_density=np.asarray(final_film_density_cmy, dtype=np.float64),
    )
    log_event(
        'refine_negative_film',
        refined_profile,
        scale_correction=stage_two_correction.scale,
        shift_correction=stage_two_correction.shift,
        stretch_correction=stage_two_correction.stretch,
        print_exposure=params.enlarger.print_exposure,
        starting_mae=starting_mae,
        ending_mae=ending_mae,
        film_density_cmy=final_film_density_cmy,
        film_density_cmy_mean=float(np.mean(final_film_density_cmy)),
        neutral_ramp_refinement=neutral_ramp_refinement,
    )
    return refined_profile


if __name__ == '__main__':
    from spektrafilm_profile_creator.core.balancing import balance_film_sensitivity, prelminary_neutral_shift
    from spektrafilm_profile_creator.core.densitometer import densitometer_normalization
    from spektrafilm_profile_creator.core.densitometer import unmix_density
    from spektrafilm_profile_creator.core.profile_transforms import remove_density_min
    from spektrafilm_profile_creator.data.loader import load_raw_profile
    from spektrafilm_profile_creator.reconstruction.dye_reconstruction import reconstruct_dye_density

    def _prepare_negative_film_profile(stock: str) -> tuple[Any, Any]:
        raw_profile = load_raw_profile(stock)
        profile = reconstruct_dye_density(
            raw_profile.as_profile(),
            model=raw_profile.recipe.dye_density_reconstruct_model,
        )
        profile = densitometer_normalization(profile)
        profile = balance_film_sensitivity(profile)
        profile = remove_density_min(profile)
        profile = prelminary_neutral_shift(profile)
        profile = unmix_density(profile)
        return raw_profile, profile

    def _negative_film_metrics(profile: Any, target_print: str, ev_ramp=(-2, -1, 0, 1, 2)):
        params, _, _ = _build_negative_film_runtime_params(profile, target_print)
        output_session, film_density_session = _create_negative_film_sessions(params)
        gray, reference = output_session.gray_ramp(params.film, ev_ramp)
        midgray_rgb, film_density_cmy = _render_midgray_rgb_and_film_density(
            output_session,
            film_density_session,
            params.film,
            print_exposure=params.enlarger.print_exposure,
        )
        reference_vector = reference.reshape((3,))
        return {
            'midgray_rgb': midgray_rgb,
            'midgray_mae': float(np.mean(np.abs(midgray_rgb - reference_vector))),
            'ramp_neutrality_mae': float(
                np.mean(np.abs(gray - np.mean(gray, axis=1, keepdims=True)))
            ),
            'film_density_mean': float(np.mean(film_density_cmy)),
            'film_density_anchor_error': float(
                abs(np.mean(film_density_cmy) - NEGATIVE_FILM_MIDGRAY_DENSITY_ANCHOR)
            ),
            'refitted_print_exposure': float(params.enlarger.print_exposure),
        }

    def main() -> None:
        stock = 'kodak_portra_400'
        raw_profile, source_profile = _prepare_negative_film_profile(stock)
        target_print = raw_profile.info.target_print
        if target_print is None:
            raise ValueError(f'{stock} does not define a target print in the raw profile info')

        before = _negative_film_metrics(source_profile, target_print)
        refined_profile = refine_negative_film(
            source_profile,
            target_print,
            stretch_curves=raw_profile.recipe.stretch_curves,
            neutral_ramp_refinement=raw_profile.recipe.neutral_ramp_refinement,
        )
        after = _negative_film_metrics(refined_profile, target_print)

        source_density_curves = np.asarray(source_profile.data.density_curves, dtype=np.float64)
        refined_density_curves = np.asarray(refined_profile.data.density_curves, dtype=np.float64)
        assert refined_density_curves.shape == source_density_curves.shape
        assert np.isfinite(after['midgray_rgb']).all()
        assert np.isfinite(after['midgray_mae'])
        assert np.isfinite(after['ramp_neutrality_mae'])
        assert np.isfinite(after['film_density_anchor_error'])
        assert np.isfinite(after['refitted_print_exposure'])

        print('negative_film refinement diagnostic')
        print(f'film_stock={raw_profile.info.stock}')
        print(f'target_print={target_print}')
        print(f'stretch_curves={raw_profile.recipe.stretch_curves}')
        print(f'neutral_ramp_refinement={raw_profile.recipe.neutral_ramp_refinement}')
        print(
            'standalone_midgray_mae '
            f'before={before["midgray_mae"]:.6f} '
            f'after={after["midgray_mae"]:.6f}'
        )
        print(
            'standalone_ramp_neutrality_mae '
            f'before={before["ramp_neutrality_mae"]:.6f} '
            f'after={after["ramp_neutrality_mae"]:.6f}'
        )
        print(
            'standalone_film_density_anchor_error '
            f'before={before["film_density_anchor_error"]:.6f} '
            f'after={after["film_density_anchor_error"]:.6f}'
        )
        print(
            'refitted_print_exposure '
            f'before={before["refitted_print_exposure"]:.6f} '
            f'after={after["refitted_print_exposure"]:.6f}'
        )
        print(
            'standalone_midgray_rgb '
            f'before={np.round(before["midgray_rgb"], 6).tolist()} '
            f'after={np.round(after["midgray_rgb"], 6).tolist()}'
        )

    main()
