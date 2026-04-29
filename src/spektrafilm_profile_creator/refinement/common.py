import copy
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import scipy

from spektrafilm.profiles.io import load_profile
from spektrafilm.runtime.api import Simulator, digest_params
from spektrafilm.runtime.params_schema import RuntimePhotoParams
from spektrafilm_profile_creator.core.profile_transforms import apply_scale_shift_stretch_density_curves
from spektrafilm_profile_creator.diagnostics.messages import log_event


MIDGRAY_VALUE = 0.184
MIDGRAY_RGB = np.array([[[MIDGRAY_VALUE, MIDGRAY_VALUE, MIDGRAY_VALUE]]], dtype=np.float64)
MIDGRAY_RGB_VECTOR = MIDGRAY_RGB.reshape((3,))


@dataclass(frozen=True)
class DensityCurvesCorrection:
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    shift: tuple[float, float, float] = (0.0, 0.0, 0.0)
    stretch: tuple[float, float, float] = (1.0, 1.0, 1.0)


ShiftCorrection = tuple[float, float, float]
RgbEvaluation = tuple[np.ndarray, np.ndarray]
MidgrayEvaluator = Callable[[ShiftCorrection], RgbEvaluation]
NeutralRampEvaluator = Callable[[DensityCurvesCorrection], RgbEvaluation]
PrintRgbEvaluator = Callable[[float, np.ndarray], np.ndarray]


def _build_runtime_params(film_profile, print_profile):
    params = RuntimePhotoParams(
        film=film_profile.clone(),
        print=load_profile(print_profile),
    )
    return params


@dataclass(slots=True)
class RuntimeSimulationSession:
    simulator: Simulator
    density_curves_target: str = 'film'
    output_film_density_cmy: bool = False
    inject_film_density_cmy: bool = False

    @classmethod
    def create(
        cls,
        params,
        *,
        density_curves_target: str = 'film',
        output_film_density_cmy: bool = False,
        inject_film_density_cmy: bool = False,
    ) -> 'RuntimeSimulationSession':
        return cls(
            simulator=Simulator(
                cls._build_simulation_params(
                    params,
                    output_film_density_cmy=output_film_density_cmy,
                    inject_film_density_cmy=inject_film_density_cmy,
                )
            ),
            density_curves_target=density_curves_target,
            output_film_density_cmy=output_film_density_cmy,
            inject_film_density_cmy=inject_film_density_cmy,
        )

    @staticmethod
    def _build_simulation_params(
        params,
        *,
        output_film_density_cmy: bool,
        inject_film_density_cmy: bool,
    ):
        working_params = copy.deepcopy(params)
        working_params.io.input_cctf_decoding = False
        working_params.io.input_color_space = 'sRGB'
        working_params.io.output_cctf_encoding = False
        working_params.debug.deactivate_spatial_effects = True
        working_params.debug.deactivate_stochastic_effects = True
        working_params.debug.debug_mode = 'off'
        working_params.debug.output_film_log_raw = False
        working_params.debug.output_film_density_cmy = False
        working_params.debug.output_print_density_cmy = False
        working_params.debug.inject_film_density_cmy = False
        working_params.settings.use_enlarger_lut = False
        working_params.settings.use_scanner_lut = False
        working_params.print_render.glare.active = False
        if output_film_density_cmy:
            working_params.debug.debug_mode = 'output'
            working_params.debug.output_film_density_cmy = True
        if inject_film_density_cmy:
            working_params.debug.debug_mode = 'inject'
            working_params.debug.inject_film_density_cmy = True
        return digest_params(working_params)

    def refresh(self, params) -> None:
        self.simulator.update_params(
            self._build_simulation_params(
                params,
                output_film_density_cmy=self.output_film_density_cmy,
                inject_film_density_cmy=self.inject_film_density_cmy,
            )
        )

    def render(
        self,
        image,
        profile,
        *,
        correction: DensityCurvesCorrection = DensityCurvesCorrection(),
        exposure_compensation_ev: float | None = None,
        print_exposure: float | None = None,
    ):
        density_curves = np.asarray(
            apply_scale_shift_stretch_density_curves(
                profile.clone(),
                correction.scale,
                correction.shift,
                correction.stretch,
            ).data.density_curves,
            dtype=np.float64,
        )
        update_kwargs = self._density_curve_update_kwargs(density_curves)
        if exposure_compensation_ev is not None:
            update_kwargs['exposure_compensation_ev'] = float(exposure_compensation_ev)
        if print_exposure is not None:
            update_kwargs['print_exposure'] = float(print_exposure)
        self.simulator.soft_update(**update_kwargs)
        return self.simulator.process(image)

    def gray_ramp(
        self,
        profile,
        ev_ramp,
        *,
        correction: DensityCurvesCorrection = DensityCurvesCorrection(),
        image=MIDGRAY_RGB,
        print_exposure: float | None = None,
    ):
        density_curves = np.asarray(
            apply_scale_shift_stretch_density_curves(
                profile.clone(),
                correction.scale,
                correction.shift,
                correction.stretch,
            ).data.density_curves,
            dtype=np.float64,
        )
        self.simulator.soft_update(**self._density_curve_update_kwargs(density_curves))
        if print_exposure is not None:
            self.simulator.soft_update(print_exposure=float(print_exposure))
        gray = np.zeros((np.size(ev_ramp), 3), dtype=np.float64)
        for index, exposure_compensation_ev in enumerate(ev_ramp):
            self.simulator.soft_update(exposure_compensation_ev=float(exposure_compensation_ev))
            gray[index] = self.simulator.process(image).flatten()
        return gray, image

    def _density_curve_update_kwargs(self, density_curves: np.ndarray) -> dict[str, np.ndarray]:
        if self.density_curves_target == 'film':
            return {'film_density_curves': density_curves}
        if self.density_curves_target == 'print':
            return {'print_density_curves': density_curves}
        raise ValueError(f'Unsupported density_curves_target: {self.density_curves_target}')


def _normalized_midgray_residual(midgray_rgb, reference_rgb):
    midgray_rgb = np.asarray(midgray_rgb, dtype=np.float64).flatten()
    reference_rgb = np.asarray(reference_rgb, dtype=np.float64).flatten()
    residual = midgray_rgb - reference_rgb
    return residual / reference_rgb * MIDGRAY_VALUE


def _gray_ramp_reference(gray, reference, anchor_axis_values, anchor_axis_value):
    gray = np.asarray(gray, dtype=np.float64)
    gray_mean = np.mean(gray, axis=1).reshape((-1, 1))
    gray_reference = np.repeat(gray_mean, 3, axis=1)
    anchor_index = np.where(np.asarray(anchor_axis_values) == anchor_axis_value)[0]
    if anchor_index.size:
        gray_reference[anchor_index[0]] = np.asarray(reference, dtype=np.float64).flatten()
    return gray_reference


def _normalized_gray_ramp_residual(gray, reference, anchor_axis_values, anchor_axis_value):
    gray_reference = _gray_ramp_reference(
        gray,
        reference,
        anchor_axis_values,
        anchor_axis_value,
    )
    residual = gray - gray_reference
    return (residual / gray_reference * MIDGRAY_VALUE).flatten()


def gray_ramp_neutral_maes(gray, reference, anchor_axis_values, anchor_axis_value) -> dict[str, float]:
    gray = np.asarray(gray, dtype=np.float64)
    gray_reference = _gray_ramp_reference(
        gray,
        reference,
        anchor_axis_values,
        anchor_axis_value,
    )
    neutral_residual = gray - gray_reference
    anchor_index = np.where(np.asarray(anchor_axis_values) == anchor_axis_value)[0]
    anchor_mae = 0.0
    if anchor_index.size:
        anchor_mae = float(
            np.mean(np.abs(gray[anchor_index[0]] - np.asarray(reference, dtype=np.float64).flatten()))
        )
    return {
        'neutral_ramp_mae': float(np.mean(np.abs(neutral_residual))),
        'midgray_neutral_mae': anchor_mae,
    }


def _centered_three_channel_delta(values: np.ndarray, start_index: int) -> np.ndarray:
    first = float(values[start_index])
    second = float(values[start_index + 1])
    return np.array([first, second, -(first + second)], dtype=np.float64)


def make_stage_two_regularization(
    *,
    target_ramp_mae: float,
    tolerated_scale_rms_centered_delta: float,
    tolerated_shift_log_exposure_rms_centered_delta: float,
    tolerated_stretch_rms_centered_delta: float | None = None,
    target_ramp_mae_softness: float | None = None,
    in_band_shape_weight: float = 0.05,
    excess_mae_weight: float = 1.0,
    midgray_anchor_weight: float = 0.0,
) -> dict[str, float]:
    target_ramp_mae = float(target_ramp_mae)
    tolerated_scale_rms_centered_delta = float(tolerated_scale_rms_centered_delta)
    tolerated_shift_log_exposure_rms_centered_delta = float(tolerated_shift_log_exposure_rms_centered_delta)

    # Scale and stretch keep the centered stage-two parameterization. Shift now
    # uses three independently fitted channel values, but the historical kwarg
    # names are preserved to avoid expanding the caller-facing config surface.
    regularization = {
        'target_ramp_mae': target_ramp_mae,
        'target_ramp_mae_softness': float(
            target_ramp_mae_softness if target_ramp_mae_softness is not None else max(target_ramp_mae * 0.1, 1e-6)
        ),
        'in_band_shape_weight': float(in_band_shape_weight),
        'excess_mae_weight': float(excess_mae_weight),
        'midgray_anchor_weight': float(midgray_anchor_weight),
        'tolerated_scale_rms_centered_delta': tolerated_scale_rms_centered_delta,
        'tolerated_shift_log_exposure_rms_centered_delta': tolerated_shift_log_exposure_rms_centered_delta,
    }
    regularization['scale'] = target_ramp_mae / tolerated_scale_rms_centered_delta
    regularization['shift'] = target_ramp_mae / tolerated_shift_log_exposure_rms_centered_delta
    if tolerated_stretch_rms_centered_delta is not None:
        tolerated_stretch_rms_centered_delta = float(tolerated_stretch_rms_centered_delta)
        regularization['tolerated_stretch_rms_centered_delta'] = tolerated_stretch_rms_centered_delta
        regularization['stretch'] = target_ramp_mae / tolerated_stretch_rms_centered_delta
    return regularization


def _stage_two_ramp_objective(ramp_residual: np.ndarray, weights) -> np.ndarray:
    target_ramp_mae = float(weights.get('target_ramp_mae', 0.0))
    if target_ramp_mae <= 0.0:
        return ramp_residual

    ramp_mae = float(np.mean(np.abs(ramp_residual)))
    softness = float(weights.get('target_ramp_mae_softness', max(target_ramp_mae * 0.1, 1e-6)))
    softness = max(softness, 1e-12)
    in_band_shape_weight = float(weights.get('in_band_shape_weight', 0.05))
    in_band_shape_weight = float(np.clip(in_band_shape_weight, 0.0, 1.0))
    excess_mae_weight = float(weights.get('excess_mae_weight', 1.0))

    excess_ramp_mae = float(np.logaddexp(0.0, (ramp_mae - target_ramp_mae) / softness) * softness)
    transition_input = float(np.clip((ramp_mae - target_ramp_mae) / softness, -60.0, 60.0))
    transition = 1.0 / (1.0 + np.exp(-transition_input))
    shape_weight = in_band_shape_weight + (1.0 - in_band_shape_weight) * transition
    quality_residual = np.array([
        excess_mae_weight * excess_ramp_mae * np.sqrt(ramp_residual.size)
    ], dtype=np.float64)
    return np.concatenate((shape_weight * ramp_residual, quality_residual))


def _stage_two_regularization(values, ramp_length, weights, fit_stretch):
    rms_residual_scale = np.sqrt(3.0 * ramp_length)
    centered_scale_delta = _centered_three_channel_delta(values, 0)
    shift_delta = np.asarray(values[2:5], dtype=np.float64)
    bias_scale = np.array([
        weights['scale'] * np.sqrt(np.mean(np.square(centered_scale_delta))) * rms_residual_scale
    ], dtype=np.float64)
    bias_shift = np.array([
        weights['shift'] * np.sqrt(np.mean(np.square(shift_delta))) * rms_residual_scale
    ], dtype=np.float64)
    if fit_stretch and 'stretch' in weights:
        centered_stretch_delta = _centered_three_channel_delta(values, 5)
        bias_stretch = np.array([
            weights['stretch'] * np.sqrt(np.mean(np.square(centered_stretch_delta))) * rms_residual_scale
        ], dtype=np.float64)
        return np.concatenate((bias_scale, bias_shift, bias_stretch))
    return np.concatenate((bias_scale, bias_shift))


def fit_gray_anchor(
    evaluate_midgray: MidgrayEvaluator,
    *,
    shift_weight: float,
    log_label: str,
) -> DensityCurvesCorrection:
    def residual_function(shift_values: np.ndarray) -> np.ndarray:
        midgray_rgb, reference_rgb = evaluate_midgray(tuple(shift_values))
        anchor_residual = _normalized_midgray_residual(midgray_rgb, reference_rgb)
        bias_shift = shift_weight * np.asarray(shift_values, dtype=np.float64)
        return np.concatenate((anchor_residual, bias_shift))

    fit = scipy.optimize.least_squares(residual_function, np.zeros(3, dtype=np.float64))
    correction = DensityCurvesCorrection(shift=tuple(fit.x))
    log_event(log_label, shift_correction=correction.shift)
    return correction


def fit_neutral_ramp(
    evaluate_neutral_ramp: NeutralRampEvaluator,
    anchor_correction: DensityCurvesCorrection,
    *,
    regularization,
    anchor_axis_values,
    anchor_axis_value,
    fit_stretch: bool = False,
    neutral_ramp_refinement: bool,
) -> DensityCurvesCorrection:
    if not neutral_ramp_refinement:
        gray, reference = evaluate_neutral_ramp(anchor_correction)
        gray_reference = _gray_ramp_reference(
            gray,
            reference,
            anchor_axis_values,
            anchor_axis_value,
        )
        neutral_maes = gray_ramp_neutral_maes(
            gray,
            reference,
            anchor_axis_values,
            anchor_axis_value,
        )
        log_event(
            'gray_ramp',
            gray=gray,
            reference=gray_reference,
            axis_values=np.asarray(anchor_axis_values, dtype=np.float64),
            anchor_axis_value=float(anchor_axis_value),
            neutral_ramp_mae=neutral_maes['neutral_ramp_mae'],
            midgray_neutral_mae=neutral_maes['midgray_neutral_mae'],
            neutral_ramp_refinement=neutral_ramp_refinement,
        )
        return anchor_correction

    anchor_axis_values = tuple(anchor_axis_values)
    anchor_index = np.where(np.asarray(anchor_axis_values) == anchor_axis_value)[0]
    anchor_shift = np.asarray(anchor_correction.shift, dtype=np.float64)
    midgray_anchor_weight = float(regularization.get('midgray_anchor_weight', 0.0))

    def build_correction(values: np.ndarray) -> DensityCurvesCorrection:
        centered_scale_delta = _centered_three_channel_delta(values, 0)
        shift_delta = np.asarray(values[2:5], dtype=np.float64)
        stretch = (1.0, 1.0, 1.0)
        if fit_stretch:
            centered_stretch_delta = _centered_three_channel_delta(values, 5)
            stretch = tuple(1.0 + centered_stretch_delta)
        return DensityCurvesCorrection(
            scale=tuple(1.0 + centered_scale_delta),
            shift=tuple(anchor_shift + shift_delta),
            stretch=stretch,
        )

    def residual_function(values: np.ndarray) -> np.ndarray:
        correction = build_correction(values)
        gray, reference = evaluate_neutral_ramp(correction)
        ramp_residual = _normalized_gray_ramp_residual(
            gray,
            reference,
            anchor_axis_values,
            anchor_axis_value,
        )
        quality_residual = _stage_two_ramp_objective(ramp_residual, regularization)
        midgray_anchor_residual = np.empty((0,), dtype=np.float64)
        if midgray_anchor_weight > 0.0 and anchor_index.size:
            midgray_anchor_residual = midgray_anchor_weight * _normalized_midgray_residual(
                gray[anchor_index[0]],
                reference,
            )
        bias = _stage_two_regularization(values, len(anchor_axis_values), regularization, fit_stretch)
        residuals = np.concatenate((quality_residual, midgray_anchor_residual, bias))
        # log_event(
        #     'neutral_ramp_fit_iteration',
        #     gray=gray,
        #     residuals=residuals,
        #     quality_residual=quality_residual,
        #     midgray_anchor_residual=midgray_anchor_residual,
        #     bias=bias,
        # )
        return residuals

    fit = scipy.optimize.least_squares(
        residual_function,
        np.zeros(5 + (2 if fit_stretch else 0), dtype=np.float64), method='trf', ftol=1e-4, xtol=1e-4, gtol=1e-4, jac='2-point',
    )
    correction = build_correction(fit.x)
    gray, reference = evaluate_neutral_ramp(correction)
    gray_reference = _gray_ramp_reference(
        gray,
        reference,
        anchor_axis_values,
        anchor_axis_value,
    )
    neutral_maes = gray_ramp_neutral_maes(
        gray,
        reference,
        anchor_axis_values,
        anchor_axis_value,
    )
    log_event(
        'gray_ramp',
        gray=gray,
        reference=gray_reference,
        axis_values=np.asarray(anchor_axis_values, dtype=np.float64),
        anchor_axis_value=float(anchor_axis_value),
        neutral_ramp_mae=neutral_maes['neutral_ramp_mae'],
        midgray_neutral_mae=neutral_maes['midgray_neutral_mae'],
        neutral_ramp_refinement=neutral_ramp_refinement,
    )
    return correction


def gray_ramp(
    params,
    ev_ramp,
    density_scale=(1, 1, 1),
    shift_correction=(0, 0, 0),
    stretch_correction=(1, 1, 1),
    *,
    session: RuntimeSimulationSession | None = None,
):
    if session is None:
        session = RuntimeSimulationSession.create(params)
    return session.gray_ramp(
        params.film,
        ev_ramp,
        correction=DensityCurvesCorrection(
            scale=density_scale,
            shift=shift_correction,
            stretch=stretch_correction,
        ),
    )
