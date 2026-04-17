from __future__ import annotations

import numpy as np

from spektrafilm_profile_creator.refinement import common as refinement_module


def test_fit_gray_anchor_returns_shift_only_correction(monkeypatch) -> None:
    solver_calls: list[np.ndarray] = []

    def fake_least_squares(_func, x0):
        x0 = np.asarray(x0, dtype=np.float64)
        solver_calls.append(x0.copy())
        return type('Fit', (), {'x': np.array([0.2, 0.3, 0.5], dtype=np.float64)})()

    monkeypatch.setattr(refinement_module, 'log_event', lambda *args, **kwargs: None)
    monkeypatch.setattr(refinement_module.scipy.optimize, 'least_squares', fake_least_squares)

    correction = refinement_module.fit_gray_anchor(
        lambda shift: (np.array([0.184, 0.184, 0.184]), np.array([0.184, 0.184, 0.184])),
        shift_weight=0.1,
        log_label='fit_gray_anchor_test',
    )

    np.testing.assert_allclose(solver_calls[0], np.array([0.0, 0.0, 0.0], dtype=np.float64))
    assert correction == refinement_module.DensityCurvesCorrection(
        scale=(1.0, 1.0, 1.0),
        shift=(0.2, 0.3, 0.5),
        stretch=(1.0, 1.0, 1.0),
    )


def test_stage_two_regularization_scales_with_square_root_of_ramp_length() -> None:
    values = np.array([0.1, -0.1, 0.05, -0.02, 0.07], dtype=np.float64)
    stage_two_regularization = refinement_module.__dict__['_stage_two_regularization']

    regularization = stage_two_regularization(
        values,
        ramp_length=9,
        weights={'scale': 0.35, 'shift': 0.15},
        fit_stretch=False,
    )

    expected = np.array([
        0.35 * np.sqrt(np.mean(np.square(np.array([0.1, -0.1, 0.0], dtype=np.float64)))) * 3.0 * np.sqrt(3.0),
        0.15 * np.sqrt(np.mean(np.square(np.array([0.05, -0.02, 0.07], dtype=np.float64)))) * 3.0 * np.sqrt(3.0),
    ])

    np.testing.assert_allclose(regularization, expected)


def test_fit_neutral_ramp_regularization_strength_scales_correction_size(monkeypatch) -> None:
    def evaluate_neutral_ramp(correction: refinement_module.DensityCurvesCorrection):
        ev = np.asarray((-1, 0, 1, 2), dtype=np.float64)
        gray = np.ones((len(ev), 3), dtype=np.float64) * 0.184
        gray[:, 0] += 0.030 + 0.07 * (correction.scale[0] - 1.0) + 0.10 * correction.shift[0]
        gray[:, 2] -= 0.025 + 0.07 * (correction.scale[2] - 1.0) + 0.10 * correction.shift[2]
        gray[:, 0] += 0.03 * (correction.stretch[0] - 1.0) * ev
        gray[:, 2] -= 0.03 * (correction.stretch[2] - 1.0) * ev
        return gray, np.array([0.184, 0.184, 0.184], dtype=np.float64)

    monkeypatch.setattr(refinement_module, 'log_event', lambda *args, **kwargs: None)
    anchor = refinement_module.DensityCurvesCorrection()

    strongly_regularized = refinement_module.fit_neutral_ramp(
        evaluate_neutral_ramp,
        anchor,
        regularization={'scale': 0.35, 'shift': 0.15, 'stretch': 1.5},
        anchor_axis_values=(-1, 0, 1, 2),
        anchor_axis_value=0,
        neutral_ramp_refinement=True,
    )
    weakly_regularized = refinement_module.fit_neutral_ramp(
        evaluate_neutral_ramp,
        anchor,
        regularization={'scale': 0.035, 'shift': 0.015, 'stretch': 0.15},
        anchor_axis_values=(-1, 0, 1, 2),
        anchor_axis_value=0,
        neutral_ramp_refinement=True,
    )

    strong_scale = np.asarray(strongly_regularized.scale, dtype=np.float64)
    weak_scale = np.asarray(weakly_regularized.scale, dtype=np.float64)
    strong_shift = np.asarray(strongly_regularized.shift, dtype=np.float64)
    weak_shift = np.asarray(weakly_regularized.shift, dtype=np.float64)

    strong_shift_norm = np.linalg.norm(strong_shift)
    weak_shift_norm = np.linalg.norm(weak_shift)
    strong_total_norm = np.linalg.norm(np.concatenate((strong_scale - 1.0, strong_shift)))
    weak_total_norm = np.linalg.norm(np.concatenate((weak_scale - 1.0, weak_shift)))

    assert strong_shift_norm < weak_shift_norm
    assert strong_total_norm < weak_total_norm


def test_fit_neutral_ramp_maps_centered_scale_and_three_channel_shift_deltas(monkeypatch) -> None:
    solver_starts: list[np.ndarray] = []
    evaluated_corrections: list[refinement_module.DensityCurvesCorrection] = []

    def evaluate_neutral_ramp(correction: refinement_module.DensityCurvesCorrection):
        evaluated_corrections.append(correction)
        gray = np.ones((3, 3), dtype=np.float64) * 0.184
        reference = np.array([0.184, 0.184, 0.184], dtype=np.float64)
        return gray, reference

    def fake_least_squares(func, x0, **_kwargs):
        x0 = np.asarray(x0, dtype=np.float64)
        solver_starts.append(x0.copy())
        final_values = np.array([0.05, -0.01, 0.02, -0.03, 0.01], dtype=np.float64)
        func(final_values)
        return type('Fit', (), {'x': final_values})()

    monkeypatch.setattr(refinement_module, 'log_event', lambda *args, **kwargs: None)
    monkeypatch.setattr(refinement_module.scipy.optimize, 'least_squares', fake_least_squares)

    anchor = refinement_module.DensityCurvesCorrection()

    correction = refinement_module.fit_neutral_ramp(
        evaluate_neutral_ramp,
        anchor,
        regularization={'scale': 0.35, 'shift': 0.15, 'stretch': 1.5},
        anchor_axis_values=(-1, 0, 1),
        anchor_axis_value=0,
        neutral_ramp_refinement=True,
    )

    np.testing.assert_allclose(solver_starts[0], np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64))
    np.testing.assert_allclose(evaluated_corrections[-1].scale, (1.05, 0.99, 0.96))
    np.testing.assert_allclose(evaluated_corrections[-1].shift, (0.02, -0.03, 0.01))
    np.testing.assert_allclose(evaluated_corrections[-1].stretch, (1.0, 1.0, 1.0))
    assert correction == evaluated_corrections[-1]


def test_fit_neutral_ramp_preserves_anchor_shift_mean(monkeypatch) -> None:
    evaluated_corrections: list[refinement_module.DensityCurvesCorrection] = []

    def evaluate_neutral_ramp(correction: refinement_module.DensityCurvesCorrection):
        evaluated_corrections.append(correction)
        gray = np.ones((3, 3), dtype=np.float64) * 0.184
        reference = np.array([0.184, 0.184, 0.184], dtype=np.float64)
        return gray, reference

    def fake_least_squares(func, x0, **_kwargs):
        x0 = np.asarray(x0, dtype=np.float64)
        func(x0)
        return type('Fit', (), {'x': x0})()

    monkeypatch.setattr(refinement_module, 'log_event', lambda *args, **kwargs: None)
    monkeypatch.setattr(refinement_module.scipy.optimize, 'least_squares', fake_least_squares)

    anchor = refinement_module.DensityCurvesCorrection(shift=(0.2, 0.3, 0.5))

    correction = refinement_module.fit_neutral_ramp(
        evaluate_neutral_ramp,
        anchor,
        regularization={'scale': 0.35, 'shift': 0.15},
        anchor_axis_values=(-1, 0, 1),
        anchor_axis_value=0,
        neutral_ramp_refinement=True,
    )

    assert evaluated_corrections[-1].shift == (0.2, 0.3, 0.5)
    assert correction.shift == (0.2, 0.3, 0.5)


def test_fit_neutral_ramp_maps_red_blue_stretch(monkeypatch) -> None:
    evaluated_corrections: list[refinement_module.DensityCurvesCorrection] = []

    def evaluate_neutral_ramp(correction: refinement_module.DensityCurvesCorrection):
        evaluated_corrections.append(correction)
        gray = np.ones((3, 3), dtype=np.float64) * 0.184
        reference = np.array([0.184, 0.184, 0.184], dtype=np.float64)
        return gray, reference

    def fake_least_squares(func, _x0, **_kwargs):
        final_values = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.2, -0.2], dtype=np.float64)
        func(final_values)
        return type('Fit', (), {'x': final_values})()

    monkeypatch.setattr(refinement_module, 'log_event', lambda *args, **kwargs: None)
    monkeypatch.setattr(refinement_module.scipy.optimize, 'least_squares', fake_least_squares)

    anchor = refinement_module.DensityCurvesCorrection()

    correction = refinement_module.fit_neutral_ramp(
        evaluate_neutral_ramp,
        anchor,
        regularization={'scale': 0.35, 'shift': 0.15, 'stretch': 1.5},
        anchor_axis_values=(-1, 0, 1),
        anchor_axis_value=0,
        fit_stretch=True,
        neutral_ramp_refinement=True,
    )

    assert evaluated_corrections[-1].stretch == (1.2, 0.8, 1.0)
    assert correction.stretch == (1.2, 0.8, 1.0)


def test_fit_neutral_ramp_adds_weighted_midgray_anchor_residual(monkeypatch) -> None:
    captured_residuals: list[np.ndarray] = []

    def evaluate_neutral_ramp(_correction: refinement_module.DensityCurvesCorrection):
        gray = np.array([
            [0.184, 0.184, 0.184],
            [0.204, 0.174, 0.184],
            [0.184, 0.184, 0.184],
        ], dtype=np.float64)
        reference = np.array([0.184, 0.184, 0.184], dtype=np.float64)
        return gray, reference

    def fake_least_squares(func, x0, **_kwargs):
        residuals = np.asarray(func(np.asarray(x0, dtype=np.float64)), dtype=np.float64)
        captured_residuals.append(residuals)
        return type('Fit', (), {'x': np.asarray(x0, dtype=np.float64)})()

    monkeypatch.setattr(refinement_module, 'log_event', lambda *args, **kwargs: None)
    monkeypatch.setattr(refinement_module.scipy.optimize, 'least_squares', fake_least_squares)

    refinement_module.fit_neutral_ramp(
        evaluate_neutral_ramp,
        refinement_module.DensityCurvesCorrection(),
        regularization={'scale': 0.35, 'shift': 0.15, 'midgray_anchor_weight': 2.0},
        anchor_axis_values=(-1, 0, 1),
        anchor_axis_value=0,
        neutral_ramp_refinement=True,
    )

    expected_anchor_residual = 2.0 * refinement_module.__dict__['_normalized_midgray_residual'](
        np.array([0.204, 0.174, 0.184], dtype=np.float64),
        np.array([0.184, 0.184, 0.184], dtype=np.float64),
    )
    np.testing.assert_allclose(captured_residuals[0][-5:-2], expected_anchor_residual)


def test_fit_neutral_ramp_logs_final_neutral_reference_ramp(monkeypatch) -> None:
    logged_events: list[tuple[str, dict[str, object]]] = []

    def evaluate_neutral_ramp(_correction: refinement_module.DensityCurvesCorrection):
        gray = np.array([
            [0.10, 0.11, 0.09],
            [0.18, 0.19, 0.17],
            [0.30, 0.33, 0.27],
        ], dtype=np.float64)
        reference = np.array([0.184, 0.184, 0.184], dtype=np.float64)
        return gray, reference

    def fake_least_squares(func, x0, **_kwargs):
        x0 = np.asarray(x0, dtype=np.float64)
        func(x0)
        return type('Fit', (), {'x': x0})()

    monkeypatch.setattr(
        refinement_module,
        'log_event',
        lambda title, *args, **kwargs: logged_events.append((title, kwargs)),
    )
    monkeypatch.setattr(refinement_module.scipy.optimize, 'least_squares', fake_least_squares)

    refinement_module.fit_neutral_ramp(
        evaluate_neutral_ramp,
        refinement_module.DensityCurvesCorrection(),
        regularization={'scale': 0.35, 'shift': 0.15},
        anchor_axis_values=(-1, 0, 1),
        anchor_axis_value=0,
        neutral_ramp_refinement=True,
    )

    assert logged_events[-1][0] == 'gray_ramp'
    np.testing.assert_allclose(
        logged_events[-1][1]['reference'],
        np.array([
            [0.10, 0.10, 0.10],
            [0.184, 0.184, 0.184],
            [0.30, 0.30, 0.30],
        ], dtype=np.float64),
    )
    expected_maes = refinement_module.gray_ramp_neutral_maes(
        np.array([
            [0.10, 0.11, 0.09],
            [0.18, 0.19, 0.17],
            [0.30, 0.33, 0.27],
        ], dtype=np.float64),
        np.array([0.184, 0.184, 0.184], dtype=np.float64),
        (-1, 0, 1),
        0,
    )
    assert logged_events[-1][1]['neutral_ramp_mae'] == expected_maes['neutral_ramp_mae']
    assert logged_events[-1][1]['midgray_neutral_mae'] == expected_maes['midgray_neutral_mae']