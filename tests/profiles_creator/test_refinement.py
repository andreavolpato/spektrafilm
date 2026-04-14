from __future__ import annotations

import numpy as np

import spektrafilm_profile_creator.refinement as refinement_module


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
        data_trustability=1.0,
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
    values = np.array([1.1, 0.9, 1.0, 0.05, -0.02, 0.01], dtype=np.float64)

    regularization = refinement_module._stage_two_regularization(
        values,
        ramp_length=9,
        weights={'scale': 0.35, 'shift': 0.15},
        fit_stretch=False,
    )

    expected = np.array([
        0.35 * 0.1 * 3.0,
        0.35 * -0.1 * 3.0,
        0.0,
        0.15 * 0.05 * 3.0,
        0.15 * -0.02 * 3.0,
        0.15 * 0.01 * 3.0,
    ])

    np.testing.assert_allclose(regularization, expected)


def test_fit_neutral_ramp_trustability_scales_correction_size(monkeypatch) -> None:
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

    high = refinement_module.fit_neutral_ramp(
        evaluate_neutral_ramp,
        anchor,
        data_trustability=1.0,
        regularization={'scale': 0.35, 'shift': 0.15, 'stretch': 1.5},
        anchor_axis_values=(-1, 0, 1, 2),
        anchor_axis_value=0,
        neutral_ramp_refinement=True,
    )
    low = refinement_module.fit_neutral_ramp(
        evaluate_neutral_ramp,
        anchor,
        data_trustability=0.1,
        regularization={'scale': 0.35, 'shift': 0.15, 'stretch': 1.5},
        anchor_axis_values=(-1, 0, 1, 2),
        anchor_axis_value=0,
        neutral_ramp_refinement=True,
    )

    high_scale = np.asarray(high.scale, dtype=np.float64)
    low_scale = np.asarray(low.scale, dtype=np.float64)
    high_shift = np.asarray(high.shift, dtype=np.float64)
    low_shift = np.asarray(low.shift, dtype=np.float64)

    high_shift_norm = np.linalg.norm(high_shift)
    low_shift_norm = np.linalg.norm(low_shift)
    high_total_norm = np.linalg.norm(np.concatenate((high_scale - 1.0, high_shift)))
    low_total_norm = np.linalg.norm(np.concatenate((low_scale - 1.0, low_shift)))

    assert high_shift_norm < low_shift_norm
    assert high_total_norm < low_total_norm


def test_fit_neutral_ramp_uses_shift_deltas_relative_to_anchor(monkeypatch) -> None:
    solver_starts: list[np.ndarray] = []
    evaluated_corrections: list[refinement_module.DensityCurvesCorrection] = []

    def evaluate_neutral_ramp(correction: refinement_module.DensityCurvesCorrection):
        evaluated_corrections.append(correction)
        gray = np.ones((3, 3), dtype=np.float64) * 0.184
        reference = np.array([0.184, 0.184, 0.184], dtype=np.float64)
        return gray, reference

    def fake_least_squares(func, x0):
        x0 = np.asarray(x0, dtype=np.float64)
        solver_starts.append(x0.copy())
        final_values = np.array([1.0, 1.0, 1.0, 0.05, -0.01, -0.02], dtype=np.float64)
        func(final_values)
        return type('Fit', (), {'x': final_values})()

    monkeypatch.setattr(refinement_module, 'log_event', lambda *args, **kwargs: None)
    monkeypatch.setattr(refinement_module.scipy.optimize, 'least_squares', fake_least_squares)

    anchor = refinement_module.DensityCurvesCorrection(shift=(0.2, 0.3, 0.5))

    correction = refinement_module.fit_neutral_ramp(
        evaluate_neutral_ramp,
        anchor,
        data_trustability=1.0,
        regularization={'scale': 0.35, 'shift': 0.15, 'stretch': 1.5},
        anchor_axis_values=(-1, 0, 1),
        anchor_axis_value=0,
        neutral_ramp_refinement=True,
    )

    np.testing.assert_allclose(solver_starts[0], np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64))
    assert evaluated_corrections[-1] == refinement_module.DensityCurvesCorrection(
        scale=(1.0, 1.0, 1.0),
        shift=(0.25, 0.29, 0.48),
        stretch=(1.0, 1.0, 1.0),
    )
    assert correction == evaluated_corrections[-1]


def test_fit_neutral_ramp_maps_red_blue_stretch(monkeypatch) -> None:
    evaluated_corrections: list[refinement_module.DensityCurvesCorrection] = []

    def evaluate_neutral_ramp(correction: refinement_module.DensityCurvesCorrection):
        evaluated_corrections.append(correction)
        gray = np.ones((3, 3), dtype=np.float64) * 0.184
        reference = np.array([0.184, 0.184, 0.184], dtype=np.float64)
        return gray, reference

    def fake_least_squares(func, _x0):
        final_values = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.2, 0.8], dtype=np.float64)
        func(final_values)
        return type('Fit', (), {'x': final_values})()

    monkeypatch.setattr(refinement_module, 'log_event', lambda *args, **kwargs: None)
    monkeypatch.setattr(refinement_module.scipy.optimize, 'least_squares', fake_least_squares)

    anchor = refinement_module.DensityCurvesCorrection()

    correction = refinement_module.fit_neutral_ramp(
        evaluate_neutral_ramp,
        anchor,
        data_trustability=1.0,
        regularization={'scale': 0.35, 'shift': 0.15, 'stretch': 1.5},
        anchor_axis_values=(-1, 0, 1),
        anchor_axis_value=0,
        fit_stretch=True,
        neutral_ramp_refinement=True,
    )

    assert evaluated_corrections[-1].stretch == (1.2, 1.0, 0.8)
    assert correction.stretch == (1.2, 1.0, 0.8)