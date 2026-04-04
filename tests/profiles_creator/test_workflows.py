from __future__ import annotations

import spektrafilm_profile_creator.workflows as workflows_module
from spektrafilm.profiles.io import Profile
from spektrafilm_profile_creator import load_raw_profile, process_profile, process_raw_profile


def _record_workflow_steps(monkeypatch, step_names: list[str], captured_steps: list[str]) -> None:
    def record_step(name: str):
        def step(profile, *_args, **_kwargs):
            captured_steps.append(name)
            return profile

        return step

    monkeypatch.setattr(workflows_module, 'log_event', lambda *args, **kwargs: None)
    for step_name in step_names:
        monkeypatch.setattr(workflows_module, step_name, record_step(step_name))


def test_process_raw_profile_returns_profile_for_negative_film(portra_400_processed_profile) -> None:
    case, profile = portra_400_processed_profile

    assert isinstance(profile, Profile)
    assert profile.info.stock == case.stock


def test_process_profile_dispatches_from_stock_string(monkeypatch) -> None:
    raw_profile = object()
    expected_profile = object()
    calls: list[tuple[str, object]] = []

    def fake_load_raw_profile(stock: str):
        calls.append(('load', stock))
        return raw_profile

    def fake_process_raw_profile(value):
        calls.append(('process', value))
        return expected_profile

    monkeypatch.setattr(workflows_module, 'load_raw_profile', fake_load_raw_profile)
    monkeypatch.setattr(workflows_module, 'process_raw_profile', fake_process_raw_profile)

    result = process_profile('kodak_portra_400')

    assert calls == [('load', 'kodak_portra_400'), ('process', raw_profile)]
    assert result is expected_profile


def test_process_raw_profile_accepts_loaded_raw_profile(monkeypatch) -> None:
    raw_profile = load_raw_profile('kodak_portra_400')
    captured_steps: list[str] = []
    _record_workflow_steps(
        monkeypatch,
        [
            'reconstruct_dye_density',
            'densitometer_normalization',
            'balance_sensitivity',
            'remove_density_min',
            'unmix_density',
            'correct_negative_curves_with_gray_ramp',
            'adjust_log_exposure',
            'replace_fitted_density_curves',
        ],
        captured_steps,
    )

    result = process_raw_profile(raw_profile)

    assert isinstance(result, Profile)
    assert result.info.stock == raw_profile.info.stock
    assert captured_steps == [
        'reconstruct_dye_density',
        'densitometer_normalization',
        'balance_sensitivity',
        'remove_density_min',
        'unmix_density',
        'correct_negative_curves_with_gray_ramp',
        'adjust_log_exposure',
        'replace_fitted_density_curves',
    ]