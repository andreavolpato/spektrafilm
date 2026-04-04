from __future__ import annotations

import spektrafilm_profile_creator.workflows as workflows_module
from spektrafilm.profiles.io import Profile, ProfileInfo
from spektrafilm_profile_creator import RawProfile, process_raw_profile


def test_process_raw_profile_routes_print_film_to_printing_workflow(monkeypatch) -> None:
    raw_profile = RawProfile(info=ProfileInfo(stock='kodak_2383', support='film', use='printing', type='negative'))
    captured_steps: list[str] = []

    def record_step(name: str):
        def step(profile, *_args, **_kwargs):
            captured_steps.append(name)
            return profile

        return step

    monkeypatch.setattr(workflows_module, 'log_event', lambda *args, **kwargs: None)
    for step_name in [
        'densitometer_normalization',
        'remove_density_min',
        'reconstruct_metameric_neutral',
        'balance_print_sensitivity',
        'unmix_density',
        'adjust_log_exposure_midgray_to_metameric_neutral',
        'replace_fitted_density_curves',
    ]:
        monkeypatch.setattr(workflows_module, step_name, record_step(step_name))

    result = process_raw_profile(raw_profile)

    assert isinstance(result, Profile)
    assert captured_steps == [
        'densitometer_normalization',
        'remove_density_min',
        'reconstruct_metameric_neutral',
        'balance_print_sensitivity',
        'unmix_density',
        'adjust_log_exposure_midgray_to_metameric_neutral',
        'replace_fitted_density_curves',
    ]