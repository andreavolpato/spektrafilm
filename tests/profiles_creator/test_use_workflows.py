from __future__ import annotations

import spektrafilm_profile_creator.workflows as workflows_module
from spektrafilm.profiles.io import Profile, ProfileInfo
from spektrafilm_profile_creator import RawProfile, RawProfileRecipe, process_raw_profile


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
        'prelminary_neutral_shift',
        'unmix_density',
        'refine_negative_print',
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
        'prelminary_neutral_shift',
        'unmix_density',
        'refine_negative_print',
        'replace_fitted_density_curves',
    ]


def test_process_raw_profile_routes_print_film_to_optional_neutral_ramp_refinement(monkeypatch) -> None:
    raw_profile = RawProfile(
        info=ProfileInfo(stock='kodak_2383', support='film', use='printing', type='negative'),
        recipe=RawProfileRecipe(target_film='kodak_vision3_250d', neutral_ramp_refinement=True),
    )
    captured_calls: list[tuple[str, bool | None]] = []

    def record_step(name: str):
        def step(profile, *_args, **_kwargs):
            captured_calls.append((name, None))
            return profile

        return step

    def record_print_refine(profile, *_args, **kwargs):
        captured_calls.append(('refine_negative_print', kwargs.get('neutral_ramp_refinement')))
        return profile

    monkeypatch.setattr(workflows_module, 'log_event', lambda *args, **kwargs: None)
    for step_name in [
        'densitometer_normalization',
        'remove_density_min',
        'reconstruct_metameric_neutral',
        'balance_print_sensitivity',
        'prelminary_neutral_shift',
        'unmix_density',
        'replace_fitted_density_curves',
    ]:
        monkeypatch.setattr(workflows_module, step_name, record_step(step_name))
    monkeypatch.setattr(workflows_module, 'refine_negative_print', record_print_refine)

    result = process_raw_profile(raw_profile)

    assert isinstance(result, Profile)
    assert captured_calls == [
        ('densitometer_normalization', None),
        ('remove_density_min', None),
        ('reconstruct_metameric_neutral', None),
        ('balance_print_sensitivity', None),
        ('prelminary_neutral_shift', None),
        ('unmix_density', None),
        ('refine_negative_print', True),
        ('replace_fitted_density_curves', None),
    ]


def test_process_raw_profile_defaults_negative_film_to_no_neutral_ramp_refinement(monkeypatch) -> None:
    raw_profile = RawProfile(info=ProfileInfo(stock='kodak_portra_400', support='film', use='filming', type='negative'))
    captured_neutral_ramp_refinement: list[bool] = []

    def record_step(_name: str):
        def step(profile, *_args, **_kwargs):
            return profile

        return step

    def record_negative_refine(profile, *_args, **kwargs):
        captured_neutral_ramp_refinement.append(kwargs.get('neutral_ramp_refinement'))
        return profile

    monkeypatch.setattr(workflows_module, 'log_event', lambda *args, **kwargs: None)
    for step_name in [
        'reconstruct_dye_density',
        'densitometer_normalization',
        'balance_film_sensitivity',
        'remove_density_min',
        'prelminary_neutral_shift',
        'unmix_density',
        'replace_fitted_density_curves',
    ]:
        monkeypatch.setattr(workflows_module, step_name, record_step(step_name))
    monkeypatch.setattr(workflows_module, 'refine_negative_film', record_negative_refine)

    result = process_raw_profile(raw_profile)

    assert isinstance(result, Profile)
    assert captured_neutral_ramp_refinement == [False]


def test_process_raw_profile_can_disable_positive_film_stage_two_refinement(monkeypatch) -> None:
    raw_profile = RawProfile(
        info=ProfileInfo(stock='kodak_ektachrome_100', support='film', use='filming', type='positive'),
        recipe=RawProfileRecipe(neutral_ramp_refinement=False),
    )
    captured_neutral_ramp_refinement: list[bool] = []

    def record_step(_name: str):
        def step(profile, *_args, **_kwargs):
            return profile

        return step

    def record_positive_refine(profile, *_args, **kwargs):
        captured_neutral_ramp_refinement.append(kwargs.get('neutral_ramp_refinement'))
        return profile

    monkeypatch.setattr(workflows_module, 'log_event', lambda *args, **kwargs: None)
    for step_name in [
        'densitometer_normalization',
        'remove_density_min',
        'reconstruct_metameric_neutral',
        'balance_film_sensitivity',
        'prelminary_neutral_shift',
        'unmix_density',
        'replace_fitted_density_curves',
    ]:
        monkeypatch.setattr(workflows_module, step_name, record_step(step_name))
    monkeypatch.setattr(workflows_module, 'refine_positive_film', record_positive_refine)

    result = process_raw_profile(raw_profile)

    assert isinstance(result, Profile)
    assert captured_neutral_ramp_refinement == [False]