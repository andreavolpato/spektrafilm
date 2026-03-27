from __future__ import annotations

from spektrafilm.profiles.io import Profile
from spektrafilm_profile_creator.core.balancing import balance_metameric_neutral, balance_sensitivity
from spektrafilm_profile_creator.core.densitometer import unmix_density
from spektrafilm_profile_creator.core.density_curves import replace_fitted_density_curves
from spektrafilm_profile_creator.core.profile_transforms import (
    adjust_log_exposure,
    align_midscale_neutral_exposures,
    remove_density_min,
)
from spektrafilm_profile_creator.data.loader import (
    load_raw_profile,
)
from spektrafilm_profile_creator.raw_profile import RawProfile
from spektrafilm_profile_creator.reconstruction.dye_reconstruction import reconstruct_dye_density
from spektrafilm_profile_creator.refinement import (
    correct_negative_curves_with_gray_ramp,
    correct_positive_curves_with_gray_ramp,
)


def _resolve_raw_profile(request: RawProfile | str) -> RawProfile:
    if isinstance(request, RawProfile):
        return request
    if isinstance(request, str):
        return load_raw_profile(request)
    raise TypeError('Expected a stock name or RawProfile')


def process_negative_film_profile(
    raw_profile: RawProfile,
) -> Profile:
    recipe = raw_profile.recipe
    gray_ramp_kwargs = dict(recipe.gray_ramp_kwargs)
    profile = raw_profile.as_profile()
    profile = remove_density_min(profile)
    profile = adjust_log_exposure(profile)
    profile = reconstruct_dye_density(profile, model=recipe.dye_density_reconstruct_model)
    profile = unmix_density(profile)
    # profile = replace_fitted_density_curves(profile)
    profile = balance_sensitivity(profile)
    # profile = replace_fitted_density_curves(profile)
    if recipe.reference_channel is not None:
        profile = align_midscale_neutral_exposures(profile, reference_channel=recipe.reference_channel)
    profile = correct_negative_curves_with_gray_ramp(profile, **gray_ramp_kwargs)
    profile = replace_fitted_density_curves(profile)
    profile = adjust_log_exposure(profile)
    return profile


def process_negative_paper_profile(
    raw_profile: RawProfile,
) -> Profile:
    recipe = raw_profile.recipe
    profile = raw_profile.as_profile()
    profile = remove_density_min(profile)
    profile = adjust_log_exposure(profile)
    profile = balance_metameric_neutral(profile)
    profile = unmix_density(profile)
    if recipe.reference_channel is not None:
        profile = align_midscale_neutral_exposures(profile, reference_channel=recipe.reference_channel)
    profile = replace_fitted_density_curves(profile)
    return profile


def process_positive_film_profile(
    raw_profile: RawProfile,
) -> Profile:
    recipe = raw_profile.recipe
    # gray_ramp_kwargs = dict(recipe.gray_ramp_kwargs)
    profile = raw_profile.as_profile()
    profile = remove_density_min(profile)
    profile = adjust_log_exposure(profile)
    profile = balance_metameric_neutral(profile)
    profile = unmix_density(profile)
    if recipe.reference_channel is not None:
        profile = align_midscale_neutral_exposures(profile, reference_channel=recipe.reference_channel)
    profile = replace_fitted_density_curves(profile)
    return profile


def process_raw_profile(raw_profile: RawProfile) -> Profile:
    if raw_profile.info.support == 'film' and raw_profile.info.type == 'negative':
        return process_negative_film_profile(raw_profile)
    if raw_profile.info.support == 'film' and raw_profile.info.type == 'positive':
        return process_positive_film_profile(raw_profile)
    if raw_profile.info.support == 'paper' and raw_profile.info.type == 'negative':
        return process_negative_paper_profile(raw_profile)    
    raise ValueError(
        'Unsupported workflow selection: '
        f'support={raw_profile.info.support}, profile_type={raw_profile.info.type}'
    )   


def process_profile(request: RawProfile | str) -> Profile:
    raw_profile = _resolve_raw_profile(request)
    return process_raw_profile(raw_profile)


__all__ = [
    'RawProfile',
    'load_raw_profile',
    'process_profile',
    'process_negative_film_profile',
    'process_negative_paper_profile',
    'process_positive_film_profile',
]
