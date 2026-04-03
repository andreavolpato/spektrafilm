from __future__ import annotations

from spektrafilm.profiles.io import Profile
from spektrafilm_profile_creator.core.balancing import reconstruct_metameric_neutral, balance_sensitivity
from spektrafilm_profile_creator.core.densitometer import unmix_density, densitometer_normalization
from spektrafilm_profile_creator.core.density_curves import replace_fitted_density_curves
from spektrafilm_profile_creator.core.profile_transforms import (
    adjust_log_exposure,
    adjust_log_exposure_midgray_to_metameric_neutral,
    remove_density_min,
)
from spektrafilm_profile_creator.data.loader import (
    load_raw_profile,
)
from spektrafilm_profile_creator.diagnostics.messages import log_event
from spektrafilm_profile_creator.raw_profile import RawProfile
from spektrafilm_profile_creator.reconstruction.dye_reconstruction import reconstruct_dye_density
from spektrafilm_profile_creator.refinement import (
    correct_negative_curves_with_gray_ramp,
    correct_positive_curves_with_gray_ramp,
)


def process_raw_profile(raw_profile: RawProfile) -> Profile:
    recipe = raw_profile.recipe
    profile = raw_profile.as_profile()
    log_event('unprocessed_profile', profile)    
    
    #########################################################################################################
    # negative film workflow
    #########################################################################################################
    if raw_profile.info.support == 'film' and raw_profile.info.type == 'negative':
        # channel density
        profile = reconstruct_dye_density(profile, model=recipe.dye_density_reconstruct_model)
        profile = densitometer_normalization(profile)
        # sensitivity
        profile = balance_sensitivity(profile)
        # density curves
        profile = remove_density_min(profile)
        profile = unmix_density(profile)
        #if recipe.reference_channel is not None:
            #profile = align_midscale_neutral_exposures(profile, reference_channel=recipe.reference_channel)
        # TODO decide on master negative and filters reference values
        profile = correct_negative_curves_with_gray_ramp(
            profile,
            target_paper=recipe.target_paper,
            data_trustability=recipe.data_trustability,
            stretch_curves=recipe.stretch_curves,
        )
        profile = adjust_log_exposure(profile) # TODO fix with density curves interpolation
        profile = replace_fitted_density_curves(profile)
        #profile = adjust_log_exposure(profile)
        return profile

    ##########################################################################################################
    # positive film workflow
    ##########################################################################################################
    if raw_profile.info.support == 'film' and raw_profile.info.type == 'positive':
        # channel density
        profile = densitometer_normalization(profile)
        profile = remove_density_min(profile, reconstruct_base_density=True) # affect also density curves
        profile = reconstruct_metameric_neutral(profile)
        # sensitivity
        profile = balance_sensitivity(profile)
        # density curves
        profile = unmix_density(profile)
        profile = adjust_log_exposure_midgray_to_metameric_neutral(profile)
        profile = correct_positive_curves_with_gray_ramp(
            profile,
            data_trustability=recipe.data_trustability,
            stretch_curves=recipe.stretch_curves,
        )
        profile = replace_fitted_density_curves(profile)
        return profile

    ##########################################################################################################
    # negative paper workflow
    ##########################################################################################################
    if raw_profile.info.support == 'paper' and raw_profile.info.type == 'negative':
        # channel density
        profile = densitometer_normalization(profile)
        profile = remove_density_min(profile, reconstruct_base_density=True) # affect also density curves
        profile = reconstruct_metameric_neutral(profile)
        # sensitivity
        # TODO use master film to balance sensitivity to the correct kodak cc values
        # density curves
        profile = unmix_density(profile)
        profile = adjust_log_exposure_midgray_to_metameric_neutral(profile)
        profile = replace_fitted_density_curves(profile)
        return profile

    
    raise ValueError(
        'Unsupported workflow selection: '
        f'support={raw_profile.info.support}, profile_type={raw_profile.info.type}'
    )


def process_profile(stock: str) -> Profile:
    raw_profile = load_raw_profile(stock)
    return process_raw_profile(raw_profile)


__all__ = [
    'RawProfile',
    'load_raw_profile',
    'process_profile',
    'process_raw_profile',
]
