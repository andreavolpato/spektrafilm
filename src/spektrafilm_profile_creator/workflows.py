from __future__ import annotations

from spektrafilm.profiles.io import Profile
from spektrafilm_profile_creator.core.balancing import (
    reconstruct_metameric_neutral, balance_film_sensitivity,
    balance_print_sensitivity,
    preliminary_match_density_curves_to_midscale_neutral_minus_base
)
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
    refine_negative_curves_with_gray_ramp,
    refine_positive_curves_with_gray_ramp,
    refine_negative_print_profile_with_neutral_ramp,
)


def process_raw_profile(raw_profile: RawProfile) -> Profile:
    recipe = raw_profile.recipe
    profile = raw_profile.as_profile()
    log_event('unprocessed_profile', profile)    
    
    #########################################################################################################
    # negative film workflow
    #########################################################################################################
    if raw_profile.info.use == 'filming' and raw_profile.info.type == 'negative':
        # channel density
        profile = reconstruct_dye_density(profile, model=recipe.dye_density_reconstruct_model)
        profile = densitometer_normalization(profile)
        # sensitivity
        profile = balance_film_sensitivity(profile)
        # density curves
        profile = remove_density_min(profile)
        profile = preliminary_match_density_curves_to_midscale_neutral_minus_base(profile)
        profile = unmix_density(profile)
        # TODO decide on master negative and filters reference values
        profile = refine_negative_curves_with_gray_ramp(
            profile,
            target_print=recipe.target_print,
            data_trustability=recipe.data_trustability,
            stretch_curves=recipe.stretch_curves,
        )
        # profile = adjust_log_exposure(profile) # TODO fix with density curves interpolation
        profile = replace_fitted_density_curves(profile)
        #profile = adjust_log_exposure(profile) # TODO make sure log_exposure is correct abd uniform across stocks
        return profile

    ##########################################################################################################
    # positive film workflow
    ##########################################################################################################
    if raw_profile.info.use == 'filming' and raw_profile.info.type == 'positive':
        # channel density
        profile = densitometer_normalization(profile)
        profile = remove_density_min(profile, reconstruct_base_density=True) # affect also density curves
        profile = reconstruct_metameric_neutral(profile)
        # sensitivity
        profile = balance_film_sensitivity(profile)
        # density curves
        profile = preliminary_match_density_curves_to_midscale_neutral_minus_base(profile,
                        correct_log_exposure_per_channel=True)
        profile = unmix_density(profile)
        profile = refine_positive_curves_with_gray_ramp(
            profile,
            data_trustability=recipe.data_trustability,
        )
        profile = replace_fitted_density_curves(profile)
        return profile

    ##########################################################################################################
    # negative paper workflow
    ##########################################################################################################
    if raw_profile.info.use == 'printing' and raw_profile.info.type == 'negative':
        # channel density
        profile = densitometer_normalization(profile)
        profile = remove_density_min(profile, reconstruct_base_density=True) # affect also density curves
        profile = reconstruct_metameric_neutral(profile)
        # sensitivity
        profile = balance_print_sensitivity(profile, target_film=recipe.target_film)
        # density curves
        profile = preliminary_match_density_curves_to_midscale_neutral_minus_base(profile, 
                        correct_log_exposure_per_channel=recipe.neutral_log_exposure_correction)
        profile = unmix_density(profile)
        profile = refine_negative_print_profile_with_neutral_ramp(profile,
                                                                  target_film=recipe.target_film,
                                                                  data_trustability=recipe.data_trustability,
                                                                  )
        profile = replace_fitted_density_curves(profile)
        return profile
    
    raise NotImplementedError(f"Workflow not implemented for profile type '{raw_profile.info.type}' and use '{raw_profile.info.use}' combination.")

def process_profile(stock: str) -> Profile:
    raw_profile = load_raw_profile(stock)
    return process_raw_profile(raw_profile)


__all__ = [
    'RawProfile',
    'load_raw_profile',
    'process_profile',
    'process_raw_profile',
]
