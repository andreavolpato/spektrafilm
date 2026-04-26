from __future__ import annotations

from spektrafilm.profiles.io import Profile
from spektrafilm_profile_creator.core.balancing import (
    reconstruct_metameric_neutral,
    balance_film_sensitivity,
    balance_print_sensitivity,
    prelminary_neutral_shift
)
from spektrafilm_profile_creator.core.densitometer import (
    unmix_density,
    densitometer_normalization,
    fill_missing_sensitivity
)
from spektrafilm_profile_creator.core.density_curves import replace_fitted_density_curves
from spektrafilm_profile_creator.core.profile_transforms import (
    remove_density_min,
)
from spektrafilm_profile_creator.data.loader import (
    load_raw_profile,
)
from spektrafilm_profile_creator.diagnostics.messages import log_event
from spektrafilm_profile_creator.raw_profile import RawProfile
from spektrafilm_profile_creator.reconstruction.dye_reconstruction import reconstruct_dye_density
from spektrafilm_profile_creator.refinement import (
    refine_negative_film,
    refine_negative_print,
    refine_positive_film,
)
from spektrafilm_profile_creator.spectral_containment import sensitivity_bandpass_hanatos2025


def process_raw_profile(raw_profile: RawProfile) -> Profile:
    recipe = raw_profile.recipe
    profile = raw_profile.as_profile()
    log_event('unprocessed_profile', profile)    
    
    #########################################################################################################
    # negative film workflow
    #########################################################################################################
    if raw_profile.info.stage == 'filming' and raw_profile.info.type == 'negative':
        # channel density
        profile = reconstruct_dye_density(profile, model=recipe.dye_density_reconstruct_model)
        profile = densitometer_normalization(profile)
        # density curves
        profile = remove_density_min(profile)
        profile = prelminary_neutral_shift(profile)
        profile = unmix_density(profile)
        # sensitivity
        profile = fill_missing_sensitivity(profile)
        profile = balance_film_sensitivity(profile)
        # final refinement
        profile = refine_negative_film(
            profile,
            target_print=raw_profile.info.target_print,
            stretch_curves=recipe.stretch_curves,
            neutral_ramp_refinement=recipe.neutral_ramp_refinement,
        )
        profile = replace_fitted_density_curves(profile)
        profile = sensitivity_bandpass_hanatos2025(profile)
        return profile

    ##########################################################################################################
    # positive film workflow
    ##########################################################################################################
    if raw_profile.info.stage == 'filming' and raw_profile.info.type == 'positive':
        # channel density
        profile = remove_density_min(profile, reconstruct_base_density=True) # affect also density curves
        profile = reconstruct_metameric_neutral(profile)
        profile = densitometer_normalization(profile)
        # density curves
        profile = prelminary_neutral_shift(profile, per_channel_shift=False)
        profile = unmix_density(profile)
        # sensitivity
        profile = fill_missing_sensitivity(profile)
        profile = balance_film_sensitivity(profile)
        # final refinement
        profile = refine_positive_film(
            profile,
            stretch_curves=recipe.stretch_curves,
            neutral_ramp_refinement=recipe.neutral_ramp_refinement,
        )
        profile = replace_fitted_density_curves(profile)
        profile = sensitivity_bandpass_hanatos2025(profile)
        return profile

    ##########################################################################################################
    # negative paper workflow
    ##########################################################################################################
    if raw_profile.info.stage == 'printing' and raw_profile.info.type == 'negative':
        # channel density
        profile = remove_density_min(profile, reconstruct_base_density=True) # affect also density curves
        profile = reconstruct_metameric_neutral(profile)
        profile = densitometer_normalization(profile)
        # density curves
        profile = prelminary_neutral_shift(profile, per_channel_shift=recipe.neutral_log_exposure_correction)
        profile = unmix_density(profile)
        # sensitivity
        profile = fill_missing_sensitivity(profile)
        profile = balance_print_sensitivity(profile, target_film=recipe.target_film)
        # final refinement
        profile = refine_negative_print(
            profile,
            target_film=recipe.target_film,
            neutral_ramp_refinement=recipe.neutral_ramp_refinement,
        )
        profile = replace_fitted_density_curves(profile)
        return profile
    
    raise NotImplementedError(f"Workflow not implemented for profile type '{raw_profile.info.type}' and stage '{raw_profile.info.stage}' combination.")

def process_profile(stock: str) -> Profile:
    raw_profile = load_raw_profile(stock)
    return process_raw_profile(raw_profile)


__all__ = [
    'RawProfile',
    'load_raw_profile',
    'process_profile',
    'process_raw_profile',
]
