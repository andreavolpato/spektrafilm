from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from opt_einsum import contract
from scipy.ndimage import gaussian_filter1d

from spektrafilm.config import SPECTRAL_SHAPE
from spektrafilm.data.profiles_loader import DensityCurvesModel
from spektrafilm.model.couplers import apply_density_correction_dir_couplers
from spektrafilm.model.density_curves import interpolate_exposure_to_density
from spektrafilm.model.grain import apply_grain
from spektrafilm.runtime.params_schema import DirCouplersParams, GrainParams
from spektrafilm.utils.conversions import density_to_light
from spektrafilm.utils.morph_curves import (
    PrintChemistryParams,
    apply_print_curves_morph,
)

FloatArray: TypeAlias = NDArray[np.float64]
ProfileType: TypeAlias = Literal["negative", "positive"]

################################################################################
# Emulsion helpers


def base_print_density_tuning(base_density, base_density_params):
    # Color: spectral base from the 3 channel mins at the status-A peaks.
    wavelengths = SPECTRAL_SHAPE.wavelengths
    status_a_max_peak = [445, 530, 610]
    base_density_channels = np.interp(status_a_max_peak, wavelengths, base_density)
    base_density_channels *= [
        base_density_params.yellow,
        base_density_params.magenta,
        base_density_params.cyan,
    ]
    spectral_min = np.interp(wavelengths, status_a_max_peak, base_density_channels)
    sigma_nm = 20
    sigma_points = sigma_nm / np.mean(np.diff(wavelengths))
    return gaussian_filter1d(spectral_min, sigma_points)


def base_film_density_tuning(base_density, base_density_params):
    """film base tuning"""
    # density shift (using CMY params)
    wavelengths = SPECTRAL_SHAPE.wavelengths
    status_m_max_peaks = [460, 555, 650]
    sigma_nm = 35
    sigma_points = sigma_nm / np.mean(np.diff(wavelengths))
    density_scale_channels = [
        base_density_params.yellow,
        base_density_params.magenta,
        base_density_params.cyan,
    ]
    spectral_density_shift = np.interp(
        wavelengths, status_m_max_peaks, density_scale_channels
    )
    density_scale = gaussian_filter1d(spectral_density_shift, sigma_points)
    # density tilt (1.0 > +0.1 density at 650nm)
    tilt_density_at_650 = base_density_params.tilt
    tilt_slope = tilt_density_at_650 / (650 - 555)
    density_tilt = tilt_slope * (wavelengths - 555) + 1
    density_tilt = np.clip(density_tilt, 0, np.inf)
    density_tilt = gaussian_filter1d(density_tilt, sigma_points)
    return base_density * base_density_params.scale * density_scale * density_tilt


def _tuned_base_density(base_density, base_density_params, is_film=False):
    """Apply the optional base-density tuning. With no params or tuning
    disabled, the base is used unchanged (the case for the film base on the
    film→print exposure path, where the print-paper base controls do not
    apply); neutral channels (all 1.0) skip the resample and only scale."""
    if base_density_params is None or not base_density_params.active:
        return base_density
    cmy_neutral = (
        base_density_params.cyan,
        base_density_params.magenta,
        base_density_params.yellow,
    ) == (1.0, 1.0, 1.0)
    # tilt is a film-base-only knob; PrintBaseParams has no tilt field.
    neutral = cmy_neutral and (base_density_params.tilt == 0.0 if is_film else True)
    if neutral:
        tuned = base_density
    else:
        if is_film:
            tuned = base_film_density_tuning(base_density, base_density_params)
        else:
            tuned = base_print_density_tuning(base_density, base_density_params)
    return tuned * base_density_params.scale


def compute_density_spectral(
    channel_density,
    density_cmy,
    base_density=None,
    base_density_params=None,
    is_film=False,
):
    density_spectral = contract(
        "ijk, lk->ijl", density_cmy, np.asarray(channel_density)
    )
    if base_density is None:
        return density_spectral
    return density_spectral + _tuned_base_density(
        base_density, base_density_params, is_film
    )


def exposure_factor(sensitivity, print_illuminant, density_spectral_midgray):
    """Print-exposure normalization factor: the reciprocal geometric-mean
    exposure of a reference midgray spectral density under the filtered
    enlarger light. Shared by the printing stage and the profile creator's
    print refinement."""
    light_midgray = density_to_light(density_spectral_midgray, print_illuminant)
    raw_midgray = contract("ijk, kl->ijl", light_midgray, sensitivity)
    raw_midgray = np.fmax(raw_midgray, 1e-10)
    # use the geometric mean to normalize the exposure
    raw_midgray_geomean = np.exp(np.mean(np.log(raw_midgray), axis=2, keepdims=True))
    return 1 / raw_midgray_geomean


# Keep black/white reference correction anchored to the stock density curves.
# Creative print-curve morphing belongs in develop_print_morph().
def develop_simple(
    log_raw,
    log_exposure,
    density_curves,
    gamma_factor=1.0,
):
    density_cmy = interpolate_exposure_to_density(
        log_raw, density_curves, log_exposure, gamma_factor
    )
    return density_cmy


def develop(
    log_raw: FloatArray,
    pixel_size_um: float,
    log_exposure: FloatArray,
    density_curves: FloatArray,
    density_curves_layers: FloatArray,
    dir_couplers: DirCouplersParams,
    grain: GrainParams,
    profile_type: ProfileType,
    gamma_factor: float = 1.0,
    bypass_grain: bool = False,
    use_fast_stats: bool = False,
) -> FloatArray:
    density_curves = np.asarray(density_curves)
    normalized_density_curves = density_curves - np.nanmin(density_curves, axis=0)

    density_cmy = develop_simple(
        log_raw,
        log_exposure,
        normalized_density_curves,
        gamma_factor=gamma_factor,
    )
    density_cmy = apply_density_correction_dir_couplers(
        density_cmy,
        log_raw,
        pixel_size_um,
        log_exposure,
        normalized_density_curves,
        dir_couplers,
        profile_type,
        gamma_factor=gamma_factor,
    )
    return apply_grain(
        density_cmy,
        pixel_size_um,
        grain,
        normalized_density_curves,
        density_curves_layers,
        profile_type,
        bypass_grain=bypass_grain,
        use_fast_stats=use_fast_stats,
    )


def develop_print_morph(
    log_raw: FloatArray,
    log_exposure: FloatArray,
    density_curves_model: DensityCurvesModel,
    chemistry: PrintChemistryParams,
    profile_type: ProfileType = "negative",
):
    density_curves_morphed = apply_print_curves_morph(
        log_exposure,
        density_curves_model,
        chemistry,
        profile_type=profile_type,
    )
    density_cmy = interpolate_exposure_to_density(
        log_raw,
        density_curves_morphed,
        log_exposure,
        gamma_factor=1.0,
    )
    return density_cmy


# Some future work notes:
# Add print dye shift in nanometers for dye absorption peaks.
# Investigate how density curves change with development conditions.
# Add a gray card border to check white balance.

if __name__ == "__main__":
    pass
