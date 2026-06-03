"""
s023 coupled-gamma print-curve morph.

Seven user-facing controls, all built on the same coupled (sigma, mu)
gamma scaling primitive:

    sigma' = sigma / gamma
    mu'    = mu    / gamma
    A'     = A

Layer mapping is by GRAIN SPEED (sensitivity threshold), not by D-logE
curve position:
  - fast sub-layer = layer with the lowest mu (lowest threshold, develops
    on the least exposure). For negative profiles this is the toe-dominant
    layer; for positive profiles it is the shoulder-dominant layer.
  - slow sub-layer = layer with the highest mu (highest threshold).
  - mid             = the middle one.

This naming follows the underlying grain population, which is the same
physical entity regardless of how the film is meant to be used.

Per-channel effective gamma is composed multiplicatively from the global
gamma factor, the per-band (fast/slow) gamma factors, and the per-channel
RGB gamma factors:

    effective_gamma_fast_R = gamma_factor * gamma_factor_fast * gamma_factor_red
    effective_gamma_fast_G = gamma_factor * gamma_factor_fast * gamma_factor_green
    effective_gamma_fast_B = gamma_factor * gamma_factor_fast * gamma_factor_blue
    effective_gamma_mid_R  = gamma_factor * gamma_factor_slow * gamma_factor_red
    effective_gamma_mid_G  = gamma_factor * gamma_factor_slow * gamma_factor_green
    effective_gamma_mid_B  = gamma_factor * gamma_factor_slow * gamma_factor_blue
    effective_gamma_slow_* = same as mid

The developer exhaustion control blends every sub-layer toward the matched
Gumbel-max CDF, uniformly across all three sub-layers of each channel. A
common horizontal offset is then applied to all three sub-layers of each
channel so developer exhaustion does not move midgray.

Properties preserved by construction:
    - A_i per sub-layer (silver+coupler) is exactly fixed
    - D(0) per channel is preserved exactly, including with developer exhaustion
    - D_max = sum A_i is preserved exactly
    - Identity at all defaults is bit-exact
"""

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import brentq as _brentq

from spektrafilm.model.density_curves import _layer_cdf_values
from spektrafilm.profiles.io import DensityCurvesModel


__all__ = [
    "PrintChemistryParams",
    "FilmChemistryParams",
    "apply_print_curves_morph",
    "apply_print_curves_morph_with_layers",
]


SIGMA_FLOOR = 0.05  # matches NormCdfsFitConfig.sigma_floor in profile-creator
_GUMBEL_LOCATION = -math.log(math.log(2.0))
_GUMBEL_WIDTH = 0.5 * math.log(2.0) * math.sqrt(2.0 * math.pi)


@dataclass(frozen=True)
class PrintChemistryParams:
    """User-facing controls for the s023 print density-curve morph.

    ``development_time`` selects which density-curve / base+fog column of a BW
    development-time family to render (None — the default — means the stock has
    no family to choose from, or "use the representative middle development").
    """

    development_time: float | None = None
    active: bool = True
    gamma_factor: float = 1.0
    gamma_factor_fast: float = 1.0
    gamma_factor_slow: float = 1.0
    gamma_factor_red: float = 1.0
    gamma_factor_green: float = 1.0
    gamma_factor_blue: float = 1.0
    developer_exhaustion: float = 0.0


@dataclass(frozen=True)
class FilmChemistryParams:
    """User-facing controls for the film density-curve chemistry.

    Mirrors :class:`PrintChemistryParams` (same s023 coupled-gamma morph,
    applied through ``apply_print_curves_morph``), with ``development_time`` as
    the leading control: it selects which density-curve / base+fog column of a
    BW development-time family the runtime renders. None (the default) means the
    stock has no family to choose from, or "use the representative middle
    development" — single-curve and color stocks always leave it None.
    """

    development_time: float | None = None
    active: bool = True
    gamma_factor: float = 1.0
    gamma_factor_fast: float = 1.0
    gamma_factor_slow: float = 1.0
    gamma_factor_red: float = 1.0
    gamma_factor_green: float = 1.0
    gamma_factor_blue: float = 1.0
    developer_exhaustion: float = 0.0


def _signed_z(z, profile_type):
    return -z if profile_type == "positive" else z


def _gumbel_matched_cdf(z):
    return np.exp(-np.exp(-(z / _GUMBEL_WIDTH + _GUMBEL_LOCATION)))


def _layer_cdf(z, profile_type, model_type='norm_cdfs', alpha=0.0, gumbel_mix=0.0):
    zs = _signed_z(z, profile_type)
    cdf = _layer_cdf_values(zs, model_type, alpha)
    if gumbel_mix > 0.0:
        cdf = (1.0 - gumbel_mix) * cdf + gumbel_mix * _gumbel_matched_cdf(zs)
    return cdf


def _evaluate_channel_density(
    log_exposure,
    centers_c,
    amplitudes_c,
    sigmas_c,
    profile_type,
    model_type='norm_cdfs',
    alphas_c=None,
    gumbel_mix_per_layer=None,
):
    x = np.asarray(log_exposure, dtype=float)
    centers_c = np.asarray(centers_c, dtype=float)
    amplitudes_c = np.asarray(amplitudes_c, dtype=float)
    sigmas_c = np.asarray(sigmas_c, dtype=float)
    if gumbel_mix_per_layer is None:
        gumbel_mix_per_layer = np.zeros(centers_c.size, dtype=float)
    if alphas_c is None:
        alphas_c = np.zeros(centers_c.size, dtype=float)

    total = np.zeros(x.size, dtype=float)
    for i in range(centers_c.size):
        z = (x - centers_c[i]) / sigmas_c[i]
        total += amplitudes_c[i] * _layer_cdf(
            z, profile_type, model_type, float(alphas_c[i]), float(gumbel_mix_per_layer[i]))
    return total


def _evaluate_fitted_density(log_exposure, density_curves_model, profile_type):
    log_exposure = np.asarray(log_exposure, dtype=float)
    model = density_curves_model
    n_channels = model.centers.shape[0]
    fitted = np.empty((log_exposure.size, n_channels), dtype=float)

    for channel_idx in range(n_channels):
        alphas_c = None if model.alphas is None else model.alphas[channel_idx]
        fitted[:, channel_idx] = _evaluate_channel_density(
            log_exposure,
            model.centers[channel_idx],
            model.amplitudes[channel_idx],
            model.sigmas[channel_idx],
            profile_type,
            model_type=model.model_type,
            alphas_c=alphas_c,
        )

    return fitted


def _speed_layer_indices(centers_c):
    """Return (i_fast, i_mid, i_slow) by ascending center (grain-speed order)."""
    order = np.argsort(np.asarray(centers_c, dtype=float))
    n = len(order)
    return int(order[0]), int(order[n // 2]), int(order[-1])


def _channel_gamma_factor(params, channel_idx):
    channel_gamma_factors = np.array(
        [
            float(params.gamma_factor_red),
            float(params.gamma_factor_green),
            float(params.gamma_factor_blue),
        ],
        dtype=float,
    )
    return float(channel_gamma_factors[channel_idx])


def _developer_exhaustion_center_offset(
    centers_c,
    amplitudes_c,
    sigmas_c,
    profile_type,
    gumbel_mix_per_layer,
    model_type='norm_cdfs',
    alphas_c=None,
):
    if np.allclose(gumbel_mix_per_layer, 0.0):
        return 0.0

    zero_exposure = np.array([0.0], dtype=float)
    target_d0 = _evaluate_channel_density(
        zero_exposure,
        centers_c,
        amplitudes_c,
        sigmas_c,
        profile_type,
        model_type=model_type,
        alphas_c=alphas_c,
        gumbel_mix_per_layer=np.zeros_like(gumbel_mix_per_layer),
    )[0]

    def residual(center_offset):
        d0_with_exhaustion = _evaluate_channel_density(
            zero_exposure,
            centers_c + center_offset,
            amplitudes_c,
            sigmas_c,
            profile_type,
            model_type=model_type,
            alphas_c=alphas_c,
            gumbel_mix_per_layer=gumbel_mix_per_layer,
        )[0]
        return float(d0_with_exhaustion - target_d0)

    r_zero = residual(0.0)
    if abs(r_zero) <= 1e-12:
        return 0.0

    lo = -0.25
    hi = 0.25
    r_lo = residual(lo)
    r_hi = residual(hi)
    for _ in range(12):
        if r_lo == 0.0:
            return float(lo)
        if r_hi == 0.0:
            return float(hi)
        if r_lo * r_hi < 0.0:
            return float(_brentq(residual, lo, hi, xtol=1e-10))
        lo *= 2.0
        hi *= 2.0
        r_lo = residual(lo)
        r_hi = residual(hi)

    return 0.0


def _morph_channel_params(density_curves_model, params, channel_idx, profile_type):
    model = density_curves_model
    centers_c = np.asarray(model.centers[channel_idx], dtype=float).copy()
    amplitudes_c = np.asarray(model.amplitudes[channel_idx], dtype=float).copy()
    sigmas_c = np.asarray(model.sigmas[channel_idx], dtype=float).copy()
    # The skew alpha is a pure shape parameter, invariant under the gamma
    # (scale) morph, so it passes through unchanged.
    alphas_c = (None if model.alphas is None
                else np.asarray(model.alphas[channel_idx], dtype=float).copy())

    i_fast, i_mid, i_slow = _speed_layer_indices(centers_c)

    gamma_factor = float(params.gamma_factor)
    channel_gamma_factor = _channel_gamma_factor(params, channel_idx)
    g_fast = gamma_factor * channel_gamma_factor * float(params.gamma_factor_fast)
    g_mid = gamma_factor * channel_gamma_factor * float(params.gamma_factor_slow)
    g_slow = gamma_factor * channel_gamma_factor * float(params.gamma_factor_slow)

    if g_fast <= 0.0 or g_mid <= 0.0 or g_slow <= 0.0:
        raise ValueError(
            "Effective gamma must remain strictly positive per channel "
            f"(channel {channel_idx}: fast={g_fast:.3f}, mid={g_mid:.3f}, slow={g_slow:.3f})."
        )

    sigmas_c[i_fast] = max(sigmas_c[i_fast] / g_fast, SIGMA_FLOOR)
    centers_c[i_fast] = centers_c[i_fast] / g_fast
    sigmas_c[i_mid] = max(sigmas_c[i_mid] / g_mid, SIGMA_FLOOR)
    centers_c[i_mid] = centers_c[i_mid] / g_mid
    sigmas_c[i_slow] = max(sigmas_c[i_slow] / g_slow, SIGMA_FLOOR)
    centers_c[i_slow] = centers_c[i_slow] / g_slow

    gumbel_mix_per_layer = np.full(centers_c.size, float(params.developer_exhaustion), dtype=float)
    centers_c = centers_c + _developer_exhaustion_center_offset(
        centers_c,
        amplitudes_c,
        sigmas_c,
        profile_type,
        gumbel_mix_per_layer,
        model_type=model.model_type,
        alphas_c=alphas_c,
    )

    return centers_c, amplitudes_c, sigmas_c, alphas_c, gumbel_mix_per_layer


def _evaluate_channel_layers(
    log_exposure,
    centers_c,
    amplitudes_c,
    sigmas_c,
    profile_type,
    model_type='norm_cdfs',
    alphas_c=None,
    gumbel_mix_per_layer=None,
):
    """Per-layer density (n_le, n_layers) for one channel — the layer-resolved
    sibling of :func:`_evaluate_channel_density`, which is just its layer sum."""
    x = np.asarray(log_exposure, dtype=float)
    centers_c = np.asarray(centers_c, dtype=float)
    amplitudes_c = np.asarray(amplitudes_c, dtype=float)
    sigmas_c = np.asarray(sigmas_c, dtype=float)
    n_layers = centers_c.size
    if gumbel_mix_per_layer is None:
        gumbel_mix_per_layer = np.zeros(n_layers, dtype=float)
    if alphas_c is None:
        alphas_c = np.zeros(n_layers, dtype=float)

    out = np.zeros((x.size, n_layers), dtype=float)
    for i in range(n_layers):
        z = (x - centers_c[i]) / sigmas_c[i]
        out[:, i] = amplitudes_c[i] * _layer_cdf(
            z, profile_type, model_type, float(alphas_c[i]), float(gumbel_mix_per_layer[i]))
    return out


def _validate_morph_params(morph_params):
    for gamma_name, value in [
        ("gamma_factor", morph_params.gamma_factor),
        ("gamma_factor_fast", morph_params.gamma_factor_fast),
        ("gamma_factor_slow", morph_params.gamma_factor_slow),
        ("gamma_factor_red", morph_params.gamma_factor_red),
        ("gamma_factor_green", morph_params.gamma_factor_green),
        ("gamma_factor_blue", morph_params.gamma_factor_blue),
    ]:
        if value <= 0.0:
            raise ValueError(f"{gamma_name} must be strictly positive (got {value}).")
    if not 0.0 <= morph_params.developer_exhaustion <= 1.0:
        raise ValueError(
            "developer_exhaustion must be in [0, 1] "
            f"(got {morph_params.developer_exhaustion})."
        )


def apply_print_curves_morph(
    log_exposure,
    density_curves_model: DensityCurvesModel,
    morph_params,
    *,
    profile_type="positive",
):
    """Apply the s023 coupled gamma morph from explicit print-curve inputs."""
    if not morph_params.active:
        return _evaluate_fitted_density(log_exposure, density_curves_model, profile_type)

    model = density_curves_model
    if model.n_layers == 0:
        raise NotImplementedError("s023 morph requires a fitted density_curves_model.")

    _validate_morph_params(morph_params)

    log_exposure = np.asarray(log_exposure, dtype=float)
    n_channels = model.centers.shape[0]
    morphed = np.empty((log_exposure.size, n_channels), dtype=float)

    for channel_idx in range(n_channels):
        centers_c, amplitudes_c, sigmas_c, alphas_c, gumbel_mix_per_layer = _morph_channel_params(
            model,
            morph_params,
            channel_idx,
            profile_type,
        )
        morphed[:, channel_idx] = _evaluate_channel_density(
            log_exposure,
            centers_c,
            amplitudes_c,
            sigmas_c,
            profile_type,
            model_type=model.model_type,
            alphas_c=alphas_c,
            gumbel_mix_per_layer=gumbel_mix_per_layer,
        )

    return morphed


def apply_print_curves_morph_with_layers(
    log_exposure,
    density_curves_model: DensityCurvesModel,
    morph_params,
    *,
    profile_type="negative",
):
    """Morph and return both the total curves and the per-layer curves.

    Like :func:`apply_print_curves_morph`, but also returns the emulsion
    sublayer breakdown — ``(total (n_le, n_ch), layers (n_le, n_layers, n_ch))``
    — so the grain path (which needs sublayers) stays consistent with the morph.
    Used to bake the film chemistry morph into a profile before develop. When
    the morph is inactive it returns the unmorphed fitted curves and layers.
    """
    model = density_curves_model
    if model.n_layers == 0:
        raise NotImplementedError("s023 morph requires a fitted density_curves_model.")
    active = bool(morph_params.active)
    if active:
        _validate_morph_params(morph_params)

    log_exposure = np.asarray(log_exposure, dtype=float)
    n_channels = model.centers.shape[0]
    n_layers = model.centers.shape[1]
    total = np.empty((log_exposure.size, n_channels), dtype=float)
    layers = np.zeros((log_exposure.size, n_layers, n_channels), dtype=float)

    for channel_idx in range(n_channels):
        if active:
            centers_c, amplitudes_c, sigmas_c, alphas_c, gumbel_mix_per_layer = _morph_channel_params(
                model, morph_params, channel_idx, profile_type,
            )
        else:
            centers_c = np.asarray(model.centers[channel_idx], dtype=float)
            amplitudes_c = np.asarray(model.amplitudes[channel_idx], dtype=float)
            sigmas_c = np.asarray(model.sigmas[channel_idx], dtype=float)
            alphas_c = (None if model.alphas is None
                        else np.asarray(model.alphas[channel_idx], dtype=float))
            gumbel_mix_per_layer = np.zeros(n_layers, dtype=float)
        per_layer = _evaluate_channel_layers(
            log_exposure,
            centers_c,
            amplitudes_c,
            sigmas_c,
            profile_type,
            model_type=model.model_type,
            alphas_c=alphas_c,
            gumbel_mix_per_layer=gumbel_mix_per_layer,
        )
        layers[:, :, channel_idx] = per_layer
        total[:, channel_idx] = per_layer.sum(axis=1)

    return total, layers
