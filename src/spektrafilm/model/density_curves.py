from dataclasses import replace

import numpy as np
import scipy
import scipy.stats
from spektrafilm.utils.fast_interp import fast_interp


################################################################################
# Parametric density-curve model (source of truth)
################################################################################
#
# A profile's density curves are defined by its DensityCurvesModel: per channel
# and per layer a (center, amplitude, sigma) — plus a skew `alpha` for the
# septic model. The sampled `density_curves` / `density_curves_layers` arrays
# the runtime consumes are DERIVED from this model by sampling it onto the
# profile's log_exposure grid (see refresh_density_curves_from_model). The
# arrays are an implementation detail of the fast_interp hot path; the model is
# the source of truth.
#
# Two per-layer sigmoid families (model_type):
#   'norm_cdfs'      Gaussian CDF Phi((x-mu)/sigma)        — no skew.
#   'sept_norm_cdfs' septic-polynomial CDF approximation with a median-
#                    preserving skew `alpha` (|alpha|<1). Transcendental-free;
#                    alpha=0 reproduces the symmetric Gaussian fit. Derived in
#                    research study b70/s010.

SEPT_K = 5.8013  # septic support width in sigmas (minimax vs Gaussian CDF)
_GAUSS_MODEL_TYPES = ('norm_cdfs',)
_SEPT_MODEL_TYPE = 'sept_norm_cdfs'


def _septic_smoothstep(v):
    """Order-7 Hermite smoothstep on [0, 1]: 35v^4 - 84v^5 + 70v^6 - 20v^7."""
    v2 = v * v
    return (v2 * v2) * (35.0 + v * (-84.0 + v * (70.0 - 20.0 * v)))


def _septic_smoothstep_deriv(v):
    """d/dv of the septic smoothstep: 140 v^3 (1 - v)^3."""
    return 140.0 * (v ** 3) * ((1.0 - v) ** 3)


def _septic_warp(z, alpha=0.0):
    """Map signed `z` (sigma units) to the warped smoothstep coordinate.

    Median-preserving skew warp v = u + alpha*u(1-u)(2u-1)^2 (vanishes at
    u=0.5, so the 0.5 crossing stays on the center for every alpha). Returns
    (v, dv/du) so callers can form both the CDF and its derivative.
    """
    u = np.clip(z / SEPT_K + 0.5, 0.0, 1.0)
    if alpha:
        two_u_m1 = 2.0 * u - 1.0
        v = np.clip(u + alpha * u * (1.0 - u) * two_u_m1 * two_u_m1, 0.0, 1.0)
        dv_du = 1.0 + alpha * (u * (u * (-16.0 * u + 24.0) - 10.0) + 1.0)
        return v, dv_du
    return u, 1.0


def _septic_cdf(z, alpha=0.0):
    """Septic CDF approximation for input `z` in sigma units, with skew alpha."""
    v, _ = _septic_warp(z, alpha)
    return _septic_smoothstep(v)


def _septic_layer(z, alpha=0.0):
    """Septic CDF and its du-derivative (for pdf / distribution models)."""
    v, dv_du = _septic_warp(z, alpha)
    return _septic_smoothstep(v), _septic_smoothstep_deriv(v) * dv_du


def _layer_cdf_values(z, model_type, alpha=0.0):
    """Per-layer sigmoid value for signed input `z`, dispatched by model_type."""
    if model_type in _GAUSS_MODEL_TYPES:
        return scipy.stats.norm.cdf(z)
    if model_type == _SEPT_MODEL_TYPE:
        return _septic_cdf(z, alpha)
    raise ValueError(
        f"unknown density-curve model_type {model_type!r}; "
        f"expected one of {(*_GAUSS_MODEL_TYPES, _SEPT_MODEL_TYPE)}"
    )


def _require_model(density_curves_model):
    if density_curves_model is None:
        raise ValueError(
            "profile has no density_curves_model; the runtime derives density "
            "curves from the parametric model and requires it to be present"
        )
    return density_curves_model


def evaluate_density_curves(density_curves_model, log_exposure, profile_type='negative'):
    """Sample the model's total density curves onto `log_exposure`.

    Returns an array shaped (n_le, n_channels). Mirrors the per-channel sum
    D(x) = sum_i A_i * cdf(+/-(x - mu_i) / sigma_i) used at fit time.
    """
    model = _require_model(density_curves_model)
    x = np.asarray(log_exposure, dtype=float)
    centers = np.asarray(model.centers, dtype=float)
    amplitudes = np.asarray(model.amplitudes, dtype=float)
    sigmas = np.asarray(model.sigmas, dtype=float)
    alphas = None if model.alphas is None else np.asarray(model.alphas, dtype=float)
    n_ch, n_layers = centers.shape
    sign = -1.0 if profile_type == 'positive' else 1.0

    out = np.zeros((x.size, n_ch), dtype=float)
    for ch in range(n_ch):
        for i in range(n_layers):
            z = sign * (x - centers[ch, i]) / sigmas[ch, i]
            alpha = float(alphas[ch, i]) if alphas is not None else 0.0
            out[:, ch] += amplitudes[ch, i] * _layer_cdf_values(z, model.model_type, alpha)
    return out


def evaluate_density_curves_layers(density_curves_model, log_exposure,
                                   profile_type='negative'):
    """Sample the model's per-layer density curves onto `log_exposure`.

    Returns an array shaped (n_le, n_layers, n_channels) — one slice per
    emulsion sub-layer, summing across the layer axis to the total curve.
    """
    model = _require_model(density_curves_model)
    x = np.asarray(log_exposure, dtype=float)
    centers = np.asarray(model.centers, dtype=float)
    amplitudes = np.asarray(model.amplitudes, dtype=float)
    sigmas = np.asarray(model.sigmas, dtype=float)
    alphas = None if model.alphas is None else np.asarray(model.alphas, dtype=float)
    n_ch, n_layers = centers.shape
    sign = -1.0 if profile_type == 'positive' else 1.0

    out = np.zeros((x.size, n_layers, n_ch), dtype=float)
    for ch in range(n_ch):
        for i in range(n_layers):
            z = sign * (x - centers[ch, i]) / sigmas[ch, i]
            alpha = float(alphas[ch, i]) if alphas is not None else 0.0
            out[:, i, ch] = amplitudes[ch, i] * _layer_cdf_values(z, model.model_type, alpha)
    return out


def refresh_density_curves_from_model(profile_data, profile_type='negative'):
    """Repopulate `density_curves` and `density_curves_layers` on `profile_data`
    from its `density_curves_model`, in place.

    The model is the source of truth; this samples it onto the profile's own
    `log_exposure` grid so every downstream consumer (develop, grain, couplers,
    Dmax probes) sees model-derived arrays. Raises if the model is absent.

    A single-layer model has no emulsion sub-layer structure, so
    `density_curves_layers` is left None (the grain path then uses its simple,
    non-sublayer branch) — matching how such stocks behaved previously.
    """
    model = _require_model(profile_data.density_curves_model)
    log_exposure = np.asarray(profile_data.log_exposure, dtype=float)
    profile_data.density_curves = evaluate_density_curves(model, log_exposure, profile_type)
    if model.n_layers > 1:
        profile_data.density_curves_layers = evaluate_density_curves_layers(
            model, log_exposure, profile_type)
    else:
        profile_data.density_curves_layers = None
    return profile_data


def select_development_time(profile_data, development_time):
    """Collapse a BW development-time family to the single member nearest
    `development_time`, in place.

    A BW family carries N density-curve columns (one per development time), a
    matching per-column base+fog (n_wl, N), and an N-row density_curves_model.
    The runtime renders one development time, so this picks one column across
    all of them — curves, layer slice, base+fog, the development_time entry, and
    the model row — leaving a normal single-curve profile (1-D base) for every
    downstream consumer. ``development_time`` None means "no choice made": the
    representative middle development is used (floor-middle when even). No-op
    for color stocks (no development_time); an N=1 family still collapses so
    its per-column base+fog becomes the 1-D base consumers expect.
    """
    times = profile_data.development_time
    curves = profile_data.density_curves
    if times is None or curves is None:
        return profile_data
    times = np.asarray(times, dtype=float)
    if development_time is None:
        index = (times.size - 1) // 2
    else:
        index = int(np.argmin(np.abs(times - float(development_time))))

    profile_data.density_curves = np.asarray(curves)[:, index:index + 1]
    if profile_data.density_curves_layers is not None:
        profile_data.density_curves_layers = (
            np.asarray(profile_data.density_curves_layers)[:, :, index:index + 1])
    base = profile_data.base_density
    if base is not None and np.asarray(base).ndim == 2:
        profile_data.base_density = np.asarray(base)[:, index]
    profile_data.development_time = times[index:index + 1]

    model = profile_data.density_curves_model
    if model is not None and model.centers is not None:
        profile_data.density_curves_model = replace(
            model,
            centers=np.asarray(model.centers)[index:index + 1],
            amplitudes=np.asarray(model.amplitudes)[index:index + 1],
            sigmas=np.asarray(model.sigmas)[index:index + 1],
            alphas=(None if model.alphas is None
                    else np.asarray(model.alphas)[index:index + 1]),
        )
    return profile_data


################################################################################
# Denstity curves
################################################################################

def interpolate_exposure_to_density(log_exposure_rgb, density_curves, log_exposure, gamma_factor):
    """
    Interpolates the exposure values to density values using the provided density curves.
    Parameters:
    log_exposure_rgb (numpy.ndarray): A 3D array of shape (height, width, n_ch) representing the log10 exposure values.
    density_curves (numpy.ndarray): A 2D array of shape (num_points, n_ch) representing the density curves for each channel.
    log_exposure (numpy.ndarray): A 1D array of logarithmic exposure values.
    gamma_factor (float): The gamma correction factor to be applied to the density characteristic curves.
    Returns:
    numpy.ndarray: A 3D array of shape (height, width, n_ch) representing the interpolated density values.
    """
    n_ch = log_exposure_rgb.shape[-1]
    if np.size(gamma_factor)==1:
        gamma_factor = [gamma_factor] * n_ch
    gamma_factor = np.array(gamma_factor)
    density_cmy = fast_interp(np.ascontiguousarray(log_exposure_rgb),
                              log_exposure[:,None]/gamma_factor[None,:],
                              density_curves)
    return density_cmy


def interp_density_cmy_layers_channel(density_cmy_ch, density_curves_ch, density_curves_layers_ch,
                                      positive_film=False):
    """Sub-layer densities for a single colour channel: ``(H, W, n_layers)``.

    ``interp_density_cmy_layers`` is just this looped over channels. Exposed
    separately so memory-sensitive callers (the grain) can build one channel's
    slab at a time instead of materializing all channels at once.
    """
    n_layers = density_curves_layers_ch.shape[1]
    stacked = np.repeat(density_cmy_ch[:, :, np.newaxis], n_layers, axis=-1)
    if positive_film:
        return fast_interp(-stacked, -density_curves_ch, density_curves_layers_ch)
    return fast_interp(stacked, density_curves_ch, density_curves_layers_ch)


def interp_density_cmy_layers(density_cmy, density_curves, density_curves_layers, positive_film=False):
    # density_curves_layers is (n_le, n_layers, n_ch); channel is the last axis.
    n_ch = density_cmy.shape[-1]
    n_layers = density_curves_layers.shape[1]
    density_cmy_layers = np.zeros((density_cmy.shape[0], density_cmy.shape[1], n_layers, n_ch)) # x,y,layer,channel
    for ch in np.arange(n_ch):
        density_cmy_layers[:,:,:,ch] = interp_density_cmy_layers_channel(
            density_cmy[:,:,ch], density_curves[:,ch], density_curves_layers[:,:,ch], positive_film)
    return density_cmy_layers
    
# This method was used for multilayer grain, but it is not used anymore
# def interpolate_layers(self, exposure_rgb):
#     density_curves_layers = density_curves_layers_model(self.log_exposure, self.parameters, self.type)
#     density_cmy_layers = np.zeros((exposure_rgb.shape[0], exposure_rgb.shape[1], 3, 3))
#     exposure = 10**(self.log_exposure)
#     for channel in np.arange(3):
#         for layer in np.arange(3):
#             density_cmy_layers[:,:,channel,layer] = np.interp(exposure_rgb[:,:,channel],
#                                                               exposure,
#                                                               density_curves_layers[:,channel,layer])
#     return density_cmy_layers

def apply_gamma_shift_correction(log_exposure, density_curves, gamma_correction, log_exposure_correction):
    dc = density_curves
    le = log_exposure
    gc = gamma_correction
    les = log_exposure_correction
    dc_out = np.zeros_like(dc)
    for i in np.arange(3):
        dc_out[:,i] = np.interp(le, le/gc[i] + les[i], dc[:,i])
    return dc_out


def remove_viewing_glare_comp(le, dc, factor=0.2, density=1.0, transition=0.3):
    """
    Removes viewing glare compensation from the density curves of print paper.
    Parameters:
    le (numpy.ndarray): The log exposure values.
    dc (numpy.ndarray): density curves of the print paper. Shape (n,3).
    factor (float, optional): The factor by which to reduce the light exposure values of the shadows. (brighter shadows). Default is 0.1.
    density (float, optional): The density value of the transition point. Default is 1.2.
    transition (float, optional): The transition density range used for Gaussian filtering. Default is 0.3.
    Returns:
    numpy.ndarray: density curves with viewing glare compensation removed.
    """
    def _measure_slope(le, density_curve, le_center, range_ev=1):
        le_delta = np.log10(2**range_ev)/2
        le_0 = le_center - le_delta
        le_1 = le_center + le_delta
        density_0 = np.interp(le_0, le, density_curve)
        density_1 = np.interp(le_1, le, density_curve)
        slope = (density_1 - density_0)/(le_1 - le_0)
        return slope    
    
    dc_mean = np.mean(dc, axis=1)
    le_center = np.interp(density, dc_mean, le)
    slope = _measure_slope(le, dc_mean, le_center)
    le_step = np.mean(np.diff(le))
    dc_out = np.zeros_like(dc)
    for i in np.arange(3):
        le_nl = np.copy(le)
        le_nl[le>le_center] -= (le[le>le_center]-le_center)*factor
        le_transition = transition/slope
        le_nl = scipy.ndimage.gaussian_filter(le_nl, le_transition/le_step)
        dc_out[:,i] = np.interp(le_nl, le, dc[:,i])
    return dc_out