import numpy as np
import scipy
import scipy.ndimage

from spektrafilm.model.density_curves import interp_density_cmy_layers_channel
from spektrafilm.model.diffusion import (
    apply_multiplicative_unsharp_mask,
    match_channels,
)
from spektrafilm.runtime.params_schema import GrainParams
from spektrafilm.utils.fast_gaussian_filter import fast_gaussian_filter
from spektrafilm.utils.fast_stats import fast_binomial, fast_poisson

################################################################################
# Grain (multilayer stochastic model)
################################################################################
#
# Grain is sampled per emulsion sub-layer with a compound Poisson-Binomial
# model (see study a90 and manuscript §10): each pixel develops a random
# number of grains, giving density fluctuations whose variance is, before any
# blur, exactly
#
#     Var[D] = D (Dmax - u D) / N ,     N = grains per pixel = pixel_area / a_grain
#
# with u the grain uniformity. The model is always multilayer: the sub-layers
# come from the density model's per-sub-layer curves; a single-layer stock is
# just the one-sub-layer case.

# Grain is stochastic noise, so float32 is ample precision and halves the
# memory of the (large, full-resolution) working arrays versus float64. The
# grain functions compute in this dtype internally and cast the result back to
# the caller's dtype on return.
_GRAIN_WORK_DTYPE = np.float32

# --- RMS-granularity parametrisation (study a90) ----------------------------
# "RMS granularity" is the catalogue number sigma_D x 1000 measured through a
# 48 µm circular aperture at net density 1.0 (ISO 6328). Pre-blur the grain is
# white at the pixel scale, so that aperture is just a pixel of area A48:
# sampling N48 = A48 / a_grain grains reproduces its variance, hence the
# resolution-independent map sigma_48^2 = D_ref (Dmax - u D_ref) a / A48. The
# coarsest sub-layer (particle_scale_sublayers == 1) is the reference: the input
# is the RMS that sub-layer would read if it were the whole emulsion.
_RMS_APERTURE_DIAMETER_UM = 48.0
_RMS_APERTURE_AREA_UM2 = float(np.pi * (_RMS_APERTURE_DIAMETER_UM / 2.0) ** 2)
_RMS_REFERENCE_NET_DENSITY = 1.0


def particle_area_from_rms_granularity(
    rms_granularity, density_max, density_min, uniformity
):
    """Coarsest-sub-layer particle area a_1 (µm²) for a target RMS granularity.

    Inverts ``sigma_48^2 = D_ref (Dmax - u D_ref) a / A48`` at the reference net
    density 1.0, with ``sigma_48 = rms_granularity / 1000``. Works per channel
    (accepts scalars or arrays). ``density_max`` is the absolute (fog-inclusive)
    channel maximum; ``D_ref = 1.0 + density_min``.
    """
    sigma48 = np.asarray(rms_granularity, dtype=float) / 1000.0
    density_min = np.asarray(density_min, dtype=float)
    density_max = np.asarray(density_max, dtype=float)
    uniformity = np.asarray(uniformity, dtype=float)
    d_ref = _RMS_REFERENCE_NET_DENSITY + density_min
    denom = np.maximum(d_ref * (density_max - uniformity * d_ref), 1e-6)
    return sigma48**2 * _RMS_APERTURE_AREA_UM2 / denom


def _realized_peak_correction(
    density_max_layers,
    density_max_total,
    particle_scale_sublayers,
    density_min,
    uniformity,
    speed_separated,
):
    """Per-channel factor on the coarsest-sub-layer area so that the *realized*
    multilayer peak RMS equals the input ``rms_granularity`` (not just what the
    coarsest sub-layer alone would read).

    Each sub-layer is an independent Selwyn bell peaking at ``Dmax_sl/2u`` with
    peak variance proportional to ``scale_sl * Dmax_sl * Dmax_tot / (4u)``. How
    those bells combine at the realized peak depends on whether the sub-layers
    are speed-separated (study a90):

    - ``speed_separated`` (camera-taking *negatives*): the fast/slow
      sub-emulsions that give exposure latitude peak at well-separated
      exposures, so the realized peak is approximately the single *dominant*
      bell -> ``max_sl(scale_sl * Dmax_sl)``.
    - otherwise (*positives* -- slides, print films, papers -- single-speed):
      the bells peak together and their variances add, so the realized peak is
      the *sum* -> ``sum_sl(scale_sl * Dmax_sl)``.

    Setting the appropriate combination equal to the datasheet sigma_in gives

        C = 4u * D_ref * (Dmax_tot - u D_ref) / ( reduce_sl(scale_sl * Dmax_sl) * Dmax_tot )

    with ``D_ref = 1.0 + density_min`` (net density 1.0) and ``reduce`` = max
    (separated) or sum (coincident). Computed from the per-sub-layer Dmax only
    (study a90); inputs are absolute (fog-inclusive). Arrays are indexed
    ``[sub-layer, channel]`` / ``[channel]``. Picking the regime per film class
    avoids the ~1.5x cross-family error a single rule would carry (study a90):
    ``max`` over-grains coincident stocks, ``sum`` under-grains separated ones.
    """
    d_ref = _RMS_REFERENCE_NET_DENSITY + density_min  # [channel]
    scales = np.asarray(particle_scale_sublayers, dtype=float)
    weighted = scales[:, None] * density_max_layers  # [sub-layer, channel]
    layer_term = (
        np.max(weighted, axis=0) if speed_separated else np.sum(weighted, axis=0)
    )
    denom = np.maximum(layer_term * density_max_total, 1e-6)
    return 4.0 * uniformity * d_ref * (density_max_total - uniformity * d_ref) / denom


def layer_particle_model(
    density,
    density_max=2.2,
    n_particles_per_pixel=10,
    grain_uniformity=0.98,
    seed=None,
    blur_particle=0.0,
    use_fast_stats=False,
):
    """Sample one emulsion sub-layer's grain at the given density field.

    Compound Poisson-Binomial: N_s ~ Poisson(N/sat) sensitised grains, each
    developed with probability D/Dmax, scaled back by the per-grain optical
    density and the saturation factor. Poisson thinning makes the realized
    density unbiased with variance D(Dmax - u D)/N (study a90).
    """
    if seed is not None:
        np.random.seed(seed)  # scipy uses np.random

    # Keep the scalar parameters as plain Python floats so a float32 `density`
    # is not silently promoted back to float64 by numpy's scalar rules; the
    # grain then stays in the caller's (float32) working dtype throughout.
    density_max = float(density_max)
    n_particles_per_pixel = float(n_particles_per_pixel)
    grain_uniformity = float(grain_uniformity)

    probability_of_development = density / density_max
    probability_of_development = np.clip(
        probability_of_development, 1e-6, 1 - 1e-6
    )  # for safe calc
    od_particle = density_max / n_particles_per_pixel

    if use_fast_stats:
        binom_rvs = fast_binomial
        poisson_rvs = fast_poisson
    else:
        binom_rvs = scipy.stats.binom.rvs
        poisson_rvs = scipy.stats.poisson.rvs
    saturation = 1 - probability_of_development * grain_uniformity * (1 - 1e-6)
    seeds = poisson_rvs(n_particles_per_pixel / saturation)
    # Cast the integer binomial counts to the working dtype, then fold in the
    # per-particle optical density and saturation in place so no extra full-size
    # temporaries are made.
    grain = binom_rvs(seeds, probability_of_development).astype(
        density.dtype, copy=False
    )
    grain *= od_particle
    grain *= saturation

    if blur_particle > 0:
        grain = fast_gaussian_filter(grain, blur_particle * np.sqrt(od_particle))
    return grain


def add_micro_structure(density_cmy_out, micro_structure, pixel_size_um):
    grain_micro_structure_blur_pixel = micro_structure[0] / pixel_size_um
    grain_micro_structure_sigma = (
        micro_structure[1] * 0.001 / pixel_size_um
    )  # grain microstructure[1] is in nm
    if grain_micro_structure_sigma > 0.05:
        # Multiplicative lognormal clumping with linear-space mean 1 and std
        # sigma. Generated directly into a single buffer instead of via
        # fast_lognormal_from_mean_std(ones, ones*sigma), which would allocate
        # several full-size arrays (two ones params + two internal mu/sigma
        # grids + the result) just to carry constants. For mean 1 the log-space
        # parameters are sigma2 = ln(1+sigma^2), mu = -sigma2/2.
        sigma2 = float(np.log1p(grain_micro_structure_sigma**2))
        clumping = np.random.standard_normal(density_cmy_out.shape)
        clumping *= np.sqrt(sigma2)
        clumping -= 0.5 * sigma2
        np.exp(clumping, out=clumping)
        if grain_micro_structure_blur_pixel > 0.4:
            clumping = fast_gaussian_filter(clumping, grain_micro_structure_blur_pixel)
        density_cmy_out *= clumping
    return density_cmy_out


def _coarsest_area_from_curves(
    density_curves_layers,
    density_min_layers,
    density_max_layers,
    density_max_fractions,
    particle_scale_sublayers,
    uniformity,
    rms_granularity,
):
    """Coarsest-sub-layer particle area from the *exact* realized multilayer peak.

    The per-pixel grain variance at exposure ``x`` is, in units of ``a1 / A48``,

        S(x) = sum_sl (scale_sl / frac_sl) * D_sl(x) * (Dmax_sl - u * D_sl(x))

    over the absolute (fog-inclusive) sub-layer densities ``D_sl(x)`` read from
    the characteristic curves. The realized peak is ``G = max_x S(x)``, so

        a1 = sigma_in^2 * A48 / G          (sigma_in = rms_granularity / 1000)

    makes that peak equal the datasheet RMS *regardless* of how the sub-layer
    bells combine: it reproduces the dominant-bell (speed-separated) and summed
    (coincident) limits automatically and, crucially, tracks the uniformity-
    driven crossover between them that no Dmax-only closed form can capture
    (study a90 — e.g. Double-X is separated at u≈0.97 but coincident at u≈0.6).
    Arrays: curves ``[le, sub, ch]``; per-layer ``[sub, ch]``; ``[channel]``.
    """
    d_abs = (
        np.nan_to_num(density_curves_layers) + density_min_layers[None, :, :]
    )  # [le, sub, ch]
    scales = np.asarray(particle_scale_sublayers, dtype=float)
    weight = scales[None, :, None] / density_max_fractions[None, :, :]  # [1, sub, ch]
    var_shape = (
        weight
        * d_abs
        * (density_max_layers[None, :, :] - uniformity[None, None, :] * d_abs)
    )
    peak = np.nanmax(np.sum(var_shape, axis=1), axis=0)  # [channel]
    sigma_in = rms_granularity / 1000.0
    return sigma_in**2 * _RMS_APERTURE_AREA_UM2 / np.maximum(peak, 1e-9)


def _layer_grain_params(
    density_max_layers,
    density_min,
    rms_granularity,
    particle_scale_sublayers,
    uniformity,
    n_ch,
    pixel_size_um,
    speed_separated=True,
    density_curves_layers=None,
):
    """Per (sub-layer, channel) grain parameters shared by the whole-array and
    streaming code paths.

    Returns ``(density_min, density_min_layers, density_max_layers,
    n_particles_per_pixel)`` where the per-layer arrays are indexed
    ``[sub-layer, channel]`` and ``density_max_layers`` already includes the
    density floor.

    The coarsest-sub-layer particle area is sized so the realized multilayer
    *peak* RMS matches the per-channel input. When ``density_curves_layers`` is
    given (the normal path) this is exact — maximised over the actual curves by
    ``_coarsest_area_from_curves``. Otherwise it falls back to the Dmax-only
    closed form (``particle_area_from_rms_granularity`` x
    ``_realized_peak_correction``), with ``speed_separated`` selecting the
    dominant-bell (max) or coincident (sum) regime. Areas are then scaled per
    sub-layer by ``particle_scale_sublayers`` (coarsest == 1).
    """
    density_max_total = np.sum(density_max_layers, axis=0)  # [channel]
    density_max_fractions = density_max_layers / density_max_total[None, :]
    density_min = match_channels(density_min, n_ch)
    uniformity = match_channels(uniformity, n_ch)
    rms_granularity = match_channels(rms_granularity, n_ch)
    density_min_layers = (
        density_max_fractions * density_min[None, :]
    )  # [sub-layer, channel]
    density_max_layers = (
        density_max_layers + density_min_layers
    )  # absolute (fog-inclusive)
    density_max_total = density_max_total + density_min  # absolute [channel]

    if density_curves_layers is not None:
        a_coarsest = _coarsest_area_from_curves(
            density_curves_layers,
            density_min_layers,
            density_max_layers,
            density_max_fractions,
            particle_scale_sublayers,
            uniformity,
            rms_granularity,
        )  # [channel]
    else:
        a_coarsest = particle_area_from_rms_granularity(
            rms_granularity, density_max_total, density_min, uniformity
        )  # [channel]
        a_coarsest = a_coarsest * _realized_peak_correction(
            density_max_layers,
            density_max_total,
            particle_scale_sublayers,
            density_min,
            uniformity,
            speed_separated,
        )
    particle_area_um2_layers = (
        a_coarsest[None, :] * np.array(particle_scale_sublayers)[:, None]
    )  # [sub-layer, channel]
    pixel_area_um2 = pixel_size_um**2
    n_particles_per_pixel = (
        pixel_area_um2 * density_max_fractions / particle_area_um2_layers
    )
    return density_min, density_min_layers, density_max_layers, n_particles_per_pixel


def _channel_sublayer_grain(
    layers_ch,
    density_max_layers_ch,
    n_particles_ch,
    grain_uniformity_ch,
    seed_base,
    blur_dye_clouds_um,
    use_fast_stats,
):
    """Sum the grain of every sub-layer of one colour channel.

    ``layers_ch`` is ``(H, W, n_layers)`` for a single channel, already offset
    by its per-layer density floor. Returns that channel's ``(H, W)`` grain.
    The first sub-layer's result seeds the accumulator so no separate zero
    buffer is allocated.
    """
    channel_grain = None
    for sl in range(layers_ch.shape[2]):
        layer = layer_particle_model(
            layers_ch[:, :, sl],
            density_max=density_max_layers_ch[sl],
            n_particles_per_pixel=n_particles_ch[sl],
            grain_uniformity=grain_uniformity_ch,
            seed=None if seed_base is None else seed_base + sl * 10,
            blur_particle=blur_dye_clouds_um,
            use_fast_stats=use_fast_stats,
        )
        if channel_grain is None:
            channel_grain = layer
        else:
            channel_grain += layer
    return channel_grain


def _finalize_grain(
    density_cmy_out,
    density_min,
    grain_micro_structure,
    pixel_size_um,
    grain_blur,
    usm_sigma=0.0,
    usm_amount=0.0,
):
    """Shared grain finishing: optical micro-structure clumping, the
    pixel-correlating blur, the mass-conserving density unsharp mask (applied on
    the absolute density, before the floor is removed, so it stays positive),
    and removal of the density floor."""
    density_cmy_out = add_micro_structure(
        density_cmy_out, grain_micro_structure, pixel_size_um
    )
    if grain_blur > 0:
        density_cmy_out = fast_gaussian_filter(density_cmy_out, grain_blur)
    if usm_amount > 0 and usm_sigma > 0:
        density_cmy_out = apply_multiplicative_unsharp_mask(
            density_cmy_out, usm_sigma, usm_amount
        )
    density_cmy_out -= density_min
    return density_cmy_out


def apply_grain_to_density_layers(
    density_cmy_layers,  # x,y,sublayers,rgb
    density_max_layers,  # [sublayers, rgb]
    pixel_size_um=10,
    rms_granularity=[6.0, 7.0, 12.0],  # rgb, 48 µm-aperture
    particle_scale_sublayers=[1.0, 0.5, 0.25],  # sublayers, coarsest == 1
    density_min=[0.03, 0.06, 0.04],
    grain_uniformity=[0.97, 0.97, 0.97],
    grain_blur=1.0,
    grain_blur_dye_clouds_um=1.0,
    grain_micro_structure=(0.1, 30),
    fixed_seed=None,
    use_fast_stats=False,
    usm_sigma=0.0,
    usm_amount=0.0,
    speed_separated=True,
    density_curves_layers=None,
):
    # Whole-array entry: grains a pre-built (H, W, sublayers, channels) stack.
    # `apply_grain` uses the streaming path below instead; this stays for direct
    # callers/tests. Output dtype matches the input. Pass `density_curves_layers`
    # (the [le, sub, ch] curves) for the exact curve-based peak; without it the
    # Dmax-only closed form is used, with `speed_separated` selecting the
    # dominant-bell (True/max) or coincident (False/sum) regime.
    out_dtype = density_cmy_layers.dtype
    n_ch = density_cmy_layers.shape[3]
    density_min, density_min_layers, density_max_layers, n_particles_per_pixel = (
        _layer_grain_params(
            density_max_layers,
            density_min,
            rms_granularity,
            particle_scale_sublayers,
            grain_uniformity,
            n_ch,
            pixel_size_um,
            speed_separated,
            density_curves_layers,
        )
    )
    grain_uniformity = match_channels(grain_uniformity, n_ch)

    # Offset each layer by its floor and grain channel-by-channel in float32.
    density_cmy_layers = density_cmy_layers.astype(_GRAIN_WORK_DTYPE, copy=False)
    density_cmy_layers += density_min_layers
    density_cmy_out = np.empty(
        density_cmy_layers.shape[0:2] + (n_ch,), dtype=_GRAIN_WORK_DTYPE
    )
    for ch in range(n_ch):
        density_cmy_out[:, :, ch] = _channel_sublayer_grain(
            density_cmy_layers[:, :, :, ch],
            density_max_layers[:, ch],
            n_particles_per_pixel[:, ch],
            grain_uniformity[ch],
            None if fixed_seed is not None else ch,
            grain_blur_dye_clouds_um,
            use_fast_stats,
        )

    density_cmy_out = _finalize_grain(
        density_cmy_out,
        density_min,
        grain_micro_structure,
        pixel_size_um,
        grain_blur,
        usm_sigma,
        usm_amount,
    )
    return density_cmy_out.astype(out_dtype, copy=False)


def apply_grain(
    density_cmy,
    pixel_size_um,
    grain: GrainParams,
    density_curves,
    density_curves_layers,
    profile_type,
    bypass_grain=False,
    use_fast_stats=False,
):
    if not grain.active or bypass_grain:
        return density_cmy

    # Streaming multilayer grain: build and grain one channel's sub-layer stack
    # at a time, so the full (H, W, sublayers, channels) array is never held —
    # only a single (H, W, sublayers) float32 slab. This is the memory-critical
    # path for full-resolution scans. A single-layer density model (no
    # per-sub-layer curves) is treated as the one-sub-layer case, built from the
    # total curve, so there is no separate single-layer approximation.
    positive_film = profile_type == "positive"
    n_ch = density_cmy.shape[-1]
    if density_curves_layers is None:
        density_curves_layers = density_curves[:, None, :]  # (n_le, 1, n_ch)
    n_sub = density_curves_layers.shape[1]

    density_max_layers = np.nanmax(density_curves_layers, axis=0)
    particle_scale_sublayers = list(grain.particle_scale_sublayers)[:n_sub]
    # Size the grain from the exact realized peak over the sub-layer curves, so
    # the realized RMS matches the input for any bell overlap / uniformity
    # (study a90); no film-class regime branch is needed.
    density_min, density_min_layers, density_max_layers, n_particles_per_pixel = (
        _layer_grain_params(
            density_max_layers,
            grain.density_min,
            grain.rms_granularity,
            particle_scale_sublayers,
            grain.uniformity,
            n_ch,
            pixel_size_um,
            density_curves_layers=density_curves_layers,
        )
    )
    grain_uniformity = match_channels(grain.uniformity, n_ch)

    density_cmy_out = np.empty(
        density_cmy.shape[0:2] + (n_ch,), dtype=_GRAIN_WORK_DTYPE
    )
    for ch in range(n_ch):
        layers_ch = interp_density_cmy_layers_channel(
            density_cmy[:, :, ch],
            density_curves[:, ch],
            density_curves_layers[:, :, ch],
            positive_film,
        ).astype(_GRAIN_WORK_DTYPE)
        layers_ch += density_min_layers[:, ch]  # offset each sub-layer by its floor
        density_cmy_out[:, :, ch] = _channel_sublayer_grain(
            layers_ch,
            density_max_layers[:, ch],
            n_particles_per_pixel[:, ch],
            grain_uniformity[ch],
            ch,
            grain.blur_dye_clouds_um,
            use_fast_stats,
        )

    density_cmy_out = _finalize_grain(
        density_cmy_out,
        density_min,
        grain.micro_structure,
        pixel_size_um,
        grain.blur,
        grain.mult_usm_sigma,
        grain.mult_usm_amount,
    )
    return density_cmy_out.astype(density_cmy.dtype, copy=False)
