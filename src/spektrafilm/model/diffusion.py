import numpy as np
import scipy.ndimage
from spektrafilm.utils.fast_gaussian_filter import fast_exponential_filter, fast_gaussian_filter
from spektrafilm.utils.numba_boost_hightlights import boost_highlights

def apply_unsharp_mask(image, sigma=0.0, amount=0.0):
    """
    Apply an unsharp mask to an image.
    
    Parameters:
    image (ndarray): The input image to be processed.
    sigma (float, optional): The standard deviation for the Gaussian sharp filter. Leave 0 if not wanted.
    amount (float, optional): The strength of the sharpening effect. Leave 0 if not wanted.
    
    Returns:
    ndarray: The processed image after applying the unsharp mask.
    """
    # image_blur = scipy.ndimage.gaussian_filter(image, sigma=(sigma, sigma, 0))
    image_blur = fast_gaussian_filter(image, sigma)
    image_sharp = image + amount * (image - image_blur)
    return image_sharp


def apply_halation_um(raw, halation, pixel_size_um):
    """Apply highlight boost, in-emulsion scatter, and back-reflection halation.

    Ordering is boost -> scatter -> halation: the boost reconstructs pre-clip
    irradiance (what actually hit the emulsion), scatter propagates that
    irradiance through the absorbing emulsion as an energy-preserving
    two-Gaussian mixture (core + tail), and halation adds back-reflected
    light as an additive sum of N Gaussians with sqrt(k)-spaced widths.

    See notes/halation_physics_model.md and notes/halation_implementation_plan.md
    for the physical derivation and parameter priors.
    """
    if not halation.active:
        return raw

    # 1. Scatter pass — energy-preserving mixture of a Gaussian core and an
    #    exponential tail (matching measured film MTFs), blended with the
    #    identity by scatter_amount to model the fraction of photons that
    #    actually scatter:
    #    E1 = (1 - s) * E0  +  s * [(1 - w_s) * G(sigma_c) * E0 + w_s * Exp(lambda_t) * E0]
    #    where s = scatter_amount. sigma_c and lambda_t are both scaled by
    #    scatter_spatial_scale; lambda_t is the decay constant of the
    #    exponential, dispatched internally to a Gaussian mixture.
    s_amount = float(halation.scatter_amount)
    s_scale = float(halation.scatter_spatial_scale)
    w_s = np.asarray(halation.scatter_tail_weight, dtype=np.float64)
    sigma_c_px = np.asarray(halation.scatter_core_um, dtype=np.float64) * s_scale / pixel_size_um
    lambda_t_px = np.asarray(halation.scatter_tail_um, dtype=np.float64) * s_scale / pixel_size_um
    if s_amount > 0 and (np.any(sigma_c_px > 0) or np.any(lambda_t_px > 0)):
        core = fast_gaussian_filter(raw, np.maximum(sigma_c_px, 1e-6))
        tail = fast_exponential_filter(raw, np.maximum(lambda_t_px, 1e-6))
        scattered = (1.0 - w_s) * core + w_s * tail
        raw = (1.0 - s_amount) * raw + s_amount * scattered

    # 2. Halation pass — additive multi-bounce sum scaled by halation_amount:
    #    E2 = E1 + halation_amount * Σ_{k=1..N} a_k * G(sigma_h * sqrt(k)) * E1
    #    with a_k = a_tot * ρ^(k-1) / Σ_j ρ^(j-1) and sigmas scaled by
    #    halation_spatial_scale.
    h_amount = float(halation.halation_amount)
    h_scale = float(halation.halation_spatial_scale)
    a_tot = np.asarray(halation.halation_strength, dtype=np.float64) * h_amount
    sigma_h_px = np.asarray(halation.halation_first_sigma_um, dtype=np.float64) * h_scale / pixel_size_um
    N = int(halation.halation_n_bounces)
    rho = float(halation.halation_bounce_decay)
    if N >= 1 and np.any(a_tot > 0) and np.any(sigma_h_px > 0):
        decay = np.array([rho ** (k - 1) for k in range(1, N + 1)], dtype=np.float64)
        decay /= decay.sum()
        halation_blur = np.zeros_like(raw)
        for k, wk in zip(range(1, N + 1), decay):
            sigma_k_px = np.maximum(sigma_h_px * np.sqrt(k), 1e-6)
            halation_blur += wk * fast_gaussian_filter(raw, sigma_k_px)
        raw = raw + a_tot * halation_blur
        if halation.halation_renormalize:
            raw = raw / (1.0 + a_tot)

    return raw

def apply_gaussian_blur(data, sigma):
    if sigma > 0:
        # return scipy.ndimage.gaussian_filter(data, (sigma, sigma, 0))
        # data = np.double(data)
        # data = np.ascontiguousarray(data)
        return fast_gaussian_filter(data, sigma)
    else:
        return data
    
def apply_gaussian_blur_um(data, sigma_um, pixel_size_um):
    sigma = sigma_um / pixel_size_um
    if sigma > 0:
        # return scipy.ndimage.gaussian_filter(data, (sigma, sigma, 0))
        # data = np.double(data)
        # data = np.ascontiguousarray(data)
        return fast_gaussian_filter(data, sigma)
    else:
        return data

def apply_diffusion_filter_mm(data, diffusion_filter_params, pixel_size_um):
    diffusion_fraction, sigma_mm, iterations, growth, decay = diffusion_filter_params
    iterations = int(iterations)
    sigma = sigma_mm * 1000 / pixel_size_um
    if sigma_mm <= 0 or sigma <= 0 or diffusion_fraction <= 0 or iterations <= 0:
        return data
    
    max_sigma = sigma * (growth ** max(iterations - 1, 0))
    image_size = min(data.shape[:2])
    if max_sigma > image_size / 6:
        print(f"Warning: diffusion filter size {max_sigma:.1f} pixels is too large for the image size {image_size}. Capping it to {image_size / 6:.1f} pixels.")
        max_sigma = image_size / 6
    
    radius = max(int(np.ceil(max_sigma * 3)), 0)
    result = np.pad(data, ((radius, radius), (radius, radius), (0, 0)), mode='reflect') if radius > 0 else data.copy()
    result_fft = np.fft.fft2(result, axes=(0, 1))
    for _ in range(iterations):
        blurred_fft = scipy.ndimage.fourier_gaussian(result_fft, sigma=(sigma, sigma, 0))
        result_fft = diffusion_fraction * blurred_fft + (1 - diffusion_fraction) * result_fft
        sigma *= growth
        diffusion_fraction *= decay
    result = np.fft.ifft2(result_fft, axes=(0, 1)).real

    if radius > 0:
        return result[radius:-radius, radius:-radius, :]
    return result


from scipy.signal import fftconvolve


# Per-family strength-independent PSF shape, as three groups
# {core, halo, bloom} each expanded into a small geometric progression of
# 2D isotropic exponentials. The full PSF is
#
#     K_s(r) = w_c * K_core(r) + w_h * K_halo(r) + w_b * K_bloom(r)
#
# with w_c + w_h + w_b = 1 and each group itself a sum of N exponentials
#
#     K_x(r) = sum_k w_x[k] * E(r; lambda_x[k]),
#     E(r; lambda) = exp(-r / lambda) / (2 * pi * lambda**2),
#
# normalised so sum_k w_x[k] = 1 inside each group. Sub-component
# wavelengths span a geometric progression in [lambda_um / spread,
# lambda_um * spread]. For core and halo the within-group weights are
# uniform — every exponential in the progression contributes equally,
# which gives a smooth log-normal-ish blur with no internal knees. For
# the bloom the within-group weights follow a power-law schedule
# w_k ∝ lambda_k**(2 - alpha): a continuous integral of this schedule
# over lambda gives a 2D radial profile decaying as r**(-alpha) at
# large r, so the discretised bloom approximates a power-law with
# tail exponent alpha — replacing the explicit Student-t we had before
# while keeping everything in one basis family (no glued seams between
# exponential and power-law components).
#
# Why this is smoother than the previous core+halo+bloom-each-a-single-
# exponential model: adjacent groups' lambda ranges overlap, so the
# transition between e.g. halo and bloom happens through a partially
# overlapping band of scales rather than at one hard handoff radius.
# Energy distributes across scales continuously and the radial profile
# loses its visible knees.
#
# Per-family values are distilled from
# notes/empirical_characterization_of_diffusion_filters.md:
#
#     resolution preservation  : glimmerglass > BPM > pro_mist > cinebloom
#     halo prominence/radius   : cinebloom >= pro_mist >= BPM > glimmerglass
#     veil / shadow lift       : cinebloom > pro_mist >= BPM > glimmerglass
#     absorption (deep blacks) : BPM only (black-particle design)
#     halo color (warmth axis) : cinebloom > BPM > pro_mist > glimmerglass
#
# Halo color: each family carries a `halo_warmth_base` that drives a
# physically-motivated, energy-conserving redistribution of weights
# inside the halo group. Warmth > 0 pushes warm light (R + slight G)
# toward the OUTER halo sub-components and cool light (B) toward the
# inner ones (and vice versa for warmth < 0) — the warm-rim / cool-core
# look that reference images of mist and bloom filters consistently
# show around practical light sources. The per-channel weights are
# clipped and renormalized so total halo energy per channel is
# preserved exactly even when the modulation is strong. The user knob
# `DiffusionFilterParams.halo_warmth` adds on top of the per-family
# base. Bloom and core stay channel-independent.
_DIFFUSION_FILTER_SHAPES: dict[str, dict] = {
    # Glimmerglass — "tight, clean, sharp-preserving, faint long tail".
    # Core-dominant: most energy in the unscattered + diffraction-limited
    # core; halo small and tight; bloom small but with a non-negligible
    # long-reach tail (alpha ~3.2) — reference images show even the
    # subtlest mist filter lifts shadows slightly far from highlights, so
    # bloom must not be vestigial. Halo neutral in colour (the empirical
    # bluish flares are specular artifacts not modelled here). The user
    # character: subtle micro-contrast smoothing, almost no veiling,
    # blacks intact, but a measurable long decay around bright sources.
    'glimmerglass': {
        'core':  {'lambda_um':  10.0, 'spread': 1.5, 'n_components': 2},
        'halo':  {'lambda_um':  50.0, 'spread': 2.0, 'n_components': 3},
        'bloom': {'lambda_um': 260.0, 'spread': 2.5, 'n_components': 4, 'alpha': 3.2},
        'w_c': 0.60, 'w_h': 0.30, 'w_b': 0.10,
        'halo_warmth_base': 0.0,
    },
    # Black Pro-Mist — "concentrated punchy halo, fast falloff, deep blacks".
    # Halo-dominant: more weight in halo than in bloom, with a tighter halo
    # scale and a steeper bloom tail than pro_mist or cinebloom — energy
    # stays close to the highlight rather than veiling the frame. Warm
    # outer halo / cool inner core with the slight yellow-green bias the
    # empirical tests pick up around practical lights.
    'black_pro_mist': {
        'core':  {'lambda_um':  16.0, 'spread': 1.5, 'n_components': 2},
        'halo':  {'lambda_um':  95.0, 'spread': 2.0, 'n_components': 3},
        'bloom': {'lambda_um': 380.0, 'spread': 2.5, 'n_components': 4, 'alpha': 3.5},
        'w_c': 0.40, 'w_h': 0.47, 'w_b': 0.13,
        'halo_warmth_base': 0.65,
    },
    # Classic Pro-Mist — "atmospheric pastel, broad halo, balanced veil".
    # Halo broader than BPM, bloom heavier than BPM, balanced halo-and-
    # bloom weight split. Less localised than BPM, less frame-wide than
    # cinebloom; the look is a soft pastel mid-air haze. Warm outer halo
    # moderate.
    'pro_mist': {
        'core':  {'lambda_um':  14.0, 'spread': 1.5, 'n_components': 2},
        'halo':  {'lambda_um': 150.0, 'spread': 2.0, 'n_components': 3},
        'bloom': {'lambda_um': 650.0, 'spread': 2.5, 'n_components': 4, 'alpha': 2.9},
        'w_c': 0.28, 'w_h': 0.42, 'w_b': 0.30,
        'halo_warmth_base': 0.40,
    },
    # CineBloom — "frame-wide reach, slow tail, retro veil".
    # Bloom-dominant: bloom carries roughly half of the scattered energy,
    # with the longest scale and the shallowest tail of the four (alpha
    # near the heavy-tailed Lévy regime). The halo is the broadest of the
    # four but proportionally less peaked than BPM/pro_mist (large lambda,
    # smaller w_h share) — the reference images show a halo that "spreads
    # rather than punches". Warmest outer rim with the strongest yellow
    # lean. The tail does most of the visual work.
    'cinebloom': {
        'core':  {'lambda_um':  20.0, 'spread': 1.5, 'n_components': 2},
        'halo':  {'lambda_um': 200.0, 'spread': 2.0, 'n_components': 3},
        'bloom': {'lambda_um': 1000.0, 'spread': 2.5, 'n_components': 4, 'alpha': 2.5},
        'w_c': 0.22, 'w_h': 0.30, 'w_b': 0.48,
        'halo_warmth_base': 0.85,
    },
}

# Per-family scaling on the shared scatter-fraction saturation table. Sets
# the overall "deflection efficiency" of the filter at a given commercial
# stop. Glimmerglass deflects fewer photons than the mist filters
# (matching its mild perceived effect); BPM keeps a bit lower scatter than
# pro_mist (its "deep blacks" character lives in low scatter + low bloom
# weight, since the model is energy-conserving — see _strength_to_scatter
# docstring). Cinebloom is the heaviest scatterer but is held below
# pro_mist's gain so its strength=2 setting stays usable: at high p_s the
# very wide / shallow-tailed bloom would otherwise wash the frame.
_DIFFUSION_FAMILY_TOTAL_GAIN: dict[str, float] = {
    'glimmerglass':   0.65,
    'black_pro_mist': 0.75,
    'pro_mist':       1.05,
    'cinebloom':      1.00,
}

DIFFUSION_FILTER_FAMILIES: tuple[str, ...] = tuple(_DIFFUSION_FILTER_SHAPES)


# Strength -> deflected fraction p_s. Tabulated at commercial filter stops
# and log2-interpolated. Matches the saturation progression listed in plan
# §5 (BPM column, since the table was originally calibrated for the
# scattered+absorbed total).
_DIFFUSION_STRENGTH_BREAKPOINTS = np.array([0.125, 0.25, 0.5, 1.0, 2.0], dtype=np.float64)
_DIFFUSION_STRENGTH_TOTAL_FRACTION = np.array([0.10, 0.20, 0.35, 0.55, 0.75], dtype=np.float64)


def _strength_to_scatter(strength: float, family: str) -> float:
    """Map filter strength and family to p_s, the deflected-photon fraction.

    The diffusion-filter model is energy-conserving: the only effect of the
    filter is to redistribute photons between unscattered (T_0 = 1 - p_s)
    and scattered (p_s, convolved with the per-family PSF) populations.
    No absorption — output integrates to the same energy as input across
    all four families. BPM's empirical "deep blacks" property is encoded
    in its lower per-family scatter gain plus a smaller bloom weight than
    pro_mist or cinebloom, not in absorption.
    """
    if strength <= 0:
        return 0.0
    log_strength = np.log2(np.clip(strength, 1e-6, None))
    log_breaks = np.log2(_DIFFUSION_STRENGTH_BREAKPOINTS)
    base_total = float(np.interp(log_strength, log_breaks, _DIFFUSION_STRENGTH_TOTAL_FRACTION))
    gain = _DIFFUSION_FAMILY_TOTAL_GAIN.get(family, 1.0)
    return float(np.clip(base_total * gain, 0.0, 0.99))


def _expand_group(group_cfg: dict, *, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Expand a {core|halo|bloom} block into (lambdas_um, weights).

    Returns sub-component decay constants and weights summing to 1 inside
    the group; the family weight w_x is multiplied in by the caller. Sub-
    component lambdas are placed on a geometric progression in
    [lambda_um / spread, lambda_um * spread]. Within-group weights are:
      - uniform for `core` and `halo` (smooth log-normal-ish blur);
      - power-law w_k ∝ lambda_k**(2 - alpha) for `bloom`, so the
        assembled bloom decays as r**(-alpha) at large r.
    """
    lambda_center = float(group_cfg['lambda_um'])
    spread = float(group_cfg.get('spread', 1.0))
    n = max(int(group_cfg.get('n_components', 1)), 1)
    if n == 1 or spread <= 1.0:
        return np.array([lambda_center], dtype=np.float64), np.array([1.0], dtype=np.float64)

    log_lo = np.log(lambda_center / spread)
    log_hi = np.log(lambda_center * spread)
    lambdas = np.exp(np.linspace(log_lo, log_hi, n))

    if kind == 'bloom':
        alpha = float(group_cfg.get('alpha', 3.0))
        weights = lambdas ** (2.0 - alpha)
    else:
        weights = np.ones_like(lambdas)
    weights = weights / weights.sum()
    return lambdas, weights


# Per-channel coefficients on the warmth axis: positive a_c means warmth>0
# enhances channel c at the OUTER halo (and suppresses it at the inner
# halo, by the symmetric gradient defined in `_halo_channel_weights`).
# The axis is biased toward the yellow-green corner (R strong, G mild, B
# suppressed-and-some), so a "warm" outer halo reads as warm-yellow
# rather than pure red — matching the empirical observations of warm
# yellow rims on cinebloom and BPM around practical lights, with the
# inner halo cooler. The amplitudes are larger than ±1 to give a visibly
# tinted halo at moderate warmth values; the clip-and-renormalize step
# in `_halo_channel_weights` keeps individual sub-component weights
# non-negative and per-channel halo energy preserved.
_HALO_CHANNEL_WARMTH_AXIS = np.array([+1.30, +0.15, -1.45], dtype=np.float64)


def _halo_channel_weights(weights: np.ndarray, warmth: float) -> np.ndarray:
    """Energy-conserving per-channel halo weight redistribution.

    Returns shape (3, N) array where each row is the halo sub-component
    weights for one of (R, G, B). Total halo energy per channel equals
    sum(weights) for any warmth, by construction:
      - g_k = symmetric "radial gradient" linearly from -1 at the
        innermost (smallest lambda) sub-component to +1 at the outermost,
        re-centred against `weights` so that sum_k weights[k] * g_k = 0.
      - per-channel coefficient a_c on the warmth axis (R warm, G mild
        warm, B cool).
      - modulation: weights[k] -> weights[k] * (1 + warmth * a_c * g_k),
        clipped to >= 0 and renormalized per channel so the channel total
        equals sum(weights). The renormalization is what enforces energy
        conservation when modulations are large enough to drive the raw
        product negative; for small warmth it is a no-op (the centred
        gradient already sums to zero against weights).

    With g going inner→outer as -1→+1 and the warmth axis biased warm
    toward (R+, G+) and cool toward B-, warmth > 0 pushes warm light
    toward the OUTER halo and cool light toward the inner halo — the
    warm-rim-with-cool-core look observed in reference images of mist
    and bloom filters. Warmth < 0 inverts this.

    Warmth is soft-clamped to [-1.5, 1.5]; at full warmth combined with
    the family base, the outer halo can be roughly twice as bright in R
    as it would be without modulation, with a corresponding suppression
    on the inner ring (and the opposite for B).
    """
    n = len(weights)
    if n < 2:
        return np.tile(weights, (3, 1))
    warmth = float(np.clip(warmth, -1.5, 1.5))
    g = np.linspace(-1.0, 1.0, n)
    g = g - np.average(g, weights=weights)
    target_total = float(np.sum(weights))
    out = np.empty((3, n), dtype=np.float64)
    for c in range(3):
        raw = weights * (1.0 + warmth * _HALO_CHANNEL_WARMTH_AXIS[c] * g)
        raw = np.maximum(raw, 0.0)
        s = raw.sum()
        if s > 0.0:
            out[c] = raw * (target_total / s)
        else:
            out[c] = weights
    return out


def _resolve_family_cfg(family: str, overrides: dict | None = None) -> dict:
    """Return the family cfg dict with optional per-group multipliers applied.

    Overrides keys (all default 1.0):
      - core_intensity, halo_intensity, bloom_intensity: scale the
        family weights w_c / w_h / w_b. The three are then renormalized
        so they still sum to 1, i.e. the kernel stays unit-normalised
        and the strength → p_s mapping is unchanged.
      - core_size, halo_size, bloom_size: scale each group's lambda_um
        uniformly (all sub-components in the group stretched together).
    """
    base = _DIFFUSION_FILTER_SHAPES[family]
    if overrides is None:
        return base
    ci = float(overrides.get('core_intensity', 1.0))
    hi = float(overrides.get('halo_intensity', 1.0))
    bi = float(overrides.get('bloom_intensity', 1.0))
    cs = float(overrides.get('core_size', 1.0))
    hs = float(overrides.get('halo_size', 1.0))
    bs = float(overrides.get('bloom_size', 1.0))
    if ci == 1.0 and hi == 1.0 and bi == 1.0 and cs == 1.0 and hs == 1.0 and bs == 1.0:
        return base
    w_c = float(base['w_c']) * max(ci, 0.0)
    w_h = float(base['w_h']) * max(hi, 0.0)
    w_b = float(base['w_b']) * max(bi, 0.0)
    total = w_c + w_h + w_b
    if total <= 0.0:
        return base
    return {
        **base,
        'core':  {**base['core'],  'lambda_um': float(base['core']['lambda_um'])  * max(cs, 1e-6)},
        'halo':  {**base['halo'],  'lambda_um': float(base['halo']['lambda_um'])  * max(hs, 1e-6)},
        'bloom': {**base['bloom'], 'lambda_um': float(base['bloom']['lambda_um']) * max(bs, 1e-6)},
        'w_c': w_c / total,
        'w_h': w_h / total,
        'w_b': w_b / total,
    }


def _bloom_max_lambda_um(family: str, overrides: dict | None = None) -> float:
    """Largest lambda in the bloom progression for a family (image-plane μm)."""
    cfg = _resolve_family_cfg(family, overrides)
    bloom = cfg['bloom']
    return float(bloom['lambda_um']) * float(bloom.get('spread', 1.0))


_OVERRIDE_KEYS = (
    'core_intensity', 'halo_intensity', 'bloom_intensity',
    'core_size', 'halo_size', 'bloom_size',
)


def _overrides_from_params(diffusion_filter) -> dict | None:
    """Pull per-group multipliers off a DiffusionFilterParams instance.

    Returns None if every multiplier is at its 1.0 default (so the fast
    path in `_resolve_family_cfg` skips the dict allocation).
    """
    out = {}
    any_set = False
    for key in _OVERRIDE_KEYS:
        v = getattr(diffusion_filter, key, 1.0)
        out[key] = float(v)
        if float(v) != 1.0:
            any_set = True
    return out if any_set else None


def _radial_components(
    radius_or_pixel_grid: np.ndarray,
    *,
    family: str,
    spatial_scale: float,
    pixel_size_um: float,
    halo_warmth: float,
    overrides: dict | None = None,
) -> dict[str, np.ndarray]:
    """Build core / per-channel halo / bloom radial contributions.

    `radius_or_pixel_grid` is in pixel units already if `pixel_size_um` is
    set to a finite value; for the analytic profile in image-plane μm,
    pass `pixel_size_um=1.0` and `radius_or_pixel_grid` already in μm.

    Returns weighted contributions K_x with the family weights w_x already
    multiplied in. The halo entry is shape (3, ...) per channel.
    """
    cfg = _resolve_family_cfg(family, overrides)
    spatial_scale = max(float(spatial_scale), 1e-6)

    core_lambdas, core_weights = _expand_group(cfg['core'], kind='core')
    halo_lambdas, halo_weights = _expand_group(cfg['halo'], kind='halo')
    bloom_lambdas, bloom_weights = _expand_group(cfg['bloom'], kind='bloom')

    halo_per_ch = _halo_channel_weights(halo_weights, halo_warmth)

    core_lambdas_px = core_lambdas * spatial_scale / pixel_size_um
    halo_lambdas_px = halo_lambdas * spatial_scale / pixel_size_um
    bloom_lambdas_px = bloom_lambdas * spatial_scale / pixel_size_um

    r = np.asarray(radius_or_pixel_grid, dtype=np.float64)

    def _exp_sum(lambdas_px: np.ndarray, weights: np.ndarray) -> np.ndarray:
        total = np.zeros_like(r)
        for wk, lk in zip(weights, lambdas_px):
            lk = max(float(lk), 1e-6)
            total += wk * np.exp(-r / lk) / (2.0 * np.pi * lk ** 2)
        return total

    core = cfg['w_c'] * _exp_sum(core_lambdas_px, core_weights)
    bloom = cfg['w_b'] * _exp_sum(bloom_lambdas_px, bloom_weights)

    halo_channels = np.stack(
        [cfg['w_h'] * _exp_sum(halo_lambdas_px, halo_per_ch[c]) for c in range(3)],
        axis=0,
    )

    return {'core': core, 'halo': halo_channels, 'bloom': bloom}


def diffusion_filter_radial_profile(
    radius_um: np.ndarray,
    *,
    family: str = 'black_pro_mist',
    spatial_scale: float = 1.0,
    halo_warmth: float = 0.0,
    overrides: dict | None = None,
) -> dict[str, np.ndarray]:
    """Analytic radial profile of the diffusion-filter PSF, unit-normalised in 2D.

    Returns each component contribution in 1/μm**2:
      - 'core', 'bloom': shape matching `radius_um` (channel-independent).
      - 'halo': shape (3, *radius_um.shape), one per channel.
      - 'total_per_channel': shape (3, *radius_um.shape), sum of all three.
    `halo_warmth` is added to the family base before the channel weights
    are redistributed. `overrides` accepts the same per-group multiplier
    keys as `_resolve_family_cfg`.
    """
    if family not in _DIFFUSION_FILTER_SHAPES:
        raise ValueError(f"Unknown diffusion filter family: {family!r}; "
                         f"available: {list(_DIFFUSION_FILTER_SHAPES)}")
    cfg = _resolve_family_cfg(family, overrides)
    effective_warmth = float(cfg.get('halo_warmth_base', 0.0)) + float(halo_warmth)
    parts = _radial_components(
        np.asarray(radius_um, dtype=np.float64),
        family=family,
        spatial_scale=spatial_scale,
        pixel_size_um=1.0,
        halo_warmth=effective_warmth,
        overrides=overrides,
    )
    total_per_channel = parts['halo'] + parts['core'][None, ...] + parts['bloom'][None, ...]
    return {
        'core': parts['core'],
        'halo': parts['halo'],
        'bloom': parts['bloom'],
        'total_per_channel': total_per_channel,
    }


def diffusion_filter_psf(
    kernel_shape: tuple[int, int],
    *,
    family: str,
    spatial_scale: float,
    pixel_size_um: float,
    halo_warmth: float = 0.0,
    overrides: dict | None = None,
) -> np.ndarray:
    """Per-channel 2D PSF for a diffusion filter.

    Returns shape (height, width, 3), each channel sum-normalised on the
    grid. The PSF is per-channel because halo weights vary per channel
    via the energy-conserving warmth redistribution; core and bloom
    contributions are channel-independent and shared across the three.
    `overrides` accepts the same per-group multiplier keys as
    `_resolve_family_cfg`.
    """
    if family not in _DIFFUSION_FILTER_SHAPES:
        raise ValueError(f"Unknown diffusion filter family: {family!r}; "
                         f"available: {list(_DIFFUSION_FILTER_SHAPES)}")
    cfg = _resolve_family_cfg(family, overrides)
    effective_warmth = float(cfg.get('halo_warmth_base', 0.0)) + float(halo_warmth)

    y, x = np.ogrid[:kernel_shape[0], :kernel_shape[1]]
    cy, cx = kernel_shape[0] // 2, kernel_shape[1] // 2
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(np.float64)

    parts = _radial_components(
        r,
        family=family,
        spatial_scale=spatial_scale,
        pixel_size_um=pixel_size_um,
        halo_warmth=effective_warmth,
        overrides=overrides,
    )
    psf = np.empty((kernel_shape[0], kernel_shape[1], 3), dtype=np.float64)
    for c in range(3):
        psf[..., c] = parts['core'] + parts['halo'][c] + parts['bloom']
        psf[..., c] /= psf[..., c].sum()
    return psf


def apply_diffusion_filter_um(image, diffusion_filter, pixel_size_um):
    """Apply a diffusion-filter PSF to an RGB image.

    Implements the energy-conserving convex combination
        E_out = (1 - p_s) * E_in  +  p_s * (K_s * E_in)
    with p_s derived from `strength` and `filter_family`. K_s is per-channel
    because the halo is colour-tinted via an energy-conserving redistribution
    across its sub-components; total scatter energy per channel = p_s.
    """
    if not diffusion_filter.active:
        return image
    if diffusion_filter.strength <= 0 or diffusion_filter.spatial_scale <= 0:
        return image
    family = diffusion_filter.filter_family
    if family not in _DIFFUSION_FILTER_SHAPES:
        raise ValueError(f"Unknown diffusion filter family: {family!r}; "
                         f"available: {list(_DIFFUSION_FILTER_SHAPES)}")

    p_s = _strength_to_scatter(diffusion_filter.strength, family)
    if p_s <= 0:
        return image

    overrides = _overrides_from_params(diffusion_filter)

    # Kernel radius: a single 2D exponential at lambda_max has 99.95% of
    # its energy inside r = 8 lambda_max (since the 2D radial CDF is
    # 1 - (1 + r/lambda) * exp(-r/lambda)). The bloom progression's
    # outermost sub-component carries a small but non-negligible weight
    # for shallow alpha, so 8 lambda_max is the right truncation budget.
    bloom_max_lambda_px = (
        _bloom_max_lambda_um(family, overrides) * diffusion_filter.spatial_scale / pixel_size_um
    )
    radius = int(np.ceil(max(8.0 * bloom_max_lambda_px, 5.0)))
    radius = min(radius, max(min(image.shape[:2]) // 2 - 1, 1))

    psf_shape = (2 * radius + 1, 2 * radius + 1)
    halo_warmth = float(getattr(diffusion_filter, 'halo_warmth', 0.0))
    psf_per_channel = diffusion_filter_psf(
        psf_shape,
        family=family,
        spatial_scale=diffusion_filter.spatial_scale,
        pixel_size_um=pixel_size_um,
        halo_warmth=halo_warmth,
        overrides=overrides,
    )

    padded = np.pad(image, ((radius, radius), (radius, radius), (0, 0)), mode='reflect')
    blurred = np.empty_like(padded)
    for channel in range(image.shape[2]):
        blurred[:, :, channel] = fftconvolve(
            padded[:, :, channel], psf_per_channel[..., channel], mode='same',
        )
    blurred = blurred[radius:-radius, radius:-radius, :]

    return (1.0 - p_s) * image + p_s * blurred



