import numpy as np
import scipy
import scipy.ndimage
from spektrafilm.model.density_curves import interp_density_cmy_layers_channel
from spektrafilm.model.diffusion import match_channels, apply_multiplicative_unsharp_mask
from spektrafilm.runtime.params_schema import GrainParams
from spektrafilm.utils.fast_stats import fast_binomial, fast_poisson
from spektrafilm.utils.fast_gaussian_filter import fast_gaussian_filter

################################################################################
# Grain (very simple model)
################################################################################

# Grain is stochastic noise, so float32 is ample precision and halves the
# memory of the (large, full-resolution) working arrays versus float64. The
# grain functions compute in this dtype internally and cast the result back to
# the caller's dtype on return.
_GRAIN_WORK_DTYPE = np.float32

def layer_particle_model(density,
                         density_max=2.2,
                         n_particles_per_pixel=10,
                         grain_uniformity=0.98,
                         seed=None,
                         blur_particle=0.0,
                         method='poisson_binomial',
                         use_fast_stats=False,
                         ):
    if seed is not None:
        np.random.seed(seed) # scipy uses np.random

    # Keep the scalar parameters as plain Python floats so a float32 `density`
    # is not silently promoted back to float64 by numpy's scalar rules; the
    # grain then stays in the caller's (float32) working dtype throughout.
    density_max = float(density_max)
    n_particles_per_pixel = float(n_particles_per_pixel)
    grain_uniformity = float(grain_uniformity)

    probability_of_development = density/density_max
    probability_of_development = np.clip(probability_of_development, 1e-6, 1-1e-6) # for safe calc
    od_particle = density_max/n_particles_per_pixel

    if method=='gamma_beta':
        gamma_rvs = scipy.stats.gamma.rvs
        beta_rvs = scipy.stats.beta.rvs
        seeds = gamma_rvs(n_particles_per_pixel/(1-grain_uniformity+1e-6), size=density.shape) * (1-grain_uniformity+1e-6)
        grain = beta_rvs(probability_of_development*n_particles_per_pixel,
                        (1-probability_of_development)*n_particles_per_pixel)*seeds*od_particle
    elif method=='poisson_binomial':
        if use_fast_stats:
            binom_rvs = fast_binomial
            poisson_rvs = fast_poisson
        else:
            binom_rvs = scipy.stats.binom.rvs
            poisson_rvs = scipy.stats.poisson.rvs
        saturation = 1 - probability_of_development*grain_uniformity*(1-1e-6)
        seeds = poisson_rvs(n_particles_per_pixel/saturation)
        # Cast the integer binomial counts to the working dtype, then fold in
        # the per-particle optical density and saturation in place so no extra
        # full-size temporaries are made.
        grain = binom_rvs(seeds, probability_of_development).astype(density.dtype, copy=False)
        grain *= od_particle
        grain *= saturation
    else:
        grain = np.zeros_like(density)

    if blur_particle>0:
        # grain = scipy.ndimage.gaussian_filter(grain, blur_particle*np.sqrt(od_particle))
        grain = fast_gaussian_filter(grain, blur_particle*np.sqrt(od_particle))
    return grain

def add_micro_structure(density_cmy_out, micro_structure, pixel_size_um):
    grain_micro_structure_blur_pixel = micro_structure[0]/pixel_size_um
    grain_micro_structure_sigma = micro_structure[1]*0.001/pixel_size_um  # grain microstructure[1] is in nm
    if grain_micro_structure_sigma > 0.05:
        # Multiplicative lognormal clumping with linear-space mean 1 and std
        # sigma. Generated directly into a single buffer instead of via
        # fast_lognormal_from_mean_std(ones, ones*sigma), which would allocate
        # several full-size arrays (two ones params + two internal mu/sigma
        # grids + the result) just to carry constants. For mean 1 the log-space
        # parameters are sigma2 = ln(1+sigma^2), mu = -sigma2/2.
        sigma2 = float(np.log1p(grain_micro_structure_sigma ** 2))
        clumping = np.random.standard_normal(density_cmy_out.shape)
        clumping *= np.sqrt(sigma2)
        clumping -= 0.5 * sigma2
        np.exp(clumping, out=clumping)
        if grain_micro_structure_blur_pixel>0.4:
            clumping = fast_gaussian_filter(clumping, grain_micro_structure_blur_pixel)
        density_cmy_out *= clumping
    return density_cmy_out

def apply_grain_to_density(density_cmy,
                           pixel_size_um=10,
                           particle_area_um2=0.2,
                           particle_scale=[1,0.8,3],
                           density_min=[0.03,0.06,0.04],
                           density_max_curves=[2.2,2.2,2.2],
                           grain_uniformity=[0.98,0.98,0.98],
                           grain_blur=1.0,
                           n_sub_layers=1,
                           fixed_seed=None,
                           usm_sigma=0.0,
                           usm_amount=0.0,
                           ):
    n_ch = density_cmy.shape[-1]
    density_min = match_channels(density_min, n_ch)
    density_max = match_channels(density_max_curves, n_ch) + density_min
    pixel_area_um2 = pixel_size_um**2
    particle_area_um2 = particle_area_um2 * match_channels(particle_scale, n_ch)
    n_particles_per_pixel = pixel_area_um2/particle_area_um2
    grain_uniformity = match_channels(grain_uniformity, n_ch)
    sigma_blur_pixel = grain_blur
    
    if fixed_seed is not None:
        seed = None
    else:
        seed = list(range(n_ch))
    
    if n_sub_layers>1:
        n_particles_per_pixel /= n_sub_layers
    
    density_cmy += density_min
    density_cmy_out = np.zeros_like(density_cmy)
    for ch in np.arange(density_cmy.shape[-1]):
        for sl in np.arange(n_sub_layers):
            density_cmy_out[:,:,ch] += layer_particle_model(density_cmy[:,:,ch],
                                                            density_max=density_max[ch],
                                                            n_particles_per_pixel=n_particles_per_pixel[ch],
                                                            grain_uniformity=grain_uniformity[ch],
                                                            seed=seed[ch] + sl*10)
    density_cmy_out /= n_sub_layers

    if sigma_blur_pixel>0.4:
        # density_cmy_out = scipy.ndimage.gaussian_filter(density_cmy_out, (sigma_blur_pixel, sigma_blur_pixel, 0))
        density_cmy_out = fast_gaussian_filter(density_cmy_out, sigma_blur_pixel)
    # Mass-conserving density USM on the absolute (positive) density, before the
    # floor is removed, so positivity holds (see diffusion.apply_multiplicative_unsharp_mask).
    if usm_amount > 0 and usm_sigma > 0:
        density_cmy_out = apply_multiplicative_unsharp_mask(density_cmy_out, usm_sigma, usm_amount)
    density_cmy_out -= density_min

    return density_cmy_out


# experimental
def _layer_grain_params(density_max_layers, density_min, particle_area_um2,
                        particle_scale, particle_scale_layers, n_ch, pixel_size_um):
    """Per (sub-layer, channel) grain parameters shared by the whole-array and
    streaming code paths.

    Returns ``(density_min, density_min_layers, density_max_layers,
    n_particles_per_pixel)`` where the per-layer arrays are indexed
    ``[sub-layer, channel]`` and ``density_max_layers`` already includes the
    density floor.
    """
    density_max_total = np.sum(density_max_layers, axis=0)            # [channel]
    density_max_fractions = density_max_layers / density_max_total[None, :]
    density_min = match_channels(density_min, n_ch)
    density_min_layers = density_max_fractions * density_min[None, :]  # [sub-layer, channel]
    density_max_layers = density_max_layers + density_min_layers

    pixel_area_um2 = pixel_size_um ** 2
    particle_area_um2_layers = (particle_area_um2
                                * match_channels(particle_scale, n_ch)[None, :]
                                * np.array(particle_scale_layers)[:, None])  # [sub-layer, channel]
    n_particles_per_pixel = pixel_area_um2 * density_max_fractions / particle_area_um2_layers
    return density_min, density_min_layers, density_max_layers, n_particles_per_pixel


def _channel_sublayer_grain(layers_ch, density_max_layers_ch, n_particles_ch,
                            grain_uniformity_ch, seed_base, blur_dye_clouds_um, use_fast_stats):
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


def _finalize_grain(density_cmy_out, density_min, grain_micro_structure, pixel_size_um,
                    grain_blur, usm_sigma=0.0, usm_amount=0.0):
    """Shared grain finishing: optical micro-structure clumping, the
    pixel-correlating blur, the mass-conserving density unsharp mask (applied on
    the absolute density, before the floor is removed, so it stays positive),
    and removal of the density floor."""
    density_cmy_out = add_micro_structure(density_cmy_out, grain_micro_structure, pixel_size_um)
    if grain_blur > 0:
        density_cmy_out = fast_gaussian_filter(density_cmy_out, grain_blur)
    if usm_amount > 0 and usm_sigma > 0:
        density_cmy_out = apply_multiplicative_unsharp_mask(density_cmy_out, usm_sigma, usm_amount)
    density_cmy_out -= density_min
    return density_cmy_out


def apply_grain_to_density_layers(density_cmy_layers, # x,y,sublayers,rgb
                                  density_max_layers, # 3x3 [sublayers,rgb]
                                  pixel_size_um=10,
                                  particle_area_um2=0.2,
                                  particle_scale=[1,0.8,3], # rgb
                                  particle_scale_layers=[3,1,0.3], # sublayers
                                  density_min=[0.03,0.06,0.04],
                                  grain_uniformity=[0.98,0.98,0.98],
                                  grain_blur=1.0,
                                  grain_blur_dye_clouds_um=1.0,
                                  grain_micro_structure=(0.1, 30),
                                  fixed_seed=None,
                                  use_fast_stats=False,
                                  usm_sigma=0.0,
                                  usm_amount=0.0,
                                  ):
    # Whole-array entry: grains a pre-built (H, W, sublayers, channels) stack.
    # `apply_grain` uses the streaming path below instead; this stays for direct
    # callers/tests. Output dtype matches the input.
    out_dtype = density_cmy_layers.dtype
    n_ch = density_cmy_layers.shape[3]
    density_min, density_min_layers, density_max_layers, n_particles_per_pixel = _layer_grain_params(
        density_max_layers, density_min, particle_area_um2,
        particle_scale, particle_scale_layers, n_ch, pixel_size_um)
    grain_uniformity = match_channels(grain_uniformity, n_ch)

    # Offset each layer by its floor and grain channel-by-channel in float32.
    density_cmy_layers = density_cmy_layers.astype(_GRAIN_WORK_DTYPE, copy=False)
    density_cmy_layers += density_min_layers
    density_cmy_out = np.empty(density_cmy_layers.shape[0:2] + (n_ch,), dtype=_GRAIN_WORK_DTYPE)
    for ch in range(n_ch):
        density_cmy_out[:, :, ch] = _channel_sublayer_grain(
            density_cmy_layers[:, :, :, ch], density_max_layers[:, ch], n_particles_per_pixel[:, ch],
            grain_uniformity[ch], None if fixed_seed is not None else ch,
            grain_blur_dye_clouds_um, use_fast_stats)

    density_cmy_out = _finalize_grain(density_cmy_out, density_min, grain_micro_structure,
                                      pixel_size_um, grain_blur, usm_sigma, usm_amount)
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

    if not grain.sublayers_active or density_curves_layers is None:
        density_max = np.nanmax(density_curves, axis=0)
        return apply_grain_to_density(
            density_cmy,
            pixel_size_um=pixel_size_um,
            particle_area_um2=grain.particle_area_um2,
            particle_scale=grain.particle_scale,
            density_min=grain.density_min,
            density_max_curves=density_max,
            grain_uniformity=grain.uniformity,
            grain_blur=grain.blur,
            n_sub_layers=grain.n_sub_layers,
            usm_sigma=grain.mult_usm_sigma,
            usm_amount=grain.mult_usm_amount,
        )

    # Streaming sub-layer grain: build and grain one channel's sub-layer stack
    # at a time, so the full (H, W, sublayers, channels) array is never held —
    # only a single (H, W, sublayers) float32 slab. This is the memory-critical
    # path for full-resolution scans. Bit-identical to feeding the whole stack
    # to apply_grain_to_density_layers (same per-channel interpolation, seeds,
    # and float32 working dtype).
    positive_film = profile_type == 'positive'
    n_ch = density_cmy.shape[-1]
    density_max_layers = np.nanmax(density_curves_layers, axis=0)
    density_min, density_min_layers, density_max_layers, n_particles_per_pixel = _layer_grain_params(
        density_max_layers, grain.density_min, grain.particle_area_um2,
        grain.particle_scale, grain.particle_scale_layers, n_ch, pixel_size_um)
    grain_uniformity = match_channels(grain.uniformity, n_ch)

    density_cmy_out = np.empty(density_cmy.shape[0:2] + (n_ch,), dtype=_GRAIN_WORK_DTYPE)
    for ch in range(n_ch):
        layers_ch = interp_density_cmy_layers_channel(
            density_cmy[:, :, ch], density_curves[:, ch], density_curves_layers[:, :, ch],
            positive_film,
        ).astype(_GRAIN_WORK_DTYPE)
        layers_ch += density_min_layers[:, ch]   # offset each sub-layer by its floor
        density_cmy_out[:, :, ch] = _channel_sublayer_grain(
            layers_ch, density_max_layers[:, ch], n_particles_per_pixel[:, ch],
            grain_uniformity[ch], ch, grain.blur_dye_clouds_um, use_fast_stats)

    density_cmy_out = _finalize_grain(density_cmy_out, density_min, grain.micro_structure,
                                      pixel_size_um, grain.blur,
                                      grain.mult_usm_sigma, grain.mult_usm_amount)
    return density_cmy_out.astype(density_cmy.dtype, copy=False)

# TODO: make grain parameter with RMS granularity

if __name__=='__main__':
    density = np.ones((128,128))*2
    g1 = layer_particle_model(density, density_max=2, n_particles_per_pixel=10, grain_uniformity=0.99, )
    g2 = layer_particle_model(density, density_max=2, n_particles_per_pixel=10, grain_uniformity=0.96, )
    print('g1 ------------------')
    print('Density Test')
    print('Mean', np.mean(g1))
    print('RMS', np.std(g1)*1000)
    print('Skewness', scipy.stats.skew(g1.flatten()))
    print('Kurtosis', scipy.stats.kurtosis(g1.flatten()))
    print('g2 ------------------')
    print('Mean', np.mean(g2))
    print('RMS', np.std(g2)*1000)
    print('Skewness', scipy.stats.skew(g2.flatten()))
    print('Kurtosis', scipy.stats.kurtosis(g2.flatten()))
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1,2)
    axs[0].imshow(g1, vmin=0, vmax=2.2)
    axs[0].set_title('Uniformity=0.99')
    axs[1].imshow(g2, vmin=0, vmax=2.2)
    axs[1].set_title('Uniformity=0.96')
    fig.suptitle('Fully saturated density with different uniformity')
    plt.show()
