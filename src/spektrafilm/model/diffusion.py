import numpy as np
import scipy.ndimage
from scipy.interpolate import PchipInterpolator
# from spectral_film_lab.utils.fast_gaussian_filter import fast_gaussian_filter
# from spectral_film_lab.utils.fft_gaussian_filter import fft_gaussian_filter

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
    image_blur = scipy.ndimage.gaussian_filter(image, sigma=(sigma, sigma, 0))
    # image_blur = fast_gaussian_filter(image, sigma)
    image_sharp = image + amount * (image - image_blur)
    return image_sharp


def apply_halation_um(raw, halation, pixel_size_um):
    """
    Apply a halation effect to an image.

    Parameters:
    raw (numpy.ndarray): The input image array with shape (height, width, channels).
    halation_size (list or tuple): The size of the halation effect for each channel.
    halation_strength (list or tuple): The strength of the halation effect for each channel.
    scattering_size (list or tuple, optional): The size of the scattering effect for each channel. Default is [0, 0, 0].
    scattering_strength (list or tuple, optional): The strength of the scattering effect for each channel. Default is [0, 0, 0].

    Returns:
    numpy.ndarray: The image array with the halation effect applied.
    """
    
    halation_size_pixel = np.array(halation.size_um) / pixel_size_um
    halation_strength = np.array(halation.strength)
    scattering_size_pixel = np.array(halation.scattering_size_um) / pixel_size_um
    scattering_strength = np.array(halation.scattering_strength)
    
    if halation.active:
        for i in np.arange(3):
            if halation_strength[i]>0:
                raw[:,:,i] += halation_strength[i]*scipy.ndimage.gaussian_filter(raw[:,:,i], halation_size_pixel[i], truncate=7)
                raw[:,:,i] /= (1+halation_strength[i])
        # if np.any(halation_strength>0):
        #     raw += fast_gaussian_filter(raw, halation_size_pixel, truncate=7)*halation_strength
        #     raw /= (1+halation_strength)
                
        for i in np.arange(3):
            if scattering_strength[i]>0:
                raw[:,:,i] += scattering_strength[i]*scipy.ndimage.gaussian_filter(raw[:,:,i], scattering_size_pixel[i], truncate=7)
                raw[:,:,i] /= (1+scattering_strength[i])
        # if np.any(scattering_strength>0):
        #     raw += fast_gaussian_filter(raw, scattering_size_pixel)*scattering_strength
        #     raw /= (1+scattering_strength)
        
    return raw

def apply_gaussian_blur(data, sigma):
    if sigma > 0:
        return scipy.ndimage.gaussian_filter(data, (sigma, sigma, 0))
        # data = np.double(data)
        # data = np.ascontiguousarray(data)
        # return fast_gaussian_filter(data, sigma)
    else:
        return data
    
def apply_gaussian_blur_um(data, sigma_um, pixel_size_um):
    sigma = sigma_um / pixel_size_um
    if sigma > 0:
        return scipy.ndimage.gaussian_filter(data, (sigma, sigma, 0))
        # data = np.double(data)
        # data = np.ascontiguousarray(data)
        # return fast_gaussian_filter(data, sigma)
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


_PROMIST_STRENGTHS = np.array([0.125, 0.25, 0.5, 1.0], dtype=np.float64)
_PROMIST_PROFILE_TABLE = np.array([
        (10.0, 55.0, 180.0, 2.1, 0.32, 0.16, 0.22),
        (14.0, 90.0, 280.0, 1.9, 0.30, 0.22, 0.34),
        (20.0, 140.0, 420.0, 1.7, 0.28, 0.30, 0.48),
        (28.0, 220.0, 640.0, 1.55, 0.24, 0.40, 0.64),
], dtype=np.float64)
_PROMIST_LOG_STRENGTHS = np.log2(_PROMIST_STRENGTHS)
_PROMIST_PROFILE_INTERPOLATORS = tuple(
    PchipInterpolator(_PROMIST_LOG_STRENGTHS, _PROMIST_PROFILE_TABLE[:, column], extrapolate=True)
    for column in range(_PROMIST_PROFILE_TABLE.shape[1])
)


def _interpolate_promist_profile(strength_value):
    query_position = np.log2(strength_value)
    return np.array([interpolator(query_position) for interpolator in _PROMIST_PROFILE_INTERPOLATORS], dtype=np.float64)


def _promist_profile_params(strength):
    if strength <= 0:
        return (0.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0)

    sigma_core_um, sigma_halo_um, sigma_bloom_um, bloom_beta, halo_weight, bloom_weight, scatter_strength = _interpolate_promist_profile(strength)
    weights = np.array(
        [
            max(1.0 - halo_weight - bloom_weight, 0.0),
            max(halo_weight, 0.0),
            max(bloom_weight, 0.0),
        ],
        dtype=np.float64,
    )
    weight_sum = np.sum(weights)
    if weight_sum <= 0:
        weights = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        weights /= weight_sum

    return (
        float(sigma_core_um),
        float(sigma_halo_um),
        float(sigma_bloom_um),
        max(float(bloom_beta), 1.01),
        float(weights[0]),
        float(weights[1]),
        float(weights[2]),
        float(scatter_strength),
    )


def _promist_scatter_profile(radius_um, strength=0.25, spatial_scale=1.0):
    radius_um = np.maximum(np.asarray(radius_um, dtype=np.float64), 0.0)
    spatial_scale = max(float(spatial_scale), 0.0)
    if strength <= 0 or spatial_scale <= 0:
        zeros = np.zeros_like(radius_um, dtype=np.float64)
        return {
            'core': zeros,
            'halo': zeros,
            'bloom': zeros,
            'total': zeros,
            'cumulative': zeros,
            'weights': (1.0, 0.0, 0.0),
            'scales_um': (0.0, 0.0, 0.0),
            'bloom_beta': 2.0,
        }

    sigma_core_um, sigma_halo_um, sigma_bloom_um, bloom_beta, core_weight, halo_weight, bloom_weight, _ = _promist_profile_params(strength)
    sigma_core_um = max(sigma_core_um * spatial_scale, 1e-6)
    sigma_halo_um = max(sigma_halo_um * spatial_scale, 1e-6)
    sigma_bloom_um = max(sigma_bloom_um * spatial_scale, 1e-6)

    core_raw = np.exp(-0.5 * (radius_um / sigma_core_um) ** 2)
    halo_raw = np.exp(-0.5 * (radius_um / sigma_halo_um) ** 2)
    bloom_raw = (1.0 + (radius_um / sigma_bloom_um) ** 2) ** (-bloom_beta)

    core_norm = 2.0 * np.pi * sigma_core_um ** 2
    halo_norm = 2.0 * np.pi * sigma_halo_um ** 2
    bloom_norm = np.pi * sigma_bloom_um ** 2 / (bloom_beta - 1.0)
    normalization = core_weight * core_norm + halo_weight * halo_norm + bloom_weight * bloom_norm

    core = core_weight * core_raw / normalization
    halo = halo_weight * halo_raw / normalization
    bloom_component = bloom_weight * bloom_raw / normalization

    cumulative_core = core_weight * core_norm * (1.0 - core_raw) / normalization
    cumulative_halo = halo_weight * halo_norm * (1.0 - halo_raw) / normalization
    cumulative_bloom = bloom_weight * bloom_norm * (1.0 - (1.0 + (radius_um / sigma_bloom_um) ** 2) ** (1.0 - bloom_beta)) / normalization
    cumulative = np.clip(cumulative_core + cumulative_halo + cumulative_bloom, 0.0, 1.0)

    return {
        'core': core,
        'halo': halo,
        'bloom': bloom_component,
        'total': core + halo + bloom_component,
        'cumulative': cumulative,
        'weights': (core_weight, halo_weight, bloom_weight),
        'scales_um': (sigma_core_um, sigma_halo_um, sigma_bloom_um),
        'bloom_beta': bloom_beta,
    }


def _promist_characteristic_radius(radius_um, cumulative, fraction):
    cumulative = np.maximum.accumulate(np.asarray(cumulative, dtype=np.float64))
    return float(np.interp(float(fraction), cumulative, np.asarray(radius_um, dtype=np.float64)))


def promist_psf(kernel_shape, strength=0.25, pixel_size_um=1.0, spatial_scale=1.0):
    spatial_scale = max(float(spatial_scale), 0.0)
    if strength <= 0 or spatial_scale <= 0:
        psf = np.zeros(kernel_shape, dtype=np.float64)
        psf[kernel_shape[0] // 2, kernel_shape[1] // 2] = 1.0
        return psf

    sigma_core_um, sigma_halo_um, sigma_bloom_um, bloom_beta, core_weight, halo_weight, bloom_weight, _ = _promist_profile_params(strength)
    sigma_core_px = max((sigma_core_um * spatial_scale) / pixel_size_um, 1e-6)
    sigma_halo_px = max((sigma_halo_um * spatial_scale) / pixel_size_um, 1e-6)
    sigma_bloom_px = max((sigma_bloom_um * spatial_scale) / pixel_size_um, 1e-6)

    y, x = np.ogrid[:kernel_shape[0], :kernel_shape[1]]
    center_y, center_x = kernel_shape[0] // 2, kernel_shape[1] // 2
    r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

    core_component = np.exp(-0.5 * (r / sigma_core_px) ** 2)
    halo_component = np.exp(-0.5 * (r / sigma_halo_px) ** 2)
    bloom_component = (1.0 + (r / sigma_bloom_px) ** 2) ** (-bloom_beta)
    psf = core_weight * core_component + halo_weight * halo_component + bloom_weight * bloom_component
    psf /= np.sum(psf)
    return psf


def apply_promist_filter(data, strength=0.25, pixel_size_um=1.0, spatial_scale=1.0, intensity=1.0):
    """Apply a Black Pro-Mist-like scattering model in linear space.

    Parameters are defined on the image plane in micrometers. `strength` is a
    float in filter-stop units such as `0.125`, `0.25`, `0.5`, or `1.0`.
    `spatial_scale=1.0` rescales the image-plane PSF footprint, and
    `intensity=1.0` keeps the reference scattered-light fraction.
    """
    spatial_scale = max(float(spatial_scale), 0.0)
    intensity = max(float(intensity), 0.0)
    intensity *= 0.5 # empirically determined scaling to keep the effect subtle at default parameters
    if strength <= 0 or spatial_scale <= 0 or intensity <= 0:
        return data

    sigma_core_um, sigma_halo_um, sigma_bloom_um, _, _, _, _, scatter_strength = _promist_profile_params(strength)
    sigma_core_px = (sigma_core_um * spatial_scale) / pixel_size_um
    sigma_halo_px = (sigma_halo_um * spatial_scale) / pixel_size_um
    sigma_bloom_px = (sigma_bloom_um * spatial_scale) / pixel_size_um
    radius = max(int(np.ceil(max(4 * sigma_core_px, 5 * sigma_halo_px, 8 * sigma_bloom_px))), 1)
    psf_shape = (2 * radius + 1, 2 * radius + 1)
    psf = promist_psf(
        psf_shape,
        strength=strength,
        pixel_size_um=pixel_size_um,
        spatial_scale=spatial_scale,
    )

    padded = np.pad(data, ((radius, radius), (radius, radius), (0, 0)), mode='reflect')
    blurred = np.empty_like(padded)
    for channel in range(data.shape[2]):
        blurred[:, :, channel] = fftconvolve(padded[:, :, channel], psf, mode='same')
    blurred = blurred[radius:-radius, radius:-radius, :]
    scatter_fraction = np.clip(scatter_strength * intensity, 0.0, 1.0)
    return (1 - scatter_fraction) * data + scatter_fraction * blurred


def _demo_promist_diffusion():
    import matplotlib.pyplot as plt

    strength_samples = np.geomspace(0.125, 1.0, 7)
    strength_grid = np.geomspace(0.125, 1.0, 121)
    _, _, max_bloom_scale_um, _, _, _, _, _ = _promist_profile_params(strength_grid[-1])
    radius_max_um = max_bloom_scale_um * 6.0
    radius_um = np.linspace(0.0, radius_max_um, 1600)
    radius_mm = radius_um / 1000.0

    strength_profiles = np.stack([
        _promist_scatter_profile(radius_um, strength=value)['total']
        for value in strength_grid
    ], axis=0)

    log_strength_profiles = np.log10(np.clip(strength_profiles, 1e-14, None))
    heatmap_vmax = np.max(log_strength_profiles)
    heatmap_vmin = heatmap_vmax - 7.0

    strength_r50 = np.empty_like(strength_grid)
    strength_r90 = np.empty_like(strength_grid)
    strength_r99 = np.empty_like(strength_grid)
    for index, value in enumerate(strength_grid):
        profile = _promist_scatter_profile(radius_um, strength=value)
        strength_r50[index] = _promist_characteristic_radius(radius_um, profile['cumulative'], 0.50)
        strength_r90[index] = _promist_characteristic_radius(radius_um, profile['cumulative'], 0.90)
        strength_r99[index] = _promist_characteristic_radius(radius_um, profile['cumulative'], 0.99)

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.8), constrained_layout=True)

    strength_profile_axis, strength_heatmap_axis, strength_metric_axis = axes

    for value in strength_samples:
        profile = _promist_scatter_profile(radius_um, strength=value)
        strength_profile_axis.plot(radius_mm, np.clip(profile['total'], 1e-14, None), linewidth=1.5, label=f'{value:g}')
    strength_profile_axis.set_title('Strength Sweep: Analytical Scatter PSF')
    strength_profile_axis.set_xlabel('radius on image plane (mm)')
    strength_profile_axis.set_ylabel('normalized PSF density')
    strength_profile_axis.set_yscale('log')
    strength_profile_axis.set_ylim(10 ** heatmap_vmin, 10 ** (heatmap_vmax + 0.2))
    strength_profile_axis.grid(alpha=0.2)
    strength_profile_axis.legend(title='strength', fontsize=8)

    strength_mesh = strength_heatmap_axis.pcolormesh(
        radius_mm,
        strength_grid,
        log_strength_profiles,
        shading='auto',
        cmap='magma',
        vmin=heatmap_vmin,
        vmax=heatmap_vmax,
    )
    strength_heatmap_axis.set_title('Strength Morphing Heatmap')
    strength_heatmap_axis.set_xlabel('radius on image plane (mm)')
    strength_heatmap_axis.set_ylabel('strength')
    strength_heatmap_axis.set_yscale('log')

    strength_metric_axis.plot(strength_grid, strength_r50 / 1000.0, linewidth=1.8, label='r50')
    strength_metric_axis.plot(strength_grid, strength_r90 / 1000.0, linewidth=1.8, label='r90')
    strength_metric_axis.plot(strength_grid, strength_r99 / 1000.0, linewidth=1.8, label='r99')
    strength_metric_axis.set_title('Strength Sweep: Encircled-Energy Radii')
    strength_metric_axis.set_xlabel('strength')
    strength_metric_axis.set_ylabel('radius (mm)')
    strength_metric_axis.set_xscale('log')
    strength_metric_axis.grid(alpha=0.2)
    strength_metric_axis.legend(fontsize=8)
    fig.colorbar(strength_mesh, ax=strength_heatmap_axis, label='log10(normalized PSF density)')
    fig.suptitle('Pro-Mist filters scatter PSF')
    backend_name = plt.get_backend().lower()
    from matplotlib import backends as mpl_backends

    backend_filter = getattr(mpl_backends, 'BackendFilter', None)
    backend_registry = getattr(mpl_backends, 'backend_registry', None)
    if backend_filter is not None and backend_registry is not None:
        interactive_backends = {name.lower() for name in backend_registry.list_builtin(backend_filter.INTERACTIVE)}
        is_interactive_backend = backend_name in interactive_backends
    else:
        is_interactive_backend = backend_name not in {
            'agg',
            'cairo',
            'pdf',
            'pgf',
            'ps',
            'svg',
            'template',
            'module://matplotlib_inline.backend_inline',
        }
    if is_interactive_backend:
        plt.show()
    return fig


if __name__ == "__main__":
    _demo_promist_diffusion()

