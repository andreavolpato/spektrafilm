import numpy as np
from scipy.ndimage import gaussian_filter
# from spectral_film_lab.utils.fast_gaussian_filter import fast_gaussian_filter
from opt_einsum import contract
from spektrafilm.model.density_curves import interpolate_exposure_to_density

def compute_density_curves_before_dir_couplers(density_curves, log_exposure, dir_couplers_matrix, high_exposure_couplers_shift=0.0, positive=False):
    """
    DIR couplers affect the same layer by increasing contrast.
    I suppose that in the design of a film this is taken into account, and the final film has well behaved density curves.
    In order to get final curves for gray ramps equal to the input data, the density curves before the effect of the couplers are needed.

    Args:
        density_curves (numpy.array): Characteristic density curves of the film after the application of DIR couplers
        log_exposure (numpy.array): The image as log_exposure
        dir_couplers_matrix (_type_): DIR couplers matrix computed with compute_dir_couplers_matrix()
        high_exposure_couplers_shift (float, optional): Related to increased inhibition at high exposures, defaults to 0.0.

    Returns:
        _type_: _description_
    """
    
    if positive:
        # We are asusming that interimage effects in positve film are acting in the silver development stage
        # We are also assuming that silver density is d_max - d
        density_curves_silver = np.nanmax(density_curves, axis=0) - density_curves
    else:
        density_curves_silver = np.copy(density_curves)
    
    dc_norm_shift = density_curves_silver + high_exposure_couplers_shift*density_curves_silver**2
    couplers_amount_curves = contract('jk, km->jm', dc_norm_shift, dir_couplers_matrix)
    log_exposure_0 = log_exposure[:,None] - couplers_amount_curves
    density_curves_corrected = np.zeros_like(density_curves)
    for i in np.arange(3):
        if positive:
            density_curves_corrected[:,i] = -np.interp(log_exposure, log_exposure_0[:,i], -density_curves[:,i])
        else:
            density_curves_corrected[:,i] = np.interp(log_exposure, log_exposure_0[:,i], density_curves[:,i])
    return density_curves_corrected


def compute_dir_couplers_matrix(amount_rgb=[0.7,0.7,0.5], layer_diffusion=1):
    """
    Compute the inhibitors matrix using a simple diffusion model across layers.

    Parameters:
    amount_rgb (list of float): Amounts of dir couplers for RGB channels. Default is [0.7,0.7,0.5]. Typically 0-1 range.
    layer_diffusion (float): Sigma for gaussian diffusion distance of dir couplers. Default is 1.

    Returns:
    numpy.ndarray: The computed inhibitors matrix. Fisrt index is the input layer, second index is the output layer.
    """
    M = np.eye(3)
    M_diffused = gaussian_filter(M, layer_diffusion, mode='constant', cval=0, axes=1)
    M_diffused /= np.sum(M_diffused, axis=1)[:, None]
    M = M_diffused *np.array(amount_rgb)[:, None]
    return M

def compute_exposure_correction_dir_couplers(log_raw, density_cmy, density_max,
                                             dir_couplers_matrix, diffusion_size_pixel,
                                             high_exposure_couplers_shift=0.0,
                                             positive=False):
    """
    Apply coupler inhibitors to the raw data based on density curves and inhibitor values.
    Coupler inhibitors are released when density is formed in the emulsion layers.
    If a layer is dense, the inhibitors are released to prevent further density formation in neighboring layers.
    Also self-inhibitors in the same layer, after spatial diffusion, can prevent further density formation
    in nearby areas, adding a local contrast effect.

    Parameters:
    raw (numpy.ndarray): The raw data to which inhibitors will be applied.
    density_cmy (numpy.ndarray): The density values for each layer.
    density_max (float): The maximum density value achievable for each layer, used for normalization.
    dir_couplers_matrix (numpy.ndarray): The inhibitors matrix. Fisrt index is the input layer, second index is the output layer.
    diffusion_size_pixel (int): The size of the gaussian filter for the diffusion of the inhibitors in xy.
    high_exposure_couplers_shift (float): if overexposure increases saturation, this will increase the inhibitors effect at higher density
    
    Returns:
    numpy.ndarray: The modified raw exposure data after applying the effect of inhibitors.
    """
    if positive:
        density_silver = density_max - density_cmy
    else:
        density_silver = np.copy(density_cmy)
    density_silver += high_exposure_couplers_shift*density_silver**2
    log_raw_correction = contract('ijk, km->ijm', density_silver, dir_couplers_matrix)
    if diffusion_size_pixel>0:
        log_raw_correction = gaussian_filter(log_raw_correction, (diffusion_size_pixel, diffusion_size_pixel, 0))
        # log_raw_correction = fast_gaussian_filter(log_raw_correction, diffusion_size_pixel)
    log_raw_corrected = log_raw - log_raw_correction
    return log_raw_corrected


def apply_density_correction_dir_couplers(
    density_cmy,
    log_raw,
    pixel_size_um,
    log_exposure,
    density_curves,
    dir_couplers,
    profile_type,
    gamma_factor=1.0,
):
    if not dir_couplers.active:
        return density_cmy

    positive = profile_type == 'positive'
    dir_couplers_amount_rgb = dir_couplers.amount * np.array(dir_couplers.ratio_rgb)
    couplers_matrix = compute_dir_couplers_matrix(
        dir_couplers_amount_rgb,
        dir_couplers.diffusion_interlayer,
    )
    density_curves_0 = compute_density_curves_before_dir_couplers(
        density_curves,
        log_exposure,
        couplers_matrix,
        dir_couplers.high_exposure_shift,
        positive=positive,
    )
    density_max = np.nanmax(density_curves, axis=0)
    diffusion_size_pixel = dir_couplers.diffusion_size_um / pixel_size_um

    # Fixed-point iteration for spatial coupling (see notes/dir_couplers_model.md):
    #     D_{k+1} = D_0(log E - G_xy * (M @ D_k))
    # One pass is exact when there is no spatial blur; with blur, a few iterations
    # resolve the edge under-correction from using a stale density on the RHS.
    n_iter = 3 if getattr(dir_couplers, 'iterate', True) else 1
    density = density_cmy
    for _ in range(n_iter):
        log_raw_0 = compute_exposure_correction_dir_couplers(
            log_raw,
            density,
            density_max,
            couplers_matrix,
            diffusion_size_pixel,
            high_exposure_couplers_shift=dir_couplers.high_exposure_shift,
            positive=positive,
        )
        density = interpolate_exposure_to_density(
            log_raw_0, density_curves_0, log_exposure, gamma_factor
        )
    return density

if __name__ == '__main__':
    # Compare 1-iteration vs 3-iteration spatial DIR coupling on a sharp edge.
    # Plots a 1D cross-section of the edge for the G channel.
    import matplotlib.pyplot as plt

    # Synthetic pre-coupler H&D curve (shared across RGB), gamma_0 = 1.2.
    log_exposure = np.linspace(-3.0, 2.0, 512)
    gamma_0 = 1.2
    d_max = 2.2
    base_curve = np.clip(gamma_0 * (log_exposure + 1.5), 0.0, d_max)
    density_curves_0 = np.tile(base_curve[:, None], (1, 3))
    density_max = np.nanmax(density_curves_0, axis=0)

    # 1D step edge in log exposure (a few rows so the 2D Gaussian behaves normally).
    H, N = 16, 256
    log_raw = np.full((H, N, 3), -0.8)
    log_raw[:, N // 2 :, :] = 0.6

    amount_rgb = np.array([0.42, 0.42, 0.42])
    couplers_matrix = compute_dir_couplers_matrix(amount_rgb, layer_diffusion=2)
    diffusion_size_pixel = 6.0

    # Warm-start: converge the 0-D (no spatial blur) fixed point to tolerance
    # so that flat regions already satisfy D = D_0(log E - M @ D). This matches
    # the real pipeline, where the incoming density has already been produced
    # by the full post-coupler H&D lookup and is therefore 0-D-consistent by
    # construction. The contraction rate here is ~gamma_0 * amount, so a fixed
    # iteration budget would leak a flat-field residual into the 1-vs-3 plot
    # once gamma_0 * amount approaches 1; iterating to tolerance keeps it
    # bounded regardless of the film stock parameters.
    density_warm = interpolate_exposure_to_density(
        log_raw, density_curves_0, log_exposure, 1.0
    )
    for _ in range(30):
        log_raw_0 = compute_exposure_correction_dir_couplers(
            log_raw, density_warm, density_max, couplers_matrix, 0.0,
        )
        density_new = interpolate_exposure_to_density(
            log_raw_0, density_curves_0, log_exposure, 1.0
        )
        if np.max(np.abs(density_new - density_warm)) < 1e-6:
            density_warm = density_new
            break
        density_warm = density_new

    def iterate(n_iter):
        density = density_warm.copy()
        for _ in range(n_iter):
            log_raw_0 = compute_exposure_correction_dir_couplers(
                log_raw, density, density_max, couplers_matrix, diffusion_size_pixel,
            )
            density = interpolate_exposure_to_density(
                log_raw_0, density_curves_0, log_exposure, 1.0
            )
        return density

    d_1 = iterate(1)
    d_3 = iterate(3)
    d_ref = iterate(15)  # converged reference

    row = H // 2
    x = np.arange(N)
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    ax_top.plot(x, d_1[row, :, 1], '--', label='1 iteration')
    ax_top.plot(x, d_3[row, :, 1], '-',  label='3 iterations')
    ax_top.plot(x, d_ref[row, :, 1], ':', label='30 iterations (converged)')
    ax_top.set_ylabel('density (G channel)')
    ax_top.set_title(
        f'DIR couplers edge response — amount={amount_rgb.tolist()}, '
        f'layer_diffusion=2, sigma_xy={diffusion_size_pixel}px'
    )
    ax_top.legend()

    ax_bot.plot(x, d_1[row, :, 1] - d_ref[row, :, 1], '--', label='1 iter - converged')
    ax_bot.plot(x, d_3[row, :, 1] - d_ref[row, :, 1], '-',  label='3 iter - converged')
    ax_bot.axhline(0, color='k', lw=0.5)
    ax_bot.set_xlabel('pixel')
    ax_bot.set_ylabel('Delta density')
    ax_bot.legend()

    plt.tight_layout()
    plt.show()
