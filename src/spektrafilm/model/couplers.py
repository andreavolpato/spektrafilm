import numpy as np
from opt_einsum import contract

from spektrafilm.runtime.params_schema import DirCouplersParams
from spektrafilm.utils.fast_gaussian_filter import fast_gaussian_filter
from spektrafilm.model.density_curves import interpolate_exposure_to_density

def compute_density_curves_before_dir_couplers(density_curves, log_exposure, dir_couplers_matrix, positive=False):
    """
    DIR couplers affect the same layer by increasing contrast.
    I suppose that in the design of a film this is taken into account, and the final film has well behaved density curves.
    In order to get final curves for gray ramps equal to the input data, the density curves before the effect of the couplers are needed.

    Args:
        density_curves (numpy.array): Characteristic density curves of the film after the application of DIR couplers
        log_exposure (numpy.array): The image as log_exposure
        dir_couplers_matrix (_type_): DIR couplers matrix computed with compute_dir_couplers_matrix()

    Returns:
        numpy.array: Corrected density curves before the effect of DIR couplers
    """
    
    if positive:
        # We are asusming that interimage effects in positve film are acting in the silver development stage
        # We are also assuming that silver density is d_max - d
        density_curves_silver = np.nanmax(density_curves, axis=0) - density_curves
    else:
        density_curves_silver = np.copy(density_curves)
    
    couplers_amount_curves = contract('jk, km->jm', density_curves_silver, dir_couplers_matrix)
    log_exposure_0 = log_exposure[:,None] - couplers_amount_curves
    density_curves_corrected = np.zeros_like(density_curves)
    for i in np.arange(3):
        if positive:
            density_curves_corrected[:,i] = -np.interp(log_exposure, log_exposure_0[:,i], -density_curves[:,i])
        else:
            density_curves_corrected[:,i] = np.interp(log_exposure, log_exposure_0[:,i], density_curves[:,i])
    return density_curves_corrected


def compute_dir_couplers_matrix(couplers_params: DirCouplersParams = DirCouplersParams()):
    """
    Compute the inhibitors matrix using a simple diffusion model across layers.

    Parameters:
    amount_rgb (list of float): Amounts of dir couplers for RGB channels. Default is [0.7,0.7,0.5]. Typically 0-1 range.
    layer_diffusion (float): Sigma for gaussian diffusion distance of dir couplers. Default is 1.

    Returns:
    numpy.ndarray: The computed inhibitors matrix.
    Row index is the donor/source layer that releases inhibitor.
    Column index is the receiving/affected layer whose exposure is reduced.
    """
    
    M_self = np.array(couplers_params.gamma_samelayer_rgb)*couplers_params.inhibition_samelayer
    M_self = np.diag(M_self)
    M_inter = np.zeros((3,3))
    # Off-diagonal terms follow the same convention: donor row, receiver column.
    M_inter[0,1] = couplers_params.gamma_interlayer_r_to_gb[0]
    M_inter[0,2] = couplers_params.gamma_interlayer_r_to_gb[1]
    M_inter[1,0] = couplers_params.gamma_interlayer_g_to_rb[0]
    M_inter[1,2] = couplers_params.gamma_interlayer_g_to_rb[1]
    M_inter[2,0] = couplers_params.gamma_interlayer_b_to_rg[0]
    M_inter[2,1] = couplers_params.gamma_interlayer_b_to_rg[1]
    M_inter *= couplers_params.inhibition_interlayer
    return M_self + M_inter

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
    # density_silver[..., k] generated in donor layer k contributes to receiver m
    # through dir_couplers_matrix[k, m].
    log_raw_correction = contract('ijk, km->ijm', density_silver, dir_couplers_matrix)
    if diffusion_size_pixel>0:
        # log_raw_correction = gaussian_filter(log_raw_correction, (diffusion_size_pixel, diffusion_size_pixel, 0))
        log_raw_correction = fast_gaussian_filter(log_raw_correction, diffusion_size_pixel)
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
    
    couplers_matrix = compute_dir_couplers_matrix(dir_couplers)
    couplers_matrix *= dir_couplers.inhibition_strength
    
    density_curves_0 = compute_density_curves_before_dir_couplers(
        density_curves,
        log_exposure,
        couplers_matrix,
        positive=positive,
    )
    density_max = np.nanmax(density_curves, axis=0)
    diffusion_size_pixel = dir_couplers.diffusion_size_um / pixel_size_um
    log_raw_0 = compute_exposure_correction_dir_couplers(
        log_raw,
        density_cmy,
        density_max,
        couplers_matrix,
        diffusion_size_pixel,
        positive=positive,
    )
    return interpolate_exposure_to_density(log_raw_0, density_curves_0, log_exposure, gamma_factor)

if __name__=='__main__':
    # # Test the raw correction coupler inhibitors
    # log_raw = np.ones((4,4,3))
    # density_cmy = np.ones((4,4,3))
    # density_max = 2.2
    # couplers_amount = [0.9,0.7,0.5]
    # diffusion_size_pixel = 2
    # log_raw = raw_correction_dir_couplers(log_raw, density_cmy, density_max, couplers_amount, diffusion_size_pixel)
    # print(log_raw)

    couplers_params = DirCouplersParams()
    M = compute_dir_couplers_matrix(couplers_params)
    print(M)