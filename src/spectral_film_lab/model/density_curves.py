import numpy as np
from spectral_film_lab.utils.fast_interp import fast_interp

################################################################################
# Denstity curves
################################################################################

def interpolate_exposure_to_density(log_exposure_rgb, density_curves, log_exposure, gamma_factor):
    """
    Interpolates the exposure values to density values using the provided density curves.
    Parameters:
    log_exposure_rgb (numpy.ndarray): A 3D array of shape (height, width, 3) representing the log10 RGB exposure values.
    density_curves (numpy.ndarray): A 2D array of shape (num_points, 3) representing the density curves for each channel.
    log_exposure (numpy.ndarray): A 1D array of logarithmic exposure values.
    gamma_factor (float): The gamma correction factor to be applied to the density characteristic curves.
    Returns:
    numpy.ndarray: A 3D array of shape (height, width, 3) representing the interpolated density values in CMY channels.
    """
    if np.size(gamma_factor)==1:
        gamma_factor = [gamma_factor, gamma_factor, gamma_factor]
    gamma_factor = np.array(gamma_factor)
    density_cmy = np.zeros((log_exposure_rgb.shape[0], log_exposure_rgb.shape[1], 3))
    # for channel in np.arange(3):
    #     sel = ~np.isnan(density_curves[:,channel])
    #     density_cmy[:,:,channel] = np.interp(log_exposure_rgb[:,:,channel],
    #                                          log_exposure[sel]/gamma_factor[channel],
    #                                          density_curves[sel,channel])
    density_cmy = fast_interp(np.ascontiguousarray(log_exposure_rgb),
                              log_exposure[:,None]/gamma_factor[None,:],
                              density_curves)
    return density_cmy
    
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


