import numpy as np
from agx_emulsion.utils.spectral_upsampling import compute_lut_spectra

# make lut with spectra covering the full xy triangle
lut_spectra = compute_lut_spectra(lut_size=192, lut_coeffs_filename='hanatos_irradiance_xy_coeffs_250304.lut')
np.save('agx_emulsion/data/luts/spectral_upsampling/irradiance_xy_tc.npy', lut_spectra)