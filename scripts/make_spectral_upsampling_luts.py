import numpy as np
from agx_emulsion.utils.spectral_upsampling import compute_lut

# make lut with spectra from rec2020 with 32x32x32 grid
lut_spectra = compute_lut(lut_size=32, color_space='ITU-R BT.2020')
np.save('agx_emulsion/data/luts/spectral_upsampling/irradiance_rec2020_32size.npy', lut_spectra)