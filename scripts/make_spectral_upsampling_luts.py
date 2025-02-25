import numpy as np
from agx_emulsion.utils.spectral_upsampling import compute_lut

# make lut with spectra from rec2020 with 32x32x32 grid
lut_spectra = compute_lut(lut_size=32, color_space='ITU-R BT.2020')
np.save('agx_emulsion/data/luts/spectral_upsampling/irradiance_rec2020_32size.npy', lut_spectra)

# # make lut with spectra from aces2065-1 with 32x32x32 grid
# lut_spectra = compute_lut(lut_size=32, color_space='ACES2065-1')
# np.save('agx_emulsion/data/luts/spectral_upsampling/irradiance_aces2065-1_32size.npy', lut_spectra)

# # make lut with spectra from aces2065-1 with 32x32x32 grid
# lut_spectra = compute_lut(lut_size=32, color_space='ProPhoto RGB')
# np.save('agx_emulsion/data/luts/spectral_upsampling/irradiance_prophoto_rgb_32size.npy', lut_spectra)
