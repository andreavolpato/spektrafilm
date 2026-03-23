import numpy as np
from spektrafilm.utils.spectral_upsampling import compute_lut_spectra
from pathlib import Path

# make lut with spectra covering the full xy triangle
lut_spectra = compute_lut_spectra(lut_size=192,
                                  lut_coeffs_filename='hanatos_irradiance_xy_coeffs_250304.lut')
output = Path(__file__).resolve().parents[1] / 'src' / 'spektrafilm' / 'data' / 'luts' / 'spectral_upsampling' / 'irradiance_xy_tc.npy'
np.save(output, lut_spectra)
