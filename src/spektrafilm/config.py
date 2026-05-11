import numpy as np
import colour

# Constants
LOG_EXPOSURE = np.linspace(-3,4,256)
SPECTRAL_SHAPE = colour.SpectralShape(380, 780, 5)

# Default color matching functions
STANDARD_OBSERVER_CMFS = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"].copy().align(SPECTRAL_SHAPE)

# Cone fundamentals paired to the same 2-degree observer regime used by the
# study-side XYZ geometry and Hanatos upsampling pipeline.
STANDARD_OBSERVER_LMS = colour.colorimetry.MSDS_CMFS_LMS[
	"Stockman & Sharpe 2 Degree Cone Fundamentals"
].copy().align(SPECTRAL_SHAPE)