"""
Example: Check midgray balance of a film profile.

This script loads a profile and computes the integrated sensitivity
under a D55 illuminant to check the balance of the film layers.
"""

import numpy as np
from spektrafilm.profiles.io import load_profile
from spektrafilm.model.illuminants import standard_illuminant

p = load_profile('kodak_portra_400_au')
ill = standard_illuminant(type='D55')
s = 10**np.double(p.data.log_sensitivity)
print(np.nansum(ill[:, None] * s, axis=0))