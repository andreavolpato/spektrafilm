"""
Example: Create and visualize film profiles.

This script demonstrates how to create a profile from raw data
and how to load and plot existing profiles.
"""

import matplotlib.pyplot as plt
from spektrafilm_profile_creator.core.profile_transforms import remove_density_min
from spektrafilm_profile_creator.plotting import plot_profile
from spektrafilm_profile_creator import load_raw_profile
from spektrafilm.profiles.io import load_profile

p = load_raw_profile('kodak_vision3_50d')
p = remove_density_min(p)
plot_profile(p)
plt.show()

p = load_profile('kodak_portra_400')
# p = load_profile('fujifilm_pro_400h')
# p = load_profile('kodak_portra_endura')
plot_profile(p)
plt.show()
