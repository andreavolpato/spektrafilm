"""
Example: Create and visualize film profiles.

This script demonstrates how to create a profile from raw data
and how to load and plot existing profiles.
"""

import matplotlib.pyplot as plt
from profiles_factory.factory import create_profile, plot_profile, remove_density_min
from spectral_film_lab.profile_store.io import load_profile

p = create_profile('kodak_vision3_50d')
p = remove_density_min(p)
plot_profile(p)
plt.show()

p = load_profile('kodak_portra_400_auc')
# p = load_profile('fujifilm_pro_400h_auc')
# p = load_profile('kodak_portra_endura')
plot_profile(p)
plt.show()
