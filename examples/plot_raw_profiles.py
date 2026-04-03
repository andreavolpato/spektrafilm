"""
Example: Plot raw film profiles.

This script demonstrates how to plot a profile from raw data
"""

import matplotlib.pyplot as plt
from spektrafilm_profile_creator.plotting import plot_profile
from spektrafilm_profile_creator import load_raw_profile

# p = load_raw_profile('kodak_vision3_50d')
p = load_raw_profile('kodak_portra_400')
# p = load_raw_profile('kodak_gold_200')
# p = load_raw_profile('fujifilm_pro_400h')
# p = load_raw_profile('kodak_portra_endura')
plot_profile(p)
plt.show()
