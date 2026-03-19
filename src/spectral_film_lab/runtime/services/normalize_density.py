from __future__ import annotations

import numpy as np


class FilmDensityNormalizer:
    def __init__(self, source_profile, density_min):
        self._source_profile = source_profile
        self._density_min = np.asarray(density_min)

    def normalize(self, density_channels: np.ndarray) -> np.ndarray:
        density_max = np.nanmax(self._source_profile.data.density_curves, axis=0)
        density_max = density_max + self._density_min
        return (density_channels + self._density_min) / density_max

    def denormalize(self, density_channels_normalized: np.ndarray) -> np.ndarray:
        density_max = np.nanmax(self._source_profile.data.density_curves, axis=0)
        density_max = density_max + self._density_min
        return density_channels_normalized * density_max - self._density_min


class PrintDensityNormalizer:
    def __init__(self, print_profile):
        self._print_profile = print_profile

    def normalize(self, density_channels: np.ndarray) -> np.ndarray:
        density_max = np.nanmax(self._print_profile.data.density_curves, axis=0)
        return density_channels / density_max

    def denormalize(self, density_channels_normalized: np.ndarray) -> np.ndarray:
        density_max = np.nanmax(self._print_profile.data.density_curves, axis=0)
        return density_channels_normalized * density_max
