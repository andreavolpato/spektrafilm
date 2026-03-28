from __future__ import annotations

import numpy as np

from spektrafilm.profiles.io import Profile, ProfileData, ProfileInfo

from tests.conftest import make_fast_test_params


def make_test_runtime_params(print_profile: str):
    return make_fast_test_params(print_profile=print_profile)


def make_test_profile(stock: str = 'diagnostic_test_stock') -> Profile:
    return Profile(
        info=ProfileInfo(stock=stock, name=stock),
        data=ProfileData(
            wavelengths=np.array([450.0, 550.0]),
            log_sensitivity=np.zeros((2, 3), dtype=float),
            channel_density=np.zeros((2, 3), dtype=float),
            base_density=np.zeros((2,), dtype=float),
            midscale_neutral_density=np.zeros((2,), dtype=float),
            log_exposure=np.array([-1.0, 1.0], dtype=float),
            density_curves=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=float),
            density_curves_layers=np.zeros((2, 3, 3), dtype=float),
        ),
    )