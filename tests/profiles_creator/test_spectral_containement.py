from __future__ import annotations

import numpy as np

from spektrafilm.profiles.io import load_profile
from spektrafilm.utils.spectral_upsampling import rgb_to_smooth_spectrum
from spektrafilm_profile_creator.spectral_containment import sensitivity_bandpass_hanatos2025


def test_sensitivity_bandpass_hanatos2025_stores_per_channel_normalized_bandpass() -> None:
    profile = load_profile('kodak_portra_400')

    updated = sensitivity_bandpass_hanatos2025(profile.clone())

    assert updated.data.bandpass_hanatos2025.shape == updated.data.log_sensitivity.shape
    assert np.all(updated.data.bandpass_hanatos2025 >= 0.0)

    midgray = np.array([[[0.184, 0.184, 0.184]]], dtype=float)
    illuminant = rgb_to_smooth_spectrum(
        midgray,
        color_space='ProPhoto RGB',
        apply_cctf_decoding=False,
        reference_illuminant=updated.info.reference_illuminant,
    )
    sensitivity = 10 ** np.asarray(updated.data.log_sensitivity, dtype=float)
    normalized_exposure = np.nansum(
        updated.data.bandpass_hanatos2025 * sensitivity * illuminant[:, None],
        axis=0,
    )

    np.testing.assert_allclose(normalized_exposure, np.ones(3), rtol=1e-6, atol=1e-6)