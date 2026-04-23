from __future__ import annotations

import numpy as np

from spektrafilm_profile_creator.core.balancing import prelminary_neutral_shift
from spektrafilm_profile_creator.core.densitometer import (
    _fill_missing_log_sensitivity_columns,
    fill_missing_sensitivity,
    unmix_density,
    unmix_sensitivity,
)
from spektrafilm_profile_creator.core.profile_transforms import remove_density_min
from spektrafilm_profile_creator.data.loader import load_raw_profile


def _prepare_profile(stock: str):
    raw_profile = load_raw_profile(stock)
    profile = raw_profile.as_profile()
    profile = remove_density_min(profile)
    profile = prelminary_neutral_shift(profile)
    profile = unmix_density(profile)
    return profile


def test_unmix_sensitivity_keeps_missing_entries_empty():
    profile = _prepare_profile('kodak_portra_400')
    published_valid = np.isfinite(np.asarray(profile.data.log_sensitivity, dtype=float))

    profile = unmix_sensitivity(profile)
    fitted = np.asarray(profile.data.log_sensitivity, dtype=float)

    assert np.isfinite(fitted[published_valid]).all()
    assert np.isnan(fitted[~published_valid]).all()


def test_fill_missing_sensitivity_simple_fills_without_touching_observed_entries():
    profile = _prepare_profile('kodak_portra_endura')
    profile = unmix_sensitivity(profile)

    sparse = np.asarray(profile.data.log_sensitivity, dtype=float)
    filled_profile = fill_missing_sensitivity(profile)
    filled = np.asarray(filled_profile.data.log_sensitivity, dtype=float)

    observed = np.isfinite(sparse)
    assert np.isfinite(filled).all()
    np.testing.assert_allclose(filled[observed], sparse[observed])


def test_fill_missing_sensitivity_simple_produces_monotone_decaying_tails():
    profile = _prepare_profile('kodak_portra_400')
    profile = unmix_sensitivity(profile)

    sparse = np.asarray(profile.data.log_sensitivity, dtype=float)
    filled = np.asarray(fill_missing_sensitivity(profile).data.log_sensitivity, dtype=float)

    for channel in range(sparse.shape[1]):
        valid_indices = np.flatnonzero(np.isfinite(sparse[:, channel]))
        left_index = valid_indices[0]
        right_index = valid_indices[-1]

        if left_index > 0:
            left_tail = filled[: left_index + 1, channel]
            assert np.all(left_tail <= filled[left_index, channel] + 1e-8)
            assert np.all(np.diff(left_tail) >= -1e-8)

        if right_index < filled.shape[0] - 1:
            right_tail = filled[right_index:, channel]
            assert np.all(right_tail <= filled[right_index, channel] + 1e-8)
            assert np.all(np.diff(right_tail) <= 1e-8)


def test_fill_missing_sensitivity_simple_uses_more_than_single_flat_edge_interval():
    wavelengths = np.array([400, 450, 500, 550, 600, 650], dtype=float)
    log_sensitivity = np.array([
        [np.nan],
        [np.nan],
        [0.0],
        [-0.01],
        [-0.8],
        [np.nan],
    ], dtype=float)

    filled = _fill_missing_log_sensitivity_columns(log_sensitivity, wavelengths)

    assert filled[0, 0] < -0.1
    assert filled[1, 0] < -0.05
    assert np.all(np.diff(filled[:3, 0]) >= -1e-8)