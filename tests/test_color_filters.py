from __future__ import annotations

import numpy as np

from spektrafilm.config import SPECTRAL_SHAPE
from spektrafilm.model import color_filters as cf


def test_library_keys_are_unique_and_loadable() -> None:
    keys = [spec.key for spec in cf.COLOR_FILTER_LIBRARY]
    assert len(keys) == len(set(keys)), "duplicate color filter keys"
    for spec in cf.COLOR_FILTER_LIBRARY:
        transmittance = cf.get_color_filter(spec.key).transmittance
        assert transmittance.shape == SPECTRAL_SHAPE.wavelengths.shape
        assert not np.isnan(transmittance).any()
        assert np.all(transmittance >= 0.0)


def test_none_returns_no_transmittance() -> None:
    assert cf.color_filter_transmittance(cf.NO_COLOR_FILTER) is None


def test_selector_enum_matches_library_with_none_first() -> None:
    members = [member.value for member in cf.CameraColorFilters]
    assert members[0] == cf.NO_COLOR_FILTER
    assert members[1:] == [spec.key for spec in cf.COLOR_FILTER_LIBRARY]


def test_transmittance_matches_loaded_filter() -> None:
    spec = cf.COLOR_FILTER_LIBRARY[0]
    np.testing.assert_array_equal(
        cf.color_filter_transmittance(spec.key),
        cf.get_color_filter(spec.key).transmittance,
    )
