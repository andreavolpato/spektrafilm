import numpy as np
import pytest

from spektrafilm import Simulator, simulate
from spektrafilm.runtime.process import AgXPhoto, photo_params, photo_process


pytestmark = pytest.mark.integration


class TestRuntimeApi:
    def test_simulate_matches_legacy_process(self, small_rgb_image, default_params):
        new_result = simulate(small_rgb_image, default_params)
        legacy_result = photo_process(small_rgb_image, default_params)

        np.testing.assert_allclose(new_result, legacy_result, atol=1e-12)

    def test_legacy_aliases_remain_available(self):
        legacy_params = photo_params()
        legacy_simulator = AgXPhoto(legacy_params)

        assert legacy_params.film.info.stock == 'kodak_portra_400'
        assert isinstance(legacy_simulator, Simulator)