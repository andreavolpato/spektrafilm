import numpy as np

from spektrafilm import Simulator, create_params, simulate
from spektrafilm.runtime.process import AgXPhoto, photo_params, photo_process


class TestRuntimeApi:
    def test_create_params_uses_runtime_defaults(self):
        params = create_params()

        assert params.film.info.stock == 'kodak_portra_400_auc'
        assert params.print.info.stock == 'kodak_portra_endura_uc'

    def test_simulator_wraps_pipeline_sections(self, default_params):
        simulator = Simulator(default_params)

        assert simulator.camera is not None
        assert simulator.film is not None
        assert simulator.print is not None

    def test_simulate_matches_legacy_process(self, small_rgb_image, default_params):
        new_result = simulate(small_rgb_image, default_params)
        legacy_result = photo_process(small_rgb_image, default_params)

        np.testing.assert_allclose(new_result, legacy_result, atol=1e-12)

    def test_legacy_aliases_remain_available(self):
        legacy_params = photo_params()
        legacy_simulator = AgXPhoto(legacy_params)

        assert legacy_params.film.info.stock == 'kodak_portra_400_auc'
        assert isinstance(legacy_simulator, Simulator)