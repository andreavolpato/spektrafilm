import numpy as np
import pytest

from spectral_film_lab.runtime.process import photo_params, photo_process

from tests.profiles_creator.create_profile_regression_baselines import (
    assert_matches_baseline,
    compute_processed_profile,
    find_case,
    load_baseline,
)


def make_runtime_params(print_profile: str):
    params = photo_params(print_profile=print_profile)
    params.debug.deactivate_spatial_effects = True
    params.debug.deactivate_stochastic_effects = True
    params.settings.use_enlarger_lut = False
    params.settings.use_scanner_lut = False
    params.io.preview_resize_factor = 1.0
    params.io.upscale_factor = 1.0
    params.io.crop = False
    params.io.full_image = False
    params.camera.auto_exposure = False
    params.camera.exposure_compensation_ev = 0.0
    return params


@pytest.fixture(scope='module')
def portra_400_processed_profile():
    case = find_case('create_profile_kodak_portra_400')
    return case, compute_processed_profile(case)


@pytest.fixture(scope='module')
def portra_endura_paper_processed_profile():
    case = find_case('create_profile_kodak_portra_endura_paper')
    return case, compute_processed_profile(case)


class TestCreateProfile:
    def test_processed_profile_matches_regression_baseline(self, portra_400_processed_profile):
        case, profile = portra_400_processed_profile
        expected = load_baseline(case.case_id)

        assert profile.info.stock == case.stock
        assert profile.info.type == case.type
        assert profile.info.support == case.support
        assert profile.info.channel_model == 'color'
        assert profile.info.densitometer == case.densitometer
        assert profile.info.reference_illuminant == case.reference_illuminant
        assert profile.info.viewing_illuminant == 'D50'

        assert_matches_baseline(case.case_id, profile, expected)

    def test_generated_processed_profile_runs_in_runtime_pipeline(self, portra_400_processed_profile):
        case, profile = portra_400_processed_profile
        params = make_runtime_params(case.runtime_print_paper)
        params.film = profile
        image = np.ones((8, 8, 3), dtype=np.float64) * 0.184

        output = np.asarray(photo_process(image, params), dtype=np.float64)

        assert output.shape == (8, 8, 3)
        assert np.isfinite(output).all()
        assert float(np.min(output)) >= 0.0
        assert float(np.max(output)) <= 1.0

    def test_processed_paper_profile_matches_regression_baseline(self, portra_endura_paper_processed_profile):
        case, profile = portra_endura_paper_processed_profile
        expected = load_baseline(case.case_id)

        assert profile.info.stock == case.stock
        assert profile.info.type == case.type
        assert profile.info.support == case.support
        assert profile.info.channel_model == 'color'
        assert profile.info.densitometer == case.densitometer
        assert profile.info.reference_illuminant == case.reference_illuminant
        assert profile.info.viewing_illuminant == case.viewing_illuminant

        assert_matches_baseline(case.case_id, profile, expected)

    def test_generated_processed_paper_profile_runs_in_runtime_pipeline(self, portra_endura_paper_processed_profile):
        case, profile = portra_endura_paper_processed_profile
        params = make_runtime_params(case.runtime_print_paper)
        params.print = profile
        image = np.ones((8, 8, 3), dtype=np.float64) * 0.184

        output = np.asarray(photo_process(image, params), dtype=np.float64)

        assert output.shape == (8, 8, 3)
        assert np.isfinite(output).all()
        assert float(np.min(output)) >= 0.0
        assert float(np.max(output)) <= 1.0