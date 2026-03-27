import numpy as np
import pytest

from spektrafilm.profiles.io import Profile
from spektrafilm.runtime.process import photo_params, photo_process
from spektrafilm_profile_creator import (
    RawProfile,
    RawProfileRecipe,
    load_raw_profile,
    process_profile,
    process_negative_film_profile,
)

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


@pytest.fixture(scope='module', name='portra_400_processed_profile')
def _portra_400_processed_profile_fixture():
    case = find_case('create_profile_kodak_portra_400')
    return case, compute_processed_profile(case)


@pytest.fixture(scope='module', name='portra_endura_paper_processed_profile')
def _portra_endura_paper_processed_profile_fixture():
    case = find_case('create_profile_kodak_portra_endura_paper')
    return case, compute_processed_profile(case)


class TestCreateProfile:
    def test_load_raw_profile_reads_yaml_info_and_recipe(self):
        raw_profile = load_raw_profile('kodak_portra_400')
        paper_raw_profile = load_raw_profile('kodak_portra_endura')

        assert isinstance(raw_profile, RawProfile)
        assert isinstance(raw_profile.recipe, RawProfileRecipe)
        assert raw_profile.recipe.dye_density_reconstruct_model == 'dmid_dmin'
        assert raw_profile.recipe.gray_ramp_kwargs == {}
        assert raw_profile.recipe.align_midscale_exposures is False
        assert paper_raw_profile.recipe.align_midscale_exposures is False

    def test_negative_workflow_returns_profile(self):
        case = find_case('create_profile_kodak_portra_400')
        raw_profile = load_raw_profile(case.stock)

        result = process_negative_film_profile(raw_profile)

        assert isinstance(result, Profile)
        assert result.info.stock == case.stock

    def test_load_raw_profile_accepts_stock_string(self):
        case = find_case('create_profile_kodak_portra_400')

        raw_profile = load_raw_profile(case.stock)

        assert raw_profile.info.stock == case.stock
        assert raw_profile.info.support == 'film'
        assert raw_profile.info.type == 'negative'
        assert raw_profile.info.channel_model == 'color'

    def test_process_profile_dispatches_from_raw_profile(self):
        case = find_case('create_profile_kodak_portra_400')
        raw_profile = load_raw_profile(case.stock)

        result = process_profile(raw_profile)

        assert isinstance(result, Profile)
        assert result.info.stock == case.stock

    def test_process_profile_dispatches_from_stock_string(self):
        case = find_case('create_profile_kodak_portra_400')

        result = process_profile(case.stock)

        assert isinstance(result, Profile)
        assert result.info.stock == case.stock

    def test_processed_profile_matches_regression_baseline(self, portra_400_processed_profile):
        case, profile = portra_400_processed_profile
        expected = load_baseline(case.case_id)
        raw_profile = load_raw_profile(case.stock)

        assert profile.info.stock == case.stock
        assert profile.info.type == raw_profile.info.type
        assert profile.info.support == raw_profile.info.support
        assert profile.info.channel_model == 'color'
        assert profile.info.densitometer == raw_profile.info.densitometer
        assert profile.info.reference_illuminant == raw_profile.info.reference_illuminant
        assert profile.info.viewing_illuminant == raw_profile.info.viewing_illuminant

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
        raw_profile = load_raw_profile(case.stock)

        assert profile.info.stock == case.stock
        assert profile.info.type == raw_profile.info.type
        assert profile.info.support == raw_profile.info.support
        assert profile.info.channel_model == 'color'
        assert profile.info.densitometer == raw_profile.info.densitometer
        assert profile.info.reference_illuminant == raw_profile.info.reference_illuminant
        assert profile.info.viewing_illuminant == raw_profile.info.viewing_illuminant

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