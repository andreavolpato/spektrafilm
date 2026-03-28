from __future__ import annotations

import numpy as np
import pytest

from spektrafilm.runtime.process import photo_process

from tests.profiles_creator.create_profile_regression_baselines import assert_matches_baseline, load_baseline
from tests.profiles_creator.helpers import make_test_runtime_params


pytestmark = [pytest.mark.slow, pytest.mark.regression]


def _assert_runtime_output_is_valid(output: np.ndarray) -> None:
    assert output.shape == (8, 8, 3)
    assert np.isfinite(output).all()
    assert float(np.min(output)) >= 0.0
    assert float(np.max(output)) <= 1.0


@pytest.mark.parametrize(
    ('fixture_name', 'support'),
    [
        ('portra_400_processed_profile', 'film'),
        ('portra_endura_paper_processed_profile', 'paper'),
    ],
    ids=['processed-film-profile', 'processed-paper-profile'],
)
def test_processed_profile_matches_regression_baseline(request, fixture_name: str, support: str) -> None:
    case, profile = request.getfixturevalue(fixture_name)
    expected = load_baseline(case.case_id)

    assert profile.info.stock == case.stock
    assert profile.info.type == 'negative'
    assert profile.info.support == support
    assert profile.info.channel_model == 'color'

    assert_matches_baseline(case.case_id, profile, expected)


@pytest.mark.parametrize(
    ('fixture_name', 'target_attr'),
    [
        ('portra_400_processed_profile', 'film'),
        ('portra_endura_paper_processed_profile', 'print'),
    ],
    ids=['processed-film-runtime', 'processed-paper-runtime'],
)
def test_processed_profile_runs_in_runtime_pipeline(request, fixture_name: str, target_attr: str) -> None:
    case, profile = request.getfixturevalue(fixture_name)
    params = make_test_runtime_params(case.runtime_print_paper)
    setattr(params, target_attr, profile)
    image = np.ones((8, 8, 3), dtype=np.float64) * 0.184

    output = np.asarray(photo_process(image, params), dtype=np.float64)

    _assert_runtime_output_is_valid(output)