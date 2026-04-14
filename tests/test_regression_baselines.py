import pytest

from tests.regression_baselines import (
    assert_matches_baseline,
    case_ids,
    compute_case_output,
    find_case,
    load_baseline,
)


pytestmark = pytest.mark.regression


class TestRegressionBaselines:
    @pytest.mark.parametrize("case_id", case_ids())
    def test_pipeline_snapshot(self, case_id):
        case = find_case(case_id)
        expected = load_baseline(case_id)
        actual = compute_case_output(case)
        assert_matches_baseline(case_id, actual, expected)
