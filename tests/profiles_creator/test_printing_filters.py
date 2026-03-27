import numpy as np

from spektrafilm.runtime.api import create_params
from spektrafilm.utils.io import read_neutral_ymc_filter_values
from spektrafilm_profile_creator.printing_filters import fit_print_filters


def test_fit_print_filters_returns_bounded_solution_and_reduces_midgray_error():
    film_profile = 'kodak_portra_400_auc'
    print_profile = 'kodak_portra_endura_uc'
    params = create_params(
        film_profile=film_profile,
        print_profile=print_profile,
        ymc_filters_from_database=False,
    )
    params.io.full_image = True

    start_y = float(params.enlarger.y_filter_neutral)
    start_m = float(params.enlarger.m_filter_neutral)

    fitted_y, fitted_m, residuals = fit_print_filters(
        params,
        iterations=1,
        stock=film_profile,
    )

    assert 0.0 <= fitted_y <= 1.0
    assert 0.0 <= fitted_m <= 1.0
    assert residuals.shape == (3,)
    assert np.isfinite(residuals).all()
    assert np.sum(np.abs(residuals)) < 1e-3

    expected_ymc = read_neutral_ymc_filter_values()[print_profile][params.enlarger.illuminant][film_profile]
    np.testing.assert_allclose(
        np.array([fitted_y, fitted_m, params.enlarger.c_filter_neutral], dtype=np.float64),
        np.array(expected_ymc, dtype=np.float64),
        rtol=0.0,
        atol=5e-4,
    )

    # fit_print_filters currently returns values without mutating the input params.
    assert float(params.enlarger.y_filter_neutral) == start_y
    assert float(params.enlarger.m_filter_neutral) == start_m