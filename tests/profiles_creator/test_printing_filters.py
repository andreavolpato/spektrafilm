import copy
from types import SimpleNamespace

import numpy as np
import pytest

from spektrafilm.runtime.api import create_params
from spektrafilm.utils.io import read_neutral_ymc_filter_values
import spektrafilm_profile_creator.printing_filters as printing_filters_module
from spektrafilm_profile_creator.printing_filters import (
    PrintFilterRegenerationConfig,
    fit_print_filter_database,
    fit_print_filters,
)


def _make_fake_filter_params(y_filter: float = 0.0, m_filter: float = 0.0, c_filter: float = 0.0):
    return SimpleNamespace(
        enlarger=SimpleNamespace(
            illuminant=None,
            y_filter_neutral=y_filter,
            m_filter_neutral=m_filter,
            c_filter_neutral=c_filter,
        )
    )


def _install_fake_filter_axes(monkeypatch) -> None:
    monkeypatch.setattr(printing_filters_module, 'PrintPapers', [SimpleNamespace(value='paper_a')])
    monkeypatch.setattr(printing_filters_module, 'Illuminants', [SimpleNamespace(value='light_a')])
    monkeypatch.setattr(
        printing_filters_module,
        'FilmStocks',
        [SimpleNamespace(value='film_a'), SimpleNamespace(value='film_b')],
    )


@pytest.mark.integration
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


@pytest.mark.unit
def test_fit_print_filter_database_skips_resolved_entries_and_does_not_mutate_inputs(monkeypatch):
    _install_fake_filter_axes(monkeypatch)

    created_params = []
    fit_calls = []

    def fake_create_params(*, film_profile, print_profile, ymc_filters_from_database):
        params = _make_fake_filter_params()
        created_params.append((film_profile, print_profile, ymc_filters_from_database, params))
        return params

    def fake_fit(params, iterations=10, stock=None, rng=None):
        del rng
        fit_calls.append(
            (
                stock,
                iterations,
                params.enlarger.illuminant,
                params.enlarger.y_filter_neutral,
                params.enlarger.m_filter_neutral,
                params.enlarger.c_filter_neutral,
            )
        )
        return (
            params.enlarger.y_filter_neutral + 0.05,
            params.enlarger.m_filter_neutral + 0.10,
            np.array([0.0, 1e-4, -1e-4], dtype=np.float64),
        )

    monkeypatch.setattr(printing_filters_module, 'create_params', fake_create_params)
    monkeypatch.setattr(printing_filters_module, 'fit_print_filters', fake_fit)

    filters = {
        'paper_a': {
            'light_a': {
                'film_a': [0.10, 0.20, 0.30],
                'film_b': [0.40, 0.50, 0.60],
            }
        }
    }
    residues = {
        'paper_a': {
            'light_a': {
                'film_a': 1e-6,
                'film_b': 1.0,
            }
        }
    }
    original_filters = copy.deepcopy(filters)
    original_residues = copy.deepcopy(residues)

    result = fit_print_filter_database(
        config=PrintFilterRegenerationConfig(
            iterations=7,
            restart_randomness=0.0,
            residue_threshold=5e-4,
            rng_seed=123,
        ),
        ymc_filters=filters,
        residues=residues,
    )

    assert len(created_params) == 1
    assert created_params[0][0:3] == ('film_b', 'paper_a', False)
    assert fit_calls == [('film_b', 7, 'light_a', 0.40, 0.50, 0.60)]
    assert result.filters['paper_a']['light_a']['film_a'] == [0.10, 0.20, 0.30]
    assert result.filters['paper_a']['light_a']['film_b'] == pytest.approx([0.45, 0.60, 0.60])
    assert result.residues['paper_a']['light_a']['film_b'] == pytest.approx(2e-4)
    assert filters == original_filters
    assert residues == original_residues
