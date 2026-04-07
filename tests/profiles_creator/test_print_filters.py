import copy
from types import SimpleNamespace

import numpy as np
import pytest

from spektrafilm.runtime.api import digest_params, init_params
from spektrafilm.utils.io import read_neutral_print_filters
import spektrafilm_profile_creator.neutral_print_filters as print_filters_module
from spektrafilm_profile_creator.neutral_print_filters import (
    DEFAULT_NEUTRAL_PRINT_FILTERS,
    NeutralPrintFilterRegenerationConfig,
    fit_neutral_print_filter_database,
    fit_neutral_print_filters,
)


def _make_fake_filter_params(y_filter: float = 0.0, m_filter: float = 0.0, c_filter: float = 0.0):
    return SimpleNamespace(
        film=SimpleNamespace(info=SimpleNamespace(is_positive=False)),
        print=SimpleNamespace(info=SimpleNamespace(is_negative=True)),
        enlarger=SimpleNamespace(
            illuminant=None,
            y_filter_neutral=y_filter,
            m_filter_neutral=m_filter,
            c_filter_neutral=c_filter,
        )
    )


def _install_fake_filter_axes(monkeypatch) -> None:
    monkeypatch.setattr(print_filters_module, 'PrintPapers', [SimpleNamespace(value='paper_a')])
    monkeypatch.setattr(print_filters_module, 'Illuminants', [SimpleNamespace(value='light_a')])
    monkeypatch.setattr(
        print_filters_module,
        'FilmStocks',
        [SimpleNamespace(value='film_a'), SimpleNamespace(value='film_b')],
    )


@pytest.mark.integration
def test_fit_neutral_print_filters_returns_bounded_solution_and_reduces_midgray_error():
    film_profile = 'kodak_portra_400'
    print_profile = 'kodak_portra_endura'
    params = init_params(
        film_profile=film_profile,
        print_profile=print_profile,
    )

    start_y = float(params.enlarger.y_filter_neutral)
    start_m = float(params.enlarger.m_filter_neutral)

    fitted_y, fitted_m, residuals = fit_neutral_print_filters(
        params,
        iterations=1,
        stock=film_profile,
    )

    # Printing filters are stored in Kodak CC units, so fitted values are expected in the tens.
    assert 0.0 <= fitted_y <= 230.0
    assert 0.0 <= fitted_m <= 230.0
    assert fitted_y >= 10.0
    assert fitted_m >= 10.0
    assert residuals.shape == (3,)
    assert np.isfinite(residuals).all()
    assert np.sum(np.abs(residuals)) < 1e-3

    # fit_neutral_print_filters currently returns values without mutating the input params.
    assert float(params.enlarger.y_filter_neutral) == start_y
    assert float(params.enlarger.m_filter_neutral) == start_m


@pytest.mark.unit
def test_fit_neutral_print_filters_skips_positive_film_on_negative_print(monkeypatch):
    params = _make_fake_filter_params(y_filter=11.0, m_filter=22.0, c_filter=33.0)
    params.film.info.is_positive = True
    params.print.info.is_negative = True

    monkeypatch.setattr(
        print_filters_module,
        'fit_neutral_print_filters_iter',
        lambda *args, **kwargs: pytest.fail('positive film on negative print should be skipped'),
    )

    fitted_y, fitted_m, residuals = fit_neutral_print_filters(params, iterations=3, stock='film_b')

    assert fitted_y == pytest.approx(11.0)
    assert fitted_m == pytest.approx(22.0)
    np.testing.assert_array_equal(residuals, np.zeros(3, dtype=np.float64))


@pytest.mark.unit
def test_fit_neutral_print_filters_uses_default_cyan_start(monkeypatch):
    params = _make_fake_filter_params(y_filter=11.0, m_filter=22.0, c_filter=0.0)
    captured_start_filters = []

    def fake_fit_iter(profile, start_filters):
        del profile
        captured_start_filters.append(tuple(start_filters))
        return float(start_filters[0]), float(start_filters[1]), float(start_filters[2]), np.zeros(3, dtype=np.float64)

    monkeypatch.setattr(print_filters_module, 'fit_neutral_print_filters_iter', fake_fit_iter)

    fitted_y, fitted_m, residuals = fit_neutral_print_filters(params, iterations=1, stock='film_a')

    assert captured_start_filters == [
        (float(DEFAULT_NEUTRAL_PRINT_FILTERS[0]), 22.0, 11.0)
    ]
    assert fitted_y == pytest.approx(11.0)
    assert fitted_m == pytest.approx(22.0)
    np.testing.assert_array_equal(residuals, np.zeros(3, dtype=np.float64))


@pytest.mark.unit
def test_digest_params_reads_neutral_print_filters_in_cmy_order():
    film_profile = 'kodak_portra_400'
    print_profile = 'kodak_portra_endura'

    params = digest_params(
        init_params(
            film_profile=film_profile,
            print_profile=print_profile,
        )
    )
    expected_filters = read_neutral_print_filters()[print_profile][params.enlarger.illuminant][film_profile]

    np.testing.assert_allclose(
        np.array(
            [
                params.enlarger.c_filter_neutral,
                params.enlarger.m_filter_neutral,
                params.enlarger.y_filter_neutral,
            ],
            dtype=np.float64,
        ),
        np.array(expected_filters, dtype=np.float64),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.unit
def test_fit_neutral_print_filter_database_skips_resolved_entries_and_does_not_mutate_inputs(monkeypatch):
    _install_fake_filter_axes(monkeypatch)

    created_params = []
    fit_calls = []

    def fake_init_params(*, film_profile, print_profile):
        params = _make_fake_filter_params()
        created_params.append((film_profile, print_profile, params))
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
            params.enlarger.y_filter_neutral + 5.0,
            params.enlarger.m_filter_neutral + 10.0,
            np.array([0.0, 1e-4, -1e-4], dtype=np.float64),
        )

    monkeypatch.setattr(print_filters_module, 'init_params', fake_init_params)
    monkeypatch.setattr(print_filters_module, 'fit_neutral_print_filters', fake_fit)

    filters = {
        'paper_a': {
            'light_a': {
                'film_a': [30.0, 20.0, 10.0],
                'film_b': [60.0, 50.0, 40.0],
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

    result = fit_neutral_print_filter_database(
        config=NeutralPrintFilterRegenerationConfig(
            iterations=7,
            restart_randomness=0.0,
            residue_threshold=5e-4,
            rng_seed=123,
        ),
        neutral_print_filters=filters,
        residues=residues,
    )

    assert len(created_params) == 1
    assert created_params[0][0:2] == ('film_b', 'paper_a')
    assert fit_calls == [('film_b', 7, 'light_a', 40.0, 50.0, 0.0)]
    assert result.filters['paper_a']['light_a']['film_a'] == [30.0, 20.0, 10.0]
    assert result.filters['paper_a']['light_a']['film_b'] == pytest.approx([0.0, 60.0, 45.0])
    assert result.residues['paper_a']['light_a']['film_b'] == pytest.approx(2e-4)
    assert filters == original_filters
    assert residues == original_residues


@pytest.mark.unit
def test_fit_neutral_print_filter_database_skips_positive_film_on_negative_print(monkeypatch):
    _install_fake_filter_axes(monkeypatch)

    fit_calls = []

    def fake_init_params(*, film_profile, print_profile):
        del print_profile
        params = _make_fake_filter_params()
        params.film.info.is_positive = film_profile == 'film_b'
        params.print.info.is_negative = True
        return params

    def fake_fit(params, iterations=10, stock=None, rng=None):
        del params, rng
        fit_calls.append((stock, iterations))
        return (45.0, 60.0, np.array([0.0, 1e-4, -1e-4], dtype=np.float64))

    monkeypatch.setattr(print_filters_module, 'init_params', fake_init_params)
    monkeypatch.setattr(print_filters_module, 'fit_neutral_print_filters', fake_fit)

    result = fit_neutral_print_filter_database(
        config=NeutralPrintFilterRegenerationConfig(
            iterations=7,
            restart_randomness=0.0,
            residue_threshold=5e-4,
            rng_seed=123,
        ),
        neutral_print_filters={
            'paper_a': {'light_a': {'film_a': [30.0, 20.0, 10.0], 'film_b': [60.0, 50.0, 40.0]}},
        },
        residues={
            'paper_a': {'light_a': {'film_a': 1.0, 'film_b': 1.0}},
        },
    )

    assert fit_calls == [('film_a', 7)]
    assert result.filters['paper_a']['light_a']['film_b'] == [60.0, 50.0, 40.0]
