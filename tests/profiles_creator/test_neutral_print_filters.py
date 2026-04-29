import copy
from types import SimpleNamespace

import numpy as np
import pytest

from spektrafilm.runtime.api import digest_params, init_params
from spektrafilm.utils.io import read_neutral_print_filters
import spektrafilm_profile_creator.neutral_print_filters as print_filters_module
from spektrafilm_profile_creator.neutral_print_filters import (
    DEFAULT_NEUTRAL_PRINT_FILTERS,
    NeutralPrintFilterFitResult,
    NeutralPrintFilterRegenerationConfig,
    fit_neutral_filter_database,
    fit_neutral_filter_entry,
    fit_neutral_filters,
)


def _make_fake_filter_params(y_filter: float = 0.0, m_filter: float = 0.0, c_filter: float = 0.0):
    return SimpleNamespace(
        film=SimpleNamespace(info=SimpleNamespace(is_positive=False, stock='test_film')),
        print=SimpleNamespace(info=SimpleNamespace(is_negative=True)),
        enlarger=SimpleNamespace(
            illuminant=None,
            print_exposure=1.0,
            y_filter_neutral=y_filter,
            m_filter_neutral=m_filter,
            c_filter_neutral=c_filter,
        )
    )


def _make_runtime_like_filter_params(y_filter: float = 0.0, m_filter: float = 0.0, c_filter: float = 0.0):
    return SimpleNamespace(
        film=SimpleNamespace(info=SimpleNamespace(is_positive=False, stock='test_film')),
        print=SimpleNamespace(info=SimpleNamespace(is_negative=True)),
        debug=SimpleNamespace(
            deactivate_spatial_effects=False,
            deactivate_stochastic_effects=False,
            debug_mode='preview',
            inject_film_density_cmy=True,
        ),
        settings=SimpleNamespace(neutral_print_filters_from_database=True),
        io=SimpleNamespace(
            input_cctf_decoding=True,
            input_color_space='ProPhoto RGB',
            output_cctf_encoding=True,
            upscale_factor=2.0,
        ),
        camera=SimpleNamespace(auto_exposure=True),
        enlarger=SimpleNamespace(
            illuminant='light_a',
            print_exposure=1.0,
            print_exposure_compensation=True,
            normalize_print_exposure=False,
            y_filter_neutral=y_filter,
            m_filter_neutral=m_filter,
            c_filter_neutral=c_filter,
        ),
        print_render=SimpleNamespace(glare=SimpleNamespace(active=True)),
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
def test_fit_neutral_filters_returns_bounded_solution_and_reduces_midgray_error():
    film_profile = 'kodak_portra_400'
    print_profile = 'kodak_portra_endura'
    params = init_params(
        film_profile=film_profile,
        print_profile=print_profile,
    )

    start_c = float(params.enlarger.c_filter_neutral)
    start_m = float(params.enlarger.m_filter_neutral)
    start_y = float(params.enlarger.y_filter_neutral)

    fit_result = fit_neutral_filters(
        params,
        iterations=1,
        normalize_print_exposure=False,
    )

    # Printing filters are stored in Kodak CC units, so fitted values are expected in the tens.
    assert 0.0 <= fit_result.c_filter <= 230.0
    assert 0.0 <= fit_result.y_filter <= 230.0
    assert 0.0 <= fit_result.m_filter <= 230.0
    assert fit_result.y_filter >= 10.0
    assert fit_result.m_filter >= 10.0
    assert fit_result.residual.shape == (3,)
    assert np.isfinite(fit_result.residual).all()
    assert fit_result.total_residual < 1e-3

    # fit_neutral_filters currently returns values without mutating the input params.
    assert float(params.enlarger.c_filter_neutral) == start_c
    assert float(params.enlarger.y_filter_neutral) == start_y
    assert float(params.enlarger.m_filter_neutral) == start_m


@pytest.mark.unit
def test_fit_neutral_filters_skips_positive_film_on_negative_print(monkeypatch):
    params = _make_fake_filter_params(y_filter=11.0, m_filter=22.0, c_filter=33.0)
    params.film.info.is_positive = True
    params.print.info.is_negative = True

    monkeypatch.setattr(
        print_filters_module,
        '_fit_once',
        lambda *args, **kwargs: pytest.fail('positive film on negative print should be skipped'),
    )

    fit_result = fit_neutral_filters(params, iterations=3)

    assert fit_result.c_filter == pytest.approx(33.0)
    assert fit_result.y_filter == pytest.approx(11.0)
    assert fit_result.m_filter == pytest.approx(22.0)
    assert fit_result.print_exposure == pytest.approx(1.0, abs=1e-3)
    np.testing.assert_array_equal(fit_result.residual, np.zeros(3, dtype=np.float64))


@pytest.mark.unit
def test_fit_neutral_filters_uses_default_cyan_start(monkeypatch):
    params = _make_fake_filter_params(y_filter=11.0, m_filter=22.0, c_filter=0.0)
    captured_start_filters = []

    def fake_solve_once(
        profile,
        start_filters,
        *,
        fit_input,
        injected_film_density_cmy,
        normalize_print_exposure,
    ):
        del profile
        del fit_input
        del injected_film_density_cmy
        del normalize_print_exposure
        captured_start_filters.append(tuple(start_filters))
        return NeutralPrintFilterFitResult(
            c_filter=float(start_filters[0]),
            m_filter=float(start_filters[1]),
            y_filter=float(start_filters[2]),
            print_exposure=1.0,
            residual=np.zeros(3, dtype=np.float64),
        )

    monkeypatch.setattr(print_filters_module, '_fit_once', fake_solve_once)

    fit_result = fit_neutral_filters(params, iterations=1)

    assert captured_start_filters == [
        (float(DEFAULT_NEUTRAL_PRINT_FILTERS[0]), 22.0, 11.0)
    ]
    assert fit_result.c_filter == pytest.approx(0.0)
    assert fit_result.y_filter == pytest.approx(11.0)
    assert fit_result.m_filter == pytest.approx(22.0)
    np.testing.assert_array_equal(fit_result.residual, np.zeros(3, dtype=np.float64))


@pytest.mark.unit
def test_prepare_fitting_profile_disables_database_neutral_filters(monkeypatch):
    params = _make_runtime_like_filter_params(y_filter=11.0, m_filter=22.0, c_filter=33.0)
    digest_flags = []
    prepare_profile_for_fitting = vars(print_filters_module)['_prepare_profile_for_fitting']

    def fake_digest(profile):
        digest_flags.append(profile.settings.neutral_print_filters_from_database)
        return profile

    monkeypatch.setattr(print_filters_module, 'digest_params', fake_digest)

    prepared = prepare_profile_for_fitting(
        params,
        injected_film_density_cmy=False,
        normalize_print_exposure=True,
    )

    assert digest_flags == [False]
    assert prepared.settings.neutral_print_filters_from_database is False
    assert prepared.debug.deactivate_spatial_effects is True
    assert prepared.debug.deactivate_stochastic_effects is True
    assert prepared.debug.debug_mode == 'off'
    assert prepared.debug.inject_film_density_cmy is False
    assert prepared.io.output_cctf_encoding is False
    assert prepared.enlarger.normalize_print_exposure is True
    assert prepared.print_render.glare.active is False


@pytest.mark.unit
def test_fit_once_uses_prepared_params_without_redigest(monkeypatch):
    params = _make_runtime_like_filter_params(y_filter=0.0, m_filter=0.0, c_filter=0.0)
    simulate_digest_flags = []
    fit_once = vars(print_filters_module)['_fit_once']

    monkeypatch.setattr(print_filters_module, 'digest_params', lambda profile: profile)

    def fake_simulate(_image, profile, digest_params_first=True, print_timings=False):
        del print_timings
        simulate_digest_flags.append(digest_params_first)
        residual = np.asarray(
            [
                profile.enlarger.m_filter_neutral - 10.0,
                profile.enlarger.y_filter_neutral - 20.0,
                profile.enlarger.print_exposure - 1.5,
            ],
            dtype=np.float64,
        ) * 1e-2
        return print_filters_module.MIDGRAY_RGB + np.reshape(residual, (1, 1, 3))

    monkeypatch.setattr(print_filters_module, 'simulate', fake_simulate)

    fit_result = fit_once(
        params,
        start_filters=(0.0, 0.0, 0.0),
        fit_input=print_filters_module.MIDGRAY_RGB,
        injected_film_density_cmy=False,
        normalize_print_exposure=False,
    )

    assert fit_result.c_filter == pytest.approx(0.0)
    assert fit_result.m_filter == pytest.approx(10.0, abs=1e-3)
    assert fit_result.y_filter == pytest.approx(20.0, abs=1e-3)
    assert fit_result.print_exposure == pytest.approx(1.5, abs=1e-3)
    assert fit_result.total_residual < 1e-6
    assert simulate_digest_flags
    assert set(simulate_digest_flags) == {False}


@pytest.mark.unit
def test_fit_once_still_optimizes_print_exposure_when_normalized(monkeypatch):
    params = _make_runtime_like_filter_params(y_filter=0.0, m_filter=0.0, c_filter=0.0)
    fit_once = vars(print_filters_module)['_fit_once']
    seen_exposures = []

    monkeypatch.setattr(print_filters_module, 'digest_params', lambda profile: profile)

    def fake_simulate(_image, profile, digest_params_first=True, print_timings=False):
        del digest_params_first
        del print_timings
        seen_exposures.append(profile.enlarger.print_exposure)
        residual = np.asarray(
            [
                profile.enlarger.m_filter_neutral - 10.0,
                profile.enlarger.y_filter_neutral - 20.0,
                profile.enlarger.print_exposure - 1.0,
            ],
            dtype=np.float64,
        ) * 1e-2
        return print_filters_module.MIDGRAY_RGB + np.reshape(residual, (1, 1, 3))

    monkeypatch.setattr(print_filters_module, 'simulate', fake_simulate)

    fit_result = fit_once(
        params,
        start_filters=(0.0, 0.0, 0.0),
        fit_input=print_filters_module.MIDGRAY_RGB,
        injected_film_density_cmy=False,
        normalize_print_exposure=True,
    )

    assert seen_exposures
    assert any(exposure != pytest.approx(1.0) for exposure in seen_exposures)
    assert fit_result.print_exposure == pytest.approx(1.0, abs=1e-3)
    assert fit_result.m_filter == pytest.approx(10.0, abs=1e-3)
    assert fit_result.y_filter == pytest.approx(20.0, abs=1e-3)
    assert fit_result.total_residual < 1e-6


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
def test_fit_neutral_filter_database_skips_resolved_entries_and_does_not_mutate_inputs(monkeypatch):
    _install_fake_filter_axes(monkeypatch)

    created_params = []
    fit_calls = []

    def fake_init_params(*, film_profile, print_profile):
        params = _make_fake_filter_params()
        created_params.append((film_profile, print_profile, params))
        return params

    def fake_fit(params, iterations=10, rng=None, normalize_print_exposure=True):
        del rng
        fit_calls.append(
            (
                iterations,
                params.enlarger.illuminant,
                params.enlarger.y_filter_neutral,
                params.enlarger.m_filter_neutral,
                params.enlarger.c_filter_neutral,
                normalize_print_exposure,
            )
        )
        return NeutralPrintFilterFitResult(
            c_filter=params.enlarger.c_filter_neutral + 1.0,
            m_filter=params.enlarger.m_filter_neutral + 10.0,
            y_filter=params.enlarger.y_filter_neutral + 5.0,
            print_exposure=1.0,
            residual=np.array([0.0, 1e-4, -1e-4], dtype=np.float64),
        )

    monkeypatch.setattr(print_filters_module, 'init_params', fake_init_params)
    monkeypatch.setattr(print_filters_module, 'fit_neutral_filters', fake_fit)

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

    result = fit_neutral_filter_database(
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
    assert fit_calls == [(7, 'light_a', 40.0, 50.0, 0.0, True)]
    assert result[0]['paper_a']['light_a']['film_a'] == [30.0, 20.0, 10.0]
    assert result[0]['paper_a']['light_a']['film_b'] == pytest.approx([1.0, 60.0, 45.0])
    assert result[1]['paper_a']['light_a']['film_b'] == pytest.approx(2e-4)
    assert filters == original_filters
    assert residues == original_residues


@pytest.mark.unit
def test_fit_neutral_filter_database_skips_positive_film_on_negative_print(monkeypatch):
    _install_fake_filter_axes(monkeypatch)

    fit_calls = []

    def fake_init_params(*, film_profile, print_profile):
        del print_profile
        params = _make_fake_filter_params()
        params.film.info.is_positive = film_profile == 'film_b'
        params.print.info.is_negative = True
        return params

    def fake_fit(params, iterations=10, rng=None, normalize_print_exposure=True):
        del params, rng
        fit_calls.append((iterations, normalize_print_exposure))
        return NeutralPrintFilterFitResult(
            c_filter=15.0,
            m_filter=45.0,
            y_filter=60.0,
            print_exposure=1.0,
            residual=np.array([0.0, 1e-4, -1e-4], dtype=np.float64),
        )

    monkeypatch.setattr(print_filters_module, 'init_params', fake_init_params)
    monkeypatch.setattr(print_filters_module, 'fit_neutral_filters', fake_fit)

    result = fit_neutral_filter_database(
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

    assert fit_calls == [(7, True)]
    assert result[0]['paper_a']['light_a']['film_b'] == [60.0, 50.0, 40.0]
    assert result[1]['paper_a']['light_a']['film_b'] == pytest.approx(0.0)


@pytest.mark.unit
def test_fit_neutral_filter_entry_updates_only_requested_combination(monkeypatch):
    created_params = []
    fit_calls = []

    def fake_init_params(*, film_profile, print_profile):
        params = _make_fake_filter_params()
        created_params.append((film_profile, print_profile, params))
        return params

    def fake_fit(params, iterations=10, rng=None, normalize_print_exposure=True):
        del rng
        fit_calls.append(
            (
                iterations,
                params.enlarger.illuminant,
                params.enlarger.y_filter_neutral,
                params.enlarger.m_filter_neutral,
                params.enlarger.c_filter_neutral,
                normalize_print_exposure,
            )
        )
        return NeutralPrintFilterFitResult(
            c_filter=params.enlarger.c_filter_neutral + 3.0,
            m_filter=params.enlarger.m_filter_neutral + 4.0,
            y_filter=params.enlarger.y_filter_neutral + 2.0,
            print_exposure=1.0,
            residual=np.array([0.0, 1e-4, -1e-4], dtype=np.float64),
        )

    monkeypatch.setattr(print_filters_module, 'init_params', fake_init_params)
    monkeypatch.setattr(print_filters_module, 'fit_neutral_filters', fake_fit)

    filters = {
        'paper_a': {
            'light_a': {
                'film_a': [30.0, 20.0, 10.0],
                'film_b': [40.0, 50.0, 60.0],
            }
        }
    }
    residues = {
        'paper_a': {
            'light_a': {
                'film_a': 1.0,
                'film_b': 1.0,
            }
        }
    }
    original_filters = copy.deepcopy(filters)
    original_residues = copy.deepcopy(residues)

    result = fit_neutral_filter_entry(
        stock='film_b',
        paper='paper_a',
        illuminant='light_a',
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
    assert fit_calls == [(7, 'light_a', 60.0, 50.0, 0.0, True)]
    assert result[0]['paper_a']['light_a']['film_a'] == [30.0, 20.0, 10.0]
    assert result[0]['paper_a']['light_a']['film_b'] == pytest.approx([3.0, 54.0, 62.0])
    assert result[1]['paper_a']['light_a']['film_a'] == pytest.approx(1.0)
    assert result[1]['paper_a']['light_a']['film_b'] == pytest.approx(2e-4)
    assert filters == original_filters
    assert residues == original_residues


@pytest.mark.unit
def test_fit_neutral_filter_entry_creates_missing_nodes(monkeypatch):
    created_params = []

    def fake_init_params(*, film_profile, print_profile):
        params = _make_fake_filter_params()
        created_params.append((film_profile, print_profile, params))
        return params

    def fake_fit(params, iterations=10, rng=None, normalize_print_exposure=True):
        del rng
        assert iterations == 5
        assert normalize_print_exposure is True
        assert params.enlarger.illuminant == 'light_a'
        assert params.enlarger.c_filter_neutral == pytest.approx(1.0)
        assert params.enlarger.m_filter_neutral == pytest.approx(2.0)
        assert params.enlarger.y_filter_neutral == pytest.approx(3.0)
        return NeutralPrintFilterFitResult(
            c_filter=4.0,
            m_filter=5.0,
            y_filter=6.0,
            print_exposure=1.0,
            residual=np.array([1e-4, -1e-4, 0.0], dtype=np.float64),
        )

    monkeypatch.setattr(print_filters_module, 'init_params', fake_init_params)
    monkeypatch.setattr(print_filters_module, 'fit_neutral_filters', fake_fit)

    result = fit_neutral_filter_entry(
        stock='film_b',
        paper='paper_a',
        illuminant='light_a',
        config=NeutralPrintFilterRegenerationConfig(
            iterations=5,
            restart_randomness=0.0,
            initial_filters=(1.0, 2.0, 3.0),
            rng_seed=123,
        ),
        neutral_print_filters={},
        residues={},
    )

    assert created_params[0][0:2] == ('film_b', 'paper_a')
    assert result[0] == {'paper_a': {'light_a': {'film_b': [4.0, 5.0, 6.0]}}}
    assert result[1] == {'paper_a': {'light_a': {'film_b': pytest.approx(2e-4)}}}

