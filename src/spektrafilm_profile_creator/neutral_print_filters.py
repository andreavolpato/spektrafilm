from __future__ import annotations

import copy
import functools
from dataclasses import dataclass

import numpy as np
from scipy import optimize

from spektrafilm.model.illuminants import Illuminants
from spektrafilm.model.stocks import FilmStocks, PrintPapers
from spektrafilm.runtime.api import digest_params, init_params, simulate
from spektrafilm.utils.io import read_neutral_print_filters, save_neutral_print_filters
from spektrafilm_profile_creator.diagnostics.messages import log_event


MIDGRAY_RGB = np.array([[[0.184, 0.184, 0.184]]], dtype=np.float64)
DEFAULT_NEUTRAL_PRINT_FILTERS = (0, 50, 50)  # kodak cc values in CMY order
DEFAULT_INITIAL_PRINT_EXPOSURE = 1.2
DEFAULT_INITIAL_PRINT_FILTERS = (0, 70, 70)  # kodak cc values in CMY order
DEFAULT_RESIDUE_THRESHOLD = 5e-4

NeutralPrintFilterDatabase = dict[str, dict[str, dict[str, list[float]]]]
NeutralPrintFilterResidueDatabase = dict[str, dict[str, dict[str, float]]]


@dataclass(frozen=True, slots=True)
class NeutralPrintFilterRegenerationConfig:
    iterations: int = 20
    restart_randomness: float = 0.5
    residue_threshold: float = DEFAULT_RESIDUE_THRESHOLD
    initial_filters: tuple[float, float, float] = DEFAULT_NEUTRAL_PRINT_FILTERS
    rng_seed: int | None = None

    def __post_init__(self) -> None:
        if self.iterations < 1:
            raise ValueError('iterations must be >= 1')
        if not 0.0 <= self.restart_randomness <= 1.0:
            raise ValueError('restart_randomness must be between 0.0 and 1.0')
        if self.residue_threshold < 0.0:
            raise ValueError('residue_threshold must be >= 0.0')


# ── Public API ─────────────────────────────────────────────────────────────────

def fit_neutral_filters(profile, iterations=10, rng=None):
    if iterations < 1:
        raise ValueError('iterations must be >= 1')

    log_event('fit_neutral_filters', stock=profile.film.info.stock)

    if _should_skip_filter_fit(profile):
        return (
            float(profile.enlarger.c_filter_neutral),
            float(profile.enlarger.m_filter_neutral),
            float(profile.enlarger.y_filter_neutral),
            np.zeros(3, dtype=np.float64),
        )

    if rng is None:
        rng = np.random.default_rng()

    start_filters = (
        float(DEFAULT_NEUTRAL_PRINT_FILTERS[0]),
        float(profile.enlarger.m_filter_neutral),
        float(profile.enlarger.y_filter_neutral),
    )
    fitted_c, fitted_m, fitted_y, residues = _fit_once(profile, start_filters=start_filters)
    for _ in range(1, iterations):
        if _total_residue(residues) < DEFAULT_RESIDUE_THRESHOLD:
            break
        start_filters = (
            float(fitted_c),
            0.5 * float(fitted_m) + float(rng.uniform(0.0, 1.0)) * 50.0,
            0.5 * float(fitted_y) + float(rng.uniform(0.0, 1.0)) * 50.0,
        )
        fitted_c, fitted_m, fitted_y, residues = _fit_once(profile, start_filters=start_filters)

    log_event(
        'fit_neutral_filters_result',
        fitted_filters=(fitted_c, fitted_m, fitted_y),
        residual=residues,
    )
    return float(fitted_c), float(fitted_m), float(fitted_y), residues


def fit_neutral_filter_entry(
    *,
    stock: str,
    paper: str,
    illuminant: str = Illuminants.lamp.value,
    config: NeutralPrintFilterRegenerationConfig | None = None,
    neutral_print_filters: NeutralPrintFilterDatabase | None = None,
    residues: NeutralPrintFilterResidueDatabase | None = None,
) -> tuple[NeutralPrintFilterDatabase, NeutralPrintFilterResidueDatabase]:
    config = config or NeutralPrintFilterRegenerationConfig()
    rng = np.random.default_rng(config.rng_seed)
    working_filters = copy.deepcopy(neutral_print_filters) if neutral_print_filters is not None else read_neutral_print_filters()
    working_residues = copy.deepcopy(residues) if residues is not None else _new_residue_database()

    _fit_entry(
        stock=stock,
        paper=paper,
        illuminant=illuminant,
        config=config,
        working_filters=working_filters,
        working_residues=working_residues,
        rng=rng,
    )
    return working_filters, working_residues


def fit_neutral_filter_database(
    config: NeutralPrintFilterRegenerationConfig | None = None,
    neutral_print_filters: NeutralPrintFilterDatabase | None = None,
    residues: NeutralPrintFilterResidueDatabase | None = None,
) -> tuple[NeutralPrintFilterDatabase, NeutralPrintFilterResidueDatabase]:
    config = config or NeutralPrintFilterRegenerationConfig()
    rng = np.random.default_rng(config.rng_seed)
    working_filters = (
        copy.deepcopy(neutral_print_filters)
        if neutral_print_filters is not None
        else _new_filter_database(config.initial_filters)
    )
    working_residues = (
        copy.deepcopy(residues)
        if residues is not None
        else _new_residue_database()
    )
    fit_count = 0
    skip_count = 0

    log_event(
        'fit_neutral_filter_database_start',
        iterations=config.iterations,
        restart_randomness=config.restart_randomness,
        residue_threshold=config.residue_threshold,
    )
    for paper in PrintPapers:
        log_event('fit_neutral_filter_database_paper', paper=paper.value)
        for light in Illuminants:
            log_event(
                'fit_neutral_filter_database_illuminant',
                paper=paper.value,
                illuminant=light.value,
            )
            for stock in FilmStocks:
                did_fit = _fit_entry(
                    stock=stock.value,
                    paper=paper.value,
                    illuminant=light.value,
                    config=config,
                    working_filters=working_filters,
                    working_residues=working_residues,
                    rng=rng,
                )
                if did_fit:
                    fit_count += 1
                else:
                    skip_count += 1

    log_event(
        'fit_neutral_filter_database_complete',
        fit_count=fit_count,
        skip_count=skip_count,
    )
    return working_filters, working_residues


def regenerate_neutral_filter_database(
    config: NeutralPrintFilterRegenerationConfig | None = None,
) -> tuple[NeutralPrintFilterDatabase, NeutralPrintFilterResidueDatabase]:
    filters, residues = fit_neutral_filter_database(config=config)
    save_neutral_print_filters(filters)
    log_event(
        'regenerate_neutral_filter_database_complete',
        combinations_saved=len(PrintPapers) * len(Illuminants) * len(FilmStocks),
    )
    return filters, residues


# ── Private helpers ────────────────────────────────────────────────────────────

def _should_skip_filter_fit(profile) -> bool:
    return profile.film.info.is_positive and profile.print.info.is_negative


def _set_neutral_filters(params, *, c_filter: float, m_filter: float, y_filter: float) -> None:
    params.enlarger.c_filter_neutral = float(c_filter)
    params.enlarger.m_filter_neutral = float(m_filter)
    params.enlarger.y_filter_neutral = float(y_filter)


def _prepare_profile_for_fitting(params):
    params.debug.deactivate_spatial_effects = True
    params.debug.deactivate_stochastic_effects = True
    params.settings.neutral_print_filters_from_database = False
    params.io.input_cctf_decoding = False
    params.io.input_color_space = 'sRGB'
    params.io.upscale_factor = 1.0
    params.camera.auto_exposure = False
    params.enlarger.print_exposure_compensation = False
    params.enlarger.normalize_print_exposure = True # makes sure print exposure 1.0 is midgray out for midgray in
    return digest_params(params)


def _total_residue(residues: np.ndarray) -> float:
    return float(np.sum(np.abs(residues)))


def _midgray_residues(working_profile, c_filter: float, values) -> np.ndarray:
    m_filter, y_filter, print_exposure = values
    _set_neutral_filters(working_profile, c_filter=c_filter, m_filter=m_filter, y_filter=y_filter)
    working_profile.enlarger.print_exposure = float(print_exposure)
    rendered_midgray = simulate(MIDGRAY_RGB, working_profile, digest_params_first=False)
    return (rendered_midgray - MIDGRAY_RGB).reshape(-1)


def _fit_once(profile, start_filters):
    working_profile = _prepare_profile_for_fitting(copy.deepcopy(profile))
    c_filter, m_filter, y_filter = (float(v) for v in start_filters)
    x0 = np.array([m_filter, y_filter, DEFAULT_INITIAL_PRINT_EXPOSURE], dtype=np.float64)
    evaluate_residues = functools.partial(_midgray_residues, working_profile, c_filter)
    initial_residual = evaluate_residues(x0)
    fit = optimize.least_squares(
        evaluate_residues,
        x0,
        bounds=([0.0, 0.0, 0.0], [230.0, 230.0, 10.0]),
        ftol=1e-6,
        xtol=1e-6,
        gtol=1e-6,
        method='trf',
    )
    log_event(
        'fit_neutral_filters_iter',
        total_residual=_total_residue(fit.fun),
        initial_residual=initial_residual,
    )
    return c_filter, float(fit.x[0]), float(fit.x[1]), fit.fun


def _new_filter_database(initial_filters=DEFAULT_INITIAL_PRINT_FILTERS) -> NeutralPrintFilterDatabase:
    return {
        paper.value: {
            light.value: {
                film.value: [float(initial_filters[0]), float(initial_filters[1]), float(initial_filters[2])]
                for film in FilmStocks
            }
            for light in Illuminants
        }
        for paper in PrintPapers
    }


def _new_residue_database(initial_value=float('inf')) -> NeutralPrintFilterResidueDatabase:
    return {
        paper.value: {
            light.value: {
                film.value: float(initial_value)
                for film in FilmStocks
            }
            for light in Illuminants
        }
        for paper in PrintPapers
    }


def _perturb_start_filters(filters, randomness, rng, initial_filters=DEFAULT_INITIAL_PRINT_FILTERS):
    _, m_filter, y_filter = filters
    randomized_m = np.clip(m_filter, 0.0, 230.0) * (1.0 - randomness) + rng.uniform(0.0, 1.0) * randomness * 50.0
    randomized_y = np.clip(y_filter, 0.0, 230.0) * (1.0 - randomness) + rng.uniform(0.0, 1.0) * randomness * 50.0
    return [float(initial_filters[0]), float(randomized_m), float(randomized_y)]


def _make_fit_params(stock, paper, illuminant, start_filters):
    params = init_params(film_profile=stock, print_profile=paper)
    params.enlarger.illuminant = illuminant
    _set_neutral_filters(
        params,
        c_filter=start_filters[0],
        m_filter=start_filters[1],
        y_filter=start_filters[2],
    )
    return params


def _fit_entry(
    *,
    stock: str,
    paper: str,
    illuminant: str,
    config: NeutralPrintFilterRegenerationConfig,
    working_filters: NeutralPrintFilterDatabase,
    working_residues: NeutralPrintFilterResidueDatabase,
    rng,
) -> bool:
    if float(working_residues[paper][illuminant][stock]) <= config.residue_threshold:
        return False

    start_filters = _perturb_start_filters(
        working_filters[paper][illuminant][stock],
        config.restart_randomness,
        rng,
        initial_filters=config.initial_filters,
    )
    params = _make_fit_params(stock, paper, illuminant, start_filters)
    if _should_skip_filter_fit(params):
        working_residues[paper][illuminant][stock] = 0.0
        return False

    fitted_c, fitted_m, fitted_y, fit_residues = fit_neutral_filters(
        params,
        iterations=config.iterations,
        rng=rng,
    )
    working_filters[paper][illuminant][stock] = [float(fitted_c), float(fitted_m), float(fitted_y)]
    working_residues[paper][illuminant][stock] = _total_residue(fit_residues)
    return True


__all__ = [
    'NeutralPrintFilterRegenerationConfig',
    'fit_neutral_filter_database',
    'fit_neutral_filter_entry',
    'fit_neutral_filters',
    'regenerate_neutral_filter_database',
]
