from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import scipy

from spektrafilm.model.illuminants import Illuminants
from spektrafilm.model.stocks import FilmStocks, PrintPapers
from spektrafilm.runtime.api import create_params, simulate
from spektrafilm.utils.io import save_neutral_print_filters
from spektrafilm_profile_creator.diagnostics.messages import log_event


MIDGRAY_RGB = np.array([[[0.184, 0.184, 0.184]]], dtype=np.float64)
DEFAULT_NEUTRAL_PRINT_FILTERS = (0, 50, 50)  # kodak cc values in CMY order
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


@dataclass(frozen=True, slots=True)
class NeutralPrintFilterRegenerationResult:
    filters: NeutralPrintFilterDatabase
    residues: NeutralPrintFilterResidueDatabase


def _prepare_fitting_profile(profile):
    working_profile = copy.deepcopy(profile)
    working_profile.debug.deactivate_spatial_effects = True
    working_profile.debug.deactivate_stochastic_effects = True
    working_profile.print_render.glare.compensation_removal_factor = 0.0
    working_profile.io.input_cctf_decoding = False
    working_profile.io.input_color_space = 'sRGB'
    working_profile.io.resize_factor = 1.0
    working_profile.io.full_image = True
    working_profile.camera.auto_exposure = False
    working_profile.enlarger.print_exposure_compensation = False
    return working_profile


def fit_neutral_print_filters_iter(profile, start_filters):
    working_profile = _prepare_fitting_profile(profile)

    def midgray_print(cmy_values, print_exposure):
        working_profile.enlarger.c_filter_neutral = cmy_values[0]
        working_profile.enlarger.m_filter_neutral = cmy_values[1]
        working_profile.enlarger.y_filter_neutral = cmy_values[2]
        working_profile.enlarger.print_exposure = print_exposure
        return simulate(MIDGRAY_RGB, working_profile)

    def evaluate_residues(values):
        residual = midgray_print([c_filter, values[0], values[1]], values[2])
        return (residual - MIDGRAY_RGB).flatten()

    c_filter, m0, y0 = start_filters
    working_profile.enlarger.c_filter_neutral = c_filter
    working_profile.enlarger.m_filter_neutral = m0
    working_profile.enlarger.y_filter_neutral = y0
    x0 = [m0, y0, 1.0]
    fit = scipy.optimize.least_squares(
        evaluate_residues,
        x0,
        bounds=([0, 0, 0], [230, 230, 10]),
        ftol=1e-6,
        xtol=1e-6,
        gtol=1e-6,
        method='trf',
    )
    log_event(
        'fit_neutral_print_filters_iter',
        total_residual=float(np.sum(np.abs(evaluate_residues(fit.x)))),
        initial_residual=evaluate_residues(x0),
    )
    return float(c_filter), float(fit.x[0]), float(fit.x[1]), evaluate_residues(fit.x)


def fit_neutral_print_filters(profile, iterations=10, stock=None, rng=None):
    if stock is None:
        log_event('fit_neutral_print_filters')
    else:
        log_event('fit_neutral_print_filters', stock=stock)
    if rng is None:
        rng = np.random.default_rng()

    c_filter = float(profile.enlarger.c_filter_neutral)
    current_m = float(profile.enlarger.m_filter_neutral)
    current_y = float(profile.enlarger.y_filter_neutral)
    for index in range(iterations):
        filter_c, filter_m, filter_y, residues = fit_neutral_print_filters_iter(
            profile,
            start_filters=(c_filter, current_m, current_y),
        )
        if np.sum(np.abs(residues)) < 1e-4 or index == iterations - 1:
            log_event(
                'fit_neutral_print_filters_result',
                fitted_filters=(filter_c, filter_m, filter_y),
                residual=residues,
            )
            break

        current_y = 0.5 * filter_y + rng.uniform(0.0, 1.0) * 50
        current_m = 0.5 * filter_m + rng.uniform(0.0, 1.0) * 50
    return filter_y, filter_m, residues


def _build_neutral_print_filter_database(initial_filters=DEFAULT_NEUTRAL_PRINT_FILTERS) -> NeutralPrintFilterDatabase:
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


def _build_neutral_print_filter_residue_database(initial_value=float('inf')) -> NeutralPrintFilterResidueDatabase:
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


def _randomize_start_filters(filters, randomness, rng):
    c_filter, m_filter, y_filter = filters
    randomized_m = np.clip(m_filter, 0.0, 230.0) * (1.0 - randomness) + rng.uniform(0.0, 1.0) * randomness * 50.0
    randomized_y = np.clip(y_filter, 0.0, 230.0) * (1.0 - randomness) + rng.uniform(0.0, 1.0) * randomness * 50.0
    return [float(c_filter), float(randomized_m), float(randomized_y)]


def _build_regeneration_params(stock, paper, illuminant, filters):
    params = create_params(
        film_profile=stock,
        print_profile=paper,
        neutral_print_filters_from_database=False,
    )
    params.enlarger.illuminant = illuminant
    params.enlarger.c_filter_neutral = float(filters[0])
    params.enlarger.m_filter_neutral = float(filters[1])
    params.enlarger.y_filter_neutral = float(filters[2])
    return params


def fit_neutral_print_filter_database(
    config: NeutralPrintFilterRegenerationConfig | None = None,
    neutral_print_filters: NeutralPrintFilterDatabase | None = None,
    residues: NeutralPrintFilterResidueDatabase | None = None,
) -> NeutralPrintFilterRegenerationResult:
    config = config or NeutralPrintFilterRegenerationConfig()
    rng = np.random.default_rng(config.rng_seed)
    working_filters = copy.deepcopy(neutral_print_filters) if neutral_print_filters is not None else _build_neutral_print_filter_database(config.initial_filters)
    working_residues = copy.deepcopy(residues) if residues is not None else _build_neutral_print_filter_residue_database()
    fit_count = 0
    skip_count = 0

    log_event(
        'fit_neutral_print_filter_database_start',
        iterations=config.iterations,
        restart_randomness=config.restart_randomness,
        residue_threshold=config.residue_threshold,
    )
    for paper in PrintPapers:
        log_event('fit_neutral_print_filter_database_paper', paper=paper.value)
        for light in Illuminants:
            log_event(
                'fit_neutral_print_filter_database_illuminant',
                paper=paper.value,
                illuminant=light.value,
            )
            for stock in FilmStocks:
                residue = float(working_residues[paper.value][light.value][stock.value])
                if residue <= config.residue_threshold:
                    skip_count += 1
                    continue

                start_filters = _randomize_start_filters(
                    working_filters[paper.value][light.value][stock.value],
                    config.restart_randomness,
                    rng,
                )
                params = _build_regeneration_params(stock.value, paper.value, light.value, start_filters)
                fitted_y, fitted_m, fit_residues = fit_neutral_print_filters(
                    params,
                    iterations=config.iterations,
                    stock=stock.value,
                    rng=rng,
                )
                working_filters[paper.value][light.value][stock.value] = [
                    float(params.enlarger.c_filter_neutral),
                    float(fitted_m),
                    float(fitted_y),
                ]
                working_residues[paper.value][light.value][stock.value] = float(np.sum(np.abs(fit_residues)))
                fit_count += 1

    log_event(
        'fit_neutral_print_filter_database_complete',
        fit_count=fit_count,
        skip_count=skip_count,
    )
    return NeutralPrintFilterRegenerationResult(filters=working_filters, residues=working_residues)


def regenerate_neutral_print_filters(
    config: NeutralPrintFilterRegenerationConfig | None = None,
) -> NeutralPrintFilterRegenerationResult:
    result = fit_neutral_print_filter_database(config=config)
    save_neutral_print_filters(result.filters)
    log_event(
        'regenerate_neutral_print_filters_complete',
        combinations_saved=len(PrintPapers) * len(Illuminants) * len(FilmStocks),
    )
    return result


__all__ = [
    'NeutralPrintFilterRegenerationConfig',
    'NeutralPrintFilterRegenerationResult',
    'fit_neutral_print_filter_database',
    'fit_neutral_print_filters',
    'regenerate_neutral_print_filters',
]