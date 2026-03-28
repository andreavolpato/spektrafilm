from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import scipy

from spektrafilm.model.illuminants import Illuminants
from spektrafilm.model.stocks import FilmStocks, PrintPapers
from spektrafilm.runtime.api import create_params, simulate
from spektrafilm.utils.io import save_ymc_filter_values
from spektrafilm_profile_creator.diagnostics.messages import log_event


MIDGRAY_RGB = np.array([[[0.184, 0.184, 0.184]]], dtype=np.float64)
DEFAULT_NEUTRAL_FILTERS = (0.90, 0.70, 0.35)
DEFAULT_RESIDUE_THRESHOLD = 5e-4

FilterDatabase = dict[str, dict[str, dict[str, list[float]]]]
ResidueDatabase = dict[str, dict[str, dict[str, float]]]


@dataclass(frozen=True, slots=True)
class PrintFilterRegenerationConfig:
    iterations: int = 20
    restart_randomness: float = 0.5
    residue_threshold: float = DEFAULT_RESIDUE_THRESHOLD
    initial_filters: tuple[float, float, float] = DEFAULT_NEUTRAL_FILTERS
    rng_seed: int | None = None

    def __post_init__(self) -> None:
        if self.iterations < 1:
            raise ValueError('iterations must be >= 1')
        if not 0.0 <= self.restart_randomness <= 1.0:
            raise ValueError('restart_randomness must be between 0.0 and 1.0')
        if self.residue_threshold < 0.0:
            raise ValueError('residue_threshold must be >= 0.0')


@dataclass(frozen=True, slots=True)
class PrintFilterRegenerationResult:
    filters: FilterDatabase
    residues: ResidueDatabase


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


def fit_print_filters_iter(profile, start_filters):
    working_profile = _prepare_fitting_profile(profile)
    c_filter = float(working_profile.enlarger.c_filter_neutral)

    def midgray_print(ymc_values, print_exposure):
        working_profile.enlarger.y_filter_neutral = ymc_values[0]
        working_profile.enlarger.m_filter_neutral = ymc_values[1]
        working_profile.enlarger.print_exposure = print_exposure
        return simulate(MIDGRAY_RGB, working_profile)

    def evaluate_residues(values):
        residual = midgray_print([values[0], values[1], c_filter], values[2])
        return (residual - MIDGRAY_RGB).flatten()

    y0, m0 = start_filters
    working_profile.enlarger.y_filter_neutral = y0
    working_profile.enlarger.m_filter_neutral = m0
    x0 = [y0, m0, 1.0]
    fit = scipy.optimize.least_squares(
        evaluate_residues,
        x0,
        bounds=([0, 0, 0], [1, 1, 10]),
        ftol=1e-6,
        xtol=1e-6,
        gtol=1e-6,
        method='trf',
    )
    log_event(
        'fit_print_filters_iter',
        total_residual=float(np.sum(np.abs(evaluate_residues(fit.x)))),
        initial_residual=evaluate_residues(x0),
    )
    return float(fit.x[0]), float(fit.x[1]), evaluate_residues(fit.x)


def fit_print_filters(profile, iterations=10, stock=None, rng=None):
    if stock is None:
        log_event('fit_print_filters')
    else:
        log_event('fit_print_filters', stock=stock)
    if rng is None:
        rng = np.random.default_rng()

    c_filter = float(profile.enlarger.c_filter_neutral)
    current_y = float(profile.enlarger.y_filter_neutral)
    current_m = float(profile.enlarger.m_filter_neutral)
    for index in range(iterations):
        filter_y, filter_m, residues = fit_print_filters_iter(profile, start_filters=(current_y, current_m))
        if np.sum(np.abs(residues)) < 1e-4 or index == iterations - 1:
            log_event(
                'fit_print_filters_result',
                fitted_filters=(filter_y, filter_m, c_filter),
                residual=residues,
            )
            break

        current_y = 0.5 * filter_y + rng.uniform(0.0, 1.0) * 0.5
        current_m = 0.5 * filter_m + rng.uniform(0.0, 1.0) * 0.5
    return filter_y, filter_m, residues


def _build_print_filter_database(initial_filters=DEFAULT_NEUTRAL_FILTERS) -> FilterDatabase:
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


def _build_residue_database(initial_value=float('inf')) -> ResidueDatabase:
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
    y_filter, m_filter, c_filter = filters
    randomized_y = np.clip(y_filter, 0.0, 1.0) * (1.0 - randomness) + rng.uniform(0.0, 1.0) * randomness
    randomized_m = np.clip(m_filter, 0.0, 1.0) * (1.0 - randomness) + rng.uniform(0.0, 1.0) * randomness
    return [float(randomized_y), float(randomized_m), float(c_filter)]


def _build_regeneration_params(stock, paper, illuminant, filters):
    params = create_params(
        film_profile=stock,
        print_profile=paper,
        ymc_filters_from_database=False,
    )
    params.enlarger.illuminant = illuminant
    params.enlarger.y_filter_neutral = float(filters[0])
    params.enlarger.m_filter_neutral = float(filters[1])
    params.enlarger.c_filter_neutral = float(filters[2])
    return params


def fit_print_filter_database(
    config: PrintFilterRegenerationConfig | None = None,
    ymc_filters: FilterDatabase | None = None,
    residues: ResidueDatabase | None = None,
) -> PrintFilterRegenerationResult:
    config = config or PrintFilterRegenerationConfig()
    rng = np.random.default_rng(config.rng_seed)
    working_filters = copy.deepcopy(ymc_filters) if ymc_filters is not None else _build_print_filter_database(config.initial_filters)
    working_residues = copy.deepcopy(residues) if residues is not None else _build_residue_database()
    fit_count = 0
    skip_count = 0

    log_event(
        'fit_print_filter_database_start',
        iterations=config.iterations,
        restart_randomness=config.restart_randomness,
        residue_threshold=config.residue_threshold,
    )
    for paper in PrintPapers:
        log_event('fit_print_filter_database_paper', paper=paper.value)
        for light in Illuminants:
            log_event(
                'fit_print_filter_database_illuminant',
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
                fitted_y, fitted_m, fit_residues = fit_print_filters(
                    params,
                    iterations=config.iterations,
                    stock=stock.value,
                    rng=rng,
                )
                working_filters[paper.value][light.value][stock.value] = [
                    float(fitted_y),
                    float(fitted_m),
                    float(params.enlarger.c_filter_neutral),
                ]
                working_residues[paper.value][light.value][stock.value] = float(np.sum(np.abs(fit_residues)))
                fit_count += 1

    log_event(
        'fit_print_filter_database_complete',
        fit_count=fit_count,
        skip_count=skip_count,
    )
    return PrintFilterRegenerationResult(filters=working_filters, residues=working_residues)


def regenerate_printing_filters(
    config: PrintFilterRegenerationConfig | None = None,
) -> PrintFilterRegenerationResult:
    result = fit_print_filter_database(config=config)
    save_ymc_filter_values(result.filters)
    log_event(
        'regenerate_printing_filters_complete',
        combinations_saved=len(PrintPapers) * len(Illuminants) * len(FilmStocks),
    )
    return result


__all__ = [
    'PrintFilterRegenerationConfig',
    'PrintFilterRegenerationResult',
    'fit_print_filter_database',
    'fit_print_filters',
    'regenerate_printing_filters',
]