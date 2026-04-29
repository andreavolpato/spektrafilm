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
DEFAULT_INITIAL_PRINT_EXPOSURE = 2.0
DEFAULT_RESIDUE_THRESHOLD = 5e-4

_NeutralPrintFilterDatabase = dict[str, dict[str, dict[str, list[float]]]]
_ResidueDatabase = dict[str, dict[str, dict[str, float]]]


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
class NeutralPrintFilterFitResult:
    c_filter: float
    m_filter: float
    y_filter: float
    print_exposure: float
    residual: np.ndarray

    @property
    def filters(self) -> tuple[float, float, float]:
        return (self.c_filter, self.m_filter, self.y_filter)

    @property
    def total_residual(self) -> float:
        return float(np.sum(np.abs(self.residual)))


# ── Public API ─────────────────────────────────────────────────────────────────

def fit_neutral_filters(
    profile,
    iterations=10,
    rng=None,
    *,
    film_density_cmy=None,
    normalize_print_exposure=True,
):
    if iterations < 1:
        raise ValueError('iterations must be >= 1')

    if film_density_cmy is None:
        fit_input = MIDGRAY_RGB
    else:
        fit_input = np.asarray(film_density_cmy, dtype=np.float64)
        if fit_input.size != 3:
            raise ValueError('film_density_cmy must contain exactly three CMY density values')
        fit_input = fit_input.reshape((1, 1, 3))
    injected_film_density_cmy = film_density_cmy is not None
    fit_mode = 'inject_film_density_cmy' if injected_film_density_cmy else 'midgray_rgb'

    log_event(
        'fit_neutral_filters',
        stock=profile.film.info.stock,
        fit_mode=fit_mode,
        normalize_print_exposure=normalize_print_exposure,
    )

    if profile.film.info.is_positive and profile.print.info.is_negative:
        return NeutralPrintFilterFitResult(
            c_filter=float(profile.enlarger.c_filter_neutral),
            m_filter=float(profile.enlarger.m_filter_neutral),
            y_filter=float(profile.enlarger.y_filter_neutral),
            print_exposure=float(profile.enlarger.print_exposure),
            residual=np.zeros(3, dtype=np.float64),
        )

    if rng is None:
        rng = np.random.default_rng()

    start_filters = (
        float(DEFAULT_NEUTRAL_PRINT_FILTERS[0]),
        float(profile.enlarger.m_filter_neutral),
        float(profile.enlarger.y_filter_neutral),
    )
    fit_result = _fit_once(
        profile,
        start_filters=start_filters,
        fit_input=fit_input,
        injected_film_density_cmy=injected_film_density_cmy,
        normalize_print_exposure=normalize_print_exposure,
    )
    for _ in range(1, iterations):
        if fit_result.total_residual < DEFAULT_RESIDUE_THRESHOLD:
            break
        start_filters = (
            float(fit_result.c_filter),
            0.5 * float(fit_result.m_filter) + float(rng.uniform(0.0, 1.0)) * 50.0,
            0.5 * float(fit_result.y_filter) + float(rng.uniform(0.0, 1.0)) * 50.0,
        )
        fit_result = _fit_once(
            profile,
            start_filters=start_filters,
            fit_input=fit_input,
            injected_film_density_cmy=injected_film_density_cmy,
            normalize_print_exposure=normalize_print_exposure,
        )

    log_event(
        'fit_neutral_filters_result',
        fitted_filters=fit_result.filters,
        print_exposure=fit_result.print_exposure,
        residual=fit_result.residual,
        fit_mode=fit_mode,
    )
    return fit_result


def fit_neutral_filter_entry(
    *,
    stock: str,
    paper: str,
    illuminant: str = Illuminants.lamp.value,
    config: NeutralPrintFilterRegenerationConfig | None = None,
    neutral_print_filters: _NeutralPrintFilterDatabase | None = None,
    residues: _ResidueDatabase | None = None,
) -> tuple[_NeutralPrintFilterDatabase, _ResidueDatabase]:
    config = config or NeutralPrintFilterRegenerationConfig()
    rng = np.random.default_rng(config.rng_seed)
    if neutral_print_filters is not None:
        working_filters = copy.deepcopy(neutral_print_filters)
    else:
        try:
            working_filters = read_neutral_print_filters()
        except FileNotFoundError:
            working_filters = {}
    working_residues = copy.deepcopy(residues) if residues is not None else {}

    _fit_database_entry(
        stock=stock,
        paper=paper,
        illuminant=illuminant,
        config=config,
        working_filters=working_filters,
        working_residues=working_residues,
        rng=rng,
    )
    return working_filters, working_residues


def regenerate_neutral_filter_entry(
    *,
    stock: str,
    paper: str,
    illuminant: str = Illuminants.lamp.value,
    config: NeutralPrintFilterRegenerationConfig | None = None,
    neutral_print_filters: _NeutralPrintFilterDatabase | None = None,
    residues: _ResidueDatabase | None = None,
) -> tuple[tuple[float, float, float], float]:
    working_filters, working_residues = fit_neutral_filter_entry(
        stock=stock,
        paper=paper,
        illuminant=illuminant,
        config=config,
        neutral_print_filters=neutral_print_filters,
        residues=residues,
    )
    save_neutral_print_filters(working_filters)

    fitted_filters = (
        working_filters.get(paper, {})
        .get(illuminant, {})
        .get(stock)
    )
    if fitted_filters is None:
        raise RuntimeError('Failed to load fitted neutral print filter entry after regeneration.')

    fitted_filters = tuple(float(value) for value in fitted_filters)
    fitted_residue = float(working_residues[paper][illuminant][stock])
    log_event(
        'regenerate_neutral_filter_entry_complete',
        stock=stock,
        paper=paper,
        illuminant=illuminant,
        fitted_filters=fitted_filters,
        residue=fitted_residue,
    )
    return fitted_filters, fitted_residue


def fit_neutral_filter_database(
    config: NeutralPrintFilterRegenerationConfig | None = None,
    neutral_print_filters: _NeutralPrintFilterDatabase | None = None,
    residues: _ResidueDatabase | None = None,
) -> tuple[_NeutralPrintFilterDatabase, _ResidueDatabase]:
    config = config or NeutralPrintFilterRegenerationConfig()
    rng = np.random.default_rng(config.rng_seed)
    working_filters = (
        copy.deepcopy(neutral_print_filters)
        if neutral_print_filters is not None
        else {
            paper.value: {
                light.value: {
                    film.value: [
                        float(config.initial_filters[0]),
                        float(config.initial_filters[1]),
                        float(config.initial_filters[2]),
                    ]
                    for film in FilmStocks
                }
                for light in Illuminants
            }
            for paper in PrintPapers
        }
    )
    working_residues = (
        copy.deepcopy(residues)
        if residues is not None
        else {
            paper.value: {
                light.value: {
                    film.value: float('inf')
                    for film in FilmStocks
                }
                for light in Illuminants
            }
            for paper in PrintPapers
        }
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
                did_fit = _fit_database_entry(
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
) -> tuple[_NeutralPrintFilterDatabase, _ResidueDatabase]:
    filters, residues = fit_neutral_filter_database(config=config)
    save_neutral_print_filters(filters)
    log_event(
        'regenerate_neutral_filter_database_complete',
        combinations_saved=len(PrintPapers) * len(Illuminants) * len(FilmStocks),
    )
    return filters, residues


# ── Private helpers ────────────────────────────────────────────────────────────


def _set_neutral_filters(params, *, c_filter: float, m_filter: float, y_filter: float) -> None:
    params.enlarger.c_filter_neutral = float(c_filter)
    params.enlarger.m_filter_neutral = float(m_filter)
    params.enlarger.y_filter_neutral = float(y_filter)


def _prepare_profile_for_fitting(
    params,
    *,
    injected_film_density_cmy: bool,
    normalize_print_exposure: bool,
):
    params.debug.deactivate_spatial_effects = True
    params.debug.deactivate_stochastic_effects = True
    params.debug.debug_mode = 'off'
    params.debug.inject_film_density_cmy = False
    params.settings.neutral_print_filters_from_database = False
    params.io.input_cctf_decoding = False
    params.io.input_color_space = 'sRGB'
    params.io.output_cctf_encoding = False
    params.io.upscale_factor = 1.0
    params.camera.auto_exposure = False
    params.enlarger.print_exposure_compensation = False
    params.enlarger.normalize_print_exposure = bool(normalize_print_exposure)
    params.print_render.glare.active = False
    if injected_film_density_cmy:
        params.debug.debug_mode = 'inject'
        params.debug.inject_film_density_cmy = True
    return digest_params(params)


def _fit_residues(working_profile, fit_input, c_filter: float, values) -> np.ndarray:
    m_filter, y_filter, print_exposure = values
    _set_neutral_filters(working_profile, c_filter=c_filter, m_filter=m_filter, y_filter=y_filter)
    working_profile.enlarger.print_exposure = float(print_exposure)
    rendered_midgray = simulate(fit_input, working_profile, digest_params_first=False)
    return (rendered_midgray - MIDGRAY_RGB).reshape(-1)


def _fit_once(
    profile,
    start_filters,
    *,
    fit_input,
    injected_film_density_cmy: bool,
    normalize_print_exposure: bool,
):
    working_profile = _prepare_profile_for_fitting(
        copy.deepcopy(profile),
        injected_film_density_cmy=injected_film_density_cmy,
        normalize_print_exposure=normalize_print_exposure,
    )
    c_filter, m_filter, y_filter = (float(v) for v in start_filters)
    x0 = np.array([m_filter, y_filter, DEFAULT_INITIAL_PRINT_EXPOSURE], dtype=np.float64)
    evaluate_residues = functools.partial(_fit_residues, working_profile, fit_input, c_filter)
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
    fit_result = NeutralPrintFilterFitResult(
        c_filter=c_filter,
        m_filter=float(fit.x[0]),
        y_filter=float(fit.x[1]),
        print_exposure=float(fit.x[2]),
        residual=np.asarray(fit.fun, dtype=np.float64),
    )
    return fit_result


def _fit_database_entry(
    *,
    stock: str,
    paper: str,
    illuminant: str,
    config: NeutralPrintFilterRegenerationConfig,
    working_filters: _NeutralPrintFilterDatabase,
    working_residues: _ResidueDatabase,
    rng,
) -> bool:
    existing_filters = (
        working_filters
        .setdefault(paper, {})
        .setdefault(illuminant, {})
        .setdefault(
            stock,
            [
                float(config.initial_filters[0]),
                float(config.initial_filters[1]),
                float(config.initial_filters[2]),
            ],
        )
    )
    residue_by_stock = working_residues.setdefault(paper, {}).setdefault(illuminant, {})
    residue_by_stock.setdefault(stock, float('inf'))

    if float(residue_by_stock[stock]) <= config.residue_threshold:
        return False

    _, m_filter, y_filter = (float(value) for value in existing_filters)
    start_filters = [
        float(config.initial_filters[0]),
        float(np.clip(m_filter, 0.0, 230.0) * (1.0 - config.restart_randomness) + rng.uniform(0.0, 1.0) * config.restart_randomness * 50.0),
        float(np.clip(y_filter, 0.0, 230.0) * (1.0 - config.restart_randomness) + rng.uniform(0.0, 1.0) * config.restart_randomness * 50.0),
    ]
    params = init_params(film_profile=stock, print_profile=paper)
    params.enlarger.illuminant = illuminant
    _set_neutral_filters(
        params,
        c_filter=start_filters[0],
        m_filter=start_filters[1],
        y_filter=start_filters[2],
    )
    if params.film.info.is_positive and params.print.info.is_negative:
        residue_by_stock[stock] = 0.0
        return False

    fit_result = fit_neutral_filters(
        params,
        iterations=config.iterations,
        rng=rng,
    )
    working_filters[paper][illuminant][stock] = [
        float(fit_result.c_filter),
        float(fit_result.m_filter),
        float(fit_result.y_filter),
    ]
    residue_by_stock[stock] = fit_result.total_residual
    return True


__all__ = [
    'NeutralPrintFilterFitResult',
    'NeutralPrintFilterRegenerationConfig',
    'fit_neutral_filter_database',
    'fit_neutral_filter_entry',
    'fit_neutral_filters',
    'regenerate_neutral_filter_entry',
    'regenerate_neutral_filter_database',
]
