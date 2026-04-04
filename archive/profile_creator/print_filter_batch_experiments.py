import copy

import numpy as np

from spektrafilm.runtime.api import create_params
from spektrafilm_profile_creator.neutral_print_filters import fit_neutral_print_filters


def fit_all_stocks(iterations=5, randomess_starting_points=0.5):
    from spektrafilm.model.illuminants import Illuminants
    from spektrafilm.model.stocks import FilmStocks, PrintPapers

    neutral_print_filters_start = {}
    residues = {}
    for paper in PrintPapers:
        neutral_print_filters_start[paper.value] = {}
        residues[paper.value] = {}
        for light in Illuminants:
            neutral_print_filters_start[paper.value][light.value] = {}
            residues[paper.value][light.value] = {}
            for film in FilmStocks:
                neutral_print_filters_start[paper.value][light.value][film.value] = [0.90, 0.70, 0.35]
                residues[paper.value][light.value][film.value] = 0.184

    neutral_print_filters_result = copy.deepcopy(neutral_print_filters_start)
    randomness = randomess_starting_points

    for paper in PrintPapers:
        print(' ' * 20)
        print('#' * 20)
        print(paper.value)
        for light in Illuminants:
            print('-' * 20)
            print(light.value)
            for stock in FilmStocks:
                if residues[paper.value][light.value][stock.value] > 5e-4:
                    y0, m0, c0 = neutral_print_filters_start[paper.value][light.value][stock.value]
                    y0 = np.clip(y0, 0, 1) * (1 - randomness) + np.random.uniform(0, 1) * randomness
                    m0 = np.clip(m0, 0, 1) * (1 - randomness) + np.random.uniform(0, 1) * randomness

                    params = create_params(
                        film_profile=stock.value,
                        print_profile=paper.value,
                        neutral_print_filters_from_database=False,
                    )
                    params.enlarger.illuminant = light.value
                    params.enlarger.y_filter_neutral = y0
                    params.enlarger.m_filter_neutral = m0
                    params.enlarger.c_filter_neutral = c0

                    y_filter, m_filter, residuals = fit_neutral_print_filters(params, iterations=iterations)
                    neutral_print_filters_result[paper.value][light.value][stock.value] = [y_filter, m_filter, c0]
                    residues[paper.value][light.value][stock.value] = np.sum(np.abs(residuals))

    return neutral_print_filters_result, residues
