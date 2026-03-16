import copy

import numpy as np

from spectral_film_lab.runtime.process import photo_params
from spectral_film_lab.model.stocks import FilmStocks, PrintPapers
from spectral_film_lab.model.illuminants import Illuminants
from profiles_creator.fitting import fit_print_filters
from spectral_film_lab.utils.io import save_ymc_filter_values


def make_ymc_filters_dictionary():
    ymc_filters_0 = {}
    residues = {}
    for paper in PrintPapers:
        ymc_filters_0[paper.value] = {}
        residues[paper.value] = {}
        for light in Illuminants:
            ymc_filters_0[paper.value][light.value] = {}
            residues[paper.value][light.value] = {}
            for film in FilmStocks:
                ymc_filters_0[paper.value][light.value][film.value] = [0.90, 0.70, 0.35]
                residues[paper.value][light.value][film.value] = 0.184
    ymc_filters = copy.copy(ymc_filters_0)
    save_ymc_filter_values(ymc_filters)
    return ymc_filters, residues


def fit_all_stocks(ymc_filters, residues, iterations=5, randomess_starting_points=0.5):
    ymc_filters_out = copy.deepcopy(ymc_filters)
    r = randomess_starting_points

    for paper in PrintPapers:
        print(" " * 20)
        print("#" * 20)
        print(paper.value)
        for light in Illuminants:
            print("-" * 20)
            print(light.value)
            for stock in FilmStocks:
                if residues[paper.value][light.value][stock.value] > 5e-4:
                    y0 = ymc_filters[paper.value][light.value][stock.value][0]
                    m0 = ymc_filters[paper.value][light.value][stock.value][1]
                    c0 = ymc_filters[paper.value][light.value][stock.value][2]
                    y0 = np.clip(y0, 0, 1) * (1 - r) + np.random.uniform(0, 1) * r
                    m0 = np.clip(m0, 0, 1) * (1 - r) + np.random.uniform(0, 1) * r

                    p = photo_params(
                        negative=stock.value,
                        print_paper=paper.value,
                        ymc_filters_from_database=False,
                    )
                    p.enlarger.illuminant = light.value
                    p.enlarger.y_filter_neutral = y0
                    p.enlarger.m_filter_neutral = m0
                    p.enlarger.c_filter_neutral = c0

                    yf, mf, res = fit_print_filters(p, iterations=iterations)
                    ymc_filters_out[paper.value][light.value][stock.value] = [yf, mf, c0]
                    residues[paper.value][light.value][stock.value] = np.sum(np.abs(res))

    return ymc_filters_out


def main() -> None:
    ymc_filters, residues = make_ymc_filters_dictionary()
    ymc_filters = fit_all_stocks(ymc_filters, residues, iterations=20)
    save_ymc_filter_values(ymc_filters)


if __name__ == "__main__":
    main()



