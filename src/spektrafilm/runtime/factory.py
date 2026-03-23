from __future__ import annotations

from functools import lru_cache

from spektrafilm.profiles.io import load_profile
from spektrafilm.runtime.params_schema import RuntimePhotoParams
from spektrafilm.utils.io import read_neutral_ymc_filter_values

@lru_cache(maxsize=1)
def _get_ymc_filters():
    return read_neutral_ymc_filter_values()


def build_runtime_params(
    film_profile: str = "kodak_portra_400_auc",
    print_profile: str = "kodak_portra_endura_uc",
    ymc_filters_from_database: bool = True,
) -> RuntimePhotoParams:
    params = RuntimePhotoParams(
        film=load_profile(film_profile),
        print=load_profile(print_profile),
    )

    if ymc_filters_from_database:
        filters = _get_ymc_filters()
        y_filter, m_filter, c_filter = filters[print_profile][params.enlarger.illuminant][film_profile]
        params.enlarger.y_filter_neutral = y_filter
        params.enlarger.m_filter_neutral = m_filter
        params.enlarger.c_filter_neutral = c_filter

    return params
