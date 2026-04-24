from __future__ import annotations

from functools import lru_cache
import numpy as np

from spektrafilm.profiles.io import load_profile
from spektrafilm.runtime.params_schema import RuntimePhotoParams
from spektrafilm.utils.io import read_neutral_print_filters


@lru_cache(maxsize=1)
def _get_neutral_print_filters():
    try:
        return read_neutral_print_filters()
    except FileNotFoundError:
        return {}


def apply_database_neutral_print_filters(
    params: RuntimePhotoParams,
    *,
    database=None,
    warn_missing: bool = True,
) -> RuntimePhotoParams:
    if not params.settings.neutral_print_filters_from_database:
        return params

    filters = _get_neutral_print_filters() if database is None else database
    stock_filters = (
        filters
        .get(params.print.info.stock, {})
        .get(params.enlarger.illuminant, {})
        .get(params.film.info.stock)
    )
    if stock_filters is not None:
        c_filter, m_filter, y_filter = (float(value) for value in stock_filters)
        params.enlarger.c_filter_neutral = c_filter
        params.enlarger.m_filter_neutral = m_filter
        params.enlarger.y_filter_neutral = y_filter
    elif warn_missing:
        print(
            f"Warning: No neutral print filters found in database for print stock {params.print.info.stock} "
            f"with illuminant {params.enlarger.illuminant} and film stock {params.film.info.stock}. "
            "Using defaults."
        )
    return params


def digest_params(params: RuntimePhotoParams, apply_stocks_specifics=True) -> RuntimePhotoParams:
    """Digest the params to prepare for use in the runtime pipeline.
    In the pipeline params should be static and not be changed.
    params.settings and params.debug should contain all the switching logic for the digesting.
    """
    params = apply_database_neutral_print_filters(params)

    if params.settings.preview_mode:
        params.enlarger.lens_blur = 0.0
        params.film_render.dir_couplers.diffusion_size_um = 0.0
        params.film_render.grain.active = False
        params.film_render.grain.agx_particle_area_um2 = 0.0
        params.film_render.grain.blur = 0.0
        # params.film_render.halation.scattering_size_um = [0.0, 0.0, 0.0]
        params.print_render.glare.blur = 0.0
        params.camera.lens_blur_um = 0.0
        params.scanner.lens_blur = 0.0
        params.scanner.unsharp_mask = (0.0, 0.0)
    
    if apply_stocks_specifics:
        params = _apply_film_specifics(params)
        params = _apply_print_specifics(params)
    
    # debug switches
    if params.debug.deactivate_spatial_effects:
        params.film_render.halation.size_um = [0, 0, 0]
        params.film_render.halation.scattering_size_um = [0, 0, 0]
        params.film_render.dir_couplers.diffusion_size_um = 0
        params.film_render.grain.blur = 0.0
        params.film_render.grain.blur_dye_clouds_um = 0.0
        params.print_render.glare.blur = 0
        params.camera.lens_blur_um = 0.0
        params.enlarger.lens_blur = 0.0
        params.enlarger.diffusion_filter = (0.0, 1.0, 1.0)
        params.scanner.lens_blur = 0.0
        params.scanner.unsharp_mask = (0.0, 0.0)

    if params.debug.deactivate_stochastic_effects:
        params.film_render.grain.active = False
        params.print_render.glare.active = False
        
    return params


def init_params(
    film_profile: str = "kodak_portra_400",
    print_profile: str = "kodak_portra_endura",
) -> RuntimePhotoParams:
    """Simple helper to build a RuntimePhotoParams with just film and print profiles specified.
    Build a runtime parameter object.
    It needs to be digested with digest_params before being used in the runtime pipeline."""

    params = RuntimePhotoParams(
        film=load_profile(film_profile),
        print=load_profile(print_profile),
    )
    return params

def _apply_film_specifics(params: RuntimePhotoParams) -> RuntimePhotoParams:
    """Apply film specific settings to the params."""
    # film overrides
    # define here all the specifics to stocks that should be applied in params.film_render
    if params.film.is_positive:
        params.film_render.dir_couplers.ratio_rgb = (0.38, 0.26, 0.17)
        
    if params.film.is_negative:
        params.film_render.dir_couplers.ratio_rgb = (0.42, 0.42, 0.42)

    # stock specifics overrides
    if params.film.info.stock == "fujifilm_velvia_100":
        params.film_render.dir_couplers.ratio_rgb *= np.ones(3) * 0.9
    if params.film.info.stock == "fujifilm_provia_100f":
        params.film_render.dir_couplers.ratio_rgb *= np.ones(3) * 1.3
    return params

def _apply_print_specifics(params: RuntimePhotoParams) -> RuntimePhotoParams:
    """Apply print specific settings to the params."""
    # define here all the specifics to stocks that should be applied in params.print_render
    return params


__all__ = [
    "apply_database_neutral_print_filters",
    "digest_params",
    "init_params",
]
