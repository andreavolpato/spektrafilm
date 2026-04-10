from __future__ import annotations

from functools import lru_cache
import numpy as np

from spektrafilm.profiles.io import load_profile
from spektrafilm.runtime.params_schema import RuntimePhotoParams
from spektrafilm.utils.io import read_neutral_print_filters


@lru_cache(maxsize=1)
def _get_neutral_print_filters():
    return read_neutral_print_filters()


def digest_params(params: RuntimePhotoParams) -> RuntimePhotoParams:
    """Digest the params to prepare for use in the runtime pipeline.
    In the pipeline params should be static and not be changed.
    params.settings and params.debug should contain all the switching logic for the digesting.
    """

    # read neutral print filters from database
    if params.settings.neutral_print_filters_from_database:
        filters = _get_neutral_print_filters()
        c_filter, m_filter, y_filter = filters[params.print.info.stock][params.enlarger.illuminant][params.film.info.stock]
        params.enlarger.c_filter_neutral = c_filter
        params.enlarger.m_filter_neutral = m_filter
        params.enlarger.y_filter_neutral = y_filter
        
    if params.settings.preview_mode:
        params.enlarger.lens_blur = 0.0
        params.film_render.dir_couplers.diffusion_size_um = 0.0
        params.film_render.grain.active = False
        params.film_render.grain.agx_particle_area_um2 = 0.0
        params.film_render.grain.blur = 0.0
        params.film_render.halation.size_um = [0.0, 0.0, 0.0]
        params.film_render.halation.scattering_size_um = [0.0, 0.0, 0.0]
        params.print_render.glare.blur = 0.0
        params.camera.lens_blur_um = 0.0
        params.scanner.lens_blur = 0.0
        params.scanner.unsharp_mask = (0.0, 0.0)

    params = _apply_film_specifics(params)
    
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



__all__ = ["digest_params", "init_params"]
