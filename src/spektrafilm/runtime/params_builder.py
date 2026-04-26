from __future__ import annotations

from functools import lru_cache

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
        # scatter/halation kernel sigmas are preserved in preview mode
        params.print_render.glare.blur = 0.0
        params.camera.lens_blur_um = 0.0
        params.scanner.lens_blur = 0.0
        params.scanner.unsharp_mask = (0.0, 0.0)
    
    if apply_stocks_specifics:
        params = _apply_film_specifics(params)
        params = _apply_print_specifics(params)
    
    # debug switches
    if params.debug.deactivate_spatial_effects:
        params.film_render.halation.scatter_core_um = (0.0, 0.0, 0.0)
        params.film_render.halation.scatter_tail_um = (0.0, 0.0, 0.0)
        params.film_render.halation.halation_first_sigma_um = (0.0, 0.0, 0.0)
        params.film_render.dir_couplers.diffusion_size_um = 0
        params.film_render.grain.blur = 0.0
        params.film_render.grain.blur_dye_clouds_um = 0.0
        params.print_render.glare.blur = 0
        params.camera.lens_blur_um = 0.0
        params.enlarger.lens_blur = 0.0
        params.enlarger.diffusion_filter.active = False
        params.camera.diffusion_filter.active = False
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
        params.film_render.dir_couplers.gamma_samelayer_rgb = (0.12, 0.08, 0.06)
        params.film_render.dir_couplers.gamma_inter_r_to_gb = (0.12, 0.06)
        params.film_render.dir_couplers.gamma_inter_g_to_rb = (0.08, 0.06)
        params.film_render.dir_couplers.gamma_inter_b_to_rg = (0.06, 0.06) # just eyeballed, optimize!
        
    if params.film.is_negative:
        params.film_render.dir_couplers.gamma_samelayer_rgb = (0.336, 0.319, 0.273)
        params.film_render.dir_couplers.gamma_interlayer_r_to_gb = (0.353, 0.302)
        params.film_render.dir_couplers.gamma_interlayer_g_to_rb = (0.154, 0.353)
        params.film_render.dir_couplers.gamma_interlayer_b_to_rg = (0.168, 0.226) # just eyeballed, optimize!

# skin - forest colors - cc blue optimization
# 0.42 alpha 
# [[0.336423, 0.353654, 0.302163],
#  [0.154796, 0.319218, 0.353513],
#  [0.168943, 0.226796, 0.273107]]
# 0.5 alpha high saturation
# [[0.341559, 0.355603, 0.305212],
#  [0.154542, 0.324590, 0.358254],
#  [0.171108, 0.225493, 0.273080]]
# reference matrix cc loss
# [[0.344306, 0.324515, 0.305428],
#  [0.142589, 0.345450, 0.360369],
#  [0.161963, 0.241596, 0.263818]]
# gamma 1.2 sigma 1.55 loss 1.37
#  [[0.53402457 0.43368797 0.23228746]
#  [0.37136105 0.4572779  0.37136105]
#  [0.23228746 0.43368797 0.53402457]]
# gamma 1.0 sigma 1.3
# [[0.48777655 0.36285359 0.14936985]
#  [0.29901809 0.40196381 0.29901809]
#  [0.14936985 0.36285359 0.48777655]]

    _apply_halation_preset(params)

    # stock specifics overrides
    if params.film.info.stock == "fujifilm_velvia_100":
        params.film_render.dir_couplers.inhibition_strength *= 0.9
    if params.film.info.stock == "fujifilm_provia_100f":
        params.film_render.dir_couplers.inhibition_strength *= 1.3
        
        
    # if params.film.info.stock == "kodak_portra_400":
    #     params.film_render.halation.scatter_core_um = (3.5, 2.2, 1.9)
    return params


# Halation low-level presets keyed by (use, antihalation). These set the
# physical baselines from the private halation implementation notes §5-§6.1;
# the user-facing knobs (scatter_amount, scatter_spatial_scale,
# halation_amount, halation_spatial_scale) remain at 1.0 and let the user
# push the effect stronger or weaker without editing the low-level
# parameters.
#
# sigma_h is set by the base material:
#   still -> triacetate, 120-140 um thick -> sigma_h ~= 65 um
#   cine  -> PET,        95-125 um thick -> sigma_h ~= 50 um
# halation_strength is set by the antihalation layer, from §5 ranges:
#   strong -> Vision3, modern colour neg: a1^R ~ 0.005-0.02
#   weak   -> older / mismatched AH:       a1^R ~ 0.02-0.08
#   no     -> rem-jet removed / redscale:  a1^R ~ 0.08-0.25
# Strength values below are a1 midpoints * 1/(1-rho) with rho=0.5 (~2x a1).
_HALATION_PRESETS: dict[tuple[str, str], dict[str, tuple[float, float, float]]] = {
    ('still', 'strong'): {'sigma_h': (65.0, 65.0, 65.0), 'strength': (0.015, 0.005, 0.0)},
    ('still', 'weak'):   {'sigma_h': (65.0, 65.0, 65.0), 'strength': (0.08,  0.02,  0.0)},
    ('still', 'no'):     {'sigma_h': (65.0, 65.0, 65.0), 'strength': (0.30,  0.10,  0.015)},
    ('cine',  'strong'): {'sigma_h': (50.0, 50.0, 50.0), 'strength': (0.015, 0.005, 0.0)},
    ('cine',  'weak'):   {'sigma_h': (50.0, 50.0, 50.0), 'strength': (0.08,  0.02,  0.0)},
    ('cine',  'no'):     {'sigma_h': (50.0, 50.0, 50.0), 'strength': (0.30,  0.10,  0.015)},
}


def _apply_halation_preset(params: RuntimePhotoParams) -> None:
    """Seed low-level halation parameters from the profile's use/antihalation tags."""
    if not params.film.is_film:
        return
    info = params.film.info
    preset = _HALATION_PRESETS.get((info.use, info.antihalation))
    if preset is None:
        return
    params.film_render.halation.halation_first_sigma_um = preset['sigma_h']
    params.film_render.halation.halation_strength = preset['strength']

def _apply_print_specifics(params: RuntimePhotoParams) -> RuntimePhotoParams:
    """Apply print specific settings to the params."""
    # define here all the specifics to stocks that should be applied in params.print_render
    return params


__all__ = [
    "apply_database_neutral_print_filters",
    "digest_params",
    "init_params",
]
