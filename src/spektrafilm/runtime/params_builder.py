from __future__ import annotations

import copy
from dataclasses import fields
from functools import lru_cache

from spektrafilm.data.presets_loader import read_coupler_presets, read_grain_presets
from spektrafilm.data.profiles_loader import load_profile
from spektrafilm.runtime.params_schema import RuntimePhotoParams
from spektrafilm.utils.io import read_neutral_print_filters

# Heavy fields shared by reference when cloning: the two Profiles carry the
# spectral arrays and density-curve models. They are swapped at selection
# time, never mutated during ordinary parameter edits, so a snapshot can
# share them safely.
_SHARED_PROFILE_FIELDS = ("film", "print")


def clone_runtime_params(params: RuntimePhotoParams) -> RuntimePhotoParams:
    """Cheap snapshot of runtime params for GUI undo / preview.

    Deep-copies the whole parameter subtree so edits to the clone never
    leak into the original, but shares the two ``Profile`` objects
    (``film``, ``print``) by reference. This keeps a snapshot O(params)
    instead of O(spectral data), the same property that made the GUI's
    old ``clone_gui_state`` cheap — without maintaining a mirror state.
    """
    kwargs = {
        f.name: getattr(params, f.name)
        if f.name in _SHARED_PROFILE_FIELDS
        else copy.deepcopy(getattr(params, f.name))
        for f in fields(params)
    }
    return RuntimePhotoParams(**kwargs)


# Fallback filter pack (C, M, Y) for film/print/illuminant combinations that
# aren't in the neutral-print-filters database — keeps any combination printable.
_DEFAULT_NEUTRAL_PRINT_FILTERS = (0.0, 50.0, 50.0)


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
) -> RuntimePhotoParams:
    if not params.settings.neutral_print_filters_from_database:
        return params

    filters = _get_neutral_print_filters() if database is None else database
    stock_filters = (
        filters.get(params.print.info.stock, {})
        .get(params.enlarger.illuminant, {})
        .get(params.film.info.stock)
    )
    if stock_filters is None:
        # Missing entry: fall back to a neutral default (C0 M50 Y50) instead of
        # leaving the params untouched, so any film/print/illuminant combination
        # stays printable even when it isn't in the database.
        c_filter, m_filter, y_filter = _DEFAULT_NEUTRAL_PRINT_FILTERS
    else:
        c_filter, m_filter, y_filter = (float(value) for value in stock_filters)
    params.enlarger.c_filter_neutral = c_filter
    params.enlarger.m_filter_neutral = m_filter
    params.enlarger.y_filter_neutral = y_filter
    return params


def digest_params(
    params: RuntimePhotoParams, apply_stocks_specifics=True
) -> RuntimePhotoParams:
    """Digest the params to prepare for use in the runtime pipeline.
    In the pipeline params should be static and not be changed.
    params.settings and params.debug should contain all the switching logic for the digesting.
    """
    params = apply_database_neutral_print_filters(params)

    if params.settings.preview_mode:
        params.enlarger.lens_blur = 0.0
        params.film_render.dir_couplers.diffusion_size_um = 0.0
        params.film_render.grain.active = False
        params.film_render.grain.rms_granularity = (0.0, 0.0, 0.0)
        params.film_render.grain.blur = 0.0
        params.film_render.grain.mult_usm_amount = 0.0  # USM is a spatial sharpen
        # scatter/halation kernel sigmas are preserved in preview mode
        params.print_render.glare.blur = 0.0
        params.camera.lens_blur_um = 0.0
        params.scanner.lens_blur = 0.0
        params.scanner.unsharp_mask = (0.0, 0.0)

    if apply_stocks_specifics:
        params = _apply_film_specifics(params)
        params = _apply_print_specifics(params)

    # debug switches
    if params.debug.lut_mode:
        params.debug.deactivate_spatial_effects = True
        params.debug.deactivate_stochastic_effects = True
        # exposure control
        params.camera.auto_exposure = False
        params.camera.exposure_compensation_ev = 0.0
        params.enlarger.print_exposure_compensation = False
        params.enlarger.print_exposure = 1.0
        # Highlight boost normalizes by the image-wide max (np.max(x)), so it is
        # an image-global transform — the same input value maps to different
        # outputs depending on the rest of the frame. That cannot be represented
        # by a static 3D LUT (a bake would freeze in the cube grid's max), so it
        # must be off in lut_mode, exactly like auto_exposure above.
        params.film_render.halation.boost_ev = 0.0
        params.scanner.white_correction = False
        params.scanner.black_correction = False
        # Per-image trims: representable in a LUT, but semantically corrections
        # for one specific negative — a LUT is image-agnostic, so they bake
        # neutral. Exposure intent enters only via the LUT creator's own
        # exposure_ev input gain (b60/n010 §9).
        params.enlarger.y_filter_shift = 0.0
        params.enlarger.m_filter_shift = 0.0
        params.enlarger.preflash_exposure = 0.0
        params.enlarger.preflash_y_filter_shift = 0.0
        params.enlarger.preflash_m_filter_shift = 0.0
        # Framing / resampling operate on the incoming image — under LUT
        # sampling that image is the cube grid itself, so either would
        # silently deform the LUT domain.
        params.io.crop = False
        params.io.upscale_factor = 1.0
        # Exact spectral compute: the internal acceleration LUTs exist for
        # interactive speed and would bake their interpolation error into
        # the sampled transform (double interpolation). Bake grids are small,
        # exact compute is affordable (same call as print_balance).
        params.settings.use_enlarger_lut = False
        params.settings.use_scanner_lut = False

    if params.debug.deactivate_spatial_effects:
        # Halation is fully spatial (scatter + back-reflection blurs); kill it
        # at the active flag as well as zeroing the kernel sigmas, so it stays
        # a no-op even if a future sigma-independent term is added inside
        # apply_halation_um.
        params.film_render.halation.active = False
        params.film_render.halation.scatter_core_um = (0.0, 0.0, 0.0)
        params.film_render.halation.scatter_tail_um = (0.0, 0.0, 0.0)
        params.film_render.halation.halation_first_sigma_um = (0.0, 0.0, 0.0)
        params.film_render.dir_couplers.diffusion_size_um = 0
        params.film_render.grain.blur = 0.0
        params.film_render.grain.blur_dye_clouds_um = 0.0
        params.film_render.grain.mult_usm_amount = 0.0  # USM is a spatial sharpen
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
    # Stock-specific overrides live in the preset TOMLs (data/presets/*.toml),
    # applied by the helpers below. Add per-stock film_render tweaks that don't
    # belong in a preset here.

    # presets
    _apply_halation_preset(params)
    _apply_grain_preset(params)
    _apply_couplers_preset(params)

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
    ("still", "strong"): {
        "sigma_h": (65.0, 65.0, 65.0),
        "strength": (0.015, 0.005, 0.0),
    },
    ("still", "weak"): {"sigma_h": (65.0, 65.0, 65.0), "strength": (0.08, 0.02, 0.0)},
    ("still", "no"): {"sigma_h": (65.0, 65.0, 65.0), "strength": (0.30, 0.10, 0.015)},
    ("cine", "strong"): {
        "sigma_h": (50.0, 50.0, 50.0),
        "strength": (0.015, 0.005, 0.0),
    },
    ("cine", "weak"): {"sigma_h": (50.0, 50.0, 50.0), "strength": (0.08, 0.02, 0.0)},
    ("cine", "no"): {"sigma_h": (50.0, 50.0, 50.0), "strength": (0.30, 0.10, 0.015)},
}


def _apply_halation_preset(params: RuntimePhotoParams) -> None:
    """Seed low-level halation parameters from the profile's use/antihalation tags."""
    if not params.film.is_film:
        return
    info = params.film.info
    preset = _HALATION_PRESETS.get((info.use, info.antihalation))
    if preset is None:
        return
    params.film_render.halation.halation_first_sigma_um = preset["sigma_h"]
    params.film_render.halation.halation_strength = preset["strength"]


GRAIN_PRESETS = read_grain_presets()
_GRAIN_DEFAULTS = GRAIN_PRESETS.get("defaults", {})


def _apply_grain_preset(params: RuntimePhotoParams) -> None:
    if params.film.info.stock not in GRAIN_PRESETS:
        return
    preset = GRAIN_PRESETS[params.film.info.stock]

    # Defaults are nested ``defaults.<channel_model>.<polarity>`` (e.g.
    # ``defaults.bw.positive``), so resolve both axes to reach the leaf table of
    # grain parameters. A missing combination yields an empty table, leaving the
    # stock preset (and the schema defaults) to stand on their own.
    channel_model = "bw" if params.film.is_bw else "color"
    polarity = "positive" if params.film.is_positive else "negative"
    defaults = _GRAIN_DEFAULTS.get(channel_model, {}).get(polarity, {})

    # apply each default, overridden by the stock preset where it provides a
    # value, then any extra preset keys that map to a grain field
    for key, value in defaults.items():
        setattr(params.film_render.grain, key, preset.get(key, value))
    for key, value in preset.items():
        if not hasattr(params.film_render.grain, key):
            continue
        setattr(params.film_render.grain, key, value)


COUPLER_PRESETS = read_coupler_presets()
_COUPLER_DEFAULTS = COUPLER_PRESETS.get("defaults", {})


def _apply_couplers_preset(params: RuntimePhotoParams) -> None:
    """Seed DIR-coupler gammas from the couplers.toml presets.

    Mirrors ``_apply_grain_preset``: a ``defaults.<channel_model>.<polarity>``
    table sets the baseline for the film's class (e.g. ``defaults.color.negative``)
    and a per-stock table, if present, overrides individual gammas. A class with
    no defaults table or a stock with no preset keeps the ``DirCouplersParams``
    schema defaults.
    """
    channel_model = "bw" if params.film.is_bw else "color"
    polarity = "positive" if params.film.is_positive else "negative"
    all_defaults = _COUPLER_DEFAULTS.get(channel_model, {}).get(polarity, {})
    defaults = {
        key: value
        for key, value in all_defaults.items()
        if hasattr(params.film_render.dir_couplers, key)
    }
    if channel_model == "color" and polarity == "negative" and params.film.is_cine:
        cine_defaults = all_defaults.get("cine", {})
        reference_illuminant = str(params.film.info.reference_illuminant).upper()
        cine_branch = "tungsten" if reference_illuminant.startswith("T") else "daylight"
        defaults.update(
            {
                key: value
                for key, value in cine_defaults.get(cine_branch, {}).items()
                if hasattr(params.film_render.dir_couplers, key)
            }
        )
    preset = COUPLER_PRESETS.get(params.film.info.stock, {})

    # apply each default, overridden by the stock preset where it provides a
    # value, then any extra preset keys that map to a coupler field
    for key, value in defaults.items():
        setattr(params.film_render.dir_couplers, key, preset.get(key, value))
    for key, value in preset.items():
        if not hasattr(params.film_render.dir_couplers, key):
            continue
        setattr(params.film_render.dir_couplers, key, value)


def _apply_print_specifics(params: RuntimePhotoParams) -> RuntimePhotoParams:
    """Apply print specific settings to the params."""
    # define here all the specifics to stocks that should be applied in params.print_render
    return params


__all__ = [
    "apply_database_neutral_print_filters",
    "clone_runtime_params",
    "digest_params",
    "init_params",
]
