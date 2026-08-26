"""Schema-fingerprint guard for the bake-neutrality contract.

The LUT creator samples the simulator through ``debug.lut_mode``, whose
digest block (``params_builder.digest_params``) must neutralize every
per-image trim and un-bakeable effect — and *only* those (the
neutral-bake contract, spektrafilm-research b60/n010 §9). That block is
hand-written and correct for today's schema; it silently rots the day a
field is added, renamed, or removed in ``params_schema`` without anyone
deciding its bake semantics.

This test pins the flattened field set of ``RuntimePhotoParams``. When
it fails:

1. Decide the new/changed field's bake semantics — one of:
   - *look* (flows into a baked LUT): nothing to do;
   - *per-image trim* or *un-bakeable*: add it to the lut_mode block in
     ``digest_params`` and to ``tests/test_lut_mode.py``;
   - *builder-owned*: set it in ``spektrafilm_lut_creator.bake.bake_params``;
   - *inert under lut_mode*: nothing to do.
2. Update ``EXPECTED_LEAF_PATHS`` below.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass

from spektrafilm.data.profiles_loader import Profile
from spektrafilm.runtime.params_builder import init_params


def leaf_paths(obj, prefix: str = "") -> list[str]:
    """Dotted paths of every leaf field, treating profiles as leaves."""
    paths: list[str] = []
    for field in fields(obj):
        value = getattr(obj, field.name)
        path = prefix + field.name
        if is_dataclass(value) and not isinstance(value, Profile):
            paths += leaf_paths(value, path + ".")
        else:
            paths.append(path)
    return paths


EXPECTED_LEAF_PATHS = [
    "camera.auto_exposure",
    "camera.auto_exposure_method",
    "camera.color_filter",
    "camera.diffusion_filter.active",
    "camera.diffusion_filter.bloom_intensity",
    "camera.diffusion_filter.bloom_size",
    "camera.diffusion_filter.core_intensity",
    "camera.diffusion_filter.core_size",
    "camera.diffusion_filter.filter_family",
    "camera.diffusion_filter.halo_intensity",
    "camera.diffusion_filter.halo_size",
    "camera.diffusion_filter.halo_warmth",
    "camera.diffusion_filter.spatial_scale",
    "camera.diffusion_filter.strength",
    "camera.exposure_compensation_ev",
    "camera.film_format_mm",
    "camera.lens_blur_um",
    "debug.deactivate_spatial_effects",
    "debug.deactivate_stochastic_effects",
    "debug.lut_mode",
    "debug.print_timings",
    "enlarger.c_filter_neutral",
    "enlarger.diffusion_filter.active",
    "enlarger.diffusion_filter.bloom_intensity",
    "enlarger.diffusion_filter.bloom_size",
    "enlarger.diffusion_filter.core_intensity",
    "enlarger.diffusion_filter.core_size",
    "enlarger.diffusion_filter.filter_family",
    "enlarger.diffusion_filter.halo_intensity",
    "enlarger.diffusion_filter.halo_size",
    "enlarger.diffusion_filter.halo_warmth",
    "enlarger.diffusion_filter.spatial_scale",
    "enlarger.diffusion_filter.strength",
    "enlarger.illuminant",
    "enlarger.lens_blur",
    "enlarger.m_filter_neutral",
    "enlarger.m_filter_shift",
    "enlarger.normalize_print_exposure",
    "enlarger.preflash_exposure",
    "enlarger.preflash_m_filter_shift",
    "enlarger.preflash_y_filter_shift",
    "enlarger.print_exposure",
    "enlarger.print_exposure_compensation",
    "enlarger.y_filter_neutral",
    "enlarger.y_filter_shift",
    "film",
    "film_render.base.active",
    "film_render.base.cyan",
    "film_render.base.magenta",
    "film_render.base.scale",
    "film_render.base.tilt",
    "film_render.base.yellow",
    "film_render.chemistry.active",
    "film_render.chemistry.developer_exhaustion",
    "film_render.chemistry.development_time",
    "film_render.chemistry.gamma_factor",
    "film_render.chemistry.gamma_factor_blue",
    "film_render.chemistry.gamma_factor_fast",
    "film_render.chemistry.gamma_factor_green",
    "film_render.chemistry.gamma_factor_red",
    "film_render.chemistry.gamma_factor_slow",
    "film_render.convert.base_percentile",
    "film_render.convert.calibration",
    "film_render.convert.exposure_compensation_ev",
    "film_render.convert.scan_illuminant",
    "film_render.dir_couplers.active",
    "film_render.dir_couplers.amount",
    "film_render.dir_couplers.diffusion_size_um",
    "film_render.dir_couplers.diffusion_tail_um",
    "film_render.dir_couplers.diffusion_tail_weight",
    "film_render.dir_couplers.gamma_interlayer_b_to_rg",
    "film_render.dir_couplers.gamma_interlayer_g_to_rb",
    "film_render.dir_couplers.gamma_interlayer_r_to_gb",
    "film_render.dir_couplers.gamma_samelayer_rgb",
    "film_render.dir_couplers.inhibition_interlayer",
    "film_render.dir_couplers.inhibition_samelayer",
    "film_render.dir_couplers.langmuir_donor_k_rgb",
    "film_render.dir_couplers.langmuir_receiver_k_rgb",
    "film_render.glare.active",
    "film_render.glare.blur",
    "film_render.glare.percent",
    "film_render.glare.roughness",
    "film_render.grain.active",
    "film_render.grain.blur",
    "film_render.grain.blur_dye_clouds_um",
    "film_render.grain.density_min",
    "film_render.grain.micro_structure",
    "film_render.grain.micro_sublayers",
    "film_render.grain.mult_usm_amount",
    "film_render.grain.mult_usm_sigma",
    "film_render.grain.particle_scale_sublayers",
    "film_render.grain.rms_granularity",
    "film_render.grain.uniformity",
    "film_render.halation.active",
    "film_render.halation.boost_ev",
    "film_render.halation.boost_range",
    "film_render.halation.halation_amount",
    "film_render.halation.halation_bounce_decay",
    "film_render.halation.halation_first_sigma_um",
    "film_render.halation.halation_n_bounces",
    "film_render.halation.halation_renormalize",
    "film_render.halation.halation_spatial_scale",
    "film_render.halation.halation_strength",
    "film_render.halation.protect_ev",
    "film_render.halation.scatter_amount",
    "film_render.halation.scatter_core_um",
    "film_render.halation.scatter_spatial_scale",
    "film_render.halation.scatter_tail_um",
    "film_render.halation.scatter_tail_weight",
    "io.crop",
    "io.crop_center",
    "io.crop_size",
    "io.input_cctf_decoding",
    "io.input_color_space",
    "io.input_gamut_compress.active",
    "io.input_gamut_compress.algorithm",
    "io.input_gamut_compress.boundary",
    "io.input_gamut_compress.hull_detail",
    "io.input_gamut_compress.knee",
    "io.output_cctf_encoding",
    "io.output_color_space",
    "io.output_gamut_compress.algorithm",
    "io.output_gamut_compress.knee",
    "io.output_gamut_compress.lightness_compression",
    "io.upscale_factor",
    "print",
    "print_render.base.active",
    "print_render.base.cyan",
    "print_render.base.magenta",
    "print_render.base.scale",
    "print_render.base.yellow",
    "print_render.chemistry.active",
    "print_render.chemistry.developer_exhaustion",
    "print_render.chemistry.development_time",
    "print_render.chemistry.gamma_factor",
    "print_render.chemistry.gamma_factor_blue",
    "print_render.chemistry.gamma_factor_fast",
    "print_render.chemistry.gamma_factor_green",
    "print_render.chemistry.gamma_factor_red",
    "print_render.chemistry.gamma_factor_slow",
    "print_render.glare.active",
    "print_render.glare.blur",
    "print_render.glare.percent",
    "print_render.glare.roughness",
    "scanner.black_correction",
    "scanner.black_level",
    "scanner.lens_blur",
    "scanner.unsharp_mask",
    "scanner.white_correction",
    "scanner.white_level",
    "settings.apply_hanatos2025_adaptation_surface",
    "settings.apply_hanatos2025_adaptation_window",
    "settings.convert_lut_resolution",
    "settings.lut_resolution",
    "settings.neutral_print_filters_from_database",
    "settings.preview_max_size",
    "settings.preview_mode",
    "settings.rgb_to_raw_method",
    "settings.spectral_gaussian_blur",
    "settings.use_convert_lut",
    "settings.use_enlarger_lut",
    "settings.use_fast_stats",
    "settings.use_scanner_lut",
    "taps.collect",
    "taps.inject",
    "workflow.route",
]


def test_params_schema_fingerprint_matches_bake_contract():
    actual = sorted(leaf_paths(init_params()))
    added = sorted(set(actual) - set(EXPECTED_LEAF_PATHS))
    removed = sorted(set(EXPECTED_LEAF_PATHS) - set(actual))
    assert actual == EXPECTED_LEAF_PATHS, (
        "RuntimePhotoParams schema changed"
        + (f" — added: {added}" if added else "")
        + (f" — removed: {removed}" if removed else "")
        + ". Decide the bake semantics of each changed field (see this "
        "test's docstring and the lut_mode block in "
        "spektrafilm/runtime/params_builder.py), then update "
        "EXPECTED_LEAF_PATHS."
    )
