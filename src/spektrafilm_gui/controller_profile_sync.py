from __future__ import annotations

from typing import Any, Mapping


PROFILE_SYNC_FIELDS: dict[str, tuple[str, ...]] = {
    'input_image': (
        'upscale_factor',
        'crop',
        'crop_center',
        'crop_size',
        'input_color_space',
        'apply_cctf_decoding',
        'spectral_upsampling_method',
        'filter_uv',
        'filter_ir',
    ),
    'grain': (
        'active',
        'sublayers_active',
        'particle_area_um2',
        'particle_scale',
        'particle_scale_layers',
        'density_min',
        'uniformity',
        'blur',
        'blur_dye_clouds_um',
        'micro_structure',
    ),
    'preflashing': (
        'exposure',
        'y_filter_shift',
        'm_filter_shift',
    ),
    'halation': (
        'active',
        'scatter_amount',
        'scatter_spatial_scale',
        'halation_amount',
        'halation_spatial_scale',
        'boost_ev',
        'protect_ev',
        'boost_range',
        'scatter_core_um',
        'scatter_tail_um',
        'scatter_tail_weight',
        'halation_strength',
        'halation_first_sigma_um',
        'halation_n_bounces',
        'halation_bounce_decay',
        'halation_renormalize',
    ),
    'couplers': (
        'active',
        'dir_couplers_amount',
        'dir_couplers_ratio',
        'dir_couplers_diffusion_um',
        'diffusion_interlayer',
        'high_exposure_shift',
    ),
    'glare': (
        'active',
        'percent',
        'roughness',
        'blur',
    ),
    'special': (
        'film_gamma_factor',
        'print_gamma_factor',
    ),
    'simulation': (
        'film_stock',
        'film_format_mm',
        'camera_lens_blur_um',
        'exposure_compensation_ev',
        'auto_exposure',
        'auto_exposure_method',
        'print_paper',
        'print_illuminant',
        'print_exposure',
        'print_exposure_compensation',
        'print_y_filter_shift',
        'print_m_filter_shift',
        'diffusion_strength',
        'diffusion_spatial_scale',
        'diffusion_intensity',
        'scan_lens_blur',
        'scan_white_correction',
        'scan_white_level',
        'scan_black_correction',
        'scan_black_level',
        'scan_unsharp_mask',
        'scan_film',
    ),
}


def apply_profile_sync_state(
    *,
    widgets: Any,
    synced_state: Any,
    profile_sync_fields: Mapping[str, tuple[str, ...]] | None = None,
) -> None:
    if profile_sync_fields is None:
        profile_sync_fields = PROFILE_SYNC_FIELDS
    for section_name, field_names in profile_sync_fields.items():
        section_widget = getattr(widgets, section_name)
        section_state = getattr(synced_state, section_name)
        for field_name in field_names:
            field_value = getattr(section_state, field_name)
            if section_name == 'simulation' and field_name == 'scan_film':
                section_widget.set_scan_film_value(field_value)
                continue
            getattr(section_widget, field_name).value = field_value