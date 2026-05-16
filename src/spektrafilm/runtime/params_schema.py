from __future__ import annotations

from dataclasses import dataclass, field

from spektrafilm.profiles.io import Profile



@dataclass
class DiffusionFilterParams:
    active: bool = False
    # filter_family selects PSF shape and absorption regime. Allowed values
    # are the keys of `_DIFFUSION_FILTER_SHAPES` in spektrafilm.model.diffusion.
    filter_family: str = "black_pro_mist"
    # commercial filter stops: 0, 1/8, 1/4, 1/2, 1, 2 (interpolated in between)
    strength: float = 0.5
    # multiplier on image-plane PSF widths (all per-group lambdas)
    spatial_scale: float = 1.0
    # additive bias to the family's halo warmth axis. The halo is energy-
    # conservingly redistributed across its sub-components per channel:
    # warmth > 0 pushes warm light (R + slight G) toward the OUTER halo
    # and cool light (B) toward the inner halo (and vice versa for
    # warmth < 0). 0 = use family default. Effective warmth is soft-
    # clamped to [-1.5, +1.5].
    halo_warmth: float = 0.0
    # Per-group fine-tune multipliers (advanced). Default 1.0 = use the
    # family preset unchanged. `*_intensity` scales the corresponding
    # group weight (w_c / w_h / w_b); the three weights are then
    # renormalized so they still sum to 1, i.e. the kernel stays
    # unit-normalised and the strength → p_s mapping is unchanged. So
    # these knobs reshuffle the relative split of energy between core,
    # halo and bloom, not the total deflected fraction. `*_size` scales
    # each group's lambda_um uniformly (all sub-components in that group
    # stretched by the same factor).
    core_intensity: float = 1.0
    core_size: float = 1.0
    halo_intensity: float = 1.0
    halo_size: float = 1.0
    bloom_intensity: float = 1.0
    bloom_size: float = 1.0


@dataclass
class CameraParams:
    exposure_compensation_ev: float = 0.0
    auto_exposure: bool = True
    auto_exposure_method: str = "center_weighted"
    lens_blur_um: float = 0.0
    film_format_mm: float = 35.0
    filter_uv: tuple[float, float, float] = (0.0, 410.0, 8.0)
    filter_ir: tuple[float, float, float] = (0.0, 675.0, 15.0)
    diffusion_filter: DiffusionFilterParams = field(default_factory=DiffusionFilterParams)


@dataclass
class EnlargerParams:
    illuminant: str = "TH-KG3"
    print_exposure: float = 1.0
    print_exposure_compensation: bool = True
    normalize_print_exposure: bool = True
    y_filter_shift: float = 0.0
    m_filter_shift: float = 0.0
    y_filter_neutral: float = 55 # kodak cc values
    m_filter_neutral: float = 65 # kodak cc values
    c_filter_neutral: float = 0 # kodak cc values
    lens_blur: float = 0.0
    diffusion_filter: DiffusionFilterParams = field(default_factory=DiffusionFilterParams)
    preflash_exposure: float = 0.0
    preflash_y_filter_shift: float = 0.0
    preflash_m_filter_shift: float = 0.0


@dataclass
class ScannerParams:
    lens_blur: float = 0.0
    white_correction: bool = False
    black_correction: bool = False
    white_level: float = 0.98
    black_level: float = 0.01
    unsharp_mask: tuple[float, float] = (0.7, 0.7)


@dataclass
class GrainParams:
    active: bool = True
    sublayers_active: bool = True
    agx_particle_area_um2: float = 0.2
    agx_particle_scale: tuple[float, float, float] = (0.8, 1.0, 2.0)
    agx_particle_scale_layers: tuple[float, float, float] = (2.5, 1.0, 0.5)
    density_min: tuple[float, float, float] = (0.07, 0.08, 0.12)
    uniformity: tuple[float, float, float] = (0.97, 0.97, 0.99)
    blur: float = 0.65
    blur_dye_clouds_um: float = 1.0
    micro_structure: tuple[float, float] = (0.2, 30)
    n_sub_layers: int = 1


@dataclass
class HalationParams:
    active: bool = True
    # high-level scalars (default 1.0 preserves the physical low-level defaults)
    scatter_amount: float = 1.0
    scatter_spatial_scale: float = 1.0
    halation_amount: float = 1.0
    halation_spatial_scale: float = 1.0
    # in-emulsion scatter — energy-preserving mixture: Gaussian core + exponential
    # tail (scatter_tail_um is the exponential decay constant, internally
    # dispatched to a Gaussian mixture by fast_exponential_filter)
    scatter_core_um: tuple[float, float, float] = (2.2, 2.0, 1.6)
    scatter_tail_um: tuple[float, float, float] = (9.3, 9.7, 9.1)
    scatter_tail_weight: tuple[float, float, float] = (0.78, 0.65, 0.67)
    # highlight boost — reconstructs pre-clip irradiance before propagation
    boost_ev: float = 0.0
    boost_range: float = 0.3
    protect_ev: float = 4.0
    # back-reflection halation — additive sum of N Gaussians with sqrt(k) widths
    halation_strength: tuple[float, float, float] = (0.05, 0.015, 0.0)
    halation_first_sigma_um: tuple[float, float, float] = (65.0, 65.0, 65.0)
    halation_n_bounces: int = 3
    halation_bounce_decay: float = 0.5
    halation_renormalize: bool = True


@dataclass
class DirCouplersParams:
    active: bool = True
    amount: float = 1.0
    inhibition_samelayer: float = 1.0
    inhibition_interlayer: float = 1.0
    gamma_samelayer_rgb: tuple[float, float, float] = (0.341, 0.324, 0.273)
    gamma_interlayer_r_to_gb: tuple[float, float] = (0.355, 0.305)
    gamma_interlayer_g_to_rb: tuple[float, float] = (0.154, 0.358)
    gamma_interlayer_b_to_rg: tuple[float, float] = (0.171, 0.225)
    diffusion_size_um: float = 20.0
    diffusion_tail_um: float = 200.0 # exponential tail for Lévy-like processes or environmental heterogeneity
    diffusion_tail_weight: float = 0.06

@dataclass
class GlareParams:
    active: bool = True
    percent: float = 0.03
    roughness: float = 0.7
    blur: float = 0.5


@dataclass
class FilmRenderingParams:
    density_curve_gamma: float = 1.0
    grain: GrainParams = field(default_factory=GrainParams)
    halation: HalationParams = field(default_factory=HalationParams)
    dir_couplers: DirCouplersParams = field(default_factory=DirCouplersParams)
    glare: GlareParams = field(default_factory=GlareParams)


@dataclass
class PrintRenderingParams:
    density_curve_gamma: float = 1.0
    glare: GlareParams = field(default_factory=GlareParams)


@dataclass
class IOParams:
    input_color_space: str = "ProPhoto RGB"
    input_cctf_decoding: bool = False
    output_color_space: str = "sRGB"
    output_cctf_encoding: bool = True
    output_clip_min: bool = True
    output_clip_max: bool = True
    crop: bool = False
    crop_center: tuple[float, float] = (0.5, 0.5)
    crop_size: tuple[float, float] = (0.1, 0.1)
    upscale_factor: float = 1.0
    scan_film: bool = False


@dataclass
class DebugParams:
    deactivate_spatial_effects: bool = False
    deactivate_stochastic_effects: bool = False
    print_timings: bool = False
    debug_mode: str = 'off' # options: 'output', 'inject', 'off', switch only one of the following at a time
    output_film_log_raw: bool = False
    output_film_density_cmy: bool = False
    output_print_density_cmy: bool = False
    inject_film_density_cmy: bool = False


@dataclass
class SettingsParams:
    rgb_to_raw_method: str = "hanatos2025"
    apply_hanatos2025_adaptation_window: bool = True
    apply_hanatos2025_adaptation_surface: bool = False
    spectral_gaussian_blur: float = 0.0
    use_enlarger_lut: bool = False
    use_scanner_lut: bool = False
    lut_resolution: int = 17
    use_fast_stats: bool = False
    preview_max_size: int = 640
    preview_mode: bool = False
    neutral_print_filters_from_database: bool = True
    

@dataclass
class RuntimePhotoParams:
    film: Profile
    print: Profile
    film_render: FilmRenderingParams = field(default_factory=FilmRenderingParams)
    print_render: PrintRenderingParams = field(default_factory=PrintRenderingParams)
    camera: CameraParams = field(default_factory=CameraParams)
    enlarger: EnlargerParams = field(default_factory=EnlargerParams)
    scanner: ScannerParams = field(default_factory=ScannerParams)
    io: IOParams = field(default_factory=IOParams)
    debug: DebugParams = field(default_factory=DebugParams)
    settings: SettingsParams = field(default_factory=SettingsParams)

    def __post_init__(self):
        if not isinstance(self.film, Profile):
            raise TypeError("film must be a Profile instance")
        if not isinstance(self.print, Profile):
            raise TypeError("print must be a Profile instance")
