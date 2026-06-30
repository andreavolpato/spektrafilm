from __future__ import annotations

from dataclasses import dataclass, field

from spektrafilm.data.profiles_loader import Profile
from spektrafilm.utils.gamut_compression import (
    InputGamutCompressSpec,
    OutputGamutCompressSpec,
)
from spektrafilm.utils.morph_curves import FilmChemistryParams, PrintChemistryParams



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
    # color_filter selects a camera taking filter from the color filter library.
    # Allowed values are the members of `CameraColorFilters` in
    # spektrafilm.model.color_filters ("none" = no filter).
    color_filter: str = "none"
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
    # pixel statistics
    rms_granularity: tuple[float, float, float] = (6, 8, 10)
    density_min: tuple[float, float, float] = (0.03, 0.03, 0.03)
    uniformity: tuple[float, float, float] = (0.97, 0.97, 0.97)
    particle_scale_sublayers: tuple[float, float, float] = (1.0, 0.5, 0.25)
    # texture
    blur: float = 0.89 # optimized to go with the mult usm below (recovers resolution), see study b80
    mult_usm_sigma: float = 0.7 # optimized to go with the blur above, see study b80
    mult_usm_amount: float = 1.5 # Multiplicative (log-domain) density unsharp mask, see study b80
    # micro substructure
    blur_dye_clouds_um: float = 2.0 # somewhat resolution of a normal microscope
    micro_structure: tuple[float, float] = (0.2, 30)
    micro_sublayers: int = 1


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
    langmuir_donor_k_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0)
    langmuir_receiver_k_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0)
    diffusion_size_um: float = 20.0
    diffusion_tail_um: float = 200.0 # exponential tail for Lévy-like processes or environmental heterogeneity
    diffusion_tail_weight: float = 0.03

@dataclass
class GlareParams:
    active: bool = True
    percent: float = 0.03
    roughness: float = 0.7
    blur: float = 0.5


@dataclass
class FilmBaseParams:
    # Film base + fog / orange mask tuning. cyan/magenta/yellow are additive
    # per-channel density shifts (value - 1) around the status-M peaks; tilt is
    # a spectral tilt of the base. All neutral (cmy=1.0, tilt=0.0) is identity.
    active: bool = True
    scale: float = 1.0
    tilt: float = 0.0
    cyan: float = 1.0
    magenta: float = 1.0
    yellow: float = 1.0


@dataclass
class PrintBaseParams:
    # Print base tuning. cyan/magenta/yellow are multiplicative scales of the
    # per-channel base mins at the status-A peaks. All neutral (cmy=1.0) is
    # identity. No tilt (film-base only).
    active: bool = True
    scale: float = 1.0
    cyan: float = 1.0
    magenta: float = 1.0
    yellow: float = 1.0


@dataclass
class ConvertFilmParams:
    # Settings for the "convert-film" stage of the invert workflow: convert a
    # scanned negative (scene-referred input RGB, in io.input_color_space) back to
    # film cmy density by inverting the scan model, so it can be injected at
    # cmy_film and printed. Activation is governed by workflow.route (the
    # "convert-film" routes), not a flag here. The film base / orange mask is
    # tuned separately in FilmRenderingParams.base. See the c40_film_inversion
    # study and its n060 (scan-illuminant design) note.
    #
    # scan_illuminant: the capture rig's light source — any name accepted by
    # spektrafilm.model.illuminants.standard_illuminant() (CIE A/D/E, FL, the CIE
    # LED-B/LED-RGB/LED-V series, measured sources, or 'BB<temp>'). It is a
    # PHYSICAL input (set it to match the rig), not a creative knob; the base is
    # the creative lever. The scan only sees illuminant(λ)·10^(-base(λ)), so they
    # share one spectral axis (n060).
    scan_illuminant: str = "D55"
    # exposure_compensation_ev: aligns the scan's absolute level to the model's
    # (the normalized-illuminant model has a fixed output scale; a real scan's
    # level is set by scanner gain). A flat RGB gain -> near-uniform density
    # offset; distinct from base, which reshapes spectrally per channel.
    exposure_compensation_ev: float = 0.0
    # base_percentile: the brightest pixels of a scanned negative are the clear /
    # unexposed film (max transmission). The "Detect base" GUI action samples the
    # mean RGB of pixels at/above this percentile and fits the film base / orange
    # mask (FilmRenderingParams.base) so the model's cmy=0 scan reproduces it, i.e.
    # the unexposed film maps to film density 0. 99.0 -> brightest 1%.
    base_percentile: float = 99.0
    # calibration: a 3x3 device-correction matrix (row-major, 9 numbers) applied
    # to the input RGB before inversion, to undo the scanner/camera colour
    # rendering vs the standard observer over the film dyes (the cross-channel
    # cast the scan_illuminant + base cannot reach — see c40 s050/s051). Stored as
    # a string so it is copy/paste-able and hand-editable; parsed by
    # spektrafilm.model.convert.parse_calibration_matrix (identity on any parse
    # failure). The "Blind calibration" GUI action fits it on the current image.
    # TEMPORARY test UX; a proper calibration flow comes later.
    calibration: str = "1 0 0  0 1 0  0 0 1"


@dataclass
class FilmRenderingParams:
    # Film chemistry: development_time selects which curve / base+fog column of a
    # BW development-time family to render (matched against
    # ProfileData.development_time); the remaining fields are the s023 morph,
    # shared with the print chemistry. Off by default for single-curve/color.
    chemistry: FilmChemistryParams = field(default_factory=FilmChemistryParams)
    grain: GrainParams = field(default_factory=GrainParams)
    halation: HalationParams = field(default_factory=HalationParams)
    dir_couplers: DirCouplersParams = field(default_factory=DirCouplersParams)
    glare: GlareParams = field(default_factory=GlareParams)
    # Film base density (film base + fog / orange mask) tuning.
    base: FilmBaseParams = field(default_factory=FilmBaseParams)
    # Convert-film: scanned-negative -> film cmy conversion settings.
    convert: ConvertFilmParams = field(default_factory=ConvertFilmParams)


@dataclass
class PrintRenderingParams:
    glare: GlareParams = field(default_factory=GlareParams)
    chemistry: PrintChemistryParams = field(default_factory=PrintChemistryParams)
    # Print base density tuning.
    base: PrintBaseParams = field(default_factory=PrintBaseParams)


@dataclass
class IOParams:
    input_color_space: str = "ProPhoto RGB"
    input_cctf_decoding: bool = False
    output_color_space: str = "sRGB"
    output_cctf_encoding: bool = True
    # Input gamut compression: smoothly pulls input chromaticities that
    # fall outside the visible spectral locus back inside (where Hanatos
    # 2025's spectral upsampling is well-defined). Baked into the
    # per-film tc_lut at build time so the per-pixel hot path is
    # untouched. See spektrafilm-research/studies/a00/a40_lut_system/n100
    # for the design.
    input_gamut_compress: InputGamutCompressSpec = field(default_factory=InputGamutCompressSpec)
    # Output gamut compression: smoothly compresses out-of-output-gamut
    # chromaticities (via the chroma knee) and above-white lightnesses
    # (via lightness_compression, a one-sided soft roll-off that leaves
    # black at 0) into the output primaries cube. With both engaged the
    # simulation output is guaranteed in [0, 1] and no downstream clip
    # is needed. See spektrafilm-research/studies/a00/a40_lut_system/n110
    # for the design and b40 for the smoothness analysis.
    output_gamut_compress: OutputGamutCompressSpec = field(default_factory=OutputGamutCompressSpec)
    crop: bool = False
    crop_center: tuple[float, float] = (0.5, 0.5)
    crop_size: tuple[float, float] = (0.1, 0.1)
    upscale_factor: float = 1.0


@dataclass
class WorkflowParams:
    # route selects which path the image takes through the pipeline stages.
    # Allowed values:
    #   "input"                          (passthrough: input -> output colour space only)
    #   "input > film > scan"
    #   "input > film > print > scan"
    #   "input > convert-film > print > scan"
    #   "input > convert-film > scan-minus-base"
    #   "input > convert-film > scan"   (convert, then scan the film WITH its base)
    # The "convert-film" routes convert a scanned negative to film cmy density
    # via the inverse-scan bridge (ConvertingStage / model.convert). The bare
    # "input" route runs no film/print/scan stage: it just colour-manages the
    # decoded input into the output colour space so it is directly viewable.
    route: str = "input > film > print > scan"

    @property
    def passthrough(self) -> bool:
        """Whether the pipeline only colour-manages the input to the output space
        (no film / print / scan): the bare "input" route."""
        return self.route == "input"

    @property
    def do_filming(self) -> bool:
        """Whether the pipeline exposes and develops the film."""
        return self.route in ("input > film > scan", "input > film > print > scan")

    @property
    def do_printing(self) -> bool:
        """Whether the pipeline exposes and develops the print."""
        return self.route in ("input > film > print > scan", "input > convert-film > print > scan")

    @property
    def scan_film(self) -> bool:
        """Whether the final scan reads the film density rather than the print."""
        return self.route in (
            "input > film > scan",
            "input > convert-film > scan-minus-base",
            "input > convert-film > scan",
        )

    @property
    def scan_minus_base(self) -> bool:
        """Whether the film scan should subtract the film base (base subtraction itself is WIP)."""
        return self.route == "input > convert-film > scan-minus-base"

    @property
    def do_convert_film(self) -> bool:
        """Whether the front-end converts a scanned negative (input RGB) directly
        to film cmy density (the inverse of the scan), instead of exposing and
        developing film. True for the "convert-film" routes."""
        return self.route in (
            "input > convert-film > print > scan",
            "input > convert-film > scan-minus-base",
            "input > convert-film > scan",
        )

    @property
    def collect_tap(self) -> str:
        """The pipeline's terminal tap. Every supported route ends at rgb_out."""
        return "rgb_out"

    @property
    def do_scan(self) -> bool:
        """Whether the pipeline runs a final scan to rgb_out. False for the bare
        "input" passthrough, which reaches rgb_out without a scan stage."""
        return self.collect_tap == "rgb_out" and not self.passthrough


@dataclass
class DebugParams:
    deactivate_spatial_effects: bool = False
    deactivate_stochastic_effects: bool = False
    print_timings: bool = False
    # When True, the pipeline behaves as a deterministic per-pixel transform
    # suitable for LUT sampling: spatial effects, stochastic effects,
    # auto-exposure, and scanner white/black/unsharp corrections are all
    # forced off, regardless of the underlying settings.
    lut_mode: bool = False


@dataclass
class TapsParams:
    """Pipeline tap configuration.

    ``inject`` and ``collect`` name the entry and exit points in the
    pipeline topology. Defaults of None mean "normal end-to-end run"
    (inject at rgb_in, collect at rgb_out).
    """
    inject: str | None = None
    collect: str | None = None


@dataclass
class SettingsParams:
    rgb_to_raw_method: str = "arctic2026beta04"
    apply_hanatos2025_adaptation_window: bool = True
    apply_hanatos2025_adaptation_surface: bool = False
    spectral_gaussian_blur: float = 0.0
    use_enlarger_lut: bool = False
    use_scanner_lut: bool = False
    use_convert_lut: bool = False
    lut_resolution: int = 17
    # The convert (scan-inverse) LUT gets its own, finer grid: its input is RGB
    # (a curved gamut inside the cube) and the print stage amplifies mid-tone
    # density error, so it needs more nodes than the density-domain scanner/enlarger
    # LUTs. The bake is one-time and only repeats when the scan model changes.
    convert_lut_resolution: int = 33
    use_fast_stats: bool = True
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
    workflow: WorkflowParams = field(default_factory=WorkflowParams)
    debug: DebugParams = field(default_factory=DebugParams)
    settings: SettingsParams = field(default_factory=SettingsParams)
    taps: TapsParams = field(default_factory=TapsParams)

    def __post_init__(self):
        if not isinstance(self.film, Profile):
            raise TypeError("film must be a Profile instance")
        if not isinstance(self.print, Profile):
            raise TypeError("print must be a Profile instance")
