from __future__ import annotations

from dataclasses import dataclass, field, fields
from types import SimpleNamespace
from typing import Any, Mapping, Optional


@dataclass
class CameraParams:
    exposure_compensation_ev: float = 0.0
    auto_exposure: bool = True
    auto_exposure_method: str = "center_weighted"
    lens_blur_um: float = 0.0
    film_format_mm: float = 35.0
    filter_uv: tuple[float, float, float] = (1.0, 410.0, 8.0)
    filter_ir: tuple[float, float, float] = (1.0, 675.0, 15.0)


@dataclass
class EnlargerParams:
    illuminant: str = "TH-KG3-L"
    print_exposure: float = 1.0
    print_exposure_compensation: bool = True
    y_filter_shift: float = 0.0
    m_filter_shift: float = 0.0
    y_filter_neutral: float = 0.9
    m_filter_neutral: float = 0.5
    c_filter_neutral: float = 0.35
    lens_blur: float = 0.0
    preflash_exposure: float = 0.0
    preflash_y_filter_shift: float = 0.0
    preflash_m_filter_shift: float = 0.0
    just_preflash: bool = False


@dataclass
class ScannerParams:
    lens_blur: float = 0.0
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
    blur: float = 0.55
    blur_dye_clouds_um: float = 1.0
    micro_structure: tuple[float, float] = (0.2, 0.5)
    n_sub_layers: int = 1


@dataclass
class HalationParams:
    active: bool = True
    strength: tuple[float, float, float] = (0.03, 0.003, 0.001)
    size_um: tuple[float, float, float] = (200.0, 200.0, 200.0)
    scattering_strength: tuple[float, float, float] = (0.01, 0.02, 0.04)
    scattering_size_um: tuple[float, float, float] = (30.0, 20.0, 15.0)


@dataclass
class DirCouplersParams:
    active: bool = True
    amount: float = 1.0
    ratio_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0)
    diffusion_interlayer: float = 2.0
    diffusion_size_um: float = 10.0
    high_exposure_shift: float = 0.0


@dataclass
class GlareParams:
    active: bool = True
    percent: float = 0.1
    roughness: float = 0.4
    blur: float = 0.5
    compensation_removal_factor: float = 0.0
    compensation_removal_density: float = 1.2
    compensation_removal_transition: float = 0.3


@dataclass
class NegativeRenderingParams:
    density_curve_gamma: float = 1.0
    base_density_scale: float = 1.0
    grain: GrainParams = field(default_factory=GrainParams)
    halation: HalationParams = field(default_factory=HalationParams)
    dir_couplers: DirCouplersParams = field(default_factory=DirCouplersParams)


@dataclass
class PrintRenderingParams:
    density_curve_gamma: float = 1.0
    base_density_scale: float = 0.4
    glare: GlareParams = field(default_factory=GlareParams)


@dataclass
class IOParams:
    input_color_space: str = "ProPhoto RGB"
    input_cctf_decoding: bool = False
    output_color_space: str = "sRGB"
    output_cctf_encoding: bool = True
    crop: bool = False
    crop_center: tuple[float, float] = (0.5, 0.5)
    crop_size: tuple[float, float] = (0.1, 1.0)
    preview_resize_factor: float = 1.0
    upscale_factor: float = 1.0
    full_image: bool = False
    compute_negative: bool = False
    compute_film_raw: bool = False


@dataclass
class DebugLuts:
    enlarger_lut: Optional[Any] = None
    scanner_lut: Optional[Any] = None


@dataclass
class DebugParams:
    deactivate_spatial_effects: bool = False
    deactivate_stochastic_effects: bool = False
    input_negative_density_cmy: bool = False
    return_negative_density_cmy: bool = False
    return_print_density_cmy: bool = False
    print_timings: bool = False
    luts: DebugLuts = field(default_factory=DebugLuts)


@dataclass
class SettingsParams:
    rgb_to_raw_method: str = "hanatos2025"
    use_camera_lut: bool = False
    use_enlarger_lut: bool = False
    use_scanner_lut: bool = False
    lut_resolution: int = 17
    use_fast_stats: bool = False


@dataclass
class RuntimePhotoParams:
    negative: Any
    print_paper: Any
    negative_render: NegativeRenderingParams = field(default_factory=NegativeRenderingParams)
    print_render: PrintRenderingParams = field(default_factory=PrintRenderingParams)
    camera: CameraParams = field(default_factory=CameraParams)
    enlarger: EnlargerParams = field(default_factory=EnlargerParams)
    scanner: ScannerParams = field(default_factory=ScannerParams)
    io: IOParams = field(default_factory=IOParams)
    debug: DebugParams = field(default_factory=DebugParams)
    settings: SettingsParams = field(default_factory=SettingsParams)


def _mapping_to_namespace(value: Any) -> Any:
    if isinstance(value, Mapping):
        return SimpleNamespace(**{k: _mapping_to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_mapping_to_namespace(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_mapping_to_namespace(v) for v in value)
    return value


def _coerce_section(value: Any, section_type: type[Any]) -> Any:
    if isinstance(value, section_type):
        return value
    if isinstance(value, Mapping):
        source = value
    elif hasattr(value, "__dict__"):
        source = vars(value)
    else:
        source = {}

    kwargs: dict[str, Any] = {}
    field_names = {f.name for f in fields(section_type)}
    for key in field_names:
        if key in source:
            kwargs[key] = source[key]

    section = section_type(**kwargs)
    if isinstance(section, DebugParams):
        section.luts = _coerce_section(section.luts, DebugLuts)
    if isinstance(section, NegativeRenderingParams):
        section.grain = _coerce_section(section.grain, GrainParams)
        section.halation = _coerce_section(section.halation, HalationParams)
        section.dir_couplers = _coerce_section(section.dir_couplers, DirCouplersParams)
    if isinstance(section, PrintRenderingParams):
        section.glare = _coerce_section(section.glare, GlareParams)
    return section


def coerce_runtime_params(params: Any) -> RuntimePhotoParams:
    if isinstance(params, RuntimePhotoParams):
        if not isinstance(params.debug.luts, DebugLuts):
            params.debug.luts = _coerce_section(params.debug.luts, DebugLuts)
        return params

    if isinstance(params, Mapping):
        source = params
    elif hasattr(params, "__dict__"):
        source = vars(params)
    else:
        raise TypeError("Unsupported params type; expected RuntimePhotoParams, mapping, or object with attributes")

    if "negative" not in source or "print_paper" not in source:
        raise ValueError("Params must include 'negative' and 'print_paper'")

    negative = source["negative"]
    print_paper = source["print_paper"]
    if isinstance(negative, Mapping):
        negative = _mapping_to_namespace(negative)
    if isinstance(print_paper, Mapping):
        print_paper = _mapping_to_namespace(print_paper)

    return RuntimePhotoParams(
        negative=negative,
        print_paper=print_paper,
        negative_render=_coerce_section(source.get("negative_render"), NegativeRenderingParams),
        print_render=_coerce_section(source.get("print_render"), PrintRenderingParams),
        camera=_coerce_section(source.get("camera"), CameraParams),
        enlarger=_coerce_section(source.get("enlarger"), EnlargerParams),
        scanner=_coerce_section(source.get("scanner"), ScannerParams),
        io=_coerce_section(source.get("io"), IOParams),
        debug=_coerce_section(source.get("debug"), DebugParams),
        settings=_coerce_section(source.get("settings"), SettingsParams),
    )
