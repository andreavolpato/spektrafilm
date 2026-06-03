"""Path-keyed presentation manifest for runtime parameters.

This is the single declaration site for how a runtime parameter appears
in the GUI — label, tooltip, range, tier — keyed by the parameter's
dotted path on
:class:`~spektrafilm.runtime.params_schema.RuntimePhotoParams`, *not* by a
mirror field in a GUI-side dataclass.

The GUI is a transparent mirror of the params schema: same field names,
same values, no GUI-only unit transforms. Whatever the schema stores is
what the widget shows. A :class:`GroupManifest` renders one runtime
parameter group as one collapsible panel; the section it drives
(``ParamsGroupSection``) reads and writes the runtime group object
directly, so there is no parallel state dataclass and no mapper.
"""

from __future__ import annotations

from dataclasses import dataclass

from spektrafilm.runtime.params_schema import (
    BaseParams,
    EnlargerParams,
    DiffusionFilterParams,
    DirCouplersParams,
    GlareParams,
    GrainParams,
    HalationParams,
    ScannerParams,
)
from spektrafilm.runtime.params_schema import CameraParams
from spektrafilm.model.illuminants import Illuminants
from spektrafilm.model.color_filters import CameraColorFilters
from spektrafilm.model.stocks import FilmStocks, PrintPapers
from spektrafilm.utils.gamut_compression import InputGamutCompressSpec, OutputGamutCompressSpec
from spektrafilm.utils.morph_curves import FilmChemistryParams, PrintChemistryParams
from spektrafilm_gui.options import (
    AutoExposureMethods,
    DiffusionFilterFamilies,
    InputGamutCompressAlgorithms,
    NapariInterpolationModes,
    OutputGamutCompressAlgorithms,
    RGBColorSpaces,
    RGBtoRAWMethod,
)


@dataclass(frozen=True)
class ParamSpec:
    """Presentation metadata for a single runtime parameter.

    ``path`` is the dotted path on ``RuntimePhotoParams`` (for example
    ``"film_render.dir_couplers.amount"``). The leaf is the field on the
    owning group dataclass; the prefix locates the group in the runtime
    tree. The editor type is inferred from the group dataclass's own type
    hints, so it is never restated here. ``label`` is optional — when
    omitted the field name itself is shown, matching the schema.
    """

    path: str
    name: str | None = None
    label: str | None = None
    tooltip: str = ""
    tier: str = "dev"
    min: float | None = None
    max: float | None = None
    step: float | None = None
    decimals: int | None = None
    enum: type | None = None  # Enum class supplying the choices for a str field

    @property
    def leaf(self) -> str:
        return self.name or self.path.rsplit(".", 1)[-1]


@dataclass(frozen=True)
class GroupManifest:
    """A runtime parameter group rendered as one collapsible GUI panel.

    ``group_path`` locates the group on ``RuntimePhotoParams`` (for
    example ``"film_render.dir_couplers"``) and ``group_cls`` is its
    dataclass type; ``fields`` declares which of its fields to expose, in
    display order. Fields of the group not listed here are passed through
    untouched, so the GUI can edit a subset of a runtime group without
    dropping the rest.

    ``panel_fields``, when set, is the subset of ``fields`` actually laid
    out in this group's own panel; the remaining ``fields`` still get
    editors built (so they stay editable, persisted, and auto-preview
    wired through this section) but are displayed by another section that
    borrows them — the same owner/view split ``SimulationSection`` uses.
    When ``None`` the panel renders every field.
    """

    title: str
    group_path: str
    group_cls: type
    fields: tuple[ParamSpec, ...]
    collapsed_by_default: bool = True
    panel_fields: tuple[ParamSpec, ...] | None = None


_DC = "film_render.dir_couplers"

DIR_COUPLERS_MANIFEST = GroupManifest(
    title="Couplers",
    group_path=_DC,
    group_cls=DirCouplersParams,
    collapsed_by_default=True,
    fields=(
        ParamSpec(f"{_DC}.active", tier="basic", tooltip="Enable DIR coupler inhibition."),
        ParamSpec(
            f"{_DC}.amount",
            tier="basic",
            min=0,
            step=0.05,
            tooltip="Global multiplier on the DIR coupler inhibition matrix. 1.0 leaves the per-channel gammas as-is.",
        ),
        ParamSpec(
            f"{_DC}.inhibition_samelayer",
            min=0,
            step=0.05,
            tooltip="Multiplier on the same-layer (diagonal) inhibition. Controls overall contrast / gamma reduction within each RGB layer.",
        ),
        ParamSpec(
            f"{_DC}.inhibition_interlayer",
            min=0,
            step=0.05,
            tooltip="Multiplier on the cross-layer (off-diagonal) inhibition. Controls saturation enhancement from interlayer DIR effects.",
        ),
        ParamSpec(
            f"{_DC}.gamma_samelayer_rgb",
            min=0,
            step=0.02,
            tooltip="Per-channel same-layer DIR gamma (R, G, B). Effective gamma reduction of each layer's density curve.",
        ),
        ParamSpec(
            f"{_DC}.gamma_interlayer_r_to_gb",
            min=0,
            step=0.02,
            tooltip="DIR inhibition from the R layer onto the G and B layers respectively (g_R->G, g_R->B).",
        ),
        ParamSpec(
            f"{_DC}.gamma_interlayer_g_to_rb",
            min=0,
            step=0.02,
            tooltip="DIR inhibition from the G layer onto the R and B layers respectively (g_G->R, g_G->B).",
        ),
        ParamSpec(
            f"{_DC}.gamma_interlayer_b_to_rg",
            min=0,
            step=0.02,
            tooltip="DIR inhibition from the B layer onto the R and G layers respectively (g_B->R, g_B->G).",
        ),
        ParamSpec(
            f"{_DC}.diffusion_size_um",
            min=0,
            step=5,
            tooltip="Sigma in um for the diffusion of the couplers, (5-20 um), controls sharpness and affects saturation.",
        ),
    ),
)


_H = "film_render.halation"

HALATION_MANIFEST = GroupManifest(
    title="Halation",
    group_path=_H,
    group_cls=HalationParams,
    collapsed_by_default=True,
    fields=(
        ParamSpec(f"{_H}.active", tier="basic", tooltip="Enable halation and in-emulsion scatter."),
        ParamSpec(
            f"{_H}.scatter_amount",
            tier="basic",
            min=0,
            step=0.05,
            tooltip="High-level scatter strength. 1.0 = full physical scatter, 0.0 = no scatter. Scales the fraction of light that undergoes in-emulsion scattering.",
        ),
        ParamSpec(
            f"{_H}.scatter_spatial_scale",
            tier="basic",
            min=0,
            step=0.1,
            tooltip="High-level scatter size multiplier (1.0 = physical defaults). Scales both core and tail sigmas.",
        ),
        ParamSpec(
            f"{_H}.halation_amount",
            tier="basic",
            min=0,
            step=0.05,
            tooltip="High-level halation strength multiplier (1.0 = physical defaults). Scales the per-channel halation amplitudes.",
        ),
        ParamSpec(
            f"{_H}.halation_spatial_scale",
            tier="basic",
            min=0,
            step=0.1,
            tooltip="High-level halation size multiplier (1.0 = physical defaults). Scales the first-bounce sigma.",
        ),
        ParamSpec(
            f"{_H}.boost_ev",
            tier="basic",
            min=0,
            step=0.5,
            tooltip="Maximum highlight boost in stops.",
        ),
        ParamSpec(
            f"{_H}.protect_ev",
            tier="basic",
            min=0,
            step=0.5,
            tooltip="Protected range above midgray for the boost onset in stops.",
        ),
        ParamSpec(
            f"{_H}.boost_range",
            tier="basic",
            min=0,
            max=1,
            step=0.05,
            tooltip="Controls how quickly the highlight boost ramps in, from 0 to 1.",
        ),
        ParamSpec(
            f"{_H}.scatter_core_um",
            min=0,
            step=0.5,
            tooltip="Sigma of the scatter core Gaussian per channel [R,G,B], in micrometers. Controls fine-scale sharpness loss in the emulsion.",
        ),
        ParamSpec(
            f"{_H}.scatter_tail_um",
            min=0,
            step=1.0,
            tooltip="Decay constant of the scatter exponential tail per channel [R,G,B], in micrometers (internally approximated by a sum of Gaussians). Controls extended low-level spread within the emulsion.",
        ),
        ParamSpec(
            f"{_H}.scatter_tail_weight",
            min=0,
            max=1,
            step=0.01,
            decimals=2,
            tooltip="Weight of the scatter tail Gaussian per channel [R,G,B] (0-1). Tail weight + core weight = 1. Higher values put more scattered light into the long tail.",
        ),
        ParamSpec(
            f"{_H}.halation_strength",
            min=0,
            step=0.005,
            decimals=3,
            tooltip="Total back-reflection halation amplitude per channel [R,G,B] (0-1). Typical red channel: weak AH 0.02-0.08, no AH 0.08-0.25. The blue channel is usually near zero.",
        ),
        ParamSpec(
            f"{_H}.halation_first_sigma_um",
            min=0,
            step=1.0,
            tooltip="Sigma of the first halation bounce per channel [R,G,B], in micrometers. Set by the base thickness (40-80 um for typical cine/still bases).",
        ),
        ParamSpec(
            f"{_H}.halation_n_bounces",
            min=1,
            max=5,
            step=1,
            tooltip="Number of multi-bounce Gaussians summed in the halation pass. Subsequent bounces use sqrt(k)-spaced widths. Typical: 2-3.",
        ),
        ParamSpec(
            f"{_H}.halation_bounce_decay",
            min=0,
            max=1,
            step=0.05,
            tooltip="Per-bounce amplitude decay ratio (rho). Physical range 0.3-0.7. Controls how fast the halation energy falls off between bounces.",
        ),
        ParamSpec(
            f"{_H}.halation_renormalize",
            tooltip="If enabled, divide by (1 + sum of bounce amplitudes) so mid-grey is preserved. If disabled, halation is purely additive and subtly lifts shadows as well as highlights.",
        ),
    ),
)


_G = "film_render.grain"

GRAIN_MANIFEST = GroupManifest(
    title="Grain",
    group_path=_G,
    group_cls=GrainParams,
    fields=(
        ParamSpec(f"{_G}.active", tier="basic", tooltip="Add grain to the negative"),
        ParamSpec(
            f"{_G}.particle_area_um2",
            tier="basic",
            min=0,
            step=0.2,
            tooltip="Area of the particles in um2, relates to ISO. Approximately 0.1 for ISO 100, 0.1 for ISO 200, 0.4 for ISO 400 and so on.",
        ),
        ParamSpec(
            f"{_G}.particle_scale",
            tooltip="Scale of particle area for the RGB layers, multiplies particle_area_um2",
        ),
        ParamSpec(
            f"{_G}.particle_scale_layers",
            min=0,
            step=0.25,
            tooltip="Scale of particle area for the sublayers in every color layer, multiplies particle_area_um2",
        ),
        ParamSpec(f"{_G}.density_min", tooltip="Minimum density of the grain, typical values (0.03-0.06)"),
        ParamSpec(f"{_G}.uniformity", tooltip="Uniformity of the grain, typical values (0.94-0.98)"),
        ParamSpec(
            f"{_G}.blur",
            tier="basic",
            min=0,
            step=0.05,
            tooltip="Sigma of gaussian blur in pixels for the grain, to be increased at high magnifications, (should be 0.8-0.9 at high resolution, reduce down to 0.6 for lower res).",
        ),
        ParamSpec(
            f"{_G}.blur_dye_clouds_um",
            min=0,
            step=0.1,
            tooltip="Scale the sigma of gaussian blur in um for the dye clouds, to be used at high magnifications, (default 1)",
        ),
        ParamSpec(
            f"{_G}.micro_structure",
            min=0,
            step=0.1,
            tooltip="Parameter for micro-structure due to clumps at the molecular level, [sigma blur of micro-structure / ultimate light-resolution (0.10 um default), size of molecular clumps in nm (30 nm default)]. Only for insane magnifications.",
        ),
    ),
)


_BD = "film_render.base"

BASE_MANIFEST = GroupManifest(
    title="Base",
    group_path=_BD,
    group_cls=BaseParams,
    fields=(
        ParamSpec(f"{_BD}.active", tier="basic", tooltip="Enable base-density (film base + fog / orange mask) tuning."),
        ParamSpec(f"{_BD}.scale", tier="basic", min=0, step=0.05, tooltip="Overall multiplier on the base density."),
        ParamSpec(f"{_BD}.cyan", tier="basic", min=0, step=0.05, tooltip="Cyan-channel scale of the base density."),
        ParamSpec(f"{_BD}.magenta", tier="basic", min=0, step=0.05, tooltip="Magenta-channel scale of the base density."),
        ParamSpec(f"{_BD}.yellow", tier="basic", min=0, step=0.05, tooltip="Yellow-channel scale of the base density."),
    ),
)


_GL = "print_render.glare"

GLARE_MANIFEST = GroupManifest(
    title="Glare",
    group_path=_GL,
    group_cls=GlareParams,
    fields=(
        ParamSpec(f"{_GL}.active", tier="basic", tooltip="Add glare to the print"),
        ParamSpec(f"{_GL}.percent", tier="basic", min=0, max=1, step=0.01, tooltip="Percentage of the glare light (typically 0.1-0.25)"),
        ParamSpec(f"{_GL}.roughness", tier="basic", min=0, max=1, step=0.05, tooltip="Roughness of the glare light (0-1)"),
        ParamSpec(f"{_GL}.blur", tier="basic", min=0, step=0.1, tooltip="Sigma of gaussian blur in pixels for the glare"),
    ),
)


_CH = "print_render.chemistry"

CHEMISTRY_MANIFEST = GroupManifest(
    title="Chemistry",
    group_path=_CH,
    group_cls=PrintChemistryParams,
    fields=(
        ParamSpec(
            f"{_CH}.development_time",
            label="Development time",
            tier="basic",
            tooltip="Development time for a BW development-time family: selects the density curve and base+fog to render. '—' uses the representative middle development; ignored for single-curve and color stocks.",
        ),
        ParamSpec(f"{_CH}.active", tier="basic", tooltip="Enable print density-curve morphing."),
        ParamSpec(
            f"{_CH}.gamma_factor",
            label="Gamma factor",
            tier="basic",
            min=0.5,
            max=2.0,
            step=0.05,
            tooltip="Global coupled gamma multiplier for the print density curves.",
        ),
        ParamSpec(
            f"{_CH}.gamma_factor_fast",
            label="Gamma factor fast",
            min=0.5,
            max=2.0,
            step=0.05,
            tooltip="Gamma factor applied to the fast sub-layer.",
        ),
        ParamSpec(
            f"{_CH}.gamma_factor_slow",
            label="Gamma factor slow",
            min=0.5,
            max=2.0,
            step=0.05,
            tooltip="Gamma factor applied to the mid and slow sub-layers.",
        ),
        ParamSpec(
            f"{_CH}.gamma_factor_red",
            label="Gamma factor red",
            min=0.5,
            max=2.0,
            step=0.02,
            tooltip="Per-channel gamma factor applied to the red channel.",
        ),
        ParamSpec(
            f"{_CH}.gamma_factor_green",
            label="Gamma factor green",
            min=0.5,
            max=2.0,
            step=0.02,
            tooltip="Per-channel gamma factor applied to the green channel.",
        ),
        ParamSpec(
            f"{_CH}.gamma_factor_blue",
            label="Gamma factor blue",
            min=0.5,
            max=2.0,
            step=0.02,
            tooltip="Per-channel gamma factor applied to the blue channel.",
        ),
        ParamSpec(
            f"{_CH}.developer_exhaustion",
            label="Developer exhaustion",
            min=0.0,
            max=1.0,
            step=0.02,
            tooltip="Blend all three print sub-layers toward the matched Gumbel shoulder while preserving midgray via a common horizontal offset.",
        ),
    ),
)


_FCH = "film_render.chemistry"

FILM_CHEMISTRY_MANIFEST = GroupManifest(
    title="Chemistry",
    group_path=_FCH,
    group_cls=FilmChemistryParams,
    fields=(
        ParamSpec(
            f"{_FCH}.development_time",
            label="Development time",
            tier="basic",
            tooltip="Development time for a BW development-time family: selects the density curve and base+fog to render. '—' uses the representative middle development; ignored for single-curve and color stocks.",
        ),
        ParamSpec(f"{_FCH}.active", tier="basic", tooltip="Enable film density-curve morphing."),
        ParamSpec(
            f"{_FCH}.gamma_factor",
            label="Gamma factor",
            tier="basic",
            min=0.5,
            max=2.0,
            step=0.05,
            tooltip="Global coupled gamma multiplier for the film density curves.",
        ),
        ParamSpec(
            f"{_FCH}.gamma_factor_fast",
            label="Gamma factor fast",
            min=0.5,
            max=2.0,
            step=0.05,
            tooltip="Gamma factor applied to the fast sub-layer.",
        ),
        ParamSpec(
            f"{_FCH}.gamma_factor_slow",
            label="Gamma factor slow",
            min=0.5,
            max=2.0,
            step=0.05,
            tooltip="Gamma factor applied to the mid and slow sub-layers.",
        ),
        ParamSpec(
            f"{_FCH}.gamma_factor_red",
            label="Gamma factor red",
            min=0.5,
            max=2.0,
            step=0.02,
            tooltip="Per-channel gamma factor applied to the red channel.",
        ),
        ParamSpec(
            f"{_FCH}.gamma_factor_green",
            label="Gamma factor green",
            min=0.5,
            max=2.0,
            step=0.02,
            tooltip="Per-channel gamma factor applied to the green channel.",
        ),
        ParamSpec(
            f"{_FCH}.gamma_factor_blue",
            label="Gamma factor blue",
            min=0.5,
            max=2.0,
            step=0.02,
            tooltip="Per-channel gamma factor applied to the blue channel.",
        ),
        ParamSpec(
            f"{_FCH}.developer_exhaustion",
            label="Developer exhaustion",
            min=0.0,
            max=1.0,
            step=0.02,
            tooltip="Blend all three film sub-layers toward the matched Gumbel shoulder while preserving midgray via a common horizontal offset.",
        ),
    ),
)


_ENL = "enlarger"

PREFLASHING_MANIFEST = GroupManifest(
    title="Preflash",
    group_path=_ENL,
    group_cls=EnlargerParams,
    collapsed_by_default=True,
    fields=(
        ParamSpec(
            f"{_ENL}.preflash_exposure",
            label="exposure",
            min=0,
            step=0.005,
            tooltip="Preflash exposure value in ev for the print",
        ),
        ParamSpec(
            f"{_ENL}.preflash_y_filter_shift",
            label="y filter shift",
            step=1,
            tooltip="Shift the Y filter of the enlarger from the neutral position for the preflash, typical values (-20-20), in Kodak CC units",
        ),
        ParamSpec(
            f"{_ENL}.preflash_m_filter_shift",
            label="m filter shift",
            step=1,
            tooltip="Shift the M filter of the enlarger from the neutral position for the preflash, typical values (-20-20), in Kodak CC units",
        ),
    ),
)


def _diffusion_manifest(title: str, group_path: str) -> GroupManifest:
    """Build a manifest for a ``DiffusionFilterParams`` group.

    The camera and enlarger stages share the same PSF dataclass, so the
    only difference between their panels is where they live in the
    runtime tree (``group_path``).
    """
    _D = group_path
    return GroupManifest(
        title=title,
        group_path=_D,
        group_cls=DiffusionFilterParams,
        fields=(
            ParamSpec(f"{_D}.active", tier="basic", tooltip="Toggle the diffusion filter (Pro-Mist family)."),
            ParamSpec(
                f"{_D}.filter_family",
                tier="basic",
                enum=DiffusionFilterFamilies,
                tooltip="PSF family. pro_mist / glimmerglass / cinebloom are transparent (energy-preserving); black_pro_mist absorbs a fraction of the deflected light, lifting shadows by reducing local contrast.",
            ),
            ParamSpec(
                f"{_D}.strength",
                tier="basic",
                min=0,
                max=2,
                step=0.125,
                tooltip="Commercial filter stop: 0, 1/8=0.125, 1/4=0.25, 1/2=0.5, 1, 2. Maps internally to the (p_s, p_a) deflected/absorbed photon fractions.",
            ),
            ParamSpec(
                f"{_D}.spatial_scale",
                min=0,
                step=0.1,
                tooltip="Multiplier on the image-plane PSF widths (all per-group lambdas). Adjust for image-format / print-size differences.",
            ),
            ParamSpec(
                f"{_D}.halo_warmth",
                min=-1.5,
                max=1.5,
                step=0.05,
                tooltip="Additive offset on the family's halo warmth axis. Positive = warm outer halo / cool inner halo. Energy-preserving per channel. 0 = use family default.",
            ),
            ParamSpec(f"{_D}.core_intensity", min=0, max=4, step=0.05, tooltip="Advanced. Multiplier on the core weight; the three group weights are renormalized to sum to 1. 1.0 = use family default."),
            ParamSpec(f"{_D}.core_size", min=0.1, max=4, step=0.05, tooltip="Advanced. Multiplier on the core lambda. 1.0 = use family default."),
            ParamSpec(f"{_D}.halo_intensity", min=0, max=4, step=0.05, tooltip="Advanced. Multiplier on the halo weight; the three group weights are renormalized to sum to 1. 1.0 = use family default."),
            ParamSpec(f"{_D}.halo_size", min=0.1, max=4, step=0.05, tooltip="Advanced. Multiplier on the halo lambda. 1.0 = use family default."),
            ParamSpec(f"{_D}.bloom_intensity", min=0, max=4, step=0.05, tooltip="Advanced. Multiplier on the bloom weight; the three group weights are renormalized to sum to 1. 1.0 = use family default."),
            ParamSpec(f"{_D}.bloom_size", min=0.1, max=4, step=0.05, tooltip="Advanced. Multiplier on the bloom lambda. 1.0 = use family default."),
        ),
    )


CAMERA_DIFFUSION_MANIFEST = _diffusion_manifest("Diffusion", "camera.diffusion_filter")
ENLARGER_DIFFUSION_MANIFEST = _diffusion_manifest("Diffusion", "enlarger.diffusion_filter")


_CAM = "camera"

CAMERA_PANEL_FIELDS = (
    ParamSpec(
        f"{_CAM}.exposure_compensation_ev",
        tier="basic",
        min=-100,
        max=100,
        step=0.25,
        tooltip="Add a bias to the auto-exposure of the camera.",
    ),
    ParamSpec(
        f"{_CAM}.auto_exposure",
        tier="basic",
        tooltip="Use the auto-exposure feature of the virtual camera.",
    ),
    ParamSpec(
        f"{_CAM}.film_format_mm",
        tier="basic",
        min=8.0,
        max=120.0,
        step=1.0,
        decimals=0,
        tooltip="Long edge of the film format in millimeters, e.g. 8, 16, 35, 60, 120.",
    ),
    ParamSpec(
        f"{_CAM}.auto_exposure_method",
        enum=AutoExposureMethods,
        tooltip="Metering method used by the virtual camera's auto-exposure.",
    ),
    ParamSpec(
        f"{_CAM}.color_filter",
        tier="basic",
        enum=CameraColorFilters,
        tooltip="Camera taking filter held in front of the lens. Its measured spectral transmittance multiplies the incoming light, so it both darkens and color-casts the exposure (e.g. Hoya Y2/YA3 yellow, R1 red, X0/X1 UV/haze). 'none' = no filter.",
    ),
)

CAMERA_MANIFEST = GroupManifest(
    title="Camera",
    group_path=_CAM,
    group_cls=CameraParams,
    collapsed_by_default=False,
    fields=CAMERA_PANEL_FIELDS,
)


_SC = "scanner"

SCANNER_MANIFEST = GroupManifest(
    title="Scanner",
    group_path=_SC,
    group_cls=ScannerParams,
    collapsed_by_default=True,
    fields=(
        ParamSpec(
            f"{_SC}.lens_blur",
            tier="basic",
            min=0,
            step=0.05,
            tooltip="Sigma of gaussian filter in pixel for the scanner lens blur.",
        ),
        ParamSpec(f"{_SC}.white_correction", tier="basic", tooltip="Enable white point correction applied to the scanner output."),
        ParamSpec(
            f"{_SC}.white_level",
            tier="basic",
            min=0,
            max=1,
            step=0.005,
            decimals=3,
            tooltip="Target white level applied when white correction is enabled.",
        ),
        ParamSpec(f"{_SC}.black_correction", tier="basic", tooltip="Enable black point correction applied to the scanner output."),
        ParamSpec(
            f"{_SC}.black_level",
            tier="basic",
            min=0,
            max=1,
            step=0.005,
            decimals=3,
            tooltip="Target black level applied when black correction is enabled.",
        ),
        ParamSpec(
            f"{_SC}.unsharp_mask",
            min=0,
            step=0.05,
            tooltip="Apply unsharp mask to the scan, [sigma in pixel, amount].",
        ),
    ),
)


_IGC = "io.input_gamut_compress"

INPUT_GAMUT_COMPRESS_MANIFEST = GroupManifest(
    title="Input gamut compress",
    group_path=_IGC,
    group_cls=InputGamutCompressSpec,
    collapsed_by_default=True,
    fields=(
        ParamSpec(f"{_IGC}.active", tier="basic", tooltip="Compress input chromaticities toward the visible spectral locus before spectral upsampling. Off passes input through unchanged."),
        ParamSpec(
            f"{_IGC}.algorithm",
            tier="basic",
            enum=InputGamutCompressAlgorithms,
            tooltip="xy: radial compression in CIE 1931 chromaticity (ACES RGC family, hue-preserving). oklch: perceptual chroma reduction at constant Oklch (L, h).",
        ),
        ParamSpec(
            f"{_IGC}.knee",
            min=0.0,
            step=0.05,
            decimals=3,
            tooltip="Reinhard knee (threshold, limit, power). Defaults (0.0, 1.0, 6.0) apply a soft full-range roll-off that asymptotes at the spectral locus boundary.",
        ),
    ),
)


_OGC = "io.output_gamut_compress"

OUTPUT_GAMUT_COMPRESS_MANIFEST = GroupManifest(
    title="Output gamut compress",
    group_path=_OGC,
    group_cls=OutputGamutCompressSpec,
    collapsed_by_default=True,
    fields=(
        ParamSpec(
            f"{_OGC}.algorithm",
            tier="basic",
            enum=OutputGamutCompressAlgorithms,
            tooltip="off: disable output gamut compression. cam16ucs (default): CAM16-UCS chroma reduction with the smoothest constant-hue behavior. oklch: perceptual chroma reduction in OkLab, preserves hue + lightness. aces_rgc: per-channel ACES RGC v1.3, matches Resolve/Nuke/OCIO. oklrab/jzazbz: alternative perceptual spaces.",
        ),
        ParamSpec(
            f"{_OGC}.knee",
            min=0.0,
            step=0.05,
            decimals=3,
            tooltip="Reinhard knee (threshold, limit, power) on normalized chroma C/C_max. Default (0.0, 1.0, 6.0) rolls off smoothly from the center with the asymptote on the output cube edge.",
        ),
    ),
)


# --- input_image: a cross-group cluster -------------------------------------
# Unlike a GroupManifest (one runtime group → one panel), input_image spans two
# runtime groups (io.* + settings.*) and is rendered across three panels that
# live on two different tabs (Input + Crop on MAIN, Spectral on ADVANCED). The
# ParamSpec paths here are rooted at the GUI's InputImageState, which holds
# `.io: IOParams` and `.settings: SettingsParams` — so "io.upscale_factor" is
# valid both on InputImageState and (structurally) on RuntimePhotoParams. One
# owner section (InputImageSection) builds all editors and renders INPUT_PANEL;
# the Crop/Spectral panels embed the remaining editors (the field sets are
# disjoint, so no editor is double-parented). `label` is kept so the visible UI
# text is unchanged by the migration to path-bound manifests.

INPUT_PANEL_FIELDS = (
    ParamSpec(
        "io.input_color_space",
        label="Input color space",
        tier="basic",
        enum=RGBColorSpaces,
        tooltip="Color space of the input image, will be internally converted to sRGB and negative values clipped",
    ),
    ParamSpec(
        "io.input_cctf_decoding",
        label="Apply CCTF decoding",
        tier="basic",
        tooltip="Apply the inverse cctf transfer function of the color space",
    ),
)

CROP_PANEL_FIELDS = (
    ParamSpec("io.upscale_factor", label="Upscale factor", min=0.0, step=0.5, tooltip="Scale image size up to increase resolution"),
    ParamSpec("io.crop", label="Crop", tooltip="Crop image to a fraction of the original size to preview details at full scale"),
    ParamSpec("io.crop_center", label="Crop center", min=0, max=1, step=0.01, tooltip="Center of the crop region in relative coordinates in x, y (0-1)"),
    ParamSpec("io.crop_size", label="Crop size", min=0, max=1, step=0.01, tooltip="Normalized size of the crop region in x, y (0,1), as fraction of the long side."),
)

SPECTRAL_PANEL_FIELDS = (
    ParamSpec(
        "settings.rgb_to_raw_method",
        label="Spectral upsampling",
        enum=RGBtoRAWMethod,
        tooltip="Method to upsample the spectral resolution of the image, hanatos2025 works on the full visible locus, mallett2019 works only on sRGB (will clip input).",
    ),
    ParamSpec(
        "settings.apply_hanatos2025_adaptation_window",
        label="hanatos2025 adaptation window",
        tooltip="Apply the hanatos2025 bandpass adaptation window when reconstructing spectra.",
    ),
    ParamSpec(
        "settings.apply_hanatos2025_adaptation_surface",
        label="hanatos2025 adaptation surface",
        tooltip="Apply the hanatos2025 surface adaptation polynomial when reconstructing spectra.",
    ),
    ParamSpec(
        "settings.spectral_gaussian_blur",
        label="Spectral gaussian blur",
        min=0,
        step=0.1,
        tooltip="Sigma in nm for Gaussian blur applied to reconstructed spectra.",
    ),
)

# All input_image fields in declaration order; the persistence (de)serializer
# derives its field↔group mapping from these paths, so the binding lives once.
INPUT_IMAGE_FIELDS = INPUT_PANEL_FIELDS + CROP_PANEL_FIELDS + SPECTRAL_PANEL_FIELDS


SPECIAL_PANEL_FIELDS = (
    ParamSpec(
        "film_channel_swap",
        label="Film channel swap",
        min=0,
        max=2,
        step=1,
    ),
    ParamSpec(
        "print_channel_swap",
        label="Print channel swap",
        min=0,
        max=2,
        step=1,
    ),
)

SPECIAL_FIELDS = SPECIAL_PANEL_FIELDS


DISPLAY_PANEL_FIELDS = (
    ParamSpec(
        "use_display_transform",
        label="Use display transform",
        tooltip="Use Pillow.ImageCms to retrive the display transform (only in Windows) and apply it to the napari viewer output, if disabled the output color space is used",
    ),
    ParamSpec(
        "gray_18_canvas",
        label="Gray 18% canvas",
        tooltip="Use neutral 18% gray as backgroung to judge the exposure and neutral colors",
    ),
    ParamSpec(
        "white_padding",
        label="White padding",
        tooltip="Expand the white border layer around the normalized preview frame, expressed as a fraction of the image long edge.",
        min=0,
        max=1,
        step=0.01,
    ),
    ParamSpec(
        "settings.preview_max_size",
        label="Preview max size",
        tooltip="max size of the long edge of the preview image in pixels",
        min=128,
        step=128,
    ),
    ParamSpec(
        "output_interpolation",
        label="Output interpolation",
        tooltip="Napari interpolation mode used to display the output layer in the viewer.",
        enum=NapariInterpolationModes,
    ),
)


SIMULATION_PROFILE_PANEL_FIELDS = (
    ParamSpec("selection.film_stock", label="Film profile", tooltip="Film stock to simulate", enum=FilmStocks),
    ParamSpec("selection.print_paper", label="Print profile", tooltip="Print paper to simulate", enum=PrintPapers),
)

SIMULATION_SPECIAL_BORROWED_FIELDS = (
    ParamSpec(
        "enlarger.illuminant",
        name="print_illuminant",
        label="Print illuminant",
        tooltip="Print illuminant to simulate",
        enum=Illuminants,
    ),
)

SIMULATION_EXPOSURE_PANEL_FIELDS = (
    ParamSpec(
        "enlarger.print_exposure",
        name="print_exposure",
        label="Print exposure",
        tooltip="Changes the exposure time set in the virtual enlarger",
        min=0,
        step=0.02,
    ),
    ParamSpec(
        "enlarger.print_exposure_compensation",
        name="print_exposure_compensation",
        label="Print auto compensation",
        tooltip="Auto adjust the print exposure for the camera exposure compensation ev",
    ),
)

SIMULATION_ENLARGER_PANEL_FIELDS = (
    ParamSpec(
        "enlarger.y_filter_shift",
        name="print_y_filter_shift",
        label="Print Y filter shift",
        tooltip="Y filter shift of the color enlarger from a neutral position, in Kodak CC units",
        min=-200,
        step=1,
    ),
    ParamSpec(
        "enlarger.m_filter_shift",
        name="print_m_filter_shift",
        label="Print M filter shift",
        tooltip="M filter shift of the color enlarger from a neutral position, in Kodak CC units",
        min=-200,
        step=1,
    ),
)

SIMULATION_OUTPUT_PANEL_FIELDS = (
    ParamSpec("io.output_color_space", label="Output color space", tooltip="Output color space of the simulation", enum=RGBColorSpaces),
    ParamSpec("workflow.saving_color_space", label="Saving color space", tooltip="Color space of the saved image file", enum=RGBColorSpaces),
    ParamSpec("workflow.saving_cctf_encoding", label="Saving CCTF encoding", tooltip="Add or not the CCTF to the saved image file"),
)

SIMULATION_FOOTER_FIELDS = (
    ParamSpec(
        "workflow.auto_preview",
        label="Auto preview",
        tooltip="trigger the preview after every change of gui parameters, use mouse scrollwheel on parameters field, read preview tooltip for details",
    ),
    ParamSpec("io.scan_film", label="Scan film", tooltip="Show a scan of the negative instead of the print"),
)

SIMULATION_FIELDS = (
    SIMULATION_PROFILE_PANEL_FIELDS
    + SIMULATION_SPECIAL_BORROWED_FIELDS
    + SIMULATION_EXPOSURE_PANEL_FIELDS
    + SIMULATION_ENLARGER_PANEL_FIELDS
    + SIMULATION_OUTPUT_PANEL_FIELDS
    + SIMULATION_FOOTER_FIELDS
)


# Registry of every group migrated to the path-keyed manifest, for tests
# and future generic wiring.
ALL_MANIFESTS = (
    DIR_COUPLERS_MANIFEST,
    HALATION_MANIFEST,
    GRAIN_MANIFEST,
    GLARE_MANIFEST,
    CHEMISTRY_MANIFEST,
    BASE_MANIFEST,
    FILM_CHEMISTRY_MANIFEST,
    PREFLASHING_MANIFEST,
    CAMERA_MANIFEST,
    CAMERA_DIFFUSION_MANIFEST,
    ENLARGER_DIFFUSION_MANIFEST,
    SCANNER_MANIFEST,
    INPUT_GAMUT_COMPRESS_MANIFEST,
    OUTPUT_GAMUT_COMPRESS_MANIFEST,
)
