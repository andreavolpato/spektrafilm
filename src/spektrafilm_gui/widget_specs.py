from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from spektrafilm_gui.options import (
    AutoExposureMethods,
    DiffusionFilterFamilies,
    FilmFormats,
    NapariInterpolationModes,
    RGBColorSpaces,
    RGBtoRAWMethod,
    RawWhiteBalance,
)
from spektrafilm.model.illuminants import Illuminants
from spektrafilm.model.stocks import FilmStocks, PrintPapers


@dataclass(frozen=True, slots=True)
class WidgetSpec:
    label: str | None = None
    tooltip: str | None = None
    min_value: float | int | None = None
    max_value: float | int | None = None
    step: float | int | None = None
    decimals: int | None = None


@dataclass(frozen=True, slots=True)
class ButtonSpec:
    text: str
    tooltip: str | None = None
    preserve_case: bool = False


GUI_SECTION_ENUMS: dict[str, dict[str, type[Enum]]] = {
    "input_image": {
        "input_color_space": RGBColorSpaces,
        "spectral_upsampling_method": RGBtoRAWMethod,
    },
    "load_raw": {
        "white_balance": RawWhiteBalance,
    },
    "display": {
        "output_interpolation": NapariInterpolationModes,
    },
    "simulation": {
        "film_stock": FilmStocks,
        "film_format_mm": FilmFormats,
        "auto_exposure_method": AutoExposureMethods,
        "camera_diffusion_filter_family": DiffusionFilterFamilies,
        "print_paper": PrintPapers,
        "print_illuminant": Illuminants,
        "output_color_space": RGBColorSpaces,
        "saving_color_space": RGBColorSpaces,
        "diffusion_filter_family": DiffusionFilterFamilies,
    },
}


GUI_WIDGET_SPECS = {
    "simulation": {
        "film_stock": WidgetSpec(label="Film profile", tooltip="Film stock to simulate"),
        "exposure_compensation_ev": WidgetSpec(
            label="Camera compensation ev",
            tooltip="Add a bias to the auto-exposure of the camera",
            min_value=-100,
            max_value=100,
            step=0.25,
        ),
        "auto_exposure": WidgetSpec(
            label="Camera auto exposure",
            tooltip="Use the auto-exposure feature of the virtual camera",
        ),
        "film_format_mm": WidgetSpec(
            label="Film format",
            tooltip="Long edge of the film format: 8mm Super8, 16mm, 35mm, 60mm, 70mm, 90mm, 120mm",
        ),
        "camera_lens_blur_um": WidgetSpec(
            label="Camera lens blur um",
            tooltip="Sigma of gaussian filter in um for the camera lens blur. About 5 um for typical lenses, down to 2-4 um for high quality lenses, used for sharp input simulations without lens blur.",
            step=0.05,
            min_value=0,
        ),
        "camera_diffusion_filter_active": WidgetSpec(
            label="Diffusion active",
            tooltip="Toggle the diffusion filter on the camera stage before film exposure.",
        ),
        "camera_diffusion_filter_family": WidgetSpec(
            label="Diffusion family",
            tooltip="PSF family used on the camera stage before film exposure.",
        ),
        "camera_diffusion_filter_strength": WidgetSpec(
            label="Diffusion strength",
            tooltip="Commercial filter stop: 0, 1/8=0.125, 1/4=0.25, 1/2=0.5, 1, 2.",
            min_value=0,
            max_value=2,
            step=0.125,
        ),
        "camera_diffusion_filter_spatial_scale": WidgetSpec(
            label="Spatial scale",
            tooltip="Multiplier on the camera-stage image-plane PSF widths.",
            min_value=0,
            step=0.1,
        ),
        "camera_diffusion_filter_halo_warmth": WidgetSpec(
            label="Halo warmth",
            tooltip="Additive offset on the camera-stage halo warmth axis. Positive = warm outer halo / cool inner halo. Negative inverts. 0 = use family default.",
            min_value=-1.5,
            max_value=1.5,
            step=0.05,
        ),
        "camera_diffusion_filter_core_intensity": WidgetSpec(
            label="Core intensity",
            tooltip="Advanced. Multiplier on the camera-stage core weight. 1.0 = use family default.",
            min_value=0,
            max_value=4,
            step=0.05,
        ),
        "camera_diffusion_filter_core_size": WidgetSpec(
            label="Core size",
            tooltip="Advanced. Multiplier on the camera-stage core lambda. 1.0 = use family default.",
            min_value=0.1,
            max_value=4,
            step=0.05,
        ),
        "camera_diffusion_filter_halo_intensity": WidgetSpec(
            label="Halo intensity",
            tooltip="Advanced. Multiplier on the camera-stage halo weight. 1.0 = use family default.",
            min_value=0,
            max_value=4,
            step=0.05,
        ),
        "camera_diffusion_filter_halo_size": WidgetSpec(
            label="Halo size",
            tooltip="Advanced. Multiplier on the camera-stage halo lambda. 1.0 = use family default.",
            min_value=0.1,
            max_value=4,
            step=0.05,
        ),
        "camera_diffusion_filter_bloom_intensity": WidgetSpec(
            label="Bloom intensity",
            tooltip="Advanced. Multiplier on the camera-stage bloom weight. 1.0 = use family default.",
            min_value=0,
            max_value=4,
            step=0.05,
        ),
        "camera_diffusion_filter_bloom_size": WidgetSpec(
            label="Bloom size",
            tooltip="Advanced. Multiplier on the camera-stage bloom lambda. 1.0 = use family default.",
            min_value=0.1,
            max_value=4,
            step=0.05,
        ),
        "print_paper": WidgetSpec(label="Print profile", tooltip="Print paper to simulate"),
        "print_illuminant": WidgetSpec(label="Print illuminant", tooltip="Print illuminant to simulate"),
        "print_exposure": WidgetSpec(
            label="Print exposure",
            tooltip="Changes the exposure time set in the virtual enlarger",
            step=0.02,
            min_value=0,
        ),
        "print_exposure_compensation": WidgetSpec(
            label="Print auto compensation",
            tooltip="Auto adjust the print exposure for the camera exposure compensation ev",
        ),
        "print_y_filter_shift": WidgetSpec(
            label="Print Y filter shift",
            tooltip="Y filter shift of the color enlarger from a neutral position, in Kodak CC units",
            step=1,
        ),
        "print_m_filter_shift": WidgetSpec(
            label="Print M filter shift",
            tooltip="M filter shift of the color enlarger from a neutral position, in Kodak CC units",
            step=1,
        ),
        "diffusion_filter_active": WidgetSpec(
            label="Diffusion active",
            tooltip="Toggle the diffusion filter (Pro-Mist family) on the print stage.",
        ),
        "diffusion_filter_family": WidgetSpec(
            label="Diffusion family",
            tooltip="PSF family. pro_mist / classic_soft / glimmerglass are transparent (energy-preserving); black_pro_mist absorbs a fraction of the deflected light, lifting shadows by reducing local contrast.",
        ),
        "diffusion_filter_strength": WidgetSpec(
            label="Diffusion strength",
            tooltip="Commercial filter stop: 0, 1/8=0.125, 1/4=0.25, 1/2=0.5, 1, 2. Maps internally to the (p_s, p_a) deflected/absorbed photon fractions.",
            min_value=0,
            max_value=2,
            step=0.125,
        ),
        "diffusion_filter_spatial_scale": WidgetSpec(
            label="Spatial scale",
            tooltip="Multiplier on the image-plane PSF widths (all per-group lambdas). Adjust for image-format / print-size differences.",
            min_value=0,
            step=0.1,
        ),
        "diffusion_filter_halo_warmth": WidgetSpec(
            label="Halo warmth",
            tooltip="Additive offset on the family's halo warmth axis. Positive = warm outer halo / cool inner halo (the look reference images show on mist and bloom filters). Negative inverts. Energy-preserving per channel. 0 = use family default.",
            min_value=-1.5,
            max_value=1.5,
            step=0.05,
        ),
        "diffusion_filter_core_intensity": WidgetSpec(
            label="Core intensity",
            tooltip="Advanced. Multiplier on the family's core weight. The three group weights (core, halo, bloom) are renormalized to sum to 1, so this reshuffles the relative split of energy between groups, not the total deflected fraction. 1.0 = use family default.",
            min_value=0,
            max_value=4,
            step=0.05,
        ),
        "diffusion_filter_core_size": WidgetSpec(
            label="Core size",
            tooltip="Advanced. Multiplier on the family's core lambda (all sub-components stretched together). 1.0 = use family default.",
            min_value=0.1,
            max_value=4,
            step=0.05,
        ),
        "diffusion_filter_halo_intensity": WidgetSpec(
            label="Halo intensity",
            tooltip="Advanced. Multiplier on the family's halo weight. The three group weights (core, halo, bloom) are renormalized to sum to 1. 1.0 = use family default.",
            min_value=0,
            max_value=4,
            step=0.05,
        ),
        "diffusion_filter_halo_size": WidgetSpec(
            label="Halo size",
            tooltip="Advanced. Multiplier on the family's halo lambda (all sub-components stretched together). 1.0 = use family default.",
            min_value=0.1,
            max_value=4,
            step=0.05,
        ),
        "diffusion_filter_bloom_intensity": WidgetSpec(
            label="Bloom intensity",
            tooltip="Advanced. Multiplier on the family's bloom weight. The three group weights (core, halo, bloom) are renormalized to sum to 1. 1.0 = use family default.",
            min_value=0,
            max_value=4,
            step=0.05,
        ),
        "diffusion_filter_bloom_size": WidgetSpec(
            label="Bloom size",
            tooltip="Advanced. Multiplier on the family's bloom lambda (all sub-components stretched together). 1.0 = use family default.",
            min_value=0.1,
            max_value=4,
            step=0.05,
        ),
        "scan_lens_blur": WidgetSpec(
            label="Scan lens blur",
            tooltip="Sigma of gaussian filter in pixel for the scanner lens blur",
            step=0.05,
            min_value=0,
        ),
        "scan_white_correction": WidgetSpec(
            label="Scan white correction",
            tooltip="Enable white point correction applied to the scanner output",
        ),
        "scan_white_level": WidgetSpec(
            label="Scan white level",
            tooltip="Target white level applied when white correction is enabled",
            min_value=0,
            max_value=1,
            step=0.005,
            decimals=3,
        ),
        "scan_black_correction": WidgetSpec(
            label="Scan black correction",
            tooltip="Enable black point correction applied to the scanner output",
        ),
        "scan_black_level": WidgetSpec(
            label="Scan black level",
            tooltip="Target black level applied when black correction is enabled",
            min_value=0,
            max_value=1,
            step=0.005,
            decimals=3,
        ),
        "scan_unsharp_mask": WidgetSpec(
            label="Scan unsharp mask",
            tooltip="Apply unsharp mask to the scan, [sigma in pixel, amount]",
            step=0.05,
            min_value=0,
        ),
        "output_color_space": WidgetSpec(label="Output color space", tooltip="Output color space of the simulation"),
        "saving_color_space": WidgetSpec(label="Saving color space", tooltip="Color space of the saved image file"),
        "saving_cctf_encoding": WidgetSpec(
            label="Saving CCTF encoding",
            tooltip="Add or not the CCTF to the saved image file",
        ),
        "auto_preview": WidgetSpec(label="Auto preview", tooltip="trigger the preview after every change of gui parameters, use mouse scrollwheel on parameters field, read preview tooltip for details"),
        "scan_film": WidgetSpec(label="Scan film", tooltip="Show a scan of the negative instead of the print"),
    },
    "display": {
        "use_display_transform": WidgetSpec(
            label="Use display transform",
            tooltip="Use Pillow.ImageCms to retrive the display transform (only in Windows) and apply it to the napari viewer output, if disabled the output color space is used",
        ),
        "gray_18_canvas": WidgetSpec(
            label="Gray 18% canvas",
            tooltip="Use neutral 18% gray as backgroung to judge the exposure and neutral colors",
        ),
        "output_interpolation": WidgetSpec(
            label="Output interpolation",
            tooltip="Napari interpolation mode used to display the output layer in the viewer.",
        ),
        "white_padding": WidgetSpec(
            label="White padding",
            tooltip="Expand the white border layer around the normalized preview frame, expressed as a fraction of the image long edge.",
            min_value=0,
            max_value=1,
            step=0.01,
        ),
        "preview_max_size": WidgetSpec(
            label="Preview max size",
            tooltip="max size of the long edge of the preview image in pixels",
            min_value=128,
            max_value=1024,
            step=128,
        ),
    },
    "special": {
        "film_gamma_factor": WidgetSpec(
            label="Film gamma factor",
            tooltip="Gamma factor of the density curves of the negative, < 1 reduce contrast, > 1 increase contrast",
            step=0.05,
            min_value=0,
        ),
        "film_channel_swap": WidgetSpec(label="Film channel swap"),
        "print_gamma_factor": WidgetSpec(
            label="Print gamma factor",
            tooltip="Gamma factor of the print paper, < 1 reduce contrast, > 1 increase contrast",
            step=0.05,
            min_value=0,
        ),
        "print_channel_swap": WidgetSpec(label="Print channel swap",
        min_value=0,
        max_value=2,
        step=1
        )
    },
    "glare": {
        "active": WidgetSpec(tooltip="Add glare to the print"),
        "percent": WidgetSpec(
            tooltip="Percentage of the glare light (typically 0.1-0.25)",
            step=0.01,
            min_value=0,
            max_value=1,
        ),
        "roughness": WidgetSpec(
            tooltip="Roughness of the glare light (0-1)",
            min_value=0,
            max_value=1,
            step=0.05,
        ),
        "blur": WidgetSpec(
            tooltip="Sigma of gaussian blur in pixels for the glare",
            min_value=0,
            step=0.1,
        ),
    },
    "halation": {
        "scatter_amount": WidgetSpec(
            tooltip="High-level scatter strength. 1.0 = full physical scatter, 0.0 = no scatter. Scales the fraction of light that undergoes in-emulsion scattering.",
            min_value=0,
            step=0.05,
        ),
        "scatter_spatial_scale": WidgetSpec(
            tooltip="High-level scatter size multiplier (1.0 = physical defaults). Scales both core and tail sigmas.",
            min_value=0,
            step=0.1,
        ),
        "halation_amount": WidgetSpec(
            tooltip="High-level halation strength multiplier (1.0 = physical defaults). Scales the per-channel halation amplitudes.",
            min_value=0,
            step=0.05,
        ),
        "halation_spatial_scale": WidgetSpec(
            tooltip="High-level halation size multiplier (1.0 = physical defaults). Scales the first-bounce sigma.",
            min_value=0,
            step=0.1,
        ),
        "boost_ev": WidgetSpec(
            tooltip="Maximum highlight boost in stops.",
            min_value=0,
            step=0.5,
        ),
        "protect_ev": WidgetSpec(
            tooltip="Protected range above midgray for the boost onset in stops.",
            min_value=0,
            step=0.5,
        ),
        "boost_range": WidgetSpec(
            tooltip="Controls how quickly the highlight boost ramps in, from 0 to 1.",
            min_value=0,
            max_value=1,
            step=0.05,
        ),
        "scatter_core_um": WidgetSpec(
            tooltip="Sigma of the scatter core Gaussian per channel [R,G,B], in micrometers. Controls fine-scale sharpness loss in the emulsion.",
            min_value=0,
            step=0.5,
        ),
        "scatter_tail_um": WidgetSpec(
            tooltip="Decay constant of the scatter exponential tail per channel [R,G,B], in micrometers (internally approximated by a sum of Gaussians). Controls extended low-level spread within the emulsion.",
            min_value=0,
            step=1.0,
        ),
        "scatter_tail_weight": WidgetSpec(
            tooltip="Weight of the scatter tail Gaussian per channel [R,G,B] (0-100, percentage). Tail weight + core weight = 100. Higher values put more scattered light into the long tail.",
            min_value=0,
            max_value=100,
            step=1,
        ),
        "halation_strength": WidgetSpec(
            tooltip="Total back-reflection halation amplitude per channel [R,G,B] (0-100, percentage). Typical red channel: weak AH 2-8, no AH 8-25. The blue channel is usually near zero.",
            min_value=0,
            max_value=100,
            step=0.5,
        ),
        "halation_first_sigma_um": WidgetSpec(
            tooltip="Sigma of the first halation bounce per channel [R,G,B], in micrometers. Set by the base thickness (40-80 um for typical cine/still bases).",
            min_value=0,
            step=1.0,
        ),
        "halation_n_bounces": WidgetSpec(
            tooltip="Number of multi-bounce Gaussians summed in the halation pass. Subsequent bounces use sqrt(k)-spaced widths. Typical: 2-3.",
            min_value=1,
            max_value=5,
            step=1,
        ),
        "halation_bounce_decay": WidgetSpec(
            tooltip="Per-bounce amplitude decay ratio (rho). Physical range 0.3-0.7. Controls how fast the halation energy falls off between bounces.",
            min_value=0,
            max_value=1,
            step=0.05,
        ),
        "halation_renormalize": WidgetSpec(
            tooltip="If enabled, divide by (1 + sum of bounce amplitudes) so mid-grey is preserved. If disabled, halation is purely additive and subtly lifts shadows as well as highlights.",
        ),
    },
    "couplers": {
        "amount": WidgetSpec(
            tooltip="Global multiplier on the DIR coupler inhibition matrix. 1.0 leaves the per-channel gammas as-is.",
            min_value=0,
            step=0.05,
        ),
        "inhibition_samelayer": WidgetSpec(
            tooltip="Multiplier on the same-layer (diagonal) inhibition. Controls overall contrast / gamma reduction within each RGB layer.",
            min_value=0,
            step=0.05,
        ),
        "inhibition_interlayer": WidgetSpec(
            tooltip="Multiplier on the cross-layer (off-diagonal) inhibition. Controls saturation enhancement from interlayer DIR effects.",
            min_value=0,
            step=0.05,
        ),
        "gamma_samelayer_rgb": WidgetSpec(
            tooltip="Per-channel same-layer DIR gamma (R, G, B). Effective gamma reduction of each layer's density curve.",
            min_value=0,
            step=0.02,
        ),
        "gamma_interlayer_r_to_gb": WidgetSpec(
            tooltip="DIR inhibition from the R layer onto the G and B layers respectively (g_R->G, g_R->B).",
            min_value=0,
            step=0.02,
        ),
        "gamma_interlayer_g_to_rb": WidgetSpec(
            tooltip="DIR inhibition from the G layer onto the R and B layers respectively (g_G->R, g_G->B).",
            min_value=0,
            step=0.02,
        ),
        "gamma_interlayer_b_to_rg": WidgetSpec(
            tooltip="DIR inhibition from the B layer onto the R and G layers respectively (g_B->R, g_B->G).",
            min_value=0,
            step=0.02,
        ),
        "diffusion_size_um": WidgetSpec(
            tooltip="Sigma in um for the diffusion of the couplers, (5-20 um), controls sharpness and affects saturation.",
            min_value=0,
            step=5,
        ),
    },
    "grain": {
        "active": WidgetSpec(tooltip="Add grain to the negative"),
        "particle_area_um2": WidgetSpec(
            tooltip="Area of the particles in um2, relates to ISO. Approximately 0.1 for ISO 100, 0.1 for ISO 200, 0.4 for ISO 400 and so on.",
            step=0.2,
            min_value=0,
        ),
        "particle_scale": WidgetSpec(tooltip="Scale of particle area for the RGB layers, multiplies particle_area_um2"),
        "particle_scale_layers": WidgetSpec(
            tooltip="Scale of particle area for the sublayers in every color layer, multiplies particle_area_um2",
            min_value=0,
            step=0.25,
        ),
        "density_min": WidgetSpec(tooltip="Minimum density of the grain, typical values (0.03-0.06)"),
        "uniformity": WidgetSpec(tooltip="Uniformity of the grain, typical values (0.94-0.98)"),
        "blur": WidgetSpec(
            tooltip="Sigma of gaussian blur in pixels for the grain, to be increased at high magnifications, (should be 0.8-0.9 at high resolution, reduce down to 0.6 for lower res).",
            min_value=0,
            step=0.05,
        ),
        "blur_dye_clouds_um": WidgetSpec(
            tooltip="Scale the sigma of gaussian blur in um for the dye clouds, to be used at high magnifications, (default 1)",
            min_value=0,
            step=0.1,
        ),
        "micro_structure": WidgetSpec(
            tooltip="Parameter for micro-structure due to clumps at the molecular level, [sigma blur of micro-structure / ultimate light-resolution (0.10 um default), size of molecular clumps in nm (30 nm default)]. Only for insane magnifications.",
            min_value=0,
            step=0.1,
        ),
    },
    "preflashing": {
        "exposure": WidgetSpec(
            tooltip="Preflash exposure value in ev for the print",
            step=0.005,
            min_value=0,
        ),
        "y_filter_shift": WidgetSpec(
            tooltip="Shift the Y filter of the enlarger from the neutral position for the preflash, typical values (-20-20), in Kodak CC units",
            step=1,
        ),
        "m_filter_shift": WidgetSpec(
            tooltip="Shift the M filter of the enlarger from the neutral position for the preflash, typical values (-20-20), in Kodak CC units",
            step=1,
        ),
    },
    "input_image": {
        "crop": WidgetSpec(label="Crop", tooltip="Crop image to a fraction of the original size to preview details at full scale"),
        "crop_center": WidgetSpec(
            label="Crop center",
            tooltip="Center of the crop region in relative coordinates in x, y (0-1)",
            step=0.01,
            min_value=0,
            max_value=1,
        ),
        "crop_size": WidgetSpec(
            label="Crop size",
            tooltip="Normalized size of the crop region in x, y (0,1), as fraction of the long side.",
            step=0.01,
            min_value=0,
            max_value=1,
        ),
        "input_color_space": WidgetSpec(
            label="Input color space",
            tooltip="Color space of the input image, will be internally converted to sRGB and negative values clipped",
        ),
        "apply_cctf_decoding": WidgetSpec(
            label="Apply CCTF decoding",
            tooltip="Apply the inverse cctf transfer function of the color space",
        ),
        "upscale_factor": WidgetSpec(label="Upscale factor", tooltip="Scale image size up to increase resolution",
                                     min_value=0.0,
                                     step=0.5,
                                     ),
        "spectral_upsampling_method": WidgetSpec(
            label="Spectral upsampling",
            tooltip="Method to upsample the spectral resolution of the image, hanatos2025 works on the full visible locus, mallett2019 works only on sRGB (will clip input).",
        ),
        "apply_hanatos2025_adaptation_window": WidgetSpec(
            label="hanatos2025 adaptation window",
            tooltip="Apply the hanatos2025 bandpass adaptation window when reconstructing spectra.",
        ),
        "apply_hanatos2025_adaptation_surface": WidgetSpec(
            label="hanatos2025 adaptation surface",
            tooltip="Apply the hanatos2025 surface adaptation polynomial when reconstructing spectra.",
        ),
        "spectral_gaussian_blur": WidgetSpec(
            label="Spectral gaussian blur",
            tooltip="Sigma in nm for Gaussian blur applied to reconstructed spectra.",
            min_value=0,
            step=0.1,
        ),
        "filter_uv": WidgetSpec(
            label="UV filter",
            tooltip="Filter UV light, (amplitude, wavelength cutoff in nm, sigma in nm). It mainly helps for avoiding UV light ruining the reds. Changing this enlarger filters neutral will be affected.",
            min_value=0,
            step=1,
        ),
        "filter_ir": WidgetSpec(
            label="IR filter",
            tooltip="Filter IR light, (amplitude, wavelength cutoff in nm, sigma in nm). Changing this enlarger filters neutral will be affected.",
            min_value=0,
            step=1,
        ),
    },
    "load_raw": {
        "white_balance": WidgetSpec(
            label="White balance",
            tooltip="Select white balance settings, if custom you can tune temperature and tint",
        ),
        "temperature": WidgetSpec(
            label="Temperature",
            tooltip="Temperature in Kelvin for the custom whitebalance, not used for the other white balance settings",
            step=100,
            min_value=1000,
        ),
        "tint": WidgetSpec(
            label="Tint",
            tooltip="Tint value for the custom white balance, not used for the other white balance settings",
            min_value=0,
            step=0.01,
        ),
        "lens_correction": WidgetSpec(label="Lens correction", tooltip="Apply lens corrections"),
    },
}


GUI_AUXILIARY_SPECS = {
    "scan_for_print": WidgetSpec(
        label="Scan for print",
        tooltip="Scan the image for print, ie white and black correction of the scanner are active, and glare is deactivated.",
    ),
}


GUI_BUTTON_SPECS = {
    "preview": ButtonSpec(
        text="PREVIEW",
        tooltip="run the simulation on a small preview and deactivates grain, halation, blurs, unsharp mask (diffusion filters are active)",
        preserve_case=True,
    ),
    "scan": ButtonSpec(
        text="SCAN",
        tooltip="Run the full simulation on the full-resolution input",
        preserve_case=True,
    ),
    "save": ButtonSpec(
        text="SAVE",
        tooltip="Save the current output layer to an image file",
        preserve_case=True,
    ),
}


EMPTY_WIDGET_SPEC = WidgetSpec()


def get_widget_spec(section_name: str, field_name: str) -> WidgetSpec:
    return GUI_WIDGET_SPECS.get(section_name, {}).get(field_name, EMPTY_WIDGET_SPEC)


def get_auxiliary_spec(name: str) -> WidgetSpec:
    return GUI_AUXILIARY_SPECS.get(name, EMPTY_WIDGET_SPEC)


def get_button_spec(name: str) -> ButtonSpec:
    return GUI_BUTTON_SPECS[name]