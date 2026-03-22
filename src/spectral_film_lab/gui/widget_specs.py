from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from spectral_film_lab.config import ENLARGER_STEPS
from spectral_film_lab.gui.options import AutoExposureMethods, RGBColorSpaces, RGBtoRAWMethod
from spectral_film_lab.model.illuminants import Illuminants
from spectral_film_lab.model.stocks import FilmStocks, PrintPapers


@dataclass(frozen=True, slots=True)
class WidgetSpec:
    tooltip: str | None = None
    min_value: float | int | None = None
    max_value: float | int | None = None
    step: float | int | None = None


GUI_SECTION_ENUMS: dict[str, dict[str, type[Enum]]] = {
    "input_image": {
        "input_color_space": RGBColorSpaces,
        "spectral_upsampling_method": RGBtoRAWMethod,
    },
    "simulation": {
        "film_stock": FilmStocks,
        "auto_exposure_method": AutoExposureMethods,
        "print_paper": PrintPapers,
        "print_illuminant": Illuminants,
        "output_color_space": RGBColorSpaces,
    },
}


GUI_WIDGET_SPECS = {
    "simulation": {
        "film_stock": WidgetSpec(tooltip="Film stock to simulate"),
        "exposure_compensation_ev": WidgetSpec(
            tooltip="Exposure compensation value in ev of the negative",
            min_value=-100,
            max_value=100,
            step=0.5,
        ),
        "auto_exposure": WidgetSpec(tooltip="Automatically adjust exposure based on the image content"),
        "film_format_mm": WidgetSpec(tooltip="Long edge of the film format in millimeters, e.g. 35mm or 60mm"),
        "camera_lens_blur_um": WidgetSpec(
            tooltip="Sigma of gaussian filter in um for the camera lens blur. About 5 um for typical lenses, down to 2-4 um for high quality lenses, used for sharp input simulations without lens blur.",
        ),
        "print_paper": WidgetSpec(tooltip="Print paper to simulate"),
        "print_illuminant": WidgetSpec(tooltip="Print illuminant to simulate"),
        "print_exposure": WidgetSpec(
            tooltip="Exposure value for the print (proportional to seconds of exposure, not ev)",
            step=0.05,
        ),
        "print_exposure_compensation": WidgetSpec(
            tooltip="Apply exposure compensation from negative exposure compensation ev, allow for changing of the negative exposure compensation while keeping constant print time.",
        ),
        "print_y_filter_shift": WidgetSpec(
            tooltip="Y filter shift of the color enlarger from a neutral position, enlarger has 170 steps",
            min_value=-ENLARGER_STEPS,
            max_value=ENLARGER_STEPS,
        ),
        "print_m_filter_shift": WidgetSpec(
            tooltip="M filter shift of the color enlarger from a neutral position, enlarger has 170 steps",
            min_value=-ENLARGER_STEPS,
            max_value=ENLARGER_STEPS,
        ),
        "scan_lens_blur": WidgetSpec(
            tooltip="Sigma of gaussian filter in pixel for the scanner lens blur",
            step=0.05,
        ),
        "scan_unsharp_mask": WidgetSpec(tooltip="Apply unsharp mask to the scan, [sigma in pixel, amount]"),
        "output_color_space": WidgetSpec(tooltip="Color space of the output image"),
        "output_cctf_encoding": WidgetSpec(
            tooltip="Apply the cctf transfer function of the color space. If false, data is linear.",
        ),
        "scan_film": WidgetSpec(tooltip="Show a scan of the negative instead of the print"),
        "compute_full_image": WidgetSpec(
            tooltip="Do not apply preview resize, compute full resolution image. Keeps the crop if active.",
        ),
    },
    "special": {
        "film_gamma_factor": WidgetSpec(
            tooltip="Gamma factor of the density curves of the negative, < 1 reduce contrast, > 1 increase contrast",
        ),
        "print_gamma_factor": WidgetSpec(
            tooltip="Gamma factor of the print paper, < 1 reduce contrast, > 1 increase contrast",
            step=0.05,
        ),
        "print_density_min_factor": WidgetSpec(
            tooltip="Minimum density factor of the print paper (0-1), make the white less white",
            min_value=0,
            max_value=1,
            step=0.2,
        ),
    },
    "glare": {
        "active": WidgetSpec(tooltip="Add glare to the print"),
        "percent": WidgetSpec(
            tooltip="Percentage of the glare light (typically 0.1-0.25)",
            step=0.05,
        ),
        "roughness": WidgetSpec(tooltip="Roughness of the glare light (0-1)"),
        "blur": WidgetSpec(tooltip="Sigma of gaussian blur in pixels for the glare"),
        "compensation_removal_factor": WidgetSpec(
            tooltip="Factor of glare compensation removal from the print, e.g. 0.2=20% underexposed print in the shadows, typical values (0.0-0.2). To be used instead of stochastic glare (i.e. when percent=0).",
            step=0.05,
        ),
        "compensation_removal_density": WidgetSpec(
            tooltip="Density of the glare compensation removal from the print, typical values (1.0-1.5).",
        ),
        "compensation_removal_transition": WidgetSpec(
            tooltip="Transition density range of the glare compensation removal from the print, typical values (0.1-0.5).",
        ),
    },
    "halation": {
        "scattering_strength": WidgetSpec(
            tooltip="Fraction of scattered light (0-100, percentage) for each channel [R,G,B]",
        ),
        "scattering_size_um": WidgetSpec(
            tooltip="Size of the scattering effect in micrometers for each channel [R,G,B], sigma of gaussian filter.",
        ),
        "halation_strength": WidgetSpec(
            tooltip="Fraction of halation light (0-100, percentage) for each channel [R,G,B]",
        ),
        "halation_size_um": WidgetSpec(
            tooltip="Size of the halation effect in micrometers for each channel [R,G,B], sigma of gaussian filter.",
        ),
    },
    "couplers": {
        "dir_couplers_amount": WidgetSpec(
            tooltip="Amount of coupler inhibitors, control saturation, typical values (0.8-1.2).",
            step=0.05,
        ),
        "dir_couplers_diffusion_um": WidgetSpec(
            tooltip="Sigma in um for the diffusion of the couplers, (5-20 um), controls sharpness and affects saturation.",
            step=5,
        ),
        "diffusion_interlayer": WidgetSpec(
            tooltip="Sigma in number of layers for diffusion across the rgb layers (typical layer thickness 3-5 um, so roughly 1.0-4.0 layers), affects saturation.",
        ),
    },
    "grain": {
        "active": WidgetSpec(tooltip="Add grain to the negative"),
        "particle_area_um2": WidgetSpec(
            tooltip="Area of the particles in um2, relates to ISO. Approximately 0.1 for ISO 100, 0.1 for ISO 200, 0.4 for ISO 400 and so on.",
            step=0.1,
        ),
        "particle_scale": WidgetSpec(tooltip="Scale of particle area for the RGB layers, multiplies particle_area_um2"),
        "particle_scale_layers": WidgetSpec(
            tooltip="Scale of particle area for the sublayers in every color layer, multiplies particle_area_um2",
        ),
        "density_min": WidgetSpec(tooltip="Minimum density of the grain, typical values (0.03-0.06)"),
        "uniformity": WidgetSpec(tooltip="Uniformity of the grain, typical values (0.94-0.98)"),
        "blur": WidgetSpec(
            tooltip="Sigma of gaussian blur in pixels for the grain, to be increased at high magnifications, (should be 0.8-0.9 at high resolution, reduce down to 0.6 for lower res).",
        ),
        "blur_dye_clouds_um": WidgetSpec(
            tooltip="Scale the sigma of gaussian blur in um for the dye clouds, to be used at high magnifications, (default 1)",
        ),
        "micro_structure": WidgetSpec(
            tooltip="Parameter for micro-structure due to clumps at the molecular level, [sigma blur of micro-structure / ultimate light-resolution (0.10 um default), size of molecular clumps in nm (30 nm default)]. Only for insane magnifications.",
        ),
    },
    "preflashing": {
        "exposure": WidgetSpec(
            tooltip="Preflash exposure value in ev for the print",
            step=0.005,
        ),
        "just_preflash": WidgetSpec(tooltip="Only apply preflash to the print, to visualize the preflash effect"),
        "y_filter_shift": WidgetSpec(
            tooltip="Shift the Y filter of the enlarger from the neutral position for the preflash, typical values (-20-20), enlarger has 170 steps",
            min_value=-ENLARGER_STEPS,
        ),
        "m_filter_shift": WidgetSpec(
            tooltip="Shift the M filter of the enlarger from the neutral position for the preflash, typical values (-20-20), enlarger has 170 steps",
            min_value=-ENLARGER_STEPS,
        ),
    },
    "input_image": {
        "preview_resize_factor": WidgetSpec(tooltip="Scale image size down (0-1) to speed up preview processing"),
        "crop": WidgetSpec(tooltip="Crop image to a fraction of the original size to preview details at full scale"),
        "crop_center": WidgetSpec(tooltip="Center of the crop region in relative coordinates in x, y (0-1)"),
        "crop_size": WidgetSpec(
            tooltip="Normalized size of the crop region in x, y (0,1), as fraction of the long side.",
        ),
        "input_color_space": WidgetSpec(
            tooltip="Color space of the input image, will be internally converted to sRGB and negative values clipped",
        ),
        "apply_cctf_decoding": WidgetSpec(
            tooltip="Apply the inverse cctf transfer function of the color space",
        ),
        "upscale_factor": WidgetSpec(tooltip="Scale image size up to increase resolution"),
        "spectral_upsampling_method": WidgetSpec(
            tooltip="Method to upsample the spectral resolution of the image, hanatos2025 works on the full visible locus, mallett2019 works only on sRGB (will clip input).",
        ),
        "filter_uv": WidgetSpec(
            tooltip="Filter UV light, (amplitude, wavelength cutoff in nm, sigma in nm). It mainly helps for avoiding UV light ruining the reds. Changing this enlarger filters neutral will be affected.",
        ),
        "filter_ir": WidgetSpec(
            tooltip="Filter IR light, (amplitude, wavelength cutoff in nm, sigma in nm). Changing this enlarger filters neutral will be affected.",
        ),
    },
}