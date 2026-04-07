from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from spektrafilm_gui.options import AutoExposureMethods, RGBColorSpaces, RGBtoRAWMethod, RawWhiteBalance
from spektrafilm.model.illuminants import Illuminants
from spektrafilm.model.stocks import FilmStocks, PrintPapers


@dataclass(frozen=True, slots=True)
class WidgetSpec:
    label: str | None = None
    tooltip: str | None = None
    min_value: float | int | None = None
    max_value: float | int | None = None
    step: float | int | None = None


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
    "simulation": {
        "film_stock": FilmStocks,
        "auto_exposure_method": AutoExposureMethods,
        "print_paper": PrintPapers,
        "print_illuminant": Illuminants,
        "output_color_space": RGBColorSpaces,
        "saving_color_space": RGBColorSpaces,
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
            label="Film format mm",
            tooltip="Long edge of the film format in millimeters, e.g. 35mm or 60mm",
        ),
        "camera_lens_blur_um": WidgetSpec(
            label="Camera lens blur um",
            tooltip="Sigma of gaussian filter in um for the camera lens blur. About 5 um for typical lenses, down to 2-4 um for high quality lenses, used for sharp input simulations without lens blur.",
            step=0.05,
            min_value=0,
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
        "scan_lens_blur": WidgetSpec(
            label="Scan lens blur",
            tooltip="Sigma of gaussian filter in pixel for the scanner lens blur",
            step=0.05,
            min_value=0,
        ),
        "scan_white_correction": WidgetSpec(
            label="Scan white correction",
            tooltip="White point correction applied to the scanner output",
            min_value=0,
            max_value=1,
            step=0.01,
        ),
        "scan_black_correction": WidgetSpec(
            label="Scan black correction",
            tooltip="Black point correction applied to the scanner output",
            min_value=0,
            max_value=1,
            step=0.01,
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
        "scan_film": WidgetSpec(label="Scan film", tooltip="Show a scan of the negative instead of the print"),
        "compute_full_image": WidgetSpec(
            label="Compute full image",
            tooltip="Do not apply preview resize, compute full resolution image. Keeps the crop if active.",
        ),
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
        "white_padding": WidgetSpec(
            label="White padding",
            tooltip="Pad the simulated output on every side with a white border expressed as a fraction of the image long edge.",
            min_value=0,
            max_value=1,
            step=0.01,
        ),
    },
    "special": {
        "film_gamma_factor": WidgetSpec(
            label="Film gamma factor",
            tooltip="Gamma factor of the density curves of the negative, < 1 reduce contrast, > 1 increase contrast",
        ),
        "film_channel_swap": WidgetSpec(label="Film channel swap"),
        "print_gamma_factor": WidgetSpec(
            label="Print gamma factor",
            tooltip="Gamma factor of the print paper, < 1 reduce contrast, > 1 increase contrast",
            step=0.05,
        ),
        "print_channel_swap": WidgetSpec(label="Print channel swap"),
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
            tooltip="Gamma value of coupler inhibitors, control saturation, typical values (0.15-0.35).",
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
            tooltip="Shift the Y filter of the enlarger from the neutral position for the preflash, typical values (-20-20), in Kodak CC units",
            step=1,
        ),
        "m_filter_shift": WidgetSpec(
            tooltip="Shift the M filter of the enlarger from the neutral position for the preflash, typical values (-20-20), in Kodak CC units",
            step=1,
        ),
    },
    "input_image": {
        "preview_resize_factor": WidgetSpec(
            label="Preview resize",
            tooltip="Scale image size down (0-1) to speed up preview processing",
            step=0.1,
        ),
        "crop": WidgetSpec(label="Crop", tooltip="Crop image to a fraction of the original size to preview details at full scale"),
        "crop_center": WidgetSpec(
            label="Crop center",
            tooltip="Center of the crop region in relative coordinates in x, y (0-1)",
            step=0.02,
        ),
        "crop_size": WidgetSpec(
            label="Crop size",
            tooltip="Normalized size of the crop region in x, y (0,1), as fraction of the long side.",
            step=0.01,
        ),
        "input_color_space": WidgetSpec(
            label="Input color space",
            tooltip="Color space of the input image, will be internally converted to sRGB and negative values clipped",
        ),
        "apply_cctf_decoding": WidgetSpec(
            label="Apply CCTF decoding",
            tooltip="Apply the inverse cctf transfer function of the color space",
        ),
        "upscale_factor": WidgetSpec(label="Upscale factor", tooltip="Scale image size up to increase resolution", step=0.5),
        "spectral_upsampling_method": WidgetSpec(
            label="Spectral upsampling",
            tooltip="Method to upsample the spectral resolution of the image, hanatos2025 works on the full visible locus, mallett2019 works only on sRGB (will clip input).",
        ),
        "filter_uv": WidgetSpec(
            label="UV filter",
            tooltip="Filter UV light, (amplitude, wavelength cutoff in nm, sigma in nm). It mainly helps for avoiding UV light ruining the reds. Changing this enlarger filters neutral will be affected.",
        ),
        "filter_ir": WidgetSpec(
            label="IR filter",
            tooltip="Filter IR light, (amplitude, wavelength cutoff in nm, sigma in nm). Changing this enlarger filters neutral will be affected.",
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
        ),
        "tint": WidgetSpec(
            label="Tint",
            tooltip="Tint value for the custom white balance, not used for the other white balance settings",
            step=0.01,
        ),
        "lens_correction": WidgetSpec(label="Lens correction", tooltip="Apply lens corrections"),
    },
}


GUI_AUXILIARY_SPECS = {
    "input_layer": WidgetSpec(label="Input layer"),
    "scan_for_print": WidgetSpec(
        label="Scan for print",
        tooltip="Scan the image for print, deactivate a few virtual paper effects, ie white and black correction of the scanner are both set to 1, and glare is deactivated. Tune them yourself without this checkbox if you want to customize the look.",
    ),
}


GUI_BUTTON_SPECS = {
    "preview": ButtonSpec(
        text="PREVIEW",
        tooltip="Run the simulation in preview mode, grain and halation are deactivated for speed",
        preserve_case=True,
    ),
    "scan": ButtonSpec(
        text="SCAN",
        tooltip="Run the simulation at full resolution and with grain and halation",
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