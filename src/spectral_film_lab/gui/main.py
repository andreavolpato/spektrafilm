import numpy as np
import napari
import json
from napari.layers import Image
from napari.types import ImageData
from napari.settings import get_settings
from magicgui import magicgui
from pathlib import Path
# import matplotlib.pyplot as plt

from spectral_film_lab.config import ENLARGER_STEPS
from spectral_film_lab.gui.magicgui_bridge import (
    apply_gui_state,
    collect_gui_state,
    DEFAULT_AUTO_EXPOSURE_METHOD,
    DEFAULT_FILM_STOCK_MEMBER,
    DEFAULT_GUI_STATE,
    DEFAULT_INPUT_COLOR_SPACE,
    DEFAULT_OUTPUT_COLOR_SPACE,
    DEFAULT_PRINT_ILLUMINANT,
    DEFAULT_PRINT_PAPER_MEMBER,
    DEFAULT_SPECTRAL_UPSAMPLING_METHOD,
    MagicGuiWidgets,
)
from spectral_film_lab.gui.params_mapper import build_params_from_state
from spectral_film_lab.utils.io import load_image_oiio
from spectral_film_lab.runtime.process import photo_process
from spectral_film_lab.profiles.io import profile_to_dict
from spectral_film_lab.utils.numba_warmup import warmup

# precompile numba functions
warmup()

# create a viewer and add a couple image layers
viewer = napari.Viewer()
viewer.window._qt_window.showMaximized()
viewer.window._qt_viewer.dockLayerControls.setVisible(False)
viewer.window._qt_viewer.dockLayerList.setVisible(False)
layer_list = viewer.window._qt_viewer.dockLayerList
settings = get_settings()
settings.appearance.theme = 'light'

# portrait = load_image_oiio('img/test/portrait_leaves_32bit_linear_prophoto_rgb.tif')
# viewer.add_image(portrait,
#                  name="portrait")

@magicgui(layout="vertical", call_button='None')
def grain(active=DEFAULT_GUI_STATE.grain.active,
          sublayers_active=DEFAULT_GUI_STATE.grain.sublayers_active,
          particle_area_um2=DEFAULT_GUI_STATE.grain.particle_area_um2,
          particle_scale=DEFAULT_GUI_STATE.grain.particle_scale,
          particle_scale_layers=DEFAULT_GUI_STATE.grain.particle_scale_layers,
          density_min=DEFAULT_GUI_STATE.grain.density_min,
          uniformity=DEFAULT_GUI_STATE.grain.uniformity,
          blur=DEFAULT_GUI_STATE.grain.blur,
          blur_dye_clouds_um=DEFAULT_GUI_STATE.grain.blur_dye_clouds_um,
          micro_structure=DEFAULT_GUI_STATE.grain.micro_structure,
          ):
    return

@magicgui(layout="vertical", call_button='None')
def input_image(preview_resize_factor=DEFAULT_GUI_STATE.input_image.preview_resize_factor,
                upscale_factor=DEFAULT_GUI_STATE.input_image.upscale_factor,
                crop=DEFAULT_GUI_STATE.input_image.crop,
                crop_center=DEFAULT_GUI_STATE.input_image.crop_center,
                crop_size=DEFAULT_GUI_STATE.input_image.crop_size,
                input_color_space=DEFAULT_INPUT_COLOR_SPACE,
                apply_cctf_decoding=DEFAULT_GUI_STATE.input_image.apply_cctf_decoding,
                spectral_upsampling_method=DEFAULT_SPECTRAL_UPSAMPLING_METHOD,
                filter_uv=DEFAULT_GUI_STATE.input_image.filter_uv,
                filter_ir=DEFAULT_GUI_STATE.input_image.filter_ir,
                ):
    return

@magicgui(layout="vertical", call_button='None')
def preflashing(exposure=DEFAULT_GUI_STATE.preflashing.exposure,
                y_filter_shift=DEFAULT_GUI_STATE.preflashing.y_filter_shift,
                m_filter_shift=DEFAULT_GUI_STATE.preflashing.m_filter_shift,
                just_preflash=DEFAULT_GUI_STATE.preflashing.just_preflash):
    return

@magicgui(layout="vertical", call_button='None')
def halation(active=DEFAULT_GUI_STATE.halation.active,
             scattering_strength=DEFAULT_GUI_STATE.halation.scattering_strength,
             scattering_size_um=DEFAULT_GUI_STATE.halation.scattering_size_um,
             halation_strength=DEFAULT_GUI_STATE.halation.halation_strength,
             halation_size_um=DEFAULT_GUI_STATE.halation.halation_size_um):
    return

@magicgui(layout="vertical", call_button='None')
def couplers(active=DEFAULT_GUI_STATE.couplers.active,
             dir_couplers_amount=DEFAULT_GUI_STATE.couplers.dir_couplers_amount,
             dir_couplers_ratio=DEFAULT_GUI_STATE.couplers.dir_couplers_ratio,
             dir_couplers_diffusion_um=DEFAULT_GUI_STATE.couplers.dir_couplers_diffusion_um,
             diffusion_interlayer=DEFAULT_GUI_STATE.couplers.diffusion_interlayer,
             high_exposure_shift=DEFAULT_GUI_STATE.couplers.high_exposure_shift):
    return

@magicgui(layout="vertical", call_button='None')
def glare(active=DEFAULT_GUI_STATE.glare.active,
          percent=DEFAULT_GUI_STATE.glare.percent,
          roughness=DEFAULT_GUI_STATE.glare.roughness,
          blur=DEFAULT_GUI_STATE.glare.blur,
          compensation_removal_factor=DEFAULT_GUI_STATE.glare.compensation_removal_factor,
          compensation_removal_density=DEFAULT_GUI_STATE.glare.compensation_removal_density,
          compensation_removal_transition=DEFAULT_GUI_STATE.glare.compensation_removal_transition):
    return

@magicgui(filename={"mode": "r"}, call_button='load image (e.g. png/exr)')
def filepicker(filename=Path("./")) -> ImageData:
    img_array = load_image_oiio(str(filename))
    img_array = img_array[...,:3]
    return img_array

@magicgui(layout="vertical", call_button='None')
def special(film_channel_swap=DEFAULT_GUI_STATE.special.film_channel_swap,
        film_gamma_factor=DEFAULT_GUI_STATE.special.film_gamma_factor,
        print_channel_swap=DEFAULT_GUI_STATE.special.print_channel_swap,
        print_gamma_factor=DEFAULT_GUI_STATE.special.print_gamma_factor,
        print_density_min_factor=DEFAULT_GUI_STATE.special.print_density_min_factor,
            ):
    return

def export_parameters(filepath, params):
    with open(filepath, 'w') as f:
        json.dump(profile_to_dict(params), f, indent=4)

def load_parameters(filepath):
    with open(filepath, 'r') as f:
        params = json.load(f)
    return params

# for details on why the `-> ImageData` return annotation works:
# https://napari.org/guides/magicgui.html#return-annotations
@magicgui(layout="vertical")
def simulation(input_layer:Image,
                    film_stock=DEFAULT_FILM_STOCK_MEMBER,
                    film_format_mm=DEFAULT_GUI_STATE.simulation.film_format_mm,
                    camera_lens_blur_um=DEFAULT_GUI_STATE.simulation.camera_lens_blur_um,
                    exposure_compensation_ev=DEFAULT_GUI_STATE.simulation.exposure_compensation_ev,
                    auto_exposure=DEFAULT_GUI_STATE.simulation.auto_exposure,
                    auto_exposure_method=DEFAULT_AUTO_EXPOSURE_METHOD,
               # print parameters
                    print=DEFAULT_PRINT_PAPER_MEMBER,
                    print_illuminant=DEFAULT_PRINT_ILLUMINANT,
                    print_exposure=DEFAULT_GUI_STATE.simulation.print_exposure,
                    print_exposure_compensation=DEFAULT_GUI_STATE.simulation.print_exposure_compensation,
                    print_y_filter_shift=DEFAULT_GUI_STATE.simulation.print_y_filter_shift,
                    print_m_filter_shift=DEFAULT_GUI_STATE.simulation.print_m_filter_shift,
            #    print_lens_blur=0.0,
               # scanner
                    scan_lens_blur=DEFAULT_GUI_STATE.simulation.scan_lens_blur,
                    scan_unsharp_mask=DEFAULT_GUI_STATE.simulation.scan_unsharp_mask,
                    output_color_space=DEFAULT_OUTPUT_COLOR_SPACE,
                    output_cctf_encoding=DEFAULT_GUI_STATE.simulation.output_cctf_encoding,
            #    return_film_log_raw=False,
                    scan_film=DEFAULT_GUI_STATE.simulation.scan_film,
                    compute_full_image=DEFAULT_GUI_STATE.simulation.compute_full_image,
               )->ImageData:    
    state = collect_gui_state(
        widgets=GUI_WIDGETS,
        film_stock=film_stock,
        film_format_mm=film_format_mm,
        camera_lens_blur_um=camera_lens_blur_um,
        exposure_compensation_ev=exposure_compensation_ev,
        auto_exposure=auto_exposure,
        auto_exposure_method=auto_exposure_method,
        print_paper=print,
        print_illuminant=print_illuminant,
        print_exposure=print_exposure,
        print_exposure_compensation=print_exposure_compensation,
        print_y_filter_shift=print_y_filter_shift,
        print_m_filter_shift=print_m_filter_shift,
        scan_lens_blur=scan_lens_blur,
        scan_unsharp_mask=scan_unsharp_mask,
        output_color_space=output_color_space,
        output_cctf_encoding=output_cctf_encoding,
        scan_film=scan_film,
        compute_full_image=compute_full_image,
    )
    params = build_params_from_state(state)

    image = np.double(input_layer.data[:,:,:3])
    scan = photo_process(image, params)
    # if params.debug.return_film_log_raw:
    #     scan = np.vstack((scan[:, :, 0], scan[:, :, 1], scan[:, :, 2]))
    scan = np.uint8(scan*255)
    return scan


GUI_WIDGETS = MagicGuiWidgets(
    input_image=input_image,
    grain=grain,
    preflashing=preflashing,
    halation=halation,
    couplers=couplers,
    glare=glare,
    special=special,
    simulation=simulation,
)


def _configure_widget(widget, *, tooltip=None, min_value=None, max_value=None, step=None):
    if tooltip is not None:
        widget.tooltip = tooltip
    if min_value is not None:
        widget.min = min_value
    if max_value is not None:
        widget.max = max_value
    if step is not None:
        widget.step = step


def _configure_widget_controls():
    _configure_widget(
        simulation.film_stock,
        tooltip='Film stock to simulate',
    )
    _configure_widget(
        simulation.exposure_compensation_ev,
        tooltip='Exposure compensation value in ev of the negative',
        min_value=-100,
        max_value=100,
        step=0.5,
    )
    _configure_widget(
        simulation.auto_exposure,
        tooltip='Automatically adjust exposure based on the image content',
    )
    _configure_widget(
        simulation.film_format_mm,
        tooltip='Long edge of the film format in millimeters, e.g. 35mm or 60mm',
    )
    _configure_widget(
        simulation.camera_lens_blur_um,
        tooltip='Sigma of gaussian filter in um for the camera lens blur. About 5 um for typical lenses, down to 2-4 um for high quality lenses, used for sharp input simulations without lens blur.',
    )
    _configure_widget(
        simulation.print,
        tooltip='Print paper to simulate',
    )
    _configure_widget(
        simulation.print_illuminant,
        tooltip='Print illuminant to simulate',
    )
    _configure_widget(
        simulation.print_exposure,
        tooltip='Exposure value for the print (proportional to seconds of exposure, not ev)',
        step=0.05,
    )
    _configure_widget(
        simulation.print_exposure_compensation,
        tooltip='Apply exposure compensation from negative exposure compensation ev, allow for changing of the negative exposure compensation while keeping constant print time.',
    )
    _configure_widget(
        simulation.print_y_filter_shift,
        tooltip='Y filter shift of the color enlarger from a neutral position, enlarger has 170 steps',
        min_value=-ENLARGER_STEPS,
        max_value=ENLARGER_STEPS,
    )
    _configure_widget(
        simulation.print_m_filter_shift,
        tooltip='M filter shift of the color enlarger from a neutral position, enlarger has 170 steps',
        min_value=-ENLARGER_STEPS,
        max_value=ENLARGER_STEPS,
    )
    _configure_widget(
        simulation.scan_lens_blur,
        tooltip='Sigma of gaussian filter in pixel for the scanner lens blur',
        step=0.05,
    )
    _configure_widget(
        simulation.scan_unsharp_mask,
        tooltip='Apply unsharp mask to the scan, [sigma in pixel, amount]',
    )
    _configure_widget(
        simulation.output_color_space,
        tooltip='Color space of the output image',
    )
    _configure_widget(
        simulation.output_cctf_encoding,
        tooltip='Apply the cctf transfer function of the color space. If false, data is linear.',
    )
    _configure_widget(
        simulation.scan_film,
        tooltip='Show a scan of the negative instead of the print',
    )
    _configure_widget(
        simulation.compute_full_image,
        tooltip='Do not apply preview resize, compute full resolution image. Keeps the crop if active.',
    )
    _configure_widget(
        simulation.call_button,
        tooltip='Run the simulation. Note: grain and halation computed only when compute_full_image is clicked.',
    )

    _configure_widget(
        special.film_gamma_factor,
        tooltip='Gamma factor of the density curves of the negative, < 1 reduce contrast, > 1 increase contrast',
    )
    _configure_widget(
        special.print_gamma_factor,
        tooltip='Gamma factor of the print paper, < 1 reduce contrast, > 1 increase contrast',
        step=0.05,
    )
    _configure_widget(
        special.print_density_min_factor,
        tooltip='Minimum density factor of the print paper (0-1), make the white less white',
        min_value=0,
        max_value=1,
        step=0.2,
    )

    _configure_widget(
        glare.active,
        tooltip='Add glare to the print',
    )
    _configure_widget(
        glare.percent,
        tooltip='Percentage of the glare light (typically 0.1-0.25)',
        step=0.05,
    )
    _configure_widget(
        glare.roughness,
        tooltip='Roughness of the glare light (0-1)',
    )
    _configure_widget(
        glare.blur,
        tooltip='Sigma of gaussian blur in pixels for the glare',
    )
    _configure_widget(
        glare.compensation_removal_factor,
        tooltip='Factor of glare compensation removal from the print, e.g. 0.2=20% underexposed print in the shadows, typical values (0.0-0.2). To be used instead of stochastic glare (i.e. when percent=0).',
        step=0.05,
    )
    _configure_widget(
        glare.compensation_removal_density,
        tooltip='Density of the glare compensation removal from the print, typical values (1.0-1.5).',
    )
    _configure_widget(
        glare.compensation_removal_transition,
        tooltip='Transition density range of the glare compensation removal from the print, typical values (0.1-0.5).',
    )

    _configure_widget(
        halation.scattering_strength,
        tooltip='Fraction of scattered light (0-100, percentage) for each channel [R,G,B]',
    )
    _configure_widget(
        halation.scattering_size_um,
        tooltip='Size of the scattering effect in micrometers for each channel [R,G,B], sigma of gaussian filter.',
    )
    _configure_widget(
        halation.halation_strength,
        tooltip='Fraction of halation light (0-100, percentage) for each channel [R,G,B]',
    )
    _configure_widget(
        halation.halation_size_um,
        tooltip='Size of the halation effect in micrometers for each channel [R,G,B], sigma of gaussian filter.',
    )

    _configure_widget(
        couplers.dir_couplers_amount,
        tooltip='Amount of coupler inhibitors, control saturation, typical values (0.8-1.2).',
        step=0.05,
    )
    _configure_widget(
        couplers.dir_couplers_diffusion_um,
        tooltip='Sigma in um for the diffusion of the couplers, (5-20 um), controls sharpness and affects saturation.',
        step=5,
    )
    _configure_widget(
        couplers.diffusion_interlayer,
        tooltip='Sigma in number of layers for diffusion across the rgb layers (typical layer thickness 3-5 um, so roughly 1.0-4.0 layers), affects saturation.',
    )

    _configure_widget(
        grain.active,
        tooltip='Add grain to the negative',
    )
    _configure_widget(
        grain.particle_area_um2,
        tooltip='Area of the particles in um2, relates to ISO. Approximately 0.1 for ISO 100, 0.1 for ISO 200, 0.4 for ISO 400 and so on.',
        step=0.1,
    )
    _configure_widget(
        grain.particle_scale,
        tooltip='Scale of particle area for the RGB layers, multiplies particle_area_um2',
    )
    _configure_widget(
        grain.particle_scale_layers,
        tooltip='Scale of particle area for the sublayers in every color layer, multiplies particle_area_um2',
    )
    _configure_widget(
        grain.density_min,
        tooltip='Minimum density of the grain, typical values (0.03-0.06)',
    )
    _configure_widget(
        grain.uniformity,
        tooltip='Uniformity of the grain, typical values (0.94-0.98)',
    )
    _configure_widget(
        grain.blur,
        tooltip='Sigma of gaussian blur in pixels for the grain, to be increased at high magnifications, (should be 0.8-0.9 at high resolution, reduce down to 0.6 for lower res).',
    )
    _configure_widget(
        grain.blur_dye_clouds_um,
        tooltip='Scale the sigma of gaussian blur in um for the dye clouds, to be used at high magnifications, (default 1)',
    )
    _configure_widget(
        grain.micro_structure,
        tooltip='Parameter for micro-structure due to clumps at the molecular level, [sigma blur of micro-structure / ultimate light-resolution (0.10 um default), size of molecular clumps in nm (30 nm default)]. Only for insane magnifications.',
    )

    _configure_widget(
        preflashing.exposure,
        tooltip='Preflash exposure value in ev for the print',
        step=0.005,
    )
    _configure_widget(
        preflashing.just_preflash,
        tooltip='Only apply preflash to the print, to visualize the preflash effect',
    )
    _configure_widget(
        preflashing.y_filter_shift,
        tooltip='Shift the Y filter of the enlarger from the neutral position for the preflash, typical values (-20-20), enlarger has 170 steps',
        min_value=-ENLARGER_STEPS,
    )
    _configure_widget(
        preflashing.m_filter_shift,
        tooltip='Shift the M filter of the enlarger from the neutral position for the preflash, typical values (-20-20), enlarger has 170 steps',
        min_value=-ENLARGER_STEPS,
    )

    _configure_widget(
        input_image.preview_resize_factor,
        tooltip='Scale image size down (0-1) to speed up preview processing',
    )
    _configure_widget(
        input_image.crop,
        tooltip='Crop image to a fraction of the original size to preview details at full scale',
    )
    _configure_widget(
        input_image.crop_center,
        tooltip='Center of the crop region in relative coordinates in x, y (0-1)',
    )
    _configure_widget(
        input_image.crop_size,
        tooltip='Normalized size of the crop region in x, y (0,1), as fraction of the long side.',
    )
    _configure_widget(
        input_image.input_color_space,
        tooltip='Color space of the input image, will be internally converted to sRGB and negative values clipped',
    )
    _configure_widget(
        input_image.apply_cctf_decoding,
        tooltip='Apply the inverse cctf transfer function of the color space',
    )
    _configure_widget(
        input_image.upscale_factor,
        tooltip='Scale image size up to increase resolution',
    )
    _configure_widget(
        input_image.spectral_upsampling_method,
        tooltip='Method to upsample the spectral resolution of the image, hanatos2025 works on the full visible locus, mallett2019 works only on sRGB (will clip input).',
    )
    _configure_widget(
        input_image.filter_uv,
        tooltip='Filter UV light, (amplitude, wavelength cutoff in nm, sigma in nm). It mainly helps for avoiding UV light ruining the reds. Changing this enlarger filters neutral will be affected.',
    )
    _configure_widget(
        input_image.filter_ir,
        tooltip='Filter IR light, (amplitude, wavelength cutoff in nm, sigma in nm). Changing this enlarger filters neutral will be affected.',
    )

def main():
    apply_gui_state(DEFAULT_GUI_STATE, widgets=GUI_WIDGETS)
    _configure_widget_controls()

    # tab1 = Container(layout='vertical', widgets=[grain, preflashing])
    viewer.window.add_dock_widget(input_image, area="right", name='input', tabify=True)
    # viewer.window.add_dock_widget(curves, area="right", name='curves', tabify=True)
    viewer.window.add_dock_widget(halation, area="right", name='halation', tabify=True)
    viewer.window.add_dock_widget(couplers, area="right", name='couplers', tabify=True)
    viewer.window.add_dock_widget(grain, area="right", name='grain', tabify=True)
    viewer.window.add_dock_widget(preflashing, area="right", name='preflash', tabify=True)
    viewer.window.add_dock_widget(glare, area="right", name='glare', tabify=True)
    viewer.window.add_dock_widget(special, area="right", name='special', tabify=True)
    viewer.window.add_dock_widget(layer_list, area="right", name='layers', tabify=True)
    viewer.window.add_dock_widget(filepicker, area="right", name='filepicker', tabify=True)
    viewer.window.add_dock_widget(simulation, area="right", name='main', tabify=False)
    napari.run()

    # TODO: use magicclass to create collapsable widgets as in https://forum.image.sc/t/widgets-alignment-in-the-plugin-when-nested-magic-class-and-magicgui-are-used/62929 


if __name__ == "__main__":
    main()




