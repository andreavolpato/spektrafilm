"""
Example: Run the full film simulation pipeline on a test image.

This script loads a test image, processes it through the film simulation
pipeline, and displays the result using matplotlib.
"""

import matplotlib.pyplot as plt
from spektrafilm import init_params, simulate, simulate_preview
from spektrafilm.utils.io import load_image_oiio, save_image_oiio


def run_simulation():
    # image = load_image_oiio('img/test/portrait_leaves_32bit_linear_prophoto_rgb.tif')
    image = load_image_oiio('C:/Users/andre/Pictures/pixls/signature_edits/darktable_exported/credit @signatureeditsco - signatureedits.com - IMG_3536.tif')
    # image = load_image_oiio('C:/Users/andre/Pictures/pixls/signature_edits/darktable_exported/Detty Studio (4).tif')

    params = init_params()
    params.film_render.grain.sublayers_active = True
    params.settings.use_enlarger_lut = True
    params.settings.use_scanner_lut = True
    
    params.camera.exposure_compensation_ev = 1.0
    params.enlarger.print_exposure = 0.5
    params.enlarger.y_filter_shift = 17
    params.enlarger.m_filter_shift = 9
    
    # params.camera.exposure_compensation_ev = 1.0
    # params.enlarger.print_exposure = 1.2
    # params.enlarger.y_filter_shift = 7
    # params.enlarger.m_filter_shift = -2
    
    params.enlarger.diffusion_filter.active = True
    params.enlarger.diffusion_filter.filter_family = "black_pro_mist"
    params.enlarger.diffusion_filter.strength = 0.25
    params.enlarger.diffusion_filter.spatial_scale = 1.0
    params.camera.film_format_mm = 35
    params.film_render.grain.agx_particle_area_um2 = 1.2
    params.io.upscale_factor = 0.5
    print_scan = simulate(image, params)
    params.io.scan_film = True
    negative_scan = simulate(image, params)

    save_image_oiio('print_scan_no_filter.jpg', print_scan)
    
    _, axs = plt.subplots(1, 2)
    axs[0].imshow(negative_scan)
    axs[0].axis('off')
    axs[0].set_title('negative')
    axs[1].imshow(print_scan)
    axs[1].axis('off')
    axs[1].set_title('print')
    plt.show()
    


if __name__ == '__main__':
    run_simulation()

