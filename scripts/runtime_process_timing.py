import matplotlib.pyplot as plt
from spectral_film_lab.utils.io import load_image_oiio
from spectral_film_lab.utils.numba_warmup import warmup
from spectral_film_lab.runtime import photo_params, photo_process

warmup()

# image = load_image_oiio('img/targets/cc_halation.png')
# image = plt.imread('img/targets/it87_test_chart_2.jpg')
# image = np.double(image[:,:,:3])/255
image = load_image_oiio('img/test/portrait_leaves_32bit_linear_prophoto_rgb.tif')
# image = [[[0.184,0.184,0.184]]]
# image = [[[0,0,0], [0.184,0.184,0.184], [1,1,1]]]
params = photo_params(print_paper='kodak_portra_endura_uc')
params.io.input_cctf_decoding = True
params.print_paper.glare.active = False
params.debug.deactivate_stochastic_effects = False
params.camera.exposure_compensation_ev = 0
params.camera.auto_exposure = True
params.io.preview_resize_factor = 0.3
params.io.upscale_factor = 3.0
params.io.full_image = False
params.io.compute_negative = False
params.negative.grain.agx_particle_area_um2 = 1
params.enlarger.preflash_exposure = 0.0
params.enlarger.print_exposure_compensation = True
params.enlarger.print_exposure = 1.0
params.negative.grain.active = False
params.debug.return_negative_density_cmy = False
params.debug.return_print_density_cmy = False

params.settings.use_fast_stats = True
params.settings.use_camera_lut = True
params.settings.use_enlarger_lut = True
params.settings.use_scanner_lut = True
params.settings.lut_resolution = 32
params.debug.print_timings = True
image = photo_process(image, params)

# plt.imshow(image[:,:,1])
plt.imshow(image)
plt.show()