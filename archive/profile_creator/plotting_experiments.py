import numpy as np
import matplotlib.pyplot as plt

from spektrafilm.profiles.io import load_profile
from spektrafilm.runtime.api import init_params, simulate


def plot_grain_chart(profile=None, film_format_mm=35):
    if profile is None:
        profile = load_profile('kodak_portra_400_auc')

    log_exposure_gradient = profile.data.log_exposure + np.log10(0.184)
    exposure = 10 ** log_exposure_gradient
    image = np.tile(exposure, (2048, 1))
    image = np.tile(image, (3, 1, 1))
    image = np.transpose(image, (1, 2, 0))

    densitometer_aperture_diameter = 48
    pixel_size = np.sqrt((densitometer_aperture_diameter / 2) ** 2 * np.pi)
    film_format_mm = np.max(image.shape) * pixel_size / 1000

    params = init_params()
    params.film = profile
    params.camera.film_format_mm = film_format_mm
    params.io.input_cctf_decoding = False
    params.camera.auto_exposure = False
    params.camera.exposure_compensation_ev = 0
    params.io.scan_film = True
    params.debug.deactivate_spatial_effects = True
    params.debug.return_film_density_cmy = True
    density_cmy = simulate(image, params)

    rms = np.std(density_cmy, axis=0) * 1000

    _, axis_density = plt.subplots()
    colors = ['tab:red', 'tab:green', 'tab:blue']
    for index in np.arange(3):
        axis_density.plot(profile.data.log_exposure, profile.data.density_curves[:, index], color=colors[index])

    axis_density.set_ylim((0, 3))
    axis_density.set_xlim((-2, 3))
    axis_density.set_ylabel('Unmixed Density (over B+F)')
    axis_density.legend(['R', 'G', 'B'])
    axis_density.set_xlabel('Log Exposure')
    axis_density.set_title('Diffuse RMS Granularity Curves')
    axis_grain = axis_density.twinx()

    for index in np.arange(3):
        axis_grain.plot(log_exposure_gradient - np.log10(0.184), rms[:, index], '--', color=colors[index])
    axis_grain.set_ylim((1, 1000))
    axis_grain.set_yscale('log')
    axis_grain.set_yticks([1, 2, 3, 5, 10, 20, 30, 50, 100], [1, 2, 3, 5, 10, 20, 30, 50, 100])
    axis_grain.grid(alpha=0.25)
    axis_grain.set_ylabel('Granularity Sigma D x1000')

    axis_grain.text(0.16, 0.95, profile.info.name, transform=axis_grain.transAxes, ha='left', va='center')
    axis_grain.text(0.16, 0.90, f'Particle area: {profile.grain.agx_particle_area_um2} $\\mu$m$^2$', transform=axis_grain.transAxes, ha='left', va='center')
    axis_grain.text(0.16, 0.85, f'Particle area scale RGB: {profile.grain.agx_particle_scale}', transform=axis_grain.transAxes, ha='left', va='center')
    axis_grain.text(0.16, 0.80, f'Particle area scale sublayers: {profile.grain.agx_particle_scale_layers}', transform=axis_grain.transAxes, ha='left', va='center')
    axis_grain.text(0.16, 0.75, f'Uniformity RGB: {profile.grain.uniformity}', transform=axis_grain.transAxes, ha='left', va='center')
    axis_grain.text(0.16, 0.70, f'Density min RGB: {profile.grain.density_min}', transform=axis_grain.transAxes, ha='left', va='center')
