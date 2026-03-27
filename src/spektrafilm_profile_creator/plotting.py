import numpy as np
import matplotlib.pyplot as plt

from spektrafilm.profiles.io import load_profile


def plot_profile(profile, unmixed=False, original=None):
    wavelengths = np.asarray(profile.data.wavelengths)
    log_exposure = np.asarray(profile.data.log_exposure)
    density_curves = np.asarray(profile.data.density_curves)
    log_sensitivity = np.asarray(profile.data.log_sensitivity)
    dye_density = np.column_stack((
        np.asarray(profile.data.channel_density),
        np.asarray(profile.data.base_density),
        np.asarray(profile.data.midscale_neutral_density),
    ))

    # Some profile payloads are channel-first Python lists; normalize to NxC for plotting.
    if log_sensitivity.ndim == 2 and log_sensitivity.shape[0] == 3 and log_sensitivity.shape[1] != 3:
        log_sensitivity = log_sensitivity.T
    if density_curves.ndim == 2 and density_curves.shape[0] == 3 and density_curves.shape[1] != 3:
        density_curves = density_curves.T
    if dye_density.ndim == 2 and dye_density.shape[0] in (3, 5) and dye_density.shape[1] not in (3, 5):
        dye_density = dye_density.T

    fig, axs = plt.subplots(1, 3)
    fig.set_tight_layout(tight='rect')
    fig.set_figheight(4)
    fig.set_figwidth(12)
    axs[0].plot(wavelengths, log_sensitivity[:, 0], color='tab:red')
    axs[0].plot(wavelengths, log_sensitivity[:, 1], color='tab:green')
    axs[0].plot(wavelengths, log_sensitivity[:, 2], color='tab:blue')
    axs[0].legend(('R', 'G', 'B'))
    axs[0].set_xlabel('Wavelength (nm)')
    if original is not None:
        original_log_sensitivity = np.asarray(original.data.log_sensitivity)
        if original_log_sensitivity.ndim == 2 and original_log_sensitivity.shape[0] == 3 and original_log_sensitivity.shape[1] != 3:
            original_log_sensitivity = original_log_sensitivity.T
        axs[0].plot(wavelengths, original_log_sensitivity[:, 0], alpha=0.5, color='tab:red', linestyle='--')
        axs[0].plot(wavelengths, original_log_sensitivity[:, 1], alpha=0.5, color='tab:green', linestyle='--')
        axs[0].plot(wavelengths, original_log_sensitivity[:, 2], alpha=0.5, color='tab:blue', linestyle='--')
    axs[0].set_ylabel('Log sensitivity')
    axs[0].set_xlim((350, 750))

    density_limit = np.nanmax(density_curves) * 1.05
    axs[1].plot(log_exposure, density_curves[:, 0], color='tab:red', label='R')
    axs[1].plot(log_exposure, density_curves[:, 1], color='tab:green', label='G')
    axs[1].plot(log_exposure, density_curves[:, 2], color='tab:blue', label='B')
    axs[1].plot([0, 0], [0, density_limit], color='gray', linewidth=1, label='Ref')
    if profile.info.is_film and profile.info.is_negative:
        log_exposure_three_stops = np.log10(2 ** 3)
        axs[1].plot(
            [-log_exposure_three_stops, -log_exposure_three_stops],
            [0, density_limit],
            color='lightgray',
            linestyle='dashed',
            linewidth=1,
            label='-3EV',
        )
    if profile.info.is_paper:
        axs[1].set_xlim((-1, 2))
    if profile.info.is_positive:
        axs[1].set_xlim((-2.5, 1.5))
    axs[1].legend()
    axs[1].set_xlabel('Log exposure')
    if unmixed:
        axs[1].set_ylabel('Layer density (over base+fog)')
    else:
        axs[1].set_ylabel('Density (status ' + profile.info.densitometer[-1] + ', over base+fog)')

    axs[2].plot(wavelengths, dye_density[:, 0], color='tab:cyan')
    axs[2].plot(wavelengths, dye_density[:, 1], color='tab:pink')
    axs[2].plot(wavelengths, dye_density[:, 2], color='gold')
    axs[2].plot(wavelengths, dye_density[:, 3], color='gray', linewidth=1, linestyle='--')
    axs[2].plot(wavelengths, dye_density[:, 4], color='gray', linewidth=1)
    axs[2].legend(('C', 'M', 'Y', 'Min', 'Mid'))
    if original is not None:
        original_dye_density = np.column_stack((
            np.asarray(original.data.channel_density),
            np.asarray(original.data.base_density),
            np.asarray(original.data.midscale_neutral_density),
        ))
        axs[2].plot(wavelengths, original_dye_density[:, 0], alpha=0.5, color='tab:cyan', linestyle='--')
        axs[2].plot(wavelengths, original_dye_density[:, 1], alpha=0.5, color='tab:pink', linestyle='--')
        axs[2].plot(wavelengths, original_dye_density[:, 2], alpha=0.5, color='gold', linestyle='--')
    axs[2].set_xlabel('Wavelength (nm)')
    axs[2].set_ylabel('Diffuse density')
    axs[2].set_xlim((350, 750))

    fig.suptitle(profile.info.name + ' - ' + profile.info.stock)


if __name__ == '__main__':
    negative_raw = load_profile('kodak_portra_400')
    negative_processed = load_profile('kodak_portra_400_auc')
    plot_profile(negative_raw)
    plot_profile(negative_processed)

    paper_raw = load_profile('kodak_portra_endura')
    paper_processed = load_profile('kodak_portra_endura_uc')
    plot_profile(paper_raw)
    plot_profile(paper_processed)
    plt.show()