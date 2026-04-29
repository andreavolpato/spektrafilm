import numpy as np
import matplotlib.pyplot as plt

from spektrafilm.profiles.io import load_profile


def _normalize_plot_payload(profile):
    wavelengths = np.asarray(profile.data.wavelengths)
    log_exposure = np.asarray(profile.data.log_exposure)
    density_curves = np.asarray(profile.data.density_curves)
    log_sensitivity = np.asarray(profile.data.log_sensitivity)
    dye_density = np.column_stack((
        np.asarray(profile.data.channel_density),
        np.asarray(profile.data.base_density),
        np.asarray(profile.data.midscale_neutral_density),
    ))

    if log_sensitivity.ndim == 2 and log_sensitivity.shape[0] == 3 and log_sensitivity.shape[1] != 3:
        log_sensitivity = log_sensitivity.T
    if density_curves.ndim == 2 and density_curves.shape[0] == 3 and density_curves.shape[1] != 3:
        density_curves = density_curves.T
    if dye_density.ndim == 2 and dye_density.shape[0] in (3, 5) and dye_density.shape[1] not in (3, 5):
        dye_density = dye_density.T

    return wavelengths, log_exposure, density_curves, log_sensitivity, dye_density


def _normalize_original_payload(original):
    if original is None:
        return None, None

    original_log_sensitivity = np.asarray(original.data.log_sensitivity)
    if original_log_sensitivity.ndim == 2 and original_log_sensitivity.shape[0] == 3 and original_log_sensitivity.shape[1] != 3:
        original_log_sensitivity = original_log_sensitivity.T

    original_dye_density = np.column_stack((
        np.asarray(original.data.channel_density),
        np.asarray(original.data.base_density),
        np.asarray(original.data.midscale_neutral_density),
    ))
    if original_dye_density.ndim == 2 and original_dye_density.shape[0] in (3, 5) and original_dye_density.shape[1] not in (3, 5):
        original_dye_density = original_dye_density.T

    return original_log_sensitivity, original_dye_density


def _prepare_profile_plot_axes(figure=None, axes=None):
    if axes is not None:
        axs = np.asarray(axes, dtype=object).ravel()
        if axs.size != 3:
            raise ValueError('plot_profile requires exactly 3 axes.')
        fig = figure if figure is not None else axs[0].figure
        for axis in axs:
            axis.clear()
    elif figure is not None:
        fig = figure
        fig.clear()
        axs = np.asarray(fig.subplots(1, 3), dtype=object).ravel()
    else:
        fig, axs = plt.subplots(1, 3)
        axs = np.asarray(axs, dtype=object).ravel()

    try:
        fig.set_layout_engine('tight')
    except AttributeError:
        fig.set_tight_layout(tight='rect')
    fig.set_size_inches(12, 4, forward=True)
    return fig, axs


def _set_log_sensitivity_ylim(axis, log_sensitivity, original_log_sensitivity=None, max_span=4.0):
    series = [np.asarray(log_sensitivity, dtype=float)]
    if original_log_sensitivity is not None:
        series.append(np.asarray(original_log_sensitivity, dtype=float))

    finite_values = np.concatenate([
        values[np.isfinite(values)]
        for values in series
        if np.isfinite(values).any()
    ])
    if finite_values.size == 0:
        return

    upper = float(np.nanmax(finite_values))
    lower = float(np.nanmin(finite_values))
    if upper - lower > max_span:
        lower = upper - max_span
    axis.set_ylim((lower, upper))


def plot_profile(profile, unmixed=False, original=None, figure=None, axes=None):
    wavelengths, log_exposure, density_curves, log_sensitivity, dye_density = _normalize_plot_payload(profile)
    original_log_sensitivity, original_dye_density = _normalize_original_payload(original)

    fig, axs = _prepare_profile_plot_axes(figure=figure, axes=axes)
    axs[0].plot(wavelengths, log_sensitivity[:, 0], color='tab:red')
    axs[0].plot(wavelengths, log_sensitivity[:, 1], color='tab:green')
    axs[0].plot(wavelengths, log_sensitivity[:, 2], color='tab:blue')
    axs[0].legend(('R', 'G', 'B'))
    axs[0].set_xlabel('Wavelength (nm)')
    if original_log_sensitivity is not None:
        axs[0].plot(wavelengths, original_log_sensitivity[:, 0], alpha=0.5, color='tab:red', linestyle='--')
        axs[0].plot(wavelengths, original_log_sensitivity[:, 1], alpha=0.5, color='tab:green', linestyle='--')
        axs[0].plot(wavelengths, original_log_sensitivity[:, 2], alpha=0.5, color='tab:blue', linestyle='--')
    axs[0].set_ylabel('Log sensitivity')
    axs[0].set_xlim((350, 750))
    _set_log_sensitivity_ylim(axs[0], log_sensitivity, original_log_sensitivity)

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
    if original_dye_density is not None:
        axs[2].plot(wavelengths, original_dye_density[:, 0], alpha=0.5, color='tab:cyan', linestyle='--')
        axs[2].plot(wavelengths, original_dye_density[:, 1], alpha=0.5, color='tab:pink', linestyle='--')
        axs[2].plot(wavelengths, original_dye_density[:, 2], alpha=0.5, color='gold', linestyle='--')
    axs[2].set_xlabel('Wavelength (nm)')
    axs[2].set_ylabel('Diffuse density')
    axs[2].set_xlim((350, 750))

    fig.suptitle(profile.info.name + ' - ' + profile.info.stock)
    return fig, axs


if __name__ == '__main__':
    negative_profile = load_profile('kodak_portra_400')
    plot_profile(negative_profile)

    paper_profile = load_profile('kodak_portra_endura')
    plot_profile(paper_profile)
    plt.show()


__all__ = [
    'plot_profile',
]