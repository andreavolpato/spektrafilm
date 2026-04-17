import numpy as np
import scipy
import matplotlib.pyplot as plt

from spektrafilm_profile_creator.data.loader import load_densitometer_data, load_raw_profile
from spektrafilm_profile_creator.diagnostics.messages import log_event


def compute_densitometer_crosstalk_matrix(densitometer_intensity, dye_density):
    crosstalk_matrix = np.zeros((3, 3))
    dye_transmittance = 10 ** (-dye_density[:, 0:3])
    for densitometer_channel in np.arange(3):
        for dye_channel in np.arange(3):
            crosstalk_matrix[densitometer_channel, dye_channel] = -np.log10(
                np.nansum(
                    densitometer_intensity[:, densitometer_channel]
                    * dye_transmittance[:, dye_channel]
                )
                / np.nansum(densitometer_intensity[:, densitometer_channel])
            )
    return crosstalk_matrix


def unmix_status_density(status_density, crosstalk_matrix):
    inverse_crosstalk = np.linalg.inv(crosstalk_matrix)
    if len(np.shape(status_density)) == 2:
        unmixed_density = np.einsum('ij,kj->ki', inverse_crosstalk, status_density)
    elif len(np.shape(status_density)) == 1:
        unmixed_density = np.dot(inverse_crosstalk, status_density)
    else:
        raise ValueError(f'status_density must be 1D or 2D, got shape {np.shape(status_density)}')
    return np.clip(unmixed_density, 0, None)


def unmix_density(profile, densitometer_intensity=None):
    data = profile.data
    density_curves = data.density_curves
    channel_density = np.asarray(data.channel_density)

    if densitometer_intensity is None:
        densitometer_intensity = load_densitometer_data(
            densitometer_type=profile.info.densitometer,
        )
    densitometer_crosstalk_matrix = compute_densitometer_crosstalk_matrix(
        densitometer_intensity,
        channel_density,
    )
    updated_profile = profile.update_data(
        density_curves=unmix_status_density(
            density_curves,
            densitometer_crosstalk_matrix,
        )
    )
    log_event(
        'unmix_density',
        updated_profile,
        densitometer_crosstalk_matrix=densitometer_crosstalk_matrix,
    )
    return updated_profile


def densitometer_normalization(profile, iterations=5):
    data = profile.data
    channel_density = np.copy(data.channel_density)
    densitometer_intensity = load_densitometer_data(
        densitometer_type=profile.info.densitometer,
    )
    
    def densitometer_measurement(normalization_constant, channel):
        channel_transmittance = 10 ** (-channel_density*normalization_constant)
        valid = np.isfinite(channel_transmittance[:, channel]) & np.isfinite(densitometer_intensity[:, channel])
        densitometer_density = -np.log10(
        np.nansum(
            densitometer_intensity[valid, channel]
            * channel_transmittance[valid, channel]
        )
        / np.nansum(densitometer_intensity[valid, channel])
        )
        return densitometer_density
    def residual(normalization_constant, channel):
        return densitometer_measurement(normalization_constant, channel) - 1.0
    
    normalization_coefficients = np.ones(3)
    for i in range(3):
        normalization_coefficients[i] = scipy.optimize.least_squares(residual, x0=1.0, args=(i,), bounds=(0.5, 2.0)).x[0]
    
    # normalization_coefficients = np.nanmax(channel_density, axis=0)

    updated_profile = profile.update_data(
        channel_density=channel_density * normalization_coefficients,
    )
    updated_profile = profile.update_info(
        fitted_cmy_midscale_neutral_density = profile.info.fitted_cmy_midscale_neutral_density / normalization_coefficients
    )
    log_event(
        'densitometer_normalization',
        updated_profile,
        normalization_coefficients=normalization_coefficients,
    )
    return updated_profile


if __name__ == '__main__':
    def _normalize_columns_to_peak(values):
        normalized = np.asarray(values, dtype=float)
        scale = np.nanmax(normalized, axis=0, keepdims=True)
        scale[~np.isfinite(scale) | (scale <= 0)] = 1.0
        return normalized / scale


    def plot_densitometer_response_and_channel_density(
        densitometer_type,
        profile,
        ax=None,
    ):
        responsivities = _normalize_columns_to_peak(
            load_densitometer_data(densitometer_type=densitometer_type)
        )
        wavelengths = np.asarray(profile.data.wavelengths)
        channel_density = np.asarray(profile.data.channel_density)

        if ax is None:
            figure, ax = plt.subplots()
        else:
            figure = ax.figure
            ax.clear()

        density_ax = ax.twinx()
        density_ax.clear()

        responsivity_colors = ('tab:red', 'tab:green', 'tab:blue')
        density_colors = ('tab:cyan', 'tab:pink', 'goldenrod')
        responsivity_labels = ('R responsivity', 'G responsivity', 'B responsivity')
        density_labels = ('C density', 'M density', 'Y density')

        for index, (color, label) in enumerate(zip(responsivity_colors, responsivity_labels)):
            ax.plot(
                wavelengths,
                responsivities[:, index],
                color=color,
                label=label,
            )
        for index, (color, label) in enumerate(zip(density_colors, density_labels)):
            density_ax.plot(
                wavelengths,
                channel_density[:, index],
                color=color,
                linestyle='--',
                label=label,
            )

        ax.set_xlim((350, 750))
        ax.set_ylim((0, 1.05))
        density_limit = np.nanmax(channel_density)
        if np.isfinite(density_limit) and density_limit > 0:
            density_ax.set_ylim((0, density_limit * 1.05))

        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Normalized densitometer responsivity')
        density_ax.set_ylabel('Diffuse channel density')
        ax.set_title(f'{densitometer_type} with {profile.info.name}')

        handles, labels = ax.get_legend_handles_labels()
        density_handles, density_legend_labels = density_ax.get_legend_handles_labels()
        ax.legend(handles + density_handles, labels + density_legend_labels, loc='upper right')
        return figure, ax, density_ax


    fig, axs = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    plot_densitometer_response_and_channel_density(
        'status_M',
        load_raw_profile('kodak_portra_400'),
        ax=axs[0],
    )
    plot_densitometer_response_and_channel_density(
        'status_A',
        load_raw_profile('kodak_portra_endura'),
        ax=axs[1],
    )

    plt.show()
