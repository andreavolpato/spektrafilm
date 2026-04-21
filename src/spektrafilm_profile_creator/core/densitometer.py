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
            responsivity = densitometer_intensity[:, densitometer_channel]
            transmittance = dye_transmittance[:, dye_channel]
            valid = np.isfinite(responsivity) & np.isfinite(transmittance)
            crosstalk_matrix[densitometer_channel, dye_channel] = -np.log10(
                np.sum(responsivity[valid] * transmittance[valid])
                / np.sum(responsivity[valid])
            )
    return crosstalk_matrix


def unmix_status_density_linear_approximation(status_density, crosstalk_matrix):
    """Linear unmixing: inverts the crosstalk matrix. Exact only for a
    delta-band densitometer; biased at high densities when the densitometer
    band spans steep slopes of the dye spectrum.
    """
    inverse_crosstalk = np.linalg.inv(crosstalk_matrix)
    if len(np.shape(status_density)) == 2:
        unmixed_density = np.einsum('ij,kj->ki', inverse_crosstalk, status_density)
    elif len(np.shape(status_density)) == 1:
        unmixed_density = np.dot(inverse_crosstalk, status_density)
    else:
        raise ValueError(f'status_density must be 1D or 2D, got shape {np.shape(status_density)}')
    return np.clip(unmixed_density, 0, None)


def forward_status_density(layer_amount, channel_density, densitometer_intensity):
    """Spectral forward model: status densities produced by given per-layer amounts.

    D_status_i = -log10( <10^(-sum_k c_k * D_k(lambda))>_{s_i(lambda)} )
    """
    spectral_density = channel_density @ layer_amount
    transmittance = 10.0 ** (-spectral_density)
    weighted_response = densitometer_intensity * transmittance[:, None]
    valid = np.isfinite(weighted_response)
    weighted = np.where(valid, weighted_response, 0.0).sum(axis=0)
    normalization = np.where(valid, densitometer_intensity, 0.0).sum(axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        return -np.log10(weighted / normalization)


def _status_density_jacobian(layer_amount, channel_density, densitometer_intensity):
    """Analytical jacobian d(D_status_i)/d(c_k) of `forward_status_density`."""
    spectral_density = channel_density @ layer_amount
    transmittance = 10.0 ** (-spectral_density)
    weighted_response = densitometer_intensity * transmittance[:, None]
    masked_response = np.where(np.isfinite(weighted_response), weighted_response, 0.0)
    masked_density = np.nan_to_num(channel_density, nan=0.0)
    total_response = masked_response.sum(axis=0)
    weighted_density = masked_response.T @ masked_density
    return weighted_density / total_response[:, None]


def unmix_status_density_nonlinear(status_density, channel_density, densitometer_intensity):
    """Recover per-layer amounts by exactly inverting the spectral measurement model.

    Solves `forward_status_density(c) = measured` per row with
    `scipy.optimize.least_squares`, warm-started from the linear inversion.
    Removes the sublinear bias that `unmix_status_density_linear_approximation`
    introduces at high densities with spectrally-variable dyes.
    """
    status_density = np.asarray(status_density)
    crosstalk_matrix = compute_densitometer_crosstalk_matrix(
        densitometer_intensity, channel_density,
    )
    inverse_crosstalk = np.linalg.inv(crosstalk_matrix)

    n_layers = channel_density.shape[1]

    def solve_one(measured):
        if not np.all(np.isfinite(measured)):
            return np.full(n_layers, np.nan)
        warm_start = inverse_crosstalk @ measured
        warm_start = np.where(np.isfinite(warm_start), warm_start, 0.0)
        warm_start = np.clip(warm_start, 0.0, None)
        result = scipy.optimize.least_squares(
            fun=lambda c: forward_status_density(c, channel_density, densitometer_intensity) - measured,
            jac=lambda c: _status_density_jacobian(c, channel_density, densitometer_intensity),
            x0=warm_start,
            bounds=(0.0, np.inf),
        )
        return result.x

    if status_density.ndim == 1:
        return solve_one(status_density)
    if status_density.ndim == 2:
        return np.stack([solve_one(row) for row in status_density], axis=0)
    raise ValueError(f'status_density must be 1D or 2D, got shape {status_density.shape}')


def unmix_density(profile, densitometer_intensity=None):
    data = profile.data
    density_curves = np.asarray(data.density_curves)
    channel_density = np.asarray(data.channel_density)
    base_density = np.asarray(data.base_density)

    if densitometer_intensity is None:
        densitometer_intensity = load_densitometer_data(
            densitometer_type=profile.info.densitometer,
        )

    # density_curves are status-density over base+fog (remove_density_min runs
    # upstream), so the effective densitometer responsivity is prefiltered by
    # the base transmittance.
    effective_densitometer = densitometer_intensity * 10.0 ** (-base_density[:, None])

    unmixed_density = unmix_status_density_nonlinear(
        density_curves, channel_density, effective_densitometer,
    )
    reconstruction_residual = np.nanmax(np.abs(
        np.stack([
            forward_status_density(row, channel_density, effective_densitometer)
            for row in unmixed_density
        ], axis=0) - density_curves
    ))

    updated_profile = profile.update_data(density_curves=unmixed_density)
    log_event(
        'unmix_density',
        updated_profile,
        reconstruction_residual=reconstruction_residual,
    )
    return updated_profile


def densitometer_normalization(profile):
    data = profile.data
    channel_density = np.copy(data.channel_density)
    base_density = np.asarray(data.base_density)
    densitometer_intensity = load_densitometer_data(
        densitometer_type=profile.info.densitometer,
    )
    effective_densitometer = densitometer_intensity * 10.0 ** (-base_density[:, None])

    def densitometer_measurement(normalization_constant, channel):
        channel_transmittance = 10 ** (-channel_density * normalization_constant)
        valid = (
            np.isfinite(channel_transmittance[:, channel])
            & np.isfinite(effective_densitometer[:, channel])
        )
        return -np.log10(
            np.sum(effective_densitometer[valid, channel] * channel_transmittance[valid, channel])
            / np.sum(effective_densitometer[valid, channel])
        )

    def residual(normalization_constant, channel):
        return densitometer_measurement(normalization_constant, channel) - 1.0

    normalization_coefficients = np.ones(3)
    for i in range(3):
        normalization_coefficients[i] = scipy.optimize.least_squares(
            residual, x0=1.0, args=(i,), bounds=(0.5, 2.0),
        ).x[0]

    updated_profile = profile.update_data(
        channel_density=channel_density * normalization_coefficients,
    )
    updated_profile = updated_profile.update_info(
        fitted_cmy_midscale_neutral_density=(
            profile.info.fitted_cmy_midscale_neutral_density / normalization_coefficients
        ),
    )
    log_event(
        'densitometer_normalization',
        updated_profile,
        normalization_coefficients=normalization_coefficients,
    )
    return updated_profile


if __name__ == '__main__':
    from spektrafilm_profile_creator.core.profile_transforms import remove_density_min

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


    def plot_linear_vs_nonlinear_unmix(stock, densitometer_type, axes):
        profile = load_raw_profile(stock)
        profile = remove_density_min(profile)
        densitometer_intensity = load_densitometer_data(densitometer_type=densitometer_type)
        channel_density = np.asarray(profile.data.channel_density)
        base_density = np.asarray(profile.data.base_density)
        status_density = np.asarray(profile.data.density_curves)
        log_exposure = np.asarray(profile.data.log_exposure)

        effective_densitometer = densitometer_intensity * 10.0 ** (-base_density[:, None])

        crosstalk_matrix = compute_densitometer_crosstalk_matrix(
            effective_densitometer, channel_density,
        )
        recovered_linear = unmix_status_density_linear_approximation(status_density, crosstalk_matrix)
        recovered_nonlinear = unmix_status_density_nonlinear(
            status_density, channel_density, effective_densitometer,
        )

        channel_colors = ('tab:cyan', 'tab:pink', 'goldenrod')
        channel_labels = ('C', 'M', 'Y')
        recovered_ax, bias_ax = axes
        recovered_ax.clear()
        bias_ax.clear()

        for index, (color, label) in enumerate(zip(channel_colors, channel_labels)):
            recovered_ax.plot(
                log_exposure, recovered_linear[:, index],
                color=color, linestyle='--', label=f'{label} linear',
            )
            recovered_ax.plot(
                log_exposure, recovered_nonlinear[:, index],
                color=color, linestyle='-', label=f'{label} nonlinear',
            )
            bias_ax.plot(
                log_exposure,
                recovered_linear[:, index] - recovered_nonlinear[:, index],
                color=color, label=label,
            )

        recovered_ax.set_ylabel('Recovered layer density')
        recovered_ax.set_title(f'{profile.info.name} / {densitometer_type}')
        recovered_ax.legend(ncol=3, fontsize=8, loc='upper left')
        recovered_ax.grid(True, alpha=0.3)

        bias_ax.axhline(0.0, color='black', linewidth=0.5)
        bias_ax.set_xlabel('log exposure')
        bias_ax.set_ylabel('Linear − nonlinear')
        bias_ax.legend(ncol=3, fontsize=8, loc='upper left')
        bias_ax.grid(True, alpha=0.3)

        max_bias = np.nanmax(np.abs(recovered_linear - recovered_nonlinear))
        bias_ax.text(
            0.98, 0.05, f'max |bias| = {max_bias:.3f}',
            transform=bias_ax.transAxes, ha='right', va='bottom', fontsize=9,
        )


    fig, axs = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    plot_densitometer_response_and_channel_density(
        'status_M', load_raw_profile('kodak_portra_400'), ax=axs[0],
    )
    plot_densitometer_response_and_channel_density(
        'status_A', load_raw_profile('kodak_portra_endura'), ax=axs[1],
    )

    unmix_fig, unmix_axs = plt.subplots(
        2, 2, figsize=(14, 8), sharex='col', constrained_layout=True,
    )
    plot_linear_vs_nonlinear_unmix('kodak_portra_400', 'status_M', unmix_axs[:, 0])
    plot_linear_vs_nonlinear_unmix('kodak_portra_endura', 'status_A', unmix_axs[:, 1])

    plt.show()
