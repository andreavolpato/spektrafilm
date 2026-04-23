import numpy as np
import scipy
from scipy.interpolate import PchipInterpolator
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


# --------------------------------------------------------------------------------
# Unmix sensitivities
#
# `profile.data.log_sensitivity[:, k]` is the per-layer log spectral quantum
# efficiency of layer k: the runtime computes the effective log-exposure seen
# by layer k from a spectral irradiance via
#
#     log_E_eff_k = log( integral over lambda of
#                        irradiance(lambda) * 10**log_sensitivity_k(lambda) )
#
# and feeds that into the per-layer H-D curve `density_curves[:, k]`. See
# `spektrafilm.utils.spectral_upsampling.rgb_to_raw_hanatos2025` and
# `rgb_to_raw_mallett2019`, which implement this integral as an einsum of
# the chromaticity-interpolated spectrum against `sensitivity = 10**log_sens`.
#
# The manufacturer publishes three *densitometric* curves -- per-wavelength
# log-sensitivities associated with each densitometer channel -- rather than
# the three *per-layer* curves the runtime needs. A published curve entangles
# every layer's response through the crosstalk of the spectral measurement.
# `unmix_sensitivity` recovers the per-layer quantum efficiencies by
# inverting that entanglement against the nonlinear spectral forward model.
# Downstream `balance_*_sensitivity` then applies the per-channel scalar
# that makes `integral(midgray * 10**log_sens) = 1` under the reference
# illuminant, which the runtime assumes.
# --------------------------------------------------------------------------------

def _evaluate_pchip_with_clamp(pchip, pchip_derivative, log_exposure_range, x):
    """Evaluate a per-layer H-D curve with clamped endpoints and zero derivative
    outside the measured `log_exposure` range. Works for scalar or array `x`.
    """
    log_E_min, log_E_max = log_exposure_range
    x_clipped = np.clip(x, log_E_min, log_E_max)
    inside = (x >= log_E_min) & (x <= log_E_max)
    value = pchip(x_clipped)
    derivative = np.where(inside, pchip_derivative(x_clipped), 0.0)
    return value, derivative


def _continue_log_sensitivity_tail(
    wavelengths,
    values,
    tail_indices,
    boundary_index,
    interior_indices,
    opposite_boundary_index,
):
    """Continue an observed band with a monotone energy-space tail that flattens far out."""
    if tail_indices.size == 0:
        return

    boundary_value = values[boundary_index]
    interior_indices = np.asarray(interior_indices, dtype=int)
    if interior_indices.size == 0:
        values[tail_indices] = boundary_value
        return

    photon_energy = 1239.841984 / np.asarray(wavelengths, dtype=float)
    candidate_indices = interior_indices[: min(interior_indices.size, 3)]
    local_slope_magnitude = np.nanmax(
        abs(
            (values[candidate_indices] - boundary_value)
            / (photon_energy[candidate_indices] - photon_energy[boundary_index])
        )
    )
    peak_index = interior_indices[np.nanargmax(values[interior_indices])]
    shape_slope_magnitude = abs(
        (values[peak_index] - boundary_value)
        / (photon_energy[peak_index] - photon_energy[boundary_index])
    )
    support_energy_scale = abs(
        photon_energy[opposite_boundary_index] - photon_energy[boundary_index]
    )
    transition_energy_scale = abs(
        photon_energy[candidate_indices[0]] - photon_energy[boundary_index]
    )
    support_energy_scale = max(
        support_energy_scale,
        transition_energy_scale,
    )
    delta_energy = abs(photon_energy[tail_indices] - photon_energy[boundary_index])
    local_drop = local_slope_magnitude * support_energy_scale * delta_energy / (
        support_energy_scale + delta_energy
    )
    target_slope_magnitude = max(local_slope_magnitude, shape_slope_magnitude)
    if target_slope_magnitude > local_slope_magnitude:
        target_drop = target_slope_magnitude * support_energy_scale * delta_energy / (
            support_energy_scale + delta_energy
        )
        blend = delta_energy / (transition_energy_scale + delta_energy)
        drop = (1.0 - blend) * local_drop + blend * target_drop
    else:
        drop = local_drop
    values[tail_indices] = boundary_value - drop


def _fill_missing_log_sensitivity_columns(log_sensitivity, wavelengths):
    """Fill missing log-sensitivity values with smooth interpolation and decaying tails."""
    filled = np.asarray(log_sensitivity, dtype=float).copy()
    wavelengths = np.asarray(wavelengths, dtype=float)
    for channel in range(filled.shape[1]):
        column_valid = np.isfinite(filled[:, channel])
        if not column_valid.any():
            continue
        if column_valid.sum() == 1:
            filled[~column_valid, channel] = filled[column_valid, channel][0]
            continue

        valid_indices = np.flatnonzero(column_valid)
        left_index = valid_indices[0]
        right_index = valid_indices[-1]
        interpolator = PchipInterpolator(
            wavelengths[column_valid],
            filled[column_valid, channel],
            extrapolate=False,
        )
        between_support = (
            (~column_valid)
            & (np.arange(filled.shape[0]) > left_index)
            & (np.arange(filled.shape[0]) < right_index)
        )
        filled[between_support, channel] = interpolator(wavelengths[between_support])

        left_interior = valid_indices[1:]
        right_interior = valid_indices[-2::-1]
        _continue_log_sensitivity_tail(
            wavelengths,
            filled[:, channel],
            np.arange(left_index),
            left_index,
            left_interior,
            right_index,
        )
        _continue_log_sensitivity_tail(
            wavelengths,
            filled[:, channel],
            np.arange(right_index + 1, filled.shape[0]),
            right_index,
            right_interior,
            left_index,
        )
    return filled


def fill_missing_sensitivity(profile):
    """Fill missing log-sensitivity values with smooth interpolation and decaying tails.

    This is intentionally separate from `unmix_sensitivity`: the physical fit
    only solves where the published curves provide data, while this helper is a
    simple post-process for workflows that require a dense spectrum.
    """
    log_sensitivity = np.asarray(profile.data.log_sensitivity, dtype=float)
    filled = _fill_missing_log_sensitivity_columns(log_sensitivity, profile.data.wavelengths)
    updated_profile = profile.update_data(log_sensitivity=filled)
    log_event(
        'fill_missing_sensitivity_simple',
        updated_profile,
        missing_count=int(np.isnan(log_sensitivity).sum()),
        remaining_missing_count=int(np.isnan(filled).sum()),
    )
    return updated_profile


def unmix_sensitivity(profile, densitometer_intensity=None):
    """Replace `log_sensitivity` with per-layer log spectral quantum efficiency.

    Physical meaning of the output:
        `10 ** log_sensitivity[:, k]` is the absorption factor layer k applies
        per wavelength. The runtime integrates it against the irradiance
        spectrum to obtain layer k's effective log-exposure,
            log_E_eff_k = log( integral over lambda of
                               irradiance(lambda) * 10**log_sensitivity_k(lambda) ),
        which is then looked up on `density_curves[:, k]`. `unmix_sensitivity`
        produces that quantity; `balance_*_sensitivity` (running after this)
        fixes the per-channel scalar so the integral equals 1 for midgray.

    Method:
        For each wavelength lambda and densitometer channel m where the
        published densitometric log-sensitivity is finite, the manufacturer's
        narrow-band sensitometric point corresponds to a broadband-equivalent
        log-exposure
            E_m(lambda) = R_m - log_sens_m_published(lambda),
        where the per-channel anchor R_m is the log-exposure at which a
        broadband exposure drives densitometer channel m to
        `d0 = profile.info.log_sensitivity_density_over_min`. R_m is found
        per channel by Brent root-finding on the nonlinear spectral forward
        model applied to the per-layer H-D curves.

        The per-layer log quantum efficiencies `q_k(lambda)` are then fit so
        that, for every (lambda, m) with data,
            forward_status_density(
                [H_k(E_m(lambda) + q_k(lambda)) for k in layers],
                channel_density,
                densitometer_intensity * 10**(-base_density[:, None]),
            )[m] = d0,
        using PchipInterpolator-based H-D curves (smooth analytical
        derivatives, clamped outside the measured log-exposure range).

        Each wavelength is solved independently from the finite published
        densitometer channels at that wavelength. Missing published entries are
        excluded from the residual and written back as `NaN` in the output.
        If a later step needs a dense spectrum, call
        `fill_missing_sensitivity_simple` explicitly.

    Preconditions:
      * `density_curves` must be per-layer (i.e., `unmix_density` has run).
      * `base_density`, `channel_density` must be available on the profile.

    Side effect: overwrites `profile.data.log_sensitivity` with the fitted
    per-layer log quantum efficiency, in the same absolute units as the input
    `log_sensitivity` (i.e., the per-wavelength shape is corrected; the
    per-channel normalization to `integral(midgray * 10**log_sens) = 1` is
    applied downstream by `balance_*_sensitivity`).
    """
    data = profile.data
    log_sensitivity_published = np.asarray(data.log_sensitivity, dtype=float)
    density_curves = np.asarray(data.density_curves, dtype=float)
    channel_density = np.asarray(data.channel_density, dtype=float)
    base_density = np.asarray(data.base_density, dtype=float)
    log_exposure = np.asarray(data.log_exposure, dtype=float)
    d0 = profile.info.log_sensitivity_density_over_min

    if densitometer_intensity is None:
        densitometer_intensity = load_densitometer_data(
            densitometer_type=profile.info.densitometer,
        )
    effective_densitometer = densitometer_intensity * 10.0 ** (-base_density[:, None])

    n_wavelengths = log_sensitivity_published.shape[0]
    valid = np.isfinite(log_sensitivity_published)
    pchips = []
    pchip_derivatives = []
    log_exposure_ranges = []
    for k in range(3):
        column = density_curves[:, k]
        column_valid = np.isfinite(column)
        if column_valid.sum() < 2:
            raise ValueError(f"Need at least 2 finite density_curve samples for layer {k}")
        pchip = PchipInterpolator(log_exposure[column_valid], column[column_valid])
        pchips.append(pchip)
        pchip_derivatives.append(pchip.derivative())
        log_exposure_ranges.append(
            (float(log_exposure[column_valid][0]), float(log_exposure[column_valid][-1]))
        )

    # Broadband anchor: R_m is the log-exposure (density-curve reference) at
    # which a broadband exposure drives densitometer channel m to d0. Found by
    # root-finding on the full nonlinear spectral forward model using the
    # per-layer H-D curves.
    def broadband_densitometer_reading(log_e, m):
        layer_density = np.array([
            _evaluate_pchip_with_clamp(
                pchips[k], pchip_derivatives[k], log_exposure_ranges[k], log_e,
            )[0]
            for k in range(3)
        ])
        layer_density = np.clip(layer_density, 0.0, None)
        d_model = forward_status_density(
            layer_density, channel_density, effective_densitometer,
        )
        return d_model[m] - d0

    anchor_lo = max(r[0] for r in log_exposure_ranges)
    anchor_hi = min(r[1] for r in log_exposure_ranges)
    broadband_anchor = np.empty(3)
    for m in range(3):
        broadband_anchor[m] = scipy.optimize.brentq(
            broadband_densitometer_reading, anchor_lo, anchor_hi, args=(m,),
        )

    # Warm start is only optimizer scaffolding for rows with missing published
    # channels. The missing values are restored to NaN after the fit.
    warm_start = _fill_missing_log_sensitivity_columns(
        log_sensitivity_published,
        data.wavelengths,
    )

    # E_m(lambda) = R_m - log_sens_m_published(lambda), pre-cached per (lam, m).
    narrow_band_exposure = broadband_anchor[None, :] - log_sensitivity_published
    def _evaluate_layers(log_sens_row, row_exposure):
        """Compute per-channel layer densities and H-D slopes for one wavelength."""
        effective_x = row_exposure[:, None] + log_sens_row[None, :]
        layer_density = np.empty((row_exposure.size, 3))
        layer_slope = np.empty((row_exposure.size, 3))
        for k in range(3):
            layer_density[:, k], layer_slope[:, k] = _evaluate_pchip_with_clamp(
                pchips[k],
                pchip_derivatives[k],
                log_exposure_ranges[k],
                effective_x[:, k],
            )
        np.clip(layer_density, 0.0, None, out=layer_density)
        return layer_density, layer_slope

    def _solve_row(row_exposure, measured_channels, warm_start_row):
        def residuals(log_sens_row):
            layer_density, _ = _evaluate_layers(log_sens_row, row_exposure)
            data_res = np.empty(measured_channels.size)
            for row_index, measured_channel in enumerate(measured_channels):
                d_model = forward_status_density(
                    layer_density[row_index],
                    channel_density,
                    effective_densitometer,
                )
                data_res[row_index] = d_model[measured_channel] - d0
            return data_res

        def jacobian(log_sens_row):
            layer_density, layer_slope = _evaluate_layers(log_sens_row, row_exposure)
            data_jac = np.empty((measured_channels.size, 3))
            for row_index, measured_channel in enumerate(measured_channels):
                sd_jac = _status_density_jacobian(
                    layer_density[row_index],
                    channel_density,
                    effective_densitometer,
                )
                data_jac[row_index] = sd_jac[measured_channel] * layer_slope[row_index]
            return data_jac

        return scipy.optimize.least_squares(
            residuals,
            warm_start_row,
            jac=jacobian,
            method='trf',
        )

    log_sensitivity_fit = np.full_like(log_sensitivity_published, np.nan)
    row_residuals = []

    for wavelength_index in range(n_wavelengths):
        measured_channels = np.flatnonzero(valid[wavelength_index])
        if measured_channels.size == 0:
            continue

        row_exposure = narrow_band_exposure[wavelength_index, measured_channels]
        result = _solve_row(row_exposure, measured_channels, warm_start[wavelength_index])
        row_fit = np.asarray(result.x, dtype=float)
        row_fit[~valid[wavelength_index]] = np.nan
        log_sensitivity_fit[wavelength_index] = row_fit
        if result.fun.size > 0:
            row_residuals.append(float(np.nanmax(np.abs(result.fun))))

    reconstruction_residual = max(row_residuals, default=0.0)

    updated_profile = profile.update_data(log_sensitivity=log_sensitivity_fit)
    log_event(
        'unmix_sensitivity',
        updated_profile,
        reconstruction_residual=reconstruction_residual,
        broadband_anchor=broadband_anchor,
        missing_count=int((~valid).sum()),
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


    def plot_sensitivity_unmix(stock, axes):
        profile = load_raw_profile(stock)
        profile = remove_density_min(profile)
        profile = unmix_density(profile)
        published = np.asarray(profile.data.log_sensitivity)
        profile = unmix_sensitivity(profile)
        fitted = np.asarray(profile.data.log_sensitivity)
        wavelengths = np.asarray(profile.data.wavelengths)

        absorption = fitted - np.nanmax(fitted, axis=0, keepdims=True)

        layer_colors = ('tab:red', 'tab:green', 'tab:blue')
        layer_labels = ('R-sensitive / C layer', 'G-sensitive / M layer', 'B-sensitive / Y layer')
        sens_ax, abs_ax = axes
        sens_ax.clear()
        abs_ax.clear()

        for index, (color, label) in enumerate(zip(layer_colors, layer_labels)):
            sens_ax.plot(
                wavelengths, published[:, index],
                color=color, linestyle='--', alpha=0.6, label=f'{label} published',
            )
            sens_ax.plot(
                wavelengths, fitted[:, index],
                color=color, linestyle='-', label=f'{label} unmixed',
            )
            abs_ax.plot(
                wavelengths, absorption[:, index],
                color=color, label=label,
            )

        sens_ax.set_ylabel('log sensitivity')
        sens_ax.set_title(f'{profile.info.name}')
        sens_ax.legend(ncol=2, fontsize=7, loc='lower center')
        sens_ax.grid(True, alpha=0.3)
        sens_ax.set_xlim((350, 750))

        abs_ax.set_xlabel('Wavelength (nm)')
        abs_ax.set_ylabel('Peak-normalized log absorption')
        abs_ax.legend(ncol=3, fontsize=8, loc='lower center')
        abs_ax.grid(True, alpha=0.3)
        abs_ax.set_xlim((350, 750))
        abs_ax.set_ylim((-3.5, 0.2))

        max_delta = np.nanmax(np.abs(fitted - published))
        sens_ax.text(
            0.98, 0.05, f'max |unmix - published| = {max_delta:.3f}',
            transform=sens_ax.transAxes, ha='right', va='bottom', fontsize=9,
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

    sens_fig, sens_axs = plt.subplots(
        2, 2, figsize=(14, 8), sharex='col', constrained_layout=True,
    )
    plot_sensitivity_unmix('kodak_portra_400', sens_axs[:, 0])
    plot_sensitivity_unmix('kodak_portra_endura', sens_axs[:, 1])

    plt.show()
