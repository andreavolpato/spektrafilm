import numpy as np

from spektrafilm_profile_creator.data.loader import load_densitometer_data
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


def unmix_density_curves(curves, crosstalk_matrix):
    inverse_crosstalk = np.linalg.inv(crosstalk_matrix)
    density_curves_raw = np.einsum('ij,kj->ki', inverse_crosstalk, curves)
    return np.clip(density_curves_raw, 0, None)


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
        density_curves=unmix_density_curves(
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


def densitometer_normalization(profile):
    data = profile.data
    density_curves = data.density_curves
    channel_density = np.asarray(data.channel_density)
    densitometer_intensity = load_densitometer_data(
        densitometer_type=profile.info.densitometer,
    )

    crosstalk_matrix = compute_densitometer_crosstalk_matrix(
        densitometer_intensity,
        channel_density,
    )
    normalization_coefficients = np.diag(crosstalk_matrix)
    updated_profile = profile.update_data(
        channel_density=channel_density / normalization_coefficients,
        density_curves=density_curves * normalization_coefficients,
    )
    log_event(
        'densitometer_normalization',
        updated_profile,
        normalization_coefficients=normalization_coefficients,
    )
    return updated_profile
