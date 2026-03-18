import importlib.resources as pkg_resources

import numpy as np
import scipy.interpolate

from spectral_film_lab.config import LOG_EXPOSURE, SPECTRAL_SHAPE
from spectral_film_lab.profiles.io import PROFILE_CHANNEL_MODELS, PROFILE_SUPPORTS, PROFILE_TYPES


def interpolate_to_common_axis(data, new_x, extrapolate=False, method='akima'):
    x = data[0]
    y = data[1]
    sorted_indexes = np.argsort(x)
    x = x[sorted_indexes]
    y = y[sorted_indexes]
    unique_index = np.unique(x, return_index=True)[1]
    x = x[unique_index]
    y = y[unique_index]
    if method == 'cubic':
        interpolator = scipy.interpolate.CubicSpline(x, y, extrapolate=extrapolate)
    elif method == 'akima':
        interpolator = scipy.interpolate.Akima1DInterpolator(x, y, extrapolate=extrapolate)
    elif method == 'linear':
        def interpolator(x_new):
            return np.interp(x_new, x, y)
    elif method == 'smoothing_spline':
        interpolator = scipy.interpolate.make_smoothing_spline(x, y)
    else:
        raise ValueError(f'Unsupported interpolation method: {method}')
    new_data = interpolator(new_x)
    return new_data


def load_csv(datapkg, filename):
    conv = lambda x: float(x) if x != b'' else None
    package = pkg_resources.files(datapkg)
    resource = package / filename
    raw_data = np.loadtxt(resource, delimiter=',', converters=conv).transpose()
    return raw_data


def load_agx_emulsion_data(stock='kodak_portra_400',
                           log_sensitivity_donor=None,
                           denisty_curves_donor=None,
                           dye_density_cmy_donor=None,
                           dye_density_min_mid_donor=None,
                           profile_type='negative',
                           support='film',
                           channel_model='color',
                           spectral_shape=SPECTRAL_SHAPE,
                           log_exposure=np.copy(LOG_EXPOSURE),
                           ):
    if profile_type not in PROFILE_TYPES:
        raise ValueError(f'Unsupported emulsion data selection: type={profile_type}')
    if support not in PROFILE_SUPPORTS:
        raise ValueError(f'Unsupported emulsion data selection: support={support}')
    if isinstance(channel_model, bool):
        channel_model = 'color' if channel_model else 'bw'
    if channel_model not in PROFILE_CHANNEL_MODELS:
        raise ValueError(f'Unsupported emulsion data selection: channel_model={channel_model}')

    if channel_model == 'bw':
        raise ValueError('Unsupported emulsion data selection: channel_model=bw. Only color datasets are available.')

    if support == 'film' and profile_type == 'negative':
        maindatapkg = 'profiles_creator.data.film.negative'
    elif support == 'film' and profile_type == 'positive':
        maindatapkg = 'profiles_creator.data.film.positive'
    elif support == 'paper':
        maindatapkg = 'profiles_creator.data.paper'
    else:
        raise ValueError(f'Unsupported emulsion data selection: channel_model={channel_model}, type={profile_type}, support={support}')

    if log_sensitivity_donor is not None:
        datapkg = maindatapkg + '.' + log_sensitivity_donor
    else:
        datapkg = maindatapkg + '.' + stock
    rootname = 'log_sensitivity_'
    log_sensitivity = np.zeros((np.size(spectral_shape.wavelengths), 3))
    channels = ['r', 'g', 'b']
    for i, channel in enumerate(channels):
        data = load_csv(datapkg, rootname + channel + '.csv')
        log_sens = interpolate_to_common_axis(data, spectral_shape.wavelengths)
        log_sensitivity[:, i] = log_sens

    if denisty_curves_donor is not None:
        datapkg = maindatapkg + '.' + denisty_curves_donor
    else:
        datapkg = maindatapkg + '.' + stock
    dh_curve_r = load_csv(datapkg, 'density_curve_r.csv')
    dh_curve_g = load_csv(datapkg, 'density_curve_g.csv')
    dh_curve_b = load_csv(datapkg, 'density_curve_b.csv')
    log_exposure_shift = (np.max(dh_curve_g[0, :]) + np.min(dh_curve_g[0, :])) / 2
    p_denc_r = interpolate_to_common_axis(dh_curve_r, log_exposure + log_exposure_shift)
    p_denc_g = interpolate_to_common_axis(dh_curve_g, log_exposure + log_exposure_shift)
    p_denc_b = interpolate_to_common_axis(dh_curve_b, log_exposure + log_exposure_shift)
    density_curves = np.array([p_denc_r, p_denc_g, p_denc_b]).transpose()

    if dye_density_cmy_donor is not None:
        datapkg = maindatapkg + '.' + dye_density_cmy_donor
    else:
        datapkg = maindatapkg + '.' + stock
    rootname = 'dye_density_'
    dye_density = np.zeros((np.size(spectral_shape.wavelengths), 5))
    channels = ['c', 'm', 'y']
    for i, channel in enumerate(channels):
        data = load_csv(datapkg, rootname + channel + '.csv')
        dye_density[:, i] = interpolate_to_common_axis(data, spectral_shape.wavelengths)
    if dye_density_min_mid_donor is not None:
        datapkg = maindatapkg + '.' + dye_density_min_mid_donor
    else:
        datapkg = maindatapkg + '.' + stock
    if support == 'film' and profile_type == 'negative':
        channels = ['min', 'mid']
        for i, channel in enumerate(channels):
            data = load_csv(datapkg, rootname + channel + '.csv')
            dye_density[:, i + 3] = interpolate_to_common_axis(data, spectral_shape.wavelengths)

    return log_sensitivity, dye_density, spectral_shape.wavelengths, density_curves, log_exposure


def load_densitometer_data(densitometer_type='status_A', spectral_shape=SPECTRAL_SHAPE):
    responsivities = np.zeros((np.size(spectral_shape.wavelengths), 3))
    channels = ['r', 'g', 'b']
    for i, channel in enumerate(channels):
        datapkg = 'profiles_creator.data.densitometer.' + densitometer_type
        filename = 'responsivity_' + channel + '.csv'
        data = load_csv(datapkg, filename)
        responsivities[:, i] = interpolate_to_common_axis(
            data,
            spectral_shape.wavelengths,
            extrapolate=False,
            method='linear',
        )
    responsivities[responsivities < 0] = 0
    responsivities /= np.nansum(responsivities, axis=0)
    return responsivities