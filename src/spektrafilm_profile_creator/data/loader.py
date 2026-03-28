import importlib.resources as pkg_resources
from collections.abc import Mapping
from functools import lru_cache

import numpy as np
import scipy.interpolate
import yaml

from spektrafilm.config import LOG_EXPOSURE, SPECTRAL_SHAPE
from spektrafilm.profiles.io import PROFILE_CHANNEL_MODELS, PROFILE_SUPPORTS, PROFILE_TYPES, ProfileData, ProfileInfo
from spektrafilm_profile_creator.raw_profile import RawProfile, RawProfileRecipe


RAW_PROFILE_FILENAME = 'profile.yaml'
_DATA_ROOT_PACKAGE = 'spektrafilm_profile_creator.data'


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


def _mapping(payload, section_name):
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise TypeError(f'Raw profile section {section_name!r} must be a mapping')
    return dict(payload)


def _load_profile_manifest(manifest, stock):
    payload = yaml.safe_load(manifest.read_text(encoding='utf-8'))
    if payload is None:
        payload = {}
    if not isinstance(payload, Mapping):
        raise TypeError(f'Raw profile manifest for {stock!r} must be a mapping')
    return dict(payload)


@lru_cache(maxsize=1)
def load_stock_catalog() -> dict[str, str]:
    catalog: dict[str, str] = {}
    data_root = pkg_resources.files(_DATA_ROOT_PACKAGE)
    for support_dir in data_root.iterdir():
        if not support_dir.is_dir() or support_dir.name not in PROFILE_SUPPORTS:
            continue
        for kind_dir in support_dir.iterdir():
            if not kind_dir.is_dir():
                continue
            data_package = f'{_DATA_ROOT_PACKAGE}.{support_dir.name}.{kind_dir.name}'
            for stock_dir in kind_dir.iterdir():
                if not stock_dir.is_dir():
                    continue
                manifest = stock_dir / RAW_PROFILE_FILENAME
                if not manifest.is_file():
                    continue
                if stock_dir.name in catalog:
                    raise ValueError(f'Duplicate stock found in raw profile catalog: {stock_dir.name!r}')
                catalog[stock_dir.name] = data_package
    return dict(sorted(catalog.items()))


def _resolve_stock_data_package(*, support, profile_type, channel_model) -> str:
    if support not in PROFILE_SUPPORTS:
        raise ValueError(f'Unsupported emulsion data selection: support={support}')
    if profile_type not in PROFILE_TYPES:
        raise ValueError(f'Unsupported emulsion data selection: type={profile_type}')
    if channel_model not in PROFILE_CHANNEL_MODELS:
        raise ValueError(f'Unsupported emulsion data selection: channel_model={channel_model}')
    kind = 'bw' if channel_model == 'bw' else profile_type
    return f'{_DATA_ROOT_PACKAGE}.{support}.{kind}'


def load_raw_profile(stock):
    data_package = load_stock_catalog().get(stock)
    if data_package is None:
        raise FileNotFoundError(f'No raw profile manifest found for stock {stock!r}')

    manifest = pkg_resources.files(data_package) / stock / RAW_PROFILE_FILENAME
    root_payload = _load_profile_manifest(manifest, stock)

    profile_payload = _mapping(root_payload.get('profile'), 'profile')
    donors_payload = _mapping(root_payload.get('donors'), 'donors')
    workflow_payload = _mapping(root_payload.get('workflow'), 'workflow')
    recipe_payload = _mapping(root_payload.get('recipe'), 'recipe')

    info = ProfileInfo(
        stock=stock,
        name=root_payload.get('name', stock),
        type=profile_payload.get('type', 'negative'),
        support=profile_payload.get('support', 'film'),
        channel_model=profile_payload.get('channel_model', 'color'),
        densitometer=profile_payload.get('densitometer', 'status_M'),
        log_sensitivity_density_over_min=profile_payload.get('log_sensitivity_density_over_min', 0.2),
        reference_illuminant=profile_payload.get('reference_illuminant', 'D55'),
        viewing_illuminant=profile_payload.get('viewing_illuminant', 'D50'),
    )
    recipe = RawProfileRecipe(
        log_sensitivity_donor=donors_payload.get('log_sensitivity'),
        density_curves_donor=donors_payload.get('density_curves'),
        dye_density_cmy_donor=donors_payload.get('dye_density_cmy'),
        dye_density_min_mid_donor=donors_payload.get('dye_density_min_mid'),
        dye_density_reconstruct_model=workflow_payload.get('dye_density_reconstruct_model', 'dmid_dmin'),
        reference_channel=recipe_payload.get('correction_reference_channel', workflow_payload.get('reference_channel')),
        target_paper=recipe_payload.get('target_paper'),
        data_trustability=recipe_payload.get('data_trustability', 1.0),
        stretch_curves=recipe_payload.get('stretch_curves', workflow_payload.get('stretch_curves', False)),
        should_process=recipe_payload.get('should_process', True),
    )
    log_sensitivity, dye_density, wavelengths, density_curves, log_exposure = load_stock_data(
        stock=stock,
        data_package=data_package,
        log_sensitivity_donor=recipe.log_sensitivity_donor,
        density_curves_donor=recipe.density_curves_donor,
        dye_density_cmy_donor=recipe.dye_density_cmy_donor,
        dye_density_min_mid_donor=recipe.dye_density_min_mid_donor,
        profile_type=info.type,
        support=info.support,
        channel_model=info.channel_model,
    )
    data = ProfileData(
        log_sensitivity=log_sensitivity,
        wavelengths=wavelengths,
        density_curves=density_curves,
        log_exposure=log_exposure,
        channel_density=np.asarray(dye_density[:, :3]),
        base_density=np.asarray(dye_density[:, 3]),
        midscale_neutral_density=np.asarray(dye_density[:, 4]),
        density_curves_layers=np.array((0, 3, 3))
    )

    return RawProfile(
        info=info,
        data=data,
        recipe=recipe,
    )


def load_stock_data(stock='kodak_portra_400',
                           log_sensitivity_donor=None,
                           density_curves_donor=None,
                           dye_density_cmy_donor=None,
                           dye_density_min_mid_donor=None,
                           profile_type='negative',
                           support='film',
                           channel_model='color',
                           data_package=None,
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

    maindatapkg = data_package or _resolve_stock_data_package(
        support=support,
        profile_type=profile_type,
        channel_model=channel_model,
    )

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

    if density_curves_donor is not None:
        datapkg = maindatapkg + '.' + density_curves_donor
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
        datapkg = 'spektrafilm_profile_creator.data.densitometer.' + densitometer_type
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


__all__ = [
    'RAW_PROFILE_FILENAME',
    'interpolate_to_common_axis',
    'load_stock_data',
    'load_csv',
    'load_densitometer_data',
    'load_raw_profile',
    'load_stock_catalog',
]