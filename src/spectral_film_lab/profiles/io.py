import copy
import importlib.resources as pkg_resources
import json
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Mapping

import numpy as np


PROFILE_TYPES = frozenset({'negative', 'positive'})
PROFILE_SUPPORTS = frozenset({'film', 'paper'})
PROFILE_CHANNEL_MODELS = frozenset({'color', 'bw'})


def normalize_channel_model(channel_model: str | bool) -> str:
    if isinstance(channel_model, bool):
        channel_model = 'color' if channel_model else 'bw'
    if channel_model not in PROFILE_CHANNEL_MODELS:
        raise ValueError(f"Unsupported channel model: {channel_model}")
    return channel_model


def _empty_vector() -> np.ndarray:
    return np.empty((0,), dtype=float)


def _empty_matrix() -> np.ndarray:
    return np.empty((0, 3), dtype=float)


def _empty_tensor() -> np.ndarray:
    return np.empty((0, 3, 3), dtype=float)


@dataclass
class ProfileInfo:
    stock: str = ''
    name: str = ''
    type: str = 'negative'
    support: str = 'film'
    channel_model: str = 'color'
    densitometer: str = 'status_M'
    log_sensitivity_density_over_min: float = 0.2
    reference_illuminant: str = 'D55'
    viewing_illuminant: str = 'D50'
    fitted_cmy_midscale_neutral_density: Any = None
    log_exposure_midscale_neutral: Any = None

    def __post_init__(self):
        if self.type not in PROFILE_TYPES:
            raise ValueError(f"Unsupported profile type: {self.type}")
        if self.support not in PROFILE_SUPPORTS:
            raise ValueError(f"Unsupported profile support: {self.support}")
        self.channel_model = normalize_channel_model(self.channel_model)

    @property
    def is_positive(self) -> bool:
        return self.type == 'positive'

    @property
    def is_negative(self) -> bool:
        return self.type == 'negative'

    @property
    def is_paper(self) -> bool:
        return self.support == 'paper'

    @property
    def is_film(self) -> bool:
        return self.support == 'film'
    
    @property
    def is_color(self) -> bool:
        return self.channel_model == 'color'
    
    @property
    def is_bw(self) -> bool:
        return self.channel_model == 'bw'


@dataclass
class ProfileData:
    log_sensitivity: np.ndarray = field(default_factory=_empty_matrix)
    wavelengths: np.ndarray = field(default_factory=_empty_vector)
    density_curves: np.ndarray = field(default_factory=_empty_matrix)
    log_exposure: np.ndarray = field(default_factory=_empty_vector)
    channel_density: np.ndarray = field(default_factory=_empty_matrix)
    base_density: np.ndarray = field(default_factory=_empty_vector)
    midscale_neutral_density: np.ndarray = field(default_factory=_empty_vector)
    density_curves_layers: np.ndarray = field(default_factory=_empty_tensor)

    def __post_init__(self):
        self.log_sensitivity = np.asarray(self.log_sensitivity, dtype=float)
        self.wavelengths = np.asarray(self.wavelengths, dtype=float)
        self.density_curves = np.asarray(self.density_curves, dtype=float)
        self.log_exposure = np.asarray(self.log_exposure, dtype=float)
        self.channel_density = np.asarray(self.channel_density, dtype=float)
        self.base_density = np.asarray(self.base_density, dtype=float)
        self.midscale_neutral_density = np.asarray(self.midscale_neutral_density, dtype=float)
        self.density_curves_layers = np.asarray(self.density_curves_layers, dtype=float)


@dataclass
class Profile:
    info: ProfileInfo = field(default_factory=ProfileInfo)
    data: ProfileData = field(default_factory=ProfileData)

    def __post_init__(self):
        if not isinstance(self.info, ProfileInfo):
            self.info = _coerce_profile_info(self.info)
        if not isinstance(self.data, ProfileData):
            self.data = _coerce_profile_data(self.data)


def _coerce_profile_info(data: Any) -> ProfileInfo:
    if isinstance(data, ProfileInfo):
        return data
    if isinstance(data, Mapping):
        source = dict(data)
    elif hasattr(data, '__dict__'):
        source = vars(data)
    else:
        raise TypeError('Unsupported profile info payload')

    kwargs = {}
    for profile_field in fields(ProfileInfo):
        if profile_field.name in source:
            kwargs[profile_field.name] = source[profile_field.name]
    return ProfileInfo(**kwargs)


def _coerce_profile_data(data: Any) -> ProfileData:
    if isinstance(data, ProfileData):
        return data
    if isinstance(data, Mapping):
        source = dict(data)
    elif hasattr(data, '__dict__'):
        source = vars(data)
    else:
        raise TypeError('Unsupported profile data payload')

    kwargs = {}
    for profile_field in fields(ProfileData):
        if profile_field.name in source:
            kwargs[profile_field.name] = source[profile_field.name]
    return ProfileData(**kwargs)


def profile_from_dict(data):
    if isinstance(data, Profile):
        return data
    if isinstance(data, Mapping) and 'info' in data and 'data' in data:
        return Profile(info=data['info'], data=data['data'])
    if isinstance(data, dict):
        return {k: profile_from_dict(v) for k, v in data.items()}
    if isinstance(data, list):
        return [profile_from_dict(v) for v in data]
    if isinstance(data, tuple):
        return tuple(profile_from_dict(v) for v in data)
    return data


def profile_to_dict(data):
    if is_dataclass(data):
        return {k: profile_to_dict(getattr(data, k)) for k in data.__dataclass_fields__}
    if isinstance(data, dict):
        return {k: profile_to_dict(v) for k, v in data.items()}
    if isinstance(data, list):
        return [profile_to_dict(v) for v in data]
    if isinstance(data, tuple):
        return [profile_to_dict(v) for v in data]
    return data


def _json_safe(data):
    if isinstance(data, dict):
        return {k: _json_safe(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_json_safe(v) for v in data]
    if isinstance(data, tuple):
        return [_json_safe(v) for v in data]
    if isinstance(data, np.ndarray):
        return _json_safe(data.tolist())
    if isinstance(data, float) and np.isnan(data):
        return None
    return data


def _validate_profile(profile, stock):
    try:
        profile.info = _coerce_profile_info(profile.info)
        data = profile.data
        valid = (
            data.log_exposure.ndim == 1
            and data.density_curves.ndim == 2
            and data.density_curves.shape[1] == 3
            and data.density_curves.shape[0] == data.log_exposure.shape[0]
            and data.log_sensitivity.ndim == 2
            and data.log_sensitivity.shape[1] == 3
            and data.wavelengths.ndim == 1
            and data.channel_density.ndim == 2
            and data.channel_density.shape[1] == 3
            and data.channel_density.shape[0] == data.wavelengths.shape[0]
            and data.base_density.ndim == 1
            and data.base_density.shape[0] == data.wavelengths.shape[0]
            and data.midscale_neutral_density.ndim == 1
            and data.midscale_neutral_density.shape[0] == data.wavelengths.shape[0]
        )
    except (AttributeError, IndexError, KeyError, TypeError):
        raise ValueError(f"Invalid profile '{stock}'") from None

    if not valid:
        raise ValueError(f"Invalid profile '{stock}'")

def save_profile(profile, suffix=''):
    profile = copy.deepcopy(profile)
    profile.info.stock = profile.info.stock + suffix
    package = pkg_resources.files('spectral_film_lab.data.profiles')
    filename = profile.info.stock + '.json'
    resource = package / filename
    print('Saving to:', filename)
    with resource.open("w") as file:
        json.dump(_json_safe(profile_to_dict(profile)), file, indent=4, allow_nan=False)

def load_profile(stock):
    package = pkg_resources.files('spectral_film_lab.data.profiles')
    filename = stock + '.json'
    resource = package / filename
    with resource.open("r") as file:
        profile = profile_from_dict(json.load(file))
    _validate_profile(profile, stock)
    return profile


# Split-architecture aliases.
load_processed_profile = load_profile
save_processed_profile = save_profile

__all__ = [
    "Profile",
    "ProfileData",
    "ProfileInfo",
    "PROFILE_CHANNEL_MODELS",
    "PROFILE_SUPPORTS",
    "PROFILE_TYPES",
    "normalize_channel_model",
    "profile_from_dict",
    "profile_to_dict",
    "load_profile",
    "save_profile",
    "load_processed_profile",
    "save_processed_profile",
]
