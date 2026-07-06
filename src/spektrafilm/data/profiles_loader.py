import copy
import warnings
from datetime import date
from importlib.metadata import PackageNotFoundError, version as distribution_version
import importlib.resources as pkg_resources
import json
from dataclasses import dataclass, field, is_dataclass, replace
from typing import Any, Mapping

import numpy as np


_PROJECT_URL = 'https://github.com/andreavolpato/spektrafilm'
_PROFILE_LICENSE_URL = f'{_PROJECT_URL}/blob/main/SPEKTRAFILM_LICENSE.txt'


PROFILE_TYPES = frozenset({'negative', 'positive'})
PROFILE_SUPPORTS = frozenset({'film', 'paper'})
PROFILE_STAGES = frozenset({'filming', 'printing'})
PROFILE_USES = frozenset({'still', 'cine'})
PROFILE_ANTIHALATION = frozenset({'strong', 'weak', 'no'})
PROFILE_CHANNEL_MODELS = frozenset({'color', 'bw'})
LEGACY_PROFILE_INFO_KEYS = frozenset({
    'fitted_cmy_midscale_neutral_density',
    'log_exposure_midscale_neutral',
})


def _package_version() -> str:
    try:
        return distribution_version('spektrafilm')
    except PackageNotFoundError:
        return '0+unknown'

def _created_date() -> str:
    return date.today().isoformat()

def _copyright_statement() -> str:
    return f"Copyright (c) {date.today().year} Andrea Volpato. Licensed under CC BY-SA 4.0."

@dataclass
class DensityCurvesModel:
    """Parametric model of the density curves.

    `centers`, `amplitudes`, `sigmas` (and `alphas`, when present) are 2D
    arrays shaped (n_channels, n_layers), or None when absent. n_layers can
    be 2, 3, ... — set by the array shape.
    Same rule as ProfileData: present, non-empty → float ndarray; else None.

    `model_type` names the per-layer sigmoid:
      - 'norm_cdfs'      sum of Gaussian CDFs (centers, amplitudes, sigmas);
                         `alphas` is None.
      - 'sept_norm_cdfs' sum of septic-polynomial CDF approximations with a
                         per-layer median-preserving skew `alphas` (|α|<1).
                         A fast, transcendental-free stand-in for the
                         Gaussian; α=0 reproduces the symmetric Gaussian fit.
    """
    model_type: str = 'norm_cdfs'
    centers: np.ndarray | None = None
    amplitudes: np.ndarray | None = None
    sigmas: np.ndarray | None = None
    alphas: np.ndarray | None = None

    def __post_init__(self):
        for name in ('centers', 'amplitudes', 'sigmas', 'alphas'):
            value = getattr(self, name)
            if value is None:
                continue
            arr = np.asarray(value, dtype=float)
            setattr(self, name, arr if arr.size else None)

    @property
    def n_channels(self) -> int:
        return self.centers.shape[0] if self.centers is not None and self.centers.ndim == 2 else 0

    @property
    def n_layers(self) -> int:
        return self.centers.shape[1] if self.centers is not None and self.centers.ndim == 2 else 0


@dataclass
class ProfileMetadata:
    version: str = field(default_factory=_package_version)
    copyright: str = field(default_factory=_copyright_statement)
    created: str = field(default_factory=_created_date)
    license: str = (
        "spektrafilm profile by Andrea Volpato, licensed under CC BY-SA 4.0. "
        "Redistribution and derivatives must credit the author, link the "
        f"project ({_PROJECT_URL}),"
        "preserve this license, and remain CC BY-SA 4.0."
        "Modifications must be noted. Full text of the license and "
        f"attribution requirements: {_PROFILE_LICENSE_URL}."
    )
    citation: str = (
        "If you use this profile in your work, please cite the spektrafilm "
        f"project: {_PROJECT_URL}, see CITATION.cff for details."
    )
    datasource: str = """
    This profile was created by processing raw measurement data from data-sheets and/or scientific papers. Original data are property of the respective holders.
    Film/photo-paper: Kodak and Fujifilm data-sheets, scientific publications, and technical material.
    Reflectance: Otsu (https://github.com/enneract/otsu2018), Munsell (https://zenodo.org/records/3269912), human skin (https://www.nist.gov/programs-projects/reflectance-measurements-human-skin), forest colors (https://zenodo.org/records/3269920), Japan colors (https://zenodo.org/records/5217752).
    All data publicly available.
    """.strip()

@dataclass
class ProfileInfo:
    stock: str = None
    name: str = None
    type: str = 'negative'
    support: str = 'film'
    stage: str = 'filming'
    use: str = 'still'
    antihalation: str = 'weak'
    target_print: str | None = None
    channel_model: str = 'color'
    densitometer: str = 'status_M'
    log_sensitivity_density_over_min: float = 0.2
    reference_illuminant: str = 'D55'
    viewing_illuminant: str = 'D50'

    @property
    def n_channels(self) -> int:
        """Number of emulsion (dye) channels: 1 for BW, 3 for color.

        Declared by `channel_model`, not inferred from array shapes. For BW the
        `density_curves` columns index development time (not channels), so the
        count cannot be read off `density_curves`; `_validate_profile` checks
        the spectral arrays (`log_sensitivity` / `channel_density`) match it.
        """
        return 1 if self.channel_model == 'bw' else 3

@dataclass
class Hanatos2025SensitivityAdaptation:
    window_params: np.ndarray | None = None
    surface_params: np.ndarray | None = None
    spectral_gaussian_blur: float = 0.0 # sigma in nm for gaussian blur of the spectra
    reference_illuminant: str = None # "D55" or "T"
    apply_window: bool = True
    apply_surface: bool = True
    active: bool = None

@dataclass
class ProfileData:
    """Spectral and characteristic data for a stock.

    Every field defaults to ``None``: an absent field is ``None``, never a fake
    empty array. Whatever is present is coerced to a float ndarray. The model is
    meant to grow — adding a new array field is a single line here and nothing
    else needs to change (``__post_init__`` coerces it, ``profile_from_dict``
    loads it, it round-trips). ``_validate_profile`` only checks the internal
    shape-consistency of the fields that are present.
    """
    # --- wavelengths
    wavelengths: np.ndarray | None = None
    log_sensitivity: np.ndarray | None = None
    channel_density: np.ndarray | None = None
    base_density: np.ndarray | None = None
    midscale_neutral_density: np.ndarray | None = None
    # --- log_exposure
    log_exposure: np.ndarray | None = None
    density_curves: np.ndarray | None = None
    density_curves_layers: np.ndarray | None = None
    density_curves_model: DensityCurvesModel | None = None
    # BW only: development time per density-curve column (the columns index
    # development time, not channels). None for color.
    development_time: np.ndarray | None = None
    # --- extra
    hanatos2025_adaptation_window_params: np.ndarray | None = None
    hanatos2025_adaptation_surface_params: np.ndarray | None = None

    def __post_init__(self):
        # One uniform rule: a present, non-empty array field becomes a float
        # ndarray; anything absent OR empty becomes None ("no data is no data",
        # whether the JSON omitted the key or gave []). density_curves_model is
        # the one structured field.
        for name in self.__dataclass_fields__:
            if name == 'density_curves_model':
                continue
            value = getattr(self, name)
            if value is None:
                continue
            arr = np.asarray(value, dtype=float)
            setattr(self, name, arr if arr.size else None)
        model = self.density_curves_model
        if model is not None and not isinstance(model, DensityCurvesModel):
            if isinstance(model, Mapping):
                self.density_curves_model = DensityCurvesModel(**dict(model))
            else:
                raise TypeError('density_curves_model must be a DensityCurvesModel or Mapping')


@dataclass
class Profile:
    metadata: ProfileMetadata = field(default_factory=ProfileMetadata)
    info: ProfileInfo = field(default_factory=ProfileInfo)
    data: ProfileData = field(default_factory=ProfileData)

    def __post_init__(self):
        if not isinstance(self.metadata, ProfileMetadata):
            raise TypeError('metadata must be a ProfileMetadata instance')
        if not isinstance(self.info, ProfileInfo):
            raise TypeError('info must be a ProfileInfo instance')
        if not isinstance(self.data, ProfileData):
            raise TypeError('data must be a ProfileData instance')

    def clone(self) -> 'Profile':
        return copy.deepcopy(self)

    def update_info(self, **changes) -> 'Profile':
        self.info = replace(self.info, **changes)
        return self

    def update_data(self, **changes) -> 'Profile':
        self.data = replace(self.data, **changes)
        return self

    def update(self, *, info=None, data=None) -> 'Profile':
        if info:
            self.update_info(**info)
        if data:
            self.update_data(**data)
        return self

    def hanatos2025_adaptation(self) -> Hanatos2025SensitivityAdaptation:
        return Hanatos2025SensitivityAdaptation(
            window_params=self.data.hanatos2025_adaptation_window_params,
            surface_params=self.data.hanatos2025_adaptation_surface_params,
            reference_illuminant=self.info.reference_illuminant,
        )
    
    @property
    def is_positive(self) -> bool:
        return self.info.type == 'positive'

    @property
    def is_negative(self) -> bool:
        return self.info.type == 'negative'

    @property
    def is_paper(self) -> bool:
        return self.info.support == 'paper'

    @property
    def is_film(self) -> bool:
        return self.info.support == 'film'
    
    @property
    def is_color(self) -> bool:
        return self.info.channel_model == 'color'
    
    @property
    def is_bw(self) -> bool:
        return self.info.channel_model == 'bw'

    @property
    def is_filming(self) -> bool:
        return self.info.stage == 'filming'

    @property
    def is_printing(self) -> bool:
        return self.info.stage == 'printing'

    @property
    def is_still(self) -> bool:
        return self.info.use == 'still'

    @property
    def is_cine(self) -> bool:
        return self.info.use == 'cine'


def _known_fields_only(cls, payload, what):
    """Keep only keys that are fields of `cls`; tolerate the rest with a warning
    instead of crashing. This is what lets the schema evolve: an experimental
    key in a profile file is ignored (not fatal) until it's promoted to a field.
    """
    known = cls.__dataclass_fields__
    unknown = sorted(k for k in payload if k not in known)
    if unknown:
        warnings.warn(
            f"Profile {what}: ignoring unknown field(s) {unknown}.",
            RuntimeWarning,
            stacklevel=3,
        )
    return {k: v for k, v in payload.items() if k in known}


def profile_from_dict(data: Any) -> Profile:
    if isinstance(data, Profile):
        return data

    if not isinstance(data, Mapping):
        raise TypeError('Unsupported profile payload')

    metadata_payload = data.get('metadata', {})
    info_payload = data.get('info', {})
    data_payload = data.get('data', {})
    if not isinstance(metadata_payload, Mapping):
        raise TypeError("Profile 'metadata' must be a mapping")
    if not isinstance(info_payload, Mapping):
        raise TypeError("Profile 'info' must be a mapping")
    if not isinstance(data_payload, Mapping):
        raise TypeError("Profile 'data' must be a mapping")

    info_payload = dict(info_payload)
    for key in LEGACY_PROFILE_INFO_KEYS:
        info_payload.pop(key, None)

    return Profile(
        metadata=ProfileMetadata(**_known_fields_only(ProfileMetadata, dict(metadata_payload), 'metadata')),
        info=ProfileInfo(**_known_fields_only(ProfileInfo, info_payload, 'info')),
        data=ProfileData(**_known_fields_only(ProfileData, dict(data_payload), 'data')),
    )


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


# Saved floats are rounded to 6 decimals: two orders of magnitude below any
# physically meaningful figure (datasheet digitization and densitometry sit
# at ~1e-2 density) yet exact to ~1e-6 through the parametric-model round
# trip, and it keeps regeneration diffs free of full-precision float noise.
_JSON_FLOAT_DECIMALS = 6


def _json_safe(data):
    if isinstance(data, dict):
        return {k: _json_safe(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_json_safe(v) for v in data]
    if isinstance(data, tuple):
        return [_json_safe(v) for v in data]
    if isinstance(data, np.ndarray):
        return _json_safe(data.tolist())
    if isinstance(data, float):
        if np.isnan(data):
            return None
        return round(data, _JSON_FLOAT_DECIMALS)
    return data


def _validate_profile_info(info, stock):
    if info.type not in PROFILE_TYPES:
        raise ValueError(f"Invalid profile '{stock}': unsupported type={info.type!r}")
    if info.support not in PROFILE_SUPPORTS:
        raise ValueError(f"Invalid profile '{stock}': unsupported support={info.support!r}")
    if info.stage not in PROFILE_STAGES:
        raise ValueError(f"Invalid profile '{stock}': unsupported stage={info.stage!r}")
    if info.use not in PROFILE_USES:
        raise ValueError(f"Invalid profile '{stock}': unsupported use={info.use!r}")
    if info.antihalation not in PROFILE_ANTIHALATION:
        raise ValueError(f"Invalid profile '{stock}': unsupported antihalation={info.antihalation!r}")
    if info.channel_model not in PROFILE_CHANNEL_MODELS:
        raise ValueError(f"Invalid profile '{stock}': unsupported channel_model={info.channel_model!r}")


# The arrays a film/paper needs to actually run. Optional features
# (density_curves_layers/model, hanatos adaptation, midscale) may be absent.
_REQUIRED_DATA_FIELDS = (
    'wavelengths', 'log_sensitivity', 'channel_density',
    'base_density', 'log_exposure', 'density_curves',
)


def _validate_profile(profile, stock):
    """Thin, permissive safety net: the run-essentials must be present, and any
    field that *is* present must be internally shape-consistent. Absent optional
    fields are fine. Channel count is declared by channel_model (1 bw / 3 color);
    the spectral arrays must match it, and only color pins density_curves columns
    to n_ch (BW columns will index development time — see b10 n050)."""
    try:
        _validate_profile_info(profile.info, stock)
        data = profile.data
        n_ch = profile.info.n_channels

        missing = [f for f in _REQUIRED_DATA_FIELDS if getattr(data, f) is None]
        if missing:
            raise ValueError(f"Invalid profile '{stock}': missing required data {missing}")

        n_wl = data.wavelengths.shape[0]
        n_le = data.log_exposure.shape[0]

        def present_ok(value, predicate):
            return value is None or predicate(value)

        valid = (
            data.wavelengths.ndim == 1
            and data.log_exposure.ndim == 1
            and data.density_curves.ndim == 2
            and data.density_curves.shape[0] == n_le
            and (profile.info.channel_model != 'color'
                 or data.density_curves.shape[1] == n_ch)
            and data.log_sensitivity.ndim == 2
            and data.log_sensitivity.shape[1] == n_ch
            and data.channel_density.ndim == 2
            and data.channel_density.shape[1] == n_ch
            and data.channel_density.shape[0] == n_wl
            and data.base_density.shape[0] == n_wl
            # base_density is 1-D spectral, except a BW development-time family
            # carries one flat spectral base per density-curve column.
            and (data.base_density.ndim == 1
                 or (profile.info.channel_model == 'bw'
                     and data.base_density.ndim == 2
                     and data.base_density.shape[1] == data.density_curves.shape[1]))
            # Optional: validated only when present.
            and present_ok(data.midscale_neutral_density,
                           lambda v: v.ndim == 1 and v.shape[0] == n_wl)
            # BW development-time family: one time per density-curve column.
            and present_ok(data.development_time,
                           lambda v: v.ndim == 1 and v.shape[0] == data.density_curves.shape[1])
        )
    except (AttributeError, IndexError, KeyError, TypeError):
        raise ValueError(f"Invalid profile '{stock}'") from None

    if not valid:
        raise ValueError(f"Invalid profile '{stock}'")

# The runtime's own profile collection. The loaders below default to it but
# accept any package of profile JSONs (e.g. the staging package's stocks).
DEFAULT_PROFILES_PACKAGE = 'spektrafilm.data.profiles'


def _scan_profile_resources(directory, resources):
    for entry in directory.iterdir():
        if entry.is_dir():
            _scan_profile_resources(entry, resources)
        elif entry.name.endswith('.json'):
            stock = entry.name[:-len('.json')]
            if stock in resources:
                raise ValueError(f'Duplicate profile found in {directory}: {stock!r}')
            resources[stock] = entry


def _profile_resources(package=DEFAULT_PROFILES_PACKAGE):
    """Map stock -> profile JSON resource, scanning ``package`` recursively:
    profiles can be organized in subfolders, the stem is the stock regardless
    of nesting."""
    resources = {}
    _scan_profile_resources(pkg_resources.files(package), resources)
    return dict(sorted(resources.items()))


def list_profiles(package=DEFAULT_PROFILES_PACKAGE):
    """Return the sorted slugs of all profiles in ``package`` (the JSON file
    stems, subfolders included; default: the bundled profiles)."""
    return list(_profile_resources(package))


def load_profile_infos(package=DEFAULT_PROFILES_PACKAGE):
    """Return ``{stock: ProfileInfo}`` for every profile in ``package``,
    sorted by stock (default: the bundled profiles). This is the data-derived
    stock catalog: consumers split it by ``info.stage`` ('filming'/'printing'),
    so adding a profile never touches code."""
    infos = {}
    for stock, resource in _profile_resources(package).items():
        with resource.open('r') as file:
            info_payload = json.load(file).get('info', {})
        infos[stock] = ProfileInfo(**_known_fields_only(ProfileInfo, info_payload, 'info'))
    return infos


def load_profile(stock, package=DEFAULT_PROFILES_PACKAGE):
    resource = _profile_resources(package).get(stock)
    if resource is None:
        raise FileNotFoundError(f'No profile found for stock {stock!r} in {package!r}')
    with resource.open("r") as file:
        profile = profile_from_dict(json.load(file))
    _validate_profile(profile, stock)
    return profile


# Split-architecture alias.
load_processed_profile = load_profile

__all__ = [
    "DensityCurvesModel",
    "Profile",
    "ProfileData",
    "ProfileInfo",
    "PROFILE_ANTIHALATION",
    "PROFILE_CHANNEL_MODELS",
    "PROFILE_STAGES",
    "PROFILE_SUPPORTS",
    "PROFILE_TYPES",
    "PROFILE_USES",
    "profile_from_dict",
    "profile_to_dict",
    "list_profiles",
    "load_profile",
    "load_profile_infos",
    "load_processed_profile",
]
