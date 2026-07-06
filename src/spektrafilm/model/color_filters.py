from dataclasses import dataclass
from enum import Enum
from functools import lru_cache

import numpy as np
import scipy.special
import matplotlib.pyplot as plt

from spektrafilm.config import SPECTRAL_SHAPE
from spektrafilm.utils.io import load_dichroic_filters, load_filter

################################################################################
# Dichroic CMY filters
################################################################################

def create_combined_dichroic_filter(wavelength, transitions, edges):
    # data from https://qd-europe.com/se/en/product/dichroic-filters-and-sets/
    dichroics = np.zeros((np.size(wavelength), 3))
    dichroics[:, 2] = scipy.special.erf((wavelength - edges[0]) / transitions[0])
    dichroics[:, 1][wavelength <= 550] = -scipy.special.erf((wavelength[wavelength <= 550] - edges[1]) / transitions[1])
    dichroics[:, 1][wavelength > 550] = scipy.special.erf((wavelength[wavelength > 550] - edges[2]) / transitions[2])
    dichroics[:, 0] = -scipy.special.erf((wavelength - edges[3]) / transitions[3])
    dichroics = dichroics / 2 + 0.5
    return dichroics


class DichroicFilters():
    def __init__(self, brand='thorlabs'):
        self.wavelengths = SPECTRAL_SHAPE.wavelengths
        self.filters = np.zeros((np.size(self.wavelengths), 3))
        if brand == 'custom':
            self.create_custom_filters()
        else:
            self.filters = load_dichroic_filters(self.wavelengths, brand)

    def plot(self):
        colors = ['tab:cyan', 'tab:pink', 'gold']
        _, ax = plt.subplots()
        for i in range(3):
            ax.plot(self.wavelengths, self.filters[:, i], color=colors[i])
        ax.set_ylabel('Transmittance')
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylim(0, 1)
        ax.set_xlim(np.min(self.wavelengths), np.max(self.wavelengths))
        ax.legend(('C', 'M', 'Y'))

    def apply(self, illuminant, filter_transmittance_values=[1, 1, 1]):
        # following durst 605 wheels values, with 170 max
        dimmed_filters = 1 - (1 - self.filters) * (1 - np.array(filter_transmittance_values))
        total_filter = np.prod(dimmed_filters, axis=1)
        filtered_illuminant = illuminant * total_filter
        return filtered_illuminant

    def apply_cc(self, illuminant, filter_cc_values=[0, 0, 0]):
        # Filter values are in Kodak CC units proportional to density, 100 units means 1.0 density, or 90% reduction in transmittance
        filter_transmittance_values = 10 ** -(np.array(filter_cc_values) / 100.0)
        return self.apply(illuminant, filter_transmittance_values=filter_transmittance_values)

    def create_custom_filters(self, edges=[516, 500, 610, 607], transitions=[12, 8, 8, 8]):
        self.filters = create_combined_dichroic_filter(self.wavelengths, transitions=transitions, edges=edges)


################################################################################
# Single-channel filters (heat absorbing, lens transmission, colored glass)
################################################################################

class GenericFilter():
    """A single spectral transmittance curve loaded from the filter database.

    Files live under ``data/filters/<filter_type>/<brand>/<name>.csv`` as two
    columns (wavelength, transmittance). Set ``data_in_percentage=True`` when
    the source curve is stored in percent rather than as a [0, 1] fraction.
    """

    def __init__(self,
                 name='KG3',
                 filter_type='heat_absorbing',
                 brand='schott',
                 data_in_percentage=False,
                 load_from_database=True):
        self.wavelengths = SPECTRAL_SHAPE.wavelengths
        self.name = name
        self.filter_type = filter_type
        self.brand = brand
        self.transmittance = np.zeros_like(self.wavelengths)
        if load_from_database:
            self.transmittance = load_filter(self.wavelengths, name, brand, filter_type,
                                             percent_transmittance=data_in_percentage)

    def plot(self, ax=None, label=None, **kwargs):
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(self.wavelengths, self.transmittance, label=label or self.name, **kwargs)
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Transmittance')
        ax.set_ylim(0, 1)
        return ax

    def apply(self, illuminant, value=1.0):
        dimmed_filter = 1 - (1 - self.transmittance) * value
        filtered_illuminant = illuminant * dimmed_filter
        return filtered_illuminant


################################################################################
# Band pass filter
################################################################################

def sigmoid_erf(x, center, width=1):
    return scipy.special.erf((x - center) / width) * 0.5 + 0.5


def compute_band_pass_filter(filter_uv=[1, 410, 8], filter_ir=[1, 675, 15]):
    amp_uv, wl_uv, width_uv = filter_uv
    amp_ir, wl_ir, width_ir = filter_ir

    amp_uv = np.clip(amp_uv, 0, 1)
    amp_ir = np.clip(amp_ir, 0, 1)

    wl = SPECTRAL_SHAPE.wavelengths
    filter_uv = 1 - amp_uv + amp_uv * sigmoid_erf(wl, wl_uv, width=width_uv)
    filter_ir = 1 - amp_ir + amp_ir * sigmoid_erf(wl, wl_ir, width=-width_ir)
    band_pass_filter = filter_uv * filter_ir
    return band_pass_filter


################################################################################
# Pre-built filter instances
################################################################################

# Dichroic CMY filter sets, by brand.
dichroic_filters = DichroicFilters()
thorlabs_dichroic_filters = DichroicFilters(brand='thorlabs')
edmund_optics_dichroic_filters = DichroicFilters(brand='edmund_optics')
durst_digital_light_dicrhoic_filters = DichroicFilters(brand='durst_digital_light')
custom_dichroic_filters = DichroicFilters(brand='custom')

# Schott heat-absorbing glass and the generic lens transmission curve.
schott_kg1_heat_filter = GenericFilter(name='KG1', filter_type='heat_absorbing', brand='schott')
schott_kg3_heat_filter = GenericFilter(name='KG3', filter_type='heat_absorbing', brand='schott')
schott_kg5_heat_filter = GenericFilter(name='KG5', filter_type='heat_absorbing', brand='schott')
generic_lens_transmission = GenericFilter(name='canon_24_f28_is', filter_type='lens_transmission',
                                          brand='canon', data_in_percentage=True)

################################################################################
# Color filter library (camera taking filters)
################################################################################

# A camera "taking" filter is a measured spectral transmittance curve that sits
# in front of the lens and multiplies the incoming light. The library below is
# the single source of truth: the GUI selector enum (CameraColorFilters) is
# derived from it, so adding a filter is two steps -- drop its CSV at
# data/filters/colored/<brand>/<name>.csv and append one ColorFilterSpec entry.

NO_COLOR_FILTER = 'none'  # selector value meaning "no taking filter"


@dataclass(frozen=True)
class ColorFilterSpec:
    key: str    # stable identifier stored in params and shown in the GUI selector
    brand: str  # subfolder under data/filters/colored/
    name: str   # csv stem under that brand folder
    label: str  # human-readable name for tooltips and documentation


COLOR_FILTER_LIBRARY: tuple[ColorFilterSpec, ...] = (
    ColorFilterSpec('hoya_x0',  'hoya', 'x0',  'Hoya X0 (yellow-green)'),
    ColorFilterSpec('hoya_x1',  'hoya', 'x1',  'Hoya X1 (green)'),
    ColorFilterSpec('hoya_y2',  'hoya', 'y2',  'Hoya Y2 (yellow)'),
    ColorFilterSpec('hoya_ya3', 'hoya', 'ya3', 'Hoya YA3 (deep yellow)'),
    ColorFilterSpec('hoya_r1',  'hoya', 'r1',  'Hoya R1 (red)'),
)

COLOR_FILTER_SPECS: dict[str, ColorFilterSpec] = {spec.key: spec for spec in COLOR_FILTER_LIBRARY}


@lru_cache(maxsize=None)
def get_color_filter(key):
    """Return the GenericFilter for a library key, loaded and cached on first use."""
    spec = COLOR_FILTER_SPECS[key]
    return GenericFilter(name=spec.name, filter_type='colored', brand=spec.brand)


def color_filter_transmittance(key):
    """Spectral transmittance for a selector value, on the SPECTRAL_SHAPE grid.

    Returns ``None`` for ``NO_COLOR_FILTER`` (the identity / no-filter case)."""
    if key == NO_COLOR_FILTER:
        return None
    return get_color_filter(key).transmittance


# GUI selector choices: "none" first, then one member per library entry. Derived
# from the library so the dropdown always matches it (as the stock enums in
# spektrafilm_gui.options are derived from the profile library).
CameraColorFilters = Enum(
    'CameraColorFilters',
    {NO_COLOR_FILTER: NO_COLOR_FILTER, **{spec.key: spec.key for spec in COLOR_FILTER_LIBRARY}},
)


def color_enlarger(light_source, filter_cc_values=(0, 65, 55), filters=custom_dichroic_filters):
    # Filter values are in Kodak CC units proportional to density, 100 units means 1.0 density, or 90% reduction in transmittance
    # cc_filter_values are in CMY order
    filter_cc_values = np.array(filter_cc_values)
    filtered_illuminant = filters.apply_cc(light_source, filter_cc_values=filter_cc_values)
    return filtered_illuminant


################################################################################

if __name__ == "__main__":
    wl = SPECTRAL_SHAPE.wavelengths

    # Dichroic CMY filters.
    durst_digital_light_dicrhoic_filters.plot()
    plt.title('Durst Digital Light Dichroic Filters')

    # Heat-absorbing glass, lens transmission and their product.
    fig_single, ax_single = plt.subplots()
    schott_kg3_heat_filter.plot(ax=ax_single, label='Schott KG3')
    generic_lens_transmission.plot(ax=ax_single, label='Canon lens transmittance')
    ax_single.plot(wl, schott_kg3_heat_filter.transmittance * generic_lens_transmission.transmittance,
                   label='Combined')
    ax_single.set_title('Heat-absorbing glass and lens transmission')
    ax_single.legend()

    # Color filter library.
    # color the line according to the filter color
    fig_colored, ax_colored = plt.subplots()
    colors = ['yellowgreen', 'green', 'gold', 'orange', 'red']
    for spec, color in zip(COLOR_FILTER_LIBRARY, colors):
        get_color_filter(spec.key).plot(ax=ax_colored, label=spec.label, color=color)
    ax_colored.set_title('Color Filter Library')
    ax_colored.set_xlim(np.min(wl), np.max(wl))
    ax_colored.legend()

    # Band-pass filter.
    fig_bp, ax_bp = plt.subplots()
    ax_bp.plot(wl, compute_band_pass_filter(), color='black', label='Band pass filter')
    ax_bp.set_xlabel('Wavelength (nm)')
    ax_bp.set_ylabel('Transmittance')
    ax_bp.set_ylim(0, 1.05)
    ax_bp.set_xlim(np.min(wl), np.max(wl))
    ax_bp.set_title('Band Pass Filter')
    ax_bp.legend()

    plt.show()
