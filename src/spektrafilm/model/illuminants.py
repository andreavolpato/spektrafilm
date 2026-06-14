import numpy as np
import colour
from enum import Enum
from spektrafilm.config import SPECTRAL_SHAPE
from spektrafilm.model.color_filters import schott_kg3_heat_filter, generic_lens_transmission


class Illuminants(Enum):
    lamp = 'TH-KG3' # tungsten halogen with heat filter
    # bulb = 'T'
    # cine = 'K75P'
    # led_rgb = 'LED-RGB1'

def available_illuminants():
    """Names accepted by :func:`standard_illuminant`.

    The union of colour's ``SDS_ILLUMINANTS`` (CIE A/D/E, FL fluorescents, the
    CIE LED-B/LED-RGB/LED-V series, HP discharge, ...) and ``SDS_LIGHT_SOURCES``
    (measured commercial sources), plus the special tokens below. ``BB<temp>``
    additionally accepts any blackbody temperature (e.g. ``'BB5000'``).
    """
    special = ['TH-KG3', 'TH-KG3-L', 'T', 'K75P', 'BB<temperature>']
    names = set(colour.SDS_ILLUMINANTS.keys()) | set(colour.SDS_LIGHT_SOURCES.keys())
    return sorted(names) + special


def black_body_spectrum(temperature):
    values = colour.colorimetry.blackbody_spectral_radiance(SPECTRAL_SHAPE.wavelengths*1e-9, temperature) # to emulate an halogen lamp
    spectral_intensity = colour.SpectralDistribution(values, domain=SPECTRAL_SHAPE)
    return spectral_intensity

def standard_illuminant(type='D65', return_class=False):
    if type[0:2]=='BB':
        temperature = np.double(type[2:])
        spectral_intensity = black_body_spectrum(temperature)
    elif type=='T':
        spectral_intensity = colour.SDS_LIGHT_SOURCES['Incandescent'].copy().align(SPECTRAL_SHAPE)
    elif type=='K75P':
        spectral_intensity = colour.SDS_LIGHT_SOURCES['Kinoton 75P'].copy().align(SPECTRAL_SHAPE)
    elif type=='TH-KG3':
        spectral_intensity = black_body_spectrum(3400)
        spectral_intensity.values = schott_kg3_heat_filter.apply(spectral_intensity.values)
    elif type=='TH-KG3-L': # enlarger source with heat filter and lens transmittance
        spectral_intensity = black_body_spectrum(3400)
        spectral_intensity.values = schott_kg3_heat_filter.apply(spectral_intensity.values)
        spectral_intensity.values = generic_lens_transmission.apply(spectral_intensity.values)
    elif type in colour.SDS_ILLUMINANTS:
        # CIE standard illuminants: A, D50/55/65/75, E, FL* fluorescents,
        # LED-B1..B5 / LED-BH1 / LED-RGB1 / LED-V1..V2, HP* discharge, etc.
        spectral_intensity = colour.SDS_ILLUMINANTS[type].copy().align(SPECTRAL_SHAPE)
    elif type in colour.SDS_LIGHT_SOURCES:
        # Measured commercial light sources (projectors, lamps, displays, etc.)
        spectral_intensity = colour.SDS_LIGHT_SOURCES[type].copy().align(SPECTRAL_SHAPE)
    else:
        raise KeyError(
            f"Unknown illuminant {type!r}. Use 'BB<temperature>' for a blackbody, "
            f"one of the special tokens (TH-KG3, TH-KG3-L, T, K75P), or any name in "
            f"colour.SDS_ILLUMINANTS / colour.SDS_LIGHT_SOURCES. "
            f"See available_illuminants()."
        )
    spectral_intensity.name = type
    # normalization
    normalization = np.sum(spectral_intensity.values) / np.size(SPECTRAL_SHAPE.wavelengths)
    spectral_intensity.values = spectral_intensity.values / normalization
    
    if return_class:
        return spectral_intensity
    else:
        return spectral_intensity[:]


if __name__=="__main__":
    import matplotlib.pyplot as plt
    ill = standard_illuminant('TH-KG3', return_class=True)
    ill_bb = standard_illuminant('BB3400', return_class=True)
    print(ill[:])
    plt.plot(ill.wavelengths, ill.values)
    plt.plot(ill_bb.wavelengths, ill_bb.values)
    plt.show()


