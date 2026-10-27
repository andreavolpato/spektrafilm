"""Narrowband RGB scan: the scanner lights the film with three LEDs and records
each band's transmittance, as an RGB-LED scanner (Scanlight, Coolscan, Pakon)
does. Checked against an independent band integral of the film's spectral
density.
"""
from __future__ import annotations

import numpy as np
import pytest

from .conftest import make_fast_test_params

from spektrafilm.config import SPECTRAL_SHAPE
from spektrafilm.model.develop import compute_density_spectral
from spektrafilm.model.illuminants import narrowband_lamps
from spektrafilm.runtime.process import simulate


def test_narrowband_lamps_peak_at_requested_wavelengths_with_unit_area():
    lamps = narrowband_lamps((665.0, 525.0, 450.0), (20.0, 20.0, 20.0))
    wavelengths = SPECTRAL_SHAPE.wavelengths
    assert lamps.shape == (wavelengths.size, 3)
    np.testing.assert_allclose(lamps.sum(axis=0), 1.0)
    np.testing.assert_array_equal(wavelengths[np.argmax(lamps, axis=0)], [665.0, 525.0, 450.0])


def test_narrowband_lamps_half_maximum_at_half_fwhm():
    lamps = narrowband_lamps((525.0, 525.0, 525.0), (20.0, 20.0, 20.0))
    wavelengths = SPECTRAL_SHAPE.wavelengths
    peak = lamps[wavelengths == 525.0][0]
    np.testing.assert_allclose(lamps[wavelengths == 535.0][0] / peak, 0.5)


def _narrowband_film_scan_params():
    params = make_fast_test_params()
    params.workflow.route = "input > film > scan"
    params.scanner.narrowband = True
    params.io.output_cctf_encoding = False
    params.taps.inject = "cmy_film"
    params.taps.collect = "rgb_out"
    return params


@pytest.mark.integration
def test_narrowband_scan_is_the_band_transmittance_of_the_film():
    params = _narrowband_film_scan_params()
    cmy = np.array([[[0.0, 0.0, 0.0], [0.5, 1.0, 1.5], [1.2, 0.3, 0.8]]])

    result = simulate(cmy.copy(), params)

    film = params.film
    density_spectral = compute_density_spectral(
        film.data.channel_density, cmy, film.data.base_density,
        base_density_params=params.film_render.base, is_film=True,
    )
    lamps = narrowband_lamps(params.scanner.narrowband_peaks_nm, params.scanner.narrowband_fwhm_nm)
    # The profiles carry no data at the grid's ends (NaN): no light passes there.
    transmittance = np.nan_to_num(10 ** -density_spectral, nan=0.0)
    expected = np.einsum("ijk,kl->ijl", transmittance, lamps)
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-9)


@pytest.mark.integration
def test_narrowband_scan_reads_more_dye_as_less_light_in_its_own_band():
    params = _narrowband_film_scan_params()
    clear = np.zeros((1, 1, 3))
    cyan = np.array([[[1.0, 0.0, 0.0]]])
    yellow = np.array([[[0.0, 0.0, 1.0]]])

    base = simulate(clear.copy(), params)[0, 0]
    with_cyan = simulate(cyan.copy(), params)[0, 0]
    with_yellow = simulate(yellow.copy(), params)[0, 0]

    # Cyan dye absorbs red: the 665 nm band loses far more than the 450 nm one.
    red_loss, blue_loss = np.log10(base / with_cyan)[[0, 2]]
    assert red_loss > 5 * blue_loss
    # Yellow dye absorbs blue: the reverse.
    red_loss, blue_loss = np.log10(base / with_yellow)[[0, 2]]
    assert blue_loss > 5 * red_loss


@pytest.mark.integration
def test_white_lamp_scan_is_unchanged_when_narrowband_is_off():
    params = _narrowband_film_scan_params()
    params.scanner.narrowband = False
    cmy = np.array([[[0.5, 1.0, 1.5]]])

    white = simulate(cmy.copy(), params)
    params.scanner.narrowband = True
    bands = simulate(cmy.copy(), params)

    assert not np.allclose(white, bands)
