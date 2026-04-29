from __future__ import annotations

import importlib.resources
import json
import math
from functools import lru_cache

import numpy as np

from spektrafilm.profiles.io import Profile
from spektrafilm.utils.spectral_upsampling import rgb_to_smooth_spectrum


_SQRT2 = np.sqrt(2.0)
_SUMMARY_FILENAME = 'summary__all_films__hanatos2025__log_mse_perchan6.json'
_ERF = np.vectorize(math.erf)


@lru_cache(maxsize=1)
def _load_summary_targets() -> dict[str, dict[str, object]]:
    package = importlib.resources.files('spektrafilm_profile_creator.data.spectral_containment')
    resource = package / _SUMMARY_FILENAME
    with resource.open('r', encoding='utf-8') as file:
        payload = json.load(file)
    return {target['stock']: target for target in payload['targets']}


def _lookup_theta_star(stock: str) -> dict[str, float]:
    targets = _load_summary_targets()
    try:
        theta_star = targets[stock]['theta_star']
    except KeyError as exc:
        raise KeyError(f"No hanatos2025 spectral containment parameters found for stock '{stock}'.") from exc
    return {key: float(value) for key, value in theta_star.items()}


def _evaluate_perchan6_bandpass(wavelengths: np.ndarray, theta_star: dict[str, float]) -> np.ndarray:
    wavelengths = np.asarray(wavelengths, dtype=float)
    c_uv = theta_star['c_uv']
    sigma_uv = theta_star['sigma_uv']
    c_ir = theta_star['c_ir']
    sigma_ir = theta_star['sigma_ir']
    c_uv_b = theta_star['c_uv_b']
    c_ir_r = theta_star['c_ir_r']

    uv_centers = np.array([c_uv, c_uv, c_uv_b], dtype=float)
    ir_centers = np.array([c_ir_r, c_ir, c_ir], dtype=float)
    uv_edge = 0.5 * (1.0 + _ERF(
        (wavelengths[:, None] - uv_centers[None, :]) / (sigma_uv * _SQRT2)
    ))
    ir_edge = 0.5 * (1.0 - _ERF(
        (wavelengths[:, None] - ir_centers[None, :]) / (sigma_ir * _SQRT2)
    ))
    return uv_edge * ir_edge


def sensitivity_bandpass_hanatos2025(profile: Profile) -> Profile:
    info = profile.info
    data = profile.data
    wavelengths = np.asarray(data.wavelengths, dtype=float)
    sensitivity = 10 ** np.asarray(data.log_sensitivity, dtype=float)

    theta_star = _lookup_theta_star(info.stock)
    bandpass = _evaluate_perchan6_bandpass(wavelengths, theta_star)

    midgray = np.array([[[0.184, 0.184, 0.184]]], dtype=float)
    illuminant = rgb_to_smooth_spectrum(
        midgray,
        color_space='ProPhoto RGB',
        apply_cctf_decoding=False,
        reference_illuminant=info.reference_illuminant,
    )

    normalization = 1.0 / np.nansum(bandpass * sensitivity * illuminant[:, None], axis=0)
    bandpass *= normalization[None, :]
    return profile.update_data(bandpass_hanatos2025=bandpass)


__all__ = ['sensitivity_bandpass_hanatos2025']

