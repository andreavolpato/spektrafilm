from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')

import matplotlib.pyplot as plt

from profiles_creator.balance import balance_metameric_neutral, balance_sensitivity
from profiles_creator.factory import (
    adjust_log_exposure,
    create_profile,
    remove_density_min,
    replace_fitted_density_curves,
    unmix_density,
)
from profiles_creator.reconstruct import reconstruct_dye_density

BASELINES_DIR = Path(__file__).resolve().parent / 'baselines'


@dataclass(frozen=True)
class CreateProfileRegressionCase:
    case_id: str
    stock: str
    type: str
    support: str
    dye_density_cmy_donor: str | None
    densitometer: str
    reference_illuminant: str
    viewing_illuminant: str
    log_sensitivity_density_over_min: float
    runtime_print_paper: str


CREATE_PROFILE_REGRESSION_CASES: tuple[CreateProfileRegressionCase, ...] = (
    CreateProfileRegressionCase(
        case_id='create_profile_kodak_portra_400',
        stock='kodak_portra_400',
        type='negative',
        support='film',
        dye_density_cmy_donor='generic_a',
        densitometer='status_M',
        reference_illuminant='D55',
        viewing_illuminant='D50',
        log_sensitivity_density_over_min=0.2,
        runtime_print_paper='kodak_portra_endura_uc',
    ),
    CreateProfileRegressionCase(
        case_id='create_profile_kodak_portra_endura_paper',
        stock='kodak_portra_endura',
        type='negative',
        support='paper',
        dye_density_cmy_donor=None,
        densitometer='status_A',
        reference_illuminant='TH-KG3-L',
        viewing_illuminant='D50',
        log_sensitivity_density_over_min=0.2,
        runtime_print_paper='kodak_portra_endura_uc',
    ),
)


def case_ids() -> list[str]:
    return [case.case_id for case in CREATE_PROFILE_REGRESSION_CASES]


def find_case(case_id: str) -> CreateProfileRegressionCase:
    for case in CREATE_PROFILE_REGRESSION_CASES:
        if case.case_id == case_id:
            return case
    raise KeyError(f'Unknown create_profile regression case_id: {case_id}')


def _process_negative_profile_for_test(profile):
    profile = remove_density_min(profile)
    profile = adjust_log_exposure(profile)
    profile = reconstruct_dye_density(profile, control_plot=False)
    profile = unmix_density(profile)
    profile = replace_fitted_density_curves(profile)
    profile = balance_sensitivity(profile)
    profile = replace_fitted_density_curves(profile)
    plt.close('all')
    return profile


def _process_paper_profile_for_test(profile):
    profile = remove_density_min(profile)
    profile = adjust_log_exposure(profile)
    profile = balance_metameric_neutral(profile)
    profile = unmix_density(profile)
    profile = replace_fitted_density_curves(profile)
    plt.close('all')
    return profile


def compute_processed_profile(case: CreateProfileRegressionCase):
    with contextlib.redirect_stdout(io.StringIO()):
        profile = create_profile(
            stock=case.stock,
            type=case.type,
            support=case.support,
            dye_density_cmy_donor=case.dye_density_cmy_donor,
            densitometer=case.densitometer,
            reference_illuminant=case.reference_illuminant,
            viewing_illuminant=case.viewing_illuminant,
            log_sensitivity_density_over_min=case.log_sensitivity_density_over_min,
        )
        if case.type == 'negative' and case.support == 'film':
            profile = _process_negative_profile_for_test(profile)
        elif case.type == 'negative' and case.support == 'paper':
            profile = _process_paper_profile_for_test(profile)
        else:
            raise KeyError(f'Unsupported create_profile regression profile: support={case.support}, type={case.type}')
    plt.close('all')
    return profile


def _sample_indices(length: int, count: int) -> np.ndarray:
    if count >= length:
        return np.arange(length, dtype=np.int64)
    return np.linspace(0, length - 1, count, dtype=np.int64)


def snapshot_profile(profile) -> dict[str, np.ndarray]:
    log_sensitivity = np.asarray(profile.data.log_sensitivity, dtype=np.float64)
    channel_density = np.asarray(profile.data.channel_density, dtype=np.float64)
    base_density = np.asarray(profile.data.base_density, dtype=np.float64)
    midscale_neutral_density = np.asarray(profile.data.midscale_neutral_density, dtype=np.float64)
    density_curves = np.asarray(profile.data.density_curves, dtype=np.float64)
    density_curves_layers = np.asarray(profile.data.density_curves_layers, dtype=np.float64)
    log_exposure = np.asarray(profile.data.log_exposure, dtype=np.float64)
    wavelengths = np.asarray(profile.data.wavelengths, dtype=np.float64)

    density_sample_idx = _sample_indices(density_curves.shape[0], 7)
    spectral_sample_idx = _sample_indices(channel_density.shape[0], 7)

    return {
        'log_sensitivity_channel_mean': np.nanmean(log_sensitivity, axis=0),
        'log_sensitivity_channel_max': np.nanmax(log_sensitivity, axis=0),
        'channel_density_sample': channel_density[spectral_sample_idx],
        'channel_density_channel_mean': np.nanmean(channel_density, axis=0),
        'base_density_sample': base_density[spectral_sample_idx],
        'midscale_neutral_density_sample': midscale_neutral_density[spectral_sample_idx],
        'density_curves_sample': density_curves[density_sample_idx],
        'density_curves_channel_mean': np.nanmean(density_curves, axis=0),
        'density_curves_channel_max': np.nanmax(density_curves, axis=0),
        'density_curves_layers_sample': density_curves_layers[density_sample_idx],
        'log_exposure_sample': log_exposure[density_sample_idx],
        'wavelength_sample': wavelengths[spectral_sample_idx],
        'fitted_cmy_midscale_neutral_density': np.asarray(profile.info.fitted_cmy_midscale_neutral_density, dtype=np.float64),
    }


def baseline_path(case_id: str) -> Path:
    return BASELINES_DIR / f'{case_id}.npz'


def save_baseline(case_id: str, profile) -> Path:
    BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    path = baseline_path(case_id)
    np.savez_compressed(path, **snapshot_profile(profile))
    return path


def load_baseline(case_id: str) -> dict[str, np.ndarray]:
    path = baseline_path(case_id)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing create_profile baseline for '{case_id}': {path}. "
            'Run scripts/regenerate_create_profile_baselines.py and commit the generated .npz file.'
        )
    data = np.load(path)
    return {key: np.asarray(data[key], dtype=np.float64) for key in data.files}


def assert_matches_baseline(
    case_id: str,
    profile,
    expected: dict[str, np.ndarray],
    *,
    rtol: float = 1e-2,
    atol: float = 1e-5,
) -> None:
    actual = snapshot_profile(profile)
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if actual_value.shape != expected_value.shape:
            raise AssertionError(
                f'{case_id}:{key} shape mismatch, actual={actual_value.shape}, expected={expected_value.shape}'
            )
        try:
            np.testing.assert_allclose(actual_value, expected_value, rtol=rtol, atol=atol)
        except AssertionError as exc:
            diff = np.abs(actual_value - expected_value)
            max_abs = float(np.nanmax(diff))
            mean_abs = float(np.nanmean(diff))
            raise AssertionError(
                f'{case_id}:{key} mismatch (max_abs={max_abs:.3e}, mean_abs={mean_abs:.3e})'
            ) from exc