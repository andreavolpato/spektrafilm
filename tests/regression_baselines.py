from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from spektrafilm.runtime.process import photo_process

from .conftest import make_fast_test_params

BASELINES_DIR = Path(__file__).resolve().parent / "baselines"


@dataclass(frozen=True)
class RegressionCase:
    case_id: str
    film_profile: str
    print_profile: str
    image_recipe: str
    output_mode: str


REGRESSION_CASES: tuple[RegressionCase, ...] = (
    RegressionCase(
        case_id="print_rgb_portra_endura_gray_ramp16",
        film_profile="kodak_portra_400_auc",
        print_profile="kodak_portra_endura_uc",
        image_recipe="gray_ramp_16",
        output_mode="print_rgb",
    ),
    RegressionCase(
        case_id="negative_density_portra_endura_gray_ramp16",
        film_profile="kodak_portra_400_auc",
        print_profile="kodak_portra_endura_uc",
        image_recipe="gray_ramp_16",
        output_mode="negative_density",
    ),
    RegressionCase(
        case_id="film_raw_portra_endura_gray_ramp16",
        film_profile="kodak_portra_400_auc",
        print_profile="kodak_portra_endura_uc",
        image_recipe="gray_ramp_16",
        output_mode="film_raw",
    ),
    RegressionCase(
        case_id="print_rgb_fuji_crystal_gray_ramp16",
        film_profile="fujifilm_pro_400h_auc",
        print_profile="fujifilm_crystal_archive_typeii_uc",
        image_recipe="gray_ramp_16",
        output_mode="print_rgb",
    ),
    RegressionCase(
        case_id="print_rgb_portra_endura_green_patch8",
        film_profile="kodak_portra_400_auc",
        print_profile="kodak_portra_endura_uc",
        image_recipe="green_patch_8",
        output_mode="print_rgb",
    ),
)


def case_ids() -> list[str]:
    return [case.case_id for case in REGRESSION_CASES]


def find_case(case_id: str) -> RegressionCase:
    for case in REGRESSION_CASES:
        if case.case_id == case_id:
            return case
    raise KeyError(f"Unknown regression case_id: {case_id}")


def build_case_image(image_recipe: str) -> np.ndarray:
    if image_recipe == "gray_ramp_16":
        ramp = np.linspace(0.01, 1.0, 16)
        image = np.ones((16, 16, 3), dtype=np.float64)
        image *= ramp[None, :, None]
        return image
    if image_recipe == "green_patch_8":
        patch = np.array([0.05, 0.40, 0.05], dtype=np.float64)
        return np.tile(patch[None, None, :], (8, 8, 1))
    raise KeyError(f"Unknown image recipe: {image_recipe}")


def compute_case_output(case: RegressionCase) -> np.ndarray:
    np.random.seed(0)
    image = build_case_image(case.image_recipe)
    params = make_fast_test_params(film_profile=case.film_profile, print_profile=case.print_profile)

    if case.output_mode == "negative_density":
        params.debug.return_film_density_cmy = True
    elif case.output_mode == "film_raw":
        params.debug.return_film_log_raw = True
    elif case.output_mode != "print_rgb":
        raise KeyError(f"Unknown output mode: {case.output_mode}")

    return np.asarray(photo_process(image, params), dtype=np.float64)


def baseline_path(case_id: str) -> Path:
    return BASELINES_DIR / f"{case_id}.npz"


def save_baseline(case_id: str, output: np.ndarray) -> Path:
    BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    path = baseline_path(case_id)
    np.savez_compressed(path, output=np.asarray(output, dtype=np.float64))
    return path


def load_baseline(case_id: str) -> np.ndarray:
    path = baseline_path(case_id)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing baseline for '{case_id}': {path}. "
            "Run scripts/regenerate_test_baselines.py and commit the generated .npz files."
        )
    data = np.load(path)
    return np.asarray(data["output"], dtype=np.float64)


def assert_matches_baseline(
    case_id: str,
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    rtol: float = 1.5e-3,
    atol: float = 1e-6,
) -> None:
    if actual.shape != expected.shape:
        raise AssertionError(
            f"{case_id}: shape mismatch, actual={actual.shape}, expected={expected.shape}"
        )

    try:
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
    except AssertionError as exc:
        diff = np.abs(actual - expected)
        max_abs = float(np.max(diff))
        mean_abs = float(np.mean(diff))
        raise AssertionError(
            f"{case_id}: snapshot mismatch (max_abs={max_abs:.3e}, mean_abs={mean_abs:.3e})."
        ) from exc
