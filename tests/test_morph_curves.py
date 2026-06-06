import numpy as np
from pytest import mark

from spektrafilm.data.profiles_loader import DensityCurvesModel
from spektrafilm.utils.morph_curves import PrintChemistryParams, apply_print_curves_morph


pytestmark = mark.unit


@mark.parametrize("profile_type", ["positive", "negative"])
def test_developer_exhaustion_preserves_midgray_at_loge_zero(profile_type):
    model = DensityCurvesModel(
        centers=np.array(
            [
                [-1.2, -0.2, 0.8],
                [-1.1, -0.1, 0.9],
                [-1.0, 0.0, 1.0],
            ],
            dtype=float,
        ),
        amplitudes=np.array(
            [
                [0.25, 0.35, 0.30],
                [0.22, 0.34, 0.33],
                [0.20, 0.32, 0.36],
            ],
            dtype=float,
        ),
        sigmas=np.array(
            [
                [0.30, 0.25, 0.35],
                [0.28, 0.24, 0.34],
                [0.26, 0.23, 0.33],
            ],
            dtype=float,
        ),
    )
    log_exposure = np.array([0.0], dtype=float)

    baseline = apply_print_curves_morph(
        log_exposure,
        model,
        PrintChemistryParams(),
        profile_type=profile_type,
    )
    exhausted = apply_print_curves_morph(
        log_exposure,
        model,
        PrintChemistryParams(developer_exhaustion=0.35),
        profile_type=profile_type,
    )

    assert np.allclose(exhausted, baseline, atol=1e-10)