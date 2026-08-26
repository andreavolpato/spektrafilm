import numpy as np
import pytest

from spektrafilm.data.profiles_loader import Profile, ProfileData, ProfileInfo
from spektrafilm.model.couplers import apply_density_correction_dir_couplers
from spektrafilm.model.develop import develop, develop_simple
from spektrafilm.model.grain import apply_grain
from spektrafilm.runtime.params_schema import (
    DirCouplersParams,
    FilmRenderingParams,
    GrainParams,
)

pytestmark = pytest.mark.unit


def _make_test_profile(profile_type: str) -> Profile:
    log_exposure = np.linspace(-3.0, 1.0, 24)
    density_curves = np.column_stack(
        [
            np.clip(log_exposure + 2.3, 0.2, 2.6),
            np.clip(log_exposure + 2.0, 0.3, 2.3),
            np.clip(log_exposure + 1.7, 0.4, 2.0),
        ]
    )
    density_curves_layers = np.stack(
        [
            density_curves * np.array([0.52, 0.48, 0.45]),
            density_curves * np.array([0.31, 0.34, 0.35]),
            density_curves * np.array([0.17, 0.18, 0.20]),
        ],
        axis=1,
    )
    return Profile(
        info=ProfileInfo(type=profile_type, support="film"),
        data=ProfileData(
            wavelengths=np.array([450.0, 550.0, 650.0]),
            log_sensitivity=np.zeros((3, 3), dtype=float),
            channel_density=np.zeros((3, 3), dtype=float),
            base_density=np.zeros((3,), dtype=float),
            midscale_neutral_density=np.zeros((3,), dtype=float),
            log_exposure=log_exposure,
            density_curves=density_curves,
            density_curves_layers=density_curves_layers,
        ),
    )


@pytest.mark.parametrize("profile_type", ["negative", "positive"])
def test_top_level_develop_matches_manual_pipeline(profile_type: str) -> None:
    profile = _make_test_profile(profile_type)
    gamma_factor = 1.15
    render = FilmRenderingParams(
        grain=GrainParams(
            active=True,
            rms_granularity=(12.0, 16.0, 20.0),
            particle_scale_sublayers=(1.0, 0.5, 0.25),
            density_min=(0.04, 0.06, 0.08),
            uniformity=(0.99, 0.98, 0.97),
            blur=0.0,
            blur_dye_clouds_um=0.0,
            micro_structure=(0.0, 0.0),
        ),
        dir_couplers=DirCouplersParams(
            active=True,
            amount=0.65,
            inhibition_samelayer=0.95,
            inhibition_interlayer=1.05,
            gamma_samelayer_rgb=(0.5, 0.4, 0.3),
            gamma_interlayer_r_to_gb=(0.3, 0.25),
            gamma_interlayer_g_to_rb=(0.2, 0.25),
            gamma_interlayer_b_to_rg=(0.15, 0.2),
            diffusion_size_um=5.0,
        ),
    )
    log_raw = np.full((5, 5, 3), -0.9, dtype=float)
    log_raw[:, :, 1] -= 0.15
    log_raw[:, :, 2] -= 0.3
    normalized_density_curves = profile.data.density_curves - np.nanmin(
        profile.data.density_curves, axis=0
    )

    result = develop(
        log_raw.copy(),
        2.5,
        profile.data.log_exposure,
        profile.data.density_curves,
        profile.data.density_curves_layers,
        render.dir_couplers,
        render.grain,
        profile.info.type,
        gamma_factor=gamma_factor,
        use_fast_stats=False,
    )
    expected = develop_simple(
        log_raw.copy(),
        profile.data.log_exposure,
        normalized_density_curves,
        gamma_factor=gamma_factor,
    )
    expected = apply_density_correction_dir_couplers(
        expected,
        log_raw.copy(),
        2.5,
        profile.data.log_exposure,
        normalized_density_curves,
        render.dir_couplers,
        profile.info.type,
        gamma_factor=gamma_factor,
    )
    expected = apply_grain(
        expected,
        2.5,
        render.grain,
        normalized_density_curves,
        profile.data.density_curves_layers,
        profile.info.type,
        use_fast_stats=False,
    )

    np.testing.assert_allclose(result, expected, atol=1e-10)
