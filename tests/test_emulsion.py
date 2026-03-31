import numpy as np
import pytest

from spektrafilm.model.couplers import apply_density_correction_dir_couplers
from spektrafilm.model.emulsion import develop, develop_simple
from spektrafilm.model.grain import apply_grain
from spektrafilm.profiles.io import Profile, ProfileData, ProfileInfo
from spektrafilm.runtime.params_schema import DirCouplersParams, FilmRenderingParams, GrainParams


pytestmark = pytest.mark.unit


def _make_test_profile(profile_type: str) -> Profile:
    log_exposure = np.linspace(-3.0, 1.0, 24)
    density_curves = np.column_stack([
        np.clip(log_exposure + 2.3, 0.2, 2.6),
        np.clip(log_exposure + 2.0, 0.3, 2.3),
        np.clip(log_exposure + 1.7, 0.4, 2.0),
    ])
    density_curves_layers = np.stack([
        density_curves * np.array([0.52, 0.48, 0.45]),
        density_curves * np.array([0.31, 0.34, 0.35]),
        density_curves * np.array([0.17, 0.18, 0.20]),
    ], axis=1)
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
    render = FilmRenderingParams(
        density_curve_gamma=1.15,
        grain=GrainParams(
            active=True,
            sublayers_active=True,
            agx_particle_area_um2=0.18,
            agx_particle_scale=(0.85, 1.0, 1.25),
            agx_particle_scale_layers=(2.2, 1.0, 0.55),
            density_min=(0.04, 0.06, 0.08),
            uniformity=(0.99, 0.98, 0.97),
            blur=0.0,
            blur_dye_clouds_um=0.0,
            micro_structure=(0.0, 0.0),
        ),
        dir_couplers=DirCouplersParams(
            active=True,
            amount=0.65,
            ratio_rgb=(1.0, 0.85, 0.7),
            diffusion_interlayer=1.1,
            diffusion_size_um=5.0,
            high_exposure_shift=0.1,
        ),
    )
    log_raw = np.full((5, 5, 3), -0.9, dtype=float)
    log_raw[:, :, 1] -= 0.15
    log_raw[:, :, 2] -= 0.3
    normalized_density_curves = profile.data.density_curves - np.nanmin(profile.data.density_curves, axis=0)

    result = develop(
        log_raw.copy(),
        2.5,
        profile.data.log_exposure,
        profile.data.density_curves,
        profile.data.density_curves_layers,
        render.dir_couplers,
        render.grain,
        profile.info.type,
        gamma_factor=render.density_curve_gamma,
        use_fast_stats=False,
    )
    expected = develop_simple(
        log_raw.copy(),
        profile.data.log_exposure,
        normalized_density_curves,
        gamma_factor=render.density_curve_gamma,
    )
    expected = apply_density_correction_dir_couplers(
        expected,
        log_raw.copy(),
        2.5,
        profile.data.log_exposure,
        normalized_density_curves,
        render.dir_couplers,
        profile.info.type,
        gamma_factor=render.density_curve_gamma,
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