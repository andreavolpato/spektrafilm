"""Channel-generic (BW / n_ch=1) behaviour of the develop/grain/coupler model.

No BW profile exists yet, so these exercise the model functions directly with
single-channel arrays: the closest validation to the eventual scan-from-negative
spike. Color (n_ch=3) is covered by the existing test_couplers/test_grain suites.
"""
import numpy as np
import pytest

from spektrafilm.model.density_curves import (
    interpolate_exposure_to_density,
    interp_density_cmy_layers,
)
from spektrafilm.model.diffusion import apply_halation_um
from spektrafilm.model.grain import (
    apply_grain,
    apply_grain_to_density,
    apply_grain_to_density_layers,
)
from spektrafilm.model.couplers import (
    apply_density_correction_dir_couplers,
    compute_dir_couplers_matrix,
)
from spektrafilm.profiles.io import DensityCurvesModel, Profile, ProfileData, ProfileInfo
from spektrafilm.runtime.params_builder import init_params, digest_params
from spektrafilm.runtime.params_schema import DirCouplersParams, GrainParams, HalationParams
from spektrafilm.runtime.pipeline import SimulationPipeline


pytestmark = pytest.mark.unit


def _synthetic_bw_negative():
    """A minimal single-channel panchromatic BW negative on the working grid."""
    wl = np.arange(380, 781, 5, dtype=float)
    sens = 0.5 * np.exp(-((wl - 560.0) / 130.0) ** 2) + 1e-3
    log_exposure = np.linspace(-3.0, 1.5, 256)
    density_curves = (0.1 + 1.9 / (1 + np.exp(-(log_exposure + 0.5) / 0.4)))[:, None]
    # The runtime derives density curves from the parametric model; give this
    # synthetic stock a one-layer Gaussian-CDF model that matches the logistic
    # above (center -0.5, amplitude 1.9, sigma ~0.64 for the 0.4 logistic scale;
    # the 0.1 base is normalized away in develop).
    density_curves_model = DensityCurvesModel(
        model_type='norm_cdfs',
        centers=[[-0.5]],
        amplitudes=[[1.9]],
        sigmas=[[0.64]],
    )
    data = ProfileData(
        wavelengths=wl,
        log_sensitivity=np.log10(sens)[:, None],
        channel_density=np.ones((wl.size, 1)),
        base_density=np.full(wl.size, 0.05),
        midscale_neutral_density=np.zeros(wl.size),
        log_exposure=log_exposure,
        density_curves=density_curves,
        density_curves_model=density_curves_model,
    )
    info = ProfileInfo(stock='synthetic_bw_pan', name='Synthetic BW Pan',
                       type='negative', support='film', stage='filming',
                       channel_model='bw', reference_illuminant='D55', viewing_illuminant='D50')
    return Profile(info=info, data=data)


@pytest.fixture
def bw_curve():
    """A single-channel negative density curve over a log-exposure axis."""
    log_exposure = np.linspace(-3.0, 1.0, 64)
    density_curves = np.clip(log_exposure + 1.5, 0.0, 2.2)[:, None]  # (64, 1)
    return log_exposure, density_curves


class TestInterpolateExposureToDensity:
    def test_single_channel_output_shape(self, bw_curve):
        log_exposure, density_curves = bw_curve
        log_raw = np.full((8, 8, 1), -0.5)
        out = interpolate_exposure_to_density(log_raw, density_curves, log_exposure, 1.0)
        assert out.shape == (8, 8, 1)
        assert np.all(np.isfinite(out))

    def test_scalar_gamma_expands_to_n_ch(self, bw_curve):
        # gamma_factor as a scalar must expand to n_ch=1, not the old [g, g, g].
        log_exposure, density_curves = bw_curve
        log_raw = np.full((4, 4, 1), -0.5)
        out = interpolate_exposure_to_density(log_raw, density_curves, log_exposure, 1.2)
        assert out.shape == (4, 4, 1)


class TestInterpDensityCmyLayers:
    def test_single_channel_layer_shape(self, bw_curve):
        log_exposure, density_curves = bw_curve
        n_le = density_curves.shape[0]
        n_layers = 3
        # (n_le, n_layers, n_ch=1): each sublayer a fraction of the channel curve.
        density_curves_layers = (
            density_curves[:, None, :] * np.array([0.5, 0.3, 0.2])[None, :, None]
        )
        density_cmy = np.full((8, 8, 1), 1.0)
        out = interp_density_cmy_layers(density_cmy, density_curves, density_curves_layers)
        assert out.shape == (8, 8, n_layers, 1)
        assert np.all(np.isfinite(out))


class TestGrainSingleChannel:
    def test_apply_grain_with_default_runtime_params_single_channel(self):
        density_curves = np.linspace(0.0, 2.2, 32)[:, None]
        density_cmy = np.full((16, 16, 1), 1.0)
        out = apply_grain(
            density_cmy,
            pixel_size_um=10,
            grain=GrainParams(sublayers_active=False),
            density_curves=density_curves,
            density_curves_layers=None,
            profile_type='negative',
        )
        assert out.shape == (16, 16, 1)
        assert np.all(np.isfinite(out))

    def test_apply_grain_to_density_single_channel(self):
        density_cmy = np.full((16, 16, 1), 1.0)
        out = apply_grain_to_density(
            density_cmy,
            pixel_size_um=10,
            particle_scale=[1.0],
            density_min=[0.03],
            density_max_curves=[2.2],
            grain_uniformity=[0.98],
            n_sub_layers=1,
        )
        assert out.shape == (16, 16, 1)
        assert np.all(np.isfinite(out))

    def test_apply_grain_to_density_layers_single_channel(self):
        # This is the allocation-bug regression: the old code sized the output on
        # the sublayer axis and only worked when sublayers == channels == 3.
        n_layers, n_ch = 3, 1
        density_cmy_layers = np.full((16, 16, n_layers, n_ch), 0.5)
        density_max_layers = np.full((n_layers, n_ch), 0.8)
        out = apply_grain_to_density_layers(
            density_cmy_layers,
            density_max_layers=density_max_layers,
            pixel_size_um=10,
            particle_scale=[1.0],
            particle_scale_layers=[3.0, 1.0, 0.3],
            density_min=[0.03],
            grain_uniformity=[0.98],
        )
        assert out.shape == (16, 16, n_ch)
        assert np.all(np.isfinite(out))


class TestHalationSingleChannel:
    def test_apply_halation_with_default_runtime_params_single_channel(self):
        raw = np.zeros((17, 17, 1), dtype=float)
        raw[8, 8, 0] = 1.0
        out = apply_halation_um(raw, HalationParams(), pixel_size_um=10.0)
        assert out.shape == raw.shape
        assert np.all(np.isfinite(out))


class TestDirCouplersSingleChannel:
    def test_matrix_is_1x1_self_inhibition_only(self):
        params = DirCouplersParams(
            inhibition_samelayer=0.9,
            inhibition_interlayer=1.1,
            gamma_samelayer_rgb=(0.5, 0.4, 0.3),
        )
        matrix = compute_dir_couplers_matrix(params, n_ch=1)
        assert matrix.shape == (1, 1)
        # Only the single layer's self-inhibition survives; no inter-image term.
        np.testing.assert_allclose(matrix[0, 0], 0.5 * 0.9)

    def test_three_channel_matrix_unchanged(self):
        # Backward compatibility: default n_ch=3 keeps the full 3x3.
        params = DirCouplersParams()
        assert compute_dir_couplers_matrix(params).shape == (3, 3)
        assert compute_dir_couplers_matrix(params, n_ch=3).shape == (3, 3)

    def test_self_inhibition_still_applies_to_bw(self, bw_curve):
        # Self-inhibition (the adjacency/edge effect) must run for single-emulsion
        # BW; only the inter-image part is dropped.
        log_exposure, density_curves = bw_curve
        ramp = np.linspace(-1.5, 0.5, 8)
        log_raw = np.tile(ramp[None, :, None], (8, 1, 1))  # spatial gradient, (8,8,1)
        density_cmy = interpolate_exposure_to_density(log_raw, density_curves, log_exposure, 1.0)
        dir_couplers = DirCouplersParams(
            active=True,
            amount=0.7,
            inhibition_samelayer=0.9,
            inhibition_interlayer=1.1,
            gamma_samelayer_rgb=(0.5, 0.4, 0.3),
            diffusion_size_um=6.0,
        )
        result = apply_density_correction_dir_couplers(
            density_cmy, log_raw, 2.0, log_exposure, density_curves, dir_couplers, 'negative',
        )
        assert result.shape == (8, 8, 1)
        assert np.all(np.isfinite(result))
        # The self-inhibition must have changed the density relative to no couplers.
        assert not np.allclose(result, density_cmy)


@pytest.mark.integration
class TestBwScanFromNegativePipeline:
    """End-to-end: a single-channel BW negative through the scan-from-negative
    path. Guards the whole channel-generic chain against regression."""

    def _run(self, *, activate_effects: bool = False):
        params = init_params()
        params.film = _synthetic_bw_negative()
        params.io.scan_film = True
        params.io.upscale_factor = 1.0
        params.io.crop = False
        params.camera.auto_exposure = False
        params.camera.exposure_compensation_ev = 0.0
        params.settings.use_enlarger_lut = False
        params.settings.use_scanner_lut = False
        params.settings.neutral_print_filters_from_database = False
        params.settings.apply_hanatos2025_adaptation_window = False
        params.settings.apply_hanatos2025_adaptation_surface = False
        params.debug.deactivate_spatial_effects = not activate_effects
        params.debug.deactivate_stochastic_effects = not activate_effects
        params = digest_params(params)

        pipe = SimulationPipeline(params)
        ramp = np.linspace(0.02, 0.95, 16)
        img = np.repeat(np.repeat(ramp[None, :, None], 6, axis=0), 3, axis=2)  # (6,16,3)
        return pipe.process(img)

    def test_runs_end_to_end_and_is_finite(self):
        out = self._run()
        assert out.shape == (6, 16, 3)
        assert np.all(np.isfinite(out))

    def test_output_is_neutral(self):
        # A neutral BW stock must produce R == G == B (no tint).
        out = self._run()
        spread = np.max(np.abs(out - out.mean(axis=2, keepdims=True)))
        assert spread < 1e-3

    def test_negative_is_monotonic_decreasing(self):
        # Scanning a negative inverts tone: brighter scene -> darker scan.
        out = self._run()
        lum = out[3].mean(axis=1)
        assert np.all(np.diff(lum) < 1e-6)

    def test_runs_with_effects_enabled(self):
        out = self._run(activate_effects=True)
        assert out.shape == (6, 16, 3)
        assert np.all(np.isfinite(out))

    def test_runs_with_luts_enabled(self):
        params = init_params(film_profile='kodak_trix')
        params.io.scan_film = True
        params.io.upscale_factor = 1.0
        params.io.crop = False
        params.camera.auto_exposure = False
        params.camera.exposure_compensation_ev = 0.0
        params.settings.use_enlarger_lut = True
        params.settings.use_scanner_lut = True
        params.settings.neutral_print_filters_from_database = False
        params.settings.apply_hanatos2025_adaptation_window = False
        params.settings.apply_hanatos2025_adaptation_surface = False
        params = digest_params(params)

        pipe = SimulationPipeline(params)
        ramp = np.linspace(0.02, 0.95, 16)
        img = np.repeat(np.repeat(ramp[None, :, None], 6, axis=0), 3, axis=2)
        out = pipe.process(img)

        assert out.shape == (6, 16, 3)
        assert np.all(np.isfinite(out))
