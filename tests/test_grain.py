import numpy as np
import pytest

from spektrafilm.model.density_curves import interp_density_cmy_layers
from spektrafilm.model.grain import apply_grain_to_density_layers
from spektrafilm.model.grain import apply_grain
from spektrafilm.runtime.params_schema import GrainParams


pytestmark = pytest.mark.unit


class TestApplyGrain:
    def test_apply_grain_returns_input_when_bypassed_or_inactive(self):
        density_cmy = np.full((3, 3, 3), 0.4)
        density_curves = np.tile(np.linspace(0.0, 2.0, 8)[:, None], (1, 3))
        density_curves_layers = np.tile(density_curves[:, None, :] / 3.0, (1, 3, 1))
        grain = GrainParams(active=False)

        inactive = apply_grain(
            density_cmy.copy(),
            4.0,
            grain,
            density_curves,
            density_curves_layers,
            "negative",
        )
        bypassed = apply_grain(
            density_cmy.copy(),
            4.0,
            GrainParams(active=True),
            density_curves,
            density_curves_layers,
            "negative",
            bypass_grain=True,
        )

        np.testing.assert_allclose(inactive, density_cmy, atol=1e-10)
        np.testing.assert_allclose(bypassed, density_cmy, atol=1e-10)

    @pytest.mark.parametrize("profile_type", ["negative", "positive"])
    def test_apply_grain_matches_layered_pipeline(self, profile_type):
        # The streaming multilayer path (apply_grain) must be bit-identical to
        # feeding the whole sub-layer stack to apply_grain_to_density_layers.
        density_cmy = np.full((4, 4, 3), [0.35, 0.55, 0.75], dtype=np.float64)
        density_curves = np.column_stack([
            np.linspace(0.0, 2.1, 10),
            np.linspace(0.0, 1.9, 10),
            np.linspace(0.0, 1.7, 10),
        ])
        density_curves_layers = np.stack([
            density_curves * np.array([0.55, 0.50, 0.45]),
            density_curves * np.array([0.30, 0.33, 0.35]),
            density_curves * np.array([0.15, 0.17, 0.20]),
        ], axis=1)
        grain = GrainParams(
            active=True,
            rms_granularity=(12.0, 16.0, 20.0),
            particle_scale_sublayers=(1.0, 0.5, 0.25),
            density_min=(0.04, 0.06, 0.08),
            uniformity=(0.99, 0.98, 0.97),
            blur=0.0,
            blur_dye_clouds_um=0.0,
            micro_structure=(0.0, 0.0),
            mult_usm_amount=0.0,
        )

        result = apply_grain(
            density_cmy.copy(),
            4.0,
            grain,
            density_curves,
            density_curves_layers,
            profile_type,
            use_fast_stats=False,
        )

        density_cmy_layers = interp_density_cmy_layers(
            density_cmy.copy(),
            density_curves,
            density_curves_layers,
            positive_film=profile_type == "positive",
        )
        expected = apply_grain_to_density_layers(
            density_cmy_layers,
            density_max_layers=np.nanmax(density_curves_layers, axis=0),
            pixel_size_um=4.0,
            rms_granularity=grain.rms_granularity,
            particle_scale_sublayers=grain.particle_scale_sublayers,
            density_min=grain.density_min,
            grain_uniformity=grain.uniformity,
            grain_blur=grain.blur,
            grain_blur_dye_clouds_um=grain.blur_dye_clouds_um,
            grain_micro_structure=grain.micro_structure,
            use_fast_stats=False,
        )

        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_realized_peak_matches_input_rms_single_sublayer(self):
        # With one sub-layer the realized-peak correction is exact: the peak of
        # the grain RMS across density equals the input rms_granularity.
        rms_in = 12.0
        aperture_px = float(np.sqrt(np.pi * 24 ** 2))  # pixel area == 48 um aperture
        density_curves = np.linspace(0.0, 2.2, 48)[:, None]
        grain = GrainParams(
            active=True, rms_granularity=(rms_in, rms_in, rms_in),
            uniformity=(0.97, 0.97, 0.97), particle_scale_sublayers=(1.0, 0.5, 0.25),
            density_min=(0.03, 0.03, 0.03), blur=0.0, blur_dye_clouds_um=0.0,
            micro_structure=(0.0, 0.0), mult_usm_amount=0.0,
        )
        peak = 0.0
        for d in np.linspace(0.05, 2.2, 20):
            img = np.full((256, 256, 1), d)
            out = apply_grain(img.copy(), aperture_px, grain,
                              density_curves, None, "negative")
            peak = max(peak, float(np.std(out[:, :, 0])) * 1000)
        assert peak == pytest.approx(rms_in, rel=0.05)

    def test_apply_grain_single_layer_density_model(self):
        # density_curves_layers=None (single-layer density model) is handled as
        # the one-sub-layer case: grain is applied, finite, mean-preserving-ish.
        density_curves = np.linspace(0.0, 2.2, 16)[:, None]
        density_cmy = np.full((8, 8, 1), 1.0)
        out = apply_grain(
            density_cmy.copy(),
            10.0,
            GrainParams(active=True, blur=0.0, micro_structure=(0.0, 0.0),
                        mult_usm_amount=0.0),
            density_curves,
            None,
            "negative",
        )
        assert out.shape == (8, 8, 1)
        assert np.all(np.isfinite(out))
