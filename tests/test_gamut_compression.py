"""Unit tests for spektrafilm.utils.gamut_compression.

The algorithms themselves are A/B-validated against ACES RGC and
coloraide in spektrafilm-research/studies/a40_lut_system/
validate_compression_against_references.py. These tests cover the
public contract (spec validation, dispatcher, identity behavior,
LUT remap shape and end-effects).
"""
from __future__ import annotations

import numpy as np
import pytest

from spektrafilm.utils.gamut_compression import (
    InputGamutCompressSpec,
    OutputGamutCompressSpec,
    compress_rgb,
    compress_rgb_aces_rgc,
    compress_rgb_cam16ucs_chroma,
    compress_rgb_jzazbz_chroma,
    compress_rgb_oklch_chroma,
    compress_rgb_oklrab_chroma,
    compress_xy,
    inscribed_locus_hull,
    reinhard_knee,
    remap_tc_lut_for_compression,
    spectral_locus_xy,
)


class TestInputGamutCompressSpec:
    def test_default_is_aces_rgc_inscribed_hull(self):
        s = InputGamutCompressSpec()
        assert s.active is True
        assert s.algorithm == "xy"
        assert s.boundary == "inscribed_hull"
        # hull_detail is the smoothness<->chroma tuning knob; its exact
        # sweet-spot value is under empirical selection (b40), so assert only
        # that it is a valid positive value, not a specific number.
        assert s.hull_detail > 0.0
        assert s.knee == (0.815, 1.0, 1.2)

    def test_inactive_constructs(self):
        s = InputGamutCompressSpec(active=False)
        assert s.active is False

    def test_locus_boundary_constructs(self):
        s = InputGamutCompressSpec(boundary="locus")
        assert s.boundary == "locus"

    def test_custom_hull_detail_constructs(self):
        s = InputGamutCompressSpec(hull_detail=6.0)
        assert s.hull_detail == 6.0

    def test_custom_knee_constructs(self):
        s = InputGamutCompressSpec(knee=(0.7, 1.5, 1.5))
        assert s.knee == (0.7, 1.5, 1.5)

    def test_invalid_algorithm_raises(self):
        # xy is the only input algorithm. jzazbz/cam16ucs were dropped from the
        # input side (measurably less smooth in-locus; still valid for output).
        for algo in ("jzazbz", "oklch", "cam16ucs"):
            with pytest.raises(ValueError, match="algorithm must be"):
                InputGamutCompressSpec(algorithm=algo)

    def test_invalid_boundary_raises(self):
        with pytest.raises(ValueError, match="boundary must be"):
            InputGamutCompressSpec(boundary="hull")

    def test_non_positive_hull_detail_raises(self):
        with pytest.raises(ValueError, match="hull_detail"):
            InputGamutCompressSpec(hull_detail=0.0)

    def test_threshold_out_of_range_raises(self):
        with pytest.raises(ValueError, match="threshold"):
            InputGamutCompressSpec(knee=(1.0, 1.0, 1.2))
        with pytest.raises(ValueError, match="threshold"):
            InputGamutCompressSpec(knee=(-0.1, 1.0, 1.2))

    def test_non_positive_limit_raises(self):
        with pytest.raises(ValueError, match="limit"):
            InputGamutCompressSpec(knee=(0.8, 0.0, 1.2))

    def test_non_positive_power_raises(self):
        with pytest.raises(ValueError, match="power"):
            InputGamutCompressSpec(knee=(0.8, 1.0, 0.0))

    def test_frozen_dataclass(self):
        s = InputGamutCompressSpec()
        with pytest.raises(Exception):
            s.active = False  # type: ignore[misc]


class TestSpectralLocus:
    def test_closed_polygon(self):
        locus = spectral_locus_xy()
        assert locus.shape[1] == 2
        assert np.allclose(locus[0], locus[-1]), "polygon must close on itself"
        assert locus.shape[0] >= 50, "should have enough vertices for a smooth locus"

    def test_in_lower_triangle(self):
        """Every locus vertex must satisfy x >= 0, y >= 0, x + y <= 1
        (modulo floating-point at the rounding edge near 580nm)."""
        locus = spectral_locus_xy()
        assert np.all(locus[:, 0] >= -1e-6)
        assert np.all(locus[:, 1] >= -1e-6)
        assert np.all(locus.sum(axis=-1) <= 1.0 + 1e-6)

    def test_cached(self):
        a = spectral_locus_xy()
        b = spectral_locus_xy()
        assert a is b, "locus polygon should be a cached singleton"


class TestReinhardKnee:
    def test_below_threshold_identity(self):
        d = np.array([0.0, 0.2, 0.5, 0.8])
        out = reinhard_knee(d, threshold=0.815, limit=1.0, power=1.2)
        np.testing.assert_array_equal(out, d)

    def test_above_threshold_strictly_below_input(self):
        d = np.array([0.9, 1.5, 5.0, 100.0])
        out = reinhard_knee(d, threshold=0.815, limit=1.0, power=1.2)
        assert np.all(out < d), "knee must compress, not stretch"

    def test_asymptotes_at_limit(self):
        """As d -> infinity the knee approaches the limit."""
        out = reinhard_knee(
            np.array(1e9), threshold=0.815, limit=1.0, power=1.2
        )
        assert abs(float(out) - 1.0) < 1e-6

    def test_continuous_at_threshold(self):
        eps = 1e-9
        below = reinhard_knee(
            np.array(0.815 - eps), threshold=0.815, limit=1.0, power=1.2,
        )
        above = reinhard_knee(
            np.array(0.815 + eps), threshold=0.815, limit=1.0, power=1.2,
        )
        assert abs(float(above) - float(below)) < 1e-6


class TestCompressXy:
    def setup_method(self):
        self.white = np.array([1 / 3, 1 / 3])
        self.spec = InputGamutCompressSpec()

    def test_inactive_is_identity(self):
        spec = InputGamutCompressSpec(active=False)
        xy = np.array([[0.7, 0.2], [0.1, 0.8]])
        out = compress_xy(xy, self.white, spec)
        np.testing.assert_array_equal(out, xy)

    def test_at_white_unchanged(self):
        out = compress_xy(self.white, self.white, self.spec)
        np.testing.assert_allclose(out, self.white, atol=1e-9)

    def test_in_locus_below_threshold_unchanged(self):
        # A point well inside the locus, below the threshold distance.
        # Pick xy ≈ (0.35, 0.36), very close to white.
        xy = np.array([0.35, 0.36])
        out = compress_xy(xy, self.white, self.spec)
        np.testing.assert_allclose(out, xy, atol=1e-9)

    def test_OOG_xy_pulled_inside_locus(self):
        # V-Gamut red corner direction, way outside the locus.
        xy = np.array([[0.73, 0.28]])
        out = compress_xy(xy, self.white, self.spec)
        # With limit=1.0 the asymptote is at the locus boundary; the
        # output should be at most ~1.0 × locus distance from white.
        d_in = np.linalg.norm(xy[0] - self.white)
        d_out = np.linalg.norm(out[0] - self.white)
        assert d_out < d_in, "OOG input should be pulled in"

    def test_locus_boundary_works(self):
        # boundary="locus" compresses against the raw locus polygon instead of
        # the default inscribed hull; both are valid.
        xy = np.array([[0.7, 0.2], [0.1, 0.8]])
        spec = InputGamutCompressSpec(boundary="locus")
        out = compress_xy(xy, self.white, spec)
        assert out.shape == xy.shape
        assert np.all(np.isfinite(out))


class TestRemapTcLutForCompression:
    def _dummy_lut(self, H=64, W=64):
        """A LUT whose value at each cell encodes the cell's tc index,
        so we can detect which cell got sampled by remap."""
        i, j = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        return np.stack(
            [i / (H - 1), j / (W - 1), 0.5 * np.ones_like(i, dtype=float)],
            axis=-1,
        ).astype(float)

    def test_inactive_is_exact_identity(self):
        lut = self._dummy_lut()
        spec = InputGamutCompressSpec(active=False)
        out = remap_tc_lut_for_compression(
            lut, np.array([1 / 3, 1 / 3]), spec,
        )
        assert np.array_equal(out, lut)

    def test_active_preserves_shape_and_dtype(self):
        lut = self._dummy_lut()
        spec = InputGamutCompressSpec()
        out = remap_tc_lut_for_compression(
            lut, np.array([1 / 3, 1 / 3]), spec,
        )
        assert out.shape == lut.shape
        assert out.dtype == lut.dtype

    def test_active_changes_some_cells(self):
        """The compression should remap at least some cells (those near
        the OOG corners), proving the remap actually fires."""
        lut = self._dummy_lut()
        spec = InputGamutCompressSpec()
        out = remap_tc_lut_for_compression(
            lut, np.array([1 / 3, 1 / 3]), spec,
        )
        assert not np.array_equal(out, lut), "remap should change some cells"

    def test_locus_boundary_works(self):
        lut = self._dummy_lut()
        spec = InputGamutCompressSpec(boundary="locus")
        out = remap_tc_lut_for_compression(
            lut, np.array([1 / 3, 1 / 3]), spec,
        )
        assert out.shape == lut.shape
        assert np.all(np.isfinite(out))

    def test_remap_handles_single_channel_bw(self):
        # The remap is channel-generic (each channel sampled independently), so a
        # 1-channel BW tc_lut must remap correctly rather than being rejected.
        H = W = 32
        i, _ = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        lut = (i / (H - 1))[..., None].astype(float)  # (32, 32, 1)
        spec = InputGamutCompressSpec()
        out = remap_tc_lut_for_compression(
            lut, np.array([1 / 3, 1 / 3]), spec,
        )
        assert out.shape == (H, W, 1)
        assert out.dtype == lut.dtype
        assert np.all(np.isfinite(out))


class TestInscribedLocusHull:
    D65 = np.array([0.3127, 0.3290])  # the fixed compression center

    def test_hull_vertices_inside_locus(self):
        from matplotlib.path import Path as MplPath
        hull = inscribed_locus_hull(self.D65, detail=11.0)
        locus = spectral_locus_xy()
        inside = MplPath(locus).contains_points(hull, radius=1e-9)
        # The hull is inscribed (R_eff <= R); only the single tightest cusp may
        # touch the boundary, so essentially every vertex is inside.
        assert np.mean(inside) > 0.99

    def test_cached(self):
        a = inscribed_locus_hull(self.D65, 11.0)
        b = inscribed_locus_hull(self.D65, 11.0)
        assert a is b, "hull should be cached per (white, detail, locus)"

    def test_higher_detail_hugs_tighter(self):
        # Larger detail keeps more modes/chroma: mean radius from white is larger.
        tight = inscribed_locus_hull(self.D65, 20.0)
        loose = inscribed_locus_hull(self.D65, 5.0)
        r_tight = float(np.linalg.norm(tight - self.D65, axis=-1).mean())
        r_loose = float(np.linalg.norm(loose - self.D65, axis=-1).mean())
        assert r_tight > r_loose

    def test_default_compression_lands_inside_locus(self):
        # The arctic2026 gate: with no out-of-locus inpainting downstream, every
        # compressed chromaticity must land inside the true locus. Sweep a dense
        # grid of wide / imaginary chromaticities (covers V-Gamut, DWG, ACEScg
        # corners and beyond).
        from matplotlib.path import Path as MplPath
        gx, gy = np.meshgrid(
            np.linspace(-0.10, 0.85, 60), np.linspace(-0.15, 1.00, 60),
        )
        xy = np.stack([gx.ravel(), gy.ravel()], axis=-1)
        spec = InputGamutCompressSpec()  # default: inscribed hull, (0.815,1,1.2)
        out = compress_xy(xy, self.D65, spec)
        inside = MplPath(spectral_locus_xy()).contains_points(out, radius=1e-6)
        assert np.all(inside), (
            f"{int((~inside).sum())} of {len(out)} compressed points "
            f"landed outside the locus"
        )


class TestOutputGamutCompressSpec:
    def test_default_is_oklch_with_full_range_soft_knee(self):
        """Default is the OkLab (oklch) perceptual-chroma algorithm with the
        current full-range soft knee. Other algorithms (aces_rgc, cam16ucs,
        jzazbz, oklrab) remain available as opt-in overrides."""
        s = OutputGamutCompressSpec()
        assert s.active is True
        assert s.algorithm == "oklch"
        assert s.knee == (0.0, 1.0, 6.0)

    def test_aces_rgc_constructs(self):
        s = OutputGamutCompressSpec(algorithm="aces_rgc")
        assert s.algorithm == "aces_rgc"

    def test_jzazbz_constructs(self):
        s = OutputGamutCompressSpec(algorithm="jzazbz")
        assert s.algorithm == "jzazbz"

    def test_cam16ucs_constructs(self):
        s = OutputGamutCompressSpec(algorithm="cam16ucs")
        assert s.algorithm == "cam16ucs"

    def test_oklrab_constructs(self):
        s = OutputGamutCompressSpec(algorithm="oklrab")
        assert s.algorithm == "oklrab"

    def test_off_is_inactive(self):
        # algorithm="off" is the canonical "skip compression" toggle;
        # the spec is frozen and exposes `active` as a derived property.
        s = OutputGamutCompressSpec(algorithm="off")
        assert s.active is False

    def test_invalid_algorithm_raises(self):
        with pytest.raises(ValueError, match="algorithm must be"):
            OutputGamutCompressSpec(algorithm="xy")

    def test_invalid_knee_raises(self):
        with pytest.raises(ValueError, match="threshold"):
            OutputGamutCompressSpec(knee=(1.0, 1.0, 1.2))
        with pytest.raises(ValueError, match="limit"):
            OutputGamutCompressSpec(knee=(0.5, -0.1, 1.2))
        with pytest.raises(ValueError, match="power"):
            OutputGamutCompressSpec(knee=(0.5, 1.0, 0.0))

    def test_frozen_dataclass(self):
        s = OutputGamutCompressSpec()
        with pytest.raises(Exception):
            s.algorithm = "off"  # type: ignore[misc]


class TestCompressRgbAcesRgc:
    knee = dict(threshold=0.815, limit=1.0, power=1.2)

    def test_in_gamut_identity(self):
        rgb = np.array([0.5, 0.5, 0.5])
        out = compress_rgb_aces_rgc(rgb, **self.knee)
        np.testing.assert_allclose(out, rgb, atol=1e-12)

    def test_below_threshold_identity(self):
        # ach = 1.0; per-channel d = (1 - 0.3)/1 = 0.7 < threshold 0.815 -> identity.
        rgb = np.array([1.0, 0.3, 0.3])
        out = compress_rgb_aces_rgc(rgb, **self.knee)
        np.testing.assert_allclose(out, rgb, atol=1e-12)

    def test_negative_channels_pulled_back_inside(self):
        """With limit=1.0 the knee's asymptote is at d=1 (c'=0), but
        for finite OOG inputs the output is just shy of the boundary.
        What matters is that negatives become non-negative."""
        rgb = np.array([1.5, -0.1, -0.05])
        out = compress_rgb_aces_rgc(rgb, **self.knee)
        # Max channel unchanged; negatives pulled to >= 0.
        assert out[0] == pytest.approx(1.5, abs=1e-9)
        assert out[1] >= 0.0
        assert out[2] >= 0.0
        # And to *meaningfully* less than the original |c|, not just barely.
        assert out[1] < 0.2
        assert out[2] < 0.2

    def test_more_negative_channels_get_more_compressed(self):
        """Stronger negative inputs land closer to the c'=0 boundary
        because d is larger and the knee's asymptote is at limit=1.0."""
        a = compress_rgb_aces_rgc(
            np.array([1.0, -0.05, -0.05]), **self.knee,
        )
        b = compress_rgb_aces_rgc(
            np.array([1.0, -1.0, -1.0]), **self.knee,
        )
        # Stronger OOG -> smaller (closer to 0) output.
        assert b[1] < a[1]
        assert b[2] < a[2]

    def test_max_channel_never_changes(self):
        """ACES RGC's per-channel formula leaves the achromatic max
        untouched. It compresses the *other* channels relative to it.
        High-amplitude pixels (max > 1) are not touched by ACES RGC; if
        the bundle is shipped via a perceptual algorithm, the
        ``lightness_compression`` handles the residual amplitude. ACES RGC
        users on its own have no amplitude knee — the simulation is
        expected to stay in [0, 1] by physical construction."""
        rgb = np.array([2.0, -0.1, 0.3])
        out = compress_rgb_aces_rgc(rgb, **self.knee)
        assert out[0] == pytest.approx(2.0, abs=1e-9)

    def test_batch_input_preserves_shape(self):
        rgb = np.random.default_rng(0).uniform(-0.2, 1.5, size=(7, 11, 3))
        out = compress_rgb_aces_rgc(rgb, **self.knee)
        assert out.shape == rgb.shape

    def test_all_zero_pixel_falls_back_to_identity(self):
        """Pixels with ach <= 0 (black or below) keep their original
        values; no chromaticity is defined to compress around."""
        rgb = np.array([[0.0, 0.0, 0.0], [-1e-13, 0.0, 0.0]])
        out = compress_rgb_aces_rgc(rgb, **self.knee)
        np.testing.assert_allclose(out, rgb, atol=1e-12)


class TestCompressRgbDispatcher:
    def test_inactive_identity(self):
        rgb = np.array([1.5, -0.1, -0.05])
        spec = OutputGamutCompressSpec(algorithm="off")
        out = compress_rgb(rgb, spec)
        np.testing.assert_array_equal(out, rgb)

    def test_active_pulls_negatives_inside(self):
        # Default algorithm is "oklch" which needs output_color_space.
        rgb = np.array([1.5, -0.1, -0.05])
        spec = OutputGamutCompressSpec()
        out = compress_rgb(rgb, spec, output_color_space="sRGB")
        assert out[1] >= -1e-3
        assert out[2] >= -1e-3

    def test_oklch_requires_output_color_space(self):
        spec = OutputGamutCompressSpec(algorithm="oklch")
        with pytest.raises(ValueError, match="output_color_space is required"):
            compress_rgb(np.array([0.5, 0.5, 0.5]), spec)

    def test_oklch_dispatches_with_output_color_space(self):
        spec = OutputGamutCompressSpec(algorithm="oklch")
        rgb = np.array([1.2, -0.05, -0.05])
        out = compress_rgb(rgb, spec, output_color_space="sRGB")
        assert out.shape == rgb.shape

    def test_jzazbz_requires_output_color_space(self):
        spec = OutputGamutCompressSpec(algorithm="jzazbz")
        with pytest.raises(ValueError, match="output_color_space is required"):
            compress_rgb(np.array([0.5, 0.5, 0.5]), spec)

    def test_jzazbz_dispatches_with_output_color_space(self):
        spec = OutputGamutCompressSpec(algorithm="jzazbz")
        rgb = np.array([1.2, -0.05, -0.05])
        out = compress_rgb(rgb, spec, output_color_space="sRGB")
        assert out.shape == rgb.shape

    def test_cam16ucs_requires_output_color_space(self):
        spec = OutputGamutCompressSpec(algorithm="cam16ucs")
        with pytest.raises(ValueError, match="output_color_space is required"):
            compress_rgb(np.array([0.5, 0.5, 0.5]), spec)

    def test_cam16ucs_dispatches_with_output_color_space(self):
        spec = OutputGamutCompressSpec(algorithm="cam16ucs")
        rgb = np.array([1.2, -0.05, -0.05])
        out = compress_rgb(rgb, spec, output_color_space="sRGB")
        assert out.shape == rgb.shape

    def test_oklrab_requires_output_color_space(self):
        spec = OutputGamutCompressSpec(algorithm="oklrab")
        with pytest.raises(ValueError, match="output_color_space is required"):
            compress_rgb(np.array([0.5, 0.5, 0.5]), spec)

    def test_oklrab_dispatches_with_output_color_space(self):
        spec = OutputGamutCompressSpec(algorithm="oklrab")
        rgb = np.array([1.2, -0.05, -0.05])
        out = compress_rgb(rgb, spec, output_color_space="sRGB")
        assert out.shape == rgb.shape


# All four perceptual-chroma algorithms (oklch, jzazbz, oklrab, cam16ucs)
# share the same algorithm shape: bisect C_max in their respective uniform
# color space and apply the Reinhard knee to ``C / C_max``. The contract
# is therefore identical across them; parametrize rather than repeat.
_PERCEPTUAL_ALGORITHMS = {
    "oklch":    (compress_rgb_oklch_chroma,    dict(threshold=0.815, limit=1.0, power=1.2)),
    "jzazbz":   (compress_rgb_jzazbz_chroma,   dict(threshold=0.815, limit=1.0, power=1.2)),
    "oklrab":   (compress_rgb_oklrab_chroma,   dict(threshold=0.95,  limit=1.0, power=2.0)),
    "cam16ucs": (compress_rgb_cam16ucs_chroma, dict(threshold=0.95,  limit=1.0, power=2.0)),
}


@pytest.mark.parametrize("algorithm", list(_PERCEPTUAL_ALGORITHMS))
class TestPerceptualChromaCompression:
    """Contract shared by all perceptual-chroma reducers. Each algorithm
    is A/B-validated against external references in
    spektrafilm-research; here we pin only the public behavior."""

    def test_in_gamut_approximate_identity(self, algorithm):
        fn, knee = _PERCEPTUAL_ALGORITHMS[algorithm]
        rgb = np.array([0.5, 0.5, 0.5])
        out = fn(rgb, output_color_space="sRGB", **knee)
        np.testing.assert_allclose(out, rgb, atol=1e-3)

    def test_pulls_negatives_inside(self, algorithm):
        fn, knee = _PERCEPTUAL_ALGORITHMS[algorithm]
        rgb = np.array([1.2, -0.1, -0.05])
        out = fn(rgb, output_color_space="sRGB", **knee)
        assert np.all(out >= -1e-3)

    def test_compresses_saturated_cyan(self, algorithm):
        fn, knee = _PERCEPTUAL_ALGORITHMS[algorithm]
        rgb = np.array([-0.2, 1.0, 1.0])
        out = fn(rgb, output_color_space="sRGB", **knee)
        assert -1e-3 <= out[0] <= 1.0 + 1e-3
        assert out[1] > 0.7  # cyan-y still
        assert out[2] > 0.7

    def test_batch_input(self, algorithm):
        fn, knee = _PERCEPTUAL_ALGORITHMS[algorithm]
        rgb = np.random.default_rng(0).uniform(-0.2, 1.3, size=(20, 3))
        out = fn(rgb, output_color_space="sRGB", **knee)
        assert out.shape == rgb.shape
        assert np.all(np.isfinite(out))


class TestPerceptualChromaAlgorithmSpecific:
    """Tests that exercise a single algorithm's specific math rather
    than the shared chroma-reduction contract above."""

    def test_oklch_table_cached_per_color_space(self):
        """The C_max accessor should hit its cache on a second call
        with the same (space, output) pair."""
        from spektrafilm.utils.gamut_compression import _get_output_c_max_table
        first = _get_output_c_max_table("oklch", "sRGB")
        second = _get_output_c_max_table("oklch", "sRGB")
        assert first is second

    def test_oklrab_lr_remap_roundtrip(self):
        """oklrab uses Ottosson's Lr remap on the lightness axis. The
        forward+inverse pair must round-trip and fix L=0, L≈1."""
        from spektrafilm.utils.gamut_compression import (
            _oklab_L_to_oklrab_Lr,
            _oklrab_Lr_to_oklab_L,
        )
        L = np.linspace(0.0, 1.0, 21)
        Lr = _oklab_L_to_oklrab_Lr(L)
        L_back = _oklrab_Lr_to_oklab_L(Lr)
        np.testing.assert_allclose(L_back, L, atol=1e-12)
        assert _oklab_L_to_oklrab_Lr(np.array(0.0)) == pytest.approx(0.0, abs=1e-12)
        assert _oklab_L_to_oklrab_Lr(np.array(1.0)) == pytest.approx(1.0, abs=1e-2)

    def test_oklch_preserves_oklab_lightness(self):
        """Representative lightness-preservation check. All four
        perceptual algorithms preserve L (or the algorithm's analog) by
        construction; we pin OkLch as the canonical reference. If a
        future change broke L preservation it would show here, and the
        same defect would surface in the other algorithms' visual QA."""
        import colour
        rgb = np.array([1.05, 0.1, 0.4])
        out = compress_rgb_oklch_chroma(
            rgb, output_color_space="sRGB",
            threshold=0.815, limit=1.0, power=1.2,
        )
        cs = colour.RGB_COLOURSPACES["sRGB"]
        xyz_in = colour.RGB_to_XYZ(
            rgb, colourspace="sRGB",
            illuminant=cs.whitepoint, apply_cctf_decoding=False,
        )
        xyz_out = colour.RGB_to_XYZ(
            out, colourspace="sRGB",
            illuminant=cs.whitepoint, apply_cctf_decoding=False,
        )
        L_in = float(colour.XYZ_to_Oklab(xyz_in)[0])
        L_out = float(colour.XYZ_to_Oklab(xyz_out)[0])
        assert abs(L_in - L_out) < 1e-3
