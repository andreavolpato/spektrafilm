"""Tests for the curated color-space registry.

Three test families per n040 §9:
1. Resolves: each entry's primaries / cctf keys exist in colour-science.
2. Round-trip: from_xyz(to_xyz(rgb)) ≈ rgb across a representative grid.
3. Mid-gray sanity: encode(decode(x)) ≈ x for x = 0.18.
"""

from __future__ import annotations

import colour
import numpy as np
import pytest

from spektrafilm_lut_creator import color_spaces as cs
from spektrafilm_lut_creator.color_spaces import ColorSpaceEntry

# Some camera-log curves in colour-science emit a RuntimeWarning when their
# piecewise branches evaluate log() on a negative intermediate (the result is
# masked / replaced downstream). BT.2100 PQ/HLG emit a ColourUsageWarning
# noting that EOTF and OETF directions are exposed under separate APIs (our
# cctf_encoding/decoding pair handles both directions correctly). Both
# warnings are informational; round-trip values are accurate within our
# tolerances.
pytestmark = [
    pytest.mark.filterwarnings("ignore::RuntimeWarning"),
    pytest.mark.filterwarnings("ignore::colour.utilities.ColourUsageWarning"),
]


_ALL_NAMES = sorted(cs._REGISTRY)
_LINEAR_NAMES = [n for n, e in cs._REGISTRY.items() if e.kind == "linear"]
_ENCODED_SDR_NAMES = [n for n, e in cs._REGISTRY.items() if e.kind == "encoded_sdr"]
_LOG_NAMES = [n for n, e in cs._REGISTRY.items() if e.kind == "log"]


def _round_trip_tolerance(kind: str) -> dict[str, float]:
    """Per-kind atol/rtol pair for RGB→XYZ→RGB round-trip assertions.

    Colour-science stores RGB↔XYZ matrices with finite precision; for some
    colourspaces (sRGB, ProPhoto, DJI D-Gamut) the forward and inverse
    matrices are not perfect inverses, so the linear round-trip costs
    ~1e-4. For *log* spaces, that linear error gets amplified through
    the inverse CCTF: a curve with a steep toe (D-Log, Cineon) can turn
    1e-4 linear into ~1e-2 in code space at deep-shadow samples. We size
    tolerances to what the backend actually delivers, not what the math
    would in principle allow.
    """
    if kind == "linear":
        return dict(atol=1e-3, rtol=1e-3)
    if kind == "encoded_sdr":
        return dict(atol=1e-3, rtol=1e-3)
    if kind == "log":
        # 1e-2 absolute = ~1% — looser than CCTF-only round-trip suggests
        # because some log curves (D-Log) have steep toes that amplify the
        # backend's matrix-inversion error there.
        return dict(atol=1e-2, rtol=1e-2)
    raise AssertionError(f"unknown kind {kind!r}")


class TestRegistryShape:
    def test_registry_is_non_empty(self):
        assert len(cs._REGISTRY) > 0

    def test_output_spaces_are_display_referred(self):
        """Output spaces must be display-referred (encoded SDR or log/HDR);
        scene-linear spaces must not appear on the output side."""
        for name in cs.list_output_spaces():
            entry = cs.get(name)
            assert entry.kind != "linear", (
                f"{name!r}: linear (scene-referred) spaces cannot be output"
            )

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown color space"):
            cs.get("Not A Real Space")


class TestRegistration:
    def test_rejects_unknown_kind(self):
        with pytest.raises(ValueError, match="kind must be"):
            cs.register(
                ColorSpaceEntry(
                    name="_test_bad_kind",
                    primaries="sRGB",
                    cctf=None,
                    kind="bogus",
                    role=("input",),
                )
            )

    def test_rejects_unknown_role(self):
        with pytest.raises(ValueError, match="role entries must be"):
            cs.register(
                ColorSpaceEntry(
                    name="_test_bad_role",
                    primaries="sRGB",
                    cctf=None,
                    kind="linear",
                    role=("flapjack",),
                )
            )

    def test_rejects_unknown_primaries(self):
        with pytest.raises(KeyError, match="primaries"):
            cs.register(
                ColorSpaceEntry(
                    name="_test_bad_primaries",
                    primaries="Not A Gamut",
                    cctf=None,
                    kind="linear",
                    role=("input",),
                )
            )

    def test_rejects_unknown_cctf(self):
        with pytest.raises(KeyError, match="cctf"):
            cs.register(
                ColorSpaceEntry(
                    name="_test_bad_cctf",
                    primaries="sRGB",
                    cctf="Not A CCTF",
                    kind="encoded_sdr",
                    role=("input",),
                )
            )

    def test_linear_must_have_none_cctf(self):
        with pytest.raises(ValueError, match="linear spaces must have cctf=None"):
            cs.register(
                ColorSpaceEntry(
                    name="_test_linear_with_cctf",
                    primaries="sRGB",
                    cctf="sRGB",
                    kind="linear",
                    role=("input",),
                )
            )

    def test_non_linear_must_have_cctf(self):
        with pytest.raises(ValueError, match="requires a cctf"):
            cs.register(
                ColorSpaceEntry(
                    name="_test_nonlinear_no_cctf",
                    primaries="sRGB",
                    cctf=None,
                    kind="encoded_sdr",
                    role=("input",),
                )
            )


@pytest.mark.parametrize("name", _ALL_NAMES)
class TestEntryResolves:
    """Every registered entry resolves in colour-science."""

    def test_primaries_resolves(self, name):
        entry = cs.get(name)
        assert entry.primaries in colour.RGB_COLOURSPACES

    def test_cctf_resolves(self, name):
        entry = cs.get(name)
        if entry.cctf is None:
            assert entry.kind == "linear"
        else:
            assert entry.cctf in colour.CCTF_ENCODINGS


@pytest.mark.parametrize("name", _ALL_NAMES)
class TestRoundTrip:
    """from_xyz(to_xyz(rgb)) ≈ rgb across a representative grid."""

    def test_round_trip_grid(self, name):
        entry = cs.get(name)
        # Log curves typically have a steep linear-to-log transition near
        # zero (the toe); a matrix round-trip error of ~1e-4 in linear
        # space gets amplified to several percent in code space inside
        # that region. We sample above the worst toe codes for log spaces,
        # which also matches actual shooting use (below the toe is below
        # black for camera-log signals).
        rng = np.random.default_rng(0)
        lo = 0.15 if entry.kind == "log" else 0.05
        rgb = rng.uniform(lo, 0.95, size=(64, 3))
        xyz = cs.to_xyz(rgb, name)
        rgb_back = cs.from_xyz(xyz, name)
        np.testing.assert_allclose(rgb_back, rgb, **_round_trip_tolerance(entry.kind))


@pytest.mark.parametrize("name", _ALL_NAMES)
class TestMidGrayCctfRoundTrip:
    """encode(decode(x)) ≈ x at mid-gray for every kind.

    Only the CCTF step is exercised here (no matrix), so tolerances stay
    tight. Most curves in colour-science round-trip to machine precision;
    a handful (notably DJI D-Log) drift by ~1e-7 due to non-cancelling
    piecewise float operations — well below LUT precision but above
    machine epsilon. We size the tolerance to the worst observed.
    """

    def test_midgray_cctf_round_trip(self, name):
        rgb = np.full((3,), 0.18)
        linear = cs.decode_cctf(rgb, name)
        back = cs.encode_cctf(linear, name)
        np.testing.assert_allclose(back, rgb, atol=1e-6)


@pytest.mark.parametrize("name", _LINEAR_NAMES)
class TestLinearCctfIsIdentity:
    def test_decode_is_identity(self, name):
        rgb = np.array([[0.1, 0.5, 0.9], [0.0, 0.18, 1.0]])
        out = cs.decode_cctf(rgb, name)
        np.testing.assert_array_equal(out, rgb)

    def test_encode_is_identity(self, name):
        rgb = np.array([[0.1, 0.5, 0.9], [0.0, 0.18, 1.0]])
        out = cs.encode_cctf(rgb, name)
        np.testing.assert_array_equal(out, rgb)


class TestCanonicalXYZ:
    """Sanity check: sRGB white maps to ≈ D65 XYZ."""

    def test_srgb_white_is_near_d65(self):
        xyz = cs.to_xyz(np.array([1.0, 1.0, 1.0]), "sRGB")
        # D65 reference XYZ in colour-science conventions (Y=1).
        np.testing.assert_allclose(xyz, [0.95046, 1.0, 1.08906], atol=1e-3)

    def test_aces_white_is_near_aces_whitepoint(self):
        xyz = cs.to_xyz(np.array([1.0, 1.0, 1.0]), "ACEScg")
        # ACES uses ~D60-ish whitepoint; Y must be ~1 and X, Z near 0.95 / 1.0.
        assert 0.94 < xyz[0] < 1.0
        assert 0.99 < xyz[1] < 1.01
        assert 0.95 < xyz[2] < 1.05


# ---------------------------------------------------------------------------
# n200 — midgray-anchored input gain
# ---------------------------------------------------------------------------


class TestInputGain:
    """Midgray-pinned gain: the input's native midgray always lands on
    the film's 0.18; ``exposure_ev`` stacks a deliberate re-exposure."""

    def test_reflectance_scale_inputs_are_identity(self):
        # SDR, camera log, and scene-linear all decode with 0.18 ≡ midgray,
        # so the midgray bridge is exactly 1.0 — no hidden re-exposure.
        for name in ("sRGB", "Panasonic V-Log", "Sony S-Log3", "ACEScg"):
            assert cs.input_midgray_gain(name) == 1.0
            assert cs.input_gain(name) == 1.0

    def test_pq_bridges_nits_to_reflectance(self):
        # PQ decodes to absolute nits (midgray_linear = 100 under the
        # SDR-reference-white convention) → gain 0.18/100.
        gain = cs.input_midgray_gain("Rec.2100 PQ")
        np.testing.assert_allclose(gain, 0.18 / 100.0, rtol=1e-9)
        # 100 nits × gain = film midgray, by construction.
        np.testing.assert_allclose(100.0 * gain, 0.18, rtol=1e-9)

    def test_hlg_bridges_nits_to_reflectance(self):
        # HLG decodes to absolute nits on the 1000-nit reference display.
        # Midgray is BT.2408's 18% grey card: HLG signal 38% → 26.24 nits.
        mid = cs.get("Rec.2100 HLG").midgray_linear
        np.testing.assert_allclose(mid, 26.238, rtol=1e-4)
        # The registry midgray must encode back to exactly signal 0.38 —
        # this pins the entry to the BT.2100 EOTF, so a colour-science
        # convention change (peak nits, OOTF) breaks loudly here.
        code = cs.encode_cctf(np.full((1, 3), mid), "Rec.2100 HLG")
        np.testing.assert_allclose(np.asarray(code).ravel(), 0.38, atol=1e-6)
        # And the bridge lands it on the film's 0.18.
        np.testing.assert_allclose(
            mid * cs.input_midgray_gain("Rec.2100 HLG"), 0.18, rtol=1e-9
        )

    def test_hlg_midgray_round_trips_through_output_gain(self):
        # Output side: film 0.18 × output_midgray_gain → 26.24 nits →
        # HLG-encoded 0.38 (not the ~0.05 near-black the missing
        # override used to produce).
        out = cs.encode_cctf(
            np.full((1, 3), 0.18 * cs.output_midgray_gain("Rec.2100 HLG")),
            "Rec.2100 HLG",
        )
        np.testing.assert_allclose(np.asarray(out).ravel(), 0.38, atol=1e-6)

    def test_exposure_ev_scales_gain(self):
        np.testing.assert_allclose(cs.input_gain("sRGB", 1.0), 2.0, rtol=1e-9)
        np.testing.assert_allclose(cs.input_gain("sRGB", -1.0), 0.5, rtol=1e-9)
        np.testing.assert_allclose(
            cs.input_gain("Rec.2100 PQ", 2.0),
            (0.18 / 100.0) * 4.0,
            rtol=1e-9,
        )

    def test_effective_input_midgray_linear(self):
        # The input-native linear value that lands on the film's 0.18:
        # the registry midgray at ev=0, halved per +1 EV.
        assert cs.effective_input_midgray_linear("Panasonic V-Log") == pytest.approx(
            0.18
        )
        assert cs.effective_input_midgray_linear("Rec.2100 PQ") == pytest.approx(100.0)
        assert cs.effective_input_midgray_linear("sRGB", 1.0) == pytest.approx(0.09)

    def test_unknown_input_color_space_raises(self):
        with pytest.raises(KeyError, match="Unknown color space"):
            cs.input_gain("Not A Real Space", 6.0)
