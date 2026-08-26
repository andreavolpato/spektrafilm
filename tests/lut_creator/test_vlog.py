"""V-Log–specific guarantees for the LUT creator.

Three families:

1. **Transfer matches the Panasonic spec** — colour-science's V-Log curve and
   V-Gamut primaries reproduce the official V-Log/V-Gamut Reference Manual
   (18% gray → 10-bit 433, 90% → 602, code 1.0 → ~46x reflection, D65 V-Gamut).
2. **Midgray configuration** — V-Log's default input gain is identity: midgray
   is pinned by construction (``input_gain = 0.18 / midgray_linear = 1``).
   This is the regression guard against the old white-anchored override,
   which attenuated the signal and pushed midgray ~2 stops down (see n200).
3. **End-to-end midgray** — a properly-exposed V-Log 18% gray, baked through a
   real vlog→sRGB cube, renders as a neutral mid-tone (output Y ≈ 0.18), not
   the near-black it became under the buggy gain. A neutral log input is also
   exposed consistently with another log curve (S-Log3).

References: Panasonic V-Log/V-Gamut Reference Manual; spektrafilm-research n200.
"""

from __future__ import annotations

import colour
import numpy as np
import pytest

from spektrafilm_lut_creator import color_spaces as cs
from spektrafilm_lut_creator.builders import BundleBuilder
from spektrafilm_lut_creator.bundles import BundleSpec
from spektrafilm_lut_creator.color_spaces import to_xyz_qa
from spektrafilm_lut_creator.qa import evaluators

# V-Log's decode evaluates log() on the unused branch of an internal np.where,
# emitting a benign RuntimeWarning; values are correct within tolerance.
pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")

VLOG = "Panasonic V-Log"
MIDGRAY = 0.18


def _ten_bit(code: float) -> int:
    return int(round(float(code) * 1023))


class TestVLogTransferMatchesPanasonicSpec:
    """colour-science's V-Log/V-Gamut must match Panasonic's published values."""

    def test_18pct_gray_encodes_to_code_433(self):
        code = float(cs.encode_cctf(np.array([MIDGRAY]), VLOG)[0])
        assert code == pytest.approx(0.42331, abs=1e-3)
        assert _ten_bit(code) == 433  # Panasonic Reference Manual, Fig. 2.2

    def test_90pct_white_encodes_to_code_602(self):
        code = float(cs.encode_cctf(np.array([0.90]), VLOG)[0])
        assert code == pytest.approx(0.58817, abs=1e-3)
        assert _ten_bit(code) == 602

    def test_black_is_code_128(self):
        # 0% reflection sits at 10-bit 128 (≈0.125); decoding it returns ~0.
        assert _ten_bit(0.125) == 128
        assert float(cs.decode_cctf(np.array([0.125]), VLOG)[0]) == pytest.approx(
            0.0, abs=1e-2
        )

    def test_code_1_decodes_to_native_headroom(self):
        white = float(cs.decode_cctf(np.array([1.0]), VLOG)[0])
        assert white == pytest.approx(46.0855, rel=1e-4)  # ≈4609% reflection
        assert np.log2(white / MIDGRAY) == pytest.approx(
            8.0, abs=0.01
        )  # ~8 stops above gray

    def test_18pct_code_round_trips_to_018(self):
        code = cs.encode_cctf(np.array([MIDGRAY]), VLOG)
        assert float(cs.decode_cctf(code, VLOG)[0]) == pytest.approx(MIDGRAY, abs=1e-3)

    def test_vgamut_primaries_and_whitepoint(self):
        space = colour.RGB_COLOURSPACES[cs.get(VLOG).primaries]
        np.testing.assert_allclose(
            space.primaries,
            [[0.730, 0.280], [0.165, 0.840], [0.100, -0.030]],
            atol=1e-4,
        )
        np.testing.assert_allclose(space.whitepoint, [0.3127, 0.3290], atol=1e-4)  # D65


class TestVLogMidgrayConfiguration:
    """The fix: V-Log's default input gain is identity — midgray pinned."""

    def test_midgray_linear_is_018(self):
        assert cs.get(VLOG).midgray_linear == MIDGRAY

    def test_default_gain_is_identity(self):
        # Regression guard: the old white-anchored override attenuated
        # V-Log ~2 stops. Midgray-pinned gain is exactly 1.0.
        assert cs.input_midgray_gain(VLOG) == pytest.approx(1.0, abs=1e-6)
        assert cs.input_gain(VLOG) == pytest.approx(1.0, abs=1e-6)

    def test_exposure_ev_scales_gain(self):
        assert cs.input_gain(VLOG, 1.0) == pytest.approx(2.0, abs=1e-6)
        assert cs.input_gain(VLOG, -2.0) == pytest.approx(0.25, abs=1e-6)

    def test_default_spec_is_midgray_pinned(self):
        spec = BundleSpec(
            film_profile="kodak_ektar_100",
            print_profiles=("kodak_ultra_endura",),
            input_color_space="vlog",
            output_color_space="srgb",
        )
        assert spec.exposure_ev == 0.0
        assert cs.input_gain(spec.input_color_space, spec.exposure_ev) == pytest.approx(
            1.0
        )


@pytest.mark.integration
class TestVLogLutMidgrayEndToEnd:
    """Bake a real vlog→sRGB cube and confirm midgray exposure is correct."""

    RES = 25

    @staticmethod
    def _midgray_output_Y(input_cs: str, resolution: int) -> float:
        """Reflectance-scale output luminance Y for the camera's 18% gray,
        applied through a freshly-baked (film, print) cube."""
        spec = BundleSpec(
            film_profile="kodak_ektar_100",
            print_profiles=("kodak_ultra_endura",),
            input_color_space=input_cs,
            output_color_space="srgb",
            topology="1lut",
            resolution=resolution,
        )
        bundle = BundleBuilder(spec).build()
        lut = bundle.luts[0][1]
        entry = cs.get(spec.input_color_space)
        # The camera's true 18% gray as a native input code (0.18 for log/SDR).
        code = np.asarray(
            cs.encode_cctf(
                np.full((1, 3), entry.midgray_linear), spec.input_color_space
            ),
            dtype=float,
        )
        out_enc = evaluators.apply_trilinear(lut.table, code)
        xyz = to_xyz_qa(out_enc, spec.output_color_space)
        return float(np.asarray(xyz)[..., 1].ravel()[0])

    def test_midgray_renders_neutral_not_crushed(self):
        y = self._midgray_output_Y("vlog", self.RES)
        offset_stops = np.log2(y / MIDGRAY)
        # Within half a stop of a faithful 18% (the small residual is the film's
        # own rendering of midgray). Crucially NOT the ~-5 stops the old
        # gain=0.25 bug produced.
        assert abs(offset_stops) < 0.5, (
            f"V-Log midgray off by {offset_stops:+.2f} stops (Y={y:.4f})"
        )
        assert y > 0.09, f"V-Log midgray crushed dark (Y={y:.4f}) — exposure regression"

    def test_midgray_matches_slog3(self):
        # A neutral 18% gray is the same XYZ in any D65-white log space, so V-Log
        # and S-Log3 (identity gain on both) must land midgray at the same output.
        y_vlog = self._midgray_output_Y("vlog", self.RES)
        y_slog3 = self._midgray_output_Y("Sony S-Log3", self.RES)
        assert abs(np.log2(y_vlog / y_slog3)) < 0.2, (
            f"V-Log midgray (Y={y_vlog:.4f}) disagrees with S-Log3 (Y={y_slog3:.4f}) "
            f"— V-Log exposed inconsistently with other log inputs"
        )
