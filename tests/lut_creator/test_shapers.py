"""Tests for the LUT wire shapers (analytic inverse + boundary behavior)."""

from __future__ import annotations

import numpy as np
import pytest

from spektrafilm_lut_creator.shapers import (
    code_to_density,
    code_to_log_e,
    density_to_code,
    log_e_to_code,
)
from spektrafilm_lut_creator.wires import DensityWire, LogEWire


class TestLogEShaper:
    def test_endpoints_map_to_unit_interval(self):
        wire = LogEWire(min=-3.0, max=2.0)
        assert log_e_to_code(np.array(-3.0), wire) == pytest.approx(0.0)
        assert log_e_to_code(np.array(2.0), wire) == pytest.approx(1.0)

    def test_midpoint(self):
        wire = LogEWire(min=-3.0, max=2.0)
        assert log_e_to_code(np.array(-0.5), wire) == pytest.approx(0.5)

    def test_inverse(self):
        wire = LogEWire(min=-3.1, max=2.4)
        log_e = np.linspace(-3.1, 2.4, 17)
        code = log_e_to_code(log_e, wire)
        np.testing.assert_allclose(code_to_log_e(code, wire), log_e, atol=1e-12)

    def test_does_not_clip_outside_range(self):
        wire = LogEWire(min=-3.0, max=2.0)
        # Values outside the declared span produce code values outside [0,1]
        assert log_e_to_code(np.array(-4.0), wire) == pytest.approx(-0.2)
        assert log_e_to_code(np.array(3.0), wire) == pytest.approx(1.2)

    def test_zero_span_raises(self):
        with pytest.raises(ValueError, match="span must be positive"):
            log_e_to_code(np.array(0.0), LogEWire(min=1.0, max=1.0))
        with pytest.raises(ValueError, match="span must be positive"):
            code_to_log_e(np.array(0.0), LogEWire(min=1.0, max=1.0))


class TestDensityShaper:
    def test_per_channel_normalization(self):
        wire = DensityWire(d_max=(2.0, 4.0, 5.0))
        density = np.array([[1.0, 2.0, 2.5]])
        code = density_to_code(density, wire)
        np.testing.assert_allclose(code, [[0.5, 0.5, 0.5]], atol=1e-12)

    def test_inverse_within_range(self):
        wire = DensityWire(d_max=(3.8, 4.1, 3.6))
        density = np.array([[0.0, 1.0, 2.0], [3.0, 2.5, 1.5], [0.5, 0.8, 0.1]])
        code = density_to_code(density, wire)
        np.testing.assert_allclose(code_to_density(code, wire), density, atol=1e-12)

    def test_clamping_above_dmax(self):
        wire = DensityWire(d_max=(2.0, 2.0, 2.0))
        density = np.array([[3.0, 1.0, -0.5]])
        code = density_to_code(density, wire)
        np.testing.assert_allclose(code, [[1.0, 0.5, 0.0]], atol=1e-12)

    def test_rejects_wrong_dmax_shape(self):
        with pytest.raises(ValueError, match="d_max must be 3-tuple"):
            density_to_code(np.zeros((1, 3)), DensityWire(d_max=(1.0, 2.0)))  # type: ignore[arg-type]

    def test_rejects_non_positive_span(self):
        with pytest.raises(ValueError, match="span .*must be all-positive"):
            density_to_code(np.zeros((1, 3)), DensityWire(d_max=(1.0, 0.0, 2.0)))

    def test_rejects_inverted_span(self):
        """d_min above d_max is degenerate; the shaper must refuse it."""
        with pytest.raises(ValueError, match="span .*must be all-positive"):
            density_to_code(
                np.zeros((1, 3)),
                DensityWire(d_max=(2.0, 2.0, 2.0), d_min=(0.0, 3.0, 0.0)),
            )

    def test_d_min_shifts_decode_floor(self):
        """With d_min=(-0.2,...) code=0 maps to D=-0.2 (the reserved fog
        headroom) and code=1 still maps to d_max."""
        wire = DensityWire(d_max=(2.0, 2.0, 2.0), d_min=(-0.2, -0.2, -0.2))
        code = np.array([[0.0, 0.5, 1.0]]).reshape(1, 3)  # one row, 3 channels
        # Per-channel decode: c=0 → -0.2, c=0.5 → midpoint of [-0.2, 2.0] = 0.9, c=1 → 2.0
        decoded = code_to_density(code, wire)
        np.testing.assert_allclose(decoded, [[-0.2, 0.9, 2.0]], atol=1e-12)

    def test_d_min_round_trip_for_below_zero_grain(self):
        """A grain sample that dips below zero must round-trip cleanly when
        it stays within the wire's [d_min, d_max] band."""
        wire = DensityWire(d_max=(3.8, 4.1, 3.6), d_min=(-0.2, -0.2, -0.2))
        density = np.array([[-0.15, 0.0, 0.05], [-0.2, 2.0, 3.6], [-0.05, 1.0, 0.5]])
        code = density_to_code(density, wire)
        # Every sample is inside the band, so nothing clips.
        assert np.all(code >= 0.0) and np.all(code <= 1.0)
        np.testing.assert_allclose(code_to_density(code, wire), density, atol=1e-12)

    def test_rejects_wrong_density_channel_count(self):
        wire = DensityWire(d_max=(2.0, 2.0, 2.0))
        with pytest.raises(ValueError, match="last axis must be 3"):
            density_to_code(np.zeros((4, 4)), wire)
        with pytest.raises(ValueError, match="last axis must be 3"):
            code_to_density(np.zeros((4, 4)), wire)
