"""1D shaper math for the LUT wires.

A shaper maps a physical quantity (log10 exposure, CMY density) onto the
[0, 1] code range that .cube-class formats demand, and back. The wire
contracts in :mod:`spektrafilm_lut_creator.wires` declare the constants;
the math here is parameterized on those constants and operates on
arbitrary-shape ndarrays.

See studies/a40_lut_system/n030_lut_package_design.md §3.
"""

from __future__ import annotations

import numpy as np

from spektrafilm_lut_creator.wires import DensityWire, LogEWire


def log_e_to_code(log_e: np.ndarray, wire: LogEWire) -> np.ndarray:
    """Encode log10(E) into [0, 1] code values.

    ``code = (log_e - wire.min) / (wire.max - wire.min)``.
    Values outside ``[wire.min, wire.max]`` map outside ``[0, 1]`` and are
    NOT clipped here; clipping is the caller's responsibility (e.g. when
    sampling values that will subsequently feed a [0,1]-only LUT format).
    """
    span = wire.max - wire.min
    if span <= 0.0:
        raise ValueError(
            f"LogEWire span must be positive: min={wire.min}, max={wire.max}"
        )
    return (np.asarray(log_e, dtype=float) - wire.min) / span


def code_to_log_e(code: np.ndarray, wire: LogEWire) -> np.ndarray:
    """Decode [0, 1] code values back to log10(E)."""
    span = wire.max - wire.min
    if span <= 0.0:
        raise ValueError(
            f"LogEWire span must be positive: min={wire.min}, max={wire.max}"
        )
    return np.asarray(code, dtype=float) * span + wire.min


def _density_wire_endpoints(wire: DensityWire) -> tuple[np.ndarray, np.ndarray]:
    """Validate + return ``(d_min, d_max)`` as length-3 float arrays."""
    d_max = np.asarray(wire.d_max, dtype=float)
    d_min = np.asarray(wire.d_min, dtype=float)
    if d_max.shape != (3,):
        raise ValueError(f"DensityWire.d_max must be 3-tuple, got shape {d_max.shape}")
    if d_min.shape != (3,):
        raise ValueError(f"DensityWire.d_min must be 3-tuple, got shape {d_min.shape}")
    if np.any(d_max - d_min <= 0.0):
        raise ValueError(
            f"DensityWire span (d_max - d_min) must be all-positive per channel; "
            f"got d_min={wire.d_min}, d_max={wire.d_max}"
        )
    return d_min, d_max


def density_to_code(density: np.ndarray, wire: DensityWire) -> np.ndarray:
    """Encode CMY density into [0, 1] code values, per-channel.

    Last axis of ``density`` is the channel axis (length 3). The encoded
    values are clamped to ``[0, 1]``; densities outside ``[d_min, d_max]``
    are truncated to the nearest endpoint (the wire is lossy outside its
    declared range).
    """
    density = np.asarray(density, dtype=float)
    if density.shape[-1] != 3:
        raise ValueError(f"density last axis must be 3, got shape {density.shape}")
    d_min, d_max = _density_wire_endpoints(wire)
    return np.clip((density - d_min) / (d_max - d_min), 0.0, 1.0)


def code_to_density(code: np.ndarray, wire: DensityWire) -> np.ndarray:
    """Decode [0, 1] code values back to CMY density, per-channel.

    Codes at 0 map to ``d_min`` (which may be slightly negative when the
    wire reserves headroom below the natural base+fog floor); codes at
    1 map to ``d_max``. No clamping is applied on decode.
    """
    code = np.asarray(code, dtype=float)
    if code.shape[-1] != 3:
        raise ValueError(f"code last axis must be 3, got shape {code.shape}")
    d_min, d_max = _density_wire_endpoints(wire)
    return code * (d_max - d_min) + d_min
