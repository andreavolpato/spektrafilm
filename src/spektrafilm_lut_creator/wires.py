"""Wire contracts for the 4-LUT chain.

Each non-RGB wire in the chain (log E, normalized density) needs an
explicit shaper that encodes it into the [0,1] range that .cube-class
formats require. The contracts live here; the shaper math lives in
``shapers.py``.

See studies/a40_lut_system/n030_lut_package_design.md §3.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LogEWire:
    """Linear-in-log shaper for log10(E) wires.

    The encoding is ``code = (log_e - min) / (max - min)``; the decode is
    the inverse. Constants are per-bundle and recorded in ``bundle.json``.
    """

    min: float
    max: float


@dataclass(frozen=True)
class DensityWire:
    """Per-channel linear normalization for CMY-density wires.

    The encoding maps ``[d_min, d_max]`` linearly onto ``[0, 1]``
    per channel and clamps:
    ``code_c = clip((D_c - d_min_c) / (d_max_c - d_min_c), 0, 1)``.
    The decode is the inverse (no clamp).

    ``d_max`` and ``d_min`` are per-channel so anisotropic ranges use
    the cube efficiently. ``d_min`` defaults to ``(0, 0, 0)`` — the
    pure-D≥0 case — and is set slightly negative (e.g. ``-0.2``) when
    the wire carries density above-base+fog and a user wants headroom
    for downstream grain models that can dip below zero (grain
    fluctuates *around* the dye density, including in the fog).
    """

    d_max: tuple[float, float, float]
    d_min: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class BoundaryWires:
    """Wires at the pipeline's internal boundary taps for a bundle.

    Carried through ``BundleBuilder._bake_sublut`` so one parameterized
    helper can bake any sub-chain by looking up the wire at its inject /
    collect taps. Fields stay ``None`` when the topology doesn't measure
    that wire — a 1-LUT bake never touches an intermediate wire, a 2-LUT
    bake measures only ``cmy_film``, etc. Input and output color spaces
    (the wires at ``rgb_in`` / ``rgb_out``) are read from
    :class:`BundleSpec` directly, not stored here.
    """

    log_e_film: LogEWire | None = None
    cmy_film: DensityWire | None = None
    log_e_print: LogEWire | None = None


@dataclass(frozen=True)
class RgbWire:
    """RGB wire identified by a registry name + CCTF state.

    ``color_space`` is a name resolvable by the
    :mod:`spektrafilm_lut_creator.color_spaces` registry. ``cctf_applied``
    indicates whether the wire carries encoded (CCTF-applied) RGB or
    decoded linear RGB.
    """

    color_space: str
    cctf_applied: bool
