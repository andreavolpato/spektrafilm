"""Loaders for the presets bundled in ``spektrafilm/data/presets``."""

from __future__ import annotations

import importlib.resources as pkg_resources
import tomllib


_GRAIN_CHANNELS = 3


def _read_presets(filename, transform):
    """Decode a preset TOML from ``spektrafilm.data.presets`` and transform it.

    ``transform`` maps the raw decoded mapping into runtime-shaped values; it
    is what distinguishes one preset family from another (grain pads to a fixed
    channel count, couplers keep each gamma's native arity).
    """
    package = pkg_resources.files('spektrafilm.data.presets')
    resource = package / filename
    with resource.open("rb") as file:
        data = tomllib.load(file)
    return transform(data)


def _to_tuples(obj):
    """Recursively convert TOML arrays to tuples, leaving scalars and tables.

    For presets whose leaf values are fixed-length per-channel vectors that map
    directly onto runtime tuple fields (e.g. the coupler gammas: a 3-vector
    same-layer term and 2-vector interlayer terms). Unlike ``_to_grain_values``
    it does not pad to a channel count — each field keeps its own arity.
    """
    if isinstance(obj, dict):
        return {key: _to_tuples(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return tuple(_to_tuples(item) for item in obj)
    return obj


def _to_grain_values(obj):
    """Recursively convert TOML grain tables to runtime values.

    Grain fields are per-channel (RGB) ``tuple[float, float, float]``; TOML
    decodes arrays as lists. Recursing (rather than flattening one level)
    keeps nested tables such as ``[defaults.color]`` intact. Leaf arrays are
    padded to ``_GRAIN_CHANNELS`` with ``None`` for any channel the preset
    omits, e.g. a single-value B&W ``rms_granularity = [14.0]`` becomes
    ``(14.0, None, None)``. A single-channel B&W emulsion reads only the first
    channel (see ``match_channels``), so the ``None`` fillers are never
    consumed — they make a partial preset explicit instead of silently
    repeating channel 0.
    """
    if isinstance(obj, dict):
        return {key: _to_grain_values(value) for key, value in obj.items()}
    if isinstance(obj, list):
        padded = list(obj) + [None] * (_GRAIN_CHANNELS - len(obj))
        return tuple(padded[:_GRAIN_CHANNELS])
    return obj


def read_grain_presets():
    """Load grain presets shipped in ``spektrafilm/data/presets``.

    Returns the decoded TOML as a mapping ``stock -> {parameter: (r, g, b)}``,
    plus a ``defaults`` table holding ``color`` / ``bw`` fallbacks. Arrays are
    converted to per-channel tuples (padded with ``None``) to match the
    runtime grain fields.
    """
    return _read_presets('grain.toml', _to_grain_values)


def read_coupler_presets():
    """Load DIR-coupler gamma presets shipped in ``spektrafilm/data/presets``.

    Returns the decoded TOML as a mapping ``stock -> {parameter: tuple}`` plus
    a nested ``defaults`` table holding
    ``defaults.<color|bw>.<positive|negative>`` fallbacks. Arrays are converted
    to tuples of their native length (3-vector same-layer gammas, 2-vector
    interlayer gammas) to match the ``DirCouplersParams`` fields.
    """
    return _read_presets('couplers.toml', _to_tuples)

