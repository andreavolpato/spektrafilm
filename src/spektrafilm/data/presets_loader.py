"""Loader for the grain presets bundled in ``spektrafilm/data/presets``."""

from __future__ import annotations

import importlib.resources as pkg_resources
import tomllib


_GRAIN_CHANNELS = 3


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
    package = pkg_resources.files('spektrafilm.data.presets')
    resource = package / 'grain_presets.toml'
    with resource.open("rb") as file:
        data = tomllib.load(file)
    return _to_grain_values(data)
