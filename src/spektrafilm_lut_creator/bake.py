"""Bake-time simulator configuration — the single recipe shared by the
builder and the QA reference path.

``bake_params`` is the one place that knows how to turn a runtime params
object into a LUT-bakeable pipeline configuration: it forces the
builder-owned fields (transport io, canonical route, taps, lut_mode) and
digests. Everything else — chemistry, bases, couplers, gamut compression,
enlarger look — flows through from the caller's ``RuntimePhotoParams``
untouched, so a GUI bake reproduces the simulator the user is looking at.
Per-image trims and un-bakeable effects are neutralized by the runtime's
own lut_mode digest block (see b60/n010 §9), not here.

When no base params are given, the simulator is built from
``init_params`` defaults with the spec's gamut-compression fields applied
— the historical CLI behavior.

See spektrafilm-research/studies/b00/b60_lut_compute_with_params/n010.
"""
from __future__ import annotations

import copy

# The canonical bake chain. Every LUT topology samples sub-chains of the
# full film -> print -> scan route; a caller's params may carry any route
# (e.g. a convert-film session), so the builder forces this one.
BAKE_ROUTE = "input > film > print > scan"


def bake_params(spec, print_profile: str, base=None):
    """Build the digested params for baking one (film, print) pair.

    Returns ``(params, digest_changes)`` where ``digest_changes`` is the
    dotted-path diff of what the lut_mode digest changed relative to the
    incoming configuration — the disclosure that lands in the bundle's
    params snapshot ("you had y_filter_shift=12 on screen, the LUT baked
    0").

    ``base`` is the caller's ``RuntimePhotoParams`` (pre-digest, e.g. the
    GUI's live object). It is deep-copied, never mutated. Its profiles are
    kept as-is — including caller-side edits — unless ``print_profile``
    names a different print stock (the multi-print bundle axis), in which
    case that print is loaded fresh. With ``base=None`` the simulator
    starts from ``init_params`` defaults and the spec's gamut-compression
    fields; with a base, gamut compression rides on ``base.io`` (the
    runtime owns it).
    """
    # Deferred runtime imports per the README boundary contract.
    from spektrafilm.data.profiles_loader import load_profile
    from spektrafilm.runtime.params_builder import digest_params, init_params
    from spektrafilm_lut_creator.color_spaces import get as get_color_space

    in_entry = get_color_space(spec.input_color_space)
    out_entry = get_color_space(spec.output_color_space)

    if base is None:
        params = init_params(
            film_profile=spec.film_profile, print_profile=print_profile,
        )
        params.io.input_gamut_compress = spec.input_gamut_compress
        params.io.output_gamut_compress = spec.output_gamut_compress
    else:
        params = copy.deepcopy(base)
        if params.print.info.stock != print_profile:
            params.print = load_profile(print_profile)

    # Builder-owned fields: transport encoding, sampling chain, bake mode.
    params.debug.lut_mode = True
    params.io.input_color_space = in_entry.primaries
    params.io.output_color_space = out_entry.primaries
    params.io.input_cctf_decoding = False
    params.io.output_cctf_encoding = False
    params.workflow.route = BAKE_ROUTE
    params.taps.inject = None
    params.taps.collect = None
    # preview_mode must be cleared *before* digest: its digest branch runs
    # ahead of the lut_mode branch and would zero kernels for the wrong
    # reason (harmless numerically, wrong provenance in the disclosure).
    params.settings.preview_mode = False

    # The disclosure baseline is a *non-lut_mode* digest of the same
    # params: what the caller's ordinary render would run with. Diffing
    # against it isolates exactly what LUT baking changed — stock presets
    # and database filter neutrals apply to both sides and cancel out.
    reference = copy.deepcopy(params)
    reference.debug.lut_mode = False
    reference = digest_params(reference)

    params = digest_params(params)
    digest_changes = _diff(_flatten(reference), _flatten(params))
    return params, digest_changes


def make_pipeline(spec, print_profile: str, base=None):
    """``SimulationPipeline`` configured for LUT baking (see bake_params)."""
    from spektrafilm.runtime.pipeline import SimulationPipeline

    params, _ = bake_params(spec, print_profile, base)
    return SimulationPipeline(params)


def params_snapshot(params, digest_changes: dict) -> dict:
    """Serializable snapshot of the digested bake params.

    The full params tree (so a bake is reproducible from ``bundle.json``
    alone), with the profile payloads replaced by their identity — the
    spectral arrays live on disk under the stock name and are reproducible
    from name + version. ``digest_changes`` (from :func:`bake_params`)
    is included as the ``digest_changes`` key: what lut_mode neutralized
    relative to the caller's configuration.
    """
    from dataclasses import asdict

    snapshot = {
        key: value
        for key, value in asdict(params).items()
        if key not in ("film", "print")
    }
    snapshot["film"] = _profile_identity(params.film)
    snapshot["print"] = _profile_identity(params.print)
    snapshot["digest_changes"] = {
        path: {"from": old, "to": new}
        for path, (old, new) in sorted(digest_changes.items())
    }
    return snapshot


def _profile_identity(profile) -> dict:
    return {
        "stock": profile.info.stock,
        "version": profile.metadata.version,
    }


def _flatten(params) -> dict:
    """Dotted-path -> leaf value for every params field outside the
    profile payloads (asdict deep-copies, so the result is a stable
    before-image even though digest mutates in place)."""
    from dataclasses import asdict

    tree = asdict(params)
    tree.pop("film", None)
    tree.pop("print", None)
    flat: dict = {}

    def walk(prefix: str, node) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                walk(f"{prefix}.{key}" if prefix else key, value)
        else:
            flat[prefix] = node

    walk("", tree)
    return flat


def _diff(before: dict, after: dict) -> dict:
    """``{path: (before, after)}`` for every changed leaf."""
    return {
        path: (before[path], after[path])
        for path in before
        if path in after and after[path] != before[path]
    }
