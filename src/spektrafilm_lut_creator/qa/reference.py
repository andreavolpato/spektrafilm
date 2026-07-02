"""Reference samples — pipeline ground truth at off-grid points.

The QA suite needs to compare ``LUT(x)`` against ``pipeline(x)`` at
many off-grid points ``x``. Running the spektrafilm pipeline is the
expensive part; we do it once per (bundle, print) and cache the
result for the lifetime of the QA run.

What's cached:

- ``rng_samples_encoded``: uniformly-distributed random RGB samples
  in the *encoded* input color space (matches what a real-world host
  would feed the LUT).
- ``pipeline_out_encoded``: spektrafilm output at those samples, in
  the *encoded* output space (matches what the LUT produces).

The cache key is a SHA256 over the bundle spec and bundle metadata.
Stored as a single ``.npz`` per (bundle, print) under
``<out_dir>/cache/``.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from spektrafilm_lut_creator.bundles import Bundle, BundleSpec
from spektrafilm_lut_creator.color_spaces import (
    decode_cctf,
    encode_cctf,
    get as get_color_space,
)


# Off-grid sample count. 50k random points over [0,1]^3 — dense enough
# to characterize a 33^3 cube's interpolation error well, cheap enough
# to compute on a laptop in seconds. Reproducible via fixed seed.
DEFAULT_OFFGRID_SAMPLES = 50_000
DEFAULT_RNG_SEED = 20260515


@dataclass
class ReferenceSamples:
    """Pipeline ground truth at off-grid sample points.

    All arrays have shape ``(M, 3)``.

    Attributes
    ----------
    rng_samples_encoded
        Input samples in the bundle's encoded input space, ``[0, 1]``.
    rng_samples_linear
        Same samples in linear input space (CCTF-decoded).
    pipeline_out_linear
        Spektrafilm pipeline output in linear output primaries,
        clipped by physics to ``[0, 1]``.
    pipeline_out_encoded
        Pipeline output in the encoded output space — the
        ground-truth values a perfect LUT would produce.
    cache_key
        SHA256 of the bundle spec + metadata used to invalidate.
    """
    rng_samples_encoded: np.ndarray
    rng_samples_linear: np.ndarray
    pipeline_out_linear: np.ndarray
    pipeline_out_encoded: np.ndarray
    cache_key: str


def _cache_key(spec: BundleSpec, bundle: Bundle, print_index: int) -> str:
    """SHA256 over the spec + bundle metadata + print id.

    Sensitive to any field that changes the pipeline output. ``print_index``
    is always interpreted as the index into ``bundle.meta.stocks.prints``
    (not into ``bundle.luts``) so the key is stable across topology
    differences in cube layout.
    """
    h = hashlib.sha256()
    # Bump when the reference-compute semantics change (gain plumbing,
    # encoding scale, etc.) so stale .npz caches don't poison QA figures
    # after a code update. "3" = midgray-pinned input gain (n200):
    # input_gain + output_midgray_gain applied in the reference path,
    # matching the builder.
    h.update(b"qa_reference_v3|")
    h.update(repr(asdict(spec)).encode("utf-8"))
    if bundle.meta.stocks is not None:
        print_name = bundle.meta.stocks.prints[print_index]
    else:
        print_name = spec.print_profiles[print_index]
    h.update(print_name.encode("utf-8"))
    h.update(json.dumps({
        "schema_version": bundle.meta.schema_version,
        "topology": bundle.meta.topology,
        "resolution": bundle.meta.resolution,
    }, sort_keys=True).encode("utf-8"))
    return h.hexdigest()[:16]


def _cache_path(cache_dir: Path, spec: BundleSpec, print_index: int) -> Path:
    """``cache/<print_index>__<short_key>.npz`` under ``cache_dir``."""
    return cache_dir / f"print{print_index:02d}__{spec.film_profile}.npz"


def compute_or_load(
    spec: BundleSpec,
    bundle: Bundle,
    print_index: int,
    cache_dir: Path,
    *,
    n_samples: int = DEFAULT_OFFGRID_SAMPLES,
    rng_seed: int = DEFAULT_RNG_SEED,
) -> ReferenceSamples:
    """Return cached reference samples or compute and cache them.

    ``cache_dir`` is created if it doesn't exist.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(spec, bundle, print_index)
    path = _cache_path(cache_dir, spec, print_index)

    if path.exists():
        try:
            data = np.load(path, allow_pickle=False)
            stored_key = str(data["cache_key"])
            if stored_key == key:
                return ReferenceSamples(
                    rng_samples_encoded=data["rng_samples_encoded"],
                    rng_samples_linear=data["rng_samples_linear"],
                    pipeline_out_linear=data["pipeline_out_linear"],
                    pipeline_out_encoded=data["pipeline_out_encoded"],
                    cache_key=stored_key,
                )
        except (OSError, KeyError, ValueError):
            # Stale or corrupt cache; recompute.
            pass

    ref = _compute(spec, bundle, print_index, n_samples=n_samples, rng_seed=rng_seed)
    np.savez(
        path,
        rng_samples_encoded=ref.rng_samples_encoded,
        rng_samples_linear=ref.rng_samples_linear,
        pipeline_out_linear=ref.pipeline_out_linear,
        pipeline_out_encoded=ref.pipeline_out_encoded,
        cache_key=np.array(key),
    )
    return ref


def _compute(
    spec: BundleSpec,
    bundle: Bundle,
    print_index: int,
    *,
    n_samples: int,
    rng_seed: int,
) -> ReferenceSamples:
    """Run the spektrafilm pipeline at ``n_samples`` random off-grid points."""
    # Deferred runtime import — matches the builder's pattern, keeps the
    # `import spektrafilm_lut_creator.qa` cost low for callers that only
    # need the result/viz layer.
    from spektrafilm.runtime.params_builder import digest_params, init_params
    from spektrafilm.runtime.pipeline import SimulationPipeline

    rng = np.random.default_rng(rng_seed)
    rng_samples_encoded = rng.uniform(0.0, 1.0, size=(n_samples, 3)).astype(np.float32)
    # Match the builder: decode, then apply the input gain (midgray
    # bridge × exposure_ev) so the pipeline sees the same linear values
    # the baked LUT was built from. Without this, an HDR or re-exposed
    # bundle's reference diverges from the LUT and every comparison
    # figure (drift maps, exposure sweeps, ΔE plots) looks broken.
    # BakeFrame holds the precomputed gains so we don't re-derive them
    # at every QA call site.
    frame = spec.bake_frame()
    rng_samples_linear = (
        decode_cctf(rng_samples_encoded, spec.input_color_space) * frame.input_gain
    ).astype(np.float32)

    in_entry = get_color_space(spec.input_color_space)
    out_entry = get_color_space(spec.output_color_space)

    print_stock = bundle.meta.stocks.prints[print_index] if bundle.meta.stocks else spec.print_profiles[print_index]

    params = init_params(film_profile=spec.film_profile, print_profile=print_stock)
    params.debug.lut_mode = True
    params.io.input_color_space = in_entry.primaries
    params.io.output_color_space = out_entry.primaries
    params.io.input_cctf_decoding = False
    params.io.output_cctf_encoding = False
    params.io.input_gamut_compress = spec.input_gamut_compress
    params.io.output_gamut_compress = spec.output_gamut_compress
    params = digest_params(params)
    pipeline = SimulationPipeline(params)

    # Pipeline expects (H, W, 3); reshape into a long strip. lut_mode
    # turns off every spatial effect so layout is purely a performance
    # knob (see grid.py docstring).
    image_in = rng_samples_linear.reshape(1, n_samples, 3)
    image_out_linear = np.asarray(pipeline.process(image_in), dtype=float).reshape(n_samples, 3)
    # Match the builder's output scaling (BakeFrame.output_gain) so
    # HDR-output QA references aren't crushed black.
    image_out_encoded = encode_cctf(
        image_out_linear * frame.output_gain, spec.output_color_space,
    )
    # Match the builder's final clip: encoded outputs land in [0,1]
    # before the LUT is written, so the reference must too.
    image_out_encoded = np.clip(image_out_encoded, 0.0, 1.0)

    return ReferenceSamples(
        rng_samples_encoded=rng_samples_encoded,
        rng_samples_linear=rng_samples_linear,
        pipeline_out_linear=image_out_linear,
        pipeline_out_encoded=image_out_encoded,
        cache_key=_cache_key(spec, bundle, print_index),
    )


def run_pipeline_at(
    spec: BundleSpec,
    bundle: Bundle,
    print_index: int,
    samples_encoded: np.ndarray,
) -> np.ndarray:
    """Run the pipeline at arbitrary encoded-input samples, return encoded output.

    For ad-hoc patterns (Planckian sweep, near-zero patches, highlight
    ramps) that aren't part of the standard reference cache. Cheap for
    small sample counts (< a few thousand).

    Both input and output are in encoded ``[0, 1]`` form to match the
    LUT-application convention.
    """
    from spektrafilm.runtime.params_builder import digest_params, init_params
    from spektrafilm.runtime.pipeline import SimulationPipeline

    samples_encoded = np.asarray(samples_encoded, dtype=np.float32).reshape(-1, 3)
    frame = spec.bake_frame()
    samples_linear = (
        decode_cctf(samples_encoded, spec.input_color_space) * frame.input_gain
    ).astype(np.float32)

    in_entry = get_color_space(spec.input_color_space)
    out_entry = get_color_space(spec.output_color_space)
    print_stock = bundle.meta.stocks.prints[print_index] if bundle.meta.stocks else spec.print_profiles[print_index]

    params = init_params(film_profile=spec.film_profile, print_profile=print_stock)
    params.debug.lut_mode = True
    params.io.input_color_space = in_entry.primaries
    params.io.output_color_space = out_entry.primaries
    params.io.input_cctf_decoding = False
    params.io.output_cctf_encoding = False
    params.io.input_gamut_compress = spec.input_gamut_compress
    params.io.output_gamut_compress = spec.output_gamut_compress
    params = digest_params(params)
    pipeline = SimulationPipeline(params)

    image_in = samples_linear.reshape(1, samples_linear.shape[0], 3)
    image_out_linear = np.asarray(pipeline.process(image_in), dtype=float).reshape(-1, 3)
    # Match the builder's output scaling so HDR-output QA references
    # aren't crushed black.
    image_out_encoded = encode_cctf(
        image_out_linear * frame.output_gain, spec.output_color_space,
    )
    return np.clip(image_out_encoded, 0.0, 1.0)
