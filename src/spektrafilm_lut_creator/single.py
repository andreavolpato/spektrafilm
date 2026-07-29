"""Single-pair LUT building — the params-first entry point.

``build_luts(params, spec)`` bakes LUTs for exactly the (film, print)
pair carried by a runtime params object. This is the API a GUI calls:
the caller's ``RuntimePhotoParams`` *is* the simulator description
(chemistry, bases, couplers, gamut compression, enlarger look), and the
:class:`LutSpec` carries only what the LUT artifact itself needs —
transport color spaces, exposure gain, topology, resolution, delivery.

Per-image trims and un-bakeable effects in the caller's params are
neutralized by the runtime's lut_mode digest (neutral-bake contract);
the neutralized values are disclosed in the bundle's params snapshot.
Multi-print bundles keep using :class:`BundleSpec` +
:class:`BundleBuilder` directly.

See spektrafilm-research/studies/b00/b60_lut_compute_with_params/n010
§6/§8.
"""

from __future__ import annotations

from dataclasses import dataclass

from spektrafilm_lut_creator.builders import BundleBuilder
from spektrafilm_lut_creator.bundles import Bundle, BundleSpec


@dataclass(frozen=True)
class LutSpec:
    """LUT-artifact description for a single (film, print) pair.

    Everything simulator-related is deliberately absent — it lives on
    the ``RuntimePhotoParams`` passed to :func:`build_luts`. Field
    semantics match their :class:`BundleSpec` counterparts.
    """

    input_color_space: str
    output_color_space: str
    topology: str = "1lut"
    resolution: int = 33
    exposure_ev: float = 0.0
    include_combinations: bool = False
    name: str = ""
    target: str | None = None
    container: str = "directory"
    ocio_config: bool = False
    qa: bool = False


def _bundle_spec(params, spec: LutSpec) -> BundleSpec:
    """The internal multi-print spec for this single pair.

    Profile identities come from the params object (no double entry);
    gamut compression is carried by ``params.io`` — the runtime owns it.
    The spec copies are ignored on the base-params bake path but kept in
    sync for the bundle name and any spec-only consumer.
    """
    return BundleSpec(
        film_profile=params.film.info.stock,
        print_profiles=(params.print.info.stock,),
        input_color_space=spec.input_color_space,
        output_color_space=spec.output_color_space,
        name=spec.name,
        topology=spec.topology,
        resolution=spec.resolution,
        exposure_ev=spec.exposure_ev,
        include_combinations=spec.include_combinations,
        target=spec.target,
        container=spec.container,
        ocio_config=spec.ocio_config,
        qa=spec.qa,
        input_gamut_compress=params.io.input_gamut_compress,
        output_gamut_compress=params.io.output_gamut_compress,
    )


def build_luts(params, spec: LutSpec) -> Bundle:
    """Bake the LUTs for the (film, print) pair in ``params``.

    ``params`` is a pre-digest ``RuntimePhotoParams`` (e.g. the GUI's
    live object built by ``build_params_from_state``). It is never
    mutated. The returned :class:`Bundle` carries the baked cubes, the
    metadata, and ``baked_params`` — the digested params the bake ran
    with, which the QA layer consumes directly.
    """
    return BundleBuilder(_bundle_spec(params, spec), base_params=params).build()


def write_bundle(bundle: Bundle, params, spec: LutSpec, out_dir=None):
    """Write a bundle built by :func:`build_luts`; returns the output path.

    Takes the same ``(params, spec)`` pair so the delivery settings
    (target, container, QA, OCIO) come from the caller's spec rather
    than a lossy reconstruction. QA — when ``spec.qa`` — runs against
    ``bundle.baked_params``, the digested params of the bake.
    """
    return BundleBuilder(_bundle_spec(params, spec), base_params=params).write(
        bundle,
        out_dir,
    )
