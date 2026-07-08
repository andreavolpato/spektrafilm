"""Bundle specification and assembled-bundle data.

A :class:`BundleSpec` is the user-facing description of what to build.
A :class:`Bundle` is the built result: one or more LUTs plus the
metadata describing them, ready to be written to disk by
:class:`spektrafilm_lut_creator.builders.BundleBuilder`.

See studies/a40_lut_system/n030_lut_package_design.md §6.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from spektrafilm.utils.gamut_compression import (
    InputGamutCompressSpec,
    OutputGamutCompressSpec,
)
from spektrafilm_lut_creator.formats import Lut
from spektrafilm_lut_creator.metadata import BundleMeta


_VALID_TOPOLOGIES = frozenset({"1lut", "2lut", "3lut", "4lut"})
_VALID_CONTAINERS = frozenset({"directory", "zip"})


@dataclass(frozen=True)
class BundleSpec:
    """User-facing description of a LUT bundle to build.

    All three topologies (``1lut``, ``2lut``, ``4lut``) are implemented.
    The bundle's
    on-disk name defaults to the canonical pattern from
    :func:`spektrafilm_lut_creator.naming.default_bundle_name`
    (``spektrafilm_<version>_<film>[_<print>]_<topology>_<in>_<out>``)
    when ``name`` is left as the empty string. Pass an explicit
    ``name`` to override.

    ``print_profiles`` is a tuple of one or more print stocks. For
    multi-print bundles, the auto-name omits the print segment (the
    bundle covers all of them — naming after one is misleading).
    """
    film_profile: str
    print_profiles: tuple[str, ...]
    input_color_space: str
    output_color_space: str
    name: str = ""
    """Bundle name. Auto-computed via
    :mod:`spektrafilm_lut_creator.naming` when empty."""
    topology: str = "1lut"
    resolution: int = 33
    target: str | None = None
    """Registry name of a :class:`DeliveryTarget`, or ``None`` for the
    generic Adobe ``.cube`` path. When set, the builder also writes a
    target-specific file (e.g., a Lumix-strict ``.cube``) and validates
    the input/output color spaces against the target's allowed set."""
    container: str = "directory"
    """How :class:`BundleBuilder.write` packages the bundle on disk.
    ``"directory"`` writes a normal folder. ``"zip"`` writes the folder
    contents, then archives that folder into a sibling ``.zip`` file and
    returns the archive path."""
    qa: bool = False
    """Whether :class:`BundleBuilder.write` should auto-run the QA suite
    after writing the bundle. The reports land at
    ``<bundle>/qa/<per-print-bundle-name>/`` (one folder per QA'd print);
    the QA cache is deleted after the run so the bundle directory stays
    ship-ready."""
    qa_print_index: int | None = None
    """Which print(s) to QA when :attr:`qa` is True. ``None`` (the
    default) runs QA for every print in :attr:`print_profiles`; an
    explicit integer runs QA for only that print. Validated against the
    bundle's print count up-front so a wrong index fails fast at spec
    construction rather than partway through a long build."""
    input_gamut_compress: InputGamutCompressSpec = field(default_factory=InputGamutCompressSpec)
    """Input gamut compression spec (algorithm + Reinhard knee parameters)
    used when baking the per-film tc_lut. Default is the ACES Reference
    Gamut Compression v1.3 cyan threshold and power with the asymptote
    limit reduced to 1.0 so the knee converges exactly at the spectral
    locus boundary (see spektrafilm-research n100 §5). Pass a
    :class:`InputGamutCompressSpec` (or a TOML table whose fields match the
    dataclass) for custom knee tuning or to disable via ``active=False``.
    The chosen spec is forwarded to ``params.io.input_gamut_compress``
    so GUI users and bundle bakes share the same code path."""
    exposure_ev: float = 0.0
    """Deliberate exposure adjustment baked into the LUT, in stops (EV).

    Midgray is pinned by construction (n200): the bake always applies
    :func:`spektrafilm_lut_creator.color_spaces.input_midgray_gain`
    (``0.18 / entry.midgray_linear`` — identity for SDR / camera-log /
    linear inputs, the nits-to-reflectance bridge for PQ / HLG) so a
    properly-exposed source midgray lands exactly on the film's 0.18.
    A camera-log 18% gray (V-Log code 0.4233, S-Log3 code 0.411, …)
    therefore always hits the film's midgray, and an sRGB source maps
    1:1 in linear reflectance.

    ``exposure_ev`` multiplies a further ``2**exposure_ev`` on top —
    the LUT-level equivalent of the camera's exposure-compensation
    dial. ``+1.0`` brightens the source one stop in the film's frame,
    midgray included; use it to deliberately push a source into the
    film's shoulder (e.g. tone-mapped SDR deliveries whose encoded 1.0
    sits only ≈2.47 stops above mid, short of the shoulder).

    Default ``0.0`` — the LUT is a strict colorimetric transform in
    exposure. Non-zero values are disclosed in ``bundle.json``
    (``input_exposure``) and the bundle README.

    White placement follows from the input curve's native headroom;
    repositioning white independently of midgray would require log
    shaping, which the LUT layer deliberately rules out (n150)."""
    output_gamut_compress: OutputGamutCompressSpec | str = field(
        default_factory=lambda: OutputGamutCompressSpec(algorithm="cam16ucs")
    )
    """Output gamut compression spec. Default ``cam16ucs`` algorithm with
    the ACES RGC cyan threshold and power, limit reduced to 1.0 so the
    knee asymptotes at the output cube edge (no hard clip needed). Pass
    an :class:`OutputGamutCompressSpec` for custom knee tuning, or pass
    an algorithm string directly (for example ``"oklch"`` or
    ``"off"``). TOML config loading still accepts a matching table."""
    ocio_config: bool = False
    """Whether the bundle includes a standalone OCIO 2 config file
    (``config.ocio``) alongside its LUT files. Opt-in: most bundles
    are consumed by grading apps that read the ``.cube`` files
    directly (Resolve, Lumix Lab, FFmpeg), and the OCIO config is only
    useful for OCIO-managed pipelines (Nuke, Maya, Houdini, Blender,
    OCIO-aware Resolve modes). Pass ``True`` to emit the config.
    See studies/a40_lut_system/n120_ocio_config_emission.md."""
    include_combinations: bool = False
    """Whether the bundle also ships every contiguous sub-chain of the
    canonical LUTs as pre-collapsed single cubes in a ``combinations/``
    subfolder. For a 4-LUT bundle that's 6 extra cubes per print
    (``l12``, ``l23``, ``l34``, ``l123``, ``l234``, ``l1234``); for
    3-LUT 3 extras; for 2-LUT 1 extra (``l12``); for 1-LUT a no-op.
    Combinations let single-LUT-slot grading apps (Resolve LUT slot,
    Lumix Lab, OBS, FFmpeg, Premiere) apply any sub-chain of the
    canonical chain as one cube — handy when the canonical 4-cube
    chain is impractical. The OCIO config (when ``ocio_config=True``)
    does *not* reference these cubes: it chains the canonical L1..LN
    directly. See studies/a40_lut_system/n130_sub_chain_combinations.md."""

    def __post_init__(self):
        # Accept either canonical registry names ("Rec.2100 PQ") or the
        # entry's short_tag slug ("rec2100pq"); normalize to canonical
        # so every downstream consumer sees the same string.
        from spektrafilm_lut_creator.color_spaces import resolve as _resolve_cs
        object.__setattr__(self, "input_color_space", _resolve_cs(self.input_color_space))
        object.__setattr__(self, "output_color_space", _resolve_cs(self.output_color_space))
        if isinstance(self.exposure_ev, bool) or not isinstance(
            self.exposure_ev, (int, float)
        ):
            raise ValueError(
                f"exposure_ev must be a number (stops); got {self.exposure_ev!r}"
            )
        object.__setattr__(self, "exposure_ev", float(self.exposure_ev))
        if isinstance(self.output_gamut_compress, str):
            object.__setattr__(
                self,
                "output_gamut_compress",
                OutputGamutCompressSpec(algorithm=self.output_gamut_compress),
            )
        if not isinstance(self.output_gamut_compress, OutputGamutCompressSpec):
            raise TypeError(
                "output_gamut_compress must be an OutputGamutCompressSpec "
                "or algorithm string"
            )
        if self.topology not in _VALID_TOPOLOGIES:
            raise ValueError(
                f"topology must be one of {sorted(_VALID_TOPOLOGIES)}, "
                f"got {self.topology!r}"
            )
        if not self.print_profiles:
            raise ValueError("print_profiles must contain at least one entry")
        # Auto-compute the canonical bundle name when not explicitly
        # given. Done here (rather than in the builder) so consumers can
        # rely on ``spec.name`` being populated immediately after
        # construction — bundle dirpaths, on-disk filenames, and the
        # report directory all derive from it.
        if not self.name:
            from spektrafilm_lut_creator.naming import default_bundle_name
            object.__setattr__(self, "name", default_bundle_name(
                film_profile=self.film_profile,
                print_profiles=self.print_profiles,
                topology=self.topology,
                input_color_space=self.input_color_space,
                output_color_space=self.output_color_space,
            ))
        # For 1lut topology with multiple print profiles, the builder
        # bakes one (film, print) cube per print and packs them all
        # into the same bundle. The 2lut / 4lut topologies share the
        # film half across prints; for 1lut each combination is
        # independent.
        if self.resolution < 2:
            raise ValueError(f"resolution must be >= 2, got {self.resolution}")
        if self.container not in _VALID_CONTAINERS:
            raise ValueError(
                f"container must be one of {sorted(_VALID_CONTAINERS)}, "
                f"got {self.container!r}"
            )
        if self.qa_print_index is not None:
            n_prints = len(self.print_profiles)
            if not 0 <= self.qa_print_index < n_prints:
                raise ValueError(
                    f"qa_print_index={self.qa_print_index} is out of range for a bundle "
                    f"with {n_prints} print(s); valid range is [0, {n_prints - 1}] or None"
                )
        if self.target is not None:
            # Deferred to avoid the module-level circular dependency
            # between bundles and delivery_targets (the target registry
            # references format names that bundles also know about).
            from spektrafilm_lut_creator.delivery_targets import get as _get_target
            target = _get_target(self.target)
            if self.input_color_space not in target.valid_inputs:
                raise ValueError(
                    f"target {self.target!r} requires input in "
                    f"{target.valid_inputs}; got {self.input_color_space!r}"
                )
            if self.output_color_space not in target.valid_outputs:
                raise ValueError(
                    f"target {self.target!r} requires output in "
                    f"{target.valid_outputs}; got {self.output_color_space!r}"
                )

    def bake_frame(self):
        """Return the :class:`BakeFrame` describing this spec's gain
        context. Convenience shortcut for
        :meth:`color_spaces.BakeFrame.from_spec`."""
        from spektrafilm_lut_creator.color_spaces import BakeFrame
        return BakeFrame.from_spec(self)


@dataclass
class Bundle:
    """A built bundle ready to write to disk.

    ``luts`` is a list of ``(relative_path, Lut)`` pairs; the path is
    relative to the bundle's root directory and is the on-disk location
    of the LUT file when written. ``meta`` is the typed metadata payload
    that becomes ``bundle.json`` on disk.

    ``baked_params`` maps each print stock to the **digested**
    ``RuntimePhotoParams`` its cubes were baked with. It is the single
    source of truth for anything that needs the bake's simulator again —
    the QA reference pipeline reads it instead of reconstructing params
    from the spec, so reference and bake cannot diverge. Runtime-only
    (not serialized); ``bundle.json`` carries the asdict snapshot.
    """
    luts: list[tuple[str, Lut]] = field(default_factory=list)
    meta: BundleMeta = field(default_factory=BundleMeta)
    baked_params: dict = field(default_factory=dict)
