"""OCIO 2 standalone config emission for spektrafilm LUT bundles.

Emits a ``config.ocio`` sibling to the bundle's ``.cube`` files. The
config makes the bundle's film simulation appear as a named OCIO
colorspace, so OCIO-managed applications (Nuke, Maya, Houdini, Blender,
Krita, OCIO-aware Resolve modes, ...) can pick it up by setting the
``OCIO`` env var to the bundle's ``config.ocio`` path.

Reference space is ACES2065-1 (AP0 / scene-linear), matching the ACES
Studio Config convention. Every emitted colorspace is asymmetric:
``from_scene_reference`` is defined, ``to_scene_reference`` is not.
That means applications can convert *into* the spektrafilm colorspace
(applying the look) but cannot invert it — inverse LUTs are M13 backlog
work.

YAML emission is hand-rolled (string templating) rather than going
through a YAML library. The output structure is tightly constrained
and the OCIO ``!<Tag>`` tag syntax mangles via PyYAML's safe loader
anyway; a hand-rolled emitter keeps the source readable and avoids a
new runtime dependency. PyOpenColorIO is required only at test time to
validate the produced file loads.

M8a + M8b scope: all four topologies (``1lut`` / ``2lut`` / ``3lut`` /
``4lut``) emit a working config, limited to color-space pairs that have
direct OCIO BuiltinTransforms (see ``_INPUT_BUILTIN`` / ``_OUTPUT_BUILTIN``).
Display + View emission (M8c) is still future work.

For multi-LUT topologies the emitter also declares the bundle's
intermediate taps (``cmy_film_<film>``, ``log_e_film_<film>``,
``log_e_print_<film>_<print>``) as asymmetric colorspaces. Users can
transform *into* them (e.g., for grain injection at the cmy_film tap)
but not out — the .cube files aren't invertible, so the inverse
direction is undefined. Wire constants (``d_min`` / ``d_max`` for
density, ``min`` / ``max`` for log_e) are embedded in each
intermediate's ``description`` so a consumer working at that tap can
decode normalized code values back to physical units.

See ``studies/a40_lut_system/n120_ocio_config_emission.md``.
"""

from __future__ import annotations

from spektrafilm_lut_creator.bundles import Bundle, BundleSpec
from spektrafilm_lut_creator.color_spaces import get as get_color_space
from spektrafilm_lut_creator.metadata import PER_PRINT_LUT_ROLES, SHARED_LUT_ROLES
from spektrafilm_lut_creator.naming import normalize_stock
from spektrafilm_lut_creator.wires import DensityWire, LogEWire

OCIO_PROFILE_VERSION = (2, 4)
"""Major.minor of the emitted ``ocio_profile_version``. OCIO 2.4 syntax
covers everything we use; older 2.x runtimes loading the config will
accept it as long as they understand the BuiltinTransforms referenced."""

REFERENCE_COLORSPACE = "ACES2065-1"
"""The config's scene-reference space. Every other colorspace declares
its transform as ``from_scene_reference`` into this space."""


# Maps registry color-space names to a sequence of (builtin_style, direction)
# pairs whose composition produces "from_scene_reference" (AP0 -> encoded).
#
# The chain is the same regardless of whether the color space appears in
# input or output role on a BundleSpec — sRGB encoded values are sRGB
# encoded values either way. Roles are enforced at the BundleSpec level by
# the color-space registry's ``role`` tuple, not by this map.
#
# Notes on direction:
# - OCIO 2.5's camera-log builtins are <SPACE>_to_ACES2065-1 (going *out
#   of* the camera space). To go AP0 -> camera-encoded we apply with
#   direction="inverse".
# - DISPLAY builtins go CIE-XYZ-D65 -> display-encoded; pair with the AP0
#   -> CIE-XYZ-D65 UTILITY transform.
#
# Empty list = identity (the registry name IS the reference).
_COLORSPACE_BUILTIN: dict[str, list[tuple[str, str]]] = {
    # ACES family.
    "ACES2065-1": [],
    "ACEScg": [("ACEScg_to_ACES2065-1", "inverse")],
    "ACEScct": [("ACEScct_to_ACES2065-1", "inverse")],
    "ACEScc": [("ACEScc_to_ACES2065-1", "inverse")],
    # Camera log spaces with direct AP0 builtins in OCIO 2.5.
    "Panasonic V-Log": [("PANASONIC_VLOG-VGAMUT_to_ACES2065-1", "inverse")],
    "Sony S-Log3": [("SONY_SLOG3-SGAMUT3_to_ACES2065-1", "inverse")],
    "Sony S-Log3 (S-Gamut3.Cine)": [
        ("SONY_SLOG3-SGAMUT3.CINE_to_ACES2065-1", "inverse")
    ],
    "ARRI LogC3 (EI800)": [("ARRI_ALEXA-LOGC-EI800-AWG_to_ACES2065-1", "inverse")],
    "ARRI LogC4": [("ARRI_LOGC4_to_ACES2065-1", "inverse")],
    "Apple Log": [("APPLE_LOG_to_ACES2065-1", "inverse")],
    "Canon Log 3": [("CANON_CLOG3-CGAMUT_to_ACES2065-1", "inverse")],
    "RED Log3G10": [("RED_LOG3G10-RWG_to_ACES2065-1", "inverse")],
    # SDR display spaces. AP0 -> CIE-XYZ-D65 (Bradford CAT) -> encoded.
    "sRGB": [
        ("UTILITY - ACES-AP0_to_CIE-XYZ-D65_BFD", "forward"),
        ("DISPLAY - CIE-XYZ-D65_to_sRGB", "forward"),
    ],
    "Rec.709": [
        ("UTILITY - ACES-AP0_to_CIE-XYZ-D65_BFD", "forward"),
        ("DISPLAY - CIE-XYZ-D65_to_REC.1886-REC.709", "forward"),
    ],
    "Rec.2020": [
        ("UTILITY - ACES-AP0_to_CIE-XYZ-D65_BFD", "forward"),
        ("DISPLAY - CIE-XYZ-D65_to_REC.1886-REC.2020", "forward"),
    ],
    "Display P3": [
        ("UTILITY - ACES-AP0_to_CIE-XYZ-D65_BFD", "forward"),
        ("DISPLAY - CIE-XYZ-D65_to_DisplayP3", "forward"),
    ],
    "DCI-P3": [
        ("UTILITY - ACES-AP0_to_CIE-XYZ-D65_BFD", "forward"),
        ("DISPLAY - CIE-XYZ-D65_to_G2.6-P3-DCI-BFD", "forward"),
    ],
    # HDR display spaces.
    "Rec.2100 PQ": [
        ("UTILITY - ACES-AP0_to_CIE-XYZ-D65_BFD", "forward"),
        ("DISPLAY - CIE-XYZ-D65_to_REC.2100-PQ", "forward"),
    ],
    "Rec.2100 HLG": [
        ("UTILITY - ACES-AP0_to_CIE-XYZ-D65_BFD", "forward"),
        ("DISPLAY - CIE-XYZ-D65_to_REC.2100-HLG-1000nit", "forward"),
    ],
    "P3-D65 PQ": [
        ("UTILITY - ACES-AP0_to_CIE-XYZ-D65_BFD", "forward"),
        ("DISPLAY - CIE-XYZ-D65_to_ST2084-P3-D65", "forward"),
    ],
}

_SUPPORTED_TOPOLOGIES = ("1lut", "2lut", "3lut", "4lut")


def is_supported(spec: BundleSpec) -> bool:
    """Cheap predicate for callers that want to decide before emitting.

    Returns True iff the emitter can produce a working config for ``spec``:
    topology is one of ``_SUPPORTED_TOPOLOGIES`` and both input/output
    spaces appear in the :data:`_COLORSPACE_BUILTIN` table.
    """
    return (
        spec.topology in _SUPPORTED_TOPOLOGIES
        and spec.input_color_space in _COLORSPACE_BUILTIN
        and spec.output_color_space in _COLORSPACE_BUILTIN
    )


def unsupported_reason(spec: BundleSpec) -> str:
    """Human-readable explanation of why ``spec`` cannot be emitted.

    Returns the empty string if ``spec`` is supported.
    """
    if spec.topology not in _SUPPORTED_TOPOLOGIES:
        return (
            f"OCIO emission does not support topology={spec.topology!r}; "
            f"supported: {list(_SUPPORTED_TOPOLOGIES)}"
        )
    if spec.input_color_space not in _COLORSPACE_BUILTIN:
        return (
            f"input color space {spec.input_color_space!r} has no OCIO "
            f"BuiltinTransform mapping; supported inputs: "
            f"{sorted(_COLORSPACE_BUILTIN)}"
        )
    if spec.output_color_space not in _COLORSPACE_BUILTIN:
        return (
            f"output color space {spec.output_color_space!r} has no OCIO "
            f"BuiltinTransform mapping; supported outputs: "
            f"{sorted(_COLORSPACE_BUILTIN)}"
        )
    return ""


def emit_ocio_config(bundle: Bundle, spec: BundleSpec) -> str:
    """Render the OCIO 2 YAML for a built bundle.

    Returns the YAML text. The caller writes it to
    ``<bundle_dir>/config.ocio``.

    Raises :class:`NotImplementedError` for unsupported topology / color
    space combinations; callers that want to skip silently should check
    :func:`is_supported` first or leave ``BundleSpec.ocio_config=False``
    (the default).
    """
    reason = unsupported_reason(spec)
    if reason:
        raise NotImplementedError(reason)

    lines: list[str] = []
    lines.extend(_header_lines(spec))
    lines.extend(_roles_block())
    lines.extend(_displays_block(spec))
    lines.extend(_colorspaces_block(bundle, spec))
    return "\n".join(lines) + "\n"


def _spektrafilm_colorspace_name(film_profile: str, print_profile: str) -> str:
    """Canonical name for the per-print spektrafilm colorspace.

    Mirrors the inline construction in :func:`_spektrafilm_colorspace_yaml`
    so display/view emitters can reference the same colorspace by name.
    """
    return (
        f"spektrafilm_{normalize_stock(film_profile)}_{normalize_stock(print_profile)}"
    )


def _spektrafilm_view_name(film_profile: str, print_profile: str) -> str:
    """Human-readable View name for the per-print spektrafilm View.

    The View name is what shows up in a colorist's viewer dropdown,
    so we humanize the stock identifiers (underscores -> spaces, title
    case) and prefix with "Spektrafilm" to disambiguate from native
    Display views like "Raw".
    """
    return f"Spektrafilm {_humanize_stock(film_profile)} / {_humanize_stock(print_profile)}"


def _humanize_stock(stock: str) -> str:
    """Turn a snake_case stock identifier into a Title Case display name.

    ``kodak_portra_400`` -> ``Kodak Portra 400``. Source profile naming
    is the limit here — "fujifilm_crystal_archive_typeii" produces
    "Fujifilm Crystal Archive Typeii"; cleaner profile names give
    cleaner View labels.
    """
    return stock.replace("_", " ").title()


# ---------------------------------------------------------------------------
# Section emitters.
# ---------------------------------------------------------------------------


def _header_lines(spec: BundleSpec) -> list[str]:
    major, minor = OCIO_PROFILE_VERSION
    prints = ", ".join(spec.print_profiles)
    lines = [
        f"ocio_profile_version: {major}.{minor}",
        "",
        f"name: {_yaml_str(spec.name)}",
        "description: |",
        f"  Standalone OCIO 2 config for the {spec.name} bundle.",
        f"  Spektrafilm: {spec.film_profile} -> {prints}",
        f"  Input: {spec.input_color_space}  Output: {spec.output_color_space}",
        f"  Reference: {REFERENCE_COLORSPACE} (AP0)",
        "  See n120_ocio_config_emission.md in spektrafilm-research for design notes.",
    ]
    if spec.include_combinations:
        # Discoverability nudge for OCIO users curious about the
        # `combinations/` folder. The config itself doesn't reference
        # those cubes — see n130 §6 for rationale.
        lines.extend(
            [
                "  Note: this bundle also ships pre-collapsed sub-chain cubes under",
                "  `combinations/` for single-LUT-slot grading apps. The OCIO config",
                "  references only the canonical chain (see n130 sec 6).",
            ]
        )
    lines.extend(
        [
            "",
            "search_path: .",
            "family_separator: /",
            "",
        ]
    )
    return lines


def _roles_block() -> list[str]:
    # OCIO 2.2+ requires `compositing_log` and `color_timing` roles. We
    # don't ship a dedicated log working space in M8a, so both point at
    # the scene-reference — functional, if not idiomatic. M8c can refine
    # if a more useful target appears.
    return [
        "roles:",
        f"  aces_interchange: {REFERENCE_COLORSPACE}",
        f"  color_timing: {REFERENCE_COLORSPACE}",
        f"  compositing_log: {REFERENCE_COLORSPACE}",
        f"  default: {REFERENCE_COLORSPACE}",
        f"  scene_linear: {REFERENCE_COLORSPACE}",
        "",
    ]


def _displays_block(spec: BundleSpec) -> list[str]:
    """Emit the OCIO ``displays:`` / ``active_displays:`` / ``active_views:``
    blocks.

    Every config declares one Display named after the bundle's output
    color space. The Display's views depend on topology:

    - **1-LUT bundles** (M8c): one View per print named
      ``Spektrafilm <Film> / <Print>`` whose colorspace is the matching
      ``spektrafilm_<film>_<print>`` — colorists pick the look from the
      viewer's View dropdown. A ``Raw`` view re-uses the bare output
      color space (no look) so the user can compare against the
      uncorrected image without leaving the config.
    - **Multi-LUT bundles** (n120 §1): only the ``Raw`` view. The value
      of multi-LUT bundles is *exposing intermediates* via colorspaces;
      a View that hides four chained stages defeats the very reason a
      colorist asked for 4-LUT in the first place.
    """
    out_cs = spec.output_color_space
    views: list[tuple[str, str]] = []

    if spec.topology == "1lut":
        for print_profile in spec.print_profiles:
            views.append(
                (
                    _spektrafilm_view_name(spec.film_profile, print_profile),
                    _spektrafilm_colorspace_name(spec.film_profile, print_profile),
                )
            )
    # Raw view is always emitted last — provides the "no look" comparison
    # point for any topology, including 1-LUT.
    views.append(("Raw", out_cs))

    lines = ["displays:", f"  {_yaml_str(out_cs)}:"]
    for view_name, cs_name in views:
        lines.append(
            f"    - !<View> {{name: {_yaml_str(view_name)}, "
            f"colorspace: {_yaml_str(cs_name)}}}"
        )
    lines.append("")
    lines.append(f"active_displays: [{_yaml_str(out_cs)}]")
    lines.append(f"active_views: [{', '.join(_yaml_str(name) for name, _ in views)}]")
    lines.append("")
    return lines


def _colorspaces_block(bundle: Bundle, spec: BundleSpec) -> list[str]:
    lines = ["colorspaces:"]
    lines.extend(_reference_colorspace_yaml())
    if spec.input_color_space != REFERENCE_COLORSPACE:
        lines.extend(_io_colorspace_yaml(spec.input_color_space, family="Input"))
    if spec.output_color_space != REFERENCE_COLORSPACE:
        lines.extend(_io_colorspace_yaml(spec.output_color_space, family="Output"))

    # Intermediates land before the final per-print spektrafilm
    # colorspaces so OCIO's reference resolution sees them when the
    # final colorspaces' from_scene_reference chains are evaluated.
    # Deduplicated by name: shared intermediates (cmy_film_<film>,
    # log_e_film_<film>) appear once even when multiple prints reference
    # them.
    emitted: set[str] = set()
    for print_profile in spec.print_profiles:
        for inter in _intermediate_specs(spec, bundle, print_profile):
            if inter["name"] in emitted:
                continue
            lines.extend(_intermediate_colorspace_yaml(spec, inter))
            emitted.add(inter["name"])

    # Final spektrafilm colorspace, one per print.
    for print_profile in spec.print_profiles:
        chain = _multilut_chain_for_print(bundle, print_profile)
        lines.extend(_spektrafilm_colorspace_yaml(spec, print_profile, chain))
    return lines


def _reference_colorspace_yaml() -> list[str]:
    entry = get_color_space(REFERENCE_COLORSPACE)
    aliases = ["lin_ap0"]
    if entry.ocio_alias and entry.ocio_alias not in aliases:
        aliases.insert(0, entry.ocio_alias)
    return [
        "  - !<ColorSpace>",
        f"    name: {_yaml_str(REFERENCE_COLORSPACE)}",
        f"    aliases: [{', '.join(_yaml_str(a) for a in aliases)}]",
        "    family: ACES",
        "    encoding: scene-linear",
        "    description: |",
        "      The Academy Color Encoding System reference primaries (AP0)",
        "      in scene-linear encoding. The reference space of this config.",
        "    isdata: false",
        "",
    ]


def _io_colorspace_yaml(name: str, *, family: str) -> list[str]:
    """Emit a ColorSpace YAML block for the bundle's input or output side.

    ``family`` selects the OCIO ``family:`` field (``"Input"`` or
    ``"Output"``); everything else — name, alias, encoding, builtin
    transform chain — comes from the color-space registry lookup.
    """
    entry = get_color_space(name)
    encoding = _encoding_for_kind(entry.kind)
    builtins = _COLORSPACE_BUILTIN[name]

    lines = [
        "  - !<ColorSpace>",
        f"    name: {_yaml_str(name)}",
    ]
    if entry.ocio_alias:
        lines.append(f"    aliases: [{_yaml_str(entry.ocio_alias)}]")
    lines.extend(
        [
            f"    family: {family}",
            f"    encoding: {encoding}",
            "    description: |",
            f"      Bundle {family.lower()} color space: {name}.",
            "    isdata: false",
            "    from_scene_reference: !<GroupTransform>",
            "      children:",
        ]
    )
    lines.extend(_builtin_transform_lines(builtins, indent="        "))
    lines.append("")
    return lines


def _multilut_chain_for_print(bundle: Bundle, print_profile: str) -> list[str]:
    """Return the ordered list of .cube relative paths that compose the
    full input -> output chain for ``print_profile``.

    Walks ``bundle.meta.luts`` in declaration order. Shared-stage cubes
    (e.g., the film LUT in 2-LUT bundles, L1 and L2 in 4-LUT bundles)
    come first; print-specific cubes follow. For 1-LUT this returns a
    single cube; for 4-LUT it returns four.
    """
    chain: list[str] = []
    for lut in bundle.meta.luts:
        if lut.role in SHARED_LUT_ROLES:
            chain.append(lut.path)
        elif lut.print_profile == print_profile and lut.role in PER_PRINT_LUT_ROLES:
            chain.append(lut.path)
    return chain


def _intermediate_specs(
    spec: BundleSpec, bundle: Bundle, print_profile: str
) -> list[dict]:
    """Topology-aware list of intermediate colorspaces to expose for ``print_profile``.

    Each dict has:
      - ``name``: OCIO colorspace name
      - ``family_suffix``: subpath under ``spektrafilm/intermediates/``
      - ``encoding``: OCIO ``encoding`` hint
      - ``description``: multi-line text including wire constants
      - ``chain_relpaths``: ordered .cube paths from input to this tap

    Returns ``[]`` for 1-LUT (no intermediates exposed). For multi-LUT,
    the returned shared-tap entries (e.g. ``cmy_film_<film>``) carry
    identical content across prints; the caller deduplicates by name.
    """
    if spec.topology == "1lut":
        return []

    film_tag = normalize_stock(spec.film_profile)
    print_tag = normalize_stock(print_profile)
    wires = bundle.meta.wires
    chain = _multilut_chain_for_print(bundle, print_profile)
    intermediates: list[dict] = []

    if spec.topology == "2lut":
        # chain = [film.cube, print.cube]; intermediate is cmy_film after film.cube.
        intermediates.append(
            {
                "name": f"cmy_film_{film_tag}",
                "family_suffix": film_tag,
                "encoding": "log",
                "description": _cmy_film_description(wires.cmy_film),
                "chain_relpaths": [chain[0]],
            }
        )

    elif spec.topology == "3lut":
        # chain = [l1.cube, l2.cube, l3_combined.cube].
        intermediates.append(
            {
                "name": f"log_e_film_{film_tag}",
                "family_suffix": film_tag,
                "encoding": "log",
                "description": _log_e_description(wires.log_e_film, stage="film"),
                "chain_relpaths": [chain[0]],
            }
        )
        intermediates.append(
            {
                "name": f"cmy_film_{film_tag}",
                "family_suffix": film_tag,
                "encoding": "log",
                "description": _cmy_film_description(wires.cmy_film),
                "chain_relpaths": [chain[0], chain[1]],
            }
        )

    elif spec.topology == "4lut":
        # chain = [l1.cube, l2.cube, l3.cube, l4.cube].
        intermediates.append(
            {
                "name": f"log_e_film_{film_tag}",
                "family_suffix": film_tag,
                "encoding": "log",
                "description": _log_e_description(wires.log_e_film, stage="film"),
                "chain_relpaths": [chain[0]],
            }
        )
        intermediates.append(
            {
                "name": f"cmy_film_{film_tag}",
                "family_suffix": film_tag,
                "encoding": "log",
                "description": _cmy_film_description(wires.cmy_film),
                "chain_relpaths": [chain[0], chain[1]],
            }
        )
        # log_e_print is per-print: L3's normalized output depends on
        # the print's exposure characteristics. Wire constants are
        # shared across prints (n090 §7), but the cube data differs.
        intermediates.append(
            {
                "name": f"log_e_print_{film_tag}_{print_tag}",
                "family_suffix": f"{film_tag}/{print_tag}",
                "encoding": "log",
                "description": _log_e_description(wires.log_e_print, stage="print"),
                "chain_relpaths": [chain[0], chain[1], chain[2]],
            }
        )

    return intermediates


def _intermediate_colorspace_yaml(spec: BundleSpec, inter: dict) -> list[str]:
    """Emit one intermediate colorspace from an ``_intermediate_specs`` dict."""
    name = inter["name"]
    family_suffix = inter["family_suffix"]
    encoding = inter["encoding"]
    description = inter["description"]
    chain = inter["chain_relpaths"]

    lines = [
        "  - !<ColorSpace>",
        f"    name: {_yaml_str(name)}",
        f"    family: spektrafilm/intermediates/{family_suffix}",
        f"    encoding: {encoding}",
        "    description: |",
    ]
    for desc_line in description.splitlines():
        lines.append(f"      {desc_line}")
    lines.extend(
        [
            "    isdata: false",
            "    from_scene_reference: !<GroupTransform>",
            "      children:",
            f"        - !<ColorSpaceTransform> {{src: {_yaml_str(REFERENCE_COLORSPACE)}, "
            f"dst: {_yaml_str(spec.input_color_space)}}}",
        ]
    )
    for cube_relpath in chain:
        lines.append(
            f"        - !<FileTransform> {{src: {_yaml_str(cube_relpath)}, "
            "interpolation: tetrahedral}"
        )
    lines.append("")
    return lines


def _spektrafilm_colorspace_yaml(
    spec: BundleSpec, print_profile: str, chain_relpaths: list[str]
) -> list[str]:
    """Emit the final spektrafilm colorspace for a (film, print) pair.

    ``chain_relpaths`` is the ordered list of .cube files composing the
    full input -> output transform. Length 1 for 1-LUT bundles; 2 for
    2-LUT; 3 for 3-LUT; 4 for 4-LUT.
    """
    film_tag = normalize_stock(spec.film_profile)
    print_tag = normalize_stock(print_profile)
    cs_name = _spektrafilm_colorspace_name(spec.film_profile, print_profile)
    out_entry = get_color_space(spec.output_color_space)
    encoding = _encoding_for_kind(out_entry.kind)
    n_cubes = len(chain_relpaths)
    topology_desc = "single combined LUT" if n_cubes == 1 else f"chained {n_cubes}-LUT"

    lines = [
        "  - !<ColorSpace>",
        f"    name: {_yaml_str(cs_name)}",
        f"    family: spektrafilm/{film_tag}/{print_tag}",
        f"    encoding: {encoding}",
        "    description: |",
        f"      Spektrafilm film simulation: {spec.film_profile} negative",
        f"      printed on {print_profile}, output as {spec.output_color_space}.",
        f"      Topology: {spec.topology} ({topology_desc}).",
        "      Asymmetric: from_scene_reference defined,",
        "      to_scene_reference undefined (no inverse LUT in this bundle).",
        "    isdata: false",
        "    from_scene_reference: !<GroupTransform>",
        "      children:",
        f"        - !<ColorSpaceTransform> {{src: {_yaml_str(REFERENCE_COLORSPACE)}, "
        f"dst: {_yaml_str(spec.input_color_space)}}}",
    ]
    for cube_relpath in chain_relpaths:
        lines.append(
            f"        - !<FileTransform> {{src: {_yaml_str(cube_relpath)}, "
            "interpolation: tetrahedral}"
        )
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Wire description helpers — embed shaper constants in the intermediate
# colorspace descriptions so a consumer working at that tap can decode
# normalized code values back to physical units (density, log10(E)).
# ---------------------------------------------------------------------------


def _cmy_film_description(wire: DensityWire | None) -> str:
    if wire is None:
        return (
            "Normalized CMY film density after development.\n"
            "Wire constants unavailable; refer to bundle.json."
        )
    d_min = ", ".join(f"{c:.4f}" for c in wire.d_min)
    d_max = ", ".join(f"{c:.4f}" for c in wire.d_max)
    return (
        "Normalized CMY film density after development (per-channel).\n"
        "Encoding: code_c = clip((D_c - d_min_c) / (d_max_c - d_min_c), 0, 1)\n"
        f"  d_min: ({d_min})\n"
        f"  d_max: ({d_max})\n"
        "Decode: D_c = code_c * (d_max_c - d_min_c) + d_min_c\n"
        "Asymmetric: from_scene_reference only. Apply grain or other\n"
        "density-domain effects here; continue the chain with the\n"
        "remaining .cube file(s) directly via FileTransform."
    )


_LOG_E_STAGE_ADVICE: dict[str, str] = {
    "film": (
        "Apply halation, light scattering, or pre-development spatial\n"
        "effects in linear-light exposure (after the 10^ decode); continue\n"
        "the chain with the remaining .cube file(s) directly via FileTransform."
    ),
    "print": (
        "Apply enlarger-stage effects (diffusion filters, dodge/burn) here;\n"
        "continue the chain with the remaining .cube file(s) directly via\n"
        "FileTransform."
    ),
}


def _log_e_description(wire: LogEWire | None, stage: str) -> str:
    """Render the OCIO description block for a normalized log10(E) wire.

    ``stage`` is ``"film"`` or ``"print"`` — selects the stage-specific
    "where to apply effects" sentence at the bottom. Everything else
    (encoding/decode formula, wire constants) is shared.
    """
    if wire is None:
        return (
            f"Normalized log10(exposure) at the {stage}.\n"
            "Wire constants unavailable; refer to bundle.json."
        )
    return (
        f"Normalized log10(exposure) at the {stage} (shared across channels).\n"
        "Encoding: code = (log_e - min) / (max - min)\n"
        f"  min: {wire.min:.4f}\n"
        f"  max: {wire.max:.4f}\n"
        "Decode: log10(E) = code * (max - min) + min; E = 10^log10(E).\n"
        "Asymmetric: from_scene_reference only. " + _LOG_E_STAGE_ADVICE[stage]
    )


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _builtin_transform_lines(
    builtins: list[tuple[str, str]], *, indent: str
) -> list[str]:
    """Emit a list of ``!<BuiltinTransform>`` YAML entries.

    Identity (empty builtins list) falls through to a single
    ``!<MatrixTransform>`` identity so the parent GroupTransform isn't
    childless — OCIO refuses empty children blocks.
    """
    if not builtins:
        return [
            f"{indent}- !<MatrixTransform> "
            "{matrix: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]}"
        ]
    out: list[str] = []
    for style, direction in builtins:
        if direction == "inverse":
            out.append(
                f"{indent}- !<BuiltinTransform> "
                f"{{style: {_yaml_str(style)}, direction: inverse}}"
            )
        else:
            out.append(f"{indent}- !<BuiltinTransform> {{style: {_yaml_str(style)}}}")
    return out


def _encoding_for_kind(kind: str) -> str:
    """Map our registry ``kind`` to OCIO 2's ``encoding`` hint.

    OCIO ``data`` is deliberately not produced here — it pairs with
    ``isdata: true`` which makes OCIO bypass colorspace transforms,
    breaking the chain.
    """
    return {
        "linear": "scene-linear",
        "encoded_sdr": "sdr-video",
        "log": "log",
    }.get(kind, "scene-linear")


def _yaml_str(value: str) -> str:
    """Quote a string for safe inline YAML emission.

    Always emits a double-quoted scalar with backslashes / quotes
    escaped — robust against names containing spaces, hyphens, parens,
    or the colon that would otherwise terminate a flow-mapping key.
    """
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'
