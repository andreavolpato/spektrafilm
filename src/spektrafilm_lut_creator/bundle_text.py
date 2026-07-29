"""Bundle-text rendering — README, cube header, README sub-blocks.

Pure formatting against :class:`BundleMeta`. No bake logic, no I/O.
Split out of :mod:`spektrafilm_lut_creator.builders` so the bake driver
stays focused on running the pipeline; consumers wanting to regenerate
the README or cube headers from an existing ``bundle.json`` can import
from here directly.

The bundle layout filenames (``README.md``, the LUT license) live here
as canonical constants — builders.py imports them when writing the
files on disk so there is one source of truth.
"""

from __future__ import annotations

import textwrap

from spektrafilm_lut_creator.metadata import BundleMeta, LutFileMeta

BUNDLE_README_FILENAME = "README.md"
LUT_LICENSE_FILENAME = "SPEKTRAFILM_LICENSE.txt"


# Human-readable labels for the README's combinations table and file-list
# descriptors. Keyed by the LutFileMeta domain / range string (which is
# the post-_TAP_TO_* alias from builders.py), so consumers reading
# bundle.json with the same vocabulary see consistent names in the
# README.
_TAP_DOMAIN_LABEL: dict[str, str] = {
    "input_rgb": "input RGB",
    "log_e_film": "log_e_film code",
    "cmy_film": "cmy_film code",
    "log_e_print": "log_e_print code",
}

_TAP_RANGE_LABEL: dict[str, str] = {
    "log_e_film": "log_e_film code",
    "cmy_film": "cmy_film code",
    "log_e_print": "log_e_print code",
    "output_rgb": "output RGB",
}


# ---------------------------------------------------------------------------
# Bundle README.
# ---------------------------------------------------------------------------


def bundle_readme_text(
    meta: BundleMeta,
    *,
    qa_results: dict[str, list] | None = None,
) -> str:
    """Render a quick-start README for the bundle root.

    ``qa_results`` is the optional ``{print: [Result, ...]}`` mapping
    produced by :meth:`BundleBuilder._run_qa`. When present, a
    "## Quality" pass/fail badge block is rendered near the top of
    the README (n090 §6.1).
    """
    prov = meta.provenance
    input_cs = meta.color_spaces.get("input")
    output_cs = meta.color_spaces.get("output")
    film_label = meta.stocks.film if meta.stocks is not None else "the film"
    if meta.stocks is not None and meta.stocks.prints:
        print_label = (
            meta.stocks.prints[0]
            if len(meta.stocks.prints) == 1
            else f"{len(meta.stocks.prints)} prints"
        )
    else:
        print_label = "a print"
    lines = [
        "# spektrafilm LUT bundle",
        "",
        "This folder contains exported LUT files plus the machine-readable metadata and license for this bundle.",
        "",
        "## What this is",
        "",
        (
            f"A physically based simulation of {film_label} on "
            f"{print_label}, calibrated against published spectral dye "
            "response and characteristic curves. The cube is the "
            "*neutral developed and printed* render — the technical "
            "transform a perfectly exposed negative would walk through "
            "to land on the print."
        ),
        "",
        "## What this is not",
        "",
        (
            "This is **not** a stylistic look-up grade. There is no "
            "creative color decision baked in beyond the film's own "
            "physics — no extra contrast, no shifted hue, no taste "
            "layer. Aesthetic grading happens *before* (input side) or "
            "*after* (output side) this LUT, not inside it."
        ),
        "",
    ]
    if qa_results:
        lines.extend(_quality_readme_section(qa_results))
    lines.extend(
        [
            "## Quick info",
            f"- Name: {meta.name}",
            f"- Topology: {meta.topology}",
            f"- Resolution: {meta.resolution}^3",
            f"- Delivery target: {meta.target or 'generic Adobe .cube'}",
            f"- Created: {prov.created}",
            f"- spektrafilm version: {prov.spektrafilm_version}",
        ]
    )
    if meta.stocks is not None:
        lines.append(f"- Film stock: {meta.stocks.film}")
        if meta.stocks.prints:
            lines.append(f"- Print stocks: {', '.join(meta.stocks.prints)}")
    if input_cs is not None:
        lines.append(
            f"- Input color space: {input_cs.name} (cctf {'on' if input_cs.cctf else 'off'})"
        )
    if meta.input_exposure is not None:
        exp = meta.input_exposure
        lines.append(
            f"- Input exposure: {exp.exposure_ev:+g} EV baked in "
            f"(linear gain {exp.gain:.4g})"
        )
    if output_cs is not None:
        lines.append(
            f"- Output color space: {output_cs.name} (cctf {'on' if output_cs.cctf else 'off'})"
        )
    lines.extend(
        [
            "",
            "## Files",
            f"- {BUNDLE_README_FILENAME}: this summary",
            "- bundle.json: full metadata payload",
            f"- {LUT_LICENSE_FILENAME}: LUT license text",
        ]
    )
    for lut in meta.luts:
        lines.append(f"- {lut.path}: {_lut_role_description(lut)}")
    lines.extend(_apply_order_block(meta.topology))
    # Combinations section (n130). Driven entirely off the bundle's
    # luts metadata: if any subchain_* entries are present, render the
    # section; otherwise skip silently.
    subchain_luts = tuple(lut for lut in meta.luts if lut.role.startswith("subchain_"))
    if subchain_luts:
        lines.extend(_combinations_readme_section(subchain_luts))
    if meta.input_exposure is not None:
        lines.extend(_input_exposure_block(meta, input_cs, output_cs))
    lines.extend(
        [
            "",
            "## Notes",
            f"- {prov.notes}",
            "- See bundle.json for the complete structured metadata.",
        ]
    )
    return "\n".join(lines) + "\n"


def _lut_role_description(lut: LutFileMeta) -> str:
    """One-line role description used in the README's "Files" listing."""
    if lut.role == "film":
        return "shared film half (L1∘L2: input RGB → normalized cmy_film density)"
    if lut.role == "print":
        return f"print half for {lut.print_profile} (L3∘L4: cmy_film → output RGB)"
    if lut.role == "combined":
        return f"full chain for {lut.print_profile}"
    if lut.role == "filming_expose":
        return "shared L1 — filming.expose (input RGB → normalized log_e_film code)"
    if lut.role == "filming_develop":
        return (
            "shared L2 — filming.develop (log_e_film code → normalized cmy_film code)"
        )
    if lut.role == "printing_expose":
        return (
            f"L3 for {lut.print_profile} — printing.expose "
            f"(cmy_film code → normalized log_e_print code)"
        )
    if lut.role == "printing_develop_scan":
        return (
            f"L4 for {lut.print_profile} — printing.develop + scanning.scan "
            f"(log_e_print code → output RGB)"
        )
    if lut.role == "printing_combined":
        return (
            f"L3 for {lut.print_profile} — printing.expose + develop + scan "
            f"(cmy_film code → output RGB)"
        )
    if lut.role.startswith("subchain_"):
        ids = lut.role[len("subchain_") :]
        scope = f"for {lut.print_profile}" if lut.print_profile else "shared"
        return (
            f"L{ids} sub-chain ({scope}) — pre-collapsed canonical stages {ids} "
            f"({_TAP_DOMAIN_LABEL.get(lut.domain, lut.domain)} → "
            f"{_TAP_RANGE_LABEL.get(lut.range, lut.range)})"
        )
    return lut.role


def _apply_order_block(topology: str) -> list[str]:
    """Render the topology-specific "Apply order" + "Working in the
    intermediate spaces" sections."""
    if topology == "2lut":
        return [
            "",
            "## Apply order",
            "Apply the film LUT first, then the matching print LUT. The two LUTs share the bundle's `cmy_film` wire — film LUT output is normalized density code in `[0, 1]` per channel, and the print LUT expects exactly that. Do not chain a film LUT from one bundle with a print LUT from another; the `d_max` constants differ.",
            "",
            "## Working in the intermediate space",
            "",
            "The `cmy_film` tap between the film and print LUTs exposes normalized film-density code. Decode via `bundle.json/wires/cmy_film` (which carries per-channel `d_min` and `d_max`) to get physical D per channel:",
            "",
            "    D = code * (d_max - d_min) + d_min",
            "",
            "Modify, re-encode to the same `[0, 1]` code range, then feed the print LUT.",
            "",
            "**Base+fog headroom.** The pipeline's `cmy_film` density is reported *above base+fog*, so the deterministic baked value is `>= 0` per channel. `d_min` is set slightly negative (e.g. -0.2) to reserve headroom below the natural floor for downstream grain models — film grain fluctuates around the dye density, including in the fog itself, so noise samples can legitimately dip below zero. Without this headroom those dips would be silently clamped at the wire's [0, 1] boundary.",
            "",
            "Useful effects to inject here:",
            "",
            "- **Grain**: density-modulated noise reproduces real film grain. Magnitude should scale with density (more grain in shadow regions for negative-positive workflows). The `cmy_film` tap is the canonical place — film grain modulates the actual silver / dye granularity, which is what density represents.",
        ]
    if topology == "3lut":
        return [
            "",
            "## Apply order",
            "Apply the three LUTs in order: L1 → L2 → L3.",
            "",
            "- L1 and L2 are shared across all prints in the bundle.",
            "- L3 is print-specific; pick the cube matching the print stock you want.",
            "",
            "Wire contracts (each cube carries [0, 1] code values; the wire describes what those codes represent physically):",
            "",
            "- After L1: normalized `log_e_film` code, with shaper (min, max) in `bundle.json/wires/log_e_film`.",
            "- After L2: normalized `cmy_film` density code, with `d_min` / `d_max` in `bundle.json/wires/cmy_film` (decode: `D = code * (d_max - d_min) + d_min`).",
            "- After L3: encoded RGB in the bundle's output color space. The `log_e_print` and `cmy_print` taps are collapsed inside L3 — they are *not* exposed as a working space.",
            "",
            "Do not cross-chain LUTs between bundles — the wire constants are stock-specific and won't line up.",
            "",
            "## Working in the intermediate spaces",
            "",
            "Two intermediate taps are exposed (after L1 and after L2). To intercept, decode via the matching wire, apply the effect in physical units, then re-encode before feeding the next LUT.",
            "",
            "- **After L1 — `log_e_film`** (light hitting the film, log-shaped): decode via `wires/log_e_film` to get log10(E), then exponentiate to recover linear-light exposure. This is the right place for spatial effects that operate on the actual light landing on the film — **halation**, **light scattering** through the emulsion, and **lens diffusion** filters (Pro-Mist, Black Pro-Mist, etc.).",
            "- **After L2 — `cmy_film` density** (developed film density, reported *above base+fog*): decode via `wires/cmy_film` (`D = code * (d_max - d_min) + d_min`) to get physical D per channel. **Grain** belongs here — film grain originates in the silver / dye granularity that density represents, so density-modulated noise at this tap is the canonical film-grain injection point. `d_min` is reserved slightly negative (e.g. -0.2) so noise samples can dip below zero — real fog grain fluctuates around base+fog, including downward — without being clipped at the [0, 1] code boundary.",
            "",
            "Enlarger-stage effects (diffusion filters at the printing light, dodge / burn masks) are **not** available in this topology because the `log_e_print` tap is collapsed inside L3. Use the 4-LUT bundle for that.",
        ]
    if topology == "4lut":
        return [
            "",
            "## Apply order",
            "Apply the four LUTs in order: L1 → L2 → L3 → L4.",
            "",
            "- L1 and L2 are shared across all prints in the bundle.",
            "- L3 and L4 are print-specific; pick the pair matching the print stock you want.",
            "",
            "Wire contracts (each cube carries [0, 1] code values; the wire describes what those codes represent physically):",
            "",
            "- After L1: normalized `log_e_film` code, with shaper (min, max) in `bundle.json/wires/log_e_film`.",
            "- After L2: normalized `cmy_film` density code, with `d_min` / `d_max` in `bundle.json/wires/cmy_film` (decode: `D = code * (d_max - d_min) + d_min`).",
            "- After L3: normalized `log_e_print` code, with shaper (min, max) in `bundle.json/wires/log_e_print`.",
            "- After L4: encoded RGB in the bundle's output color space.",
            "",
            "Do not cross-chain LUTs between bundles — the wire constants are stock-specific and won't line up.",
            "",
            "## Working in the intermediate spaces",
            "",
            "Each of the three intermediate taps exposes a normalized `[0, 1]` code. To intercept, decode via the matching wire, apply the effect in physical units, then re-encode before feeding the next LUT.",
            "",
            "- **After L1 — `log_e_film`** (light hitting the film, log-shaped): decode via `wires/log_e_film` to get log10(E), then exponentiate to recover linear-light exposure. This is the right place for spatial effects that operate on the actual light landing on the film — **halation**, **light scattering** through the emulsion, and **lens diffusion** filters (Pro-Mist, Black Pro-Mist, etc.).",
            "- **After L2 — `cmy_film` density** (developed film density, reported *above base+fog*): decode via `wires/cmy_film` (`D = code * (d_max - d_min) + d_min`) to get physical D per channel. **Grain** belongs here — film grain originates in the silver / dye granularity that density represents, so density-modulated noise at this tap is the canonical film-grain injection point. `d_min` is reserved slightly negative (e.g. -0.2) so noise samples can dip below zero — real fog grain fluctuates around base+fog, including downward — without being clipped at the [0, 1] code boundary.",
            "- **After L3 — `log_e_print`** (light hitting the print, log-shaped): decode via `wires/log_e_print` to get log10(E) at the print. Enlarger-stage effects belong here — **enlarger diffusion filters** (soft-focus, baseboard scatter), simulated **dodge / burn masks**, and any other manipulation of the printing light.",
        ]
    return []


def _input_exposure_block(meta, input_cs, output_cs) -> list[str]:
    """Render the "Input exposure" disclosure section (n200)."""
    exp = meta.input_exposure
    film_mid_gray = 0.18 * exp.gain
    return [
        "",
        "## Input exposure",
        "",
        (
            f"This bundle was baked with `exposure_ev = "
            f"{exp.exposure_ev:+g}` — a deliberate linear gain of "
            f"{exp.gain:.4g} on top of the default midgray-pinned "
            f"mapping. Every input linear value gets the same "
            f"multiplier, middle gray included: a source whose "
            f"native mid-gray is 0.18 linear renders as if exposed "
            f"{abs(exp.exposure_ev):g} stop(s) "
            f"{'over' if exp.exposure_ev > 0 else 'under'} — the film "
            f"sees mid-gray at {film_mid_gray:.4g} linear instead of "
            f"the photographic 0.18 reference."
        ),
        "",
        (
            "**Disclosure.** With `exposure_ev` set, the LUT is no "
            "longer a strict colorimetric "
            f"`{input_cs.name if input_cs else 'input'} → "
            f"{output_cs.name if output_cs else 'output'}` "
            "transform — it bakes in the exposure gain so the "
            "source walks more (or less) of the film's latitude. "
            "See spektrafilm-research n200 for the rationale."
        ),
    ]


def _quality_readme_section(qa_results: dict[str, list]) -> list[str]:
    """Render the README's "## Quality" pass/fail badge block (n090 §6.1).

    Produces one row per QA test, surfacing the test's pass/fail/info
    status and its headline ``short_summary()``. For multi-print
    bundles every print gets its own subsection so the colorist can
    audit each chain independently.
    """
    badges = {"PASS": "✓ PASS", "FAIL": "✗ FAIL", "INFO": "· INFO"}
    lines = ["## Quality", ""]
    multi_print = len(qa_results) > 1
    for print_profile, results in qa_results.items():
        if multi_print:
            lines.extend([f"### {print_profile}", ""])
        n_pass = sum(1 for r in results if r.passed is True)
        n_fail = sum(1 for r in results if r.passed is False)
        n_info = sum(1 for r in results if r.passed is None)
        lines.extend(
            [
                f"{n_pass} pass · {n_fail} fail · {n_info} info "
                f"— full report at `qa/<print-folder>/report.md`.",
                "",
                "| Test | Status | Headline |",
                "|------|--------|----------|",
            ]
        )
        for r in results:
            lines.append(f"| `{r.name}` | {badges[r.status()]} | {r.short_summary()} |")
        lines.append("")
    return lines


def _combinations_readme_section(
    subchain_luts: tuple[LutFileMeta, ...],
) -> list[str]:
    """Render the README's "Pre-collapsed sub-chains" section.

    Lists every ``subchain_*`` cube in the bundle with its domain →
    range mapping and print-specificity. Generated from the bundle's
    own metadata so the README never lists a cube that wasn't baked.
    """
    lines = [
        "",
        "## Pre-collapsed sub-chains",
        "",
        "In addition to the canonical LUTs above, this bundle ships every "
        "contiguous sub-chain of the canonical chain in `combinations/`. Each "
        "combination cube collapses two or more canonical stages into a "
        "single transform, for grading apps that only have one LUT slot "
        "(Resolve LUT slot, Lumix Lab, OBS, FFmpeg, Premiere).",
        "",
        "| Cube | Maps | Print-specific |",
        "|------|------|----------------|",
    ]
    for lut in subchain_luts:
        ids = lut.role[len("subchain_") :]
        label = f"l{ids}"
        domain_label = _TAP_DOMAIN_LABEL.get(lut.domain, lut.domain)
        range_label = _TAP_RANGE_LABEL.get(lut.range, lut.range)
        print_specific = "yes" if lut.print_profile else "no"
        lines.append(f"| {label} | {domain_label} → {range_label} | {print_specific} |")
    lines.extend(
        [
            "",
            "The canonical L1..LN cubes in the bundle root remain the recommended "
            "chain — they expose every intermediate tap for grain, halation, "
            "diffusion, and enlarger-stage manipulation. Use the combinations "
            "when you want a single-cube application of a particular sub-chain. "
            "Wire contracts and decode formulas for each tap are the same as in "
            "the apply-order section above, and recorded in `bundle.json/wires`.",
        ]
    )
    return lines


# ---------------------------------------------------------------------------
# Cube file headers.
# ---------------------------------------------------------------------------


def cube_header_lines(meta: BundleMeta, rel_path: str) -> list[str]:
    """Render the bundle's provenance into ``# ``-prefixable comment lines
    suitable for the top of a ``.cube`` file.

    The lines are deliberately short and human-readable: a user opening
    the file in a text editor should immediately see what it is, how it
    was made, the license, and how to cite. Long fields wrap to multiple
    comment lines.
    """
    prov = meta.provenance
    sep = "=" * 76
    lines: list[str] = [sep, "spektrafilm LUT"]
    lines.append(f"Bundle:  {meta.name}  ({meta.topology}, {meta.resolution}^3)")
    if meta.stocks is not None:
        lines.append(f"Film:    {meta.stocks.film}")
        if meta.stocks.prints:
            lines.append(f"Print:   {', '.join(meta.stocks.prints)}")
    in_cs = meta.color_spaces.get("input")
    out_cs = meta.color_spaces.get("output")
    if in_cs is not None:
        lines.append(f"Input:   {in_cs.name}  (cctf {'on' if in_cs.cctf else 'off'})")
    if out_cs is not None:
        lines.append(f"Output:  {out_cs.name}  (cctf {'on' if out_cs.cctf else 'off'})")
    lines.append(f"Created: {prov.created}")
    lines.append(f"spektrafilm: {prov.spektrafilm_version}")
    lines.append(f"Project: {prov.project_url}")
    lines.append("")
    lines.append(prov.copyright)
    lines.append("")
    lines.extend(_wrap_field("License", prov.license))
    lines.append("")
    lines.extend(_wrap_field("Citation", prov.citation))
    lines.append("")
    lines.extend(_wrap_field("Notes", prov.notes))
    lines.append("")
    this_lut = next((lut for lut in meta.luts if lut.path == rel_path), None)
    if this_lut is not None and this_lut.role != "combined":
        lines.append(
            f"Role:    {this_lut.role}  (domain={this_lut.domain} → range={this_lut.range})"
        )
    lines.append(f"File: {rel_path}  (see sibling bundle.json for full metadata)")
    lines.append(sep)
    return lines


def _wrap_field(label: str, text: str, width: int = 76) -> list[str]:
    """Soft-wrap a labeled paragraph onto an indented multi-line comment block.

    URLs (and any other long token) are kept whole on their own line so they
    stay clickable in editors that auto-link bare URLs.
    """
    wrapped = textwrap.wrap(
        text,
        width=width - len(label) - 2,
        break_long_words=False,
        break_on_hyphens=False,
    )
    if not wrapped:
        return [f"{label}: "]
    first, *rest = wrapped
    indent = " " * (len(label) + 2)
    return [f"{label}: {first}"] + [f"{indent}{line}" for line in rest]
