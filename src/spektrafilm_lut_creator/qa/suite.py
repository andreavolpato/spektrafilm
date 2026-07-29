"""Suite runner + markdown report emission.

``run(spec, bundle, out_dir)`` is the one-call entry point used by
``explore_lut.py`` and any other caller. It assembles the
:class:`QAContext`, executes the default test list, writes
``report.md`` and the figures, and returns the list of
:class:`Result` objects.

The report is markdown-only — renders in VS Code, GitHub, any reader.
HTML / PDF can be produced downstream from the same content.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from spektrafilm_lut_creator import color_spaces
from spektrafilm_lut_creator.bundles import Bundle, BundleSpec
from spektrafilm_lut_creator.formats import Lut
from spektrafilm_lut_creator.grid import cube_grid
from spektrafilm_lut_creator.qa import evaluators, reference
from spektrafilm_lut_creator.qa.result import Result
from spektrafilm_lut_creator.qa.tests import DEFAULT_TESTS

# Public re-export so callers can compose their own suites.
DEFAULT_SUITE: tuple[Callable[[QAContext], Result], ...] = DEFAULT_TESTS


@dataclass
class QAContext:
    """Per-LUT context handed to every test function.

    Held read-only by tests; suite-level objects don't mutate it.

    Attributes
    ----------
    spec
        The :class:`BundleSpec` used to build the bundle. Tests use it
        to re-invoke the pipeline for ad-hoc patterns.
    bundle
        The built :class:`Bundle`. Carries metadata + the LUT list.
    print_index
        Which LUT in ``bundle.luts`` this context is for.
    print_name
        Convenience: the print stock name (e.g.,
        ``"kodak_portra_endura"``).
    lut
        The :class:`Lut` being QA'd.
    grid_input
        Shape ``(N³, 3)`` of on-grid input samples in the encoded
        input space. Constant for a given resolution.
    grid_output
        Shape ``(N³, 3)`` of on-grid output samples — the cube table
        flattened in Adobe canonical order.
    reference
        :class:`ReferenceSamples` holding the off-grid pipeline ground
        truth used by ``off_grid_identity``. Computed once per
        ``run()`` call and cached on disk.
    frame
        :class:`color_spaces.BakeFrame` derived from the spec — gain
        context bound to encode-input / decode-output convenience
        methods. Tests use ``ctx.frame.encode_input(...)`` /
        ``ctx.frame.decode_output_to_xyz(...)`` instead of threading
        ``exposure_ev`` through each helper.
    out_dir
        Root output directory for this QA pass.
    figures_dir
        ``out_dir / "figures"`` — where tests save ONGs.
    """

    spec: BundleSpec
    bundle: Bundle
    print_index: int
    print_name: str
    lut: Lut
    grid_input: np.ndarray
    grid_output: np.ndarray
    reference: reference.ReferenceSamples
    frame: color_spaces.BakeFrame
    out_dir: Path
    figures_dir: Path


def list_tests() -> list[str]:
    """Names of the tests in the default suite, in execution order."""
    return [fn.__name__ for fn in DEFAULT_SUITE]


def _print_name(bundle: Bundle, print_index: int) -> str:
    """Resolve the human-readable print name for the QA print index.

    The ``print_index`` always means "which print", indexed into
    ``bundle.meta.stocks.prints`` — the *bundle.luts* indexing is
    topology-dependent (1-LUT: 1:1; 2-LUT: film at 0 then prints).
    """
    if bundle.meta.stocks is not None:
        prints = bundle.meta.stocks.prints
        if print_index < 0 or print_index >= len(prints):
            raise IndexError(
                f"print_index {print_index} out of range for bundle with "
                f"{len(prints)} prints"
            )
        return prints[print_index]
    # Fallback: no stocks recorded; assume 1-LUT and use the LUT path.
    return bundle.luts[print_index][0]


def _effective_lut(bundle: Bundle, print_index: int) -> tuple[str, Lut]:
    """Return ``(label, Lut)`` for the LUT to QA at ``print_index``.

    Composes the canonical L1..LN chain via trilinear interpolation
    (the host-default mode), matching what a user actually deploys
    when they apply the bundle's cubes in order. Returns a virtual
    combined :class:`Lut` keyed on the final-stage canonical cube's
    filename.

    Cubes are looked up by :attr:`LutFileMeta.role` rather than by
    index into ``bundle.luts`` — bundles with
    :attr:`BundleSpec.include_combinations` interleave extra cubes
    into that list, so positional indexing would pick the wrong cube.
    """
    topology = bundle.meta.topology
    print_name = _print_name(bundle, print_index)
    by_path: dict[str, Lut] = {p: l for p, l in bundle.luts}

    def find(role: str, print_profile: str | None = None) -> tuple[str, Lut]:
        for entry in bundle.meta.luts:
            if entry.role == role and entry.print_profile == print_profile:
                return entry.path, by_path[entry.path]
        raise LookupError(
            f"no canonical LUT in bundle with role={role!r}, print_profile={print_profile!r}"
        )

    if topology == "1lut":
        path, lut = find("combined", print_name)
        return path, lut
    if topology == "2lut":
        _, film = find("film")
        path, print_lut = find("print", print_name)
        composed = _compose_film_print(film, print_lut, bundle.meta.resolution)
        return path, composed
    if topology == "3lut":
        _, l1 = find("filming_expose")
        _, l2 = find("filming_develop")
        path, l3 = find("printing_combined", print_name)
        composed = _compose_3lut(l1, l2, l3, bundle.meta.resolution)
        return path, composed
    if topology == "4lut":
        _, l1 = find("filming_expose")
        _, l2 = find("filming_develop")
        _, l3 = find("printing_expose", print_name)
        path, l4 = find("printing_develop_scan", print_name)
        composed = _compose_4lut(l1, l2, l3, l4, bundle.meta.resolution)
        return path, composed
    raise NotImplementedError(f"QA does not yet handle topology={topology!r}")


def _compose_film_print(film_lut: Lut, print_lut: Lut, resolution: int) -> Lut:
    """Sample film_lut ∘ print_lut at a ``resolution^3`` cube grid.

    The composition is evaluated with trilinear interpolation (the
    host-default mode). The resulting :class:`Lut` table is what a
    user gets by applying the two cubes in order — encoded input RGB
    → cmy_film code → encoded output RGB — and is suitable for the
    standard QA harness without further awareness of the 2-LUT
    structure.
    """
    grid = cube_grid(resolution)
    cmy_codes = evaluators.apply_trilinear(film_lut.table, grid)
    rgb_encoded = evaluators.apply_trilinear(print_lut.table, cmy_codes)
    table = rgb_encoded.reshape(resolution, resolution, resolution, 3)
    title = (
        f"{film_lut.title} + {print_lut.title}"
        if film_lut.title and print_lut.title
        else "2-lut chain"
    )
    return Lut(table=table, title=title)


def _compose_3lut(l1: Lut, l2: Lut, l3: Lut, resolution: int) -> Lut:
    """Sample L1∘L2∘L3 at a ``resolution^3`` cube grid.

    Three trilinear hops total: encoded input RGB → log_e_film code →
    cmy_film code → encoded output RGB. One interpolation hop fewer than
    4-LUT, so off-grid error is between 2-LUT and 4-LUT.
    """
    grid = cube_grid(resolution)
    log_e_film_code = evaluators.apply_trilinear(l1.table, grid)
    cmy_film_code = evaluators.apply_trilinear(l2.table, log_e_film_code)
    rgb_encoded = evaluators.apply_trilinear(l3.table, cmy_film_code)
    table = rgb_encoded.reshape(resolution, resolution, resolution, 3)
    title = (
        f"{l1.title} + {l2.title} + {l3.title}"
        if all(lut.title for lut in (l1, l2, l3))
        else "3-lut chain"
    )
    return Lut(table=table, title=title)


def _compose_4lut(l1: Lut, l2: Lut, l3: Lut, l4: Lut, resolution: int) -> Lut:
    """Sample L1∘L2∘L3∘L4 at a ``resolution^3`` cube grid.

    Each stage is evaluated with trilinear interpolation (4 hops of
    interpolation total). The output is what a user gets by applying
    the four cubes in order: encoded input RGB → log_e_film code →
    cmy_film code → log_e_print code → encoded output RGB.

    Note that 4-stage trilinear chains accumulate more interpolation
    error than 1- or 2-LUT bundles at the same per-cube resolution,
    so the QA's ``off_grid_identity`` test typically reports higher
    ΔE for 4-LUT — that's the cost of the modular structure, not a
    bug in the bake.
    """
    grid = cube_grid(resolution)
    log_e_film_code = evaluators.apply_trilinear(l1.table, grid)
    cmy_film_code = evaluators.apply_trilinear(l2.table, log_e_film_code)
    log_e_print_code = evaluators.apply_trilinear(l3.table, cmy_film_code)
    rgb_encoded = evaluators.apply_trilinear(l4.table, log_e_print_code)
    table = rgb_encoded.reshape(resolution, resolution, resolution, 3)
    parts = [lut.title for lut in (l1, l2, l3, l4) if lut.title]
    title = " + ".join(parts) if parts else "4-lut chain"
    return Lut(table=table, title=title)


def run(
    spec: BundleSpec,
    bundle: Bundle,
    out_dir: Path | str,
    *,
    suite: Sequence[Callable[[QAContext], Result]] | None = None,
    print_index: int = 0,
) -> list[Result]:
    """Run the QA suite against one print's LUT chain in the bundle.

    Parameters
    ----------
    spec, bundle
        Both required: ``spec`` is the source-of-truth for pipeline
        invocation, ``bundle`` carries the cube and metadata.
    out_dir
        Directory to write ``report.md``, ``figures/*.png``, and
        ``cache/*.npz`` into. Created if missing.
    suite
        Optional alternative test list. Each item is a function
        taking a :class:`QAContext` and returning a :class:`Result`.
    print_index
        Which print to QA. For ``1lut`` bundles this
        indexes directly into ``bundle.luts``; for ``2lut``
        the print_index selects which print's chain (shared film +
        that print's LUT) to QA.

    Returns
    -------
    list[Result]
        The full result set for the run, in suite order.
    """
    suite = tuple(suite) if suite is not None else DEFAULT_SUITE
    spec_obj = spec
    bundle_obj = bundle
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not bundle_obj.luts:
        raise ValueError("bundle has no LUTs to QA")

    # Materialize the effective LUT for QA. For 2-LUT bundles, this is
    # the chain (film → print) sampled at the bundle's cube resolution
    # — what users will actually deploy. For 1-LUT bundles, it's just
    # the print's combined LUT.
    rel_path, lut = _effective_lut(bundle_obj, print_index)
    print_name = _print_name(bundle_obj, print_index)

    n = lut.resolution
    grid_input = cube_grid(n)
    grid_output = lut.table.reshape(n**3, 3)

    print(f"[qa] computing reference samples for print {print_index} ({print_name})...")
    ref = reference.compute_or_load(spec_obj, bundle_obj, print_index, cache_dir)
    print(
        f"[qa]   cache key={ref.cache_key}  samples={ref.rng_samples_encoded.shape[0]}"
    )

    ctx = QAContext(
        spec=spec_obj,
        bundle=bundle_obj,
        print_index=print_index,
        print_name=print_name,
        lut=lut,
        grid_input=grid_input,
        grid_output=grid_output,
        reference=ref,
        frame=spec_obj.bake_frame(),
        out_dir=out_dir,
        figures_dir=figures_dir,
    )

    results: list[Result] = []
    for fn in suite:
        print(f"[qa] running {fn.__name__}...")
        try:
            result = fn(ctx)
        except Exception as exc:  # noqa: BLE001 — one failing test should not abort the suite
            result = Result(
                name=fn.__name__,
                summary={"error": str(exc)},
                interpretation=f"Test raised {type(exc).__name__}: {exc}",
                passed=False,
            )
            print(f"[qa]   FAILED: {exc}")
        results.append(result)

    report_path = out_dir / "report.md"
    write_report(results, ctx, report_path)
    # Render the same content as a self-contained HTML page next to
    # report.md. Cheap (~100ms) and the colorist can double-click it.
    from spektrafilm_lut_creator.qa.html_export import report_md_to_html

    html_path = report_md_to_html(report_path)

    n_pass = sum(1 for r in results if r.passed is True)
    n_fail = sum(1 for r in results if r.passed is False)
    n_info = sum(1 for r in results if r.passed is None)
    print(f"[qa] summary: {n_pass} pass, {n_fail} fail, {n_info} info")
    if n_fail:
        for r in results:
            if r.passed is False:
                print(f"[qa]   FAIL  {r.name}  {r.short_summary()}")
    print(f"[qa] wrote {report_path}, {html_path.name}")
    return results


# ---------------------------------------------------------------------------
# Markdown report emission.
# ---------------------------------------------------------------------------


def write_report(results: list[Result], ctx: QAContext, path: Path) -> None:
    """Write ``report.md`` summarizing the QA run.

    The layout is the same every time:

    1. Run header (bundle, print, color spaces, resolution).
    2. Summary table — one row per test, headline number(s).
    3. Per-test sections: heading, units, summary table, figure,
       reference-values list, interpretation paragraph.

    Pass/fail is intentionally **not** rendered in the report — the
    reader judges the numbers themselves against ``reference_values``
    where the test exposes them. Pass/fail still drives the console
    log and any CI integrations via :meth:`Result.passed`. Literature
    citations and prior art live in each test function's docstring,
    not in the report — the bundle output is for the bundle's numbers.

    Markdown renders cleanly in VS Code, GitHub, and any reader. PDF
    export is a downstream concern.
    """
    bundle = ctx.bundle
    spec = ctx.spec
    lines: list[str] = []

    lines.append(f"# QA report — `{bundle.meta.name}`")
    lines.append("")
    lines.append(f"- **Print**: `{ctx.print_name}`")
    lines.append(f"- **Film**: `{spec.film_profile}`")
    lines.append(f"- **Input color space**: `{spec.input_color_space}`")
    lines.append(f"- **Output color space**: `{spec.output_color_space}`")
    lines.append(
        f"- **Topology**: `{bundle.meta.topology}`  ·  **Resolution**: `{spec.resolution}^3`"
    )
    lines.append(
        f"- **spektrafilm version**: `{bundle.meta.provenance.spektrafilm_version}`"
    )
    lines.append(f"- **Generated**: `{bundle.meta.provenance.created}`")
    lines.append("")

    # Summary table.
    lines.append("## Summary")
    lines.append("")
    lines.append("| Test | Headline numbers |")
    lines.append("|---|---|")
    for r in results:
        summary = r.short_summary()
        # Escape pipes inside the summary cell.
        summary = summary.replace("|", "\\|")
        lines.append(f"| [{r.name}](#{_anchor(r.name)}) | {summary} |")
    lines.append("")

    # Per-test sections.
    for r in results:
        lines.append(f"## {r.name}")
        lines.append("")
        if r.units:
            lines.append(f"**Units**: {r.units}")
            lines.append("")

        # Summary key/value table.
        if r.summary:
            lines.append("| Metric | Value |")
            lines.append("|---|---|")
            for key, val in r.summary.items():
                rendered = _render_value(val)
                lines.append(f"| `{key}` | {rendered} |")
            lines.append("")

        if r.figure_path is not None:
            rel = _relative(r.figure_path, path.parent)
            lines.append(f"![{r.name}]({rel})")
            lines.append("")

        if r.reference_values:
            lines.append("**Reference values:**")
            for key, ref_text in r.reference_values.items():
                lines.append(f"- `{key}` {ref_text}")
            lines.append("")

        if r.interpretation:
            lines.append(r.interpretation)
            lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        f"Generated by `spektrafilm_lut_creator.qa` "
        f"(spektrafilm {bundle.meta.provenance.spektrafilm_version}). "
        f"See `studies/a40_lut_system/n080_lut_quality_and_visualization.md` "
        f"for design context."
    )
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def _render_value(val) -> str:
    """Render a summary value for the markdown table."""
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, (int, np.integer)):
        return str(int(val))
    if isinstance(val, float):
        return f"{val:.4g}"
    return f"`{val}`"


def _anchor(name: str) -> str:
    """GitHub-flavored markdown anchor for a heading."""
    return name.replace("_", "-").lower()


def _relative(target: Path, base: Path) -> str:
    """Path of ``target`` relative to ``base`` as a forward-slash string."""
    target = Path(target).resolve()
    base = Path(base).resolve()
    try:
        return target.relative_to(base).as_posix()
    except ValueError:
        # Targets outside base — fall back to absolute path string.
        return target.as_posix()
