"""Command-line interface for the spektrafilm LUT creator.

Two subcommands:

- ``build`` — produce one bundle. Flags cover the common
  :class:`BundleSpec` fields; ``--from spec.toml`` loads a full spec
  from disk. CLI flags override TOML values when both are present.
- ``list KIND`` — print registry contents (film, print, input, output,
  target) one name per line, suitable for shell pipelines.

Color-space arguments accept either the canonical registry name
(``"Panasonic V-Log"``) or its short-tag slug (``vlog``). Slugs are
sourced verbatim from the registry's :attr:`ColorSpaceEntry.short_tag`
field — there is no parallel slug table to maintain.

Run ``spektrafilm-lut --help`` for the user-facing reference.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import tomllib
from importlib import resources
from pathlib import Path

from spektrafilm.utils.gamut_compression import (
    InputGamutCompressSpec,
    OutputGamutCompressSpec,
)
from spektrafilm_lut_creator import color_spaces, delivery_targets
from spektrafilm_lut_creator.builders import BundleBuilder
from spektrafilm_lut_creator.bundles import BundleSpec

_LIST_KINDS = ("film", "print", "input", "output", "target")


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a shell exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "build":
        return _cmd_build(args)
    if args.command == "list":
        return _cmd_list(args.kind)
    parser.print_help(sys.stderr)
    return 2


# ---------------------------------------------------------------------------
# Argument parsing.
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spektrafilm-lut",
        description="Bake spektrafilm LUT bundles for external grading tools.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser(
        "build",
        help="Build one LUT bundle.",
        description=(
            "Build one LUT bundle. Required: --film, --print, --input, "
            "--output. Color spaces accept canonical names "
            '("Panasonic V-Log") or short-tag slugs (vlog) — see '
            "`spektrafilm-lut list input` / `list output`."
        ),
    )
    build.add_argument(
        "--from",
        dest="from_toml",
        metavar="FILE",
        type=Path,
        help=(
            "Load BundleSpec fields from a TOML file. Field names match "
            "the BundleSpec dataclass 1:1. CLI flags override TOML values."
        ),
    )
    build.add_argument("--name", help="Bundle name; auto-computed when omitted.")
    build.add_argument("--film", help="Film profile slug, e.g. kodak_portra_400.")
    build.add_argument(
        "--print",
        dest="prints",
        action="append",
        metavar="PRINT",
        help="Print profile slug. Repeat for multi-print bundles.",
    )
    build.add_argument(
        "--input",
        dest="input_cs",
        help='Input color space (name or slug), e.g. "Panasonic V-Log" or vlog.',
    )
    build.add_argument(
        "--output",
        dest="output_cs",
        help="Output color space (name or slug), e.g. sRGB or srgb.",
    )
    build.add_argument(
        "--topology",
        choices=("1lut", "2lut", "3lut", "4lut"),
        help="Bundle topology (default: 1lut).",
    )
    build.add_argument(
        "--resolution",
        type=int,
        metavar="N",
        help="Cube resolution N (default: 33). Common: 17, 33, 65.",
    )
    build.add_argument(
        "--target",
        metavar="NAME",
        help=(
            "Delivery target (e.g. lumix_realtime_vlog). When set, the "
            "bundle also writes a target-specific file and validates "
            "input/output against the target's allowed set."
        ),
    )
    build.add_argument(
        "--container",
        choices=("directory", "zip"),
        help="On-disk packaging (default: directory).",
    )
    build.add_argument(
        "--exposure-ev",
        type=float,
        metavar="EV",
        help=(
            "Deliberate exposure baked into the LUT, in stops. "
            "Default 0: midgray-pinned colorimetric behavior."
        ),
    )
    build.add_argument(
        "--qa",
        action="store_true",
        help="Run the QA suite after writing the bundle.",
    )
    build.add_argument(
        "--qa-print-index",
        type=int,
        metavar="I",
        help="Run QA only for print I (default: all prints in the bundle).",
    )
    build.add_argument(
        "--ocio-config",
        action="store_true",
        help="Emit a config.ocio alongside the .cube files.",
    )
    build.add_argument(
        "--combinations",
        action="store_true",
        help=(
            "For multi-LUT bundles, also ship every contiguous sub-chain "
            "as a pre-collapsed single cube in combinations/."
        ),
    )
    build.add_argument(
        "--out",
        type=Path,
        required=True,
        metavar="DIR",
        help=("Output directory. The bundle is written inside DIR/<bundle-name>/."),
    )

    listing = subparsers.add_parser(
        "list",
        help="List registry contents.",
        description="Print one name per line, sorted; suitable for shell pipelines.",
    )
    listing.add_argument(
        "kind",
        choices=_LIST_KINDS,
        metavar="KIND",
        help=f"What to list. One of: {', '.join(_LIST_KINDS)}.",
    )

    return parser


# ---------------------------------------------------------------------------
# `build` subcommand.
# ---------------------------------------------------------------------------


def _cmd_build(args: argparse.Namespace) -> int:
    fields = _load_toml_fields(args.from_toml) if args.from_toml else {}
    _merge_cli_overrides(fields, args)

    try:
        spec = _make_spec(fields)
    except (KeyError, ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    builder = BundleBuilder(spec)
    bundle = builder.build()
    out_path = builder.write(bundle, Path(args.out) / spec.name)
    print(f"[done] {out_path}")
    return 0


def _make_spec(fields: dict) -> BundleSpec:
    """Build a :class:`BundleSpec` from a flat dict, resolving slugs and
    nested specs (gamut compression).

    Required fields are validated by the dataclass itself; we only handle
    the type-conversion concerns the CLI introduces (slug → canonical
    name, TOML tables → dataclass instances).
    """
    if "film_profile" not in fields:
        raise ValueError("missing --film (or `film_profile` in TOML).")
    if "print_profiles" not in fields or not fields["print_profiles"]:
        raise ValueError("missing --print (or `print_profiles` in TOML).")
    if "input_color_space" not in fields:
        raise ValueError("missing --input (or `input_color_space` in TOML).")
    if "output_color_space" not in fields:
        raise ValueError("missing --output (or `output_color_space` in TOML).")

    fields["print_profiles"] = tuple(fields["print_profiles"])
    fields["input_color_space"] = resolve_color_space(
        fields["input_color_space"],
        role="input",
    )
    fields["output_color_space"] = resolve_color_space(
        fields["output_color_space"],
        role="output",
    )

    if "input_gamut_compress" in fields and isinstance(
        fields["input_gamut_compress"], dict
    ):
        fields["input_gamut_compress"] = InputGamutCompressSpec(
            **fields["input_gamut_compress"]
        )
    if "output_gamut_compress" in fields and isinstance(
        fields["output_gamut_compress"], dict
    ):
        fields["output_gamut_compress"] = OutputGamutCompressSpec(
            **fields["output_gamut_compress"]
        )

    valid = {f.name for f in dataclasses.fields(BundleSpec)}
    unknown = set(fields) - valid
    if unknown:
        raise ValueError(
            f"unknown BundleSpec field(s): {sorted(unknown)}. Valid: {sorted(valid)}."
        )
    return BundleSpec(**fields)


def _load_toml_fields(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"error: --from {path} does not exist or is not a file.")
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def _merge_cli_overrides(fields: dict, args: argparse.Namespace) -> None:
    """Overlay non-None CLI values onto ``fields`` in place.

    CLI wins over TOML so a base spec can be parameterized at the
    command line (``--from base.toml --resolution 65``).
    """
    cli_to_field = {
        "name": "name",
        "film": "film_profile",
        "prints": "print_profiles",
        "input_cs": "input_color_space",
        "output_cs": "output_color_space",
        "topology": "topology",
        "resolution": "resolution",
        "target": "target",
        "container": "container",
        "exposure_ev": "exposure_ev",
        "qa_print_index": "qa_print_index",
    }
    for cli_attr, field_name in cli_to_field.items():
        value = getattr(args, cli_attr, None)
        if value is not None:
            fields[field_name] = value
    # Boolean store_true flags: presence ≡ True. They can't override a
    # TOML `true` back to False, but no current use case needs that.
    for cli_attr, field_name in (
        ("qa", "qa"),
        ("ocio_config", "ocio_config"),
        ("combinations", "include_combinations"),
    ):
        if getattr(args, cli_attr, False):
            fields[field_name] = True


def resolve_color_space(value: str, *, role: str) -> str:
    """Resolve a CLI argument to a canonical registry name.

    Accepts either the canonical name or the entry's ``short_tag``
    slug. Errors include the role and a hint to the ``list`` subcommand.
    """
    try:
        return color_spaces.resolve(value)
    except KeyError:
        raise ValueError(
            f"unknown {role} color space {value!r}. Try `spektrafilm-lut list {role}`."
        ) from None


# ---------------------------------------------------------------------------
# `list` subcommand.
# ---------------------------------------------------------------------------


def _cmd_list(kind: str) -> int:
    if kind == "film":
        names = _list_profiles_by_stage("filming")
        for name in names:
            print(name)
        return 0
    if kind == "print":
        names = _list_profiles_by_stage("printing")
        for name in names:
            print(name)
        return 0
    if kind == "target":
        names = delivery_targets.list_targets()
        for name in names:
            print(name)
        return 0
    if kind in ("input", "output"):
        # Color spaces print as ``<canonical>  <slug>`` so both forms
        # — the canonical registry name (``"Panasonic V-Log"``) and
        # the short-tag slug (``vlog``) — accepted by ``--input`` /
        # ``--output`` are visible at a glance. Padded to a column so
        # the slugs line up; the slug is still the last whitespace-
        # delimited token, so pipelines that want it can read it with
        # ``awk '{print $NF}'``.
        names = (
            color_spaces.list_input_spaces()
            if kind == "input"
            else color_spaces.list_output_spaces()
        )
        width = max((len(n) for n in names), default=0)
        for name in names:
            slug = color_spaces.get(name).short_tag
            print(f"{name:<{width}}  {slug}")
        return 0
    # argparse enforces choices, this branch is unreachable.
    return 2


def _list_profiles_by_stage(stage: str) -> list[str]:
    """Walk the ``spektrafilm.data.profiles`` package and return the
    stocks whose ``info.stage`` matches ``stage`` (``"filming"`` or
    ``"printing"``), sorted alphabetically.

    We inspect the JSON directly rather than calling :func:`load_profile`
    so the list command stays fast even with many profiles installed.
    """
    package = resources.files("spektrafilm.data.profiles")
    names: list[str] = []
    for entry in package.iterdir():
        if not entry.name.endswith(".json"):
            continue
        with entry.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        info = data.get("info", {})
        if info.get("stage") == stage:
            names.append(info.get("stock", entry.stem))
    return sorted(names)


if __name__ == "__main__":
    sys.exit(main())
