"""Filename + bundle-name normalization helpers.

Centralized so :class:`spektrafilm_lut_creator.bundles.BundleSpec` can
auto-compute a canonical default ``name`` in its ``__post_init__``
without importing ``builders`` (which would create a cycle —
``builders`` already imports ``bundles``).

The canonical bundle name shape is::

    spektrafilm_<version>_<film>[_<print>]_<topology>_<input>_<output>

For single-print bundles ``<print>`` is the normalized print stock
tag. For multi-print bundles it becomes ``<N>printpack`` (e.g.
``3printpack``) — the count communicates the pack's scope without
falsely naming after one print. Every tag is normalized:

- ``<version>``: ``0.3.2`` → ``v032``
- ``<film>`` / ``<print>``: brand prefix stripped, first two
  underscore-separated tokens fused (``kodak_portra_400`` →
  ``portra400``; ``fujifilm_crystal_archive_typeii`` →
  ``crystalarchive``)
- ``<topology>``: passes through (``1lut`` / ``2lut`` / ``4lut``)
- ``<input>`` / ``<output>``: the color-space registry's ``short_tag``
  (``Panasonic V-Log`` → ``vlog``; ``Rec.2020`` → ``rec2020``)
"""

from __future__ import annotations

from spektrafilm_lut_creator.metadata import _spektrafilm_version

# Brand prefixes stripped from profile stock names when building canonical
# camera-safe filename tags. Order matters when prefixes share a stem
# (none here yet). Extend as new vendors arrive.
_BRAND_PREFIXES: tuple[str, ...] = (
    "kodak",
    "fujifilm",
    "fuji",
    "ilford",
    "agfa",
    "cinestill",
    "polaroid",
    "ferrania",
    "lomography",
)

_TOPOLOGY_TAGS: dict[str, str] = {
    "1lut": "1lut",
    "2lut": "2lut",
    "3lut": "3lut",
    "4lut": "4lut",
}


def normalize_stock(stock: str) -> str:
    """``kodak_portra_400`` → ``portra400``; ``fujifilm_c200`` → ``c200``;
    ``fujifilm_crystal_archive_typeii`` → ``crystalarchive``.

    Strips the brand prefix (if recognized) and fuses the first two
    remaining underscore-separated segments without delimiter.
    """
    parts = stock.lower().split("_")
    if parts and parts[0] in _BRAND_PREFIXES:
        parts = parts[1:]
    return "".join(parts[:2])


def normalize_version(version: str) -> str:
    """``0.3.2`` → ``v032``; ``0.4.1.dev0+abc`` → ``v041``.

    Strips PEP 440 dev / local-version suffixes, drops dots, prepends ``v``.
    """
    base = version.split("+", 1)[0].split(".dev", 1)[0]
    digit_groups = [g for g in base.split(".") if g.isdigit()]
    return "v" + "".join(digit_groups)


def topology_tag(topology: str) -> str:
    """Short topology identifier for filenames/bundle names.

    Today this is an identity for the canonical short forms (``1lut`` /
    ``2lut`` / ``4lut``). Kept as a thin indirection so future variants
    (e.g. ``4lut-special_label``) can collapse to a family tag here
    without touching call sites.
    """
    return _TOPOLOGY_TAGS.get(topology, topology)


def default_bundle_name(
    film_profile: str,
    print_profiles: tuple[str, ...],
    topology: str,
    input_color_space: str,
    output_color_space: str,
) -> str:
    """Compute the canonical default bundle name from a spec's components.

    The print slot carries the normalized print tag for single-print
    bundles and ``<N>printpack`` (e.g. ``3printpack``) for multi-print
    bundles — keeps the count discoverable from the filename without
    misleadingly naming the pack after one of its prints.
    """
    # Lazy import to avoid a cycle with the registry (color_spaces
    # imports nothing from this package's core layout but its registry
    # is populated at module-load time so the import is light).
    from spektrafilm_lut_creator.color_spaces import short_tag as _cs_short_tag

    v_tag = normalize_version(_spektrafilm_version())
    film_tag = normalize_stock(film_profile)
    topo_tag = topology_tag(topology)
    in_tag = _cs_short_tag(input_color_space)
    out_tag = _cs_short_tag(output_color_space)

    parts = ["spektrafilm", v_tag, film_tag]
    if len(print_profiles) == 1:
        parts.append(normalize_stock(print_profiles[0]))
    else:
        parts.append(f"{len(print_profiles)}printpack")
    parts.extend([topo_tag, in_tag, out_tag])
    return "_".join(parts)


def lut_filename(
    *,
    film_profile: str,
    version_tag: str,
    print_profile: str | None = None,
    suffix: str | None = None,
    subdir: str | None = None,
    ext: str = ".cube",
) -> str:
    """Build the canonical on-disk filename for one cube in a bundle.

    Pattern: ``lut_{version_tag}_{film}[_{print}][_{suffix}]{ext}``,
    optionally prefixed with ``<subdir>/``. Every canonical cube the
    builder emits comes from this one helper:

    - **1-LUT combined**: ``film_profile + print_profile``, no suffix
      → ``lut_v032_portra400_endura.cube``.
    - **2-LUT film half**: ``film_profile`` only, ``suffix="film"``
      → ``lut_v032_portra400_film.cube``.
    - **2-LUT print half**: ``film + print``, ``suffix="print"``
      → ``lut_v032_portra400_endura_print.cube``.
    - **4-LUT L1 / L2** (shared): ``film_profile`` only, ``suffix="l1"``
      or ``"l2"``.
    - **4-LUT L3 / L4** (per-print): ``film + print``, ``suffix="l3"``
      or ``"l4"``.
    - **3-LUT L3 (collapsed)**: ``film + print``, ``suffix="l3"``.
    - **Sub-chain combinations** (n130): ``film [+ print]``,
      ``suffix="l12"``/``"l1234"``/…, ``subdir="combinations"``.

    Stocks are normalized via :func:`normalize_stock`; the version tag
    is expected pre-normalized by :func:`normalize_version`.
    """
    parts = ["lut", version_tag, normalize_stock(film_profile)]
    if print_profile is not None:
        parts.append(normalize_stock(print_profile))
    if suffix:
        parts.append(suffix)
    name = "_".join(parts) + ext
    return f"{subdir}/{name}" if subdir else name


def lut_title(
    *,
    film_profile: str,
    version_tag: str,
    print_profile: str | None = None,
    suffix: str | None = None,
) -> str:
    """Compact title for a baked cube — same shape as
    :func:`lut_filename` minus the ``lut_`` prefix and extension.

    Used as the ``TITLE "..."`` line inside ``.cube`` files. Mirrors
    the filename pattern so a user reading a cube's title can recognize
    the file it came from.
    """
    parts = [version_tag, normalize_stock(film_profile)]
    if print_profile is not None:
        parts.append(normalize_stock(print_profile))
    if suffix:
        parts.append(suffix)
    return "_".join(parts)


def per_print_qa_folder_name(
    film_profile: str,
    print_profile: str,
) -> str:
    """Folder name for one print's QA report inside ``<bundle>/qa/``.

    Shape: ``<film>_<print>`` (e.g. ``portra160_portraendura``). The
    parent bundle directory already carries the spektrafilm version,
    topology, and color-space pair; the QA folder only needs to
    disambiguate by what changes *within* a bundle — the print.
    """
    film_tag = normalize_stock(film_profile)
    print_tag = normalize_stock(print_profile)
    return f"{film_tag}_{print_tag}"
