"""Panasonic Lumix ``.cube`` variant.

The Real-Time LUT feature on Lumix bodies (S5II, S5IIX, S9, S1II/IIE,
GH7, and similar) accepts ``.cube`` files with a vendor-specific
``#LUMIXPHOTOSTYLE`` comment line that declares the LUT's expected input
encoding. The layout below is the format verified to load on these
bodies (mirroring a working reference script supplied by the user)::

    TITLE "..."
    #LUMIXPHOTOSTYLE VLOG
    LUT_3D_SIZE <n>
    DOMAIN_MIN 0 0 0
    DOMAIN_MAX 1 1 1
    <blank line>
    <data: r g b>

It differs from Adobe's spec wording in two practical ways: the
``#LUMIXPHOTOSTYLE`` tag is required right after ``TITLE``, and
``LUT_3D_SIZE`` precedes the ``DOMAIN_*`` lines.

Provenance comments are intentionally **not** emitted in this format.
Lumix's parser is verified to tolerate the photo-style tag, but we have
no field evidence it tolerates additional comment lines — keep the
header minimal for maximum compatibility. The full provenance still
lives in the sibling ``bundle.json``.

See [n080](../../../../spektrafilm-research/studies/a40_lut_system/n080_lut_quality_and_visualization.md)
for the broader LUT-export plan.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from spektrafilm_lut_creator.formats import Lut, register

# Registry color-space name → Lumix photo-style tag. Add new entries
# only after a real camera has accepted them; the inventory is
# intentionally minimal and grown by field-verification.
LUMIX_PHOTOSTYLE_BY_INPUT: Mapping[str, str] = {
    "Panasonic V-Log": "VLOG",
}


# Fixed-decimal data formatting (Lumix-reference style). 6 places is
# below in-camera display precision and matches the working script.
_VALUE_FORMAT = "{:.6f}"


class LumixCubeFormat:
    """Lumix-compatible ``.cube`` writer (registers as ``"lumix"``)."""

    name = "lumix"
    extensions = (".cube",)

    def write(
        self,
        lut: Lut,
        path: Path,
        *,
        photo_style_tag: str | None = None,
        header_lines: list[str] | None = None,  # ignored; kept for protocol parity
    ) -> None:
        del header_lines  # Lumix-strict mode: no extra comments
        n = lut.resolution
        flat = np.asarray(lut.table, dtype=float).reshape(n**3, 3)
        lines: list[str] = []
        if lut.title:
            lines.append(f'TITLE "{lut.title}"')
        if photo_style_tag:
            lines.append(f"#LUMIXPHOTOSTYLE {photo_style_tag}")
        lines.append(f"LUT_3D_SIZE {n}")
        lines.append(
            "DOMAIN_MIN " + " ".join(_VALUE_FORMAT.format(v) for v in lut.domain_min)
        )
        lines.append(
            "DOMAIN_MAX " + " ".join(_VALUE_FORMAT.format(v) for v in lut.domain_max)
        )
        lines.append("")  # blank line before data (matches working reference)
        for r, g, b in flat:
            lines.append(" ".join(_VALUE_FORMAT.format(v) for v in (r, g, b)))
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")

    def read(self, path: Path) -> Lut:
        # Lumix files are valid Adobe-spec cubes (the photo-style tag is
        # a comment to the spec). Delegate to the base reader.
        from spektrafilm_lut_creator.formats.cube import CubeFormat

        return CubeFormat().read(path)


register(LumixCubeFormat())
