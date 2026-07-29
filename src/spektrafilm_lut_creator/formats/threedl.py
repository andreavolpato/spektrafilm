"""Autodesk / Discreet ``.3dl`` 3D LUT writer.

Format reference: the Lustre / Autodesk ``.3dl`` text format used by
Smoke, Flame, and other Discreet-lineage tools. The structure is::

    # Optional comment lines starting with '#'
    <shape line: N ascending integer code values, space-separated>
    <N^3 lines of 'R G B' integer triplets in [0, 2^bit_depth - 1]>

The shape line declares both the cube resolution (its length) and the
input bit-depth scale (its max value). The data lines hold the encoded
output triplets at each grid point. Ordering matches Adobe ``.cube``:
``R`` varies fastest, then ``G``, then ``B``.

This plugin emits the 10-bit Autodesk dialect: the shape line spans
``[0, 1023]`` in evenly-spaced steps, and output values are scaled
into ``[0, 1023]``. 10-bit is the dialect Flame / Smoke have shipped
with for decades; 12- and 16-bit dialects exist but are workflow-
specific and not required for v1.

See [n090 §4.1](../../../../spektrafilm-research/studies/a40_lut_system/n090_industry_grade_bundles.md)
for the multi-format roadmap.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from spektrafilm_lut_creator.formats import Lut, register

_BIT_DEPTH = 10
_MAX_CODE = (1 << _BIT_DEPTH) - 1  # 1023


class ThreeDLFormat:
    """Autodesk ``.3dl`` 10-bit writer (registers as ``"3dl"``)."""

    name = "3dl"
    extensions = (".3dl",)

    def write(
        self,
        lut: Lut,
        path: Path,
        *,
        header_lines: list[str] | None = None,
    ) -> None:
        n = lut.resolution
        flat = np.asarray(lut.table, dtype=float).reshape(n**3, 3)

        lines: list[str] = []
        if header_lines:
            for raw in header_lines:
                lines.append(f"# {raw}" if raw else "#")

        # Shape line: N values evenly spaced over [0, 1023].
        shape = np.round(np.linspace(0, _MAX_CODE, n)).astype(int)
        lines.append(" ".join(str(v) for v in shape))

        # Data: N^3 integer triplets in [0, 1023], R-fastest then G then B
        # (the same ordering convention as Adobe .cube).
        scaled = np.clip(
            np.round(flat * _MAX_CODE).astype(int),
            0,
            _MAX_CODE,
        )
        for r, g, b in scaled:
            lines.append(f"{r} {g} {b}")

        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")

    def read(self, path: Path) -> Lut:
        shape_values: list[int] | None = None
        triplets: list[tuple[int, int, int]] = []

        for raw in Path(path).read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if shape_values is None:
                # First non-comment line is the shape declaration.
                shape_values = [int(v) for v in parts]
                continue
            if len(parts) != 3:
                raise ValueError(
                    f"{path}: expected 3 integers per data line, got {parts!r}"
                )
            triplets.append((int(parts[0]), int(parts[1]), int(parts[2])))

        if shape_values is None:
            raise ValueError(f"{path}: missing shape line")
        n = len(shape_values)
        if len(triplets) != n**3:
            raise ValueError(
                f"{path}: body has {len(triplets)} entries, expected size**3 = {n**3}"
            )
        max_code = max(shape_values[-1], 1)
        table = np.asarray(triplets, dtype=float).reshape(n, n, n, 3) / float(max_code)
        return Lut(table=table, domain_min=(0.0, 0.0, 0.0), domain_max=(1.0, 1.0, 1.0))


register(ThreeDLFormat())
