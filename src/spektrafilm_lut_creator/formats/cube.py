"""Adobe .cube 3D LUT reader / writer.

Wire-level reference: the "Cube LUT Specification 1.0" (Adobe). Header
fields supported: ``TITLE``, ``DOMAIN_MIN``, ``DOMAIN_MAX``,
``LUT_3D_SIZE``. Body is one ``r g b`` triplet per line, with R varying
fastest, then G, then B.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from spektrafilm_lut_creator.formats import Lut, register

_VALUE_FORMAT = "{:.10g}"


class CubeFormat:
    name = "cube"
    extensions = (".cube",)

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
                # Allow callers to pass blank "section break" entries; format
                # everything else as a comment so any parser ignores it.
                lines.append(f"# {raw}" if raw else "#")
        if lut.title:
            lines.append(f'TITLE "{lut.title}"')
        lines.append(
            "DOMAIN_MIN " + " ".join(_VALUE_FORMAT.format(v) for v in lut.domain_min)
        )
        lines.append(
            "DOMAIN_MAX " + " ".join(_VALUE_FORMAT.format(v) for v in lut.domain_max)
        )
        lines.append(f"LUT_3D_SIZE {n}")
        for r, g, b in flat:
            lines.append(" ".join(_VALUE_FORMAT.format(v) for v in (r, g, b)))
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")

    def read(self, path: Path) -> Lut:
        title = ""
        domain_min = (0.0, 0.0, 0.0)
        domain_max = (1.0, 1.0, 1.0)
        size: int | None = None
        values: list[tuple[float, float, float]] = []

        for raw in Path(path).read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            head, _, rest = line.partition(" ")
            head_upper = head.upper()
            if head_upper == "TITLE":
                title = rest.strip().strip('"')
            elif head_upper == "DOMAIN_MIN":
                domain_min = _parse_triplet(rest)
            elif head_upper == "DOMAIN_MAX":
                domain_max = _parse_triplet(rest)
            elif head_upper == "LUT_3D_SIZE":
                size = int(rest.strip())
            elif head_upper == "LUT_1D_SIZE":
                raise ValueError(f"1D .cube LUT not supported (LUT_1D_SIZE in {path})")
            else:
                # Body line: three floats.
                values.append(_parse_triplet(line))

        if size is None:
            raise ValueError(f"{path}: missing LUT_3D_SIZE header")
        if len(values) != size**3:
            raise ValueError(
                f"{path}: body has {len(values)} entries, expected size**3 = {size**3}"
            )
        table = np.asarray(values, dtype=float).reshape(size, size, size, 3)
        return Lut(
            table=table, domain_min=domain_min, domain_max=domain_max, title=title
        )


def _parse_triplet(text: str) -> tuple[float, float, float]:
    parts = text.split()
    if len(parts) != 3:
        raise ValueError(f"expected 3 floats, got {len(parts)} in {text!r}")
    return (float(parts[0]), float(parts[1]), float(parts[2]))


register(CubeFormat())
