"""Hald-CLUT PNG writer.

A Hald CLUT (Sasaki, 2007) packs a 3D LUT into a square PNG image by
treating it as an image of side ``N²`` pixels for a LUT of side ``N²``
cells. The standard form requires the cube resolution to be a perfect
square: a *level-L* Hald CLUT has cube resolution ``L²`` and image
dimensions ``L²×L² × L²×L²`` pixels (equivalently, an ``L⁶``-pixel
square).

Common levels:

================  ===================  ====================  ==============
Level ``L``       Cube resolution      Image dimensions      Pixels (= N³)
================  ===================  ====================  ==============
4                 16                   16 × 16 → 256 × 256   4 096
6                 36                   36 × 36 → 1 296 × 1   46 656
8                 64                   64 × 64 → 4 096 × 1   262 144
================  ===================  ====================  ==============

Memory layout: pixel ``(x, y)`` of the image corresponds to LUT cell
``(r, g, b)`` where ``index = y * (L²)² + x``, with ``r`` varying
fastest, then ``g``, then ``b`` — same ordering as Adobe ``.cube``.
This matches Niku Sasaki's reference implementation and the layout
expected by Photoshop's *Color Lookup* filter and ImageMagick's
``hald:`` device.

Encoding: 8-bit per channel, sRGB-curve-free. 8-bit is the de-facto
Hald CLUT format every consumer (Photoshop *Color Lookup*, OBS *Apply
LUT*, ImageMagick ``hald:``) expects; 16-bit RGB-PNG support is
patchy across these tools, and the 8-bit round-trip error
(``1/255`` per channel ≈ ``0.4%``) is acceptable for the workflows
Hald ONGs are used in. The PNG carries the LUT values verbatim;
consumers apply the relevant transfer functions themselves.

See [n090 §4.1](../../../../spektrafilm-research/studies/a40_lut_system/n090_industry_grade_bundles.md)
for the multi-format roadmap.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from spektrafilm_lut_creator.formats import Lut, register

_MAX_8BIT = (1 << 8) - 1


class HaldPNGFormat:
    """Hald-CLUT PNG writer (registers as ``"hald_png"``).

    Restriction: ``lut.resolution`` must be a perfect square (``16``,
    ``25``, ``36``, ``49``, ``64``, …). Other resolutions raise
    :class:`ValueError`. Bake at ``resolution=64`` (level 8) for the
    standard Hald CLUT consumers like Photoshop expect.
    """

    name = "hald_png"
    extensions = (".png",)

    def write(
        self,
        lut: Lut,
        path: Path,
        *,
        header_lines: list[str] | None = None,
    ) -> None:
        del header_lines  # PNG metadata isn't part of the Hald contract
        from PIL import Image

        n = lut.resolution
        level = _hald_level(n)
        side = n * level  # == L³ == sqrt(N³)
        flat = np.asarray(lut.table, dtype=float).reshape(n**3, 3)
        image_array = (
            np.clip(np.round(flat * _MAX_8BIT), 0, _MAX_8BIT)
            .astype(np.uint8)
            .reshape(side, side, 3)
        )
        Image.fromarray(image_array, mode="RGB").save(path, format="PNG")

    def read(self, path: Path) -> Lut:
        from PIL import Image

        with Image.open(path) as im:
            im.load()
            if im.mode != "RGB":
                im = im.convert("RGB")
            arr = np.asarray(im)
        side = arr.shape[0]
        if arr.shape != (side, side, 3):
            raise ValueError(
                f"{path}: expected a square RGB image, got shape {arr.shape}"
            )
        # side = L³, so cube resolution N = L² = side**(2/3).
        level = round(side ** (1.0 / 3.0))
        if level**3 != side:
            raise ValueError(
                f"{path}: image side {side} is not a Hald-CLUT level "
                f"(L³ with integer L)"
            )
        n = level * level
        table = arr.reshape(n, n, n, 3).astype(float) / float(_MAX_8BIT)
        return Lut(table=table, domain_min=(0.0, 0.0, 0.0), domain_max=(1.0, 1.0, 1.0))


def _hald_level(resolution: int) -> int:
    """Return the Hald level ``L`` such that ``L² == resolution``, or
    raise :class:`ValueError` if the resolution isn't a perfect square.
    """
    level = int(math.isqrt(resolution))
    if level * level != resolution:
        raise ValueError(
            f"Hald CLUT requires a perfect-square cube resolution "
            f"(16, 25, 36, 49, 64, …); got {resolution}. Bake at "
            f"resolution=64 (level 8) for the standard form."
        )
    return level


register(HaldPNGFormat())
