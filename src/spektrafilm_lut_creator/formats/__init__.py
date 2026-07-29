"""LUT format plugin registry and the canonical :class:`Lut` data class.

Each format plugin exposes a :class:`LutFormat`-conforming object via
:func:`register`. v1 ships ``cube`` only; other formats (``3dl``,
``hald_png``, ``ocio_config``) register themselves the same way later.

See studies/a40_lut_system/n030_lut_package_design.md §8.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass
class Lut:
    """A 3D LUT in unit-cube form.

    ``table`` has shape ``(N, N, N, 3)`` with indexing ``table[b, g, r, :]``
    — i.e. ``table.reshape(-1, 3)`` is in Adobe canonical .cube order
    (R fastest, then G, then B). Values are typically in ``[0, 1]`` but
    the dataclass does not enforce it.
    """

    table: np.ndarray
    domain_min: tuple[float, float, float] = (0.0, 0.0, 0.0)
    domain_max: tuple[float, float, float] = (1.0, 1.0, 1.0)
    title: str = ""

    @property
    def resolution(self) -> int:
        if self.table.ndim != 4 or self.table.shape[-1] != 3:
            raise ValueError(
                f"Lut.table must have shape (N, N, N, 3), got {self.table.shape}"
            )
        n = self.table.shape[0]
        if self.table.shape[:3] != (n, n, n):
            raise ValueError(
                f"Lut.table must be a cube along its first three axes, got {self.table.shape}"
            )
        return n


class LutFormat(Protocol):
    """Protocol for LUT format plugins."""

    name: str
    extensions: tuple[str, ...]

    def write(
        self, lut: Lut, path: Path, *, header_lines: list[str] | None = None
    ) -> None: ...
    def read(self, path: Path) -> Lut: ...


LUT_FORMATS: dict[str, LutFormat] = {}


def register(fmt: LutFormat) -> None:
    """Register a format plugin under its :attr:`name`."""
    LUT_FORMATS[fmt.name] = fmt


def get_format(name: str) -> LutFormat:
    """Return the registered plugin for ``name`` or raise ``KeyError``."""
    try:
        return LUT_FORMATS[name]
    except KeyError:
        raise KeyError(
            f"Unknown LUT format {name!r}. Registered: {sorted(LUT_FORMATS)}"
        ) from None


# Import side-effect: register the built-in plugins.
from spektrafilm_lut_creator.formats import cube as _cube  # noqa: E402,F401
from spektrafilm_lut_creator.formats import hald_png as _hald_png  # noqa: E402,F401
from spektrafilm_lut_creator.formats import lumix as _lumix  # noqa: E402,F401
from spektrafilm_lut_creator.formats import threedl as _threedl  # noqa: E402,F401
