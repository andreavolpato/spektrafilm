"""Off-grid LUT evaluators.

Industry hosts evaluate ``.cube`` LUTs by interpolating the 3D table at
off-grid sample positions. The two dominant methods:

- **Trilinear**: Premiere, FFmpeg, OBS, most OCIO defaults. Linear
  interpolation in each of the three cube axes (8 corner taps per
  sample).
- **Tetrahedral**: DaVinci Resolve, Baselight, FilmLight. Decomposes
  each cube cell into 6 tetrahedra and interpolates linearly inside the
  relevant one (4 corner taps per sample). Better preservation of
  achromatic axis behavior than trilinear.

A bake that looks correct under trilinear can show small but visible
differences under tetrahedral and vice versa — this is why off-grid
QA must test both. See e.g. Kirk, Truelight whitepapers (FilmLight).

Both evaluators take a LUT table of shape ``(N, N, N, 3)`` indexed
``[b, g, r, :]`` (Adobe canonical) and a flat sample array of shape
``(M, 3)`` of RGB values in ``[0, 1]``, and return ``(M, 3)``.
"""

from __future__ import annotations

import numpy as np


def apply_trilinear(table: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """Apply a 3D LUT to ``samples`` using trilinear interpolation.

    Parameters
    ----------
    table
        Shape ``(N, N, N, 3)`` indexed ``[b, g, r, :]`` (matches
        :class:`spektrafilm_lut_creator.formats.Lut.table`).
    samples
        Shape ``(M, 3)``, values in ``[0, 1]``. Out-of-range samples
        are clamped.

    Returns
    -------
    np.ndarray
        Shape ``(M, 3)``. dtype follows ``table.dtype`` promoted to
        float.
    """
    table = np.asarray(table, dtype=float)
    samples = np.clip(np.asarray(samples, dtype=float), 0.0, 1.0)
    n = table.shape[0]
    # Lattice-space coordinates: a sample at 0.0 lands on grid index 0,
    # a sample at 1.0 lands on grid index N-1.
    lattice = samples * (n - 1)
    i0 = np.floor(lattice).astype(int)
    i0 = np.clip(i0, 0, n - 2)  # leave room for i0+1
    i1 = i0 + 1
    frac = lattice - i0  # shape (M, 3), in [0, 1]

    fr, fg, fb = frac[:, 0], frac[:, 1], frac[:, 2]
    r0, r1 = i0[:, 0], i1[:, 0]
    g0, g1 = i0[:, 1], i1[:, 1]
    b0, b1 = i0[:, 2], i1[:, 2]

    # Eight corner taps. Indexing is table[b, g, r].
    c000 = table[b0, g0, r0]
    c100 = table[b0, g0, r1]
    c010 = table[b0, g1, r0]
    c110 = table[b0, g1, r1]
    c001 = table[b1, g0, r0]
    c101 = table[b1, g0, r1]
    c011 = table[b1, g1, r0]
    c111 = table[b1, g1, r1]

    # Bilinear in R, G:
    c00 = c000 * (1 - fr)[:, None] + c100 * fr[:, None]
    c01 = c001 * (1 - fr)[:, None] + c101 * fr[:, None]
    c10 = c010 * (1 - fr)[:, None] + c110 * fr[:, None]
    c11 = c011 * (1 - fr)[:, None] + c111 * fr[:, None]
    c0 = c00 * (1 - fg)[:, None] + c10 * fg[:, None]
    c1 = c01 * (1 - fg)[:, None] + c11 * fg[:, None]
    # And along B:
    return c0 * (1 - fb)[:, None] + c1 * fb[:, None]


def apply_tetrahedral(table: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """Apply a 3D LUT to ``samples`` using tetrahedral interpolation.

    Each cube cell is decomposed into 6 tetrahedra sharing the main
    diagonal (V000 → V111). Each sample picks the tetrahedron whose
    fractional-coordinate ordering matches its own, then linearly
    interpolates the 4 corners of that tetrahedron.

    See Kirk, "Tetrahedral Interpolation" (FilmLight Truelight
    whitepapers), and the standard reference in the OCIO and Resolve
    documentation.

    Parameters
    ----------
    table
        Shape ``(N, N, N, 3)`` indexed ``[b, g, r, :]``.
    samples
        Shape ``(M, 3)``, values in ``[0, 1]``. Out-of-range clamped.

    Returns
    -------
    np.ndarray
        Shape ``(M, 3)``.
    """
    table = np.asarray(table, dtype=float)
    samples = np.clip(np.asarray(samples, dtype=float), 0.0, 1.0)
    n = table.shape[0]
    lattice = samples * (n - 1)
    i0 = np.clip(np.floor(lattice).astype(int), 0, n - 2)
    frac = lattice - i0
    fr, fg, fb = frac[:, 0], frac[:, 1], frac[:, 2]
    r0, g0, b0 = i0[:, 0], i0[:, 1], i0[:, 2]
    r1, g1, b1 = r0 + 1, g0 + 1, b0 + 1

    # Pre-fetch the eight corner values for each sample.
    c000 = table[b0, g0, r0]
    c100 = table[b0, g0, r1]
    c010 = table[b0, g1, r0]
    c110 = table[b0, g1, r1]
    c001 = table[b1, g0, r0]
    c101 = table[b1, g0, r1]
    c011 = table[b1, g1, r0]
    c111 = table[b1, g1, r1]

    # The six tetrahedra share the V000 → V111 diagonal. Each is
    # selected by the ordering of (fr, fg, fb). Standard table:
    #   fr>=fg>=fb : V000, V100, V110, V111
    #   fr>=fb>=fg : V000, V100, V101, V111
    #   fb>=fr>=fg : V000, V001, V101, V111
    #   fg>=fr>=fb : V000, V010, V110, V111
    #   fg>=fb>=fr : V000, V010, V011, V111
    #   fb>=fg>=fr : V000, V001, V011, V111
    out = np.empty_like(c000)

    m1 = (fr >= fg) & (fg >= fb)
    m2 = (fr >= fb) & (fb > fg)
    m3 = (fb > fr) & (fr >= fg)
    m4 = (fg > fr) & (fr >= fb)
    m5 = (fg >= fb) & (fb > fr)
    m6 = (fb > fg) & (fg > fr)

    def lerp4(mask, c_a, c_b, c_c, c_d, w_a, w_b, w_c, w_d):
        if not np.any(mask):
            return
        out[mask] = (
            c_a[mask] * w_a[mask, None]
            + c_b[mask] * w_b[mask, None]
            + c_c[mask] * w_c[mask, None]
            + c_d[mask] * w_d[mask, None]
        )

    # Barycentric weights inside each tetrahedron.
    # Edge ordering: V000 -> V_first -> V_second -> V111 with fractions
    # that move along R, then G, then B (etc.) so weights are
    # 1-largest_axis, largest_axis-middle_axis, middle_axis-smallest_axis, smallest_axis.
    # Example for fr>=fg>=fb: V000 weight = 1-fr; V100 = fr-fg; V110 = fg-fb; V111 = fb.
    lerp4(m1, c000, c100, c110, c111, 1.0 - fr, fr - fg, fg - fb, fb)
    lerp4(m2, c000, c100, c101, c111, 1.0 - fr, fr - fb, fb - fg, fg)
    lerp4(m3, c000, c001, c101, c111, 1.0 - fb, fb - fr, fr - fg, fg)
    lerp4(m4, c000, c010, c110, c111, 1.0 - fg, fg - fr, fr - fb, fb)
    lerp4(m5, c000, c010, c011, c111, 1.0 - fg, fg - fb, fb - fr, fr)
    lerp4(m6, c000, c001, c011, c111, 1.0 - fb, fb - fg, fg - fr, fr)

    return out
