"""3D sampling-grid generation and pipeline-image reshaping.

Adobe .cube canonical order: R varies fastest, then G, then B. A flat
grid of shape ``(N**3, 3)`` lists those samples in that order. The
spektrafilm pipeline expects ``(H, W, 3)`` image input; the builder
reshapes the grid into an image, runs the pipeline, and reshapes back.

The shape choice for ``(H, W)`` doesn't affect output correctness when
``lut_mode`` is active (all spatial effects are off), but it does affect
performance — long, thin shapes can stall numba inner loops. The
default ``H = resolution, W = resolution ** 2`` is a reasonable
compromise.

See studies/a40_lut_system/n030_lut_package_design.md §7.
"""

from __future__ import annotations

import numpy as np


def cube_grid(resolution: int) -> np.ndarray:
    """Return a flat sampling grid of shape ``(resolution ** 3, 3)`` in
    Adobe .cube canonical order (R fastest, then G, then B).

    Samples are evenly spaced in ``[0, 1]`` along each axis.
    """
    if resolution < 2:
        raise ValueError(f"resolution must be >= 2, got {resolution}")
    axis = np.linspace(0.0, 1.0, resolution)
    # meshgrid with 'ij' indexing and (r, g, b) ordering then ravel-as-(b,g,r)
    # would also work; the explicit nested form below is clearer and matches
    # Adobe's spec wording.
    r, g, b = np.meshgrid(axis, axis, axis, indexing="ij")
    # Transpose so b is the slowest (outermost) and r the fastest (innermost),
    # then flatten C-order. After transpose r,g,b each have shape (N,N,N)
    # indexed as [b, g, r].
    r = r.transpose(2, 1, 0)
    g = g.transpose(2, 1, 0)
    b = b.transpose(2, 1, 0)
    return np.stack([r.ravel(), g.ravel(), b.ravel()], axis=-1)


def grid_as_image(grid: np.ndarray, resolution: int) -> np.ndarray:
    """Reshape a flat ``(N**3, 3)`` grid into an ``(H, W, 3)`` image.

    Default layout is ``(resolution, resolution ** 2, 3)``; the caller
    can re-reshape further if a different aspect is desired.
    """
    expected = (resolution**3, 3)
    if grid.shape != expected:
        raise ValueError(
            f"grid shape {grid.shape} does not match (resolution**3, 3) = {expected}"
        )
    return grid.reshape(resolution, resolution**2, 3)


def image_as_grid(image: np.ndarray, resolution: int) -> np.ndarray:
    """Inverse of :func:`grid_as_image`: ``(H, W, 3)`` → ``(N**3, 3)``.

    The image must contain ``resolution ** 3`` samples in C order."""
    if image.shape[-1] != 3:
        raise ValueError(f"image last axis must be 3, got shape {image.shape}")
    n_samples = int(np.prod(image.shape[:-1]))
    if n_samples != resolution**3:
        raise ValueError(
            f"image has {n_samples} samples, expected resolution**3 = {resolution**3}"
        )
    return image.reshape(resolution**3, 3)
