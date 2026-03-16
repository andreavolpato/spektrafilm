"""Stable runtime boundary used by spectral_film_lab consumers.

This first step delegates to existing agx_emulsion functions to avoid behavior changes.
"""

from __future__ import annotations

from typing import Any

from spectral_film_lab.runtime.process import photo_params, photo_process
from spectral_film_lab.profile_store.io import load_profile, save_profile


def make_runtime_params(
    negative: str = "kodak_portra_400",
    print_paper: str = "kodak_portra_endura_uc",
    ymc_filters_from_database: bool = True,
) -> Any:
    """Build runtime parameters for one simulation configuration."""
    return photo_params(
        negative=negative,
        print_paper=print_paper,
        ymc_filters_from_database=ymc_filters_from_database,
    )


def run_photo_process(image: Any, params: Any) -> Any:
    """Run one photo simulation pass."""
    return photo_process(image, params)


def load_processed_profile(stock: str) -> Any:
    """Load a processed profile by stock name."""
    return load_profile(stock)


def save_processed_profile(profile: Any, suffix: str = "") -> None:
    """Save a processed profile object."""
    save_profile(profile, suffix=suffix)
