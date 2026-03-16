"""Public API for raw-curve to processed-profile generation."""

from __future__ import annotations

from typing import Any

from profiles_creator.correct import correct_negative_curves_with_gray_ramp
from profiles_creator.factory import (
    create_profile,
    process_negative_profile,
    process_paper_profile,
)
from profiles_creator.fitting import fit_print_filters


__all__ = [
    "create_profile",
    "process_negative_profile",
    "process_paper_profile",
    "correct_negative_curves_with_gray_ramp",
    "fit_print_filters",
]


def create_raw_profile(*args: Any, **kwargs: Any) -> Any:
    """Compatibility alias for create_profile naming in the split architecture."""
    return create_profile(*args, **kwargs)

