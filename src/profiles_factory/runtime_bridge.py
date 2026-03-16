"""Bridge from profiles_factory to spectral_film_lab runtime API."""

from __future__ import annotations

from typing import Any

from spectral_film_lab.runtime.api import make_runtime_params, run_photo_process


__all__ = ["make_runtime_params", "run_photo_process"]


def run_with_runtime(image: Any, params: Any) -> Any:
    """Small helper to keep runtime access centralized for future extraction."""
    return run_photo_process(image, params)
