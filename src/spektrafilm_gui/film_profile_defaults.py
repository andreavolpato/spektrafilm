from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Mapping

from spektrafilm.profiles.io import load_profile
from spektrafilm_gui.state import GuiState, clone_gui_state


@dataclass(frozen=True, slots=True)
class GuiStateOverride:
    section_name: str
    changes: Mapping[str, object]


NEGATIVE_FILM_DEFAULT_OVERRIDES = (
    GuiStateOverride(
        section_name='couplers',
        changes={'dir_couplers_ratio': (1.0, 1.0, 1.0)},
    ),
    GuiStateOverride(
        section_name='simulation',
        changes={'scan_film': False},
    ),
)

POSITIVE_FILM_DEFAULT_OVERRIDES = (
    GuiStateOverride(
        section_name='couplers',
        changes={'dir_couplers_ratio': (1.0, 0.65, 0.35)},
    ),
    GuiStateOverride(
        section_name='simulation',
        changes={'scan_film': True},
    ),
)

PROFILE_TYPE_DEFAULT_OVERRIDES = {
    'negative': NEGATIVE_FILM_DEFAULT_OVERRIDES,
    'positive': POSITIVE_FILM_DEFAULT_OVERRIDES,
}


@lru_cache(maxsize=None)
def default_overrides_for_film_stock(film_stock: str) -> tuple[GuiStateOverride, ...]:
    profile_type = load_profile(film_stock).info.type
    return PROFILE_TYPE_DEFAULT_OVERRIDES.get(profile_type, ())


def apply_gui_state_overrides(state: GuiState, overrides: tuple[GuiStateOverride, ...]) -> GuiState:
    next_state = clone_gui_state(state)
    for override in overrides:
        section_state = getattr(next_state, override.section_name)
        setattr(next_state, override.section_name, replace(section_state, **override.changes))
    return next_state


def override_section_names(overrides: tuple[GuiStateOverride, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    section_names: list[str] = []
    for override in overrides:
        if override.section_name in seen:
            continue
        seen.add(override.section_name)
        section_names.append(override.section_name)
    return tuple(section_names)


__all__ = [
    'GuiStateOverride',
    'NEGATIVE_FILM_DEFAULT_OVERRIDES',
    'POSITIVE_FILM_DEFAULT_OVERRIDES',
    'PROFILE_TYPE_DEFAULT_OVERRIDES',
    'default_overrides_for_film_stock',
    'apply_gui_state_overrides',
    'override_section_names',
]