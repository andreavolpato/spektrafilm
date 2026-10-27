from __future__ import annotations

from typing import Any, Iterable


PROFILE_SYNC_SECTION_NAMES = (
    'grain',
    'halation',
    'couplers',
    'chemistry',
    'film_chemistry',
    'glare',
    'special',
    'simulation',
    'camera',
    'preflashing',
    'enlarger_diffusion',
    'camera_diffusion',
    'scanner',
    'input_image',
)


def apply_profile_sync_state(
    *,
    widgets: Any,
    synced_state: Any,
    profile_sync_section_names: Iterable[str] | None = None,
) -> None:
    if profile_sync_section_names is None:
        profile_sync_section_names = PROFILE_SYNC_SECTION_NAMES
    for section_name in profile_sync_section_names:
        section_widget = getattr(widgets, section_name)
        section_state = getattr(synced_state, section_name)
        set_state = getattr(section_widget, 'set_state', None)
        if callable(set_state):
            section_widget.set_state(section_state)
