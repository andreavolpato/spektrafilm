from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from typing import Any

from spektrafilm.profiles.io import Profile, ProfileData, ProfileInfo


@dataclass(frozen=True, slots=True)
class RawProfileRecipe:
    log_sensitivity_donor: str | None = None
    density_curves_donor: str | None = None
    dye_density_cmy_donor: str | None = None
    dye_density_min_mid_donor: str | None = None
    dye_density_reconstruct_model: str = 'dmid_dmin'
    apply_gray_ramp: bool = False
    gray_ramp_kwargs: dict[str, Any] = field(default_factory=dict)
    align_midscale_exposures: bool = False
    reference_channel: str | None = None
    target_paper: str | None = None
    data_trustability: float = 1.0
    correction_reference_channel: str | None = None
    should_process: bool = True


@dataclass
class RawProfile:
    info: ProfileInfo = field(default_factory=ProfileInfo)
    data: ProfileData = field(default_factory=ProfileData)
    recipe: RawProfileRecipe = field(default_factory=RawProfileRecipe)

    def __post_init__(self):
        if not isinstance(self.info, ProfileInfo):
            raise TypeError('info must be a ProfileInfo instance')
        if not isinstance(self.data, ProfileData):
            raise TypeError('data must be a ProfileData instance')
        if not isinstance(self.recipe, RawProfileRecipe):
            raise TypeError('recipe must be a RawProfileRecipe instance')

    def clone(self) -> 'RawProfile':
        return copy.deepcopy(self)

    def update_info(self, **changes) -> 'RawProfile':
        self.info = replace(self.info, **changes)
        return self

    def update_data(self, **changes) -> 'RawProfile':
        self.data = replace(self.data, **changes)
        return self

    def update_recipe(self, **changes) -> 'RawProfile':
        if 'gray_ramp_kwargs' in changes and changes['gray_ramp_kwargs'] is not None:
            changes['gray_ramp_kwargs'] = dict(changes['gray_ramp_kwargs'])
        self.recipe = replace(self.recipe, **changes)
        return self

    def update(self, *, info=None, data=None, recipe=None) -> 'RawProfile':
        if info:
            self.update_info(**info)
        if data:
            self.update_data(**data)
        if recipe:
            self.update_recipe(**recipe)
        return self

    def as_profile(self) -> Profile:
        return Profile(
            info=copy.deepcopy(self.info),
            data=copy.deepcopy(self.data),
        )


__all__ = [
    'RawProfile',
    'RawProfileRecipe',
]