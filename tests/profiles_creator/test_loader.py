from __future__ import annotations

import pytest

from spektrafilm_profile_creator import RawProfile, RawProfileRecipe, load_raw_profile


pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    (
        'stock',
        'expected_support',
        'expected_reference_channel',
        'expected_target_paper',
        'expected_data_trustability',
        'expected_stretch_curves',
    ),
    [
        ('kodak_portra_400', 'film', None, 'kodak_portra_endura_uc', 1.0, False),
        ('kodak_portra_endura', 'paper', None, None, None, None),
        ('fujifilm_c200', 'film', 'green', None, None, None),
        ('fujifilm_pro_400h', 'film', 'mid', None, None, None),
    ],
    ids=['portra-film', 'portra-paper', 'fuji-c200', 'fuji-pro-400h'],
)
def test_load_raw_profile_reads_expected_info_and_recipe(
    stock: str,
    expected_support: str,
    expected_reference_channel: str | None,
    expected_target_paper: str | None,
    expected_data_trustability: float | None,
    expected_stretch_curves: bool | None,
) -> None:
    raw_profile = load_raw_profile(stock)

    assert isinstance(raw_profile, RawProfile)
    assert isinstance(raw_profile.recipe, RawProfileRecipe)
    assert raw_profile.info.stock == stock
    assert raw_profile.info.support == expected_support
    assert raw_profile.info.type == 'negative'
    assert raw_profile.info.channel_model == 'color'
    assert raw_profile.recipe.reference_channel == expected_reference_channel

    if expected_target_paper is not None:
        assert raw_profile.recipe.dye_density_reconstruct_model == 'dmid_dmin'
        assert raw_profile.recipe.target_paper == expected_target_paper
        assert raw_profile.recipe.data_trustability == pytest.approx(expected_data_trustability)
        assert raw_profile.recipe.stretch_curves is expected_stretch_curves